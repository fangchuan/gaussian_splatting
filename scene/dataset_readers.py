#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

import open3d as o3d
from utils.pano_utils import Equirec2Cube

class CameraInfo(NamedTuple):
    uid: int
    R: np.array   # c2w
    T: np.array   #w2c
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    print(f'Reading {len(cam_extrinsics)} cameras')
    print(f'Camera intrinsic: {cam_intrinsics}')
    print(f'Images folder: {images_folder}')
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    if 'nx' not in vertices.dtype.names:
        normals = np.zeros_like(positions)
    else:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def fetchOpen3DPly(path:str, T:np.array=None):
    """ load point cloud

    Args:
        path (str): point cloud path
        T (np.array, optional): 4x4 transform. Defaults to None.

    Returns:
        _type_: _description_
    """
    plydata = o3d.io.read_point_cloud(path)
    positions = np.asarray(plydata.points)
    colors = np.asarray(plydata.colors)

    if plydata.has_normals():
        normals = np.asarray(plydata.normals)
    else:
        normals = np.zeros_like(positions)
    
    # # rotate points if T is given
    # if T is not None:
    #     positions = np.transpose(positions)
    #     positions = T @ np.vstack((positions, np.ones((1, positions.shape[1]))))
    #     positions = np.transpose(positions)[:,:3]
    #     positions = np.transpose(positions)

    #     normals = np.transpose(normals)
    #     normals = T[:3,:3] @ normals
    #     normals = np.transpose(normals)

    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

# StructureD 3D data set
"""root_folder
    source_views
        scene_00000_492165.png
        scene_00001_906323.png
        ...
    source_depths
        scene_00000_492165.png
        scene_00001_906323.png
        ...
    source_layouts
        scene_00000_492165.txt
        scene_00001_906323.txt
        ...
    source_cameras
        scene_00000_492165.txt
        scene_00001_906323.txt
        ...
    target_views
        scene_00000_492165_0.png
        scene_00000_492165_1.png
        scene_00000_492165_2.png
        ...
    target_cameras
        scene_00000_492165_0.txt
        scene_00000_492165_1.txt
        scene_00000_492165_2.txt
        ...
    points3d.ply
"""
def read_split_file(split_file:str):
    splits = []
    with open(split_file, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            splits.append(line.strip())
    return splits

def read_camera_pose_file(cam_pose_file:str):
    cam_poses = []
    with open(cam_pose_file, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            cam_poses.append(np.array([float(x) for x in line.strip().split(" ")]))
    return cam_poses

def readST3DSceneInfo(scene_path:str, is_eval:bool, is_use_cubemap:bool=False):
    cam_infos_lst = []
    rgb_images_lst = [fn for fn in os.listdir(os.path.join(scene_path)) if fn.startswith('image_')]
    rgb_images_lst.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    cam_pose_file = os.path.join(scene_path, "cameras.txt")
    cam_poses = read_camera_pose_file(cam_pose_file)

    assert len(cam_poses) == len(rgb_images_lst)

    if is_use_cubemap:
        e2c = Equirec2Cube(equ_h=512, equ_w=1024, face_w=256)
        # R_raw_cubemap = R.from_rotvec(-np.pi/2 * np.array([1, 0, 0])).as_matrix()
        R_raw_cubemap = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        R_cubemap_raw = R_raw_cubemap.transpose()
        T_cubemap_raw = np.eye(4)
        T_cubemap_raw[:3,:3] = R_cubemap_raw
    else:
        T_cubemap_raw = None

    for idx, img_file in enumerate(rgb_images_lst):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{} \n".format(idx+1, len(rgb_images_lst)))
        sys.stdout.flush()

        image_path = os.path.join(scene_path, img_file)
        image_name = img_file
        image = Image.open(image_path)

        # cubemaps
        equi_image = np.array(image.convert("RGB"))
        cubemap_imgs = e2c.run(equi_image)
        cubemap_poses = e2c.cubemap_poses()
        if not is_use_cubemap:
            # convert c2w to w2c
            cam_center = -cam_poses[idx][:3]
            print(f'image_name: {image_name} cam_center: {cam_center}')
            cam_infos_lst.append(CameraInfo(uid=idx, R=np.eye(3), T=cam_center, FovY=180, FovX=360, image=image,
                    image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
        else:
            raw_cam_center_w_c = cam_poses[idx][:3]
            new_cam_center_w_c = T_cubemap_raw[:3,:3] @ raw_cam_center_w_c + T_cubemap_raw[:3,3]
            for i in range(6):
                cubemap_img = cubemap_imgs[:,i*256:(i+1)*256,:]
                cubemap_img = Image.fromarray(cubemap_img)
                T_ref_subview = cubemap_poses[i]
                cubemap_dir = os.path.join(scene_path, 'cubemap_images')
                if not os.path.exists(cubemap_dir):
                    os.makedirs(cubemap_dir)
                cubemap_img_name = f'{image_name[:-4]}_{i}.png'
                cubemap_img_path = os.path.join(cubemap_dir, cubemap_img_name)
                cubemap_img.save(cubemap_img_path)
                R_ref_subview = T_ref_subview[:3, :3]
                R_w_c = R_ref_subview
                cam_center_c_w = -R_w_c.transpose() @ new_cam_center_w_c
                print(f'image_name: {cubemap_img_name} cam_center_w_c: {new_cam_center_w_c}')
                cam_infos_lst.append(CameraInfo(uid=idx*6+i, R=R_w_c, T=cam_center_c_w, FovY=90, FovX=90, image=cubemap_img,
                                    image_path=cubemap_img_path, image_name=cubemap_img_name, width=cubemap_img.size[0], height=cubemap_img.size[1]))
    sys.stdout.write('\n')

    train_split_file = os.path.join(scene_path, "train.txt")
    test_split_file = os.path.join(scene_path, "test.txt")
    train_splits = read_split_file(train_split_file)

    if is_eval:
        if not is_use_cubemap:
            train_cam_infos = [c for idx, c in enumerate(cam_infos_lst) if c.image_name in train_splits or c.image_name == "image_0.png"]
            # print(f'train_splits: {train_splits}')
            if os.path.exists(test_split_file):
                test_splits = read_split_file(test_split_file)
                test_cam_infos = [c for idx, c in enumerate(cam_infos_lst) if c.image_name in test_splits]
                # print(f'test_splits: {test_splits}')
        else:
            train_cubemap_img_names = [f'{train_img_name[:-4]}_{i}.png' for train_img_name in train_splits for i in range(6)]
            train_cam_infos = [c for idx, c in enumerate(cam_infos_lst) if c.image_name in train_cubemap_img_names or "image_0" in c.image_name ]
            if os.path.exists(test_split_file):
                test_splits = read_split_file(test_split_file)
                test_cubemap_img_names = [f'{test_img_name[:-4]}_{i}.png' for test_img_name in test_splits for i in range(6)]
                test_cam_infos = [c for idx, c in enumerate(cam_infos_lst) if c.image_name in test_cubemap_img_names]
    else:
        train_cam_infos = cam_infos_lst
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(scene_path, "points3d.ply")
    if is_use_cubemap:
        o3d_raw_pcl = o3d.io.read_point_cloud(ply_path)
        raw_points = o3d_raw_pcl.points
        raw_points = np.asarray(raw_points)
        raw_points = np.transpose(raw_points)
        points_in_cam = R_cubemap_raw @ raw_points
        o3d_raw_pcl.points = o3d.utility.Vector3dVector(np.transpose(points_in_cam))
        ply_path = os.path.join(scene_path, "points3d_cubemap.ply")
        o3d.io.write_point_cloud(ply_path, o3d_raw_pcl)
    # if not os.path.exists(ply_path):
    #     # Since this data set has no colmap data, we start with random points
    #     num_pts = 100_000
    #     print(f"Generating random point cloud ({num_pts})...")
        
    #     # We create random points inside the bounds of the synthetic Blender scenes
    #     xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
    #     shs = np.random.random((num_pts, 3)) / 255.0
    #     pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    #     storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchOpen3DPly(ply_path, T=T_cubemap_raw)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                            ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "ST3D": readST3DSceneInfo
}