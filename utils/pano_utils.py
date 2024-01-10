import os
import sys

import numpy as np
import cv2
import open3d as o3d
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation as R

def np_coorx2u(coorx, coorW=1024):
    return ((coorx + 0.5) / coorW - 0.5) * 2 * np.pi


def np_coory2v(coory, coorH=512):
    return -((coory + 0.5) / coorH - 0.5) * np.pi


def vis_color_pointcloud(rgb_img_filepath:str, depth_img_filepath:str, saved_color_pcl_filepath:str)->o3d.geometry.PointCloud:
    """
    :param rgb_img_filepath: rgb panorama image filepath
    :param depth_img_filepath: depth panorama image filepath
    :param saved_color_pcl_filepath: saved color point cloud filepath
    :return: o3d.geometry.PointCloud
    """

    def get_unit_spherical_map():
        h = 512
        w = 1024

        coorx, coory = np.meshgrid(np.arange(w), np.arange(h))
        us = np_coorx2u(coorx, w)
        vs = np_coory2v(coory, h)

        X = np.expand_dims(np.cos(vs) * np.sin(us), 2)
        Z = np.expand_dims(np.sin(vs), 2)
        Y = np.expand_dims(np.cos(vs) * np.cos(us), 2)
        unit_map = np.concatenate([X, Y, Z], axis=2)

        return unit_map

    def display_inlier_outlier(cloud, ind):
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)

        print("Showing outliers (red) and inliers (gray): ")
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw([inlier_cloud, outlier_cloud])

    assert os.path.exists(rgb_img_filepath), 'rgb panorama doesnt exist!!!'
    assert os.path.exists(depth_img_filepath), 'depth panorama doesnt exist!!!'

    raw_depth_img = cv2.imread(depth_img_filepath, cv2.IMREAD_UNCHANGED)
    if len(raw_depth_img.shape) == 3:
        raw_depth_img = cv2.cvtColor(raw_depth_img, cv2.COLOR_BGR2GRAY)
    depth_img = np.asarray(raw_depth_img)
    if np.isnan(depth_img.any()) or len(depth_img[depth_img > 0]) == 0:
        print('empyt depth image')
        exit(-1)

    raw_rgb_img = cv2.imread(rgb_img_filepath, cv2.IMREAD_UNCHANGED)
    rgb_img = cv2.cvtColor(raw_rgb_img, cv2.COLOR_BGR2RGB)
    if rgb_img.shape[2] == 4:
        rgb_img = rgb_img[:, :, :3]
    if np.isnan(rgb_img.any()) or len(rgb_img[rgb_img > 0]) == 0:
        print('empyt rgb image')
        exit(-1)
    color = np.clip(rgb_img, 0.0, 255.0) / 255.0
    # print(f'raw_rgb shape: {rgb_img.shape} color shape: {color.shape}, ')

    depth_img = np.expand_dims((depth_img / 1000.0), axis=2)
    # normalized_depth_img = depth_img / np.max(depth_img)
    pointcloud = depth_img * get_unit_spherical_map()

    o3d_pointcloud = o3d.geometry.PointCloud()
    o3d_pointcloud.points = o3d.utility.Vector3dVector(pointcloud.reshape(-1, 3))
    o3d_pointcloud.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3))
    # must constrain normals pointing towards camera
    o3d_pointcloud.estimate_normals()
    o3d_pointcloud.orient_normals_towards_camera_location(camera_location=(0, 0, 0))
    # remove outliers
    # cl, ind = o3d_pointcloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # display_inlier_outlier(o3d_pointcloud, ind)
    o3d.io.write_point_cloud(saved_color_pcl_filepath, o3d_pointcloud)
    return o3d_pointcloud



# Based on https://github.com/sunset1995/py360convert
class Equirec2Cube:
    def __init__(self, equ_h, equ_w, face_w):
        '''
        equ_h: int, height of the equirectangular image
        equ_w: int, width of the equirectangular image
        face_w: int, the length of each face of the cubemap
        '''

        self.equ_h = equ_h
        self.equ_w = equ_w
        self.face_w = face_w
        self.subview_poses = []

        self._xyzcube()
        self._xyz2coor()

        # For convert R-distance to Z-depth for CubeMaps
        cosmap = 1 / np.sqrt((2 * self.grid[..., 0]) ** 2 + (2 * self.grid[..., 1]) ** 2 + 1)
        self.cosmaps = np.concatenate(6 * [cosmap], axis=1)[..., np.newaxis]

    def _xyzcube(self):
        '''
        Compute the xyz cordinates of the unit cube in [F R B L U D] format.
        '''
        self.xyz = np.zeros((self.face_w, self.face_w * 6, 3), np.float32)
        rng = np.linspace(-0.5, 0.5, num=self.face_w, dtype=np.float32)
        self.grid = np.stack(np.meshgrid(rng, -rng), -1)

        # Front face (y = 0.5)
        self.xyz[:, 0 * self.face_w:1 * self.face_w, [0, 2]] = self.grid
        self.xyz[:, 0 * self.face_w:1 * self.face_w, 1] = 0.5
        T = np.eye(4)
        self.subview_poses.append(T)

        # Right face (x = 0.5)
        self.xyz[:, 1 * self.face_w:2 * self.face_w, [1, 2]] = self.grid[:, ::-1]
        self.xyz[:, 1 * self.face_w:2 * self.face_w, 0] = 0.5
        T = np.eye(4)
        T[:3,:3] = R.from_rotvec(np.pi/2 * np.array([0, 1, 0])).as_matrix()
        self.subview_poses.append(T)

        # Back face (y = -0.5)
        self.xyz[:, 2 * self.face_w:3 * self.face_w, [0, 2]] = self.grid[:, ::-1]
        self.xyz[:, 2 * self.face_w:3 * self.face_w, 1] = -0.5
        T = np.eye(4)
        T[:3,:3] = R.from_rotvec(np.pi * np.array([0, 1, 0])).as_matrix()
        self.subview_poses.append(T)

        # Left face (x = -0.5)
        self.xyz[:, 3 * self.face_w:4 * self.face_w, [1, 2]] = self.grid
        self.xyz[:, 3 * self.face_w:4 * self.face_w, 0] = -0.5
        T = np.eye(4)
        T[:3,:3] = R.from_rotvec(-np.pi/2 * np.array([0, 1, 0])).as_matrix()
        self.subview_poses.append(T)

        # Up face (z = 0.5)
        self.xyz[:, 4 * self.face_w:5 * self.face_w, [0, 1]] = self.grid[::-1, :]
        self.xyz[:, 4 * self.face_w:5 * self.face_w, 2] = 0.5
        T = np.eye(4)
        T[:3,:3] = R.from_rotvec(np.pi/2 * np.array([1, 0, 0])).as_matrix()
        self.subview_poses.append(T)

        # Down face (z = -0.5)
        self.xyz[:, 5 * self.face_w:6 * self.face_w, [0, 1]] = self.grid
        self.xyz[:, 5 * self.face_w:6 * self.face_w, 2] = -0.5
        T = np.eye(4)
        T[:3,:3] = R.from_rotvec(-np.pi/2 * np.array([1, 0, 0])).as_matrix()
        self.subview_poses.append(T)

        # # Front face (z = 0.5)
        # self.xyz[:, 0 * self.face_w:1 * self.face_w, [0, 1]] = self.grid
        # self.xyz[:, 0 * self.face_w:1 * self.face_w, 2] = 0.5
        # T = np.eye(4)
        # self.subview_poses.append(T)

        # # Right face (x = 0.5)
        # self.xyz[:, 1 * self.face_w:2 * self.face_w, [2, 1]] = self.grid[:, ::-1]
        # self.xyz[:, 1 * self.face_w:2 * self.face_w, 0] = 0.5
        # T[:3,:3] = R.from_rotvec(-np.pi/2 * np.array([0, 1, 0])).as_matrix()
        # self.subview_poses.append(T)

        # # Back face (z = -0.5)
        # self.xyz[:, 2 * self.face_w:3 * self.face_w, [0, 1]] = self.grid[:, ::-1]
        # self.xyz[:, 2 * self.face_w:3 * self.face_w, 2] = -0.5
        # T[:3,:3] = R.from_rotvec(-np.pi * np.array([0, 1, 0])).as_matrix()
        # self.subview_poses.append(T)

        # # Left face (x = -0.5)
        # self.xyz[:, 3 * self.face_w:4 * self.face_w, [2, 1]] = self.grid
        # self.xyz[:, 3 * self.face_w:4 * self.face_w, 0] = -0.5
        # T[:3,:3] = R.from_rotvec(np.pi/2 * np.array([0, 1, 0])).as_matrix()
        # self.subview_poses.append(T)

        # # Up face (y = 0.5)
        # self.xyz[:, 4 * self.face_w:5 * self.face_w, [0, 2]] = self.grid[::-1, :]
        # self.xyz[:, 4 * self.face_w:5 * self.face_w, 1] = 0.5
        # T[:3,:3] = R.from_rotvec(np.pi/2 * np.array([1, 0, 0])).as_matrix()
        # self.subview_poses.append(T)

        # # Down face (y = -0.5)
        # self.xyz[:, 5 * self.face_w:6 * self.face_w, [0, 2]] = self.grid
        # self.xyz[:, 5 * self.face_w:6 * self.face_w, 1] = -0.5
        # T[:3,:3] = R.from_rotvec(-np.pi/2 * np.array([1, 0, 0])).as_matrix()
        # self.subview_poses.append(T)

    def cubemap_poses(self):
        # return T_ref_subview
        assert len(self.subview_poses) == 6
        return np.array(self.subview_poses)

    def _xyz2coor(self):

        # x, y, z to longitude and latitude
        x, y, z = np.split(self.xyz, 3, axis=-1)
        lon = np.arctan2(x, y)
        c = np.sqrt(x ** 2 + y ** 2)
        lat = np.arctan2(z, c)

        # longitude and latitude to equirectangular coordinate
        self.coor_x = (lon / (2 * np.pi) + 0.5) * self.equ_w - 0.5
        self.coor_y = (-lat / np.pi + 0.5) * self.equ_h - 0.5

    def sample_equirec(self, e_img, order=0):
        pad_u = np.roll(e_img[[0]], self.equ_w // 2, axis=1)
        pad_d = np.roll(e_img[[-1]], self.equ_w // 2, axis=1)
        e_img = np.concatenate([e_img, pad_d, pad_u], 0)
        # pad_l = e_img[:, [0]]
        # pad_r = e_img[:, [-1]]
        # e_img = np.concatenate([e_img, pad_l, pad_r], 1)

        return map_coordinates(e_img, [self.coor_y, self.coor_x],
                               order=order, mode='wrap')[..., 0]

    def run(self, equ_img, equ_dep=None):

        h, w = equ_img.shape[:2]
        if h != self.equ_h or w != self.equ_w:
            equ_img = cv2.resize(equ_img, (self.equ_w, self.equ_h))
            if equ_dep is not None:
                equ_dep = cv2.resize(equ_dep, (self.equ_w, self.equ_h), interpolation=cv2.INTER_NEAREST)

        cube_img = np.stack([self.sample_equirec(equ_img[..., i], order=1)
                             for i in range(equ_img.shape[2])], axis=-1)

        if equ_dep is not None:
            cube_dep = np.stack([self.sample_equirec(equ_dep[..., i], order=0)
                                 for i in range(equ_dep.shape[2])], axis=-1)
            cube_dep = cube_dep * self.cosmaps

        if equ_dep is not None:
            return cube_img, cube_dep
        else:
            return cube_img