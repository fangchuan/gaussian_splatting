import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline
from diffusers.utils import load_image
import os

import open3d as o3d
from scipy.spatial.transform import Rotation as R

def predict_depthes(image_folder: str):
    """
    :param image_folder: cubemap images folder

    :return:
    """
    # 1. load image
    # 2. predict depth
    # 3. save depth

    pipe = DiffusionPipeline.from_pretrained(
        "Bingxin/Marigold",
        custom_pipeline="marigold_depth_estimation"
        # torch_dtype=torch.float16,                # (optional) Run with half-precision (16-bit float).
    )

    pipe.to("cuda")

    # images_lst = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    images_lst = ['image_0_0.png', 'image_0_1.png', 'image_0_2.png', 'image_0_3.png', 'image_0_4.png', 'image_0_5.png']

    for image_name in images_lst:
        img_path_or_url = os.path.join(image_folder, image_name)
        image: Image.Image = load_image(img_path_or_url)

        pipeline_output = pipe(
            image,                  # Input image.
            # denoising_steps=10,     # (optional) Number of denoising steps of each inference pass. Default: 10.
            # ensemble_size=10,       # (optional) Number of inference passes in the ensemble. Default: 10.
            # processing_res=768,     # (optional) Maximum resolution of processing. If set to 0: will not resize at all. Defaults to 768.
            # match_input_res=True,   # (optional) Resize depth prediction to match input resolution.
            # batch_size=0,           # (optional) Inference batch size, no bigger than `num_ensemble`. If set to 0, the script will automatically decide the proper batch size. Defaults to 0.
            # color_map="Spectral",   # (optional) Colormap used to colorize the depth map. Defaults to "Spectral".
            # show_progress_bar=True, # (optional) If true, will show progress bars of the inference progress.
        )

        depth: np.ndarray = pipeline_output.depth_np                    # Predicted depth map
        depth_colored: Image.Image = pipeline_output.depth_colored      # Colorized prediction

        # Save as uint16 PNG
        depth_uint16 = (depth * 65535.0).astype(np.uint16)
        saved_depth_path = os.path.join(image_folder, "{}_depth.png".format(image_name.split(".")[0]))
        Image.fromarray(depth_uint16).save(saved_depth_path, mode="I;16")

        # Save colorized depth map
        # depth_colored.save("./depth_colored.png")


def unproject_depth(depth_img_filepath:str, hfov:float, WIDTH:int):
    # hfov = 90.0 * np.pi / 180.0
    K = np.array([
        [1 / np.tan(hfov / 2.), 0., 0., 0.],
        [0., 1 / np.tan(hfov / 2.), 0., 0.],
        [0., 0.,  1, 0],
        [0., 0., 0, 1]])
    
    depth_img = np.asarray(Image.open(depth_img_filepath))
    depth_img = depth_img.astype(np.float32)
    depth_img = np.expand_dims((depth_img/65535.0).astype(np.float32),axis=2)
    # Now get an approximation for the true world coordinates -- see if they make sense
    # [-1, 1] for x and [1, -1] for y as array indexing is y-down while world is y-up
    xs, ys = np.meshgrid(np.linspace(-1,1,WIDTH), np.linspace(-1,1,WIDTH))
    depth = depth_img.reshape(1,WIDTH,WIDTH)
    xs = xs.reshape(1,WIDTH,WIDTH)
    ys = ys.reshape(1,WIDTH,WIDTH)

    # Unproject
    # negate depth as the camera looks along -Z
    # positive depth as the camera looks along Z
    xys = np.vstack((xs * depth , ys * depth, depth, np.ones(depth.shape)))
    xys = xys.reshape(4, -1)
    xy_c0 = np.matmul(np.linalg.inv(K), xys)
    # xys_T = np.transpose(xys)

    return xy_c0

def reconstruct_pointcloud(image_folder:str):
    # images_lst = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f)) and f.endswith("_depth.png")]
    images_lst = ['image_0_0.png', 'image_0_1.png', 'image_0_2.png', 'image_0_3.png', 'image_0_4.png', 'image_0_5.png']
    # images_lst.sort(key=lambda x: int(x.split("_")[1]))
    total_pointcloud = o3d.geometry.PointCloud()
    for image_name in images_lst:
        rgb_img_path = os.path.join(image_folder, image_name)
        depth_img_path = os.path.join(image_folder, "{}_depth.png".format(image_name.split(".")[0]))
        rgb_img = np.asarray(Image.open(rgb_img_path))
        rgb_img = rgb_img.astype(np.float32) / 255.0
        subview_pcl = unproject_depth(depth_img_path, hfov=90.0 * np.pi / 180.0, WIDTH=256)

        subview_idx = int(image_name.split("_")[2].split(".")[0])
        # chose front as reference view
        T_ref_subview = np.eye(4)
        if subview_idx == 1:
            T_ref_subview[:3, :3] = R.from_rotvec(np.pi/2 * np.array([0, 1, 0])).as_matrix()
        elif subview_idx == 2:
            T_ref_subview[:3, :3] = R.from_rotvec(np.pi * np.array([0, 1, 0])).as_matrix()
        elif subview_idx == 3:
            T_ref_subview[:3, :3] = R.from_rotvec(-np.pi/2 * np.array([0, 1, 0])).as_matrix()
        elif subview_idx == 4:
            T_ref_subview[:3, :3] = R.from_rotvec(np.pi/2 * np.array([1, 0, 0])).as_matrix()
        elif subview_idx == 5:
            T_ref_subview[:3, :3] = R.from_rotvec(-np.pi/2 * np.array([1, 0, 0])).as_matrix()
        subview_pointcloud = T_ref_subview @ subview_pcl

        subview_pointcloud_T = np.transpose(subview_pointcloud)
        o3d_pointcloud = o3d.geometry.PointCloud()
        o3d_pointcloud.points = o3d.utility.Vector3dVector(subview_pointcloud_T[:,:3])
        o3d_pointcloud.colors = o3d.utility.Vector3dVector(rgb_img.reshape(-1, 3))
        o3d.io.write_point_cloud(os.path.join(image_folder, "{}_points.ply".format(image_name.split(".")[0])), o3d_pointcloud)

        total_pointcloud += o3d_pointcloud


    # v_pointclouds = np.concatenate((v_pointclouds), axis=1)
    # v_pointclouds_T = np.transpose(v_pointclouds)
    # o3d_pointcloud = o3d.geometry.PointCloud()
    # o3d_pointcloud.points = o3d.utility.Vector3dVector(v_pointclouds_T[:,:3])
    o3d.io.write_point_cloud(os.path.join(image_folder, "image_0.ply"), total_pointcloud)



if __name__ == "__main__":
    image_folder = "/mnt/nas_3dv/hdd1/datasets/Structured3d/SPGS_2/scene_00000_492165/cubemap_images"
    # saved_depth_folder = "/mnt/nas_3dv/hdd1/datasets/Structured3d/SPGS_2/scene_00000_492165/depths"
    # predict_depthes(image_folder)
    reconstruct_pointcloud(image_folder)

    # rotate raw pointcloud around x axis
    raw_pcl_path = "/mnt/nas_3dv/hdd1/datasets/Structured3d/SPGS_2/scene_00000_492165/points3d.ply"
    o3d_raw_pcl = o3d.io.read_point_cloud(raw_pcl_path)
    R_raw_cam = R.from_rotvec(-np.pi/2 * np.array([1, 0, 0])).as_matrix()
    R_cam_raw = R_raw_cam.transpose()
    raw_points = o3d_raw_pcl.points
    raw_points = np.asarray(raw_points)
    raw_points = np.transpose(raw_points)
    print(f'Rotation matrix: {R_cam_raw}')
    points_in_cam = R_cam_raw @ raw_points
    o3d_raw_pcl.points = o3d.utility.Vector3dVector(np.transpose(points_in_cam))
    o3d.io.write_point_cloud(os.path.join(image_folder, "raw_points.ply"), o3d_raw_pcl)

    print("Done!")