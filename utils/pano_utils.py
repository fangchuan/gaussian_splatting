import os
import sys

import numpy as np
import cv2
import open3d as o3d

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
        Y = np.expand_dims(np.sin(vs), 2)
        Z = np.expand_dims(np.cos(vs) * np.cos(us), 2)
        unit_map = np.concatenate([X, Z, Y], axis=2)

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

