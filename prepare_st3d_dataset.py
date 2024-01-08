import os
import sys
sys.path.append('.')
sys.path.append('..')

import argparse
import shutil
from tqdm import tqdm
from glob import glob

import numpy as np
from utils.pano_utils import vis_color_pointcloud

def read_PNVS_split_file(split_file:str):
    splits = {}
    with open(split_file, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            # print(line.strip())
            data = line.strip().split(" ")
            # print(data)
            assert len(data) == 3, "Invalid split file format!"
            # splits.append(data[0]+"_"+data[1]+"_"+data[2])
            room_name = data[0]+"_"+data[1]
            if room_name not in splits:
                splits[room_name] = []
                splits[room_name].append(int(data[2]))
            else:
                splits[room_name].append(int(data[2]))
    return splits

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
def process_dataset(raw_dataset_dir:str, output_dir:str, is_verbose:bool):
    src_target_map_dict = {}

    splits = ['easy', 'hard']

    for split in splits:
        dataset_dir = os.path.join(raw_dataset_dir, split)            
        source_view_folder = os.path.join(dataset_dir, "source_image")
        source_depth_folder = os.path.join(dataset_dir, "source_depth")
        source_cam_folder = os.path.join(dataset_dir, "source_camera")
        source_layout_folder = os.path.join(dataset_dir, "source_layout")
        target_view_folder = os.path.join(dataset_dir, "target_image")
        target_cam_folder = os.path.join(dataset_dir, "target_camera")

        train_split_file = os.path.join(dataset_dir, "train.txt")
        test_split_file = os.path.join(dataset_dir, "val.txt")
        train_splits = read_PNVS_split_file(train_split_file)
        test_splits = read_PNVS_split_file(test_split_file)

        src_rgb_img_lst = [ fname for fname in os.listdir(source_view_folder) if fname.endswith(".png") ]
        src_rgb_img_lst.sort()
        src_img_name_lst = [ fname.split(".")[0] for fname in src_rgb_img_lst ]
        
        for img_name in tqdm(src_img_name_lst):
            src_rgb_img_path = os.path.join(source_view_folder, img_name + ".png")
            src_depth_img_path = os.path.join(source_depth_folder, img_name + ".png")
            src_cam_path = os.path.join(source_cam_folder, img_name + ".txt")
            src_layout_path = os.path.join(source_layout_folder, img_name + ".txt")
            tgt_rgb_img_path_lst = glob(os.path.join(target_view_folder, img_name + "_*.png"))
            tgt_cam_path_lst = glob(os.path.join(target_cam_folder, img_name + "_*.txt"))
            assert len(tgt_rgb_img_path_lst) == len(tgt_cam_path_lst), "Number of target images and cameras are not equal!"

            scene_output_dir = os.path.join(output_dir, img_name)
            if not os.path.exists(scene_output_dir):
                os.makedirs(scene_output_dir)

            # load camera pose
            original_cam_center = np.loadtxt(src_cam_path)
            original_cam_center = original_cam_center[:3] * 0.001
            # cameras_lst = []

            saved_src_img_path = os.path.join(scene_output_dir, "image_0.png")
            saved_src_depth_path = os.path.join(scene_output_dir, "depth_0.png")
            saved_cams_path = os.path.join(scene_output_dir, "cameras.txt")
            saved_src_layout_path = os.path.join(scene_output_dir, "layout.txt")
            if not os.path.exists(saved_src_img_path) or not os.path.exists(saved_src_depth_path) or not os.path.exists(saved_src_layout_path):
                shutil.copyfile(src_rgb_img_path, saved_src_img_path)
                shutil.copyfile(src_depth_img_path, saved_src_depth_path)
                # np.savetxt(saved_src_cam_path, np.zeros((3, )))
                # cameras_lst.append(np.zeros((3, )))
                with open(saved_cams_path, 'a') as f:
                    f.write("0.0 0.0 0.0\n")
                shutil.copyfile(src_layout_path, saved_src_layout_path)

            saved_ply_path = os.path.join(scene_output_dir, "points3d.ply")
            # get scene point cloud
            pointcloud = vis_color_pointcloud(rgb_img_filepath=src_rgb_img_path, depth_img_filepath=src_depth_img_path, saved_color_pcl_filepath=saved_ply_path)

            for tgt_img in tgt_rgb_img_path_lst:
                if img_name not in src_target_map_dict:
                    src_target_map_dict[img_name] = 1
                else:
                    src_target_map_dict[img_name] += 1
                
                tgt_img_name = os.path.basename(tgt_img).split(".")[0]
                tgt_img_id = int(tgt_img_name.split("_")[-1])
                tgt_cam_path = os.path.join(target_cam_folder, tgt_img_name + ".txt")
                saved_tgt_img_path = os.path.join(scene_output_dir, "image_%d.png" % src_target_map_dict[img_name])
                # saved_tgt_cam_path = os.path.join(scene_output_dir, "target_camera_%d.txt" % src_target_map_dict[img_name])

                # load camera pose
                tgt_cam_center = np.loadtxt(tgt_cam_path)
                tgt_cam_center = tgt_cam_center[:3] * 0.001
                tgt_cam_center = tgt_cam_center - original_cam_center

                shutil.copyfile(tgt_img, saved_tgt_img_path)
                # np.savetxt(saved_tgt_cam_path, tgt_cam_center)
                # cameras_lst.append(tgt_cam_center)
                with open(saved_cams_path, 'a') as f:
                    f.write("%f %f %f\n" % (tgt_cam_center[0], tgt_cam_center[1], tgt_cam_center[2]))

                # train/test split for this room
                new_train_split_file = os.path.join(scene_output_dir, "train.txt")
                new_test_split_file = os.path.join(scene_output_dir, "test.txt")
                if img_name == 'scene_00000_492165':
                    print(f'train_splits: {train_splits[img_name]}, target_id: {tgt_img_id}')
                if img_name in train_splits:
                    if tgt_img_id in train_splits[img_name]:
                        with open(new_train_split_file, 'a') as f:
                            f.write("image_%d.png\n" % src_target_map_dict[img_name])
                if img_name in test_splits:
                    if tgt_img_id in test_splits[img_name]:
                        with open(new_test_split_file, 'a') as f:
                            f.write("image_%d.png\n" % src_target_map_dict[img_name])
            
            # # save camera centers
            # saved_cam_path = os.path.join(scene_output_dir, "cameras.txt")
            # with open(saved_cam_path, 'w') as f:
            #     for cam in cameras_lst:
            #         f.write("%f %f %f\n" % (cam[0], cam[1], cam[2]))

    # print averaging number of target views in each scene
    print("Number of scenes: %d" % len(src_target_map_dict.keys()))
    print("Average number of target views in each scene: %f" % (sum(src_target_map_dict.values()) / len(src_target_map_dict.keys())))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="/mnt/nas_3dv/hdd1/datasets/Structured3d/PNVS/")
    parser.add_argument("--output_path", default="/mnt/nas_3dv/hdd1/datasets/Structured3d/SPGS_2/")
    parser.add_argument("--verbose", default=True, type=bool)
    args = parser.parse_args()

    raw_dataset_dir = args.dataset_path
    output_dir = args.output_path
    is_verbose = args.verbose

    if not os.path.exists(raw_dataset_dir):
        print(f"Input directory {raw_dataset_dir} doesn't exists!")
        exit(-1)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading Indoor Scene")
    process_dataset(raw_dataset_dir, output_dir, is_verbose)
#    for scene_id in tqdm(scene_ids):

#         if scene_id in INVALID_SCENES_LST:
#             continue

#         room_type_lst = None
#         # parse scene annotation
#         scene_anno_3d_filepath = os.path.join(raw_dataset_dir, scene_id, 'annotation_3d.json')
#         if not os.path.isfile(scene_anno_3d_filepath):
#             INVALID_SCENES_LST.append(scene_id)
#             continue
#         else:
#             scene_anno_3d_dict = json.load(open(scene_anno_3d_filepath, 'r'))
#             room_type_lst = scene_anno_3d_dict['semantics']

#         scene_dir = os.path.join(raw_dataset_dir, scene_id, '2D_rendering')
#         for room_id in np.sort(os.listdir(scene_dir)):

#             room_str = '%s_%s' % (scene_id, room_id)
#             if room_str in INVALID_ROOMS_LST:
#                 continue
#             room_type_str = 'undefined'
#             if room_type_lst is not None:
#                 for rt in room_type_lst:
#                     if rt['ID'] == int(room_id):
#                         room_type_str = rt['type']
#                         break

#             # print(f'Processing room: {room_str}')

#             room_path = os.path.join(scene_dir, room_id, "panorama")
#             source_img_path = os.path.join(room_path, "full", "rgb_rawlight.png")
#             source_cor_path = os.path.join(room_path, "layout.txt")
#             source_cam_pos_path = os.path.join(room_path, "camera_xyz.txt")
