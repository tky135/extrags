import os
import shutil
import sys
sys.path.append(os.getcwd())
from loguru import logger
from typing import Union, Set, Dict, List, Any, Tuple, Optional, Iterator
import argparse
import loguru
import yaml
import json
import glob
import cv2
import shutil
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
import re
import joblib
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from pathlib import Path
from hdmap_base.e2dm.planet import Planet, Point, Vector
from scipy.spatial import cKDTree
from shapely.geometry import Polygon as sPolygon, Point as sPoint
from scipy import interpolate
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

import open3d as o3d
assert o3d.__version__ >= '0.14'

import pkg_resources
from data_warehouse.file_service import DataWareHouse
version = pkg_resources.get_distribution('data_warehouse_py').version
print("data warehouse version is ", version)

from worker_base.utils import read_json, write_json, remove
from copy import deepcopy
import ditu.topbind as tb
import ditu.utils as ditu_utils
import transforms3d

from worker_base.config import Config
from worker_base.worker import Worker
from worker_base.io import IO
import worker_base.geom_utils as geom_utils
import worker_base.utils as worker_utils
import requests
import time
from PIL import Image
version = pkg_resources.get_distribution('data_warehouse_py').version
print("data warehouse version is ", version)

DATA_WAREHOUSE_USERNAME='antbear-detection-mapping@proj'
DATA_WAREHOUSE_PASSWORD='defc0120'

def interpolate_pose(src_ts_list, src_qwc_list, src_center_list,
                    target_ts_list):
    '''
    src_ts_list: [float(timestamp), ...], 轨迹时间戳List
    src_qwc_list: [np.array([qx, qy, qz, qw]).reshape(1, 4), ...], 轨迹四元数List
    src_center_list: [np.array([x, y, z]).reshape(1, 3), ...], 轨迹位置List
    target_ts_list: [float(timestamp), ...], 待插值时间戳List

    返回： 插值后时间戳List, 四元数， 姿态。仅支持内插。     
    '''                 
    src_center_list = np.concatenate(src_center_list, axis=0)
    src_qwc_list = np.concatenate(src_qwc_list, axis=0)
    interp_f_center = interpolate.interp1d(
        src_ts_list, src_center_list, axis=0, fill_value='extrapolate')
    interp_f_qwc = Slerp(np.array(src_ts_list), R.from_quat(src_qwc_list))
    valid_ts_list = np.array(target_ts_list)
    valid_ts_list = valid_ts_list[np.logical_and(
        valid_ts_list > src_ts_list[0], valid_ts_list < src_ts_list[-1])]
    qwc_list = interp_f_qwc(valid_ts_list)
    center_list = interp_f_center(valid_ts_list)

    return valid_ts_list, qwc_list, center_list

def convert_array_to_pil(depth_map):
    # Input: depth_map -> HxW numpy array with depth values 
    # Output: colormapped_im -> HxW numpy array with colorcoded depth values
    mask = depth_map!=0
    disp_map = 1/depth_map
    vmax = np.percentile(disp_map[mask], 95)
    vmin = np.percentile(disp_map[mask], 5)
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    mask = np.repeat(np.expand_dims(mask,-1), 3, -1)
    colormapped_im = (mapper.to_rgba(disp_map)[:, :, :3] * 255).astype(np.uint8)
    colormapped_im[~mask] = 255
    return colormapped_im


def find_closest_timestep(timestamp, timestamp_list) -> int:
    # import ipdb; ipdb.set_trace()
    int_timestamp = np.array([int(val) for val in timestamp_list])
    return np.argmin(np.abs(int_timestamp - int(timestamp)))


def enu2lla(enus, anchor_lla=None):
    assert anchor_lla is not None
    T_ecef_enu = tb.utils.T_ecef_enu(
        lon=anchor_lla[0], lat=anchor_lla[1], alt=anchor_lla[2])
    ecefs = tb.utils.apply_transform(T_ecef_enu, enus)
    llas = tb.utils.ecef2lla(ecefs)
    return llas

def convert_ecef_to_pcd(lidar_loc_points_ecef, lidar_loc_points_colored, anchor_lla=None):
    # save pcd for visualize
    points_all_lla = tb.utils.ecef2lla(lidar_loc_points_ecef)
    center_lla = points_all_lla[0] if anchor_lla is None else anchor_lla
    points_all_enu = lla2enu(points_all_lla, anchor_lla=center_lla)[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_all_enu)
    pcd.colors = o3d.utility.Vector3dVector(lidar_loc_points_colored / 255.0)
    return pcd, anchor_lla

def lla2enu(llas, anchor_lla=None):
    assert anchor_lla is not None
    ecef = tb.utils.lla2ecef(llas)
    T_ecef_enu = tb.utils.T_ecef_enu(
        lon=anchor_lla[0], lat=anchor_lla[1], alt=anchor_lla[2])
    T_enu_ecef = np.linalg.inv(T_ecef_enu)
    enus = tb.utils.apply_transform(T_enu_ecef, ecef)
    return enus, T_enu_ecef

class Pipeline(Worker):

    def __init__(
            self,
            *,
            config: Config,
            # workdir
            workdir: Optional[str] = None,
            workdir_root: Optional[str] = None,
            # flags
            skip_downloading: bool = False,
            skip_uploading: bool = True,  # default not to upload
            skip_cleaning: bool = True,  # default not to clean workdir
            only_download_inputs: bool = False,
            only_download_outputs: bool = False,
    ):
        super().__init__(
            config=config,
            workdir=workdir,
            workdir_root=workdir_root,
            skip_downloading=skip_downloading,
            skip_uploading=skip_uploading,
            skip_cleaning=skip_cleaning,
            only_download_inputs=only_download_inputs,
            only_download_outputs=only_download_outputs,
        )

        self.camera_names_dict = {
            "camera_front_wide": "cameraF100",
            "camera_left_front": "cameraLF100",
            "camera_right_front": "cameraRF100",
            "camera_front_far": "cameraF30",
            "camera_left_rear": "cameraLR100",
            "camera_right_rear": "cameraRR100",
            "camera_rear_mid": "cameraR50"
        }

        self.camera_names_inverse_dict = {
            "cameraF100": "camera_front_wide",
            "cameraLF100": "camera_left_front",
            "cameraRF100": "camera_right_front",
            "cameraF30": "camera_front_far",
            "cameraLR100": "camera_left_rear",
            "cameraRR100":"camera_right_rear",
            "cameraR50": "camera_rear_mid"
        }

        self.camera_name_list = [
            "cameraF30",
            "cameraF100",
            "cameraLF100",
            "cameraRF100",
            "cameraLR100",
            "cameraRR100",
            "cameraR50"
        ]

        self.dw = DataWareHouse(print_mode=True, retry_time=5, username=DATA_WAREHOUSE_USERNAME, password=DATA_WAREHOUSE_PASSWORD, env=self.env)

        self._to_json_excludes = [
            'camera_names_dict', 'io', 'camera_calib_dict', 'dw'
        ]
        
        self.images_upload_url = "https://{}/api/v1/common-file/upload?storage_type=oss_beijing&bucket={}&file_path={}/{}/keyframes/{}/{}"
        self.parameter_upload_url = "https://{}/api/v1/common-file/upload?storage_type=oss_beijing&bucket={}&file_path={}/{}/parameter_undist.json"
        self.package_type_bucket_dict = {
            8: "worker-result-devcar-online",
            11: "worker-result-lacar-offline"
        }

    @property
    def io(self) -> IO:
        return IO(self.config, patch_data_warehouse=False)

    @property
    def task_type(self):
        return 'cvg'

    def download_loc_points(self):
        # download loc points
        self.lidar_points_medb_url = f'http://{self.config.data_warehouse}/api/v1/data-express/download/mesh?task_id={self.ddgt_task_id}&file_name='
        self.loc_points_url = f'http://{self.config.data_warehouse}/api/v1/data-express/download/mesh?task_id={self.ddgt_task_id}&file_name=lidar_loc_medb_file_names.json'
        output_dir = os.path.abspath(self.input_dir)
        lidar_loc_medb_file = f"{output_dir}/lidar_loc_medb_file_names.json"
        download_list = [{
            'uri' : self.loc_points_url,
            'relpath' : True,
            'symlink' : lidar_loc_medb_file
        }]
        self.io.download(
            workdir=self.input_dir,
            task_name='cvg_download_lidar_loc_medb_files_json',
            download_list=download_list
        )

        # read lidar_loc_medb_file_name
        with open(lidar_loc_medb_file, 'r') as f:
            lidar_loc_medb_file_info = json.load(f)
        
        lidar_output_dir = os.path.join(output_dir, "lidar_loc_medb")
        os.system(f"mkdir -p {lidar_output_dir}")
        lidar_download_list = []
        for file_path in lidar_loc_medb_file_info['file_names']:
            lidar_download_list.append({
                'uri' : self.lidar_points_medb_url + file_path,
                'relpath' : True,
                'symlink' : os.path.join(lidar_output_dir, file_path)
            })

        self.io.download(
            workdir=self.input_dir,
            task_name='cvg_download_lidar_loc_medb',
            download_list=lidar_download_list
        )

        # merge medb and convert medb to pcd
        points_all, colors_all = None, None
        for index, filename in enumerate(lidar_loc_medb_file_info['file_names']):
            medb_path = f"{self.input_dir}/lidar_loc_medb/{filename}"
            medb = tb.Medb(medb_path)
            if index == 0:
                points_all = medb.points()
                colors_all = medb.colors()
                continue 
            
            points_all = np.vstack((points_all, medb.points()))
            colors_all = np.vstack((colors_all, medb.colors()))

        self.lidar_loc_medb_path = f"{self.input_dir}/lidar_loc_medb/lidar_loc_points.medb"
        tb.Medb.dump_cloud(self.lidar_loc_medb_path, points=points_all, colors=colors_all)

        # points_all_lla = tb.utils.ecef2lla(points_all)
        # center_lla = points_all_lla[0]
        # points_all_enu = lla2enu(points_all_lla, anchor_lla=center_lla)[0]
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points_all_enu)
        # pcd.colors = o3d.utility.Vector3dVector(colors_all / 255.0)
        # o3d.io.write_point_cloud(f"{self.input_dir}/lidar_loc_medb/lidar_loc_points.pcd", pcd)

    def _download_inputs_impl(self):

        try:
            # download loc points
            self.download_loc_points()
        except:
            logger.info("download loc_points failed!!!")

        # download package data
        for idx in range(len(self.packagename_list)):
            packagename = self.packagename_list[idx]
            packagetype = self.packagetype_list[idx]

            # download package data
            package_data_path = os.path.join(self.input_dir.replace('input', 'package_data'), packagename)
            package_data_url = self.package_data_url.format(self.config.data_warehouse, packagename)
            self.dw.download_dir(package_data_path, package_data_url)

            # # download images, undist_images and segs
            # valid_camera_list = self.download_images_and_segs(packagename)

            # assert len(valid_camera_list) == 7, f"{packagename} dose not have 7 cams!!"

            # # download cloud_body
            # cloud_body_path = os.path.join(self.input_dir.replace('input', 'package_data'), packagename, 'cloud_body')
            # cloud_body_url = self.cloud_body_url.format(self.config.data_warehouse, packagename)
            # self.dw.download_dir(cloud_body_path, cloud_body_url)

    def preprocess_data(self):
        lidar_flag = os.path.exists(f"{self.input_dir}/lidar_loc_medb/lidar_loc_points.medb")
        if lidar_flag:
            # load lidar loc points
            lidar_loc_medb = tb.Medb(f"{self.input_dir}/lidar_loc_medb/lidar_loc_points.medb")
            lidar_loc_points_ecef = lidar_loc_medb.points()
            lidar_loc_colors = np.zeros_like(lidar_loc_points_ecef)
            lidar_loc_colors_count = np.zeros(len(lidar_loc_points_ecef))
            lidar_loc_labels = np.zeros((len(lidar_loc_points_ecef), 124))
        else:
            lidar_loc_points_ecef_tree, lidar_loc_points_ecef_road = [], []
            lidar_loc_points_colored_tree, lidar_loc_points_colored_road = [], []

        lidar_ts_offset = 0 # 16700

        # compute depth map
        for packagename in self.packagename_list:
            print("working on : ", packagename)
            package_data_root_path = os.path.join(self.input_dir.replace('input', 'package_data'), packagename)
            # load image_pose.txt.json
            image_pose_path = os.path.join(self.input_dir.replace('input', 'package_data'), packagename, "Image_Pose_new.txt.json")
            with open(image_pose_path, "r") as f:
                image_pose_info = json.load(f)
            
            image_pose_ts_list = []
            for feat in image_pose_info["features"]:
                image_pose_ts_list.append(feat["properties"]["timestamp"])
            # find shortest ts
            count = 1e8
            shortest_timestamp_list = []
            for cam_name in self.camera_name_list:
                images_list = glob.glob(os.path.join(self.input_dir.replace('input', 'package_data'), packagename, cam_name, 'keyframes', 'undist_images/*.jpg'))
                if count > len(images_list):
                    count = len(images_list)
                    # find ts
                    shortest_timestamp_list = []
                    for image_path in images_list:
                        shortest_timestamp_list.append(image_path.split('/')[-1].split('.')[0]) 
            # update Image_Pose_new
            image_pose_updated_dict = {"features": [], "type": "FeatureCollection"}
            for feat in image_pose_info["features"]:
                image_pose_ts = feat["properties"]["timestamp"]
                if image_pose_ts in shortest_timestamp_list:
                    image_pose_updated_dict["features"].append(feat)
            with open(os.path.join(self.input_dir.replace('input', 'package_data'), packagename, "Image_Pose_new_filtered.txt.json"), "w") as f:
                json.dump(image_pose_updated_dict, f, indent=4)
            image_pose_info = image_pose_updated_dict
            
            print("load egopose ...")
            egopose_ts_dict = {}
            src_ts_list, src_qwc_list, src_center_list = [], [], []
            for feat in image_pose_info["features"]:
                Rwc_quat_wxyz = feat["properties"]["extrinsic"]["Rwc_quat_wxyz"]
                rotation = R.from_quat(np.array(Rwc_quat_wxyz)[[1, 2, 3, 0]]).as_matrix()
                center = feat["properties"]["extrinsic"]["center"]
                egopose_wi = np.eye(4)
                egopose_wi[:3, :3] = rotation
                egopose_wi[:3, 3] = center

                timestamp = feat["properties"]["timestamp"]
                egopose_ts_dict[timestamp] = egopose_wi

                src_ts_list.append(int(timestamp))
                src_qwc_list.append(np.array(Rwc_quat_wxyz)[[1, 2, 3, 0]].reshape(1, 4))
                src_center_list.append(np.array(center).reshape(1, 3))

            lidar_points_dir = os.path.join(package_data_root_path, "cloud_body")
            lidar_points_pcd_list = glob.glob(os.path.join(lidar_points_dir, "*.pcd"))

            # find lidar timestamps
            lidar_timestamps = []
            for lidar_path in lidar_points_pcd_list:
                lidar_timestamps.append(int(lidar_path.split('/')[-1].split('.')[0]) - lidar_ts_offset)

            # interpolate ts
            print("interpolate egopose w.r.t lidar timestamp ...")
            valid_ts_list, qwc_list, center_list = interpolate_pose(src_ts_list, src_qwc_list, src_center_list, lidar_timestamps)
            interp_lidar_egopose_dict = {}
            for idx, t in enumerate(valid_ts_list):
                rotation = qwc_list[idx].as_matrix()
                center = np.array(center_list[idx])
                transformation = np.eye(4, dtype=np.float32)
                transformation[:3, :3] = rotation
                transformation[:3, 3] = center

                ts = str(valid_ts_list[idx])
                interp_lidar_egopose_dict[ts] = transformation
            lidar_timestamps = valid_ts_list

            # import ipdb; ipdb.set_trace()

            for cam_name in self.camera_name_list:
                print("working on camera name ", cam_name)
                depth_vis_dir = os.path.join(package_data_root_path, cam_name, "keyframes/depth_vis")
                os.system(f"mkdir -p {depth_vis_dir}")

                # load intrinsic, extrinsic
                with open(os.path.join(package_data_root_path, cam_name, "calibration.json"), 'r') as f:
                    calib_info = json.load(f)

                distort_coef = np.array(calib_info["DistCoef"])
                image_width, image_height = calib_info["image_width"], calib_info["image_height"]
                hoodline = int(calib_info["hood_line"])

                intrinsic = np.array(calib_info["camera_matrix"]["data"]).reshape(calib_info["camera_matrix"]["rows"], -1)
                cam_to_ego = np.linalg.inv(np.array(calib_info["Tci"]["data"]).reshape(calib_info["Tci"]["rows"], -1))

                timestamp_list = [str(ts) for ts in sorted(src_ts_list)]

                # compute depth
                depth_map_npy = {}
                for idx, ts in tqdm(enumerate(timestamp_list), total=len(timestamp_list), ncols=100, desc=f"[{cam_name}] computing depth maps"):
                    # load image
                    image_path = os.path.abspath(os.path.join(package_data_root_path, cam_name, f"keyframes/undist_images/{ts}.jpg"))
                    image = np.array(pil.open(image_path))

                    # load seg image
                    seg_image_path = os.path.abspath(os.path.join(package_data_root_path, cam_name, f"keyframes/seg_images/{ts}.png"))
                    label = np.array(pil.open(seg_image_path))

                    if lidar_flag:
                        # colorize lidar_loc_points_ecef
                        image_egopose_iw = np.linalg.inv(egopose_ts_dict[ts])
                        lidar_loc_points_imgtsbody = (image_egopose_iw[:3, :3] @ lidar_loc_points_ecef.T + image_egopose_iw[:3, 3:4]).T
                        ego_to_cam = np.linalg.inv(cam_to_ego)
                        lidar_loc_points_cam = (ego_to_cam[:3, :3] @ lidar_loc_points_imgtsbody.T + ego_to_cam[:3, 3:4]).T
                        lidar_loc_points_img = (intrinsic @ lidar_loc_points_cam.T).T
                        lidar_loc_depth = lidar_loc_points_img[:, 2]
                        loc_cam_points = lidar_loc_points_img[:, :2] / (lidar_loc_depth[:, None] + 1e-6)
                        valid_lidar_loc_mask = (
                            (loc_cam_points[:, 0] >= 0) & (loc_cam_points[:, 0] < image_width)
                            & (loc_cam_points[:, 1] >= 0) & (loc_cam_points[:, 1] < hoodline)
                            & (lidar_loc_depth > 0)
                        )
                        filtered_loc_cam_points = loc_cam_points[valid_lidar_loc_mask]
                        lidar_loc_colors[valid_lidar_loc_mask] += image[np.int32(filtered_loc_cam_points[:, 1]), np.int32(filtered_loc_cam_points[:, 0])]
                        lidar_loc_colors_count[valid_lidar_loc_mask] += 1
                        # lidar_loc_labels[valid_lidar_loc_mask][label[np.int32(filtered_loc_cam_points[:, 1]), np.int32(filtered_loc_cam_points[:, 0])]] += 1
                        lidar_loc_labels[np.arange(len(lidar_loc_labels))[valid_lidar_loc_mask], label[np.int32(filtered_loc_cam_points[:, 1]), np.int32(filtered_loc_cam_points[:, 0])]] += 1

                    # find correspoding lidar ts
                    lidar_ts_idx = find_closest_timestep(ts, lidar_timestamps)
                    lidar_ts = str(int(lidar_timestamps[lidar_ts_idx]))
                    lidar_egopose_wi = interp_lidar_egopose_dict[lidar_ts]

                    # load lidar pcd
                    lidar_pcd_path = os.path.join(package_data_root_path, f"cloud_body/{str(int(lidar_timestamps[lidar_ts_idx]) + lidar_ts_offset)}.pcd")
                    lidar_points = np.array(o3d.io.read_point_cloud(lidar_pcd_path).points)
                    lidar_points_ecef = (lidar_egopose_wi[:3, :3] @ lidar_points.T + lidar_egopose_wi[:3, 3:4]).T

                    # load image egopose
                    image_egopose_iw = np.linalg.inv(egopose_ts_dict[ts])
                    lidar_points_imgtsbody = (image_egopose_iw[:3, :3] @ lidar_points_ecef.T + image_egopose_iw[:3, 3:4]).T
                    ego_to_cam = np.linalg.inv(cam_to_ego)
                    lidar_points_cam = (ego_to_cam[:3, :3] @ lidar_points_imgtsbody.T + ego_to_cam[:3, 3:4]).T
                    lidar_points_img = (intrinsic @ lidar_points_cam.T).T

                    depth = lidar_points_img[:, 2]
                    cam_points = lidar_points_img[:, :2] / (depth[:, None] + 1e-6)
                    valid_mask = (
                        (cam_points[:, 0] >= 0) & (cam_points[:, 0] < image_width)
                        & (cam_points[:, 1] >= 0) & (cam_points[:, 1] < hoodline)
                        & (depth > 0)
                    )

                    depth = depth[valid_mask]
                    cam_points = cam_points[valid_mask]
                    depth_map = np.zeros((image_height, image_width))
                    depth_map[np.int32(cam_points[:, 1]), np.int32(cam_points[:, 0])] = depth
                    depth_mask = depth_map > 0
                    depth_map_npy[ts] = {
                        "cam_points": cam_points,
                        "depth": depth,
                        "image_width": image_width,
                        "image_height": image_height
                    }

                    if valid_mask.sum() == 0:
                        # import ipdb; ipdb.set_trace()
                        # load image
                        image_path = os.path.join(package_data_root_path, cam_name, f"keyframes/undist_images/{ts}.jpg")
                        image = np.array(pil.open(image_path))
                        merged_image = image * (1.0 - depth_mask[:, :, None])
                        pil.fromarray(np.uint8(merged_image)).save(os.path.join(depth_vis_dir, f"{ts}.jpg"))
                        continue

                    # colorize depth map
                    colored_depth_map = convert_array_to_pil(depth_map)

                    # merge image
                    merged_image = image * (1.0 - depth_mask[:, :, None]) + colored_depth_map * depth_mask[:, :, None]
                    pil.fromarray(np.uint8(merged_image)).save(os.path.join(depth_vis_dir, f"{ts}.jpg"))

                    if not lidar_flag:
                        # color lidar points
                        # filter ground points
                        # import ipdb; ipdb.set_trace()
                        lidar_points_labels = label[np.int32(cam_points[:, 1]), np.int32(cam_points[:, 0])]
                        road_mask = (lidar_points_labels == 21) | ((35 <= lidar_points_labels) & (lidar_points_labels <= 56)) | (lidar_points_labels == 13) | (lidar_points_labels == 14) | (lidar_points_labels == 74)
                        tree_mask = (1.0 - road_mask).astype(bool)
                        lidar_loc_points_ecef_tree.append(lidar_points_ecef[valid_mask][tree_mask])
                        lidar_loc_points_colored_tree.append(image[np.int32(cam_points[:, 1]), np.int32(cam_points[:, 0])][tree_mask])
                        lidar_loc_points_ecef_road.append(lidar_points_ecef[valid_mask][road_mask])
                        lidar_loc_points_colored_road.append(image[np.int32(cam_points[:, 1]), np.int32(cam_points[:, 0])][road_mask])

                np.save(os.path.join(package_data_root_path, cam_name, f"keyframes/{cam_name}_depthmap.npy"), depth_map_npy)

        if lidar_flag:
            # avg colors
            lidar_loc_colors /= (lidar_loc_colors_count[:, None] + 1e-8)
            lidar_loc_labels = np.argmax(lidar_loc_labels, axis=-1)

            tb.Medb.dump_cloud(f"{self.input_dir.replace('input', 'package_data')}/lidar_loc_points_colored.medb", points=lidar_loc_points_ecef, colors=lidar_loc_colors)

            # split road and tree points
            lidar_loc_points_roadmask = (lidar_loc_labels == 21) | ((35 <= lidar_loc_labels) & (lidar_loc_labels <= 56)) | (lidar_loc_labels == 13) | (lidar_loc_labels == 14) | (lidar_loc_labels == 74) 
            lidar_loc_points_ecef_road = lidar_loc_points_ecef[lidar_loc_points_roadmask]
            lidar_loc_points_road_colors = lidar_loc_colors[lidar_loc_points_roadmask]
            tb.Medb.dump_cloud(f"{self.input_dir.replace('input', 'package_data')}/lidar_loc_points_colored_road.medb", points=lidar_loc_points_ecef_road, colors=lidar_loc_points_road_colors)

            lidar_loc_points_treemask = (1.0 - lidar_loc_points_roadmask).astype(bool)
            lidar_loc_points_ecef_tree = lidar_loc_points_ecef[lidar_loc_points_treemask]
            lidar_loc_points_tree_colors = lidar_loc_colors[lidar_loc_points_treemask]
            tb.Medb.dump_cloud(f"{self.input_dir.replace('input', 'package_data')}/lidar_loc_points_colored_tree.medb", points=lidar_loc_points_ecef_tree, colors=lidar_loc_points_tree_colors)

            pcd_tree, anchor_lla = convert_ecef_to_pcd(lidar_loc_points_ecef_tree, lidar_loc_points_tree_colors, anchor_lla=None)
            o3d.io.write_point_cloud(f"{self.input_dir.replace('input', 'package_data')}/lidar_loc_points_colored_tree.pcd", pcd_tree)

            pcd_road, _ = convert_ecef_to_pcd(lidar_loc_points_ecef_road, lidar_loc_points_road_colors, anchor_lla=anchor_lla)
            o3d.io.write_point_cloud(f"{self.input_dir.replace('input', 'package_data')}/lidar_loc_points_colored_road.pcd", pcd_road)
        else:
            # import ipdb; ipdb.set_trace()
            lidar_loc_points_ecef_tree = np.concatenate(lidar_loc_points_ecef_tree)
            lidar_loc_points_colored_tree = np.concatenate(lidar_loc_points_colored_tree)
            lidar_loc_points_ecef_road = np.concatenate(lidar_loc_points_ecef_road)
            lidar_loc_points_colored_road = np.concatenate(lidar_loc_points_colored_road)
            lidar_loc_points_ecef = np.concatenate((lidar_loc_points_ecef_tree, lidar_loc_points_ecef_road), axis = 0)
            lidar_loc_colors = np.concatenate((lidar_loc_points_colored_tree, lidar_loc_points_colored_road), axis = 0)

            tb.Medb.dump_cloud(f"{self.input_dir.replace('input', 'package_data')}/lidar_loc_points_colored.medb", points=lidar_loc_points_ecef, colors=lidar_loc_colors)
            tb.Medb.dump_cloud(f"{self.input_dir.replace('input', 'package_data')}/lidar_loc_points_colored_tree.medb", points=lidar_loc_points_ecef_tree, colors=lidar_loc_points_colored_tree)
            tb.Medb.dump_cloud(f"{self.input_dir.replace('input', 'package_data')}/lidar_loc_points_colored_road.medb", points=lidar_loc_points_ecef_road, colors=lidar_loc_points_colored_road)

            pcd_tree, anchor_lla = convert_ecef_to_pcd(lidar_loc_points_ecef_tree, lidar_loc_points_colored_tree, anchor_lla=None)
            o3d.io.write_point_cloud(f"{self.input_dir.replace('input', 'package_data')}/lidar_loc_points_colored_tree.pcd", pcd_tree)

            pcd_road, _ = convert_ecef_to_pcd(lidar_loc_points_ecef_road, lidar_loc_points_colored_road, anchor_lla=anchor_lla)
            o3d.io.write_point_cloud(f"{self.input_dir.replace('input', 'package_data')}/lidar_loc_points_colored_road.pcd", pcd_road)


    def _upload_outputs_impl(self, *, only_export_upload_tasks: bool = False):
        '''
        sesf.upload_outputs()
        3. _upload_outputs_impl：上传任务输出
        https://momenta.feishu.cn/wiki/wikcnhmNU0qZ4Vunlb88CdOnAvb
        '''
        # import ipdb; ipdb.set_trace()
        # 0. upload url
        url_format = f"http://{self.config.data_warehouse}/api/v1/data-express/upload/mesh?task_id={self.ddgt_task_id}&file_name="
        
        # 1. upload pth
        upload_list = []
        pth_root_path = "/home/dylan.wu/data/Staging_test/omnire/debug_0923/mask_loss_0.5_7view"
        pth_path = os.path.join(pth_root_path, 'checkpoint_final.pth')
        upload_list.append({
            "url": url_format + "checkpoint_final.pth",
            "path": pth_path
        })

        # 2. upload Image Pose
        input_data_path = "/home/dylan.wu/data/Custom_data/ddld80417133-LYRIQ-087070-2023-08-21-17-54-56"
        image_pose_txt_json_path = os.path.join(input_data_path, 'Image_Pose.txt_cpy.json')
        upload_list.append({
            "url": url_format + f"{self.package_name}/Image_Pose.txt.json",
            "path": image_pose_txt_json_path
        })

        # 3. upload keyframes
        for camera_name in self.camera_names_inverse_dict:
            upload_list.append({
                "url": url_format + self.package_name + f"/{camera_name}_keyframes.json",
                "path": os.path.join(self.output_dir, self.package_name, f"{camera_name}_keyframes.json")
            })

        # 4. upload meta info
        upload_list.append({
                "url": url_format + "meta.json",
                "path": os.path.join(self.output_dir, "meta.json")
            })

        self.io.upload(
            workdir=self.output_dir, 
            task_name='final_upload',
            upload_list=upload_list,
            only_export=only_export_upload_tasks
        )

    def _download_outputs_impl(self):
        raise Exception('not implemented')

    def _consume_task_impl(self):
        '''
        self.consume_task()
        '''
        self.banner(' CONSUME_TASK ')
        logger.info(
            f'env: {self.env}, task_id: {self.task_id} (type:{type(self.task_id)}), task: {json.dumps(self.task, indent=4)}'
        )
        write_json(
            f'{self.input_dir}/task.json',
            {
                **self.task,
                'env': self.env,
            },
        )

        if self.task['package_type_list'][0] == 8:
            self.package_data_url = "https://{}/api/v1/common-file/download?storage_type=oss_beijing&bucket=worker-result-devcar-online&file_path={}"
            self.keyframe_info_url = "https://{}/api/v1/common-file/download?storage_type=oss_beijing&bucket=worker-result-devcar-online&file_path={}/{}/keyframes/all_key_frame.json"
            self.image_pose_url = "https://{}/api/v1/common-file/download?storage_type=oss_beijing&bucket=worker-result-devcar-online&file_path={}/Image_Pose_new.txt.json"
            self.images_url = "https://{}/api/v1/common-file/download?storage_type=oss_beijing&bucket=worker-result-devcar-online&file_path={}/{}/keyframes/all_key_frame.json"
            self.undist_images_url = "https://{}/api/v1/common-file/download?storage_type=oss_beijing&bucket=worker-result-devcar-online&file_path={}/{}/keyframes/undist_images/{}.jpg"
            self.seg_images_url = "https://{}/api/v1/common-file/download?storage_type=oss_beijing&bucket=worker-result-devcar-online&file_path={}/{}/keyframes/seg_images/{}.png"
            self.parameter_url = "https://{}/api/v1/common-file/download?storage_type=oss_beijing&bucket=worker-result-devcar-online&file_path={}/{}/parameter.json"
            self.calib_download_url = "https://{}/api/v1/common-file/download?storage_type=oss_beijing&bucket=worker-result-devcar-online&file_path={}/calibration/{}.yaml"
            self.cloud_body_url = "https://{}/api/v1/common-file/download?storage_type=oss_beijing&bucket=worker-result-devcar-online&file_path={}/cloud_body"
        elif self.task['package_type_list'][0] == 11:
            self.package_data_url = "https://{}/api/v1/common-file/download?storage_type=oss_beijing&bucket=worker-result-lacar-offline&file_path={}"
            self.keyframe_info_url = "https://{}/api/v1/common-file/download?storage_type=oss_beijing&bucket=worker-result-lacar-offline&file_path={}/{}/keyframes/all_key_frame.json"
            self.image_pose_url = "https://{}/api/v1/common-file/download?storage_type=oss_beijing&bucket=worker-result-lacar-online&file_path={}/Image_Pose_new.txt.json"
            self.images_url = "https://{}/api/v1/common-file/download?storage_type=oss_beijing&bucket=worker-result-lacar-offline&file_path={}/{}/keyframes/images/{}.jpg"
            self.undist_images_url = "https://{}/api/v1/common-file/download?storage_type=oss_beijing&bucket=worker-result-lacar-offline&file_path={}/{}/keyframes/undist_images/{}.jpg"
            self.seg_images_url = "https://{}/api/v1/common-file/download?storage_type=oss_beijing&bucket=worker-result-lacar-offline&file_path={}/{}/keyframes/seg_images/{}.png"
            self.parameter_url = "https://{}/api/v1/common-file/download?storage_type=oss_beijing&bucket=worker-result-lacar-offline&file_path={}/{}/parameter.json"
            self.calib_download_url = "https://{}/api/v1/common-file/download?storage_type=oss_beijing&bucket=worker-result-lacar-offline&file_path={}/calibration/{}.yaml"
            self.cloud_body_url = "https://{}/api/v1/common-file/download?storage_type=oss_beijing&bucket=worker-result-lacar-offline&file_path={}/cloud_body"
        else:
            print("unknown package type:", self.task['package_type'])


        self.ddgt_task_id = str(self.task.get('task_id')).split('_')[1]
        self.packagename_list = self.task.get("package_list", None)
        self.packagetype_list = self.task.get("package_type_list", None)
        if not self.skip_downloading:
            self.download_inputs()

        self.preprocess_data()

        # import ipdb; ipdb.set_trace()

        # start train jobs
        # data_root = os.path.join(self.input_dir.replace('input', ''))
        # output_root = self.output_dir
        # cmd = [
        #     'cd ./third_party/controllable-video-gen',
        #     '&&',
        #     f'./run.sh -r {data_root} -o {output_root} -i package_data'
        # ]

        # cmd = f'{" ".join(cmd)}'
        # logger.info(cmd)

        # os.system(cmd)

        # self.banner('61.upload_output')
        # self.upload_outputs()
        # self.take_snapshot('upload_outputs')

        self.banner('70.final')

        if self.config.debug:
            logger.error('debugging...')
            # TODO, extra export

        return {
            'task_timestamp': self.timers[0].start,
            'task_timestamp_readable': self.timers[0].Start,
        }


def main():
    Pipeline.main(
        sys.argv,
        prog='/usr/bin/python3.8 pipeline.py',
        description=('pipeline for cvg'),
    )


if __name__ == '__main__':
    main()
