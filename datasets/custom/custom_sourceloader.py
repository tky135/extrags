from typing import Dict
import logging
import os
import json
import joblib
import glob
import numpy as np
from tqdm import trange, tqdm
from PIL import Image
from scipy import interpolate
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from omegaconf import OmegaConf
import open3d as o3d
import cv2
from scipy.spatial import cKDTree

import torch
from torch import Tensor

from pytorch3d.transforms import matrix_to_quaternion
from datasets.base.scene_dataset import ModelType
from datasets.base.lidar_source import SceneLidarSource
from datasets.base.pixel_source import ScenePixelSource, CameraData

logger = logging.getLogger()

# define each class's node type
OBJECT_CLASS_NODE_MAPPING = {
    "Vehicle": ModelType.RigidNodes,
    "Pedestrian": ModelType.SMPLNodes,
    "Cyclist": ModelType.DeformableNodes
}
SMPLNODE_CLASSES = ["Pedestrian"]

# OpenCV to Dataset coordinate transformation
# opencv coordinate system: x right, y down, z front
# waymo coordinate system: x front, y left, z up
OPENCV2DATASET = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
)

# Waymo Camera List:
# 0: front_camera
# 1: front_left_camera
# 2: front_right_camera
# 3: left_camera
# 4: right_camera
AVAILABLE_CAM_LIST = [0, 1, 2, 3, 4]

class CustomCameraData(CameraData):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_all_filelist(self):
        """
        Create file lists for all data files.
        e.g., img files, feature files, etc.
        """
        # ---- define filepaths ---- #
        img_filepaths = []
        dynamic_mask_filepaths, sky_mask_filepaths = [], []
        human_mask_filepaths, vehicle_mask_filepaths = [], []
        road_mask_filepaths = []

        # Note: we assume all the files in waymo dataset are synchronized
        for t in self.timestamp_list:
            img_filepaths.append(
                os.path.join(self.data_path, self.cam_name, 'keyframes', 'undist_images', f"{t}.jpg")
            )
            dynamic_mask_filepaths.append(
                os.path.join(
                    os.path.join(self.data_path, self.cam_name, 'keyframes', 'seg_images', f"{t}.png")
                )
            )
            human_mask_filepaths.append(
                os.path.join(
                    os.path.join(self.data_path, self.cam_name, 'keyframes', 'seg_images', f"{t}.png")
                )
            )
            vehicle_mask_filepaths.append(
                os.path.join(
                    os.path.join(self.data_path, self.cam_name, 'keyframes', 'seg_images', f"{t}.png")
                )
            )
            sky_mask_filepaths.append(
                os.path.join(self.data_path, self.cam_name, 'keyframes', 'seg_images', f"{t}.png")
            )
            road_mask_filepaths.append(
                os.path.join(self.data_path, self.cam_name, 'keyframes', 'seg_images', f"{t}.png")
            )

        self.img_filepaths = np.array(img_filepaths)
        self.dynamic_mask_filepaths = np.array(dynamic_mask_filepaths)
        self.human_mask_filepaths = np.array(human_mask_filepaths)
        self.vehicle_mask_filepaths = np.array(vehicle_mask_filepaths)
        self.sky_mask_filepaths = np.array(sky_mask_filepaths)
        self.road_mask_filepaths = np.array(road_mask_filepaths)

        with open(os.path.join(self.data_path, self.cam_name, "calibration.json")) as f:
            calib_info = json.load(f)
        
        self.cam_height = calib_info['ground_vec'][1]

    def load_egocar_mask(self):
        # compute egocar mask from hoodline
        # try:
        height, width = self.calib_info['image_height'], self.calib_info['image_width']
        egocar_mask_np = np.zeros((height, width))
        hoodline = int(self.calib_info['hood_line'])
        self.hoodline = int(hoodline)
        egocar_mask_np[hoodline:,...] = 255
        egocar_mask = Image.fromarray(egocar_mask_np).convert("L").resize(
            (self.load_size[1], self.load_size[0]), Image.BILINEAR
        )
        self.egocar_mask = torch.from_numpy(np.array(egocar_mask) > 0).float()
        # except:
        #     self.egocar_mask = None

    def load_dynamic_masks(self):
        dynamic_masks = []
        for ix, fname in tqdm(enumerate(self.dynamic_mask_filepaths), desc="Loading dynamic masks",
            dynamic_ncols=True, total=len(self.dynamic_mask_filepaths),
        ):
            label = np.asarray(Image.open(fname))
            dyn_mask = ((0 <= label) & (label <= 1)) | ((32 <= label) & (label <= 34)) | (105 == label) | ((109 <= label) & (label <= 120))
            dyn_mask = Image.fromarray(dyn_mask * 255.).convert('L')
            dyn_mask = dyn_mask.resize(
                (self.load_size[1], self.load_size[0]), Image.BILINEAR
            )
            dynamic_masks.append(np.array(dyn_mask) > 0)
        self.dynamic_masks = torch.from_numpy(np.stack(dynamic_masks, axis=0)).float()

        human_masks = []
        for ix, fname in tqdm(enumerate(self.human_mask_filepaths), desc="Loading human masks",
            dynamic_ncols=True, total=len(self.human_mask_filepaths),
        ):
            label = np.asarray(Image.open(fname))
            human_mask = ((30 <= label) & (label <= 31)) 
            human_mask = Image.fromarray(human_mask * 255.).convert('L')
            human_mask = human_mask.resize(
                (self.load_size[1], self.load_size[0]), Image.BILINEAR
            )
            human_masks.append(np.array(human_mask) > 0)
        self.human_masks = torch.from_numpy(np.stack(human_masks, axis=0)).float()

        vehicle_masks = []
        for ix, fname in tqdm(enumerate(self.vehicle_mask_filepaths), desc="Loading vehicle masks",
            dynamic_ncols=True, total=len(self.vehicle_mask_filepaths),
        ):
            label = np.asarray(Image.open(fname))
            vehicle_mask = ((106 <= label) & (label <= 110)) | ((112 <= label) & (label <= 115))
            vehicle_mask = Image.fromarray(vehicle_mask * 255.).convert('L')
            vehicle_mask = vehicle_mask.resize(
                (self.load_size[1], self.load_size[0]), Image.BILINEAR
            )
            vehicle_masks.append(np.array(vehicle_mask) > 0)
        self.vehicle_masks = torch.from_numpy(np.stack(vehicle_masks, axis=0)).float()

    def load_sky_masks(self):
        sky_masks = []
        for ix, fname in tqdm(enumerate(self.sky_mask_filepaths), desc="Loading sky masks",
            dynamic_ncols=True, total=len(self.sky_mask_filepaths),
        ):
            label = np.asarray(Image.open(fname))
            sky_mask = label == 61
            sky_mask = Image.fromarray(sky_mask * 255.).convert('L')
            sky_mask = sky_mask.resize(
                (self.load_size[1], self.load_size[0]), Image.BILINEAR
            )
            sky_masks.append(np.array(sky_mask) > 0)
        self.sky_masks = torch.from_numpy(np.stack(sky_masks, axis=0)).float()
    
    def load_road_masks(self):
        road_masks = []
        for idx, fname in tqdm(enumerate(self.road_mask_filepaths), desc="Loading road masks",
            dynamic_ncols=True, total=len(self.road_mask_filepaths),
        ):
            label = np.asarray(Image.open(fname))
            road_mask = (label == 21) | ((35 <= label) & (label <= 56)) | (label == 13) | (label == 14)
            road_mask = Image.fromarray(road_mask * 255.).convert('L')
            road_mask = road_mask.resize(
                (self.load_size[1], self.load_size[0]), Image.BILINEAR
            )
            road_masks.append(np.array(road_mask) > 0)
        self.road_masks = torch.from_numpy(np.stack(road_masks, axis=0)).float()
        
    def load_calibrations(self):
        """
        Load the camera intrinsics, extrinsics, timestamps, etc.
        Compute the camera-to-world matrices, ego-to-world matrices, etc.
        """
        # load camera intrinsics
        # 1d Array of [f_u, f_v, c_u, c_v, k{1, 2}, p{1, 2}, k{3}].
        # ====!! we did not use distortion parameters for simplicity !!====
        # to be improved!!
        # load calibration.json
        with open(os.path.join(self.data_path, self.cam_name, "calibration.json")) as f:
            calib_info = json.load(f)

        distort_coef = np.array(calib_info["DistCoef"])
        image_width, image_height = calib_info["image_width"], calib_info["image_height"]

        intrinsic = np.array(calib_info['camera_matrix']['data']).reshape(calib_info['camera_matrix']['rows'], -1)
        # intrinsic = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(intrinsic, distort_coef, (image_width,image_height), None)
        
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        k1, k2, p1, p2, k3 = 0, 0, 0, 0, 0

        # scale intrinsics w.r.t. load size
        fx, fy = (
            fx * self.load_size[1] / self.original_size[1],
            fy * self.load_size[0] / self.original_size[0],
        )
        cx, cy = (
            cx * self.load_size[1] / self.original_size[1],
            cy * self.load_size[0] / self.original_size[0],
        )
        _intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        _distortions = np.array([k1, k2, p1, p2, k3])

        # load camera extrinsics
        cam_to_ego = np.linalg.inv(np.array(calib_info['Tci']['data']).reshape(calib_info['Tci']['rows'], -1))

        # because we use opencv coordinate system to generate camera rays,
        # we need a transformation matrix to covnert rays from opencv coordinate
        # system to waymo coordinate system.
        # opencv coordinate system: x right, y down, z front
        # waymo coordinate system: x front, y left, z up
        cam_to_ego = cam_to_ego # @ OPENCV2DATASET

        # compute per-image poses and intrinsics
        cam_to_worlds, ego_to_worlds = [], []
        intrinsics, distortions = [], []

        # we tranform the camera poses w.r.t. the first timestep to make the translation vector of
        # the first ego pose as the origin of the world coordinate system.
        # ego_to_world_start = np.loadtxt(
        #     os.path.join(self.data_path, "ego_pose", f"{self.start_timestep:03d}.txt")
        # )

        with open(os.path.join(self.data_path, 'Image_Pose.txt.json.from_lidar'), 'r') as f:
            image_pose_info = json.load(f)
        # json_path = os.path.join(self.data_path, 'lidar_pose.json')
        self.egopose_ts_dict = {}
        # with open(json_path, 'r') as f:
        #     for line in tqdm(f):
        #         info = json.loads(line)
        #         ts = float(info['timestamp_us'])
        #         position = info['position']
        #         Rwc_quat_wxyz = info['orientation']
        #         rotation = R.from_quat(np.array(Rwc_quat_wxyz)[[1, 2, 3, 0]]).as_matrix()
        #         ego_pose_wi = np.eye(4)
        #         ego_pose_wi[:3, :3] = rotation
        #         ego_pose_wi[:3, 3] = position
        #         self.egopose_ts_dict[ts] = ego_pose_wi
        # ego_pose_wi_start = self.egopose_ts_dict[min_timestamp]
        # lidar_to_worlds = {}
        # src_qwc_list, src_center_list = [], []

        # for t in self.egopose_ts_dict:
        #     ego_pose_wi_current = self.egopose_ts_dict[t]
        #     # compute ego_to_world transformation
        #     lidar_to_world = np.linalg.inv(ego_pose_wi_start) @ ego_pose_wi_current
        #     # lidar_to_worlds.append(lidar_to_world)
        #     lidar_to_worlds[t] = torch.from_numpy(lidar_to_world).float()

        #     qwc = R.from_matrix(lidar_to_world[:3, :3]).as_quat().reshape(1, 4)
        #     center = lidar_to_world[:3, 3].reshape(1, 3)
        #     src_qwc_list.append(qwc)
        #     src_center_list.append(center)

        # # interpolate ts
        # src_ts_list = [int(val) for val in list(lidar_to_worlds.keys())]
        # target_ts_list = [int(val) for val in self.timestamp_list]
        # valid_ts_list, qwc_list, center_list = self.interpolate_pose(
        #     src_ts_list,
        #     src_qwc_list,
        #     src_center_list,
        #     target_ts_list
        # )

        # interp_lidar_to_worlds = {}
        # for idx, t in enumerate(valid_ts_list):
        #     rotation = qwc_list[idx].as_matrix()
        #     center = np.array(center_list[idx])
        #     transformation = np.eye(4, dtype=np.float32)
        #     transformation[:3, :3] = rotation
        #     transformation[:3, 3] = center

        #     ts = str(valid_ts_list[idx])
        #     interp_lidar_to_worlds[ts] = torch.from_numpy(transformation).float()
        for feat in image_pose_info["features"]:
            Rwc_quat_wxyz = feat["properties"]["extrinsic"]["Rwc_quat_wxyz"]
            rotation = R.from_quat(np.array(Rwc_quat_wxyz)[[1, 2, 3, 0]]).as_matrix()
            center = feat['properties']['extrinsic']['center']
            ego_pose_wi = np.eye(4)
            ego_pose_wi[:3, :3] = rotation
            ego_pose_wi[:3, 3] = center

            timestamp = feat["properties"]["timestamp"]
            self.egopose_ts_dict[timestamp] = ego_pose_wi

        # use one of the egopose as world coord
        min_timestamp = min(list(self.egopose_ts_dict.keys()))
        ego_pose_wi_start = self.egopose_ts_dict[min_timestamp]

        # for t in self.egopose_ts_dict:
        for t in self.timestamp_list:
            ego_pose_wi_current = self.egopose_ts_dict[t]
            # compute ego_to_world transformation
            ego_to_world = np.linalg.inv(ego_pose_wi_start) @ ego_pose_wi_current
            ego_to_worlds.append(ego_to_world)
            # transformation:
            #   (opencv_cam -> waymo_cam -> waymo_ego_vehicle) -> current_world
            cam2world = ego_to_world @ cam_to_ego
            cam_to_worlds.append(cam2world)
            intrinsics.append(_intrinsics)
            distortions.append(_distortions)

        self.intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0)).float()
        self.distortions = torch.from_numpy(np.stack(distortions, axis=0)).float()
        self.cam_to_worlds = torch.from_numpy(np.stack(cam_to_worlds, axis=0)).float()
        self.calib_info = calib_info

    @classmethod
    def get_camera2worlds(cls, data_path: str, cam_id: str, start_timestep: int, end_timestep: int) -> torch.Tensor:
        """
        Returns camera-to-world matrices for the specified camera and time range.

        Args:
            data_path (str): Path to the dataset.
            cam_id (str): Camera ID.
            start_timestep (int): Start timestep.
            end_timestep (int): End timestep.

        Returns:
            torch.Tensor: Camera-to-world matrices of shape (num_frames, 4, 4).
        """
        # Load camera extrinsics
        cam_to_ego = np.loadtxt(os.path.join(data_path, "extrinsics", f"{cam_id}.txt"))
        cam_to_ego = cam_to_ego @ OPENCV2DATASET

        # Load ego poses and compute camera-to-world matrices
        cam_to_worlds = []
        ego_to_world_start = np.loadtxt(os.path.join(data_path, "ego_pose", f"{start_timestep:03d}.txt"))
        
        for t in range(start_timestep, end_timestep):
            ego_to_world_current = np.loadtxt(os.path.join(data_path, "ego_pose", f"{t:03d}.txt"))
            ego_to_world = np.linalg.inv(ego_to_world_start) @ ego_to_world_current
            cam2world = ego_to_world @ cam_to_ego
            cam_to_worlds.append(cam2world)

        return torch.from_numpy(np.stack(cam_to_worlds, axis=0)).float()

class CustomPixelSource(ScenePixelSource):
    def __init__(
        self,
        dataset_name: str,
        pixel_data_config: OmegaConf,
        data_path: str,
        start_timestep: int,
        end_timestep: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(dataset_name, pixel_data_config, device=device)
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.load_data()

    def load_cameras(self):
        self._timesteps = torch.arange(self.start_timestep, self.end_timestep)
        self.register_normalized_timestamps()
        # cam_id2int = {
        #     'cameraF30': 0,
        #     'cameraF100': 1,
        #     'cameraLF100': 2,
        #     'cameraRF100': 3,
        #     'cameraLR100': 4,
        #     'cameraRR100': 5,
        #     'cameraR50': 6,
        # }
        for idx, cam_id in enumerate(self.camera_list):
            logger.info(f"Loading camera {cam_id}")

            camera = CustomCameraData(
                dataset_name=self.dataset_name,
                data_path=self.data_path,
                cam_id=cam_id,
                start_timestep=self.start_timestep,
                end_timestep=self.end_timestep,
                load_dynamic_mask=self.data_cfg.load_dynamic_mask,
                load_sky_mask=self.data_cfg.load_sky_mask,
                load_road_mask=self.data_cfg.load_road_mask,
                downscale_when_loading=self.data_cfg.downscale_when_loading[idx],
                undistort=self.data_cfg.undistort,
                buffer_downscale=self.buffer_downscale,
                device=self.device,
            )
            camera.load_time(self.normalized_time)
            # camera_int = cam_id2int[camera.cam_name]
            # unique_img_idx = torch.arange(len(camera), device=self.device) * len(self.camera_list) + 
            unique_img_idx = torch.arange(len(camera), device=self.device) * len(self.camera_list) + idx
            camera.set_unique_ids(
                unique_cam_idx = idx,
                unique_img_idx = unique_img_idx
            )
            logger.info(f"Camera {camera.cam_name} loaded.")
            self.camera_data[cam_id] = camera
    
    def load_objects(self):
        pass

    # def load_objects(self):
    #     """
    #     get ground truth bounding boxes of the dynamic objects

    #     instances_info = {
    #         "0": # simplified instance id
    #             {
    #                 "id": str,
    #                 "class_name": str,
    #                 "frame_annotations": {
    #                     "frame_idx": List,
    #                     "obj_to_world": List,
    #                     "box_size": List,
    #             },
    #         ...
    #     }
    #     frame_instances = {
    #         "0": # frame idx
    #             List[int] # list of simplified instance ids
    #         ...
    #     }
    #     """
    #     instances_info_path = os.path.join(self.data_path, "instances", "instances_info.json")
    #     frame_instances_path = os.path.join(self.data_path, "instances", "frame_instances.json")
    #     with open(instances_info_path, "r") as f:
    #         instances_info = json.load(f)
    #     with open(frame_instances_path, "r") as f:
    #         frame_instances = json.load(f)
    #     # get pose of each instance at each frame
    #     # shape (num_frames, num_instances, 4, 4)
    #     num_instances = len(instances_info)
    #     num_full_frames = len(frame_instances)
    #     instances_pose = np.zeros((num_full_frames, num_instances, 4, 4))
    #     instances_size = np.zeros((num_full_frames, num_instances, 3))
    #     instances_true_id = np.arange(num_instances)
    #     instances_model_types = np.ones(num_instances) * -1
        
    #     ego_to_world_start = np.loadtxt(
    #         os.path.join(self.data_path, "ego_pose", f"{self.start_timestep:03d}.txt")
    #     )
    #     for k, v in instances_info.items():
    #         instances_model_types[int(k)] = OBJECT_CLASS_NODE_MAPPING[v["class_name"]]
    #         for frame_idx, obj_to_world, box_size in zip(v["frame_annotations"]["frame_idx"], v["frame_annotations"]["obj_to_world"], v["frame_annotations"]["box_size"]):
    #             # the first ego pose as the origin of the world coordinate system.
    #             obj_to_world = np.array(obj_to_world).reshape(4, 4)
    #             obj_to_world = np.linalg.inv(ego_to_world_start) @ obj_to_world
    #             instances_pose[frame_idx, int(k)] = np.array(obj_to_world)
    #             instances_size[frame_idx, int(k)] = np.array(box_size)
        
    #     # get frame valid instances
    #     # shape (num_frames, num_instances)
    #     per_frame_instance_mask = np.zeros((num_full_frames, num_instances))
    #     for frame_idx, valid_instances in frame_instances.items():
    #         per_frame_instance_mask[int(frame_idx), valid_instances] = 1
        
    #     # select the frames that are in the range of start_timestep and end_timestep
    #     instances_pose = torch.from_numpy(instances_pose[self.start_timestep:self.end_timestep]).float()
    #     instances_size = torch.from_numpy(instances_size[self.start_timestep:self.end_timestep]).float()
    #     instances_true_id = torch.from_numpy(instances_true_id).long()
    #     instances_model_types = torch.from_numpy(instances_model_types).long()
    #     per_frame_instance_mask = torch.from_numpy(per_frame_instance_mask[self.start_timestep:self.end_timestep]).bool()
        
    #     # filter out the instances that are not visible in selected frames
    #     ins_frame_cnt = per_frame_instance_mask.sum(dim=0)
    #     instances_pose = instances_pose[:, ins_frame_cnt > 0]
    #     instances_size = instances_size[:, ins_frame_cnt > 0]
    #     instances_true_id = instances_true_id[ins_frame_cnt > 0]
    #     instances_model_types = instances_model_types[ins_frame_cnt > 0]
    #     per_frame_instance_mask = per_frame_instance_mask[:, ins_frame_cnt > 0]
        
    #     # assign to the class
    #     # (num_frames, num_instances, 4, 4)
    #     self.instances_pose = instances_pose
    #     # (num_instances, 3)
    #     self.instances_size = instances_size.sum(0) / per_frame_instance_mask.sum(0).unsqueeze(-1)
    #     # (num_frames, num_instances)
    #     self.per_frame_instance_mask = per_frame_instance_mask
    #     # (num_instances)
    #     self.instances_true_id = instances_true_id
    #     # (num_instances)
    #     self.instances_model_types = instances_model_types
        
    #     if self.data_cfg.load_smpl:
    #         # Collect camera-to-world matrices for all available cameras
    #         cam_to_worlds = {}
    #         for cam_id in AVAILABLE_CAM_LIST:
    #             cam_to_worlds[cam_id] = WaymoCameraData.get_camera2worlds(
    #                 self.data_path, 
    #                 str(cam_id), 
    #                 self.start_timestep, 
    #                 self.end_timestep
    #             )

    #         # load SMPL parameters
    #         smpl_dict = joblib.load(os.path.join(self.data_path, "humanpose", "smpl.pkl"))
    #         frame_num = self.end_timestep - self.start_timestep
            
    #         smpl_human_all = {}
    #         for fi in tqdm(range(self.start_timestep, self.end_timestep), desc="Loading SMPL"):
    #             for instance_id, ins_smpl in smpl_dict.items():
    #                 if instance_id not in smpl_human_all:
    #                     smpl_human_all[instance_id] = {
    #                         "smpl_quats": torch.zeros((frame_num, 24, 4), dtype=torch.float32),
    #                         "smpl_trans": torch.zeros((frame_num, 3), dtype=torch.float32),
    #                         "smpl_betas": torch.zeros((frame_num, 10), dtype=torch.float32),
    #                         "frame_valid": torch.zeros((frame_num), dtype=torch.bool)
    #                     }
    #                     smpl_human_all[instance_id]["smpl_quats"][:, :, 0] = 1.0
    #                 if ins_smpl["valid_mask"][fi]:
    #                     betas = ins_smpl["smpl"]["betas"][fi]
    #                     smpl_human_all[instance_id]["smpl_betas"][fi - self.start_timestep] = betas
                        
    #                     body_pose = ins_smpl["smpl"]["body_pose"][fi]
    #                     smpl_orient = ins_smpl["smpl"]["global_orient"][fi]
    #                     cam_depend = ins_smpl["selected_cam_idx"][fi].item()
                        
    #                     c2w = cam_to_worlds[cam_depend][fi - self.start_timestep]
    #                     world_orient = c2w[:3, :3].to(smpl_orient.device) @ smpl_orient.squeeze()
    #                     smpl_quats = matrix_to_quaternion(
    #                         torch.cat([world_orient[None, ...], body_pose], dim=0)
    #                     )
                        
    #                     ii = instances_info[str(instance_id)]['frame_annotations']["frame_idx"].index(fi)
    #                     o2w = np.array(
    #                         instances_info[str(instance_id)]['frame_annotations']["obj_to_world"][ii]
    #                     )
    #                     o2w = torch.from_numpy(
    #                         np.linalg.inv(ego_to_world_start) @ o2w
    #                     )
    #                     # box_size = instances_info[str(instance_id)]['frame_annotations']["box_size"][ii]
                        
    #                     smpl_human_all[instance_id]["smpl_quats"][fi - self.start_timestep] = smpl_quats
    #                     smpl_human_all[instance_id]["smpl_trans"][fi - self.start_timestep] = o2w[:3, 3]
    #                     smpl_human_all[instance_id]["frame_valid"][fi - self.start_timestep] = True

    #         self.smpl_human_all = smpl_human_all
            
class CustomLiDARSource(SceneLidarSource):
    def __init__(
        self,
        lidar_data_config: OmegaConf,
        data_path: str,
        start_timestep: int,
        end_timestep: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(lidar_data_config, device=device)
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.create_all_filelist()
        self.load_data()

    def create_all_filelist(self):
        """
        Create a list of all the files in the dataset.
        e.g., a list of all the lidar scans in the dataset.
        """
        # load lidar ts list
        lidar_list = sorted(glob.glob(os.path.join(self.data_path, "cloud_body/*.pcd")))
        timestamp_list = []
        for fpath in lidar_list:
            timestamp_list.append(fpath.split('/')[-1].replace(".pcd", ""))
        self.timestamp_list = timestamp_list
        self.lidar_filepaths = np.array(lidar_list)

    def interpolate_pose(self, src_ts_list, src_qwc_list, src_center_list,
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

    def load_calibrations(self):
        """
        Load the calibration files of the dataset.
        e.g., lidar to world transformation matrices.
        """

        with open(os.path.join(self.data_path, 'Image_Pose.txt.json.from_lidar'), 'r') as f:
            image_pose_info = json.load(f)
        self.egopose_ts_dict = {}
        for feat in image_pose_info["features"]:
            Rwc_quat_wxyz = feat["properties"]["extrinsic"]["Rwc_quat_wxyz"]
            rotation = R.from_quat(np.array(Rwc_quat_wxyz)[[1, 2, 3, 0]]).as_matrix()
            center = feat['properties']['extrinsic']['center']
            ego_pose_wi = np.eye(4)
            ego_pose_wi[:3, :3] = rotation
            ego_pose_wi[:3, 3] = center

            timestamp = feat["properties"]["timestamp"]
            self.egopose_ts_dict[timestamp] = ego_pose_wi
        
        self.egopose_ts_list = list(self.egopose_ts_dict.keys())
        # use one of the egopose as world coord
        min_timestamp = min(list(self.egopose_ts_dict.keys()))
        ego_pose_wi_start = self.egopose_ts_dict[min_timestamp]
        lidar_to_worlds = {}
        src_qwc_list, src_center_list = [], []

        for t in self.egopose_ts_dict:
            ego_pose_wi_current = self.egopose_ts_dict[t]
            # compute ego_to_world transformation
            lidar_to_world = np.linalg.inv(ego_pose_wi_start) @ ego_pose_wi_current
            # lidar_to_worlds.append(lidar_to_world)
            lidar_to_worlds[t] = torch.from_numpy(lidar_to_world).float()

            qwc = R.from_matrix(lidar_to_world[:3, :3]).as_quat().reshape(1, 4)
            center = lidar_to_world[:3, 3].reshape(1, 3)
            src_qwc_list.append(qwc)
            src_center_list.append(center)

        # interpolate ts
        src_ts_list = [int(val) for val in list(lidar_to_worlds.keys())]
        target_ts_list = [int(val) for val in self.timestamp_list]
        valid_ts_list, qwc_list, center_list = self.interpolate_pose(
            src_ts_list,
            src_qwc_list,
            src_center_list,
            target_ts_list
        )

        interp_lidar_to_worlds = {}
        for idx, t in enumerate(valid_ts_list):
            rotation = qwc_list[idx].as_matrix()
            center = np.array(center_list[idx])
            transformation = np.eye(4, dtype=np.float32)
            transformation[:3, :3] = rotation
            transformation[:3, 3] = center

            ts = str(valid_ts_list[idx])
            interp_lidar_to_worlds[ts] = torch.from_numpy(transformation).float()

        # self.lidar_to_worlds = torch.from_numpy(
        #     np.stack(lidar_to_worlds, axis=0)
        # ).float()
        self.valid_timestamp_list = list(interp_lidar_to_worlds.keys())
        self.lidar_to_worlds = interp_lidar_to_worlds

    def load_lidar(self):
        """
        Load the lidar data of the dataset from the filelist.
        """
        origins, directions, ranges, laser_ids = [], [], [], []
        # in waymo, we simplify timestamps as the time indices
        timesteps = []

        accumulated_num_original_rays = 0
        accumulated_num_rays = 0
        for t in trange(
            0, len(self.valid_timestamp_list), desc="Loading lidar", dynamic_ncols=True
        ):
            lidar_filepaths = os.path.join(self.data_path, f"cloud_body/{self.valid_timestamp_list[t]}.pcd")
            lidar_points = np.array(o3d.io.read_point_cloud(lidar_filepaths).points)
            original_length = len(lidar_points)
            accumulated_num_original_rays += original_length

            lidar_origins = torch.from_numpy(np.zeros_like(lidar_points)).float()
            lidar_points = torch.from_numpy(lidar_points).float()
            lidar_ids = torch.from_numpy(np.zeros((original_length,))).float()
            # we don't collect intensities and elongations for now

            # select lidar points based on a truncated ego-forward-directional range
            # this is to make sure most of the lidar points are within the range of the camera
            valid_mask = torch.ones_like(lidar_origins[:, 0]).bool()
            if self.data_cfg.truncated_max_range is not None:
                valid_mask = lidar_points[:, 0] < self.data_cfg.truncated_max_range
            if self.data_cfg.truncated_min_range is not None:
                valid_mask = valid_mask & (
                    lidar_points[:, 0] > self.data_cfg.truncated_min_range
                )
            lidar_origins = lidar_origins[valid_mask]
            lidar_points = lidar_points[valid_mask]
            lidar_ids = lidar_ids[valid_mask]
            # transform lidar points from lidar coordinate system to world coordinate system
            lidar_to_world_pose = self.lidar_to_worlds[self.valid_timestamp_list[t]]
            lidar_origins = (
                lidar_to_world_pose[:3, :3] @ lidar_origins.T
                + lidar_to_world_pose[:3, 3:4]
            ).T
            lidar_points = (
                lidar_to_world_pose[:3, :3] @ lidar_points.T
                + lidar_to_world_pose[:3, 3:4]
            ).T
           
            # compute lidar directions
            lidar_directions = lidar_points - lidar_origins
            lidar_ranges = torch.norm(lidar_directions, dim=-1, keepdim=True)
            lidar_directions = lidar_directions / lidar_ranges
            # we use time indices as the timestamp for waymo dataset
            lidar_timestamp = torch.ones_like(lidar_ranges).squeeze(-1) * t
            accumulated_num_rays += len(lidar_ranges)

            origins.append(lidar_origins)
            directions.append(lidar_directions)
            ranges.append(lidar_ranges)
            laser_ids.append(lidar_ids)
            # we use time indices as the timestamp for waymo dataset
            timesteps.append(lidar_timestamp)

        logger.info(
            f"Number of lidar rays: {accumulated_num_rays} "
            f"({accumulated_num_rays / accumulated_num_original_rays * 100:.2f}% of "
            f"{accumulated_num_original_rays} original rays)"
        )
        logger.info("Filter condition:")
        logger.info(f"  only_use_top_lidar: {self.data_cfg.only_use_top_lidar}")
        logger.info(f"  truncated_max_range: {self.data_cfg.truncated_max_range}")
        logger.info(f"  truncated_min_range: {self.data_cfg.truncated_min_range}")

        self.origins = torch.cat(origins, dim=0)
        self.directions = torch.cat(directions, dim=0)
        self.ranges = torch.cat(ranges, dim=0)
        self.laser_ids = torch.cat(laser_ids, dim=0)
        self.visible_masks = torch.zeros_like(self.ranges).squeeze().bool()
        self.road_masks = torch.zeros_like(self.ranges).squeeze().bool()
        self.colors = torch.ones_like(self.directions)

        # the underscore here is important.
        self._timesteps = torch.cat(timesteps, dim=0)
        self.register_normalized_timestamps()

    def to(self, device: torch.device):
        super().to(device)

    def get_lidar_rays(self, time_idx: int) -> Dict[str, Tensor]:
        """
        Get the of rays for rendering at the given timestep.
        Args:
            time_idx: the index of the lidar scan to render.
        Returns:
            a dict of the sampled rays.
        """
        origins = self.origins[self.timesteps == time_idx]
        directions = self.directions[self.timesteps == time_idx]
        ranges = self.ranges[self.timesteps == time_idx]
        normalized_time = self.normalized_time[self.timesteps == time_idx]
        # flows = self.flows[self.timesteps == time_idx]
        return {
            "lidar_origins": origins,
            "lidar_viewdirs": directions,
            "lidar_ranges": ranges,
            "lidar_normed_time": normalized_time,
            "lidar_mask": self.timesteps == time_idx,
            # "lidar_flows": flows,
        }


    def find_closest_timestep(self, timestamp: float) -> int:
        """
        Find the closest timestep to the given timestamp.
        Args:
            normed_timestamp: the normalized timestamp to find the closest timestep for.
        Returns:
            the closest timestep to the given timestamp.
        """
        # import ipdb; ipdb.set_trace()
        int_timestamp = torch.tensor([int(val) for val in self.valid_timestamp_list])
        return torch.argmin(
            torch.abs(int_timestamp - timestamp)
        )

    def get_all_lidar_rays(self) -> Dict[str, Tensor]:
        """
        Get the of rays for rendering at the given timestep.
        Args:
            time_idx: the index of the lidar scan to render.
        Returns:
            a dict of the sampled rays.
        """
        return {
            "lidar_origins": self.origins,
            "lidar_viewdirs": self.directions,
            "lidar_ranges": self.ranges,
            "lidar_normed_time": self.normalized_time,
        }
    def get_road_lidarpoints(self):
        return self.road_point_neus
    def delete_invisible_pts(self) -> None:
        """
        Clear the unvisible points.
        """
        if self.visible_masks is not None:
            num_bf = self.origins.shape[0]
            self.origins = self.origins[self.visible_masks]
            self.directions = self.directions[self.visible_masks]
            self.ranges = self.ranges[self.visible_masks]
            self._timesteps = self._timesteps[self.visible_masks]
            self._normalized_time = self._normalized_time[self.visible_masks]
            self.colors = self.colors[self.visible_masks]
            logger.info(
                f"[Lidar] {num_bf - self.visible_masks.sum()} out of {num_bf} points are cleared. {self.visible_masks.sum()} points left."
            )
            self.visible_masks = None
            # save visible lidar points
            # lidar_points = (
            #         self.origins
            #         + self.directions * self.ranges
            #     )
            # lidar_points[:, 2] += 1.44
            # import open3d as o3d
            # pc = o3d.geometry.PointCloud()
            # pc.points = o3d.utility.Vector3dVector(lidar_points.cpu().numpy())
            # o3d.io.write_point_cloud("lidar_points.pcd", pc)
        else:
            logger.info("[Lidar] No unvisible points to clear.")