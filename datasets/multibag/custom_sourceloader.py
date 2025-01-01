from typing import Dict
import logging
import os
import json
import joblib
import glob
import copy
import numpy as np
import ditu.topbind as tb
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
from datasets.dataset_meta import DATASETS_CONFIG

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
AVAILABLE_CAM_LIST = [0, 1, 2, 3, 4, 5, 6]

class CustomCameraData(CameraData):
    def __init__(
        self,
        dataset_name: str,
        data_path: str,
        cam_id: int,
        bag_id: int,
        anchor_pose: np.array = None,
        # the start timestep to load
        start_timestep: int = 0,
        # the end timestep to load
        end_timestep: int = None,
        # whether to load the dynamic masks
        load_dynamic_mask: bool = False,
        # whether to load the sky masks
        load_sky_mask: bool = False,
        # the size to load the images
        downscale_when_loading: float = 1.0,
        # whether to undistort the images
        undistort: bool = False,
        # whether to use buffer sampling
        buffer_downscale: float = 1.0,
        # the device to move the camera to
        device: torch.device = torch.device("cpu"),
    ):
        self.dataset_name = dataset_name
        self.cam_id = cam_id
        self.bag_id = bag_id
        self.anchor_pose = anchor_pose
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.undistort = undistort
        self.buffer_downscale = buffer_downscale
        self.device = device
        self.cam_name = DATASETS_CONFIG[dataset_name][cam_id]["camera_name"]
        self.depth_map_path = os.path.join(self.data_path, self.cam_name, "keyframes", f"{self.cam_name}_depthmap.npy")

        # get current cam timestamp
        Image_Pose_new_filtered_path = os.path.join(self.data_path, "Image_Pose_new_filtered.txt.json") 
        Image_Pose_new_path = os.path.join(self.data_path, "Image_Pose_new.txt.json") 
        self.image_pose_filtered_path = Image_Pose_new_filtered_path if os.path.exists(Image_Pose_new_filtered_path) else Image_Pose_new_path

        with open(self.image_pose_filtered_path, 'r') as f:
            image_pose_info = json.load(f)
        self.timestamp_list = []
        for idx, finfo in enumerate(image_pose_info["features"]):
            if idx < self.start_timestep:
                continue
            if idx >= self.end_timestep:
                break
            self.timestamp_list.append(finfo["properties"]["timestamp"])
        self.timestamp_list = sorted(self.timestamp_list)
        
        with open(os.path.join(self.data_path, self.cam_name, "calibration.json"), "r") as f:
            calib_info = json.load(f)
        self.original_size = (calib_info["image_height"], calib_info["image_width"])
        self.load_size = [
            int(self.original_size[0] / downscale_when_loading),
            int(self.original_size[1] / downscale_when_loading),
        ]
        self.downscale_when_loading = downscale_when_loading

        # Load the images, dynamic masks, sky masks, etc.
        self.create_all_filelist()
        self.load_calibrations()
        # self.load_images()
        self.images = None
        self.load_egocar_mask()
        self.dynamic_masks = None
        self.sky_masks = None
        self.human_masks = None
        self.vehicle_masks = None
        self.road_masks = None
        self.lidar_depth_maps = None
        # if load_dynamic_mask:
        #     self.load_dynamic_masks()
        # if load_sky_mask:
        #     self.load_sky_masks()
        # self.lidar_depth_maps = None # will be loaded by: self.load_depth()
        # self.load_depth()
        self.image_error_maps = None # will be built by: self.build_image_error_buffer()
        self.to(self.device)
        self.downscale_factor = 1.0
        

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
            road_mask_filepaths.append(
                os.path.join(
                    os.path.join(self.data_path, self.cam_name, 'keyframes', 'seg_images', f"{t}.png")
                )
            )
            sky_mask_filepaths.append(
                os.path.join(self.data_path, self.cam_name, 'keyframes', 'seg_images', f"{t}.png")
            )
        self.img_filepaths = np.array(img_filepaths)
        self.dynamic_mask_filepaths = np.array(dynamic_mask_filepaths)
        self.human_mask_filepaths = np.array(human_mask_filepaths)
        self.vehicle_mask_filepaths = np.array(vehicle_mask_filepaths)
        self.road_mask_filepaths = np.array(road_mask_filepaths)
        self.sky_mask_filepaths = np.array(sky_mask_filepaths)

        with open(os.path.join(self.data_path, self.cam_name, "calibration.json")) as f:
            calib_info = json.load(f)
        
        self.cam_height = calib_info['ground_vec'][1]

    def load_egocar_mask(self):
        # compute egocar mask from hoodline
        # try:
        height, width = self.calib_info['image_height'], self.calib_info['image_width']
        egocar_mask_np = np.zeros((height, width))
        hoodline = int(self.calib_info['hood_line'])
        egocar_mask_np[hoodline:,...] = 255
        egocar_mask = Image.fromarray(egocar_mask_np).convert("L").resize(
            (self.load_size[1], self.load_size[0]), Image.BILINEAR
        )
        self.egocar_mask = torch.from_numpy(np.array(egocar_mask) > 0).float()
        self.hoodline = int(hoodline / self.downscale_when_loading)
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

        road_masks = []
        for ix, fname in tqdm(enumerate(self.road_mask_filepaths), desc="Loading road masks",
            dynamic_ncols=True, total=len(self.road_mask_filepaths),
        ):
            label = np.asarray(Image.open(fname))
            road_mask = (label == 21) | ((35 <= label) & (label <= 56)) | (label == 13) | (label == 14) | (label == 74)
            road_mask = Image.fromarray(road_mask * 255.).convert('L')
            road_mask = road_mask.resize(
                (self.load_size[1], self.load_size[0]), Image.BILINEAR
            )
            road_masks.append(np.array(road_mask) > 0)
        self.road_masks = torch.from_numpy(np.stack(road_masks, axis=0)).float()

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
        intrinsics, distortions = [], []

        # we tranform the camera poses w.r.t. the first timestep to make the translation vector of
        # the first ego pose as the origin of the world coordinate system.
        # ego_to_world_start = np.loadtxt(
        #     os.path.join(self.data_path, "ego_pose", f"{self.start_timestep:03d}.txt")
        # )

        with open(self.image_pose_filtered_path, 'r') as f:
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

        # use one of the egopose as world coord
        if self.anchor_pose is None:
            min_timestamp = min(list(self.egopose_ts_dict.keys()))
            self.anchor_pose = copy.deepcopy(self.egopose_ts_dict[min_timestamp])

        # for t in self.egopose_ts_dict:
        cam_to_worlds = []
        ego_to_worlds = []
        for t in self.timestamp_list:
            ego_pose_wi_current = self.egopose_ts_dict[t]
            # compute ego_to_world transformation
            ego_to_world = np.linalg.inv(self.anchor_pose) @ ego_pose_wi_current
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
        self.ego_to_wrolds = torch.from_numpy(np.stack(ego_to_worlds, axis=0)).float()
        self.calib_info = calib_info

    def load_depth(self):
        # import ipdb; ipdb.set_trace()
        # load depth npy
        depth_map_npy = np.load(self.depth_map_path, allow_pickle=True).item()

        lidar_depth_maps = []
        for ts in self.timestamp_list:
            depth_map = torch.zeros(self.load_size[0], self.load_size[1])
            if ts in depth_map_npy:
                info = depth_map_npy[ts]
                cam_points = torch.tensor(info['cam_points'])
                cam_points[:, 0] /= self.downscale_when_loading
                cam_points[:, 1] /= self.downscale_when_loading
                depth = torch.tensor(info['depth']).float()
                depth_map[torch.minimum(torch.tensor(depth_map.shape[0] - 1), cam_points[:, 1].long()), torch.minimum(torch.tensor(depth_map.shape[1] - 1), cam_points[:, 0].long())] = depth
            lidar_depth_maps.append(depth_map)
        self.lidar_depth_maps = torch.stack(lidar_depth_maps, dim = 0)
        

class CustomPixelSource(ScenePixelSource):
    def __init__(
        self,
        dataset_name: str,
        pixel_data_config: OmegaConf,
        data_path: list,
        start_timestep: int,
        end_timestep: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(dataset_name, pixel_data_config, device=device)
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.load_data()
    
    @property
    def num_cams(self) -> int:
        """
        Returns:
            the number of cameras in the dataset
        """
        return len(self.camera_data.keys())

    @property
    def num_imgs(self) -> int:
        """
        Returns:
            the number of images in the dataset
        """
        total_num_imgs = 0
        for cam in self.camera_data:
            total_num_imgs += len(self.camera_data[cam])
        return total_num_imgs

    @property
    def num_cams_per_bag(self) -> int:
        """
        Returns:
            the number of cameras in the dataset
        """
        return len(self.camera_list)


    @property
    def num_timesteps(self) -> int:
        """
        Returns:
            the number of image timesteps in the dataset
        """
        if isinstance(self._timesteps, list):
            return len(self._timesteps)
        else:
            total_count = 0
            for key in self._timesteps:
                total_count += len(self._timesteps[key])
            return total_count

    def to(self, device: torch.device) -> "ScenePixelSource":
        """
        Move the dataset to the given device.
        Args:
            device: the device to move the dataset to.
        """
        self.device = device
        if self._timesteps is not None:
            for key in self._timesteps:
                self._timesteps[key] = self._timesteps[key].to(device)
        if self._normalized_time is not None:
            self._normalized_time = self._normalized_time.to(device)
        if self.instances_pose is not None:
            self.instances_pose = self.instances_pose.to(device)
        if self.instances_size is not None:
            self.instances_size = self.instances_size.to(device)
        if self.per_frame_instance_mask is not None:
            self.per_frame_instance_mask = self.per_frame_instance_mask.to(device)
        if self.instances_model_types is not None:
            self.instances_model_types = self.instances_model_types.to(device)
        return self

    def load_data(self):
        """
        A general function to load all data.
        """
        self.load_cameras()
        # self.build_image_error_buffer()
        logger.info("[Pixel] All Pixel Data loaded.")
        
        if self.data_cfg.load_objects:
            self.load_objects()
            logger.info("[Pixel] All Object Annotations loaded.")
        
        # set initial downscale factor
        for cam_id in self.camera_data:
            self.camera_data[cam_id].set_downscale_factor(self._downscale_factor)


    def load_cameras(self):
        # self._timesteps = torch.arange(self.start_timestep, self.end_timestep)
        # self.register_normalized_timestamps()

        scene_img_count, scene_camera_count = 0, 0
        self.anchor_pose = None
        self.frame_idx_cam_id_convert_map = {}
        self.cam_id_scene_name_map = {}
        self._timesteps = {}
        for bag_idx, scene_path in enumerate(self.data_path):
            cur_scene_img_count = 0
            scene_name = scene_path.split('/')[-1]
            _timesteps = None
            for idx, cam_id in enumerate(self.camera_list):
                logger.info(f"Loading camera {cam_id} in bag {bag_idx}")
                camera = CustomCameraData(
                    dataset_name=self.dataset_name,
                    data_path=scene_path,
                    cam_id=cam_id,
                    anchor_pose=self.anchor_pose,
                    start_timestep=self.start_timestep,
                    end_timestep=self.end_timestep,
                    load_dynamic_mask=self.data_cfg.load_dynamic_mask,
                    load_sky_mask=self.data_cfg.load_sky_mask,
                    downscale_when_loading=self.data_cfg.downscale_when_loading[idx],
                    undistort=self.data_cfg.undistort,
                    buffer_downscale=self.buffer_downscale,
                    device=self.device,
                    bag_id=bag_idx
                )
                # camera.load_time(self.normalized_time)
                unique_img_idx = torch.arange(len(camera), device=self.device) * len(self.camera_list) + idx + scene_img_count
                camera.set_unique_ids(
                    unique_cam_idx = idx + scene_camera_count,
                    unique_img_idx = unique_img_idx
                )
                cur_scene_img_count += len(unique_img_idx)
                logger.info(f"Camera {camera.cam_name} in bag {bag_idx} loaded.")
                self.camera_data[cam_id + scene_camera_count] = camera
                self.anchor_pose = camera.anchor_pose
                self.cam_id_scene_name_map[cam_id + scene_camera_count] = scene_name
                if _timesteps is None:
                    _timesteps = [int(val) for val in camera.timestamp_list]
                camera.load_time(torch.tensor([float(val)/len(_timesteps) for val in range(len(_timesteps))]))
                
                for uii in unique_img_idx:
                    frame_idx_in_total = (uii - idx) // len(self.camera_list)
                    cur_frame_idx = frame_idx_in_total - scene_img_count // len(self.camera_list)
                    self.frame_idx_cam_id_convert_map[(frame_idx_in_total.item(), cam_id)] = (cur_frame_idx.item(), cam_id + scene_camera_count)
        
            scene_img_count += cur_scene_img_count
            scene_camera_count += len(self.camera_list)
            self._timesteps[scene_name] = torch.tensor(_timesteps)
        
        self.register_normalized_timestamps()
    
    def load_objects(self):
        pass

    def register_normalized_timestamps(self) -> None:
        # normalized timestamps are between 0 and 1
        self.timestep_map = {}
        total_num_timesteps = self.num_timesteps
        count = 0
        for key in self._timesteps:
            for ts in self._timesteps[key]:
                count_key = float(count) / total_num_timesteps
                self.timestep_map[count_key] = (key, ts)
                count += 1
        
        self._normalized_time = torch.tensor(list(self.timestep_map.keys())).to(self.device)
        self._unique_normalized_timestamps = self._normalized_time.unique()

    def parse_img_idx(self, img_idx: int):
        """
        Parse the image index to the camera index and frame index.
        Args:
            img_idx: the image index.
        Returns:
            cam_idx: the camera index.
            frame_idx: the frame index.
        """
        cam_idx = img_idx % self.num_cams_per_bag
        frame_idx = img_idx // self.num_cams_per_bag
        return cam_idx, frame_idx

    def get_image(self, img_idx: int) -> Dict[str, Tensor]:
        """
        Get the rays for rendering the given image index.
        Args:
            img_idx: the image index.
        Returns:
            a dict containing the rays for rendering the given image index.
        """
        cam_idx, frame_idx = self.parse_img_idx(img_idx)
        cur_frame_idx, unique_cam_idx = self.frame_idx_cam_id_convert_map[(frame_idx, cam_idx)]
        return self.camera_data[unique_cam_idx].get_image(cur_frame_idx)

class CustomLiDARSource(SceneLidarSource):
    def __init__(
        self,
        lidar_data_config: OmegaConf,
        data_path: list,
        anchor_pose,
        start_timestep: int,
        end_timestep: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(lidar_data_config, device=device)
        self.data_path = data_path
        self.data_root_path = os.path.dirname(data_path[0])
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.anchor_pose = anchor_pose
        self.medb_path = os.path.join(self.data_root_path, "lidar_loc_points_colored_tree.medb")
        self.create_all_filelist()
        self.load_data()

    def create_all_filelist(self):
        """
        Create a list of all the files in the dataset.
        e.g., a list of all the lidar scans in the dataset.
        """
        self.timestamp_dict = {}
        self.lidar_filepaths = {}
        for bag_idx, scene_path in enumerate(self.data_path):
            scene_name = scene_path.split("/")[-1]
            # load lidar ts list
            lidar_list = sorted(glob.glob(os.path.join(scene_path, "cloud_body/*.pcd")))
            timestamp_list = []
            update_lidar_list = []
            for idx, fpath in enumerate(lidar_list):
                if idx < self.start_timestep:
                    continue 
                if idx > self.end_timestep:
                    continue
                update_lidar_list.append(fpath)
                timestamp_list.append(fpath.split('/')[-1].replace(".pcd", ""))
            self.timestamp_dict[scene_name] = timestamp_list
            self.lidar_filepaths[scene_name] = np.array(lidar_list)

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
        pass

    # def load_calibrations(self):
    #     """
    #     Load the calibration files of the dataset.
    #     e.g., lidar to world transformation matrices.
    #     """

    #     self.egopose_scene_ts_dict = {}
    #     self.valid_timestamp_dict = {}
    #     self.lidar_to_worlds_dict = {}
    #     self.max_timestamp_count = 0
    #     for idx, scene_path in enumerate(self.data_path):
            
    #         scene_name = scene_path.split("/")[-1]
    #         with open(self.image_pose_filtered_path, 'r') as f:
    #             image_pose_info = json.load(f)
            
    #         egopose_ts_dict = {}
    #         for feat in image_pose_info["features"]:
    #             Rwc_quat_wxyz = feat["properties"]["extrinsic"]["Rwc_quat_wxyz"]
    #             rotation = R.from_quat(np.array(Rwc_quat_wxyz)[[1, 2, 3, 0]]).as_matrix()
    #             center = feat['properties']['extrinsic']['center']
    #             ego_pose_wi = np.eye(4)
    #             ego_pose_wi[:3, :3] = rotation
    #             ego_pose_wi[:3, 3] = center

    #             timestamp = feat["properties"]["timestamp"]
    #             egopose_ts_dict[timestamp] = ego_pose_wi
            
    #         self.egopose_scene_ts_dict[scene_name] = egopose_ts_dict

    #         egopose_ts_list = list(egopose_ts_dict.keys())
    #         if self.anchor_pose is None:
    #             # use one of the egopose as world coord
    #             min_timestamp = min(list(egopose_ts_dict.keys()))
    #             self.anchor_pose = egopose_ts_dict[min_timestamp]
    #         lidar_to_worlds = {}
    #         src_qwc_list, src_center_list = [], []

    #         for t in egopose_ts_dict:
    #             ego_pose_wi_current = egopose_ts_dict[t]
    #             # compute ego_to_world transformation
    #             lidar_to_world = np.linalg.inv(self.anchor_pose) @ ego_pose_wi_current
    #             # lidar_to_worlds.append(lidar_to_world)
    #             lidar_to_worlds[t] = torch.from_numpy(lidar_to_world).float()

    #             qwc = R.from_matrix(lidar_to_world[:3, :3]).as_quat().reshape(1, 4)
    #             center = lidar_to_world[:3, 3].reshape(1, 3)
    #             src_qwc_list.append(qwc)
    #             src_center_list.append(center)

    #         # interpolate ts
    #         src_ts_list = [int(val) for val in list(lidar_to_worlds.keys())]
    #         target_ts_list = [int(val) for val in self.timestamp_dict[scene_name]]
    #         valid_ts_list, qwc_list, center_list = self.interpolate_pose(
    #             src_ts_list,
    #             src_qwc_list,
    #             src_center_list,
    #             target_ts_list
    #         )

    #         interp_lidar_to_worlds = {}
    #         for idx, t in enumerate(valid_ts_list):
    #             rotation = qwc_list[idx].as_matrix()
    #             center = np.array(center_list[idx])
    #             transformation = np.eye(4, dtype=np.float32)
    #             transformation[:3, :3] = rotation
    #             transformation[:3, 3] = center

    #             ts = str(valid_ts_list[idx])
    #             interp_lidar_to_worlds[ts] = torch.from_numpy(transformation).float()

    #         # self.lidar_to_worlds = torch.from_numpy(
    #         #     np.stack(lidar_to_worlds, axis=0)
    #         # ).float()
    #         self.valid_timestamp_dict[scene_name] = list(interp_lidar_to_worlds.keys())
    #         self.lidar_to_worlds_dict[scene_name] = interp_lidar_to_worlds
    #         self.max_timestamp_count = max(self.max_timestamp_count, len(interp_lidar_to_worlds))

    def load_lidar(self):
        """
        Load the lidar data of the dataset from the filelist.
        """
        lidar_loc_medb = tb.Medb(self.medb_path)
        lidar_loc_points_ecef = lidar_loc_medb.points()
        lidar_loc_colors = lidar_loc_medb.colors() / 255.

        # convert ecef to anchor_pose
        ecef2world = np.linalg.inv(self.anchor_pose)
        lidar_loc_points_world = (
            ecef2world[:3, :3] @ lidar_loc_points_ecef.T 
            + ecef2world[:3, 3:4]
        ).T 

        # import ipdb; ipdb.set_trace()
        self.lidar_loc_points = torch.tensor(lidar_loc_points_world).float()
        self.colors = torch.tensor(lidar_loc_colors).float()

        logger.info(
            f"Number of lidar rays: {len(self.lidar_loc_points)}"
        )

    def to(self, device: torch.device):
        super().to(device)

    def get_aabb(self) -> Tensor:
        """
        Returns:
            aabb_min, aabb_max: the min and max of the axis-aligned bounding box of the scene
        Note:
            we assume the lidar points are already in the world coordinate system
            we first downsample the lidar points, then compute the aabb by taking the
            given percentiles of the lidar coordinates in each dimension.
        """
        assert (
            self.lidar_loc_points is not None
            and self.colors is not None
        ), "Lidar points not loaded, cannot compute aabb."
        logger.info("[Lidar] Computing auto AABB based on downsampled lidar points....")

        lidar_pts = self.lidar_loc_points

        # downsample the lidar points by uniformly sampling a subset of them
        lidar_pts = lidar_pts[
            torch.randperm(len(lidar_pts))[
                : int(len(lidar_pts) / self.data_cfg.lidar_downsample_factor)
            ]
        ]
        # compute the aabb by taking the given percentiles of the lidar coordinates in each dimension
        aabb_min = torch.quantile(lidar_pts, self.data_cfg.lidar_percentile, dim=0)
        aabb_max = torch.quantile(lidar_pts, 1 - self.data_cfg.lidar_percentile, dim=0)
        del lidar_pts
        torch.cuda.empty_cache()

        # usually the lidar's height is very small, so we slightly increase the height of the aabb
        if aabb_max[-1] < 20:
            aabb_max[-1] = 20.0
        aabb = torch.tensor([*aabb_min, *aabb_max])
        logger.info(f"[Lidar] Auto AABB from LiDAR: {aabb}")
        return aabb

    @property
    def pts_xyz(self) -> Tensor:
        """
        Returns:
            the xyz coordinates of the lidar points.
            shape: (num_lidar_points, 3)
        """
        return self.lidar_loc_points

    @property
    def num_points(self) -> int:
        """
        Returns:
            the number of lidar points in the dataset.
        """
        return self.lidar_loc_points.size(0)

    def find_closest_timestep(self, timestamp: float, scene_name=None) -> int:
        """
        Find the closest timestep to the given timestamp.
        Args:
            normed_timestamp: the normalized timestamp to find the closest timestep for.
        Returns:
            the closest timestep to the given timestamp.
        """
        # import ipdb; ipdb.set_trace()
        if scene_name is None:
            int_timestamp = torch.tensor([int(val) for val in self.valid_timestamp_list])
            return torch.argmin(
                torch.abs(int_timestamp - timestamp)
            )
        else:
            int_timestamp = torch.tensor([int(val) for val in self.valid_timestamp_dict[scene_name]])
            lidar_min_idx = torch.argmin(torch.abs(int_timestamp - timestamp))
            lidar_timestamp = self.valid_timestamp_dict[scene_name][lidar_min_idx]
            return self.lidar_timestamp_scene_name_map[scene_name][lidar_timestamp], lidar_timestamp

    def get_all_lidar_rays(self) -> Dict[str, Tensor]:
        """
        Get the of rays for rendering at the given timestep.
        Args:
            time_idx: the index of the lidar scan to render.
        Returns:
            a dict of the sampled rays.
        """
        return self.lidar_loc_points


