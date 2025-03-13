
import torch
import torchvision
import torch.nn.functional as F
import contextlib
import numpy as np
import os
from PIL import Image
from functools import partial
from einops import rearrange, repeat
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    AutoencoderKL,
    ModelMixin,
)

from transformers import (
    CLIPTokenizer, 
    CLIPTextModel,
)
from magicdrive.misc.common import move_to
from magicdrive.networks.unet_addon_rawbox import BEVControlNetModel
from magicdrive.networks.unet_2d_condition_multiview import UNet2DConditionModelMultiview
from magicdrive.pipeline.pipeline_bev_controlnet import StableDiffusionBEVControlNetPipeline
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.core.bbox.structures.utils import get_box_type
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.map_api import locations as LOCATIONS
from nuscenes import NuScenes
from mmdet3d.datasets.pipelines.loading import *
from mmdet3d.datasets.pipelines.formating import *
from mmdet3d.datasets.pipelines.transforms_3d import *
from magicdrive.dataset.pipeline import *
from magicdrive.dataset.utils import collate_fn
from mmcv.parallel.data_container import DataContainer
import datetime
import random
prefix_current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
prefix_current_time = prefix_current_time + f"_{os.environ['exp_name']}" + f"_{os.environ['scene_idx']}"
if not os.path.exists(prefix_current_time):
    os.makedirs(prefix_current_time)

NameMapping = {
        "movable_object.barrier": "barrier",
        "vehicle.bicycle": "bicycle",
        "vehicle.bus.bendy": "bus",
        "vehicle.bus.rigid": "bus",
        "vehicle.car": "car",
        "vehicle.construction": "construction_vehicle",
        "vehicle.motorcycle": "motorcycle",
        "human.pedestrian.adult": "pedestrian",
        "human.pedestrian.child": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "movable_object.trafficcone": "traffic_cone",
        "vehicle.trailer": "trailer",
        "vehicle.truck": "truck",
    }

import lpips
import torch.nn as nn
import torch
from torch import Tensor
from torch.nn import functional as F


def calc_mean_std(feat: Tensor, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, "The input feature should be 4D tensor."
    b, c = size[:2]
    feat_var = feat.reshape(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(b, c, 1, 1)
    feat_mean = feat.reshape(b, c, -1).mean(dim=2).reshape(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat: Tensor, style_feat: Tensor):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def wavelet_blur(image: Tensor, radius: int):
    """
    Apply wavelet blur to the input tensor.
    """
    # input shape: (1, 3, H, W)
    # convolution kernel
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    # add channel dimensions to the kernel to make it a 4D tensor
    kernel = kernel[None, None]
    # repeat the kernel across all input channels
    kernel = kernel.repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    # apply convolution
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    return output


def wavelet_decomposition(image: Tensor, levels=5):
    """
    Apply wavelet decomposition to the input tensor.
    This function only returns the low frequency & the high frequency.
    """
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2**i
        low_freq = wavelet_blur(image, radius)
        high_freq += image - low_freq
        image = low_freq

    return high_freq, low_freq


def wavelet_reconstruction(content_feat: Tensor, style_feat: Tensor):
    """
    Apply wavelet decomposition, so that the content will have the same color as the style.
    """
    # calculate the wavelet decomposition of the content feature
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat)
    del content_low_freq
    # calculate the wavelet decomposition of the style feature
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)
    del style_high_freq
    # reconstruct the content feature with the style's high frequency
    return content_high_freq + style_low_freq


def normalize_image(image:torch.Tensor):
    return normalize_image_v2(image)
def normalize_image_v1(image:torch.Tensor):
    """
    image with shape [b, 6, 3, 224, 400]
    """
    mean = image.mean(dim=[0, 1, 3, 4], keepdim=True)
    std = image.std(dim=[0, 1, 3, 4], keepdim=True)
    image = (image - mean) / std
    return image

def normalize_image_v2(image:torch.Tensor):
    """
    image with shape [b, 6, 3, 224, 400]
    """
    mean = image.mean(dim=[0, 3, 4], keepdim=True)
    std = image.std(dim=[0, 3, 4], keepdim=True)
    image = (image - mean) / std
    return image

def normalize_image_v0(image:torch.Tensor):
    return image
def normalize_image_v3(image:torch.Tensor):
    mean = image.mean(dim=[0, 3, 4], keepdim=True)
    image = image - mean
    return image


def move_to(obj, device, filter=lambda x: True):
    if torch.is_tensor(obj):
        if filter(obj):
            return obj.to(device)
        else:
            return obj
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device, filter)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device, filter))
        return res
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, DataContainer):
        return move_to(obj.data, device, filter)
    elif isinstance(obj, LiDARInstance3DBoxes):
        return move_to(obj.tensor, device, filter)
    elif obj is None:
        return obj
    else:
        raise TypeError(f"Invalid type {obj.__class__} for move_to.")

def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a quaternion [w, x, y, z] into a 3x3 rotation matrix.
    
    Parameters
    ----------
    quaternion : array-like of shape (4,)
        The quaternion to convert, in the order [w, x, y, z].
        
    Returns
    -------
    R : numpy.ndarray of shape (3, 3)
        The corresponding 3x3 rotation matrix.
    """
    q = np.array(quaternion, dtype=float, copy=True)
    # Normalize the quaternion to avoid numerical issues
    q /= np.linalg.norm(q)

    w, x, y, z = q

    # Compute the rotation matrix
    # Reference for this specific formula:
    # R = [[1 - 2y^2 - 2z^2,     2xy - 2wz,       2xz + 2wy],
    #      [2xy + 2wz,           1 - 2x^2 - 2z^2, 2yz - 2wx],
    #      [2xz - 2wy,           2yz + 2wx,       1 - 2x^2 - 2y^2]]
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y]
    ], dtype=float)

    return R

def dilate(image: torch.Tensor, kernel_size: int = 3, gaussian: bool=False) -> torch.Tensor:
    """
    Dilates a binary 2D image tensor using a square kernel of given size.
    
    Args:
        image: 2D tensor of shape (H, W) with values 0 and 1.
        kernel_size: Size of the square kernel (must be odd).
    
    Returns:
        Dilated 2D tensor with same shape as input.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size should be odd for symmetric dilation.")
    
    # Ensure image is float and add batch & channel dimensions
    if len(image.shape) == 2:
        img = image.float().unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
    else:
        img = image
    assert len(img.shape) == 4, "Input image should have shape (H, W) or (1, 1, H, W)." 
    
    # Create kernel (structuring element)
    kernel = torch.ones(1, 1, kernel_size, kernel_size, dtype=torch.float32, device=image.device)
    if gaussian:
        kernel = create_gaussian_kernel(kernel_size, image.device)
    
    kernel = kernel.repeat(img.shape[1], img.shape[1], 1, 1)

    
    # Apply convolution with padding to maintain size
    padding = kernel_size // 2
    convolved = torch.nn.functional.conv2d(img, kernel, padding=padding)
    
    # Threshold to get binary values and remove added dimensions
    if not gaussian:
        dilated = (convolved > 0).float().squeeze(0).squeeze(0)
    else:
        dilated = convolved.float()
    
    return dilated

def create_gaussian_kernel(kernel_size, image_device):
    """
    Creates a Gaussian kernel with the specified size.

    Args:
        kernel_size (int): The size of the kernel (must be odd).
        image_device (torch.device): The device of the image tensor (to place kernel on same device).

    Returns:
        torch.Tensor: A Gaussian kernel tensor of shape (1, 1, kernel_size, kernel_size).
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd to have a center.")

    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8 # Heuristic for sigma, you can adjust this
    if sigma < 0.1: # To prevent very small sigma for small kernels
        sigma = 0.1

    center = kernel_size // 2
    x, y = torch.meshgrid(torch.arange(0, kernel_size, dtype=torch.float32),
                           torch.arange(0, kernel_size, dtype=torch.float32), indexing='ij')
    gaussian_kernel = torch.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))

    # Normalize to make the kernel sum to 1
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    # gaussian_kernel = gaussian_kernel / gaussian_kernel[center, center] * 1.0
    # import ipdb ; ipdb.set_trace()
    # Reshape to (1, 1, kernel_size, kernel_size) and set dtype and device
    gaussian_kernel = gaussian_kernel.reshape(1, 1, kernel_size, kernel_size).to(dtype=torch.float32, device=image_device)

    return gaussian_kernel

def visualize_class_map(pred_map, save_path="segmentation.png"):
    """
    Convert a tensor [8, 200, 200] representing class scores/probabilities
    into a color-coded segmentation map and save as a PNG image.

    Args:
        pred_map (torch.Tensor): A tensor of shape [8, H, W]
                                 Typically the output of a network, 
                                 e.g. logits or probabilities for each class.
        save_path (str): Where to save the PNG image.

    Returns:
        None (saves an image to disk)
    """
    # pred_map shape: [8, 200, 200]
    # Take argmax across the first dimension (channel dim), 
    # which results in shape [200, 200] with values in [0 .. 7].
    class_indices = torch.argmax(pred_map, dim=0).cpu().numpy()  # shape: (200, 200)

    # Define a color palette for 8 classes (R, G, B).
    # You can customize these colors.
    palette = [
        (0,   0,   0  ),  # Class 0 -> Black
        (128, 0,   0  ),  # Class 1 -> Maroon
        (0,   128, 0  ),  # Class 2 -> Green
        (0,   0,   128),  # Class 3 -> Navy
        (128, 128, 0 ),  # Class 4 -> Olive
        (128, 0,   128),  # Class 5 -> Purple
        (0,   128, 128),  # Class 6 -> Teal
        (255, 165, 0  ),  # Class 7 -> Orange
    ]

    # Create an empty RGB image (H, W, 3) to store colors
    height, width = class_indices.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Map each class index to its corresponding color
    for c, color in enumerate(palette):
        color_image[class_indices == c] = color
    color_image[color_image.shape[0] // 2, color_image.shape[1] // 2] = (255, 255, 255)
    # Convert to PIL image and save
    img = Image.fromarray(color_image, mode="RGB")
    img.save(save_path)
    print(f"Segmentation map saved at: {save_path}")

def sample_gaussian_around_t(t, sigma=10):
    """
    Draws a single integer sample from a normal (Gaussian) distribution
    centered at t, clamped to the range [0, 999].
    
    :param t:     The mean of the distribution (an integer in [0, 999]).
    :param sigma: The standard deviation for the Gaussian distribution.
    :return:      An integer sample in the range [0, 999].
    """
    # Sample from a Gaussian distribution
    sample = random.gauss(mu=t, sigma=sigma)
    # Round the sample to the nearest integer
    sample_rounded = int(round(sample))
    # Clamp the result to [0, 999]
    clamped_sample = max(0, min(999, sample_rounded))
    
    return clamped_sample
def concat_6_view(x):
    # assuming batch size = 1
    # x: [1, 6, 3, 224, 400]
    top_row = torch.cat([x[0, i] for i in range(3)], dim=2)
    bottom_row = torch.cat([x[0, i] for i in range(3, 6)], dim=2)
    return torch.cat([top_row, bottom_row], dim=1)


def obtain_sensor2top(
    nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type="lidar"
):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f"{os.getcwd()}/")[-1]  # relative path
    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }
    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = quaternion_to_rotation_matrix(l2e_r_s)
    e2g_r_s_mat = quaternion_to_rotation_matrix(e2g_r_s)
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    )
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
    return sweep
def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims
    dimensions.
    """
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

class ControlnetUnetWrapper(ModelMixin):
    """As stated in https://github.com/huggingface/accelerate/issues/668, we
    should not use accumulate provided by accelerator, but create a wrapper to
    two modules.
    """

    def __init__(self, controlnet, unet, weight_dtype=torch.float16,
                 unet_in_fp16=True) -> None:
        super().__init__()
        self.controlnet = controlnet
        self.unet = unet
        self.weight_dtype = weight_dtype
        self.unet_in_fp16 = unet_in_fp16

    def forward(self, noisy_latents, timesteps, camera_param,
                encoder_hidden_states, encoder_hidden_states_uncond,
                controlnet_image, **kwargs):
        N_cam = noisy_latents.shape[1]
        kwargs = move_to(
            kwargs, self.weight_dtype, lambda x: x.dtype == torch.float32)

        # fmt: off
        down_block_res_samples, mid_block_res_sample, \
        encoder_hidden_states_with_cam = self.controlnet(
            noisy_latents,  # b, N_cam, 4, H/8, W/8
            timesteps,  # b
            camera_param=camera_param,  # b, N_cam, 189
            encoder_hidden_states=encoder_hidden_states,  # b, len, 768
            encoder_hidden_states_uncond=encoder_hidden_states_uncond,  # 1, len, 768
            controlnet_cond=controlnet_image,  # b, 26, 200, 200
            return_dict=False,
            **kwargs,
        )
        # fmt: on

        # starting from here, we use (B n) as batch_size
        noisy_latents = rearrange(noisy_latents, "b n ... -> (b n) ...")
        if timesteps.ndim == 1:
            timesteps = repeat(timesteps, "b -> (b n)", n=N_cam)

        # Predict the noise residual
        # NOTE: Since we fix most of the model, we cast the model to fp16 and
        # disable autocast to prevent it from falling back to fp32. Please
        # enable autocast on your customized/trainable modules.
        context = contextlib.nullcontext
        context_kwargs = {}
        if self.unet_in_fp16:
            context = torch.cuda.amp.autocast
            context_kwargs = {"enabled": False}
        with context(**context_kwargs):
            model_pred = self.unet(
                noisy_latents,  # b x n, 4, H/8, W/8
                timesteps.reshape(-1),  # b x n
                encoder_hidden_states=encoder_hidden_states_with_cam.to(
                    dtype=self.weight_dtype
                ),  # b x n, len + 1, 768
                # TODO: during training, some camera param are masked.
                down_block_additional_residuals=[
                    sample.to(dtype=self.weight_dtype)
                    for sample in down_block_res_samples
                ],  # all intermedite have four dims: b x n, c, h, w
                mid_block_additional_residual=mid_block_res_sample.to(
                    dtype=self.weight_dtype
                ),  # b x n, 1280, h, w. we have 4 x 7 as mid_block_res
            ).sample

        model_pred = rearrange(model_pred, "(b n) ... -> b n ...", n=N_cam)
        return model_pred
class MagicDrive:
    def __init__(self, sd_path, checkpoint_path, version="trainval", nuscenes_root='/train-syncdata/kaiyuan.tan/nuscenes-download/nuscenes'):

        self.checkpoint_path = checkpoint_path
        self.sd_path = sd_path
        self.nuscenes_root = nuscenes_root
        self.device = torch.device('cuda:0')
        self.weight_dtype = torch.float16
        self.version = version

        self.lpips_loss = lpips.LPIPS(net='vgg').to(self.device) # input scaled to [-1, 1]

        self.unet = UNet2DConditionModelMultiview.from_pretrained(f'{self.checkpoint_path}', subfolder="unet").to(self.device, dtype=self.weight_dtype)
        self.controlnet = BEVControlNetModel.from_pretrained(f'{self.checkpoint_path}', subfolder="controlnet").to(self.device, dtype=self.weight_dtype)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.sd_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.sd_path, subfolder="text_encoder").to(self.device, dtype=self.weight_dtype)
        self.vae = AutoencoderKL.from_pretrained(self.sd_path, subfolder="vae").to(self.device, dtype=self.weight_dtype)

        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.controlnet.requires_grad_(False)
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.sd_path, subfolder="scheduler") # removed noise during step

        self.pipeline = StableDiffusionBEVControlNetPipeline.from_pretrained(
            self.sd_path,
            unet=self.unet,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            controlnet=self.controlnet,
            safety_checker=None,
            feature_extractor=None,
            torch_dtype=torch.float16
        ) 

        # self.nusc_database = NuScenes(version='v1.0-mini', dataroot=self.nuscenes_root)
        self.nusc_database = NuScenes(version=f'v1.0-{self.version}', dataroot=self.nuscenes_root)


        self.controlnet_unet = ControlnetUnetWrapper(self.controlnet, self.unet)

        self.maps = {}
        for location in LOCATIONS:
            self.maps[location] = NuScenesMap(self.nuscenes_root, location)


        # configurations temp
        self.patch_size = (100, 100)
        self.force_all_boxes = True
        self.with_velocity = True
        self.CLASSES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        self.box_type_3d, self.box_mode_3d = get_box_type("LiDAR")
        self.template = 'A driving scene image at {location}. {description}.'
        self.bbox_mode = 'all-xyz'
        self.mgd_order = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']


        data_cfg_224x400 = [{'type': 'LoadMultiViewImageFromFiles', 'to_float32': True}, 
        {'type': 'LoadAnnotations3D', 'with_bbox_3d': True, 'with_label_3d': True, 'with_attr_label': False}, 
        {'type': 'ImageAug3D', 'final_dim': [224, 400], 'resize_lim': [0.25, 0.25], 'bot_pct_lim': [0.0, 0.0], 'rot_lim': [0.0, 0.0], 'rand_flip': False, 'is_train': False}, 
        {'type': 'GlobalRotScaleTrans', 'resize_lim': [1.0, 1.0], 'rot_lim': [0.0, 0.0], 'trans_lim': 0, 'is_train': True}, 
        {'type': 'ObjectNameFilterM', 'classes': ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']}, 
        {'type': 'LoadBEVSegmentationM', 'dataset_root': self.nuscenes_root, 'xbound': [-50.0, 50.0, 0.5], 'ybound': [-50.0, 50.0, 0.5], 'classes': ['drivable_area', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area', 'road_divider', 'lane_divider', 'road_block'], 'object_classes': None, 'aux_data': None, 'cache_file': None}, 
        {'type': 'ReorderMultiViewImagesM', 'order': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'], 'safe': False}, 
        {'type': 'ImageNormalize', 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}, 
        {'type': 'DefaultFormatBundle3D', 'classes': ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']}, 
        {'type': 'Collect3D', 'keys': ['img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_masks_bev'], 'meta_keys': ['camera_intrinsics', 'lidar2ego', 'lidar2camera', 'camera2lidar', 'lidar2image', 'img_aug_matrix'], 'meta_lis_keys': ['timeofday', 'location', 'description', 'filename', 'token']}]


        data_cfg_424x800 = [{'type': 'LoadMultiViewImageFromFiles', 'to_float32': True}, 
        {'type': 'LoadAnnotations3D', 'with_bbox_3d': True, 'with_label_3d': True, 'with_attr_label': False}, 
        {'type': 'ImageAug3D', 'final_dim': [424, 800], 'resize_lim': [0.25, 0.25], 'bot_pct_lim': [0.0, 0.0], 'rot_lim': [0.0, 0.0], 'rand_flip': False, 'is_train': False}, 
        {'type': 'GlobalRotScaleTrans', 'resize_lim': [1.0, 1.0], 'rot_lim': [0.0, 0.0], 'trans_lim': 0, 'is_train': True}, 
        {'type': 'ObjectNameFilterM', 'classes': ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']}, 
        {'type': 'LoadBEVSegmentationM', 'dataset_root': self.nuscenes_root, 'xbound': [-50.0, 50.0, 0.25], 'ybound': [-50.0, 50.0, 0.25], 'classes': ['drivable_area', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area', 'road_divider', 'lane_divider', 'road_block'], 'object_classes': None, 'aux_data': None, 'cache_file': None}, 
        {'type': 'ReorderMultiViewImagesM', 'order': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'], 'safe': False}, 
        {'type': 'ImageNormalize', 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}, 
        {'type': 'DefaultFormatBundle3D', 'classes': ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']}, 
        {'type': 'Collect3D', 'keys': ['img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_masks_bev'], 'meta_keys': ['camera_intrinsics', 'lidar2ego', 'lidar2camera', 'camera2lidar', 'lidar2image', 'img_aug_matrix'], 'meta_lis_keys': ['timeofday', 'location', 'description', 'filename', 'token']}]


        # data_cfg_424x800_train = [{'type': 'LoadMultiViewImageFromFiles', 'to_float32': True}, 
        # {'type': 'LoadAnnotations3D', 'with_bbox_3d': True, 'with_label_3d': True, 'with_attr_label': False}, 
        # {'type': 'ImageAug3D', 'final_dim': [424, 800], 'resize_lim': [0.5, 0.5], 'bot_pct_lim': [0.0, 0.0], 'rot_lim': None, 'rand_flip': False, 'is_train': False}, 
        # {'type': 'GlobalRotScaleTrans', 'resize_lim': [1.0, 1.0], 'rot_lim': [0.0, 0.0], 'trans_lim': 0, 'is_train': True}, 
        # {'type': 'ObjectNameFilterM', 'classes': ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']}, 
        # {'type': 'LoadBEVSegmentationM', 'dataset_root': 'data/nuscenes/', 'xbound': [-50.0, 50.0, 0.25], 'ybound': [-50.0, 50.0, 0.25], 'classes': ['drivable_area', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area', 'road_divider', 'lane_divider', 'road_block'], 'object_classes': None, 'aux_data': None, 'cache_file': 'data/nuscenes_mmdet3d_2/../nuscenes_map_aux/train_26x400x400_map_aux_full.h5'}, 
        # {'type': 'RandomFlip3DwithViews', 'flip_ratio': 0.0, 'direction': None}, 
        # {'type': 'ReorderMultiViewImagesM', 'order': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'], 'safe': False}, 
        # {'type': 'ImageNormalize', 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}, 
        # {'type': 'DefaultFormatBundle3D', 'classes': ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']}, 
        # {'type': 'Collect3D', 'keys': ['img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_masks_bev'], 'meta_keys': ['camera_intrinsics', 'lidar2ego', 'lidar2camera', 'camera2lidar', 'lidar2image', 'img_aug_matrix'], 'meta_lis_keys': ['timeofday', 'location', 'description', 'filename', 'token']}]
        
        # data_cfg_424x800_test = [{'type': 'LoadMultiViewImageFromFiles', 'to_float32': True}, 
        # {'type': 'LoadAnnotations3D', 'with_bbox_3d': True, 'with_label_3d': True, 'with_attr_label': False}, 
        # {'type': 'ImageAug3D', 'final_dim': [424, 800], 'resize_lim': [0.5, 0.5], 'bot_pct_lim': [0.0, 0.0], 'rot_lim': [0.0, 0.0], 'rand_flip': False, 'is_train': False}, 
        # {'type': 'GlobalRotScaleTrans', 'resize_lim': [1.0, 1.0], 'rot_lim': [0.0, 0.0], 'trans_lim': 0, 'is_train': True}, 
        # {'type': 'ObjectNameFilterM', 'classes': ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']}, 
        # {'type': 'LoadBEVSegmentationM', 'dataset_root': 'data/nuscenes/', 'xbound': [-50.0, 50.0, 0.25], 'ybound': [-50.0, 50.0, 0.25], 'classes': ['drivable_area', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area', 'road_divider', 'lane_divider', 'road_block'], 'object_classes': None, 'aux_data': None, 'cache_file': 'data/nuscenes_mmdet3d_2/../nuscenes_map_aux/val_26x400x400_map_aux_full.h5'}, 
        # {'type': 'ReorderMultiViewImagesM', 'order': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'], 'safe': False}, 
        # {'type': 'ImageNormalize', 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}, 
        # {'type': 'DefaultFormatBundle3D', 'classes': ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']}, 
        # {'type': 'Collect3D', 'keys': ['img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_masks_bev'], 'meta_keys': ['camera_intrinsics', 'lidar2ego', 'lidar2camera', 'camera2lidar', 'lidar2image', 'img_aug_matrix'], 'meta_lis_keys': ['timeofday', 'location', 'description', 'filename', 'token']}]

        self.data_pipeline_cfg = data_cfg_224x400

        self.data_pipeline = []

        self.info_cache = {}


    def get_prefix(self):
        return prefix_current_time

    def save_image0_from_pixels(self, pixels, path):
        pixels = concat_6_view(pixels.detach().float().cpu())

        pixels = (pixels / 2 + 0.5).clamp(0, 1) * 255
        pixels = pixels.type(torch.uint8)
        base_dir = os.path.dirname(os.path.join(prefix_current_time, path))
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        torchvision.io.write_png(pixels, os.path.join(prefix_current_time, path))
    def save_image0_from_latents(self, latents, path):
        with torch.no_grad():
            latents = latents.detach()
            bs = len(latents)
            latents = 1 / self.vae.config.scaling_factor * latents
            latents = rearrange(latents, 'b c ... -> (b c) ...')
            image = self.vae.decode(latents).sample
            image = rearrange(image, '(b c) ... -> b c ...', b=bs)
            image = (image / 2 + 0.5).clamp(0, 1)
            image = concat_6_view(image.cpu())
            image = image * 255
            image = image.type(torch.uint8)
            torchvision.io.write_png(image, os.path.join(prefix_current_time, path))


    def set_scene(self, scene_idx):
        return
        # self.scene_idx = scene_idx
        # self.camera_param = None
        # self.input_ids = None
        # self.uncond_ids = None
        
        # sample_token = 'd2033528dce44366925af0c0eefee398'

        # self.scene_token = self.nusc_database.scene[scene_idx]['token']
        # self.log_token = self.nusc_database.get('log', self.nusc_database.get('scene', self.scene_token)['log_token'])['token']
        # self.location = self.nusc_database.get('log', self.log_token)['location']
        # self.map = self.maps[self.location]

        # first_sample_token = self.nusc_database.get('scene', self.scene_token)['first_sample_token']
        # lidar_data_token = self.nusc_database.get('sample', first_sample_token)['data']['LIDAR_TOP']
        # calibrated_sensor = self.nusc_database.get('calibrated_sensor', self.nusc_database.get('sample_data', lidar_data_token)['calibrated_sensor_token'])
        # lidar2ego_rotation, lidar2ego_translation = calibrated_sensor['rotation'], calibrated_sensor['translation']

        # lidar2ego = np.eye(4)
        # lidar2ego[:3, :3] = quaternion_to_rotation_matrix(lidar2ego_rotation)
        # lidar2ego[:3, 3] = lidar2ego_translation
        # self.lidar2ego = lidar2ego

        # self.cam2lidar = []
        # self.intrinsics = []
        # self.cam_name = []

        # self.mgd_order = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
        # for cam_name in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
        #     cam_token = self.nusc_database.get('sample_data', self.nusc_database.get('sample', first_sample_token)['data'][cam_name])['token']
        #     cam = self.nusc_database.get('calibrated_sensor', self.nusc_database.get('sample_data', cam_token)['calibrated_sensor_token'])
        #     cam2ego_rotation, cam2ego_translation = cam['rotation'], cam['translation']
        #     cam2ego = np.eye(4)
        #     cam2ego[:3, :3] = quaternion_to_rotation_matrix(cam2ego_rotation)
        #     cam2ego[:3, 3] = cam2ego_translation

        #     cam2lidar = np.linalg.inv(self.lidar2ego) @ cam2ego
        #     self.cam2lidar.append(cam2lidar)
        #     self.intrinsics.append(np.array(cam['camera_intrinsic']).astype(np.float32))

        # # masks = self.maps[self.location].get_map_mask(

        # # )

    def _add_noise(self, latents, noise, timesteps):
        if timesteps.ndim == 2:
            B, N = latents.shape[:2]
            bc2b = partial(rearrange, pattern="b n ... -> (b n) ...")
            b2bc = partial(rearrange, pattern="(b n) ... -> b n ...", b=B)
        elif timesteps.ndim == 1:
            def bc2b(x): return x
            def b2bc(x): return x
        noisy_latents = self.noise_scheduler.add_noise(
            bc2b(latents), bc2b(noise), bc2b(timesteps)
        )
        noisy_latents = b2bc(noisy_latents)
        return noisy_latents

    def _remove_noise(self, noisy_latents, noise, timesteps):
        if timesteps.ndim == 2:
            B, N = noisy_latents.shape[:2]
            bc2b = partial(rearrange, pattern="b n ... -> (b n) ...")
            b2bc = partial(rearrange, pattern="(b n) ... -> b n ...", b=B)
        elif timesteps.ndim == 1:
            def bc2b(x): return x
            def b2bc(x): return x
        restored_latents = self.noise_scheduler.remove_noise(
            bc2b(noisy_latents), bc2b(noise), bc2b(timesteps)
        )
        restored_latents = b2bc(restored_latents)
        return restored_latents
    def get_info(self, sample_token, shift_x):
        print(f"current sample token: {sample_token}")
        if sample_token not in self.info_cache:
            info = self.get_sample_info(sample_token)
            info['shift_x'] = shift_x
            info = self.data_pipeline_fn(info)
            info = self.collate_fn_preprocess(info)
            self.info_cache[sample_token] = info
        return self.info_cache[sample_token]
    
    def visualize_info_bbox(self, batch, save_prefix="./"):
        bboxes = batch['kwargs']['bboxes_3d_data']['bboxes'].clone()

        # bbox projection to image
        import cv2
        def project_3d_to_image(bboxes, lidar2img, img_shape):
            """
            bboxes: Tensor [N,8,3] (x,y,z in LiDAR坐标系)
            lidar2img: Tensor [4,4] 投影矩阵
            img_shape: (H, W, C) 图像尺寸
            return: 
                projected_boxes: List[N][8][2] 图像坐标 (过滤不可见点)
                valid_mask: Tensor [N,8] 有效点标记
            """
            # 齐次坐标扩展
            ones = torch.ones(bboxes.shape[0], 8, 1, device=bboxes.device)
            homo_coords = torch.cat([bboxes, ones], dim=-1)  # [N,8,4]

            # 坐标系转换
            camera_coords = torch.einsum('ij,nkj->nki', lidar2img, homo_coords)  # [N,8,4]

            # 透视除法 (排除深度<=0的点)
            z = camera_coords[..., 2]
            valid_depth = z > 1e-5  # 深度有效性判断
            image_coords = camera_coords / z.unsqueeze(-1)  # [N,8,4]

            # 坐标有效性判断
            x, y = image_coords[..., 0], image_coords[..., 1]
            valid_x = (x >= 0) & (x < img_shape[1])
            valid_y = (y >= 0) & (y < img_shape[0])
            valid_mask = valid_depth & valid_x & valid_y  # [N,8]

            return image_coords[..., :2].cpu().numpy(), valid_mask.cpu().numpy()
        
        def draw_3d_boxes(img_tensor, projected_boxes, valid_mask, color=(0,255,0), thickness=10):
            """
            参数说明：
            img_tensor: torch.Tensor [3, H, W] (RGB格式)
            projected_boxes: np.ndarray [N, 8, 2] 投影后的图像坐标
            valid_mask: np.ndarray [N, 8] 有效点标记
            返回：
            torch.Tensor [3, H, W] (与输入同格式)
            """
            # 转换张量为OpenCV格式
            img_np = img_tensor.permute(1, 2, 0).contiguous().detach().cpu().numpy()
            img_np = (img_np + 1) / 2  # 反归一化
            if img_np.max() <= 1.0:  # 自动检测归一化输入
                img_np = (img_np * 255).clip(0, 255)
            img_cv = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)

            # 绘制逻辑
            for box, mask in zip(projected_boxes, valid_mask):
                visible_edges = []
                
                # 底面四边形有效性判断
                if np.sum(mask[[0,1,2,3]]) >= 3:
                    visible_edges += [(0,1),(1,2),(2,3),(3,0)]
                
                # 顶面四边形有效性判断
                if np.sum(mask[[4,5,6,7]]) >= 3:
                    visible_edges += [(4,5),(5,6),(6,7),(7,4)]
                
                # 立柱有效性判断
                vertical_edges = [(i,i+4) for i in range(4)]
                for i,j in vertical_edges:
                    if mask[i] and mask[j]:
                        visible_edges.append((i,j))

                # 坐标转换与绘制
                for (i,j) in visible_edges:
                    pt1 = tuple(np.round(box[i]).astype(int))
                    pt2 = tuple(np.round(box[j]).astype(int))
                    img_cv = cv2.line(img_cv, pt1, pt2, color, thickness)

            # 转换回原始张量格式
            result_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            if img_tensor.dtype == torch.float32:  # 保持输入数据类型
                result_rgb = result_rgb.astype(np.float32) / 255.0
                result_rgb = result_rgb * 2 - 1
                
            return torch.from_numpy(result_rgb).permute(2, 0, 1).to(img_tensor.device)
        
        lidar2images = batch['meta_data']['lidar2image'][0].data    
        lidar_bboxes_all = bboxes[0]
        bbox_height = lidar_bboxes_all[:, :, 1, 2] - lidar_bboxes_all[:, :, 0, 2]

        lidar_bboxes_all[..., 2] -= bbox_height.unsqueeze(-1) / 2
        # mgd assumes a translation of 0.5 in z
        
        with torch.no_grad():
            for i in range(6):
                lidar_bboxes = lidar_bboxes_all[i]
                image_bboxes, valid_mask = project_3d_to_image(lidar_bboxes, lidar2images[i].to(lidar_bboxes.device), (900, 1600))
                img = F.interpolate(batch['pixel_values'][0][i].detach().unsqueeze(0), size=(900, 1600), mode='bilinear').squeeze(0)
                img = draw_3d_boxes(img, image_bboxes, valid_mask)
                import torchvision
                img = img * 0.5 + 0.5
                img = img * 255
                img = img.cpu().type(torch.uint8)
                torchvision.io.write_png(img, f"{save_prefix}_bbox_{i}.png")

    def get_loss(self, pred_rgb, sample_token, timestep, shift_x, step, method, inpainting_mask=None, **kwargs):
        """
        pred_rgb: [1, 6, 3, 224, 400]
        """
        # if step % 100 == 0:
        #     save_dict = {'pred_rgb': pred_rgb, 'sample_token': sample_token, 'timestep': timestep, 'shift_x': shift_x, 'step': step, 'method': method, 'inpainting_mask': inpainting_mask}
        #     torch.save(save_dict, f"save_dict_{kwargs['scene_idx']}_{step}_{sample_token}.pt")
        
        # get sample info
        info = self.get_info(sample_token, shift_x)

        # hack pixel_values with pred_rgb
        info['pixel_values'] = pred_rgb

        # move to device
        batch = move_to(info, self.device)
        inpainting_mask = inpainting_mask.to(self.device)

        # visualize bev segmentation map
        # visualize_class_map(batch['bev_map_with_aux'][0], "segmentation.png")

        # hack bboxes TODO
        bboxes = batch['kwargs']['bboxes_3d_data']['bboxes'].clone()
        bboxes[:, :, :, :, 0] += shift_x
        batch['kwargs']['bboxes_3d_data']['bboxes'] = bboxes
        


        # self.visualize_info_bbox(batch, save_prefix=f"bbox_{kwargs['scene_idx']}_{step}_{sample_token}")
        # set controlnet to unconditional
        self.controlnet_unet.train()
        self.controlnet_unet.controlnet.drop_cond_ratio = 0.0

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]
        encoder_hidden_states_uncond = self.text_encoder(
            batch
            ["uncond_ids"])[0]

        # bev seg for conditioning
        controlnet_image = batch["bev_map_with_aux"].to(
            dtype=self.weight_dtype)
        

        camera_param = batch["camera_param"].to(self.weight_dtype)
        N_cam = batch["pixel_values"].shape[1]


        image_x = batch['pixel_values'].clip(-1, 1)
        # Convert images to latent space
        sds_img_vae = self.vae.encode(
            rearrange(image_x, "b n c h w -> (b n) c h w").to(
                dtype=self.weight_dtype
            )
        ).latent_dist.sample()
        sds_img_vae = sds_img_vae * self.vae.config.scaling_factor
        sds_img_vae = rearrange(sds_img_vae, "(b n) c h w -> b n c h w", n=N_cam)

        inpainting_mask_vae = F.interpolate(inpainting_mask, size=sds_img_vae.shape[-2:], mode='bilinear').unsqueeze(0).half()
        inpainting_mask_vae = torch.cat([inpainting_mask_vae, inpainting_mask_vae[:, :, :1, :, :]], dim=2)
        # self.save_image0_from_pixels(inpainting_mask_vae[:, :, :3, :, :], f"inpainting_mask_vae_{step}.png")


        # import ipdb ; ipdb.set_trace()

        # save_dict = {'pred_rgb': pred_rgb, 'sample_token': sample_token, 'timestep': timestep, 'shift_x': shift_x, 'step': step, 'method': method, 'inpainting_mask': inpainting_mask}
        # torch.save(save_dict, f"save_dict.pt")
        if method == 'sds':
            batch_size = 1

            camera_param_batched = camera_param.repeat(batch_size, 1, 1, 1)
            encoder_hidden_states_batched = encoder_hidden_states.repeat(batch_size, 1, 1)
            # encoder_hidden_states_uncond_batched = encoder_hidden_states_uncond.repeat(batch_size, 1, 1)
            controlnet_image_batched = controlnet_image.repeat(batch_size, 1, 1, 1)
            bboxes_3d_data_batched = batch['kwargs']['bboxes_3d_data']
            bboxes_3d_data_batched['bboxes'] = bboxes_3d_data_batched['bboxes'].repeat(batch_size, 1, 1, 1, 1)
            bboxes_3d_data_batched['classes'] = bboxes_3d_data_batched['classes'].repeat(batch_size, 1, 1)
            bboxes_3d_data_batched['masks'] = bboxes_3d_data_batched['masks'].repeat(batch_size, 1, 1)


            sds_img_vae_batched = sds_img_vae.repeat(batch_size, 1, 1, 1, 1)
            
            # sds_img_vae = sds_img_vae * (1 - mask) + input_latents * mask
            # sds_img_vae = sds_img_vae * inpainting_mask_vae.half() + input_latents * (1 - inpainting_mask_vae).half()
            # if step % 100 == 0:
            #     self.save_image0_from_latents(sds_img_vae, f"sds_img_vae_{step}.png")


            with torch.no_grad():

                # add noise at timestep t
                noise = torch.randn_like(sds_img_vae_batched)
                noisy_latents = self._add_noise(sds_img_vae_batched, noise, torch.tensor(timestep).unsqueeze(0).to(self.device))


                # predict noise residual epsilon
                self.controlnet_unet.controlnet.drop_cond_ratio = 0.0
                model_pred = self.controlnet_unet(
                    noisy_latents, torch.tensor(timestep).unsqueeze(0).to(self.device).repeat(batch_size), camera_param_batched, encoder_hidden_states_batched, encoder_hidden_states_uncond,
                    controlnet_image_batched,
                    bboxes_3d_data=bboxes_3d_data_batched,
                )



                self.controlnet_unet.controlnet.drop_cond_ratio = 1.0
                model_pred_uncod = self.controlnet_unet(
                    noisy_latents, torch.tensor(timestep).unsqueeze(0).to(self.device).repeat(batch_size), camera_param_batched, encoder_hidden_states_batched, encoder_hidden_states_uncond,
                    controlnet_image_batched,
                    bboxes_3d_data=bboxes_3d_data_batched,
                )

                model_pred = model_pred_uncod + 2.0 * (model_pred - model_pred_uncod) # cfg


                
                # method 0: direct mse with denoised latent
                target_latent = self._remove_noise(noisy_latents, model_pred, torch.tensor(timestep).unsqueeze(0).to(self.device).repeat(batch_size)).half()
                target_latent = target_latent.mean(dim=0, keepdim=True)
                
                # method 1: from SDS pseudo code
                # g = torch.dot((model_pred.half() - noise).detach().flatten(), sds_img_vae.flatten())
                # g.backward()



                # method 2: from DreamGaussian
                # wt = 1.0 - self.noise_scheduler.alphas_cumprod[timestep].item()
                # grad = wt * (model_pred.half() - noise)
                # grad = torch.nan_to_num(grad)
                # target_latent = (sds_img_vae - grad).detach()
            loss = 0.5 * F.mse_loss(sds_img_vae, target_latent.detach(), reduction='sum')

            # make sure the image is consistent with the decoder output
            # image consistency loss with sds_img_vae
            # with torch.no_grad():
            #     target_latent_detached = sds_img_vae.detach()
            #     bs = len(target_latent_detached)
            #     target_latent_detached = 1 / self.vae.config.scaling_factor * target_latent_detached
            #     target_latent_detached = rearrange(target_latent_detached, 'b c ... -> (b c) ...')
            #     target_img = self.vae.decode(target_latent_detached).sample
            #     target_img = rearrange(target_img, '(b c) ... -> b c ...', b=bs)
            # image_consis_loss = F.mse_loss(target_img.detach(), image_x.half(), reduction='mean') * 2e4
            # loss += image_consis_loss



            # # image consistency loss with target_latent
            # with torch.no_grad():
            #     target_latent_detached = target_latent.detach()
            #     bs = len(target_latent_detached)
            #     target_latent_detached = 1 / self.vae.config.scaling_factor * target_latent_detached
            #     target_latent_detached = rearrange(target_latent_detached, 'b c ... -> (b c) ...')
            #     target_img = self.vae.decode(target_latent_detached).sample
            #     target_img = rearrange(target_img, '(b c) ... -> b c ...', b=bs)
            # image_consis_loss = F.mse_loss(target_img.detach(), image_x.half(), reduction='mean') * 2e4
            # loss += image_consis_loss

            # print("step: {}, loss: {}, image_consis_loss: {}, timestep: {}".format(step, loss, image_consis_loss, timestep))
            
            return loss, {'target_latent': target_latent.detach(), 'sds_img_vae': sds_img_vae.detach()}
        elif method == 'direct':
            with torch.no_grad():
                noise = torch.randn_like(sds_img_vae)
                noisy_latents = self._add_noise(sds_img_vae, noise, torch.tensor(timestep).unsqueeze(0).to(self.device))
                # denoising loop
                step_ratio = timestep // 50
                timesteps = (np.arange(0, 50) * step_ratio).round()[::-1].copy().astype(np.int64)
                self.noise_scheduler.set_timesteps(timesteps=timesteps)
                for i in timesteps:
                    self.controlnet_unet.controlnet.drop_cond_ratio = 0.0
                    model_pred = self.controlnet_unet(
                        noisy_latents, torch.tensor(i).unsqueeze(0).to(self.device), camera_param, encoder_hidden_states,
                        encoder_hidden_states_uncond, controlnet_image,
                        **batch['kwargs'],
                    )
                    self.controlnet_unet.controlnet.drop_cond_ratio = 1.0
                    model_pred_uncod = self.controlnet_unet(
                        noisy_latents, torch.tensor(i).unsqueeze(0).to(self.device), camera_param, encoder_hidden_states,
                        encoder_hidden_states_uncond, controlnet_image,
                        **batch['kwargs'],
                    )

                    model_pred = model_pred_uncod + 2 * (model_pred - model_pred_uncod)
                    output = self.noise_scheduler.step(model_pred[0], i, noisy_latents[0], inpainting_mask_vae=inpainting_mask_vae[0].half(), gt_vae_latents=sds_img_vae[0], stochastic=False)
                    noisy_latents = output['prev_sample'].unsqueeze(0).half()
                self.save_image0_from_latents(noisy_latents, f"noisy_latents_{step}_{sample_token}.png")
            noisy_latents = noisy_latents.detach()
            bs = len(noisy_latents)
            noisy_latents = 1 / self.vae.config.scaling_factor * noisy_latents
            noisy_latents = rearrange(noisy_latents, 'b c ... -> (b c) ...')
            image = self.vae.decode(noisy_latents).sample
            image = rearrange(image, '(b c) ... -> b c ...', b=bs)
            # image = (image / 2 + 0.5).clamp(0, 1)

            return F.mse_loss(image, batch['pixel_values'].half(), reduction='mean'), {'target_image': image.detach()}
        # elif method == 'multistep':
        #     with torch.no_grad():
        #         num_ts = 10
        #         step_ratio = timestep // num_ts
        #         timesteps = (np.arange(0, num_ts) * step_ratio).round()[::-1].copy().astype(np.int64)
        #         self.noise_scheduler.set_timesteps(timesteps=timesteps)

        #         noise = torch.randn_like(sds_img_vae)
        #         noisy_latents = self._add_noise(sds_img_vae, noise, torch.tensor(timesteps[0]).unsqueeze(0).to(self.device))
        #         # denoising loop
        #         for i in timesteps:
        #             self.controlnet_unet.controlnet.drop_cond_ratio = 0.0
        #             model_pred = self.controlnet_unet(
        #                 noisy_latents, torch.tensor(i).unsqueeze(0).to(self.device), camera_param, encoder_hidden_states,
        #                 encoder_hidden_states_uncond, controlnet_image,
        #                 **batch['kwargs'],
        #             )
        #             self.controlnet_unet.controlnet.drop_cond_ratio = 1.0
        #             model_pred_uncod = self.controlnet_unet(
        #                 noisy_latents, torch.tensor(i).unsqueeze(0).to(self.device), camera_param, encoder_hidden_states,
        #                 encoder_hidden_states_uncond, controlnet_image,
        #                 **batch['kwargs'],
        #             )

        #             model_pred = model_pred_uncod + 2 * (model_pred - model_pred_uncod)
        #             output = self.noise_scheduler.step(model_pred[0], i, noisy_latents[0], inpainting_mask_vae=inpainting_mask_vae[0].half(), gt_vae_latents=sds_img_vae[0])
        #             # noisy_latents = output['pred_original_sample'].unsqueeze(0).half()
        #             noisy_latents = output['prev_sample'].unsqueeze(0).half()
        #         self.save_image0_from_latents(noisy_latents, f"noisy_latents_{step}.png")
        #     noisy_latents = noisy_latents.detach()
        #     noisy_latents_sds = noisy_latents.clone()
        #     bs = len(noisy_latents)
        #     noisy_latents = 1 / self.vae.config.scaling_factor * noisy_latents
        #     noisy_latents = rearrange(noisy_latents, 'b c ... -> (b c) ...')
        #     image = self.vae.decode(noisy_latents).sample
        #     image = rearrange(image, '(b c) ... -> b c ...', b=bs)
        #     # image = (image / 2 + 0.5).clamp(0, 1)


        #     image = normalize_image(image.float())
        #     batch['pixel_values'] = normalize_image(batch['pixel_values'])


        #     loss_mse = F.mse_loss(batch['pixel_values'], image.float(), reduction='mean')

        #     loss_l1 = F.l1_loss(batch['pixel_values'], image.float(), reduction='mean')

        #     # loss_latent_mse = F.mse_loss(sds_img_vae, noisy_latents_sds.detach(), reduction='sum')

        #     loss_lpips = self.lpips_loss(batch['pixel_values'].float().squeeze(0), image.float().squeeze(0)).mean()
        #     wt = 1.0 - self.noise_scheduler.alphas_cumprod[timestep].item()
        #     total_loss = loss_l1 + 0.1 * loss_lpips
        #     return total_loss, {'target_image': image.detach(), 'target_latent': noisy_latents_sds.detach()}
        elif method == 'anneal':
            with torch.no_grad():
                resample = 10
                R_guide = 10

                self.noise_scheduler.set_timesteps(num_inference_steps=25)
                timesteps = self.noise_scheduler.timesteps
                print(timesteps)
                x_t = torch.randn_like(sds_img_vae)
                for i, t in enumerate(timesteps):
                    if i < 25:
                        for r in range(resample):
                            self.controlnet_unet.controlnet.drop_cond_ratio = 0.0
                            model_pred = self.controlnet_unet(
                                x_t, torch.tensor(t).unsqueeze(0).to(self.device), camera_param, encoder_hidden_states,
                                encoder_hidden_states_uncond, controlnet_image,
                                **batch['kwargs'],
                            )
                            self.controlnet_unet.controlnet.drop_cond_ratio = 1.0
                            model_pred_uncod = self.controlnet_unet(
                                x_t, torch.tensor(t).unsqueeze(0).to(self.device), camera_param, encoder_hidden_states,
                                encoder_hidden_states_uncond, controlnet_image,
                                **batch['kwargs'],
                            )

                            model_pred = model_pred_uncod + 2 * (model_pred - model_pred_uncod)
                            if r < R_guide:
                                output = self.noise_scheduler.step(model_pred[0], t, x_t[0], inpainting_mask_vae=inpainting_mask_vae[0].half(), gt_vae_latents=sds_img_vae[0])
                            else:
                                output = self.noise_scheduler.step(model_pred[0], t, x_t[0])
                            x_t_1 = output['prev_sample'].unsqueeze(0).half()
                            x0 = output['pred_original_sample'].unsqueeze(0).half().detach()
                            self.save_image0_from_latents(x0, f"noisy_latents_{t}_{r}.png")
                            x_t = self._add_noise(x0, torch.randn_like(x0), torch.tensor(t).unsqueeze(0).to(self.device))
                        
                        x_t = x_t_1
                    else:
                        self.controlnet_unet.controlnet.drop_cond_ratio = 0.0
                        model_pred = self.controlnet_unet(
                            x_t, torch.tensor(t).unsqueeze(0).to(self.device), camera_param, encoder_hidden_states,
                            encoder_hidden_states_uncond, controlnet_image,
                            **batch['kwargs'],
                        )
                        self.controlnet_unet.controlnet.drop_cond_ratio = 1.0
                        model_pred_uncod = self.controlnet_unet(
                            x_t, torch.tensor(t).unsqueeze(0).to(self.device), camera_param, encoder_hidden_states,
                            encoder_hidden_states_uncond, controlnet_image,
                            **batch['kwargs'],
                        )
                        model_pred = model_pred_uncod + 2 * (model_pred - model_pred_uncod)
                        output = self.noise_scheduler.step(model_pred[0], t, x_t[0])
                        x_t_1 = output['prev_sample'].unsqueeze(0).half()
                        x0 = output['pred_original_sample'].unsqueeze(0).half().detach()
                        self.save_image0_from_latents(x0, f"noisy_latents_{t}.png")
                        x_t = x_t_1
        elif method == 'repaint':
            with torch.no_grad():
                resample = 5

                self.noise_scheduler.set_timesteps(num_inference_steps=25)
                timesteps = self.noise_scheduler.timesteps
                print(timesteps)
                x_t = torch.randn_like(sds_img_vae)
                for i, t in enumerate(timesteps):
                    num_resample = 1 if i < 15 else 5
                    for r in range(num_resample):
                        x_t_1_known = self._add_noise(sds_img_vae, torch.randn_like(sds_img_vae), torch.tensor(t).unsqueeze(0).to(self.device)) if i < len(timesteps) - 1 else sds_img_vae # mask sure the last step matches the known input exactly
                        # x_t_1_known = sds_img_vae
                        self.controlnet_unet.controlnet.drop_cond_ratio = 0.0
                        model_pred = self.controlnet_unet(
                            x_t, torch.tensor(t).unsqueeze(0).to(self.device), camera_param, encoder_hidden_states,
                            encoder_hidden_states_uncond, controlnet_image,
                            **batch['kwargs'],
                        )
                        self.controlnet_unet.controlnet.drop_cond_ratio = 1.0
                        model_pred_uncod = self.controlnet_unet(
                            x_t, torch.tensor(t).unsqueeze(0).to(self.device), camera_param, encoder_hidden_states,
                            encoder_hidden_states_uncond, controlnet_image,
                            **batch['kwargs'],
                        )

                        model_pred = model_pred_uncod + 2 * (model_pred - model_pred_uncod)
                        output = self.noise_scheduler.step(model_pred[0], t, x_t[0], inpainting_mask_vae=inpainting_mask_vae[0].half(), gt_vae_latents=sds_img_vae[0])
                        x_t_1_unknown = output['prev_sample'].unsqueeze(0).half()

                        ### saving
                        x_0 = output['pred_original_sample'].unsqueeze(0).half().detach()
                        x_0 = x_0 * inpainting_mask_vae + sds_img_vae * (1 - inpainting_mask_vae)
                        self.save_image0_from_latents(x_0, f"noisy_latents_{t}_{r}.png")
                        ###
                        x_t_1 = x_t_1_unknown * inpainting_mask_vae + x_t_1_known * (1 - inpainting_mask_vae)
                        if i < len(timesteps) - 1 and r < resample - 1:
                            beta_t_1 = self.noise_scheduler.betas[t - 1]
                            x_t = x_t_1 * (1 - beta_t_1) ** 0.5 + torch.randn_like(x_t_1) * beta_t_1 ** 0.5
                    
                    x_t = x_t_1
        elif method == 'multistep':
            with torch.no_grad():
                resample = kwargs['resample']

                num_ts = kwargs['num_ts']
                step_ratio = kwargs['ts'] // num_ts
                timesteps = (np.arange(0, num_ts) * step_ratio).round()[::-1].copy().astype(np.int64)
                self.noise_scheduler.set_timesteps(timesteps=timesteps)

                noise = torch.randn_like(sds_img_vae)
                x_t = self._add_noise(sds_img_vae, noise, torch.tensor(timesteps[0]).unsqueeze(0).to(self.device))

                if kwargs['inmask_g_scale'] == 0:
                    inmask_g_scale = np.zeros(len(timesteps))
                else:
                    inmask_g_scale = np.linspace(1, 0, len(timesteps)) ** kwargs['inmask_g_scale']

                # x_t = torch.randn_like(sds_img_vae)
                for i, t in enumerate(timesteps):
                    current_inpainting_mask = (inpainting_mask_vae[0].half() - inmask_g_scale[i]).clip(0, 1)
                    # inmask_g_scale = inmask_g_scale * 0.5
                    for r in range(resample):
                        self.controlnet_unet.controlnet.drop_cond_ratio = 0.0
                        model_pred = self.controlnet_unet(
                            x_t, torch.tensor(t).unsqueeze(0).to(self.device), camera_param, encoder_hidden_states,
                            encoder_hidden_states_uncond, controlnet_image,
                            **batch['kwargs'],
                        )
                        self.controlnet_unet.controlnet.drop_cond_ratio = 1.0
                        model_pred_uncod = self.controlnet_unet(
                            x_t, torch.tensor(t).unsqueeze(0).to(self.device), camera_param, encoder_hidden_states,
                            encoder_hidden_states_uncond, controlnet_image,
                            **batch['kwargs'],
                        )

                        model_pred = model_pred_uncod + kwargs['cfg_scale'] * (model_pred - model_pred_uncod)
                        output = self.noise_scheduler.step(model_pred[0], t, x_t[0], inpainting_mask_vae=current_inpainting_mask, gt_vae_latents=sds_img_vae[0], stochastic=kwargs['stochastic'])
                        x_t_1 = output['prev_sample'].unsqueeze(0).half()
                        x0 = output['pred_original_sample'].unsqueeze(0).half().detach()
                        # self.save_image0_from_latents(x0, f"noisy_latents_{t}_{r}.png")
                        if t == 0 or r == resample - 1:
                            break
                        # x_t = self._add_noise(x0, torch.randn_like(x0), torch.tensor(t).unsqueeze(0).to(self.device))

                        # problematic (less noise is added)
                        # beta_t_1 = self.noise_scheduler.betas[t - 1]
                        # x_t = x_t_1 * (1 - beta_t_1) ** 0.5 + torch.randn_like(x_t_1) * beta_t_1 ** 0.5

                        timestep_t_1 = timesteps[i + 1]
                        alpha_cum_t_1 = self.noise_scheduler.alphas_cumprod[timestep_t_1].item()
                        alpha_cum_t = self.noise_scheduler.alphas_cumprod[t].item()
                        x_t = x_t_1 * (alpha_cum_t / alpha_cum_t_1) ** 0.5 + torch.randn_like(x_t_1) * (1 - (alpha_cum_t / alpha_cum_t_1)) ** 0.5
                    
                    
                    x_t = x_t_1
            noisy_latents = x0.detach()
            noisy_latents_sds = noisy_latents.clone()
            bs = len(noisy_latents)
            noisy_latents = 1 / self.vae.config.scaling_factor * noisy_latents
            noisy_latents = rearrange(noisy_latents, 'b c ... -> (b c) ...')
            image = self.vae.decode(noisy_latents).sample
            image = rearrange(image, '(b c) ... -> b c ...', b=bs)
            # image = (image / 2 + 0.5).clamp(0, 1)

            unnorm_img = image.detach().clone()


            # style_shifted_image = wavelet_reconstruction(image.squeeze(0), batch['pixel_values'].squeeze(0)).unsqueeze(0)
            # self.save_image0_from_pixels(style_shifted_image, f"style_shifted_image_{step}.png")
            # self.save_image0_from_pixels(batch['pixel_values'], f"input_image_{step}.png")
            # self.save_image0_from_pixels(image, f"targe_image_{step}.png")
            # import ipdb ; ipdb.set_trace()

            image = normalize_image(image.float())
            batch['pixel_values'] = normalize_image(batch['pixel_values'])


            loss_mse = F.mse_loss(batch['pixel_values'], image.float(), reduction='mean')

            loss_l1 = F.l1_loss(batch['pixel_values'], image.float(), reduction='mean')

            loss_latent_mse = F.mse_loss(sds_img_vae, noisy_latents_sds.detach(), reduction='sum')

            loss_lpips = self.lpips_loss(batch['pixel_values'].float().squeeze(0), image.float().squeeze(0)).mean()


            w_latent_mse = 0.0
            w_l1 = 5.0
            w_mse = 0.0
            w_lpips = 50.0

            # wt = 1.0 - self.noise_scheduler.alphas_cumprod[timestep].item()
            total_loss = loss_latent_mse * 0.1 * w_latent_mse + loss_l1 * 1e4 * w_l1 + loss_mse * 2e4 * w_mse + loss_lpips * 1e3 * w_lpips
            return total_loss, {'unnorm_target_image': unnorm_img, 'target_image': image.detach(), 'target_latent': noisy_latents_sds.detach()}


        else:
            raise Exception(f"method {method} not supported")
    def get_sample_info(self, sample_token):
        sample = self.nusc_database.get('sample', sample_token)
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd_rec = self.nusc_database.get("sample_data", sample["data"]["LIDAR_TOP"])
        cs_record = self.nusc_database.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
        pose_record = self.nusc_database.get("ego_pose", sd_rec["ego_pose_token"])
        location = self.nusc_database.get(
            "log", self.nusc_database.get("scene", sample["scene_token"])["log_token"]
        )["location"]
        lidar_path, boxes, _ = self.nusc_database.get_sample_data(lidar_token)
        description = self.nusc_database.get("scene", sample["scene_token"])["description"]
        timeofday = self.nusc_database.get(
            "log", self.nusc_database.get("scene", sample["scene_token"])["log_token"]
        )["logfile"][5:]

        info = {
            "lidar_path": lidar_path,
            "token": sample["token"],
            "sweeps": [],
            "cams": dict(),
            "lidar2ego_translation": cs_record["translation"],
            "lidar2ego_rotation": cs_record["rotation"],
            "ego2global_translation": pose_record["translation"],
            "ego2global_rotation": pose_record["rotation"],
            "timestamp": sample["timestamp"],
            "location": location,
            "description": description,
            "timeofday": timeofday,
        }

        l2e_r = info["lidar2ego_rotation"]
        l2e_t = info["lidar2ego_translation"]
        e2g_r = info["ego2global_rotation"]
        e2g_t = info["ego2global_translation"]
        l2e_r_mat = quaternion_to_rotation_matrix(l2e_r)
        e2g_r_mat = quaternion_to_rotation_matrix(e2g_r)

        # obtain 6 image's information per frame
        camera_types = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]
        for cam in camera_types:
            cam_token = sample["data"][cam]
            cam_path, _, camera_intrinsics = self.nusc_database.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(
                self.nusc_database, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
            )
            cam_info.update(camera_intrinsics=camera_intrinsics)
            info["cams"].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = self.nusc_database.get("sample_data", sample["data"]["LIDAR_TOP"])
        sweeps = []
        while len(sweeps) < 10:
            if not sd_rec["prev"] == "":
                sweep = obtain_sensor2top(
                    self.nusc_database, sd_rec["prev"], l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, "lidar"
                )
                sweeps.append(sweep)
                sd_rec = self.nusc_database.get("sample_data", sd_rec["prev"])
            else:
                break
        info["sweeps"] = sweeps
        # obtain annotation
        annotations = [
            self.nusc_database.get("sample_annotation", token) for token in sample["anns"]
        ]
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(
            -1, 1
        )
        velocity = np.array(
            [self.nusc_database.box_velocity(token)[:2] for token in sample["anns"]]
        )
        valid_flag = np.array(
            [
                (anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0
                for anno in annotations
            ],
            dtype=bool,
        ).reshape(-1)
        # convert velo from global to lidar
        for i in range(len(boxes)):
            velo = np.array([*velocity[i], 0.0])
            velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
            velocity[i] = velo[:2]

        names = [b.name for b in boxes]
        for i in range(len(names)):
            if names[i] in NameMapping:
                names[i] = NameMapping[names[i]]
        names = np.array(names)
        # we need to convert rot to SECOND format.
        gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
        assert len(gt_boxes) == len(
            annotations
        ), f"{len(gt_boxes)}, {len(annotations)}"
        info["gt_boxes"] = gt_boxes
        info["gt_names"] = names
        info["gt_velocity"] = velocity.reshape(-1, 2)
        info["num_lidar_pts"] = np.array([a["num_lidar_pts"] for a in annotations])
        info["num_radar_pts"] = np.array([a["num_radar_pts"] for a in annotations])
        info["visibility"] = np.array([a["visibility_token"] for a in annotations], dtype=np.uint8)
        info["valid_flag"] = valid_flag




        data = dict(
            token=info["token"],
            sample_idx=info['token'],
            lidar_path=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"],
            location=info["location"],
        )
        add_key = [
            "description",
            "timeofday",
            "visibility",
            "flip_gt",
        ]
        for key in add_key:
            if key in info:
                data[key] = info[key]

        # ego to global transform
        ego2global = np.eye(4).astype(np.float32)
        ego2global[:3, :3] = quaternion_to_rotation_matrix(
            info["ego2global_rotation"])
        ego2global[:3, 3] = info["ego2global_translation"]
        data["ego2global"] = ego2global

        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = quaternion_to_rotation_matrix(
            info["lidar2ego_rotation"])
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        data["lidar2ego"] = lidar2ego

        # if self.modality["use_camera"]:
        data["image_paths"] = []
        data["lidar2camera"] = []
        data["lidar2image"] = []
        data["camera2ego"] = []
        data["camera_intrinsics"] = []
        data["camera2lidar"] = []

        for _, camera_info in info["cams"].items():
            data["image_paths"].append(camera_info["data_path"])

            # lidar to camera transform
            lidar2camera_r = np.linalg.inv(
                camera_info["sensor2lidar_rotation"])
            lidar2camera_t = (
                camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
            )
            lidar2camera_rt = np.eye(4).astype(np.float32)
            lidar2camera_rt[:3, :3] = lidar2camera_r.T
            lidar2camera_rt[3, :3] = -lidar2camera_t
            data["lidar2camera"].append(lidar2camera_rt.T)

            # camera intrinsics
            camera_intrinsics = np.eye(4).astype(np.float32)
            camera_intrinsics[:3, :3] = camera_info["camera_intrinsics"]
            data["camera_intrinsics"].append(camera_intrinsics)

            # lidar to image transform
            lidar2image = camera_intrinsics @ lidar2camera_rt.T
            data["lidar2image"].append(lidar2image)

            # camera to ego transform
            camera2ego = np.eye(4).astype(np.float32)
            camera2ego[:3, :3] = quaternion_to_rotation_matrix(
                camera_info["sensor2ego_rotation"]
            )
            camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
            data["camera2ego"].append(camera2ego)

            # camera to lidar transform
            camera2lidar = np.eye(4).astype(np.float32)
            camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
            camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
            data["camera2lidar"].append(camera2lidar)

        annos, mask = self.get_annotations(info)
        if "visibility" in data:
            data["visibility"] = data["visibility"][mask]
        data["ann_info"] = annos


        data["img_fields"] = []
        data["bbox3d_fields"] = []
        data["pts_mask_fields"] = []
        data["pts_seg_fields"] = []
        data["bbox_fields"] = []
        data["mask_fields"] = []
        data["seg_fields"] = []
        data["box_type_3d"] = self.box_type_3d
        data["box_mode_3d"] = self.box_mode_3d

        return data

    def get_bev_map_from_lidar2w(self, info):
        import ipdb ; ipdb.set_trace()
        pass

    def get_annotations(self, info):
        if self.force_all_boxes:
            mask = np.ones_like(info["valid_flag"])
        elif self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info["gt_velocity"][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # haotian: this is an important change: from 0.5, 0.5, 0.5 -> 0.5, 0.5, 0
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
        ).convert_to(get_box_type("LiDAR")[1])

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )
        return anns_results, mask
    def collate_fn_preprocess(self, info):

        ret_dict = collate_fn(examples=[info], template=self.template, tokenizer=self.tokenizer, is_train=True, bbox_mode=self.bbox_mode)
        return ret_dict

    def prepare_data_pipeline(self):
        data_pipeline_cfg = self.data_pipeline_cfg.copy()
        for processor in data_pipeline_cfg:
            name = processor.pop("type")
            processor = eval(name)(**processor)
            self.data_pipeline.append(processor)

    def data_pipeline_fn(self, info):
        for processor in self.data_pipeline:
            info = processor(info)
        return info


def find_files_by_prefix(directory, prefixes):
    """
    Find files in a directory that start with any of the specified numeric prefixes
    
    Args:
        directory (str): Path to search in
        prefixes (list): List of integer prefixes to match
    
    Returns:
        list: Full paths of matching files
    """
    str_prefixes = [str(p) for p in prefixes]
    matches = []
    
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file():
                filename = entry.name
                if any(filename.startswith(prefix) for prefix in str_prefixes):
                    matches.append(entry.path)
    
    return matches
def argument_search():
    import time
    resample_l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
    
    ts_l = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    num_ts_l = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    inmask_g_scale_l = [0.1, 0.2, 0,5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


    mgd = MagicDrive(sd_path="pretrained/stable-diffusion-v1-5", checkpoint_path="pretrained/SDv1.5mv-rawbox_2023-09-07_18-39_224x400")
    mgd.set_scene(605)
    mgd.prepare_data_pipeline()
    save_dict = torch.load("save_dict.pt")
    for i in range(len(inmask_g_scale_l)):
        kwargs = {
            "resample": 5,
            "num_ts": 25,
            "ts": 200,
            "inmask_g_scale": 1.0
        }
        current_time = time.time()
        return_img = mgd.get_loss(pred_rgb=save_dict['pred_rgb'], sample_token=save_dict['sample_token'], timestep=200, shift_x=3, step=0, method='repaint2', inpainting_mask=save_dict['inpainting_mask'], **kwargs)
        elapsed_time = time.time() - current_time
        mgd.save_image0_from_latents(return_img, f"repaint2_{inmask_g_scale_l[i]}_{elapsed_time}.png")


    



if __name__ == "__main__":
    mgd = MagicDrive(sd_path="pretrained/stable-diffusion-v1-5", checkpoint_path="pretrained/SDv1.5mv-rawbox_2023-09-07_18-39_224x400")
    # mgd = MagicDrive(sd_path="pretrained/stable-diffusion-v1-5", checkpoint_path="pretrained/large_mgd")
    mgd.set_scene(605)
    mgd.prepare_data_pipeline()

    # hack pixel_values
    # recon_image = []
    # recon_image_name = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
    # for name in recon_image_name:
    #     image = torchvision.io.read_image(f"recon_shift/045_{name}.png").float() / 255.
    #     image = 2 * (image - 0.5)
    #     image = F.interpolate(image.unsqueeze(0), size=(224, 400), mode='bilinear', align_corners=False).squeeze(0)
    #     recon_image.append(image)
    # recon_image = torch.stack(recon_image).unsqueeze(0)
    # mgd.save_image0_from_pixels(recon_image, "image0_recon.png")

    # 556a590f3e1741fb87d7e4dc8ea687e4

    method = 'multistep'
    pt_path_prefix = "/home/kaiyuan.tan"
    iter_list = [31500, 31900, 30400, 30600, 31600, 32200]
    scene_idx = [40, 605, 516]
    pt_prefix = [("save_dict_" + str(i) + "_" + str(j), i, j) for i in scene_idx for j in iter_list]
    pt_path = find_files_by_prefix(pt_path_prefix, [pt_prefix[i][0] for i in range(len(pt_prefix))])
    iterable = [(pt_path[i], int(pt_path[i].split("_")[-3]), int(pt_path[i].split("_")[-2])) for i in range(len(pt_path))]
    for pt_path, scene, it in iterable:
        print(pt_path, scene, it)
        save_dict = torch.load(pt_path)
        # save_dict['pred_rgb'] = F.interpolate(save_dict['pred_rgb'].squeeze(0), size=(424, 800), mode='bilinear', align_corners=False).unsqueeze(0)
        if method == 'direct':
            mgd.get_loss(pred_rgb=save_dict['pred_rgb'], sample_token=save_dict['sample_token'], timestep=500, shift_x=save_dict['shift_x'], step=it, method='direct', inpainting_mask=torch.ones_like(save_dict['inpainting_mask']), scene_idx=0)
        elif method in ['sds', 'multistep', 'anneal', 'repaint', 'repaint2']:
            # sds
            image_params = torch.nn.Parameter(save_dict['pred_rgb'].to(self.device), requires_grad=True)
            optimizer = torch.optim.Adam([image_params], lr=1e-2)


            for i in range(1, 201):
                image_params_masked = image_params * save_dict['inpainting_mask'].to(self.device) + save_dict['pred_rgb'].to(self.device) * (1 - save_dict['inpainting_mask'].to(self.device))
                optimizer.zero_grad()
                # t = sample_gaussian_around_t(50 * (1000 - i) // 1000, sigma=30) + 50
                # t = (1000 - i)
                # t = random.randint(50, 100)
                t = 50
                # t = random.randint(1, 999)
                kwargs = {
                    "resample": 2,
                    "num_ts": 10,
                    "ts": t,
                    "inmask_g_scale": 1, 
                    "stochastic": False,
                    "cfg_scale": 2.0,
                    "scene_idx": 605
                }
                # kwargs = {
                #     "resample": 4,
                #     "num_ts": 5,
                #     "ts": 300,
                #     "inmask_g_scale": 0.5,
                #     "stochastic": False,
                #     "cfg_scale": 2.0
                # }
                inpainting_mask = dilate(save_dict['inpainting_mask'], 45, True)
                loss, ret_dict = mgd.get_loss(pred_rgb=image_params_masked, sample_token=save_dict['sample_token'], timestep=None, shift_x=save_dict['shift_x'], step=i, method=method, inpainting_mask=inpainting_mask, **kwargs)
                target_latent = ret_dict['target_latent']
                loss.backward()

                optimizer.step()
                # print(f"step {i}, loss {loss.item()}, image_consis_loss {image_consis_loss.item()} t {t}")
                # if i % 50 == 0:
                    # self.save_image0_from_latents(sds_img_vae, f"sds_img_vae_{i}.png")

                mgd.save_image0_from_pixels(image_params.clip(-1, 1), f"{scene}_{it}_input.png")
                mgd.save_image0_from_latents(target_latent, f"{scene}_{it}_target.png")
                mgd.save_image0_from_pixels(inpainting_mask.unsqueeze(0) * 2 - 1, f"{scene}_{it}_mask.png")
                    # mgd.save_image0_from_latents(ret_dict['sds_img_vae'], f"sds_img_vae_{i}.png")
                break



                # evaluation of target image

                # with torch.no_grad():
                #     target_image = ret_dict['unnorm_target_image'] * (1 - save_dict['inpainting_mask'].to(self.device))
                #     input_image = image_params_masked * (1 - save_dict['inpainting_mask'].to(self.device))
                #     consistency = F.l1_loss(target_image, input_image)
                #     print(f"consistency loss: {consistency.item() * 10000}")
                #     sds_scores = []
                #     for t in [50, 100, 200, 500]:
                #         sds_loss, _ = mgd.get_loss(pred_rgb=target_image, sample_token=save_dict['sample_token'], timestep=t, shift_x=3, step=i, method='sds', inpainting_mask=save_dict['inpainting_mask'])
                #         sds_scores.append(sds_loss.item())
                #     print(f"sds scores: {sds_scores}")
            
