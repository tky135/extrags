"""
Filename: 3dgs.py

Author: Ziyu Chen (ziyu.sjtu@gmail.com)

Description:
Unofficial implementation of 3DGS based on the work by Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 
This implementation is modified from the nerfstudio GaussianSplattingModel.

- Original work by Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis.
- Codebase reference: nerfstudio GaussianSplattingModel (https://github.com/nerfstudio-project/nerfstudio/blob/gaussian-splatting/nerfstudio/models/gaussian_splatting.py)

Original paper: https://arxiv.org/abs/2308.04079
"""

from typing import Dict, List, Tuple
from omegaconf import OmegaConf
import logging

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from models.gaussians.basics import *

logger = logging.getLogger()
def normalize(v):
    """Normalizes a 3D vector."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(norm > 0, v / norm, v)

def normals_to_quaternions(normals):
    # Ensure the normals are unit vectors
    normals = normalize(normals)
    
    # Reference normal vector [0, 0, 1] (z-axis)
    ref_vector = np.array([0, 0, 1])
    
    # Compute the axis of rotation (cross product of each normal and ref_vector)
    rotation_axis = np.cross(ref_vector, normals)
    rotation_axis = normalize(rotation_axis)  # Normalize the axis
    
    # Compute the angle between the reference vector and each normal (dot product)
    dot_product = np.dot(normals, ref_vector)
    angles = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Ensure the values are within [-1, 1]
    
    # Convert axis-angle to quaternions
    half_angles = angles / 2.0
    sin_half_angles = np.sin(half_angles)
    
    quaternions = np.zeros((normals.shape[0], 4))
    quaternions[:, 0] = np.cos(half_angles)  # Scalar part (w)
    quaternions[:, 1:] = rotation_axis * sin_half_angles[:, None]  # Vector part (x, y, z)
    
    # Handle cases where the normal is already [0, 0, 1] (aligned with z-axis)
    aligned_mask = np.isclose(normals, [0, 0, 1]).all(axis=1)
    quaternions[aligned_mask] = np.array([1, 0, 0, 0])  # Identity quaternion for no rotation
    
    return quaternions

def quaternion_multiply(q1, q2):
    """Multiplies two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def quaternion_conjugate(q):
    """Returns the conjugate of a quaternion."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def apply_quaternion_rotation(quaternions, points):
    """Applies a quaternion rotation to a set of 3D points."""
    rotated_points = []
    
    for q, p in zip(quaternions, points):
        # Convert the point to a quaternion (pure quaternion with w=0)
        p_quat = np.array([0, p[0], p[1], p[2]])
        
        # Perform the rotation: p' = q * p * q^(-1)
        q_conjugate = quaternion_conjugate(q)
        rotated_p_quat = quaternion_multiply(quaternion_multiply(q, p_quat), q_conjugate)
        
        # Extract the rotated point (ignore the w component)
        rotated_points.append(rotated_p_quat[1:])
    
    return np.array(rotated_points)
class VanillaGaussians(nn.Module):

    def __init__(
        self,
        class_name: str,
        ctrl: OmegaConf,
        reg: OmegaConf = None,
        networks: OmegaConf = None,
        scene_scale: float = 30.,
        scene_origin: torch.Tensor = torch.zeros(3),
        num_train_images: int = 300,
        device: torch.device = torch.device("cuda"),
        **kwargs
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.ctrl_cfg = ctrl
        self.reg_cfg = reg
        self.networks_cfg = networks
        self.scene_scale = scene_scale
        self.scene_origin = scene_origin
        self.num_train_images = num_train_images
        self.step = 0
        
        self.device = device
        self.ball_gaussians=self.ctrl_cfg.get("ball_gaussians", False)
        self.gaussian_2d = self.ctrl_cfg.get("gaussian_2d", False)
        if 'sdf_grad' in kwargs:
            self.sdf_grad = kwargs['sdf_grad']
        
        # for evaluation
        self.in_test_set = False
        
        # init models
        self.xys_grad_norm = None
        self.max_2Dsize = None
        self.under_ground = None
        self._means = torch.zeros(1, 3, device=self.device)
        if self.ball_gaussians:
            self._scales = torch.zeros(1, 1, device=self.device)
        else:
            if self.gaussian_2d:
                self._scales = torch.zeros(1, 2, device=self.device)
            else:
                self._scales = torch.zeros(1, 3, device=self.device)
        self._quats = torch.zeros(1, 4, device=self.device)
        self._opacities = torch.zeros(1, 1, device=self.device)
        self._features_dc = torch.zeros(1, 3, device=self.device)
        self._features_rest = torch.zeros(1, num_sh_bases(self.sh_degree) - 1, 3, device=self.device)
        self.ground_gs = False
        
    @property
    def sh_degree(self):
        return self.ctrl_cfg.sh_degree

    def create_from_pcd(self, init_means: torch.Tensor, init_colors: torch.Tensor, from_lidar: torch.Tensor, init_opacity: float = 0.1) -> None:
        self._means = Parameter(init_means)
        
        distances, _ = k_nearest_sklearn(self._means.data, 4)
        
        
        # estimate normals
        pcd_data = self._means.data.cpu().numpy()
        import open3d as o3d 
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_data)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=10))
        normals = np.asarray(pcd.normals)
        mask = normals[:, 2] < 0
        normals[mask] = -normals[mask]
        
        
        o3d.io.write_point_cloud("test.ply", pcd)
        quats = normals_to_quaternions(normals)
        
        
        # verification
        # tmp = np.zeros_like(normals)
        # tmp[:, 2] = 1
        # import ipdb ; ipdb.set_trace()
        # apply_quaternion_rotation(quats, tmp)
        # calculate normal
        
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True).to(self.device)

        if self.ball_gaussians:
            self._scales = Parameter(torch.log(avg_dist.repeat(1, 1)))
        else:
            if self.gaussian_2d:
                # 如果是2dgs用估计的法向量
                self._quats = Parameter(torch.from_numpy(quats).float().to(self.device))
                self._scales = Parameter(torch.log(avg_dist.repeat(1, 2)))
            else:
                self._scales = Parameter(torch.log(avg_dist.repeat(1, 3)))
                self._quats = Parameter(random_quat_tensor(self.num_points).to(self.device))
        dim_sh = num_sh_bases(self.sh_degree)

        fused_color = RGB2SH(init_colors) # float range [0, 1] 
        shs = torch.zeros((fused_color.shape[0], dim_sh, 3)).float().to(self.device)
        if self.sh_degree > 0:
            shs[:, 0, :3] = fused_color
            shs[:, 1:, 3:] = 0.0
        else:
            shs[:, 0, :3] = torch.logit(init_colors, eps=1e-10)
        self._features_dc = Parameter(shs[:, 0, :])
        self._features_rest = Parameter(shs[:, 1:, :])
        self._opacities = Parameter(torch.logit(init_opacity * torch.ones(self.num_points, 1, device=self.device)))
        self.from_lidar = from_lidar.float()
    
    def add_points(self, init_means: torch.Tensor, init_colors: torch.Tensor, from_lidar: torch.Tensor) -> None:
        new_means = Parameter(init_means)
        distances, _ = k_nearest_sklearn(new_means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True).to(self.device)
        if self.ball_gaussians:
            new_scales = Parameter(torch.log(avg_dist.repeat(1, 1)))
        else:
            if self.gaussian_2d:
                new_scales = Parameter(torch.log(avg_dist.repeat(1, 2)))
            else:
                new_scales = Parameter(torch.log(avg_dist.repeat(1, 3)))
        new_quats = Parameter(random_quat_tensor(new_means.shape[0]).to(self.device))
        dim_sh = num_sh_bases(self.sh_degree)

        fused_color = RGB2SH(init_colors) # float range [0, 1] 
        shs = torch.zeros((fused_color.shape[0], dim_sh, 3)).float().to(self.device)
        if self.sh_degree > 0:
            shs[:, 0, :3] = fused_color
            shs[:, 1:, 3:] = 0.0
        else:
            shs[:, 0, :3] = torch.logit(init_colors, eps=1e-10)
        new_features_dc = Parameter(shs[:, 0, :])
        new_features_rest = Parameter(shs[:, 1:, :])
        new_opacities = Parameter(torch.logit(0.1 * torch.ones(new_means.shape[0], 1, device=self.device)))
        new_from_lidar = from_lidar.float()
        
        self._means = Parameter(torch.cat([self._means.detach(), new_means], dim=0))
        self._scales = Parameter(torch.cat([self._scales.detach(), new_scales], dim=0))
        self._quats = Parameter(torch.cat([self._quats.detach(), new_quats], dim=0))
        self._features_dc = Parameter(torch.cat([self._features_dc.detach(), new_features_dc], dim=0))
        self._features_rest = Parameter(torch.cat([self._features_rest.detach(), new_features_rest], dim=0))
        self._opacities = Parameter(torch.cat([self._opacities.detach(), new_opacities], dim=0))
        self.from_lidar = torch.cat([self.from_lidar, new_from_lidar], dim=0)
        
        
    @property
    def colors(self):
        if self.sh_degree > 0:
            return SH2RGB(self._features_dc)
        else:
            return torch.sigmoid(self._features_dc)
    @property
    def shs_0(self):
        return self._features_dc
    @property
    def shs_rest(self):
        return self._features_rest
    @property
    def num_points(self):
        return self._means.shape[0]
    @property
    def get_scaling(self):
        if self.ball_gaussians:
            if self.gaussian_2d:
                scaling = torch.exp(self._scales).repeat(1, 2)
                scaling = torch.cat([scaling, torch.zeros_like(scaling[..., :1])], dim=-1)
                return scaling
            else:
                return torch.exp(self._scales).repeat(1, 3)
        else:
            if self.gaussian_2d:
                scaling = torch.exp(self._scales)
                scaling = torch.cat([scaling[..., :2], torch.zeros_like(scaling[..., :1])], dim=-1)
                return scaling
            else:
                return torch.exp(self._scales)
    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacities)
    @property
    def get_quats(self):
        return self.quat_act(self._quats)
    
    def quat_act(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True)
    
    def preprocess_per_train_step(self, step: int):
        self.step = step
        
    def postprocess_per_train_step(
        self,
        step: int,
        optimizer: torch.optim.Optimizer,
        radii: torch.Tensor,
        xys_grad: torch.Tensor,
        last_size: int,
    ) -> None:
        self.after_train(radii, xys_grad, last_size)
        if step % self.ctrl_cfg.refine_interval == 0:
            self.refinement_after(step, optimizer)

    def after_train(
        self,
        radii: torch.Tensor,
        xys_grad: torch.Tensor,
        last_size: int,
    ) -> None:
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (radii > 0).flatten()
            full_mask = torch.zeros(self.num_points, device=radii.device, dtype=torch.bool)
            full_mask[self.filter_mask] = visible_mask
            
            grads = xys_grad.norm(dim=-1)
            if self.xys_grad_norm is None:
                self.xys_grad_norm = torch.zeros(self.num_points, device=grads.device, dtype=grads.dtype)
                self.xys_grad_norm[self.filter_mask] = grads
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[full_mask] = self.vis_counts[full_mask] + 1
                self.xys_grad_norm[full_mask] = grads[visible_mask] + self.xys_grad_norm[full_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros(self.num_points, device=radii.device, dtype=torch.float32)
            newradii = radii[visible_mask]
            self.max_2Dsize[full_mask] = torch.maximum(
                self.max_2Dsize[full_mask], newradii / float(last_size)
            )
        
    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            self.class_prefix+"xyz": [self._means],
            self.class_prefix+"sh_dc": [self._features_dc],
            self.class_prefix+"sh_rest": [self._features_rest],
            self.class_prefix+"opacity": [self._opacities],
            self.class_prefix+"scaling": [self._scales],
            self.class_prefix+"rotation": [self._quats],
        }
    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return self.get_gaussian_param_groups()

    def refinement_after(self, step, optimizer: torch.optim.Optimizer) -> None:
        assert step == self.step
        if self.step <= self.ctrl_cfg.warmup_steps:
            return
        with torch.no_grad():
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.ctrl_cfg.reset_alpha_interval
            do_densification = (
                self.step < self.ctrl_cfg.stop_split_at
                and self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval)
            )
            # split & duplicate
            print(f"Class {self.class_prefix} current points: {self.num_points} @ step {self.step}")
            if do_densification:
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                
                avg_grad_norm = self.xys_grad_norm / self.vis_counts
                high_grads = (avg_grad_norm > self.ctrl_cfg.densify_grad_thresh).squeeze()
                
                splits = (
                    self.get_scaling.max(dim=-1).values > \
                        self.ctrl_cfg.densify_size_thresh * self.scene_scale
                ).squeeze()
                if self.step < self.ctrl_cfg.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.ctrl_cfg.split_screen_size).squeeze()
                splits &= high_grads
                nsamps = self.ctrl_cfg.n_split_samples
                (
                    split_means,
                    split_feature_dc,
                    split_feature_rest,
                    split_opacities,
                    split_scales,
                    split_quats,
                    split_under_ground
                ) = self.split_gaussians(splits, nsamps)

                dups = (
                    self.get_scaling.max(dim=-1).values <= \
                        self.ctrl_cfg.densify_size_thresh * self.scene_scale
                ).squeeze()
                dups &= high_grads
                (
                    dup_means,
                    dup_feature_dc,
                    dup_feature_rest,
                    dup_opacities,
                    dup_scales,
                    dup_quats,
                    dep_under_ground
                ) = self.dup_gaussians(dups)
                
                self._means = Parameter(torch.cat([self._means.detach(), split_means, dup_means], dim=0))
                # self.colors_all = Parameter(torch.cat([self.colors_all.detach(), split_colors, dup_colors], dim=0))
                self._features_dc = Parameter(torch.cat([self._features_dc.detach(), split_feature_dc, dup_feature_dc], dim=0))
                self._features_rest = Parameter(torch.cat([self._features_rest.detach(), split_feature_rest, dup_feature_rest], dim=0))
                self._opacities = Parameter(torch.cat([self._opacities.detach(), split_opacities, dup_opacities], dim=0))
                self._scales = Parameter(torch.cat([self._scales.detach(), split_scales, dup_scales], dim=0))
                self._quats = Parameter(torch.cat([self._quats.detach(), split_quats, dup_quats], dim=0))
                self.from_lidar = torch.cat([self.from_lidar, torch.zeros_like(split_scales[:, 0]), torch.zeros_like(dup_scales[:, 0])], dim=0)
                
                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [self.max_2Dsize, torch.zeros_like(split_scales[:, 0]), torch.zeros_like(dup_scales[:, 0])],
                    dim=0,
                )
                self.under_ground = torch.cat([self.under_ground, split_under_ground, dep_under_ground], dim=0)
                
                split_idcs = torch.where(splits)[0]
                param_groups = self.get_gaussian_param_groups()
                dup_in_optim(optimizer, split_idcs, param_groups, n=nsamps)

                dup_idcs = torch.where(dups)[0]
                param_groups = self.get_gaussian_param_groups()
                dup_in_optim(optimizer, dup_idcs, param_groups, 1)

            # cull NOTE: Offset all the opacity reset logic by refine_every so that we don't
                # save checkpoints right when the opacity is reset (saves every 2k)
            if self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval):
                deleted_mask = self.cull_gaussians()
                param_groups = self.get_gaussian_param_groups()
                remove_from_optim(optimizer, deleted_mask, param_groups)
            print(f"Class {self.class_prefix} left points: {self.num_points}")
            logger.info("number of lidar points: " + str(self.from_lidar.sum()))
            print("number of lidar points: ", self.from_lidar.sum())
                    
            # reset opacity
            if self.step % reset_interval == self.ctrl_cfg.refine_interval and self.step < self.ctrl_cfg.stop_split_at and not self.ground_gs:
                print("resetting opacity at step", self.step)
                # NOTE: in nerfstudio, reset_value = cull_alpha_thresh * 0.8
                    # we align to original repo of gaussians spalting
                reset_value = torch.min(self.get_opacity.data,
                                        torch.ones_like(self._opacities.data) * 0.01)
                self._opacities.data = torch.logit(reset_value)
                # reset the exp of optimizer
                for group in optimizer.param_groups:
                    if group["name"] == self.class_prefix+"opacity":
                        old_params = group["params"][0]
                        param_state = optimizer.state[old_params]
                        param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                        param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

    def cull_gaussians(self):
        """
        This function deletes gaussians with under a certain opacity threshold
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (self.get_opacity.data < self.ctrl_cfg.cull_alpha_thresh).squeeze()
        if self.step > self.ctrl_cfg.reset_alpha_interval or self.ground_gs:
            # cull huge ones
            if self.ground_gs:
                toobigs = (
                    torch.exp(self._scales).max(dim=-1).values > 5
                ).squeeze()
            else:
                toobigs = (
                    torch.exp(self._scales).max(dim=-1).values > 
                    self.ctrl_cfg.cull_scale_thresh * self.scene_scale
                ).squeeze()
            culls = culls | toobigs
            if self.step < self.ctrl_cfg.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                culls = culls | (self.max_2Dsize > self.ctrl_cfg.cull_screen_size).squeeze()
        
        # cull underground ones
        if self.under_ground is not None:
            culls = culls | self.under_ground.squeeze(-1)
            
        # only cull too bigs for ground_gs
        if self.ground_gs:
            culls = toobigs
        self._means = Parameter(self._means[~culls].detach())
        self._scales = Parameter(self._scales[~culls].detach())
        self._quats = Parameter(self._quats[~culls].detach())
        # self.colors_all = Parameter(self.colors_all[~culls].detach())
        self._features_dc = Parameter(self._features_dc[~culls].detach())
        self._features_rest = Parameter(self._features_rest[~culls].detach())
        self._opacities = Parameter(self._opacities[~culls].detach())
        self.from_lidar = self.from_lidar[~culls]
        
        # 不更新self.under_ground, base会更新

        print(f"     Cull: {n_bef - self.num_points}")
        return culls

    def split_gaussians(self, split_mask: torch.Tensor, samps: int) -> Tuple:
        """
        This function splits gaussians that are too large
        """

        n_splits = split_mask.sum().item()
        print(f"    Split: {n_splits}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            self.get_scaling[split_mask].repeat(samps, 1) * centered_samples
            # torch.exp(self._scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quat_act(self._quats[split_mask])  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self._means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        # new_colors_all = self.colors_all[split_mask].repeat(samps, 1, 1)
        new_feature_dc = self._features_dc[split_mask].repeat(samps, 1)
        new_feature_rest = self._features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self._opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self._scales[split_mask]) / size_fac).repeat(samps, 1)
        self._scales[split_mask] = torch.log(torch.exp(self._scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self._quats[split_mask].repeat(samps, 1)
        new_under_ground = self.under_ground[split_mask].repeat(samps, 1)
        return new_means, new_feature_dc, new_feature_rest, new_opacities, new_scales, new_quats, new_under_ground

    def dup_gaussians(self, dup_mask: torch.Tensor) -> Tuple:
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        print(f"      Dup: {n_dups}")
        dup_means = self._means[dup_mask]
        # dup_colors = self.colors_all[dup_mask]
        dup_feature_dc = self._features_dc[dup_mask]
        dup_feature_rest = self._features_rest[dup_mask]
        dup_opacities = self._opacities[dup_mask]
        dup_scales = self._scales[dup_mask]
        dup_quats = self._quats[dup_mask]
        dup_under_ground = self.under_ground[dup_mask]
        return dup_means, dup_feature_dc, dup_feature_rest, dup_opacities, dup_scales, dup_quats, dup_under_ground
    def generate_offset_points(self, points, offset=0.01):
        """
        Generate offset points by adding/subtracting offset to x,y coordinates
        
        Args:
            points (torch.Tensor): Input points tensor of shape [N, 3]
            offset (float): Offset value to add/subtract (default: 0.01)
            
        Returns:
            torch.Tensor: Offset points tensor of shape [N, 4, 3]
        """
        # Create offset matrix [4, 2] for x,y coordinates
        offsets = torch.tensor([
            [-offset, -offset],  # bottom-left
            [-offset, offset],   # top-left
            [offset, -offset],   # bottom-right
            [offset, offset]     # top-right
        ], device=points.device, dtype=points.dtype)
        
        # Expand points to [N, 1, 3] and offsets to [1, 4, 2]
        expanded_points = points.unsqueeze(1)  # [N, 1, 3]
        expanded_offsets = offsets.unsqueeze(0)  # [1, 4, 2]
        
        # Create output tensor [N, 4, 3]
        result = expanded_points.expand(-1, 4, -1).clone()
        
        # Add offsets to x,y coordinates
        result[..., :2] += expanded_offsets
        
        return result
    def estimate_plane_normals(self, points):
        """
        Estimate normal vectors for multiple planes using SVD.
        
        Args:
            points: Tensor of shape [N, 4, 3] where:
                N is the batch size
                4 is the number of points per plane
                3 is the xyz coordinates
        
        Returns:
            normals: Tensor of shape [N, 3] containing unit normal vectors
        """
        # Center the points by subtracting the mean
        # Shape: [N, 4, 3] -> [N, 1, 3]
        centroids = torch.mean(points, dim=1, keepdim=True)
        # Shape: [N, 4, 3]
        centered_points = points - centroids
        
        # Compute the covariance matrix for each set of points
        # Shape: [N, 3, 3]
        covariance = torch.bmm(centered_points.transpose(1, 2), centered_points)
        
        # Perform batch SVD
        # u: [N, 3, 3], s: [N, 3], v: [N, 3, 3]
        u, s, v = torch.svd(covariance)
        
        # The normal vector is the last right singular vector (column of v)
        # Shape: [N, 3]
        normals = v[..., -1]
        
        # Normalize the vectors to unit length
        normals = F.normalize(normals, p=2, dim=-1)
        
        return normals
    def knn_normal(self, points):
        with torch.no_grad():
            # distances, indices = k_nearest_sklearn(points, 32)
            offseted_points = self.generate_offset_points(points, 0.05).reshape(-1, 3)
            elevation = self.sdf_network((offseted_points + self.omnire_w2neus_w[:3, 3].cuda()) @ torch.linalg.inv(self.scale_mat[:3, :3]), output_height=True)[:, :1] * 9
            offseted_points[:, 2] = elevation.squeeze()
            offseted_points = offseted_points.reshape(-1, 4, 3)
            return self.estimate_plane_normals(offseted_points)
            
            # normals = torch.zeros_like(points)
    def quaternion_to_vector(self, quaternions, reference_vector=None):
        """
        Convert quaternions to normal vectors in a differentiable way.
        
        Args:
            quaternions: Tensor of shape [..., 4] containing quaternions in format (w, x, y, z)
                        where w is the scalar component and (x, y, z) is the vector part.
                        Quaternions should be normalized (unit quaternions).
            reference_vector: Optional tensor of shape [..., 3] containing the reference vector
                            to rotate. Defaults to [0, 0, 1] if None.
        
        Returns:
            Tensor of shape [..., 3] containing the rotated vectors.
        """
        if not torch.is_tensor(quaternions):
            quaternions = torch.tensor(quaternions, dtype=torch.float32)
        
        # Ensure quaternions are normalized
        quaternions = F.normalize(quaternions, p=2, dim=-1)
        
        # Extract components
        w = quaternions[..., 0]
        x = quaternions[..., 1]
        y = quaternions[..., 2]
        z = quaternions[..., 3]
        
        if reference_vector is None:
            # Default to rotating [0, 0, 1]
            reference_vector = torch.zeros_like(quaternions[..., :3])
            reference_vector[..., 2] = 1.0
        
        # Compute rotation using quaternion multiplication
        # v' = q * v * q^(-1)
        # For unit quaternions, q^(-1) = q_conjugate = [w, -x, -y, -z]
        
        # Pre-compute common terms
        xx = x * x
        yy = y * y
        zz = z * z
        wx = w * x
        wy = w * y
        wz = w * z
        xy = x * y
        xz = x * z
        yz = y * z
        
        # Rotation matrix elements
        r00 = 1 - 2 * (yy + zz)
        r01 = 2 * (xy - wz)
        r02 = 2 * (xz + wy)
        
        r10 = 2 * (xy + wz)
        r11 = 1 - 2 * (xx + zz)
        r12 = 2 * (yz - wx)
        
        r20 = 2 * (xz - wy)
        r21 = 2 * (yz + wx)
        r22 = 1 - 2 * (xx + yy)
        
        # Reshape for batch matrix multiplication
        rotation_matrix = torch.stack([
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1)
        ], dim=-2)
        
        # Perform rotation
        rotated_vector = torch.matmul(rotation_matrix, reference_vector.unsqueeze(-1)).squeeze(-1)
        
        # Normalize output vector
        result = F.normalize(rotated_vector, p=2, dim=-1)
        
        return result
    def get_gaussians(self, cam: dataclass_camera) -> Dict:
        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        self.filter_mask = filter_mask
        
        # get colors of gaussians
        colors = torch.cat((self._features_dc[:, None, :], self._features_rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = self._means.detach() - cam.camtoworlds.data[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])
            
        activated_opacities = self.get_opacity
        activated_scales = self.get_scaling
        activated_rotations = self.get_quats
        actovated_colors = rgbs
        
        # collect gaussians information
        if not self.training:
            output_means = self._means
        else:
            output_means = self._means.detach() * self.from_lidar.unsqueeze(-1) + self._means * (1.0 - self.from_lidar).unsqueeze(-1)
        if self.ground_gs:
            # 验证
            # self.sdf_network.sdf((output_means + self.omnire_w2neus_w[:3, 3].cuda()) @ torch.linalg.inv(self.scale_mat[:3, :3]))
            if self.sdf_grad:
                elevation = self.sdf_network((output_means + self.omnire_w2neus_w[:3, 3].cuda()) @ torch.linalg.inv(self.scale_mat[:3, :3]), output_height=True)[:, :1] * 9
            else:
                with torch.no_grad():
                    elevation = (self.sdf_network((output_means + self.omnire_w2neus_w[:3, 3].cuda()) @ torch.linalg.inv(self.scale_mat[:3, :3]), output_height=True)[:, :1] * 9).detach()
            output_means = torch.cat([output_means[:, :2], elevation], dim=-1)
            if self.step % 10 == 0:
                normal_align_loss = 1 - torch.abs(torch.sum(self.quaternion_to_vector(activated_rotations[filter_mask]) * self.knn_normal(output_means[filter_mask]), dim=-1))
                normal_align_loss = normal_align_loss.mean()
            else:
                normal_align_loss = torch.tensor(0.0).to(self.device)
            # normals = 
            gs_dict = dict(
                    _means=output_means[filter_mask],
                    _opacities=activated_opacities[filter_mask],
                    _rgbs=actovated_colors[filter_mask],
                    _scales=activated_scales[filter_mask].clip(0, 0.05),
                    _quats=activated_rotations[filter_mask],
                    align_error=normal_align_loss,
                )
            # 计算法向量
            
        else:
            gs_dict = dict(
                    _means=output_means[filter_mask],
                    _opacities=activated_opacities[filter_mask],
                    _rgbs=actovated_colors[filter_mask],
                    _scales=activated_scales[filter_mask],
                    _quats=activated_rotations[filter_mask],
                )
        # if self.step < 10000:
        #     gs_dict = dict(
        #         _means=self._means[filter_mask].detach(),
        #         _opacities=activated_opacities[filter_mask],
        #         _rgbs=actovated_colors[filter_mask],
        #         _scales=activated_scales[filter_mask],
        #         _quats=activated_rotations[filter_mask],
        #     )
        # else:
        #     gs_dict = dict(
        #         _means=self._means[filter_mask],
        #         _opacities=activated_opacities[filter_mask],
        #         _rgbs=actovated_colors[filter_mask],
        #         _scales=activated_scales[filter_mask],
        #         _quats=activated_rotations[filter_mask],
        #     )
        
        # check nan and inf in gs_dict
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                print(f"NaN detected in gaussian {k} at step {self.step}")
                import ipdb ; ipdb.set_trace()

            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in gaussian {k} at step {self.step}")
                
        return gs_dict
    
    def compute_reg_loss(self):
        loss_dict = {}
        sharp_shape_reg_cfg = self.reg_cfg.get("sharp_shape_reg", None)
        if sharp_shape_reg_cfg is not None:
            w = sharp_shape_reg_cfg.w
            max_gauss_ratio = sharp_shape_reg_cfg.max_gauss_ratio
            step_interval = sharp_shape_reg_cfg.step_interval
            if self.step % step_interval == 0:
                # scale regularization
                scale_exp = self.get_scaling
                if self.gaussian_2d:
                    scale_exp = scale_exp[:, :2]
                    max_gauss_ratio = 5.0
                scale_reg = torch.maximum(scale_exp.amax(dim=-1) / (scale_exp.amin(dim=-1) + 1e-8), torch.tensor(max_gauss_ratio)) - max_gauss_ratio
                scale_reg = scale_reg.mean() * w
                loss_dict["sharp_shape_reg"] = scale_reg

        flatten_reg = self.reg_cfg.get("flatten", None)
        if flatten_reg is not None:
            sclaings = self.get_scaling
            min_scale, _ = torch.min(sclaings, dim=1)
            min_scale = torch.clamp(min_scale, 0, 30)
            flatten_loss = torch.abs(min_scale).mean()
            loss_dict["flatten"] = flatten_loss * flatten_reg.w
        
        sparse_reg = self.reg_cfg.get("sparse_reg", None)
        if sparse_reg and self.step > sparse_reg.start_step and self.ground_gs:
            if (self.cur_radii > 0).sum():
                opacity = torch.sigmoid(self._opacities)
                opacity = opacity.clamp(1e-6, 1-1e-6)
                log_opacity = opacity * torch.log(opacity)
                log_one_minus_opacity = (1-opacity) * torch.log(1 - opacity)
                sparse_loss = -1 * (log_opacity + log_one_minus_opacity)[self.cur_radii > 0].mean()
                loss_dict["sparse_reg"] = sparse_loss * sparse_reg.w

        # compute the max of scaling
        max_s_square_reg = self.reg_cfg.get("max_s_square_reg", None)
        if max_s_square_reg is not None and not self.ball_gaussians:
            loss_dict["max_s_square"] = torch.mean((self.get_scaling.max(dim=1).values) ** 2) * max_s_square_reg.w
        return loss_dict
    def state_dict(self, sdf=False) -> Dict:
        state_dict = super().state_dict()
        if sdf:
            with torch.no_grad():
                elevation = (self.sdf_network((state_dict['_means'] + self.omnire_w2neus_w[:3, 3].cuda()) @ torch.linalg.inv(self.scale_mat[:3, :3]), output_height=True)[:, :1] * 9).detach()
                state_dict['_means'][:, 2] = elevation.squeeze()
        return state_dict
    def load_state_dict(self, state_dict: Dict, **kwargs) -> str:
        N = state_dict["_means"].shape[0]
        self._means = Parameter(torch.zeros((N,) + self._means.shape[1:], device=self.device))
        self._scales = Parameter(torch.zeros((N,) + self._scales.shape[1:], device=self.device))
        self._quats = Parameter(torch.zeros((N,) + self._quats.shape[1:], device=self.device))
        self._features_dc = Parameter(torch.zeros((N,) + self._features_dc.shape[1:], device=self.device))
        self._features_rest = Parameter(torch.zeros((N,) + self._features_rest.shape[1:], device=self.device))
        self._opacities = Parameter(torch.zeros((N,) + self._opacities.shape[1:], device=self.device))
        msg = super().load_state_dict(state_dict, **kwargs)
        return msg
    
    def export_gaussians_to_ply(self, alpha_thresh: float) -> Dict:
        means = self._means
        direct_color = self.colors
        
        activated_opacities = self.get_opacity
        mask = activated_opacities.squeeze() > alpha_thresh
        return {
            "positions": means[mask],
            "colors": direct_color[mask],
        }