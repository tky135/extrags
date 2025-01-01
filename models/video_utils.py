from typing import Literal, Dict, List, Optional, Callable
from tqdm import tqdm, trange
import numpy as np
import os
import logging
import imageio

import torch
from torch import Tensor
from torch.nn import functional as F
from skimage.metrics import structural_similarity as ssim
from datasets.base import SplitWrapper
from models.trainers.base import BasicTrainer
from utils.visualization import (
    to8b,
    depth_visualizer,
)
from torcheval.metrics.image import FrechetInceptionDistance

logger = logging.getLogger()

def get_numpy(x: Tensor) -> np.ndarray:
    return x.detach().squeeze().cpu().numpy()

def non_zero_mean(x: Tensor) -> float:
    return sum(x) / len(x) if len(x) > 0 else -1

def compute_psnr(prediction: Tensor, target: Tensor) -> float:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between the prediction and target tensors.

    Args:
        prediction (torch.Tensor): The predicted tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        float: The PSNR value between the prediction and target tensors.
    """
    if not isinstance(prediction, Tensor):
        prediction = Tensor(prediction)
    if not isinstance(target, Tensor):
        target = Tensor(target).to(prediction.device)
    return (-10 * torch.log10(F.mse_loss(prediction, target))).item()



def render(
    dataset: SplitWrapper,
    trainer: BasicTrainer,
    save_path: str,
    layout: Callable,
    num_timestamps: int,
    keys: List[str] = [
        "gt_rgbs",
        "rgbs",
        "Background_rgbs",
        "RigidNodes_rgbs",
        "DeformableNodes_rgbs",
        "SMPLNodes_rgbs",
        "depths",
        "Background_depths",
        "RigidNodes_depths",
        "DeformableNodes_depths",
        "SMPLNodes_depths",
        "mask",
        "lidar_on_images",
        "rgb_sky_blend",
        "rgb_sky",
        "rgb_error_maps"
    ],
    num_cams: int = 3,
    save_images: bool = True,
    fps: int = 10,
    compute_metrics: bool = False,
    compute_error_map: bool = False,
    vis_indices: Optional[List[int]] = None,
    lane_shift: bool = False,
):
    """
    Renders a dataset utilizing a specified render function.

    Parameters:
        dataset: Dataset to render.
        trainer: Gaussian trainer, includes gaussian models and rendering modules
        compute_metrics: Optional; if True, the function will compute and return metrics. Default is False.
        compute_error_map: Optional; if True, the function will compute and return error maps. Default is False.
        vis_indices: Optional; if not None, the function will only render the specified indices. Default is None.
    """
    output_dict = {k: [] for k in keys}
    output_dict.update({'cam_names': []})
    if num_timestamps > 1:
        video_writer = {k: imageio.get_writer(save_path.replace(".mp4", f"_{k}.mp4"), mode="I", fps=fps) for k in keys}
    else:
        video_writer = {k: imageio.get_writer(save_path.replace(".mp4", f"_{k}.png"), mode="I") for k in keys}

    if compute_metrics:
        psnrs, ssim_scores, lpipss = [], [], []
        masked_psnrs, masked_ssims = [], []
        human_psnrs, human_ssims = [], []
        vehicle_psnrs, vehicle_ssims = [], []
        occupied_psnrs, occupied_ssims = [], []
        fid = FrechetInceptionDistance(device=torch.device("cuda"))
        

    # with torch.no_grad():
    camera_downscale = trainer._get_downscale_factor()
    num_images = 0
    num_frames = 0
    
    
    
    # dataset setup
    indices = list(range(len(dataset))) if vis_indices is None else vis_indices
    dataset.mode = "sequential"
    dataset.camera_downscale = camera_downscale
    dataset.available_indices = indices
    dataiter = dataset.get_iterator(num_workers=4, prefetch_factor=2)

    for i in tqdm(indices, desc=f"rendering {dataset.split}", dynamic_ncols=True):
        image_infos, cam_infos = next(dataiter)
        if os.environ.get("DATASET") == 'kitti360/4cams' and cam_infos['cam_id'].flatten()[0].item() >= 1:
            continue
        for k, v in image_infos.items():
            if isinstance(v, Tensor):
                image_infos[k] = v[0].cuda(non_blocking=True)
        for k, v in cam_infos.items():
            if isinstance(v, Tensor):
                cam_infos[k] = v[0].cuda(non_blocking=True)
        # render the image
        results = trainer(image_infos, cam_infos)
        
        # ------------- clip rgb ------------- #
        for k, v in results.items():
            if isinstance(v, Tensor) and "rgb" in k:
                results[k] = v.clamp(0., 1.)
        
        # ------------- cam names ------------- #
        output_dict['cam_names'].append(cam_infos["cam_name"][0])

        # ------------- rgb ------------- #
        rgb = results["rgb"]
        output_dict['rgbs'].append(get_numpy(rgb))
        
        # if not os.path.exists(".output_results"):
        #     os.makedirs(".output_results")
        # imageio.imwrite(f".output_results/{i:03d}.png", to8b(get_numpy(rgb)))
        if "pixels" in image_infos and 'gt_rgbs' in keys:
            output_dict['gt_rgbs'].append(get_numpy(image_infos["pixels"]))
            
        green_background = torch.tensor([0.0, 177, 64]) / 255.0
        green_background = green_background.to(rgb.device)
        if "Background_rgb" in results and "Background_rgbs" in keys:
            Background_rgb = results["Background_rgb"] * results[
                "Background_opacity"
            ] + green_background * (1 - results["Background_opacity"])
            output_dict['Background_rgbs'].append(get_numpy(Background_rgb))
        if "RigidNodes_rgb" in results and 'RigidNodes_rgbs' in keys:
            RigidNodes_rgb = results["RigidNodes_rgb"] * results[
                "RigidNodes_opacity"
            ] + green_background * (1 - results["RigidNodes_opacity"])
            output_dict['RigidNodes_rgbs'].append(get_numpy(RigidNodes_rgb))
        if "DeformableNodes_rgb" in results and 'DeformableNodes_rgbs' in keys:
            DeformableNodes_rgb = results["DeformableNodes_rgb"] * results[
                "DeformableNodes_opacity"
            ] + green_background * (1 - results["DeformableNodes_opacity"])
            output_dict['DeformableNodes_rgbs'].append(get_numpy(DeformableNodes_rgb))
        if "SMPLNodes_rgb" in results and 'SMPLNodes_rgbs' in keys:
            SMPLNodes_rgb = results["SMPLNodes_rgb"] * results[
                "SMPLNodes_opacity"
            ] + green_background * (1 - results["SMPLNodes_opacity"])
            output_dict['SMPLNodes_rgbs'].append(get_numpy(SMPLNodes_rgb))
        if "Dynamic_rgb" in results and 'Dynamic_rgbs' in keys:
            Dynamic_rgb = results["Dynamic_rgb"] * results[
                "Dynamic_opacity"
            ] + green_background * (1 - results["Dynamic_opacity"])
            output_dict['Dynamic_rgbs'].append(get_numpy(Dynamic_rgb))
        if 'rgb_error_maps' in keys:
            # cal mean squared error
            error_map = (rgb - image_infos["pixels"]) ** 2
            error_map = error_map.mean(dim=-1, keepdim=True)
            # scale
            error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min())
            error_map = error_map.repeat_interleave(3, dim=-1)
            output_dict['rgb_error_maps'].append(get_numpy(error_map))
        if "rgb_sky_blend" in results and 'rgb_sky_blend' in keys:
            output_dict['rgb_sky_blend'].append(get_numpy(results["rgb_sky_blend"]))
        if "rgb_sky" in results and 'rgb_sky' in keys:
            output_dict['rgb_sky'].append(get_numpy(results["rgb_sky"]))
        # ------------- depth ------------- #
        if 'depths' in keys:
            depth = results['3dgs']["depth"]
            output_dict['depths'].append(get_numpy(depth))
        # ------------- mask ------------- #
        if "opacity" in results and 'mask' in keys:
            output_dict['mask'].append(get_numpy(results["opacity"]))
        if "Background_depth" in results and 'Background_depths' in keys:
            output_dict['Background_depths'].append(get_numpy(results["Background_depth"]))
        if "RigidNodes_depth" in results and 'RigidNodes_depths' in keys:
            output_dict["RigidNodes_depths"].append(get_numpy(results["RigidNodes_depth"]))
        if "DeformableNodes_depth" in results and 'DeformableNodes_depths' in keys:
            output_dict["DeformableNodes_depths"].append(get_numpy(results["DeformableNodes_depth"]))
        if "SMPLNodes_depth" in results and 'SMPLNodes_depths' in keys:
            output_dict["SMPLNodes_depths"].append(get_numpy(results["SMPLNodes_depth"]))
        if "Dynamic_depth" in results and 'Dynamic_depths' in keys:
            output_dict["Dynamic_depths"].append(get_numpy(results["Dynamic_depth"]))
        if "sky_masks" in image_infos and 'gt_sky_masks' in keys:
            output_dict["sky_masks"].append(get_numpy(image_infos["sky_masks"]))
        
        # ------------- lidar ------------- #
        if "lidar_depth_map" in image_infos and 'lidar_on_images' in keys:
            depth_map = image_infos["lidar_depth_map"]
            depth_img = depth_map.cpu().numpy()
            depth_img = depth_visualizer(depth_img, depth_img > 0)
            mask = (depth_map.unsqueeze(-1) > 0).cpu().numpy()
            lidar_on_image = image_infos["pixels"].cpu().numpy() * (1 - mask) + depth_img * mask
            output_dict["lidar_on_images"].append(lidar_on_image)
        del results

        if compute_metrics:
            fid.update(images=rgb[None, ...].permute(0, 3, 1, 2).clip(0, 1), is_real=False)
            fid.update(images=image_infos["pixels"][None, ...].permute(0, 3, 1, 2).clip(0, 1), is_real=True)
            valid_mask = (1.0 - image_infos['human_masks']) * (1.0 - image_infos['vehicle_masks']) * (1.0 - image_infos['dynamic_masks'])
            if 'egocar_masks' in image_infos:
                valid_mask *= (1.0 - image_infos['egocar_masks'])
            valid_mask = valid_mask.unsqueeze(-1)
            psnr = compute_psnr(rgb * valid_mask, image_infos["pixels"] * valid_mask)
            ssim_score = ssim(
                get_numpy(rgb),
                get_numpy(image_infos["pixels"]),
                data_range=1.0,
                channel_axis=-1,
            )
            lpips = trainer.lpips(
                rgb[None, ...].permute(0, 3, 1, 2).clip(0, 1),
                image_infos["pixels"][None, ...].permute(0, 3, 1, 2).clip(0, 1)
            )
            logger.info(f"Frame {i}: PSNR {psnr:.4f}, SSIM {ssim_score:.4f}")
            psnrs.append(psnr)
            ssim_scores.append(ssim_score)
            lpipss.append(lpips.item())
            
            if "sky_masks" in image_infos:
                occupied_mask = ~get_numpy(image_infos["sky_masks"]).astype(bool)
                if occupied_mask.sum() > 0:
                    occupied_psnrs.append(
                        compute_psnr(
                            rgb[occupied_mask], image_infos["pixels"][occupied_mask]
                        )
                    )
                    occupied_ssims.append(
                        ssim(
                            get_numpy(rgb),
                            get_numpy(image_infos["pixels"]),
                            data_range=1.0,
                            channel_axis=-1,
                            full=True,
                        )[1][occupied_mask].mean()
                    )

            if "dynamic_masks" in image_infos:
                dynamic_mask = get_numpy(image_infos["dynamic_masks"]).astype(bool)
                if dynamic_mask.sum() > 0:
                    masked_psnrs.append(
                        compute_psnr(
                            rgb[dynamic_mask], image_infos["pixels"][dynamic_mask]
                        )
                    )
                    masked_ssims.append(
                        ssim(
                            get_numpy(rgb),
                            get_numpy(image_infos["pixels"]),
                            data_range=1.0,
                            channel_axis=-1,
                            full=True,
                        )[1][dynamic_mask].mean()
                    )
            
            if "human_masks" in image_infos:
                human_mask = get_numpy(image_infos["human_masks"]).astype(bool)
                if human_mask.sum() > 0:
                    human_psnrs.append(
                        compute_psnr(
                            rgb[human_mask], image_infos["pixels"][human_mask]
                        )
                    )
                    human_ssims.append(
                        ssim(
                            get_numpy(rgb),
                            get_numpy(image_infos["pixels"]),
                            data_range=1.0,
                            channel_axis=-1,
                            full=True,
                        )[1][human_mask].mean()
                    )
            
            if "vehicle_masks" in image_infos:
                vehicle_mask = get_numpy(image_infos["vehicle_masks"]).astype(bool)
                if vehicle_mask.sum() > 0:
                    vehicle_psnrs.append(
                        compute_psnr(
                            rgb[vehicle_mask], image_infos["pixels"][vehicle_mask]
                        )
                    )
                    vehicle_ssims.append(
                        ssim(
                            get_numpy(rgb),
                            get_numpy(image_infos["pixels"]),
                            data_range=1.0,
                            channel_axis=-1,
                            full=True,
                        )[1][vehicle_mask].mean()
                    )
        num_images += 1
        if num_images % num_cams == 0:
            # write to video
            for k in keys:
                try:
                    if len(output_dict[k]) == 0:
                        continue
                    # handle special cases
                    cam_frames = output_dict[k]
                    if "mask" in k:
                        cam_frames = [np.stack([frame, frame, frame], axis=-1) for frame in cam_frames]
                    elif "depth" in k:
                        cam_frames = [depth_visualizer(frame, None) for frame in cam_frames]
                    frame = layout(cam_frames, output_dict['cam_names'])
                    frame = to8b(frame)
                    video_writer[k].append_data(frame)
                    output_dict[k] = []
                    if save_images:
                        os.makedirs(save_path.replace(".mp4", f"_{k}"), exist_ok=True)
                        imageio.imwrite(
                            save_path.replace(".mp4", f"_{k}/{num_frames:03d}_.png"),
                            frame,
                        )
                except Exception as e:
                    print(e)
                    import ipdb ; ipdb.set_trace()
            output_dict['cam_names'] = []
            num_frames += 1
                
    # messy aggregation...
    metrics_dict = {}
    metrics_dict["psnr"] = non_zero_mean(psnrs) if compute_metrics else -1
    metrics_dict["ssim"] = non_zero_mean(ssim_scores) if compute_metrics else -1
    metrics_dict["lpips"] = non_zero_mean(lpipss) if compute_metrics else -1
    # metrics_dict["fid"] = fid.compute().item() if compute_metrics else -1
    metrics_dict["occupied_psnr"] = non_zero_mean(occupied_psnrs) if compute_metrics else -1
    metrics_dict["occupied_ssim"] = non_zero_mean(occupied_ssims) if compute_metrics else -1
    metrics_dict["masked_psnr"] = non_zero_mean(masked_psnrs) if compute_metrics else -1
    metrics_dict["masked_ssim"] = non_zero_mean(masked_ssims) if compute_metrics else -1
    metrics_dict["human_psnr"] = non_zero_mean(human_psnrs) if compute_metrics else -1
    metrics_dict["human_ssim"] = non_zero_mean(human_ssims) if compute_metrics else -1
    metrics_dict["vehicle_psnr"] = non_zero_mean(vehicle_psnrs) if compute_metrics else -1
    metrics_dict["vehicle_ssim"] = non_zero_mean(vehicle_ssims) if compute_metrics else -1
    
    
    num_samples = len(dataset.split_indices) if vis_indices is None else len(vis_indices)
    logger.info(f"Eval over {num_samples} images:")
    logger.info(f"\t Full Image  PSNR: {metrics_dict['psnr']:.4f}")
    logger.info(f"\t Full Image  SSIM: {metrics_dict['ssim']:.4f}")
    logger.info(f"\t Full Image LPIPS: {metrics_dict['lpips']:.4f}")
    # logger.info(f"\t              FID: {metrics_dict['fid']:.4f}")
    logger.info(f"\t     Non-Sky PSNR: {metrics_dict['occupied_psnr']:.4f}")
    logger.info(f"\t     Non-Sky SSIM: {metrics_dict['occupied_ssim']:.4f}")
    logger.info(f"\tDynamic-Only PSNR: {metrics_dict['masked_psnr']:.4f}")
    logger.info(f"\tDynamic-Only SSIM: {metrics_dict['masked_ssim']:.4f}")
    logger.info(f"\t  Human-Only PSNR: {metrics_dict['human_psnr']:.4f}")
    logger.info(f"\t  Human-Only SSIM: {metrics_dict['human_ssim']:.4f}")
    logger.info(f"\tVehicle-Only PSNR: {metrics_dict['vehicle_psnr']:.4f}")
    logger.info(f"\tVehicle-Only SSIM: {metrics_dict['vehicle_ssim']:.4f}")

    return metrics_dict
