from omegaconf import OmegaConf
import numpy as np
import os
import time
import wandb
import random
import imageio
import logging
import argparse
import json
import torch
import torch.nn.functional as F
import torchvision
from tools.eval import do_evaluation
from utils.misc import import_str
from utils.backup import backup_project
from utils.logging import MetricLogger, setup_logging
from models.video_utils import render
from datasets.driving_dataset import DrivingDataset

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
shift_x_l = []

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


def clone_tensor_dict_deep(tensor_dict):
    cloned_dict = {}
    for k, v in tensor_dict.items():
        if isinstance(v, dict):
            cloned_dict[k] = clone_tensor_dict_deep(v)
        elif hasattr(v, 'clone'):  # checks if it's a tensor-like object
            cloned_dict[k] = v.clone()
        else:
            cloned_dict[k] = v
    return cloned_dict
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
def set_seeds(seed=42):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def setup(args):
    # get config
    
    # cli > dataset config > config 
    cfg = OmegaConf.load(args.config_file)
    
    # parse datasets
    args_from_cli = OmegaConf.from_cli(args.opts)
    dataset_type = args_from_cli.get("dataset")
    assert dataset_type is not None, "Please specify dataset type in cli"
        
    dataset_cfg = OmegaConf.load(
        os.path.join("configs", "datasets", f"{dataset_type}.yaml")
    )
    # merge data
    cfg = OmegaConf.merge(cfg, dataset_cfg)
    
    # merge cli
    cfg = OmegaConf.merge(cfg, args_from_cli)
    log_dir = os.path.join(args.output_root, args.project, args.run_name)

    dataset_type = cfg.data.data_root.split('/')[-1]
    scene_idx = str(cfg.data.scene_idx)

    # shift_x_dict = json.load(open("shift_x.json"))
    shift_x_l.extend([-float(os.environ.get('shift_x'))])
    # if scene_idx in shift_x_dict[dataset_type]:
    #     shift_x_l.extend(shift_x_dict[dataset_type][scene_idx])
    # else:
    #     shift_x_l.extend([3.0])
    
    # update config and create log dir
    cfg.log_dir = log_dir
    cfg.trainer.log_dir = log_dir
    cfg.trainer.dataset = cfg.data.dataset
    cfg.trainer.scene_idx = scene_idx
    cfg.trainer.version = dataset_type
    os.makedirs(log_dir, exist_ok=True)
    for folder in ["images", "videos", "metrics", "configs_bk", "buffer_maps", "backup"]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)
    
    # update environment variables
    os.environ.update({
        "LOG_DIR": log_dir,
        "DATASET": dataset_type,
    })
    # setup wandb
    if args.enable_wandb:
        # sometimes wandb fails to init in cloud machines, so we give it several (many) tries
        while (
            wandb.init(
                project=args.project,
                # entity=args.entity,
                sync_tensorboard=True,
                settings=wandb.Settings(start_method="fork"),
            )
            is not wandb.run
        ):
            continue
        wandb.run.name = args.run_name
        wandb.run.save()
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
        wandb.config.update(args)

    # setup random seeds
    set_seeds(cfg.seed)

    global logger
    setup_logging(output=log_dir, level=logging.INFO, time_string=current_time)
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    # save config
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    saved_cfg_path = os.path.join(log_dir, "config.yaml")
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
        
    # also save a backup copy
    saved_cfg_path_bk = os.path.join(log_dir, "configs_bk", f"config_{current_time}.yaml")
    with open(saved_cfg_path_bk, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    logger.info(f"Full config saved to {saved_cfg_path}, and {saved_cfg_path_bk}")
    
    # Backup codes
    backup_project(
        os.path.join(log_dir, 'backup'), "./", 
        ["configs", "datasets", "models", "utils", "tools"], 
        [".py", ".h", ".cpp", ".cuh", ".cu", ".sh", ".yaml"]
    )
    return cfg

def main(args):
    cfg = setup(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build dataset
    # cfg.data: # dict_keys(['data_root', 'dataset', 'scene_idx', 'start_timestep', 'end_timestep', 'preload_device', 'pixel_source', 'lidar_source'])
    dataset = DrivingDataset(data_cfg=cfg.data)

    # setup trainer
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set.split_indices),
        num_full_images=len(dataset.full_image_set.split_indices),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        dataset_obj=dataset,
        device=device
    )
    
    # NOTE: If resume, gaussians will be loaded from checkpoint
    #       If not, gaussians will be initialized from dataset
    if args.resume_from is not None:
        trainer.init_gaussians_from_dataset(dataset, fast_run=True)
        trainer.resume_from_checkpoint(
            ckpt_path=args.resume_from,
            load_only_model=True
        )
        state_dict = torch.load(args.resume_from, map_location="cpu")
        if 'Ground_gs' not in state_dict['models'] and trainer.ground_method == 'rsg':
            trainer.neus23dgs()
        del state_dict
        logger.info(
            f"Resuming training from {args.resume_from}, starting at step {trainer.step}"
        )
        if trainer.export_neus_2dgs:
            print("exporting neus to 2dgs")
            trainer.neus23dgs(output=True)
            trainer.ground_method = "rsg"
            trainer.save_checkpoint(
                    log_dir=cfg.log_dir,
                    save_only_model=True,
                    is_final=False,
                    ckpt_name="checkpoint_neus_2dgs",
                )
    else:
        trainer.init_gaussians_from_dataset(dataset=dataset)
        logger.info(
            f"Training from scratch, initializing gaussians from dataset, starting at step {trainer.step}"
        )
    
    if args.enable_viewer:
        # a simple viewer for background visualization
        trainer.init_viewer(port=args.viewer_port)
    
    # define render keys
    render_keys = [
        "gt_rgbs",
        "rgbs",
        "Background_rgbs",
        "Dynamic_rgbs",
        "RigidNodes_rgbs",
        "DeformableNodes_rgbs",
        "SMPLNodes_rgbs",
        "depths",
        "Background_depths",
        "Dynamic_depths",
        "RigidNodes_depths",
        "DeformableNodes_depths",
        "SMPLNodes_depths",
        "mask",
        "lidar_on_images",
        "rgb_sky_blend",
        "rgb_sky",
        "rgb_error_maps",
    ]
    # setup optimizer  
    trainer.initialize_optimizer()

    print("optimizer_all optimizing parameters: ")
    for group in trainer.optimizer_all.param_groups:
        print(group['name'])
    print("optimizer_diffusion optimizing parameters: ")
    for group in trainer.optimizer_diffusion.param_groups:
        print(group['name'])
    
    
    # setup metric logger
    metrics_file = os.path.join(cfg.log_dir, "metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    all_iters = np.arange(trainer.step, trainer.num_iters + 1)
    
    # DEBUG USE
    # do_evaluation(
    #     step=0,
    #     cfg=cfg,
    #     trainer=trainer,
    #     dataset=dataset,
    #     render_keys=render_keys,
    #     args=args,
    # )
    dataset.train_image_set.mode = "random"
    trainiter = dataset.train_image_set.get_iterator(num_workers=4, prefetch_factor=2)

    # dataset.diff_image_set.mode = "sequential_infinite"
    # dataset.diff_image_set.mode = "sequential"
    # diffiter = dataset.diff_image_set.get_iterator(num_workers=4, prefetch_factor=2)
    # image_params = torch.nn.Parameter(torch.zeros((1, 6, 3, 224, 400)).cuda(), requires_grad=True)
    # optimizer = torch.optim.Adam([image_params], lr=1e-2)




    # diff loss configurations
    diff_method = 'multistep'
    diff_start = 30000
    trainer.diff_start = diff_start
    diff_freq = 10
    mgd_order = trainer.mgd.mgd_order
    cam_name2mgd_order = {cam_name: i for i, cam_name in enumerate(mgd_order)}


    if diff_method == 'direct':
        # diff loss buffers
        buffer_shift = dict()
        buffer_mask = dict()
        buffer_cam_shift_vector_w = dict()
    
    if diff_method in ['sds', 'multistep']:
        dataset.diff_image_set.mode = "sequential_infinite"
        diffiter = dataset.diff_image_set.get_iterator(num_workers=4, prefetch_factor=2)


    # do a evaluation first
    # do_evaluation(
    #     step=30000,
    #     cfg=cfg,
    #     trainer=trainer,
    #     dataset=dataset,
    #     render_keys=['rgbs', 'gt_rgbs', 'lidar_on_images'],
    #     args=args,
    #     shift_x=max(shift_x_l)
    # )

    sample2shift_vec = dict()
    for step in metric_logger.log_every(all_iters, cfg.logging.print_freq):

        # update shift set every 2000 steps
        if step > diff_start and step % diff_start == 0 and diff_method == 'direct':
            raise Exception
        # if diff_method == 'direct':
            # update shift buffer
            dataset.diff_image_set.mode = "sequential"
            diffiter = dataset.diff_image_set.get_iterator(num_workers=4, prefetch_factor=2)

            shift_x = random.choice(shift_x_l)
            timestep = 50


            timestamp_counter = 0

            while True:
                try:
                    image_6_views = {}
                    inpainting_mask_6_views = {}
                    current_sample_token = None
                    cam_shift_vector_w = None
                    for cam_id in range(trainer.n_camera):
                        diff_image_infos, diff_cam_infos = next(diffiter)
                        for k, v in diff_image_infos.items():
                            if isinstance(v, torch.Tensor):
                                diff_image_infos[k] = v[0].cuda(non_blocking=True)
                        for k, v in diff_cam_infos.items():
                            if isinstance(v, torch.Tensor):
                                diff_cam_infos[k] = v[0].cuda(non_blocking=True)
                        if 'road_masks' in diff_image_infos:
                            diff_image_infos['road_masks'][-1, -1] = 1.0
                        if current_sample_token is None:
                            current_sample_token = diff_image_infos['sample_tokens'][0]
                        else:
                            assert diff_image_infos['sample_tokens'][0] == current_sample_token


                        # get shift vector in world coordinate
                        if diff_cam_infos['cam_name'][0] == 'CAM_FRONT': # TODO Assuming x axis of CAM_FRONT is the same as LiDAR
                            c2w = diff_cam_infos['camera_to_world'].clone()
                            cam_x_axis_w = c2w[:3, :3] @ torch.tensor([1, 0, 0], device=c2w.device, dtype=c2w.dtype)
                            cam_shift_vector_w = -cam_x_axis_w * shift_x
                            buffer_cam_shift_vector_w[current_sample_token] = cam_shift_vector_w.clone().cpu()
                        
                        # shift camera
                        c2w = diff_cam_infos['camera_to_world'].clone()
                        c2w[:3, 3] += cam_shift_vector_w
                        diff_cam_infos['camera_to_world'] = c2w
                        
                        diff_outputs = trainer(diff_image_infos, diff_cam_infos, is_diffusion_step=True)
                        image_6_views[diff_cam_infos['cam_name'][0]] = diff_outputs['rgb'].permute(2, 0, 1).detach()
                        
                        # dilate inpainting mask
                        inpainting_mask = (diff_outputs['RigidNodes_opacity'] > 0.1).float().squeeze()
                        inpainting_mask = dilate(inpainting_mask, kernel_size=17)

                        inpainting_mask_6_views[diff_cam_infos['cam_name'][0]] = inpainting_mask.unsqueeze(0).detach().repeat(3, 1, 1)
                    image_6_views_l = [image_6_views[cam_name] for cam_name in trainer.mgd.mgd_order]
                    inpainting_mask_6_views_l = [inpainting_mask_6_views[cam_name] for cam_name in trainer.mgd.mgd_order]
                    image_6_views_ts = torch.stack(image_6_views_l, dim=0)
                    inpainting_mask_6_views_ts = torch.stack(inpainting_mask_6_views_l, dim=0)
                    
                    image_6_views_ts = F.interpolate(image_6_views_ts, size=(224, 400), mode='bilinear', align_corners=False)
                    inpainting_mask_6_views_ts = F.interpolate(inpainting_mask_6_views_ts, size=(224, 400), mode='nearest')
                    # image_6_views_ts = F.interpolate(image_6_views_ts, size=(424, 800), mode='bilinear', align_corners=False)
                    # inpainting_mask_6_views_ts = F.interpolate(inpainting_mask_6_views_ts, size=(424, 800), mode='nearest')
                    
                    image_6_views_ts = (image_6_views_ts.unsqueeze(0) - 0.5) * 2.0
                    trainer.mgd.save_image0_from_pixels(image_6_views_ts, f'image_6_views_{timestamp_counter}.png')
                    trainer.mgd.save_image0_from_pixels(inpainting_mask_6_views_ts.unsqueeze(0), f'inpainting_mask_6_views_{timestamp_counter}.png')
                    with torch.no_grad():
                        loss, retdict = trainer.mgd.get_loss(image_6_views_ts, current_sample_token, timestep, shift_x=shift_x, step=timestamp_counter, method=diff_method, inpainting_mask=inpainting_mask_6_views_ts)
                        buffer_shift[current_sample_token] = retdict['target_image'].cpu()
                        buffer_mask[current_sample_token] = inpainting_mask_6_views_ts.cpu()
                    timestamp_counter += 1

                except StopIteration:
                    break
        # ----------------------------------------------------------------------------
        # ----------------------------     Validate     ------------------------------
        if step % cfg.logging.vis_freq == 0 and cfg.logging.vis_freq > 0:
            trainer.set_eval()
            logger.info("Visualizing...")
            vis_timestep = np.linspace(
                0,
                dataset.num_img_timesteps,
                trainer.num_iters // cfg.logging.vis_freq + 1,
                endpoint=False,
                dtype=int,
            )[step // cfg.logging.vis_freq]
            # with torch.no_grad():
            metrics_dict = render(
                dataset=dataset.full_image_set,
                trainer=trainer,
                save_path=os.path.join(
                    cfg.log_dir, "images", f"step_{step}.mp4"
                ),
                layout=dataset.layout,
                num_timestamps=1,
                keys=render_keys,
                num_cams=trainer.n_camera,
                save_images=False,
                fps=cfg.render.fps,
                compute_metrics=True,
                compute_error_map=cfg.render.vis_error,
                vis_indices=[
                    vis_timestep * trainer.n_camera + i
                    for i in range(trainer.n_camera)
                ],
            )
            if args.enable_wandb:
                wandb.log(
                    {
                        "image_metrics/psnr": metrics_dict["psnr"],
                        "image_metrics/ssim": metrics_dict["ssim"],
                        "image_metrics/occupied_psnr": metrics_dict["occupied_psnr"],
                        "image_metrics/occupied_ssim": metrics_dict["occupied_ssim"],
                    }
                )
            torch.cuda.empty_cache()
        #----------------------------------------------------------------------------
        #----------------------------  training step  -------------------------------
        # prepare for training
        trainer.set_train()
        trainer.preprocess_per_train_step(step=step)
        next_iter = False
        while next_iter == False:
            loss_dict = {}
            if step > diff_start:
                trainer.optimizer_zero_grad(is_diffusion_step=True) # zero grad
            else:
                trainer.optimizer_zero_grad(is_diffusion_step=False)
            # get data
            train_step_camera_downscale = trainer._get_downscale_factor()
            if dataset.train_image_set.camera_downscale != train_step_camera_downscale:
                dataset.train_image_set.camera_downscale = train_step_camera_downscale
            # sample training data
            image_infos, cam_infos = next(trainiter)
            for k, v in image_infos.items():
                if isinstance(v, torch.Tensor):
                    image_infos[k] = v[0].cuda(non_blocking=True)
            for k, v in cam_infos.items():
                if isinstance(v, torch.Tensor):
                    cam_infos[k] = v[0].cuda(non_blocking=True)
            if 'road_masks' in image_infos:
                image_infos['road_masks'][-1, -1] = 1.0
            
            # forward & backward
            outputs = trainer(image_infos, cam_infos)
            if step % 100 == 0:
                save_rgb = outputs['rgb'].permute(2, 0, 1).detach().cpu() * 255
                save_rgb = save_rgb.type(torch.uint8)
                _cam_name = cam_infos['cam_name'][0]
                torchvision.io.write_png(save_rgb, os.path.join(trainer.mgd.get_prefix(), f'rgb_{_cam_name}_{step}.png'))
            trainer.update_visibility_filter()

            loss_dict.update(trainer.compute_losses(
                outputs=outputs,
                image_infos=image_infos,
                cam_infos=cam_infos,
            ))

            

            if step > diff_start and diff_method == 'direct' and image_infos['is_key_frame'].flatten().item() is True and image_infos['sample_tokens'][0] in buffer_shift:
                # diffusion loss
                raise Exception
                cam_name = cam_infos['cam_name'][0]
                target_img = buffer_shift[image_infos['sample_tokens'][0]][0][cam_name2mgd_order[cam_name]]
                target_img = target_img.cuda()

                cam_shift = buffer_cam_shift_vector_w[image_infos['sample_tokens'][0]]
                cam_shift = cam_shift.cuda()

                c2w = cam_infos['camera_to_world'].clone()
                c2w[:3, 3] += cam_shift
                cam_infos['camera_to_world'] = c2w

                outputs_shift = trainer(image_infos, cam_infos)
                pred_rgb = outputs_shift['rgb'].permute(2, 0, 1).unsqueeze(0)
                pred_rgb = F.interpolate(pred_rgb, size=(224, 400), mode='bilinear', align_corners=False)
                pred_rgb = (pred_rgb - 0.5) * 2.0
                img_idx = image_infos['img_idx'].flatten()[0]
                trainer.mgd.save_image0_from_pixels(pred_rgb.unsqueeze(1).repeat(1, 6, 1, 1, 1), f'img_{img_idx}_diff_training at_{step}.png')
                trainer.mgd.save_image0_from_pixels(target_img.unsqueeze(0).unsqueeze(0).repeat(1, 6, 1, 1, 1), f'img_{img_idx}_diff_training_target at_{step}.png')

                inpainting_mask = buffer_mask[image_infos['sample_tokens'][0]][cam_name2mgd_order[cam_name]]
                inpainting_mask = inpainting_mask.reshape(pred_rgb.shape).cuda()
                trainer.mgd.save_image0_from_pixels(inpainting_mask.unsqueeze(0).repeat(1, 6, 1, 1, 1), f'img_{img_idx}_inpainting_mask_{step}.png')
                diff_loss = F.mse_loss(pred_rgb * inpainting_mask, target_img.reshape(pred_rgb.shape).float() * inpainting_mask, reduction='mean')

                loss_dict['diff_loss'] = diff_loss




            # check nan or inf
            for k, v in loss_dict.items():
                if torch.isnan(v).any():
                    print(k)
                    import ipdb ; ipdb.set_trace()
                    raise ValueError(f"NaN detected in loss {k} at step {step}")
                if torch.isinf(v).any():
                    raise ValueError(f"Inf detected in loss {k} at step {step}")
            if step > diff_start:
                trainer.backward(loss_dict, is_diffusion_step=True)
            else:
                trainer.backward(loss_dict, is_diffusion_step=False)
            
            next_iter = True        
        # after training step
        trainer.postprocess_per_train_step(step=step, diff_grad=True)

        #----------------------------------------------------------------------------
        #-------------------------------  logging  ----------------------------------
        with torch.no_grad():
            # cal stats
            metric_dict = trainer.compute_metrics(
                outputs=outputs,
                image_infos=image_infos,
            )
        metric_logger.update(**{"train_metrics/"+k: v.item() for k, v in metric_dict.items()})
        metric_logger.update(**{"train_stats/gaussian_num_" + k: v for k, v in trainer.get_gaussian_count().items()})
        metric_logger.update(**{"losses/"+k: v.item() for k, v in loss_dict.items()})
        metric_logger.update(**{"train_stats/lr_optimizer_all" + group['name']: group['lr'] for group in trainer.optimizer_all.param_groups})
        metric_logger.update(**{"train_stats/lr_optimizer_diffusion" + group['name']: group['lr'] for group in trainer.optimizer_diffusion.param_groups})
        if args.enable_wandb:
            wandb.log({k: v.avg for k, v in metric_logger.meters.items()})

        #----------------------------------------------------------------------------
        #----------------------------     Saving     --------------------------------
        do_save = step > 0 and (
            (step % cfg.logging.saveckpt_freq == 0) or (step == trainer.num_iters)
        ) and (args.resume_from is None)
        if do_save:  
            trainer.export_ply(output_path=os.path.join(cfg.log_dir, f"step_{step}.ply"))
            trainer.save_checkpoint(
                log_dir=cfg.log_dir,
                save_only_model=True,
                is_final=step == trainer.num_iters,
            )
        # don't save the first time to avoid overwriting
        args.resume_from = None
        #----------------------------------------------------------------------------
        #------------------------    Cache Image Error    ---------------------------
        if (
            step > 0 and trainer.optim_general.cache_buffer_freq > 0
            and step % trainer.optim_general.cache_buffer_freq == 0
        ):
            logger.info("Caching image error...")
            trainer.set_eval()
            dataset.pixel_source.update_downscale_factor(
                1 / dataset.pixel_source.buffer_downscale
            )
            render_results = render_images(
                trainer=trainer,
                dataset=dataset.full_image_set,
            )
            dataset.pixel_source.reset_downscale_factor()
            dataset.pixel_source.update_image_error_maps(render_results)

            # save error maps
            merged_error_video = dataset.pixel_source.get_image_error_video(
                dataset.layout
            )
            imageio.mimsave(
                os.path.join(
                    cfg.log_dir, "buffer_maps", f"buffer_maps_{step}.mp4"
                ),
                merged_error_video,
                fps=cfg.render.fps,
            )
            logger.info("Done caching rgb error maps")



        #----------------------------------------------------------------------------
        #------------------------    SDSDiffusion Loss    ---------------------------
        trainer.set_train()
        trainer.preprocess_per_train_step(step=step)
        loss_dict = {}
        trainer.optimizer_zero_grad(is_diffusion_step=True) # zero grad
        # get data
        train_step_camera_downscale = trainer._get_downscale_factor()
        if dataset.train_image_set.camera_downscale != train_step_camera_downscale:
            dataset.train_image_set.camera_downscale = train_step_camera_downscale

        if step > diff_start and diff_method in ['sds', 'multistep'] and step % diff_freq == 0:
        # if step > diff_start and diff_method == 'sds' and image_infos['is_key_frame'].flatten().item() is True:
            shift_x = random.choice(shift_x_l)
            random_camera = random.choice([0, 1, 2, 3, 4, 5])
            # random_camera = random.choice([3])
            if diff_method == 'sds':
                lower_bound = 50
                upper_bound = 100
                diff_loss_weight = 5e-4
            elif diff_method == 'multistep':
                lower_bound = 50
                upper_bound = 500
                diff_loss_weight = 1e-4 / 2
            


            # if step > 32000:
            #     diff_loss_weight = 5e-4
            # if step > 34000:
            #     diff_loss_weight = 1e-4
            # if step > 36000:
            #     diff_loss_weight = 5e-5
            # if step > 38000:
            #     diff_loss_weight = 1e-5

            # if step > 32000:
            #     diff_loss_weight = 0.0
            # if step < 32000:
            #     upper_bound = 200
            #     diff_loss_weight = 1e-3
            # elif step < 36000:
            #     upper_bound = 100
            #     diff_loss_weight = 1e-4
            # else:
            #     upper_bound = 75
            #     lower_bound = 25
            #     diff_loss_weight = 1e-5

            timestep = random.randint(lower_bound, upper_bound)
            # timestep = sample_gaussian_around_t(int((40000. - step) / 10000. * 275 + 25), sigma=30)
            image_6_views = {}
            inpainting_mask_6_views = {}
            current_sample_token = None
            cam_shift_vector_w = None

            # deal with not-chosen cameras
            diff_image_infos_l, diff_cam_infos_l = [], []
            for _ in range(trainer.n_camera):
                img_info, cam_info = next(diffiter)
                diff_image_infos_l.append(img_info)
                diff_cam_infos_l.append(cam_info)
            for cam_id in range(trainer.n_camera):
                # if cam_id == random_camera:
                #     continue
                diff_image_infos, diff_cam_infos = clone_tensor_dict_deep(diff_image_infos_l[cam_id]), clone_tensor_dict_deep(diff_cam_infos_l[cam_id])
                for k, v in diff_image_infos.items():
                    if isinstance(v, torch.Tensor):
                        diff_image_infos[k] = v[0].cuda(non_blocking=True)
                for k, v in diff_cam_infos.items():
                    if isinstance(v, torch.Tensor):
                        diff_cam_infos[k] = v[0].cuda(non_blocking=True)
                if 'road_masks' in diff_image_infos:
                    diff_image_infos['road_masks'][-1, -1] = 1.0
                if current_sample_token is None:
                    current_sample_token = diff_image_infos['sample_tokens'][0]
                else:
                    assert diff_image_infos['sample_tokens'][0] == current_sample_token


                # get shift vector in world coordinate
                if diff_cam_infos['cam_name'][0] == 'CAM_FRONT': # TODO Assuming x axis of CAM_FRONT is the same as LiDAR
                    c2w = diff_cam_infos['camera_to_world'].clone()
                    cam_shift_vector_w = torch.tensor([-shift_x, 0, 0], device=c2w.device, dtype=c2w.dtype)

                    frame_idx = diff_image_infos['frame_idx'].flatten()[0].item()

                    lidar2w = dataset.lidar_source.lidar_to_worlds[frame_idx].to(device=c2w.device, dtype=c2w.dtype)
                    lidar_shift_vector = lidar2w[:3, :3] @ cam_shift_vector_w

                    if current_sample_token in sample2shift_vec:
                        assert torch.allclose(sample2shift_vec[current_sample_token], lidar_shift_vector.cpu())
                    else:
                        sample2shift_vec[current_sample_token] = lidar_shift_vector.cpu()
                    
                
                # shift camera
                c2w = diff_cam_infos['camera_to_world'].clone()
                c2w[:3, 3] += cam_shift_vector_w
                diff_cam_infos['camera_to_world'] = c2w
                
                diff_outputs = trainer(diff_image_infos, diff_cam_infos, is_diffusion_step=True)
                if cam_id == random_camera:
                    image_6_views[diff_cam_infos['cam_name'][0]] = diff_outputs['rgb'].permute(2, 0, 1)# .detach()
                else:
                    image_6_views[diff_cam_infos['cam_name'][0]] = diff_outputs['rgb'].permute(2, 0, 1).detach()
                
                # dilate inpainting mask
                inpainting_mask = diff_outputs['uncertainty']['rgb_gaussians'].float().squeeze()
                # inpainting_mask = (diff_outputs['RigidNodes_opacity'] > 0.1).float().squeeze()
                # inpainting_mask = dilate(inpainting_mask, kernel_size=11, gaussian=True)

                inpainting_mask_6_views[diff_cam_infos['cam_name'][0]] = inpainting_mask.unsqueeze(0).detach().repeat(3, 1, 1)
            
            # deal with chosen camera
            for cam_id in range(trainer.n_camera):
                if cam_id != random_camera:
                    continue
                diff_image_infos, diff_cam_infos = clone_tensor_dict_deep(diff_image_infos_l[cam_id]), clone_tensor_dict_deep(diff_cam_infos_l[cam_id])
                for k, v in diff_image_infos.items():
                    if isinstance(v, torch.Tensor):
                        diff_image_infos[k] = v[0].cuda(non_blocking=True)
                for k, v in diff_cam_infos.items():
                    if isinstance(v, torch.Tensor):
                        diff_cam_infos[k] = v[0].cuda(non_blocking=True)
                if 'road_masks' in diff_image_infos:
                    diff_image_infos['road_masks'][-1, -1] = 1.0
                if current_sample_token is None:
                    current_sample_token = diff_image_infos['sample_tokens'][0]
                else:
                    assert diff_image_infos['sample_tokens'][0] == current_sample_token


                # get shift vector in world coordinate
                if diff_cam_infos['cam_name'][0] == 'CAM_FRONT': # TODO Assuming x axis of CAM_FRONT is the same as LiDAR
                    c2w = diff_cam_infos['camera_to_world'].clone()
                    cam_shift_vector_w = torch.tensor([-shift_x, 0, 0], device=c2w.device, dtype=c2w.dtype)
                
                # shift camera
                c2w = diff_cam_infos['camera_to_world'].clone()
                c2w[:3, 3] += cam_shift_vector_w
                diff_cam_infos['camera_to_world'] = c2w
                
                diff_outputs = trainer(diff_image_infos, diff_cam_infos, is_diffusion_step=True)
                if cam_id == random_camera:
                    image_6_views[diff_cam_infos['cam_name'][0]] = diff_outputs['rgb'].permute(2, 0, 1)# .detach()
                else:
                    image_6_views[diff_cam_infos['cam_name'][0]] = diff_outputs['rgb'].permute(2, 0, 1).detach()
                
                # dilate inpainting mask
                inpainting_mask = diff_outputs['uncertainty']['rgb_gaussians'].float().squeeze()
                # inpainting_mask = (diff_outputs['RigidNodes_opacity'] > 0.1).float().squeeze()
                # inpainting_mask = dilate(inpainting_mask, kernel_size=33)
                # inpainting_mask = dilate(inpainting_mask, kernel_size=11, gaussian=True)


                inpainting_mask_6_views[diff_cam_infos['cam_name'][0]] = inpainting_mask.unsqueeze(0).detach().repeat(3, 1, 1)

            
            image_6_views_l = [image_6_views[cam_name] for cam_name in trainer.mgd.mgd_order]
            inpainting_mask_6_views_l = [inpainting_mask_6_views[cam_name] for cam_name in trainer.mgd.mgd_order]
            image_6_views_ts = torch.stack(image_6_views_l, dim=0)
            inpainting_mask_6_views_ts = torch.stack(inpainting_mask_6_views_l, dim=0)
            image_6_views_ts = F.interpolate(image_6_views_ts, size=(224, 400), mode='bilinear', align_corners=False)
            inpainting_mask_6_views_ts = F.interpolate(inpainting_mask_6_views_ts, size=(224, 400), mode='bilinear')
            # image_6_views_ts = F.interpolate(image_6_views_ts, size=(424, 800), mode='bilinear', align_corners=False)
            # inpainting_mask_6_views_ts = F.interpolate(inpainting_mask_6_views_ts, size=(424, 800), mode='bilinear')
            
            inpainting_mask_6_views_ts = (1 - inpainting_mask_6_views_ts).clip(0, 1)
            inpainting_mask_6_views_ts = dilate(inpainting_mask_6_views_ts, kernel_size=45, gaussian=True).clip(0, 1)
            image_6_views_ts = (image_6_views_ts.unsqueeze(0) - 0.5) * 2.0

            # rint = 0
            rint = random.randint(0, 5)
            frame_idx = diff_image_infos['frame_idx'].flatten()[0].item()
            if rint == 0:
                with torch.no_grad():
                    blended_img = image_6_views_ts * inpainting_mask_6_views_ts.unsqueeze(0) + (1 - inpainting_mask_6_views_ts.unsqueeze(0)) * torch.tensor([-1, 1, -1], device=image_6_views_ts.device).view(1, 1, 3, 1, 1)
                    trainer.mgd.save_image0_from_pixels(blended_img, f'blended_{frame_idx}_{step}_{random_camera}.png')
                    trainer.mgd.save_image0_from_pixels(image_6_views_ts, f'image_views_{frame_idx}_{step}_{random_camera}.png')
                    trainer.mgd.save_image0_from_pixels((inpainting_mask_6_views_ts.unsqueeze(0) - 0.5) * 2.0, f'inpainting_mask_{frame_idx}_{step}_{random_camera}.png')
            

            inpainting_mask_6_views_ts_ss = inpainting_mask_6_views_ts.reshape(image_6_views_ts.shape)
            image_6_views_ts = image_6_views_ts * inpainting_mask_6_views_ts_ss + (1 - inpainting_mask_6_views_ts_ss) * image_6_views_ts

            kwargs = {
                "resample": 2,
                "num_ts": 10,
                "ts": timestep,
                "inmask_g_scale": 2.0,
                "stochastic": False,
                "cfg_scale": 2.0,
                "scene_idx": trainer.scene_idx
            }

            loss, retdict = trainer.mgd.get_loss(image_6_views_ts, current_sample_token, None, shift_x=shift_x, step=step, method=diff_method, inpainting_mask=torch.ones_like(inpainting_mask_6_views_ts), **kwargs)

            loss *= diff_loss_weight



            print(f'diff loss: {loss}')
            loss_dict = {}
            loss_dict['diff_loss'] = loss
            target_img_latent = retdict['target_latent'].detach()
            if rint == 0:
                trainer.mgd.save_image0_from_latents(target_img_latent, f'target_img_latent_{frame_idx}_{step}_{timestep}.png')


            # kwargs = {
            #     "resample": 2,
            #     "num_ts": 10,
            #     "ts": 1000,
            #     "inmask_g_scale": 0
            # }
            # if True:
            #     kwargs = {
            #     "resample": 3,
            #     "num_ts": 25,
            #     "ts": 300,
            #     "inmask_g_scale": 0.5,
            #     "stochastic": False,
            #     "cfg_scale": 2.0
            # }
            #     loss, retdict = trainer.mgd.get_loss(image_6_views_ts, current_sample_token, None, shift_x=shift_x, step=step, method=diff_method, inpainting_mask=inpainting_mask_6_views_ts, **kwargs)

            #     loss *= diff_loss_weight



            #     print(f'diff loss: {loss}')
            #     loss_dict = {}
            #     loss_dict['diff_loss'] = loss
            #     target_img_latent = retdict['target_latent'].detach()
            #     if True:
            #         trainer.mgd.save_image0_from_pixels(image_6_views_ts, f'image_views_{frame_idx}_{step}_{random_camera}.png')
            #         trainer.mgd.save_image0_from_latents(target_img_latent, f'target_img_latent_{frame_idx}_{step}_{timestep}.png')


            # check nan or inf
            for k, v in loss_dict.items():
                if torch.isnan(v).any():
                    raise ValueError(f"NaN detected in loss {k} at step {step}")
                if torch.isinf(v).any():
                    raise ValueError(f"Inf detected in loss {k} at step {step}")
            trainer.backward(loss_dict, is_diffusion_step=True)

            # after training step
            trainer.postprocess_per_train_step(step=step, diff_grad=True, do_refinement=False)

            
    with open(os.path.join(trainer.log_dir, "sample2shift_vec.json"), 'w') as f:
        json.dump({k: v.tolist() for k, v in sample2shift_vec.items()}, f)
    logger.info("Training done!")
    
    if trainer.export_neus_2dgs:
    #     print("exporting neus to 2dgs")
    #     trainer.neus23dgs(output=True)
    #     trainer.ground_method = "rsg"
        trainer.save_checkpoint(
                log_dir=cfg.log_dir,
                save_only_model=True,
                is_final=False,
                ckpt_name="checkpoint_neus_2dgs",
            )

    do_evaluation(
        step=step,
        cfg=cfg,
        trainer=trainer,
        dataset=dataset,
        render_keys=['rgbs', 'gt_rgbs', 'lidar_on_images'],
        args=args,
        shift_x=max(shift_x_l)
    )
    
    if args.enable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)
    
    return step

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Gaussian Splatting for a single scene")
    parser.add_argument("--config_file", help="path to config file", type=str)
    parser.add_argument("--output_root", default="./work_dirs/", help="path to save checkpoints and logs", type=str)
    
    # eval
    parser.add_argument("--resume_from", default=None, help="path to checkpoint to resume from", type=str)
    parser.add_argument("--render_video_postfix", type=str, default=None, help="an optional postfix for video")    
    
    # wandb logging part
    parser.add_argument("--enable_wandb", action="store_true", help="enable wandb logging")
    parser.add_argument("--entity", default="ziyc", type=str, help="wandb entity name")
    parser.add_argument("--project", default="drivestudio", type=str, help="wandb project name, also used to enhance log_dir")
    parser.add_argument("--run_name", default="omnire", type=str, help="wandb run name, also used to enhance log_dir")
    
    # viewer
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=8080, help="viewer port")
    
    # misc
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    # args.enable_wandb = True
    final_step = main(args)
