from typing import List, Optional
from omegaconf import OmegaConf
import os
import time
import json
import wandb
import logging
import argparse

import torch
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from utils.logging import setup_logging
from models.trainers import BasicTrainer
from models.video_utils import render

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

# @torch.no_grad()
def do_evaluation(
    step: int = 0,
    cfg: OmegaConf = None,
    trainer: BasicTrainer = None,
    dataset: DrivingDataset = None,
    args: argparse.Namespace = None,
    render_keys: Optional[List[str]] = None,
    post_fix: str = "",
    log_metrics: bool = True
):
    trainer.set_eval()

    logger.info("Evaluating Pixels...")
    if dataset.test_image_set is not None and cfg.render.render_test:
        logger.info("Evaluating Test Set Pixels...")
        if args.render_video_postfix is None:
            video_output_pth = f"{cfg.log_dir}/videos{post_fix}/test_set_{step}.mp4"
        else:
            video_output_pth = (
                f"{cfg.log_dir}/videos{post_fix}/test_set_{step}_{args.render_video_postfix}.mp4"
            )
        if "multibag" in os.environ.get("DATASET"):
            vis_timestamps = min(300, dataset.num_img_timesteps)
            vis_indices = range(vis_timestamps * dataset.pixel_source.num_cams_per_bag)
        else:
            vis_indices = None
        metrics_dict = render(
            dataset=dataset.test_image_set,
            trainer=trainer,
            save_path=video_output_pth,
            layout=dataset.layout,
            num_timestamps=dataset.num_test_timesteps,
            keys=render_keys,
            num_cams=dataset.pixel_source.num_cams_per_bag if "multibag" in os.environ.get("DATASET") else dataset.pixel_source.num_cams,
            save_images=True,
            fps=cfg.render.fps,
            compute_metrics=True,
            compute_error_map=cfg.render.vis_error,
            vis_indices=vis_indices,
            lane_shift=False,
        )
        
        if log_metrics:
            eval_dict = {}
            for k, v in metrics_dict.items():
                if k in [
                    "psnr",
                    "ssim",
                    "lpips",
                    "fid",
                    "occupied_psnr",
                    "occupied_ssim",
                    "masked_psnr",
                    "masked_ssim",
                    "human_psnr",
                    "human_ssim",
                    "vehicle_psnr",
                    "vehicle_ssim",
                ]:
                    eval_dict[f"image_metrics/test/{k}"] = v
            if args.enable_wandb:
                wandb.log(eval_dict)
            test_metrics_file = f"{cfg.log_dir}/metrics{post_fix}/images_test_{current_time}.json"
            with open(test_metrics_file, "w") as f:
                json.dump(eval_dict, f)
            logger.info(f"Image evaluation metrics saved to {test_metrics_file}")
        torch.cuda.empty_cache()
        
    if cfg.render.render_full:
        logger.info("Evaluating Full Set...")
        if args.render_video_postfix is None:
            video_output_pth = f"{cfg.log_dir}/videos{post_fix}/full_set_{step}.mp4"
        else:
            video_output_pth = (
                f"{cfg.log_dir}/videos{post_fix}/full_set_{step}_{args.render_video_postfix}.mp4"
            )
        if "multibag" in os.environ.get("DATASET"):
            vis_timestamps = min(300, dataset.num_img_timesteps)
            vis_indices = range(vis_timestamps * dataset.pixel_source.num_cams_per_bag)
        else:
            vis_indices = None
        metrics_dict = render(
            dataset=dataset.full_image_set,
            trainer=trainer,
            save_path=video_output_pth,
            layout=dataset.layout,
            num_timestamps=dataset.num_img_timesteps,
            keys=render_keys,
            num_cams=dataset.pixel_source.num_cams_per_bag if "multibag" in os.environ.get("DATASET") else dataset.pixel_source.num_cams,
            save_images=True,
            fps=cfg.render.fps,
            compute_metrics=True,
            compute_error_map=cfg.render.vis_error,
            vis_indices=vis_indices,
            lane_shift=False,
        )
        
        if log_metrics:
            eval_dict = {}
            for k, v in metrics_dict.items():
                
                if k in [
                    "psnr",
                    "ssim",
                    "lpips",
                    "fid",
                    "occupied_psnr",
                    "occupied_ssim",
                    "masked_psnr",
                    "masked_ssim",
                    "human_psnr",
                    "human_ssim",
                    "vehicle_psnr",
                    "vehicle_ssim",
                ]:
                    eval_dict[f"image_metrics/full/{k}"] = v
            if args.enable_wandb:
                wandb.log(eval_dict)
            full_metrics_file = f"{cfg.log_dir}/metrics{post_fix}/images_full_{current_time}.json"
            with open(full_metrics_file, "w") as f:
                json.dump(eval_dict, f)
            logger.info(f"Image evaluation metrics saved to {full_metrics_file}")

        torch.cuda.empty_cache()
    
    render_novel_cfg = cfg.render.get("render_novel", None)
    if render_novel_cfg is not None:
        logger.info("Rendering novel views...")
        if "multibag" in os.environ.get("DATASET"):
            vis_timestamps = min(300, dataset.num_img_timesteps)
            vis_indices = range(vis_timestamps * dataset.pixel_source.num_cams_per_bag)
        else:
            vis_indices = None
        if args.render_video_postfix is None:
            video_output_pth = f"{cfg.log_dir}/videos{post_fix}/shift_set_{step}.mp4"
        else:
            video_output_pth = (
                f"{cfg.log_dir}/videos{post_fix}/shift_set_{step}_{args.render_video_postfix}.mp4"
            )
        metrics_dict = render(
            dataset=dataset.full_image_set,
            trainer=trainer,
            save_path=video_output_pth,
            layout=dataset.layout,
            num_timestamps=dataset.num_img_timesteps,
            keys=render_keys,
            num_cams=dataset.pixel_source.num_cams_per_bag if "multibag" in os.environ.get("DATASET") else dataset.pixel_source.num_cams,
            save_images=True,
            fps=cfg.render.fps,
            compute_metrics=True,
            compute_error_map=cfg.render.vis_error,
            vis_indices=vis_indices,
            lane_shift=True,
        )
        
        if log_metrics:
            eval_dict = {}
            for k, v in metrics_dict.items():
                
                if k in [
                    "psnr",
                    "ssim",
                    "lpips",
                    "fid",
                    "occupied_psnr",
                    "occupied_ssim",
                    "masked_psnr",
                    "masked_ssim",
                    "human_psnr",
                    "human_ssim",
                    "vehicle_psnr",
                    "vehicle_ssim",
                ]:
                    eval_dict[f"image_metrics/shift/{k}"] = v
            if args.enable_wandb:
                wandb.log(eval_dict)
            shift_metrics_file = f"{cfg.log_dir}/metrics{post_fix}/images_shift_{current_time}.json"
            with open(shift_metrics_file, "w") as f:
                json.dump(eval_dict, f)
            logger.info(f"Image evaluation metrics saved to {shift_metrics_file}")
        torch.cuda.empty_cache()
def main(args):
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    export_neus_2dgs = cfg.trainer.export_neus_2dgs
    os.environ.update({
        "LOG_DIR": log_dir,
        "DATASET": cfg.dataset,
    })
    args.enable_wandb = False
    for folder in ["videos_eval", "metrics_eval"]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # setup logging
    global logger
    setup_logging(output=log_dir, level=logging.INFO, time_string=current_time)

    # build dataset
    dataset = DrivingDataset(data_cfg=cfg.data)

    # setup trainer
    cfg.trainer.dataset = cfg.data.dataset
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set.split_indices),
        num_full_images=len(dataset.full_image_set.split_indices),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
        dataset_obj=dataset,
    )
    trainer.init_gaussians_from_dataset(dataset, fast_run=True)
    # Resume from checkpoint
    trainer.resume_from_checkpoint(
        ckpt_path=args.resume_from,
        load_only_model=True
    )
    logger.info(
        f"Resuming training from {args.resume_from}, starting at step {trainer.step}"
    )
    
    # if export_neus_2dgs:
    #     print("exporting neus to 2dgs")
    #     # trainer.neus23dgs(output=True)
    #     # trainer.ground_method = "rsg"
    #     # trainer.save_checkpoint(
    #     #         log_dir=cfg.log_dir,
    #     #         save_only_model=True,
    #     #         is_final=False,
    #     #         ckpt_name="checkpoint_neus_2dgs",
    #     #     )
    if args.enable_viewer:
        # a simple viewer for background visualization
        trainer.init_viewer(port=args.viewer_port)
    
    # define render keys
    render_keys = [
        "gt_rgbs",
        "rgbs",
        # "Background_rgbs",
        # "RigidNodes_rgbs",
        # "DeformableNodes_rgbs",
        # "SMPLNodes_rgbs",
        # "depths",
        # "Background_depths",
        # "RigidNodes_depths",
        # "DeformableNodes_depths",
        # "SMPLNodes_depths",
        # "mask",
        # "lidar_on_images",
        # "rgb_sky_blend",
        # "rgb_sky",
        # "rgb_error_maps"
    ]
    
    if args.save_catted_videos:
        cfg.logging.save_seperate_video = False
    
    do_evaluation(
        step=trainer.step,
        cfg=cfg,
        trainer=trainer,
        dataset=dataset,
        render_keys=render_keys,
        args=args,
        post_fix="_eval"
    )
    
    if args.enable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Gaussian Splatting for a single scene")    
    # eval
    parser.add_argument("--resume_from", default=None, help="path to checkpoint to resume from", type=str, required=True)
    parser.add_argument("--render_video_postfix", type=str, default=None, help="an optional postfix for video")    
    parser.add_argument("--save_catted_videos", type=bool, default=False, help="visualize lidar on image")
    
    # viewer
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=8080, help="viewer port")
        
    # misc
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    main(args)