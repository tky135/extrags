import logging
import os
import contextlib
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.random
import torchvision
import torch.nn.functional as F
from einops import rearrange, repeat

from diffusers import (
    ModelMixin,
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers.optimization import get_scheduler

from ..misc.common import load_module, convert_outputs_to_fp16, move_to
from .base_runner import BaseRunner
from .utils import smart_param_count
# from magicdrive.runner.utils import concat_6_views
import random
import torch
import numpy as np
from PIL import Image

import datetime

prefix_current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
import os
prefix_current_time = prefix_current_time + f"_{os.environ['exp_name']}"
if not os.path.exists(prefix_current_time):
    os.makedirs(prefix_current_time)


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

def img_m11_to_01(img):
    return img * 0.5 + 0.5




class ControlnetUnetWrapper(ModelMixin):
    """As stated in https://github.com/huggingface/accelerate/issues/668, we
    should not use accumulate provided by accelerator, but create a wrapper to
    two modules.
    """

    def __init__(self, controlnet, unet, weight_dtype=torch.float32,
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
        import ipdb ; ipdb.set_trace()
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


class MultiviewRunner(BaseRunner):
    def __init__(self, cfg, accelerator, train_set, val_set) -> None:
        super().__init__(cfg, accelerator, train_set, val_set)
    def save_image0_from_pixels(self, pixels, path):
        pixels = concat_6_view(pixels.detach().float().cpu())

        pixels = (pixels / 2 + 0.5).clamp(0, 1) * 255
        pixels = pixels.type(torch.uint8)
        torchvision.io.write_png(pixels, os.path.join(prefix_current_time, path))
    def save_image0_from_latents(self, latents, path):
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

    def _init_fixed_models(self, cfg):
        # fmt: off
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="vae")
        self.noise_scheduler = DDPMScheduler.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="scheduler")
        # fmt: on

    def _init_trainable_models(self, cfg):
        # fmt: off
        # fmt: on
        # 0113 modified to load pretrained unet and controlnet
        model_cls = load_module(cfg.model.unet_module)
        self.unet = model_cls.from_pretrained('pretrained/SDv1.5mv-rawbox_2023-09-07_18-39_224x400', subfolder="unet")

        model_cls = load_module(cfg.model.model_module)
        self.controlnet = model_cls.from_pretrained('pretrained/SDv1.5mv-rawbox_2023-09-07_18-39_224x400', subfolder="controlnet")
    def _set_model_trainable_state(self, train=True):
        # set trainable status
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.controlnet.train(train)
        self.unet.requires_grad_(False)
        for name, mod in self.unet.trainable_module.items():
            logging.debug(
                f"[MultiviewRunner] set {name} to requires_grad = True")
            mod.requires_grad_(train)

    def set_optimizer_scheduler(self):
        # optimizer and lr_schedulers
        if self.cfg.runner.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # Optimizer creation
        params_to_optimize = list(self.controlnet.parameters())
        unet_params = self.unet.trainable_parameters
        param_count = smart_param_count(unet_params)
        logging.info(
            f"[MultiviewRunner] add {param_count} params from unet to optimizer.")
        params_to_optimize += unet_params
        self.optimizer = optimizer_class(
            params_to_optimize,
            lr=self.cfg.runner.learning_rate,
            betas=(self.cfg.runner.adam_beta1, self.cfg.runner.adam_beta2),
            weight_decay=self.cfg.runner.adam_weight_decay,
            eps=self.cfg.runner.adam_epsilon,
        )

        # lr scheduler
        self._calculate_steps()
        # fmt: off
        self.lr_scheduler = get_scheduler(
            self.cfg.runner.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.runner.lr_warmup_steps * self.cfg.runner.gradient_accumulation_steps,
            num_training_steps=self.cfg.runner.max_train_steps * self.cfg.runner.gradient_accumulation_steps,
            num_cycles=self.cfg.runner.lr_num_cycles,
            power=self.cfg.runner.lr_power,
        )
        # fmt: on

    def prepare_device(self):
        self.controlnet_unet = ControlnetUnetWrapper(self.controlnet, self.unet)
        # accelerator
        ddp_modules = (
            self.controlnet_unet,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        )
        ddp_modules = self.accelerator.prepare(*ddp_modules)
        (
            self.controlnet_unet,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        ) = ddp_modules

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Move vae, unet and text_encoder to device and cast to weight_dtype
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        if self.cfg.runner.unet_in_fp16 and self.weight_dtype == torch.float16:
            self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
            # move optimized params to fp32. TODO: is this necessary?
            if self.cfg.model.use_fp32_for_unet_trainable:
                for name, mod in self.unet.trainable_module.items():
                    logging.debug(f"[MultiviewRunner] set {name} to fp32")
                    mod.to(dtype=torch.float32)
                    mod._original_forward = mod.forward
                    # autocast intermediate is necessary since others are fp16
                    mod.forward = torch.cuda.amp.autocast(
                        dtype=torch.float16)(mod.forward)
                    # we ensure output is always fp16
                    mod.forward = convert_outputs_to_fp16(mod.forward)
            else:
                raise TypeError(
                    "There is an error/bug in accumulation wrapper, please "
                    "make all trainable param in fp32.")
        controlnet_unet = self.accelerator.unwrap_model(self.controlnet_unet)
        controlnet_unet.weight_dtype = self.weight_dtype
        controlnet_unet.unet_in_fp16 = self.cfg.runner.unet_in_fp16

        with torch.no_grad():
            self.accelerator.unwrap_model(self.controlnet).prepare(
                self.cfg,
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder
            )

        # We need to recalculate our total training steps as the size of the
        # training dataloader may have changed.
        self._calculate_steps()

    def _save_model(self, root=None):
        if root is None:
            root = self.cfg.log_root
        # if self.accelerator.is_main_process:
        controlnet = self.accelerator.unwrap_model(self.controlnet)
        controlnet.save_pretrained(
            os.path.join(root, self.cfg.model.controlnet_dir))
        unet = self.accelerator.unwrap_model(self.unet)
        unet.save_pretrained(os.path.join(root, self.cfg.model.unet_dir))
        logging.info(f"Save your model to: {root}")

    def _train_one_stop(self, batch):
        """
        batch: 
            pixel_values [b, N_cam, 3, H, W],
            camera_param [b, N_cam, 3, 7],
            input_ids [b, 33(text_len)], 文字描述
            uncond_ids [b, 33(text_len)], 无条件描述
            bev_map_with_aux [b, 8, 200, 200]
            
            
        """
        print(batch['meta_data']['metas'][0].data['token'])
        visualize_class_map(batch['bev_map_with_aux'][0], "segmentation.png")

        # hack pixel_values
        recon_image = []
        recon_image_name = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
        for name in recon_image_name:
            image = torchvision.io.read_image(f"recon_origin/045_{name}.png").float() / 255.
            image = 2 * (image - 0.5)
            image = F.interpolate(image.unsqueeze(0), size=(224, 400), mode='bilinear', align_corners=False).squeeze(0)
            recon_image.append(image)
        recon_image = torch.stack(recon_image).unsqueeze(0)
        self.save_image0_from_pixels(recon_image, "image0_recon.png")
        batch['pixel_values'] = recon_image.to(batch['pixel_values'].device).to(batch['pixel_values'].dtype)


        bboxes = batch['kwargs']['bboxes_3d_data']['bboxes']
        # bboxes[:, :, :, :, 0] += 3.
        batch['kwargs']['bboxes_3d_data']['bboxes'] = bboxes
        


        self.controlnet_unet.train()
        self.controlnet_unet.controlnet.drop_cond_ratio = 0.0
        with self.accelerator.accumulate(self.controlnet_unet):

            # Get the text embedding for conditioning
            encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]
            encoder_hidden_states_uncond = self.text_encoder(
                batch
                ["uncond_ids"])[0]

            controlnet_image = batch["bev_map_with_aux"].to(
                dtype=self.weight_dtype)
            

            camera_param = batch["camera_param"].to(self.weight_dtype)


            N_cam = batch["pixel_values"].shape[1]
            self.save_image0_from_pixels(batch["pixel_values"], "image0.png")

            ### add artifacts to batch['pixel_values]
            print("====================Begin inpainting hack at _train_one_stop====================")
            # method #1: gaussian noise in image space
            # batch["pixel_values"] = batch["pixel_values"] * 0.5 + torch.randn_like(batch["pixel_values"]) * 0.5

            # method #3: replace patch with gaussian noise
            # batch['pixel_values'][:, :, :, 112: 112+32, 200: 200+32] = torch.randn_like(batch['pixel_values'][:, :, :, 112: 112+32, 200: 200+32])


            # method #4: replace patch with mean value
            # batch['pixel_values'][:, :, :, 112: 112+32, 200: 200+32] = torch.randn_like(batch['pixel_values'][:, :, :, 112: 112+32, 200: 200+32], dim=(3, 4), keepdim=True).repeat(1, 1, 1, 32, 32)
            # self.save_image0_from_pixels(batch["pixel_values"], "image0_disturbed.png")

            # method #5: replace patch with zeros/randn
            inpainting_mask = torch.zeros_like(batch['pixel_values'])
            # inpainting_mask[:, 4, :, 112:112+64, 28:172] = 1  # mask of shifted image
            inpainting_mask[:, 2, :, 30:, :] = 1
            # for i in range(224):
            #     for j in range(400):
            #         if i % 32 == 0 and j % 32 == 0 and i + 32 < 224 and j + 32 < 400:
            #             inpainting_mask[:, :, :, i:i + 8, j:j+8] = 1
            # inpainting_mask[:, :, :, 112: 112+32, 200: 200+32] = 1
            # batch['pixel_values'][inpainting_mask == 1] = torch.zeros_like(batch['pixel_values'][inpainting_mask == 1])

            input_pixels = batch['pixel_values'].clone()

            # resize inpainting mask to 28 x 50
            inpainting_mask_vae = F.interpolate(inpainting_mask[0], size=(28, 50), mode='nearest').unsqueeze(0)
            inpainting_mask_vae = torch.cat([inpainting_mask_vae[:, :, :1, :, :], inpainting_mask_vae], dim=2)
            print("====================End of inpainting hack at _train_one_stop====================")

            # print("====================Begin inverse inference hack at _train_one_stop====================")
            # Convert images to latent space
            latents = self.vae.encode(
                rearrange(batch["pixel_values"], "b n c h w -> (b n) c h w").to(
                    dtype=self.weight_dtype
                )
            ).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            latents = rearrange(latents, "(b n) c h w -> b n c h w", n=N_cam)

            input_latents = latents.clone()

            self.save_image0_from_latents(latents, "image0_decoded.png")

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            # make sure we use same noise for different views, only take the
            # first
            if self.cfg.model.train_with_same_noise:
                noise = repeat(noise[:, 0], "b ... -> b r ...", r=N_cam)

            bsz = latents.shape[0]
            
            timesteps = torch.tensor([200]).long()
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self._add_noise(latents, noise, timesteps)
            # noisy_latents = torch.randn_like(latents)
            self.save_image0_from_latents(noisy_latents, "image0_noisy.png")

            # denoising loop
            # assuming batch size = 1
            with torch.no_grad():
                for i in range(int(timesteps[0]), 0, -1):
                    model_pred = self.controlnet_unet(
                        noisy_latents, torch.tensor(i).unsqueeze(0).cuda(), camera_param, encoder_hidden_states,
                        encoder_hidden_states_uncond, controlnet_image,
                        **batch['kwargs'],
                    )
                    output = self.noise_scheduler.step(model_pred[0], i, noisy_latents[0], inpaint_mask=inpainting_mask_vae[0], inpaint_condition=input_latents[0])
                    # output = self.noise_scheduler.step(model_pred[0], i, noisy_latents[0])
                    noisy_latents = output['prev_sample'].unsqueeze(0).half()
                    pred_x0 = output['pred_original_sample'].half()
                    self.save_image0_from_latents(pred_x0.unsqueeze(0).detach(), f"image0_pred_{i}.png")

            print("====================End of inverse inference hack at _train_one_stop====================")
            import ipdb ; ipdb.set_trace()


            # print("====================Begin SDS hack at _train_one_stop====================")
            # image_params = torch.nn.Parameter(torch.randn_like(batch['pixel_values']), requires_grad=True)

            # optimizer = torch.optim.Adam([image_params], lr=1e-2)


            # for i in range(10000):
            #     optimizer.zero_grad()
            #     # if i < 2500:
            #     #     t = int(torch.randint(1, 1000, (1,)).long().item())
            #     # elif i < 5000:
            #     #     t = int(torch.randint(1, 500, (1,)).long().item())
            #     # elif i < 7500:
            #     #     t = int(torch.randint(1, 250, (1,)).long().item())
            #     # else:
            #     #     t = int(torch.randint(1, 200, (1,)).long().item())
            #     t = torch.randint(1, 1000, (1,)).long().item()
            #     image_x = image_params
            #     # image_x = image_params * inpainting_mask + image_params.detach() * (1 - inpainting_mask)
            #     # image_x = torch.sigmoid(image_params)
            #     # image_x = inpainting_gt_img
            #     # Convert images to latent space
            #     sds_img_vae = self.vae.encode(
            #         rearrange(image_x, "b n c h w -> (b n) c h w").to(
            #             dtype=self.weight_dtype
            #         )
            #     ).latent_dist.sample()
            #     sds_img_vae = sds_img_vae * self.vae.config.scaling_factor
            #     sds_img_vae = rearrange(sds_img_vae, "(b n) c h w -> b n c h w", n=N_cam)

            #     # sds_img_vae = sds_img_vae * inpainting_mask_vae.half() + input_latents * (1 - inpainting_mask_vae).half()

            #     # self.save_image0_from_latents(sds_img_vae, "sds_img_vae.png")


            #     with torch.no_grad():
            #         # add noise at timestep t
            #         noisy_latents = self._add_noise(sds_img_vae, torch.randn_like(sds_img_vae), torch.tensor(t).unsqueeze(0).cuda())


            #         # predict noise residual epsilon
            #         model_pred = self.controlnet_unet(
            #             noisy_latents, torch.tensor(t).unsqueeze(0).cuda(), camera_param, encoder_hidden_states,
            #             encoder_hidden_states_uncond, controlnet_image,
            #             **batch['kwargs'],
            #         )

            #         target_latent = self._remove_noise(noisy_latents, model_pred, torch.tensor(t).unsqueeze(0).cuda()).half()
                    
            #         # method 1: from pseudo code
            #         # g = torch.dot((model_pred.half() - noise).detach().flatten(), sds_img_vae.flatten())
            #         # g.backward()
            #         w = 1. - self.noise_scheduler.alphas[t].cuda()
            #         grad = w * (model_pred.half() - noise)
            #         grad = torch.nan_to_num(grad)
            #         target = (sds_img_vae - grad).detach()
            #     loss = 0.5 * F.mse_loss(sds_img_vae, target_latent.detach(), reduction='sum')
            #     loss.backward()

            #     optimizer.step()
            #     print(f"step {i}, loss {loss.item()}, t {t}")
            #     if i % 100 == 0:
            #         self.save_image0_from_latents(sds_img_vae, f"sds_img_vae_{i}.png")
            #         self.save_image0_from_pixels(image_x, f"sds_img_{i}.png")
            #         self.save_image0_from_latents(target_latent, f"sds_img_remove_noise_target_{i}_{t}.png")
            #         # self.save_image0_from_latents(target.half(), f"sds_img_gradient_target_{i}_{t}_grad.png")







            # print("====================End of SDS hack at _train_one_stop====================")
            # import ipdb ; ipdb.set_trace()



            print("====================Begin SJC hack at _train_one_stop====================")

            batch_size = 1
            image_params = torch.nn.Parameter(batch['pixel_values'].cuda(), requires_grad=True)
            # image_params = torch.nn.Parameter(torch.zeros_like(batch['pixel_values']).cuda(), requires_grad=True)

            optimizer = torch.optim.Adam([image_params], lr=1e-2)


            camera_param_batched = camera_param.repeat(batch_size, 1, 1, 1)
            encoder_hidden_states_batched = encoder_hidden_states.repeat(batch_size, 1, 1)
            # encoder_hidden_states_uncond_batched = encoder_hidden_states_uncond.repeat(batch_size, 1, 1)
            controlnet_image_batched = controlnet_image.repeat(batch_size, 1, 1, 1)
            bboxes_3d_data_batched = batch['kwargs']['bboxes_3d_data']
            bboxes_3d_data_batched['bboxes'] = bboxes_3d_data_batched['bboxes'].repeat(batch_size, 1, 1, 1, 1)
            bboxes_3d_data_batched['classes'] = bboxes_3d_data_batched['classes'].repeat(batch_size, 1, 1)
            bboxes_3d_data_batched['masks'] = bboxes_3d_data_batched['masks'].repeat(batch_size, 1, 1)


            for i in range(5001):
                

                optimizer.zero_grad()
                # if i < 2500:
                #     t = int(torch.randint(1, 1000, (1,)).long().item())
                # elif i < 5000:
                #     t = int(torch.randint(1, 500, (1,)).long().item())
                # elif i < 7500:
                #     t = int(torch.randint(1, 250, (1,)).long().item())
                # else:
                #     t = int(torch.randint(1, 200, (1,)).long().item())
                # t = torch.randint(300, 500, (1,)).long().item()
                t = sample_gaussian_around_t(200 * (5000 - i) // 5000, sigma=30)
                # image_x = torch.tanh(image_params)
                # image_x = image_params * inpainting_mask + image_params.detach() * (1 - inpainting_mask)
                image_x = image_params.clip(-1, 1)

                # mask_1d = torch.zeros(28 * 50, dtype=torch.float16).cuda()

                # # Randomly select 1024 distinct indices out of 200*200
                # indices = torch.randperm(28 * 50)[:64]

                # # Set those indices to 1
                # mask_1d[indices] = 1.0

                # # Reshape to 200 x 200
                # mask = mask_1d.view(28, 50).reshape(1, 1, 1, 28, 50).repeat(1, 6, 4, 1, 1)


                # image_x = image_x * mask + batch['pixel_values'] * (1 - mask)
                

                # image_x = image_params * inpainting_mask + image_params.detach() * (1 - inpainting_mask)
                # image_x = torch.sigmoid(image_params)
                # image_x = inpainting_gt_img
                # Convert images to latent space
                sds_img_vae = self.vae.encode(
                    rearrange(image_x, "b n c h w -> (b n) c h w").to(
                        dtype=self.weight_dtype
                    )
                ).latent_dist.sample()
                sds_img_vae = sds_img_vae * self.vae.config.scaling_factor
                sds_img_vae = rearrange(sds_img_vae, "(b n) c h w -> b n c h w", n=N_cam)
                sds_img_vae_batched = sds_img_vae.repeat(batch_size, 1, 1, 1, 1)
                
                # sds_img_vae = sds_img_vae * (1 - mask) + input_latents * mask
                sds_img_vae = sds_img_vae * inpainting_mask_vae.half() + input_latents * (1 - inpainting_mask_vae).half()

                # self.save_image0_from_latents(sds_img_vae, "sds_img_vae.png")


                with torch.no_grad():

                    # add noise at timestep t
                    noisy_latents = self._add_noise(sds_img_vae_batched, torch.randn_like(sds_img_vae_batched), torch.tensor(t).unsqueeze(0).cuda())


                    # predict noise residual epsilon
                    self.controlnet_unet.controlnet.drop_cond_ratio = 0.0
                    import ipdb ; ipdb.set_trace()
                    model_pred = self.controlnet_unet(
                        noisy_latents, torch.tensor(t).unsqueeze(0).cuda().repeat(batch_size), camera_param_batched, encoder_hidden_states_batched, encoder_hidden_states_uncond,
                        controlnet_image_batched,
                        bboxes_3d_data=bboxes_3d_data_batched,
                    )



                    self.controlnet_unet.controlnet.drop_cond_ratio = 1.0
                    model_pred_uncod = self.controlnet_unet(
                        noisy_latents, torch.tensor(t).unsqueeze(0).cuda().repeat(batch_size), camera_param_batched, encoder_hidden_states_batched, encoder_hidden_states_uncond,
                        controlnet_image_batched,
                        bboxes_3d_data=bboxes_3d_data_batched,
                    )

                    model_pred = model_pred_uncod + 2.0 * (model_pred - model_pred_uncod)


                    

                    target_latent = self._remove_noise(noisy_latents, model_pred, torch.tensor(t).unsqueeze(0).cuda().repeat(batch_size)).half()
                    target_latent = target_latent.mean(dim=0, keepdim=True)
                    
                    # method 1: from SDS pseudo code
                    # g = torch.dot((model_pred.half() - noise).detach().flatten(), sds_img_vae.flatten())
                    # g.backward()
                    # w = 1. - self.noise_scheduler.alphas[t].cuda()
                    # grad = w * (model_pred.half() - noise)
                    # grad = torch.nan_to_num(grad)
                    # target = (sds_img_vae - grad).detach()
                loss = 0.5 * F.mse_loss(sds_img_vae, target_latent.detach(), reduction='sum')

                # make sure the image is consistent with the decoder output
                # image consistency loss
                with torch.no_grad():
                    target_latent_detached = sds_img_vae.detach()
                    bs = len(target_latent_detached)
                    target_latent_detached = 1 / self.vae.config.scaling_factor * target_latent_detached
                    target_latent_detached = rearrange(target_latent_detached, 'b c ... -> (b c) ...')
                    target_img = self.vae.decode(target_latent_detached).sample
                    target_img = rearrange(target_img, '(b c) ... -> b c ...', b=bs)
                image_consis_loss = F.mse_loss(target_img.detach(), image_x.half(), reduction='mean') * 2e4
                loss += image_consis_loss
                loss.backward()

                optimizer.step()
                print(f"step {i}, loss {loss.item()} t {t}")
                # print(f"step {i}, loss {loss.item()}, image_consis_loss {image_consis_loss.item()} t {t}")
                if i % 100 == 0:
                    self.save_image0_from_latents(sds_img_vae, f"sds_img_vae_{i}.png")
                    self.save_image0_from_pixels(image_x, f"sds_img_{i}.png")
                    self.save_image0_from_latents(target_latent, f"sds_img_remove_noise_target_{i}_{t}.png")
                    # self.save_image0_from_latents(target.half(), f"sds_img_gradient_target_{i}_{t}_grad.png")
            








            print("====================End of SJC hack at _train_one_stop====================")
            import ipdb ; ipdb.set_trace()





            N_cam = batch["pixel_values"].shape[1]

            # Convert images to latent space
            latents = self.vae.encode(
                rearrange(batch["pixel_values"], "b n c h w -> (b n) c h w").to(
                    dtype=self.weight_dtype
                )
            ).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            latents = rearrange(latents, "(b n) c h w -> b n c h w", n=N_cam)

            # embed camera params, in (B, 6, 3, 7), out (B, 6, 189)
            # camera_emb = self._embed_camera(batch["camera_param"])
            camera_param = batch["camera_param"].to(self.weight_dtype)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            # make sure we use same noise for different views, only take the
            # first
            if self.cfg.model.train_with_same_noise:
                noise = repeat(noise[:, 0], "b ... -> b r ...", r=N_cam)

            bsz = latents.shape[0]
            # Sample a random timestep for each image
            if self.cfg.model.train_with_same_t:
                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
            else:
                timesteps = torch.stack([torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ) for _ in range(N_cam)], dim=1)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self._add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]
            encoder_hidden_states_uncond = self.text_encoder(
                batch
                ["uncond_ids"])[0]

            controlnet_image = batch["bev_map_with_aux"].to(
                dtype=self.weight_dtype)

            model_pred = self.controlnet_unet(
                noisy_latents, timesteps, camera_param, encoder_hidden_states,
                encoder_hidden_states_uncond, controlnet_image,
                **batch['kwargs'],
            )

            # Get the target for loss depending on the prediction type
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(
                    latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                )

            loss = F.mse_loss(
                model_pred.float(), target.float(), reduction='none')
            loss = loss.mean()

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                params_to_clip = self.controlnet_unet.parameters()
                self.accelerator.clip_grad_norm_(
                    params_to_clip, self.cfg.runner.max_grad_norm
                )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(
                set_to_none=self.cfg.runner.set_grads_to_none)

        return loss
