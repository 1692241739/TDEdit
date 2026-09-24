# tdedit_core.py

import os
import time
import torch
from diffusers import LCMScheduler
from PIL import Image, ImageFont, ImageDraw
import torch.nn.functional as nnf
from typing import Optional, Union, Tuple, List, Callable, Dict, Any
import abc
import utils_text.ptp_utils as ptp_utils
import utils
import numpy as np
import utils_text.seq_aligner as seq_aligner
import math
import random
import torch.nn.functional as F
import cv2
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
)
from utils.unet_drag.unet_2d_condition import UNet2DConditionModel
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin, StableDiffusionLoraLoaderMixin
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from utils import MODEL_INIT_LOCK
from utils.ui_utils import apply_drag
from utils_drag.hole_fill_modes import DEFAULT_HOLE_FILL_MODE, normalize_hole_fill_mode
from tdedit_paths import MODEL_PATH
import inspect
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from utils.unet_drag.unet_2d_condition import UNet2DConditionOutput

# 设置环境变量
os.environ['GRADIO_TEMP_DIR'] = '/tmp'

# 全局配置
LOW_RESOURCE = False
MAX_NUM_WORDS = 77
model_id_or_path = MODEL_PATH
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
# --------------------------
# Pipeline 和类定义
# --------------------------

# (此处保留所有原来的 override_forward, EditPipeline, AttentionControl 等类定义)
# 为了代码完整性，我将原来的 EditPipeline 类和相关辅助函数放在这里

logger = logging.get_logger(__name__)


def _count_drag_pairs(points) -> int:
    """Count valid drag handle/target pairs from alternating XY points."""
    if points is None:
        return 0

    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()

    rows = []
    if isinstance(points, np.ndarray):
        if points.ndim == 2:
            rows = points
        elif points.ndim == 1 and points.shape[0] >= 2:
            if points.dtype == object and isinstance(points[0], (list, tuple, np.ndarray)):
                rows = list(points)
            else:
                rows = [points]
    elif isinstance(points, (list, tuple)):
        if len(points) > 0 and isinstance(points[0], (list, tuple, np.ndarray)):
            rows = points
        elif len(points) >= 2:
            rows = [points]

    valid_points = 0
    for row in rows:
        try:
            x = float(row[0])
            y = float(row[1])
        except Exception:
            continue
        if np.isfinite(x) and np.isfinite(y):
            valid_points += 1

    return valid_points // 2


def _normalize_prompt_signature(prompt) -> str:
    if prompt is None:
        return ""
    if isinstance(prompt, (list, tuple)):
        return " || ".join(str(item).strip() for item in prompt)
    return str(prompt).strip()


def _has_prompt_delta(source_prompt, target_prompt) -> bool:
    return _normalize_prompt_signature(source_prompt) != _normalize_prompt_signature(target_prompt)


def override_forward(self):
    def forward(
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
            down_block_additional_residuals: Optional[tuple[torch.Tensor]] = None,
            mid_block_additional_residual: Optional[torch.Tensor] = None,
            down_intrablock_additional_residuals: Optional[tuple[torch.Tensor]] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            return_dict: bool = True,
            return_intermediates: bool = False,
            iter_cur=0, phase="sample",
    ) -> Union[UNet2DConditionOutput, tuple]:
        default_overall_up_factor = 2 ** self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                forward_upsample_size = True
                break

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        if self.config.addition_embed_type == "image_hint":
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )

        sample = self.conv_in(sample)

        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        is_adapter = down_intrablock_additional_residuals is not None
        
        if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                additional_residuals = {}
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    iter_cur=iter_cur, phase=phase,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    sample += down_intrablock_additional_residuals.pop(0)

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()
            for down_block_res_sample, down_block_additional_residual in zip(
                    down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)
            down_block_res_samples = new_down_block_res_samples

        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    iter_cur=iter_cur, phase=phase,
                )
            else:
                sample = self.mid_block(sample, emb)

            if (
                    is_adapter
                    and len(down_intrablock_additional_residuals) > 0
                    and sample.shape == down_intrablock_additional_residuals[0].shape
            ):
                sample += down_intrablock_additional_residuals.pop(0)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        all_intermediate_features = [sample]

        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    iter_cur=iter_cur, phase=phase
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )
            all_intermediate_features.append(sample)
        
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if return_intermediates:
            return sample, all_intermediate_features
        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)

    return forward

class EditPipeline(DiffusionPipeline, TextualInversionLoaderMixin, StableDiffusionLoraLoaderMixin):
    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: LCMScheduler,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()
        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning("Safety checker disabled.")

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def modify_unet_forward(self):
        self.unet.forward = override_forward(self.unet)

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale
            adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            
            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt

            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(self, prompt, strength):
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

    def prepare_extra_step_kwargs(self, generator):
        extra_step_kwargs = {}
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    def get_timesteps(self, num_inference_steps, strength, device):
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        return timesteps, num_inference_steps - t_start

    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, denoise_model, generator=None):
        image = image.to(device=device, dtype=dtype)
        batch_size = image.shape[0]

        if image.shape[1] == 4:
            init_latents = image
        else:
            if isinstance(generator, list):
                init_latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = self.vae.encode(image).latent_dist.sample(generator)
            init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
             init_latents = torch.cat([init_latents] * num_images_per_prompt, dim=0)
        else:
            init_latents = torch.cat([init_latents] * num_images_per_prompt, dim=0)

        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        clean_latents = init_latents

        if denoise_model:
            init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
            latents = init_latents
        else:
            latents = noise

        return latents, clean_latents

    def __call__(
        self,
        prompt: Union[str, List[str]],
        source_prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]]=None,
        positive_prompt: Union[str, List[str]]=None,
        image: PipelineImageInput = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        original_inference_steps: Optional[int]  = 50,
        guidance_scale: Optional[float] = 7.5,
        source_guidance_scale: Optional[float] = 1,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        denoise_model: Optional[bool] = True,
        mask: PipelineImageInput = None,
        selected_points: Optional[list] = None,
        # --- 彻底删除: lam, latent_lr, n_pix_step, lora_path, use_lora, use_fastdrag ---
        visualize_drag: Optional[bool] = False,
        drag_type: Optional[str] = "Hybrid-Rigid",
        influence_range: Optional[float] = 0.0,
        pointcloud_domain: Optional[str] = "auto",
        mask_backend_mode: Optional[str] = "sam_refined",
        anchor_strategy_3d: Optional[str] = "auto",
        ddcm_eta: Optional[float] = 1.0,
        controller: Optional[Any] = None,
        return_layout_image: Optional[bool] = False,
        edit_mode: Optional[str] = "joint",
        hole_fill_mode: Optional[str] = DEFAULT_HOLE_FILL_MODE,
        use_expanded_subject_fill: Optional[bool] = True,
        expanded_subject_fill_px: Optional[int] = 6,
        use_drag_guided_prefill: Optional[bool] = False,
        enable_3d_subject_scope_fill: Optional[bool] = False,
        drag_layout_latents: Optional[bool] = True,
        drag_target_latents: Optional[bool] = None,
        drag_clean_latents: Optional[bool] = True,
        drag_target_q_layout_mix: Optional[bool] = True,
        enable_ref_target_denoise_mix: Optional[bool] = False,
        enable_ref_kv_injection: Optional[bool] = True,
        ref_target_denoise_mix_max: Optional[float] = 0.35,
        ref_target_denoise_mix_start: Optional[float] = 0.60,
        joint_target_refine_mix: Optional[bool] = False,
        joint_target_refine_mix_start: Optional[float] = 0.30,
        joint_target_refine_mix_max_out: Optional[float] = 0.45,
        joint_target_refine_mix_max_in: Optional[float] = 0.10,
        timing_collector: Optional[Dict[str, float]] = None,
    ):
        with torch.no_grad():
            def _timing_now():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                return time.perf_counter()

            timing_start = _timing_now() if isinstance(timing_collector, dict) else None
            source_image_pil = image
            hole_fill_mode = normalize_hole_fill_mode(hole_fill_mode, default=DEFAULT_HOLE_FILL_MODE)
            use_expanded_subject_fill = bool(use_expanded_subject_fill)
            try:
                expanded_subject_fill_px = int(round(float(expanded_subject_fill_px)))
            except Exception:
                expanded_subject_fill_px = 6
            expanded_subject_fill_px = max(0, min(48, expanded_subject_fill_px))
            use_drag_guided_prefill = bool(use_drag_guided_prefill)
            enable_3d_subject_scope_fill = bool(enable_3d_subject_scope_fill)
            drag_layout_latents = bool(drag_layout_latents)
            if drag_target_latents is None:
                drag_target_latents = drag_layout_latents
            else:
                drag_target_latents = bool(drag_target_latents)
            drag_clean_latents = bool(drag_clean_latents)
            drag_target_q_layout_mix = bool(drag_target_q_layout_mix)
            enable_ref_target_denoise_mix = bool(enable_ref_target_denoise_mix)
            enable_ref_kv_injection = bool(enable_ref_kv_injection)
            try:
                ref_target_denoise_mix_max = float(ref_target_denoise_mix_max)
            except Exception:
                ref_target_denoise_mix_max = 0.35
            ref_target_denoise_mix_max = max(0.0, min(1.0, ref_target_denoise_mix_max))
            try:
                ref_target_denoise_mix_start = float(ref_target_denoise_mix_start)
            except Exception:
                ref_target_denoise_mix_start = 0.60
            ref_target_denoise_mix_start = max(0.0, min(0.95, ref_target_denoise_mix_start))
            joint_target_refine_mix = bool(joint_target_refine_mix)
            try:
                joint_target_refine_mix_start = float(joint_target_refine_mix_start)
            except Exception:
                joint_target_refine_mix_start = 0.30
            joint_target_refine_mix_start = max(0.0, min(0.95, joint_target_refine_mix_start))
            try:
                joint_target_refine_mix_max_out = float(joint_target_refine_mix_max_out)
            except Exception:
                joint_target_refine_mix_max_out = 0.45
            joint_target_refine_mix_max_out = max(0.0, min(1.0, joint_target_refine_mix_max_out))
            try:
                joint_target_refine_mix_max_in = float(joint_target_refine_mix_max_in)
            except Exception:
                joint_target_refine_mix_max_in = 0.10
            joint_target_refine_mix_max_in = max(0.0, min(1.0, joint_target_refine_mix_max_in))
            edit_mode = str(edit_mode or "joint").strip().lower()
            drag_pair_count = _count_drag_pairs(selected_points)
            has_drag_points = drag_pair_count > 0
            has_text_delta = _has_prompt_delta(source_prompt, prompt)
            effective_text_only = has_text_delta and (not has_drag_points)
            effective_joint_edit = has_text_delta and has_drag_points
            mask_backend_mode = str(mask_backend_mode or "sam_refined").strip().lower()
            if mask_backend_mode in {"user_mask", "user", "raw_user_mask", "raw"}:
                mask_backend_mode = "user_mask"
            else:
                mask_backend_mode = "sam_refined"
            if controller is not None:
                setattr(controller, "enable_ref_kv_injection", bool(enable_ref_kv_injection))
                setattr(controller, "drag_target_q_layout_mix", bool(drag_target_q_layout_mix))
            
            # Mask 预处理
            if mask is None:
                mask_np = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
            else:
                mask_np = mask.astype(np.uint8)

            # 1. Check inputs
            self.check_inputs(prompt, strength)

            # 2. Define call parameters
            batch_size = 1 if isinstance(prompt, str) else len(prompt)
            device = self._execution_device
            do_classifier_free_guidance = guidance_scale > 1.0
            # 3. Encode prompts (移除 lora_scale)
            prompt_embeds_tuple = self.encode_prompt(
                prompt, device, num_images_per_prompt, do_classifier_free_guidance,
                negative_prompt=negative_prompt, prompt_embeds=prompt_embeds, lora_scale=None,
            )
            source_prompt_embeds_tuple = self.encode_prompt(
                source_prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                prompt_embeds=None,
                lora_scale=None,
            )
            
            if prompt_embeds_tuple[1] is not None:
                prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])
            else:
                prompt_embeds = prompt_embeds_tuple[0]
            
            if source_prompt_embeds_tuple[1] is not None:
                source_prompt_embeds = torch.cat([source_prompt_embeds_tuple[1], source_prompt_embeds_tuple[0]])
            else:
                source_prompt_embeds = source_prompt_embeds_tuple[0]

            # 4. Preprocess image
            image = self.image_processor.preprocess(image)

            # 5. Set Timesteps
            self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device, original_inference_steps=original_inference_steps)
            timesteps, num_inference_steps= self.get_timesteps(num_inference_steps, strength, device)
            latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

            # 6. Prepare Latents
            latents, clean_latents = self.prepare_latents(
                image, latent_timestep, batch_size, num_images_per_prompt, prompt_embeds.dtype, device, denoise_model, generator
            )

            # 保存 source 路径（未拖拽）用于后续双锚融合
            source_latents = latents.clone()
            source_clean_latents = clean_latents.clone()
            try:
                drag_t_idx = int(latent_timestep[0].detach().cpu().item())
                alpha_prod_t_drag = float(self.scheduler.alphas_cumprod[drag_t_idx].detach().cpu().item())
            except Exception:
                alpha_prod_t_drag = None
            
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator)
            generator = extra_step_kwargs.pop("generator", None)
            drag_preserve_mask_lat = None
            if timing_start is not None:
                timing_after_prepare = _timing_now()
                timing_collector["preparation_sec"] = float(timing_after_prepare - timing_start)
            # --- [删除了]: 原先这里有一大段为了 DragDiffusion 准备 concat_target_input 的代码 ---
            # --- [删除了]: scale_model_input, torch.stack/cat 等操作 ---

            # 统一五分支采样路径：Text/Drag/Joint 仅在输入预处理上区分。
            # 历史的 text 三分支路径已移除，固定走五分支实现。
            if edit_mode in {"text", "drag", "joint"}:
                # 7. Apply Drag
                # 7.1 对带噪 latent 应用拖拽，并返回可复用几何计划。
                # layout/target 两分支共享同一份 dragged_latents，但可分别选择是否采用。
                drag_plan = None
                if drag_layout_latents or drag_target_latents:
                    dragged_latents, drag_plan = apply_drag(
                        source_image=source_image_pil,
                        latents=latents,
                        mask=mask_np,
                        points=selected_points,
                        device=device,
                        drag_type=drag_type,
                        influence_range=influence_range,
                        visualize_drag=visualize_drag,
                        pointcloud_domain=pointcloud_domain,
                        mask_backend_mode=mask_backend_mode,
                        anchor_strategy_3d=anchor_strategy_3d,
                        return_plan=True,
                        hole_fill_mode=hole_fill_mode,
                        use_expanded_subject_fill=use_expanded_subject_fill,
                        expanded_subject_fill_px=expanded_subject_fill_px,
                        use_drag_guided_prefill=use_drag_guided_prefill,
                        enable_3d_subject_scope_fill=enable_3d_subject_scope_fill,
                        vae=self.vae,
                        clean_latents=clean_latents,
                        alpha_prod_t=alpha_prod_t_drag,
                    )
                else:
                    dragged_latents = latents.clone()

                # 7.2 对 clean latents 应用拖拽，作为 DDCM 的正确锚点。
                # 若 layout latent 未拖拽但 clean latent 需要拖拽，则在这里补建 drag_plan。
                if drag_clean_latents:
                    if drag_plan is not None:
                        dragged_clean_latents = apply_drag(
                            source_image=source_image_pil,
                            latents=clean_latents,
                            mask=mask_np,
                            points=selected_points,
                            device=device,
                            drag_type=drag_type,
                            influence_range=influence_range,
                            visualize_drag=False,
                            pointcloud_domain=pointcloud_domain,
                            mask_backend_mode=mask_backend_mode,
                            anchor_strategy_3d=anchor_strategy_3d,
                            drag_plan=drag_plan,
                            hole_fill_mode=hole_fill_mode,
                            use_expanded_subject_fill=use_expanded_subject_fill,
                            expanded_subject_fill_px=expanded_subject_fill_px,
                            use_drag_guided_prefill=use_drag_guided_prefill,
                            enable_3d_subject_scope_fill=enable_3d_subject_scope_fill,
                            vae=self.vae,
                            clean_latents=clean_latents,
                            alpha_prod_t=1.0,
                        )
                    else:
                        # noisy 分支未拖拽时，首次真实的 drag 发生在 clean latents 上；
                        # 这里接管 visualize_drag，避免前端勾选后却没有任何 drag_process 输出。
                        dragged_clean_latents, drag_plan = apply_drag(
                            source_image=source_image_pil,
                            latents=clean_latents,
                            mask=mask_np,
                            points=selected_points,
                            device=device,
                            drag_type=drag_type,
                            influence_range=influence_range,
                            visualize_drag=visualize_drag,
                            pointcloud_domain=pointcloud_domain,
                            mask_backend_mode=mask_backend_mode,
                            anchor_strategy_3d=anchor_strategy_3d,
                            return_plan=True,
                            hole_fill_mode=hole_fill_mode,
                            use_expanded_subject_fill=use_expanded_subject_fill,
                            expanded_subject_fill_px=expanded_subject_fill_px,
                            use_drag_guided_prefill=use_drag_guided_prefill,
                            enable_3d_subject_scope_fill=enable_3d_subject_scope_fill,
                            vae=self.vae,
                            clean_latents=clean_latents,
                            alpha_prod_t=1.0,
                        )
                else:
                    dragged_clean_latents = clean_latents.clone()

                if effective_joint_edit and joint_target_refine_mix:
                    mask_candidate_full = None
                    if isinstance(drag_plan, dict):
                        m_start = drag_plan.get("m_start_full_overall")
                        m_pseudo = drag_plan.get("m_pseudo_full_overall")
                        if isinstance(m_start, np.ndarray):
                            mask_candidate_full = np.asarray(m_start, dtype=np.float32)
                        if isinstance(m_pseudo, np.ndarray):
                            if mask_candidate_full is None:
                                mask_candidate_full = np.asarray(m_pseudo, dtype=np.float32)
                            else:
                                mask_candidate_full = np.maximum(
                                    mask_candidate_full,
                                    np.asarray(m_pseudo, dtype=np.float32),
                                )
                    if mask_candidate_full is None:
                        mask_candidate_full = np.asarray(mask_np, dtype=np.float32)

                    if (
                        isinstance(mask_candidate_full, np.ndarray)
                        and mask_candidate_full.ndim == 2
                        and float(np.sum(mask_candidate_full > 0.5)) > 0.0
                    ):
                        mask_t = torch.from_numpy((mask_candidate_full > 0.5).astype(np.float32)).to(
                            device=device,
                            dtype=torch.float32,
                        )
                        mask_t = mask_t.unsqueeze(0).unsqueeze(0)
                        mask_t = F.interpolate(
                            mask_t,
                            size=(dragged_latents.shape[-2], dragged_latents.shape[-1]),
                            mode="nearest",
                        )
                        mask_t = F.max_pool2d(mask_t, kernel_size=3, stride=1, padding=1)
                        drag_preserve_mask_lat = torch.clamp(mask_t, 0.0, 1.0)

                # 7.1 Mask 可视化与保存逻辑 (Debug用)
                # 拖拽可视化由 drag_process/mask_process 统一输出

                torch.cuda.empty_cache()

                if timing_start is not None:
                    timing_after_geometry = _timing_now()
                    timing_collector["geometry_anchor_sec"] = float(timing_after_geometry - timing_after_prepare)

                clean_source_latents = source_clean_latents.half()
                clean_layout_latents = dragged_clean_latents.half()
                num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

                # 固定统一五分支：
                # source    : source prompt + source(noisy) latents
                # mutual    : source prompt + source(noisy) latents
                # reference : target prompt + source(noisy) latents
                # layout    : source prompt + (dragged or source)(noisy) latents
                # target    : target prompt + (dragged or source)(noisy) latents
                # clean_* latents 只作为 DDCM 的 x0 锚点
                source_branch_latents = source_latents.half()
                mutual_branch_latents = source_latents.half()
                reference_branch_latents = source_latents.half()
                layout_branch_latents = (dragged_latents if drag_layout_latents else source_latents).half()
                target_branch_latents = (dragged_latents if drag_target_latents else source_latents).half()

                if not has_drag_points:
                    print("[FiveBranch] no drag pairs -> final decode uses reference branch (ref).")
                with self.progress_bar(total=num_inference_steps) as progress_bar:
                    pred_x0_out = None
                    for i, t in enumerate(timesteps):
                        progress = i / max(len(timesteps) - 1, 1)

                        source_latent_input = torch.cat([source_branch_latents] * 2) if do_classifier_free_guidance else source_branch_latents
                        mutual_latent_input = torch.cat([mutual_branch_latents] * 2) if do_classifier_free_guidance else mutual_branch_latents
                        reference_latent_input = torch.cat([reference_branch_latents] * 2) if do_classifier_free_guidance else reference_branch_latents
                        layout_latent_input = torch.cat([layout_branch_latents] * 2) if do_classifier_free_guidance else layout_branch_latents
                        target_latent_input = torch.cat([target_branch_latents] * 2) if do_classifier_free_guidance else target_branch_latents

                        source_latent_input = self.scheduler.scale_model_input(source_latent_input, t)
                        mutual_latent_input = self.scheduler.scale_model_input(mutual_latent_input, t)
                        reference_latent_input = self.scheduler.scale_model_input(reference_latent_input, t)
                        layout_latent_input = self.scheduler.scale_model_input(layout_latent_input, t)
                        target_latent_input = self.scheduler.scale_model_input(target_latent_input, t)

                        if do_classifier_free_guidance:
                            concat_latent_model_input = torch.stack(
                                [
                                    source_latent_input[0], mutual_latent_input[0], reference_latent_input[0], layout_latent_input[0], target_latent_input[0],
                                    source_latent_input[1], mutual_latent_input[1], reference_latent_input[1], layout_latent_input[1], target_latent_input[1],
                                ],
                                dim=0,
                            )
                            concat_prompt_embeds = torch.stack(
                                [
                                    source_prompt_embeds[0], source_prompt_embeds[0], prompt_embeds[0], source_prompt_embeds[0], prompt_embeds[0],
                                    source_prompt_embeds[1], source_prompt_embeds[1], prompt_embeds[1], source_prompt_embeds[1], prompt_embeds[1],
                                ],
                                dim=0,
                            )
                        else:
                            concat_latent_model_input = torch.cat(
                                [source_latent_input, mutual_latent_input, reference_latent_input, layout_latent_input, target_latent_input],
                                dim=0,
                            )
                            concat_prompt_embeds = torch.cat(
                                [source_prompt_embeds, source_prompt_embeds, prompt_embeds, source_prompt_embeds, prompt_embeds],
                                dim=0,
                            )

                        concat_noise_pred = self.unet(
                            sample=concat_latent_model_input, timestep=t,
                            cross_attention_kwargs=cross_attention_kwargs,
                            encoder_hidden_states=concat_prompt_embeds,
                            iter_cur=i,
                        ).sample

                        if do_classifier_free_guidance:
                            (
                                source_noise_pred_uncond, mutual_noise_pred_uncond, reference_noise_pred_uncond, layout_noise_pred_uncond, target_noise_pred_uncond,
                                source_noise_pred_text, mutual_noise_pred_text, reference_noise_pred_text, layout_noise_pred_text, target_noise_pred_text,
                            ) = concat_noise_pred.chunk(10, dim=0)

                            source_noise_pred = source_noise_pred_uncond + source_guidance_scale * (source_noise_pred_text - source_noise_pred_uncond)
                            mutual_noise_pred = mutual_noise_pred_uncond + source_guidance_scale * (mutual_noise_pred_text - mutual_noise_pred_uncond)
                            reference_noise_pred = reference_noise_pred_uncond + guidance_scale * (reference_noise_pred_text - reference_noise_pred_uncond)
                            layout_noise_pred = layout_noise_pred_uncond + source_guidance_scale * (layout_noise_pred_text - layout_noise_pred_uncond)
                            target_noise_pred = target_noise_pred_uncond + guidance_scale * (target_noise_pred_text - target_noise_pred_uncond)
                        else:
                            (source_noise_pred, mutual_noise_pred, reference_noise_pred, layout_noise_pred, target_noise_pred) = concat_noise_pred.chunk(5, dim=0)

                        noise = torch.randn(
                            target_branch_latents.shape,
                            dtype=target_branch_latents.dtype,
                            device=target_branch_latents.device,
                            generator=generator,
                        )
                        alpha_prod_t = self.scheduler.alphas_cumprod[t]
                        sqrt_alpha = alpha_prod_t ** 0.5
                        sqrt_beta = (1 - alpha_prod_t) ** 0.5
                        pred_xs = None
                        pred_xs_source = (source_branch_latents - sqrt_beta * source_noise_pred) / sqrt_alpha
                        pred_xs_layout = (layout_branch_latents - sqrt_beta * layout_noise_pred) / sqrt_alpha
                        pred_xs_mutual = (mutual_branch_latents - sqrt_beta * mutual_noise_pred) / sqrt_alpha

                        source_ddcm_latents = source_branch_latents
                        layout_source_latents = layout_branch_latents
                        if enable_ref_target_denoise_mix:
                            denom = max(1e-6, 1.0 - ref_target_denoise_mix_start)
                            detail_ramp = float(min(1.0, max(0.0, (progress - ref_target_denoise_mix_start) / denom)))
                            detail_mix = detail_ramp * ref_target_denoise_mix_max
                        else:
                            detail_mix = 0.0
                        layout_to_target_noise = (1.0 - detail_mix) * layout_noise_pred + detail_mix * reference_noise_pred

                        _, reference_branch_latents, pred_reference = ddcm_sampler(
                            self.scheduler, source_ddcm_latents, reference_branch_latents, t,
                            source_noise_pred, reference_noise_pred, clean_source_latents,
                            noise=noise, to_next=False, eta=ddcm_eta,
                            cur_step=i, total_steps=num_inference_steps, **extra_step_kwargs
                        )
                        source_branch_latents, mutual_branch_latents, pred_mutual = ddcm_sampler(
                            self.scheduler, source_ddcm_latents, mutual_branch_latents, t,
                            source_noise_pred, mutual_noise_pred, clean_source_latents,
                            noise=noise, to_next=False, eta=ddcm_eta,
                            cur_step=i, total_steps=num_inference_steps, **extra_step_kwargs
                        )
                        _, target_branch_latents, pred_x0 = ddcm_sampler(
                            self.scheduler, layout_source_latents, target_branch_latents, t,
                            layout_to_target_noise, target_noise_pred, clean_layout_latents,
                            noise=noise, to_next=False, eta=ddcm_eta,
                            cur_step=i, total_steps=num_inference_steps, **extra_step_kwargs
                        )
                        if effective_joint_edit and joint_target_refine_mix:
                            denom_joint = max(1e-6, 1.0 - joint_target_refine_mix_start)
                            joint_ramp = float(
                                min(1.0, max(0.0, (progress - joint_target_refine_mix_start) / denom_joint))
                            )
                            mix_out = joint_ramp * joint_target_refine_mix_max_out
                            mix_in = joint_ramp * joint_target_refine_mix_max_in
                            if (mix_out > 0.0) or (mix_in > 0.0):
                                if drag_preserve_mask_lat is None:
                                    blend_w = 0.0
                                else:
                                    mask_drag = drag_preserve_mask_lat.to(
                                        device=target_branch_latents.device,
                                        dtype=target_branch_latents.dtype,
                                    )
                                    blend_w = mix_in * mask_drag + mix_out * (1.0 - mask_drag)

                                if not (isinstance(blend_w, float) and blend_w == 0.0):
                                    target_branch_latents = target_branch_latents + blend_w * (
                                        reference_branch_latents - target_branch_latents
                                    )
                                    pred_x0 = pred_x0 + blend_w * (pred_reference - pred_x0)
                        _, layout_branch_latents, _ = ddcm_sampler(
                            self.scheduler, layout_source_latents, layout_source_latents, t,
                            layout_noise_pred, layout_noise_pred, clean_layout_latents,
                            noise=noise, to_next=True, eta=ddcm_eta,
                            cur_step=i, total_steps=num_inference_steps, **extra_step_kwargs
                        )
                        # 无拖拽点时，最终输出切到 reference 分支；五分支本身仍按原逻辑完整计算。
                        pred_x0_out = pred_reference if (not has_drag_points) else pred_x0

                        if controller is not None and getattr(controller, "local_blend", None) is not None and i > 0:
                            alpha_prod_t_cb = self.scheduler.alphas_cumprod[t]
                            mutual_branch_latents, reference_branch_latents = controller.step_callback(
                                i, t, source_branch_latents, reference_branch_latents, mutual_branch_latents, alpha_prod_t_cb
                            )

                        if callback is not None:
                            try:
                                callback(i, t, {
                                    "pred_xs": pred_xs,
                                    "pred_xs_source": pred_xs_source,
                                    "pred_xs_layout": pred_xs_layout,
                                    "pred_xs_mutual": pred_xs_mutual,
                                    "pred_x0": pred_x0,
                                    "pred_reference": pred_reference,
                                    "pred_mutual": pred_mutual,
                                    "latents": target_branch_latents,
                                    "layout_latents": layout_branch_latents,
                                    "reference_latents": reference_branch_latents,
                                    "mutual_latents": mutual_branch_latents,
                                    "controller": controller,
                                    "source_image": source_image_pil,
                                })
                            except Exception as e:
                                print(f"[Callback] step {i} failed: {e}")

                        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                            progress_bar.update()
                        if callback is not None and (i + 1) % 4 == 0:
                            torch.cuda.empty_cache()

            if timing_start is not None:
                timing_after_denoise = _timing_now()
                timing_collector["denoising_sec"] = float(timing_after_denoise - timing_after_geometry)

            # Decode 图像
            # `pred_x0` is the final target-branch x0 prediction.
            # `pred_xs_layout` is the layout-branch x0 prediction for debugging (geometry-only view).
            layout_latent_for_decode = pred_xs_layout if pred_xs_layout is not None else pred_reference
            decode_latent = pred_x0_out if pred_x0_out is not None else pred_x0
            if not output_type == "latent":
                image = self.vae.decode(decode_latent / self.vae.config.scaling_factor, return_dict=False)[0]
                image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
                if return_layout_image:
                    image_layout = self.vae.decode(layout_latent_for_decode / self.vae.config.scaling_factor, return_dict=False)[0]
                    image_layout, has_nsfw_concept_layout = self.run_safety_checker(image_layout, device, prompt_embeds.dtype)
                else:
                    image_layout = None
                    has_nsfw_concept_layout = None
            else:
                image = decode_latent
                has_nsfw_concept = None
                image_layout = layout_latent_for_decode if return_layout_image else None
                has_nsfw_concept_layout = None

            if has_nsfw_concept is None:
                do_denormalize = [True] * image.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

            image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
            if image_layout is not None:
                image_layout = self.image_processor.postprocess(image_layout, output_type=output_type, do_denormalize=do_denormalize)

            if timing_start is not None:
                timing_after_decode = _timing_now()
                timing_collector["decode_postprocess_sec"] = float(timing_after_decode - timing_after_denoise)
                timing_collector["pipeline_total_sec"] = float(timing_after_decode - timing_start)

            if not return_dict:
                return (image, has_nsfw_concept)

        layout_output = (
            StableDiffusionPipelineOutput(images=image_layout, nsfw_content_detected=has_nsfw_concept_layout)
            if image_layout is not None
            else None
        )
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept), layout_output

# --------------------------
# 辅助函数 (DDCM Sampler, LocalBlend, AttentionControl 等)
# --------------------------

def ddcm_sampler(scheduler, x_s, x_t, timestep, e_s, e_t, x_0, noise, to_next=True, decay_factor=0.9, cur_step=0, total_steps=10, eta=1.0):
    """
    按 InfEdit 原版 DDCM 公式，支持通过 eta 控制每步随机性。
    decay_factor/cur_step/total_steps 保留为兼容参数，不参与计算。
    """
    if scheduler.num_inference_steps is None:
        raise ValueError("Number of inference steps is 'None'")
    if scheduler.step_index is None:
        scheduler._init_step_index(timestep)

    prev_step_index = scheduler.step_index + 1
    if prev_step_index < len(scheduler.timesteps):
        prev_timestep = scheduler.timesteps[prev_step_index]
    else:
        prev_timestep = timestep

    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    eta = float(max(0.0, min(1.0, eta)))
    variance = beta_prod_t_prev
    std_dev_t = eta * variance
    noise = std_dev_t ** 0.5 * noise

    # DDCM核心计算（x_0已经是拖拽后的版本）
    e_c = (x_s - alpha_prod_t ** 0.5 * x_0) / (1 - alpha_prod_t) ** 0.5
    pred_x0 = x_0 + ((x_t - x_s) - beta_prod_t ** 0.5 * (e_t - e_s)) / alpha_prod_t ** 0.5

    eps = (e_t - e_s) + e_c
    dir_xt = (beta_prod_t_prev - std_dev_t) ** 0.5 * eps

    if len(scheduler.timesteps) > 1:
        prev_xt = alpha_prod_t_prev ** 0.5 * pred_x0 + dir_xt + noise
        prev_xs = alpha_prod_t_prev ** 0.5 * x_0 + dir_xt + noise

    else:
        prev_xt = pred_x0
        prev_xs = x_0

    if to_next:
        scheduler._step_index += 1

    return prev_xs, prev_xt, pred_x0

# (LocalBlend, AttentionControl, AttentionStore, AttentionControlEdit, AttentionRefine 定义)
# 由于代码量巨大，这里省略详细实现，假设用户直接复制原文件中这些类的代码到这里。
# 关键点：需要包含 LocalBlend, AttentionControl, AttentionStore, AttentionControlEdit, AttentionRefine

class LocalBlend:
    def get_mask(self, x_t, maps, word_idx, thresh, i):
        maps = maps * word_idx.reshape(1, 1, 1, 1, -1)
        maps = (maps[:, :, :, :, 1:self.len - 1]).mean(0, keepdim=True)
        maps = (maps).max(-1)[0]
        maps = nnf.interpolate(maps, size=(x_t.shape[2:]))
        maps = maps / maps.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
        mask = maps > thresh
        return mask

    def save_image(self, mask, i, caption):
        image = mask[0, 0, :, :]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.cpu().numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        if not os.path.exists(f"inter/{caption}"):
           os.mkdir(f"inter/{caption}") 
        ptp_utils.save_images(image, f"inter/{caption}/{i}.jpg")
    
    def generate_typical_sizes(self, latent_size):
        sizes = []
        current_h, current_w = latent_size
        for _ in range(4):
            rounded_h = int(math.ceil(current_h))
            rounded_w = int(math.ceil(current_w))
            sizes.append((rounded_h, rounded_w))
            if _ < 3:
                current_h = current_h / 2
                current_w = current_w / 2
        return sizes

    def get_feature_map_size(self, sequence_length, latent_size):
        typical_sizes = self.generate_typical_sizes(latent_size)
        for feat_h, feat_w in typical_sizes:
            if feat_h * feat_w == sequence_length:
                return feat_h, feat_w
        for feat_h in range(1, int(sequence_length ** 0.5) + 1):
            if sequence_length % feat_h == 0:
                feat_w = sequence_length // feat_h
                if abs(feat_h / feat_w - latent_size[0] / latent_size[1]) < 0.5:
                    return feat_h, feat_w
        sqrt_len = int(sequence_length ** 0.5)
        return sqrt_len, sqrt_len
    
    def __call__(self, i, x_s, x_t, x_m, attention_store, alpha_prod, temperature=0.15, use_xm=False, drag_mask=None):
        """
        【改进点6】拖拽感知的LocalBlend
        - 新增drag_mask参数：拖拽区域的mask
        - 拖拽区域内优先保持x_t（拖拽+编辑后的结果）
        """
        # 对齐 InfEdit：使用 down_cross[2:4] + up_cross[:3]
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        h, w = x_t.shape[2], x_t.shape[3]
        h, w = ((h + 1) // 2 + 1) // 2, ((w + 1) // 2 + 1) // 2
        maps = [item.reshape(2, -1, 1, h // int((h * w / item.shape[-2])**0.5),  w // int((h * w / item.shape[-2])**0.5), MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps_s = maps[0,:]
        maps_m = maps[1,:]
        thresh_e = temperature / alpha_prod ** (0.5)
        if thresh_e < self.thresh_e:
          thresh_e = self.thresh_e
        thresh_m = self.thresh_m
        mask_e = self.get_mask(x_t, maps_m, self.alpha_e, thresh_e, i)
        mask_m = self.get_mask(x_t, maps_s, (self.alpha_m - self.alpha_me), thresh_m, i)
        mask_me = self.get_mask(x_t, maps_m, self.alpha_me, self.thresh_e, i)
        if self.save_inter:
            self.save_image(mask_e,i,"mask_e")
            self.save_image(mask_m,i,"mask_m")
            self.save_image(mask_me,i,"mask_me")
        if self.alpha_e.sum() == 0:
          x_t_out = x_t
        else:
          x_t_out = torch.where(mask_e, x_t, x_m)
        x_t_out = torch.where(mask_m, x_s, x_t_out)
        if use_xm:
          x_t_out = torch.where(mask_me, x_m, x_t_out)

        # 【改进点6续】与拖拽mask的融合
        if drag_mask is not None:
            # 将drag_mask调整到latent分辨率
            if drag_mask.shape[2:] != x_t.shape[2:]:
                drag_mask_resized = nnf.interpolate(
                    drag_mask.unsqueeze(0).unsqueeze(0) if drag_mask.ndim == 2 else drag_mask,
                    size=x_t.shape[2:],
                    mode='nearest'
                )
            else:
                drag_mask_resized = drag_mask

            # 拖拽区域内优先保持x_t（拖拽+编辑后的结果）
            # 这确保了拖拽效果不会被LocalBlend的mask_m覆盖
            x_t_out = torch.where(drag_mask_resized > 0.5, x_t, x_t_out)

        return x_m, x_t_out

    def __init__(self,thresh_e = 0.3, thresh_m = 0.3, save_inter = False):
        self.thresh_e = thresh_e
        self.thresh_m = thresh_m
        self.save_inter = save_inter
    def set_map(self, ms, alpha, alpha_e, alpha_m, len):
        self.m = ms
        self.alpha = alpha
        self.alpha_e = alpha_e
        self.alpha_m = alpha_m
        alpha_me = alpha_e.to(torch.bool) & alpha_m.to(torch.bool)
        self.alpha_me = alpha_me.to(torch.float)
        self.len = len

class AttentionControl(abc.ABC):
    def step_callback(self, x_t): return x_t
    def between_steps(self): return
    @property
    def num_uncond_att_layers(self): return self.num_att_layers if LOW_RESOURCE else 0
    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str): raise NotImplementedError
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                # 兼容 CFG / 非 CFG：
                # - CFG: 仅修改 conditional half
                # - 非 CFG: 修改整批
                branch_count = getattr(self, "batch_size", None)
                if isinstance(branch_count, int) and branch_count > 0 and attn.shape[0] % branch_count == 0:
                    if attn.shape[0] % (2 * branch_count) == 0:
                        cond_start = attn.shape[0] // 2
                        attn[cond_start:] = self.forward(attn[cond_start:], is_cross, place_in_unet)
                    else:
                        attn = self.forward(attn, is_cross, place_in_unet)
                else:
                    h = attn.shape[0]
                    attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers // 2 + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
    def self_attn_forward(self, q, k, v, num_heads): return q, k, v # Default

class EmptyControl(AttentionControl):
    def forward(self, attn, is_cross: bool, place_in_unet: str): return attn
    def self_attn_forward(self, q, k, v, num_heads):
        return q, k, v

class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # 对齐 InfEdit：存储所有低分辨率 attention（不做 layer_counters 门控）
        if attn.shape[1] <= 32 ** 2:
            self.step_store[key].append(attn)
        return attn
    @torch.no_grad()
    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.full_attention_store_target = self.full_step_store_target
        self.full_attention_store_reference = self.full_step_store_reference
        self.step_store = self.get_empty_store()
        self.full_step_store_target = self.get_empty_store()
        self.full_step_store_reference = self.get_empty_store()
    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention
    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.full_step_store_target = self.get_empty_store()
        self.full_step_store_reference = self.get_empty_store()
        self.attention_store = {}
        self.full_attention_store_target = {} 
        self.full_attention_store_reference = {}
    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.full_step_store_target = self.get_empty_store()
        self.full_step_store_reference = self.get_empty_store()
        self.attention_store = {}
        self.full_attention_store_target = {}
        self.full_attention_store_reference = {}

class AttentionControlEdit(AttentionStore, abc.ABC):
    def step_callback(self, i, t, x_s, x_t, x_m, alpha_prod):
        if (self.local_blend is not None) and (i > 0):
            use_xm = self._at_last_effective_step()
            x_m, x_t = self.local_blend(i, x_s, x_t, x_m, self.attention_store, alpha_prod, use_xm=use_xm)
        return x_m, x_t
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace): raise NotImplementedError

    def _schedule_progress(self) -> float:
        total_steps = max(1, int(getattr(self, "num_steps", 1)))
        return (int(self.cur_step) + int(getattr(self, "start_steps", 0)) + 1) * 1.0 / total_steps

    def _at_last_effective_step(self) -> bool:
        total_steps = int(getattr(self, "num_steps", 0))
        return (int(self.cur_step) + int(getattr(self, "start_steps", 0)) + 1) == total_steps

    def _resolve_schedule_value(self, value, fallback=0.7):
        if isinstance(value, (float, int)):
            return float(value)
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return float(value[-1])
        if isinstance(value, dict) and len(value) > 0:
            try:
                v = next(iter(value.values()))
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    return float(v[-1])
                if isinstance(v, (float, int)):
                    return float(v)
            except Exception:
                pass
        return float(fallback)

    def _hard_switch_alpha(self, before_branch_enabled: bool) -> float:
        return 0.0 if bool(before_branch_enabled) else 1.0

    def _four_branch_self_attn_blend(self, q, k, v, num_heads, text_alpha, drag_alpha):
        """
        四分支自注意力共享（source/reference/layout/target）：
        - text_alpha: source/reference 对的 self schedule
        - drag_alpha: layout/target 对的 self schedule
        """
        num_branches = q.shape[0] // num_heads
        if num_branches != 4:
            return q, k, v

        q0, q1, q2, q3 = q[:num_heads], q[num_heads:2 * num_heads], q[2 * num_heads:3 * num_heads], q[3 * num_heads:4 * num_heads]
        k0, k1, k2, k3 = k[:num_heads], k[num_heads:2 * num_heads], k[2 * num_heads:3 * num_heads], k[3 * num_heads:4 * num_heads]
        v0, v1, v2, v3 = v[:num_heads], v[num_heads:2 * num_heads], v[2 * num_heads:3 * num_heads], v[3 * num_heads:4 * num_heads]

        q_ref = (1.0 - text_alpha) * q0 + text_alpha * q1
        q_mix_layout = bool(getattr(self, "drag_target_q_layout_mix", True))
        q_tar = ((1.0 - drag_alpha) * q2 + drag_alpha * q3) if q_mix_layout else q3
        k_ref = (1.0 - text_alpha) * k0 + text_alpha * k1
        k_lay = (1.0 - drag_alpha) * k0 + drag_alpha * k2
        v_lay = (1.0 - drag_alpha) * v0 + drag_alpha * v2

        q_out = torch.cat([q0, q_ref, q2, q_tar], dim=0)
        k_out = torch.cat([k0, k_ref, k_lay, k1], dim=0)
        v_out = torch.cat([v0, v1, v_lay, v1], dim=0)
        return q_out, k_out, v_out

    def _five_branch_self_attn_blend(self, q, k, v, num_heads, text_alpha, drag_alpha, apply_drag_kv):
        """
        五分支（source/mutual/reference/layout/target）：
        - 前三分支使用 text self schedule（InfEdit 三分支逻辑）。
        - 后两分支使用 drag self schedule（layout/target）。
        """
        num_branches = q.shape[0] // num_heads
        if num_branches != 5:
            return q, k, v

        q_s, q_m, q_r, q_l, q_t = (
            q[:num_heads],
            q[num_heads:2 * num_heads],
            q[2 * num_heads:3 * num_heads],
            q[3 * num_heads:4 * num_heads],
            q[4 * num_heads:5 * num_heads],
        )
        k_s, k_m, k_r, k_l, k_t = (
            k[:num_heads],
            k[num_heads:2 * num_heads],
            k[2 * num_heads:3 * num_heads],
            k[3 * num_heads:4 * num_heads],
            k[4 * num_heads:5 * num_heads],
        )
        v_s, v_m, v_r, v_l, v_t = (
            v[:num_heads],
            v[num_heads:2 * num_heads],
            v[2 * num_heads:3 * num_heads],
            v[3 * num_heads:4 * num_heads],
            v[4 * num_heads:5 * num_heads],
        )

        tri_alpha = float(text_alpha)
        drag_alpha = float(min(1.0, max(0.0, drag_alpha)))
        q_mix_layout = bool(getattr(self, "drag_target_q_layout_mix", True))

        # before: q=[s,s,s], k=[s,s,s], v=[s,s,r]
        q_s_before, q_m_before, q_r_before = q_s, q_s, q_s
        k_s_before, k_m_before, k_r_before = k_s, k_s, k_s
        v_s_before, v_m_before, v_r_before = v_s, v_s, v_r
        # after: q=[s,r,r], k=[s,s,r], v=[s,s,r]
        q_s_after, q_m_after, q_r_after = q_s, q_r, q_r
        k_s_after, k_m_after, k_r_after = k_s, k_s, k_r
        v_s_after, v_m_after, v_r_after = v_s, v_s, v_r

        q_s_out = (1.0 - tri_alpha) * q_s_before + tri_alpha * q_s_after
        q_m_out = (1.0 - tri_alpha) * q_m_before + tri_alpha * q_m_after
        q_r_out = (1.0 - tri_alpha) * q_r_before + tri_alpha * q_r_after
        k_s_out = (1.0 - tri_alpha) * k_s_before + tri_alpha * k_s_after
        k_m_out = (1.0 - tri_alpha) * k_m_before + tri_alpha * k_m_after
        v_s_out = (1.0 - tri_alpha) * v_s_before + tri_alpha * v_s_after
        v_m_out = (1.0 - tri_alpha) * v_m_before + tri_alpha * v_m_after
        k_r_out = (1.0 - tri_alpha) * k_r_before + tri_alpha * k_r_after
        v_r_out = (1.0 - tri_alpha) * v_r_before + tri_alpha * v_r_after

        # Drag 分支（layout/target）：
        # 恢复窗口门控：仅在注入窗口内改 layout/target 自注意力，窗口外保持原路径。
        if apply_drag_kv:
            q_l_out = q_l
            detail_alpha_target = float(min(1.0, max(0.0, (drag_alpha - 0.25) / 0.75)))
            enable_ref_kv_injection = bool(getattr(self, "enable_ref_kv_injection", True))
            k_l_out, v_l_out = k_l, v_l

            q_t_out = ((1.0 - drag_alpha) * q_l + drag_alpha * q_t) if q_mix_layout else q_t
            if enable_ref_kv_injection:
                # 约束：仅“ref 替换”部分用 ref；非 ref 部分保持 target 自身 K/V。
                k_t_out = (1.0 - detail_alpha_target) * k_t + detail_alpha_target * k_r
                v_t_out = (1.0 - detail_alpha_target) * v_t + detail_alpha_target * v_r
            else:
                # No ref injection should restore the target branch's own K/V,
                # instead of additionally binding it to the layout branch.
                k_t_out = k_t
                v_t_out = v_t
        else:
            q_l_out, k_l_out, v_l_out = q_l, k_l, v_l
            q_t_out, k_t_out, v_t_out = q_t, k_t, v_t

        return (
            torch.cat([q_s_out, q_m_out, q_r_out, q_l_out, q_t_out], dim=0),
            torch.cat([k_s_out, k_m_out, k_r_out, k_l_out, k_t_out], dim=0),
            torch.cat([v_s_out, v_m_out, v_r_out, v_l_out, v_t_out], dim=0),
        )

    def _in_injection_window(self) -> bool:
        """
        start_step/start_layer 对应的注入窗口：
        - self-attn: 控制 ref->target KV 注入
        - cross-attn: 控制 layout->target 拖拽注入

        注意：当前计数器 `cur_att_layer` 只在 cross-attn 的 controller 调用中递增，
        因而它已经对应到每个 transformer block 的层索引（无需再做 //2）。
        """
        return (self.cur_step in self.step_idx) and (self.cur_att_layer in self.layer_idx)

    def self_attn_forward(self, q, k, v, num_heads):
        """
        四 schedule 版本：
        - text_self_replace_steps: source/mutual/reference 三分支
        - drag_self_replace_steps: layout/target 两分支
        """
        progress = self._schedule_progress()

        text_self_center = self._resolve_schedule_value(
            self.text_self_replace_steps, fallback=self._resolve_schedule_value(self.self_replace_steps, fallback=0.7)
        )
        use_after_text_self = text_self_center <= progress
        text_alpha = 1.0 if use_after_text_self else 0.0
        drag_self_center = self._resolve_schedule_value(
            self.drag_self_replace_steps,
            fallback=self._resolve_schedule_value(self.self_replace_steps, fallback=0.7),
        )
        use_after_drag_self = drag_self_center <= progress
        drag_alpha = 1.0 if use_after_drag_self else 0.0
        apply_drag_kv = self._in_injection_window()

        num_branches = q.shape[0] // num_heads
        if num_branches == 5:
            return self._five_branch_self_attn_blend(
                q, k, v, num_heads, text_alpha, drag_alpha, apply_drag_kv
            )

        if num_branches == 4:
            return self._four_branch_self_attn_blend(q, k, v, num_heads, text_alpha, drag_alpha)

        if num_branches == 3:
            # 3分支兼容路径（旧逻辑保留）
            # 非 CFG（3 分支）
            q_before = torch.cat([q[:num_heads], q[:num_heads], q[:num_heads]])
            k_before = torch.cat([k[:num_heads], k[:num_heads], k[:num_heads]])
            v_before = torch.cat([v[:num_heads * 2], v[:num_heads]])

            # 对齐 InfEdit 三分支：after 时第三分支 query 取第二分支
            q_after = torch.cat([q[:num_heads * 2], q[num_heads:2 * num_heads]])
            k_after = torch.cat([k[:num_heads * 2], k[:num_heads]])
            v_after = torch.cat([v[:num_heads * 2], v[:num_heads]])

            q = (1.0 - text_alpha) * q_before + text_alpha * q_after
            k = (1.0 - text_alpha) * k_before + text_alpha * k_after
            v = (1.0 - text_alpha) * v_before + text_alpha * v_after
            return q, k, v

        # CFG（uncond + cond）
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)

        num_branches_cfg = qu.shape[0] // num_heads
        if num_branches_cfg == 5:
            qu, ku, vu = self._five_branch_self_attn_blend(
                qu, ku, vu, num_heads, text_alpha, drag_alpha, apply_drag_kv
            )
            qc, kc, vc = self._five_branch_self_attn_blend(
                qc, kc, vc, num_heads, text_alpha, drag_alpha, apply_drag_kv
            )
            return torch.cat([qu, qc], dim=0), torch.cat([ku, kc], dim=0), torch.cat([vu, vc], dim=0)

        if num_branches_cfg == 4:
            qu, ku, vu = self._four_branch_self_attn_blend(qu, ku, vu, num_heads, text_alpha, drag_alpha)
            qc, kc, vc = self._four_branch_self_attn_blend(qc, kc, vc, num_heads, text_alpha, drag_alpha)
            return torch.cat([qu, qc], dim=0), torch.cat([ku, kc], dim=0), torch.cat([vu, vc], dim=0)

        if num_branches_cfg != 3:
            return q, k, v

        qu_before = torch.cat([qu[:num_heads], qu[:num_heads], qu[:num_heads]])
        qc_before = torch.cat([qc[:num_heads], qc[:num_heads], qc[:num_heads]])
        ku_before = torch.cat([ku[:num_heads], ku[:num_heads], ku[:num_heads]])
        kc_before = torch.cat([kc[:num_heads], kc[:num_heads], kc[:num_heads]])
        vu_before = torch.cat([vu[:num_heads * 2], vu[:num_heads]])
        vc_before = torch.cat([vc[:num_heads * 2], vc[:num_heads]])

        # 对齐 InfEdit 三分支：after 时第三分支 query 取第二分支
        qu_after = torch.cat([qu[:num_heads * 2], qu[num_heads:2 * num_heads]])
        qc_after = torch.cat([qc[:num_heads * 2], qc[num_heads:2 * num_heads]])
        ku_after = torch.cat([ku[:num_heads * 2], ku[:num_heads]])
        kc_after = torch.cat([kc[:num_heads * 2], kc[:num_heads]])
        vu_after = torch.cat([vu[:num_heads * 2], vu[:num_heads]])
        vc_after = torch.cat([vc[:num_heads * 2], vc[:num_heads]])

        qu = (1.0 - text_alpha) * qu_before + text_alpha * qu_after
        qc = (1.0 - text_alpha) * qc_before + text_alpha * qc_after
        ku = (1.0 - text_alpha) * ku_before + text_alpha * ku_after
        kc = (1.0 - text_alpha) * kc_before + text_alpha * kc_after
        vu = (1.0 - text_alpha) * vu_before + text_alpha * vu_after
        vc = (1.0 - text_alpha) * vc_before + text_alpha * vc_after
        return torch.cat([qu, qc], dim=0), torch.cat([ku, kc], dim=0), torch.cat([vu, vc], dim=0)

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        """
        四 schedule 版本：
        - text_cross_replace_steps: 控制 source/mutual/reference 文本路径
        - drag_cross_replace_steps: 控制 layout/target 拖拽路径
        """
        if is_cross:
            progress = self._schedule_progress()
            text_cross_center = self._resolve_schedule_value(
                self.text_cross_replace_steps, fallback=self._resolve_schedule_value(self.cross_replace_steps, fallback=0.7)
            )
            drag_cross_center = self._resolve_schedule_value(self.drag_cross_replace_steps, fallback=text_cross_center)
            use_before_text_cross = text_cross_center >= progress
            use_before_drag_cross = drag_cross_center >= progress

            h = attn.shape[0] // self.batch_size
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])

            # 四分支主逻辑（source/reference/layout/target）：
            # - reference: 对齐 copy2 的 target 逻辑，走 source->reference 的 schedule 混合
            # - target: 维持 target<-layout 的替换
            if self.batch_size == 4:
                alpha_keep_ref = self._hard_switch_alpha(use_before_text_cross)
                alpha_keep_target = self._hard_switch_alpha(use_before_drag_cross)

                attn_source, attn_reference, attn_layout, attn_target = attn[0], attn[1], attn[2], attn[3]

                attn_source_to_reference = self.replace_cross_attention(attn_source, attn_reference)
                attn_layout_to_target = self.replace_cross_attention(attn_layout, attn_target)

                attn[1] = (1.0 - alpha_keep_ref) * attn_source_to_reference + alpha_keep_ref * attn_reference
                attn[3] = (1.0 - alpha_keep_target) * attn_layout_to_target + alpha_keep_target * attn_target

                key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
                self.full_step_store_target[key].append(attn[3])
                self.full_step_store_reference[key].append(attn[1])

                attn_store = torch.cat([attn_layout_to_target, attn_source_to_reference], dim=0)
                attn_store = attn_store.reshape(2 * h, *attn_store.shape[2:])
                attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
                super(AttentionControlEdit, self).forward(attn_store, is_cross, place_in_unet)
                return attn

            # 五分支：source/mutual/reference/layout/target
            if self.batch_size == 5:
                alpha_keep_target = self._hard_switch_alpha(use_before_drag_cross)
                apply_drag_cross = self._in_injection_window()

                attn_source, attn_mutual, attn_reference, attn_layout, attn_target = (
                    attn[0], attn[1], attn[2], attn[3], attn[4]
                )

                # InfEdit 同款文本路径：
                # source->reference 作为 base_store，mutual->reference 作为 replace_new
                attn_source_to_reference = self.replace_cross_attention(attn_source, attn_reference)
                attn_mutual_to_reference = self.replace_cross_attention(attn_mutual, attn_reference)

                # Drag 路径保持 layout->target
                attn_layout_to_target = self.replace_cross_attention(attn_layout, attn_target)

                # 不让 start_step/start_layer 影响 cross-attn 文本路径：
                # 文本路径仍仅由 text_* schedule 控制。
                if use_before_text_cross:
                    # 对齐 InfEdit：early 阶段写入 replace_new（masa/mutual -> target/reference）。
                    attn[2] = attn_mutual_to_reference

                # target: 恢复窗口门控，窗口内按 schedule 混合，窗口外保持 target 自身。
                if apply_drag_cross:
                    attn[4] = (1.0 - alpha_keep_target) * attn_layout_to_target + alpha_keep_target * attn_target
                else:
                    attn[4] = attn_target

                key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
                self.full_step_store_target[key].append(attn[4])
                self.full_step_store_reference[key].append(attn[2])

                # LocalBlend 严格按 InfEdit 的两路 attention store：base_store + replace_new
                attn_store = torch.cat([attn_source_to_reference, attn_mutual_to_reference], dim=0)
                attn_store = attn_store.reshape(2 * h, *attn_store.shape[2:])
                attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
                super(AttentionControlEdit, self).forward(attn_store, is_cross, place_in_unet)
                return attn

            attn_base, attn_replace, attn_masa = attn[0], attn[1], attn[2]
            attn_base_store = self.replace_cross_attention(attn_base, attn_replace)
            attn_replace_new = self.replace_cross_attention(attn_masa, attn_replace)

            # 对齐 InfEdit 三分支：early 阶段写入 replace_new（masa->target）。
            if use_before_text_cross:
                attn[1] = attn_replace_new

            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            self.full_step_store_target[key].append(attn[1])
            self.full_step_store_reference[key].append(attn[2])

            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
            # 对齐 InfEdit：LocalBlend 的 store 使用 base_store + replace_new
            attn_store = torch.cat([attn_base_store, attn_replace_new])
            attn_store = attn_store.reshape(2 * h, *attn_store.shape[2:])
            super(AttentionControlEdit, self).forward(attn_store, is_cross, place_in_unet)
        return attn

    def __init__(self, prompts, num_steps: int,start_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend],
                 branch_count: Optional[int] = None,
                 attn_switch_mode: str = "hard",
                 text_cross_replace_steps: Optional[Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]]] = None,
                 text_self_replace_steps: Optional[Union[float, Tuple[float, float]]] = None,
                 drag_cross_replace_steps: Optional[Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]]] = None,
                 drag_self_replace_steps: Optional[Union[float, Tuple[float, float]]] = None,
                 drag_target_q_layout_mix: Optional[bool] = True):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = int(branch_count) if branch_count is not None else (len(prompts) + 1)
        self.self_replace_steps = self_replace_steps
        self.cross_replace_steps = cross_replace_steps
        self.text_cross_replace_steps = cross_replace_steps if text_cross_replace_steps is None else text_cross_replace_steps
        self.text_self_replace_steps = self_replace_steps if text_self_replace_steps is None else text_self_replace_steps
        self.drag_cross_replace_steps = cross_replace_steps if drag_cross_replace_steps is None else drag_cross_replace_steps
        self.drag_self_replace_steps = self_replace_steps if drag_self_replace_steps is None else drag_self_replace_steps
        self.drag_target_q_layout_mix = bool(drag_target_q_layout_mix)
        self.num_steps = num_steps
        self.start_steps = start_steps
        self.local_blend = local_blend
        self.ref_target_denoise_mix_start = 0.60
        self.enable_ref_kv_injection = True
        # 软切换逻辑已移除，固定使用硬切换。
        self.attn_switch_mode = "hard"

class AttentionRefine(AttentionControlEdit):
    def replace_cross_attention(self, attn_masa, att_replace):
        attn_masa_replace = attn_masa[:, :, self.mapper].squeeze()
        attn_replace = attn_masa_replace * self.alphas + att_replace * (1 - self.alphas)        
        return attn_replace

    def __init__(self, prompts, prompt_specifiers, num_steps: int,start_steps: int, 
                 cross_replace_steps: float, self_replace_steps: float,
                 start_step: int = 1, start_layer: int = 10, layer_idx: Optional[int] = None,
                 step_idx: Optional[int] = None, total_steps: Optional[int] = 10,
                 local_blend: Optional[LocalBlend] = None,
                 branch_count: Optional[int] = None,
                 attn_switch_mode: str = "hard",
                 text_cross_replace_steps: Optional[float] = None,
                 text_self_replace_steps: Optional[float] = None,
                 drag_cross_replace_steps: Optional[float] = None,
                 drag_self_replace_steps: Optional[float] = None,
                 drag_target_q_layout_mix: Optional[bool] = True,
                 original_height: Optional[int] = 512, original_width: Optional[int] = 512,
                 downsample_factor: Optional[int] = 8,):
        super(AttentionRefine, self).__init__(
            prompts, num_steps, start_steps, cross_replace_steps, self_replace_steps,
            local_blend, branch_count=branch_count, attn_switch_mode=attn_switch_mode,
            text_cross_replace_steps=text_cross_replace_steps,
            text_self_replace_steps=text_self_replace_steps,
            drag_cross_replace_steps=drag_cross_replace_steps,
            drag_self_replace_steps=drag_self_replace_steps,
            drag_target_q_layout_mix=drag_target_q_layout_mix,
        )
        self.mapper, alphas, ms, alpha_e, alpha_m = seq_aligner.get_refinement_mapper(prompts, prompt_specifiers, tokenizer, encoder, device)
        self.mapper, alphas, ms = self.mapper.to(device), alphas.to(device).to(torch_dtype), ms.to(device).to(torch_dtype)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])
        self.ms = ms.reshape(ms.shape[0], 1, 1, ms.shape[1])
        ms = ms.to(device)
        alpha_e = alpha_e.to(device)
        alpha_m = alpha_m.to(device)
        t_len = len(tokenizer(prompts[1])["input_ids"])
        if self.local_blend is not None:
            self.local_blend.set_map(ms, alphas, alpha_e, alpha_m, t_len)
        self.total_steps = total_steps
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, 16))  
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))  
        self.original_height = original_height
        self.original_width = original_width
        self.downsample_factor = downsample_factor

# [已删除] add_noise_and_save, save_metadata, save_manually - UI 中未使用

# --------------------------
# 模型加载与初始化
# --------------------------

# 初始化模型变量
scheduler = None
pipe = None
tokenizer = None
encoder = None
vae = None
original_unet = None

def load_models():
    global scheduler, pipe, tokenizer, encoder, vae, original_unet

    with MODEL_INIT_LOCK:
        # ========================================================
        # [核心修复] 防止模型重复加载导致爆显存/段错误
        # ========================================================
        if pipe is not None:
            print(f"✅ [load_models] 模型已存在于内存中，跳过加载。")
            return
        # ========================================================

        print(f"进程 {os.getpid()} 正在加载模型到 {device}...")
        print("Loading models...")
        
        # 添加 use_auth_token=False 或删除该参数（新版 diffusers 已弃用，这也是你报错警告的原因）
        # 如果你的环境必须验证 HuggingFace Token，请保持原样，否则建议设为 False
        scheduler = LCMScheduler.from_pretrained(
            model_id_or_path, 
            subfolder="scheduler"
        )
        
        pipe = EditPipeline.from_pretrained(
            model_id_or_path, 
            scheduler=scheduler, 
            torch_dtype=torch_dtype,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        )
        
        tokenizer = pipe.tokenizer
        encoder = pipe.text_encoder
        vae = pipe.vae
        
        original_unet = UNet2DConditionModel.from_pretrained(
            model_id_or_path,
            subfolder="unet",
            torch_dtype=torch.float16,
        ).to(device)
        
        if torch.cuda.is_available():
            pipe = pipe.to(device)
            
        pipe.unet = original_unet
        print("Models loaded successfully.")

# --------------------------
# 推理函数 Inference
# --------------------------

def inference(img, source_prompt, target_prompt,
              positive_prompt, negative_prompt,
              guidance_s, guidance_t,
              num_inference_steps, seed, strength,
              start_step, start_layer,
              cross_replace_steps, self_replace_steps,
              denoise, mask, selected_points,
              visualize_process, visualize_drag,
              drag_type, influence_range, pointcloud_domain="auto", image_name=None,
              mask_backend_mode="sam_refined",
              anchor_strategy_3d="auto",
              low_randomness=False, save_layout=True,
              local_blend_word="", mutual_blend_word="",
              ref_target_denoise_mix=False,
              ref_kv_injection=True,
              ref_target_denoise_mix_max=0.35,
              ref_target_denoise_mix_start=0.60,
              joint_target_refine_mix=False,
              joint_target_refine_mix_start=0.30,
              joint_target_refine_mix_max_out=0.45,
              joint_target_refine_mix_max_in=0.10,
              local_blend_thresh_e=0.3, local_blend_thresh_m=0.3,
              attn_switch_mode="hard",
              edit_mode="joint",
              hole_fill_mode=DEFAULT_HOLE_FILL_MODE,
              use_expanded_subject_fill=True,
              expanded_subject_fill_px=6,
              use_drag_guided_prefill=False,
              enable_3d_subject_scope_fill=False,
              drag_layout_latents=True,
              drag_target_latents=None,
              drag_clean_latents=True,
              drag_target_q_layout_mix=True,
              debug_artifacts=None,
              text_cross_replace_steps=None, text_self_replace_steps=None,
              drag_cross_replace_steps=None, drag_self_replace_steps=None,
              timing_collector=None,
              **kwargs):
    """
    简化后的推理函数。
    删除了: source_word, target_word, local, reference
    visualize_process: 保存每步过程图到 debug_files/denoise_steps
    visualize_drag: 输出拖拽调试到 debug_files/{mask_process,drag_process}
    pointcloud_domain: 3D 点云运行域（latent / image / auto）
    """
    global pipe, original_unet

    # 兼容旧调用命名
    if "low_randomness" in kwargs:
        low_randomness = kwargs["low_randomness"]
    if "pointcloud_domain" in kwargs:
        pointcloud_domain = kwargs["pointcloud_domain"]
    if "mask_backend_mode" in kwargs:
        mask_backend_mode = kwargs["mask_backend_mode"]
    if "anchor_strategy_3d" in kwargs:
        anchor_strategy_3d = kwargs["anchor_strategy_3d"]
    if "local_blend_word" in kwargs:
        local_blend_word = kwargs["local_blend_word"]
    if "mutual_blend_word" in kwargs:
        mutual_blend_word = kwargs["mutual_blend_word"]
    if "local_blend_thresh_e" in kwargs:
        local_blend_thresh_e = kwargs["local_blend_thresh_e"]
    if "local_blend_thresh_m" in kwargs:
        local_blend_thresh_m = kwargs["local_blend_thresh_m"]
    if "ref_target_denoise_mix" in kwargs:
        ref_target_denoise_mix = kwargs["ref_target_denoise_mix"]
    if "ref_kv_injection" in kwargs:
        ref_kv_injection = kwargs["ref_kv_injection"]
    if "ref_target_denoise_mix_max" in kwargs:
        ref_target_denoise_mix_max = kwargs["ref_target_denoise_mix_max"]
    if "ref_target_denoise_mix_start" in kwargs:
        ref_target_denoise_mix_start = kwargs["ref_target_denoise_mix_start"]
    if "joint_target_refine_mix" in kwargs:
        joint_target_refine_mix = kwargs["joint_target_refine_mix"]
    if "joint_target_refine_mix_start" in kwargs:
        joint_target_refine_mix_start = kwargs["joint_target_refine_mix_start"]
    if "joint_target_refine_mix_max_out" in kwargs:
        joint_target_refine_mix_max_out = kwargs["joint_target_refine_mix_max_out"]
    if "joint_target_refine_mix_max_in" in kwargs:
        joint_target_refine_mix_max_in = kwargs["joint_target_refine_mix_max_in"]
    if "attn_switch_mode" in kwargs:
        attn_switch_mode = kwargs["attn_switch_mode"]
    if "edit_mode" in kwargs:
        edit_mode = kwargs["edit_mode"]
    if "hole_fill_mode" in kwargs:
        hole_fill_mode = kwargs["hole_fill_mode"]
    if "use_expanded_subject_fill" in kwargs:
        use_expanded_subject_fill = kwargs["use_expanded_subject_fill"]
    if "expanded_subject_fill_px" in kwargs:
        expanded_subject_fill_px = kwargs["expanded_subject_fill_px"]
    if "use_drag_guided_prefill" in kwargs:
        use_drag_guided_prefill = kwargs["use_drag_guided_prefill"]
    if "enable_3d_subject_scope_fill" in kwargs:
        enable_3d_subject_scope_fill = kwargs["enable_3d_subject_scope_fill"]
    if "drag_layout_latents" in kwargs:
        drag_layout_latents = kwargs["drag_layout_latents"]
    if "drag_target_latents" in kwargs:
        drag_target_latents = kwargs["drag_target_latents"]
    if "drag_clean_latents" in kwargs:
        drag_clean_latents = kwargs["drag_clean_latents"]
    if "drag_target_q_layout_mix" in kwargs:
        drag_target_q_layout_mix = kwargs["drag_target_q_layout_mix"]
    if "text_cross_replace_steps" in kwargs:
        text_cross_replace_steps = kwargs["text_cross_replace_steps"]
    if "text_self_replace_steps" in kwargs:
        text_self_replace_steps = kwargs["text_self_replace_steps"]
    if "drag_cross_replace_steps" in kwargs:
        drag_cross_replace_steps = kwargs["drag_cross_replace_steps"]
    if "drag_self_replace_steps" in kwargs:
        drag_self_replace_steps = kwargs["drag_self_replace_steps"]
    def _has_bracket_markup(text):
        s = str(text or "")
        return ("[" in s) or ("]" in s)

    def _short_text(text, limit=120):
        s = str(text or "").replace("\n", " ").strip()
        if len(s) <= limit:
            return s
        return s[:limit] + "..."

    def _normalize_optional_schedule(v):
        if v is None:
            return None
        try:
            val = float(v)
        except Exception:
            return None
        return float(max(0.0, min(1.0, val)))

    local_blend_word = str(local_blend_word or "").strip()
    mutual_blend_word = str(mutual_blend_word or "").strip()
    local_blend_thresh_e = float(local_blend_thresh_e)
    local_blend_thresh_m = float(local_blend_thresh_m)
    start_step = int(round(float(start_step)))
    start_layer = int(round(float(start_layer)))
    start_step = max(0, start_step)
    start_layer = max(0, min(15, start_layer))
    ref_target_denoise_mix = bool(ref_target_denoise_mix)
    ref_kv_injection = bool(ref_kv_injection)
    ref_target_denoise_mix_max = max(0.0, min(1.0, float(ref_target_denoise_mix_max)))
    ref_target_denoise_mix_start = max(0.0, min(0.95, float(ref_target_denoise_mix_start)))
    joint_target_refine_mix = bool(joint_target_refine_mix)
    joint_target_refine_mix_start = max(0.0, min(0.95, float(joint_target_refine_mix_start)))
    joint_target_refine_mix_max_out = max(0.0, min(1.0, float(joint_target_refine_mix_max_out)))
    joint_target_refine_mix_max_in = max(0.0, min(1.0, float(joint_target_refine_mix_max_in)))
    drag_layout_latents = bool(drag_layout_latents)
    if drag_target_latents is None:
        drag_target_latents = drag_layout_latents
    else:
        drag_target_latents = bool(drag_target_latents)
    drag_clean_latents = bool(drag_clean_latents)
    drag_target_q_layout_mix = bool(drag_target_q_layout_mix)
    attn_switch_mode = "hard"
    edit_mode = str(edit_mode or "joint").strip().lower()
    text_cross_replace_steps = _normalize_optional_schedule(text_cross_replace_steps)
    text_self_replace_steps = _normalize_optional_schedule(text_self_replace_steps)
    drag_cross_replace_steps = _normalize_optional_schedule(drag_cross_replace_steps)
    drag_self_replace_steps = _normalize_optional_schedule(drag_self_replace_steps)
    mask_backend_mode = str(mask_backend_mode or "sam_refined").strip().lower()
    if mask_backend_mode in {"user_mask", "user", "raw_user_mask", "raw"}:
        mask_backend_mode = "user_mask"
    else:
        mask_backend_mode = "sam_refined"
    hole_fill_mode = normalize_hole_fill_mode(hole_fill_mode, default=DEFAULT_HOLE_FILL_MODE)
    use_expanded_subject_fill = bool(use_expanded_subject_fill)
    try:
        expanded_subject_fill_px = int(round(float(expanded_subject_fill_px)))
    except Exception:
        expanded_subject_fill_px = 6
    expanded_subject_fill_px = max(0, min(48, expanded_subject_fill_px))
    use_drag_guided_prefill = bool(use_drag_guided_prefill)
    enable_3d_subject_scope_fill = bool(enable_3d_subject_scope_fill)
    collect_debug_artifacts = isinstance(debug_artifacts, dict)
    save_step_images = bool(visualize_process)
    collect_final_only = collect_debug_artifacts and (not save_step_images)
    enable_debug_callback = save_step_images or collect_debug_artifacts
    if collect_debug_artifacts:
        debug_artifacts.clear()

    drag_pair_count = _count_drag_pairs(selected_points)
    has_drag_points = drag_pair_count > 0
    has_text_delta = _has_prompt_delta(source_prompt, target_prompt)
    effective_text_only = has_text_delta and (not has_drag_points)
    effective_joint_edit = has_text_delta and has_drag_points
    effective_edit_kind = "joint" if effective_joint_edit else "drag" if has_drag_points else "text" if has_text_delta else "identity"

    # 延迟加载
    if pipe is None:
        load_models()

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()

    # 1. 直接使用 original_unet
    pipe.unet = original_unet.to(device, dtype=torch.float16)

    pipe.modify_unet_forward()
    img_pil = Image.fromarray(img)
    original_height, original_width = img_pil.height, img_pil.width

    if denoise is False: strength = 1.0
    num_denoise_num = math.trunc(num_inference_steps * strength)
    num_start = num_inference_steps - num_denoise_num
    ddcm_eta = 0.0 if bool(low_randomness) else 1.0
    if num_denoise_num <= 0:
        num_denoise_num = 1
    # 避免 start_step 超出有效去噪步导致前端“滑了没反应”的感知。
    start_step = min(start_step, num_denoise_num - 1)

    # 固定使用五分支。
    branch_count = 5
    if effective_edit_kind == "text":
        print(
            f"[TextDebug][inference] image={image_name} denoise={bool(denoise)} "
            f"strength={float(strength):.3f} steps={int(num_inference_steps)} denoise_steps={int(num_denoise_num)} "
            f"branch_count={int(branch_count)} "
            f"drag_pairs={int(drag_pair_count)} "
            f"src_has_bracket={_has_bracket_markup(source_prompt)} tgt_has_bracket={_has_bracket_markup(target_prompt)} "
            f"attn_switch_mode={attn_switch_mode} "
            f"text_cross={text_cross_replace_steps} text_self={text_self_replace_steps} "
            f"drag_cross={drag_cross_replace_steps} drag_self={drag_self_replace_steps} "
            f"ref_kv_injection={ref_kv_injection} "
            f"local_blend='{local_blend_word}' mutual_blend='{mutual_blend_word}' "
            f"thresh_e={float(local_blend_thresh_e):.2f} thresh_m={float(local_blend_thresh_m):.2f}"
        )
        print(f"[TextDebug][inference] source_prompt={_short_text(source_prompt)}")
        print(f"[TextDebug][inference] target_prompt={_short_text(target_prompt)}")
    elif effective_edit_kind == "joint":
        print(
            f"[JointDebug][inference] image={image_name} denoise={bool(denoise)} "
            f"drag_pairs={int(drag_pair_count)} text_delta={has_text_delta} "
            f"branch_count={int(branch_count)} "
            f"drag_layout_latents={drag_layout_latents} drag_target_latents={drag_target_latents} "
            f"drag_clean_latents={drag_clean_latents} "
            f"joint_target_refine_mix={joint_target_refine_mix} "
            f"joint_mix_start={float(joint_target_refine_mix_start):.2f} "
            f"joint_mix_max_out={float(joint_target_refine_mix_max_out):.2f} "
            f"joint_mix_max_in={float(joint_target_refine_mix_max_in):.2f}"
        )
    elif effective_edit_kind == "drag":
        print(
            f"[DragDebug][inference] image={image_name} denoise={bool(denoise)} "
            f"drag_pairs={int(drag_pair_count)} branch_count={int(branch_count)} "
            f"ref_kv_injection={ref_kv_injection} "
            f"drag_cross={drag_cross_replace_steps} drag_self={drag_self_replace_steps} "
            f"drag_layout_latents={drag_layout_latents} drag_target_latents={drag_target_latents} "
            f"drag_clean_latents={drag_clean_latents} "
            f"drag_target_q_layout_mix={drag_target_q_layout_mix} "
            f"use_drag_guided_prefill={use_drag_guided_prefill}"
        )
    else:
        print(
            f"[EditDebug][inference] image={image_name} denoise={bool(denoise)} "
            f"drag_pairs={int(drag_pair_count)} text_delta={has_text_delta} "
            f"branch_count={int(branch_count)}"
        )
    local_blend = LocalBlend(
        thresh_e=local_blend_thresh_e,
        thresh_m=local_blend_thresh_m,
        save_inter=False,
    )
    try:
        controller = AttentionRefine(
            [source_prompt, target_prompt], [[local_blend_word, mutual_blend_word]],
            num_inference_steps, num_start,
            cross_replace_steps=cross_replace_steps,
            self_replace_steps=self_replace_steps,
            start_step=start_step, start_layer=start_layer,
            total_steps=num_denoise_num,
            local_blend=local_blend,
            branch_count=branch_count,
            attn_switch_mode=attn_switch_mode,
            text_cross_replace_steps=text_cross_replace_steps,
            text_self_replace_steps=text_self_replace_steps,
            drag_cross_replace_steps=drag_cross_replace_steps,
            drag_self_replace_steps=drag_self_replace_steps,
            drag_target_q_layout_mix=drag_target_q_layout_mix,
            original_height=original_height, original_width=original_width
        )
    except Exception as e:
        print(f"[LocalBlend Fallback] mapper failed ({e}), fallback to empty blend words.")
        controller = AttentionRefine(
            [source_prompt, target_prompt], [["", ""]],
            num_inference_steps, num_start,
            cross_replace_steps=cross_replace_steps,
            self_replace_steps=self_replace_steps,
            start_step=start_step, start_layer=start_layer,
            total_steps=num_denoise_num,
            local_blend=local_blend,
            branch_count=branch_count,
            attn_switch_mode=attn_switch_mode,
            text_cross_replace_steps=text_cross_replace_steps,
            text_self_replace_steps=text_self_replace_steps,
            drag_cross_replace_steps=drag_cross_replace_steps,
            drag_self_replace_steps=drag_self_replace_steps,
            drag_target_q_layout_mix=drag_target_q_layout_mix,
            original_height=original_height, original_width=original_width
        )
    controller.ref_target_denoise_mix_start = ref_target_denoise_mix_start
    controller.enable_ref_kv_injection = bool(ref_kv_injection)
    controller.drag_target_q_layout_mix = bool(drag_target_q_layout_mix)
    ptp_utils.register_attention_control(pipe, controller, False)

    debug_callback = None
    latest_branch_images = {}
    debug_dir = None
    denoise_debug_dir = None
    if enable_debug_callback:
        if save_step_images:
            # 统一扁平保存到 debug_files/denoise_steps，文件名前缀区分样本与步数
            debug_dir = os.path.join("debug_files", "denoise_steps")
            os.makedirs(debug_dir, exist_ok=True)
            denoise_debug_dir = os.path.abspath(debug_dir)

        def debug_callback(step_idx, timestep, data):
            try:
                if collect_final_only:
                    try:
                        if int(step_idx) < int(num_denoise_num - 1):
                            return
                    except Exception:
                        pass
                step_name = f"step_{step_idx:03d}"
                pred_xs_source = data.get("pred_xs_source")
                pred_xs_layout = data.get("pred_xs_layout")
                pred_xs_mutual = data.get("pred_xs_mutual")
                pred_x0 = data.get("pred_x0")
                pred_reference = data.get("pred_reference")

                pred_x0_img = None
                pred_reference_img = None
                pred_xs_source_img = None
                pred_xs_layout_img = None
                pred_xs_mutual_img = None

                if pred_xs_source is not None:
                    with torch.no_grad():
                        decoded_src_source = pipe.vae.decode(pred_xs_source / pipe.vae.config.scaling_factor, return_dict=False)[0]
                        decoded_src_source = pipe.image_processor.postprocess(decoded_src_source, output_type="pil", do_denormalize=[True] * decoded_src_source.shape[0])
                        pred_xs_source_img = decoded_src_source[0]
                        latest_branch_images["source"] = pred_xs_source_img.copy()

                if pred_xs_layout is not None:
                    with torch.no_grad():
                        decoded_src_layout = pipe.vae.decode(pred_xs_layout / pipe.vae.config.scaling_factor, return_dict=False)[0]
                        decoded_src_layout = pipe.image_processor.postprocess(decoded_src_layout, output_type="pil", do_denormalize=[True] * decoded_src_layout.shape[0])
                        pred_xs_layout_img = decoded_src_layout[0]
                        latest_branch_images["layout"] = pred_xs_layout_img.copy()

                if pred_xs_mutual is not None:
                    with torch.no_grad():
                        decoded_src_mutual = pipe.vae.decode(pred_xs_mutual / pipe.vae.config.scaling_factor, return_dict=False)[0]
                        decoded_src_mutual = pipe.image_processor.postprocess(decoded_src_mutual, output_type="pil", do_denormalize=[True] * decoded_src_mutual.shape[0])
                        pred_xs_mutual_img = decoded_src_mutual[0]
                        latest_branch_images["mutual"] = pred_xs_mutual_img.copy()

                if pred_x0 is not None:
                    with torch.no_grad():
                        decoded = pipe.vae.decode(pred_x0 / pipe.vae.config.scaling_factor, return_dict=False)[0]
                        decoded = pipe.image_processor.postprocess(decoded, output_type="pil", do_denormalize=[True] * decoded.shape[0])
                        pred_x0_img = decoded[0]
                        latest_branch_images["target"] = pred_x0_img.copy()

                if pred_reference is not None:
                    with torch.no_grad():
                        decoded_reference = pipe.vae.decode(pred_reference / pipe.vae.config.scaling_factor, return_dict=False)[0]
                        decoded_reference = pipe.image_processor.postprocess(decoded_reference, output_type="pil", do_denormalize=[True] * decoded_reference.shape[0])
                        pred_reference_img = decoded_reference[0]
                        latest_branch_images["reference"] = pred_reference_img.copy()

                if save_step_images:
                    concat_items = []
                    if pred_xs_source_img is not None:
                        concat_items.append((pred_xs_source_img, "Source"))
                    if pred_xs_mutual_img is not None:
                        concat_items.append((pred_xs_mutual_img, "Mutual"))
                    if pred_reference_img is not None:
                        concat_items.append((pred_reference_img, "Reference"))
                    if pred_xs_layout_img is not None:
                        concat_items.append((pred_xs_layout_img, "Layout"))
                    if pred_x0_img is not None:
                        concat_items.append((pred_x0_img, "Target"))

                    if len(concat_items) > 0 and debug_dir is not None:
                        base_w, base_h = concat_items[0][0].size
                        resized = []
                        labels = []
                        for im, label in concat_items:
                            if im.size != (base_w, base_h):
                                resized.append(im.resize((base_w, base_h), Image.BILINEAR))
                            else:
                                resized.append(im)
                            labels.append(label)

                        gap = 10
                        canvas_w = base_w * len(resized) + gap * (len(resized) - 1)

                        font_size = max(14, base_w // 18)
                        try:
                            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                        except Exception:
                            font = ImageFont.load_default()
                        text_bar_h = font_size + 16

                        canvas_h = base_h + text_bar_h
                        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

                        draw = ImageDraw.Draw(canvas)
                        x = 0
                        for im, label in zip(resized, labels):
                            # label at top
                            draw.text((x + 4, 4), label, fill=(0, 0, 0), font=font)
                            # image below
                            canvas.paste(im, (x, text_bar_h))
                            x += base_w + gap

                        canvas.save(os.path.join(debug_dir, f"{step_name}.png"))
            except Exception as e:
                print(f"[Debug] step {step_idx} save failed: {e}")

    # 2. 调用 Pipe
    results, results_layout = pipe(
        prompt=target_prompt,
        source_prompt=source_prompt,
        negative_prompt=negative_prompt, positive_prompt=positive_prompt,
        image=img_pil,
        num_inference_steps=num_inference_steps,
        strength=strength,
        guidance_scale=guidance_t, source_guidance_scale=guidance_s,
        denoise_model=denoise, callback=debug_callback,
        mask=mask, selected_points=selected_points,
        visualize_drag=visualize_drag,
        drag_type=drag_type,
        influence_range=influence_range,
        pointcloud_domain=pointcloud_domain,
        mask_backend_mode=mask_backend_mode,
        anchor_strategy_3d=anchor_strategy_3d,
        ddcm_eta=ddcm_eta,
        controller=controller,
        return_layout_image=bool(save_layout),
        edit_mode=edit_mode,
        hole_fill_mode=hole_fill_mode,
        use_expanded_subject_fill=use_expanded_subject_fill,
        expanded_subject_fill_px=expanded_subject_fill_px,
        use_drag_guided_prefill=use_drag_guided_prefill,
        enable_3d_subject_scope_fill=enable_3d_subject_scope_fill,
        drag_layout_latents=drag_layout_latents,
        drag_target_latents=drag_target_latents,
        drag_clean_latents=drag_clean_latents,
        drag_target_q_layout_mix=drag_target_q_layout_mix,
        enable_ref_target_denoise_mix=ref_target_denoise_mix,
        enable_ref_kv_injection=ref_kv_injection,
        ref_target_denoise_mix_max=ref_target_denoise_mix_max,
        ref_target_denoise_mix_start=ref_target_denoise_mix_start,
        joint_target_refine_mix=joint_target_refine_mix,
        joint_target_refine_mix_start=joint_target_refine_mix_start,
        joint_target_refine_mix_max_out=joint_target_refine_mix_max_out,
        joint_target_refine_mix_max_in=joint_target_refine_mix_max_in,
        timing_collector=timing_collector,
    )
    layout_img = results_layout.images[0] if (results_layout is not None) else None
    if collect_debug_artifacts:
        debug_artifacts["denoise_steps_dir"] = denoise_debug_dir
        debug_artifacts["final_branch_images"] = {k: v.copy() for k, v in latest_branch_images.items()}
        final_branch_dir = os.path.join("debug_files", "denoise_steps", "final_branches")
        final_branch_abs = os.path.abspath(final_branch_dir)
        os.makedirs(final_branch_dir, exist_ok=True)
        for name in os.listdir(final_branch_dir):
            path = os.path.join(final_branch_dir, name)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

        branch_order = ["source", "mutual", "reference", "layout", "target"]
        final_branch_paths = {}
        for key in branch_order:
            img_obj = latest_branch_images.get(key)
            if img_obj is None:
                continue
            out_path = os.path.join(final_branch_dir, f"{key}.png")
            try:
                img_obj.save(out_path)
                final_branch_paths[key] = os.path.abspath(out_path)
            except Exception as exc:
                print(f"[Debug] save final branch '{key}' failed: {exc}")
        debug_artifacts["final_branches_dir"] = final_branch_abs
        debug_artifacts["final_branch_paths"] = final_branch_paths

    return results.images[0], layout_img
