import os
from copy import deepcopy

import cv2
import gradio as gr
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from PIL.ImageOps import exif_transpose

from utils_drag.drag_processor import process_drag_request
from utils.metadata_utils import load_metadata
from tdedit_paths import META_DIR


def clear_all(length=480):
    return (
        gr.Image.update(value=None, height=length, width=length, interactive=True),
        gr.Image.update(value=None, height=length, width=length, interactive=False),
        gr.Image.update(value=None, height=length, width=length, interactive=False),
        [],
        None,
        None,
    )


def mask_image(image, mask, color=[255, 0, 0], alpha=0.5):
    if image.shape[-1] == 4:
        image = image[..., :3]

    out = deepcopy(image)
    img = deepcopy(image)

    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    img[mask > 0] = color
    out = cv2.addWeighted(img, alpha, out, 1 - alpha, 0, out)
    return out


def store_img(
    canvas,
    length,
    user_study_mode,
    image_filename,
    use_meta_data=False,
    meta_data_type="json",
    save_path=META_DIR,
):
    if canvas is None:
        return None, [], None, None, None, None, None, None

    img = canvas["background"]
    if img is None:
        return None, [], None, None, None, None, None, None

    height, width = img.shape[:2]
    new_width = length
    new_height = int(length * height / width)
    img_pil = Image.fromarray(img).convert("RGB")
    img_resized = img_pil.resize((new_width, new_height), Image.BILINEAR)
    img_array = np.array(img_resized)

    source_prompt, target_prompt, mask, selected_points = None, None, None, []

    if use_meta_data:
        try:
            source_prompt, target_prompt, mask, points, _, _, _, _, _, _, _, _, _, _ = load_metadata(
                image_filename, meta_data_type, save_path
            )
            if mask is not None:
                mask = cv2.resize(mask.astype(np.float32), (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                mask = np.uint8(mask > 0)
            else:
                mask = np.zeros((new_height, new_width), dtype=np.uint8)
            selected_points = points if points is not None else []
        except Exception as e:
            print(f"Metadata loading failed: {e}")
            mask = np.zeros((new_height, new_width), dtype=np.uint8)
    else:
        if len(canvas["layers"]) > 0:
            mask = np.float32(canvas["layers"][0][:, :, 3]) / 255.0
        else:
            mask = np.zeros((height, width), dtype=np.float32)

        mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        mask = np.uint8(mask > 0)
        selected_points = []

    if mask.sum() > 0:
        masked_img = mask_image(img_array, 1 - mask, color=[0, 0, 0], alpha=0.7)
    else:
        masked_img = img_array.copy()

    debug_depth_dir = "debug_files/depth_process"
    os.makedirs(debug_depth_dir, exist_ok=True)
    output_path = os.path.join(debug_depth_dir, "eval_temp.jpg")
    Image.fromarray(img_array).save(output_path, "PNG")

    return (
        img_array,
        selected_points,
        masked_img,
        mask,
        new_width,
        new_height,
        source_prompt if source_prompt else "",
        target_prompt if target_prompt else "",
    )


def display_image_with_metadata(
    file,
    length,
    use_meta_data,
    meta_data_type,
    save_path,
    drag_type="Hybrid-Rigid",
    use_origin_point=False,
):
    if file is None:
        return [None] * 18
    filename = os.path.basename(file.name)

    img = Image.open(file.name).convert("RGB")
    img = exif_transpose(img)

    height, width = img.size
    new_width = length
    new_height = int(length * height / width)
    img_resized = img.resize((new_width, new_height), Image.BILINEAR)
    img_array = np.array(img_resized)

    debug_depth_dir = "debug_files/depth_process"
    os.makedirs(debug_depth_dir, exist_ok=True)
    output_path = os.path.join(debug_depth_dir, "eval_temp.jpg")
    img_resized.save(output_path, "PNG")

    source_prompt, target_prompt, mask, selected_points = None, None, None, []
    current_drag_type = drag_type
    origin_point_distance, shield_distance, reletive_distance = 20.0, 30.0, 0.0
    influence_ratio, influence_range, non_influence_range = 1.0, 0.7, 0.7
    cross_replace_steps, self_replace_steps = 0.7, 0.7

    if use_meta_data:
        try:
            (
                source_prompt,
                target_prompt,
                mask,
                points,
                loaded_drag_type,
                _,
                _,
                _,
                _,
                _,
                influence_range,
                _,
                cross_replace_steps,
                self_replace_steps,
            ) = load_metadata(filename, meta_data_type, save_path)
            if mask is not None:
                mask = cv2.resize(mask.astype(np.float32), (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                mask = np.uint8(mask > 0)
                masked_img = mask_image(img_array, 1 - mask, color=[0, 0, 0], alpha=0.7)
            else:
                mask = np.zeros((new_height, new_width), dtype=np.uint8)
                masked_img = img_array.copy()
            selected_points = points if points is not None else []
            if loaded_drag_type:
                current_drag_type = loaded_drag_type
        except Exception:
            masked_img = img_array.copy()
            mask = np.zeros((new_height, new_width), dtype=np.uint8)
    else:
        masked_img = img_array.copy()
        mask = np.zeros((new_height, new_width), dtype=np.uint8)

    canvas_output = {"background": img_array, "layers": [], "composite": img_array}
    return (
        canvas_output,
        masked_img,
        img_array,
        filename,
        source_prompt,
        target_prompt,
        mask,
        selected_points,
        current_drag_type,
        "Combined",
        origin_point_distance,
        shield_distance,
        reletive_distance,
        influence_ratio,
        influence_range,
        non_influence_range,
        cross_replace_steps,
        self_replace_steps,
    )


def get_origin_point_from_mask(mask):
    if mask.ndim != 2:
        return (mask.shape[1] // 2, mask.shape[0] // 2)
    mask_uint8 = (mask == 1).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (mask.shape[1] // 2, mask.shape[0] // 2)

    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return (mask.shape[1] // 2, mask.shape[0] // 2)

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def get_points(img, sel_pix, mask, drag_type, use_origin_point, evt: gr.SelectData):
    sel_pix.append(evt.index)
    points = []

    for idx, point in enumerate(sel_pix):
        if idx % 2 == 0:
            cv2.circle(img, tuple(point), 8, (255, 0, 0), -1)
            points.append(tuple(point))
        else:
            cv2.circle(img, tuple(point), 8, (0, 0, 255), -1)
            points.append(tuple(point))

        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 2, tipLength=0.1)
            points = []

    return img if isinstance(img, np.ndarray) else np.array(img)


def preprocess_image(image, device, dtype=torch.float32):
    image = torch.from_numpy(image).float() / 127.5 - 1
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device, dtype)
    return image


def apply_drag(
    source_image,
    latents,
    mask,
    points,
    device,
    drag_type,
    influence_range,
    visualize_drag=False,
    pointcloud_domain="auto",
    drag_plan=None,
    return_plan=False,
    hole_fill_mode="sgf",
    use_expanded_subject_fill=True,
    expanded_subject_fill_px=6,
    use_drag_guided_prefill=False,
    mask_backend_mode="sam_refined",
    anchor_strategy_3d="auto",
    enable_3d_subject_scope_fill=False,
    vae=None,
    clean_latents=None,
    alpha_prod_t=None,
):
    if isinstance(source_image, torch.Tensor):
        src_img_np = source_image.detach().cpu()
        if src_img_np.ndim == 4:
            src_img_np = src_img_np[0]
        if src_img_np.shape[0] in [1, 3, 4]:
            src_img_np = src_img_np.permute(1, 2, 0).numpy()
        else:
            src_img_np = src_img_np.numpy()
        if src_img_np.min() < 0:
            src_img_np = (src_img_np + 1) / 2 * 255
        elif src_img_np.max() <= 2.0:
            src_img_np = src_img_np * 255
    elif isinstance(source_image, np.ndarray):
        src_img_np = source_image.copy()
        if src_img_np.max() <= 2.0:
            src_img_np = src_img_np * 255
    else:
        src_img_np = np.array(source_image)

    src_img_np = np.clip(src_img_np, 0, 255).astype(np.uint8)

    handle_points = []
    target_points = []
    num_pairs = len(points) // 2

    for i in range(num_pairs):
        start_pt = points[2 * i]
        end_pt = points[2 * i + 1]
        handle_points.append([start_pt[1], start_pt[0]])
        target_points.append([end_pt[1], end_pt[0]])

    if len(handle_points) == 0:
        print("[Apply Drag] No valid drag pairs found.")
        if return_plan:
            return latents, drag_plan
        return latents

    if return_plan:
        updated_latents, resolved_plan = process_drag_request(
            latents=latents,
            source_image_np=src_img_np,
            user_drawn_mask=mask,
            handle_points=handle_points,
            target_points=target_points,
            drag_mode=drag_type,
            rigid_ratio=influence_range,
            visualize_drag=visualize_drag,
            pointcloud_domain=pointcloud_domain,
            device=device,
            drag_plan=drag_plan,
            return_plan=True,
            hole_fill_mode=hole_fill_mode,
            use_expanded_subject_fill=use_expanded_subject_fill,
            expanded_subject_fill_px=expanded_subject_fill_px,
            use_drag_guided_prefill=use_drag_guided_prefill,
            mask_backend_mode=mask_backend_mode,
            anchor_strategy_3d=anchor_strategy_3d,
            enable_3d_subject_scope_fill=enable_3d_subject_scope_fill,
            vae=vae,
            clean_latents=clean_latents,
            alpha_prod_t=alpha_prod_t,
        )
    else:
        updated_latents = process_drag_request(
            latents=latents,
            source_image_np=src_img_np,
            user_drawn_mask=mask,
            handle_points=handle_points,
            target_points=target_points,
            drag_mode=drag_type,
            rigid_ratio=influence_range,
            visualize_drag=visualize_drag,
            pointcloud_domain=pointcloud_domain,
            device=device,
            drag_plan=drag_plan,
            return_plan=False,
            hole_fill_mode=hole_fill_mode,
            use_expanded_subject_fill=use_expanded_subject_fill,
            expanded_subject_fill_px=expanded_subject_fill_px,
            use_drag_guided_prefill=use_drag_guided_prefill,
            mask_backend_mode=mask_backend_mode,
            anchor_strategy_3d=anchor_strategy_3d,
            enable_3d_subject_scope_fill=enable_3d_subject_scope_fill,
            vae=vae,
            clean_latents=clean_latents,
            alpha_prod_t=alpha_prod_t,
        )

    if return_plan:
        return updated_latents.half(), resolved_plan
    return updated_latents.half()


__all__ = [
    "clear_all",
    "mask_image",
    "store_img",
    "display_image_with_metadata",
    "load_metadata",
    "get_origin_point_from_mask",
    "get_points",
    "preprocess_image",
    "apply_drag",
]
