import os
import gc
import json
import argparse
import numpy as np
from PIL import Image
import torch
import torch.multiprocessing as mp  # 导入多进程模块
from pathlib import Path
import fcntl
import time
import random
import re
import cv2
from PIL import ImageDraw, ImageFont
from utils_drag.hole_fill_modes import (
    HOLE_FILL_MODE_CHOICES,
    default_hole_fill_mode_for_mode,
    normalize_hole_fill_mode,
)

def _safe_float(value, default, min_value=None, max_value=None):
    """安全转 float，失败时回退 default，并可选做范围裁剪。"""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    if min_value is not None:
        val = max(min_value, val)
    if max_value is not None:
        val = min(max_value, val)
    return val


def _word_exists_in_prompt(word, prompt):
    w = str(word or "").strip().lower()
    p = str(prompt or "").strip().lower()
    if (not w) or (not p):
        return False
    tokens = re.findall(r"[a-z0-9]+", p)
    if w in tokens:
        return True
    return f" {w} " in f" {p} "


def _resolve_hole_fill_mode_for_mode(mode, raw_value):
    if raw_value in {"", "auto", None}:
        raw_value = default_hole_fill_mode_for_mode(mode)
    return normalize_hole_fill_mode(raw_value)


def _has_bracket_markup(text):
    s = str(text or "")
    return ("[" in s) or ("]" in s)


def _short_text(text, limit=120):
    s = str(text or "").replace("\n", " ").strip()
    if len(s) <= limit:
        return s
    return s[:limit] + "..."


def _pick_first_nonempty(*values):
    for val in values:
        if val is None:
            continue
        if isinstance(val, str):
            if val.strip():
                return val.strip()
            continue
        return val
    return ""


def _split_annotation_entry(entry):
    """
    统一兼容两种 JSON 结构：
    1) 新结构: {"source": {...}, "modified": {...}, "user_study": {...}}
    2) 旧结构: 扁平单层字典
    """
    if isinstance(entry, dict) and ("source" in entry or "modified" in entry or "user_study" in entry):
        source = entry.get("source") if isinstance(entry.get("source"), dict) else {}
        modified = entry.get("modified") if isinstance(entry.get("modified"), dict) else {}
        user_study = entry.get("user_study") if isinstance(entry.get("user_study"), dict) else {}
        return source, modified, user_study
    if isinstance(entry, dict):
        return dict(entry), {}, {}
    return {}, {}, {}


def _resolve_annotation_entry(entry, view_mode="effective"):
    """
    按视图返回当前样本的有效标注：
    - source: 只看 source
    - modified/effective: source 与 modified 合并（modified 覆盖 source），
      若 modified 为空则回退 source
    - user_study: source 与 user_study 合并（user_study 覆盖 source），
      若 user_study 为空则回退 source
    """
    source, modified, user_study = _split_annotation_entry(entry)
    merged = dict(source)
    if isinstance(modified, dict):
        merged.update(modified)
    user_merged = dict(source)
    if isinstance(user_study, dict):
        user_merged.update(user_study)

    if view_mode == "source":
        return dict(source), source, modified
    if view_mode == "user_study":
        if isinstance(user_study, dict) and len(user_study) > 0:
            return dict(user_merged), source, modified
        return dict(source), source, modified
    if view_mode in {"modified", "effective"}:
        if isinstance(modified, dict) and len(modified) > 0:
            return dict(merged), source, modified
        return dict(source), source, modified
    return dict(source), source, modified


def _draw_preview_like_ui(image_np, mask_np, points_xy, drag_type):
    """
    近似前端“4. 预览与加点”的显示：
    1) Mask 区域高亮，背景变暗
    2) 仅 Rigid 模式叠加质心+范围圈
    3) 绘制控制点与箭头
    """
    if image_np is None:
        return None
    vis = image_np.copy()

    mask_u8 = None
    if mask_np is not None:
        mask_u8 = (np.asarray(mask_np) > 0).astype(np.uint8)
        if int(mask_u8.sum()) > 0:
            alpha = 0.7
            img_f = vis.astype(np.float32)
            shadow = img_f * (1.0 - alpha)
            m3 = mask_u8[..., None]
            vis = (img_f * m3 + shadow * (1 - m3)).astype(np.uint8)

            guide_modes = {"Rigid", "2D-Rigid", "3D-Rigid"}
            drag_type_norm = str(drag_type or "").strip()
            if drag_type_norm in guide_modes:
                comp_u8 = (mask_u8 * 255).astype(np.uint8)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(comp_u8, connectivity=8)
                min_area = max(36, int(round(mask_u8.shape[0] * mask_u8.shape[1] * 2.5e-4)))
                for i in range(1, num_labels):
                    area = int(stats[i, cv2.CC_STAT_AREA])
                    if area < min_area:
                        continue
                    cx, cy = int(centroids[i][0]), int(centroids[i][1])
                    _, _, w, h, _ = stats[i]
                    # 与后端 detect_rigid_intent 严格一致：
                    # backend radius=((w+h)/4)*(4/9)，阈值 relative_dist<0.4
                    # => 判定圈半径系数 = 0.4*(4/9) = 8/45
                    zone_radius = int(((w + h) / 4.0) * (8.0 / 45.0))
                    cv2.circle(vis, (cx, cy), max(1, zone_radius), (0, 255, 0), 2)
                    cv2.circle(vis, (cx, cy), 6, (255, 255, 0), -1)
                    cv2.circle(vis, (cx, cy), 7, (0, 0, 0), 1)

    points = points_xy if isinstance(points_xy, list) else []
    if len(points) > 0:
        pairs = []
        for idx, pt in enumerate(points):
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                continue
            x, y = int(pt[0]), int(pt[1])
            if idx % 2 == 0:
                cv2.circle(vis, (x, y), 8, (255, 0, 0), -1)
            else:
                cv2.circle(vis, (x, y), 8, (0, 0, 255), -1)
            pairs.append((x, y))
            if len(pairs) == 2:
                cv2.arrowedLine(vis, pairs[0], pairs[1], (255, 255, 255), 2, tipLength=0.1)
                pairs = []
    return vis


def _build_concat_preview_result(source_image_np, result_pil, mask_np, points_xy, drag_type, source_prompt, target_prompt):
    left_np = _draw_preview_like_ui(source_image_np, mask_np, points_xy, drag_type)
    if left_np is None:
        return result_pil

    left_pil = Image.fromarray(left_np)
    res_pil = result_pil if isinstance(result_pil, Image.Image) else Image.fromarray(np.array(result_pil))
    res_pil_resized = res_pil.resize(left_pil.size, Image.BILINEAR)

    font_size = max(16, left_pil.size[0] // 25)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    text_bar_h = font_size + 16

    w, h = left_pil.size
    canvas_img = Image.new("RGB", (w * 2 + 10, h + text_bar_h), (255, 255, 255))
    canvas_img.paste(left_pil, (0, text_bar_h))
    canvas_img.paste(res_pil_resized, (w + 10, text_bar_h))

    draw = ImageDraw.Draw(canvas_img)
    src_label = f"Source: {source_prompt}" if source_prompt else "Source"
    tgt_label = f"Target: {target_prompt}" if target_prompt else "Target"
    draw.text((4, 4), src_label, fill=(0, 0, 0), font=font)
    draw.text((w + 14, 4), tgt_label, fill=(0, 0, 0), font=font)
    return canvas_img


def parse_args():
    parser = argparse.ArgumentParser(description="TDEdit 多卡并行批量评估脚本")
    # 基础路径
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mapping_file", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["text", "drag", "joint"], default="joint")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--fixed_first_n", type=int, default=-1, help="固定取前N张，<=0 表示不启用")
    parser.add_argument("--random_sample_size", type=int, default=-1, help="随机抽样数量，<=0 表示不抽样")
    parser.add_argument("--random_sample_seed", type=int, default=42, help="随机抽样种子")
    
    # 并行配置
    parser.add_argument("--device", nargs='+', type=int, default=[0], help="显卡ID列表，例如 --device 0 1 2")

    # 扩散核心参数 (略，保持和你之前提供的一致)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--guidance_s", type=float, default=1.0)
    parser.add_argument("--guidance_t", type=float, default=2.0)
    parser.add_argument("--positive_prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--cross_replace_steps", type=float, default=0.7)
    parser.add_argument("--cross_replace_steps_sm", type=float, default=-1.0)
    parser.add_argument("--cross_replace_steps_lt", type=float, default=-1.0)
    parser.add_argument("--self_replace_steps", type=float, default=0.7)
    parser.add_argument("--text_cross_replace_steps", type=float, default=-1.0, help=">=0 时覆盖 Text Branch 的 cross schedule")
    parser.add_argument("--text_self_replace_steps", type=float, default=-1.0, help=">=0 时覆盖 Text Branch 的 self schedule")
    parser.add_argument(
        "--drag_cross_replace_steps",
        type=float,
        default=0.0,
        help="Drag Branch 的 cross schedule；默认 0.0，即关闭 layout early mix",
    )
    parser.add_argument(
        "--drag_self_replace_steps",
        type=float,
        default=-1.0,
        help=">=0 时覆盖 Drag Branch 的 self schedule（即 GSAC 起点）",
    )
    parser.add_argument("--local_blend_thresh_e", type=float, default=0.3)
    parser.add_argument("--local_blend_thresh_m", type=float, default=0.3)
    parser.add_argument("--start_step", type=int, default=1)
    parser.add_argument("--start_layer", type=int, default=10)
    parser.add_argument(
        "--ref_target_denoise_mix",
        dest="ref_target_denoise_mix",
        action="store_true",
        help="启用后期 layout-side 与 reference 细节噪声混合，用于 layout->target DDCM（默认开启）",
    )
    parser.add_argument(
        "--no-ref_target_denoise_mix",
        dest="ref_target_denoise_mix",
        action="store_false",
        help="关闭后期 layout-side 与 reference 细节噪声混合",
    )
    parser.add_argument(
        "--ref_kv_injection",
        dest="ref_kv_injection",
        action="store_true",
        help="启用 target self-attn 上的 reference K/V 注入（默认开启）",
    )
    parser.add_argument(
        "--no-ref_kv_injection",
        dest="ref_kv_injection",
        action="store_false",
        help="关闭 target self-attn 上的 reference K/V 注入",
    )
    parser.add_argument(
        "--drag_target_q_layout_mix",
        dest="drag_target_q_layout_mix",
        action="store_true",
        help="启用 target Q 与 layout Q 的调度混合（LQM，默认开启）",
    )
    parser.add_argument(
        "--no-drag_target_q_layout_mix",
        dest="drag_target_q_layout_mix",
        action="store_false",
        help="关闭 target-layout Q 混合（LQM 消融）",
    )
    parser.add_argument("--ref_target_denoise_mix_max", type=float, default=0.7)
    parser.add_argument("--ref_target_denoise_mix_start", type=float, default=0.9)
    parser.add_argument(
        "--joint_target_refine_mix",
        dest="joint_target_refine_mix",
        action="store_true",
        help="Joint 模式启用 drag-aware 的 reference->target 蒸馏（默认关闭）",
    )
    parser.add_argument(
        "--no-joint_target_refine_mix",
        dest="joint_target_refine_mix",
        action="store_false",
        help="Joint 模式关闭 drag-aware 的 reference->target 蒸馏",
    )
    parser.add_argument("--joint_target_refine_mix_start", type=float, default=0.30)
    parser.add_argument("--joint_target_refine_mix_max_out", type=float, default=0.45)
    parser.add_argument("--joint_target_refine_mix_max_in", type=float, default=0.10)
    parser.add_argument("--drag_type", type=str, default="2D-Non-Rigid")
    parser.add_argument("--influence_range", type=float, default=0.5)
    parser.add_argument(
        "--drag_layout_latents",
        dest="drag_layout_latents",
        action="store_true",
        help="对 layout 分支的 noisy latents 应用拖拽（target 默认跟随 layout）",
    )
    parser.add_argument(
        "--no-drag_layout_latents",
        dest="drag_layout_latents",
        action="store_false",
        help="保持 layout 分支的 noisy latents 为未拖拽版本",
    )
    parser.add_argument(
        "--drag_target_latents",
        dest="drag_target_latents",
        action="store_true",
        help="对 target 分支的 noisy latents 应用拖拽（默认跟随 layout）",
    )
    parser.add_argument(
        "--no-drag_target_latents",
        dest="drag_target_latents",
        action="store_false",
        help="保持 target 分支的 noisy latents 为未拖拽版本",
    )
    parser.add_argument(
        "--drag_clean_latents",
        dest="drag_clean_latents",
        action="store_true",
        help="对 clean latents 应用拖拽，作为 DDCM 的 x0 锚点（默认开启）",
    )
    parser.add_argument(
        "--no-drag_clean_latents",
        dest="drag_clean_latents",
        action="store_false",
        help="保持 clean latents 为未拖拽版本",
    )
    parser.add_argument(
        "--pointcloud_domain",
        type=str,
        choices=["auto", "latent", "image"],
        default="image",
        help="与前端默认对齐：Point Cloud Domain 默认 image",
    )
    parser.add_argument(
        "--attn_switch_mode",
        type=str,
        default="hard",
        help="注意力切换模式固定为 hard（保留参数仅为兼容旧命令）",
    )
    parser.add_argument(
        "--hole_fill_mode",
        type=str,
        choices=["auto", *HOLE_FILL_MODE_CHOICES],
        default="auto",
        help=(
            "补洞策略。canonical: "
            + "/".join(HOLE_FILL_MODE_CHOICES)
            + "；默认按 mode 自动选择：text=sgf, drag=sgf, joint=sgf"
        ),
    )
    parser.add_argument(
        "--use_expanded_subject_fill",
        dest="use_expanded_subject_fill",
        action="store_true",
        help="启用 mask 扩展补全（默认开启）",
    )
    parser.add_argument(
        "--no-use_expanded_subject_fill",
        dest="use_expanded_subject_fill",
        action="store_false",
        help="关闭 mask 扩展补全",
    )
    parser.add_argument(
        "--expanded_subject_fill_px",
        type=int,
        default=6,
        help="mask 扩展像素（0-48）",
    )
    parser.set_defaults(use_expanded_subject_fill=True)
    parser.add_argument(
        "--use_drag_guided_prefill",
        dest="use_drag_guided_prefill",
        action="store_true",
        help="启用拖拽方向背景预填充（默认开启）",
    )
    parser.add_argument(
        "--no-use_drag_guided_prefill",
        dest="use_drag_guided_prefill",
        action="store_false",
        help="关闭拖拽方向背景预填充",
    )
    parser.set_defaults(use_drag_guided_prefill=True)
    parser.set_defaults(denoise=True)
    parser.set_defaults(ref_target_denoise_mix=True)
    parser.set_defaults(ref_kv_injection=True)
    parser.set_defaults(drag_target_q_layout_mix=True)
    parser.set_defaults(joint_target_refine_mix=False)
    parser.set_defaults(drag_layout_latents=False)
    parser.set_defaults(drag_target_latents=None)
    parser.set_defaults(drag_clean_latents=True)
    parser.add_argument("--denoise", dest="denoise", action="store_true")
    parser.add_argument("--no-denoise", dest="denoise", action="store_false")
    parser.add_argument("--low_randomness", action="store_true")
    parser.add_argument("--visualize_process", action="store_true")
    parser.add_argument("--visualize_drag", action="store_true")
    parser.add_argument("--skip_no_points", action="store_true", help="在 drag/joint 模式下跳过没有 points 的图片")
    parser.add_argument("--save_concat_preview", action="store_true", help="保存前端预览+结果拼接图")
    parser.add_argument(
        "--concat_as_result",
        action="store_true",
        help="与 --save_concat_preview 一起使用：将拼接图保存到 results 目录，替代原始结果图",
    )
    parser.add_argument(
        "--save_reference_as_result",
        action="store_true",
        help="兼容旧参数：等价于 --result_branch reference",
    )
    parser.add_argument(
        "--result_branch",
        type=str,
        choices=["auto", "target", "reference", "layout", "mutual", "source"],
        default="auto",
        help="保存到 results 目录的最终分支；auto 会按 topology 自动选择",
    )
    parser.add_argument(
        "--annotation_view",
        type=str,
        choices=["effective", "source", "modified", "user_study"],
        default="modified",
        help="读取标注时使用哪一视图：effective/source/modified/user_study（默认 modified）",
    )
    parser.add_argument(
        "--use_saved_drag_params",
        action="store_true",
        help="启用后，优先使用 JSON 里保存的 drag_type/influence_range（仅 drag/joint 生效）",
    )
    saved_type_group = parser.add_mutually_exclusive_group()
    saved_type_group.add_argument(
        "--use_saved_drag_type",
        dest="use_saved_drag_type",
        action="store_true",
        help="仅从 JSON 读取 drag_type；未显式设置时继承 --use_saved_drag_params。",
    )
    saved_type_group.add_argument(
        "--no-use_saved_drag_type",
        dest="use_saved_drag_type",
        action="store_false",
        help="不从 JSON 读取 drag_type。",
    )
    saved_influence_group = parser.add_mutually_exclusive_group()
    saved_influence_group.add_argument(
        "--use_saved_influence_range",
        dest="use_saved_influence_range",
        action="store_true",
        help="仅从 JSON 读取 influence_range；未显式设置时继承 --use_saved_drag_params。",
    )
    saved_influence_group.add_argument(
        "--no-use_saved_influence_range",
        dest="use_saved_influence_range",
        action="store_false",
        help="不从 JSON 读取 influence_range。",
    )
    parser.set_defaults(use_saved_drag_type=None, use_saved_influence_range=None)
    # 向后兼容旧参数名
    parser.add_argument(
        "--use_saved_tuning_params",
        dest="use_saved_drag_params",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()
    # 软切换逻辑已移除，统一强制 hard。
    args.attn_switch_mode = "hard"
    args.hole_fill_mode = _resolve_hole_fill_mode_for_mode(args.mode, getattr(args, "hole_fill_mode", "auto"))

    return args


def resolve_result_branch(args) -> str:
    raw = str(getattr(args, "result_branch", "auto") or "auto").strip().lower()
    if raw == "auto":
        if bool(getattr(args, "save_reference_as_result", False)):
            return "reference"
        if str(getattr(args, "mode", "") or "").strip().lower() == "text":
            return "reference"
        return "target"
    return raw

# ==========================================
# 进程执行函数
# ==========================================
def worker_inference(gpu_id, image_keys, all_data, args, timing_store=None):
    """每个显卡进程独立执行的任务"""
    # 【关键修复 1】物理隔绝显卡：必须在导入推理模块之前设置
    # 这确保了该进程及其加载的所有库只能“看到”这一张显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 【关键修复 2】延迟导入：确保模型加载动作发生在环境变量生效之后
    # 这样 tdedit_core 内部的 torch.device("cuda") 才会指向正确的物理卡
    from tdedit_core import inference, load_models
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_model_load_start = time.perf_counter()
    load_models()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_seconds = float(time.perf_counter() - t_model_load_start)
    model_load_peak_memory_bytes = int(
        torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    )
    print(f"🚀 显卡 {gpu_id} 已启动，负责 {len(image_keys)} 张图片")
    
    # 目录准备
    res_dir = os.path.join(args.output_dir, "results")
    concat_dir = os.path.join(args.output_dir, "results_concat")
    tmp_depth_dir = os.path.join(args.output_dir, f"temp_depth_gpu{gpu_id}")
    os.makedirs(res_dir, exist_ok=True)
    if args.save_concat_preview and (not args.concat_as_result):
        os.makedirs(concat_dir, exist_ok=True)
    os.makedirs(tmp_depth_dir, exist_ok=True)
    result_branch = resolve_result_branch(args)
    if len(image_keys) > 0:
        print(f"[run_batch] GPU {gpu_id} result_branch={result_branch}")

    # 循环处理分给该进程的 key
    image_edit_times = {}
    image_peak_memory_bytes = {}
    image_stage_times = {}
    for idx, img_id in enumerate(image_keys):
        raw_entry = all_data[img_id]
        item, source_item, modified_item = _resolve_annotation_entry(
            raw_entry,
            view_mode=args.annotation_view,
        )

        # =======================================================
        # 【新增】 跳过逻辑放在这里 (读取图片之前，节省时间)
        # =======================================================
        if args.mode in ["drag", "joint"] and args.skip_no_points:
            # 获取 points，如果没有则为空列表
            raw_points = item.get("points", [])
            # 如果 points 为 None 或者 空列表，则跳过
            if not raw_points or len(raw_points) == 0:
                print(f"⏩ GPU {gpu_id} 跳过: {img_id} (没有 points 数据)")
                continue
        # =======================================================

        img_rel = _pick_first_nonempty(
            item.get("image_path"),
            raw_entry.get("image_path") if isinstance(raw_entry, dict) else "",
            source_item.get("image_path") if isinstance(source_item, dict) else "",
            modified_item.get("image_path") if isinstance(modified_item, dict) else "",
            f"images/{img_id}.png",
        )
        img_full_path = os.path.join(args.data_root, img_rel)
        
        if not os.path.isfile(img_full_path):
            print(f"   ⚠️ GPU {gpu_id} 跳过: 找不到文件 {img_id}")
            continue

        raw_img_pil = Image.open(img_full_path).convert("RGB")
        temp_depth_path = os.path.join(tmp_depth_dir, f"{img_id}_tmp.jpg")
        raw_img_pil.save(temp_depth_path)
        raw_img_np = np.array(raw_img_pil)

        mask_rel = _pick_first_nonempty(
            item.get("mask_path"),
            raw_entry.get("mask_path") if isinstance(raw_entry, dict) else "",
            source_item.get("mask_path") if isinstance(source_item, dict) else "",
            modified_item.get("mask_path") if isinstance(modified_item, dict) else "",
        )
        mask_np = None
        if mask_rel:
            mask_full_path = os.path.join(args.data_root, mask_rel)
            if os.path.isfile(mask_full_path):
                mask_np = (np.array(Image.open(mask_full_path).convert("L")) > 127).astype(np.uint8)
        if mask_np is None:
            mask_np = np.zeros(raw_img_np.shape[:2], dtype=np.uint8)

        source_prompt = _pick_first_nonempty(
            item.get("source_prompt"),
            item.get("original_prompt"),
            raw_entry.get("source_prompt") if isinstance(raw_entry, dict) else "",
            raw_entry.get("original_prompt") if isinstance(raw_entry, dict) else "",
            source_item.get("source_prompt") if isinstance(source_item, dict) else "",
            source_item.get("original_prompt") if isinstance(source_item, dict) else "",
            modified_item.get("source_prompt") if isinstance(modified_item, dict) else "",
            modified_item.get("original_prompt") if isinstance(modified_item, dict) else "",
        )
        target_prompt = _pick_first_nonempty(
            item.get("target_prompt"),
            item.get("editing_prompt"),
            raw_entry.get("target_prompt") if isinstance(raw_entry, dict) else "",
            raw_entry.get("editing_prompt") if isinstance(raw_entry, dict) else "",
            source_item.get("target_prompt") if isinstance(source_item, dict) else "",
            source_item.get("editing_prompt") if isinstance(source_item, dict) else "",
            modified_item.get("target_prompt") if isinstance(modified_item, dict) else "",
            modified_item.get("editing_prompt") if isinstance(modified_item, dict) else "",
        )
        if not target_prompt:
            target_prompt = source_prompt
        if not source_prompt:
            print(f"⏩ GPU {gpu_id} 跳过: {img_id} (empty source prompt)")
            continue

        selected_points = _pick_first_nonempty(
            item.get("points"),
            raw_entry.get("points") if isinstance(raw_entry, dict) else [],
            source_item.get("points") if isinstance(source_item, dict) else [],
            modified_item.get("points") if isinstance(modified_item, dict) else [],
        )
        if not isinstance(selected_points, list):
            selected_points = []

        blended_word = str(
            _pick_first_nonempty(
                item.get("blended_word"),
                raw_entry.get("blended_word") if isinstance(raw_entry, dict) else "",
                source_item.get("blended_word") if isinstance(source_item, dict) else "",
                modified_item.get("blended_word") if isinstance(modified_item, dict) else "",
            ) or ""
        ).strip()
        bw_parts = blended_word.split() if blended_word else []
        # 与 InfEdit 评估逻辑一致：优先取 blended_word 第二个词做 local。
        if len(bw_parts) >= 2:
            local_blend_word = bw_parts[1]
        elif len(bw_parts) == 1:
            local_blend_word = bw_parts[0]
        else:
            local_blend_word = ""
        mutual_blend_word = str(
            _pick_first_nonempty(
                item.get("mutual_word", item.get("mutual", "")),
                raw_entry.get("mutual_word", raw_entry.get("mutual", "")) if isinstance(raw_entry, dict) else "",
                source_item.get("mutual_word", source_item.get("mutual", "")) if isinstance(source_item, dict) else "",
                modified_item.get("mutual_word", modified_item.get("mutual", "")) if isinstance(modified_item, dict) else "",
            ) or ""
        ).strip()

        if args.mode != "text":
            if (not mutual_blend_word) and len(bw_parts) >= 2 and (bw_parts[0] == bw_parts[1]):
                mutual_blend_word = bw_parts[0]
            # 词不在 target prompt 中时，禁用该词，避免 LocalBlend 对齐崩溃
            if not _word_exists_in_prompt(local_blend_word, target_prompt):
                local_blend_word = ""
            if not _word_exists_in_prompt(mutual_blend_word, target_prompt):
                mutual_blend_word = ""

        # 默认先使用全局参数（由 run_eval.py 按 mode 算好）
        # 说明：
        # 1) strength 不再从 JSON 读取
        # 2) drag_type 与 influence_range 可独立决定是否从 JSON 读取；
        #    未显式设置新开关时，保持旧版 use_saved_drag_params 的兼容行为。
        if args.strength is None:
            if args.mode == "text":
                final_strength = 1.0
            elif args.mode == "drag":
                final_strength = 0.7
            else:
                final_strength = 0.7
        else:
            final_strength = float(args.strength)
        final_drag_type = args.drag_type
        final_influence_range = float(args.influence_range)

        use_saved_drag_type = (
            bool(args.use_saved_drag_params)
            if args.use_saved_drag_type is None
            else bool(args.use_saved_drag_type)
        )
        use_saved_influence_range = (
            bool(args.use_saved_drag_params)
            if args.use_saved_influence_range is None
            else bool(args.use_saved_influence_range)
        )

        if args.mode in ["drag", "joint"] and use_saved_drag_type:
            item_drag_type = str(item.get("drag_type", "") or "").strip()
            if item_drag_type:
                final_drag_type = item_drag_type

        if args.mode in ["drag", "joint"] and use_saved_influence_range:
            final_influence_range = _safe_float(
                item.get("influence_range", final_influence_range),
                default=final_influence_range,
                min_value=0.0,
                max_value=1.0,
            )

        if args.mode == "text":
            selected_points = []
            # 对齐 InfEdit PIE text：只使用 blended_word 解析的 local(target blend)，
            # mutual(source blend) 固定为空字符串。
            mutual_blend_word = ""
        elif args.mode == "drag":
            target_prompt = source_prompt
            # 与 UI Drag 模式对齐：不走词级 LocalBlend 引导
            local_blend_word = ""
            mutual_blend_word = ""

        run_ref_target_denoise_mix = bool(args.ref_target_denoise_mix)
        run_ref_kv_injection = bool(args.ref_kv_injection)
        run_drag_layout_latents = bool(args.drag_layout_latents)
        run_drag_target_latents = (
            run_drag_layout_latents if args.drag_target_latents is None else bool(args.drag_target_latents)
        )
        run_drag_clean_latents = bool(args.drag_clean_latents)
        if args.mode == "text":
            # text-only 自动屏蔽后两分支相关路径：固定解码 reference 分支。
            run_ref_target_denoise_mix = False
            run_ref_kv_injection = False
            run_drag_layout_latents = False
            run_drag_target_latents = False
            run_drag_clean_latents = False

        try:
            print(f"[{idx+1}/{len(image_keys)}] GPU {gpu_id} 处理中: {img_id}")
            if args.mode == "text":
                print(
                    f"[TextDebug][run_batch] id={img_id} "
                    f"src_has_bracket={_has_bracket_markup(source_prompt)} "
                    f"tgt_has_bracket={_has_bracket_markup(target_prompt)} "
                    f"src_len={len(str(source_prompt))} tgt_len={len(str(target_prompt))} "
                    f"attn_switch_mode={args.attn_switch_mode} "
                    f"local_blend='{local_blend_word}' mutual_blend='{mutual_blend_word}' "
                    f"denoise={bool(args.denoise)} strength={float(final_strength):.3f} steps={int(args.steps)}"
                )
                print(f"[TextDebug][run_batch] source_prompt={_short_text(source_prompt)}")
                print(f"[TextDebug][run_batch] target_prompt={_short_text(target_prompt)}")
            # 调用推理函数
            # CUDA kernels are asynchronous. Synchronize at both boundaries so
            # the reported latency is wall-clock GPU latency rather than CPU
            # dispatch time, and reset the allocator peak for each image.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t_edit_start = time.perf_counter()
            text_cross_steps = None if float(args.text_cross_replace_steps) < 0.0 else float(args.text_cross_replace_steps)
            text_self_steps = None if float(args.text_self_replace_steps) < 0.0 else float(args.text_self_replace_steps)
            drag_cross_steps = None if float(args.drag_cross_replace_steps) < 0.0 else float(args.drag_cross_replace_steps)
            drag_self_steps = None if float(args.drag_self_replace_steps) < 0.0 else float(args.drag_self_replace_steps)
            debug_artifacts = {} if result_branch != "target" else None
            stage_timing_collector = {}
            res_img, _ = inference(
                img=raw_img_np,
                source_prompt=source_prompt,
                target_prompt=target_prompt,
                positive_prompt=args.positive_prompt,
                negative_prompt=args.negative_prompt,
                guidance_s=args.guidance_s,
                guidance_t=args.guidance_t,
                num_inference_steps=args.steps,
                seed=args.seed,
                strength=final_strength,
                start_step=args.start_step,
                start_layer=args.start_layer,
                cross_replace_steps=args.cross_replace_steps,
                self_replace_steps=args.self_replace_steps,
                text_cross_replace_steps=text_cross_steps,
                text_self_replace_steps=text_self_steps,
                drag_cross_replace_steps=drag_cross_steps,
                drag_self_replace_steps=drag_self_steps,
                denoise=args.denoise,
                mask=mask_np,
                selected_points=selected_points,
                visualize_process=args.visualize_process,
                visualize_drag=args.visualize_drag,
                drag_type=final_drag_type,
                influence_range=final_influence_range,
                drag_layout_latents=run_drag_layout_latents,
                drag_target_latents=run_drag_target_latents,
                drag_clean_latents=run_drag_clean_latents,
                pointcloud_domain=args.pointcloud_domain,
                attn_switch_mode=args.attn_switch_mode,
                hole_fill_mode=args.hole_fill_mode,
                use_expanded_subject_fill=bool(args.use_expanded_subject_fill),
                expanded_subject_fill_px=int(args.expanded_subject_fill_px),
                use_drag_guided_prefill=bool(args.use_drag_guided_prefill),
                image_name=img_id,
                low_randomness=args.low_randomness,
                save_layout=False,
                local_blend_word=local_blend_word,
                mutual_blend_word=mutual_blend_word,
                local_blend_thresh_e=args.local_blend_thresh_e,
                local_blend_thresh_m=args.local_blend_thresh_m,
                ref_target_denoise_mix=run_ref_target_denoise_mix,
                ref_kv_injection=run_ref_kv_injection,
                drag_target_q_layout_mix=bool(args.drag_target_q_layout_mix),
                ref_target_denoise_mix_max=float(args.ref_target_denoise_mix_max),
                ref_target_denoise_mix_start=float(args.ref_target_denoise_mix_start),
                joint_target_refine_mix=bool(args.joint_target_refine_mix),
                joint_target_refine_mix_start=float(args.joint_target_refine_mix_start),
                joint_target_refine_mix_max_out=float(args.joint_target_refine_mix_max_out),
                joint_target_refine_mix_max_in=float(args.joint_target_refine_mix_max_in),
                edit_mode=args.mode,
                debug_artifacts=debug_artifacts,
                timing_collector=stage_timing_collector,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            edit_seconds = float(time.perf_counter() - t_edit_start)
            image_peak_memory_bytes[str(img_id)] = int(
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            )
            image_stage_times[str(img_id)] = {
                str(key): float(value) for key, value in stage_timing_collector.items()
            }

            image_to_save = res_img
            if result_branch != "target":
                branch_img = None
                if isinstance(debug_artifacts, dict):
                    final_branch_images = debug_artifacts.get("final_branch_images", {})
                    if isinstance(final_branch_images, dict):
                        branch_img = final_branch_images.get(result_branch)
                if branch_img is not None:
                    image_to_save = branch_img
                else:
                    print(
                        f"[run_batch] id={img_id} branch '{result_branch}' missing, fallback to target output."
                    )

            # 保存结果（可选：拼接前端预览 + 最终图）
            out_path_raw = os.path.join(res_dir, f"{img_id}.png")
            if args.save_concat_preview:
                concat_img = _build_concat_preview_result(
                    source_image_np=raw_img_np,
                    result_pil=image_to_save,
                    mask_np=mask_np,
                    points_xy=selected_points,
                    drag_type=final_drag_type,
                    source_prompt=source_prompt,
                    target_prompt=target_prompt,
                )
                if args.concat_as_result:
                    concat_img.save(out_path_raw)
                else:
                    image_to_save.save(out_path_raw)
                    concat_img.save(os.path.join(concat_dir, f"{img_id}.png"))
            else:
                image_to_save.save(out_path_raw)
            image_edit_times[str(img_id)] = edit_seconds
            
        except Exception as e:
            print(f"❌ GPU {gpu_id} 处理 {img_id} 失败: {e}")
        finally:
            # Drop per-image tensors before asking the allocator to release
            # cached blocks. Merely calling empty_cache while the previous
            # result/debug objects remain referenced does not prevent gradual
            # growth in long joint-edit sweeps.
            res_img = None
            image_to_save = None
            debug_artifacts = None
            stage_timing_collector = None
            gc.collect()
            torch.cuda.empty_cache()
            if os.path.exists(temp_depth_path):
                os.remove(temp_depth_path)

    # 清理该 GPU 的临时文件夹
    try:
        if not os.listdir(tmp_depth_dir):
            os.rmdir(tmp_depth_dir)
    except:
        pass
    if timing_store is not None:
        timing_store[gpu_id] = {
            "times": image_edit_times,
            "peak_memory_bytes": image_peak_memory_bytes,
            "stage_times_sec": image_stage_times,
            "model_load_seconds": model_load_seconds,
            "model_load_peak_memory_bytes": model_load_peak_memory_bytes,
        }
    print(f"✅ 显卡 {gpu_id} 任务全部完成。")



def update_master_config(args):
    # 1. 确定大 JSON 的存放位置
    # 根据你的目录结构：OUT_DIR 是根目录往后数第4级
    # EXP_ROOT / Dataset / Method / SubDir / Timestamp
    out_path = Path(args.output_dir)
    exp_root = out_path.parents[3] # 获取 3_experiments_results 目录
    master_json_path = os.path.join(exp_root, "master_configs.json")
    
    # 2. 准备本次实验的数据
    # 提取时间戳（即当前文件夹名）
    exp_id = out_path.name 
    # 获取数据集类型（从路径判断或从 args 逻辑判断）
    dataset_name = "DragBench" if "1_Drag-Bench" in str(out_path) else "PIE-Bench"
    mode_name = args.mode # "drag", "text", "joint"
    
    current_config = vars(args).copy()
    from datetime import datetime
    current_config["recorded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 3. 读取并更新 (带文件锁保护)
    db = {}
    if os.path.exists(master_json_path):
        try:
            with open(master_json_path, "r", encoding="utf-8") as f:
                db = json.load(f)
        except:
            db = {}

    # 初始化层级结构
    if dataset_name not in db: db[dataset_name] = {}
    if mode_name not in db[dataset_name]: db[dataset_name][mode_name] = {}
    
    # 存入数据
    db[dataset_name][mode_name][exp_id] = current_config

    # 4. 写入文件
    with open(master_json_path, "w", encoding="utf-8") as f:
        # 尝试加锁，防止多个任务同时结束时写入冲突（虽然 main 只运行一次，但养成好习惯）
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(db, f, indent=4, ensure_ascii=False)
            fcntl.flock(f, fcntl.LOCK_UN)
        except:
            json.dump(db, f, indent=4, ensure_ascii=False)

# ==========================================
# 主入口
# ==========================================
def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    args.code_project_name = project_root.name
    args.code_project_root = str(project_root)
    args.code_entry_script = str(Path(__file__).resolve())
    
    # 首先保存当前文件夹里的小 JSON ( experiment_config.json )
    os.makedirs(args.output_dir, exist_ok=True)
    local_config_path = os.path.join(args.output_dir, "experiment_config.json")
    with open(local_config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    # 然后更新根目录的大 JSON ( master_configs.json )
    try:
        update_master_config(args)
        print(f"📖 Master Registry 已更新。")
    except Exception as e:
        print(f"⚠️ Master Registry 更新失败: {e}")


    # 加载数据
    with open(args.mapping_file, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    
    all_keys = list(all_data.keys())
    end_pos = args.end_idx if args.end_idx != -1 else len(all_keys)
    selected_keys = all_keys[args.start_idx : end_pos]

    # drag/joint + skip_no_points 时，采样池先过滤无点样本，避免产出数量不足
    if args.mode in ["drag", "joint"] and args.skip_no_points:
        eligible_keys = []
        for k in selected_keys:
            raw_entry = all_data.get(k, {})
            item, _, _ = _resolve_annotation_entry(raw_entry, view_mode=args.annotation_view)
            raw_points = item.get("points", [])
            if isinstance(raw_points, list) and len(raw_points) > 0:
                eligible_keys.append(k)
        sample_pool = eligible_keys
    else:
        sample_pool = list(selected_keys)

    if args.fixed_first_n > 0:
        if len(sample_pool) == 0:
            selected_keys = []
        else:
            n = min(int(args.fixed_first_n), len(sample_pool))
            selected_keys = sample_pool[:n]
        print(
            f"📌 固定前N启用: fixed_first_n={args.fixed_first_n}, selected={len(selected_keys)}"
        )
    elif args.random_sample_size > 0:
        rng = random.Random(int(args.random_sample_seed))

        if len(sample_pool) == 0:
            selected_keys = []
        else:
            n = min(int(args.random_sample_size), len(sample_pool))
            selected_keys = rng.sample(sample_pool, n)
            selected_keys.sort()

        print(
            f"🎲 随机抽样启用: sample_size={args.random_sample_size}, "
            f"seed={args.random_sample_seed}, selected={len(selected_keys)}"
        )
    
    num_gpus = len(args.device)
    print(f"🚀 准备多卡并行处理，显卡列表: {args.device}，共 {len(selected_keys)} 张图片")

    # 数据分片
    shards = np.array_split(selected_keys, num_gpus)
    
    # 必须使用 spawn 方法来启动 CUDA 相关子进程
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    timing_store = manager.dict()
    
    processes = []
    for i in range(num_gpus):
        gpu_id = args.device[i]
        shard = shards[i].tolist()
        p = mp.Process(target=worker_inference, args=(gpu_id, shard, all_data, args, timing_store))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    merged_times = {}
    merged_peak_memory = {}
    merged_stage_times = {}
    model_load_seconds = []
    model_load_peak_memory_bytes = []
    for shard_payload in timing_store.values():
        if not isinstance(shard_payload, dict):
            continue
        # Backward compatibility with timing stores written by older workers.
        if "times" in shard_payload:
            shard_times = shard_payload.get("times", {})
            shard_memory = shard_payload.get("peak_memory_bytes", {})
            shard_stages = shard_payload.get("stage_times_sec", {})
            if "model_load_seconds" in shard_payload:
                model_load_seconds.append(float(shard_payload["model_load_seconds"]))
            if "model_load_peak_memory_bytes" in shard_payload:
                model_load_peak_memory_bytes.append(int(shard_payload["model_load_peak_memory_bytes"]))
        else:
            shard_times = shard_payload
            shard_memory = {}
            shard_stages = {}
        for image_id, sec in shard_times.items():
            try:
                merged_times[str(image_id)] = float(sec)
            except Exception:
                continue
        for image_id, value in shard_memory.items():
            try:
                merged_peak_memory[str(image_id)] = int(value)
            except Exception:
                continue
        for image_id, values in shard_stages.items():
            if isinstance(values, dict):
                merged_stage_times[str(image_id)] = {
                    str(key): float(value) for key, value in values.items()
                }

    sorted_times = dict(sorted(merged_times.items(), key=lambda kv: kv[0]))
    sorted_peak_memory = dict(sorted(merged_peak_memory.items(), key=lambda kv: kv[0]))
    avg_edit = float(sum(sorted_times.values()) / len(sorted_times)) if sorted_times else 0.0
    peak_allocated = max(sorted_peak_memory.values()) if sorted_peak_memory else 0
    editing_time_payload = {
        "unit": "seconds",
        "description": "Per-image editing latency measured inside the model inference call with CUDA synchronization at both boundaries (single GPU worker; model loading and image saving excluded).",
        "image_times_sec": sorted_times,
        "peak_memory_allocated_bytes": sorted_peak_memory,
        "stage_times_sec": dict(sorted(merged_stage_times.items(), key=lambda kv: kv[0])),
        "summary": {
            "count": int(len(sorted_times)),
            "avg_editing_sec": avg_edit,
            "max_peak_memory_allocated_bytes": int(peak_allocated),
            "max_peak_memory_allocated_gib": float(peak_allocated / (1024 ** 3)),
            "model_load_seconds_by_worker": model_load_seconds,
            "max_model_load_peak_memory_allocated_bytes": int(max(model_load_peak_memory_bytes) if model_load_peak_memory_bytes else 0),
            "max_model_load_peak_memory_allocated_gib": float((max(model_load_peak_memory_bytes) if model_load_peak_memory_bytes else 0) / (1024 ** 3)),
        },
    }
    editing_time_path = os.path.join(args.output_dir, "editing_times.json")
    with open(editing_time_path, "w", encoding="utf-8") as f:
        json.dump(editing_time_payload, f, indent=4, ensure_ascii=False)
    print(f"⏱️ 编辑耗时已写入: {editing_time_path} (count={len(sorted_times)}, avg={avg_edit:.4f}s)")

    print(f"\n✅ 批量并行处理全部完成。结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main()
