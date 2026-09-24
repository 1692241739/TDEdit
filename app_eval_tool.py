import argparse
import os
import sys


def _env_to_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def build_launch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TDEdit Gradio UI launcher")
    parser.add_argument("--host", default=os.environ.get("TDEDIT_GRADIO_HOST", "0.0.0.0"))

    default_port_raw = os.environ.get("TDEDIT_GRADIO_PORT", os.environ.get("GRADIO_SERVER_PORT", "7890"))
    try:
        default_port = int(default_port_raw)
    except (TypeError, ValueError):
        default_port = 7890
    parser.add_argument("--port", type=int, default=default_port)

    parser.set_defaults(share=_env_to_bool("TDEDIT_GRADIO_SHARE", default=False))
    parser.add_argument("--share", dest="share", action="store_true")
    parser.add_argument("--no-share", dest="share", action="store_false")

    parser.set_defaults(queue=True)
    parser.add_argument("--queue", dest="queue", action="store_true")
    parser.add_argument("--no-queue", dest="queue", action="store_false")

    parser.add_argument(
        "--disable-localhost-fallback",
        action="store_true",
        help="Disable automatic _frontend=False retry when localhost accessibility check fails.",
    )
    parser.add_argument(
        "--allowed-path",
        action="append",
        default=[],
        help="Additional allowed path for Gradio file access, can be passed multiple times.",
    )
    parser.add_argument(
        "--no-dataset-allowed-paths",
        action="store_true",
        help="Do not auto-append dataset roots from DATASET_CONFIG into allowed_paths.",
    )
    return parser


# CLI help must not construct Gradio, load CUDA libraries, or require models.
if __name__ == "__main__" and any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    build_launch_parser().parse_args()

import gradio as gr
import json
import re
import shutil
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch
from datetime import datetime
from tdedit_paths import DRAGBENCH_ROOT, PIEBENCH_ROOT, PROCESS_ROOT

# ==========================================
# 1. 导入核心工具
# ==========================================
from tdedit_core import inference, load_models
from utils.segment_utils import get_interactive_masks_batch, preload_sam2
from utils_drag.hole_fill_modes import (
    HOLE_FILL_MODE_CHOICES as CANONICAL_HOLE_FILL_MODE_CHOICES,
    default_hole_fill_mode_for_mode,
    normalize_hole_fill_mode,
)
# ==========================================
# 2. 数据集配置
# ==========================================
DATASET_CONFIG = {
    "DragBench": {
        "root": DRAGBENCH_ROOT,
        "json_files": ["dragbench.json"]
    },
    "PIE-Bench": {
        "root": PIEBENCH_ROOT,
        "json_files": ["pie_bench.json"]
    }
}
DEFAULT_DATASET_NAME = "DragBench"

MODIFIED_MASK_DIR = "masks"
DRAG_TYPE_CHOICES = [
    "2D-Hybrid",
    "2D-Non-Rigid",
    "2D-Rigid",
    "3D-Hybrid",
    "3D-Non-Rigid",
    "3D-Rigid",
]
DEFAULT_DRAG_TYPE = "2D-Non-Rigid"
DEFAULT_INFLUENCE_RANGE = 0.5
DEFAULT_LOCAL_BLEND_WORD = ""
DEFAULT_MUTUAL_BLEND_WORD = ""
DEFAULT_LOCAL_BLEND_THRESH_E = 0.6
DEFAULT_LOCAL_BLEND_THRESH_M = 0.6
DEFAULT_DRAG_CROSS_REPLACE_STEPS = 0.0
DEFAULT_SHOW_ORIGINAL_MASK = False
INFERENCE_MASK_MODE_CHOICES = ["SAM Refined", "User Mask"]
DEFAULT_INFERENCE_MASK_MODE = "SAM Refined"
HOLE_FILL_MODE_CHOICES = list(CANONICAL_HOLE_FILL_MODE_CHOICES)
DEFAULT_EXPANDED_SUBJECT_FILL = True
DEFAULT_DRAG_GUIDED_PREFILL = False
DEFAULT_3D_SUBJECT_SCOPE_FILL = False
DEFAULT_EXPANDED_SUBJECT_FILL_PX = 2
DEFAULT_EXPANDED_SUBJECT_FILL_PX_BY_MODE = {
    "Joint": 2,
    "Text": 2,
    "Drag": 4,
}
POINTCLOUD_DOMAIN_CHOICES = ["latent", "image"]
DEFAULT_POINTCLOUD_DOMAIN = "image"
DEFAULT_REF_TARGET_DENOISE_MIX = True
DEFAULT_REF_TARGET_DENOISE_MIX_MAX = 0.70
DEFAULT_REF_TARGET_DENOISE_MIX_START = 0.30
DEFAULT_FRONTEND_DRAG_LAYOUT_LATENTS = False
DEFAULT_FRONTEND_DRAG_TARGET_LATENTS = False
DEFAULT_FRONTEND_DRAG_CLEAN_LATENTS = True
DEFAULT_DRAG_TARGET_Q_LAYOUT_MIX = True
DEFAULT_FRONTEND_REF_KV_INJECTION = True
DATASET_VIEW_CHOICES = ["source", "modified", "user_study"]
DEFAULT_DATASET_VIEW = "modified"
USER_STUDY_FLAG_KEY = "is_user_study"
EDIT_MODE_CHOICES = ["Joint", "Text", "Drag"]
DEFAULT_EDIT_MODE = "Drag"
DEFAULT_HOLE_FILL_MODE = default_hole_fill_mode_for_mode(DEFAULT_EDIT_MODE)
PAPER_MODE_FULL = "Paper Mode"
PAPER_MODE_LITE = "Paper Mode Lite"
RUN_MODE_CHOICES = ["Normal Mode", PAPER_MODE_FULL, PAPER_MODE_LITE]
DEFAULT_RUN_MODE = "Normal Mode"
DEFAULT_STRENGTH_BY_MODE = {
    "Joint": 0.7,
    "Text": 1.0,
    "Drag": 0.75,
}
DEFAULT_TARGET_GUIDANCE_BY_MODE = {
    "Joint": 2.0,
    "Text": 2.3,
    "Drag": 1.0,
}
DEFAULT_STEPS_BY_MODE = {
    "Joint": 15,
    "Text": 15,
    "Drag": 17,
}
# 连通域最小面积（用于过滤零碎噪点，避免被识别为独立组件）
COMPONENT_MIN_AREA_BASE = 36
COMPONENT_MIN_AREA_RATIO = 2.5e-4
# 额外按“相对最大连通域面积”过滤小碎块，避免前端仍显示零星噪点
COMPONENT_MIN_AREA_RELATIVE_TO_LARGEST = 0.05

PAPER_PROCESS_ROOT = PROCESS_ROOT
PAPER_REGISTRY_NAME = "_paper_registry.json"
PAPER_DEBUG_SUBDIRS = ("mask_process", "drag_process", "final_results")
PAPER_EXCLUDED_MASK_KEYWORDS = ("effect_mask", "insert_mask", "vacate_mask", "trail_mask")
PAPER_EXCLUDED_DEBUG_REGEXES = (
    r"sam_comp\d+_summary",
    r"__panel_\d+",
)
PAPER_EXCLUDED_SPLIT_TILE_REGEXES = (
    # 按需添加需跳过的子图规则；当前默认不过滤。
)
PAPER_FORCE_SINGLE_DEBUG_REGEXES = (
    # 这类图在 paper 里保留整图即可，不应按拼图拆子图。
    r"hole_masks",
)
PAPER_NO_TEXT_SINGLE_ENV = "TDEDIT_PAPER_NO_TEXT_SINGLE"
PAPER_COMPOSITE_NAME_HINTS = (
    "summary_analysis",
    "summary_overview",
    "trace_compare",
    "masks_cutouts",
    "hole_masks",
    "fill_process",
    "fill_scopes",
    "final_preview",
    "component_story",
    "intent_story",
    "full_depth",
    "component_depths",
    "rotated_depths",
    "rgb_projection",
)
PAPER_COMPOSITE_GRID_RULES = (
    (r"(?:^|_)3drotate_03_trace_compare", 1, 4),
    (r"(?:^|_)3drotate_02_masks_cutouts", 1, 4),
    (r"(?:^|_)3drotate_01_fill_scopes", 1, 4),
    (r"(?:^|_)3drotate_04_final_preview", 1, 4),
    (r"trace_compare", 2, 2),
    (r"masks_cutouts", 2, 3),
    (r"hole_masks", 2, 2),
    (r"fill_process", 1, 4),
    (r"fill_scopes", 1, 4),
    (r"final_preview", 1, 2),
    (r"component_story", 1, 3),
    (r"intent_story", 1, 3),
    (r"summary_analysis", 2, 3),
    (r"summary_overview", 2, 4),
    (r"sam_comp\d+_analysis", 2, 3),
    (r"full_depth", 1, 4),
    (r"component_depths", 1, 4),
    (r"rotated_depths", 1, 3),
    (r"rgb_projection", 1, 3),
)
# 前端预览中仅 Rigid 模式显示“意图识别圈/质心点”可视引导；
# Hybrid 与 Non-Rigid 统一不显示该类圈，避免误导。
GUIDE_VIS_DRAG_TYPES = {"Rigid", "2D-Rigid", "3D-Rigid"}
# 同一个 user mask 连通域内强制单主体，不同连通域仍独立处理。
FORCE_SINGLE_COMPONENT_PER_MASK = True


def _to_split_annotation_entry(entry):
    """
    Normalize an older flat entry into the new split format:
    {
      "source": {...},
      "modified": {...},
      "user_study": {...}
    }
    """
    if isinstance(entry, dict) and ("source" in entry or "modified" in entry or "user_study" in entry):
        source = entry.get("source")
        modified = entry.get("modified")
        user_study = entry.get("user_study")
        return {
            "source": source if isinstance(source, dict) else {},
            "modified": modified if isinstance(modified, dict) else {},
            "user_study": user_study if isinstance(user_study, dict) else {},
        }
    return {
        "source": entry if isinstance(entry, dict) else {},
        "modified": {},
        "user_study": {},
    }


def _get_merged_annotation_entry(split_entry, overlay_key):
    source = split_entry.get("source", {})
    overlay = split_entry.get(overlay_key, {})
    eff = dict(source)
    if isinstance(overlay, dict):
        eff.update(overlay)
    return eff


def _annotation_entry_has_content(entry):
    if not isinstance(entry, dict):
        return False
    for value in entry.values():
        if isinstance(value, str):
            if value.strip():
                return True
            continue
        if isinstance(value, (list, dict, tuple, set)):
            if len(value) > 0:
                return True
            continue
        if value is not None:
            return True
    return False


def _resolve_annotation_entry_for_view(split_entry, dataset_view_mode=DEFAULT_DATASET_VIEW):
    source = split_entry.get("source", {})
    modified = split_entry.get("modified", {})
    user_study = split_entry.get("user_study", {})
    has_modified = _annotation_entry_has_content(modified)
    has_user_study = _annotation_entry_has_content(user_study)

    requested_view = str(dataset_view_mode or DEFAULT_DATASET_VIEW).strip().lower()
    if requested_view not in DATASET_VIEW_CHOICES:
        requested_view = DEFAULT_DATASET_VIEW

    active_view = requested_view
    if requested_view == "modified" and not has_modified:
        active_view = "source"
    elif requested_view == "user_study" and not has_user_study:
        active_view = "source"

    if active_view == "source":
        entry = dict(source)
    elif active_view == "modified":
        entry = _get_merged_annotation_entry({"source": source, "modified": modified}, "modified")
    else:
        entry = _get_merged_annotation_entry({"source": source, "user_study": user_study}, "user_study")
    return entry, active_view, has_modified, has_user_study


def _normalize_drag_type(value):
    v = str(value or "").strip()
    return v if v in DRAG_TYPE_CHOICES else DEFAULT_DRAG_TYPE


def _normalize_influence_range(value):
    try:
        val = float(value)
    except (TypeError, ValueError):
        return DEFAULT_INFLUENCE_RANGE
    return float(np.clip(val, 0.0, 1.0))


def _normalize_blend_word(value):
    return str(value or "").strip()


def _normalize_blend_thresh(value, default_value):
    try:
        val = float(value)
    except (TypeError, ValueError):
        return float(default_value)
    return float(np.clip(val, 0.0, 1.0))


def _resolve_blend_words_from_entry(entry):
    if not isinstance(entry, dict):
        return DEFAULT_LOCAL_BLEND_WORD, DEFAULT_MUTUAL_BLEND_WORD

    local_word = _normalize_blend_word(entry.get("local_blend_word", ""))
    mutual_word = _normalize_blend_word(entry.get("mutual_word", entry.get("mutual", "")))
    blended_word = _normalize_blend_word(entry.get("blended_word", ""))

    parts = blended_word.split() if blended_word else []
    if not local_word:
        if len(parts) >= 2:
            local_word = parts[1]
        elif len(parts) == 1:
            local_word = parts[0]
    if (not mutual_word) and len(parts) >= 2 and (parts[0] == parts[1]):
        mutual_word = parts[0]

    return local_word, mutual_word


def _normalize_edit_mode(value):
    v = str(value or "").strip()
    return v if v in EDIT_MODE_CHOICES else DEFAULT_EDIT_MODE


def _normalize_inference_mask_mode(value):
    v = str(value or "").strip()
    return v if v in INFERENCE_MASK_MODE_CHOICES else DEFAULT_INFERENCE_MASK_MODE


def _default_strength_for_mode(edit_mode):
    mode = _normalize_edit_mode(edit_mode)
    return float(DEFAULT_STRENGTH_BY_MODE.get(mode, DEFAULT_STRENGTH_BY_MODE[DEFAULT_EDIT_MODE]))


def _default_target_guidance_for_mode(edit_mode):
    mode = _normalize_edit_mode(edit_mode)
    return float(
        DEFAULT_TARGET_GUIDANCE_BY_MODE.get(
            mode,
            DEFAULT_TARGET_GUIDANCE_BY_MODE[DEFAULT_EDIT_MODE],
        )
    )


def _default_steps_for_mode(edit_mode):
    mode = _normalize_edit_mode(edit_mode)
    return int(DEFAULT_STEPS_BY_MODE.get(mode, DEFAULT_STEPS_BY_MODE[DEFAULT_EDIT_MODE]))


def _default_expanded_subject_fill_px_for_mode(edit_mode):
    mode = _normalize_edit_mode(edit_mode)
    return int(
        DEFAULT_EXPANDED_SUBJECT_FILL_PX_BY_MODE.get(
            mode,
            DEFAULT_EXPANDED_SUBJECT_FILL_PX_BY_MODE[DEFAULT_EDIT_MODE],
        )
    )


def _default_denoise_for_mode(edit_mode):
    return True


def _default_hole_fill_mode_for_mode(edit_mode):
    mode = _normalize_edit_mode(edit_mode)
    return default_hole_fill_mode_for_mode(mode)


def _apply_mode_defaults(edit_mode, source_prompt_text, target_prompt_text):
    mode = _normalize_edit_mode(edit_mode)
    source_prompt_text = str(source_prompt_text or "")
    target_prompt_text = str(target_prompt_text or "")

    if mode == "Drag":
        target_prompt_text = source_prompt_text

    return (
        target_prompt_text,
        _default_strength_for_mode(mode),
        _default_target_guidance_for_mode(mode),
    )


def _is_modified_entry_user_edited(split_entry):
    source = split_entry.get("source", {}) if isinstance(split_entry, dict) else {}
    modified = split_entry.get("modified", {}) if isinstance(split_entry, dict) else {}
    if not isinstance(modified, dict) or len(modified) == 0:
        return False

    mask_rel = str(modified.get("mask_path", "") or "")
    if mask_rel.startswith(f"{MODIFIED_MASK_DIR}/"):
        return True

    if modified.get("points", []) != source.get("points", []):
        return True
    if str(modified.get("source_prompt", "")) != str(source.get("source_prompt", "")):
        return True
    if str(modified.get("target_prompt", "")) != str(source.get("target_prompt", "")):
        return True
    if str(modified.get("blended_word", "")) != str(source.get("blended_word", "")):
        return True
    if str(modified.get("mutual_word", modified.get("mutual", ""))) != str(source.get("mutual_word", source.get("mutual", ""))):
        return True

    src_drag = _normalize_drag_type(source.get("drag_type", DEFAULT_DRAG_TYPE))
    mod_drag = _normalize_drag_type(modified.get("drag_type", DEFAULT_DRAG_TYPE))
    if src_drag != mod_drag:
        return True

    src_range = _normalize_influence_range(
        source.get("influence_range", DEFAULT_INFLUENCE_RANGE)
    )
    mod_range = _normalize_influence_range(
        modified.get("influence_range", DEFAULT_INFLUENCE_RANGE)
    )
    if abs(src_range - mod_range) > 1e-9:
        return True

    src_image_rel = str(source.get("image_path", ""))
    mod_image_rel = str(modified.get("image_path", src_image_rel))
    if mod_image_rel != src_image_rel:
        return True

    return False


def _dump_json_with_compact_points(json_path, data):
    """
    Keep points list compact when writing json.
    """
    placeholders = {}

    def _replace(node):
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if k == "points" and isinstance(v, list):
                    key = f"___POINTS_HOLDER_{len(placeholders)}___"
                    placeholders[key] = json.dumps(v, ensure_ascii=False)
                    out[k] = key
                else:
                    out[k] = _replace(v)
            return out
        if isinstance(node, list):
            return [_replace(v) for v in node]
        return node

    replaced = _replace(data)
    full_json_str = json.dumps(replaced, indent=4, ensure_ascii=False)
    for key, compact in placeholders.items():
        full_json_str = full_json_str.replace(f'"{key}"', compact)

    with open(json_path, "w", encoding="utf-8") as f:
        f.write(full_json_str)

# ==========================================
# 3. 基础图像处理工具
# ==========================================

def apply_mask_overlay(image_np, mask_np):
    if mask_np is None or mask_np.sum() == 0: return image_np.copy()
    mask_binary = (mask_np > 0).astype(np.uint8)
    alpha = 0.7
    img_f = image_np.astype(float)
    shadow = img_f * (1 - alpha)
    m_3d = mask_binary[..., None]
    final = img_f * m_3d + shadow * (1 - m_3d)
    return final.astype(np.uint8)


def _sanitize_path_component(text, default_value="sample"):
    raw = str(text or "").strip()
    if not raw:
        return default_value
    cleaned = re.sub(r"[^0-9A-Za-z\-\._]+", "_", raw)
    cleaned = cleaned.strip("_.")
    return cleaned or default_value


def _sha1_of_bytes(raw_bytes):
    hasher = hashlib.sha1()
    hasher.update(raw_bytes)
    return hasher.hexdigest()


def _sha1_of_ndarray(arr):
    if arr is None:
        return ""
    arr_np = np.asarray(arr)
    return _sha1_of_bytes(arr_np.tobytes())


def _normalize_mask_u8(mask_like, fallback_hw):
    h, w = int(fallback_hw[0]), int(fallback_hw[1])
    if mask_like is None:
        return np.zeros((h, w), dtype=np.uint8)
    mask_np = np.asarray(mask_like)
    if mask_np.ndim > 2:
        mask_np = mask_np[..., 0]
    if mask_np.shape[:2] != (h, w):
        mask_np = cv2.resize(mask_np.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    if mask_np.max() <= 1.0:
        mask_u8 = (mask_np > 0.5).astype(np.uint8) * 255
    else:
        mask_u8 = (mask_np > 127).astype(np.uint8) * 255
    return mask_u8.astype(np.uint8)


def _to_pil_image(image_obj):
    if image_obj is None:
        return None
    if isinstance(image_obj, Image.Image):
        return image_obj
    image_np = np.asarray(image_obj)
    if image_np.dtype != np.uint8:
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    return Image.fromarray(image_np)


def _draw_drag_points_overlay(image_np, points):
    canvas = np.asarray(image_np).copy()
    pts = points if isinstance(points, list) else []
    pair = []
    for idx, pt in enumerate(pts):
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        x, y = int(pt[0]), int(pt[1])
        color = (255, 0, 0) if idx % 2 == 0 else (0, 0, 255)
        cv2.circle(canvas, (x, y), 8, color, -1)
        pair.append((x, y))
        if len(pair) == 2:
            cv2.arrowedLine(canvas, pair[0], pair[1], (255, 255, 255), 2, tipLength=0.1)
            pair = []
    return canvas


def _get_drag_type_from_run_config(run_config):
    if isinstance(run_config, dict):
        return str(run_config.get("drag_type", "") or "")
    return ""


def _render_mask_overlay_with_guides(image_np, mask_np, drag_type):
    """
    仅在 Rigid 模式复用前端同款可视化（质心 + 绿色圈），其余模式保持纯 mask 叠加。
    """
    mask01 = (np.asarray(mask_np) > 0).astype(np.uint8)
    drag_type_norm = str(drag_type or "").strip()
    if drag_type_norm in GUIDE_VIS_DRAG_TYPES:
        return draw_visual_guides(image_np, mask01, drag_type_norm)
    return apply_mask_overlay(image_np, mask01)


def _load_paper_registry(root_dir):
    registry_path = os.path.join(root_dir, PAPER_REGISTRY_NAME)
    if not os.path.isfile(registry_path):
        return registry_path, {"version": 1, "records": []}
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("registry is not dict")
        records = data.get("records", [])
        if not isinstance(records, list):
            records = []
        data["records"] = records
        data.setdefault("version", 1)
        return registry_path, data
    except Exception as exc:
        print(f"[paper_mode] load registry failed: {exc}")
        return registry_path, {"version": 1, "records": []}


def _save_paper_registry(registry_path, data):
    tmp_path = f"{registry_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, registry_path)


def _next_record_id(records, image_slug):
    pattern = re.compile(rf"^{re.escape(image_slug)}_(\d{{4}})$")
    max_seq = 0
    for rec in records:
        rid = str(rec.get("record_id", ""))
        m = pattern.match(rid)
        if m:
            max_seq = max(max_seq, int(m.group(1)))
    return f"{image_slug}_{max_seq + 1:04d}"


def _find_existing_record(records, image_slug, param_hash):
    for rec in records:
        if rec.get("image_slug") == image_slug and rec.get("param_hash") == param_hash:
            return rec
    return None


def _reset_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def _prepare_paper_debug_workspace(debug_root):
    for sub in PAPER_DEBUG_SUBDIRS:
        _reset_dir(os.path.join(debug_root, sub))


def _find_first_file_with_keywords(base_dir, keywords):
    if not os.path.isdir(base_dir):
        return None
    keys = [str(k).lower() for k in keywords if str(k).strip()]
    hits = []
    for root, _, files in os.walk(base_dir):
        for fname in files:
            lname = fname.lower()
            if all(k in lname for k in keys):
                hits.append(os.path.join(root, fname))
    if not hits:
        return None
    hits.sort()
    return hits[0]


def _infer_regular_grid_for_image(image_path, base_size=512):
    def _looks_like_composite_name(name_lower):
        if not name_lower:
            return False
        if any(k in name_lower for k in PAPER_COMPOSITE_NAME_HINTS):
            return True
        for patt, _, _ in PAPER_COMPOSITE_GRID_RULES:
            if re.search(patt, name_lower):
                return True
        return False

    def _grid_from_rows_cols(rows, cols, w, h, min_tile=96):
        rows = int(rows)
        cols = int(cols)
        if rows <= 0 or cols <= 0:
            return None
        if rows * cols <= 1:
            return None
        if rows > 8 or cols > 8:
            return None
        if (w % cols) != 0 or (h % rows) != 0:
            return None
        tile_w = int(w // cols)
        tile_h = int(h // rows)
        if tile_w < int(min_tile) or tile_h < int(min_tile):
            return None
        return int(rows), int(cols), int(tile_w), int(tile_h)

    try:
        with Image.open(image_path) as im:
            w, h = im.size
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    if w <= base_size and h <= base_size:
        return None
    name_lower = os.path.basename(str(image_path)).lower()
    composite_hint = _looks_like_composite_name(name_lower)

    # 优先：按“已知拼图命名规则”匹配网格（支持任意 tile 尺寸，不限 512 系列）。
    for patt, rows, cols in PAPER_COMPOSITE_GRID_RULES:
        if not re.search(patt, name_lower):
            continue
        matched = _grid_from_rows_cols(rows, cols, w, h, min_tile=96)
        if matched is not None:
            return matched

    # 仅允许“等尺寸方格拼图”自动切分，避免把非规则总结图误拆成大量子图。
    # 典型拼图来自 512x512 网格，也兼容少量其它常见方格边长。
    tile_candidates = [1024, 768, 640, 576, int(base_size), 448, 384, 320, 256]
    for tile in tile_candidates:
        if tile <= 0:
            continue
        tile = int(tile)
        if (w % tile) != 0 or (h % tile) != 0:
            continue
        cols = w // tile
        rows = h // tile
        if rows * cols <= 1:
            continue
        if w < base_size * 2 and h < base_size * 2:
            continue
        if rows > 8 or cols > 8:
            continue
        return int(rows), int(cols), tile, tile

    # 兜底：仅对“命名明显是拼图”的图，尝试常见布局（避免误拆普通单图）。
    if composite_hint:
        common_layouts = (
            (2, 4), (2, 3), (2, 2),
            (1, 4), (1, 3), (1, 2),
            (3, 3), (3, 2), (4, 2),
        )
        for rows, cols in common_layouts:
            matched = _grid_from_rows_cols(rows, cols, w, h, min_tile=96)
            if matched is not None:
                return matched

    return None


def _save_paper_process_record(
    process_root,
    image_name,
    source_image_np,
    user_mask_np,
    display_mask_np,
    drag_points,
    result_img,
    layout_img,
    source_prompt_text,
    target_prompt_text,
    run_config,
    debug_root,
    debug_artifacts=None,
    lite_mode=False,
):
    os.makedirs(process_root, exist_ok=True)
    image_slug = _sanitize_path_component(image_name or "unnamed")
    lite_manifest_filename = f"{image_slug}__record_manifest.txt"
    drag_type_for_vis = _get_drag_type_from_run_config(run_config)

    mask_u8 = _normalize_mask_u8(user_mask_np, source_image_np.shape[:2])
    display_mask_u8 = None
    if display_mask_np is not None:
        try:
            display_mask_u8 = _normalize_mask_u8(display_mask_np, source_image_np.shape[:2])
        except Exception:
            display_mask_u8 = None
    signature = {
        "image_name": str(image_name or ""),
        "image_slug": image_slug,
        "source_prompt": str(source_prompt_text or ""),
        "target_prompt": str(target_prompt_text or ""),
        "drag_points": drag_points if isinstance(drag_points, list) else [],
        "source_image_sha1": _sha1_of_ndarray(source_image_np),
        "user_mask_sha1": _sha1_of_ndarray(mask_u8),
        "display_mask_sha1": _sha1_of_ndarray(display_mask_u8) if display_mask_u8 is not None else "",
        "run_config": run_config,
    }
    param_hash = _sha1_of_bytes(json.dumps(signature, ensure_ascii=False, sort_keys=True).encode("utf-8"))

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    registry_path, registry = _load_paper_registry(process_root)
    records = registry.get("records", [])
    existing = _find_existing_record(records, image_slug, param_hash)
    reused = False
    if existing is not None and os.path.isdir(str(existing.get("record_dir", ""))):
        record_id = str(existing["record_id"])
        record_dir = str(existing["record_dir"])
        existing["last_run_at"] = now_str
        existing["run_count"] = int(existing.get("run_count", 1)) + 1
        reused = True
    else:
        record_id = _next_record_id(records, image_slug)
        record_dir = os.path.join(process_root, record_id)
        records.append(
            {
                "record_id": record_id,
                "record_dir": record_dir,
                "image_name": str(image_name or ""),
                "image_slug": image_slug,
                "param_hash": param_hash,
                "created_at": now_str,
                "last_run_at": now_str,
                "run_count": 1,
            }
        )

    os.makedirs(record_dir, exist_ok=True)
    if lite_mode:
        ordered_dir = record_dir
        debug_raw_dir = None
        legacy_ordered_dir = os.path.join(record_dir, "ordered")
        legacy_debug_raw_dir = os.path.join(record_dir, "debug_raw")
        if os.path.isdir(legacy_ordered_dir):
            shutil.rmtree(legacy_ordered_dir, ignore_errors=True)
        if os.path.isdir(legacy_debug_raw_dir):
            shutil.rmtree(legacy_debug_raw_dir, ignore_errors=True)

        ordered_file_pattern = re.compile(r"^\d{2}_\d{2}_.+")
        for fname in os.listdir(record_dir):
            if (
                ("record_manifest" in fname)
                and fname.lower().endswith((".json", ".txt"))
            ):
                if fname != lite_manifest_filename:
                    file_path = os.path.join(record_dir, fname)
                    if os.path.isfile(file_path):
                        try:
                            os.remove(file_path)
                        except Exception:
                            pass
                continue
            if not ordered_file_pattern.match(fname):
                continue
            file_path = os.path.join(record_dir, fname)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass
    else:
        ordered_dir = os.path.join(record_dir, "ordered")
        debug_raw_dir = os.path.join(record_dir, "debug_raw")
        _reset_dir(ordered_dir)
        _reset_dir(debug_raw_dir)

    group_idx = 1
    group_item_counts = {}

    def _next_group():
        nonlocal group_idx
        gid = int(group_idx)
        group_idx += 1
        return gid

    def _next_item_in_group(group_id):
        curr = int(group_item_counts.get(group_id, 0)) + 1
        group_item_counts[group_id] = curr
        return curr

    def _build_ordered_name(group_id, item_id, label, ext):
        raw_label = str(label or "").strip()
        if not raw_label:
            raw_label = "item"
        # 允许中文等可读标签，仅替换文件系统不安全字符。
        safe_label = re.sub(r"[\\/:*?\"<>|\r\n\t]+", "_", raw_label)
        safe_label = re.sub(r"\s+", "_", safe_label).strip(" ._")
        if not safe_label:
            safe_label = "item"
        if lite_mode:
            # Lite 下文件名追加图片名，方便脱离目录后仍可追踪来源。
            return f"{int(group_id):02d}_{int(item_id):02d}_{safe_label}__{image_slug}{ext}"
        return f"{int(group_id):02d}_{int(item_id):02d}_{safe_label}{ext}"

    def _save_generated(label, pil_image, group_id, ext=".png"):
        if pil_image is None:
            return None
        item_id = _next_item_in_group(group_id)
        fname = _build_ordered_name(group_id, item_id, label, ext)
        out_path = os.path.join(ordered_dir, fname)
        pil_image.save(out_path)
        return out_path

    def _save_from_file(label, src_path, group_id):
        if not src_path or (not os.path.isfile(src_path)):
            return None
        ext = os.path.splitext(src_path)[1] or ".png"
        item_id = _next_item_in_group(group_id)
        fname = _build_ordered_name(group_id, item_id, label, ext)
        out_path = os.path.join(ordered_dir, fname)
        shutil.copy2(src_path, out_path)
        return out_path

    def _save_split_tiles_from_file(label_prefix, src_path, group_id, force_rows=None, force_cols=None):
        if not src_path or (not os.path.isfile(src_path)):
            return 0
        try:
            with Image.open(src_path) as src_img:
                w, h = src_img.size
                if force_rows is not None and force_cols is not None:
                    rows = int(force_rows)
                    cols = int(force_cols)
                    if rows <= 0 or cols <= 0 or (w % cols) != 0 or (h % rows) != 0:
                        return 0
                    tile_w = w // cols
                    tile_h = h // rows
                else:
                    inferred = _infer_regular_grid_for_image(src_path, base_size=512)
                    if inferred is None:
                        return 0
                    rows, cols, tile_w, tile_h = inferred

                tile_count = 0
                ext = os.path.splitext(src_path)[1] or ".png"
                for r in range(rows):
                    for c in range(cols):
                        left = c * tile_w
                        top = r * tile_h
                        tile = src_img.crop((left, top, left + tile_w, top + tile_h))
                        tile_count += 1
                        tile_label = f"{label_prefix}_{tile_count:02d}"
                        if _is_excluded_split_tile(tile_label.lower()):
                            continue
                        item_id = _next_item_in_group(group_id)
                        fname = _build_ordered_name(group_id, item_id, tile_label, ext)
                        out_path = os.path.join(ordered_dir, fname)
                        tile.save(out_path)
                return tile_count
        except Exception:
            return 0

    def _save_native_panel_files(label_prefix, src_path, group_id):
        if not src_path or (not os.path.isfile(src_path)):
            return 0
        src_dir = os.path.dirname(src_path)
        stem = os.path.splitext(os.path.basename(src_path))[0]
        def _semantic_match_key(name_lower):
            key = re.sub(r"^\d+_", "", str(name_lower or "").strip().lower())
            # 忽略语义名中的中间两位序号（如 3DRigid_04_fill_scopes vs 3DRigid_01_fill_scopes）。
            key = re.sub(r"(?<=_)\d{2}(?=_)", "", key)
            key = re.sub(r"_+", "_", key).strip("_")
            return key

        target_key = _semantic_match_key(stem)

        panel_hits = []
        try:
            for fname in os.listdir(src_dir):
                lower = fname.lower()
                if not lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    continue
                m = re.search(r"__panel_(\d+)", lower)
                if m is None:
                    continue
                panel_root = lower[:m.start()]
                if _semantic_match_key(panel_root) != target_key:
                    continue
                panel_hits.append((int(m.group(1)), os.path.join(src_dir, fname)))
        except Exception:
            return 0
        if not panel_hits:
            return 0
        panel_hits.sort(key=lambda x: (int(x[0]), str(x[1])))
        saved_count = 0
        for panel_idx, panel_path in panel_hits:
            ext = os.path.splitext(panel_path)[1] or ".png"
            tile_label = f"{label_prefix}_{int(panel_idx):02d}"
            if _is_excluded_split_tile(tile_label.lower()):
                continue
            item_id = _next_item_in_group(group_id)
            fname = _build_ordered_name(group_id, item_id, tile_label, ext)
            out_path = os.path.join(ordered_dir, fname)
            shutil.copy2(panel_path, out_path)
            saved_count += 1
        return saved_count

    src_pil = _to_pil_image(source_image_np)
    res_pil = _to_pil_image(result_img)
    layout_pil = _to_pil_image(layout_img)
    mask_vis = Image.fromarray(mask_u8)
    user_overlay_np = _render_mask_overlay_with_guides(source_image_np, mask_u8, drag_type_for_vis)
    user_overlay = Image.fromarray(user_overlay_np)
    points_overlay = Image.fromarray(_draw_drag_points_overlay(np.asarray(user_overlay), drag_points))

    input_group = _next_group()
    _save_generated("原始图片", src_pil, group_id=input_group)
    if lite_mode:
        # Lite 仅保留 01_01 和 01_04，预留中间编号。
        _next_item_in_group(input_group)
        _next_item_in_group(input_group)
    else:
        _save_generated("User掩码", mask_vis, group_id=input_group)
        _save_generated("User掩码叠加原图", user_overlay, group_id=input_group)
    _save_generated("叠加拖拽点", points_overlay, group_id=input_group)

    mask_dir = os.path.join(debug_root, "mask_process")
    drag_dir = os.path.join(debug_root, "drag_process")
    final_dir = os.path.join(debug_root, "final_results")
    saved_source_paths = set()

    def _mark_saved_source(src_path):
        if src_path and os.path.isfile(src_path):
            saved_source_paths.add(os.path.abspath(src_path))

    def _label_from_debug_path(src_path):
        rel_path = os.path.relpath(src_path, debug_root)
        rel_no_ext = os.path.splitext(rel_path)[0]
        normalized = rel_no_ext.replace("\\", "/").replace("/", "__")
        return _sanitize_path_component(normalized, default_value="debug")

    def _is_excluded_debug_image(lower_name):
        if any(k in lower_name for k in PAPER_EXCLUDED_MASK_KEYWORDS):
            return True
        for patt in PAPER_EXCLUDED_DEBUG_REGEXES:
            if re.search(patt, lower_name):
                return True
        return False

    def _is_excluded_split_tile(lower_label):
        for patt in PAPER_EXCLUDED_SPLIT_TILE_REGEXES:
            if re.search(patt, lower_label):
                return True
        return False

    def _force_single_debug_image(lower_name):
        for patt in PAPER_FORCE_SINGLE_DEBUG_REGEXES:
            if re.search(patt, lower_name):
                return True
        return False

    def _iter_image_files(base_dir):
        if not os.path.isdir(base_dir):
            return []
        items = []
        for root, _, files in os.walk(base_dir):
            for fname in files:
                lower = fname.lower()
                if not lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    continue
                if _is_excluded_debug_image(lower):
                    continue
                items.append(os.path.join(root, fname))
        items.sort()
        return items

    def _save_debug_image_bundle(src_path):
        if not src_path or (not os.path.isfile(src_path)):
            return
        src_abs = os.path.abspath(src_path)
        if src_abs in saved_source_paths:
            return
        label = _label_from_debug_path(src_abs)
        name_lower = os.path.basename(src_abs).lower()
        bundle_group = _next_group()
        grid = None if _force_single_debug_image(name_lower) else _infer_regular_grid_for_image(src_abs, base_size=512)
        if grid is not None:
            rows, cols, _, _ = grid
            if _save_from_file(f"拼图_{label}", src_abs, group_id=bundle_group):
                _mark_saved_source(src_abs)
            native_saved = _save_native_panel_files(f"子图_{label}", src_abs, group_id=bundle_group)
            if native_saved <= 0:
                _save_split_tiles_from_file(
                    f"子图_{label}",
                    src_abs,
                    group_id=bundle_group,
                    force_rows=rows,
                    force_cols=cols,
                )
        else:
            if _save_from_file(f"单图_{label}", src_abs, group_id=bundle_group):
                _mark_saved_source(src_abs)

    def _find_all_component_mask_files(base_dir):
        if not os.path.isdir(base_dir):
            return []
        items = []
        for root, _, files in os.walk(base_dir):
            for fname in files:
                lower = fname.lower()
                if not lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    continue
                if "comp" not in lower or "mask" not in lower:
                    continue
                # 过滤分析/汇总可视化图（带文字/标题），但保留真实 component mask（如 sam_comp01_mask.jpg）。
                if (
                    ("summary" in lower)
                    or ("analysis" in lower)
                    or ("overview" in lower)
                    or ("subject" in lower)
                ) and ("final_selected_mask" not in lower):
                    continue
                if any(k in lower for k in PAPER_EXCLUDED_MASK_KEYWORDS):
                    continue
                items.append(os.path.join(root, fname))
        items.sort()
        return items

    component_mask_paths = _find_all_component_mask_files(mask_dir)
    summary_mask_path = (
        _find_first_file_with_keywords(mask_dir, ["sam_summary_mask"])
        or _find_first_file_with_keywords(mask_dir, ["summary", "mask"])
    )
    final_selected_mask_path = _find_first_file_with_keywords(mask_dir, ["final_selected_mask"])
    raw_mask_path = _find_first_file_with_keywords(mask_dir, ["00_mask"])
    subject_overlay_path = (
        _find_first_file_with_keywords(mask_dir, ["summary", "subject"])
        or _find_first_file_with_keywords(mask_dir, ["summary", "overview"])
        or _find_first_file_with_keywords(mask_dir, ["comp01", "subject"])
    )
    mask_stage_group = _next_group()

    def _all_components_mask_u8(mask_like):
        mask01 = (np.asarray(mask_like) > 0).astype(np.uint8)
        if mask01.ndim != 2:
            return None
        if int(np.count_nonzero(mask01)) == 0:
            return None
        return (mask01 * 255).astype(np.uint8)

    def _extract_subject_mask_from_subject_preview(path):
        if not path or (not os.path.isfile(path)):
            return None
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        fg = (gray > 8).astype(np.uint8)
        return _all_components_mask_u8(fg)

    def _merge_masks_from_files(mask_paths):
        if not isinstance(mask_paths, list) or len(mask_paths) == 0:
            return None
        merged = np.zeros(source_image_np.shape[:2], dtype=np.uint8)
        for fp in mask_paths:
            try:
                gray = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                m_u8 = _normalize_mask_u8(gray, source_image_np.shape[:2])
                merged = np.maximum(merged, (m_u8 > 0).astype(np.uint8) * 255)
            except Exception:
                continue
        return _all_components_mask_u8(merged)

    def _sanitize_subject_mask_u8(mask_like, user_mask_like=None):
        """
        清理“主体mask”里的可视化噪点（典型是左上角标题文字）：
        1) 移除小连通域；
        2) 若存在 user mask，则优先保留与 user mask 重叠的连通域。
        """
        m_u8 = _normalize_mask_u8(mask_like, source_image_np.shape[:2])
        m_bin = (m_u8 > 0).astype(np.uint8)
        if int(np.count_nonzero(m_bin)) == 0:
            return None

        h, w = m_bin.shape[:2]
        min_area = max(24, int(round(h * w * 1.5e-4)))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m_bin, connectivity=8)
        if num_labels <= 1:
            return _all_components_mask_u8(m_bin)

        user_bin = None
        if user_mask_like is not None:
            user_u8 = _normalize_mask_u8(user_mask_like, source_image_np.shape[:2])
            user_bin = (user_u8 > 0).astype(np.uint8)
        has_user_constraint = user_bin is not None and int(np.count_nonzero(user_bin)) > 0

        keep = np.zeros_like(m_bin, dtype=np.uint8)
        for lid in range(1, num_labels):
            area = int(stats[lid, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            comp = (labels == lid).astype(np.uint8)
            if has_user_constraint:
                overlap = int(np.count_nonzero(np.logical_and(comp > 0, user_bin > 0)))
                if overlap <= 0:
                    continue
            keep[comp > 0] = 1

        # 兜底：若过滤后为空，保留最大连通域，避免整张图被清空。
        if int(np.count_nonzero(keep)) == 0:
            largest_id = None
            largest_area = -1
            for lid in range(1, num_labels):
                area = int(stats[lid, cv2.CC_STAT_AREA])
                if area > largest_area:
                    largest_area = area
                    largest_id = lid
            if largest_id is not None:
                keep[labels == largest_id] = 1

        return _all_components_mask_u8(keep)

    # 主体掩码优先级（与你前端显示保持一致）：
    # 1) 前端 display_mask_state（即你在 preview 里看到的主体掩码）
    # 2) 03_sam_summary_mask（仅作为后备）
    # 3) final_selected_mask / 00_mask（最后的文件回退）
    # 4) comp masks 合并 / subject 可视化图提取
    # 5) user mask（最终兜底）
    subject_mask_u8 = _all_components_mask_u8(display_mask_u8)
    subject_mask_u8 = _sanitize_subject_mask_u8(subject_mask_u8, user_mask_like=mask_u8)

    if subject_mask_u8 is None:
        for subject_mask_path in [summary_mask_path, final_selected_mask_path, raw_mask_path]:
            if not subject_mask_path or (not os.path.isfile(subject_mask_path)):
                continue
            try:
                subject_mask_gray = cv2.imread(subject_mask_path, cv2.IMREAD_GRAYSCALE)
                if subject_mask_gray is None:
                    continue
                candidate = _normalize_mask_u8(subject_mask_gray, source_image_np.shape[:2])
                candidate = _all_components_mask_u8(candidate)
                candidate = _sanitize_subject_mask_u8(candidate, user_mask_like=mask_u8)
                if candidate is not None:
                    subject_mask_u8 = candidate
                    break
            except Exception:
                continue

    if subject_mask_u8 is None and len(component_mask_paths) > 0:
        subject_mask_u8 = _merge_masks_from_files(component_mask_paths)
        subject_mask_u8 = _sanitize_subject_mask_u8(subject_mask_u8, user_mask_like=mask_u8)

    if subject_mask_u8 is None:
        subject_mask_u8 = _extract_subject_mask_from_subject_preview(subject_overlay_path)
        subject_mask_u8 = _sanitize_subject_mask_u8(subject_mask_u8, user_mask_like=mask_u8)

    if subject_mask_u8 is None:
        subject_mask_u8 = _all_components_mask_u8(mask_u8)
        subject_mask_u8 = _sanitize_subject_mask_u8(subject_mask_u8, user_mask_like=mask_u8)

    if subject_mask_u8 is not None:
        if lite_mode:
            # Lite 仅保留 02_02 和 02_03，预留 02_01。
            _next_item_in_group(mask_stage_group)
        else:
            _save_generated(
                "主体掩码",
                Image.fromarray(subject_mask_u8),
                group_id=mask_stage_group,
            )
        subject_overlay = _render_mask_overlay_with_guides(source_image_np, subject_mask_u8, drag_type_for_vis)
        _save_generated(
            "主体掩码叠加原图",
            Image.fromarray(subject_overlay),
            group_id=mask_stage_group,
        )
        subject_with_points = _draw_drag_points_overlay(subject_overlay, drag_points)
        _save_generated(
            "主体掩码叠加原图+拖拽点",
            Image.fromarray(subject_with_points),
            group_id=mask_stage_group,
        )
    elif subject_overlay_path and os.path.isfile(subject_overlay_path):
        # 兜底：如果没找到主体mask，再退回旧可视化图，避免流程中断。
        try:
            subject_bgr = cv2.imread(subject_overlay_path, cv2.IMREAD_COLOR)
            if subject_bgr is not None:
                subject_rgb = cv2.cvtColor(subject_bgr, cv2.COLOR_BGR2RGB)
                subject_mask_fallback_u8 = _extract_subject_mask_from_subject_preview(subject_overlay_path)
                subject_points_base = subject_rgb
                if subject_mask_fallback_u8 is not None:
                    if lite_mode:
                        _next_item_in_group(mask_stage_group)
                    else:
                        _save_generated(
                            "主体掩码",
                            Image.fromarray(subject_mask_fallback_u8),
                            group_id=mask_stage_group,
                        )
                    subject_overlay_base = _render_mask_overlay_with_guides(
                        source_image_np,
                        subject_mask_fallback_u8,
                        drag_type_for_vis,
                    )
                    _save_generated(
                        "主体掩码叠加原图",
                        Image.fromarray(subject_overlay_base),
                        group_id=mask_stage_group,
                    )
                    subject_points_base = subject_overlay_base
                elif lite_mode:
                    _next_item_in_group(mask_stage_group)
                    _save_generated(
                        "主体掩码叠加原图",
                        Image.fromarray(subject_rgb),
                        group_id=mask_stage_group,
                    )
                subject_with_points = _draw_drag_points_overlay(subject_points_base, drag_points)
                _save_generated(
                    "主体掩码叠加原图+拖拽点",
                    Image.fromarray(subject_with_points),
                    group_id=mask_stage_group,
                )
                _mark_saved_source(subject_overlay_path)
            else:
                if lite_mode:
                    _next_item_in_group(mask_stage_group)
                    if _save_from_file("主体掩码叠加原图", subject_overlay_path, group_id=mask_stage_group):
                        _mark_saved_source(subject_overlay_path)
                if _save_from_file("主体掩码叠加原图+拖拽点", subject_overlay_path, group_id=mask_stage_group):
                    _mark_saved_source(subject_overlay_path)
        except Exception:
            if lite_mode:
                _next_item_in_group(mask_stage_group)
                if _save_from_file("主体掩码叠加原图", subject_overlay_path, group_id=mask_stage_group):
                    _mark_saved_source(subject_overlay_path)
            if _save_from_file("主体掩码叠加原图+拖拽点", subject_overlay_path, group_id=mask_stage_group):
                _mark_saved_source(subject_overlay_path)

    if not lite_mode:
        bg_fill_path = os.path.join(mask_dir, "05_background_filled_rgb.jpg")
        if _save_from_file("背景填充图", bg_fill_path, group_id=mask_stage_group):
            _mark_saved_source(bg_fill_path)
        drag_comp_path = os.path.join(mask_dir, "05_final_composed_rgb.jpg")
        if _save_from_file("拖拽合成图", drag_comp_path, group_id=mask_stage_group):
            _mark_saved_source(drag_comp_path)

        for base_dir in (mask_dir, drag_dir):
            for fp in _iter_image_files(base_dir):
                _save_debug_image_bundle(fp)

    final_branch_paths = {}
    if isinstance(debug_artifacts, dict):
        final_branch_paths = debug_artifacts.get("final_branch_paths", {}) or {}
    branch_order = [
        ("source", "五分支_Source"),
        ("mutual", "五分支_Mutual"),
        ("reference", "五分支_Reference"),
        ("layout", "五分支_Layout"),
        ("target", "五分支_Target"),
    ]
    branch_group = None
    for key, label in branch_order:
        branch_path = final_branch_paths.get(key)
        if branch_path and os.path.isfile(branch_path):
            if branch_group is None:
                branch_group = _next_group()
            _save_from_file(label, branch_path, group_id=branch_group)

    if not lite_mode:
        output_group = _next_group()
        if res_pil is not None:
            _save_generated("最终结果图", res_pil, group_id=output_group)
        if layout_pil is not None:
            _save_generated("Layout分支图", layout_pil, group_id=output_group)
        _save_from_file("拼接对比图", os.path.join(final_dir, "concat.png"), group_id=output_group)

    if not lite_mode:
        for sub in PAPER_DEBUG_SUBDIRS:
            src_sub = os.path.join(debug_root, sub)
            dst_sub = os.path.join(debug_raw_dir, sub)
            if os.path.isdir(src_sub):
                shutil.copytree(src_sub, dst_sub, dirs_exist_ok=True)
        mask_raw_dir = os.path.join(debug_raw_dir, "mask_process")
        if os.path.isdir(mask_raw_dir):
            for fname in os.listdir(mask_raw_dir):
                lower = fname.lower()
                if _is_excluded_debug_image(lower):
                    file_path = os.path.join(mask_raw_dir, fname)
                    if os.path.isfile(file_path):
                        try:
                            os.remove(file_path)
                        except Exception:
                            pass

    debug_summary = {
        "has_final_branches": bool(final_branch_paths),
        "final_branch_count": int(len(final_branch_paths)),
    }

    debug_raw_dir_abs = os.path.abspath(debug_raw_dir) if debug_raw_dir else None

    manifest = {
        "record_id": record_id,
        "record_dir": os.path.abspath(record_dir),
        "created_at": now_str,
        "reused_record": reused,
        "param_hash": param_hash,
        "image_name": str(image_name or ""),
        "image_slug": image_slug,
        "source_prompt": str(source_prompt_text or ""),
        "target_prompt": str(target_prompt_text or ""),
        "signature": signature,
        "result_folders": {
            "record_dir": os.path.abspath(record_dir),
            "ordered_dir": os.path.abspath(ordered_dir),
            "debug_raw_dir": debug_raw_dir_abs,
        },
        "debug_root": os.path.abspath(debug_root),
        "debug_raw_dir": debug_raw_dir_abs,
        "debug_artifacts": debug_summary,
    }
    os.makedirs(record_dir, exist_ok=True)
    if lite_mode:
        manifest_path = os.path.join(record_dir, lite_manifest_filename)
        for legacy_name in ("record_manifest.json", "record_manifest.txt", f"{image_slug}__record_manifest.json"):
            legacy_path = os.path.join(record_dir, legacy_name)
            if (legacy_path != manifest_path) and os.path.isfile(legacy_path):
                try:
                    os.remove(legacy_path)
                except Exception:
                    pass
        lite_lines = [
            str(source_prompt_text or ""),
            str(target_prompt_text or ""),
            str(drag_type_for_vis or ""),
        ]
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lite_lines) + "\n")
    else:
        manifest_path = os.path.join(record_dir, "record_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    _save_paper_registry(registry_path, registry)
    return {
        "record_id": record_id,
        "record_dir": os.path.abspath(record_dir),
        "manifest_path": os.path.abspath(manifest_path),
        "reused_record": reused,
        "param_hash": param_hash,
    }

# ==========================================
# 4. 核心逻辑函数 (可视化、点位、IO)
# ==========================================

def _get_component_min_area(mask_shape):
    h, w = int(mask_shape[0]), int(mask_shape[1])
    return max(COMPONENT_MIN_AREA_BASE, int(round(h * w * COMPONENT_MIN_AREA_RATIO)))


def _resolve_component_min_area(mask_input, min_area=None):
    """
    自适应连通域阈值：
    - 基础阈值：与图像尺寸相关
    - 相对阈值：最大连通域面积的一定比例
    """
    mask_u8 = (np.asarray(mask_input) > 0).astype(np.uint8)
    if mask_u8.ndim != 2:
        return int(min_area) if min_area is not None else COMPONENT_MIN_AREA_BASE

    base_min = int(min_area) if min_area is not None else _get_component_min_area(mask_u8.shape)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return base_min

    largest_area = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
    rel_min = int(round(largest_area * COMPONENT_MIN_AREA_RELATIVE_TO_LARGEST))
    return max(base_min, rel_min)


def _remove_small_components(mask_input, min_area=None, keep_largest_if_empty=False):
    """
    过滤过小连通域，返回 uint8(H, W) in {0,1}.
    """
    if mask_input is None:
        return None

    mask_u8 = (np.asarray(mask_input) > 0).astype(np.uint8)
    if mask_u8.ndim != 2:
        return mask_u8

    min_area = _resolve_component_min_area(mask_u8, min_area=min_area)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8

    out = np.zeros_like(mask_u8, dtype=np.uint8)
    largest_label = 0
    largest_area = 0
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area > largest_area:
            largest_area = area
            largest_label = label_id
        if area >= min_area:
            out[labels == label_id] = 1

    if keep_largest_if_empty and int(np.count_nonzero(out)) == 0 and largest_label > 0:
        out[labels == largest_label] = 1
    return out

# --- 4.2 核心渲染引擎 ---
def draw_visual_guides(image, mask, drag_type):
    """
    【核心渲染函数 - 升级版】
    1. 叠加 Mask (背景变暗)
    2. 如果是 Rigid，支持多连通域显示 (显示多个质心和圈)
    """
    if image is None: return None
    vis_img = image.copy()
    
    if mask is None or mask.sum() == 0:
        return vis_img

    # 1. 过滤过小连通域 + 叠加 Mask (Mask区域高亮，背景变暗)
    mask_float = mask if mask.max() <= 1.0 else mask / 255.0
    min_area = _resolve_component_min_area(
        mask_float, min_area=_get_component_min_area(mask_float.shape)
    )
    mask_binary = _remove_small_components(mask_float, min_area=min_area, keep_largest_if_empty=False)
    if int(np.count_nonzero(mask_binary)) == 0:
        return vis_img
    vis_img = apply_mask_overlay(vis_img, mask_binary.astype(np.float32))

    # 2. 仅 Rigid 模式下绘制辅助线；Hybrid/Non-Rigid 不绘制圈与质心
    drag_type_norm = str(drag_type or "").strip()
    if drag_type_norm in GUIDE_VIS_DRAG_TYPES:
        mask_u8 = (mask_binary * 255).astype(np.uint8)
        # 使用连通域分析：不同 user mask 连通域独立显示。
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

        # 遍历每个连通域 (Label 0 是背景，跳过)
        for i in range(1, num_labels):
            # 获取当前区域的质心
            cx, cy = int(centroids[i][0]), int(centroids[i][1])

            # 获取包围盒信息 (用于计算圈的大小)
            x, y, w, h, area = stats[i]

            # 过滤太小的噪点区域
            if area < min_area:
                continue

            # 计算显示半径
            radius = (w + h) / 4.0
            # 与后端 detect_rigid_intent 严格一致：
            # backend radius=((w+h)/4)*(4/9)，阈值 relative_dist<0.4
            # => 判定圈半径系数 = 0.4*(4/9) = 8/45
            zone_radius = max(1, int(radius * (8.0 / 45.0)))

            # 绘制视觉引导
            cv2.circle(vis_img, (cx, cy), zone_radius, (0, 255, 0), 2)
            # 画黄点 (Centroid)
            cv2.circle(vis_img, (cx, cy), 6, (255, 255, 0), -1)
            cv2.circle(vis_img, (cx, cy), 7, (0, 0, 0), 1)

    return vis_img
# --- 4.3 交互回调函数 ---

def extract_canvas_state(canvas):
    """从画布提取背景图和用户 mask。"""
    if canvas is None:
        return None, None
    img = canvas.get("background")
    if img is None:
        return None, None

    img_array = img if img.shape[-1] == 3 else img[..., :3]
    img_array = img_array.copy()

    layers = canvas.get("layers") or []
    if len(layers) > 0:
        mask = np.float32(layers[0][:, :, 3]) / 255.0
    else:
        mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.float32)
    mask = (mask > 0).astype(np.uint8)
    return img_array, mask


def refine_user_mask_with_sam(image_np, user_mask_np):
    """
    使用 SAM 对用户涂抹 mask 做前端显示精炼（只用于显示，不改变推理输入 mask）。
    """
    if image_np is None or user_mask_np is None:
        return user_mask_np

    user_mask_u8 = (user_mask_np > 0).astype(np.uint8) * 255
    if np.count_nonzero(user_mask_u8) < 10:
        return (user_mask_u8 > 0).astype(np.uint8)

    min_area = _resolve_component_min_area(
        user_mask_u8, min_area=_get_component_min_area(user_mask_u8.shape)
    )
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(user_mask_u8, connectivity=8)
    all_seed_points = []
    all_hint_masks = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp_hint = (labels == label_id).astype(np.uint8) * 255
        cx = int(round(float(centroids[label_id][0])))
        cy = int(round(float(centroids[label_id][1])))
        h, w = comp_hint.shape[:2]
        cx = int(np.clip(cx, 0, w - 1))
        cy = int(np.clip(cy, 0, h - 1))
        if comp_hint[cy, cx] == 0:
            ys, xs = np.where(comp_hint > 0)
            if len(xs) == 0:
                continue
            mid = len(xs) // 2
            cx, cy = int(xs[mid]), int(ys[mid])
        all_seed_points.append(np.array([[cx, cy]], dtype=np.int32))
        all_hint_masks.append(comp_hint)

    if len(all_seed_points) == 0:
        filtered_user = _remove_small_components(
            user_mask_u8, min_area=min_area, keep_largest_if_empty=True
        )
        return filtered_user.astype(np.uint8)

    sam_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        preload_sam2(device=sam_device)
        comp_sam_masks = get_interactive_masks_batch(
            image=image_np,
            all_handle_points=all_seed_points,
            device=sam_device,
            all_user_hint_masks=all_hint_masks,
            debug_dir=None,
            enable_debug=False,
        )
    except Exception as exc:
        print(f"[refine_user_mask_with_sam] fallback to user mask: {exc}")
        return (user_mask_u8 > 0).astype(np.uint8)

    subject_mask_u8 = np.zeros_like(user_mask_u8)
    for comp_idx, sam_mask in enumerate(comp_sam_masks):
        if sam_mask is None:
            continue
        sam_u8 = sam_mask.astype(np.uint8)
        if sam_u8.max() <= 1:
            sam_u8 = sam_u8 * 255
        sam_u8 = (sam_u8 > 127).astype(np.uint8) * 255
        hint_u8 = all_hint_masks[comp_idx]
        subject_u8 = ((sam_u8 > 0) & (hint_u8 > 0)).astype(np.uint8) * 255
        if FORCE_SINGLE_COMPONENT_PER_MASK:
            # 约束：单个 user 连通域内部最多一个主体；
            # 若 SAM 切成多块，优先保留包含 seed 的连通域，其次保留最大连通域。
            sub_bin = (subject_u8 > 0).astype(np.uint8)
            n_sub, sub_labels, sub_stats, _ = cv2.connectedComponentsWithStats(sub_bin, connectivity=8)
            if n_sub > 2:
                keep_label = 0
                try:
                    seed = all_seed_points[comp_idx]
                    sx = int(round(float(seed[0][0])))
                    sy = int(round(float(seed[0][1])))
                    h_sub, w_sub = sub_bin.shape[:2]
                    sx = int(np.clip(sx, 0, w_sub - 1))
                    sy = int(np.clip(sy, 0, h_sub - 1))
                    keep_label = int(sub_labels[sy, sx])
                except Exception:
                    keep_label = 0

                if keep_label <= 0:
                    largest_area = -1
                    largest_label = 0
                    for lid in range(1, n_sub):
                        area = int(sub_stats[lid, cv2.CC_STAT_AREA])
                        if area > largest_area:
                            largest_area = area
                            largest_label = lid
                    keep_label = largest_label

                if keep_label > 0:
                    kept = np.zeros_like(sub_bin, dtype=np.uint8)
                    kept[sub_labels == keep_label] = 1
                    subject_u8 = (kept * 255).astype(np.uint8)
                else:
                    subject_u8 = hint_u8.copy()
        subject_mask_u8 = np.maximum(subject_mask_u8, subject_u8)

    filtered_subject = _remove_small_components(
        subject_mask_u8, min_area=min_area, keep_largest_if_empty=True
    )
    return filtered_subject.astype(np.uint8)


def _resolve_display_mask(image_np, user_mask, show_original_mask):
    if user_mask is None:
        return None
    if bool(show_original_mask):
        return (user_mask > 0).astype(np.uint8)
    return refine_user_mask_with_sam(image_np, user_mask)


def update_mask_and_visualize(
    canvas,
    drag_type_value,
    source_image_state,
    drag_points_state,
    show_original_mask,
    inference_mask_mode,
):
    """[按钮回调] 确认 Mask -> 更新可视化。"""
    canvas_img, user_mask = extract_canvas_state(canvas)

    # 优先使用已有原图，避免被画布污染
    img_array = source_image_state if source_image_state is not None else (canvas_img.copy() if canvas_img is not None else None)
    if img_array is None:
        return None, gr.update(value=None), None, None, []

    mode_is_user_mask = _normalize_inference_mask_mode(inference_mask_mode) == "User Mask"
    effective_show_original_mask = bool(show_original_mask) or mode_is_user_mask
    display_mask = _resolve_display_mask(img_array, user_mask, effective_show_original_mask)
    vis_img = draw_visual_guides(img_array, display_mask, drag_type_value)

    current_points = drag_points_state if isinstance(drag_points_state, list) else []
    if len(current_points) > 0:
        vis_img = redraw_points_on_load(vis_img, current_points)

    return (
        img_array,
        gr.update(value=vis_img, height=img_array.shape[0]),
        display_mask,
        user_mask,
        list(current_points)
    )

def reset_to_base_state(original_image, mask, drag_type_value):
    """[按钮回调] 撤销/重置 -> 调用核心渲染"""
    vis_img = draw_visual_guides(original_image, mask, drag_type_value)
    return vis_img, []

def get_points(img, sel_pix, evt: gr.SelectData):
    """
    处理点击事件，维护点位列表 (纯点对模式，无第0点)
    """
    if img is None:
        return None
    # 始终在拷贝上绘制，避免污染原图/状态
    img_np = img if isinstance(img, np.ndarray) else np.array(img)
    vis_img = img_np.copy()

    sel_pix.append(evt.index)
    points = []

    for idx, point in enumerate(sel_pix):
        if idx % 2 == 0:
            # Start Point (Red)
            cv2.circle(vis_img, tuple(point), 8, (255, 0, 0), -1)
            points.append(tuple(point))
        else:
            # End Point (Blue)
            cv2.circle(vis_img, tuple(point), 8, (0, 0, 255), -1)
            points.append(tuple(point))
        
        if len(points) == 2:
            cv2.arrowedLine(vis_img, points[0], points[1], (255, 255, 255), 2, tipLength=0.1)
            points = []

    return vis_img

def safe_get_points(img, sel_pix, evt: gr.SelectData):
    if img is None: return None
    x, y = evt.index
    h, w = img.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h: return img
    return get_points(img, sel_pix, evt)

def redraw_points_on_load(img, points):
    """加载历史数据时重绘点"""
    vis_img = img.copy()
    class MockEvent:
        def __init__(self, x, y): self.index = [x, y]
    temp_pts = []
    for pt in points:
        vis_img = get_points(vis_img, temp_pts, MockEvent(pt[0], pt[1]))
    return vis_img

# --- 4.4 数据存取函数 ---

# 1. 函数定义去掉 use_origin_point
def load_annotation_data(
    dataset_name,
    selected_image_name,
    drag_type_value=DEFAULT_DRAG_TYPE,
    influence_range_value=DEFAULT_INFLUENCE_RANGE,
    edit_mode_value=DEFAULT_EDIT_MODE,
    dataset_view_mode=DEFAULT_DATASET_VIEW,
    show_original_mask=DEFAULT_SHOW_ORIGINAL_MASK,
):
    if not selected_image_name:
        return (
            gr.update(value=None),
            gr.update(value=None),
            None,
            "",
            "",
            gr.update(value=DEFAULT_LOCAL_BLEND_WORD),
            gr.update(value=DEFAULT_MUTUAL_BLEND_WORD),
            gr.update(value=DEFAULT_LOCAL_BLEND_THRESH_E),
            gr.update(value=DEFAULT_LOCAL_BLEND_THRESH_M),
            None,
            None,
            [],
            "",
            gr.update(value=DEFAULT_DRAG_TYPE),
            gr.update(value=DEFAULT_INFLUENCE_RANGE),
            gr.update(value=_default_strength_for_mode(edit_mode_value)),
            gr.update(value=_default_target_guidance_for_mode(edit_mode_value)),
            "未选择文件",
        )
    cfg = DATASET_CONFIG[dataset_name]
    root = cfg["root"]
    raw_img = np.array(Image.open(os.path.join(root, "images", selected_image_name)).convert("RGB"))

    # 保持原始大小，不做 resize
    resized_img = raw_img
    h, w = resized_img.shape[:2]

    source_prompt_text, target_prompt_text, drag_points_xy, saved_mask = "", "", [], None
    mask_path_from_json = ""
    image_id = os.path.splitext(selected_image_name)[0]
    active_view = "source"
    loaded_drag_type = DEFAULT_DRAG_TYPE
    loaded_influence_range = DEFAULT_INFLUENCE_RANGE
    loaded_local_blend_word = DEFAULT_LOCAL_BLEND_WORD
    loaded_mutual_blend_word = DEFAULT_MUTUAL_BLEND_WORD
    loaded_local_blend_thresh_e = DEFAULT_LOCAL_BLEND_THRESH_E
    loaded_local_blend_thresh_m = DEFAULT_LOCAL_BLEND_THRESH_M
    normalized_edit_mode = _normalize_edit_mode(edit_mode_value)
    # 编辑模式不再强制覆盖数据视图；Text 仅清空拖拽点。
    effective_dataset_view_mode = str(dataset_view_mode or DEFAULT_DATASET_VIEW).strip().lower()
    if effective_dataset_view_mode not in DATASET_VIEW_CHOICES:
        effective_dataset_view_mode = DEFAULT_DATASET_VIEW

    found_data = False
    for j_name in cfg["json_files"]:
        if found_data:
            break
        json_path = os.path.join(root, j_name)
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if image_id in data:
                        split_entry = _to_split_annotation_entry(data[image_id])
                        entry, active_view, _, _ = _resolve_annotation_entry_for_view(
                            split_entry,
                            dataset_view_mode=effective_dataset_view_mode,
                        )
                        source_prompt_text = entry.get("source_prompt", "")
                        target_prompt_text = entry.get("target_prompt", "")
                        # 与评估脚本口径对齐：Text 模式回读 blend word。
                        # 若当前视图（常见是 modified）把词保存为空，则回退到 source 词。
                        if normalized_edit_mode == "Text":
                            entry_local_word, entry_mutual_word = _resolve_blend_words_from_entry(entry)
                            source_local_word, source_mutual_word = _resolve_blend_words_from_entry(
                                split_entry.get("source", {})
                            )
                            loaded_local_blend_word = entry_local_word or source_local_word
                            # 对齐 InfEdit PIE text：source blend(mutual) 固定不使用。
                            loaded_mutual_blend_word = DEFAULT_MUTUAL_BLEND_WORD
                        else:
                            loaded_local_blend_word = DEFAULT_LOCAL_BLEND_WORD
                            loaded_mutual_blend_word = DEFAULT_MUTUAL_BLEND_WORD
                        drag_points_xy = entry.get("points", [])
                        mask_path_from_json = entry.get("mask_path", "")
                        loaded_drag_type = _normalize_drag_type(
                            entry.get("drag_type", loaded_drag_type)
                        )
                        loaded_influence_range = _normalize_influence_range(
                            entry.get(
                                "influence_range",
                                loaded_influence_range,
                            )
                        )
                        loaded_local_blend_thresh_e = _normalize_blend_thresh(
                            entry.get("local_blend_thresh_e", loaded_local_blend_thresh_e),
                            DEFAULT_LOCAL_BLEND_THRESH_E,
                        )
                        loaded_local_blend_thresh_m = _normalize_blend_thresh(
                            entry.get("local_blend_thresh_m", loaded_local_blend_thresh_m),
                            DEFAULT_LOCAL_BLEND_THRESH_M,
                        )
                        found_data = True
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                print(f"[load_annotation_data] read {json_path} failed: {exc}")

    if mask_path_from_json:
        m_path = os.path.join(root, mask_path_from_json)
    else:
        m_path = os.path.join(root, "masks", f"{image_id}.png")
    saved_mask = None
    if os.path.exists(m_path):
        saved_mask = (np.array(Image.open(m_path).convert("L")) > 127).astype(np.uint8)

    if not isinstance(drag_points_xy, list):
        drag_points_xy = []
    if normalized_edit_mode == "Text":
        drag_points_xy = []

    user_mask = np.zeros((h, w), dtype=np.uint8)
    if saved_mask is not None:
        user_mask = cv2.resize(saved_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    display_mask = _resolve_display_mask(resized_img, user_mask, show_original_mask)
    
    # === 调用核心渲染 ===
    vis_img = draw_visual_guides(resized_img, display_mask, loaded_drag_type)
    
    if drag_points_xy:
        vis_img = redraw_points_on_load(vis_img, drag_points_xy)

    target_prompt_text, mode_strength_default, mode_target_guidance_default = _apply_mode_defaults(
        edit_mode_value, source_prompt_text, target_prompt_text
    )

    # Drag/Joint 模式保持前端旧策略（不回读 blend word）；Text 模式保留回读值。
    if normalized_edit_mode != "Text":
        loaded_local_blend_word = DEFAULT_LOCAL_BLEND_WORD
        loaded_mutual_blend_word = DEFAULT_MUTUAL_BLEND_WORD

    return (
        gr.update(value=resized_img, height=h+100), 
        gr.update(value=vis_img, height=h), 
        resized_img, source_prompt_text, target_prompt_text,
        gr.update(value=loaded_local_blend_word),
        gr.update(value=loaded_mutual_blend_word),
        gr.update(value=loaded_local_blend_thresh_e),
        gr.update(value=loaded_local_blend_thresh_m),
        display_mask,
        user_mask, drag_points_xy, selected_image_name,
        gr.update(value=loaded_drag_type),
        gr.update(value=loaded_influence_range),
        gr.update(value=mode_strength_default),
        gr.update(value=mode_target_guidance_default),
        f"已加载: {selected_image_name} ({active_view})",
    )

def on_drag_type_change(
    drag_type_value,
    influence_range_value,
    strength_value,
    edit_mode_value,
    source_image_state,
    source_prompt_state,
    target_prompt_state,
    local_blend_word_state,
    mutual_blend_word_state,
    local_blend_thresh_e_state,
    local_blend_thresh_m_state,
    display_mask_state,
    user_mask_state,
    drag_points_state,
    selected_image_name_state,
    suppress_drag_type_change_once_state,
):
    selected_name = selected_image_name_state
    normalized_drag_type = _normalize_drag_type(drag_type_value)
    normalized_range = _normalize_influence_range(influence_range_value)

    # 由图片加载写回 Drag Type 时，会触发一次 drag_type.change。
    # 即使是抑制事件，也执行轻量重绘，避免手工切换时出现“偶发不刷新样式”。
    if suppress_drag_type_change_once_state:
        preview_update = gr.update()
        if source_image_state is not None:
            mask_for_render = (
                display_mask_state
                if display_mask_state is not None
                else (
                    user_mask_state
                    if user_mask_state is not None
                    else np.zeros(source_image_state.shape[:2], dtype=np.uint8)
                )
            )
            vis_img = draw_visual_guides(source_image_state, mask_for_render, normalized_drag_type)
            if isinstance(drag_points_state, list) and len(drag_points_state) > 0:
                vis_img = redraw_points_on_load(vis_img, drag_points_state)
            preview_update = gr.update(value=vis_img, height=source_image_state.shape[0])

        return (
            gr.update(),
            preview_update,
            source_image_state,
            source_prompt_state,
            target_prompt_state,
            gr.update(value=_normalize_blend_word(local_blend_word_state)),
            gr.update(value=_normalize_blend_word(mutual_blend_word_state)),
            gr.update(value=_normalize_blend_thresh(local_blend_thresh_e_state, DEFAULT_LOCAL_BLEND_THRESH_E)),
            gr.update(value=_normalize_blend_thresh(local_blend_thresh_m_state, DEFAULT_LOCAL_BLEND_THRESH_M)),
            display_mask_state,
            user_mask_state,
            drag_points_state if isinstance(drag_points_state, list) else [],
            selected_image_name_state,
            gr.update(value=normalized_drag_type),
            gr.update(value=normalized_range),
            gr.update(value=strength_value),
            gr.update(),
            False,
        )

    if not selected_name:
        return (
            gr.update(value=None),
            gr.update(value=None),
            None,
            "",
            "",
            gr.update(value=DEFAULT_LOCAL_BLEND_WORD),
            gr.update(value=DEFAULT_MUTUAL_BLEND_WORD),
            gr.update(value=DEFAULT_LOCAL_BLEND_THRESH_E),
            gr.update(value=DEFAULT_LOCAL_BLEND_THRESH_M),
            None,
            None,
            [],
            "",
            gr.update(value=normalized_drag_type),
            gr.update(value=normalized_range),
            gr.update(value=_default_strength_for_mode(edit_mode_value)),
            "未选择文件",
            False,
        )

    # 关键：这里不再触发二次 load_annotation_data，避免图片切换后因 drag_type 联动导致重复加载卡顿。
    # 如果图片尚未加载完成，直接保持现状并同步控件值。
    if source_image_state is None:
        return (
            gr.update(),
            gr.update(),
            source_image_state,
            source_prompt_state,
            target_prompt_state,
            gr.update(value=_normalize_blend_word(local_blend_word_state)),
            gr.update(value=_normalize_blend_word(mutual_blend_word_state)),
            gr.update(value=_normalize_blend_thresh(local_blend_thresh_e_state, DEFAULT_LOCAL_BLEND_THRESH_E)),
            gr.update(value=_normalize_blend_thresh(local_blend_thresh_m_state, DEFAULT_LOCAL_BLEND_THRESH_M)),
            display_mask_state,
            user_mask_state,
            drag_points_state if isinstance(drag_points_state, list) else [],
            selected_image_name_state or selected_name,
            gr.update(value=normalized_drag_type),
            gr.update(value=normalized_range),
            gr.update(value=strength_value),
            f"已切换 Drag Type，等待图片加载: {selected_name}",
            False,
        )

    mask_for_render = (
        display_mask_state
        if display_mask_state is not None
        else (
            user_mask_state
            if user_mask_state is not None
            else np.zeros(source_image_state.shape[:2], dtype=np.uint8)
        )
    )
    vis_img = draw_visual_guides(source_image_state, mask_for_render, normalized_drag_type)
    if isinstance(drag_points_state, list) and len(drag_points_state) > 0:
        vis_img = redraw_points_on_load(vis_img, drag_points_state)

    h = source_image_state.shape[0]
    return (
        gr.update(),
        gr.update(value=vis_img, height=h),
        source_image_state,
        source_prompt_state,
        target_prompt_state,
        gr.update(value=_normalize_blend_word(local_blend_word_state)),
        gr.update(value=_normalize_blend_word(mutual_blend_word_state)),
        gr.update(value=_normalize_blend_thresh(local_blend_thresh_e_state, DEFAULT_LOCAL_BLEND_THRESH_E)),
        gr.update(value=_normalize_blend_thresh(local_blend_thresh_m_state, DEFAULT_LOCAL_BLEND_THRESH_M)),
        display_mask_state,
        user_mask_state,
        drag_points_state,
        selected_name,
        gr.update(value=normalized_drag_type),
        gr.update(value=normalized_range),
        gr.update(value=strength_value),
        f"已保留当前编辑（仅切换 Drag Type）: {selected_name}",
        False,
    )


def on_dataset_modify_options_change(
    dataset_name,
    selected_image_name,
    drag_type_value,
    influence_range_value,
    edit_mode_value,
    dataset_view_mode,
    show_original_mask,
):
    loaded = load_annotation_data(
        dataset_name,
        selected_image_name,
        drag_type_value,
        influence_range_value,
        edit_mode_value,
        dataset_view_mode,
        show_original_mask,
    )
    return (*loaded, True)


def on_show_original_mask_change(
    show_original_mask,
    inference_mask_mode,
    source_image_state,
    user_mask_state,
    drag_type_value,
    drag_points_state,
):
    if source_image_state is None:
        return gr.update(), None

    mode_norm = _normalize_inference_mask_mode(inference_mask_mode)
    if user_mask_state is None:
        display_mask = np.zeros(source_image_state.shape[:2], dtype=np.uint8)
    elif mode_norm == "User Mask":
        display_mask = (np.asarray(user_mask_state) > 0).astype(np.uint8)
    else:
        display_mask = _resolve_display_mask(source_image_state, user_mask_state, show_original_mask)

    vis_img = draw_visual_guides(source_image_state, display_mask, drag_type_value)
    if isinstance(drag_points_state, list) and len(drag_points_state) > 0:
        vis_img = redraw_points_on_load(vis_img, drag_points_state)

    return gr.update(value=vis_img, height=source_image_state.shape[0]), display_mask


def on_inference_mask_mode_change(
    inference_mask_mode,
    source_image_state,
    user_mask_state,
    drag_type_value,
    drag_points_state,
):
    if source_image_state is None:
        return gr.update(), None

    mode_norm = _normalize_inference_mask_mode(inference_mask_mode)
    if user_mask_state is None:
        display_mask = np.zeros(source_image_state.shape[:2], dtype=np.uint8)
    elif mode_norm == "User Mask":
        display_mask = (np.asarray(user_mask_state) > 0).astype(np.uint8)
    else:
        display_mask = refine_user_mask_with_sam(source_image_state, user_mask_state)

    vis_img = draw_visual_guides(source_image_state, display_mask, drag_type_value)
    if isinstance(drag_points_state, list) and len(drag_points_state) > 0:
        vis_img = redraw_points_on_load(vis_img, drag_points_state)
    return gr.update(value=vis_img, height=source_image_state.shape[0]), display_mask


def clear_current_annotations(dataset_name, selected_image_name_state, source_image_state, dataset_view_mode):
    if not selected_image_name_state or source_image_state is None: 
        return [None] * 5 + ["未选择图片"]
    
    cfg = DATASET_CONFIG[dataset_name]
    root = cfg["root"]
    image_id = os.path.splitext(selected_image_name_state)[0]

    for j_name in cfg["json_files"]:
        json_path = os.path.join(root, j_name)
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if image_id in data:
                    split_entry = _to_split_annotation_entry(data[image_id])
                    branch_key = "user_study" if str(dataset_view_mode or "").strip().lower() == "user_study" else "modified"
                    branch_entry = split_entry.get(branch_key, {})
                    branch_mask_rel = branch_entry.get("mask_path", "")
                    # 仅删除用户另存到 masks/ 下的掩码，避免误删 source 掩码
                    if isinstance(branch_mask_rel, str) and branch_mask_rel.startswith(f"{MODIFIED_MASK_DIR}/"):
                        branch_mask_path = os.path.join(root, branch_mask_rel)
                        if os.path.exists(branch_mask_path):
                            try:
                                os.remove(branch_mask_path)
                            except OSError as exc:
                                print(f"[clear_current_annotations] remove {branch_mask_path} failed: {exc}")

                    # 只清空当前分支，不触碰 source
                    split_entry[branch_key] = {}
                    data[image_id] = split_entry
                    _dump_json_with_compact_points(json_path, data)
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                print(f"[clear_current_annotations] update {json_path} failed: {exc}")

    h, w = source_image_state.shape[:2]
    canvas_img = source_image_state.copy()
    preview_img = source_image_state.copy()
    return (
        gr.update(value=canvas_img), 
        preview_img, 
        np.zeros((h, w), dtype=np.uint8),
        np.zeros((h, w), dtype=np.uint8), 
        [],
        f"已清空 {('user_study' if str(dataset_view_mode or '').strip().lower() == 'user_study' else 'modified')} 标注: {image_id}"
    )

def save_annotation(
    dataset_name,
    selected_image_name_state,
    source_prompt,
    target_prompt,
    local_blend_word_value,
    mutual_blend_word_value,
    local_blend_thresh_e_value,
    local_blend_thresh_m_value,
    drag_points_state,
    user_mask_state,
    drag_type_value,
    influence_range_value,
    edit_mode_value,
    eval_mode,
    dataset_view_mode,
):
    if not selected_image_name_state:
        return "保存失败：未选择文件"
    cfg = DATASET_CONFIG[dataset_name]
    root = cfg["root"]
    image_id = os.path.splitext(selected_image_name_state)[0]

    modified_mask_dir = os.path.join(root, MODIFIED_MASK_DIR)
    os.makedirs(modified_mask_dir, exist_ok=True)
    modified_mask_rel = f"{MODIFIED_MASK_DIR}/{image_id}.png"
    mask_saved = False
    if user_mask_state is not None:
        Image.fromarray((user_mask_state * 255).astype(np.uint8)).save(os.path.join(root, modified_mask_rel))
        mask_saved = True

    normalized_drag_type = _normalize_drag_type(drag_type_value)
    normalized_range = _normalize_influence_range(influence_range_value)
    normalized_local_blend_word = _normalize_blend_word(local_blend_word_value)
    normalized_mutual_blend_word = _normalize_blend_word(mutual_blend_word_value)
    normalized_local_blend_thresh_e = _normalize_blend_thresh(
        local_blend_thresh_e_value, DEFAULT_LOCAL_BLEND_THRESH_E
    )
    normalized_local_blend_thresh_m = _normalize_blend_thresh(
        local_blend_thresh_m_value, DEFAULT_LOCAL_BLEND_THRESH_M
    )
    normalized_edit_mode = _normalize_edit_mode(edit_mode_value)
    normalized_dataset_view = str(dataset_view_mode or DEFAULT_DATASET_VIEW).strip().lower()
    if normalized_dataset_view not in DATASET_VIEW_CHOICES:
        normalized_dataset_view = DEFAULT_DATASET_VIEW
    normalized_points = drag_points_state if isinstance(drag_points_state, list) else []
    eval_mode = bool(eval_mode)
    target_branch_key = normalized_dataset_view if normalized_dataset_view in {"modified", "user_study"} else "modified"

    for j_name in cfg["json_files"]:
        json_path = os.path.join(root, j_name)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {}
        except json.JSONDecodeError as exc:
            print(f"[save_annotation] malformed json {json_path}: {exc}")
            data = {}
        except OSError as exc:
            print(f"[save_annotation] read {json_path} failed: {exc}")
            data = {}

        split_entry = _to_split_annotation_entry(data.get(image_id, {}))
        source_entry = split_entry.get("source", {})
        image_rel = source_entry.get("image_path", f"images/{selected_image_name_state}")

        target_entry = split_entry.get(target_branch_key, {})
        if not isinstance(target_entry, dict):
            target_entry = {}

        if eval_mode:
            # Eval 模式：保存 mask + 控制点 + drag 参数，不更新 prompt
            target_entry.update({
                "points": normalized_points,
                "mask_path": modified_mask_rel if mask_saved else target_entry.get("mask_path", ""),
                "image_path": image_rel,
                "drag_type": normalized_drag_type,
                "influence_range": normalized_range,
                "blended_word": normalized_local_blend_word,
                "local_blend_word": normalized_local_blend_word,
                "mutual_word": normalized_mutual_blend_word,
                "local_blend_thresh_e": normalized_local_blend_thresh_e,
                "local_blend_thresh_m": normalized_local_blend_thresh_m,
            })
        else:
            target_entry.update({
                "source_prompt": source_prompt,
                "points": normalized_points,
                "mask_path": modified_mask_rel if mask_saved else target_entry.get("mask_path", ""),
                "image_path": image_rel,
                "drag_type": normalized_drag_type,
                "influence_range": normalized_range,
                "blended_word": normalized_local_blend_word,
                "local_blend_word": normalized_local_blend_word,
                "mutual_word": normalized_mutual_blend_word,
                "local_blend_thresh_e": normalized_local_blend_thresh_e,
                "local_blend_thresh_m": normalized_local_blend_thresh_m,
            })
            # Drag 模式只保存拖拽相关标注，不覆盖已有 target_prompt。
            if normalized_edit_mode != "Drag":
                target_entry["target_prompt"] = target_prompt
        if target_branch_key == "user_study":
            target_entry[USER_STUDY_FLAG_KEY] = 1
        split_entry[target_branch_key] = target_entry
        data[image_id] = split_entry
        _dump_json_with_compact_points(json_path, data)

    if eval_mode:
        return f"保存成功（Eval 模式：写入 {target_branch_key}，保存 mask + 控制点 + drag 参数）！ID: {image_id}"
    return f"保存成功（写入 {target_branch_key}）！ID: {image_id}"

# --- 修改点 1：更新列表逻辑，构建画廊数据 ---
def update_image_list(dataset_name, hide_labeled):
    cfg = DATASET_CONFIG[dataset_name]
    root = cfg["root"]
    img_dir = os.path.join(root, "images")
    if not os.path.exists(img_dir): 
        # 返回：画廊数据，隐藏下拉框更新，状态列表
        return [], gr.update(choices=[], value=None), []
    
    all_imgs = [f for f in sorted(os.listdir(img_dir)) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if hide_labeled:
        labeled_ids = set()
        for j_name in cfg["json_files"]:
            json_path = os.path.join(root, j_name)
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for img_id, content in data.items():
                            split_entry = _to_split_annotation_entry(content)
                            if _is_modified_entry_user_edited(split_entry):
                                labeled_ids.add(img_id)
                except (OSError, json.JSONDecodeError, TypeError) as exc:
                    print(f"[update_image_list] read {json_path} failed: {exc}")
        all_imgs = [f for f in all_imgs if os.path.splitext(f)[0] not in labeled_ids]
    
    # 构建画廊数据：[(绝对路径, 文件名), (绝对路径, 文件名)...]
    gallery_data = [(os.path.join(img_dir, f), f) for f in all_imgs]
    
    first_val = all_imgs[0] if all_imgs else None
    
    # 输出 1: Gallery的值, 输出 2: 隐藏Dropdown的更新, 输出 3: 状态列表
    return gallery_data, gr.update(choices=all_imgs, value=first_val), all_imgs

# 新增辅助函数：处理画廊点击
def on_gallery_select(evt: gr.SelectData, image_name_list):
    # evt.index 是被点击图片的索引
    # 我们直接从状态列表中取文件名，绝对安全，不会触发 URL 校验
    if image_name_list and evt.index < len(image_name_list):
        return image_name_list[evt.index]
    return None
# ==========================================
# 5. UI 构建
# ==========================================

css = """
.fixed-height img { 
    object-fit: contain !important; 
}

/* --- 画廊外框：保留这个，这是右边的滚动条 --- */
.gallery-scroll {
    height: 350px !important;
    overflow-y: auto !important;
    display: block !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
}

/* --- 新增：隐藏画廊内部的滚动条 (左边那个) --- */
/* 针对 Chrome/Edge/Safari */
.gallery-scroll *::-webkit-scrollbar {
    width: 0 !important;
    height: 0 !important;
    display: none !important;
}

/* 针对 Firefox */
.gallery-scroll * {
    scrollbar-width: none !important;
}
"""

with gr.Blocks(css=css, title="TDEdit Tool") as demo:
    gr.Markdown("# 🛠️ TDEdit Evaluation & Annotation Tool")
    
    current_image_name_list = gr.State([])
    selected_image_name_state = gr.State("")
    source_image_state = gr.State(None)
    display_mask_state = gr.State(None)
    user_mask_state = gr.State(None)
    drag_points_state = gr.State([])
    suppress_drag_type_change_once = gr.State(False)
    
    with gr.Row():
        with gr.Column(scale=60):
            with gr.Group():
                with gr.Row():
                    dataset_name_dropdown = gr.Dropdown(
                        label="1. 数据集",
                        choices=list(DATASET_CONFIG.keys()),
                        value=DEFAULT_DATASET_NAME,
                        scale=3,
                    )
                    hide_labeled_checkbox = gr.Checkbox(label="🚫 隐藏已标注", value=False, scale=1)
            
            # --- 修改点 2：UI 替换 ---
            # 1. 把原来的 image_dd 设为 visible=False (作为逻辑中转，不要删)
            selected_image_name_dropdown = gr.Dropdown(
                label="当前选中文件",
                interactive=True,
                visible=False,
                allow_custom_value=True,
            )
            
            # 2. 新增 Gallery 组件 (方案一)
            image_gallery = gr.Gallery(
                label="2. 选择图片 (点击选择)", 
                columns=[4],
                rows=[2],
                object_fit="contain",
                height=350,  
                allow_preview=False,
                elem_classes=["gallery-scroll"]  # <--- 关键！加上这个
            )
            
            canvas = gr.ImageEditor(type="numpy", label="3. 绘制 Mask", brush=gr.Brush(colors=["#000000"], color_mode="fixed", default_size=80), height=512, elem_classes=["fixed-height"])
            btn_process_mask = gr.Button("⬇️ 确认涂抹", variant="secondary")
            with gr.Row():
                refresh_btn = gr.Button("🔄 刷新")
                prev_btn = gr.Button("⬅️ 上一张")
                next_btn = gr.Button("下一张 ➡️")
            
            preview_image = gr.Image(type="numpy", label="4. 预览与加点", interactive=True, height=512, elem_classes=["fixed-height"])
            
            with gr.Row():
                undo_button = gr.Button("🧹 清空控制点 / 重绘")
                clear_btn = gr.Button("🗑️ 清空当前图片所有标注", variant="secondary")
                save_btn = gr.Button("💾 保存标注结果", variant="primary")
            
            run_btn = gr.Button("🚀 Run Inference", variant="stop")
            image_out = gr.Image(label="Output Image")
            image_layout = gr.Image(label="Layout", visible=False)
            status_label = gr.Label(label="系统状态")

        with gr.Column(scale=40):
            with gr.Tab("UAC options"):
                source_prompt = gr.Textbox(label="Source prompt")
                target_prompt = gr.Textbox(label="Target prompt")
                gr.Markdown("**Core Toggles**")
                with gr.Row():
                    denoise = gr.Checkbox(label="Denoising Mode", value=True)
                    low_randomness = gr.Checkbox(label="Low Randomness (eta=0)", value=False)
                with gr.Row():
                    source_guidance_scale = gr.Slider(label="Source GS", value=1.0, minimum=1, maximum=10)
                    target_guidance_scale = gr.Slider(
                        label="Target GS",
                        value=_default_target_guidance_for_mode(DEFAULT_EDIT_MODE),
                        minimum=1,
                        maximum=10,
                    )
                positive_prompt = gr.Textbox(label="Positive Prompt", value="")
                negative_prompt = gr.Textbox(label="Negative Prompt", value="")
                with gr.Row():
                    local_blend_word = gr.Textbox(
                        label="Target blend",
                        value=DEFAULT_LOCAL_BLEND_WORD,
                        placeholder="",
                    )
                    local_blend_thresh_e = gr.Slider(
                        label="Target blend thresh",
                        value=DEFAULT_LOCAL_BLEND_THRESH_E,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                    )
                with gr.Row():
                    mutual_blend_word = gr.Textbox(
                        label="Source blend",
                        value=DEFAULT_MUTUAL_BLEND_WORD,
                        placeholder="",
                    )
                    local_blend_thresh_m = gr.Slider(
                        label="Source blend thresh",
                        value=DEFAULT_LOCAL_BLEND_THRESH_M,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                    )
                with gr.Row():
                    text_cross_replace_steps = gr.Slider(label="Text Cross Schedule", value=0.7, maximum=1)
                    text_self_replace_steps = gr.Slider(label="Text Self Schedule", value=0.7, maximum=1)
                with gr.Row():
                    drag_self_replace_steps = gr.Slider(
                        label="GSAC起点(进度, drag_self_replace_steps)",
                        value=0.7,
                        maximum=1,
                    )
                with gr.Row():
                    strength = gr.Slider(
                        label="Strength",
                        value=_default_strength_for_mode(DEFAULT_EDIT_MODE),
                        maximum=1,
                    )
                with gr.Row():
                    edit_mode = gr.Radio(
                        label="Edit Mode",
                        choices=EDIT_MODE_CHOICES,
                        value=DEFAULT_EDIT_MODE,
                    )
                with gr.Row():
                    drag_type = gr.Dropdown(
                        label='Drag Type',
                        choices=DRAG_TYPE_CHOICES,
                        value=DEFAULT_DRAG_TYPE
                    )
                with gr.Row():
                    influence_range = gr.Slider(
                        label="Influence Range", 
                        value=DEFAULT_INFLUENCE_RANGE, minimum=0, maximum=1, step=0.01,
                        info="值越大影响范围越大，值越小影响范围越局部。该语义在2D和3D模式中一致。"
                    )
                gr.Markdown("**Dataset View**")
                with gr.Row():
                    run_mode = gr.Radio(
                        label="Run Mode",
                        choices=RUN_MODE_CHOICES,
                        value=DEFAULT_RUN_MODE,
                    )
                with gr.Row():
                    dataset_view_mode = gr.Radio(
                        label="标注视图",
                        choices=DATASET_VIEW_CHOICES,
                        value=DEFAULT_DATASET_VIEW,
                    )
                with gr.Row():
                    eval_mode_checkbox = gr.Checkbox(
                        label="Eval Mode（保存 Mask + 控制点，不改 Prompt）",
                        value=True,
                    )
                with gr.Row():
                    visualize_drag = gr.Checkbox(label="Visualize Drag", value=False)
                    visualize_process = gr.Checkbox(label="Visualize Process", value=False)
                    show_original_mask = gr.Checkbox(label="Show Original Mask", value=DEFAULT_SHOW_ORIGINAL_MASK)
                with gr.Row():
                    inference_mask_mode = gr.Radio(
                        label="Inference Mask Mode",
                        choices=INFERENCE_MASK_MODE_CHOICES,
                        value=DEFAULT_INFERENCE_MASK_MODE
                    )
                with gr.Row():
                    pointcloud_domain = gr.Radio(
                        label="3D点云模式 (Point Cloud Domain)",
                        choices=POINTCLOUD_DOMAIN_CHOICES,
                        value=DEFAULT_POINTCLOUD_DOMAIN,
                    )
                with gr.Row():
                    drag_layout_latents = gr.Checkbox(
                        label="拖拽 Layout 分支",
                        value=DEFAULT_FRONTEND_DRAG_LAYOUT_LATENTS,
                    )
                    drag_target_latents = gr.Checkbox(
                        label="拖拽 Target 分支",
                        value=DEFAULT_FRONTEND_DRAG_TARGET_LATENTS,
                    )
                with gr.Row():
                    drag_target_q_layout_mix = gr.Checkbox(
                        label="Target Q混合Layout",
                        value=DEFAULT_DRAG_TARGET_Q_LAYOUT_MIX,
                    )
                    ref_kv_injection = gr.Checkbox(
                        label="Ref K/V替换",
                        value=DEFAULT_FRONTEND_REF_KV_INJECTION,
                    )
                with gr.Row():
                    ref_target_denoise_mix = gr.Checkbox(
                        label="Ref-Target去噪混合",
                        value=DEFAULT_REF_TARGET_DENOISE_MIX,
                    )
                with gr.Row():
                    ref_target_denoise_mix_max = gr.Slider(
                        label="Ref-Target混合上限",
                        value=DEFAULT_REF_TARGET_DENOISE_MIX_MAX,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                    )
                    ref_target_denoise_mix_start = gr.Slider(
                        label="Ref-Target混合起点(进度)",
                        value=DEFAULT_REF_TARGET_DENOISE_MIX_START,
                        minimum=0.0,
                        maximum=0.95,
                        step=0.01,
                    )
                with gr.Row():
                    hole_fill_mode = gr.Radio(
                        label="Hole Fill Mode",
                        choices=HOLE_FILL_MODE_CHOICES,
                        value=DEFAULT_HOLE_FILL_MODE,
                    )
                with gr.Row():
                    use_expanded_subject_fill = gr.Checkbox(
                        label="Mask大补全",
                        value=DEFAULT_EXPANDED_SUBJECT_FILL,
                    )
                    use_drag_guided_prefill = gr.Checkbox(
                        label="方向预填充",
                        value=DEFAULT_DRAG_GUIDED_PREFILL,
                    )
                    enable_3d_subject_scope_fill = gr.Checkbox(
                        label="3D SubjectScope补洞",
                        value=DEFAULT_3D_SUBJECT_SCOPE_FILL,
                    )
                    expanded_subject_fill_px = gr.Slider(
                        label="Mask大扩张(px)",
                        value=_default_expanded_subject_fill_px_for_mode(DEFAULT_EDIT_MODE),
                        minimum=0,
                        maximum=48,
                        step=1,
                    )

            with gr.Tab("Base Model Config"):
                with gr.Row():
                    start_step = gr.Slider(label="Ref Inject Start Step", value=1, minimum=0, maximum=15, step=1)
                    start_layer = gr.Slider(label="Ref Inject Start Layer", value=10, minimum=0, maximum=15, step=1)

    # --- 事件绑定 ---
    # 1. 刷新事件：现在的 outputs 多了一个 image_gallery
    refresh_events = [dataset_name_dropdown.change, refresh_btn.click, hide_labeled_checkbox.change]
    for event in refresh_events:
        event(
            update_image_list, 
            inputs=[dataset_name_dropdown, hide_labeled_checkbox], 
            # 注意顺序：Gallery, Dropdown, State
            outputs=[image_gallery, selected_image_name_dropdown, current_image_name_list]
        )

    # 2. 新增：画廊点击事件 -> 更新隐藏的 Dropdown
    # 当点击画廊时，把选中的文件名传给 selected_image_name_dropdown，
    # selected_image_name_dropdown 会自动触发后续的加载逻辑
    image_gallery.select(
        on_gallery_select,
        inputs=[current_image_name_list],  # <--- 这里加上 current_image_name_list
        outputs=selected_image_name_dropdown
    )

    load_event_outputs = [
        canvas, preview_image, source_image_state, source_prompt, target_prompt,
        local_blend_word, mutual_blend_word, local_blend_thresh_e, local_blend_thresh_m,
        display_mask_state, user_mask_state, drag_points_state, selected_image_name_state,
        drag_type, influence_range,
        strength,
        target_guidance_scale,
        status_label,
        suppress_drag_type_change_once,
    ]

    def navigate(direction, current_loaded_name, image_name_list):
        if not image_name_list:
            return None
        if current_loaded_name not in image_name_list:
            return image_name_list[0] if direction == "next" else image_name_list[-1]
        idx = image_name_list.index(current_loaded_name)
        new_idx = (idx + 1) % len(image_name_list) if direction == "next" else (idx - 1) % len(image_name_list)
        return image_name_list[new_idx]

    def navigate_and_load(
        direction,
        current_loaded_name,
        image_name_list,
        dataset_name,
        drag_type_value,
        influence_range_value,
        edit_mode_value,
        dataset_view_mode,
        show_original_mask_value,
    ):
        target_name = navigate(direction, current_loaded_name, image_name_list)
        return on_dataset_modify_options_change(
            dataset_name,
            target_name,
            drag_type_value,
            influence_range_value,
            edit_mode_value,
            dataset_view_mode,
            show_original_mask_value,
        )

    prev_btn.click(
        lambda c_loaded, l, ds, dt, ir, em, dvm, som: navigate_and_load("prev", c_loaded, l, ds, dt, ir, em, dvm, som),
        [selected_image_name_state, current_image_name_list, dataset_name_dropdown, drag_type, influence_range, edit_mode, dataset_view_mode, show_original_mask],
        load_event_outputs,
    )
    next_btn.click(
        lambda c_loaded, l, ds, dt, ir, em, dvm, som: navigate_and_load("next", c_loaded, l, ds, dt, ir, em, dvm, som),
        [selected_image_name_state, current_image_name_list, dataset_name_dropdown, drag_type, influence_range, edit_mode, dataset_view_mode, show_original_mask],
        load_event_outputs,
    )

    selected_image_name_dropdown.change(
        on_dataset_modify_options_change,
        inputs=[dataset_name_dropdown, selected_image_name_dropdown, drag_type, influence_range, edit_mode, dataset_view_mode, show_original_mask],
        outputs=load_event_outputs
    )

    drag_type.change(
        on_drag_type_change,
        inputs=[
            drag_type, influence_range, strength, edit_mode,
            source_image_state, source_prompt, target_prompt,
            local_blend_word, mutual_blend_word, local_blend_thresh_e, local_blend_thresh_m,
            display_mask_state,
            user_mask_state, drag_points_state, selected_image_name_state,
            suppress_drag_type_change_once
        ],
        outputs=[
            canvas, preview_image, source_image_state, source_prompt, target_prompt,
            local_blend_word, mutual_blend_word, local_blend_thresh_e, local_blend_thresh_m,
            display_mask_state, user_mask_state, drag_points_state, selected_image_name_state,
            drag_type, influence_range,
            strength,
            status_label, suppress_drag_type_change_once
        ]
    )

    dataset_view_mode.change(
        on_dataset_modify_options_change,
        inputs=[dataset_name_dropdown, selected_image_name_state, drag_type, influence_range, edit_mode, dataset_view_mode, show_original_mask],
        outputs=load_event_outputs,
    )

    edit_mode.change(
        on_dataset_modify_options_change,
        inputs=[dataset_name_dropdown, selected_image_name_state, drag_type, influence_range, edit_mode, dataset_view_mode, show_original_mask],
        outputs=load_event_outputs,
    )

    edit_mode.change(
        lambda mode_value: gr.update(value=_default_denoise_for_mode(mode_value)),
        inputs=[edit_mode],
        outputs=[denoise],
    )
    edit_mode.change(
        lambda mode_value: gr.update(value=_default_expanded_subject_fill_px_for_mode(mode_value)),
        inputs=[edit_mode],
        outputs=[expanded_subject_fill_px],
    )
    edit_mode.change(
        lambda mode_value: gr.update(value=_default_hole_fill_mode_for_mode(mode_value)),
        inputs=[edit_mode],
        outputs=[hole_fill_mode],
    )

    show_original_mask.change(
        on_show_original_mask_change,
        inputs=[show_original_mask, inference_mask_mode, source_image_state, user_mask_state, drag_type, drag_points_state],
        outputs=[preview_image, display_mask_state],
    )
    inference_mask_mode.change(
        on_inference_mask_mode_change,
        inputs=[inference_mask_mode, source_image_state, user_mask_state, drag_type, drag_points_state],
        outputs=[preview_image, display_mask_state],
    )

    clear_btn.click(
        clear_current_annotations, 
        [dataset_name_dropdown, selected_image_name_state, source_image_state, dataset_view_mode], 
        [canvas, preview_image, display_mask_state, user_mask_state, drag_points_state, status_label]
    )

    btn_process_mask.click(
        update_mask_and_visualize, 
        inputs=[canvas, drag_type, source_image_state, drag_points_state, show_original_mask, inference_mask_mode], 
        outputs=[source_image_state, preview_image, display_mask_state, user_mask_state, drag_points_state]
    )

    preview_image.select(safe_get_points, [preview_image, drag_points_state], [preview_image])
    
    undo_button.click(
        reset_to_base_state, 
        inputs=[source_image_state, display_mask_state, drag_type], 
        outputs=[preview_image, drag_points_state]
    )
    
    save_btn.click(
        save_annotation,
        [
            dataset_name_dropdown, selected_image_name_state,
            source_prompt, target_prompt,
            local_blend_word, mutual_blend_word, local_blend_thresh_e, local_blend_thresh_m,
            drag_points_state, user_mask_state,
            drag_type, influence_range, edit_mode, eval_mode_checkbox, dataset_view_mode,
        ],
        status_label
    )

    def run_inference_wrapper(
        source_image_state, source_prompt_text, target_prompt_text,
        local_blend_word_text, mutual_blend_word_text,
        local_blend_thresh_e_value, local_blend_thresh_m_value,
        positive_prompt, negative_prompt,
        source_guidance_scale, target_guidance_scale,
        strength, run_mode, edit_mode, start_step, start_layer,
        text_cross_replace_steps, text_self_replace_steps,
        drag_self_replace_steps,
        denoise, low_randomness,
        display_mask_state, user_mask_state, drag_points_state,
        visualize_process, visualize_drag,
        drag_type, influence_range, pointcloud_domain,
        drag_layout_latents, drag_target_latents,
        drag_target_q_layout_mix,
        ref_kv_injection,
        ref_target_denoise_mix,
        ref_target_denoise_mix_max, ref_target_denoise_mix_start,
        hole_fill_mode,
        use_expanded_subject_fill, use_drag_guided_prefill, enable_3d_subject_scope_fill, expanded_subject_fill_px,
        selected_image_name_state,
        inference_mask_mode,
        show_original_mask,
    ):
        if source_image_state is None:
            return None, None

        debug_root = "debug_files"
        mask_debug_dir = os.path.join(debug_root, "mask_process")
        drag_debug_dir = os.path.join(debug_root, "drag_process")

        run_mode_norm = str(run_mode or "").strip()
        paper_mode = run_mode_norm in {PAPER_MODE_FULL, PAPER_MODE_LITE}
        paper_mode_lite = run_mode_norm == PAPER_MODE_LITE
        if paper_mode and (not paper_mode_lite):
            _prepare_paper_debug_workspace(debug_root)

        normalized_mode = _normalize_edit_mode(edit_mode)
        mode_text = normalized_mode == "Text"
        run_steps = _default_steps_for_mode(normalized_mode)
        run_seed = 0 if mode_text else 42
        run_start_step = int(round(float(start_step)))
        run_start_layer = int(round(float(start_layer)))
        run_start_step = max(0, min(15, run_start_step))
        run_start_layer = max(0, min(15, run_start_layer))
        drag_cross_replace_steps = float(DEFAULT_DRAG_CROSS_REPLACE_STEPS)
        run_denoise = bool(denoise)
        run_source_prompt = source_prompt_text
        run_target_prompt = target_prompt_text
        raw_drag_points = drag_points_state if isinstance(drag_points_state, list) else []
        run_points = [] if mode_text else raw_drag_points
        if normalized_mode == "Drag":
            run_target_prompt = run_source_prompt
        run_has_drag_points = isinstance(run_points, list) and len(run_points) >= 2
        run_has_text_delta = str(run_source_prompt or "").strip() != str(run_target_prompt or "").strip()
        if mode_text and len(raw_drag_points) >= 2:
            print("[run_inference] edit_mode=text: drag points are ignored. Switch to Drag/Joint to enable drag.")
        inference_mask_mode = _normalize_inference_mask_mode(inference_mask_mode)
        mask_backend_mode = "user_mask" if inference_mask_mode == "User Mask" else "sam_refined"
        hole_fill_mode = normalize_hole_fill_mode(
            hole_fill_mode,
            default=_default_hole_fill_mode_for_mode(normalized_mode),
        )
        # Lite 模式只保留最小产出，不跑中间可视化调试逻辑。
        run_visualize_process = bool(visualize_process) and (not paper_mode_lite)
        run_visualize_drag = (
            bool(visualize_drag) or (paper_mode and (not paper_mode_lite))
        ) and run_has_drag_points
        if bool(visualize_drag) and (not run_visualize_drag):
            print("[run_inference] visualize_drag disabled: no valid drag pairs after mode filtering.")
        if run_visualize_drag:
            for folder in [mask_debug_dir, drag_debug_dir]:
                os.makedirs(folder, exist_ok=True)
        frontend_drag_layout_latents = bool(drag_layout_latents)
        frontend_drag_target_latents = bool(drag_target_latents)
        frontend_drag_target_q_layout_mix = bool(drag_target_q_layout_mix)
        frontend_ref_kv_injection = bool(ref_kv_injection)
        frontend_drag_clean_latents = DEFAULT_FRONTEND_DRAG_CLEAN_LATENTS
        # 先按 mode 预处理输入，再根据预处理后的真实状态决定 text-only 词级引导。
        if run_has_text_delta and (not run_has_drag_points):
            run_local_blend_word = _normalize_blend_word(local_blend_word_text)
            # 对齐 InfEdit PIE text：source blend(mutual) 固定为空。
            run_mutual_blend_word = DEFAULT_MUTUAL_BLEND_WORD
        else:
            run_local_blend_word = DEFAULT_LOCAL_BLEND_WORD
            run_mutual_blend_word = DEFAULT_MUTUAL_BLEND_WORD
        run_local_blend_thresh_e = _normalize_blend_thresh(local_blend_thresh_e_value, DEFAULT_LOCAL_BLEND_THRESH_E)
        run_local_blend_thresh_m = _normalize_blend_thresh(local_blend_thresh_m_value, DEFAULT_LOCAL_BLEND_THRESH_M)
        debug_artifacts = {} if paper_mode else None
        empty_mask_u8 = np.zeros(source_image_state.shape[:2], dtype=np.uint8)

        # Drag 模式保持 102 老逻辑：
        # 前端预览可继续显示精炼后的 mask，但实际推理只传原始 user mask，
        # 由后端执行唯一一次 SAM 精炼，避免前后端双重精炼导致主体拆分和意图漂移。
        if normalized_mode.lower() == "drag":
            mask_for_inference = _normalize_mask_u8(
                user_mask_state if user_mask_state is not None else empty_mask_u8,
                source_image_state.shape[:2],
            )
            mask_backend_mode = "sam_refined"
        # 推理mask与前端预览解耦：
        # - SAM Refined: 优先使用已缓存 display_mask（show_original_mask=False时），否则现场重算SAM精炼mask
        # - User Mask: 直接使用用户原始涂抹mask
        elif inference_mask_mode == "User Mask":
            mask_for_inference = _normalize_mask_u8(
                user_mask_state if user_mask_state is not None else empty_mask_u8,
                source_image_state.shape[:2],
            )
        else:
            if (display_mask_state is not None) and (not bool(show_original_mask)):
                mask_for_inference = _normalize_mask_u8(display_mask_state, source_image_state.shape[:2])
            else:
                refined_mask = refine_user_mask_with_sam(source_image_state, user_mask_state)
                mask_for_inference = _normalize_mask_u8(
                    refined_mask if refined_mask is not None else empty_mask_u8,
                    source_image_state.shape[:2],
                )

        # 调用简化后的 inference 函数
        # Paper 模式下：单图默认关闭文字标注（点云类可视化除外，由下游函数自行保留）。
        prev_paper_no_text_flag = os.environ.get(PAPER_NO_TEXT_SINGLE_ENV)
        if paper_mode:
            os.environ[PAPER_NO_TEXT_SINGLE_ENV] = "1"
        else:
            os.environ.pop(PAPER_NO_TEXT_SINGLE_ENV, None)
        try:
            result_img, layout_img = inference(
                img=source_image_state,
                source_prompt=run_source_prompt,
                target_prompt=run_target_prompt,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                guidance_s=source_guidance_scale,
                guidance_t=target_guidance_scale,
                num_inference_steps=run_steps,
                seed=run_seed,
                strength=float(strength),
                start_step=run_start_step,
                start_layer=run_start_layer,
                cross_replace_steps=text_cross_replace_steps,
                self_replace_steps=text_self_replace_steps,
                text_cross_replace_steps=text_cross_replace_steps,
                text_self_replace_steps=text_self_replace_steps,
                drag_cross_replace_steps=drag_cross_replace_steps,
                drag_self_replace_steps=drag_self_replace_steps,
                denoise=run_denoise,
                mask=mask_for_inference,
                selected_points=run_points,
                visualize_process=run_visualize_process,
                visualize_drag=run_visualize_drag,
                drag_type=drag_type,
                influence_range=influence_range,
                pointcloud_domain=pointcloud_domain,
                attn_switch_mode="hard",
                ref_kv_injection=frontend_ref_kv_injection,
                ref_target_denoise_mix=ref_target_denoise_mix,
                ref_target_denoise_mix_max=ref_target_denoise_mix_max,
                ref_target_denoise_mix_start=ref_target_denoise_mix_start,
                hole_fill_mode=hole_fill_mode,
                use_expanded_subject_fill=use_expanded_subject_fill,
                expanded_subject_fill_px=expanded_subject_fill_px,
                use_drag_guided_prefill=use_drag_guided_prefill,
                enable_3d_subject_scope_fill=enable_3d_subject_scope_fill,
                image_name=selected_image_name_state,
                low_randomness=low_randomness,
                local_blend_word=run_local_blend_word,
                mutual_blend_word=run_mutual_blend_word,
                local_blend_thresh_e=run_local_blend_thresh_e,
                local_blend_thresh_m=run_local_blend_thresh_m,
                edit_mode=normalized_mode.lower(),
                debug_artifacts=debug_artifacts,
                mask_backend_mode=mask_backend_mode,
                drag_layout_latents=frontend_drag_layout_latents,
                drag_target_latents=frontend_drag_target_latents,
                drag_target_q_layout_mix=frontend_drag_target_q_layout_mix,
                drag_clean_latents=frontend_drag_clean_latents,
            )
        finally:
            if prev_paper_no_text_flag is None:
                os.environ.pop(PAPER_NO_TEXT_SINGLE_ENV, None)
            else:
                os.environ[PAPER_NO_TEXT_SINGLE_ENV] = prev_paper_no_text_flag

        # === 保存结果到 final_results ===
        if not paper_mode_lite:
            final_dir = os.path.join(debug_root, "final_results")
            try:
                os.makedirs(final_dir, exist_ok=True)

                # 清理旧文件，确保 final_results 中只有 3 张图
                for name in os.listdir(final_dir):
                    path = os.path.join(final_dir, name)
                    if os.path.isfile(path) and name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                        try:
                            os.remove(path)
                        except OSError as exc:
                            print(f"[final_results] remove {path} failed: {exc}")

                # 原图与结果图 (PIL Image)
                src_pil = Image.fromarray(source_image_state)
                res_pil = result_img if isinstance(result_img, Image.Image) else Image.fromarray(np.array(result_img))

                # 1. 保存原图
                src_pil.save(os.path.join(final_dir, "original.png"))
                # 2. 保存结果图
                res_pil.save(os.path.join(final_dir, "result.png"))

                # 3. 生成拼接图
                # 左图：原图叠加 mask + 拖拽点 + source prompt
                mask_for_vis = _normalize_mask_u8(
                    mask_for_inference,
                    source_image_state.shape[:2],
                )
                left_np = _render_mask_overlay_with_guides(source_image_state, mask_for_vis, drag_type)

                # 绘制拖拽点和箭头
                if run_points:
                    pairs = []
                    for idx, pt in enumerate(run_points):
                        if idx % 2 == 0:
                            cv2.circle(left_np, tuple(pt), 8, (255, 0, 0), -1)
                            pairs.append(tuple(pt))
                        else:
                            cv2.circle(left_np, tuple(pt), 8, (0, 0, 255), -1)
                            pairs.append(tuple(pt))
                        if len(pairs) == 2:
                            cv2.arrowedLine(left_np, pairs[0], pairs[1], (255, 255, 255), 2, tipLength=0.1)
                            pairs = []

                left_pil = Image.fromarray(left_np)
                # 统一尺寸：以左图为基准，缩放右图
                res_pil_resized = res_pil.resize(left_pil.size, Image.BILINEAR)

                # Paper 模式全量禁用图片文字标注；Normal 模式保留文字条。
                text_bar_h = 0
                font = None
                if not paper_mode:
                    font_size = max(16, left_pil.size[0] // 25)
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                    except OSError:
                        font = ImageFont.load_default()
                    text_bar_h = font_size + 16

                # 创建拼接画布
                w, h = left_pil.size
                canvas_w = w * 2 + 10  # 中间留 10px 间隔
                canvas_h = h + text_bar_h
                canvas_img = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

                # 贴左图
                canvas_img.paste(left_pil, (0, text_bar_h))
                # 贴右图
                canvas_img.paste(res_pil_resized, (w + 10, text_bar_h))

                # Normal 模式保留文字标注，Paper 模式不添加任何文字。
                if not paper_mode:
                    draw = ImageDraw.Draw(canvas_img)
                    src_label = f"Source: {source_prompt_text}" if source_prompt_text else "Source"
                    tgt_label = f"Target: {target_prompt_text}" if target_prompt_text else "Target"
                    draw.text((4, 4), src_label, fill=(0, 0, 0), font=font)
                    draw.text((w + 14, 4), tgt_label, fill=(0, 0, 0), font=font)

                # 4. 保存拼接图
                canvas_img.save(os.path.join(final_dir, "concat.png"))
                print(f"[final_results] 已保存到 {final_dir}/original.png, result.png, concat.png")
            except Exception as e:
                print(f"[final_results] 保存失败: {e}")

        if paper_mode:
            try:
                run_config = {
                    "edit_mode": normalized_mode.lower(),
                    "source_prompt": str(source_prompt_text or ""),
                    "target_prompt": str(target_prompt_text or ""),
                    "positive_prompt": str(positive_prompt or ""),
                    "negative_prompt": str(negative_prompt or ""),
                    "steps": int(run_steps),
                    "seed": int(run_seed),
                    "strength": float(strength),
                    "guidance_s": float(source_guidance_scale),
                    "guidance_t": float(target_guidance_scale),
                    "start_step": int(run_start_step),
                    "start_layer": int(run_start_layer),
                    "text_cross_replace_steps": float(text_cross_replace_steps),
                    "text_self_replace_steps": float(text_self_replace_steps),
                    "drag_cross_replace_steps": float(drag_cross_replace_steps),
                    "drag_self_replace_steps": float(drag_self_replace_steps),
                    "denoise": bool(run_denoise),
                    "low_randomness": bool(low_randomness),
                    "drag_type": str(drag_type),
                    "influence_range": float(influence_range),
                    "pointcloud_domain": str(pointcloud_domain),
                    "hole_fill_mode": str(hole_fill_mode),
                    "use_expanded_subject_fill": bool(use_expanded_subject_fill),
                    "use_drag_guided_prefill": bool(use_drag_guided_prefill),
                    "enable_3d_subject_scope_fill": bool(enable_3d_subject_scope_fill),
                    "expanded_subject_fill_px": int(expanded_subject_fill_px),
                    "inference_mask_mode": str(inference_mask_mode),
                    "mask_backend_mode": str(mask_backend_mode),
                    "ref_target_denoise_mix": bool(ref_target_denoise_mix),
                    "ref_kv_injection": bool(frontend_ref_kv_injection),
                    "ref_target_denoise_mix_max": float(ref_target_denoise_mix_max),
                    "ref_target_denoise_mix_start": float(ref_target_denoise_mix_start),
                    "drag_layout_latents": bool(frontend_drag_layout_latents),
                    "drag_target_latents": bool(frontend_drag_target_latents),
                    "drag_target_q_layout_mix": bool(frontend_drag_target_q_layout_mix),
                    "drag_clean_latents": bool(frontend_drag_clean_latents),
                    "visualize_drag": bool(run_visualize_drag),
                    "visualize_process": bool(run_visualize_process),
                    "selected_image_name": str(selected_image_name_state or ""),
                    "paper_mode_lite": bool(paper_mode_lite),
                }
                paper_record = _save_paper_process_record(
                    process_root=PAPER_PROCESS_ROOT,
                    image_name=selected_image_name_state,
                    source_image_np=source_image_state,
                    user_mask_np=user_mask_state,
                    display_mask_np=display_mask_state,
                    drag_points=run_points,
                    result_img=result_img,
                    layout_img=layout_img,
                    source_prompt_text=source_prompt_text,
                    target_prompt_text=target_prompt_text,
                    run_config=run_config,
                    debug_root=debug_root,
                    debug_artifacts=debug_artifacts,
                    lite_mode=paper_mode_lite,
                )
                print(
                    f"[paper_mode] saved record: {paper_record['record_dir']} "
                    f"(reused={paper_record['reused_record']}, hash={paper_record['param_hash'][:8]})"
                )
            except Exception as exc:
                print(f"[paper_mode] save process record failed: {exc}")

        return result_img, layout_img

    run_btn.click(
        run_inference_wrapper,
        inputs=[
            source_image_state, source_prompt, target_prompt,
            local_blend_word, mutual_blend_word,
            local_blend_thresh_e, local_blend_thresh_m,
            positive_prompt, negative_prompt,
            source_guidance_scale, target_guidance_scale,
            strength, run_mode, edit_mode, start_step, start_layer,
            text_cross_replace_steps, text_self_replace_steps,
            drag_self_replace_steps,
            denoise, low_randomness,
            display_mask_state, user_mask_state, drag_points_state,
            visualize_process, visualize_drag,
            drag_type, influence_range, pointcloud_domain,
            drag_layout_latents, drag_target_latents,
            drag_target_q_layout_mix,
            ref_kv_injection,
            ref_target_denoise_mix,
            ref_target_denoise_mix_max, ref_target_denoise_mix_start,
            hole_fill_mode,
            use_expanded_subject_fill, use_drag_guided_prefill, enable_3d_subject_scope_fill, expanded_subject_fill_px,
            selected_image_name_state,
            inference_mask_mode,
            show_original_mask,
        ],
        outputs=[image_out, image_layout]
    )

    # 1. 定义一个用于 UI 调用的模型加载包装函数
    def auto_load_model_wrapper():
        print(">>> 正在后台自动加载模型，请稍候...")
        load_models()
        return "✅ 模型加载完成，系统就绪 (Ready)"

    # 2. 页面加载时：先刷新图片列表 (速度快，立刻显示)
    demo.load(
        update_image_list, 
        inputs=[dataset_name_dropdown, hide_labeled_checkbox], 
        outputs=[image_gallery, selected_image_name_dropdown, current_image_name_list]
    )

    # 3. 页面加载时：并发触发模型加载 (速度慢，在后台运行，完成后更新状态栏)
    demo.load(
        auto_load_model_wrapper,
        inputs=None,
        outputs=[status_label],
        queue=True # 允许在队列中运行，避免阻塞 UI 渲染
    )

def _resolve_allowed_paths(args: argparse.Namespace):
    allowed = []
    if not args.no_dataset_allowed_paths:
        allowed.extend(cfg["root"] for cfg in DATASET_CONFIG.values())
    allowed.extend(args.allowed_path or [])
    # keep order + deduplicate
    return list(dict.fromkeys(allowed))


def launch_ui(args: argparse.Namespace) -> None:
    if args.port < 1 or args.port > 65535:
        raise ValueError(f"Invalid --port={args.port}, expected [1, 65535]")

    os.makedirs("debug_files", exist_ok=True)

    allow_list = _resolve_allowed_paths(args)
    launch_kwargs = dict(
        server_name=args.host,
        server_port=args.port,
        allowed_paths=allow_list,
    )
    runnable_demo = demo.queue() if args.queue else demo

    print(
        f"启动 Gradio 服务... (host={args.host}, port={args.port}, "
        f"share={args.share}, queue={args.queue})"
    )
    try:
        runnable_demo.launch(share=args.share, **launch_kwargs)
    except ValueError as e:
        err = str(e)
        localhost_err = ("localhost is not accessible" in err) or ("shareable link must be created" in err)
        if (not args.share) and (not args.disable_localhost_fallback) and localhost_err:
            print("[Gradio] localhost 检查失败，自动切换 _frontend=False 重试...")
            runnable_demo.launch(share=False, _frontend=False, **launch_kwargs)
        else:
            raise


def main() -> None:
    parser = build_launch_parser()
    args = parser.parse_args()
    launch_ui(args)


if __name__ == "__main__":
    main()
