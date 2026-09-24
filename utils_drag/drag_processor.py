# drag_processor.py
import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2
import math
from PIL import Image as PILImage
from utils.segment_utils import get_interactive_mask, get_interactive_masks_batch
from utils_drag import rotate_3d_processor
from utils_drag.depth_estimator import estimate_depth
from utils_drag.hole_fill_modes import (
    DEFAULT_HOLE_FILL_MODE,
    normalize_hole_fill_mode,
)
# ==========================================
# 全局配置
# ==========================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MASK_DEBUG_ROOT = os.path.join(PROJECT_ROOT, "debug_files", "mask_process")
DRAG_DEBUG_ROOT = os.path.join(PROJECT_ROOT, "debug_files", "drag_process")
NON_RIGID_METHOD = "RATIO"
COMPONENT_MIN_AREA_BASE = 36
COMPONENT_MIN_AREA_RATIO = 2.5e-4
COMPONENT_MIN_AREA_RELATIVE_TO_LARGEST = 0.05
DEFAULT_EXPANDED_SUBJECT_FILL_ENABLED = True
DEFAULT_EXPANDED_SUBJECT_FILL_PX = 6
MAX_EXPANDED_SUBJECT_FILL_PX = 48
DEFAULT_DRAG_GUIDED_PREFILL_ENABLED = False
MASK_BACKEND_MODE_CHOICES = ("sam_refined", "user_mask")
DEFAULT_MASK_BACKEND_MODE = "sam_refined"
THREE_D_ANCHOR_STRATEGY_CHOICES = (
    "auto",
    "balanced_edge",
)
DEFAULT_THREE_D_ANCHOR_STRATEGY = "auto"
SUBPIXEL_CONSTRAINT_MODE_CHOICES = ("uniform_neighbor", "bilinear_iterative")
DEFAULT_SUBPIXEL_CONSTRAINT_MODE = "bilinear_iterative"
_LAMA_MODEL = None
_LAMA_INIT_FAILED = False


def _paper_no_text_single_enabled():
    v = os.environ.get("TDEDIT_PAPER_NO_TEXT_SINGLE")
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_hole_fill_mode(mode):
    return normalize_hole_fill_mode(mode, default=DEFAULT_HOLE_FILL_MODE)


def _normalize_expanded_subject_fill(use_expanded_subject_fill, expanded_subject_fill_px):
    use_expand = bool(use_expanded_subject_fill)
    try:
        expand_px = int(round(float(expanded_subject_fill_px)))
    except Exception:
        expand_px = int(DEFAULT_EXPANDED_SUBJECT_FILL_PX)
    expand_px = int(np.clip(expand_px, 0, MAX_EXPANDED_SUBJECT_FILL_PX))
    if expand_px <= 0:
        use_expand = False
    return use_expand, expand_px


def _normalize_drag_guided_prefill(use_drag_guided_prefill):
    return bool(use_drag_guided_prefill)


def _normalize_mask_backend_mode(mask_backend_mode):
    v = str(mask_backend_mode or DEFAULT_MASK_BACKEND_MODE).strip().lower()
    if v in {"user_mask", "user", "raw_user_mask", "raw"}:
        return "user_mask"
    if v not in MASK_BACKEND_MODE_CHOICES:
        return DEFAULT_MASK_BACKEND_MODE
    return v


def _normalize_3d_anchor_strategy(strategy):
    v = str(strategy or DEFAULT_THREE_D_ANCHOR_STRATEGY).strip().lower()
    if v in {"", "auto", "default"}:
        return "auto"
    if v in {"balanced", "balanced_edge", "balanced edge", "edge_balanced"}:
        return "balanced_edge"
    # 历史兼容：旧的 opposite 策略统一折叠到 balanced_edge。
    if v in {"drag_opposite", "drag opposite", "direction_opposite", "direction opposite"}:
        return "balanced_edge"
    if v in {
        "front_opposite",
        "front opposite",
        "phase2_front_opposite",
        "hybrid_front_opposite",
        "hybrid front opposite",
    }:
        return "balanced_edge"
    return DEFAULT_THREE_D_ANCHOR_STRATEGY


def _normalize_subpixel_constraint_mode(mode):
    v = str(mode or DEFAULT_SUBPIXEL_CONSTRAINT_MODE).strip().lower()
    if v in {"v2", "bilinear", "bilinear_iter", "bilinear_iterative"}:
        return "bilinear_iterative"
    if v in {"uniform_neighbor", "uniform_4neighbor", "uniform_4nn", "old", "v1"}:
        return "uniform_neighbor"
    if v not in SUBPIXEL_CONSTRAINT_MODE_CHOICES:
        return DEFAULT_SUBPIXEL_CONSTRAINT_MODE
    return v


def _get_subpixel_constraint_mode():
    return _normalize_subpixel_constraint_mode(
        os.environ.get("TDEDIT_SUBPIXEL_CONSTRAINT_MODE", DEFAULT_SUBPIXEL_CONSTRAINT_MODE)
    )


LATENT_WARP_AA_ENABLED = True
LATENT_WARP_AA_SCALE = 2
NONRIGID_RIGID_BLEND = 1.0
HYBRID_ANCHOR_FOLLOW_GAIN = 0.72
HYBRID_CONTROL_GLOBAL_FOLLOW_GAIN = 0.35
NONRIGID_CONTROL_GLOBAL_FOLLOW_GAIN = 0.0
NONRIGID_CONTROL_RADIUS_SCALE = 1.0

# ✅ 统一颜色定义（BGR格式，用于 OpenCV）
VIS_HANDLE_COLOR = (0, 0, 255)      # 红色: 起点/Handle
VIS_TARGET_COLOR = (255, 0, 0)      # 蓝色: 终点/Target
VIS_ARROW_COLOR = (255, 255, 255)   # 白色: 箭头
VIS_PIVOT_COLOR = (0, 255, 255)     # 黄色: 支点
VIS_ANCHOR_COLOR = (255, 255, 0)    # 青色: 锚点
VIS_FONT_SCALE = 0.6                # 统一字体大小

SUPPORTED_CANONICAL_DRAG_MODES = (
    "2D-Rigid",
    "2D-Non-Rigid",
    "2D-Hybrid",
    "3D-Rigid",
    "3D-Non-Rigid",
    "3D-Hybrid",
)

DRAG_MODE_ALIASES = {
    # 旧版本命名
    "Rigid": "2D-Rigid",
    "Non-Rigid": "2D-Non-Rigid",
    "Hybrid": "2D-Hybrid",
    "3D-Rotate": "3D-Rigid",
    "Stretch": "2D-Non-Rigid",
    "Hybrid-Rigid": "2D-Hybrid",
    # 新版本命名
    "2D-Rigid": "2D-Rigid",
    "2D-Non-Rigid": "2D-Non-Rigid",
    "2D-Hybrid": "2D-Hybrid",
    "3D-Rigid": "3D-Rigid",
    "3D-Non-Rigid": "3D-Non-Rigid",
    "3D-Hybrid": "3D-Hybrid",
}


def normalize_drag_mode(drag_mode):
    """统一拖拽类型命名，兼容旧字符串。"""
    mode = str(drag_mode).strip() if drag_mode is not None else ""
    canonical_mode = DRAG_MODE_ALIASES.get(mode)
    if canonical_mode is None:
        canonical_mode = "2D-Hybrid"
        print(f"[Drag Mode] Unknown mode '{drag_mode}', fallback to '{canonical_mode}'.")
    return canonical_mode


def split_drag_mode(drag_mode):
    """返回 (规范化模式, 维度域, 子模式)。"""
    canonical_mode = normalize_drag_mode(drag_mode)
    domain, base_mode = canonical_mode.split("-", 1)
    return canonical_mode, domain, base_mode


def _coord_scale_image_to_latent(size_img, size_lat):
    """
    坐标缩放（image -> latent）：
    与 align_corners=True 的网格坐标系保持一致。
    """
    si = int(size_img)
    sl = int(size_lat)
    if si > 1 and sl > 1:
        return float(sl - 1) / float(si - 1)
    return float(sl) / float(max(si, 1))


def _coord_scale_latent_to_image(size_lat, size_img):
    """
    坐标缩放（latent -> image）：
    与 align_corners=True 的网格坐标系保持一致。
    """
    sl = int(size_lat)
    si = int(size_img)
    if sl > 1 and si > 1:
        return float(si - 1) / float(sl - 1)
    return float(si) / float(max(sl, 1))


def _infer_image_size_from_latent_and_scale(size_lat, scale_lat_per_img):
    """
    由 latent 尺度与坐标缩放反推 image 尺度。
    """
    sl = max(1, int(size_lat))
    s = float(scale_lat_per_img)
    if sl <= 1:
        return 1
    if s > 1e-8:
        est = int(round((float(sl - 1) / s) + 1.0))
        if est > 0:
            return est
    return sl


def _grid_sample_latents_aa(latents, norm_grid, mode="bilinear", padding_mode="border", align_corners=True):
    """
    对 latent warp 做轻量抗混叠:
    - 先上采样 latent 与 grid
    - 在高分辨率做 grid_sample
    - 再回采样到原分辨率
    """
    if (not LATENT_WARP_AA_ENABLED) or mode != "bilinear" or LATENT_WARP_AA_SCALE <= 1:
        return F.grid_sample(
            latents.float(),
            norm_grid.unsqueeze(0),
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        ).to(latents.dtype)

    bsz, ch, h, w = latents.shape
    scale = int(LATENT_WARP_AA_SCALE)
    hh, ww = h * scale, w * scale

    lat_hi = F.interpolate(
        latents.float(), size=(hh, ww), mode="bilinear", align_corners=False
    )
    grid_hi = F.interpolate(
        norm_grid.permute(2, 0, 1).unsqueeze(0),
        size=(hh, ww),
        mode="bilinear",
        align_corners=align_corners,
    ).permute(0, 2, 3, 1)

    warped_hi = F.grid_sample(
        lat_hi,
        grid_hi,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    warped = F.interpolate(
        warped_hi, size=(h, w), mode="bilinear", align_corners=False
    )
    return warped.to(latents.dtype)


def _sanitize_xy_points(points):
    """将输入坐标清洗为 float32 的 Nx2，过滤 None/NaN/Inf。"""
    if points is None:
        return np.zeros((0, 2), dtype=np.float32)

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

    clean = []
    for row in rows:
        try:
            x = float(row[0])
            y = float(row[1])
        except Exception:
            continue
        if np.isfinite(x) and np.isfinite(y):
            clean.append([x, y])

    if len(clean) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(clean, dtype=np.float32)


# ==========================================
# 1. 核心修复算法
# ==========================================
def fill_background_holes(image_input, hole_mask, forbidden_mask=None, device=None, radius=5):
    """伪量化 OpenCV Telea 修复算法"""
    original_device = device
    if original_device is None:
        if isinstance(image_input, torch.Tensor): 
            original_device = image_input.device
        else: 
            original_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 统一转为 Numpy
    if isinstance(image_input, torch.Tensor):
        if image_input.ndim == 4:
            feat_np = image_input[0].detach().cpu().float().numpy()
        else:
            feat_np = image_input.detach().cpu().float().numpy()
        is_tensor_output = True
    else:
        feat_np = image_input.astype(np.float32)
        if feat_np.ndim == 3:
            feat_np = feat_np.transpose(2, 0, 1)
        is_tensor_output = False
    C, H, W = feat_np.shape
    # Mask 准备
    def prep_mask_numpy(m):
        if m is None: return np.zeros((H, W), dtype=np.uint8)
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().float()
            if m.ndim == 4: m = m[0, 0]
            elif m.ndim == 3: m = m[0]
            m = m.numpy()
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        return (m > 0.5).astype(np.uint8)
    mask_uint8 = prep_mask_numpy(hole_mask)
    if np.sum(mask_uint8) == 0:
        return image_input
    # forbidden 区域也并入 inpaint 输入掩码，避免修复时从主体边缘“偷纹理”回填到空洞
    forbidden_uint8 = prep_mask_numpy(forbidden_mask)
    if np.sum(forbidden_uint8) > 0:
        forbid_k = max(3, int(radius * 2 + 1))
        if forbid_k % 2 == 0:
            forbid_k += 1
        forbid_kernel = np.ones((forbid_k, forbid_k), np.uint8)
        forbidden_dilated = cv2.dilate(forbidden_uint8, forbid_kernel, iterations=1)
        inpaint_mask_uint8 = np.logical_or(mask_uint8 > 0, forbidden_dilated > 0).astype(np.uint8) * 255
    else:
        inpaint_mask_uint8 = mask_uint8 * 255
    # 逐通道修复
    filled_channels = []
    for i in range(C):
        channel = feat_np[i]
        c_min, c_max = channel.min(), channel.max()
        c_range = c_max - c_min + 1e-6
        
        channel_u8 = ((channel - c_min) / c_range * 255).astype(np.uint8)
        inpainted_u8 = cv2.inpaint(channel_u8, inpaint_mask_uint8, inpaintRadius=int(radius), flags=cv2.INPAINT_TELEA)
        inpainted_float = (inpainted_u8.astype(np.float32) / 255.0) * c_range + c_min
        
        mask_bool = (mask_uint8 > 0)
        final_channel = channel.copy()
        final_channel[mask_bool] = inpainted_float[mask_bool]
        filled_channels.append(final_channel)
    result_np = np.stack(filled_channels, axis=0)
    if is_tensor_output:
        out_tensor = torch.from_numpy(result_np).float().to(original_device)
        return out_tensor.unsqueeze(0)
    else:
        result_hwc = result_np.transpose(1, 2, 0)
        if result_hwc.max() <= 1.05:
            return (np.clip(result_hwc, 0, 1) * 255).astype(np.uint8)
        else:
            return np.clip(result_hwc, 0, 255).astype(np.uint8)


def _get_lama_model():
    global _LAMA_MODEL, _LAMA_INIT_FAILED
    if _LAMA_MODEL is not None:
        return _LAMA_MODEL
    if _LAMA_INIT_FAILED:
        return None
    try:
        from simple_lama_inpainting import SimpleLama
        _LAMA_MODEL = SimpleLama()
        print("[SGF/LaMa] backend=SimpleLaMa initialized.")
    except Exception as e:
        _LAMA_INIT_FAILED = True
        print(f"[SGF/LaMa] unavailable, fallback to Telea ({e})")
        return None
    return _LAMA_MODEL


def _run_lama_inpaint_rgb(image_rgb_u8, hole_mask_u8):
    lama = _get_lama_model()
    if lama is None:
        return None

    img = np.asarray(image_rgb_u8, dtype=np.uint8)
    if img.ndim != 3 or img.shape[2] < 3:
        return None
    if img.shape[2] > 3:
        img = img[:, :, :3]

    mask = (np.asarray(hole_mask_u8) > 0).astype(np.uint8) * 255
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    out = lama(PILImage.fromarray(img), PILImage.fromarray(mask))
    out_np = np.asarray(out)
    if out_np.ndim == 2:
        out_np = np.repeat(out_np[:, :, None], 3, axis=2)
    if out_np.shape[2] > 3:
        out_np = out_np[:, :, :3]
    # SimpleLaMa may pad/round non-multiple-of-eight inputs and return the
    # padded raster. Restore the caller's pixel grid before applying the
    # original-resolution hole mask.
    if out_np.shape[:2] != img.shape[:2]:
        out_np = cv2.resize(out_np, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    return np.clip(out_np, 0, 255).astype(np.uint8)


def _encode_rgb_to_latents_via_vae(image_rgb_u8, vae, device, dtype, target_hw=None):
    if vae is None:
        return None

    try:
        vae_param = next(vae.parameters())
        vae_device = vae_param.device
        vae_dtype = vae_param.dtype
    except Exception:
        vae_device = torch.device(device)
        vae_dtype = torch.float32

    img = np.asarray(image_rgb_u8, dtype=np.uint8)
    img_t = torch.from_numpy(img).to(device=vae_device, dtype=torch.float32)
    img_t = img_t.permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    img_t = img_t.to(dtype=vae_dtype)

    with torch.no_grad():
        latent = vae.encode(img_t).latent_dist.sample()
        scale = float(getattr(getattr(vae, "config", None), "scaling_factor", 0.18215))
        latent = latent * scale

    latent = latent.to(device=device, dtype=dtype)
    if target_hw is not None:
        th, tw = int(target_hw[0]), int(target_hw[1])
        if latent.shape[-2:] != (th, tw):
            latent = F.interpolate(latent.float(), size=(th, tw), mode="bilinear", align_corners=False).to(dtype=dtype)
    return latent


def _compose_clean_prior_into_current_latents(
    latents,
    clean_prior_latents,
    hole_mask_full,
    clean_latents=None,
    alpha_prod_t=None,
):
    if clean_prior_latents is None:
        return latents

    _, _, h_lat, w_lat = latents.shape
    hole_lat_u8 = _mask_to_binary_uint8(hole_mask_full, target_hw=(h_lat, w_lat))
    if int(np.sum(hole_lat_u8 > 0)) <= 0:
        return latents

    hole_t = torch.from_numpy((hole_lat_u8 > 0).astype(np.float32)).to(device=latents.device).view(1, 1, h_lat, w_lat) > 0.5

    prior = clean_prior_latents
    if torch.is_tensor(prior):
        prior_t = prior.to(device=latents.device, dtype=torch.float32)
    else:
        prior_np = np.asarray(prior, dtype=np.float32)
        prior_t = torch.from_numpy(prior_np).to(device=latents.device, dtype=torch.float32)
    if prior_t.ndim == 3:
        prior_t = prior_t.unsqueeze(0)
    if prior_t.shape[-2:] != (h_lat, w_lat):
        prior_t = F.interpolate(prior_t.float(), size=(h_lat, w_lat), mode="bilinear", align_corners=False)

    fill_t = prior_t
    if (clean_latents is not None) and (alpha_prod_t is not None):
        clean_ref = clean_latents.to(device=latents.device, dtype=torch.float32)
        if clean_ref.shape[-2:] != (h_lat, w_lat):
            clean_ref = F.interpolate(clean_ref.float(), size=(h_lat, w_lat), mode="bilinear", align_corners=False)
        alpha = float(np.clip(float(alpha_prod_t), 0.0, 1.0))
        if alpha < 1.0 - 1e-6:
            sqrt_alpha = math.sqrt(alpha)
            sqrt_beta = math.sqrt(max(1.0 - alpha, 1e-8))
            eps_t = (latents.float() - sqrt_alpha * clean_ref) / max(sqrt_beta, 1e-6)
            fill_t = sqrt_alpha * prior_t + sqrt_beta * eps_t

    out = torch.where(hole_t.expand_as(latents), fill_t.to(dtype=latents.dtype), latents)
    return out


def _fill_background_holes_lama_sgf(
    latents,
    source_image_np,
    hole_mask_full,
    forbidden_mask_full,
    device,
    vae=None,
    clean_latents=None,
    alpha_prod_t=None,
):
    if vae is None:
        return None, None, None
    if not isinstance(source_image_np, np.ndarray) or source_image_np.ndim != 3:
        return None, None, None

    h_img, w_img = int(source_image_np.shape[0]), int(source_image_np.shape[1])
    hole_u8 = _mask_to_binary_uint8(hole_mask_full, target_hw=(h_img, w_img))
    if int(np.sum(hole_u8 > 0)) <= 0:
        return latents, source_image_np.astype(np.uint8), None

    lama_mask_u8 = hole_u8
    forbidden_u8 = _mask_to_binary_uint8(forbidden_mask_full, target_hw=(h_img, w_img))
    if int(np.sum(forbidden_u8 > 0)) > 0:
        # 与 Telea 路径一致：forbidden 区域并入 inpaint 输入掩码，避免从主体边缘“偷纹理”。
        forbid_k = 11
        forbid_kernel = np.ones((forbid_k, forbid_k), np.uint8)
        forbidden_dilated = cv2.dilate(forbidden_u8, forbid_kernel, iterations=1)
        lama_mask_u8 = np.logical_or(hole_u8 > 0, forbidden_dilated > 0).astype(np.uint8)

    lama_rgb_raw = _run_lama_inpaint_rgb(source_image_np, lama_mask_u8)
    if lama_rgb_raw is None:
        return None, None, None

    # 仅把“真实背景洞”写回，forbidden 只用于限制 donor，不直接改最终背景底图。
    lama_rgb = source_image_np.astype(np.uint8).copy()
    hole_bool = hole_u8 > 0
    lama_rgb[hole_bool] = lama_rgb_raw[hole_bool]

    clean_prior = _encode_rgb_to_latents_via_vae(
        image_rgb_u8=lama_rgb,
        vae=vae,
        device=device,
        dtype=latents.dtype,
        target_hw=(latents.shape[-2], latents.shape[-1]),
    )
    if clean_prior is None:
        return None, None, None

    composed = _compose_clean_prior_into_current_latents(
        latents=latents,
        clean_prior_latents=clean_prior,
        hole_mask_full=hole_mask_full,
        clean_latents=clean_latents,
        alpha_prod_t=alpha_prod_t,
    )
    print(
        f"[Background Fill][SGF/LaMa] hole={int(np.sum(hole_u8 > 0))}, "
        f"forbidden={int(np.sum(forbidden_u8 > 0))}"
    )
    return composed, lama_rgb, clean_prior.detach().cpu().float()


def fill_holes_with_nearest_boundary(image_input, hole_mask, forbidden_mask=None, device=None, return_residual=False):
    """
    使用“空洞边界最近邻像素”做填充，优先保持局部纹理连续性。
    - hole_mask: 需要填充的区域
    - forbidden_mask: 禁止作为 donor 的区域（仅限制取样来源）
    """
    original_device = device
    original_dtype = None
    original_ndim = None
    np_input_is_hwc = False

    if original_device is None:
        if isinstance(image_input, torch.Tensor):
            original_device = image_input.device
        else:
            original_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(image_input, torch.Tensor):
        x = image_input
        original_dtype = x.dtype
        original_ndim = x.ndim
        if x.ndim == 4:
            feat_np = x[0].detach().cpu().float().numpy()
        elif x.ndim == 3:
            feat_np = x.detach().cpu().float().numpy()
        else:
            raise ValueError("image_input tensor ndim must be 3 or 4")
        is_tensor_output = True
    else:
        arr = np.asarray(image_input)
        original_dtype = arr.dtype
        original_ndim = arr.ndim
        if arr.ndim == 3:
            np_input_is_hwc = True
            feat_np = arr.astype(np.float32).transpose(2, 0, 1)
        elif arr.ndim == 2:
            np_input_is_hwc = False
            feat_np = arr.astype(np.float32)[None, ...]
        else:
            raise ValueError("image_input numpy ndim must be 2 or 3")
        is_tensor_output = False

    C, H, W = feat_np.shape

    def _prep_mask(mask_like):
        if mask_like is None:
            return np.zeros((H, W), dtype=np.uint8)
        if isinstance(mask_like, torch.Tensor):
            m = mask_like.detach().cpu().float()
            if m.ndim == 4:
                m = m[0, 0]
            elif m.ndim == 3:
                m = m[0]
            m = m.numpy()
        else:
            m = np.asarray(mask_like, dtype=np.float32)
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        return (m > 0.5).astype(np.uint8)

    hole_u8 = _prep_mask(hole_mask)
    if int(np.sum(hole_u8)) <= 0:
        residual_zero = np.zeros((H, W), dtype=np.float32)
        if return_residual:
            return image_input, residual_zero, {"strategy": "nearest_boundary", "filled": 0, "reason": "no_hole"}
        return image_input

    forbidden_u8 = _prep_mask(forbidden_mask)
    donor_mask = (hole_u8 == 0)
    if int(np.sum(forbidden_u8)) > 0:
        donor_mask = donor_mask & (forbidden_u8 == 0)

    if int(np.sum(donor_mask)) <= 0:
        residual_full = hole_u8.astype(np.float32)
        if return_residual:
            return image_input, residual_full, {"strategy": "nearest_boundary", "filled": 0, "reason": "no_donor"}
        return image_input

    try:
        from scipy import ndimage
        _, nearest_idx = ndimage.distance_transform_edt(~donor_mask, return_indices=True)
    except Exception:
        filled = fill_background_holes(
            image_input=image_input,
            hole_mask=hole_u8.astype(np.float32),
            forbidden_mask=forbidden_u8.astype(np.float32) if int(np.sum(forbidden_u8)) > 0 else None,
            device=original_device,
            radius=3,
        )
        residual_zero = np.zeros((H, W), dtype=np.float32)
        if return_residual:
            return filled, residual_zero, {"strategy": "fallback_telea", "filled": int(np.sum(hole_u8)), "reason": "ndimage_unavailable"}
        return filled

    hy, hx = np.where(hole_u8 > 0)
    out_np = feat_np.copy()
    if hy.size > 0:
        ny = nearest_idx[0, hy, hx]
        nx = nearest_idx[1, hy, hx]
        out_np[:, hy, hx] = feat_np[:, ny, nx]

    if is_tensor_output:
        out_t = torch.from_numpy(out_np).to(device=original_device, dtype=original_dtype)
        if original_ndim == 4:
            out_t = out_t.unsqueeze(0)
        residual = np.zeros((H, W), dtype=np.float32)
        if return_residual:
            return out_t, residual, {"strategy": "nearest_boundary", "filled": int(hy.size), "reason": "ok"}
        return out_t

    if np_input_is_hwc:
        out_arr = out_np.transpose(1, 2, 0)
    else:
        out_arr = out_np[0]
    if np.issubdtype(original_dtype, np.integer):
        out_arr = np.clip(out_arr, 0, 255).astype(original_dtype)
    else:
        out_arr = out_arr.astype(original_dtype)
    residual = np.zeros((H, W), dtype=np.float32)
    if return_residual:
        return out_arr, residual, {"strategy": "nearest_boundary", "filled": int(hy.size), "reason": "ok"}
    return out_arr


def fill_holes_with_bnni(
    image_input,
    hole_mask,
    donor_mask,
    device=None,
    local_boundary_only=True,
    boundary_smooth=True,
    smooth_band_px=2,
    smooth_sigma=0.9,
    smooth_strength=0.28,
    hf_keep_near=0.78,
    hf_keep_far=0.42,
    hf_near_px=1.25,
    hf_far_px=4.0,
    propagate_iters=72,
    return_debug=False,
):
    """
    BNNI: Boundary Nearest Neighbor Interpolation
    - 仅从 donor_mask 指定类别取样（主体洞->主体 donor，背景洞->背景 donor）
    - 默认优先使用“当前 hole 连通域边界上的 donor”（更局部、更符合方向约束）
    - 当前实现采用 FastDrag-like 原理：
      行/列方向最近邻插值传播（vertical/horizontal）+ 最近邻 donor 残差兜底
    """
    original_device = device
    original_dtype = None
    original_ndim = None
    np_input_is_hwc = False

    if original_device is None:
        if isinstance(image_input, torch.Tensor):
            original_device = image_input.device
        else:
            original_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(image_input, torch.Tensor):
        x = image_input
        original_dtype = x.dtype
        original_ndim = x.ndim
        if x.ndim == 4:
            feat_np = x[0].detach().cpu().float().numpy()
        elif x.ndim == 3:
            feat_np = x.detach().cpu().float().numpy()
        else:
            raise ValueError("image_input tensor ndim must be 3 or 4")
        is_tensor_output = True
    else:
        arr = np.asarray(image_input)
        original_dtype = arr.dtype
        original_ndim = arr.ndim
        if arr.ndim == 3:
            np_input_is_hwc = True
            feat_np = arr.astype(np.float32).transpose(2, 0, 1)
        elif arr.ndim == 2:
            np_input_is_hwc = False
            feat_np = arr.astype(np.float32)[None, ...]
        else:
            raise ValueError("image_input numpy ndim must be 2 or 3")
        is_tensor_output = False

    C, H, W = feat_np.shape

    def _prep_mask(mask_like):
        if mask_like is None:
            return np.zeros((H, W), dtype=np.uint8)
        if isinstance(mask_like, torch.Tensor):
            m = mask_like.detach().cpu().float()
            if m.ndim == 4:
                m = m[0, 0]
            elif m.ndim == 3:
                m = m[0]
            m = m.numpy()
        else:
            m = np.asarray(mask_like, dtype=np.float32)
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        return (m > 0.5).astype(np.uint8)

    hole_u8 = _prep_mask(hole_mask)
    donor_u8 = _prep_mask(donor_mask)
    if int(np.sum(hole_u8)) <= 0:
        if return_debug:
            return image_input, np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32), {"filled": 0, "residual": 0, "strategy": "bnni", "reason": "no_hole"}
        return image_input
    if int(np.sum(donor_u8)) <= 0:
        residual = hole_u8.astype(np.float32)
        if return_debug:
            return image_input, np.zeros((H, W), dtype=np.float32), residual, {"filled": 0, "residual": int(np.sum(residual > 0.5)), "strategy": "bnni", "reason": "no_donor"}
        return image_input

    out_np = feat_np.copy()
    filled_u8 = np.zeros((H, W), dtype=np.uint8)

    try:
        from scipy import ndimage
    except Exception:
        ndimage = None

    if ndimage is None:
        # scipy 不可用时退化到最近边界实现（仍受 donor 限制）
        forbid = (1.0 - donor_u8.astype(np.float32))
        out_fallback, residual_fb, _ = fill_holes_with_nearest_boundary(
            image_input=image_input,
            hole_mask=hole_u8.astype(np.float32),
            forbidden_mask=forbid,
            device=original_device,
            return_residual=True,
        )
        if return_debug:
            filled_mask = np.logical_and(hole_u8 > 0, residual_fb <= 0.5).astype(np.float32)
            return out_fallback, filled_mask, residual_fb.astype(np.float32), {
                "filled": int(np.sum(filled_mask > 0.5)),
                "residual": int(np.sum(residual_fb > 0.5)),
                "strategy": "bnni_fallback",
                "mode": "fastdrag_like",
            }
        return out_fallback

    def _nearest_along_axis(query_coords, donor_coords):
        # donor_coords 已排序，返回每个 query 在该轴上的最近 donor 坐标
        idx = np.searchsorted(donor_coords, query_coords)
        idx_lo = np.clip(idx - 1, 0, donor_coords.size - 1)
        idx_hi = np.clip(idx, 0, donor_coords.size - 1)
        c_lo = donor_coords[idx_lo]
        c_hi = donor_coords[idx_hi]
        choose_lo = np.abs(query_coords - c_lo) <= np.abs(c_hi - query_coords)
        return np.where(choose_lo, c_lo, c_hi)

    def _select_donor_column(known_mask, x, relax_radius):
        donor_ys = np.where(known_mask[:, x])[0]
        if donor_ys.size > 0:
            return int(x), donor_ys

        best = None
        for d in range(1, relax_radius + 1):
            xl = x - d
            xr = x + d
            if xl >= 0:
                ys_l = np.where(known_mask[:, xl])[0]
                if ys_l.size > 0:
                    best = (int(xl), ys_l)
                    break
            if xr < W:
                ys_r = np.where(known_mask[:, xr])[0]
                if ys_r.size > 0:
                    best = (int(xr), ys_r)
                    break
        return best

    def _select_donor_row(known_mask, y, relax_radius):
        donor_xs = np.where(known_mask[y, :])[0]
        if donor_xs.size > 0:
            return int(y), donor_xs

        best = None
        for d in range(1, relax_radius + 1):
            yu = y - d
            yd = y + d
            if yu >= 0:
                xs_u = np.where(known_mask[yu, :])[0]
                if xs_u.size > 0:
                    best = (int(yu), xs_u)
                    break
            if yd < H:
                xs_d = np.where(known_mask[yd, :])[0]
                if xs_d.size > 0:
                    best = (int(yd), xs_d)
                    break
        return best

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hole_u8, connectivity=8)
    for lid in range(1, num_labels):
        if int(stats[lid, cv2.CC_STAT_AREA]) <= 0:
            continue

        comp = (labels == lid).astype(np.uint8)
        comp_bool = comp > 0
        if not np.any(comp_bool):
            continue

        local_donor = donor_u8.copy()
        if bool(local_boundary_only):
            ring = cv2.dilate(comp * 255, np.ones((3, 3), np.uint8), iterations=1)
            ring = np.logical_and(ring > 0, comp <= 0).astype(np.uint8)
            local_donor = np.logical_and(ring > 0, donor_u8 > 0).astype(np.uint8)
            if int(np.sum(local_donor)) <= 0:
                ring_expand = ring.copy()
                for _ in range(4):
                    ring_expand = cv2.dilate(ring_expand * 255, np.ones((3, 3), np.uint8), iterations=1)
                    ring_expand = np.logical_and(ring_expand > 0, comp <= 0).astype(np.uint8)
                    local_donor = np.logical_and(ring_expand > 0, donor_u8 > 0).astype(np.uint8)
                    if int(np.sum(local_donor)) > 0:
                        break
            if int(np.sum(local_donor)) <= 0:
                local_donor = donor_u8.copy()

        if int(np.sum(local_donor)) <= 0:
            continue

        # FastDrag-like: 沿列/沿行做最近 donor 插值传播；允许已填像素继续作为 donor 扩散
        known = local_donor > 0
        relax_radius = max(2, min(32, int(round(0.06 * min(H, W)))))
        it_cap = int(max(8, min(int(propagate_iters), 2 * max(H, W))))

        for _ in range(it_cap):
            remaining = np.logical_and(comp_bool, ~known)
            if not np.any(remaining):
                break

            changed = False

            # pass-1: vertical
            xs = np.where(np.any(remaining, axis=0))[0]
            for x in xs.tolist():
                ys = np.where(remaining[:, x])[0]
                if ys.size <= 0:
                    continue
                donor_sel = _select_donor_column(known, int(x), relax_radius)
                if donor_sel is None:
                    continue
                donor_x, donor_ys = donor_sel
                y_src = _nearest_along_axis(ys, donor_ys)
                out_np[:, ys, x] = out_np[:, y_src, donor_x]
                known[ys, x] = True
                filled_u8[ys, x] = 1
                changed = True

            remaining = np.logical_and(comp_bool, ~known)
            if not np.any(remaining):
                break

            # pass-2: horizontal
            ys = np.where(np.any(remaining, axis=1))[0]
            for y in ys.tolist():
                xs = np.where(remaining[y, :])[0]
                if xs.size <= 0:
                    continue
                donor_sel = _select_donor_row(known, int(y), relax_radius)
                if donor_sel is None:
                    continue
                donor_y, donor_xs = donor_sel
                x_src = _nearest_along_axis(xs, donor_xs)
                out_np[:, y, xs] = out_np[:, donor_y, x_src]
                known[y, xs] = True
                filled_u8[y, xs] = 1
                changed = True

            if not changed:
                break

        # 残差兜底：最近 donor 直接赋值（仍限制在 local_donor 语义域）
        remain = np.logical_and(comp_bool, ~known)
        if np.any(remain):
            donor_for_fallback = known
            if int(np.sum(donor_for_fallback)) > 0:
                _, nearest_idx = ndimage.distance_transform_edt(~donor_for_fallback, return_indices=True)
                ry, rx = np.where(remain)
                ny = nearest_idx[0, ry, rx]
                nx = nearest_idx[1, ry, rx]
                valid = donor_for_fallback[ny, nx]
                if np.any(valid):
                    ryv, rxv = ry[valid], rx[valid]
                    nyv, nxv = ny[valid], nx[valid]
                    out_np[:, ryv, rxv] = out_np[:, nyv, nxv]
                    known[ryv, rxv] = True
                    filled_u8[ryv, rxv] = 1

    residual_u8 = np.logical_and(hole_u8 > 0, filled_u8 <= 0).astype(np.float32)

    # BNNI 填充后仅在“填充边界带”做轻量平滑，减少拼接块感；
    # 深处纹理不动，避免整体发糊。
    if bool(boundary_smooth) and int(np.sum(filled_u8 > 0)) > 0:
        fill_bin = (filled_u8 > 0).astype(np.uint8)
        dist_in = cv2.distanceTransform(fill_bin, cv2.DIST_L2, 3)
        band = max(1, int(smooth_band_px))
        alpha = np.clip((float(band) - dist_in) / max(1.0, float(band)), 0.0, 1.0).astype(np.float32)
        alpha = alpha * float(np.clip(smooth_strength, 0.0, 1.0))
        if np.any(alpha > 1e-5):
            sigma = max(0.1, float(smooth_sigma))
            for ci in range(C):
                ch = out_np[ci].astype(np.float32)
                ch_blur = cv2.GaussianBlur(ch, (0, 0), sigmaX=sigma, sigmaY=sigma)
                ch = ch * (1.0 - alpha) + ch_blur * alpha
                out_np[ci] = ch

    if is_tensor_output:
        out_t = torch.from_numpy(out_np).to(device=original_device, dtype=original_dtype)
        if original_ndim == 4:
            out_t = out_t.unsqueeze(0)
        if return_debug:
            return out_t, filled_u8.astype(np.float32), residual_u8.astype(np.float32), {
                "filled": int(np.sum(filled_u8 > 0)),
                "residual": int(np.sum(residual_u8 > 0.5)),
                "strategy": "bnni",
                "local_boundary_only": bool(local_boundary_only),
                "mode": "fastdrag_like",
            }
        return out_t

    if np_input_is_hwc:
        out_arr = out_np.transpose(1, 2, 0)
    else:
        out_arr = out_np[0]
    if np.issubdtype(original_dtype, np.integer):
        out_arr = np.clip(out_arr, 0, 255).astype(original_dtype)
    else:
        out_arr = out_arr.astype(original_dtype)
    if return_debug:
        return out_arr, filled_u8.astype(np.float32), residual_u8.astype(np.float32), {
            "filled": int(np.sum(filled_u8 > 0)),
            "residual": int(np.sum(residual_u8 > 0.5)),
            "strategy": "bnni",
            "local_boundary_only": bool(local_boundary_only),
            "mode": "fastdrag_like",
        }
    return out_arr


def fill_holes_with_bnni_scanline(
    image_input,
    hole_mask,
    donor_mask,
    device=None,
    primary_axis="vertical",
    secondary_axis="horizontal",
    relax_radius=8,
    return_debug=False,
):
    """
    扫描线版 BNNI（强调“线性拉伸感”）：
    - 仅从 donor_mask 取样
    - 先按 primary_axis 扫描，再按 secondary_axis 扫描补残差
    - 常用于主体内部空洞，先形成可见线条，再交给普通 BNNI 补边角
    """
    original_device = device
    original_dtype = None
    original_ndim = None
    np_input_is_hwc = False

    if original_device is None:
        if isinstance(image_input, torch.Tensor):
            original_device = image_input.device
        else:
            original_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(image_input, torch.Tensor):
        x = image_input
        original_dtype = x.dtype
        original_ndim = x.ndim
        if x.ndim == 4:
            feat_np = x[0].detach().cpu().float().numpy()
        elif x.ndim == 3:
            feat_np = x.detach().cpu().float().numpy()
        else:
            raise ValueError("image_input tensor ndim must be 3 or 4")
        is_tensor_output = True
    else:
        arr = np.asarray(image_input)
        original_dtype = arr.dtype
        original_ndim = arr.ndim
        if arr.ndim == 3:
            np_input_is_hwc = True
            feat_np = arr.astype(np.float32).transpose(2, 0, 1)
        elif arr.ndim == 2:
            np_input_is_hwc = False
            feat_np = arr.astype(np.float32)[None, ...]
        else:
            raise ValueError("image_input numpy ndim must be 2 or 3")
        is_tensor_output = False

    C, H, W = feat_np.shape

    def _prep_mask(mask_like):
        if mask_like is None:
            return np.zeros((H, W), dtype=np.uint8)
        if isinstance(mask_like, torch.Tensor):
            m = mask_like.detach().cpu().float()
            if m.ndim == 4:
                m = m[0, 0]
            elif m.ndim == 3:
                m = m[0]
            m = m.numpy()
        else:
            m = np.asarray(mask_like, dtype=np.float32)
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        return (m > 0.5).astype(np.uint8)

    hole_u8 = _prep_mask(hole_mask)
    donor_u8 = _prep_mask(donor_mask)
    if int(np.sum(hole_u8)) <= 0:
        if return_debug:
            return image_input, np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32), {"filled": 0, "residual": 0, "strategy": "bnni_scanline", "reason": "no_hole"}
        return image_input
    if int(np.sum(donor_u8)) <= 0:
        residual = hole_u8.astype(np.float32)
        if return_debug:
            return image_input, np.zeros((H, W), dtype=np.float32), residual, {"filled": 0, "residual": int(np.sum(residual > 0.5)), "strategy": "bnni_scanline", "reason": "no_donor"}
        return image_input

    src_np = feat_np
    out_np = feat_np.copy()
    filled_u8 = np.zeros((H, W), dtype=np.uint8)

    def _scan_axis(remain_comp_u8, axis):
        nonlocal out_np, filled_u8
        if axis not in ("vertical", "horizontal"):
            return remain_comp_u8

        remain = remain_comp_u8.copy()
        if axis == "vertical":
            xs = np.where(np.any(remain > 0, axis=0))[0]
            for x in xs.tolist():
                ys = np.where(remain[:, x] > 0)[0]
                if ys.size <= 0:
                    continue
                donor_ys = np.where(donor_u8[:, x] > 0)[0]
                donor_x = int(x)
                if donor_ys.size <= 0:
                    found = False
                    for d in range(1, int(max(1, relax_radius)) + 1):
                        xl = x - d
                        xr = x + d
                        cand = []
                        if xl >= 0:
                            ys_l = np.where(donor_u8[:, xl] > 0)[0]
                            if ys_l.size > 0:
                                cand.append((d, int(xl), ys_l))
                        if xr < W:
                            ys_r = np.where(donor_u8[:, xr] > 0)[0]
                            if ys_r.size > 0:
                                cand.append((d, int(xr), ys_r))
                        if len(cand) > 0:
                            _, donor_x, donor_ys = sorted(cand, key=lambda t: (t[0], abs(t[1] - x)))[0]
                            found = True
                            break
                    if not found:
                        continue
                idx = np.searchsorted(donor_ys, ys)
                idx_lo = np.clip(idx - 1, 0, donor_ys.size - 1)
                idx_hi = np.clip(idx, 0, donor_ys.size - 1)
                y_lo = donor_ys[idx_lo]
                y_hi = donor_ys[idx_hi]
                choose_lo = np.abs(ys - y_lo) <= np.abs(y_hi - ys)
                y_src = np.where(choose_lo, y_lo, y_hi)
                out_np[:, ys, x] = src_np[:, y_src, donor_x]
                filled_u8[ys, x] = 1
                remain[ys, x] = 0
            return remain

        ys = np.where(np.any(remain > 0, axis=1))[0]
        for y in ys.tolist():
            xs = np.where(remain[y, :] > 0)[0]
            if xs.size <= 0:
                continue
            donor_xs = np.where(donor_u8[y, :] > 0)[0]
            donor_y = int(y)
            if donor_xs.size <= 0:
                found = False
                for d in range(1, int(max(1, relax_radius)) + 1):
                    yu = y - d
                    yd = y + d
                    cand = []
                    if yu >= 0:
                        xs_u = np.where(donor_u8[yu, :] > 0)[0]
                        if xs_u.size > 0:
                            cand.append((d, int(yu), xs_u))
                    if yd < H:
                        xs_d = np.where(donor_u8[yd, :] > 0)[0]
                        if xs_d.size > 0:
                            cand.append((d, int(yd), xs_d))
                    if len(cand) > 0:
                        _, donor_y, donor_xs = sorted(cand, key=lambda t: (t[0], abs(t[1] - y)))[0]
                        found = True
                        break
                if not found:
                    continue
            idx = np.searchsorted(donor_xs, xs)
            idx_lo = np.clip(idx - 1, 0, donor_xs.size - 1)
            idx_hi = np.clip(idx, 0, donor_xs.size - 1)
            x_lo = donor_xs[idx_lo]
            x_hi = donor_xs[idx_hi]
            choose_lo = np.abs(xs - x_lo) <= np.abs(x_hi - xs)
            x_src = np.where(choose_lo, x_lo, x_hi)
            out_np[:, y, xs] = src_np[:, donor_y, x_src]
            filled_u8[y, xs] = 1
            remain[y, xs] = 0
        return remain

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hole_u8, connectivity=8)
    for lid in range(1, num_labels):
        if int(stats[lid, cv2.CC_STAT_AREA]) <= 0:
            continue
        comp = (labels == lid).astype(np.uint8)
        remain = comp.copy()

        axes = []
        if primary_axis in ("vertical", "horizontal"):
            axes.append(primary_axis)
        if secondary_axis in ("vertical", "horizontal") and secondary_axis not in axes:
            axes.append(secondary_axis)
        if len(axes) == 0:
            axes = ["vertical", "horizontal"]

        for ax in axes:
            remain = _scan_axis(remain, ax)
            if int(np.sum(remain)) <= 0:
                break

    residual_u8 = np.logical_and(hole_u8 > 0, filled_u8 <= 0).astype(np.float32)

    if is_tensor_output:
        out_t = torch.from_numpy(out_np).to(device=original_device, dtype=original_dtype)
        if original_ndim == 4:
            out_t = out_t.unsqueeze(0)
        if return_debug:
            return out_t, filled_u8.astype(np.float32), residual_u8.astype(np.float32), {
                "filled": int(np.sum(filled_u8 > 0)),
                "residual": int(np.sum(residual_u8 > 0.5)),
                "strategy": "bnni_scanline",
                "primary_axis": str(primary_axis),
                "secondary_axis": str(secondary_axis),
            }
        return out_t

    if np_input_is_hwc:
        out_arr = out_np.transpose(1, 2, 0)
    else:
        out_arr = out_np[0]
    if np.issubdtype(original_dtype, np.integer):
        out_arr = np.clip(out_arr, 0, 255).astype(original_dtype)
    else:
        out_arr = out_arr.astype(original_dtype)
    if return_debug:
        return out_arr, filled_u8.astype(np.float32), residual_u8.astype(np.float32), {
            "filled": int(np.sum(filled_u8 > 0)),
            "residual": int(np.sum(residual_u8 > 0.5)),
            "strategy": "bnni_scanline",
            "primary_axis": str(primary_axis),
            "secondary_axis": str(secondary_axis),
        }
    return out_arr


def _shift_no_wrap_4d(src, dy, dx):
    """
    无环绕平移采样：dst(y,x)=src(y+dy,x+dx)。
    """
    if (not torch.is_tensor(src)) or src.ndim != 4:
        raise ValueError("src must be Tensor(1,C,H,W)")
    _, _, H, W = src.shape
    out = torch.zeros_like(src)
    if dy >= 0:
        src_y0, src_y1 = dy, H
        dst_y0, dst_y1 = 0, H - dy
    else:
        src_y0, src_y1 = 0, H + dy
        dst_y0, dst_y1 = -dy, H
    if dx >= 0:
        src_x0, src_x1 = dx, W
        dst_x0, dst_x1 = 0, W - dx
    else:
        src_x0, src_x1 = 0, W + dx
        dst_x0, dst_x1 = -dx, W
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return out
    out[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = src[:, :, src_y0:src_y1, src_x0:src_x1]
    return out


def fill_holes_with_stretch_propagation(
    image_input,
    hole_mask,
    donor_mask,
    flow_xy=None,
    device=None,
    max_iters=96,
    directional_strength=0.75,
    local_band_ratio=0.12,
    local_band_min=3,
    local_band_max=24,
    return_debug=False,
):
    """
    拉伸式补全（非整块位移）：
    1) 仅允许空洞邻域 donor（近场）
    2) 迭代把边界已知特征向空洞内部传播（类似被拉伸）
    3) 方向权重由 flow_xy 提供，仅作为“传播偏置”而非直接平移
    """
    original_device = device
    original_dtype = None
    original_ndim = None
    np_input_is_hwc = False

    if original_device is None:
        if isinstance(image_input, torch.Tensor):
            original_device = image_input.device
        else:
            original_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(image_input, torch.Tensor):
        x = image_input
        original_dtype = x.dtype
        original_ndim = x.ndim
        if x.ndim == 4:
            feat_np = x[0].detach().cpu().float().numpy()
        elif x.ndim == 3:
            feat_np = x.detach().cpu().float().numpy()
        else:
            raise ValueError("image_input tensor ndim must be 3 or 4")
        is_tensor_output = True
    else:
        arr = np.asarray(image_input)
        original_dtype = arr.dtype
        original_ndim = arr.ndim
        if arr.ndim == 3:
            np_input_is_hwc = True
            feat_np = arr.astype(np.float32).transpose(2, 0, 1)
        elif arr.ndim == 2:
            np_input_is_hwc = False
            feat_np = arr.astype(np.float32)[None, ...]
        else:
            raise ValueError("image_input numpy ndim must be 2 or 3")
        is_tensor_output = False

    C, H, W = feat_np.shape

    def _prep_mask(mask_like):
        if mask_like is None:
            return np.zeros((H, W), dtype=np.uint8)
        if isinstance(mask_like, torch.Tensor):
            m = mask_like.detach().cpu().float()
            if m.ndim == 4:
                m = m[0, 0]
            elif m.ndim == 3:
                m = m[0]
            m = m.numpy()
        else:
            m = np.asarray(mask_like, dtype=np.float32)
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        return (m > 0.5).astype(np.uint8)

    hole_u8 = _prep_mask(hole_mask)
    donor_u8 = _prep_mask(donor_mask)
    if int(np.sum(hole_u8)) <= 0:
        if return_debug:
            return image_input, np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32), {"filled": 0, "residual": 0, "iters": 0}
        return image_input

    # donor 仅保留“空洞附近”区域，避免远距离纹理污染
    ys, xs = np.where(hole_u8 > 0)
    if ys.size > 0:
        span = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1), 1)
    else:
        span = 1
    band = int(np.clip(round(float(span) * float(local_band_ratio)), int(local_band_min), int(local_band_max)))
    k = max(3, int(2 * band + 1))
    if k % 2 == 0:
        k += 1
    near_hole = cv2.dilate((hole_u8 * 255).astype(np.uint8), np.ones((k, k), np.uint8), iterations=1) > 0
    donor_local_u8 = ((donor_u8 > 0) & near_hole & (hole_u8 == 0)).astype(np.uint8)
    if int(np.sum(donor_local_u8)) <= 0:
        donor_local_u8 = ((donor_u8 > 0) & (hole_u8 == 0)).astype(np.uint8)
    if int(np.sum(donor_local_u8)) <= 0:
        if return_debug:
            return image_input, np.zeros((H, W), dtype=np.float32), hole_u8.astype(np.float32), {"filled": 0, "residual": int(np.sum(hole_u8)), "iters": 0}
        return image_input

    feat_t = torch.from_numpy(feat_np).to(device=original_device, dtype=torch.float32).unsqueeze(0)
    hole_t = torch.from_numpy(hole_u8.astype(np.float32)).to(device=original_device).view(1, 1, H, W) > 0.5
    known_t = torch.from_numpy(donor_local_u8.astype(np.float32)).to(device=original_device).view(1, 1, H, W) > 0.5
    known_t = known_t & (~hole_t)

    out_t = feat_t.clone()
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1),  (1, 0),  (1, 1)]

    flow_x = float(flow_xy[0]) if flow_xy is not None else 0.0
    flow_y = float(flow_xy[1]) if flow_xy is not None else 0.0
    flow_norm = math.sqrt(flow_x * flow_x + flow_y * flow_y)
    if flow_norm > 1e-6:
        ux, uy = flow_x / flow_norm, flow_y / flow_norm
    else:
        ux, uy = 0.0, 0.0

    kernel_neighbors = torch.ones((1, 1, 3, 3), device=original_device, dtype=torch.float32)
    kernel_neighbors[:, :, 1, 1] = 0.0
    iters_used = 0

    for it in range(int(max_iters)):
        remaining = hole_t & (~known_t)
        if int(remaining.sum().item()) <= 0:
            iters_used = it
            break
        neigh_cnt = F.conv2d(known_t.float(), kernel_neighbors, padding=1)
        frontier = remaining & (neigh_cnt > 0)
        if int(frontier.sum().item()) <= 0:
            iters_used = it
            break

        sum_feat = torch.zeros_like(out_t)
        sum_w = torch.zeros((1, 1, H, W), device=original_device, dtype=torch.float32)
        for dy, dx in offsets:
            neigh_valid = _shift_no_wrap_4d(known_t.float(), dy, dx) > 0.5
            if int(neigh_valid.sum().item()) <= 0:
                continue
            neigh_feat = _shift_no_wrap_4d(out_t, dy, dx)
            base = 1.0 / (math.sqrt(float(dx * dx + dy * dy)) + 1e-6)

            if flow_norm > 1e-6:
                vec_x = float(-dx)
                vec_y = float(-dy)
                vec_n = math.sqrt(vec_x * vec_x + vec_y * vec_y) + 1e-6
                align = max(0.0, (ux * (vec_x / vec_n) + uy * (vec_y / vec_n)))
                dir_factor = 1.0 + float(directional_strength) * align
            else:
                dir_factor = 1.0

            w = neigh_valid.float() * (base * dir_factor)
            sum_feat = sum_feat + neigh_feat * w
            sum_w = sum_w + w

        can_fill = frontier & (sum_w > 1e-6)
        if int(can_fill.sum().item()) <= 0:
            iters_used = it
            break
        fill_val = sum_feat / (sum_w + 1e-6)
        out_t = torch.where(can_fill.expand_as(out_t), fill_val, out_t)
        known_t = known_t | can_fill
        iters_used = it + 1

    filled_t = hole_t & known_t
    residual_t = hole_t & (~known_t)
    filled_u8 = filled_t.float().squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    residual_u8 = residual_t.float().squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)

    # 最近边界兜底（仍限制 donor_local，不走 Telea）
    if int(np.sum(residual_u8 > 0.5)) > 0:
        forbid_local = (1.0 - donor_local_u8.astype(np.float32))
        out_t, residual2_u8, _ = fill_holes_with_nearest_boundary(
            image_input=out_t,
            hole_mask=residual_u8,
            forbidden_mask=forbid_local,
            device=original_device,
            return_residual=True,
        )
        filled_u8 = np.logical_or(filled_u8 > 0.5, np.logical_and(residual_u8 > 0.5, residual2_u8 <= 0.5)).astype(np.float32)
        residual_u8 = residual2_u8.astype(np.float32)

    out_np = out_t.squeeze(0).detach().cpu().numpy()

    if is_tensor_output:
        out_ret = torch.from_numpy(out_np).to(device=original_device, dtype=original_dtype)
        if original_ndim == 4:
            out_ret = out_ret.unsqueeze(0)
        if return_debug:
            return out_ret, filled_u8, residual_u8, {
                "filled": int(np.sum(filled_u8 > 0.5)),
                "residual": int(np.sum(residual_u8 > 0.5)),
                "iters": int(iters_used),
                "band": int(band),
            }
        return out_ret

    if np_input_is_hwc:
        out_arr = out_np.transpose(1, 2, 0)
    else:
        out_arr = out_np[0]
    if np.issubdtype(original_dtype, np.integer):
        out_arr = np.clip(out_arr, 0, 255).astype(original_dtype)
    else:
        out_arr = out_arr.astype(original_dtype)
    if return_debug:
        return out_arr, filled_u8, residual_u8, {
            "filled": int(np.sum(filled_u8 > 0.5)),
            "residual": int(np.sum(residual_u8 > 0.5)),
            "iters": int(iters_used),
            "band": int(band),
        }
    return out_arr


def _mask_to_binary_uint8(mask_input, target_hw=None):
    """将输入 mask 统一转换为 0/1 的 uint8(H, W)。"""
    if mask_input is None:
        if target_hw is None:
            return np.zeros((1, 1), dtype=np.uint8)
        return np.zeros(target_hw, dtype=np.uint8)

    if isinstance(mask_input, torch.Tensor):
        mask_np = mask_input.detach().cpu().float().numpy()
    else:
        mask_np = np.asarray(mask_input)

    if mask_np.ndim == 4:
        mask_np = mask_np[0, 0]
    elif mask_np.ndim == 3:
        # 兼容 [1,H,W] / [H,W,1] / [C,H,W]
        if mask_np.shape[0] <= 4 and mask_np.shape[1] > 4 and mask_np.shape[2] > 4:
            mask_np = mask_np[0]
        else:
            mask_np = mask_np[..., 0]

    if target_hw is not None and tuple(mask_np.shape[:2]) != tuple(target_hw):
        h_t, w_t = int(target_hw[0]), int(target_hw[1])
        mask_np = cv2.resize(mask_np.astype(np.float32), (w_t, h_t), interpolation=cv2.INTER_NEAREST)

    max_v = float(np.max(mask_np)) if mask_np.size > 0 else 0.0
    thr = 0.5 if max_v <= 1.0 else 127.0
    return (mask_np > thr).astype(np.uint8)


def infer_subject_fill_scope(mask_input, close_ratio=0.10, max_close=31, max_fill_dist_ratio=0.16):
    """
    自动推断主体最终范围，并返回需要补洞的区域（仅主体域内）。
    返回:
      scope_mask: float32(H,W), 0/1
      hole_mask:  float32(H,W), 0/1
      info:       调试信息
    """
    mask_u8 = _mask_to_binary_uint8(mask_input)
    H, W = mask_u8.shape[:2]
    fg_area = int(np.sum(mask_u8 > 0))
    if fg_area == 0:
        empty = np.zeros((H, W), dtype=np.float32)
        return empty, empty, {
            "kernel": 0,
            "mask_area": 0,
            "scope_area": 0,
            "hole_pixels": 0,
            "max_fill_dist": 0.0,
        }

    ys, xs = np.where(mask_u8 > 0)
    bbox_w = int(xs.max() - xs.min() + 1)
    bbox_h = int(ys.max() - ys.min() + 1)
    obj_span = max(bbox_w, bbox_h)

    k = max(3, int(round(float(obj_span) * float(close_ratio))))
    if k % 2 == 0:
        k += 1
    k = int(np.clip(k, 3, int(max_close)))
    if k % 2 == 0:
        k = max(3, k - 1)
    kernel_close = np.ones((k, k), np.uint8)

    mask_255 = mask_u8 * 255
    mask_closed = cv2.morphologyEx(mask_255, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scope_u8 = np.zeros_like(mask_u8)
    if contours:
        cv2.drawContours(scope_u8, contours, -1, color=1, thickness=-1)
    else:
        scope_u8 = (mask_closed > 0).astype(np.uint8)

    # 仅允许在原主体附近扩展，防止跨越到远处背景区域。
    outside_u8 = (mask_u8 == 0).astype(np.uint8)
    dist_to_mask = cv2.distanceTransform(outside_u8, cv2.DIST_L2, 5)
    max_fill_dist = max(2.0, float(obj_span) * float(max_fill_dist_ratio))
    candidate_new = (scope_u8 > 0) & (mask_u8 == 0) & (dist_to_mask <= max_fill_dist)

    scope_u8 = np.where((mask_u8 > 0) | candidate_new, 1, 0).astype(np.uint8)
    scope_four_dir_inner = _compute_four_direction_subject_interior_u8(scope_u8)
    hole_u8 = np.logical_and((scope_u8 > 0) & (mask_u8 == 0), scope_four_dir_inner > 0).astype(np.uint8)

    info = {
        "kernel": int(k),
        "mask_area": int(np.sum(mask_u8)),
        "scope_area": int(np.sum(scope_u8)),
        "hole_pixels": int(np.sum(hole_u8)),
        "max_fill_dist": float(max_fill_dist),
        "hole_rule": "four_direction_subject_enclosed",
    }
    return scope_u8.astype(np.float32), hole_u8.astype(np.float32), info


def _shift_tensor_integer(feat_4d, shift_x, shift_y):
    """
    对 4D 张量做整数平移（零填充），并返回有效位掩码。
    feat_4d: (1, C, H, W)
    返回:
      shifted:    (1, C, H, W)
      valid_mask: (1, 1, H, W) bool，表示目标像素是否由源像素搬运而来
    """
    if not torch.is_tensor(feat_4d) or feat_4d.ndim != 4:
        raise ValueError("feat_4d must be a torch Tensor with shape (1,C,H,W)")

    _, _, H, W = feat_4d.shape
    dx = int(round(float(shift_x)))
    dy = int(round(float(shift_y)))

    shifted = torch.zeros_like(feat_4d)
    valid_mask = torch.zeros((1, 1, H, W), device=feat_4d.device, dtype=torch.bool)

    src_x0 = max(0, -dx)
    src_x1 = min(W, W - dx)
    src_y0 = max(0, -dy)
    src_y1 = min(H, H - dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(W, W + dx)
    dst_y0 = max(0, dy)
    dst_y1 = min(H, H + dy)

    if src_x1 <= src_x0 or src_y1 <= src_y0 or dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return shifted, valid_mask

    shifted[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = feat_4d[:, :, src_y0:src_y1, src_x0:src_x1]
    valid_mask[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = True
    return shifted, valid_mask


def _iterative_masked_neighbor_fill(feat_4d, valid_mask_4d, fill_mask_4d, max_iters=64):
    """
    在给定填充域内做邻域扩散补全（不跨域）：
    - 仅允许 fill_mask 内像素被填充
    - 从 valid_mask 的已知像素向外扩散
    """
    if not torch.is_tensor(feat_4d) or feat_4d.ndim != 4:
        raise ValueError("feat_4d must be Tensor(1,C,H,W)")
    if not torch.is_tensor(valid_mask_4d) or valid_mask_4d.ndim != 4:
        raise ValueError("valid_mask_4d must be Tensor(1,1,H,W)")
    if not torch.is_tensor(fill_mask_4d) or fill_mask_4d.ndim != 4:
        raise ValueError("fill_mask_4d must be Tensor(1,1,H,W)")

    out = feat_4d
    valid = valid_mask_4d.bool().clone()
    fill_domain = fill_mask_4d.bool()

    C = int(out.shape[1])
    dev = out.device
    dtype = out.dtype
    kernel_neighbors = torch.ones((1, 1, 3, 3), device=dev, dtype=dtype)
    kernel_neighbors[:, :, 1, 1] = 0.0
    kernel_feat = kernel_neighbors.expand(C, 1, 3, 3).contiguous()

    filled_total = 0
    for _ in range(int(max_iters)):
        remaining = fill_domain & (~valid)
        if int(remaining.sum().item()) == 0:
            break

        valid_f = valid.to(dtype)
        neighbor_cnt = F.conv2d(valid_f, kernel_neighbors, padding=1)
        updatable = remaining & (neighbor_cnt > 0)
        if int(updatable.sum().item()) == 0:
            break

        neighbor_sum = F.conv2d(out * valid_f, kernel_feat, padding=1, groups=C)
        fill_val = neighbor_sum / (neighbor_cnt + 1e-6)
        out = torch.where(updatable.expand_as(out), fill_val, out)
        valid = valid | updatable
        filled_total += int(updatable.sum().item())

    return out, valid, filled_total


def _estimate_motion_shift_latent_from_points(handles_xy, targets_xy, scale_x, scale_y):
    """
    从图像坐标控制点估计该连通域平均位移（latent 像素）。
    返回 (shift_x_lat, shift_y_lat) 或 None。
    """
    h_pts = _sanitize_xy_points(handles_xy)
    t_pts = _sanitize_xy_points(targets_xy)
    n = min(h_pts.shape[0], t_pts.shape[0])
    if n <= 0:
        return None
    delta = t_pts[:n] - h_pts[:n]
    if delta.shape[0] == 0:
        return None
    mean_dx_img = float(np.mean(delta[:, 0]))
    mean_dy_img = float(np.mean(delta[:, 1]))
    return (
        mean_dx_img * float(scale_x),
        mean_dy_img * float(scale_y),
    )


def fill_subject_holes_within_scope(
    image_input,
    subject_mask,
    device=None,
    radius=5,
    close_ratio=0.10,
    max_close=31,
    max_fill_dist_ratio=0.16,
    source_image_input=None,
    source_subject_mask=None,
    source_use_background=False,
    motion_shift_xy=None,
    diffusion_max_iters=64,
    return_hole_mask=False,
):
    """
    在自动识别到的主体范围内补洞，并显式禁止从主体外部取样。
    返回:
      filled_image, refined_scope_mask, scope_info
    """
    scope_mask, hole_mask, scope_info = infer_subject_fill_scope(
        subject_mask,
        close_ratio=close_ratio,
        max_close=max_close,
        max_fill_dist_ratio=max_fill_dist_ratio,
    )
    if int(scope_info.get("hole_pixels", 0)) <= 0:
        if return_hole_mask:
            return image_input, scope_mask, scope_info, hole_mask
        return image_input, scope_mask, scope_info

    # ===== Subject-BNNI 策略（不使用位移搬运）=====
    # Stage A: (disabled) motion/source transport
    # Stage B1: 局部 BNNI
    # Stage B2: 全局 BNNI 残差补齐

    # 非 Tensor 输入（如 RGB np）沿用原 Telea 流程
    if not isinstance(image_input, torch.Tensor):
        forbidden_mask = (scope_mask <= 0.5).astype(np.float32)
        filled = fill_background_holes(
            image_input=image_input,
            hole_mask=hole_mask,
            forbidden_mask=forbidden_mask,
            device=device,
            radius=radius,
        )
        if return_hole_mask:
            return filled, scope_mask, scope_info, hole_mask
        return filled, scope_mask, scope_info

    feat = image_input
    feat_was_3d = False
    if feat.ndim == 3:
        feat = feat.unsqueeze(0)
        feat_was_3d = True
    if feat.ndim != 4:
        forbidden_mask = (scope_mask <= 0.5).astype(np.float32)
        filled = fill_background_holes(
            image_input=image_input,
            hole_mask=hole_mask,
            forbidden_mask=forbidden_mask,
            device=device,
            radius=radius,
        )
        if return_hole_mask:
            return filled, scope_mask, scope_info, hole_mask
        return filled, scope_mask, scope_info

    feat_dtype = feat.dtype
    feat_dev = feat.device
    feat_work = feat.float().clone()
    _, _, Hf, Wf = feat_work.shape

    scope_u8 = _mask_to_binary_uint8(scope_mask, target_hw=(Hf, Wf))
    hole_u8 = _mask_to_binary_uint8(hole_mask, target_hw=(Hf, Wf))
    scope_t = torch.from_numpy(scope_u8.astype(np.float32)).to(device=feat_dev).view(1, 1, Hf, Wf) > 0.5
    hole_t = torch.from_numpy(hole_u8.astype(np.float32)).to(device=feat_dev).view(1, 1, Hf, Wf) > 0.5
    valid_t = scope_t & (~hole_t)

    stage_transport_filled = 0
    stage_diffuse_filled = 0
    stage_telea_filled = 0
    stage_bnni_scanline_filled = 0
    stage_bnni_local_filled = 0
    stage_bnni_global_filled = 0

    # Stage B0: 扫描线 BNNI（先竖后横，先建立线条结构）
    donor_bnni = valid_t.float().squeeze(0).squeeze(0)
    hole_bnni = hole_t.float().squeeze(0).squeeze(0)
    feat_work, filled_bnni_scanline, _, _ = fill_holes_with_bnni_scanline(
        image_input=feat_work,
        hole_mask=hole_bnni,
        donor_mask=donor_bnni,
        device=feat_dev,
        primary_axis="vertical",
        secondary_axis=None,
        return_debug=True,
    )
    if isinstance(feat_work, torch.Tensor):
        feat_work = feat_work.to(dtype=torch.float32, device=feat_dev)
    filled_bnni_scanline_u8 = _mask_to_binary_uint8(filled_bnni_scanline, target_hw=(Hf, Wf))
    filled_bnni_scanline_t = torch.from_numpy(filled_bnni_scanline_u8.astype(np.float32)).to(device=feat_dev).view(1, 1, Hf, Wf) > 0.5
    valid_t = valid_t | filled_bnni_scanline_t
    stage_bnni_scanline_filled = int(filled_bnni_scanline_t.sum().item())

    # Stage B1: 局部 BNNI（补扫描线剩余）
    remain_t = hole_t & (~valid_t)
    remain_n = int(remain_t.sum().item())
    if remain_n > 0:
        feat_work, filled_bnni_local, _, _ = fill_holes_with_bnni(
            image_input=feat_work,
            hole_mask=remain_t.float().squeeze(0).squeeze(0),
            donor_mask=donor_bnni,
            device=feat_dev,
            local_boundary_only=True,
            boundary_smooth=False,
            smooth_band_px=1,
            smooth_sigma=0.6,
            smooth_strength=0.0,
            hf_keep_near=1.0,
            hf_keep_far=0.92,
            propagate_iters=64,
            return_debug=True,
        )
        if isinstance(feat_work, torch.Tensor):
            feat_work = feat_work.to(dtype=torch.float32, device=feat_dev)
        filled_bnni_local_u8 = _mask_to_binary_uint8(filled_bnni_local, target_hw=(Hf, Wf))
        filled_bnni_local_t = torch.from_numpy(filled_bnni_local_u8.astype(np.float32)).to(device=feat_dev).view(1, 1, Hf, Wf) > 0.5
        valid_t = valid_t | filled_bnni_local_t
        stage_bnni_local_filled = int(filled_bnni_local_t.sum().item())

    # Stage B2: 全局 BNNI 残差补齐（仍然 donor=主体内部，禁止主体外取样）
    remain_t = hole_t & (~valid_t)
    remain_n = int(remain_t.sum().item())
    if remain_n > 0:
        feat_work, filled_bnni_global, _, _ = fill_holes_with_bnni(
            image_input=feat_work,
            hole_mask=remain_t.float().squeeze(0).squeeze(0),
            donor_mask=donor_bnni,
            device=feat_dev,
            local_boundary_only=False,
            boundary_smooth=False,
            smooth_band_px=1,
            smooth_sigma=0.6,
            smooth_strength=0.0,
            hf_keep_near=1.0,
            hf_keep_far=0.92,
            propagate_iters=64,
            return_debug=True,
        )
        if isinstance(feat_work, torch.Tensor):
            feat_work = feat_work.to(dtype=torch.float32, device=feat_dev)
        filled_bnni_global_u8 = _mask_to_binary_uint8(filled_bnni_global, target_hw=(Hf, Wf))
        filled_bnni_global_t = torch.from_numpy(filled_bnni_global_u8.astype(np.float32)).to(device=feat_dev).view(1, 1, Hf, Wf) > 0.5
        valid_t = valid_t | filled_bnni_global_t
        stage_bnni_global_filled = int(filled_bnni_global_t.sum().item())

    stage_diffuse_filled = int(stage_bnni_scanline_filled + stage_bnni_local_filled + stage_bnni_global_filled)

    out = feat_work.to(dtype=feat_dtype)
    if feat_was_3d:
        out = out.squeeze(0)

    scope_info = dict(scope_info)
    scope_info.update({
        "strategy": "subject_bnni_fill_only",
        "stage_transport_filled": int(stage_transport_filled),
        "stage_diffuse_filled": int(stage_diffuse_filled),
        "stage_bnni_scanline_filled": int(stage_bnni_scanline_filled),
        "stage_bnni_local_filled": int(stage_bnni_local_filled),
        "stage_bnni_global_filled": int(stage_bnni_global_filled),
        "stage_telea_filled": int(stage_telea_filled),
        "remaining_after_fill": int((hole_t & (~valid_t)).sum().item()),
    })
    filled = out
    if return_hole_mask:
        return filled, scope_mask, scope_info, hole_mask
    return filled, scope_mask, scope_info


def interpolation_fill_subject(x):
    """主体内部裂缝修复"""
    if x.dim() != 4: return x
    out = x.clone()
    device = x.device
    kernel = torch.ones((1, 1, 3, 3), device=device)
    kernel[:, :, 1, 1] = 0
    is_zero = (torch.abs(out).sum(dim=1, keepdim=True) < 1e-6).float()
    if is_zero.sum() == 0: return out
    for c in range(x.shape[1]):
        feat = out[:, c:c+1]
        neighbor_sum = F.conv2d(feat, kernel, padding=1)
        is_valid = (torch.abs(feat) > 1e-6).float()
        valid_count = F.conv2d(is_valid, kernel, padding=1)
        filled_val = neighbor_sum / (valid_count + 1e-8)
        fillable = (valid_count > 0) & (torch.abs(feat) < 1e-6)
        out[:, c:c+1] = torch.where(fillable, filled_val, feat)
    return out


def _latent_mask_to_fullres(mask_input, target_hw):
    """
    将 latent 尺度 mask 安全上采样到图像尺度，返回 float32(H, W) in {0,1}.
    """
    h_img, w_img = int(target_hw[0]), int(target_hw[1])
    if torch.is_tensor(mask_input):
        mt = mask_input.detach().float()
        if mt.ndim == 2:
            mt = mt.unsqueeze(0).unsqueeze(0)
        elif mt.ndim == 3:
            mt = mt.unsqueeze(1)
        mt = F.interpolate(mt, size=(h_img, w_img), mode='nearest')
        out = mt.squeeze().cpu().numpy().astype(np.float32)
    else:
        out = np.asarray(mask_input, dtype=np.float32)
        if out.shape != (h_img, w_img):
            out = cv2.resize(out, (w_img, h_img), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    return (out > 0.5).astype(np.float32)


def _mask_to_fullres_vis_float(mask_input, target_hw):
    """
    可视化专用：转换到图像分辨率 float32 mask（保留软边，不做二值化）。
    """
    h_img, w_img = int(target_hw[0]), int(target_hw[1])
    if torch.is_tensor(mask_input):
        out = mask_input.detach().float().cpu().numpy()
    else:
        out = np.asarray(mask_input, dtype=np.float32)

    out = np.squeeze(out).astype(np.float32)
    if out.ndim != 2:
        out = np.asarray(out, dtype=np.float32)
        if out.ndim >= 3:
            out = np.squeeze(out[..., 0]).astype(np.float32)
        else:
            out = np.zeros((h_img, w_img), dtype=np.float32)

    if out.size <= 0:
        return np.zeros((h_img, w_img), dtype=np.float32)

    if float(np.max(out)) > 1.5:
        out = out / 255.0
    out = np.clip(out, 0.0, 1.0).astype(np.float32)

    if out.shape != (h_img, w_img):
        h_src, w_src = int(out.shape[0]), int(out.shape[1])
        interp = cv2.INTER_AREA if (h_src > h_img and w_src > w_img) else cv2.INTER_LINEAR
        out = cv2.resize(out, (w_img, h_img), interpolation=interp).astype(np.float32)
        out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)


def _downsample_image_mask_to_latent(mask_input, target_hw, threshold=0.5):
    """
    将 image 分辨率二值 mask 下采样到 latent 分辨率，并返回 {0,1} float32。
    - 下采样优先使用 INTER_AREA，保证由高分辨率 mask 派生。
    """
    h_t, w_t = int(target_hw[0]), int(target_hw[1])
    if torch.is_tensor(mask_input):
        arr = mask_input.detach().float().cpu().numpy()
    else:
        arr = np.asarray(mask_input, dtype=np.float32)

    arr = np.squeeze(arr).astype(np.float32)
    if arr.ndim != 2:
        arr = np.asarray(arr, dtype=np.float32).reshape(h_t, w_t)

    if arr.shape != (h_t, w_t):
        h_s, w_s = int(arr.shape[0]), int(arr.shape[1])
        interp = cv2.INTER_AREA if (h_s >= h_t and w_s >= w_t) else cv2.INTER_LINEAR
        arr = cv2.resize(arr, (w_t, h_t), interpolation=interp).astype(np.float32)
    return (arr > float(threshold)).astype(np.float32)


def _build_drag_plan(
    drag_mode,
    m_start_full_overall,
    m_pseudo_full_overall,
    background_hole_full,
    component_norm_grids,
    component_masks_end,
    component_masks_start=None,
    component_handles=None,
    component_targets=None,
    background_hole_base_full=None,
    hole_fill_mode=DEFAULT_HOLE_FILL_MODE,
    use_drag_guided_prefill=DEFAULT_DRAG_GUIDED_PREFILL_ENABLED,
    enable_subject_hole_fill=True,
    subject_hole_mask_lat=None,
    background_filled_clean_latent=None,
):
    """
    将一次高分辨率几何/分割计算固化为可复用计划，用于 second-pass clean latent。
    """
    grids_cpu = []
    for g in component_norm_grids:
        if torch.is_tensor(g):
            grids_cpu.append(g.detach().cpu().float())
        else:
            grids_cpu.append(None)

    masks_cpu = []
    for m in component_masks_end:
        if torch.is_tensor(m):
            masks_cpu.append(m.detach().cpu().float())
        else:
            masks_cpu.append(torch.tensor(m, dtype=torch.float32))

    masks_start_cpu = []
    for m in (component_masks_start or []):
        if torch.is_tensor(m):
            masks_start_cpu.append(m.detach().cpu().float())
        else:
            masks_start_cpu.append(torch.tensor(m, dtype=torch.float32))

    handles_cpu = []
    for pts in (component_handles or []):
        pts_arr = _sanitize_xy_points(pts)
        handles_cpu.append(pts_arr.astype(np.float32))

    targets_cpu = []
    for pts in (component_targets or []):
        pts_arr = _sanitize_xy_points(pts)
        targets_cpu.append(pts_arr.astype(np.float32))

    return {
        "version": 1,
        "drag_mode": str(drag_mode),
        "hole_fill_mode": _normalize_hole_fill_mode(hole_fill_mode),
        "use_drag_guided_prefill": _normalize_drag_guided_prefill(use_drag_guided_prefill),
        "enable_subject_hole_fill": bool(enable_subject_hole_fill),
        "m_start_full_overall": np.asarray(m_start_full_overall, dtype=np.float32),
        "m_pseudo_full_overall": np.asarray(m_pseudo_full_overall, dtype=np.float32),
        "background_hole_full": np.asarray(background_hole_full, dtype=np.float32),
        "background_hole_base_full": np.asarray(
            background_hole_full if background_hole_base_full is None else background_hole_base_full,
            dtype=np.float32,
        ),
        "subject_hole_mask_lat": (
            None
            if subject_hole_mask_lat is None
            else np.asarray(subject_hole_mask_lat, dtype=np.float32)
        ),
        "background_filled_clean_latent": (
            None
            if background_filled_clean_latent is None
            else (
                background_filled_clean_latent.detach().cpu().float().numpy()
                if torch.is_tensor(background_filled_clean_latent)
                else np.asarray(background_filled_clean_latent, dtype=np.float32)
            )
        ),
        "component_norm_grids": grids_cpu,
        "component_masks_start": masks_start_cpu,
        "component_masks_end": masks_cpu,
        "component_handles": handles_cpu,
        "component_targets": targets_cpu,
    }


def _apply_drag_guided_prefill_to_latents(
    latents,
    background_hole_full,
    pseudo_subject_full,
    component_masks_start,
    component_masks_end,
    component_handles,
    component_targets,
    device,
):
    bsz, _, h_lat, w_lat = latents.shape
    if bsz != 1:
        raise ValueError(f"Only batch size 1 is supported in drag-guided prefill, got {bsz}")

    h_img, w_img = int(background_hole_full.shape[0]), int(background_hole_full.shape[1])
    scale_y = _coord_scale_image_to_latent(h_img, h_lat)
    scale_x = _coord_scale_image_to_latent(w_img, w_lat)

    latents_float = latents.float()
    guided_background = latents_float.clone()
    bg_hole_lat_u8 = _mask_to_binary_uint8(background_hole_full, target_hw=(h_lat, w_lat))
    bg_hole_lat_t = torch.from_numpy(bg_hole_lat_u8.astype(np.float32)).to(device=device).view(1, 1, h_lat, w_lat) > 0.5
    pseudo_obj_lat_u8 = _mask_to_binary_uint8(pseudo_subject_full, target_hw=(h_lat, w_lat))
    pseudo_obj_lat_t = torch.from_numpy(pseudo_obj_lat_u8.astype(np.float32)).to(device=device).view(1, 1, h_lat, w_lat) > 0.5
    pseudo_bg_lat_t = (~pseudo_obj_lat_t).float()

    guided_cover = torch.zeros((1, 1, h_lat, w_lat), device=device, dtype=torch.bool)
    guided_prefill_px = 0
    comp_count = min(
        len(component_masks_start or []),
        len(component_masks_end or []),
        len(component_handles or []),
        len(component_targets or []),
    )
    for comp_idx in range(comp_count):
        m_start_ref = component_masks_start[comp_idx]
        m_end_ref = component_masks_end[comp_idx]
        handles_ref = component_handles[comp_idx]
        targets_ref = component_targets[comp_idx]

        start_lat_u8 = _mask_to_binary_uint8(m_start_ref, target_hw=(h_lat, w_lat))
        end_lat_u8 = _mask_to_binary_uint8(m_end_ref, target_hw=(h_lat, w_lat))
        start_lat_t = torch.from_numpy(start_lat_u8.astype(np.float32)).to(device=device).view(1, 1, h_lat, w_lat) > 0.5
        end_lat_t = torch.from_numpy(end_lat_u8.astype(np.float32)).to(device=device).view(1, 1, h_lat, w_lat) > 0.5

        vacate_lat_t = start_lat_t & (~end_lat_t) & bg_hole_lat_t
        if int(vacate_lat_t.sum().item()) == 0:
            continue

        comp_shift_lat = _estimate_motion_shift_latent_from_points(
            handles_xy=handles_ref,
            targets_xy=targets_ref,
            scale_x=scale_x,
            scale_y=scale_y,
        )
        if comp_shift_lat is None:
            continue

        shifted_src, shifted_src_valid = _shift_tensor_integer(latents_float, comp_shift_lat[0], comp_shift_lat[1])
        shifted_bg_mask, shifted_bg_valid = _shift_tensor_integer(pseudo_bg_lat_t, comp_shift_lat[0], comp_shift_lat[1])
        source_bg_valid = shifted_bg_valid & (shifted_bg_mask > 0.5)
        fill_t = vacate_lat_t & shifted_src_valid & source_bg_valid
        fill_n = int(fill_t.sum().item())
        if fill_n <= 0:
            continue

        guided_background = torch.where(fill_t.expand_as(guided_background), shifted_src, guided_background)
        guided_cover = guided_cover | fill_t
        guided_prefill_px += fill_n

    residual_hole_t = bg_hole_lat_t & (~guided_cover)
    residual_hole_n = int(residual_hole_t.sum().item())
    return (
        guided_background.to(dtype=latents.dtype),
        residual_hole_t.float().squeeze(0).squeeze(0),
        pseudo_obj_lat_t.float().squeeze(0).squeeze(0),
        guided_prefill_px,
        residual_hole_n,
    )


def _apply_drag_plan_to_latents_sgf(
    latents,
    drag_plan,
    device,
    clean_latents=None,
    alpha_prod_t=None,
    hole_fill_mode=DEFAULT_HOLE_FILL_MODE,
):
    """
    SGF 回放逻辑（内置实现）：
    1) 背景：expand hole + Telea
    2) 主体：插值后 subject-scope 补洞
    """
    bsz, _, h_lat, w_lat = latents.shape
    if bsz != 1:
        raise ValueError(f"Only batch size 1 is supported in drag plan replay, got {bsz}")
    mode = _normalize_hole_fill_mode(hole_fill_mode)
    use_lama_mode = (mode == "lama")
    plan_enable_subject_hole_fill = bool(
        drag_plan.get("enable_subject_hole_fill", True) if isinstance(drag_plan, dict) else True
    )
    plan_subject_hole_lat = np.zeros((h_lat, w_lat), dtype=np.float32)
    if isinstance(drag_plan, dict):
        plan_subject_hole_lat_raw = drag_plan.get("subject_hole_mask_lat")
        if plan_subject_hole_lat_raw is not None:
            plan_subject_hole_lat = (_mask_to_binary_uint8(plan_subject_hole_lat_raw, target_hw=(h_lat, w_lat)) > 0).astype(np.float32)

    m_start_full_overall = np.asarray(
        drag_plan.get("m_start_full_overall", np.zeros((h_lat, w_lat), dtype=np.float32)),
        dtype=np.float32,
    )
    m_pseudo_full_overall = np.asarray(
        drag_plan.get("m_pseudo_full_overall", np.zeros_like(m_start_full_overall)),
        dtype=np.float32,
    )

    bg_hole_full_plan = drag_plan.get("background_hole_full")
    if bg_hole_full_plan is not None:
        bg_hole_mask = np.asarray(bg_hole_full_plan, dtype=np.float32)
        print(f"[Background Fill/Replay][SGF] use precomputed hole, area={int(np.sum(bg_hole_mask > 0.5))}")
    else:
        bg_hole_mask, bg_expand_info = _expand_background_fill_hole_mask(m_start_full_overall)
        if bool(bg_expand_info.get("enabled", False)):
            print(
                f"[Background Fill/Replay][SGF] expand_px={bg_expand_info.get('expand_px', 0)}, "
                f"k={bg_expand_info.get('kernel', 0)}, area={bg_expand_info.get('area_before', 0)}->{bg_expand_info.get('area_after', 0)}"
            )

    plan_use_drag_guided_prefill = _normalize_drag_guided_prefill(
        drag_plan.get("use_drag_guided_prefill", DEFAULT_DRAG_GUIDED_PREFILL_ENABLED)
        if isinstance(drag_plan, dict)
        else DEFAULT_DRAG_GUIDED_PREFILL_ENABLED
    )
    background_latents = None
    plan_bg_clean = drag_plan.get("background_filled_clean_latent") if isinstance(drag_plan, dict) else None
    if use_lama_mode and (plan_bg_clean is not None):
        try:
            background_latents = _compose_clean_prior_into_current_latents(
                latents=latents,
                clean_prior_latents=plan_bg_clean,
                hole_mask_full=bg_hole_mask,
                clean_latents=clean_latents,
                alpha_prod_t=alpha_prod_t,
            )
            print("[Background Fill/Replay][SGF/LaMa] use plan clean prior.")
        except Exception as e:
            background_latents = None
            print(f"[Background Fill/Replay][SGF/LaMa] plan prior failed, fallback Telea ({e})")

    if background_latents is None:
        if plan_use_drag_guided_prefill:
            guided_background, residual_hole_mask, pseudo_obj_mask, guided_prefill_px, residual_hole_n = (
                _apply_drag_guided_prefill_to_latents(
                    latents=latents,
                    background_hole_full=bg_hole_mask,
                    pseudo_subject_full=m_pseudo_full_overall,
                    component_masks_start=drag_plan.get("component_masks_start", []),
                    component_masks_end=drag_plan.get("component_masks_end", []),
                    component_handles=drag_plan.get("component_handles", []),
                    component_targets=drag_plan.get("component_targets", []),
                    device=device,
                )
            )
            if guided_prefill_px > 0:
                print(
                    f"[Background Fill/Replay][SGF] drag-guided prefill: filled={guided_prefill_px}, residual={residual_hole_n}"
                )
            background_latents = fill_background_holes(
                image_input=guided_background,
                hole_mask=residual_hole_mask,
                forbidden_mask=pseudo_obj_mask,
                device=device,
            )
        else:
            background_latents = fill_background_holes(
                image_input=latents,
                hole_mask=bg_hole_mask,
                forbidden_mask=m_pseudo_full_overall,
                device=device,
            )

    final_latents = background_latents.clone()
    plan_drag_mode = str(drag_plan.get("drag_mode", ""))
    grid_interp_mode = "nearest" if plan_drag_mode.startswith("3D-") else "bilinear"

    grids = drag_plan.get("component_norm_grids", [])
    masks_end = drag_plan.get("component_masks_end", [])
    comp_count = min(len(grids), len(masks_end))

    for idx in range(comp_count):
        grid_ref = grids[idx]
        mask_ref = masks_end[idx]

        if torch.is_tensor(mask_ref):
            refined_mask = mask_ref.to(device=device, dtype=latents.dtype)
        else:
            refined_mask = torch.tensor(mask_ref, device=device, dtype=latents.dtype)
        if refined_mask.ndim == 4:
            refined_mask = refined_mask[0, 0]
        elif refined_mask.ndim == 3:
            refined_mask = refined_mask[0]
        if refined_mask.shape != (h_lat, w_lat):
            refined_mask = F.interpolate(
                refined_mask.view(1, 1, refined_mask.shape[0], refined_mask.shape[1]).float(),
                size=(h_lat, w_lat),
                mode="nearest",
            ).squeeze(0).squeeze(0).to(latents.dtype)

        if torch.is_tensor(grid_ref):
            norm_grid = grid_ref.to(device=device, dtype=torch.float32)
            if norm_grid.ndim == 4:
                norm_grid = norm_grid[0]
            warped_latent = _grid_sample_latents_aa(
                latents,
                norm_grid,
                mode=grid_interp_mode,
                padding_mode="border",
                align_corners=True,
            )
        else:
            warped_latent = latents.clone()

        if plan_enable_subject_hole_fill:
            refined_latent = interpolation_fill_subject(warped_latent)
        else:
            refined_latent = warped_latent

        m_new = refined_mask.unsqueeze(0).unsqueeze(0)
        final_latents = refined_latent * m_new + final_latents * (1 - m_new)

    if plan_enable_subject_hole_fill:
        subject_end_lat_u8 = np.zeros((h_lat, w_lat), dtype=np.uint8)
        for mask_ref in masks_end:
            if torch.is_tensor(mask_ref):
                m_np = mask_ref.detach().cpu().float().numpy()
            else:
                m_np = np.asarray(mask_ref, dtype=np.float32)
            if m_np.ndim == 4:
                m_np = m_np[0, 0]
            elif m_np.ndim == 3:
                m_np = m_np[0]
            m_u8 = _mask_to_binary_uint8(m_np, target_hw=(h_lat, w_lat))
            subject_end_lat_u8 = np.maximum(subject_end_lat_u8, m_u8)

        if int(np.sum(subject_end_lat_u8 > 0)) > 0:
            _, subject_hole_lat_raw, _ = infer_subject_fill_scope(
                subject_end_lat_u8.astype(np.float32),
                close_ratio=0.10,
                max_close=max(7, int(round(0.35 * min(h_lat, w_lat)))),
                max_fill_dist_ratio=0.18,
            )
            hole_inside_lat = np.logical_and(subject_hole_lat_raw > 0.5, subject_end_lat_u8 > 0).astype(np.float32)
            hole_inside_lat = np.maximum(
                hole_inside_lat,
                np.logical_and(plan_subject_hole_lat > 0.5, subject_end_lat_u8 > 0).astype(np.float32),
            ).astype(np.float32)
            donor_inside_lat_subject = np.logical_and(subject_end_lat_u8 > 0, hole_inside_lat <= 0.5).astype(np.float32)
            print(
                f"[Subject Fill/Replay][SGF] hole_inside_lat={int(np.sum(hole_inside_lat > 0.5))}, "
                f"hole_plan_lat={int(np.sum(plan_subject_hole_lat > 0.5))}, "
                f"donor_inside_lat={int(np.sum(donor_inside_lat_subject > 0.5))}"
            )
            if int(np.sum(hole_inside_lat > 0.5)) > 0:
                final_latents, _, residual_in_lat_1, info_in_1 = fill_holes_with_bnni(
                    image_input=final_latents,
                    hole_mask=hole_inside_lat.astype(np.float32),
                    donor_mask=donor_inside_lat_subject.astype(np.float32),
                    device=device,
                    local_boundary_only=True,
                    return_debug=True,
                )
                print(
                    f"[Subject Fill/Replay][SGF] pass1 "
                    f"filled={info_in_1.get('filled',0)}, residual={info_in_1.get('residual',0)}"
                )
                if int(np.sum(residual_in_lat_1 > 0.5)) > 0:
                    final_latents, _, _, info_in_2 = fill_holes_with_bnni(
                        image_input=final_latents,
                        hole_mask=residual_in_lat_1.astype(np.float32),
                        donor_mask=donor_inside_lat_subject.astype(np.float32),
                        device=device,
                        local_boundary_only=False,
                        return_debug=True,
                    )
                    print(
                        f"[Subject Fill/Replay][SGF] pass2 "
                        f"filled={info_in_2.get('filled',0)}, residual={info_in_2.get('residual',0)}"
                    )
    else:
        print("[Subject Fill/Replay][SGF] disabled by drag plan flag.")

    if torch.is_tensor(final_latents):
        final_latents = final_latents.to(dtype=latents.dtype)
    return final_latents


def _apply_drag_plan_to_latents(latents, drag_plan, device, clean_latents=None, alpha_prod_t=None):
    """
    仅在 latent 空间回放既定拖拽计划：
    - 不再触发 SAM / 深度估计 / 3D图像投影
    - 可用于 noisy/clean 双分支复用
    """
    bsz, _, h_lat, w_lat = latents.shape
    if bsz != 1:
        # 当前链路始终单图推理；若未来支持批量，这里需要扩展。
        raise ValueError(f"Only batch size 1 is supported in drag plan replay, got {bsz}")

    plan_hole_fill_mode = _normalize_hole_fill_mode(
        drag_plan.get("hole_fill_mode", DEFAULT_HOLE_FILL_MODE) if isinstance(drag_plan, dict) else DEFAULT_HOLE_FILL_MODE
    )
    if plan_hole_fill_mode in {"sgf", "lama"}:
        return _apply_drag_plan_to_latents_sgf(
            latents=latents,
            drag_plan=drag_plan,
            device=device,
            clean_latents=clean_latents,
            alpha_prod_t=alpha_prod_t,
            hole_fill_mode=plan_hole_fill_mode,
        )

    m_start_full_overall = np.asarray(
        drag_plan.get("m_start_full_overall", np.zeros((h_lat, w_lat), dtype=np.float32)),
        dtype=np.float32,
    )
    m_pseudo_full_overall = np.asarray(
        drag_plan.get("m_pseudo_full_overall", np.zeros_like(m_start_full_overall)),
        dtype=np.float32,
    )
    bg_hole_full_plan = drag_plan.get("background_hole_full")
    bg_hole_full_mask = None
    if bg_hole_full_plan is not None:
        bg_hole_full_mask = np.asarray(bg_hole_full_plan, dtype=np.float32)
        bg_hole_mask = _downsample_image_mask_to_latent(
            bg_hole_full_mask,
            target_hw=(h_lat, w_lat),
            threshold=0.10,
        )
        print(
            f"[Background Fill/Replay] use precomputed hole, area={int(np.sum(bg_hole_mask > 0.5))}"
        )
    else:
        bg_hole_full_mask, bg_expand_info = _expand_background_fill_hole_mask(m_start_full_overall)
        bg_hole_mask = _downsample_image_mask_to_latent(
            bg_hole_full_mask,
            target_hw=(h_lat, w_lat),
            threshold=0.10,
        )
        if bool(bg_expand_info.get("enabled", False)):
            print(
                f"[Background Fill/Replay] expand_px={bg_expand_info.get('expand_px', 0)}, "
                f"k={bg_expand_info.get('kernel', 0)}, area={bg_expand_info.get('area_before', 0)}->{bg_expand_info.get('area_after', 0)}"
            )

    # 回放路径与主路径保持一致：
    # - inner 仅由未扩张主体洞定义（主体 donor）
    # - 扩张只影响 outside（背景补全）
    def _stabilize_hole_mask(mask01):
        raw = (np.asarray(mask01) > 0.5).astype(np.float32)
        smoothed = _smooth_binary_mask_shape(
            raw,
            close_ratio=0.004,
            open_ratio=0.003,
            blur_ratio=0.003,
        )
        return np.logical_or(smoothed > 0.5, raw > 0.5).astype(np.float32)

    total_hole_full = _stabilize_hole_mask(
        bg_hole_full_mask if bg_hole_full_mask is not None else m_start_full_overall
    )

    total_hole_lat_u8 = _mask_to_binary_uint8(
        _downsample_image_mask_to_latent(total_hole_full, target_hw=(h_lat, w_lat), threshold=0.10),
        target_hw=(h_lat, w_lat),
    )

    plan_drag_mode = str(drag_plan.get("drag_mode", ""))
    grid_interp_mode = 'nearest' if plan_drag_mode.startswith("3D-") else 'bilinear'
    grids = drag_plan.get("component_norm_grids", [])
    masks_end = drag_plan.get("component_masks_end", [])
    subject_end_lat_u8 = np.zeros((h_lat, w_lat), dtype=np.uint8)
    for mask_ref in masks_end:
        if torch.is_tensor(mask_ref):
            m_np = mask_ref.detach().cpu().float().numpy()
        else:
            m_np = np.asarray(mask_ref, dtype=np.float32)
        if m_np.ndim == 4:
            m_np = m_np[0, 0]
        elif m_np.ndim == 3:
            m_np = m_np[0]
        m_u8 = _mask_to_binary_uint8(m_np, target_hw=(h_lat, w_lat))
        subject_end_lat_u8 = np.maximum(subject_end_lat_u8, m_u8)

    if int(np.sum(subject_end_lat_u8 > 0)) > 0:
        _, subject_hole_lat_raw, _ = infer_subject_fill_scope(
            subject_end_lat_u8.astype(np.float32),
            close_ratio=0.10,
            max_close=max(7, int(round(0.35 * min(h_lat, w_lat)))),
            max_fill_dist_ratio=0.18,
        )
    else:
        subject_hole_lat_raw = np.zeros((h_lat, w_lat), dtype=np.float32)

    hole_inside_lat = np.logical_and(subject_hole_lat_raw > 0.5, subject_end_lat_u8 > 0).astype(np.float32)
    hole_outside_lat = (total_hole_lat_u8 > 0).astype(np.float32)
    hole_total_lat = (total_hole_lat_u8 > 0).astype(np.float32)
    donor_all_lat = (total_hole_lat_u8 <= 0).astype(np.float32)
    donor_inside_lat_subject = np.logical_and(subject_end_lat_u8 > 0, hole_inside_lat <= 0.5).astype(np.float32)

    if plan_hole_fill_mode in {"bnni_all"}:
        print(f"[Background Fill/Replay] mode={plan_hole_fill_mode} (global BNNI)")
        background_latents = latents.clone()
        if int(np.sum(hole_total_lat > 0.5)) > 0 and int(np.sum(donor_all_lat > 0.5)) > 0:
            background_latents, _, residual_all_1, _ = fill_holes_with_bnni(
                image_input=background_latents,
                hole_mask=hole_total_lat.astype(np.float32),
                donor_mask=donor_all_lat.astype(np.float32),
                device=device,
                local_boundary_only=True,
                return_debug=True,
            )
            if int(np.sum(residual_all_1 > 0.5)) > 0:
                background_latents, _, residual_all_2, _ = fill_holes_with_bnni(
                    image_input=background_latents,
                    hole_mask=residual_all_1.astype(np.float32),
                    donor_mask=donor_all_lat.astype(np.float32),
                    device=device,
                    local_boundary_only=False,
                    return_debug=True,
                )
                residual_global = residual_all_2
            else:
                residual_global = residual_all_1
        else:
            residual_global = hole_total_lat.astype(np.float32)

        print(f"[Background Fill/Replay] residual={int(np.sum(residual_global > 0.5))}")
    else:
        print(f"[Background Fill/Replay] mode={plan_hole_fill_mode} (inner donor: subject-only)")
        background_latents = fill_background_holes(
            image_input=latents,
            hole_mask=hole_outside_lat,
            forbidden_mask=m_pseudo_full_overall,
            device=device,
        )

        if int(np.sum(hole_inside_lat > 0.5)) > 0:
            background_latents, _, residual_in_lat_1, _ = fill_holes_with_bnni(
                image_input=background_latents,
                hole_mask=hole_inside_lat.astype(np.float32),
                donor_mask=donor_inside_lat_subject,
                device=device,
                local_boundary_only=True,
                return_debug=True,
            )
            if int(np.sum(residual_in_lat_1 > 0.5)) > 0:
                background_latents, _, _, _ = fill_holes_with_bnni(
                    image_input=background_latents,
                    hole_mask=residual_in_lat_1.astype(np.float32),
                    donor_mask=donor_inside_lat_subject,
                    device=device,
                    local_boundary_only=False,
                    return_debug=True,
                )

    final_latents = background_latents.clone()
    comp_count = min(len(grids), len(masks_end))

    for idx in range(comp_count):
        grid_ref = grids[idx]
        mask_ref = masks_end[idx]

        if torch.is_tensor(mask_ref):
            refined_mask = mask_ref.to(device=device, dtype=latents.dtype)
        else:
            refined_mask = torch.tensor(mask_ref, device=device, dtype=latents.dtype)
        if refined_mask.ndim == 4:
            refined_mask = refined_mask[0, 0]
        elif refined_mask.ndim == 3:
            refined_mask = refined_mask[0]
        if refined_mask.shape != (h_lat, w_lat):
            refined_mask = F.interpolate(
                refined_mask.view(1, 1, refined_mask.shape[0], refined_mask.shape[1]).float(),
                size=(h_lat, w_lat),
                mode='nearest',
            ).squeeze(0).squeeze(0).to(latents.dtype)

        if torch.is_tensor(grid_ref):
            norm_grid = grid_ref.to(device=device, dtype=torch.float32)
            if norm_grid.ndim == 4:
                norm_grid = norm_grid[0]
            warped_latent = _grid_sample_latents_aa(
                latents,
                norm_grid,
                mode=grid_interp_mode,
                padding_mode='border',
                align_corners=True,
            )
        else:
            warped_latent = latents.clone()

        # 回放路径保持几何一致：不做主体 scope 改写，只复用既定的 end mask。
        refined_latent = interpolation_fill_subject(warped_latent)

        m_new = refined_mask.unsqueeze(0).unsqueeze(0)
        final_latents = refined_latent * m_new + final_latents * (1 - m_new)

    return final_latents
# ==========================================
# 2. Mask 工具函数
# ==========================================
def get_mask_info_numpy(mask_np):
    """获取 Mask 的质心和包围盒"""
    if mask_np.dtype != np.uint8:
        mask_u8 = (mask_np > 0.5).astype(np.uint8) * 255
    else:
        mask_u8 = mask_np
    M = cv2.moments(mask_u8)
    if M["m00"] == 0:
        h, w = mask_np.shape[:2]
        return w/2.0, h/2.0, 0, 0, w, h
    
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    y_indices, x_indices = np.nonzero(mask_u8 > 127)
    if len(y_indices) == 0:
        h, w = mask_np.shape[:2]
        return cx, cy, 0, 0, w, h
        
    min_x, max_x = np.min(x_indices), np.max(x_indices)
    min_y, max_y = np.min(y_indices), np.max(y_indices)
    
    return cx, cy, min_x, min_y, max_x, max_y

def dilate_mask(mask_np, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask_np, kernel, iterations=1)


def _to_odd_kernel(ksize):
    ksize = max(3, int(ksize))
    return ksize if ksize % 2 == 1 else ksize + 1


def _get_component_min_area(mask_shape):
    h, w = int(mask_shape[0]), int(mask_shape[1])
    return max(COMPONENT_MIN_AREA_BASE, int(round(h * w * COMPONENT_MIN_AREA_RATIO)))


def _resolve_component_min_area(mask_input, min_area=None):
    """
    自适应连通域面积阈值：
    - 基础阈值：与图像尺寸相关
    - 相对阈值：最大连通域面积的一定比例
    """
    mask_bin = (np.asarray(mask_input) > 0).astype(np.uint8)
    if mask_bin.ndim != 2:
        return int(min_area) if min_area is not None else COMPONENT_MIN_AREA_BASE

    base_min = int(min_area) if min_area is not None else _get_component_min_area(mask_bin.shape)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if num_labels <= 1:
        return base_min

    largest_area = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
    rel_min = int(round(largest_area * COMPONENT_MIN_AREA_RELATIVE_TO_LARGEST))
    return max(base_min, rel_min)


def _remove_small_components(mask_input, min_area=None, keep_largest_if_empty=False):
    """
    过滤过小连通域，返回 uint8(H, W) in {0,255}.
    """
    if mask_input is None:
        return None

    mask_u8 = np.asarray(mask_input)
    if mask_u8.ndim != 2:
        return mask_u8.astype(np.uint8)

    mask_bin = (mask_u8 > 0).astype(np.uint8)
    min_area = _resolve_component_min_area(mask_bin, min_area=min_area)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if num_labels <= 1:
        return mask_bin.astype(np.uint8) * 255

    out = np.zeros_like(mask_bin, dtype=np.uint8)
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

    return out.astype(np.uint8) * 255


def _clean_binary_mask_u8(mask_u8, close_k=3, open_k=3, min_area=0, keep_largest_if_empty=False):
    """
    轻量净化二值 mask（u8 in {0,1} or {0,255}）:
    close -> open -> remove small components
    """
    if mask_u8 is None:
        return None
    m = (np.asarray(mask_u8) > 0).astype(np.uint8)
    if close_k and int(close_k) >= 3:
        k = _to_odd_kernel(int(close_k))
        m = (cv2.morphologyEx((m * 255).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)) > 127).astype(np.uint8)
    if open_k and int(open_k) >= 3:
        k = _to_odd_kernel(int(open_k))
        m = (cv2.morphologyEx((m * 255).astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8)) > 127).astype(np.uint8)
    if int(min_area) > 0:
        m = (_remove_small_components((m * 255).astype(np.uint8), min_area=int(min_area), keep_largest_if_empty=keep_largest_if_empty) > 127).astype(np.uint8)
    return m


def _distance_to_seed_pixels(seed_u8):
    """
    返回每个像素到 seed(=1) 的 L2 距离图（float32）。
    seed 为空时返回很大值。
    """
    seed = (np.asarray(seed_u8) > 0).astype(np.uint8)
    if int(np.sum(seed)) <= 0:
        return np.ones_like(seed, dtype=np.float32) * 1e6
    inv = (seed <= 0).astype(np.uint8)
    return cv2.distanceTransform(inv, cv2.DIST_L2, 5).astype(np.float32)


def _extract_enclosed_holes_u8(mask_u8):
    """
    提取 mask 内部封闭空洞（不与图像边界连通的背景区域）。
    返回 u8 {0,1}.
    """
    m = (np.asarray(mask_u8) > 0).astype(np.uint8)
    H, W = m.shape[:2]
    if int(np.sum(m)) <= 0:
        return np.zeros((H, W), dtype=np.uint8)

    bg = (m <= 0).astype(np.uint8)
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(bg, connectivity=8)
    if num_labels <= 1:
        return np.zeros((H, W), dtype=np.uint8)

    border_labels = set()
    border_labels.update(np.unique(labels[0, :]).tolist())
    border_labels.update(np.unique(labels[H - 1, :]).tolist())
    border_labels.update(np.unique(labels[:, 0]).tolist())
    border_labels.update(np.unique(labels[:, W - 1]).tolist())

    holes = np.zeros((H, W), dtype=np.uint8)
    for lid in range(1, num_labels):
        if lid in border_labels:
            continue
        holes[labels == lid] = 1
    return holes


def _compute_four_direction_subject_interior_u8(subject_u8):
    """
    返回“上下左右四个方向都还能看到主体”的像素区域。
    用于严格定义“主体内部”，避免把边缘/外侧区域误判成内部空洞。
    """
    subject = (np.asarray(subject_u8) > 0).astype(np.uint8)
    H, W = subject.shape[:2]
    if H <= 0 or W <= 0 or int(np.sum(subject)) <= 0:
        return np.zeros((H, W), dtype=np.uint8)

    row_cumsum = np.cumsum(subject, axis=1, dtype=np.int32)
    row_total = row_cumsum[:, -1:]
    has_left = np.zeros((H, W), dtype=bool)
    has_left[:, 1:] = row_cumsum[:, :-1] > 0
    has_right = (row_total - row_cumsum) > 0

    col_cumsum = np.cumsum(subject, axis=0, dtype=np.int32)
    col_total = col_cumsum[-1:, :]
    has_up = np.zeros((H, W), dtype=bool)
    has_up[1:, :] = col_cumsum[:-1, :] > 0
    has_down = (col_total - col_cumsum) > 0

    return (has_left & has_right & has_up & has_down).astype(np.uint8)


def _split_hole_inner_outer_precise(hole_u8, subject_support_u8, ratio_thr=0.78):
    """
    严格的内外洞划分：
    只有当 hole 像素在上下左右四个方向都被主体包围时，才视为主体内部空洞。
    """
    hole = (np.asarray(hole_u8) > 0).astype(np.uint8)
    subject = (np.asarray(subject_support_u8) > 0).astype(np.uint8)
    if int(np.sum(hole)) <= 0:
        z = np.zeros_like(hole, dtype=np.float32)
        return z, z

    subject_four_dir_inner = _compute_four_direction_subject_interior_u8(subject)
    inner = np.logical_and(hole > 0, subject_four_dir_inner > 0).astype(np.uint8)
    outer = np.logical_and(hole > 0, inner <= 0).astype(np.uint8)
    return inner.astype(np.float32), outer.astype(np.float32)


def _dilate_binary_mask(mask01, ksize):
    """
    对二值 mask 做一次膨胀，输入/输出均为 {0,1} float32。
    """
    k = _to_odd_kernel(ksize)
    kernel = np.ones((k, k), np.uint8)
    u8 = (mask01 > 0.5).astype(np.uint8) * 255
    out = cv2.dilate(u8, kernel, iterations=1)
    return (out > 127).astype(np.float32)


def _expand_background_fill_hole_mask(mask_input, expand_ratio=0.015, min_expand_px=1, max_expand_px=8):
    """
    对背景补洞区域做适度外扩，减少边缘残留伪影。
    返回: (expanded_mask01, info)
    """
    mask_u8 = _mask_to_binary_uint8(mask_input)
    H, W = mask_u8.shape[:2]
    area = int(np.sum(mask_u8 > 0))
    if area <= 0:
        return np.zeros((H, W), dtype=np.float32), {
            "enabled": False,
            "reason": "empty_mask",
            "expand_px": 0,
            "kernel": 0,
            "area_before": 0,
            "area_after": 0,
        }

    ys, xs = np.where(mask_u8 > 0)
    bbox_w = int(xs.max() - xs.min() + 1)
    bbox_h = int(ys.max() - ys.min() + 1)
    obj_span = max(bbox_w, bbox_h, 1)

    expand_px = int(np.clip(round(float(obj_span) * float(expand_ratio)), int(min_expand_px), int(max_expand_px)))
    k = _to_odd_kernel(2 * expand_px + 1)
    expanded = _dilate_binary_mask(mask_u8.astype(np.float32), k)

    return expanded.astype(np.float32), {
        "enabled": True,
        "expand_px": int(expand_px),
        "kernel": int(k),
        "obj_span": int(obj_span),
        "area_before": int(area),
        "area_after": int(np.sum(expanded > 0.5)),
    }


def _compute_overall_end_mask(all_results, H_img, W_img, device):
    """
    统一计算全局 end mask（图像分辨率）。
    优先读取每个组件 debug_info 中的 m_end_full；缺失时回退到 norm_grid 重建。
    """
    m_end_overall = np.zeros((H_img, W_img), dtype=np.float32)
    comp_num = len(all_results.get("grids", []))
    debug_list = all_results.get("debug_infos", [])
    for comp_idx in range(comp_num):
        debug_info = debug_list[comp_idx] if comp_idx < len(debug_list) else {}
        m_end_full = debug_info.get("m_end_full") if isinstance(debug_info, dict) else None
        if isinstance(m_end_full, np.ndarray):
            if m_end_full.shape[:2] != (H_img, W_img):
                m_end_full = _latent_mask_to_fullres(m_end_full, target_hw=(H_img, W_img))
            else:
                m_end_full = np.clip(m_end_full.astype(np.float32), 0.0, 1.0)
            m_end_overall = np.maximum(m_end_overall, m_end_full.astype(np.float32))
            continue

        norm_grid = all_results["grids"][comp_idx]
        m_start = all_results.get("masks_start", [])[comp_idx] if comp_idx < len(all_results.get("masks_start", [])) else None
        if norm_grid is None or m_start is None:
            continue
        if torch.is_tensor(norm_grid):
            grid_ref = norm_grid.detach().to(device=device, dtype=torch.float32)
        else:
            grid_ref = torch.from_numpy(np.asarray(norm_grid, dtype=np.float32)).to(device=device)
        if grid_ref.ndim == 4:
            grid_ref = grid_ref[0]
        if grid_ref.ndim != 3 or grid_ref.shape[-1] != 2:
            continue

        grid_tensor = grid_ref.permute(2, 0, 1).unsqueeze(0)
        grid_up = F.interpolate(grid_tensor, size=(H_img, W_img), mode="bilinear", align_corners=True)
        grid_up = grid_up.permute(0, 2, 3, 1)
        m_start_t = torch.from_numpy(np.asarray(m_start, dtype=np.float32)).to(device).view(1, 1, H_img, W_img)
        m_end_t = F.grid_sample(m_start_t, grid_up, mode="nearest", align_corners=True)
        m_end_overall = np.maximum(m_end_overall, m_end_t.squeeze().detach().cpu().numpy().astype(np.float32))
    return np.clip(m_end_overall, 0.0, 1.0).astype(np.float32)


def _collect_subject_hole_mask_from_results(all_results, target_hw):
    """汇总各组件 debug_info 中记录的主体内部空洞掩码。"""
    h, w = int(target_hw[0]), int(target_hw[1])
    merged = np.zeros((h, w), dtype=np.float32)
    if not isinstance(all_results, dict):
        return merged

    nested_keys = ("phase1", "phase2", "phase1_3d", "phase2_3d_nonrigid", "phase2_nonrigid")
    for dbg in all_results.get("debug_infos", []):
        if not isinstance(dbg, dict):
            continue
        cand = dbg.get("subject_hole_mask_full")
        if isinstance(cand, np.ndarray):
            c_u8 = _mask_to_binary_uint8(cand, target_hw=(h, w))
            merged = np.maximum(merged, (c_u8 > 0).astype(np.float32))
        for k in nested_keys:
            sub = dbg.get(k)
            if not isinstance(sub, dict):
                continue
            cand2 = sub.get("subject_hole_mask_full")
            if isinstance(cand2, np.ndarray):
                c_u8 = _mask_to_binary_uint8(cand2, target_hw=(h, w))
                merged = np.maximum(merged, (c_u8 > 0).astype(np.float32))
    return merged.astype(np.float32)


def _collect_subject_hole_lat_mask_from_results(all_results, target_hw):
    """汇总各组件 debug_info 中记录的主体内部空洞（latent 尺度）掩码。"""
    h, w = int(target_hw[0]), int(target_hw[1])
    merged = np.zeros((h, w), dtype=np.float32)
    if not isinstance(all_results, dict):
        return merged

    nested_keys = ("phase1", "phase2", "phase1_3d", "phase2_3d_nonrigid", "phase2_nonrigid")
    for dbg in all_results.get("debug_infos", []):
        if not isinstance(dbg, dict):
            continue
        cand = dbg.get("subject_hole_mask_lat")
        if cand is not None:
            c_u8 = _mask_to_binary_uint8(cand, target_hw=(h, w))
            merged = np.maximum(merged, (c_u8 > 0).astype(np.float32))
        for k in nested_keys:
            sub = dbg.get(k)
            if not isinstance(sub, dict):
                continue
            cand2 = sub.get("subject_hole_mask_lat")
            if cand2 is not None:
                c_u8 = _mask_to_binary_uint8(cand2, target_hw=(h, w))
                merged = np.maximum(merged, (c_u8 > 0).astype(np.float32))
    return merged.astype(np.float32)


def _smooth_binary_mask_shape(mask01, close_ratio=0.010, open_ratio=0.008, blur_ratio=0.006):
    """
    对二值形状做轻量轮廓平滑：
    - close: 填小凹陷
    - open : 去小凸起
    - blur+threshold: 软化锯齿边
    返回 float32 {0,1} mask
    """
    if mask01 is None:
        return mask01
    m = np.asarray(mask01, dtype=np.float32)
    if m.ndim != 2:
        return m

    H, W = m.shape[:2]
    if H <= 2 or W <= 2:
        return (m > 0.5).astype(np.float32)

    u8 = ((m > 0.5).astype(np.uint8) * 255)
    if int(np.count_nonzero(u8)) == 0:
        return np.zeros_like(m, dtype=np.float32)

    k_close = _to_odd_kernel(close_ratio * min(H, W))
    k_open = _to_odd_kernel(open_ratio * min(H, W))
    k_blur = _to_odd_kernel(blur_ratio * min(H, W))

    ker_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    ker_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, ker_close, iterations=1)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, ker_open, iterations=1)

    blur = cv2.GaussianBlur(u8.astype(np.float32), (k_blur, k_blur), 0)
    u8 = (blur > 120.0).astype(np.uint8) * 255

    # 去掉很小的毛刺连通域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((u8 > 127).astype(np.uint8), connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_area = int(np.max(areas)) if areas.size > 0 else 0
        keep_min = max(36, int(0.05 * max_area))
        clean = np.zeros_like(u8, dtype=np.uint8)
        for lid in range(1, num_labels):
            if int(stats[lid, cv2.CC_STAT_AREA]) >= keep_min:
                clean[labels == lid] = 255
        if int(np.count_nonzero(clean)) > 0:
            u8 = clean

    return (u8 > 127).astype(np.float32)

def extract_mask_components(hint_mask, source_image_np, sam_pts_xy):
    """从手绘 Mask 中提取所有连通域"""
    H_img, W_img = source_image_np.shape[:2]
    components = []
    if hint_mask is None:
        return components
    hint_mask_u8 = (np.asarray(hint_mask) > 0).astype(np.uint8) * 255
    min_area = _resolve_component_min_area(
        hint_mask_u8, min_area=_get_component_min_area(hint_mask_u8.shape)
    )
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(hint_mask_u8, connectivity=8)
    if num_labels <= 1:
        return components
    removed_small = 0
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            removed_small += 1
            continue
        comp_mask_bool = (labels == label_id)
        comp_mask_u8 = comp_mask_bool.astype(np.uint8) * 255
        handle_indices = []
        for h_idx, h_pt in enumerate(sam_pts_xy):
            hx, hy = int(h_pt[0]), int(h_pt[1])
            if 0 <= hx < W_img and 0 <= hy < H_img:
                if comp_mask_bool[hy, hx]:
                    handle_indices.append(h_idx)
        if len(handle_indices) == 0:
            continue
        components.append({
            'mask_u8': comp_mask_u8,
            'handle_indices': handle_indices,
            'hint_mask': comp_mask_u8
        })
    if removed_small > 0:
        print(f"[Mask Components] Removed {removed_small} tiny components (min_area={min_area})")
    return components

def prune_inactive_masks(mask_np, handle_points_np, radius=20):
    """剔除没有控制点的连通域"""
    if mask_np.max() <= 1.05: 
        mask_u8 = (mask_np * 255).astype(np.uint8)
    else: 
        mask_u8 = mask_np.astype(np.uint8)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 2: 
        return mask_np

    min_area = _resolve_component_min_area(
        mask_u8, min_area=_get_component_min_area(mask_u8.shape)
    )
    
    active_label_ids = set()
    h, w = mask_u8.shape
    
    for pt in handle_points_np:
        px, py = int(pt[0]), int(pt[1])
        
        if 0 <= px < w and 0 <= py < h:
            lid = labels[py, px]
            if lid > 0 and int(stats[lid, cv2.CC_STAT_AREA]) >= min_area:
                active_label_ids.add(lid)
                continue
        
        found_neighbor = False
        for dy in range(-radius, radius + 1):
            if found_neighbor: break
            for dx in range(-radius, radius + 1):
                nx, ny = px + dx, py + dy
                if 0 <= nx < w and 0 <= ny < h:
                    lid = labels[ny, nx]
                    if lid > 0 and int(stats[lid, cv2.CC_STAT_AREA]) >= min_area:
                        active_label_ids.add(lid)
                        found_neighbor = True
                        break
    
    if len(active_label_ids) == 0:
        return mask_np 
        
    new_mask_u8 = np.zeros_like(mask_u8)
    for lid in active_label_ids:
        new_mask_u8[labels == lid] = 255
        
    if mask_np.max() <= 1.05: 
        return (new_mask_u8 / 255.0).astype(np.float32)
    else: 
        return new_mask_u8


# ==========================================
# 3. 刚性变形 - 意图识别与调度
# ==========================================
def detect_rigid_intent(handle_points_np, mask_np, force_mode=None):
    """【刚性变形】意图识别"""
    mask_for_intent = _remove_small_components(
        mask_np,
        min_area=_get_component_min_area(mask_np.shape[:2]),
        keep_largest_if_empty=True,
    )
    cx, cy, min_x, min_y, max_x, max_y = get_mask_info_numpy(mask_for_intent)
    centroid = np.array([cx, cy])
    
    if len(handle_points_np) == 0: 
        h_center = centroid
    else: 
        h_center = np.mean(handle_points_np, axis=0)
    
    # ✅ 内联：计算旋转支点（OPPOSITE 模式）
    pivot_point = 2 * centroid - h_center
    
    # 强制模式
    if force_mode == "ROTATION":
        print(f"[Rigid Intent] Rotation (FORCED, pivot={pivot_point})")
        return "ROTATION", {'action': 'Rotation', 'pivot_img': pivot_point}
    elif force_mode == "TRANSLATION":
        print(f"[Rigid Intent] Translation (FORCED)")
        return "TRANSLATION", {'action': 'Translation', 'pivot_img': None}
    
    # 自动判断
    # 判定圈在上次基础上再缩小 1/3（总系数从原始值变为 4/9）。
    radius = (((max_x - min_x) + (max_y - min_y)) / 4.0) * (4.0 / 9.0) + 1e-6
    dist = np.linalg.norm(h_center - centroid)
    relative_dist = dist / radius
    
    if relative_dist < 0.4:
        print(f"[Rigid Intent] Translation (AUTO, dist={relative_dist:.2f})")
        return "TRANSLATION", {'action': 'Translation', 'pivot_img': None}
    else:
        print(f"[Rigid Intent] Rotation (AUTO, dist={relative_dist:.2f}, pivot={pivot_point})")
        return "ROTATION", {'action': 'Rotation', 'pivot_img': pivot_point}

def _rigid_01_translation(src_pts, tgt_pts, H, W, device):
    """【刚性-01】平移变形"""
    shift = torch.mean(tgt_pts - src_pts, dim=0)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device), 
        torch.arange(W, device=device), 
        indexing='ij'
    )
    src_x = grid_x - shift[0]
    src_y = grid_y - shift[1]
    return torch.stack([src_x, src_y], dim=2).float()

def _rigid_02_rotation(src_pts, tgt_pts, pivot, H, W, device):
    """【刚性-02】旋转变形"""
    # pivot 是 numpy array，需要转换
    pivot_tensor = torch.tensor(pivot, device=device).float()
    
    v_src = src_pts[0] - pivot_tensor
    v_tgt = tgt_pts[0] - pivot_tensor
    theta_src = torch.atan2(v_src[1], v_src[0])
    theta_tgt = torch.atan2(v_tgt[1], v_tgt[0])
    d_theta = theta_tgt - theta_src 
    cos_t = torch.cos(d_theta)
    sin_t = torch.sin(d_theta)
    
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device), 
        torch.arange(W, device=device), 
        indexing='ij'
    )
    rel_x = grid_x - pivot_tensor[0]
    rel_y = grid_y - pivot_tensor[1]
    rot_x = rel_x * cos_t + rel_y * sin_t
    rot_y = -rel_x * sin_t + rel_y * cos_t
    src_x = rot_x + pivot_tensor[0]
    src_y = rot_y + pivot_tensor[1]
    return torch.stack([src_x, src_y], dim=2).float()


def _fit_similarity_transform_np(src_pts, dst_pts):
    """
    最小二乘拟合 2D 相似变换（row-vector 形式）:
      dst = scale * src @ R^T + t
    返回: (scale, R, t, mean_residual)
    """
    src = np.asarray(src_pts, dtype=np.float32).reshape(-1, 2)
    dst = np.asarray(dst_pts, dtype=np.float32).reshape(-1, 2)
    if src.shape[0] < 2 or dst.shape[0] < 2:
        return None, None, None, np.inf

    n = min(src.shape[0], dst.shape[0])
    src = src[:n]
    dst = dst[:n]

    mu_s = np.mean(src, axis=0)
    mu_d = np.mean(dst, axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d

    var_s = float(np.sum(src_c ** 2))
    if var_s < 1e-6:
        return None, None, None, np.inf

    cov = src_c.T @ dst_c / float(max(n, 1))
    U, S, Vt = np.linalg.svd(cov)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    scale = float(np.sum(S) / (var_s / float(max(n, 1)) + 1e-6))
    t = mu_d - scale * (R @ mu_s)
    pred = (scale * (src @ R.T)) + t
    residual = float(np.mean(np.linalg.norm(pred - dst, axis=1)))
    return float(scale), R.astype(np.float32), t.astype(np.float32), residual


def _build_similarity_inverse_grid(scale_factor, rot_mat, trans_vec, H, W, device):
    """
    构建相似变换的逆采样网格（用于 grid_sample）:
      forward: x' = s * x @ R^T + t
      inverse: x  = (x' - t) @ R / s
    """
    s = float(scale_factor)
    if not np.isfinite(s) or abs(s) < 1e-6:
        s = 1.0

    R = np.asarray(rot_mat, dtype=np.float32).reshape(2, 2)
    t = np.asarray(trans_vec, dtype=np.float32).reshape(2)
    R_t = torch.from_numpy(R).to(device=device, dtype=torch.float32)
    t_t = torch.from_numpy(t).to(device=device, dtype=torch.float32)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    out_xy = torch.stack([grid_x, grid_y], dim=2)  # target coords
    src_xy = torch.matmul(out_xy - t_t, R_t) / s
    return src_xy

def process_rigid_deformation(
    comp_sam_pts, comp_targets_xy, mask_for_calc,
    src_xy, tgt_xy, H_lat, W_lat, device,
    latents, comp_mask_start, comp_m_start_full,
    scale_x, scale_y, force_mode=None
):
    """
    【刚性变形】完整流程封装
    新增：返回变换后的控制点位置，供 Hybrid 模式使用
    """
    debug_info = {'m_start_full': comp_m_start_full}
    
    # 1. 意图识别
    intent_type, intent_params = detect_rigid_intent(comp_sam_pts, mask_for_calc, force_mode=force_mode)
    debug_info['intent_type'] = intent_type
    debug_info['sub_action'] = intent_params['action']
    
    # 2. 支点坐标转换
    pivot_img = intent_params['pivot_img']
    pivot_lat = np.array([pivot_img[0] * scale_x, pivot_img[1] * scale_y]) if pivot_img is not None else None
    
    # 3. Grid 计算 + 变换后控制点
    if intent_type == "TRANSLATION":
        grid_abs = _rigid_01_translation(src_xy, tgt_xy, H_lat, W_lat, device)
        # 计算平移后的控制点
        shift = torch.mean(tgt_xy - src_xy, dim=0)
        transformed_src_xy = src_xy + shift
    else:  # ROTATION
        grid_abs = _rigid_02_rotation(src_xy, tgt_xy, pivot_lat, H_lat, W_lat, device)
        # 计算旋转后的控制点
        pivot_tensor = torch.tensor(pivot_lat, device=device).float()
        v_src = src_xy[0] - pivot_tensor
        v_tgt = tgt_xy[0] - pivot_tensor
        theta_src = torch.atan2(v_src[1], v_src[0])
        theta_tgt = torch.atan2(v_tgt[1], v_tgt[0])
        d_theta = theta_tgt - theta_src
        cos_t = torch.cos(d_theta)
        sin_t = torch.sin(d_theta)
        
        # 控制点正向变换：从源位置到旋转后位置（逆时针旋转 d_theta）
        # 注意：这与 Grid 采样的反向映射方向相反
        # 正向旋转矩阵: [cos, -sin; sin, cos]
        transformed_src_xy = torch.zeros_like(src_xy)
        for i in range(src_xy.shape[0]):
            rel = src_xy[i] - pivot_tensor
            transformed_src_xy[i, 0] = rel[0] * cos_t - rel[1] * sin_t + pivot_tensor[0]
            transformed_src_xy[i, 1] = rel[0] * sin_t + rel[1] * cos_t + pivot_tensor[1]
        
        debug_info['rotation_angle'] = torch.rad2deg(d_theta).item()
    
    # 4. 归一化 Grid
    norm_grid = torch.zeros_like(grid_abs)
    norm_grid[..., 0] = 2.0 * grid_abs[..., 0] / (W_lat - 1) - 1.0
    norm_grid[..., 1] = 2.0 * grid_abs[..., 1] / (H_lat - 1) - 1.0
    
    # 5. Warp Latent
    warped_latents = _grid_sample_latents_aa(
        latents,
        norm_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )
    
    # 6. Warp Mask
    comp_mask_end = F.grid_sample(
        comp_mask_start.view(1, 1, H_lat, W_lat),
        norm_grid.unsqueeze(0),
        mode='bilinear', align_corners=True
    ).squeeze()
    
    # 7. 保存调试信息
    debug_info['norm_grid'] = norm_grid
    debug_info['pivot_img'] = pivot_img
    debug_info['pivot_lat'] = pivot_lat
    debug_info['warped_latents'] = warped_latents.clone()
    debug_info['mask_end'] = comp_mask_end.clone()
    debug_info['transformed_src_xy'] = transformed_src_xy  # 新增：变换后的控制点
    
    # 刚性变形没有锚点
    empty_anchors = np.zeros((0, 2), dtype=np.float32)
    
    return warped_latents, comp_mask_end, "Rigid", pivot_img, empty_anchors, debug_info


# ==========================================
# 4. 非刚性变形 - 意图识别与调度
# ==========================================
def detect_nonrigid_intent(handle_points, target_points, mask_np, influence_range=0.5):
    """
    【非刚性变形】意图识别（新版）

    意图集合：
    - SINGLE_SIDE_NORMAL: 单边模式-普通
    - SINGLE_SIDE_AXIS: 单边模式-轴向（垂直方向：向上/向下）
    - BILATERAL_UNIFORM: 双边均匀拉伸/压缩（支持多点分组）
    - UNIFORM_SCALE: 均匀缩放（保形）
    - FREE_STRETCH: 自由拉伸兜底
    - EMPTY: 空 mask
    """
    def _fit_similarity(src_pts, dst_pts):
        """最小二乘相似变换拟合，返回 (scale, R, t, mean_residual)。"""
        src = np.asarray(src_pts, dtype=np.float32).reshape(-1, 2)
        dst = np.asarray(dst_pts, dtype=np.float32).reshape(-1, 2)
        if src.shape[0] < 2:
            return None, None, None, np.inf

        mu_s = np.mean(src, axis=0)
        mu_d = np.mean(dst, axis=0)
        src_c = src - mu_s
        dst_c = dst - mu_d
        var_s = float(np.sum(src_c ** 2))
        if var_s < 1e-6:
            return None, None, None, np.inf

        cov = src_c.T @ dst_c / float(src.shape[0])
        U, S, Vt = np.linalg.svd(cov)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        scale = float(np.sum(S) / (var_s / float(src.shape[0]) + 1e-6))
        t = mu_d - scale * (R @ mu_s)
        pred = (scale * (src @ R.T)) + t
        residual = float(np.mean(np.linalg.norm(pred - dst, axis=1)))
        return scale, R, t, residual

    if mask_np.dtype != np.uint8:
        mask_np = (mask_np * 255).astype(np.uint8)

    handle_points = np.asarray(handle_points, dtype=np.float32).reshape(-1, 2)
    target_points = np.asarray(target_points, dtype=np.float32).reshape(-1, 2)
    n_pts = min(handle_points.shape[0], target_points.shape[0])
    if n_pts == 0:
        return "EMPTY", {"desc": "No control points"}
    handle_points = handle_points[:n_pts]
    target_points = target_points[:n_pts]

    y_idxs, x_idxs = np.nonzero(mask_np > 127)
    if len(y_idxs) == 0:
        return "EMPTY", {"desc": "Empty Mask"}

    all_pts = np.stack([x_idxs, y_idxs], axis=1).astype(np.float32)
    center_of_mass = np.mean(all_pts, axis=0)
    h_center = np.mean(handle_points, axis=0)
    disps = target_points - handle_points
    disp_norm = np.linalg.norm(disps, axis=1)
    motion_mag = float(np.mean(disp_norm)) if disp_norm.size > 0 else 0.0
    unit_disps = disps / (disp_norm[:, None] + 1e-6)
    avg_vec = np.mean(unit_disps, axis=0)
    consistency = float(np.linalg.norm(avg_vec))
    mean_disp = np.mean(disps, axis=0)
    mean_disp_norm = float(np.linalg.norm(mean_disp))
    mean_disp_unit = mean_disp / (mean_disp_norm + 1e-6)

    # 物体尺度估计
    min_xy = np.min(all_pts, axis=0)
    max_xy = np.max(all_pts, axis=0)
    obj_w = float(max_xy[0] - min_xy[0] + 1e-6)
    obj_h = float(max_xy[1] - min_xy[1] + 1e-6)
    obj_diag = float(np.sqrt(obj_w ** 2 + obj_h ** 2) + 1e-6)

    # mask PCA 主轴（用于几何分析）
    rel_obj = all_pts - center_of_mass
    cov_obj = np.cov(rel_obj.T) if rel_obj.shape[0] >= 3 else np.eye(2, dtype=np.float32)
    evals_obj, evecs_obj = np.linalg.eigh(cov_obj)
    order_obj = np.argsort(evals_obj)[::-1]
    evals_obj = evals_obj[order_obj]
    obj_major_axis = evecs_obj[:, order_obj[0]].astype(np.float32)
    if np.linalg.norm(obj_major_axis) < 1e-6:
        obj_major_axis = np.array([1.0, 0.0], dtype=np.float32)
    obj_major_axis = obj_major_axis / (np.linalg.norm(obj_major_axis) + 1e-6)
    obj_minor_axis = np.array([-obj_major_axis[1], obj_major_axis[0]], dtype=np.float32)
    elongation = float(np.sqrt((evals_obj[0] + 1e-6) / (evals_obj[1] + 1e-6)))

    # handle 主轴（用于双边分组）
    rel_h = handle_points - center_of_mass
    if n_pts >= 2:
        cov_h = np.cov(rel_h.T)
        evals_h, evecs_h = np.linalg.eigh(cov_h)
        handle_axis = evecs_h[:, np.argmax(evals_h)].astype(np.float32)
    else:
        handle_axis = mean_disp_unit.astype(np.float32)
    if np.linalg.norm(handle_axis) < 1e-6:
        handle_axis = obj_major_axis.copy()
    handle_axis = handle_axis / (np.linalg.norm(handle_axis) + 1e-6)
    handle_perp_axis = np.array([-handle_axis[1], handle_axis[0]], dtype=np.float32)

    # 坐标轴夹角（单边轴向判定）
    axis_candidates = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([-1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([0.0, -1.0], dtype=np.float32),
    ]
    min_axis_angle = 90.0
    for axis in axis_candidates:
        dot = float(np.clip(np.dot(mean_disp_unit, axis), -1.0, 1.0))
        min_axis_angle = min(min_axis_angle, float(np.degrees(np.arccos(dot))))

    # 同侧判定（支持多点）
    radial_vec = h_center - center_of_mass
    same_side_ratio = 0.0
    same_side_ratio_soft = 0.0
    if np.linalg.norm(radial_vec) >= 1e-3:
        radial_unit = radial_vec / (np.linalg.norm(radial_vec) + 1e-6)
        proj = np.dot(rel_h, radial_unit)
        soft_thr = -0.02 * obj_diag
        hard_thr = 0.06 * obj_diag
        same_side_ratio_soft = float(np.mean(proj > soft_thr))
        same_side_ratio = float(np.mean(proj > hard_thr))

    # 额外同侧判定（沿拖拽方向轴）：
    # 解决“多点同向但均值接近质心”时 radial_vec 退化导致的误判。
    motion_side_ratio = 0.0
    motion_side_ratio_soft = 0.0
    motion_side_valid = False
    if mean_disp_norm >= 1e-3:
        side_proj_motion = np.dot(rel_h, mean_disp_unit)
        side_center = float(np.mean(side_proj_motion))
        if abs(side_center) >= 0.015 * obj_diag:
            motion_side_valid = True
            side_sign = 1.0 if side_center >= 0.0 else -1.0
            signed_proj = side_proj_motion * side_sign
            motion_soft_thr = -0.02 * obj_diag
            motion_hard_thr = 0.04 * obj_diag
            motion_side_ratio_soft = float(np.mean(signed_proj > motion_soft_thr))
            motion_side_ratio = float(np.mean(signed_proj > motion_hard_thr))

    is_same_side_motion = motion_side_valid and (
        (motion_side_ratio >= 0.72)
        or (motion_side_ratio_soft >= 0.90 and motion_side_ratio >= 0.55)
    )

    is_same_side = (
        (same_side_ratio >= 0.80)
        or (same_side_ratio_soft >= 0.90 and same_side_ratio >= 0.67)
        or is_same_side_motion
    )

    # 位移方向一致性：单侧模式需要“多数点同向、冲突点少”
    disp_dir_cos = np.sum(unit_disps * mean_disp_unit[None, :], axis=1)
    disp_dir_agree = float(np.mean(disp_dir_cos > 0.35))
    disp_dir_conflict = float(np.mean(disp_dir_cos < -0.15))
    single_side_motion_ok = (
        (disp_dir_agree >= 0.72)
        and (disp_dir_conflict <= 0.20)
        and (consistency >= 0.42)
    )
    single_side_candidate = (
        (is_same_side and single_side_motion_ok)
        or (
            consistency > 0.72
            and disp_dir_agree > 0.78
            and same_side_ratio_soft >= 0.70
            and disp_dir_conflict < 0.12
        )
        or (
            # 多点同向兜底：3 点以内且方向高度一致时，优先单边而不是 FREE_STRETCH
            n_pts <= 3
            and disp_dir_agree >= 0.82
            and disp_dir_conflict <= 0.15
            and consistency >= 0.55
            and (is_same_side_motion or min_axis_angle < 12.0)
        )
    )
    coherent_one_direction = (
        consistency >= 0.88
        and disp_dir_agree >= 0.90
        and disp_dir_conflict <= 0.08
    )

    # 双边分组特征
    h_proj = np.dot(rel_h, handle_axis)
    d_proj = np.dot(disps, handle_axis)
    d_perp = np.dot(disps, handle_perp_axis)
    pos_mask = h_proj > 0
    neg_mask = h_proj < 0
    pos_n = int(np.sum(pos_mask))
    neg_n = int(np.sum(neg_mask))
    side_balance = float(min(pos_n, neg_n) / max(1, n_pts))

    # 控制点空间覆盖度（用于区分“单侧多点” vs “全局缩放”）
    if n_pts >= 2:
        angles = np.mod(np.arctan2(rel_h[:, 1], rel_h[:, 0]), 2.0 * np.pi)
        angles_sorted = np.sort(angles)
        angle_gaps = np.diff(np.concatenate([angles_sorted, angles_sorted[:1] + 2.0 * np.pi]))
        max_gap = float(np.max(angle_gaps)) if angle_gaps.size > 0 else (2.0 * np.pi)
        coverage_deg = float((2.0 * np.pi - max_gap) * 180.0 / np.pi)

        sector_num = 8
        sector_size = (2.0 * np.pi) / sector_num
        sector_ids = np.floor(angles / sector_size).astype(np.int32) % sector_num
        occupied_sectors = int(np.unique(sector_ids).size)
        occ_set = set(int(v) for v in np.unique(sector_ids).tolist())
        opposite_covered = any(((s + sector_num // 2) % sector_num) in occ_set for s in occ_set)
    else:
        coverage_deg = 0.0
        occupied_sectors = 1 if n_pts == 1 else 0
        opposite_covered = False

    # 公共参数
    dynamic_step = 3 + int(influence_range * 30)
    keep_ratio = np.clip(0.4 - (influence_range * 0.38), 0.02, 0.5)
    params = {
        "all_pts": all_pts,
        "mask_u8": mask_np,
        "center_of_mass": center_of_mass,
        "h_center": h_center,
        "handle_points": handle_points,
        "target_points": target_points,
        "influence_range": float(influence_range),
        "dynamic_step": int(dynamic_step),
        "keep_ratio": float(keep_ratio),
        "avg_vec": avg_vec.astype(np.float32),
        "obj_diag": float(obj_diag),
        "obj_major_axis": obj_major_axis,
        "obj_minor_axis": obj_minor_axis,
        "elongation": float(elongation),
        "handle_axis": handle_axis,
        "handle_perp_axis": handle_perp_axis,
        "min_axis_angle": float(min_axis_angle),
        "same_side_ratio": float(same_side_ratio),
        "same_side_ratio_soft": float(same_side_ratio_soft),
        "motion_side_ratio": float(motion_side_ratio),
        "motion_side_ratio_soft": float(motion_side_ratio_soft),
        "is_same_side_motion": bool(is_same_side_motion),
        "consistency": float(consistency),
        "disp_dir_agree": float(disp_dir_agree),
        "disp_dir_conflict": float(disp_dir_conflict),
        "single_side_motion_ok": bool(single_side_motion_ok),
        "motion_mag": float(motion_mag),
        "coverage_deg": float(coverage_deg),
        "occupied_sectors": int(occupied_sectors),
        "opposite_covered": bool(opposite_covered),
    }

    # 0) 运动量太小 -> 自由拉伸兜底
    if motion_mag < 0.35:
        print(f"[Non-Rigid Intent] FREE_STRETCH (tiny motion={motion_mag:.3f})")
        return "FREE_STRETCH", params

    # 1) 先尝试均匀缩放（保形）
    # 说明：若先判双边，会把很多“整体内收/外扩”截走，导致缩放识别率偏低。
    global_coverage_ok_relaxed = (
        (
            (coverage_deg >= 150.0)
            and (occupied_sectors >= 3)
            and bool(opposite_covered)
        )
        or (
            (n_pts >= 4)
            and (coverage_deg >= 115.0)
            and (occupied_sectors >= 3)
        )
    )
    if n_pts >= 3 and side_balance >= 0.10 and global_coverage_ok_relaxed:
        scale, R, t, sim_res = _fit_similarity(handle_points, target_points)
        if scale is not None:
            rel_src = handle_points - center_of_mass
            rel_tgt = target_points - center_of_mass
            src_r = np.linalg.norm(rel_src, axis=1)
            tgt_r = np.linalg.norm(rel_tgt, axis=1)
            # 近质心点的径向方向不稳定，容易拉低缩放判定置信度
            valid_r = src_r > max(1.0, obj_diag * 0.06)
            if np.sum(valid_r) >= 2:
                ratios = tgt_r[valid_r] / (src_r[valid_r] + 1e-6)
                ratio_med = float(np.median(ratios))
                ratio_cv = float(np.std(ratios) / (abs(ratio_med) + 1e-6))
            else:
                ratio_med = float(scale)
                ratio_cv = 1.0

            valid_idx = np.where(valid_r)[0]
            if valid_idx.size >= 2:
                rel_src_v = rel_src[valid_idx]
                disps_v = disps[valid_idx]
                src_r_v = np.linalg.norm(rel_src_v, axis=1)
                disp_r_v = np.linalg.norm(disps_v, axis=1)
                radial_unit = rel_src_v / (src_r_v[:, None] + 1e-6)
                disp_unit = disps_v / (disp_r_v[:, None] + 1e-6)
                radial_dot = np.sum(disp_unit * radial_unit, axis=1)
                tangential_abs = np.abs(
                    disp_unit[:, 0] * radial_unit[:, 1] - disp_unit[:, 1] * radial_unit[:, 0]
                )
                radial_align = float(np.mean(np.abs(radial_dot)))
                tangential_mean = float(np.mean(tangential_abs))
                radial_sign_consistency = float(np.abs(np.mean(np.sign(radial_dot + 1e-6))))
            else:
                radial_align = 0.0
                tangential_mean = 1.0
                radial_sign_consistency = 0.0

            sim_res_norm = float(sim_res / (obj_diag + 1e-6))
            if (
                (0.40 <= scale <= 2.30)
                and ratio_cv < 0.38
                and sim_res_norm < 0.15
                and radial_align > 0.55
                and tangential_mean < 0.86
                and radial_sign_consistency > 0.15
                and not coherent_one_direction
            ):
                params["scale_factor"] = float(scale)
                params["scale_rotation"] = R.astype(np.float32)
                params["scale_translation"] = t.astype(np.float32)
                params["scale_fit_residual"] = float(sim_res)
                params["scale_fit_residual_norm"] = float(sim_res_norm)
                params["scale_radial_align"] = float(radial_align)
                params["scale_tangential"] = float(tangential_mean)
                params["scale_radial_sign_consistency"] = float(radial_sign_consistency)
                print(
                    f"[Non-Rigid Intent] UNIFORM_SCALE (pre-bilateral, s={scale:.3f}, "
                    f"cv={ratio_cv:.3f}, err={sim_res_norm:.3f}, "
                    f"rad={radial_align:.3f}, tan={tangential_mean:.3f}, "
                    f"cov={coverage_deg:.1f}°, sec={occupied_sectors})"
                )
                return "UNIFORM_SCALE", params

    # 2) 双边均匀拉伸/压缩（支持多点）
    if pos_n >= 1 and neg_n >= 1 and side_balance >= 0.25:
        pos_mean = float(np.mean(d_proj[pos_mask])) if pos_n > 0 else 0.0
        neg_mean = float(np.mean(d_proj[neg_mask])) if neg_n > 0 else 0.0
        opposite_motion = (pos_mean * neg_mean) < -0.20
        axial_energy = float(np.mean(np.abs(d_proj)))
        perp_energy = float(np.mean(np.abs(d_perp)))
        perp_ratio = perp_energy / (axial_energy + 1e-6)
        mag_ratio = min(abs(pos_mean), abs(neg_mean)) / (max(abs(pos_mean), abs(neg_mean)) + 1e-6)

        if opposite_motion and perp_ratio < 0.45 and mag_ratio > 0.30:
            outward_score = float(np.mean(np.sign(h_proj + 1e-6) * d_proj))
            bilateral_type = "STRETCH" if outward_score > 0 else "COMPRESS"
            params["bilateral_type"] = bilateral_type
            params["bilateral_side_balance"] = side_balance
            params["bilateral_axis_perp_ratio"] = perp_ratio
            print(
                f"[Non-Rigid Intent] BILATERAL_UNIFORM ({bilateral_type}, "
                f"balance={side_balance:.2f}, perp_ratio={perp_ratio:.2f})"
            )
            return "BILATERAL_UNIFORM", params

    # 3) 均匀缩放（保形，严格版本兜底）
    # 关键约束：
    # - 至少 3 个点（避免 2 点退化拟合）
    # - 控制点必须有“全局覆盖”而非单侧聚集
    global_coverage_ok = (
        (coverage_deg >= 150.0)
        and (occupied_sectors >= 3)
        and bool(opposite_covered)
    )
    if n_pts >= 3 and side_balance >= 0.14 and global_coverage_ok:
        scale, R, t, sim_res = _fit_similarity(handle_points, target_points)
        if scale is not None:
            rel_src = handle_points - center_of_mass
            rel_tgt = target_points - center_of_mass
            src_r = np.linalg.norm(rel_src, axis=1)
            tgt_r = np.linalg.norm(rel_tgt, axis=1)
            valid_r = src_r > max(1.0, obj_diag * 0.06)
            if np.sum(valid_r) >= 2:
                ratios = tgt_r[valid_r] / (src_r[valid_r] + 1e-6)
                ratio_med = float(np.median(ratios))
                ratio_cv = float(np.std(ratios) / (abs(ratio_med) + 1e-6))
            else:
                ratio_med = float(scale)
                ratio_cv = 1.0

            # 额外判据：缩放位移应主要沿“质心->控制点”的径向，切向成分不能过大。
            valid_idx = np.where(valid_r)[0]
            if valid_idx.size >= 2:
                rel_src_v = rel_src[valid_idx]
                disps_v = disps[valid_idx]
                src_r_v = np.linalg.norm(rel_src_v, axis=1)
                disp_r_v = np.linalg.norm(disps_v, axis=1)
                radial_unit = rel_src_v / (src_r_v[:, None] + 1e-6)
                disp_unit = disps_v / (disp_r_v[:, None] + 1e-6)
                radial_dot = np.sum(disp_unit * radial_unit, axis=1)
                tangential_abs = np.abs(
                    disp_unit[:, 0] * radial_unit[:, 1] - disp_unit[:, 1] * radial_unit[:, 0]
                )
                radial_align = float(np.mean(np.abs(radial_dot)))
                tangential_mean = float(np.mean(tangential_abs))
                radial_sign_consistency = float(np.abs(np.mean(np.sign(radial_dot + 1e-6))))
            else:
                radial_align = 0.0
                tangential_mean = 1.0
                radial_sign_consistency = 0.0

            sim_res_norm = float(sim_res / (obj_diag + 1e-6))
            if (
                (0.50 <= scale <= 2.00)
                and ratio_cv < 0.28
                and sim_res_norm < 0.12
                and radial_align > 0.64
                and tangential_mean < 0.74
                and radial_sign_consistency > 0.24
                and not coherent_one_direction
            ):
                params["scale_factor"] = float(scale)
                params["scale_rotation"] = R.astype(np.float32)
                params["scale_translation"] = t.astype(np.float32)
                params["scale_fit_residual"] = float(sim_res)
                params["scale_fit_residual_norm"] = float(sim_res_norm)
                params["scale_radial_align"] = float(radial_align)
                params["scale_tangential"] = float(tangential_mean)
                params["scale_radial_sign_consistency"] = float(radial_sign_consistency)
                print(
                    f"[Non-Rigid Intent] UNIFORM_SCALE (s={scale:.3f}, "
                    f"cv={ratio_cv:.3f}, err={sim_res_norm:.3f}, "
                    f"rad={radial_align:.3f}, tan={tangential_mean:.3f}, "
                    f"cov={coverage_deg:.1f}°, sec={occupied_sectors})"
                )
                return "UNIFORM_SCALE", params

    # 4) 单边模式（普通/轴向）优先于径向缩放兜底
    if single_side_candidate:
        axis_vertical = (
            min_axis_angle < 10.0
            and abs(float(mean_disp_unit[1])) >= abs(float(mean_disp_unit[0]))
        )
        if axis_vertical:
            params["single_side_mode"] = "AXIS"
            print(
                f"[Non-Rigid Intent] SINGLE_SIDE_AXIS "
                f"(same_side={same_side_ratio:.2f}, motion_side={motion_side_ratio:.2f}, "
                f"agree={disp_dir_agree:.2f}, conflict={disp_dir_conflict:.2f}, axis={min_axis_angle:.1f}°)"
            )
            return "SINGLE_SIDE_AXIS", params

        params["single_side_mode"] = "NORMAL"
        print(
            f"[Non-Rigid Intent] SINGLE_SIDE_NORMAL "
            f"(same_side={same_side_ratio:.2f}, motion_side={motion_side_ratio:.2f}, "
            f"agree={disp_dir_agree:.2f}, conflict={disp_dir_conflict:.2f}, axis={min_axis_angle:.1f}°)"
        )
        return "SINGLE_SIDE_NORMAL", params

    # 5) 径向缩放兜底：优先避免误判为 FREE_STRETCH
    # 典型场景：多点向中心收缩/远离中心扩张，但相似变换拟合被少数点干扰。
    radial_cov_ok = (coverage_deg >= 90.0 and occupied_sectors >= 3)
    if n_pts >= 3 and radial_cov_ok and (not coherent_one_direction):
        rel_src = handle_points - center_of_mass
        rel_tgt = target_points - center_of_mass
        src_r = np.linalg.norm(rel_src, axis=1)
        tgt_r = np.linalg.norm(rel_tgt, axis=1)
        valid_r = src_r > max(1.0, obj_diag * 0.08)

        if np.sum(valid_r) >= 2:
            rel_src_v = rel_src[valid_r]
            disps_v = disps[valid_r]
            src_r_v = np.linalg.norm(rel_src_v, axis=1)
            disp_r_v = np.linalg.norm(disps_v, axis=1)
            radial_unit = rel_src_v / (src_r_v[:, None] + 1e-6)
            disp_unit = disps_v / (disp_r_v[:, None] + 1e-6)
            radial_dot = np.sum(disp_unit * radial_unit, axis=1)
            tangential_abs = np.abs(
                disp_unit[:, 0] * radial_unit[:, 1] - disp_unit[:, 1] * radial_unit[:, 0]
            )

            radial_align = float(np.mean(np.abs(radial_dot)))
            tangential_mean = float(np.mean(tangential_abs))
            radial_mean = float(np.mean(radial_dot))
            radial_sign_consistency = float(np.abs(np.mean(np.sign(radial_dot + 1e-6))))

            if (
                radial_align > 0.50
                and tangential_mean < 0.92
                and abs(radial_mean) > 0.18
                and radial_sign_consistency > 0.10
            ):
                ratios = tgt_r[valid_r] / (src_r[valid_r] + 1e-6)
                ratio_med = float(np.median(ratios))
                ratio_cv = float(np.std(ratios) / (abs(ratio_med) + 1e-6))

                fit_scale, fit_R, fit_t, fit_res = _fit_similarity(handle_points, target_points)
                if fit_scale is None:
                    fit_scale = ratio_med
                    fit_R = np.eye(2, dtype=np.float32)
                    fit_t = np.zeros((2,), dtype=np.float32)
                    fit_res = 0.0

                scale_final = float(fit_scale)
                # 保证收缩/扩张方向与径向统计一致
                if radial_mean < 0:
                    scale_final = min(scale_final, ratio_med, 0.98)
                else:
                    scale_final = max(scale_final, ratio_med, 1.02)

                scale_final = float(np.clip(scale_final, 0.40, 2.30))
                params["scale_factor"] = scale_final
                params["scale_rotation"] = np.asarray(fit_R, dtype=np.float32)
                params["scale_translation"] = np.asarray(fit_t, dtype=np.float32).reshape(2)
                params["scale_fit_residual"] = float(fit_res)
                params["scale_fit_residual_norm"] = float(fit_res / (obj_diag + 1e-6))
                params["scale_radial_align"] = float(radial_align)
                params["scale_tangential"] = float(tangential_mean)
                params["scale_radial_sign_consistency"] = float(radial_sign_consistency)
                params["scale_ratio_cv"] = float(ratio_cv)
                print(
                    f"[Non-Rigid Intent] UNIFORM_SCALE (radial-fallback, s={scale_final:.3f}, "
                    f"rad={radial_align:.3f}, tan={tangential_mean:.3f}, "
                    f"mean={radial_mean:.3f}, cv={ratio_cv:.3f}, cov={coverage_deg:.1f}°)"
                )
                return "UNIFORM_SCALE", params

    # 6) 兜底：自由拉伸
    print(
        f"[Non-Rigid Intent] FREE_STRETCH (fallback, "
        f"consistency={consistency:.2f}, same_side={same_side_ratio:.2f}, "
        f"agree={disp_dir_agree:.2f}, conflict={disp_dir_conflict:.2f})"
    )
    return "FREE_STRETCH", params

def _nonrigid_01_center_pinned(params):
    """【非刚性-01】Center Pinned 模式"""
    all_pts = params['all_pts']
    center_of_mass = params['center_of_mass']
    
    desc = f"Free Stretch (Center Pinned, Deg={params['influence_range']:.2f})"
    dists = np.linalg.norm(all_pts - center_of_mass, axis=1)
    sorted_idx = np.argsort(dists)
    num_keep = max(5, int(len(all_pts) * params['keep_ratio']))
    anchors = all_pts[sorted_idx[:num_keep]] 
    return anchors[::params['dynamic_step']], desc


def _merge_anchor_sets(*anchor_sets):
    """合并多个锚点集合并去重，统一返回 (N, 2) float32。"""
    valid_sets = []
    for arr in anchor_sets:
        if arr is None:
            continue
        arr_np = np.asarray(arr, dtype=np.float32)
        if arr_np.ndim != 2 or arr_np.shape[0] == 0:
            continue
        if arr_np.shape[1] < 2:
            continue
        valid_sets.append(arr_np[:, :2])

    if len(valid_sets) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    merged = np.vstack(valid_sets)
    merged = np.unique(np.round(merged).astype(np.int32), axis=0).astype(np.float32)
    return merged


def _build_outer_shell_anchors(mask_u8, handle_points, influence_range, min_keep=20, max_keep=260):
    """
    构建“外边界保护锚点”：
    - 仅从主体最外层边界采样
    - 优先保留离 handle 更远的边界点，避免把运动侧完全锁死
    """
    if mask_u8 is None:
        return np.zeros((0, 2), dtype=np.float32), {"reason": "mask_none"}

    mask_bin = (mask_u8 > 127).astype(np.uint8)
    if np.sum(mask_bin) == 0:
        return np.zeros((0, 2), dtype=np.float32), {"reason": "empty_mask"}

    # ring_width 越小，越强调“只锁最外层边缘”
    ring_width = int(np.clip(round(1 + (1.0 - float(influence_range)) * 2.0), 1, 4))
    k = np.ones((ring_width * 2 + 1, ring_width * 2 + 1), np.uint8)
    eroded = cv2.erode(mask_bin, k, iterations=1)
    shell = (mask_bin > 0) & (eroded == 0)

    ys, xs = np.where(shell)
    if xs.size == 0:
        ys, xs = np.where(mask_bin > 0)
        if xs.size == 0:
            return np.zeros((0, 2), dtype=np.float32), {"reason": "no_shell_points"}

    shell_pts = np.stack([xs, ys], axis=1).astype(np.float32)

    hp = np.asarray(handle_points, dtype=np.float32).reshape(-1, 2) if handle_points is not None else np.zeros((0, 2), dtype=np.float32)
    if hp.shape[0] > 0:
        diff = shell_pts[:, None, :] - hp[None, :, :]
        dists = np.min(np.linalg.norm(diff, axis=2), axis=1)
    else:
        dists = np.ones((shell_pts.shape[0],), dtype=np.float32)

    # influence_range 越大（影响范围越大）=> 保留更少外边界锚点
    q = float(np.clip(35.0 + 35.0 * float(influence_range), 35.0, 70.0))
    thr = float(np.percentile(dists, q)) if dists.size > 0 else 0.0
    keep_mask = dists >= thr

    if int(np.sum(keep_mask)) < int(min_keep):
        relax_q = max(18.0, q - 18.0)
        thr = float(np.percentile(dists, relax_q)) if dists.size > 0 else 0.0
        keep_mask = dists >= thr

    kept = shell_pts[keep_mask] if np.any(keep_mask) else shell_pts
    kept_dist = dists[keep_mask] if np.any(keep_mask) else dists

    target_keep = int(np.clip(round(220 - 120 * float(influence_range)), min_keep, max_keep))
    if kept.shape[0] > target_keep:
        order = np.argsort(kept_dist)[::-1]
        kept = kept[order[:target_keep]]

    step = max(1, int(round(1 + 2 * float(influence_range))))
    kept = np.unique(np.round(kept).astype(np.int32), axis=0).astype(np.float32)
    if step > 1 and kept.shape[0] > min_keep:
        kept = kept[::step]

    if kept.shape[0] < min_keep and shell_pts.shape[0] > 0:
        order = np.argsort(dists)[::-1][:min_keep]
        kept = shell_pts[order]

    info = {
        "ring_width": int(ring_width),
        "distance_percentile": float(q),
        "distance_threshold": float(thr),
        "shell_points": int(shell_pts.shape[0]),
        "anchor_count": int(kept.shape[0]),
    }
    return kept.astype(np.float32), info


def _build_outer_shell_transport_pairs(
    mask_u8,
    handle_points,
    target_points,
    center_of_mass,
    influence_range=0.5,
    max_pairs=220,
):
    """
    构建“外壳厚度守恒”约束点对（src->tgt）：
    - 只选 handle 外侧的壳层点（相对质心更外）
    - 外侧壳层点与最近 handle 同位移，避免外壳被额外拉伸/压缩
    """
    if mask_u8 is None:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), {"reason": "mask_none"}

    handles = np.asarray(handle_points, dtype=np.float32).reshape(-1, 2)
    targets = np.asarray(target_points, dtype=np.float32).reshape(-1, 2)
    n = min(handles.shape[0], targets.shape[0])
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), {"reason": "no_controls"}

    handles = handles[:n]
    targets = targets[:n]
    disps = targets - handles
    disp_mag = np.linalg.norm(disps, axis=1)
    if np.max(disp_mag) < 0.5:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), {"reason": "tiny_motion"}

    mask_bin = (mask_u8 > 127).astype(np.uint8)
    if np.sum(mask_bin) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), {"reason": "empty_mask"}

    h_img, w_img = mask_bin.shape[:2]
    ring_width = int(np.clip(round(1 + (1.0 - float(influence_range)) * 1.5), 1, 3))
    k = np.ones((ring_width * 2 + 1, ring_width * 2 + 1), np.uint8)
    eroded = cv2.erode(mask_bin, k, iterations=1)
    shell = (mask_bin > 0) & (eroded == 0)

    ys, xs = np.where(shell)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), {"reason": "no_shell"}

    shell_pts = np.stack([xs, ys], axis=1).astype(np.float32)
    diff = shell_pts[:, None, :] - handles[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    nn_idx = np.argmin(dists, axis=1)
    nn_dist = np.min(dists, axis=1)

    obj_diag = float(np.sqrt((np.max(xs) - np.min(xs) + 1e-6) ** 2 + (np.max(ys) - np.min(ys) + 1e-6) ** 2) + 1e-6)
    # 统一语义：range 越大，外壳跟随半径越小（额外约束更局部）
    # 保持 range=0.5 时与旧行为接近。
    influence_radius = float(np.clip(obj_diag * (0.20 + 0.35 * (1.0 - float(influence_range))), 8.0, 220.0))

    radial = handles[nn_idx] - np.asarray(center_of_mass, dtype=np.float32)[None, :]
    radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
    radial_unit = radial / (radial_norm + 1e-6)
    outside_score = np.sum((shell_pts - handles[nn_idx]) * radial_unit, axis=1)

    # 仅保留“更外侧”的壳层点，并限制在控制点邻域内
    sel = (outside_score > 0.0) & (nn_dist <= influence_radius) & (disp_mag[nn_idx] > 0.5)
    if int(np.sum(sel)) < 12:
        sel = (outside_score > -1.5) & (nn_dist <= influence_radius) & (disp_mag[nn_idx] > 0.5)

    if not np.any(sel):
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), {"reason": "no_valid_shell_pairs"}

    src_pts = shell_pts[sel]
    idx_sel = nn_idx[sel]
    scores = outside_score[sel] + 0.25 * disp_mag[idx_sel]
    tgt_pts = src_pts + disps[idx_sel]
    tgt_pts[:, 0] = np.clip(tgt_pts[:, 0], 0, w_img - 1)
    tgt_pts[:, 1] = np.clip(tgt_pts[:, 1], 0, h_img - 1)

    if src_pts.shape[0] > max_pairs:
        sampled = _sample_anchors_spatially_balanced(src_pts, scores, max_anchors=int(max_pairs))
        if sampled.shape[0] > 0:
            sampled_i32 = np.unique(np.round(sampled).astype(np.int32), axis=0)
            # 回查 sampled 点在 src_pts 中的最近索引
            diff_s = src_pts[:, None, :] - sampled_i32[None, :, :].astype(np.float32)
            idx_back = np.argmin(np.linalg.norm(diff_s, axis=2), axis=0)
            src_pts = src_pts[idx_back]
            tgt_pts = tgt_pts[idx_back]

    src_pts = np.unique(np.round(src_pts).astype(np.int32), axis=0).astype(np.float32)
    if src_pts.shape[0] > 0:
        # 重新按最近点回查 tgt，保证一一对应
        diff_re = shell_pts[None, :, :] - src_pts[:, None, :]
        idx_re = np.argmin(np.linalg.norm(diff_re, axis=2), axis=1)
        nn_re = nn_idx[idx_re]
        tgt_pts = src_pts + disps[nn_re]
        tgt_pts[:, 0] = np.clip(tgt_pts[:, 0], 0, w_img - 1)
        tgt_pts[:, 1] = np.clip(tgt_pts[:, 1], 0, h_img - 1)
    else:
        tgt_pts = np.zeros((0, 2), dtype=np.float32)

    info = {
        "ring_width": int(ring_width),
        "influence_radius": float(influence_radius),
        "pair_count": int(src_pts.shape[0]),
    }
    return src_pts.astype(np.float32), tgt_pts.astype(np.float32), info


def _build_similarity_shell_pairs(
    mask_u8,
    scale_factor,
    scale_rotation,
    scale_translation,
    max_pairs=260,
):
    """
    为 UNIFORM_SCALE 构建“壳层相似变换配对”：
    - 从主体外壳采样点作为 src
    - 用相似变换预测这些点的 tgt
    这样在少控制点时也能保持整体保形，不至于扭曲。
    """
    if mask_u8 is None:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            {"reason": "mask_none"},
        )

    mask_bin = (mask_u8 > 127).astype(np.uint8)
    if np.sum(mask_bin) == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            {"reason": "empty_mask"},
        )

    h_img, w_img = mask_bin.shape[:2]
    k = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask_bin, k, iterations=1)
    shell = (mask_bin > 0) & (eroded == 0)
    ys, xs = np.where(shell)
    if xs.size == 0:
        ys, xs = np.where(mask_bin > 0)
        if xs.size == 0:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
                {"reason": "no_shell_points"},
            )

    src_pts = np.stack([xs, ys], axis=1).astype(np.float32)
    if src_pts.shape[0] > int(max_pairs):
        scores = np.ones((src_pts.shape[0],), dtype=np.float32)
        src_pts = _sample_anchors_spatially_balanced(src_pts, scores, max_anchors=int(max_pairs))

    src_pts = np.unique(np.round(src_pts).astype(np.int32), axis=0).astype(np.float32)
    if src_pts.shape[0] == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            {"reason": "no_sampled_points"},
        )

    s = float(scale_factor)
    R = np.asarray(scale_rotation, dtype=np.float32).reshape(2, 2)
    t = np.asarray(scale_translation, dtype=np.float32).reshape(2)
    tgt_pts = s * (src_pts @ R.T) + t[None, :]
    tgt_pts[:, 0] = np.clip(tgt_pts[:, 0], 0, w_img - 1)
    tgt_pts[:, 1] = np.clip(tgt_pts[:, 1], 0, h_img - 1)

    info = {
        "pair_count": int(src_pts.shape[0]),
        "scale_factor": float(s),
    }
    return src_pts.astype(np.float32), tgt_pts.astype(np.float32), info


def _build_free_stretch_guard_pairs(
    all_pts,
    center_of_mass,
    handle_points,
    influence_range=0.5,
    max_pairs=52,
):
    """
    FREE_STRETCH 专用“内部恒等约束点对”：
    - 仅在主体内部采样（避开边缘）
    - 尽量远离控制点（避免压制拖拽）
    - src=tgt，不引入额外目标，只用于抑制过拉
    """
    pts = np.asarray(all_pts, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] == 0:
        z = np.zeros((0, 2), dtype=np.float32)
        return z, z, {"reason": "empty_points"}

    center = np.asarray(center_of_mass, dtype=np.float32).reshape(2)
    rel = pts - center[None, :]
    rad = np.linalg.norm(rel, axis=1)
    if rad.size == 0:
        z = np.zeros((0, 2), dtype=np.float32)
        return z, z, {"reason": "no_radius"}

    # 只取中内层，避免边缘锁死
    q_lo = 10.0
    q_hi = float(np.clip(58.0 - 20.0 * float(influence_range), 34.0, 58.0))
    r_lo = float(np.percentile(rad, q_lo))
    r_hi = float(np.percentile(rad, q_hi))
    keep_rad = (rad >= r_lo) & (rad <= r_hi)
    cand = pts[keep_rad]
    cand_rad = rad[keep_rad]

    if cand.shape[0] < 12:
        r_mid = float(np.percentile(rad, 55.0))
        keep_mid = rad <= r_mid
        cand = pts[keep_mid]
        cand_rad = rad[keep_mid]
        if cand.shape[0] < 12:
            cand = pts
            cand_rad = rad

    hp = np.asarray(handle_points, dtype=np.float32).reshape(-1, 2)
    if hp.shape[0] > 0 and cand.shape[0] > 0:
        hp = np.unique(np.round(hp).astype(np.int32), axis=0).astype(np.float32)
        diff = cand[:, None, :] - hp[None, :, :]
        d_min = np.min(np.linalg.norm(diff, axis=2), axis=1)

        # range 越大（更自由）→ guard 越少；range 越小（更稳）→ guard 越多
        q_far = float(np.clip(52.0 + 28.0 * float(influence_range), 52.0, 80.0))
        thr_far = float(np.percentile(d_min, q_far))
        keep_far = d_min >= thr_far
        if int(np.sum(keep_far)) >= 12:
            cand = cand[keep_far]
            cand_rad = cand_rad[keep_far]
            d_min = d_min[keep_far]
    else:
        d_min = np.ones((cand.shape[0],), dtype=np.float32)
        thr_far = 0.0

    if cand.shape[0] == 0:
        z = np.zeros((0, 2), dtype=np.float32)
        return z, z, {"reason": "no_candidates"}

    keep_n = int(np.clip(round(float(max_pairs) * (1.0 - 0.55 * float(influence_range))), 10, max_pairs))
    scores = 0.60 * (1.0 / (cand_rad + 1e-3)) + 0.40 * d_min
    sampled = _sample_anchors_spatially_balanced(cand, scores, max_anchors=keep_n)
    sampled = np.unique(np.round(sampled).astype(np.int32), axis=0).astype(np.float32)

    info = {
        "pair_count": int(sampled.shape[0]),
        "max_pairs": int(max_pairs),
        "selected_pairs": int(keep_n),
        "q_rad": [float(q_lo), float(q_hi)],
        "distance_threshold": float(thr_far),
    }
    return sampled, sampled.copy(), info


def _nonrigid_02_root_pinned(params):
    """【非刚性-02】单边模式（普通 / 轴向）

    核心逻辑：
    - 锚点位置由【控制点位置】决定基础方向
    - 运动方向决定是否需要L形（斜向运动才需要）

    控制点在上方 + 向下压 → 只固定底部
    控制点在右上 + 斜向右上拉 → 固定底部+左侧（L形）
    控制点在右上 + 主要向下压 → 只固定底部（不需要L形）
    """
    all_pts = params['all_pts']
    h_center = params['h_center']
    center_of_mass = params['center_of_mass']
    handle_points = params['handle_points']
    influence_range = params['influence_range']

    # ========================================
    # 计算控制点相对于质心的位置（决定基础锚点方向）
    # ========================================
    handle_center = np.mean(handle_points, axis=0)
    relative_pos = handle_center - center_of_mass

    # 预处理边界信息
    pts_centered = all_pts - center_of_mass
    proj_x = pts_centered[:, 0]
    proj_y = pts_centered[:, 1]
    x_min, x_max = np.min(proj_x), np.max(proj_x)
    y_min, y_max = np.min(proj_y), np.max(proj_y)
    x_range = x_max - x_min + 1e-6
    y_range = y_max - y_min + 1e-6

    # 控制点在物体的相对位置 (0=中心, ±1=边缘)
    rel_x = relative_pos[0] / (x_range / 2) if x_range > 1e-3 else 0
    rel_y = relative_pos[1] / (y_range / 2) if y_range > 1e-3 else 0

    # ========================================
    # 计算运动方向
    # ========================================
    target_points = params.get('target_points')
    if target_points is not None and len(target_points) > 0:
        motion_vec = np.mean(target_points - handle_points, axis=0)
    else:
        motion_vec = params.get('avg_vec', np.array([0.0, -1.0]))

    motion_norm = np.linalg.norm(motion_vec)
    if motion_norm < 1e-3:
        motion_vec = np.array([0.0, -1.0])
        motion_norm = 1.0

    motion_unit = motion_vec / motion_norm
    motion_x, motion_y = motion_unit[0], motion_unit[1]

    print(f"[Single-Side] Control pos: ({rel_x:.2f}, {rel_y:.2f}), Motion: ({motion_x:.2f}, {motion_y:.2f})")

    # ========================================
    # 【核心判断】区分拉伸/挤压 vs 普通模式
    # ========================================
    # 定义：运动方向与4个轴（上下左右）的夹角
    # - 夹角 < 10度：挤压或拉伸模式（单边锚点）
    # - 夹角 >= 10度：普通模式（L形锚点）

    # 4个轴的单位向量
    axis_up = np.array([0.0, -1.0])
    axis_down = np.array([0.0, 1.0])
    axis_left = np.array([-1.0, 0.0])
    axis_right = np.array([1.0, 0.0])

    # 计算运动方向与各轴的夹角
    def angle_to_axis(motion, axis):
        dot = np.dot(motion, axis)
        return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

    angle_to_up = angle_to_axis(motion_unit, axis_up)
    angle_to_down = angle_to_axis(motion_unit, axis_down)
    angle_to_left = angle_to_axis(motion_unit, axis_left)
    angle_to_right = angle_to_axis(motion_unit, axis_right)

    # 找到最小夹角
    min_angle = min(angle_to_up, angle_to_down, angle_to_left, angle_to_right)
    axis_threshold = 10  # 度
    forced_mode = str(params.get('single_side_mode', '')).upper()

    up_or_down_axis_aligned = (
        abs(motion_y) >= abs(motion_x)
        and min(angle_to_up, angle_to_down) < axis_threshold
    )

    # 判断是否为轴向运动（垂直方向：向上/向下）
    if forced_mode == "AXIS":
        is_axis_aligned = up_or_down_axis_aligned
    elif forced_mode == "NORMAL":
        is_axis_aligned = False
    else:
        is_axis_aligned = up_or_down_axis_aligned

    print(
        f"[Single-Side] Axis angles: Up={angle_to_up:.1f}°, Down={angle_to_down:.1f}°, "
        f"Left={angle_to_left:.1f}°, Right={angle_to_right:.1f}°, Min={min_angle:.1f}°, "
        f"forced={forced_mode or 'AUTO'}, vertical={up_or_down_axis_aligned}"
    )

    # 轴向模式支持垂直方向：
    # - dot <= 0: 视为压缩（向内按）
    # - dot > 0: 视为拉伸（向外拉）
    is_stretch = False
    is_compress = False

    if is_axis_aligned:
        # 计算控制点偏移方向（从质心指向控制点）
        if np.linalg.norm(relative_pos) > 1e-3:
            handle_dir = relative_pos / np.linalg.norm(relative_pos)
            # 计算运动方向与控制点偏移的点积
            dot_product = np.dot(motion_unit, handle_dir)

            if dot_product <= 0:
                is_compress = True
                print(f"[Single-Side] Axis COMPRESS mode (dot={dot_product:.2f})")
            else:
                is_stretch = True
                print(f"[Single-Side] Axis STRETCH mode (dot={dot_product:.2f})")
        else:
            # 退化场景：控制点接近质心，按垂直方向决定压缩/拉伸，避免退回 NORMAL。
            if motion_y >= 0.0:
                is_compress = True
                print("[Single-Side] Axis fallback to COMPRESS (handle near center)")
            else:
                is_stretch = True
                print("[Single-Side] Axis fallback to STRETCH (handle near center)")
    else:
        print(f"[Single-Side] NORMAL mode (not axis-aligned, min_angle={min_angle:.1f}° >= {axis_threshold}°)")

    # ========================================
    # 【挤压模式验证】预先检查可用锚点区域是否足够
    # 如果不够，退回到普通模式（L形锚点）
    # ========================================
    min_edge_coverage_ratio = 0.25  # 最小边缘覆盖率（25%）
    max_edge_angle_for_compress = 20  # 挤压模式的最大边缘角度偏离（与水平/垂直）

    if is_compress:
        # 预计算目标边缘的有效覆盖率
        def compute_valid_edge_coverage(pts_centered, edge_type, total_range):
            """
            计算边缘上角度合适的区域占总边缘的比例
            edge_type: 'bottom', 'top', 'left', 'right'
            """
            proj_x_local = pts_centered[:, 0]
            proj_y_local = pts_centered[:, 1]

            if edge_type in ['bottom', 'top']:
                # 水平边缘
                x_min_l, x_max_l = np.min(proj_x_local), np.max(proj_x_local)
                x_range_l = x_max_l - x_min_l + 1e-6

                # 提取边缘轮廓
                num_bins = min(50, max(10, int(x_range_l / 5)))
                bins = np.linspace(x_min_l, x_max_l, num_bins + 1)
                contour_pts = []

                for j in range(num_bins):
                    bin_m = (proj_x_local >= bins[j]) & (proj_x_local < bins[j+1])
                    if bin_m.sum() > 0:
                        if edge_type == 'bottom':
                            idx = np.argmax(proj_y_local[bin_m])
                        else:
                            idx = np.argmin(proj_y_local[bin_m])
                        pts_in_bin = pts_centered[bin_m]
                        contour_pts.append(pts_in_bin[idx])

                if len(contour_pts) < 3:
                    return 0.0

                contour_pts = np.array(contour_pts)

                # 计算局部斜率
                valid_count = 0
                for k in range(len(contour_pts)):
                    k_prev = max(0, k - 1)
                    k_next = min(len(contour_pts) - 1, k + 1)
                    if k_prev == k_next:
                        valid_count += 1
                        continue
                    dx = contour_pts[k_next][0] - contour_pts[k_prev][0]
                    dy = contour_pts[k_next][1] - contour_pts[k_prev][1]
                    if abs(dx) < 1e-6:
                        angle = 90
                    else:
                        angle = abs(np.degrees(np.arctan(dy / dx)))
                    if angle <= max_edge_angle_for_compress:
                        valid_count += 1

                return valid_count / len(contour_pts)

            else:
                # 垂直边缘
                y_min_l, y_max_l = np.min(proj_y_local), np.max(proj_y_local)
                y_range_l = y_max_l - y_min_l + 1e-6

                num_bins = min(50, max(10, int(y_range_l / 5)))
                bins = np.linspace(y_min_l, y_max_l, num_bins + 1)
                contour_pts = []

                for j in range(num_bins):
                    bin_m = (proj_y_local >= bins[j]) & (proj_y_local < bins[j+1])
                    if bin_m.sum() > 0:
                        if edge_type == 'left':
                            idx = np.argmin(proj_x_local[bin_m])
                        else:
                            idx = np.argmax(proj_x_local[bin_m])
                        pts_in_bin = pts_centered[bin_m]
                        contour_pts.append(pts_in_bin[idx])

                if len(contour_pts) < 3:
                    return 0.0

                contour_pts = np.array(contour_pts)

                valid_count = 0
                for k in range(len(contour_pts)):
                    k_prev = max(0, k - 1)
                    k_next = min(len(contour_pts) - 1, k + 1)
                    if k_prev == k_next:
                        valid_count += 1
                        continue
                    dx = contour_pts[k_next][0] - contour_pts[k_prev][0]
                    dy = contour_pts[k_next][1] - contour_pts[k_prev][1]
                    if abs(dy) < 1e-6:
                        angle = 90
                    else:
                        angle = abs(np.degrees(np.arctan(dx / dy)))
                    if angle <= max_edge_angle_for_compress:
                        valid_count += 1

                return valid_count / len(contour_pts)

        # 轴向压缩：按垂直方向选择受压边
        target_edge = 'bottom' if motion_y >= 0.0 else 'top'

        # 计算有效覆盖率
        coverage = compute_valid_edge_coverage(pts_centered, target_edge, x_range if target_edge in ['bottom', 'top'] else y_range)
        print(f"[Single-Side] Compress validation: {target_edge} edge coverage = {coverage:.1%}")

        if coverage < min_edge_coverage_ratio:
            # 覆盖率不足，退回普通模式
            print(f"[Single-Side] Compress coverage too low ({coverage:.1%} < {min_edge_coverage_ratio:.0%}), fallback to NORMAL mode")
            is_compress = False
            is_axis_aligned = False

    # ========================================
    # 根据控制点位置确定可能的固定边
    # ========================================
    pos_threshold = 0.2

    # 控制点在上方(y<0) → 可能固定底部
    # 控制点在下方(y>0) → 可能固定顶部
    could_fix_bottom = rel_y < -pos_threshold
    could_fix_top = rel_y > pos_threshold

    # 控制点在右侧(x>0) → 可能固定左侧
    # 控制点在左侧(x<0) → 可能固定右侧
    could_fix_left = rel_x > pos_threshold
    could_fix_right = rel_x < -pos_threshold

    # ========================================
    # 根据拉伸/挤压模式决定锚点策略
    # ========================================

    # 确定固定边
    fix_bottom = False
    fix_top = False
    fix_left = False
    fix_right = False

    if is_compress:
        # ========================================
        # 【挤压模式】：只固定单边，锚点在运动方向的前方
        # 例如：高跟鞋鞋跟往下按 → 只固定底部边缘
        # ========================================
        if motion_y >= 0.0:
            fix_bottom = True
            print(f"[Single-Side] COMPRESS mode -> bottom edge only")
        else:
            fix_top = True
            print(f"[Single-Side] COMPRESS mode -> top edge only")

    elif is_stretch:
        # ========================================
        # 【拉伸模式】：单边锚点，固定运动方向反方向的边
        # 注：当前轴向拉伸已启用（垂直方向）
        # ========================================

        # 根据运动方向确定固定哪条边（运动的反方向）
        if motion_y < 0:  # 向上拉 → 固定底部
            fix_bottom = True
        else:
            fix_top = True

        print(f"[Single-Side] STRETCH mode -> vertical single edge only")

    else:
        # ========================================
        # 【普通模式】：单边反侧（位置+方向联合）
        # 不再使用 L 形双边；更符合“圆形直觉”
        # 例如：控制点在上侧（左上+右上）-> 固定底部单边
        # ========================================
        pos_norm = np.linalg.norm(relative_pos)
        if pos_norm > 1e-3:
            pos_unit = relative_pos / pos_norm
        else:
            pos_unit = np.array([0.0, 0.0], dtype=np.float32)

        # 位置主导、方向辅助
        fuse = 0.72 * pos_unit + 0.28 * motion_unit
        fuse_norm = np.linalg.norm(fuse)
        if fuse_norm < 1e-6:
            if pos_norm > 1e-3:
                fuse = pos_unit
            else:
                # 退化：没有明确位置时，用运动反方向作为“受力侧”
                fuse = motion_unit
            fuse_norm = np.linalg.norm(fuse) + 1e-6
        fuse = fuse / fuse_norm

        # opposite 是固定侧方向
        opposite = -fuse

        # 多点时，用控制点分布形态做轻偏置：
        # 横向分布更宽 -> 优先上下边；纵向分布更高 -> 优先左右边。
        if handle_points.shape[0] >= 2:
            rel_hp = handle_points - center_of_mass
            spread_x = float(np.std(rel_hp[:, 0]))
            spread_y = float(np.std(rel_hp[:, 1]))
            if spread_x > spread_y * 1.25:
                y_score = 0.70 * pos_unit[1] + 0.30 * motion_unit[1]
                if abs(y_score) > 0.08:
                    opposite = np.array([0.0, 1.0 if y_score < 0 else -1.0], dtype=np.float32)
            elif spread_y > spread_x * 1.25:
                x_score = 0.70 * pos_unit[0] + 0.30 * motion_unit[0]
                if abs(x_score) > 0.08:
                    opposite = np.array([1.0 if x_score < 0 else -1.0, 0.0], dtype=np.float32)

        # 清零后只启用一条边
        fix_bottom = False
        fix_top = False
        fix_left = False
        fix_right = False

        if abs(opposite[1]) >= abs(opposite[0]):
            if opposite[1] > 0:
                fix_bottom = True
            else:
                fix_top = True
        else:
            if opposite[0] > 0:
                fix_right = True
            else:
                fix_left = True

        print(
            f"[Single-Side] NORMAL mode -> single opposite edge "
            f"(fuse=({fuse[0]:.2f},{fuse[1]:.2f}), opposite=({opposite[0]:.2f},{opposite[1]:.2f}))"
        )

    # 如果没有任何边被选中，默认固定底部
    if not fix_bottom and not fix_top and not fix_left and not fix_right:
        fix_bottom = True

    print(f"[Single-Side] Fix edges: Bottom={fix_bottom}, Top={fix_top}, Left={fix_left}, Right={fix_right}")

    # ========================================
    # influence_range 控制弧长和厚度（越大越自由）
    # ========================================
    arc_ratio = 1.0 - influence_range * 0.7  # 0→100%, 1→30%
    num_layers = max(1, 4 - int(influence_range * 3))  # 0→4层, 1→1层
    # 轴向模式与全局语义一致：
    # range 越大 -> 固定锚点越少（弧更短）；range 越小 -> 固定锚点越多（弧更长）
    axis_arc_ratio = float(np.clip(1.0 - 0.7 * float(influence_range), 0.30, 1.0))

    all_edge_points = []
    modes = []

    # ========================================
    # 【通用】边缘角度过滤参数和函数
    # 在拉伸/挤压模式下，过滤角度偏离太大的边缘点
    # ========================================
    max_edge_angle_deviation = 20  # 最大允许的边缘角度偏离（度）

    def filter_edge_by_angle(pts_in_region, proj_main, proj_secondary, edge_type, is_axis_mode):
        """
        根据边缘角度过滤锚点区域

        参数:
            pts_in_region: 区域内的所有点
            proj_main: 主轴投影（水平边缘用x，垂直边缘用y）
            proj_secondary: 次轴投影（水平边缘用y，垂直边缘用x）
            edge_type: 'bottom', 'top', 'left', 'right'
            is_axis_mode: 是否为轴向模式（拉伸/挤压）

        返回:
            过滤后的 (pts, proj_main, proj_secondary) 或原始数据
        """
        if not is_axis_mode or len(pts_in_region) < 3:
            return pts_in_region, proj_main, proj_secondary

        # 确定排序和边缘提取方向
        is_horizontal = edge_type in ['bottom', 'top']

        if is_horizontal:
            # 水平边缘：按x排序，提取y极值点
            main_min, main_max = np.min(proj_main), np.max(proj_main)
            main_range = main_max - main_min + 1e-6
            num_bins = min(50, max(10, int(main_range / 5)))
            bins = np.linspace(main_min, main_max, num_bins + 1)

            contour_pts = []
            contour_main = []

            for j in range(num_bins):
                bin_m = (proj_main >= bins[j]) & (proj_main < bins[j+1])
                if bin_m.sum() > 0:
                    if edge_type == 'bottom':
                        idx = np.argmax(proj_secondary[bin_m])
                    else:
                        idx = np.argmin(proj_secondary[bin_m])
                    pts_in_bin = pts_in_region[bin_m]
                    contour_pts.append(pts_in_bin[idx])
                    contour_main.append(proj_main[bin_m][idx])

            if len(contour_pts) < 3:
                return pts_in_region, proj_main, proj_secondary

            contour_pts = np.array(contour_pts)
            contour_main = np.array(contour_main)

            # 计算局部斜率角度
            angles = []
            for k in range(len(contour_pts)):
                k_prev = max(0, k - 1)
                k_next = min(len(contour_pts) - 1, k + 1)
                if k_prev == k_next:
                    angles.append(0)
                    continue
                dx = contour_pts[k_next][0] - contour_pts[k_prev][0]
                dy = contour_pts[k_next][1] - contour_pts[k_prev][1]
                if abs(dx) < 1e-6:
                    angles.append(90)
                else:
                    angles.append(abs(np.degrees(np.arctan(dy / dx))))
            angles = np.array(angles)

            # 从两端向中心收缩，找到角度合适的范围
            valid_start = 0
            valid_end = len(angles)

            for j in range(len(angles)):
                if angles[j] <= max_edge_angle_deviation:
                    valid_start = j
                    break

            for j in range(len(angles) - 1, -1, -1):
                if angles[j] <= max_edge_angle_deviation:
                    valid_end = j + 1
                    break

            if valid_start < valid_end and valid_end - valid_start >= 2:
                new_main_min = contour_main[valid_start]
                new_main_max = contour_main[valid_end - 1]

                # 重新过滤点
                new_mask = (proj_main >= new_main_min) & (proj_main <= new_main_max)
                pts_filtered = pts_in_region[new_mask]
                proj_main_filtered = proj_main[new_mask]
                proj_secondary_filtered = proj_secondary[new_mask]

                removed_left = valid_start
                removed_right = len(angles) - valid_end
                if removed_left > 0 or removed_right > 0:
                    print(f"[Single-Side] {edge_type} edge angle filter: removed {removed_left} left, {removed_right} right bins")

                return pts_filtered, proj_main_filtered, proj_secondary_filtered

        else:
            # 垂直边缘：按y排序，提取x极值点
            main_min, main_max = np.min(proj_main), np.max(proj_main)
            main_range = main_max - main_min + 1e-6
            num_bins = min(50, max(10, int(main_range / 5)))
            bins = np.linspace(main_min, main_max, num_bins + 1)

            contour_pts = []
            contour_main = []

            for j in range(num_bins):
                bin_m = (proj_main >= bins[j]) & (proj_main < bins[j+1])
                if bin_m.sum() > 0:
                    if edge_type == 'left':
                        idx = np.argmin(proj_secondary[bin_m])
                    else:
                        idx = np.argmax(proj_secondary[bin_m])
                    pts_in_bin = pts_in_region[bin_m]
                    contour_pts.append(pts_in_bin[idx])
                    contour_main.append(proj_main[bin_m][idx])

            if len(contour_pts) < 3:
                return pts_in_region, proj_main, proj_secondary

            contour_pts = np.array(contour_pts)
            contour_main = np.array(contour_main)

            # 计算局部斜率角度（相对于垂直线）
            angles = []
            for k in range(len(contour_pts)):
                k_prev = max(0, k - 1)
                k_next = min(len(contour_pts) - 1, k + 1)
                if k_prev == k_next:
                    angles.append(0)
                    continue
                dx = contour_pts[k_next][0] - contour_pts[k_prev][0]
                dy = contour_pts[k_next][1] - contour_pts[k_prev][1]
                if abs(dy) < 1e-6:
                    angles.append(90)
                else:
                    angles.append(abs(np.degrees(np.arctan(dx / dy))))
            angles = np.array(angles)

            valid_start = 0
            valid_end = len(angles)

            for j in range(len(angles)):
                if angles[j] <= max_edge_angle_deviation:
                    valid_start = j
                    break

            for j in range(len(angles) - 1, -1, -1):
                if angles[j] <= max_edge_angle_deviation:
                    valid_end = j + 1
                    break

            if valid_start < valid_end and valid_end - valid_start >= 2:
                new_main_min = contour_main[valid_start]
                new_main_max = contour_main[valid_end - 1]

                new_mask = (proj_main >= new_main_min) & (proj_main <= new_main_max)
                pts_filtered = pts_in_region[new_mask]
                proj_main_filtered = proj_main[new_mask]
                proj_secondary_filtered = proj_secondary[new_mask]

                removed_left = valid_start
                removed_right = len(angles) - valid_end
                if removed_left > 0 or removed_right > 0:
                    print(f"[Single-Side] {edge_type} edge angle filter: removed {removed_left} left, {removed_right} right bins")

                return pts_filtered, proj_main_filtered, proj_secondary_filtered

        return pts_in_region, proj_main, proj_secondary

    # 是否启用边缘角度过滤（只在拉伸/挤压模式下启用）
    use_edge_angle_filter = is_stretch or is_compress

    # ========================================
    # 判断是否为L形（需要连续弧）
    # ========================================
    is_L_shape = (fix_bottom or fix_top) and (fix_left or fix_right)

    if is_L_shape:
        # ========================================
        # L形连续弧：从角落出发，沿边缘扩展
        # ========================================

        # 确定角落位置
        if fix_bottom and fix_left:
            corner_x, corner_y = x_min, y_max  # 左下角
            x_direction = 1   # 向右扩展
            y_direction = -1  # 向上扩展
            modes = ["Bottom", "Left"]
        elif fix_bottom and fix_right:
            corner_x, corner_y = x_max, y_max  # 右下角
            x_direction = -1  # 向左扩展
            y_direction = -1  # 向上扩展
            modes = ["Bottom", "Right"]
        elif fix_top and fix_left:
            corner_x, corner_y = x_min, y_min  # 左上角
            x_direction = 1   # 向右扩展
            y_direction = 1   # 向下扩展
            modes = ["Top", "Left"]
        else:  # fix_top and fix_right
            corner_x, corner_y = x_max, y_min  # 右上角
            x_direction = -1  # 向左扩展
            y_direction = 1   # 向下扩展
            modes = ["Top", "Right"]

        # 计算弧长（从角落沿两边扩展的长度）
        arc_len_x = x_range * arc_ratio
        arc_len_y = y_range * arc_ratio

        # --- 水平边缘部分（底部或顶部）---
        if x_direction > 0:
            x_start, x_end = corner_x, corner_x + arc_len_x
        else:
            x_start, x_end = corner_x - arc_len_x, corner_x

        mask_h = (proj_x >= x_start) & (proj_x <= x_end)
        pts_h = all_pts[mask_h]
        proj_x_h = proj_x[mask_h]
        proj_y_h = proj_y[mask_h]

        # 应用边缘角度过滤
        edge_type_h = "bottom" if fix_bottom else "top"
        pts_h, proj_x_h, proj_y_h = filter_edge_by_angle(pts_h, proj_x_h, proj_y_h, edge_type_h, use_edge_angle_filter)

        if len(pts_h) > 0:
            num_bins = max(10, int(arc_len_x / 5))
            bins = np.linspace(np.min(proj_x_h), np.max(proj_x_h), num_bins + 1)

            for i in range(num_bins):
                bin_mask = (proj_x_h >= bins[i]) & (proj_x_h < bins[i+1])
                if bin_mask.sum() == 0:
                    continue
                pts_in_bin = pts_h[bin_mask]
                proj_y_in_bin = proj_y_h[bin_mask]
                n_select = min(num_layers, len(pts_in_bin))
                if fix_bottom:
                    sorted_idx = np.argsort(proj_y_in_bin)[-n_select:]  # y最大 = 底部
                else:
                    sorted_idx = np.argsort(proj_y_in_bin)[:n_select]   # y最小 = 顶部
                all_edge_points.append(pts_in_bin[sorted_idx])

        # --- 垂直边缘部分（左侧或右侧）---
        if y_direction > 0:
            y_start, y_end = corner_y, corner_y + arc_len_y
        else:
            y_start, y_end = corner_y - arc_len_y, corner_y

        mask_v = (proj_y >= y_start) & (proj_y <= y_end)
        pts_v = all_pts[mask_v]
        proj_x_v = proj_x[mask_v]
        proj_y_v = proj_y[mask_v]

        # 应用边缘角度过滤
        edge_type_v = "left" if fix_left else "right"
        pts_v, proj_y_v, proj_x_v = filter_edge_by_angle(pts_v, proj_y_v, proj_x_v, edge_type_v, use_edge_angle_filter)

        if len(pts_v) > 0:
            num_bins = max(10, int(arc_len_y / 5))
            bins = np.linspace(np.min(proj_y_v), np.max(proj_y_v), num_bins + 1)

            for i in range(num_bins):
                bin_mask = (proj_y_v >= bins[i]) & (proj_y_v < bins[i+1])
                if bin_mask.sum() == 0:
                    continue
                pts_in_bin = pts_v[bin_mask]
                proj_x_in_bin = proj_x_v[bin_mask]
                n_select = min(num_layers, len(pts_in_bin))
                if fix_left:
                    sorted_idx = np.argsort(proj_x_in_bin)[:n_select]   # x最小 = 左侧
                else:
                    sorted_idx = np.argsort(proj_x_in_bin)[-n_select:]  # x最大 = 右侧
                all_edge_points.append(pts_in_bin[sorted_idx])

    else:
        # ========================================
        # 单边缘：从质心延长线与边缘的交点开始，向两边延伸
        # 这个交点就是离控制点最远的位置（对侧）
        # ========================================

        # 计算对侧方向（从控制点穿过质心的延长线方向）
        if np.linalg.norm(relative_pos) < 1e-3:
            opposite_dir = np.array([0.0, 1.0])  # 默认向下
        else:
            opposite_dir = -relative_pos / np.linalg.norm(relative_pos)

        # 控制点中心相对于质心的投影（用于备用计算）
        handle_proj_x = relative_pos[0]
        handle_proj_y = relative_pos[1]

        # 厚度过滤阈值
        global_filter_ratio = arc_ratio * 0.35
        # ========================================
        # 【挤压模式】边缘角度过滤参数
        # 只在挤压模式下启用，过滤角度偏离太大的边缘点
        # ========================================
        # 最大允许的边缘角度偏离（度）
        # 底部/顶部边缘：边缘应该接近水平，超过此角度的点被过滤
        # 左侧/右侧边缘：边缘应该接近垂直，超过此角度的点被过滤
        max_edge_angle_deviation = 20 if is_compress else 35  # 度

        def compute_local_edge_slope(pts_sorted_by_axis, axis='x'):
            """
            计算边缘点的局部斜率（用于判断边缘角度）
            返回每个点的局部斜率角度（相对于水平/垂直的偏离度数）
            """
            n = len(pts_sorted_by_axis)
            if n < 3:
                return np.zeros(n)

            angles = np.zeros(n)
            for i in range(n):
                # 使用前后各一个点来计算局部斜率
                i_prev = max(0, i - 1)
                i_next = min(n - 1, i + 1)

                if i_prev == i_next:
                    angles[i] = 0
                    continue

                p_prev = pts_sorted_by_axis[i_prev]
                p_next = pts_sorted_by_axis[i_next]

                dx = p_next[0] - p_prev[0]
                dy = p_next[1] - p_prev[1]

                if axis == 'x':
                    # 水平边缘，计算与水平线的夹角
                    if abs(dx) < 1e-6:
                        angles[i] = 90  # 垂直
                    else:
                        angles[i] = abs(np.degrees(np.arctan(dy / dx)))
                else:
                    # 垂直边缘，计算与垂直线的夹角
                    if abs(dy) < 1e-6:
                        angles[i] = 90  # 水平
                    else:
                        angles[i] = abs(np.degrees(np.arctan(dx / dy)))

            return angles

        # --- 底部边缘 (y最大) ---
        if fix_bottom:
            y_max_global = np.max(proj_y)
            y_global_threshold = y_max_global - y_range * global_filter_ratio

            if is_axis_aligned:
                arc_len_x = x_range * axis_arc_ratio
                arc_center_x = 0.5 * (x_min + x_max)
            else:
                if is_compress:
                    # 非轴向挤压：相对质心可用更靠控制点的中心，但弧长仍由统一 arc_ratio 控制
                    arc_len_x = x_range * arc_ratio
                    arc_center_x = handle_proj_x
                else:
                    arc_len_x = x_range * arc_ratio
                    # 计算延长线与底部边缘(y=y_max)的交点
                    if abs(opposite_dir[1]) > 1e-6 and opposite_dir[1] > 0:
                        # 向下的方向才能与底部相交
                        t = y_max / opposite_dir[1]
                        arc_center_x = opposite_dir[0] * t
                    else:
                        # 方向不对，用简单的对侧逻辑
                        arc_center_x = -handle_proj_x  # 质心的反方向

            # 限制在边界内
            arc_center_x = np.clip(arc_center_x, x_min, x_max)

            # 从中心向两边延伸
            x_arc_start = arc_center_x - arc_len_x / 2
            x_arc_end = arc_center_x + arc_len_x / 2

            # 确保不超出边界，如果超出则平移
            if x_arc_start < x_min:
                x_arc_end += (x_min - x_arc_start)
                x_arc_start = x_min
            if x_arc_end > x_max:
                x_arc_start -= (x_arc_end - x_max)
                x_arc_end = x_max
            x_arc_start = max(x_arc_start, x_min)
            x_arc_end = min(x_arc_end, x_max)

            print(f"[Single-Side] Bottom edge: center_x={arc_center_x:.1f}, range=[{x_arc_start:.1f}, {x_arc_end:.1f}]")

            if is_axis_aligned:
                # 轴向模式下不做底部窄带阈值，直接从整列里取每个x-bin的边界极值，避免只剩中段
                combined_mask = (proj_x >= x_arc_start) & (proj_x <= x_arc_end)
            else:
                combined_mask = (proj_y >= y_global_threshold) & (proj_x >= x_arc_start) & (proj_x <= x_arc_end)
            pts_in_region = all_pts[combined_mask]
            proj_x_region = proj_x[combined_mask]
            proj_y_region = proj_y[combined_mask]

            if len(pts_in_region) > 0:
                # 挤压模式：识别所有“角度合格(<=20°)”的连续底边段，每一段都放锚点
                compress_segments_x = None
                if is_compress and len(pts_in_region) >= 3:
                    edge_span = float(max(1e-6, x_arc_end - x_arc_start))
                    num_bins_contour = min(64, max(12, int(edge_span / 4)))
                    contour_bins = np.linspace(x_arc_start, x_arc_end, num_bins_contour + 1)

                    contour_pts = []
                    contour_proj_x = []
                    for j in range(num_bins_contour):
                        if j == num_bins_contour - 1:
                            bin_m = (proj_x_region >= contour_bins[j]) & (proj_x_region <= contour_bins[j + 1])
                        else:
                            bin_m = (proj_x_region >= contour_bins[j]) & (proj_x_region < contour_bins[j + 1])
                        if bin_m.sum() == 0:
                            continue

                        idx_in_bin = np.where(bin_m)[0]
                        max_y_idx = idx_in_bin[np.argmax(proj_y_region[bin_m])]
                        contour_pts.append(pts_in_region[max_y_idx])
                        contour_proj_x.append(float(proj_x_region[max_y_idx]))

                    if len(contour_pts) >= 3:
                        contour_pts = np.asarray(contour_pts, dtype=np.float32)
                        contour_proj_x = np.asarray(contour_proj_x, dtype=np.float32)
                        edge_angles = compute_local_edge_slope(contour_pts, axis='x')
                        valid_mask = edge_angles <= max_edge_angle_deviation

                        # 连续有效段（至少2个轮廓bin）全部保留
                        min_seg_bins = 2
                        seg_indices = []
                        seg_start = None
                        for j, is_valid in enumerate(valid_mask):
                            if is_valid and seg_start is None:
                                seg_start = j
                            elif (not is_valid) and seg_start is not None:
                                seg_end = j - 1
                                if seg_end - seg_start + 1 >= min_seg_bins:
                                    seg_indices.append((seg_start, seg_end))
                                seg_start = None
                        if seg_start is not None:
                            seg_end = len(valid_mask) - 1
                            if seg_end - seg_start + 1 >= min_seg_bins:
                                seg_indices.append((seg_start, seg_end))

                        if len(seg_indices) > 0:
                            raw_segments = []
                            for seg_start, seg_end in seg_indices:
                                seg_x0 = float(contour_proj_x[seg_start])
                                seg_x1 = float(contour_proj_x[seg_end])
                                if seg_x0 <= seg_x1:
                                    raw_segments.append((seg_x0, seg_x1))
                                else:
                                    raw_segments.append((seg_x1, seg_x0))

                            raw_segments.sort(key=lambda it: it[0])
                            merged_segments = []
                            for seg_x0, seg_x1 in raw_segments:
                                if not merged_segments or seg_x0 > merged_segments[-1][1] + 1.0:
                                    merged_segments.append([seg_x0, seg_x1])
                                else:
                                    merged_segments[-1][1] = max(merged_segments[-1][1], seg_x1)

                            compress_segments_x = [(float(a), float(b)) for a, b in merged_segments if b - a >= 1e-3]
                            seg_str = ", ".join([f"[{a:.1f}, {b:.1f}]" for a, b in compress_segments_x])
                            print(f"[Single-Side] Compress valid bottom segments ({len(compress_segments_x)}): {seg_str}")
                        else:
                            print(
                                f"[Single-Side] Compress angle filter: no valid bottom segment (<= {max_edge_angle_deviation:.0f}°)"
                            )

                # 生成锚点
                if len(pts_in_region) > 0:
                    if is_compress:
                        def append_bottom_bins(local_pts, local_proj_x, local_proj_y, min_bins=4):
                            if len(local_pts) == 0:
                                return 0

                            span = float(np.max(local_proj_x) - np.min(local_proj_x))
                            if span < 1e-6:
                                keep_n_local = min(max(1, num_layers), len(local_pts))
                                local_sel = np.argsort(local_proj_y)[-keep_n_local:]
                                all_edge_points.append(local_pts[local_sel])
                                return int(keep_n_local)

                            num_bins_local = max(min_bins, int(span / 5))
                            bins_local = np.linspace(np.min(local_proj_x), np.max(local_proj_x), num_bins_local + 1)
                            added = 0
                            for bi in range(num_bins_local):
                                if bi == num_bins_local - 1:
                                    bin_mask_local = (local_proj_x >= bins_local[bi]) & (local_proj_x <= bins_local[bi + 1])
                                else:
                                    bin_mask_local = (local_proj_x >= bins_local[bi]) & (local_proj_x < bins_local[bi + 1])
                                if bin_mask_local.sum() == 0:
                                    continue
                                pts_bin = local_pts[bin_mask_local]
                                y_bin = local_proj_y[bin_mask_local]
                                n_sel = min(num_layers, len(pts_bin))
                                sel_idx = np.argsort(y_bin)[-n_sel:]
                                all_edge_points.append(pts_bin[sel_idx])
                                added += int(len(sel_idx))
                            return added

                        added_total = 0
                        if compress_segments_x is not None and len(compress_segments_x) > 0:
                            for seg_i, (seg_x0, seg_x1) in enumerate(compress_segments_x):
                                seg_mask = (proj_x_region >= seg_x0) & (proj_x_region <= seg_x1)
                                if not np.any(seg_mask):
                                    continue
                                added = append_bottom_bins(
                                    pts_in_region[seg_mask],
                                    proj_x_region[seg_mask],
                                    proj_y_region[seg_mask],
                                    min_bins=3,
                                )
                                added_total += int(added)
                                if added > 0:
                                    print(
                                        f"[Single-Side] Compress anchors on segment#{seg_i + 1}: "
                                        f"x=[{seg_x0:.1f}, {seg_x1:.1f}], added={added}"
                                    )

                        # 没有有效段时回退到底边整体采样，避免无锚点
                        if added_total == 0:
                            append_bottom_bins(pts_in_region, proj_x_region, proj_y_region, min_bins=10)

                    else:
                        num_bins = max(10, int((x_arc_end - x_arc_start) / 5))
                        bins = np.linspace(x_arc_start, x_arc_end, num_bins + 1)

                        for i in range(num_bins):
                            bin_mask = (proj_x_region >= bins[i]) & (proj_x_region < bins[i+1])
                            if bin_mask.sum() == 0:
                                continue
                            pts_in_bin = pts_in_region[bin_mask]
                            proj_y_in_bin = proj_y_region[bin_mask]
                            n_select = min(num_layers, len(pts_in_bin))
                            sorted_idx = np.argsort(proj_y_in_bin)[-n_select:]
                            all_edge_points.append(pts_in_bin[sorted_idx])

            modes.append("Bottom")

        # --- 顶部边缘 (y最小) ---
        if fix_top:
            y_min_global = np.min(proj_y)
            y_global_threshold = y_min_global + y_range * global_filter_ratio

            if is_axis_aligned:
                arc_len_x = x_range * axis_arc_ratio
                arc_center_x = 0.5 * (x_min + x_max)
            else:
                if is_compress:
                    arc_len_x = x_range * arc_ratio
                    arc_center_x = handle_proj_x
                else:
                    arc_len_x = x_range * arc_ratio
                    if abs(opposite_dir[1]) > 1e-6 and opposite_dir[1] < 0:
                        t = y_min / opposite_dir[1]
                        arc_center_x = opposite_dir[0] * t
                    else:
                        arc_center_x = -handle_proj_x

            arc_center_x = np.clip(arc_center_x, x_min, x_max)

            x_arc_start = arc_center_x - arc_len_x / 2
            x_arc_end = arc_center_x + arc_len_x / 2

            if x_arc_start < x_min:
                x_arc_end += (x_min - x_arc_start)
                x_arc_start = x_min
            if x_arc_end > x_max:
                x_arc_start -= (x_arc_end - x_max)
                x_arc_end = x_max
            x_arc_start = max(x_arc_start, x_min)
            x_arc_end = min(x_arc_end, x_max)

            print(f"[Single-Side] Top edge: center_x={arc_center_x:.1f}, range=[{x_arc_start:.1f}, {x_arc_end:.1f}]")

            if is_axis_aligned:
                combined_mask = (proj_x >= x_arc_start) & (proj_x <= x_arc_end)
            else:
                combined_mask = (proj_y <= y_global_threshold) & (proj_x >= x_arc_start) & (proj_x <= x_arc_end)
            pts_in_region = all_pts[combined_mask]
            proj_x_region = proj_x[combined_mask]
            proj_y_region = proj_y[combined_mask]

            if len(pts_in_region) > 0:
                num_bins = max(10, int((x_arc_end - x_arc_start) / 5))
                bins = np.linspace(x_arc_start, x_arc_end, num_bins + 1)

                for i in range(num_bins):
                    bin_mask = (proj_x_region >= bins[i]) & (proj_x_region < bins[i+1])
                    if bin_mask.sum() == 0:
                        continue
                    pts_in_bin = pts_in_region[bin_mask]
                    proj_y_in_bin = proj_y_region[bin_mask]
                    n_select = min(num_layers, len(pts_in_bin))
                    sorted_idx = np.argsort(proj_y_in_bin)[:n_select]
                    all_edge_points.append(pts_in_bin[sorted_idx])

            modes.append("Top")

        # --- 左侧边缘 (x最小) ---
        if fix_left:
            x_min_global = np.min(proj_x)
            x_global_threshold = x_min_global + x_range * global_filter_ratio

            if is_axis_aligned:
                arc_len_y = y_range * axis_arc_ratio
                arc_center_y = 0.5 * (y_min + y_max)
            else:
                if is_compress:
                    arc_len_y = y_range * arc_ratio
                    arc_center_y = handle_proj_y
                else:
                    arc_len_y = y_range * arc_ratio
                    if abs(opposite_dir[0]) > 1e-6 and opposite_dir[0] < 0:
                        t = x_min / opposite_dir[0]
                        arc_center_y = opposite_dir[1] * t
                    else:
                        arc_center_y = -handle_proj_y

            arc_center_y = np.clip(arc_center_y, y_min, y_max)

            y_arc_start = arc_center_y - arc_len_y / 2
            y_arc_end = arc_center_y + arc_len_y / 2

            if y_arc_start < y_min:
                y_arc_end += (y_min - y_arc_start)
                y_arc_start = y_min
            if y_arc_end > y_max:
                y_arc_start -= (y_arc_end - y_max)
                y_arc_end = y_max
            y_arc_start = max(y_arc_start, y_min)
            y_arc_end = min(y_arc_end, y_max)

            print(f"[Single-Side] Left edge: center_y={arc_center_y:.1f}, range=[{y_arc_start:.1f}, {y_arc_end:.1f}]")

            if is_axis_aligned:
                combined_mask = (proj_y >= y_arc_start) & (proj_y <= y_arc_end)
            else:
                combined_mask = (proj_x <= x_global_threshold) & (proj_y >= y_arc_start) & (proj_y <= y_arc_end)
            pts_in_region = all_pts[combined_mask]
            proj_x_region = proj_x[combined_mask]
            proj_y_region = proj_y[combined_mask]

            if len(pts_in_region) > 0:
                num_bins = max(10, int((y_arc_end - y_arc_start) / 5))
                bins = np.linspace(y_arc_start, y_arc_end, num_bins + 1)

                for i in range(num_bins):
                    bin_mask = (proj_y_region >= bins[i]) & (proj_y_region < bins[i+1])
                    if bin_mask.sum() == 0:
                        continue
                    pts_in_bin = pts_in_region[bin_mask]
                    proj_x_in_bin = proj_x_region[bin_mask]
                    n_select = min(num_layers, len(pts_in_bin))
                    sorted_idx = np.argsort(proj_x_in_bin)[:n_select]
                    all_edge_points.append(pts_in_bin[sorted_idx])

            modes.append("Left")

        # --- 右侧边缘 (x最大) ---
        if fix_right:
            x_max_global = np.max(proj_x)
            x_global_threshold = x_max_global - x_range * global_filter_ratio

            if is_axis_aligned:
                arc_len_y = y_range * axis_arc_ratio
                arc_center_y = 0.5 * (y_min + y_max)
            else:
                if is_compress:
                    arc_len_y = y_range * arc_ratio
                    arc_center_y = handle_proj_y
                else:
                    arc_len_y = y_range * arc_ratio
                    if abs(opposite_dir[0]) > 1e-6 and opposite_dir[0] > 0:
                        t = x_max / opposite_dir[0]
                        arc_center_y = opposite_dir[1] * t
                    else:
                        arc_center_y = -handle_proj_y

            arc_center_y = np.clip(arc_center_y, y_min, y_max)

            y_arc_start = arc_center_y - arc_len_y / 2
            y_arc_end = arc_center_y + arc_len_y / 2

            if y_arc_start < y_min:
                y_arc_end += (y_min - y_arc_start)
                y_arc_start = y_min
            if y_arc_end > y_max:
                y_arc_start -= (y_arc_end - y_max)
                y_arc_end = y_max
            y_arc_start = max(y_arc_start, y_min)
            y_arc_end = min(y_arc_end, y_max)

            print(f"[Single-Side] Right edge: center_y={arc_center_y:.1f}, range=[{y_arc_start:.1f}, {y_arc_end:.1f}]")

            if is_axis_aligned:
                combined_mask = (proj_y >= y_arc_start) & (proj_y <= y_arc_end)
            else:
                combined_mask = (proj_x >= x_global_threshold) & (proj_y >= y_arc_start) & (proj_y <= y_arc_end)
            pts_in_region = all_pts[combined_mask]
            proj_x_region = proj_x[combined_mask]
            proj_y_region = proj_y[combined_mask]

            if len(pts_in_region) > 0:
                num_bins = max(10, int((y_arc_end - y_arc_start) / 5))
                bins = np.linspace(y_arc_start, y_arc_end, num_bins + 1)

                for i in range(num_bins):
                    bin_mask = (proj_y_region >= bins[i]) & (proj_y_region < bins[i+1])
                    if bin_mask.sum() == 0:
                        continue
                    pts_in_bin = pts_in_region[bin_mask]
                    proj_x_in_bin = proj_x_region[bin_mask]
                    n_select = min(num_layers, len(pts_in_bin))
                    sorted_idx = np.argsort(proj_x_in_bin)[-n_select:]
                    all_edge_points.append(pts_in_bin[sorted_idx])

            modes.append("Right")

    # ========================================
    # 合并锚点
    # ========================================
    if len(all_edge_points) > 0:
        anchors = np.vstack(all_edge_points)
        anchors = np.unique(anchors, axis=0)
    else:
        # 退化：取底部10%
        y_max_val = np.max(proj_y)
        threshold_val = y_max_val - y_range * 0.1
        anchors = all_pts[proj_y >= threshold_val]
        modes = ["Fallback"]

    mode_str = "+".join(modes)
    shape_type = "L-Arc" if is_L_shape else "Edge"
    mode_label = "Axis" if is_axis_aligned else "Normal"
    desc = f"SingleSide-{mode_label} ({shape_type}, {mode_str}, arc={arc_ratio:.0%}, layers={num_layers}, n={len(anchors)}, Deg={influence_range:.2f})"

    return anchors, desc

def _nonrigid_single_side_normal(params):
    """单边模式-普通。"""
    local = dict(params)
    local["single_side_mode"] = "NORMAL"
    return _nonrigid_02_root_pinned(local)


def _nonrigid_single_side_axis(params):
    """单边模式-轴向。"""
    local = dict(params)
    local["single_side_mode"] = "AXIS"
    return _nonrigid_02_root_pinned(local)


def _nonrigid_03_squeeze(params):
    """【非刚性-03】双边均匀拉伸/压缩（支持多点）。"""
    all_pts = params["all_pts"]
    center_of_mass = params["center_of_mass"]
    influence_range = float(params["influence_range"])

    # 使用分组主轴，稳定双边均匀效果（左/右或上/下）
    axis = np.asarray(params.get("handle_axis", [1.0, 0.0]), dtype=np.float32)
    axis = axis / (np.linalg.norm(axis) + 1e-6)
    perp = np.array([-axis[1], axis[0]], dtype=np.float32)

    rel = all_pts - center_of_mass
    proj_axis = np.dot(rel, axis)
    proj_perp = np.dot(rel, perp)
    axis_range = float(np.max(proj_axis) - np.min(proj_axis) + 1e-6)
    perp_range = float(np.max(proj_perp) - np.min(proj_perp) + 1e-6)

    # 双边模式使用“中性轴长条锚点”：
    # range 小 -> 锚点条更长、点更多；range 大 -> 锚点条更短、点更少。
    strip_ratio = float(np.clip(1.0 - 0.70 * influence_range, 0.30, 1.0))
    axis_band = axis_range * (0.02 + 0.04 * (1.0 - influence_range))  # 中性轴厚度（较窄）
    perp_half = 0.5 * perp_range * strip_ratio                      # 条带长度的一半

    center_mask = (np.abs(proj_axis) <= axis_band) & (np.abs(proj_perp) <= perp_half)
    cand_pts = all_pts[center_mask]
    cand_axis = proj_axis[center_mask]
    cand_perp = proj_perp[center_mask]

    # 候选不足时，先放宽条带，再退化为“全局近中性轴”采样
    if cand_pts.shape[0] < 10:
        axis_band_relax = axis_band * 1.8
        perp_half_relax = min(0.5 * perp_range, perp_half * 1.25)
        relax_mask = (np.abs(proj_axis) <= axis_band_relax) & (np.abs(proj_perp) <= perp_half_relax)
        cand_pts = all_pts[relax_mask]
        cand_axis = proj_axis[relax_mask]
        cand_perp = proj_perp[relax_mask]
        axis_band = axis_band_relax
        perp_half = perp_half_relax

    if cand_pts.shape[0] < 10:
        near_axis_order = np.argsort(np.abs(proj_axis))
        take_n = min(len(all_pts), int(np.clip(30 + 70 * strip_ratio, 30, 100)))
        sel = near_axis_order[:take_n]
        cand_pts = all_pts[sel]
        cand_axis = proj_axis[sel]
        cand_perp = proj_perp[sel]

    # 沿中线方向分桶，保证是一条“长条”而不是中间一团
    layers = max(1, 4 - int(influence_range * 3))  # range 小更多层，range 大更少层
    bin_count = int(np.clip(round(10 + 38 * strip_ratio), 10, 48))
    pmin = float(np.min(cand_perp)) if cand_perp.size > 0 else -perp_half
    pmax = float(np.max(cand_perp)) if cand_perp.size > 0 else perp_half
    if pmax - pmin < 1e-4:
        pmin, pmax = -perp_half, perp_half
    bins = np.linspace(pmin, pmax, bin_count + 1)

    picked = []
    for bi in range(bin_count):
        if bi == bin_count - 1:
            bin_m = (cand_perp >= bins[bi]) & (cand_perp <= bins[bi + 1])
        else:
            bin_m = (cand_perp >= bins[bi]) & (cand_perp < bins[bi + 1])
        if np.sum(bin_m) == 0:
            continue
        pts_bin = cand_pts[bin_m]
        axis_bin = np.abs(cand_axis[bin_m])
        n_pick = min(layers, len(pts_bin))
        idx = np.argsort(axis_bin)[:n_pick]  # 优先贴近中性轴
        picked.append(pts_bin[idx])

    if len(picked) > 0:
        anchors = np.vstack(picked).astype(np.float32)
        anchors = np.unique(np.round(anchors).astype(np.int32), axis=0).astype(np.float32)
    else:
        anchors = np.zeros((0, 2), dtype=np.float32)

    # 上限保护（防止极端大 mask 产生过多锚点）
    max_keep = int(np.clip(round(20 + 95 * strip_ratio), 20, 120))
    if anchors.shape[0] > max_keep:
        anchor_rel = anchors - center_of_mass[None, :]
        anchor_axis = np.abs(np.dot(anchor_rel, axis))
        anchor_perp = np.abs(np.dot(anchor_rel, perp))
        # 既偏好靠近中性轴，也兼顾长条覆盖
        scores = 1.0 / (anchor_axis + 0.15 * anchor_perp + 1e-3)
        anchors = _sample_anchors_spatially_balanced(anchors, scores, max_anchors=max_keep)
        anchors = np.unique(np.round(anchors).astype(np.int32), axis=0).astype(np.float32)

    bilateral_type = str(params.get("bilateral_type", "UNIFORM")).upper()
    desc = (
        f"BilateralUniform-{bilateral_type} "
        f"(stripe={strip_ratio:.2f}, band={axis_band:.1f}, layers={layers}, n={len(anchors)}, Deg={influence_range:.2f})"
    )
    return anchors.astype(np.float32), desc


def _nonrigid_04_scaling(params):
    """【非刚性-04】均匀缩放（保形）模式。"""
    influence_range = float(params["influence_range"])
    # 均匀缩放模式不添加固定锚点，避免中心锚点抵消缩放意图。
    anchors = np.zeros((0, 2), dtype=np.float32)
    s = float(params.get("scale_factor", 1.0))
    desc = f"UniformScale (s={s:.3f}, n={len(anchors)}, Deg={influence_range:.2f})"
    return anchors.astype(np.float32), desc


def _nonrigid_05_chaos(params):
    """【非刚性-05】自由拉伸（兜底）模式。"""
    all_pts = np.asarray(params["all_pts"], dtype=np.float32).reshape(-1, 2)
    center_of_mass = np.asarray(params["center_of_mass"], dtype=np.float32).reshape(2)
    influence_range = float(params["influence_range"])
    handle_points = np.asarray(
        params.get("handle_points", np.zeros((0, 2), dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1, 2)
    if all_pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), "FreeStretch-Centroid (empty)"

    # 仅固定“质心锚点”：自由拉伸时给出稳定核心，但尽量不约束边缘
    d_center = np.linalg.norm(all_pts - center_of_mass[None, :], axis=1)
    if handle_points.shape[0] > 0:
        # 若控制点恰好压在质心附近，放宽一圈再选，避免控制点与固定点冲突
        h = np.unique(np.round(handle_points).astype(np.int32), axis=0).astype(np.float32)
        diff = all_pts[:, None, :] - h[None, :, :]
        d_handle = np.min(np.linalg.norm(diff, axis=2), axis=1)
        avoid_thr = float(np.clip(2.0 + 4.0 * influence_range, 2.0, 7.0))
        candidates = np.where(d_handle >= avoid_thr)[0]
        if candidates.size > 0:
            local = candidates[np.argmin(d_center[candidates])]
        else:
            local = int(np.argmin(d_center))
    else:
        local = int(np.argmin(d_center))

    anchors = all_pts[local: local + 1].astype(np.float32)
    desc = (
        f"FreeStretch-Centroid (n={len(anchors)}, "
        f"center=({float(center_of_mass[0]):.1f},{float(center_of_mass[1]):.1f}), "
        f"Deg={influence_range:.2f})"
    )
    return anchors, desc


def _sample_grid_bilinear_xy(grid_abs, x, y):
    """
    在绝对坐标网格上做双线性采样，返回 [src_x, src_y]。
    grid_abs: (H, W, 2)
    x, y: float（目标空间坐标）
    """
    H, W = grid_abs.shape[0], grid_abs.shape[1]
    x = float(np.clip(x, 0.0, W - 1.0))
    y = float(np.clip(y, 0.0, H - 1.0))

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, W - 1)
    y1 = min(y0 + 1, H - 1)
    dx = x - x0
    dy = y - y0

    v00 = grid_abs[y0, x0]
    v10 = grid_abs[y0, x1]
    v01 = grid_abs[y1, x0]
    v11 = grid_abs[y1, x1]

    top = (1.0 - dx) * v00 + dx * v10
    bottom = (1.0 - dx) * v01 + dx * v11
    return (1.0 - dy) * top + dy * bottom

def _enforce_subpixel_handle_constraints_uniform_neighbor(grid_abs, src_xy, tgt_xy):
    """
    对 MLS 结果做“亚像素约束”：
    1) 在控制点四邻域上施加同一delta，保证双线性采样点命中；
    2) 最近整点做硬钉住，进一步抑制量化误差。
    返回:
        corrected_grid, mean_err_before(latent), mean_err_after(latent)
    """
    if not isinstance(grid_abs, torch.Tensor):
        return grid_abs, 0.0, 0.0

    if src_xy is None or tgt_xy is None or src_xy.numel() == 0 or tgt_xy.numel() == 0:
        return grid_abs, 0.0, 0.0

    H, W = grid_abs.shape[0], grid_abs.shape[1]
    device = grid_abs.device
    dtype = grid_abs.dtype

    corrected = grid_abs.clone()
    accum = torch.zeros_like(corrected)
    count = torch.zeros((H, W, 1), device=device, dtype=dtype)
    residual_before = []

    pair_count = min(int(src_xy.shape[0]), int(tgt_xy.shape[0]))
    for i in range(pair_count):
        tx = float(tgt_xy[i, 0].item())
        ty = float(tgt_xy[i, 1].item())
        sx = float(src_xy[i, 0].item())
        sy = float(src_xy[i, 1].item())

        tx = float(np.clip(tx, 0.0, W - 1.0))
        ty = float(np.clip(ty, 0.0, H - 1.0))
        sx = float(np.clip(sx, 0.0, W - 1.0))
        sy = float(np.clip(sy, 0.0, H - 1.0))

        desired = torch.tensor([sx, sy], device=device, dtype=dtype)
        sampled = _sample_grid_bilinear_xy(corrected, tx, ty)
        delta = desired - sampled
        residual_before.append(float(torch.norm(delta, p=2).item()))

        x0 = int(math.floor(tx))
        y0 = int(math.floor(ty))
        x1 = min(x0 + 1, W - 1)
        y1 = min(y0 + 1, H - 1)
        neighbors = {(x0, y0), (x1, y0), (x0, y1), (x1, y1)}
        for nx, ny in neighbors:
            accum[ny, nx] += delta
            count[ny, nx, 0] += 1.0

    has_update = count > 0
    correction = accum / torch.clamp(count, min=1.0)
    corrected = corrected + correction * has_update.to(dtype)

    # 二次硬钉住：最近整点直接映射到源控制点
    for i in range(pair_count):
        tx = float(np.clip(float(tgt_xy[i, 0].item()), 0.0, W - 1.0))
        ty = float(np.clip(float(tgt_xy[i, 1].item()), 0.0, H - 1.0))
        sx = float(np.clip(float(src_xy[i, 0].item()), 0.0, W - 1.0))
        sy = float(np.clip(float(src_xy[i, 1].item()), 0.0, H - 1.0))
        xi = int(round(tx))
        yi = int(round(ty))
        corrected[yi, xi, 0] = sx
        corrected[yi, xi, 1] = sy

    residual_after = []
    for i in range(pair_count):
        tx = float(np.clip(float(tgt_xy[i, 0].item()), 0.0, W - 1.0))
        ty = float(np.clip(float(tgt_xy[i, 1].item()), 0.0, H - 1.0))
        sx = float(np.clip(float(src_xy[i, 0].item()), 0.0, W - 1.0))
        sy = float(np.clip(float(src_xy[i, 1].item()), 0.0, H - 1.0))
        desired = torch.tensor([sx, sy], device=device, dtype=dtype)
        sampled = _sample_grid_bilinear_xy(corrected, tx, ty)
        residual_after.append(float(torch.norm(desired - sampled, p=2).item()))

    mean_before = float(np.mean(residual_before)) if len(residual_before) > 0 else 0.0
    mean_after = float(np.mean(residual_after)) if len(residual_after) > 0 else 0.0
    return corrected, mean_before, mean_after


def _enforce_subpixel_handle_constraints_bilinear_iterative(grid_abs, src_xy, tgt_xy):
    """
    对 MLS 结果做“亚像素约束”：
    1) 按双线性权重迭代修正四邻域，避免“平均分摊 delta”带来的反向误差；
    2) 仅在目标点非常接近整点时做额外硬钉住，抑制量化误差；
    3) 若修正后控制误差反而更大，则回滚到原始 grid。
    返回:
        corrected_grid, mean_err_before(latent), mean_err_after(latent)
    """
    if not isinstance(grid_abs, torch.Tensor):
        return grid_abs, 0.0, 0.0

    if src_xy is None or tgt_xy is None or src_xy.numel() == 0 or tgt_xy.numel() == 0:
        return grid_abs, 0.0, 0.0

    H, W = grid_abs.shape[0], grid_abs.shape[1]
    device = grid_abs.device
    dtype = grid_abs.dtype

    pair_count = min(int(src_xy.shape[0]), int(tgt_xy.shape[0]))
    if pair_count <= 0:
        return grid_abs, 0.0, 0.0

    corrected = grid_abs.clone()

    def _collect_residuals(grid_ref):
        residuals = []
        for i in range(pair_count):
            tx = float(np.clip(float(tgt_xy[i, 0].item()), 0.0, W - 1.0))
            ty = float(np.clip(float(tgt_xy[i, 1].item()), 0.0, H - 1.0))
            sx = float(np.clip(float(src_xy[i, 0].item()), 0.0, W - 1.0))
            sy = float(np.clip(float(src_xy[i, 1].item()), 0.0, H - 1.0))
            desired = torch.tensor([sx, sy], device=device, dtype=dtype)
            sampled = _sample_grid_bilinear_xy(grid_ref, tx, ty)
            residuals.append(float(torch.norm(desired - sampled, p=2).item()))
        return residuals

    residual_before = _collect_residuals(corrected)
    num_iters = 3

    for _ in range(num_iters):
        accum = torch.zeros_like(corrected)
        count = torch.zeros((H, W, 1), device=device, dtype=dtype)

        for i in range(pair_count):
            tx = float(np.clip(float(tgt_xy[i, 0].item()), 0.0, W - 1.0))
            ty = float(np.clip(float(tgt_xy[i, 1].item()), 0.0, H - 1.0))
            sx = float(np.clip(float(src_xy[i, 0].item()), 0.0, W - 1.0))
            sy = float(np.clip(float(src_xy[i, 1].item()), 0.0, H - 1.0))

            desired = torch.tensor([sx, sy], device=device, dtype=dtype)
            sampled = _sample_grid_bilinear_xy(corrected, tx, ty)
            delta = desired - sampled

            x0 = int(math.floor(tx))
            y0 = int(math.floor(ty))
            x1 = min(x0 + 1, W - 1)
            y1 = min(y0 + 1, H - 1)
            dx = tx - float(x0)
            dy = ty - float(y0)
            bilinear_neighbors = [
                (x0, y0, (1.0 - dx) * (1.0 - dy)),
                (x1, y0, dx * (1.0 - dy)),
                (x0, y1, (1.0 - dx) * dy),
                (x1, y1, dx * dy),
            ]
            sum_sq = float(sum((w * w) for _, _, w in bilinear_neighbors))
            if sum_sq <= 1e-8:
                continue

            for nx, ny, w in bilinear_neighbors:
                if w <= 1e-8:
                    continue
                w_t = torch.tensor(float(w), device=device, dtype=dtype)
                proposal = (w_t / float(sum_sq)) * delta
                accum[ny, nx] += w_t * proposal
                count[ny, nx, 0] += w_t

        has_update = count > 0
        correction = accum / torch.clamp(count, min=1e-6)
        corrected = corrected + correction * has_update.to(dtype)

    for i in range(pair_count):
        tx = float(np.clip(float(tgt_xy[i, 0].item()), 0.0, W - 1.0))
        ty = float(np.clip(float(tgt_xy[i, 1].item()), 0.0, H - 1.0))
        sx = float(np.clip(float(src_xy[i, 0].item()), 0.0, W - 1.0))
        sy = float(np.clip(float(src_xy[i, 1].item()), 0.0, H - 1.0))
        frac_dx = abs(tx - round(tx))
        frac_dy = abs(ty - round(ty))
        if max(frac_dx, frac_dy) > 0.12:
            continue
        xi = int(round(tx))
        yi = int(round(ty))
        corrected[yi, xi, 0] = sx
        corrected[yi, xi, 1] = sy

    residual_after = _collect_residuals(corrected)
    mean_before = float(np.mean(residual_before)) if len(residual_before) > 0 else 0.0
    mean_after = float(np.mean(residual_after)) if len(residual_after) > 0 else 0.0
    if mean_after > mean_before + 1e-4:
        return grid_abs, mean_before, mean_before
    return corrected, mean_before, mean_after


def _enforce_subpixel_handle_constraints(grid_abs, src_xy, tgt_xy):
    mode = _get_subpixel_constraint_mode()
    if mode == "bilinear_iterative":
        return _enforce_subpixel_handle_constraints_bilinear_iterative(grid_abs, src_xy, tgt_xy)
    return _enforce_subpixel_handle_constraints_uniform_neighbor(grid_abs, src_xy, tgt_xy)


def _build_single_side_coherent_targets(src_xy, tgt_xy, intent_type):
    """
    单侧拉伸专用：在保持整体轮廓平滑的前提下，抑制不合理过拉。
    核心规则：
    - 侧边短位移点：保持原目标，避免被“带着过拉”
    - 中间短位移点：当左右两侧都更长且同向时，可允许被抬升（必要过拉）
    - 垂直主轴分量做轻平滑，减少局部鼓包/凹陷
    """
    if not isinstance(src_xy, torch.Tensor) or not isinstance(tgt_xy, torch.Tensor):
        return tgt_xy, {"enabled": False, "reason": "invalid_tensor"}
    if src_xy.numel() == 0 or tgt_xy.numel() == 0:
        return tgt_xy, {"enabled": False, "reason": "empty_points"}

    pair_count = min(int(src_xy.shape[0]), int(tgt_xy.shape[0]))
    if pair_count <= 0:
        return tgt_xy, {"enabled": False, "reason": "no_pairs"}

    src = src_xy[:pair_count]
    tgt = tgt_xy[:pair_count]
    disp = tgt - src
    mean_disp = torch.mean(disp, dim=0)
    mean_norm = float(torch.norm(mean_disp, p=2).item())
    if mean_norm < 1e-6:
        return tgt_xy, {"enabled": False, "reason": "tiny_motion"}
    if pair_count < 3:
        return tgt_xy, {"enabled": False, "reason": "few_points_keep_original"}

    if str(intent_type).upper() == "SINGLE_SIDE_AXIS":
        axis = torch.tensor([0.0, 1.0], device=src.device, dtype=src.dtype)
    else:
        axis = mean_disp / (torch.norm(mean_disp, p=2) + 1e-6)

    perp = torch.stack([-axis[1], axis[0]])
    cross_pos = torch.sum(src * perp.view(1, 2), dim=1)
    order = torch.argsort(cross_pos)

    disp_axis = torch.sum(disp * axis.view(1, 2), dim=1)
    disp_perp = torch.sum(disp * perp.view(1, 2), dim=1)

    d_axis_sorted = disp_axis[order].clone()
    d_perp_sorted = disp_perp[order].clone()
    # ========= 1D 序列正则化：先把位移剖面变平滑 =========
    d_axis_sm = d_axis_sorted.clone()
    d_perp_sm = d_perp_sorted.clone()
    for _ in range(3):
        sm_axis = d_axis_sm.clone()
        sm_perp = d_perp_sm.clone()
        if pair_count >= 2:
            sm_axis[0] = 0.75 * d_axis_sm[0] + 0.25 * d_axis_sm[1]
            sm_axis[-1] = 0.75 * d_axis_sm[-1] + 0.25 * d_axis_sm[-2]
            sm_perp[0] = 0.80 * d_perp_sm[0] + 0.20 * d_perp_sm[1]
            sm_perp[-1] = 0.80 * d_perp_sm[-1] + 0.20 * d_perp_sm[-2]
        if pair_count > 2:
            sm_axis[1:-1] = 0.25 * d_axis_sm[:-2] + 0.50 * d_axis_sm[1:-1] + 0.25 * d_axis_sm[2:]
            sm_perp[1:-1] = 0.25 * d_perp_sm[:-2] + 0.50 * d_perp_sm[1:-1] + 0.25 * d_perp_sm[2:]
        d_axis_sm = sm_axis
        d_perp_sm = sm_perp

    # ========= 短位移保护：短点更贴近原目标，避免被带着过拉 =========
    abs_axis = torch.abs(d_axis_sorted)
    abs_sorted, _ = torch.sort(abs_axis)
    q_idx = int(np.clip(round(0.35 * max(0, pair_count - 1)), 0, pair_count - 1))
    short_thr = abs_sorted[q_idx]
    short_mask = abs_axis <= (short_thr + 1e-6)

    keep_short_axis = torch.tensor(0.88, device=src.device, dtype=src.dtype)
    keep_long_axis = torch.tensor(0.46, device=src.device, dtype=src.dtype)
    keep_short_perp = torch.tensor(0.76, device=src.device, dtype=src.dtype)
    keep_long_perp = torch.tensor(0.54, device=src.device, dtype=src.dtype)

    keep_axis = torch.where(short_mask, keep_short_axis, keep_long_axis)
    keep_perp = torch.where(short_mask, keep_short_perp, keep_long_perp)

    d_axis_new_sorted = keep_axis * d_axis_sorted + (1.0 - keep_axis) * d_axis_sm
    d_perp_new_sorted = keep_perp * d_perp_sorted + (1.0 - keep_perp) * d_perp_sm

    # ========= 仅对“中间 valley”做过拉提升（允许必要过拉） =========
    center_boost_count = 0
    shape_smooth_count = 0
    for i in range(1, pair_count - 1):
        left = d_axis_sorted[i - 1]
        cur = d_axis_sorted[i]
        right = d_axis_sorted[i + 1]

        left_s = float(torch.sign(left).item())
        right_s = float(torch.sign(right).item())
        cur_s = float(torch.sign(cur).item())
        same_dir_lr = (left_s != 0.0) and (right_s != 0.0) and (left_s == right_s)
        same_dir_all = same_dir_lr and (cur_s == 0.0 or cur_s == left_s)

        left_abs = float(torch.abs(left).item())
        cur_abs = float(torch.abs(cur).item())
        right_abs = float(torch.abs(right).item())
        ref_abs = 0.5 * (left_abs + right_abs)
        valley_margin = max(0.22, 0.15 * max(left_abs, right_abs))
        center_short = cur_abs + valley_margin < ref_abs

        if same_dir_all and center_short:
            ref_signed = 0.5 * (d_axis_new_sorted[i - 1] + d_axis_new_sorted[i + 1])
            boosted = d_axis_new_sorted[i] + 0.92 * (ref_signed - d_axis_new_sorted[i])
            cap_abs = max(left_abs, right_abs) * 1.12 + 0.16
            if float(boosted.item()) >= 0.0:
                boosted = torch.clamp(
                    boosted,
                    min=d_axis_new_sorted[i],
                    max=torch.tensor(cap_abs, device=src.device, dtype=src.dtype),
                )
            else:
                boosted = torch.clamp(
                    boosted,
                    min=torch.tensor(-cap_abs, device=src.device, dtype=src.dtype),
                    max=d_axis_new_sorted[i],
                )
            d_axis_new_sorted[i] = boosted
            center_boost_count += 1
        else:
            shape_smooth_count += 1

    max_axis_mag = torch.max(torch.abs(d_axis_sorted))
    axis_limit = torch.maximum(max_axis_mag * 1.30, torch.tensor(0.6, device=src.device, dtype=src.dtype))
    d_axis_new_sorted = torch.clamp(d_axis_new_sorted, min=-axis_limit, max=axis_limit)

    d_axis_new = torch.zeros_like(disp_axis)
    d_perp_new = torch.zeros_like(disp_perp)
    d_axis_new[order] = d_axis_new_sorted
    d_perp_new[order] = d_perp_new_sorted

    coherent_disp = d_axis_new.view(-1, 1) * axis.view(1, 2) + d_perp_new.view(-1, 1) * perp.view(1, 2)
    coherent_tgt = src + coherent_disp

    out = tgt_xy.clone()
    out[:pair_count] = coherent_tgt

    residual = float(torch.mean(torch.norm(coherent_tgt - tgt, dim=1)).item())
    info = {
        "enabled": True,
        "pair_count": int(pair_count),
        "axis_x": float(axis[0].item()),
        "axis_y": float(axis[1].item()),
        "coherent_policy": "center_valley_adaptive",
        "center_boost_count": int(center_boost_count),
        "shape_smooth_count": int(shape_smooth_count),
        "avg_target_adjustment": float(residual),
    }
    return out, info

def _augment_single_side_curve_constraints(src_xy, tgt_xy, intent_type, max_extra=12):
    """
    单侧模式附加曲线约束：
    - 在相邻控制点间自动生成中间约束点
    - 位移使用 Catmull-Rom 平滑插值，抑制“中间塌陷”
    """
    if not isinstance(src_xy, torch.Tensor) or not isinstance(tgt_xy, torch.Tensor):
        return (
            torch.zeros((0, 2), device=src_xy.device if isinstance(src_xy, torch.Tensor) else "cpu", dtype=torch.float32),
            torch.zeros((0, 2), device=src_xy.device if isinstance(src_xy, torch.Tensor) else "cpu", dtype=torch.float32),
            {"enabled": False, "reason": "invalid_tensor"},
        )

    if src_xy.numel() == 0 or tgt_xy.numel() == 0:
        z = torch.zeros((0, 2), device=src_xy.device, dtype=src_xy.dtype)
        return z, z.clone(), {"enabled": False, "reason": "empty_points"}

    pair_count = min(int(src_xy.shape[0]), int(tgt_xy.shape[0]))
    if pair_count < 2:
        z = torch.zeros((0, 2), device=src_xy.device, dtype=src_xy.dtype)
        return z, z.clone(), {"enabled": False, "reason": "few_points"}

    src = src_xy[:pair_count]
    tgt = tgt_xy[:pair_count]
    disp = tgt - src
    mean_disp = torch.mean(disp, dim=0)
    mean_norm = float(torch.norm(mean_disp, p=2).item())
    if mean_norm < 1e-6:
        z = torch.zeros((0, 2), device=src_xy.device, dtype=src_xy.dtype)
        return z, z.clone(), {"enabled": False, "reason": "tiny_motion"}

    if str(intent_type).upper() == "SINGLE_SIDE_AXIS":
        axis = torch.tensor([0.0, 1.0], device=src.device, dtype=src.dtype)
    else:
        axis = mean_disp / (torch.norm(mean_disp, p=2) + 1e-6)
    perp = torch.stack([-axis[1], axis[0]])

    cross_pos = torch.sum(src * perp.view(1, 2), dim=1)
    order = torch.argsort(cross_pos)
    src_o = src[order]
    disp_o = disp[order]

    # Catmull-Rom 样条插值（向量版）
    def _catmull_rom(p0, p1, p2, p3, t):
        t2 = t * t
        t3 = t2 * t
        return 0.5 * (
            2.0 * p1
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )

    extra_src_list = []
    extra_tgt_list = []
    total_segments = max(1, pair_count - 1)

    for i in range(total_segments):
        s0 = src_o[i]
        s1 = src_o[i + 1]
        seg_len = float(torch.norm(s1 - s0, p=2).item())
        if seg_len < 0.8:
            continue

        d0 = disp_o[max(i - 1, 0)]
        d1 = disp_o[i]
        d2 = disp_o[i + 1]
        d3 = disp_o[min(i + 2, pair_count - 1)]

        n_sub = 2 if seg_len >= 4.0 else 1
        for k in range(1, n_sub + 1):
            t = float(k / (n_sub + 1))
            tt = torch.tensor(t, device=src.device, dtype=src.dtype)
            src_p = (1.0 - tt) * s0 + tt * s1

            disp_cr = _catmull_rom(d0, d1, d2, d3, tt)
            disp_lin = (1.0 - tt) * d1 + tt * d2
            disp_p = 0.70 * disp_cr + 0.30 * disp_lin

            # 防止插值过冲导致新鼓包：按局部邻域幅度限幅
            ax1 = torch.sum(d1 * axis)
            ax2 = torch.sum(d2 * axis)
            pr1 = torch.sum(d1 * perp)
            pr2 = torch.sum(d2 * perp)
            ax_cap = torch.maximum(torch.abs(ax1), torch.abs(ax2)) * 1.12 + 0.12
            pr_cap = torch.maximum(torch.abs(pr1), torch.abs(pr2)) * 1.20 + 0.10

            ax_v = torch.sum(disp_p * axis)
            pr_v = torch.sum(disp_p * perp)
            ax_v = torch.clamp(ax_v, min=-ax_cap, max=ax_cap)
            pr_v = torch.clamp(pr_v, min=-pr_cap, max=pr_cap)
            disp_p = ax_v * axis + pr_v * perp

            tgt_p = src_p + disp_p
            extra_src_list.append(src_p)
            extra_tgt_list.append(tgt_p)

    if len(extra_src_list) == 0:
        z = torch.zeros((0, 2), device=src_xy.device, dtype=src_xy.dtype)
        return z, z.clone(), {"enabled": False, "reason": "no_extra_points"}

    extra_src = torch.stack(extra_src_list, dim=0)
    extra_tgt = torch.stack(extra_tgt_list, dim=0)

    if int(extra_src.shape[0]) > int(max_extra):
        keep_idx = torch.linspace(
            0,
            int(extra_src.shape[0]) - 1,
            int(max_extra),
            device=extra_src.device,
        ).round().long()
        keep_idx = torch.unique(keep_idx)
        extra_src = extra_src[keep_idx]
        extra_tgt = extra_tgt[keep_idx]

    info = {
        "enabled": True,
        "pair_count": int(pair_count),
        "extra_pair_count": int(extra_src.shape[0]),
        "policy": "catmull_rom_curve_constraints",
    }
    return extra_src, extra_tgt, info


def _smooth_grid_displacement_field(grid_abs, strength=0.32, iters=2):
    """
    对绝对采样网格做位移场平滑（轻量拉普拉斯），用于抑制局部折角。
    """
    if not isinstance(grid_abs, torch.Tensor):
        return grid_abs, {"enabled": False, "reason": "invalid_grid"}

    H, W = int(grid_abs.shape[0]), int(grid_abs.shape[1])
    if H < 3 or W < 3:
        return grid_abs, {"enabled": False, "reason": "small_grid"}

    strength = float(np.clip(strength, 0.0, 0.85))
    iters = int(max(0, iters))
    if iters <= 0 or strength <= 1e-6:
        return grid_abs, {"enabled": False, "reason": "zero_strength"}

    yy, xx = torch.meshgrid(
        torch.arange(H, device=grid_abs.device, dtype=grid_abs.dtype),
        torch.arange(W, device=grid_abs.device, dtype=grid_abs.dtype),
        indexing='ij'
    )
    identity = torch.stack([xx, yy], dim=-1)
    disp = grid_abs - identity

    for _ in range(iters):
        sm = disp.clone()
        sm[1:-1, 1:-1] = (
            0.40 * disp[1:-1, 1:-1]
            + 0.15 * disp[:-2, 1:-1]
            + 0.15 * disp[2:, 1:-1]
            + 0.15 * disp[1:-1, :-2]
            + 0.15 * disp[1:-1, 2:]
        )
        disp = (1.0 - strength) * disp + strength * sm

    out = identity + disp
    info = {
        "enabled": True,
        "strength": float(strength),
        "iters": int(iters),
    }
    return out, info


def solve_mls_affine_grid(src_pts, tgt_pts, H, W, device, weight_power=4.0):
    """
    【非刚性变形】Moving Least Squares (MLS) 仿射网格求解器
    
    参数：
        src_pts: (N, 2) 变形前的控制点位置（当前状态）
        tgt_pts: (N, 2) 变形后的目标位置（原始状态，用于反向采样）
        H, W: 输出网格的高度和宽度
        device: torch 设备
        
    返回：
        grid_abs: (H, W, 2) 绝对坐标网格
    """
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    p = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1).float()
    N, M = src_pts.shape[0], p.shape[0]
    
    # 权重计算：次幂越高越局部，越低越平滑
    weight_power = float(np.clip(weight_power, 1.6, 6.0))
    distances = torch.norm(p.unsqueeze(1) - src_pts.unsqueeze(0), dim=2)
    weights = 1.0 / (distances ** weight_power + 1e-8)
    sum_w = weights.sum(dim=1, keepdim=True)
    
    # 加权质心
    p_star = (weights.unsqueeze(2) * src_pts.unsqueeze(0)).sum(dim=1) / sum_w
    q_star = (weights.unsqueeze(2) * tgt_pts.unsqueeze(0)).sum(dim=1) / sum_w
    
    # 去中心化
    p_hat = src_pts.unsqueeze(0) - p_star.unsqueeze(1)
    q_hat = tgt_pts.unsqueeze(0) - q_star.unsqueeze(1)
    
    # 仿射矩阵求解
    weights_expanded = weights.unsqueeze(2).unsqueeze(3)
    p_hat_T = p_hat.unsqueeze(2)
    q_hat_reshaped = q_hat.unsqueeze(3)
    
    A_numerator = (weights_expanded * torch.matmul(q_hat_reshaped, p_hat_T)).sum(dim=1)
    A_denominator = (weights_expanded * torch.matmul(p_hat_T.transpose(2, 3), p_hat_T)).sum(dim=1)
    A_denominator_reg = A_denominator + torch.eye(2, device=device).unsqueeze(0) * 1e-5
    
    try:
        A = torch.linalg.solve(A_denominator_reg.transpose(1, 2), A_numerator.transpose(1, 2)).transpose(1, 2)
    except Exception as e:
        print(f"[MLS] linalg.solve failed: {e}, using identity matrix")
        A = torch.eye(2, device=device).unsqueeze(0).expand(M, 2, 2)
    
    # 应用变换
    p_centered = p - p_star
    transformed = torch.matmul(A, p_centered.unsqueeze(2)).squeeze(2)
    return (transformed + q_star).reshape(H, W, 2)

def process_nonrigid_deformation(
    comp_sam_pts, comp_targets_xy, mask_for_calc,
    src_xy, tgt_xy, H_lat, W_lat, device,
    latents, comp_mask_start, comp_m_start_full,
    scale_x, scale_y, rigid_ratio
):
    """
    【非刚性变形】完整流程封装
    """
    debug_info = {'m_start_full': comp_m_start_full}
    
    # 1. Mask 转换
    m_u8 = (mask_for_calc * 255).astype(np.uint8) if mask_for_calc.dtype != np.uint8 else mask_for_calc
    
    # 2. 意图识别
    intent_type, intent_params = detect_nonrigid_intent(
        comp_sam_pts, comp_targets_xy, m_u8, influence_range=rigid_ratio
    )
    debug_info['intent_type'] = intent_type
    debug_info['sub_action'] = intent_type
    
    # 3. 锚点生成（内联调度）
    if intent_type == "EMPTY":
        anchors_img = np.zeros((0, 2), dtype=np.float32)
        method_desc = "Empty Mask"
    else:
        anchor_funcs = {
            # 新版语义
            "SINGLE_SIDE_NORMAL": _nonrigid_single_side_normal,
            "SINGLE_SIDE_AXIS": _nonrigid_single_side_axis,
            "BILATERAL_UNIFORM": _nonrigid_03_squeeze,
            "UNIFORM_SCALE": _nonrigid_04_scaling,
            "FREE_STRETCH": _nonrigid_05_chaos,
            # 兼容旧语义（避免历史日志/缓存触发 KeyError）
            "CENTER_PINNED": _nonrigid_01_center_pinned,
            "ROOT_PINNED": _nonrigid_02_root_pinned,
            "SQUEEZE": _nonrigid_03_squeeze,
            "SCALING": _nonrigid_04_scaling,
            "CHAOS": _nonrigid_05_chaos,
        }
        anchor_fn = anchor_funcs.get(intent_type, _nonrigid_05_chaos)
        anchors_img, method_desc = anchor_fn(intent_params)

    # ✅ 确保 anchors_img 是正确的 (N, 2) 形状
    if anchors_img is None:
        anchors_img = np.zeros((0, 2), dtype=np.float32)
    elif not isinstance(anchors_img, np.ndarray):
        anchors_img = np.array(anchors_img, dtype=np.float32)

    # 确保是二维数组
    if anchors_img.ndim == 1:
        if len(anchors_img) == 0:
            anchors_img = np.zeros((0, 2), dtype=np.float32)
        else:
            anchors_img = anchors_img.reshape(-1, 2)

    # UNIFORM_SCALE 专用直通路径：
    # 直接使用“等比例缩放 + 旋转 + 平移”相似变换，
    # 避免 MLS 在该意图下引入不必要的局部形变。
    if intent_type == "UNIFORM_SCALE":
        src_np = src_xy.detach().cpu().numpy().astype(np.float32)
        tgt_np = tgt_xy.detach().cpu().numpy().astype(np.float32)
        s_lat, R_lat, t_lat, fit_err_lat = _fit_similarity_transform_np(src_np, tgt_np)
        if s_lat is None:
            # 理论上 UNIFORM_SCALE 至少有 2 点；这里做鲁棒回退。
            s_lat = float(np.clip(intent_params.get("scale_factor", 1.0), 0.30, 3.00))
            R_lat = np.eye(2, dtype=np.float32)
            if src_np.shape[0] > 0 and tgt_np.shape[0] > 0:
                n = min(src_np.shape[0], tgt_np.shape[0])
                t_lat = np.mean(tgt_np[:n] - s_lat * src_np[:n], axis=0).astype(np.float32)
            else:
                t_lat = np.zeros((2,), dtype=np.float32)
            if src_np.shape[0] > 0 and tgt_np.shape[0] > 0:
                n = min(src_np.shape[0], tgt_np.shape[0])
                fit_err_lat = float(np.mean(np.linalg.norm(src_np[:n] - tgt_np[:n], axis=1)))
            else:
                fit_err_lat = 0.0

        grid_abs = _build_similarity_inverse_grid(
            scale_factor=float(s_lat),
            rot_mat=R_lat,
            trans_vec=t_lat,
            H=H_lat,
            W=W_lat,
            device=device,
        )

        # 关键：UNIFORM_SCALE 不再做本地硬钉点修正。
        # 该修正会把全局相似变换拉成局部折叠，导致多连通域/碎片。
        residuals = []
        pair_count = min(int(src_xy.shape[0]), int(tgt_xy.shape[0]))
        for i in range(pair_count):
            tx = float(np.clip(float(tgt_xy[i, 0].item()), 0.0, W_lat - 1.0))
            ty = float(np.clip(float(tgt_xy[i, 1].item()), 0.0, H_lat - 1.0))
            sx = float(np.clip(float(src_xy[i, 0].item()), 0.0, W_lat - 1.0))
            sy = float(np.clip(float(src_xy[i, 1].item()), 0.0, H_lat - 1.0))
            sampled = _sample_grid_bilinear_xy(grid_abs, tx, ty)
            if torch.is_tensor(sampled):
                sampled_np = sampled.detach().cpu().numpy().astype(np.float32)
            else:
                sampled_np = np.asarray(sampled, dtype=np.float32)
            residuals.append(float(np.linalg.norm(np.array([sx, sy], dtype=np.float32) - sampled_np)))
        ctrl_err_before_lat = float(np.mean(residuals)) if len(residuals) > 0 else 0.0
        ctrl_err_after_lat = ctrl_err_before_lat

        norm_grid = torch.zeros_like(grid_abs)
        norm_grid[..., 0] = 2.0 * grid_abs[..., 0] / (W_lat - 1) - 1.0
        norm_grid[..., 1] = 2.0 * grid_abs[..., 1] / (H_lat - 1) - 1.0

        warped_latents = _grid_sample_latents_aa(
            latents,
            norm_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )
        comp_mask_end = F.grid_sample(
            comp_mask_start.view(1, 1, H_lat, W_lat),
            norm_grid.unsqueeze(0),
            mode='nearest', align_corners=True
        ).squeeze()

        anchors_img = np.zeros((0, 2), dtype=np.float32)
        debug_info['anchors_img'] = anchors_img
        debug_info['method_desc'] = (
            f"UniformScale-SimilarityDirect (s={float(s_lat):.3f}, fit={float(fit_err_lat):.3f})"
        )
        debug_info['uniform_scale_direct'] = {
            "enabled": True,
            "scale_factor_lat": float(s_lat),
            "rotation_lat": np.asarray(R_lat, dtype=np.float32),
            "translation_lat": np.asarray(t_lat, dtype=np.float32),
            "fit_error_lat": float(fit_err_lat),
        }
        debug_info['norm_grid'] = norm_grid
        debug_info['pivot_img'] = None
        debug_info['warped_latents'] = warped_latents.clone()
        debug_info['mask_end'] = comp_mask_end.clone()
        debug_info['ctrl_err_before_lat'] = float(ctrl_err_before_lat)
        debug_info['ctrl_err_after_lat'] = float(ctrl_err_after_lat)
        debug_info['ctrl_err_after_smooth_lat'] = float(ctrl_err_after_lat)
        debug_info['single_side_smooth'] = {"enabled": False, "reason": "uniform_scale_direct"}
        debug_info['mls_weight_power'] = 0.0
        scale_safe = max(float(scale_x), 1e-6)
        debug_info['ctrl_err_before_img'] = float(ctrl_err_before_lat / scale_safe)
        debug_info['ctrl_err_after_img'] = float(ctrl_err_after_lat / scale_safe)

        print(
            f"  [Non-Rigid] Sub-action: UNIFORM_SCALE, SimilarityDirect "
            f"(s={float(s_lat):.3f}, fit={float(fit_err_lat):.3f}), Anchors: 0, "
            f"CtrlErr(lat): {float(ctrl_err_before_lat):.3f}->{float(ctrl_err_after_lat):.3f}"
        )
        return warped_latents, comp_mask_end, "Non-Rigid", None, anchors_img, debug_info

    # FREE_STRETCH 细分 profile：
    # - HARD_CONTROL_GLOBAL: 控制点全局分布，优先“全部到位+平滑”，禁用会冲突的附加锚/guard
    # - CENTROID_GUARD: 其他自由拉伸场景沿用质心+内部guard策略
    free_stretch_profile = "NA"
    if intent_type == "FREE_STRETCH":
        ctrl_n = int(min(len(comp_sam_pts), len(comp_targets_xy)))
        coverage_deg = float(intent_params.get("coverage_deg", 0.0))
        opposite_covered = bool(intent_params.get("opposite_covered", False))
        same_side_soft = float(intent_params.get("same_side_ratio_soft", intent_params.get("same_side_ratio", 0.0)))
        consistency = float(intent_params.get("consistency", 0.0))

        # 典型“可解结构”：点在轮廓上有全局覆盖且不是明显单侧，这时用 hard-control 更稳定
        is_global_layout = (
            ctrl_n >= 4
            and coverage_deg >= 135.0
            and opposite_covered
            and same_side_soft <= 0.88
            and consistency >= 0.20
        )
        if is_global_layout:
            free_stretch_profile = "HARD_CONTROL_GLOBAL"
            anchors_img = np.zeros((0, 2), dtype=np.float32)
            method_desc = f"{method_desc}|HardCtrl"
        else:
            free_stretch_profile = "CENTROID_GUARD"

    debug_info["free_stretch_profile"] = free_stretch_profile

    # 3.1 全模式补充“外壳静止锚点”：远离控制点的外边界尽量保持不动，避免少点时整体扭曲。
    # 但单边/双边结构模式应优先遵守其专用锚点语义，避免把远端（如鞋尖）误锁死。
    shell_anchor_skip_modes = {
        "SINGLE_SIDE_NORMAL",
        "SINGLE_SIDE_AXIS",
        "BILATERAL_UNIFORM",
        "UNIFORM_SCALE",
        "FREE_STRETCH",
    }
    if intent_type in shell_anchor_skip_modes:
        shell_keep_anchors = np.zeros((0, 2), dtype=np.float32)
        shell_anchor_info = {"reason": f"disabled_for_{intent_type}"}
    else:
        shell_anchor_min_keep = 40 if len(comp_sam_pts) <= 2 else 20
        shell_anchor_max_keep = 220 if len(comp_sam_pts) <= 2 else 140
        shell_keep_anchors, shell_anchor_info = _build_outer_shell_anchors(
            mask_u8=m_u8,
            handle_points=comp_sam_pts,
            influence_range=rigid_ratio,
            min_keep=shell_anchor_min_keep,
            max_keep=shell_anchor_max_keep,
        )
    anchors_img = _merge_anchor_sets(anchors_img, shell_keep_anchors)

    debug_info['anchors_img'] = anchors_img
    debug_info['method_desc'] = method_desc
    debug_info['outer_shell_anchors'] = shell_anchor_info
    debug_info['outer_shell_anchor_count'] = int(shell_keep_anchors.shape[0]) if isinstance(shell_keep_anchors, np.ndarray) else 0

    # 4. 锚点坐标转换到 Latent 空间
    if len(anchors_img) > 0:
        anchors_lat = torch.from_numpy(anchors_img.copy()).float().to(device)
        anchors_lat[:, 0] *= scale_x
        anchors_lat[:, 1] *= scale_y
    else:
        anchors_lat = torch.zeros((0, 2), device=device, dtype=torch.float32)

    single_side_modes = {"SINGLE_SIDE_NORMAL", "SINGLE_SIDE_AXIS"}
    control_src_xy = src_xy
    control_tgt_xy = tgt_xy
    if intent_type in single_side_modes:
        coherent_tgt_xy, single_side_info = _build_single_side_coherent_targets(
            src_xy=src_xy,
            tgt_xy=tgt_xy,
            intent_type=intent_type,
        )
        control_tgt_xy = coherent_tgt_xy
        curve_target_xy = control_tgt_xy
        curve_src_xy, curve_tgt_xy, curve_info = _augment_single_side_curve_constraints(
            src_xy=control_src_xy,
            tgt_xy=curve_target_xy,
            intent_type=intent_type,
            max_extra=12,
        )
    else:
        single_side_info = {"enabled": False, "reason": "not_single_side"}
        curve_src_xy = torch.zeros((0, 2), device=src_xy.device, dtype=src_xy.dtype)
        curve_tgt_xy = torch.zeros((0, 2), device=src_xy.device, dtype=src_xy.dtype)
        curve_info = {"enabled": False, "reason": "not_single_side"}
    debug_info["single_side_coherent_targets"] = single_side_info
    debug_info["single_side_curve_constraints"] = curve_info
    debug_info["single_side_target_mode"] = "coherent_target_main"

    # 5. 外壳厚度守恒约束（外侧壳层随最近控制点平移）
    # UNIFORM_SCALE 使用专用“相似变换壳层配对”，避免与最近点平移规则冲突。
    if intent_type == "UNIFORM_SCALE":
        shell_src_img = np.zeros((0, 2), dtype=np.float32)
        shell_tgt_img = np.zeros((0, 2), dtype=np.float32)
        shell_info = {"reason": "disabled_for_uniform_scale"}
    elif intent_type == "FREE_STRETCH":
        shell_src_img = np.zeros((0, 2), dtype=np.float32)
        shell_tgt_img = np.zeros((0, 2), dtype=np.float32)
        shell_info = {"reason": f"disabled_for_{intent_type}"}
    else:
        shell_src_img, shell_tgt_img, shell_info = _build_outer_shell_transport_pairs(
            mask_u8=m_u8,
            handle_points=comp_sam_pts,
            target_points=comp_targets_xy,
            center_of_mass=intent_params.get("center_of_mass", np.mean(comp_sam_pts, axis=0) if len(comp_sam_pts) > 0 else np.array([0.0, 0.0])),
            influence_range=rigid_ratio,
            max_pairs=220,
        )
    debug_info["outer_shell_transport"] = shell_info
    debug_info["outer_shell_pair_count"] = int(shell_src_img.shape[0])

    # 5.1 均匀缩放专用保形配对：按拟合相似变换推动外壳，抑制“圆变扭曲”。
    if intent_type == "UNIFORM_SCALE":
        sim_src_img, sim_tgt_img, sim_info = _build_similarity_shell_pairs(
            mask_u8=m_u8,
            scale_factor=float(intent_params.get("scale_factor", 1.0)),
            scale_rotation=intent_params.get("scale_rotation", np.eye(2, dtype=np.float32)),
            scale_translation=intent_params.get("scale_translation", np.zeros((2,), dtype=np.float32)),
            max_pairs=260 if len(comp_sam_pts) <= 3 else 200,
        )
    else:
        sim_src_img = np.zeros((0, 2), dtype=np.float32)
        sim_tgt_img = np.zeros((0, 2), dtype=np.float32)
        sim_info = {"reason": "not_uniform_scale"}
    debug_info["uniform_scale_shell_pairs"] = sim_info
    debug_info["uniform_scale_shell_pair_count"] = int(sim_src_img.shape[0])

    # 5.2 FREE_STRETCH 专用内部恒等约束（抑制过拉，不改变控制点目标）
    if intent_type == "FREE_STRETCH" and free_stretch_profile != "HARD_CONTROL_GLOBAL":
        free_src_img, free_tgt_img, free_info = _build_free_stretch_guard_pairs(
            all_pts=intent_params.get("all_pts", np.zeros((0, 2), dtype=np.float32)),
            center_of_mass=intent_params.get(
                "center_of_mass",
                np.mean(comp_sam_pts, axis=0) if len(comp_sam_pts) > 0 else np.array([0.0, 0.0], dtype=np.float32),
            ),
            handle_points=comp_sam_pts,
            influence_range=rigid_ratio,
            max_pairs=56 if len(comp_sam_pts) <= 4 else 44,
        )
    elif intent_type == "FREE_STRETCH":
        free_src_img = np.zeros((0, 2), dtype=np.float32)
        free_tgt_img = np.zeros((0, 2), dtype=np.float32)
        free_info = {"reason": "disabled_for_hard_control_global"}
    else:
        free_src_img = np.zeros((0, 2), dtype=np.float32)
        free_tgt_img = np.zeros((0, 2), dtype=np.float32)
        free_info = {"reason": "not_free_stretch"}
    debug_info["free_stretch_guard_pairs"] = free_info
    debug_info["free_stretch_guard_pair_count"] = int(free_src_img.shape[0])

    moving_src_xy = control_src_xy
    moving_tgt_xy = control_tgt_xy

    if curve_src_xy.shape[0] > 0:
        moving_src_xy = torch.cat([moving_src_xy, curve_src_xy], dim=0)
        moving_tgt_xy = torch.cat([moving_tgt_xy, curve_tgt_xy], dim=0)

    if shell_src_img.shape[0] > 0:
        shell_src_lat = torch.from_numpy(shell_src_img.copy()).float().to(device)
        shell_tgt_lat = torch.from_numpy(shell_tgt_img.copy()).float().to(device)
        shell_src_lat[:, 0] *= scale_x
        shell_src_lat[:, 1] *= scale_y
        shell_tgt_lat[:, 0] *= scale_x
        shell_tgt_lat[:, 1] *= scale_y
        moving_src_xy = torch.cat([moving_src_xy, shell_src_lat], dim=0)
        moving_tgt_xy = torch.cat([moving_tgt_xy, shell_tgt_lat], dim=0)
    else:
        shell_src_lat = torch.zeros((0, 2), device=device, dtype=torch.float32)
        shell_tgt_lat = torch.zeros((0, 2), device=device, dtype=torch.float32)

    if sim_src_img.shape[0] > 0:
        sim_src_lat = torch.from_numpy(sim_src_img.copy()).float().to(device)
        sim_tgt_lat = torch.from_numpy(sim_tgt_img.copy()).float().to(device)
        sim_src_lat[:, 0] *= scale_x
        sim_src_lat[:, 1] *= scale_y
        sim_tgt_lat[:, 0] *= scale_x
        sim_tgt_lat[:, 1] *= scale_y
        moving_src_xy = torch.cat([moving_src_xy, sim_src_lat], dim=0)
        moving_tgt_xy = torch.cat([moving_tgt_xy, sim_tgt_lat], dim=0)

    if free_src_img.shape[0] > 0:
        free_src_lat = torch.from_numpy(free_src_img.copy()).float().to(device)
        free_tgt_lat = torch.from_numpy(free_tgt_img.copy()).float().to(device)
        free_src_lat[:, 0] *= scale_x
        free_src_lat[:, 1] *= scale_y
        free_tgt_lat[:, 0] *= scale_x
        free_tgt_lat[:, 1] *= scale_y
        moving_src_xy = torch.cat([moving_src_xy, free_src_lat], dim=0)
        moving_tgt_xy = torch.cat([moving_tgt_xy, free_tgt_lat], dim=0)
    else:
        free_src_lat = torch.zeros((0, 2), device=device, dtype=torch.float32)
        free_tgt_lat = torch.zeros((0, 2), device=device, dtype=torch.float32)

    # 6. MLS Grid 计算
    mls_weight_power = 4.0
    if intent_type in single_side_modes:
        # 单侧模式降低局部性，提升轮廓连续性（更接近平滑圆弧）
        mls_weight_power = float(np.clip(2.0 + 1.1 * float(rigid_ratio), 2.0, 3.2))
    elif intent_type == "FREE_STRETCH":
        if free_stretch_profile == "HARD_CONTROL_GLOBAL":
            # 全局分布控制点：降低局部性，配合硬约束迭代可同时兼顾“到位+平滑”
            mls_weight_power = float(np.clip(2.6 + 0.7 * float(rigid_ratio), 2.6, 3.3))
        else:
            # 其他自由拉伸：提高局部性，减少互相牵连导致的过拉
            mls_weight_power = float(np.clip(3.8 + 1.2 * float(rigid_ratio), 3.8, 5.0))
    elif intent_type == "BILATERAL_UNIFORM":
        mls_weight_power = float(np.clip(2.4 + 1.0 * float(rigid_ratio), 2.2, 3.4))

    pts_current_state = torch.cat([moving_tgt_xy, anchors_lat], dim=0)
    pts_original_state = torch.cat([moving_src_xy, anchors_lat], dim=0)
    grid_abs = solve_mls_affine_grid(
        pts_current_state,
        pts_original_state,
        H_lat,
        W_lat,
        device,
        weight_power=mls_weight_power,
    )

    # 关键修复：MLS 后追加亚像素硬约束，减少“差一点点到不了目标点”的误差
    grid_abs, ctrl_err_before_lat, ctrl_err_after_lat = _enforce_subpixel_handle_constraints(
        grid_abs=grid_abs,
        src_xy=control_src_xy,
        tgt_xy=control_tgt_xy,
    )
    if intent_type in single_side_modes or intent_type == "FREE_STRETCH":
        if intent_type in single_side_modes:
            smooth_strength = float(np.clip(0.28 + 0.20 * (1.0 - float(rigid_ratio)), 0.25, 0.45))
            smooth_iters = 2 if int(control_src_xy.shape[0]) <= 3 else 1
        elif free_stretch_profile == "HARD_CONTROL_GLOBAL":
            # 关键：平滑与硬约束交替，避免“平滑后跑偏”或“纯硬约束锯齿”
            alt_steps = 3 if int(control_src_xy.shape[0]) <= 5 else 2
            smooth_strength = float(np.clip(0.20 + 0.12 * (1.0 - float(rigid_ratio)), 0.20, 0.32))

            # 先把控制点精确压到位，再交替平滑
            for _ in range(2):
                grid_abs, _, _ = _enforce_subpixel_handle_constraints(
                    grid_abs=grid_abs,
                    src_xy=control_src_xy,
                    tgt_xy=control_tgt_xy,
                )
            for _ in range(alt_steps):
                grid_abs, _ = _smooth_grid_displacement_field(
                    grid_abs=grid_abs,
                    strength=smooth_strength,
                    iters=1,
                )
                grid_abs, _, _ = _enforce_subpixel_handle_constraints(
                    grid_abs=grid_abs,
                    src_xy=control_src_xy,
                    tgt_xy=control_tgt_xy,
                )
            grid_abs, _, ctrl_err_after_smooth_lat = _enforce_subpixel_handle_constraints(
                grid_abs=grid_abs,
                src_xy=control_src_xy,
                tgt_xy=control_tgt_xy,
            )
            single_side_smooth_info = {
                "enabled": True,
                "mode": "free_stretch_hard_control_global",
                "strength": float(smooth_strength),
                "iters": int(alt_steps),
            }
            # 跳过下方通用平滑分支
            smooth_strength = None
            smooth_iters = 0
        else:
            # 自由拉伸：只做轻平滑，避免把严格控制点结果再次“抹回去”
            smooth_strength = float(np.clip(0.14 + 0.10 * (1.0 - float(rigid_ratio)), 0.14, 0.24))
            smooth_iters = 1

        if smooth_strength is not None:
            grid_abs, single_side_smooth_info = _smooth_grid_displacement_field(
                grid_abs=grid_abs,
                strength=smooth_strength,
                iters=smooth_iters,
            )
            if intent_type in single_side_modes and curve_src_xy.shape[0] > 0:
                grid_abs, _, _ = _enforce_subpixel_handle_constraints(
                    grid_abs=grid_abs,
                    src_xy=torch.cat([control_src_xy, curve_src_xy], dim=0),
                    tgt_xy=torch.cat([control_tgt_xy, curve_tgt_xy], dim=0),
                )
            if (
                intent_type in single_side_modes
                and shell_src_lat.shape[0] > 0
            ):
                # 单边模式平滑后重压“外壳随动”约束：外层跟着走，但不参与二次拉伸
                grid_abs, _, _ = _enforce_subpixel_handle_constraints(
                    grid_abs=grid_abs,
                    src_xy=shell_src_lat,
                    tgt_xy=shell_tgt_lat,
                )
            if intent_type == "FREE_STRETCH" and anchors_lat.shape[0] > 0:
                # 保证“质心固定点”在平滑后依然锁定
                grid_abs, _, _ = _enforce_subpixel_handle_constraints(
                    grid_abs=grid_abs,
                    src_xy=anchors_lat,
                    tgt_xy=anchors_lat,
                )
            if intent_type == "FREE_STRETCH" and free_src_lat.shape[0] > 0:
                # 保证内部 guard 约束不被平滑破坏，抑制中间区过拉
                grid_abs, _, _ = _enforce_subpixel_handle_constraints(
                    grid_abs=grid_abs,
                    src_xy=free_src_lat,
                    tgt_xy=free_tgt_lat,
                )
            grid_abs, _, ctrl_err_after_smooth_lat = _enforce_subpixel_handle_constraints(
                grid_abs=grid_abs,
                src_xy=control_src_xy,
                tgt_xy=control_tgt_xy,
            )
            if intent_type == "FREE_STRETCH":
                single_side_smooth_info["mode"] = (
                    "free_stretch_centroid"
                    if free_stretch_profile != "HARD_CONTROL_GLOBAL"
                    else "free_stretch_hard_control_global"
                )
    else:
        single_side_smooth_info = {"enabled": False, "reason": "not_single_side_mode"}
        ctrl_err_after_smooth_lat = ctrl_err_after_lat
    
    # 7. 归一化 Grid
    norm_grid = torch.zeros_like(grid_abs)
    norm_grid[..., 0] = 2.0 * grid_abs[..., 0] / (W_lat - 1) - 1.0
    norm_grid[..., 1] = 2.0 * grid_abs[..., 1] / (H_lat - 1) - 1.0
    
    # 8. Warp Latent
    warped_latents = _grid_sample_latents_aa(
        latents,
        norm_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )
    
    # 9. Warp Mask
    comp_mask_end = F.grid_sample(
        comp_mask_start.view(1, 1, H_lat, W_lat),
        norm_grid.unsqueeze(0),
        mode='bilinear', align_corners=True
    ).squeeze()
    
    # 10. 保存调试信息
    subpixel_constraint_mode = _get_subpixel_constraint_mode()
    debug_info['norm_grid'] = norm_grid
    debug_info['pivot_img'] = None  # 非刚性没有支点
    debug_info['warped_latents'] = warped_latents.clone()
    debug_info['mask_end'] = comp_mask_end.clone()
    debug_info['subpixel_constraint_mode'] = str(subpixel_constraint_mode)
    debug_info['ctrl_err_before_lat'] = ctrl_err_before_lat
    debug_info['ctrl_err_after_lat'] = ctrl_err_after_lat
    debug_info['single_side_smooth'] = single_side_smooth_info
    debug_info['ctrl_err_after_smooth_lat'] = float(ctrl_err_after_smooth_lat)
    debug_info['mls_weight_power'] = float(mls_weight_power)
    # 换算到图像空间（x尺度），便于日志理解
    scale_safe = max(float(scale_x), 1e-6)
    debug_info['ctrl_err_before_img'] = float(ctrl_err_before_lat / scale_safe)
    debug_info['ctrl_err_after_img'] = float(ctrl_err_after_lat / scale_safe)
    
    print(
        f"  [Non-Rigid] Sub-action: {intent_type}, {method_desc}, Anchors: {len(anchors_img)}, "
        f"SubPx={subpixel_constraint_mode}, "
        f"CtrlErr(lat): {ctrl_err_before_lat:.3f}->{ctrl_err_after_lat:.3f}->{ctrl_err_after_smooth_lat:.3f}"
    )
    
    return warped_latents, comp_mask_end, "Non-Rigid", None, anchors_img, debug_info


def _postprocess_projected_rgb_depth_mask(warped_rgb_np, warped_depth_np, rotated_mask_u8):
    """
    投影后轻量平滑（非3D结构补全）：
    1) 闭运算找主体内部小孔洞
    2) RGB 用 inpaint 填洞，Depth 用最近邻深度补洞
    3) 边缘轻微模糊，减小锯齿
    """
    if rotated_mask_u8 is None:
        return warped_rgb_np, warped_depth_np, rotated_mask_u8

    mask_u8 = rotated_mask_u8.astype(np.uint8)
    H, W = mask_u8.shape[:2]
    if H < 2 or W < 2:
        return warped_rgb_np, warped_depth_np, mask_u8
    if np.sum(mask_u8 > 127) == 0:
        return warped_rgb_np, warped_depth_np, mask_u8

    k = max(3, int(round(0.01 * min(H, W))))
    if k % 2 == 0:
        k += 1
    k = int(np.clip(k, 3, 9))
    kernel_close = np.ones((k, k), np.uint8)
    mask_closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel_close)

    internal_holes = (mask_closed > 0) & (mask_u8 == 0)
    hole_count = int(np.sum(internal_holes))

    if hole_count > 0:
        # 深度补洞：最近邻复制（仅内部小孔）
        if isinstance(warped_depth_np, np.ndarray) and warped_depth_np.shape[:2] == (H, W):
            try:
                from scipy import ndimage
                valid = mask_u8 > 0
                dist, nearest_idx = ndimage.distance_transform_edt(~valid, return_indices=True)
                holes_near = internal_holes & (dist <= max(2, k))
                hy, hx = np.where(holes_near)
                if hy.size > 0:
                    ny = nearest_idx[0, hy, hx]
                    nx = nearest_idx[1, hy, hx]
                    warped_depth_np[hy, hx] = warped_depth_np[ny, nx]
                    internal_holes = holes_near
                    hole_count = int(hy.size)
            except Exception:
                pass

        # RGB补洞：仅对内部孔洞 inpaint，不扩展主体轮廓
        if isinstance(warped_rgb_np, np.ndarray) and warped_rgb_np.shape[:2] == (H, W):
            hole_mask_u8 = internal_holes.astype(np.uint8) * 255
            if np.any(hole_mask_u8 > 0):
                try:
                    warped_bgr = cv2.cvtColor(warped_rgb_np, cv2.COLOR_RGB2BGR)
                    inpainted_bgr = cv2.inpaint(
                        warped_bgr, hole_mask_u8, inpaintRadius=3, flags=cv2.INPAINT_TELEA
                    )
                    warped_rgb_np = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
                except Exception:
                    pass

        mask_u8[internal_holes] = 255

    # 边缘轻微抗锯齿
    if isinstance(warped_rgb_np, np.ndarray) and warped_rgb_np.shape[:2] == (H, W):
        edge_kernel = np.ones((3, 3), np.uint8)
        mask_eroded = cv2.erode(mask_u8, edge_kernel, iterations=1)
        edge_pixels = (mask_u8 > 0) & (mask_eroded == 0)
        if np.any(edge_pixels):
            blurred_rgb = cv2.GaussianBlur(warped_rgb_np, (3, 3), 0.5)
            warped_rgb_np[edge_pixels] = blurred_rgb[edge_pixels]

    if hole_count > 0:
        print(f"[3D Raster Smooth] internal_holes_filled={hole_count}, kernel={k}")

    return warped_rgb_np, warped_depth_np, mask_u8


def _projective_3d_warp_from_points(
    latents, H_lat, W_lat, device,
    old_xy, new_xy, new_z,
    scale_x, scale_y,
    source_image_np=None,
    mask_start_full=None,
    need_image_outputs=True,
    enable_subject_scope_fill=False,
):
    """
    基于3D投影结果执行重映射（非MLS）:
    - 输入为每个前景点的 old/new 2D投影与 new_z
    - 通过 z-buffer 生成 source->target 的稀疏可见映射
    - 在 image / latent 两个分辨率分别构建采样映射
    """
    old_xy = np.asarray(old_xy, dtype=np.float32).reshape(-1, 2)
    new_xy = np.asarray(new_xy, dtype=np.float32).reshape(-1, 2)
    new_z = np.asarray(new_z, dtype=np.float32).reshape(-1)

    if old_xy.shape[0] == 0:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H_lat, device=device),
            torch.arange(W_lat, device=device),
            indexing='ij'
        )
        norm_grid = torch.zeros((H_lat, W_lat, 2), device=device, dtype=torch.float32)
        norm_grid[..., 0] = 2.0 * grid_x.float() / (W_lat - 1) - 1.0
        norm_grid[..., 1] = 2.0 * grid_y.float() / (H_lat - 1) - 1.0
        if isinstance(mask_start_full, np.ndarray):
            H_img, W_img = mask_start_full.shape[:2]
        elif isinstance(source_image_np, np.ndarray):
            H_img, W_img = source_image_np.shape[:2]
        else:
            H_img = _infer_image_size_from_latent_and_scale(H_lat, scale_y)
            W_img = _infer_image_size_from_latent_and_scale(W_lat, scale_x)
        empty_m = np.zeros((H_img, W_img), dtype=np.float32)
        empty_rgb = np.zeros((H_img, W_img, 3), dtype=np.uint8) if isinstance(source_image_np, np.ndarray) else None
        empty_depth = np.zeros((H_img, W_img), dtype=np.float32)
        empty_debug = {
            "subject_scope_mask_full": empty_m.astype(np.float32),
            "subject_hole_mask_full": empty_m.astype(np.float32),
            "subject_scope_mask_lat": np.zeros((H_lat, W_lat), dtype=np.float32),
            "subject_hole_mask_lat": np.zeros((H_lat, W_lat), dtype=np.float32),
        }
        return latents.clone(), torch.zeros((H_lat, W_lat), device=device), norm_grid, empty_m, empty_rgb, empty_depth, (empty_m * 255).astype(np.uint8), empty_debug

    if isinstance(source_image_np, np.ndarray):
        H_img, W_img = source_image_np.shape[:2]
    elif isinstance(mask_start_full, np.ndarray):
        H_img, W_img = mask_start_full.shape[:2]
    else:
        H_img = _infer_image_size_from_latent_and_scale(H_lat, scale_y)
        W_img = _infer_image_size_from_latent_and_scale(W_lat, scale_x)

    order = np.argsort(new_z)

    warped_rgb_np = None
    warped_depth_np = None
    rotated_mask_u8 = None
    m_end_full = None
    completion_info_img = {'filled_count': 0, 'reason': 'skipped'}

    proj_debug = {
        "subject_scope_mask_full": np.zeros((H_img, W_img), dtype=np.float32),
        "subject_hole_mask_full": np.zeros((H_img, W_img), dtype=np.float32),
        "subject_scope_mask_lat": np.zeros((H_lat, W_lat), dtype=np.float32),
        "subject_hole_mask_lat": np.zeros((H_lat, W_lat), dtype=np.float32),
    }

    if bool(need_image_outputs):
        # ===== image-resolution z-buffer =====
        z_buf_img = np.full((H_img, W_img), -np.inf, dtype=np.float32)
        src_map_x_img = np.zeros((H_img, W_img), dtype=np.float32)
        src_map_y_img = np.zeros((H_img, W_img), dtype=np.float32)
        warped_rgb_np = np.zeros((H_img, W_img, 3), dtype=np.uint8) if isinstance(source_image_np, np.ndarray) else None

        for idx in order:
            nx = int(round(float(new_xy[idx, 0])))
            ny = int(round(float(new_xy[idx, 1])))
            if not (0 <= nx < W_img and 0 <= ny < H_img):
                continue
            z_val = float(new_z[idx])
            if z_val > z_buf_img[ny, nx]:
                z_buf_img[ny, nx] = z_val
                sx = float(np.clip(old_xy[idx, 0], 0, W_img - 1))
                sy = float(np.clip(old_xy[idx, 1], 0, H_img - 1))
                src_map_x_img[ny, nx] = sx
                src_map_y_img[ny, nx] = sy
                if warped_rgb_np is not None:
                    sx_i = int(round(sx))
                    sy_i = int(round(sy))
                    warped_rgb_np[ny, nx] = source_image_np[sy_i, sx_i]

        valid_img = z_buf_img > -np.inf
        completion_info_img = {'filled_count': 0, 'reason': 'disabled'}

        if warped_rgb_np is not None:
            remapped_rgb = cv2.remap(
                source_image_np,
                src_map_x_img.astype(np.float32),
                src_map_y_img.astype(np.float32),
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REFLECT,
            )
            warped_rgb_np = np.zeros_like(remapped_rgb)
            warped_rgb_np[valid_img] = remapped_rgb[valid_img]

        rotated_mask_u8 = (valid_img.astype(np.uint8) * 255)

        warped_depth_np = np.zeros((H_img, W_img), dtype=np.float32)
        warped_depth_np[valid_img] = z_buf_img[valid_img]

        warped_rgb_np, warped_depth_np, rotated_mask_u8 = _postprocess_projected_rgb_depth_mask(
            warped_rgb_np=warped_rgb_np,
            warped_depth_np=warped_depth_np,
            rotated_mask_u8=rotated_mask_u8,
        )

        # 主体范围兜底：自动识别最终主体范围，并把内部空洞并入主体mask
        if bool(enable_subject_scope_fill):
            try:
                scope_img, hole_img, scope_info_img = infer_subject_fill_scope(
                    rotated_mask_u8,
                    close_ratio=0.10,
                    max_close=35,
                    max_fill_dist_ratio=0.18,
                )
                proj_debug["subject_scope_mask_full"] = np.asarray(scope_img, dtype=np.float32)
                proj_debug["subject_hole_mask_full"] = np.asarray(hole_img, dtype=np.float32)
                hole_n_img = int(scope_info_img.get("hole_pixels", 0))
                if hole_n_img > 0:
                    if warped_rgb_np is not None:
                        img_radius = max(2, int(round(0.018 * max(H_img, W_img))))
                        warped_rgb_np = fill_background_holes(
                            image_input=warped_rgb_np,
                            hole_mask=hole_img,
                            forbidden_mask=(scope_img <= 0.5).astype(np.float32),
                            device=device,
                            radius=img_radius,
                        )
                    rotated_mask_u8 = np.maximum(rotated_mask_u8, (scope_img * 255.0).astype(np.uint8))
                    print(
                        f"[3D Subject Scope][Image] holes_filled={hole_n_img}, "
                        f"k={scope_info_img.get('kernel', 0)}"
                    )
            except Exception as e:
                print(f"[3D Subject Scope][Image] failed: {e}")

        m_end_full = (rotated_mask_u8.astype(np.float32) / 255.0)

    # ===== latent-resolution z-buffer =====
    old_lat_x = old_xy[:, 0] * float(scale_x)
    old_lat_y = old_xy[:, 1] * float(scale_y)
    new_lat_x = new_xy[:, 0] * float(scale_x)
    new_lat_y = new_xy[:, 1] * float(scale_y)

    z_buf_lat = np.full((H_lat, W_lat), -np.inf, dtype=np.float32)
    src_map_x_lat = np.zeros((H_lat, W_lat), dtype=np.float32)
    src_map_y_lat = np.zeros((H_lat, W_lat), dtype=np.float32)

    for idx in order:
        nx = int(round(float(new_lat_x[idx])))
        ny = int(round(float(new_lat_y[idx])))
        if not (0 <= nx < W_lat and 0 <= ny < H_lat):
            continue
        z_val = float(new_z[idx])
        if z_val > z_buf_lat[ny, nx]:
            z_buf_lat[ny, nx] = z_val
            src_map_x_lat[ny, nx] = float(np.clip(old_lat_x[idx], 0, W_lat - 1))
            src_map_y_lat[ny, nx] = float(np.clip(old_lat_y[idx], 0, H_lat - 1))

    valid_lat = z_buf_lat > -np.inf
    completion_info_lat = {'filled_count': 0, 'reason': 'disabled'}

    grid_x_np, grid_y_np = np.meshgrid(
        np.arange(W_lat, dtype=np.float32),
        np.arange(H_lat, dtype=np.float32),
        indexing='xy'
    )
    sample_x_np = grid_x_np.copy()
    sample_y_np = grid_y_np.copy()
    sample_x_np[valid_lat] = src_map_x_lat[valid_lat]
    sample_y_np[valid_lat] = src_map_y_lat[valid_lat]

    img_fill_n = int(completion_info_img.get('filled_count', 0))
    lat_fill_n = int(completion_info_lat.get('filled_count', 0))
    if img_fill_n > 0 or lat_fill_n > 0:
        print(f"[3D Completion] image_filled={img_fill_n}, latent_filled={lat_fill_n}")

    norm_grid_np = np.zeros((H_lat, W_lat, 2), dtype=np.float32)
    norm_grid_np[..., 0] = 2.0 * sample_x_np / (W_lat - 1) - 1.0
    norm_grid_np[..., 1] = 2.0 * sample_y_np / (H_lat - 1) - 1.0
    norm_grid = torch.from_numpy(norm_grid_np).to(device).float()

    warped_latents = F.grid_sample(
        latents.float(), norm_grid.unsqueeze(0),
        mode='nearest', padding_mode='border', align_corners=True
    ).to(latents.dtype)

    mask_lat_np = valid_lat.astype(np.float32)
    if bool(enable_subject_scope_fill):
        try:
            scope_lat, hole_lat, scope_info_lat = infer_subject_fill_scope(
                mask_lat_np.astype(np.float32),
                close_ratio=0.12,
                max_close=max(5, int(round(0.35 * min(H_lat, W_lat)))),
                max_fill_dist_ratio=0.22,
            )
            scope_lat = (scope_lat > 0.5).astype(np.float32)
            hole_lat = np.logical_and(hole_lat > 0.5, scope_lat > 0.5).astype(np.float32)
            proj_debug["subject_scope_mask_lat"] = np.asarray(scope_lat, dtype=np.float32)
            proj_debug["subject_hole_mask_lat"] = np.asarray(hole_lat, dtype=np.float32)
            # 关键：开启 subject scope 时，将主体内部洞并入最终主体 mask，确保后续 latent 合成可见。
            mask_lat_np = np.maximum(mask_lat_np, scope_lat).astype(np.float32)

            scope_full_from_lat = _latent_mask_to_fullres(scope_lat, target_hw=(H_img, W_img))
            hole_full_from_lat = _latent_mask_to_fullres(hole_lat, target_hw=(H_img, W_img))
            proj_debug["subject_scope_mask_full"] = np.maximum(
                np.asarray(proj_debug.get("subject_scope_mask_full", np.zeros((H_img, W_img), dtype=np.float32)), dtype=np.float32),
                np.asarray(scope_full_from_lat, dtype=np.float32),
            )
            proj_debug["subject_hole_mask_full"] = np.maximum(
                np.asarray(proj_debug.get("subject_hole_mask_full", np.zeros((H_img, W_img), dtype=np.float32)), dtype=np.float32),
                np.asarray(hole_full_from_lat, dtype=np.float32),
            )
            hole_lat_n = int(np.sum(hole_lat > 0.5))
            if hole_lat_n > 0:
                print(
                    f"[3D Subject Scope][Latent] holes={hole_lat_n}, "
                    f"k={scope_info_lat.get('kernel', 0)}"
                )
        except Exception as e:
            print(f"[3D Subject Scope][Latent] failed: {e}")
    comp_mask_end = torch.from_numpy(mask_lat_np).to(device).float()

    return warped_latents, comp_mask_end, norm_grid, m_end_full, warped_rgb_np, warped_depth_np, rotated_mask_u8, proj_debug


def _sample_nearest_foreground_depth(depth_map, fg_mask_u8, x, y, fallback_depth, max_radius=24):
    """
    在前景mask中为(x,y)寻找最近有效深度；若找不到则回退fallback_depth。
    返回: (sampled_depth, sampled_xy, sample_dist_px)
    """
    H, W = depth_map.shape[:2]
    xi = int(np.clip(round(float(x)), 0, W - 1))
    yi = int(np.clip(round(float(y)), 0, H - 1))

    if fg_mask_u8[yi, xi] > 127:
        return float(depth_map[yi, xi]), (xi, yi), 0.0

    r = max(1, int(max_radius))
    x0 = max(0, xi - r)
    x1 = min(W, xi + r + 1)
    y0 = max(0, yi - r)
    y1 = min(H, yi + r + 1)

    patch_mask = fg_mask_u8[y0:y1, x0:x1] > 127
    if not np.any(patch_mask):
        return float(fallback_depth), (xi, yi), -1.0

    ys, xs = np.where(patch_mask)
    xs = xs + x0
    ys = ys + y0
    d2 = (xs.astype(np.float32) - float(x)) ** 2 + (ys.astype(np.float32) - float(y)) ** 2
    j = int(np.argmin(d2))
    sx = int(xs[j])
    sy = int(ys[j])
    return float(depth_map[sy, sx]), (sx, sy), float(np.sqrt(float(d2[j])))


def _sample_anchors_spatially_balanced(candidate_xy, candidate_scores, max_anchors=220):
    """
    对候选锚点做空间均衡采样，避免锚点扎堆。
    """
    if candidate_xy is None or len(candidate_xy) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    pts = np.asarray(candidate_xy, dtype=np.float32)
    scores = np.asarray(candidate_scores, dtype=np.float32).reshape(-1)
    n = pts.shape[0]
    if scores.shape[0] != n:
        scores = np.ones((n,), dtype=np.float32)

    if n <= max_anchors:
        return pts

    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    span_x = max(float(x_max - x_min), 1.0)
    span_y = max(float(y_max - y_min), 1.0)

    grid_n = int(np.clip(round(np.sqrt(max_anchors) * 1.8), 12, 40))
    cell_w = span_x / float(grid_n)
    cell_h = span_y / float(grid_n)

    best_idx_per_cell = {}
    for i in range(n):
        cx = int(np.clip((pts[i, 0] - x_min) / (cell_w + 1e-6), 0, grid_n - 1))
        cy = int(np.clip((pts[i, 1] - y_min) / (cell_h + 1e-6), 0, grid_n - 1))
        key = (cx, cy)
        prev = best_idx_per_cell.get(key)
        if prev is None or scores[i] > scores[prev]:
            best_idx_per_cell[key] = i

    selected = np.array(sorted(best_idx_per_cell.values()), dtype=np.int32)
    if selected.size == 0:
        order = np.argsort(scores)[::-1][:max_anchors]
        return pts[order].astype(np.float32)

    if selected.size > max_anchors:
        order = np.argsort(scores[selected])[::-1][:max_anchors]
        selected = selected[order]
    elif selected.size < max_anchors:
        mask = np.ones((n,), dtype=bool)
        mask[selected] = False
        remain = np.where(mask)[0]
        if remain.size > 0:
            need = min(max_anchors - selected.size, remain.size)
            fill = remain[np.argsort(scores[remain])[::-1][:need]]
            selected = np.concatenate([selected, fill], axis=0)

    return pts[selected].astype(np.float32)


def _build_drag_direction_profile(handle_points_xy, target_points_xy, min_motion_px=1.5):
    """
    统计拖拽方向画像（仅使用有效拖拽对）。
    返回:
      - handles_xy: 有效 handle 点 (M,2)
      - dirs_unit: 对应单位方向向量 (M,2)
      - weights: 方向权重（按拖拽位移）(M,)
      - centroid: 按权重聚合的控制点中心
      - mean_dir_unit/global_available: 全局主方向及其可用性
    """
    handles = _sanitize_xy_points(handle_points_xy)
    targets = _sanitize_xy_points(target_points_xy)
    pair_n = min(handles.shape[0], targets.shape[0])
    if pair_n <= 0:
        return {
            'enabled': False,
            'reason': 'empty_pairs',
            'pair_count': int(pair_n),
            'active_count': 0,
        }

    handles = handles[:pair_n]
    targets = targets[:pair_n]
    drags = targets - handles
    motion = np.linalg.norm(drags, axis=1)
    active = motion > float(min_motion_px)
    active_count = int(np.sum(active))
    if active_count <= 0:
        return {
            'enabled': False,
            'reason': 'tiny_motion',
            'pair_count': int(pair_n),
            'active_count': int(active_count),
            'mean_motion_px': float(np.mean(motion)) if motion.size > 0 else 0.0,
        }

    handles_a = handles[active].astype(np.float32)
    drags_a = drags[active].astype(np.float32)
    motion_a = motion[active].astype(np.float32)

    dirs_unit = drags_a / (motion_a[:, None] + 1e-6)
    weights = motion_a / (float(np.sum(motion_a)) + 1e-6)
    centroid = np.sum(handles_a * weights[:, None], axis=0).astype(np.float32)

    mean_vec = np.sum(dirs_unit * weights[:, None], axis=0)
    mean_vec_norm = float(np.linalg.norm(mean_vec))
    global_available = bool(mean_vec_norm >= 0.12)
    if global_available:
        mean_dir_unit = (mean_vec / (mean_vec_norm + 1e-6)).astype(np.float32)
    else:
        mean_dir_unit = np.zeros((2,), dtype=np.float32)

    span_x = float(np.max(handles_a[:, 0]) - np.min(handles_a[:, 0])) if handles_a.shape[0] > 0 else 0.0
    span_y = float(np.max(handles_a[:, 1]) - np.min(handles_a[:, 1])) if handles_a.shape[0] > 0 else 0.0
    motion_span_px = max(float(np.hypot(span_x, span_y)), float(np.percentile(motion_a, 75.0)), 1.0)

    return {
        'enabled': True,
        'reason': 'ok',
        'pair_count': int(pair_n),
        'active_count': int(active_count),
        'handles_xy': handles_a.astype(np.float32),
        'dirs_unit': dirs_unit.astype(np.float32),
        'weights': weights.astype(np.float32),
        'centroid': centroid.astype(np.float32),
        'mean_dir_unit': mean_dir_unit.astype(np.float32),
        'global_available': bool(global_available),
        'mean_dir_consensus': float(mean_vec_norm),
        'mean_motion_px': float(np.mean(motion_a)),
        'motion_span_px': float(motion_span_px),
    }


def _compute_drag_opposition_scores(candidate_xy, drag_profile):
    """
    计算候选点对“拖拽反方向”的匹配分数，范围 [0, 1]。
    1 越倾向于固定在拖拽相反侧，0 越不匹配。
    """
    pts = np.asarray(candidate_xy, dtype=np.float32)
    if pts.ndim == 1 and pts.shape[0] >= 2:
        pts = pts.reshape(1, 2)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32), {
            'enabled': False,
            'reason': 'empty_candidates',
        }
    n_total = int(pts.shape[0])
    if pts.shape[1] > 2:
        pts = pts[:, :2]
    finite_mask = np.isfinite(pts).all(axis=1)
    if not np.any(finite_mask):
        return np.full((n_total,), 0.0, dtype=np.float32), {
            'enabled': False,
            'reason': 'non_finite_candidates',
        }
    pts_eval = pts[finite_mask]
    if not bool(drag_profile.get('enabled', False)):
        return np.full((n_total,), 0.5, dtype=np.float32), {
            'enabled': False,
            'reason': str(drag_profile.get('reason', 'disabled')),
        }

    handles = np.asarray(drag_profile.get('handles_xy', np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    dirs_unit = np.asarray(drag_profile.get('dirs_unit', np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    weights = np.asarray(drag_profile.get('weights', np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if handles.shape[0] == 0 or dirs_unit.shape[0] == 0 or weights.shape[0] == 0:
        return np.full((n_total,), 0.5, dtype=np.float32), {
            'enabled': False,
            'reason': 'invalid_profile',
        }

    rel = pts_eval[:, None, :] - handles[None, :, :]  # (N, M, 2)
    rel_norm = np.linalg.norm(rel, axis=2) + 1e-6

    # cos(theta): theta 是候选点方向 vs (-drag_dir)，1 表示完全在反方向
    cos_opp = -np.sum(rel * dirs_unit[None, :, :], axis=2) / rel_norm
    local_cos_score = np.clip((cos_opp + 1.0) * 0.5, 0.0, 1.0)

    dist_scale = max(float(drag_profile.get('motion_span_px', 1.0)) * 0.20, 8.0)
    dist_gain = 1.0 - np.exp(-rel_norm / dist_scale)
    local_score = local_cos_score * dist_gain
    weighted_local = np.sum(local_score * weights[None, :], axis=1)

    global_score = np.full((pts_eval.shape[0],), 0.5, dtype=np.float32)
    if bool(drag_profile.get('global_available', False)):
        centroid = np.asarray(drag_profile.get('centroid', np.zeros((2,), dtype=np.float32)), dtype=np.float32)
        mean_dir_unit = np.asarray(drag_profile.get('mean_dir_unit', np.zeros((2,), dtype=np.float32)), dtype=np.float32)
        rel_g = pts_eval - centroid[None, :]
        rel_g_norm = np.linalg.norm(rel_g, axis=1) + 1e-6
        cos_g = -np.sum(rel_g * mean_dir_unit[None, :], axis=1) / rel_g_norm
        global_score = np.clip((cos_g + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)

    # 适度凸显“反方向区域”
    direction_score = np.clip(0.58 * weighted_local + 0.42 * global_score, 0.0, 1.0)
    direction_score = np.power(direction_score, 1.25).astype(np.float32)

    full_direction_score = np.zeros((n_total,), dtype=np.float32)
    full_direction_score[finite_mask] = direction_score

    info = {
        'enabled': True,
        'mean': float(np.mean(direction_score)),
        'q25': float(np.percentile(direction_score, 25.0)),
        'q50': float(np.percentile(direction_score, 50.0)),
        'q75': float(np.percentile(direction_score, 75.0)),
        'valid_count': int(np.sum(finite_mask)),
        'invalid_count': int(n_total - int(np.sum(finite_mask))),
    }
    return full_direction_score, info


def _select_opposite_edge_anchor_indices(
    points_xy,
    points_z,
    handle_points_xy,
    target_points_xy,
    influence_ratio=0.5,
    max_keep=96,
    exclude_indices=None,
    sector_num=18,
):
    """
    选择“拖拽反方向边缘环带”锚点索引（用于保底锚点）：
    - 严格从边缘带中选
    - 反方向作为强偏好（非过硬阈值，避免坍缩成一小块）
    - 小 z（更靠前）作为稳定先验
    - 扇区轮询采样，保证覆盖范围
    """
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32), {'enabled': False, 'reason': 'empty_points'}

    z_vals = np.asarray(points_z, dtype=np.float32).reshape(-1) if points_z is not None else None
    if z_vals is not None and z_vals.shape[0] != pts.shape[0]:
        z_vals = None

    finite_mask = np.isfinite(pts).all(axis=1)
    if z_vals is not None:
        finite_mask &= np.isfinite(z_vals)
    if not np.any(finite_mask):
        return np.zeros((0,), dtype=np.int32), {'enabled': False, 'reason': 'non_finite_points'}

    valid_idx = np.where(finite_mask)[0].astype(np.int32)
    pts_valid = pts[valid_idx]
    z_valid = z_vals[valid_idx] if z_vals is not None else None

    if exclude_indices is not None:
        ex = np.asarray(exclude_indices, dtype=np.int32).reshape(-1)
        ex = ex[(ex >= 0) & (ex < pts.shape[0])]
        if ex.size > 0:
            keep_mask = ~np.isin(valid_idx, np.unique(ex))
            valid_idx = valid_idx[keep_mask]
            pts_valid = pts_valid[keep_mask]
            z_valid = z_valid[keep_mask] if z_valid is not None else None
    if pts_valid.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32), {'enabled': False, 'reason': 'all_excluded'}

    drag_profile = _build_drag_direction_profile(handle_points_xy, target_points_xy, min_motion_px=1.5)
    if not bool(drag_profile.get('enabled', False)):
        return np.zeros((0,), dtype=np.int32), {
            'enabled': False,
            'reason': str(drag_profile.get('reason', 'drag_disabled')),
        }

    mean_dir_unit = np.asarray(drag_profile.get('mean_dir_unit', np.zeros((2,), dtype=np.float32)), dtype=np.float32)
    if float(np.linalg.norm(mean_dir_unit)) < 1e-6:
        dirs_unit = np.asarray(drag_profile.get('dirs_unit', np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
        if dirs_unit.shape[0] == 0:
            return np.zeros((0,), dtype=np.int32), {'enabled': False, 'reason': 'no_drag_direction'}
        mean_dir_unit = dirs_unit[0]
    mean_dir_norm = float(np.linalg.norm(mean_dir_unit))
    if mean_dir_norm < 1e-6:
        return np.zeros((0,), dtype=np.int32), {'enabled': False, 'reason': 'zero_drag_direction'}
    mean_dir_unit = (mean_dir_unit / (mean_dir_norm + 1e-6)).astype(np.float32)

    centroid = np.asarray(drag_profile.get('centroid', np.mean(pts_valid, axis=0)), dtype=np.float32).reshape(2)
    rel = pts_valid - centroid[None, :]
    radial = np.linalg.norm(rel, axis=1)
    radial_norm = radial / (float(np.max(radial)) + 1e-6)

    opposite_axis = (-mean_dir_unit).astype(np.float32)
    cos_opp = np.sum(rel * opposite_axis[None, :], axis=1) / (radial + 1e-6)
    cos_score = np.clip((cos_opp + 1.0) * 0.5, 0.0, 1.0)

    influence_ratio = float(np.clip(influence_ratio, 0.0, 1.0))

    # 边缘带：只要主体边缘附近，避免锚点掉到内部
    xi = np.round(pts_valid[:, 0]).astype(np.int32)
    yi = np.round(pts_valid[:, 1]).astype(np.int32)
    x0 = int(np.min(xi))
    y0 = int(np.min(yi))
    x1 = int(np.max(xi))
    y1 = int(np.max(yi))
    canvas_w = max(1, x1 - x0 + 3)
    canvas_h = max(1, y1 - y0 + 3)
    local_x = np.clip(xi - x0 + 1, 0, canvas_w - 1)
    local_y = np.clip(yi - y0 + 1, 0, canvas_h - 1)
    occ = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    occ[local_y, local_x] = 255
    dist_in = cv2.distanceTransform((occ > 0).astype(np.uint8), cv2.DIST_L2, 3)
    edge_dist = dist_in[local_y, local_x].astype(np.float32)
    edge_band_px = float(np.clip(2.4 + 2.2 * influence_ratio, 2.2, 4.8))
    edge_sel = edge_dist <= edge_band_px

    min_keep = max(24, int(round(float(max_keep) * 0.34)))
    if int(np.sum(edge_sel)) < min_keep:
        edge_sel = edge_dist <= (edge_band_px * 1.8)
    if int(np.sum(edge_sel)) < max(18, int(round(float(max_keep) * 0.24))):
        edge_sel = edge_dist <= (edge_band_px * 2.5)

    cand_idx = valid_idx[edge_sel]
    if cand_idx.size == 0:
        return np.zeros((0,), dtype=np.int32), {
            'enabled': False,
            'reason': 'empty_after_edge_band',
            'edge_band_px': float(edge_band_px),
        }

    cand_rel = rel[edge_sel]
    cand_rad_norm = radial_norm[edge_sel]
    cand_cos = cos_score[edge_sel]
    cand_edge_dist = edge_dist[edge_sel]

    if z_valid is not None:
        z_cand = z_valid[edge_sel]
        z_pref_small = (float(np.max(z_cand)) - z_cand) / (float(np.max(z_cand) - np.min(z_cand)) + 1e-6)
    else:
        z_pref_small = np.full((cand_idx.shape[0],), 0.5, dtype=np.float32)

    # 反方向强偏好 + 小z + 外层 + 更贴边
    edge_pref = 1.0 - np.clip(cand_edge_dist / (edge_band_px * 2.6 + 1e-6), 0.0, 1.0)
    score = (
        0.46 * cand_cos.astype(np.float32)
        + 0.30 * z_pref_small.astype(np.float32)
        + 0.16 * cand_rad_norm.astype(np.float32)
        + 0.08 * edge_pref.astype(np.float32)
    ).astype(np.float32)

    # 方向门控：聚焦“反方向角落”而不是反方向整侧边
    proj_opp = np.sum(cand_rel * opposite_axis[None, :], axis=1)
    proj_gate = float(
        np.percentile(
            proj_opp,
            float(np.clip(66.0 + 10.0 * influence_ratio, 62.0, 82.0)),
        )
    )
    cos_gate = float(np.clip(np.percentile(cand_cos, 62.0 + 10.0 * influence_ratio), 0.56, 0.88))
    preferred = (cand_cos >= cos_gate) & (proj_opp >= proj_gate)
    preferred_min = max(10, int(round(float(max_keep) * 0.12)))
    if int(np.sum(preferred)) >= preferred_min:
        local_sel = preferred
    else:
        local_sel = cand_cos >= float(np.clip(np.percentile(cand_cos, 54.0), 0.45, 0.75))
        if int(np.sum(local_sel)) < preferred_min:
            local_sel = np.ones_like(preferred, dtype=bool)

    cand_idx = cand_idx[local_sel]
    cand_rel = cand_rel[local_sel]
    score = score[local_sel]
    cand_cos = cand_cos[local_sel]
    cand_edge_dist = cand_edge_dist[local_sel]

    keep_floor = max(14, int(round(float(max_keep) * 0.18)))
    keep_cap = int(max(keep_floor + 1, round(float(max_keep) * 0.62)))
    keep_n = int(
        np.clip(
            round(float(max_keep) * (0.34 + 0.16 * (1.0 - influence_ratio))),
            keep_floor,
            keep_cap,
        )
    )
    keep_n = min(keep_n, int(cand_idx.shape[0]))
    if keep_n <= 0:
        return np.zeros((0,), dtype=np.int32), {'enabled': False, 'reason': 'zero_keep'}

    sector_num = int(np.clip(sector_num, 12, 48))
    ang = np.arctan2(cand_rel[:, 1], cand_rel[:, 0])
    sector_ids = np.floor((ang + np.pi) / (2.0 * np.pi) * sector_num).astype(np.int32)
    sector_ids = np.clip(sector_ids, 0, sector_num - 1)

    sector_lists = []
    for sid in range(sector_num):
        pos = np.where(sector_ids == sid)[0]
        if pos.size == 0:
            sector_lists.append(np.zeros((0,), dtype=np.int32))
            continue
        order = pos[np.argsort(score[pos])[::-1]]
        sector_lists.append(order.astype(np.int32))

    selected_local = []
    cursor = [0 for _ in range(sector_num)]
    active_any = True
    while len(selected_local) < keep_n and active_any:
        active_any = False
        for sid in range(sector_num):
            lst = sector_lists[sid]
            c = cursor[sid]
            if c < lst.shape[0]:
                selected_local.append(int(lst[c]))
                cursor[sid] += 1
                active_any = True
                if len(selected_local) >= keep_n:
                    break

    if len(selected_local) == 0:
        selected_local = np.argsort(score)[::-1][:keep_n].astype(np.int32).tolist()
    selected_local = np.unique(np.asarray(selected_local, dtype=np.int32))
    if selected_local.size > keep_n:
        selected_local = selected_local[np.argsort(score[selected_local])[::-1][:keep_n]]

    edge_idx = np.unique(cand_idx[selected_local].astype(np.int32))
    info = {
        'enabled': True,
        'reason': 'ok',
        'selected_count': int(edge_idx.shape[0]),
        'candidate_count': int(np.sum(edge_sel)),
        'keep_n': int(keep_n),
        'opposite_cos_gate': float(cos_gate),
        'opposite_proj_gate': float(proj_gate),
        'preferred_count': int(np.sum(preferred)),
        'edge_band_px': float(edge_band_px),
        'edge_candidate_count': int(np.sum(edge_sel)),
        'edge_dist_mean': float(np.mean(cand_edge_dist)) if cand_edge_dist.size > 0 else 0.0,
        'opposite_cos_mean': float(np.mean(cand_cos)) if cand_cos.size > 0 else 0.0,
        'sector_num': int(sector_num),
    }
    return edge_idx, info


def _select_rotation_invariant_anchors_3d(points_3d, points_xy, handle_points_3d,
                                          influence_ratio=0.5, max_anchors=220,
                                          handle_points_xy=None, target_points_xy=None):
    """
    旋转不变的3D锚点选择（方向+深度）：
    - 远离控制点（避免把会被拖动区域作为锚）
    - 靠外层（以质心半径约束）
    - 反拖拽方向边缘环带硬约束（确保反方向边缘有锚点）
    - 小 z 软优先（不把反方向整体否掉）
    - 空间均衡采样
    """
    pts3d = np.asarray(points_3d, dtype=np.float32)
    ptsxy = np.asarray(points_xy, dtype=np.float32)
    handles3d = np.asarray(handle_points_3d, dtype=np.float32)

    if pts3d.ndim != 2 or pts3d.shape[0] == 0 or ptsxy.ndim != 2 or ptsxy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), {
            'strategy': 'ROTATION_INVARIANT_CENTROID_HANDLE',
            'strategy_name': 'Centroid+HandleInvariant',
            'reason': 'empty_points',
        }

    if handles3d.ndim != 2 or handles3d.shape[0] == 0:
        handles3d = np.zeros((1, 3), dtype=np.float32)

    # 各向同性3D距离：在刚性旋转下保持不变
    diff = pts3d[:, None, :] - handles3d[None, :, :]
    min_handle_dist = np.min(np.linalg.norm(diff, axis=2), axis=1)
    radial_dist = np.linalg.norm(pts3d, axis=1)  # 与质心的半径

    influence_ratio = float(np.clip(influence_ratio, 0.0, 1.0))
    # influence_ratio 越大（影响范围越大）=> 锚点预算越小，避免把主体锁死拖不动
    budget_scale = 0.18 + 0.82 * (1.0 - influence_ratio)
    min_budget = min(int(max_anchors), 18)
    anchor_budget = int(np.clip(round(float(max_anchors) * budget_scale), min_budget, int(max_anchors)))
    anchor_budget = max(1, anchor_budget)

    q_handle = 55.0 + 15.0 * influence_ratio
    q_radial = 50.0 + 10.0 * influence_ratio

    h_thr = float(np.percentile(min_handle_dist, q_handle))
    r_thr = float(np.percentile(radial_dist, q_radial))
    sel = (min_handle_dist >= h_thr) & (radial_dist >= r_thr)

    min_keep = max(16, int(min(anchor_budget, ptsxy.shape[0]) * 0.28))
    if int(sel.sum()) < min_keep:
        h_relax = float(np.percentile(min_handle_dist, max(45.0, q_handle - 12.0)))
        r_relax = float(np.percentile(radial_dist, max(40.0, q_radial - 10.0)))
        sel = (min_handle_dist >= h_relax) & (radial_dist >= r_relax)
    if int(sel.sum()) < min_keep:
        sel = (min_handle_dist >= float(np.percentile(min_handle_dist, 65.0)))
    if int(sel.sum()) < min_keep:
        sel = (radial_dist >= float(np.percentile(radial_dist, 55.0)))
    if int(sel.sum()) == 0:
        sel = np.ones((ptsxy.shape[0],), dtype=bool)

    cand_indices = np.where(sel)[0].astype(np.int32)
    cand_xy = ptsxy[cand_indices]
    cand_h = min_handle_dist[cand_indices]
    cand_r = radial_dist[cand_indices]

    h_norm = cand_h / (float(np.max(cand_h)) + 1e-6) if cand_h.size > 0 else np.zeros((0,), dtype=np.float32)
    r_norm = cand_r / (float(np.max(cand_r)) + 1e-6) if cand_r.size > 0 else np.zeros((0,), dtype=np.float32)

    # 旧版几何分数，作为兜底
    base_score = (0.70 * h_norm + 0.30 * r_norm).astype(np.float32)
    anchors_base = _sample_anchors_spatially_balanced(cand_xy, base_score, max_anchors=anchor_budget)
    base_indices = (
        _map_xy_controls_to_indices(ptsxy, anchors_base)
        if isinstance(anchors_base, np.ndarray) and anchors_base.ndim == 2 and anchors_base.shape[0] > 0
        else np.zeros((0,), dtype=np.int32)
    )
    if base_indices.size > 0:
        base_indices = np.unique(base_indices.astype(np.int32))

    drag_profile = _build_drag_direction_profile(handle_points_xy, target_points_xy, min_motion_px=1.5)
    full_direction_score = np.full((ptsxy.shape[0],), 0.5, dtype=np.float32)
    direction_info = {'enabled': False, 'reason': str(drag_profile.get('reason', 'disabled'))}
    if bool(drag_profile.get('enabled', False)):
        full_direction_score, direction_info = _compute_drag_opposition_scores(ptsxy, drag_profile)
        # 方向模式下，提高锚点预算，避免数量过低导致范围不够
        drag_budget_floor = int(np.clip(96 + 56 * (1.0 - influence_ratio), 96, 152))
        anchor_budget = int(max(anchor_budget, min(int(max_anchors), drag_budget_floor)))

    control_indices = (
        _map_xy_controls_to_indices(ptsxy, handle_points_xy)
        if handle_points_xy is not None
        else np.zeros((0,), dtype=np.int32)
    )
    if control_indices.size > 0:
        control_indices = np.unique(control_indices.astype(np.int32))

    opposite_edge_indices = np.zeros((0,), dtype=np.int32)
    opposite_edge_info = {'enabled': False, 'reason': 'drag_disabled'}
    drag_enabled = bool(direction_info.get('enabled', False))
    if drag_enabled:
        opposite_edge_indices, opposite_edge_info = _select_opposite_edge_anchor_indices(
            points_xy=ptsxy,
            points_z=pts3d[:, 2].astype(np.float32),
            handle_points_xy=handle_points_xy,
            target_points_xy=target_points_xy,
            influence_ratio=influence_ratio,
            max_keep=anchor_budget,
            exclude_indices=control_indices,
            sector_num=18,
        )
        if opposite_edge_indices.size > 0:
            opposite_edge_indices = np.unique(opposite_edge_indices.astype(np.int32))

    # 全局边缘带（用于主补点池）：确保“锚点在边缘且范围足够”
    xi_all = np.round(ptsxy[:, 0]).astype(np.int32)
    yi_all = np.round(ptsxy[:, 1]).astype(np.int32)
    x0_all = int(np.min(xi_all))
    y0_all = int(np.min(yi_all))
    x1_all = int(np.max(xi_all))
    y1_all = int(np.max(yi_all))
    canvas_w_all = max(1, x1_all - x0_all + 3)
    canvas_h_all = max(1, y1_all - y0_all + 3)
    local_x_all = np.clip(xi_all - x0_all + 1, 0, canvas_w_all - 1)
    local_y_all = np.clip(yi_all - y0_all + 1, 0, canvas_h_all - 1)
    occ_all = np.zeros((canvas_h_all, canvas_w_all), dtype=np.uint8)
    occ_all[local_y_all, local_x_all] = 255
    dist_all = cv2.distanceTransform((occ_all > 0).astype(np.uint8), cv2.DIST_L2, 3)
    edge_dist_all = dist_all[local_y_all, local_x_all].astype(np.float32)
    edge_band_main = float(np.clip(2.6 + 2.4 * influence_ratio, 2.4, 5.2))
    edge_mask = edge_dist_all <= edge_band_main
    if int(np.sum(edge_mask)) < max(36, int(round(anchor_budget * 0.82))):
        edge_mask = edge_dist_all <= (edge_band_main * 1.8)
    if int(np.sum(edge_mask)) < max(24, int(round(anchor_budget * 0.55))):
        edge_mask = edge_dist_all <= (edge_band_main * 2.5)

    # 全局评分：边缘+小z为主，方向仅软偏好
    z_all = pts3d[:, 2].astype(np.float32)
    z_pref_all = np.clip(
        (float(np.max(z_all)) - z_all) / (float(np.max(z_all) - np.min(z_all)) + 1e-6),
        0.0,
        1.0,
    )
    h_all_norm = min_handle_dist / (float(np.max(min_handle_dist)) + 1e-6)
    r_all_norm = radial_dist / (float(np.max(radial_dist)) + 1e-6)
    edge_pref_all = 1.0 - np.clip(edge_dist_all / (edge_band_main * 2.8 + 1e-6), 0.0, 1.0)
    global_score_all = (
        0.23 * h_all_norm
        + 0.15 * r_all_norm
        + 0.37 * z_pref_all
        + 0.25 * edge_pref_all
    ).astype(np.float32)
    if drag_enabled:
        global_score_all = np.clip(0.86 * global_score_all + 0.14 * full_direction_score, 0.0, 1.0).astype(np.float32)

    # 第1层：反方向边缘环带“保底”，但只占总预算一部分，避免全局塌缩成一小团
    opposite_keep_indices = np.zeros((0,), dtype=np.int32)
    if drag_enabled and opposite_edge_indices.size > 0:
        reserve_ratio = 0.62 + 0.22 * (1.0 - influence_ratio)
        reserve_n = int(np.clip(round(anchor_budget * reserve_ratio), 24, min(anchor_budget, 128)))
        reserve_n = min(reserve_n, int(opposite_edge_indices.shape[0]))
        if reserve_n > 0:
            opp_xy = ptsxy[opposite_edge_indices]
            opp_score = np.clip(
                0.78 * global_score_all[opposite_edge_indices] + 0.22 * full_direction_score[opposite_edge_indices],
                0.0,
                1.0,
            ).astype(np.float32)
            opp_selected_xy = _sample_anchors_spatially_balanced(opp_xy, opp_score, max_anchors=reserve_n)
            opposite_keep_indices = _map_xy_controls_to_indices(ptsxy, opp_selected_xy)
            if opposite_keep_indices.size > 0:
                opposite_keep_indices = np.unique(opposite_keep_indices.astype(np.int32))
                opposite_keep_indices = opposite_keep_indices[np.isin(opposite_keep_indices, opposite_edge_indices)]

    merged_indices = opposite_keep_indices.copy()
    edge_fill_indices = np.zeros((0,), dtype=np.int32)
    extra_indices = np.zeros((0,), dtype=np.int32)
    final_budget = int(anchor_budget)

    # 拖拽启用时走“角落模式”：只保留反方向角落边缘，不再补整圈外边缘
    if drag_enabled:
        corner_budget = int(
            np.clip(
                round(anchor_budget * (0.50 + 0.18 * (1.0 - influence_ratio))),
                24,
                min(anchor_budget, 92),
            )
        )
        final_budget = int(corner_budget)

        if merged_indices.size == 0 and opposite_edge_indices.size > 0:
            merged_indices = opposite_edge_indices.copy()

        if merged_indices.size < corner_budget:
            dir_thr = float(np.percentile(full_direction_score, 70.0))
            corner_pool = np.where(edge_mask & (full_direction_score >= dir_thr))[0].astype(np.int32)
            if corner_pool.size > 0:
                if control_indices.size > 0:
                    corner_pool = corner_pool[~np.isin(corner_pool, control_indices)]
                if merged_indices.size > 0:
                    corner_pool = corner_pool[~np.isin(corner_pool, merged_indices)]
            if corner_pool.size > 0:
                need_n = int(min(max(0, corner_budget - merged_indices.size), corner_pool.size))
                if need_n > 0:
                    corner_score = np.clip(
                        0.66 * global_score_all[corner_pool] + 0.34 * full_direction_score[corner_pool],
                        0.0,
                        1.0,
                    ).astype(np.float32)
                    corner_xy = _sample_anchors_spatially_balanced(
                        ptsxy[corner_pool],
                        corner_score,
                        max_anchors=need_n,
                    )
                    edge_fill_indices = _map_xy_controls_to_indices(ptsxy, corner_xy)
                    if edge_fill_indices.size > 0:
                        edge_fill_indices = np.unique(edge_fill_indices.astype(np.int32))
                        edge_fill_indices = edge_fill_indices[np.isin(edge_fill_indices, corner_pool)]
                        if merged_indices.size > 0:
                            merged_indices = np.concatenate([merged_indices, edge_fill_indices], axis=0)
                        else:
                            merged_indices = edge_fill_indices.copy()

    else:
        # 第2层：从全局边缘带补齐预算（覆盖范围大，且保证在边缘）
        remain_budget = int(max(0, anchor_budget - merged_indices.shape[0]))
        if remain_budget > 0:
            edge_pool = np.where(edge_mask)[0].astype(np.int32)
            if edge_pool.size > 0:
                if control_indices.size > 0:
                    edge_pool = edge_pool[~np.isin(edge_pool, control_indices)]
                if merged_indices.size > 0:
                    edge_pool = edge_pool[~np.isin(edge_pool, merged_indices)]
            if edge_pool.size > 0:
                edge_fill_xy = _sample_anchors_spatially_balanced(
                    ptsxy[edge_pool],
                    global_score_all[edge_pool],
                    max_anchors=remain_budget,
                )
                edge_fill_indices = _map_xy_controls_to_indices(ptsxy, edge_fill_xy)
                if edge_fill_indices.size > 0:
                    edge_fill_indices = np.unique(edge_fill_indices.astype(np.int32))
                    edge_fill_indices = edge_fill_indices[np.isin(edge_fill_indices, edge_pool)]

        if merged_indices.size > 0 and edge_fill_indices.size > 0:
            merged_indices = np.concatenate([merged_indices, edge_fill_indices], axis=0)
        elif merged_indices.size == 0 and edge_fill_indices.size > 0:
            merged_indices = edge_fill_indices.copy()

        # 第3层：若仍不足，从外层环补点（仍然偏好边缘/小z）
        remain_budget_2 = int(max(0, anchor_budget - merged_indices.shape[0]))
        if remain_budget_2 > 0:
            radial_thr_outer = float(np.percentile(radial_dist, 56.0))
            extra_pool = np.where(radial_dist >= radial_thr_outer)[0].astype(np.int32)
            if extra_pool.size > 0:
                if control_indices.size > 0:
                    extra_pool = extra_pool[~np.isin(extra_pool, control_indices)]
                if merged_indices.size > 0:
                    extra_pool = extra_pool[~np.isin(extra_pool, merged_indices)]
            if extra_pool.size > 0:
                extra_xy = _sample_anchors_spatially_balanced(
                    ptsxy[extra_pool],
                    global_score_all[extra_pool],
                    max_anchors=remain_budget_2,
                )
                extra_indices = _map_xy_controls_to_indices(ptsxy, extra_xy)
                if extra_indices.size > 0:
                    extra_indices = np.unique(extra_indices.astype(np.int32))
                    extra_indices = extra_indices[np.isin(extra_indices, extra_pool)]

        if merged_indices.size > 0 and extra_indices.size > 0:
            merged_indices = np.concatenate([merged_indices, extra_indices], axis=0)
        elif merged_indices.size == 0 and extra_indices.size > 0:
            merged_indices = extra_indices.copy()

        # 数量下限保护（仅非角落模式）
        floor_n = max(24, int(round(anchor_budget * 0.45)))
        if merged_indices.size < floor_n:
            rescue_pool = np.where(edge_mask)[0].astype(np.int32)
            if rescue_pool.size == 0:
                rescue_pool = np.arange(ptsxy.shape[0], dtype=np.int32)
            if control_indices.size > 0:
                rescue_pool = rescue_pool[~np.isin(rescue_pool, control_indices)]
            if merged_indices.size > 0:
                rescue_pool = rescue_pool[~np.isin(rescue_pool, merged_indices)]
            rescue_need = int(min(max(0, floor_n - merged_indices.size), rescue_pool.size))
            if rescue_need > 0:
                order = np.argsort(global_score_all[rescue_pool])[::-1][:rescue_need]
                rescue_indices = rescue_pool[order].astype(np.int32)
                merged_indices = np.concatenate([merged_indices, rescue_indices], axis=0) if merged_indices.size > 0 else rescue_indices

    if merged_indices.size == 0:
        merged_indices = base_indices.copy()

    if merged_indices.size > 0:
        merged_indices = np.unique(merged_indices.astype(np.int32))
        if control_indices.size > 0:
            merged_indices = merged_indices[~np.isin(merged_indices, control_indices)]
        merged_indices = merged_indices[(merged_indices >= 0) & (merged_indices < ptsxy.shape[0])]

    # 最终预算裁剪：优先保留全局高分 + 空间均匀
    if merged_indices.size > final_budget:
        final_xy = _sample_anchors_spatially_balanced(
            ptsxy[merged_indices],
            global_score_all[merged_indices],
            max_anchors=final_budget,
        )
        final_idx = _map_xy_controls_to_indices(ptsxy, final_xy)
        if final_idx.size > 0:
            final_idx = np.unique(final_idx.astype(np.int32))
            final_idx = final_idx[np.isin(final_idx, merged_indices)]
            merged_indices = final_idx

    anchors = ptsxy[merged_indices].astype(np.float32) if merged_indices.size > 0 else np.zeros((0, 2), dtype=np.float32)

    strategy = 'ROTATION_INVARIANT_OPPOSITE_CORNER_Z_SMALL' if drag_enabled else 'ROTATION_INVARIANT_CENTROID_HANDLE'
    strategy_name = 'OppositeCorner+SmallZ+HandleInvariant' if drag_enabled else 'Centroid+HandleInvariant'
    info = {
        'strategy': strategy,
        'strategy_name': strategy_name,
        'candidate_count': int(cand_xy.shape[0]),
        'anchor_count': int(anchors.shape[0]),
        'anchor_budget': int(anchor_budget),
        'max_anchors': int(max_anchors),
        'q_handle': float(q_handle),
        'q_radial': float(q_radial),
        'thr_handle': float(h_thr),
        'thr_radial': float(r_thr),
        'score_weights': {
            'handle_dist': 0.23,
            'radial_dist': 0.15,
            'small_z_prior': 0.37,
            'edge_prior': 0.25,
            'drag_direction_soft': 0.14 if drag_enabled else 0.0,
        },
        'drag_direction_enabled': bool(drag_enabled),
        'drag_direction_info': direction_info,
        'opposite_edge_ring_info': opposite_edge_info,
        'opposite_edge_ring_count': int(opposite_edge_indices.shape[0]),
        'opposite_edge_reserved_count': int(opposite_keep_indices.shape[0]),
        'opposite_edge_ring_xy': (
            ptsxy[opposite_edge_indices].astype(np.float32).tolist()
            if opposite_edge_indices.shape[0] > 0
            else []
        ),
        'anchor_budget_effective': int(final_budget),
        'corner_only_mode': bool(drag_enabled),
        'corner_budget': int(final_budget),
        'edge_band_main_px': float(edge_band_main),
        'edge_pool_count': int(np.sum(edge_mask)),
        'edge_fill_count': int(edge_fill_indices.shape[0]),
        'extra_anchor_count': int(extra_indices.shape[0]),
        'base_fallback_count': int(base_indices.shape[0]),
    }
    return anchors, info


def _estimate_hybrid_phase2_front_axis_3d(
    points_3d,
    control_points_3d,
    control_points_xy=None,
    target_points_xy=None,
):
    """
    估计 Hybrid Phase2 的“前向轴”：
    - 主来源：旋转后控制点相对主体质心方向（通常对应被拖拽的前脸/鼻部）
    - 辅助来源：2D 拖拽平均方向映射到 3D XY 平面
    """
    pts3d = np.asarray(points_3d, dtype=np.float32).reshape(-1, 3)
    ctrls3d = np.asarray(control_points_3d, dtype=np.float32).reshape(-1, 3)
    info = {
        'enabled': False,
        'reason': 'init',
    }
    if pts3d.shape[0] == 0:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        info.update({'reason': 'empty_points'})
        return axis, info

    centroid3d = np.mean(pts3d, axis=0).astype(np.float32)
    vec_ctrl = np.zeros((3,), dtype=np.float32)
    if ctrls3d.shape[0] > 0:
        vec_ctrl = (np.mean(ctrls3d, axis=0) - centroid3d).astype(np.float32)

    vec_drag = np.zeros((3,), dtype=np.float32)
    handles = _sanitize_xy_points(control_points_xy)
    targets = _sanitize_xy_points(target_points_xy)
    pair_n = int(min(handles.shape[0], targets.shape[0]))
    if pair_n > 0:
        disp = targets[:pair_n] - handles[:pair_n]
        motion = np.linalg.norm(disp, axis=1)
        active = motion > 1.0
        if np.any(active):
            disp_mean = np.mean(disp[active], axis=0).astype(np.float32)
            vec_drag = np.array([disp_mean[0], -disp_mean[1], 0.0], dtype=np.float32)

    n_ctrl = float(np.linalg.norm(vec_ctrl))
    n_drag = float(np.linalg.norm(vec_drag))
    if n_ctrl < 1e-6 and n_drag < 1e-6:
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        reason = 'fallback_default_x'
    elif n_ctrl < 1e-6:
        vec = vec_drag
        reason = 'use_drag_only'
    elif n_drag < 1e-6:
        vec = vec_ctrl
        reason = 'use_control_only'
    else:
        vec = (0.82 * vec_ctrl + 0.18 * vec_drag).astype(np.float32)
        reason = 'blend_control_drag'

    n_vec = float(np.linalg.norm(vec))
    if n_vec < 1e-6:
        vec = vec_ctrl if n_ctrl >= n_drag else vec_drag
        n_vec = float(np.linalg.norm(vec))
    if n_vec < 1e-6:
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        n_vec = 1.0
        reason = 'fallback_after_blend'

    axis = (vec / (n_vec + 1e-6)).astype(np.float32)
    info.update({
        'enabled': True,
        'reason': reason,
        'centroid_3d': centroid3d.tolist(),
        'vec_control': vec_ctrl.tolist(),
        'vec_drag': vec_drag.tolist(),
        'norm_control': float(n_ctrl),
        'norm_drag': float(n_drag),
        'front_axis_3d': axis.tolist(),
    })
    return axis, info


def _select_phase2_orientation_opposite_anchors_3d(
    points_3d,
    points_xy,
    handle_points_3d,
    front_axis_3d,
    influence_ratio=0.5,
    max_anchors=220,
    exclude_indices=None,
    sector_num=18,
):
    """
    Hybrid Phase2 专用锚点：
    - 仅从边缘带选点
    - 按“朝向反侧”打分（不使用 z 值先验）
    - 保持空间均匀覆盖，避免集中成一小坨
    """
    pts3d = np.asarray(points_3d, dtype=np.float32).reshape(-1, 3)
    ptsxy = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    handles3d = np.asarray(handle_points_3d, dtype=np.float32).reshape(-1, 3)
    info = {
        'strategy': 'PHASE2_ORIENTATION_OPPOSITE_EDGE',
        'strategy_name': 'Phase2OrientationOppositeEdge',
        'enabled': False,
        'reason': 'init',
    }
    if pts3d.shape[0] == 0 or ptsxy.shape[0] == 0:
        info['reason'] = 'empty_points'
        return np.zeros((0, 2), dtype=np.float32), info
    if pts3d.shape[0] != ptsxy.shape[0]:
        n = int(min(pts3d.shape[0], ptsxy.shape[0]))
        pts3d = pts3d[:n]
        ptsxy = ptsxy[:n]
    if handles3d.shape[0] == 0:
        handles3d = np.zeros((1, 3), dtype=np.float32)

    axis = np.asarray(front_axis_3d, dtype=np.float32).reshape(-1)
    if axis.shape[0] < 3:
        info['reason'] = 'invalid_axis'
        return np.zeros((0, 2), dtype=np.float32), info
    axis = axis[:3]
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-6:
        info['reason'] = 'zero_axis'
        return np.zeros((0, 2), dtype=np.float32), info
    axis = (axis / (axis_norm + 1e-6)).astype(np.float32)

    n_all = ptsxy.shape[0]
    valid_mask = np.isfinite(ptsxy).all(axis=1) & np.isfinite(pts3d).all(axis=1)
    valid_idx = np.where(valid_mask)[0].astype(np.int32)
    if valid_idx.size == 0:
        info['reason'] = 'non_finite_points'
        return np.zeros((0, 2), dtype=np.float32), info

    if exclude_indices is not None:
        ex = np.asarray(exclude_indices, dtype=np.int32).reshape(-1)
        ex = ex[(ex >= 0) & (ex < n_all)]
        if ex.size > 0:
            keep_mask = ~np.isin(valid_idx, np.unique(ex))
            valid_idx = valid_idx[keep_mask]
    if valid_idx.size == 0:
        info['reason'] = 'all_excluded'
        return np.zeros((0, 2), dtype=np.float32), info

    p3 = pts3d[valid_idx]
    p2 = ptsxy[valid_idx]

    diff_h = p3[:, None, :] - handles3d[None, :, :]
    min_handle_dist = np.min(np.linalg.norm(diff_h, axis=2), axis=1)

    centroid3d = np.mean(p3, axis=0).astype(np.float32)
    rel3d = p3 - centroid3d[None, :]
    rel3d_norm = np.linalg.norm(rel3d, axis=1) + 1e-6
    radial3d = rel3d_norm.astype(np.float32)
    cos_opp = -np.sum(rel3d * axis[None, :], axis=1) / rel3d_norm
    orient_score = np.clip((cos_opp + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)

    xi = np.round(p2[:, 0]).astype(np.int32)
    yi = np.round(p2[:, 1]).astype(np.int32)
    x0 = int(np.min(xi))
    y0 = int(np.min(yi))
    x1 = int(np.max(xi))
    y1 = int(np.max(yi))
    w = max(1, x1 - x0 + 3)
    h = max(1, y1 - y0 + 3)
    lx = np.clip(xi - x0 + 1, 0, w - 1)
    ly = np.clip(yi - y0 + 1, 0, h - 1)
    occ = np.zeros((h, w), dtype=np.uint8)
    occ[ly, lx] = 255
    dist_in = cv2.distanceTransform((occ > 0).astype(np.uint8), cv2.DIST_L2, 3)
    edge_dist = dist_in[ly, lx].astype(np.float32)
    influence_ratio = float(np.clip(influence_ratio, 0.0, 1.0))
    edge_band = float(np.clip(2.6 + 2.4 * influence_ratio, 2.4, 5.4))
    edge_mask = edge_dist <= edge_band
    if int(np.sum(edge_mask)) < 24:
        edge_mask = edge_dist <= (edge_band * 1.8)
    if int(np.sum(edge_mask)) < 16:
        edge_mask = np.ones_like(edge_mask, dtype=bool)

    cand_local = np.where(edge_mask)[0].astype(np.int32)
    if cand_local.size == 0:
        info['reason'] = 'empty_edge_pool'
        return np.zeros((0, 2), dtype=np.float32), info

    q_opp = float(np.clip(56.0 + 18.0 * influence_ratio, 52.0, 82.0))
    opp_thr = float(np.percentile(orient_score[cand_local], q_opp))
    opp_local = cand_local[orient_score[cand_local] >= opp_thr]
    min_opp = max(12, int(round(float(max_anchors) * 0.12)))
    if opp_local.size < min_opp:
        opp_thr = float(np.percentile(orient_score[cand_local], 50.0))
        opp_local = cand_local[orient_score[cand_local] >= opp_thr]
    if opp_local.size == 0:
        opp_local = cand_local

    h_norm = min_handle_dist / (float(np.max(min_handle_dist)) + 1e-6)
    r_norm = radial3d / (float(np.max(radial3d)) + 1e-6)
    score = (
        0.60 * orient_score
        + 0.23 * h_norm.astype(np.float32)
        + 0.17 * r_norm.astype(np.float32)
    ).astype(np.float32)

    budget_scale = 0.24 + 0.76 * (1.0 - influence_ratio)
    budget = int(np.clip(round(float(max_anchors) * budget_scale), 24, int(max_anchors)))
    budget = min(budget, int(valid_idx.shape[0]))
    if budget <= 0:
        info['reason'] = 'zero_budget'
        return np.zeros((0, 2), dtype=np.float32), info

    selected_xy = _sample_anchors_spatially_balanced(
        p2[opp_local],
        score[opp_local],
        max_anchors=min(int(opp_local.size), budget),
    )
    selected_local = _map_xy_controls_to_indices(p2, selected_xy)
    if selected_local.size > 0:
        selected_local = np.unique(selected_local.astype(np.int32))

    if selected_local.size < budget:
        fill_need = int(min(max(0, budget - selected_local.size), cand_local.size))
        if fill_need > 0:
            fill_pool = cand_local
            if selected_local.size > 0:
                fill_pool = fill_pool[~np.isin(fill_pool, selected_local)]
            if fill_pool.size > 0:
                fill_xy = _sample_anchors_spatially_balanced(
                    p2[fill_pool],
                    score[fill_pool],
                    max_anchors=min(int(fill_pool.size), fill_need),
                )
                fill_local = _map_xy_controls_to_indices(p2, fill_xy)
                if fill_local.size > 0:
                    fill_local = np.unique(fill_local.astype(np.int32))
                    selected_local = (
                        np.concatenate([selected_local, fill_local], axis=0)
                        if selected_local.size > 0
                        else fill_local
                    )

    if selected_local.size > 0:
        selected_local = np.unique(selected_local.astype(np.int32))
        selected_local = selected_local[(selected_local >= 0) & (selected_local < p2.shape[0])]
    if selected_local.size == 0:
        order = np.argsort(score[cand_local])[::-1][:budget]
        selected_local = cand_local[order].astype(np.int32)

    if selected_local.size > budget:
        keep_xy = _sample_anchors_spatially_balanced(
            p2[selected_local],
            score[selected_local],
            max_anchors=budget,
        )
        keep_local = _map_xy_controls_to_indices(p2, keep_xy)
        if keep_local.size > 0:
            keep_local = np.unique(keep_local.astype(np.int32))
            keep_local = keep_local[np.isin(keep_local, selected_local)]
            selected_local = keep_local

    final_idx = valid_idx[selected_local] if selected_local.size > 0 else np.zeros((0,), dtype=np.int32)
    anchors = ptsxy[final_idx].astype(np.float32) if final_idx.size > 0 else np.zeros((0, 2), dtype=np.float32)
    info.update({
        'enabled': True,
        'reason': 'ok',
        'front_axis_3d': axis.tolist(),
        'candidate_count': int(cand_local.size),
        'opposite_pool_count': int(opp_local.size),
        'anchor_budget': int(budget),
        'anchor_count': int(anchors.shape[0]),
        'opposite_orientation_threshold': float(opp_thr),
        'edge_band_px': float(edge_band),
        'sector_num': int(np.clip(sector_num, 12, 48)),
    })
    return anchors, info


def _select_3d_nonrigid_anchor_points(
    points_3d,
    points_xy,
    handle_points_3d,
    handle_points_xy,
    target_points_xy,
    influence_ratio=0.5,
    max_anchors=220,
    anchor_strategy_3d=DEFAULT_THREE_D_ANCHOR_STRATEGY,
):
    requested_strategy = _normalize_3d_anchor_strategy(anchor_strategy_3d)
    # 统一禁用 Non-Rigid 的对侧选点逻辑，固定为 balanced_edge。
    _ = handle_points_xy, target_points_xy
    anchors_img, anchor_select_info = _select_rotation_invariant_anchors_3d(
        points_3d=points_3d,
        points_xy=points_xy,
        handle_points_3d=handle_points_3d,
        influence_ratio=influence_ratio,
        max_anchors=max_anchors,
        handle_points_xy=None,
        target_points_xy=None,
    )
    anchor_select_info = dict(anchor_select_info)
    anchor_select_info["requested_anchor_strategy_3d"] = requested_strategy
    anchor_select_info["effective_anchor_strategy_3d"] = "balanced_edge"
    anchor_select_info["opposite_anchor_bias_enabled"] = False
    anchor_select_info["strategy_override"] = "balanced_edge_without_drag_opposition"
    if requested_strategy not in {"auto", "balanced_edge"}:
        anchor_select_info["requested_strategy_fallback_reason"] = (
            "opposite_anchor_selection_disabled"
        )
    return anchors_img, anchor_select_info


def _select_3d_hybrid_phase2_anchor_points(
    points_3d,
    points_xy,
    handle_points_3d,
    handle_points_xy,
    target_points_xy,
    front_axis_3d,
    influence_ratio=0.5,
    max_anchors=220,
    exclude_indices=None,
    anchor_strategy_3d=DEFAULT_THREE_D_ANCHOR_STRATEGY,
):
    requested_strategy = _normalize_3d_anchor_strategy(anchor_strategy_3d)
    # 统一禁用 Hybrid Phase2 的“对侧选点”逻辑，固定为 balanced_edge。
    # 这样避免前向轴估计/拖拽方向偏置带来的锚点不稳定。
    _ = handle_points_xy, target_points_xy, front_axis_3d, exclude_indices
    anchors_img, anchor_select_info = _select_rotation_invariant_anchors_3d(
        points_3d=points_3d,
        points_xy=points_xy,
        handle_points_3d=handle_points_3d,
        influence_ratio=influence_ratio,
        max_anchors=max_anchors,
        handle_points_xy=None,
        target_points_xy=None,
    )
    anchor_select_info = dict(anchor_select_info)
    anchor_select_info["requested_anchor_strategy_3d"] = requested_strategy
    anchor_select_info["effective_anchor_strategy_3d"] = "balanced_edge"
    anchor_select_info["phase2_opposite_anchor_bias_enabled"] = False
    anchor_select_info["strategy_override"] = "balanced_edge_without_drag_opposition"
    if requested_strategy != "balanced_edge":
        anchor_select_info["requested_strategy_fallback_reason"] = (
            "phase2_opposite_anchor_selection_disabled"
        )
    return anchors_img, anchor_select_info


def _enforce_control_targets_on_points_3d(
    pts3d_original,
    pts3d_deformed,
    points_xy,
    handle_points_xy,
    target_points_xy,
    target_points_3d,
    control_indices=None,
    pair_active_mask=None,
    radius_px=10.0,
    min_neighbors=16,
    max_neighbors=96,
    falloff_px=6.0,
    blend_strength=1.0,
    rigid_core_ratio=0.45,
    rigid_core_min_px=3.0,
    global_follow_ratio=2.6,
    global_follow_gain=0.0,
):
    """
    控制点强制到位（局部簇）：
    - 控制点中心点严格落到目标3D位置
    - 控制点邻域点按高斯权重向目标位移对齐，避免“只动一个点”
    """
    pts0 = np.asarray(pts3d_original, dtype=np.float32).reshape(-1, 3)
    pts1 = np.asarray(pts3d_deformed, dtype=np.float32).reshape(-1, 3)
    xy = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    handles = _sanitize_xy_points(handle_points_xy)
    targets = _sanitize_xy_points(target_points_xy)
    tgt3d = np.asarray(target_points_3d, dtype=np.float32).reshape(-1, 3)

    n = int(min(pts0.shape[0], pts1.shape[0], xy.shape[0]))
    info = {
        'enabled': False,
        'reason': 'init',
        'pair_count': 0,
        'active_pair_count': 0,
        'mean_handle_to_index_px': 0.0,
        'max_handle_to_index_px': 0.0,
        'mean_cluster_size': 0.0,
        'mean_core_cluster_size': 0.0,
        'mean_global_cluster_size': 0.0,
    }
    if n <= 0:
        info['reason'] = 'empty_points'
        return pts1, info
    if pts0.shape[0] != n:
        pts0 = pts0[:n]
    if pts1.shape[0] != n:
        pts1 = pts1[:n]
    if xy.shape[0] != n:
        xy = xy[:n]

    pair_n = int(min(handles.shape[0], targets.shape[0], tgt3d.shape[0]))
    info['pair_count'] = pair_n
    if pair_n <= 0:
        info['reason'] = 'empty_pairs'
        return pts1, info

    if control_indices is None:
        ctrl_idx_hint = np.full((pair_n,), -1, dtype=np.int32)
    else:
        ctrl_idx_raw = np.asarray(control_indices, dtype=np.int32).reshape(-1)
        ctrl_idx_hint = np.full((pair_n,), -1, dtype=np.int32)
        take_n = min(pair_n, ctrl_idx_raw.shape[0])
        if take_n > 0:
            ctrl_idx_hint[:take_n] = ctrl_idx_raw[:take_n]

    if pair_active_mask is None:
        active_mask = np.ones((pair_n,), dtype=bool)
    else:
        m = np.asarray(pair_active_mask, dtype=bool).reshape(-1)
        active_mask = np.ones((pair_n,), dtype=bool)
        take_n = min(pair_n, m.shape[0])
        if take_n > 0:
            active_mask[:take_n] = m[:take_n]

    radius_px = float(max(radius_px, 2.0))
    falloff_px = float(max(falloff_px, 1.0))
    min_neighbors = int(max(1, min_neighbors))
    max_neighbors = int(max(min_neighbors, max_neighbors))
    blend_strength = float(np.clip(blend_strength, 0.0, 1.5))
    rigid_core_ratio = float(np.clip(rigid_core_ratio, 0.0, 1.0))
    rigid_core_min_px = float(max(rigid_core_min_px, 0.0))
    global_follow_ratio = float(np.clip(global_follow_ratio, 1.0, 6.0))
    global_follow_gain = float(np.clip(global_follow_gain, 0.0, 1.0))

    handle_to_idx = []
    cluster_sizes = []
    core_cluster_sizes = []
    global_cluster_sizes = []
    active_count = 0

    for i in range(pair_n):
        if not bool(active_mask[i]):
            continue
        active_count += 1
        hxy = handles[i]
        t3 = tgt3d[i]

        center_idx = int(ctrl_idx_hint[i])
        d2 = (xy[:, 0] - float(hxy[0])) ** 2 + (xy[:, 1] - float(hxy[1])) ** 2
        if d2.size == 0:
            continue
        nearest_idx = int(np.argmin(d2))
        if center_idx < 0 or center_idx >= n:
            center_idx = nearest_idx
        else:
            hint_dist = float(np.sqrt(float(d2[center_idx])))
            # hint 与真实 handle 偏差过大时，强制回退到最近点，避免锁错位置
            if hint_dist > max(radius_px * 1.8, 12.0):
                center_idx = nearest_idx

        handle_to_idx.append(float(np.sqrt(float(d2[center_idx]))))

        neighbor_idx = np.where(d2 <= (radius_px * radius_px))[0].astype(np.int32)
        if neighbor_idx.size < min_neighbors:
            k = int(min(max_neighbors, n))
            if k > 0:
                if k >= n:
                    neighbor_idx = np.arange(n, dtype=np.int32)
                else:
                    neighbor_idx = np.argpartition(d2, k - 1)[:k].astype(np.int32)
        if neighbor_idx.size == 0:
            pts1[center_idx] = t3
            cluster_sizes.append(1)
            core_cluster_sizes.append(1)
            global_cluster_sizes.append(1)
            continue

        # 可选：把控制点残差传播到更大范围（默认关闭，避免局部组织被过度拉扯）。
        shift_global = t3 - pts1[center_idx]
        global_radius_px = max(radius_px * global_follow_ratio, radius_px + 2.0)
        global_idx = np.where(d2 <= (global_radius_px * global_radius_px))[0].astype(np.int32)
        if global_idx.size < min_neighbors:
            k_global = int(min(max_neighbors * 2, n))
            if k_global > 0:
                if k_global >= n:
                    global_idx = np.arange(n, dtype=np.int32)
                else:
                    global_idx = np.argpartition(d2, k_global - 1)[:k_global].astype(np.int32)
        if global_idx.size > 0 and global_follow_gain > 1e-6:
            sigma2_global = float((global_radius_px * 0.6) ** 2)
            wg = np.exp(-d2[global_idx] / (2.0 * sigma2_global + 1e-6)).astype(np.float32)
            wg = np.clip(wg * global_follow_gain, 0.0, 1.0)[:, None]
            pts1[global_idx] = pts1[global_idx] + shift_global[None, :] * wg
            global_cluster_sizes.append(int(global_idx.size))
        else:
            global_cluster_sizes.append(0)

        sigma2 = float(falloff_px * falloff_px)
        w = np.exp(-d2[neighbor_idx] / (2.0 * sigma2 + 1e-6)).astype(np.float32)
        w = np.clip(w, 0.0, 1.0)

        # 用原始点位定义“目标位移簇”，避免多轮叠加带来形变破坏。
        shift = t3 - pts0[center_idx]
        cluster_target = pts0[neighbor_idx] + shift[None, :]
        alpha = (blend_strength * w)[:, None].astype(np.float32)
        alpha = np.clip(alpha, 0.0, 1.0)

        # 控制点邻域核心区使用硬约束，保证关键部位（如鼻子）整体刚性跟随目标点。
        core_radius_px = max(rigid_core_min_px, radius_px * rigid_core_ratio)
        core_mask = d2[neighbor_idx] <= (core_radius_px * core_radius_px)
        if np.any(core_mask):
            alpha[core_mask] = np.maximum(alpha[core_mask], 0.85)
            core_cluster_sizes.append(int(np.sum(core_mask)))
        else:
            core_cluster_sizes.append(0)

        pts1[neighbor_idx] = pts1[neighbor_idx] * (1.0 - alpha) + cluster_target * alpha

        # 中心点严格到目标
        pts1[center_idx] = t3
        cluster_sizes.append(int(neighbor_idx.size))

    info.update({
        'enabled': True,
        'reason': 'ok',
        'active_pair_count': int(active_count),
        'mean_handle_to_index_px': float(np.mean(handle_to_idx)) if len(handle_to_idx) > 0 else 0.0,
        'max_handle_to_index_px': float(np.max(handle_to_idx)) if len(handle_to_idx) > 0 else 0.0,
        'mean_cluster_size': float(np.mean(cluster_sizes)) if len(cluster_sizes) > 0 else 0.0,
        'mean_core_cluster_size': float(np.mean(core_cluster_sizes)) if len(core_cluster_sizes) > 0 else 0.0,
        'mean_global_cluster_size': float(np.mean(global_cluster_sizes)) if len(global_cluster_sizes) > 0 else 0.0,
        'radius_px': float(radius_px),
        'falloff_px': float(falloff_px),
        'blend_strength': float(blend_strength),
        'rigid_core_ratio': float(rigid_core_ratio),
        'rigid_core_min_px': float(rigid_core_min_px),
        'global_follow_ratio': float(global_follow_ratio),
        'global_follow_gain': float(global_follow_gain),
    })
    return pts1.astype(np.float32), info


def _safe_unit_vec(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < float(eps):
        return None
    return v / n


def _rotation_matrix_from_vec_to_vec(v_from, v_to, eps=1e-8):
    """
    计算将 v_from 最小旋转到 v_to 的 3x3 旋转矩阵（Rodrigues）。
    """
    a = _safe_unit_vec(v_from, eps=eps)
    b = _safe_unit_vec(v_to, eps=eps)
    if a is None or b is None:
        return np.eye(3, dtype=np.float64), {
            'ok': False,
            'reason': 'zero_vector',
            'angle_deg': 0.0,
            'dot': 1.0,
        }

    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    angle_rad = float(np.arccos(c))
    angle_deg = float(np.degrees(angle_rad))

    if c > 1.0 - eps:
        return np.eye(3, dtype=np.float64), {
            'ok': True,
            'reason': 'already_aligned',
            'angle_deg': angle_deg,
            'dot': c,
        }

    if c < -1.0 + eps:
        basis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(a, basis))) > 0.9:
            basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(a, basis)
        axis_u = _safe_unit_vec(axis, eps=eps)
        if axis_u is None:
            axis_u = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        R = -np.eye(3, dtype=np.float64) + 2.0 * np.outer(axis_u, axis_u)
        return R, {
            'ok': True,
            'reason': 'opposite_direction',
            'angle_deg': angle_deg,
            'dot': c,
            'axis': axis_u.tolist(),
        }

    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )
    R = np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - c) / (s * s + eps))
    return R, {
        'ok': True,
        'reason': 'rodrigues',
        'angle_deg': angle_deg,
        'dot': c,
        'axis': (_safe_unit_vec(v, eps=eps) if s > eps else np.zeros(3)).tolist(),
    }


def _fit_rigid_transform_kabsch_3d(src_pts, tgt_pts, weights=None, eps=1e-8):
    """
    在 3D 上拟合刚体变换（R, t），最小化 ||R*src + t - tgt||。
    返回:
      - R: (3,3)
      - t: (3,)
      - info: 诊断信息
    """
    src = np.asarray(src_pts, dtype=np.float64).reshape(-1, 3)
    tgt = np.asarray(tgt_pts, dtype=np.float64).reshape(-1, 3)
    n = int(min(src.shape[0], tgt.shape[0]))
    if n <= 0:
        return np.eye(3, dtype=np.float64), np.zeros((3,), dtype=np.float64), {
            'ok': False,
            'reason': 'empty_points',
            'pair_count': 0,
        }
    src = src[:n]
    tgt = tgt[:n]

    finite = np.isfinite(src).all(axis=1) & np.isfinite(tgt).all(axis=1)
    if int(np.sum(finite)) <= 0:
        return np.eye(3, dtype=np.float64), np.zeros((3,), dtype=np.float64), {
            'ok': False,
            'reason': 'non_finite_points',
            'pair_count': int(n),
        }
    src = src[finite]
    tgt = tgt[finite]
    n = int(src.shape[0])
    if n <= 0:
        return np.eye(3, dtype=np.float64), np.zeros((3,), dtype=np.float64), {
            'ok': False,
            'reason': 'empty_after_finite_filter',
            'pair_count': 0,
        }

    if weights is None:
        w = np.ones((n,), dtype=np.float64)
    else:
        w_raw = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w_raw.shape[0] < n:
            pad = np.ones((n - w_raw.shape[0],), dtype=np.float64)
            w = np.concatenate([w_raw, pad], axis=0)
        else:
            w = w_raw[:n]
        w = np.where(np.isfinite(w), w, 0.0)
        w = np.clip(w, 0.0, None)
        if float(np.sum(w)) <= eps:
            w = np.ones((n,), dtype=np.float64)
    w = w / (float(np.sum(w)) + eps)

    src_mu = np.sum(src * w[:, None], axis=0)
    tgt_mu = np.sum(tgt * w[:, None], axis=0)
    src_c = src - src_mu[None, :]
    tgt_c = tgt - tgt_mu[None, :]

    if n == 1:
        R = np.eye(3, dtype=np.float64)
        t = (tgt_mu - src_mu).astype(np.float64)
        return R, t, {
            'ok': True,
            'reason': 'single_point_translation_only',
            'pair_count': int(n),
            'det_R': 1.0,
            'rmse': float(np.linalg.norm((src + t[None, :]) - tgt)),
        }

    H = (src_c * w[:, None]).T @ tgt_c
    try:
        U, S, Vt = np.linalg.svd(H, full_matrices=True)
    except Exception:
        return np.eye(3, dtype=np.float64), np.zeros((3,), dtype=np.float64), {
            'ok': False,
            'reason': 'svd_failed',
            'pair_count': int(n),
        }
    R = Vt.T @ U.T
    det_R = float(np.linalg.det(R))
    reflection_fixed = False
    if det_R < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
        det_R = float(np.linalg.det(R))
        reflection_fixed = True
    t = (tgt_mu - (R @ src_mu)).astype(np.float64)

    src_warp = (R @ src.T).T + t[None, :]
    rmse = float(np.sqrt(np.mean(np.sum((src_warp - tgt) ** 2, axis=1))))
    return R, t, {
        'ok': bool(np.isfinite(R).all() and np.isfinite(t).all()),
        'reason': 'kabsch_weighted',
        'pair_count': int(n),
        'det_R': float(det_R),
        'reflection_fixed': bool(reflection_fixed),
        'singular_values': np.asarray(S, dtype=np.float64).tolist(),
        'rmse': float(rmse),
    }


def _matrix_to_yaw_pitch_deg(R):
    """
    将旋转矩阵近似分解为本项目 yaw/pitch（仅用于日志/标题）。
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    yaw = np.degrees(np.arctan2(R[0, 2], R[0, 0]))
    pitch = np.degrees(np.arcsin(np.clip(R[2, 1], -1.0, 1.0)))
    return float(yaw), float(pitch)


def _map_xy_controls_to_indices(points_xy, control_xy):
    """
    将控制点(2D)映射到点云数组索引。返回 int32 索引数组（可重复）。
    """
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    ctrls = np.asarray(control_xy, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] == 0 or ctrls.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)

    idx_list = []
    for c in ctrls:
        d2 = (pts[:, 0] - float(c[0])) ** 2 + (pts[:, 1] - float(c[1])) ** 2
        if d2.size == 0:
            continue
        idx_list.append(int(np.argmin(d2)))
    if len(idx_list) == 0:
        return np.zeros((0,), dtype=np.int32)
    return np.asarray(idx_list, dtype=np.int32)


def _estimate_arc_axis_z_centroid(
    xs,
    ys,
    depth_values,
    cx,
    cy,
    depth_min,
    depth_range,
    z_scale,
    min_points=96,
):
    """
    估计更接近“物体轴心/圆心”的 Z 中心（用于 3D 坐标系居中）。
    思路：
    - 将可见表面深度映射到 z_surface in [0, z_scale]
    - 在 (r^2, z) 上做稳健一元二次近似：z ~= a * r^2 + b
    - 对圆弧近似下，轴心深度约为 z_center ~= b + 1/(2a)
    """
    z_mid = float(z_scale / 2.0)
    info = {
        'mode': 'mid_range_fallback',
        'enabled': False,
        'reason': 'init',
        'z_centroid_mid': float(z_mid),
        'z_centroid': float(z_mid),
    }

    xs = np.asarray(xs, dtype=np.float64).reshape(-1)
    ys = np.asarray(ys, dtype=np.float64).reshape(-1)
    depth_vals = np.asarray(depth_values, dtype=np.float64).reshape(-1)
    n = int(min(xs.shape[0], ys.shape[0], depth_vals.shape[0]))
    if n < max(24, int(min_points)):
        info['reason'] = 'too_few_points'
        return z_mid, info
    if depth_range <= 1e-6 or z_scale <= 1e-6:
        info['reason'] = 'invalid_depth_range_or_scale'
        return z_mid, info

    xs = xs[:n]
    ys = ys[:n]
    depth_vals = depth_vals[:n]
    z_surface = (depth_vals - float(depth_min)) / float(depth_range) * float(z_scale)
    rel_x = xs - float(cx)
    rel_y = ys - float(cy)
    r2 = rel_x * rel_x + rel_y * rel_y

    finite = np.isfinite(r2) & np.isfinite(z_surface)
    if int(np.sum(finite)) < max(24, int(min_points)):
        info['reason'] = 'non_finite_points'
        return z_mid, info
    r2 = r2[finite]
    z_surface = z_surface[finite]

    r2_q98 = float(np.percentile(r2, 98.0))
    if r2_q98 <= 1e-6:
        info['reason'] = 'degenerate_radius_distribution'
        return z_mid, info

    inlier = r2 <= r2_q98
    if int(np.sum(inlier)) < 24:
        info['reason'] = 'too_few_inliers'
        return z_mid, info
    r2_fit = r2[inlier]
    z_fit = z_surface[inlier]

    r2_scale = float(np.percentile(r2_fit, 80.0)) + 1e-6
    w = 1.0 / (1.0 + r2_fit / r2_scale)
    X = np.stack([r2_fit, np.ones_like(r2_fit)], axis=1)
    Xw = X * np.sqrt(w)[:, None]
    yw = z_fit * np.sqrt(w)
    try:
        coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    except Exception:
        info['reason'] = 'lstsq_failed'
        return z_mid, info

    a = float(coef[0])
    b = float(coef[1])
    if (not np.isfinite(a)) or (not np.isfinite(b)):
        info['reason'] = 'non_finite_coef'
        return z_mid, info
    if a <= 1e-6:
        info['reason'] = 'non_positive_curvature'
        info['coef_a'] = float(a)
        info['coef_b'] = float(b)
        return z_mid, info

    pred = X @ coef
    rmse = float(np.sqrt(np.mean((pred - z_fit) ** 2)))
    z_std = float(np.std(z_fit)) + 1e-6
    nrmse = float(rmse / z_std)

    radius_est = float(1.0 / (2.0 * a))
    if (not np.isfinite(radius_est)) or radius_est <= 0.0:
        info['reason'] = 'invalid_radius_est'
        info['coef_a'] = float(a)
        info['coef_b'] = float(b)
        return z_mid, info

    # 圆弧轴心候选（通常位于可见表面之后）
    z_axis_raw = float(b + radius_est)
    z_med = float(np.median(z_surface))
    z_hi = float(np.percentile(z_surface, 95.0))
    xy_radius = float(np.sqrt(r2_q98))
    max_shift = float(max(0.70 * z_scale, 1.25 * xy_radius))
    z_axis_clamped = float(np.clip(z_axis_raw, z_med, z_hi + max_shift))
    z_center = float(max(z_mid, z_axis_clamped))

    info.update({
        'mode': 'arc_axis_center',
        'enabled': True,
        'reason': 'ok',
        'coef_a': float(a),
        'coef_b': float(b),
        'fit_rmse': float(rmse),
        'fit_nrmse': float(nrmse),
        'radius_est': float(radius_est),
        'z_axis_raw': float(z_axis_raw),
        'z_axis_clamped': float(z_axis_clamped),
        'z_centroid': float(z_center),
        'sample_count': int(n),
        'inlier_count': int(np.sum(inlier)),
    })
    return z_center, info


def process_3d_rigid_unified_deformation(
    comp_sam_pts, comp_targets_xy, mask_for_calc,
    src_xy, tgt_xy, H_lat, W_lat, device,
    latents, comp_mask_start, comp_m_start_full,
    scale_x, scale_y, depth_map,
    source_image_np=None, component_id=None,
    force_mode=None,
    enable_subject_scope_fill=False,
):
    """
    3D-Rigid 统一后端（意图识别: 平移/最小旋转 + projective z-buffer）。
    """
    debug_info = {
        'm_start_full': comp_m_start_full,
        'component_id': component_id,
    }

    comp_sam_pts = np.asarray(comp_sam_pts, dtype=np.float32).reshape(-1, 2)
    comp_targets_xy = np.asarray(comp_targets_xy, dtype=np.float32).reshape(-1, 2)

    if isinstance(source_image_np, np.ndarray):
        H_img, W_img = source_image_np.shape[:2]
    else:
        H_img = _infer_image_size_from_latent_and_scale(H_lat, scale_y)
        W_img = _infer_image_size_from_latent_and_scale(W_lat, scale_x)

    m_u8 = (mask_for_calc * 255).astype(np.uint8) if mask_for_calc.dtype != np.uint8 else mask_for_calc.copy()
    if m_u8.max() <= 1:
        m_u8 = (m_u8 * 255).astype(np.uint8)

    handle_points_yx = []
    for pt in comp_sam_pts:
        hx = int(np.clip(pt[0], 0, W_img - 1))
        hy = int(np.clip(pt[1], 0, H_img - 1))
        handle_points_yx.append([hy, hx])

    filter_save_dir = None
    if rotate_3d_processor.ENABLE_3D_DEBUG:
        # 统一写到当前调试目录，直接覆盖同名文件，不再创建 depth_comp* 子目录
        filter_save_dir = rotate_3d_processor.ensure_debug_dir()

    try:
        filtered_mask_u8, depth_filter_info = rotate_3d_processor.filter_background_by_depth(
            m_u8, depth_map, handle_points_yx,
            distance_threshold=None,
            save_dir=filter_save_dir,
            source_image_np=source_image_np,
        )
        if np.sum(filtered_mask_u8 > 127) < 20:
            filtered_mask_u8 = m_u8
    except Exception as e:
        filtered_mask_u8 = m_u8
        depth_filter_info = {'filtered': False, 'reason': f'depth_filter_error:{e}'}

    ys, xs = np.where(filtered_mask_u8 > 127)
    if len(xs) == 0:
        debug_info.update({
            'intent_type': '3D_RIGID_EMPTY_MASK',
            'sub_action': '3D_RIGID_EMPTY_MASK',
            'anchors_img': np.zeros((0, 2), dtype=np.float32),
            'depth_filter_info': depth_filter_info,
            'sam_mask': filtered_mask_u8,
            'depth_map': depth_map,
        })
        return latents.clone(), comp_mask_start.clone(), "3D-Rigid", None, np.zeros((0, 2), dtype=np.float32), debug_info

    depth_values = depth_map[ys, xs].astype(np.float32)
    depth_min = float(depth_values.min())
    depth_max = float(depth_values.max())
    depth_range = depth_max - depth_min + 1e-6

    cx = float(np.mean(xs))
    cy = float(np.mean(ys))

    x_range = max(float(xs.max() - xs.min()), 1.0)
    y_range = max(float(ys.max() - ys.min()), 1.0)
    xy_range = max(x_range, y_range)

    depth_std = float(np.std(depth_values))
    depth_mean = float(np.mean(depth_values))
    depth_cv = depth_std / (depth_mean + 1e-6)
    if depth_cv < 0.05:
        z_scale_factor = 0.4
    elif depth_cv < 0.10:
        z_scale_factor = 0.7
    elif depth_cv < 0.20:
        z_scale_factor = 1.0
    else:
        z_scale_factor = 1.5

    z_scale = xy_range * z_scale_factor
    z_centroid, z_center_info = _estimate_arc_axis_z_centroid(
        xs=xs,
        ys=ys,
        depth_values=depth_values,
        cx=cx,
        cy=cy,
        depth_min=depth_min,
        depth_range=depth_range,
        z_scale=z_scale,
        min_points=96,
    )

    xs_f = xs.astype(np.float32)
    ys_f = ys.astype(np.float32)
    zs = (depth_values - depth_min) / depth_range * z_scale - z_centroid
    pts3d = np.stack([xs_f - cx, -(ys_f - cy), zs], axis=1).astype(np.float32)
    old_xy = np.stack([xs_f, ys_f], axis=1).astype(np.float32)

    num_pairs = min(len(comp_sam_pts), len(comp_targets_xy))
    if num_pairs == 0:
        debug_info.update({
            'intent_type': '3D_RIGID_EMPTY',
            'sub_action': '3D_RIGID_EMPTY',
            'anchors_img': np.zeros((0, 2), dtype=np.float32),
            'depth_filter_info': depth_filter_info,
            'sam_mask': filtered_mask_u8,
            'depth_map': depth_map,
        })
        return latents.clone(), comp_mask_start.clone(), "3D-Rigid", None, np.zeros((0, 2), dtype=np.float32), debug_info

    pair_src = comp_sam_pts[:num_pairs]
    pair_tgt = comp_targets_xy[:num_pairs]

    handle_3d_orig = []
    target_3d_vis = []
    for i in range(num_pairs):
        hx, hy = float(pair_src[i][0]), float(pair_src[i][1])
        tx, ty = float(pair_tgt[i][0]), float(pair_tgt[i][1])
        hxi = int(np.clip(round(hx), 0, W_img - 1))
        hyi = int(np.clip(round(hy), 0, H_img - 1))
        h_depth = float(depth_map[hyi, hxi])
        hz = (h_depth - depth_min) / depth_range * z_scale - z_centroid
        h3 = np.array([hx - cx, -(hy - cy), hz], dtype=np.float32)
        handle_3d_orig.append(h3)
        target_3d_vis.append([tx - cx, -(ty - cy), hz])

    handle_3d_orig = np.stack(handle_3d_orig, axis=0)
    target_3d_vis = np.asarray(target_3d_vis, dtype=np.float32)

    intent_type_2d, intent_params = detect_rigid_intent(
        pair_src,
        filtered_mask_u8,
        force_mode=force_mode,
    )
    intent_type_2d = str(intent_type_2d).upper()
    if intent_type_2d not in ("TRANSLATION", "ROTATION"):
        intent_type_2d = "ROTATION"
    debug_info['rigid_intent'] = intent_type_2d
    debug_info['rigid_intent_params'] = dict(intent_params) if isinstance(intent_params, dict) else {}

    pair_disp = np.linalg.norm(pair_tgt - pair_src, axis=1)
    ref_idx = int(np.argmax(pair_disp))
    ref_h = pair_src[ref_idx]
    ref_t = pair_tgt[ref_idx]

    rotation_axis = '3D-Direct'
    rotation_angle = 0.0
    rotation_yaw_deg, rotation_pitch_deg = 0.0, 0.0
    R = np.eye(3, dtype=np.float32)
    shift_xy = np.array([0.0, 0.0], dtype=np.float32)
    shift_3d = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    if intent_type_2d == "TRANSLATION":
        shift_xy = np.mean(pair_tgt - pair_src, axis=0).astype(np.float32)
        shift_3d = np.array([float(shift_xy[0]), float(-shift_xy[1]), 0.0], dtype=np.float32)
        rotation_axis = '3D-Translation'
        pts3d_rot = pts3d + shift_3d[None, :]
        rot_x = np.clip(old_xy[:, 0] + shift_xy[0], 0, W_img - 1)
        rot_y = np.clip(old_xy[:, 1] + shift_xy[1], 0, H_img - 1)
        rot_xy = np.stack([rot_x, rot_y], axis=1).astype(np.float32)
        handle_3d_rot = handle_3d_orig + shift_3d[None, :]
        rotated_handles_xy = np.stack(
            [
                np.clip(pair_src[:, 0] + shift_xy[0], 0, W_img - 1),
                np.clip(pair_src[:, 1] + shift_xy[1], 0, H_img - 1),
            ],
            axis=1,
        ).astype(np.float32)
        angle_debug_info = {
            'phase1_rotation_solver': 'translation_intent_shift',
            'translation_shift_xy': shift_xy.astype(np.float32).tolist(),
            'translation_shift_3d': shift_3d.astype(np.float32).tolist(),
            'rotation_skipped': True,
            'yaw_deg': 0.0,
            'pitch_deg': 0.0,
        }
        print(f"  [3D-Rigid] Translation intent: dx={shift_xy[0]:.2f}, dy={shift_xy[1]:.2f}")
    else:
        ref_hx = int(np.clip(round(float(ref_h[0])), 0, W_img - 1))
        ref_hy = int(np.clip(round(float(ref_h[1])), 0, H_img - 1))
        ref_h_depth = float(depth_map[ref_hy, ref_hx])
        ref_h_z = (ref_h_depth - depth_min) / depth_range * z_scale - z_centroid
        ref_h_3d = np.array([float(ref_h[0] - cx), -float(ref_h[1] - cy), ref_h_z], dtype=np.float64)
        ref_t_3d = np.array([float(ref_t[0] - cx), -float(ref_t[1] - cy), ref_h_z], dtype=np.float64)

        R_direct, direct_info = _rotation_matrix_from_vec_to_vec(ref_h_3d, ref_t_3d)
        if (not np.isfinite(R_direct).all()) or (not bool(direct_info.get('ok', True))):
            R = np.eye(3, dtype=np.float32)
            rotation_angle = 0.0
            rotation_yaw_deg, rotation_pitch_deg = 0.0, 0.0
            angle_debug_info = {
                'phase1_rotation_solver': 'identity_fallback',
                'phase1_rotation_from_vec_to_vec': direct_info,
                'handle_ref_3d': ref_h_3d.tolist(),
                'target_ref_3d': ref_t_3d.tolist(),
                'yaw_deg': 0.0,
                'pitch_deg': 0.0,
            }
            print(
                "  [3D-Rigid] Direct rotation fallback: identity "
                f"(reason={direct_info.get('reason', 'unknown')})"
            )
        else:
            R = R_direct.astype(np.float32)
            rotation_angle = float(direct_info.get('angle_deg', 0.0))
            rotation_yaw_deg, rotation_pitch_deg = _matrix_to_yaw_pitch_deg(R)
            h3_after = (R @ ref_h_3d.astype(np.float32)).astype(np.float32)
            align_err = float(np.linalg.norm(h3_after - ref_t_3d.astype(np.float32)))
            angle_debug_info = {
                'phase1_rotation_solver': 'vec_to_vec_min_rotation',
                'phase1_rotation_from_vec_to_vec': direct_info,
                'handle_ref_3d': ref_h_3d.tolist(),
                'target_ref_3d': ref_t_3d.tolist(),
                'handle_after_rotation_3d': h3_after.tolist(),
                'handle_target_alignment_error': align_err,
                'yaw_deg': float(rotation_yaw_deg),
                'pitch_deg': float(rotation_pitch_deg),
            }
            print(
                f"  [3D-Rigid] Direct rotation: angle={rotation_angle:.2f}deg, "
                f"align_err={align_err:.3f}"
            )

        pts3d_rot = (R @ pts3d.T).T
        rot_x = np.clip(pts3d_rot[:, 0] + cx, 0, W_img - 1)
        rot_y = np.clip(-pts3d_rot[:, 1] + cy, 0, H_img - 1)
        rot_xy = np.stack([rot_x, rot_y], axis=1).astype(np.float32)
        handle_3d_rot = (R @ handle_3d_orig.T).T.astype(np.float32)
        rotated_handles_xy = np.stack(
            [
                np.clip(handle_3d_rot[:, 0] + cx, 0, W_img - 1),
                np.clip(-handle_3d_rot[:, 1] + cy, 0, H_img - 1),
            ],
            axis=1,
        ).astype(np.float32)

    control_indices = _map_xy_controls_to_indices(old_xy, pair_src)
    for i, nearest_idx in enumerate(control_indices):
        if i >= num_pairs:
            break
        nidx = int(nearest_idx)
        if nidx < 0 or nidx >= pts3d_rot.shape[0]:
            continue
        pts3d_rot[nidx] = handle_3d_rot[i]
        rot_xy[nidx] = rotated_handles_xy[i]

    # 3D-Rigid 暂不做额外局部锁定，避免局部纹理被破坏。
    control_lock_info = {
        'enabled': False,
        'reason': 'disabled_for_3d_rigid',
        'pair_count': int(num_pairs),
        'active_pair_count': 0,
    }
    control_lock_radius_px = 0.0
    control_lock_falloff_px = 0.0
    min_neighbors_runtime = 0
    max_neighbors_runtime = 0
    rotated_handles_xy_locked = rotated_handles_xy.copy()

    p1_new_z = pts3d_rot[:, 2].astype(np.float32)
    need_image_outputs = bool(rotate_3d_processor.ENABLE_3D_DEBUG)
    warped_latents, comp_mask_end, norm_grid, m_end_full, phase1_rgb, phase1_depth, rotated_mask_u8, proj_debug = (
        _projective_3d_warp_from_points(
            latents=latents,
            H_lat=H_lat,
            W_lat=W_lat,
            device=device,
            old_xy=old_xy,
            new_xy=rot_xy,
            new_z=p1_new_z,
            scale_x=scale_x,
            scale_y=scale_y,
            source_image_np=source_image_np,
            mask_start_full=comp_m_start_full,
            need_image_outputs=need_image_outputs,
            enable_subject_scope_fill=enable_subject_scope_fill,
        )
    )

    if not isinstance(m_end_full, np.ndarray):
        m_end_full = _latent_mask_to_fullres(comp_mask_end, target_hw=(H_img, W_img))
    if not isinstance(rotated_mask_u8, np.ndarray):
        rotated_mask_u8 = (m_end_full > 0.5).astype(np.uint8) * 255

    save_dir = rotate_3d_processor.ensure_debug_dir()
    if rotate_3d_processor.ENABLE_3D_DEBUG and save_dir is not None:
        comp_tag = int(component_id) + 1 if component_id is not None else 1
        if isinstance(source_image_np, np.ndarray) and source_image_np.shape[0] >= H_img and source_image_np.shape[1] >= W_img:
            colors_rgb = source_image_np[ys, xs].astype(np.float32) / 255.0
        else:
            z_norm = (zs - float(np.min(zs))) / (float(np.max(zs) - np.min(zs)) + 1e-6)
            colors_rgb = np.stack([z_norm, z_norm, z_norm], axis=1).astype(np.float32)
        try:
            handle_3d_vis = np.nan_to_num(handle_3d_orig, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            target_3d_vis_safe = np.nan_to_num(target_3d_vis, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            rotated_handle_3d_vis = np.nan_to_num(handle_3d_rot, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            rotate_3d_processor.save_3d_point_cloud_visualization(
                pts3d, pts3d_rot, colors_rgb,
                rotation_axis, float(rotation_angle), save_dir,
                centroid_3d=[0.0, 0.0, 0.0],
                handle_3d=handle_3d_vis,
                target_3d=target_3d_vis_safe,
                handle_rotated_3d=rotated_handle_3d_vis,
                output_name=f"depth_05_3d_point_cloud_rigid_comp{comp_tag}.png",
                yaw_deg=rotation_yaw_deg,
                pitch_deg=rotation_pitch_deg,
                image_hw=(H_img, W_img),
                image_center_xy=(cx, cy),
            )
        except Exception as e:
            print(f"[3D-Rigid] Point cloud vis failed: {e}")

    if intent_type_2d == "TRANSLATION":
        sub_action = f"Translation(dx={shift_xy[0]:.1f},dy={shift_xy[1]:.1f})"
        intent_type_debug = '3D_RIGID_TRANSLATION'
        method_desc = '3D rigid translation (intent-aware)'
        solver_tag = 'intent_translation'
    else:
        sub_action = f"Yaw={rotation_yaw_deg:.1f},Pitch={rotation_pitch_deg:.1f}"
        intent_type_debug = '3D_RIGID_ROTATION'
        method_desc = '3D direct minimal rotation'
        solver_tag = '3D-direct'

    debug_info.update({
        'intent_type': intent_type_debug,
        'sub_action': sub_action,
        'method_desc': method_desc,
        'deformation_space': '3D',
        'warp_backend': 'PROJECTIVE_ZBUFFER_3D',
        'anchors_img': np.zeros((0, 2), dtype=np.float32),
        'comp_sam_pts': pair_src.astype(np.float32),
        'comp_targets_xy': pair_tgt.astype(np.float32),
        'rotated_handle_points_xy': rotated_handles_xy_locked.astype(np.float32),
        'norm_grid': norm_grid,
        'warped_latents': warped_latents.clone(),
        'mask_end': comp_mask_end.clone(),
        'm_end_full': m_end_full,
        'rotated_mask': rotated_mask_u8,
        'rotated_rgb': phase1_rgb,
        'rotated_depth': phase1_depth,
        'subject_scope_mask_full': np.asarray((proj_debug or {}).get('subject_scope_mask_full', np.zeros((H_img, W_img), dtype=np.float32)), dtype=np.float32),
        'subject_hole_mask_full': np.asarray((proj_debug or {}).get('subject_hole_mask_full', np.zeros((H_img, W_img), dtype=np.float32)), dtype=np.float32),
        'subject_scope_mask_lat': np.asarray((proj_debug or {}).get('subject_scope_mask_lat', np.zeros((H_lat, W_lat), dtype=np.float32)), dtype=np.float32),
        'subject_hole_mask_lat': np.asarray((proj_debug or {}).get('subject_hole_mask_lat', np.zeros((H_lat, W_lat), dtype=np.float32)), dtype=np.float32),
        'sam_mask': filtered_mask_u8,
        'depth_map': depth_map,
        'depth_filter_info': depth_filter_info,
        'z_center_info': z_center_info,
        'z_scale_factor': float(z_scale_factor),
        'z_scale': float(z_scale),
        'z_centroid': float(z_centroid),
        'control_lock_info': control_lock_info,
        'control_lock_radius_px': float(control_lock_radius_px),
        'control_lock_falloff_px': float(control_lock_falloff_px),
        'control_lock_radius_scale': float(NONRIGID_CONTROL_RADIUS_SCALE),
        'control_global_follow_gain': float(NONRIGID_CONTROL_GLOBAL_FOLLOW_GAIN),
        'control_lock_neighbors': {
            'min': int(min_neighbors_runtime),
            'max': int(max_neighbors_runtime),
        },
        'centroid': (int(round(cx)), int(round(cy))),
        'rotation_axis': rotation_axis,
        'rotation_angle_deg': float(rotation_angle),
        'rotation_yaw_deg': rotation_yaw_deg,
        'rotation_pitch_deg': rotation_pitch_deg,
        'rotation_ref_index': ref_idx,
        'angle_computation': angle_debug_info,
    })
    print(f"  [3D-Rigid] {sub_action} (solver={solver_tag})")
    return warped_latents, comp_mask_end, "3D-Rigid", None, np.zeros((0, 2), dtype=np.float32), debug_info


def process_3d_nonrigid_deformation(
    comp_sam_pts, comp_targets_xy, mask_for_calc,
    src_xy, tgt_xy, H_lat, W_lat, device,
    latents, comp_mask_start, comp_m_start_full,
    scale_x, scale_y, influence_ratio,
    depth_map, source_image_np=None, component_id=None,
    image_to_runtime_scale=1.0,
    anchor_strategy_3d=DEFAULT_THREE_D_ANCHOR_STRATEGY,
    enable_subject_scope_fill=False,
):
    """
    3D 非刚性变形（统一3D空间）:
    1. 在3D点云空间（x, y, z）估计局部位移场
    2. 依据3D距离自动选取锚点（远离控制点/低影响区）
    3. 将3D变形投影回2D，通过z-buffer重映射（非MLS）
    """
    debug_info = {
        'm_start_full': comp_m_start_full,
        'component_id': component_id,
    }

    if isinstance(source_image_np, np.ndarray):
        H_img, W_img = source_image_np.shape[:2]
    else:
        H_img = _infer_image_size_from_latent_and_scale(H_lat, scale_y)
        W_img = _infer_image_size_from_latent_and_scale(W_lat, scale_x)

    m_u8 = (mask_for_calc * 255).astype(np.uint8) if mask_for_calc.dtype != np.uint8 else mask_for_calc.copy()
    if m_u8.max() <= 1:
        m_u8 = (m_u8 * 255).astype(np.uint8)

    handle_points_yx = []
    for pt in comp_sam_pts:
        hx = int(np.clip(pt[0], 0, W_img - 1))
        hy = int(np.clip(pt[1], 0, H_img - 1))
        handle_points_yx.append([hy, hx])

    filter_save_dir = None
    if rotate_3d_processor.ENABLE_3D_DEBUG:
        # 统一写到当前调试目录，直接覆盖同名文件，不再创建 depth_comp* 子目录
        filter_save_dir = rotate_3d_processor.ensure_debug_dir()

    try:
        filtered_mask_u8, depth_filter_info = rotate_3d_processor.filter_background_by_depth(
            m_u8, depth_map, handle_points_yx,
            distance_threshold=None,
            save_dir=filter_save_dir,
            source_image_np=source_image_np,
        )
        if np.sum(filtered_mask_u8 > 127) < 20:
            filtered_mask_u8 = m_u8
    except Exception as e:
        filtered_mask_u8 = m_u8
        depth_filter_info = {'filtered': False, 'reason': f'depth_filter_error:{e}'}

    ys, xs = np.where(filtered_mask_u8 > 127)
    if len(xs) == 0:
        debug_info.update({
            'intent_type': '3D_NR_EMPTY_MASK',
            'sub_action': '3D_NR_EMPTY_MASK',
            'anchors_img': np.zeros((0, 2), dtype=np.float32),
            'depth_filter_info': depth_filter_info,
            'sam_mask': filtered_mask_u8,
            'depth_map': depth_map,
        })
        return latents.clone(), comp_mask_start.clone(), "3D-Non-Rigid", None, np.zeros((0, 2), dtype=np.float32), debug_info

    depth_values = depth_map[ys, xs].astype(np.float32)
    depth_min = float(depth_values.min())
    depth_max = float(depth_values.max())
    depth_range = depth_max - depth_min + 1e-6

    cx = float(np.mean(xs))
    cy = float(np.mean(ys))

    x_range = max(float(xs.max() - xs.min()), 1.0)
    y_range = max(float(ys.max() - ys.min()), 1.0)
    xy_range = max(x_range, y_range)

    depth_std = float(np.std(depth_values))
    depth_mean = float(np.mean(depth_values))
    depth_cv = depth_std / (depth_mean + 1e-6)
    if depth_cv < 0.05:
        z_scale_factor = 0.4
    elif depth_cv < 0.10:
        z_scale_factor = 0.7
    elif depth_cv < 0.20:
        z_scale_factor = 1.0
    else:
        z_scale_factor = 1.5

    z_scale = xy_range * z_scale_factor
    z_centroid, z_center_info = _estimate_arc_axis_z_centroid(
        xs=xs,
        ys=ys,
        depth_values=depth_values,
        cx=cx,
        cy=cy,
        depth_min=depth_min,
        depth_range=depth_range,
        z_scale=z_scale,
        min_points=96,
    )

    xs_f = xs.astype(np.float32)
    ys_f = ys.astype(np.float32)
    zs = (depth_values - depth_min) / depth_range * z_scale - z_centroid
    pts3d = np.stack([xs_f - cx, -(ys_f - cy), zs], axis=1).astype(np.float32)

    num_pairs = min(len(comp_sam_pts), len(comp_targets_xy))
    if num_pairs == 0:
        debug_info.update({
            'intent_type': '3D_NR_EMPTY',
            'sub_action': '3D_NR_EMPTY',
            'anchors_img': np.zeros((0, 2), dtype=np.float32),
            'depth_filter_info': depth_filter_info,
            'sam_mask': filtered_mask_u8,
            'depth_map': depth_map,
        })
        return latents.clone(), comp_mask_start.clone(), "3D-Non-Rigid", None, np.zeros((0, 2), dtype=np.float32), debug_info

    handle_3d = []
    delta_3d = []
    target_depth_sampling = []
    # latent 域默认分辨率更低，直接按真实比例会过弱；保留 0.25 下限维持可达性。
    runtime_scale = float(np.clip(float(image_to_runtime_scale), 0.25, 1.0))
    sample_radius_base = float(max(24.0, 0.08 * float(min(H_img, W_img))))
    sample_radius_runtime = int(np.clip(round(sample_radius_base * runtime_scale), 3, 96))
    for i in range(num_pairs):
        hx, hy = float(comp_sam_pts[i][0]), float(comp_sam_pts[i][1])
        tx, ty = float(comp_targets_xy[i][0]), float(comp_targets_xy[i][1])

        hxi = int(np.clip(round(hx), 0, W_img - 1))
        hyi = int(np.clip(round(hy), 0, H_img - 1))

        h_depth = float(depth_map[hyi, hxi])
        sampled_depth, sampled_xy, sample_dist = _sample_nearest_foreground_depth(
            depth_map=depth_map,
            fg_mask_u8=filtered_mask_u8,
            x=tx,
            y=ty,
            fallback_depth=h_depth,
            max_radius=sample_radius_runtime,
        )
        sampled_depth = float(sampled_depth)
        # 目标 z 保持 >= 控制点 z：小于时锁回 handle z，大于时保留目标 z
        t_depth = float(max(sampled_depth, h_depth))
        target_depth_sampling.append({
            'pair_idx': int(i),
            'sample_xy': (int(sampled_xy[0]), int(sampled_xy[1])),
            'sample_dist_px': float(sample_dist),
            'mode': 'sample_ge_handle_z',
            'handle_depth': float(h_depth),
            'sampled_depth': float(sampled_depth),
            'final_depth': float(t_depth),
            'clamped_to_handle': bool(sampled_depth < h_depth),
        })

        hz = (h_depth - depth_min) / depth_range * z_scale - z_centroid
        tz = (t_depth - depth_min) / depth_range * z_scale - z_centroid

        h3 = np.array([hx - cx, -(hy - cy), hz], dtype=np.float32)
        t3 = np.array([tx - cx, -(ty - cy), tz], dtype=np.float32)
        handle_3d.append(h3)
        delta_3d.append(t3 - h3)

    handle_3d = np.stack(handle_3d, axis=0)
    delta_3d = np.stack(delta_3d, axis=0)

    old_xy = np.stack([xs_f, ys_f], axis=1).astype(np.float32)
    anchor_handle_xy = None
    anchor_target_xy = None
    anchors_img, anchor_select_info = _select_rotation_invariant_anchors_3d(
        points_3d=pts3d,
        points_xy=old_xy,
        handle_points_3d=handle_3d,
        influence_ratio=influence_ratio,
        max_anchors=220,
        handle_points_xy=anchor_handle_xy,
        target_points_xy=anchor_target_xy,
    )
    anchor_select_info = dict(anchor_select_info)
    anchor_select_info['opposite_anchor_bias_enabled'] = False
    anchor_select_info['strategy_override'] = 'balanced_edge_without_drag_opposition'

    control_indices = _map_xy_controls_to_indices(old_xy, comp_sam_pts[:num_pairs])
    anchor_indices = _map_xy_controls_to_indices(old_xy, anchors_img)
    if anchor_indices.size > 0:
        if control_indices.size > 0:
            anchor_indices = anchor_indices[~np.isin(anchor_indices, control_indices)]
        anchor_indices = np.unique(anchor_indices.astype(np.int32))
    if anchor_indices.size > 0:
        anchors_img = old_xy[anchor_indices].astype(np.float32)
    else:
        anchors_img = np.zeros((0, 2), dtype=np.float32)
    anchor_pts3d = pts3d[anchor_indices] if anchor_indices.size > 0 else np.zeros((0, 3), dtype=np.float32)

    influence_ratio = float(np.clip(influence_ratio, 0.0, 1.0))
    obj_diag_3d = float(np.sqrt(x_range ** 2 + y_range ** 2 + z_scale ** 2) + 1e-6)
    influence_radius = (0.12 + 0.88 * influence_ratio) * obj_diag_3d
    sigma = max(influence_radius * 0.45, 1e-3)

    z_metric_weight = 1.35
    target_3d = handle_3d + delta_3d
    use_rigid_residual = False
    compare_sgf_enabled = False
    nonrigid_solver_mode = "sgf_direct_kernel"
    rigid_residual_info = {
        'enabled': False,
        'mode': 'sgf_direct_kernel',
        'blend': 0.0,
        'fit_info': None,
    }
    compare_sgf_info = {
        'enabled': False,
        'reason': 'disabled_or_not_applicable',
    }

    pts3d_field_base = pts3d.astype(np.float32)
    handle_field_3d = handle_3d.astype(np.float32)
    delta_field_3d = delta_3d.astype(np.float32)
    anchor_field_3d = anchor_pts3d.astype(np.float32)
    pts3d_original_for_lock = pts3d.astype(np.float32)

    if use_rigid_residual:
        fit_weights = np.linalg.norm(delta_3d[:, :2], axis=1).astype(np.float64)
        if float(np.max(fit_weights)) <= 1e-6:
            fit_weights = np.linalg.norm(delta_3d, axis=1).astype(np.float64)
        fit_weights = np.clip(fit_weights, 1e-4, None)

        R_nr, t_nr, fit_info_nr = _fit_rigid_transform_kabsch_3d(
            src_pts=handle_3d.astype(np.float64),
            tgt_pts=target_3d.astype(np.float64),
            weights=fit_weights,
        )
        rigid_blend = float(NONRIGID_RIGID_BLEND)

        if bool(fit_info_nr.get('ok', False)):
            pts3d_rigid_full = ((R_nr @ pts3d.astype(np.float64).T).T + t_nr[None, :]).astype(np.float32)
            handle_rigid_full = ((R_nr @ handle_3d.astype(np.float64).T).T + t_nr[None, :]).astype(np.float32)
            if anchor_pts3d.shape[0] > 0:
                anchor_rigid_full = ((R_nr @ anchor_pts3d.astype(np.float64).T).T + t_nr[None, :]).astype(np.float32)
            else:
                anchor_rigid_full = np.zeros((0, 3), dtype=np.float32)

            pts3d_field_base = (pts3d + (pts3d_rigid_full - pts3d) * rigid_blend).astype(np.float32)
            handle_field_3d = (handle_3d + (handle_rigid_full - handle_3d) * rigid_blend).astype(np.float32)
            delta_field_3d = (target_3d - handle_field_3d).astype(np.float32)
            if anchor_pts3d.shape[0] > 0:
                anchor_field_3d = (anchor_pts3d + (anchor_rigid_full - anchor_pts3d) * rigid_blend).astype(np.float32)
            else:
                anchor_field_3d = np.zeros((0, 3), dtype=np.float32)
            pts3d_original_for_lock = pts3d_field_base
            nonrigid_solver_mode = "rigid_plus_residual"
            yaw_fit, pitch_fit = _matrix_to_yaw_pitch_deg(R_nr)
            rigid_motion_norm = np.linalg.norm((pts3d_field_base - pts3d), axis=1).astype(np.float32)
            residual_norm = np.linalg.norm(delta_field_3d, axis=1).astype(np.float32)
            rigid_residual_info = {
                'enabled': True,
                'mode': nonrigid_solver_mode,
                'blend': float(rigid_blend),
                'fit_info': fit_info_nr,
                'translation_3d': np.asarray(t_nr, dtype=np.float64).tolist(),
                'yaw_deg': float(yaw_fit),
                'pitch_deg': float(pitch_fit),
                'rigid_motion_mean': float(np.mean(rigid_motion_norm)) if rigid_motion_norm.size > 0 else 0.0,
                'rigid_motion_max': float(np.max(rigid_motion_norm)) if rigid_motion_norm.size > 0 else 0.0,
                'residual_mean': float(np.mean(residual_norm)) if residual_norm.size > 0 else 0.0,
                'residual_max': float(np.max(residual_norm)) if residual_norm.size > 0 else 0.0,
            }
        else:
            compare_sgf_enabled = False
            rigid_residual_info = {
                'enabled': False,
                'mode': 'sgf_direct_kernel_fallback',
                'blend': float(rigid_blend),
                'fit_info': fit_info_nr,
            }

    def _solve_local_nonrigid(base_pts3d, local_handles_3d, local_delta_3d, local_anchor_pts3d):
        local_anchor_soft_strength = 0.0
        local_anchor_soft_total_mass = 0.0
        if local_anchor_pts3d.shape[0] > 0:
            handle_n = int(local_handles_3d.shape[0])
            anchor_n = int(local_anchor_pts3d.shape[0])
            anchor_total_ratio = float(np.clip(0.35 + 0.45 * (1.0 - influence_ratio), 0.20, 0.80))
            local_anchor_soft_total_mass = anchor_total_ratio * float(max(handle_n, 1))
            local_anchor_soft_strength = float(local_anchor_soft_total_mass / float(max(anchor_n, 1)))
            local_anchor_soft_strength = float(np.clip(local_anchor_soft_strength, 0.004, 0.22))
            constraint_pts3d = np.concatenate([local_handles_3d, local_anchor_pts3d], axis=0)
            constraint_delta3d = np.concatenate([local_delta_3d, np.zeros_like(local_anchor_pts3d)], axis=0)
            constraint_w = np.concatenate(
                [
                    np.ones((local_handles_3d.shape[0],), dtype=np.float32),
                    np.full((local_anchor_pts3d.shape[0],), local_anchor_soft_strength, dtype=np.float32),
                ],
                axis=0,
            )
        else:
            constraint_pts3d = local_handles_3d
            constraint_delta3d = local_delta_3d
            constraint_w = np.ones((local_handles_3d.shape[0],), dtype=np.float32)

        diff = base_pts3d[:, None, :] - constraint_pts3d[None, :, :]
        diff_metric = diff.copy()
        diff_metric[:, :, 2] *= z_metric_weight
        dists = np.linalg.norm(diff_metric, axis=2)
        weights = np.exp(-(dists ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
        weights *= constraint_w[None, :]
        sum_w = np.sum(weights, axis=1, keepdims=True) + 1e-6
        local_weights = weights / sum_w
        disp = local_weights @ constraint_delta3d

        diff_ctrl = base_pts3d[:, None, :] - local_handles_3d[None, :, :]
        diff_ctrl[:, :, 2] *= z_metric_weight
        min_ctrl_dist = np.min(np.linalg.norm(diff_ctrl, axis=2), axis=1)
        local_sigma_gate_scale = 0.69 if local_anchor_pts3d.shape[0] > 0 else 0.75
        local_sigma_gate = max(influence_radius * local_sigma_gate_scale, 1e-3)
        gate = np.exp(-(min_ctrl_dist ** 2) / (2.0 * local_sigma_gate ** 2)).astype(np.float32)
        disp *= gate[:, None]
        disp_norm = np.linalg.norm(disp, axis=1).astype(np.float32)

        return (base_pts3d + disp).astype(np.float32), {
            'anchor_soft_strength': float(local_anchor_soft_strength),
            'anchor_soft_total_mass': float(local_anchor_soft_total_mass),
            'sigma_gate_scale': float(local_sigma_gate_scale),
            'sigma_gate': float(local_sigma_gate),
            'disp_mean': float(np.mean(disp_norm)) if disp_norm.size > 0 else 0.0,
            'disp_max': float(np.max(disp_norm)) if disp_norm.size > 0 else 0.0,
        }

    pts3d_new, local_field_info = _solve_local_nonrigid(
        pts3d_field_base,
        handle_field_3d,
        delta_field_3d,
        anchor_field_3d,
    )
    anchor_soft_strength = float(local_field_info.get('anchor_soft_strength', 0.0))
    anchor_soft_total_mass = float(local_field_info.get('anchor_soft_total_mass', 0.0))
    sigma_gate_scale = float(local_field_info.get('sigma_gate_scale', 0.75))
    sigma_gate = float(local_field_info.get('sigma_gate', max(influence_radius * sigma_gate_scale, 1e-3)))

    sgf_field_info = None
    sgf_lock_info = None
    pts3d_compare_sgf = None
    pts3d_compare_sgf_raw = None
    if compare_sgf_enabled:
        pts3d_compare_sgf_raw, sgf_field_info = _solve_local_nonrigid(
            pts3d.astype(np.float32),
            handle_3d.astype(np.float32),
            delta_3d.astype(np.float32),
            anchor_pts3d.astype(np.float32),
        )

    # 备份版行为：不做额外锚点后处理，仅保留核权重里的软约束。
    anchor_lock_gain = 0.0
    anchor_lock_snap = 0.0
    anchor_lock_info = {
        'enabled': False,
        'reason': 'disabled_use_backup_soft_anchor',
        'hard_locked_anchor_count': 0,
        'anchor_drift_mean_before': 0.0,
        'anchor_drift_mean_after': 0.0,
    }
    control_lock_radius_img = float(np.clip((6.5 + 9.0 * influence_ratio) * NONRIGID_CONTROL_RADIUS_SCALE, 6.5, 24.0))
    runtime_lock_scale = float(max(np.sqrt(runtime_scale), 0.45))
    control_lock_radius_px = float(np.clip(control_lock_radius_img * runtime_lock_scale, 3.0, 16.0))
    control_lock_falloff_px = float(np.clip(control_lock_radius_px * 0.55, 1.5, 8.5))
    neighbor_scale = float(max(np.sqrt(runtime_scale), 0.50))
    min_neighbors_runtime = int(np.clip(round(12.0 * neighbor_scale), 4, 18))
    max_neighbors_runtime = int(np.clip(round(110.0 * neighbor_scale), max(min_neighbors_runtime, 24), 120))
    if pts3d_compare_sgf_raw is not None:
        pts3d_compare_sgf, sgf_lock_info = _enforce_control_targets_on_points_3d(
            pts3d_original=pts3d.astype(np.float32),
            pts3d_deformed=pts3d_compare_sgf_raw,
            points_xy=old_xy,
            handle_points_xy=comp_sam_pts[:num_pairs],
            target_points_xy=comp_targets_xy[:num_pairs],
            target_points_3d=target_3d,
            control_indices=control_indices,
            pair_active_mask=np.ones((num_pairs,), dtype=bool),
            radius_px=control_lock_radius_px,
            min_neighbors=min_neighbors_runtime,
            max_neighbors=max_neighbors_runtime,
            falloff_px=control_lock_falloff_px,
            blend_strength=1.0,
            rigid_core_ratio=0.35,
            rigid_core_min_px=2.5,
            global_follow_gain=float(NONRIGID_CONTROL_GLOBAL_FOLLOW_GAIN),
        )
    pts3d_new, control_lock_info = _enforce_control_targets_on_points_3d(
        pts3d_original=pts3d_original_for_lock,
        pts3d_deformed=pts3d_new,
        points_xy=old_xy,
        handle_points_xy=comp_sam_pts[:num_pairs],
        target_points_xy=comp_targets_xy[:num_pairs],
        target_points_3d=target_3d,
        control_indices=control_indices,
        pair_active_mask=np.ones((num_pairs,), dtype=bool),
        radius_px=control_lock_radius_px,
        min_neighbors=min_neighbors_runtime,
        max_neighbors=max_neighbors_runtime,
        falloff_px=control_lock_falloff_px,
        blend_strength=1.0,
        rigid_core_ratio=0.35,
        rigid_core_min_px=2.5,
        global_follow_gain=float(NONRIGID_CONTROL_GLOBAL_FOLLOW_GAIN),
    )
    if pts3d_compare_sgf is not None:
        sgf_tdedit_dist = np.linalg.norm((pts3d_compare_sgf - pts3d_new), axis=1).astype(np.float32)
        compare_sgf_info = {
            'enabled': True,
            'reason': 'ok',
            'sgf_field_info': sgf_field_info,
            'sgf_control_lock_info': sgf_lock_info,
            'sgf_vs_tdedit_mean_l2': float(np.mean(sgf_tdedit_dist)) if sgf_tdedit_dist.size > 0 else 0.0,
            'sgf_vs_tdedit_max_l2': float(np.max(sgf_tdedit_dist)) if sgf_tdedit_dist.size > 0 else 0.0,
        }
    new_x = np.clip(pts3d_new[:, 0] + cx, 0, W_img - 1)
    new_y = np.clip(-pts3d_new[:, 1] + cy, 0, H_img - 1)
    new_xy = np.stack([new_x, new_y], axis=1).astype(np.float32)

    new_z = pts3d_new[:, 2].astype(np.float32)
    need_image_outputs = bool(rotate_3d_processor.ENABLE_3D_DEBUG)
    warped_latents, comp_mask_end, norm_grid, m_end_full, warped_rgb_np, warped_depth_np, rotated_mask_u8, proj_debug = (
        _projective_3d_warp_from_points(
            latents=latents,
            H_lat=H_lat,
            W_lat=W_lat,
            device=device,
            old_xy=old_xy,
            new_xy=new_xy,
            new_z=new_z,
            scale_x=scale_x,
            scale_y=scale_y,
            source_image_np=source_image_np,
            mask_start_full=comp_m_start_full,
            need_image_outputs=need_image_outputs,
            enable_subject_scope_fill=enable_subject_scope_fill,
        )
    )

    # 3D点云可视化：Non-Rigid 显示变形前后（非刚性本身不含全局旋转，角度置0）
    save_dir = rotate_3d_processor.ensure_debug_dir()
    if rotate_3d_processor.ENABLE_3D_DEBUG and save_dir is not None:
        comp_tag = int(component_id) + 1 if component_id is not None else 1
        if isinstance(source_image_np, np.ndarray) and source_image_np.shape[0] >= H_img and source_image_np.shape[1] >= W_img:
            colors_rgb = source_image_np[ys, xs].astype(np.float32) / 255.0
        else:
            # 无原图时使用深度灰度伪色，避免点云可视化缺失
            z_norm = (zs - float(np.min(zs))) / (float(np.max(zs) - np.min(zs)) + 1e-6)
            colors_rgb = np.stack([z_norm, z_norm, z_norm], axis=1).astype(np.float32)
        extra_rot_pts, extra_rot_cols = (None, None)
        try:
            def _collect_control_positions(points3d_arr):
                points3d_arr = np.asarray(points3d_arr, dtype=np.float32).reshape(-1, 3)
                out = handle_3d.astype(np.float32).copy()
                take_n = int(min(out.shape[0], control_indices.shape[0]))
                for i in range(take_n):
                    idx = int(control_indices[i])
                    if 0 <= idx < points3d_arr.shape[0]:
                        out[i] = points3d_arr[idx]
                return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            handle_3d_vis = np.nan_to_num(handle_3d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            target_3d_vis = np.nan_to_num(target_3d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            rotated_handle_3d_vis = _collect_control_positions(pts3d_new)
            if handle_3d_vis.shape[0] > 0 and target_3d_vis.shape[0] > 0:
                h0 = handle_3d_vis[0]
                t0 = target_3d_vis[0]
                rh0 = rotated_handle_3d_vis[0]
                print(
                    "[3D-Non-Rigid][PointCloudVis] Consistency check "
                    f"h0=({h0[0]:.2f},{h0[1]:.2f},{h0[2]:.2f}) "
                    f"t0=({t0[0]:.2f},{t0[1]:.2f},{t0[2]:.2f}) "
                    f"new_h0=({rh0[0]:.2f},{rh0[1]:.2f},{rh0[2]:.2f})"
                )
            anchor_pts3d_new = (
                pts3d_new[anchor_indices]
                if anchor_indices.size > 0
                else np.zeros((0, 3), dtype=np.float32)
            )
            rotate_3d_processor.save_3d_point_cloud_visualization(
                pts3d, pts3d_new, colors_rgb,
                "Y", 0.0, save_dir,
                centroid_3d=[0.0, 0.0, 0.0],
                handle_3d=handle_3d_vis,
                target_3d=target_3d_vis,
                handle_rotated_3d=rotated_handle_3d_vis,
                anchor_points_3d=anchor_pts3d,
                anchor_points_rotated_3d=anchor_pts3d_new,
                output_name=f"depth_05_3d_point_cloud_nonrigid_comp{comp_tag}.png",
                extra_rotated_points_3d=extra_rot_pts,
                extra_colors_rgb=extra_rot_cols,
                image_hw=(H_img, W_img),
                image_center_xy=(cx, cy),
            )
            if pts3d_compare_sgf is not None:
                sgf_handle_3d_vis = _collect_control_positions(pts3d_compare_sgf)
                anchor_pts3d_sgf = (
                    pts3d_compare_sgf[anchor_indices]
                    if anchor_indices.size > 0
                    else np.zeros((0, 3), dtype=np.float32)
                )
                rotate_3d_processor.save_3d_point_cloud_visualization(
                    pts3d, pts3d_compare_sgf, colors_rgb,
                    "Y", 0.0, save_dir,
                    centroid_3d=[0.0, 0.0, 0.0],
                    handle_3d=handle_3d_vis,
                    target_3d=target_3d_vis,
                    handle_rotated_3d=sgf_handle_3d_vis,
                    anchor_points_3d=anchor_pts3d,
                    anchor_points_rotated_3d=anchor_pts3d_sgf,
                    third_points_3d=pts3d_new,
                    handle_third_3d=rotated_handle_3d_vis,
                    target_third_3d=target_3d_vis,
                    third_row_title="Rigid+Residual",
                    output_name=f"depth_05_3d_point_cloud_nonrigid_compare_comp{comp_tag}.png",
                    image_hw=(H_img, W_img),
                    image_center_xy=(cx, cy),
                )
        except Exception as e:
            print(f"[3D-Non-Rigid] Point cloud vis failed: {e}")

    mean_abs_dz = float(np.mean(np.abs(delta_3d[:, 2]))) if delta_3d.size > 0 else 0.0
    mean_abs_dz_residual = float(np.mean(np.abs(delta_field_3d[:, 2]))) if delta_field_3d.size > 0 else 0.0
    anchor_strategy_name = str(anchor_select_info.get('strategy_name', 'Centroid+HandleInvariant'))
    anchor_drift_before = float(anchor_lock_info.get('anchor_drift_mean_before', 0.0))
    anchor_drift_after = float(anchor_lock_info.get('anchor_drift_mean_after', 0.0))
    solver_tag = "Rigid+Residual" if nonrigid_solver_mode == "rigid_plus_residual" else "SGFKernel"
    rigid_rmse = 0.0
    if bool(rigid_residual_info.get('enabled', False)):
        rigid_rmse = float((rigid_residual_info.get('fit_info') or {}).get('rmse', 0.0))
    anchors_desc = (
        f"3D anchors={len(anchors_img)}, strategy={anchor_strategy_name}, "
        f"radius3d={influence_radius:.1f}, mean|dz|={mean_abs_dz:.2f}, "
        f"res|dz|={mean_abs_dz_residual:.2f}, solver={solver_tag}, "
        f"anchorDrift={anchor_drift_before:.2f}->{anchor_drift_after:.2f}"
    )
    if bool(rigid_residual_info.get('enabled', False)):
        anchors_desc += (
            f", rigidRMSE={rigid_rmse:.2f}, "
            f"resMean={float(rigid_residual_info.get('residual_mean', 0.0)):.2f}"
        )
    if bool(compare_sgf_info.get('enabled', False)):
        anchors_desc += f", sgfDelta={float(compare_sgf_info.get('sgf_vs_tdedit_mean_l2', 0.0)):.2f}"
    # 非 Hybrid 的 3D-Non-Rigid 没有 phase2 前向轴；保留安全默认值避免 NameError。
    front_axis_3d = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    front_axis_info = {
        'enabled': False,
        'reason': 'nonrigid_no_phase2_axis',
    }
    debug_info.update({
        'intent_type': '3D_LOCAL_DEFORM',
        'sub_action': f"3D_LOCAL_DEFORM(R={influence_ratio:.2f})",
        'method_desc': anchors_desc,
        'deformation_space': '3D',
        'warp_backend': 'PROJECTIVE_ZBUFFER_3D',
        'anchors_img': anchors_img,
        'comp_sam_pts': comp_sam_pts.astype(np.float32),
        'comp_targets_xy': comp_targets_xy.astype(np.float32),
        'norm_grid': norm_grid,
        'pivot_img': None,
        'warped_latents': warped_latents.clone(),
        'mask_end': comp_mask_end.clone(),
        'm_end_full': m_end_full,
        'rotated_mask': rotated_mask_u8,
        'rotated_rgb': warped_rgb_np,
        'rotated_depth': warped_depth_np,
        'subject_scope_mask_full': np.asarray((proj_debug or {}).get('subject_scope_mask_full', np.zeros((H_img, W_img), dtype=np.float32)), dtype=np.float32),
        'subject_hole_mask_full': np.asarray((proj_debug or {}).get('subject_hole_mask_full', np.zeros((H_img, W_img), dtype=np.float32)), dtype=np.float32),
        'subject_scope_mask_lat': np.asarray((proj_debug or {}).get('subject_scope_mask_lat', np.zeros((H_lat, W_lat), dtype=np.float32)), dtype=np.float32),
        'subject_hole_mask_lat': np.asarray((proj_debug or {}).get('subject_hole_mask_lat', np.zeros((H_lat, W_lat), dtype=np.float32)), dtype=np.float32),
        'sam_mask': filtered_mask_u8,
        'depth_map': depth_map,
        'depth_filter_info': depth_filter_info,
        'z_center_info': z_center_info,
        'z_scale_factor': float(z_scale_factor),
        'z_scale': float(z_scale),
        'z_centroid': float(z_centroid),
        'target_depth_sampling': target_depth_sampling,
        'mean_abs_dz': mean_abs_dz,
        'mean_abs_dz_residual': mean_abs_dz_residual,
        'z_metric_weight': z_metric_weight,
        'sigma_gate_scale': float(sigma_gate_scale),
        'sigma_gate': float(sigma_gate),
        'influence_radius_3d': influence_radius,
        'influence_ratio': influence_ratio,
        'nonrigid_solver_mode': nonrigid_solver_mode,
        'nonrigid_use_rigid_residual': bool(use_rigid_residual),
        'nonrigid_rigid_residual_info': rigid_residual_info,
        'nonrigid_compare_sgf': compare_sgf_info,
        'nonrigid_local_field_info': local_field_info,
        'anchor_strategy': str(anchor_select_info.get('strategy', 'ROTATION_INVARIANT_CENTROID_HANDLE')),
        'anchor_select_info': anchor_select_info,
        'phase2_front_axis_3d': np.asarray(front_axis_3d, dtype=np.float32).reshape(3).tolist(),
        'phase2_front_axis_info': front_axis_info,
        'anchor_hard_constraint_count': 0,
        'anchor_soft_constraint_count': int(anchor_indices.size),
        'anchor_soft_strength': float(anchor_soft_strength),
        'anchor_soft_total_mass': float(anchor_soft_total_mass),
        'anchor_lock_info': anchor_lock_info,
        'anchor_lock_gain': float(anchor_lock_gain),
        'anchor_lock_snap': float(anchor_lock_snap),
        'runtime_scale': float(runtime_scale),
        'depth_sample_radius_px': int(sample_radius_runtime),
        'control_lock_info': control_lock_info,
        'control_lock_radius_px': float(control_lock_radius_px),
        'control_lock_falloff_px': float(control_lock_falloff_px),
        'control_lock_neighbors': {
            'min': int(min_neighbors_runtime),
            'max': int(max_neighbors_runtime),
        },
        'centroid': (int(round(cx)), int(round(cy))),
    })

    print(f"  [3D-Non-Rigid] {anchors_desc}")
    return warped_latents, comp_mask_end, "3D-Non-Rigid", None, anchors_img, debug_info


def process_3d_hybrid_unified_deformation(
    comp_sam_pts, comp_targets_xy, mask_for_calc,
    src_xy, tgt_xy, H_lat, W_lat, device,
    latents, comp_mask_start, comp_m_start_full,
    scale_x, scale_y, influence_ratio,
    depth_map, source_image_np=None, component_id=None,
    image_to_runtime_scale=1.0,
    anchor_strategy_3d=DEFAULT_THREE_D_ANCHOR_STRATEGY,
    enable_subject_scope_fill=False,
):
    """
    3D-Hybrid（统一3D空间，两阶段执行）:
    Phase 1: 3D刚性旋转
    Phase 2: 旋转后3D点云上的非刚性局部拉伸
    最终输出:
    - 两阶段完整 debug（用于两阶段可视化）
    - 合成后的最终网格（用于后续合成）
    """
    debug_info = {
        'm_start_full': comp_m_start_full,
        'component_id': component_id,
    }

    comp_sam_pts = np.asarray(comp_sam_pts, dtype=np.float32).reshape(-1, 2)
    comp_targets_xy = np.asarray(comp_targets_xy, dtype=np.float32).reshape(-1, 2)

    if isinstance(source_image_np, np.ndarray):
        H_img, W_img = source_image_np.shape[:2]
    else:
        H_img = _infer_image_size_from_latent_and_scale(H_lat, scale_y)
        W_img = _infer_image_size_from_latent_and_scale(W_lat, scale_x)
    # latent 域默认分辨率更低，直接按真实比例会过弱；保留 0.25 下限维持可达性。
    runtime_scale = float(np.clip(float(image_to_runtime_scale), 0.25, 1.0))

    m_u8 = (mask_for_calc * 255).astype(np.uint8) if mask_for_calc.dtype != np.uint8 else mask_for_calc.copy()
    if m_u8.max() <= 1:
        m_u8 = (m_u8 * 255).astype(np.uint8)

    handle_points_yx = []
    for pt in comp_sam_pts:
        hx = int(np.clip(pt[0], 0, W_img - 1))
        hy = int(np.clip(pt[1], 0, H_img - 1))
        handle_points_yx.append([hy, hx])

    filter_save_dir = None
    if rotate_3d_processor.ENABLE_3D_DEBUG:
        # 统一写到当前调试目录，直接覆盖同名文件，不再创建 depth_comp* 子目录
        filter_save_dir = rotate_3d_processor.ensure_debug_dir()

    try:
        filtered_mask_u8, depth_filter_info = rotate_3d_processor.filter_background_by_depth(
            m_u8, depth_map, handle_points_yx,
            distance_threshold=None,
            save_dir=filter_save_dir,
            source_image_np=source_image_np,
        )
        if np.sum(filtered_mask_u8 > 127) < 20:
            filtered_mask_u8 = m_u8
    except Exception as e:
        filtered_mask_u8 = m_u8
        depth_filter_info = {'filtered': False, 'reason': f'depth_filter_error:{e}'}

    ys, xs = np.where(filtered_mask_u8 > 127)
    if len(xs) == 0:
        debug_info.update({
            'intent_type': '3D_HYBRID_EMPTY_MASK',
            'sub_action': '3D_HYBRID_EMPTY_MASK',
            'anchors_img': np.zeros((0, 2), dtype=np.float32),
            'depth_filter_info': depth_filter_info,
            'sam_mask': filtered_mask_u8,
            'depth_map': depth_map,
        })
        return latents.clone(), comp_mask_start.clone(), "3D-Hybrid", None, np.zeros((0, 2), dtype=np.float32), debug_info

    depth_values = depth_map[ys, xs].astype(np.float32)
    depth_min = float(depth_values.min())
    depth_max = float(depth_values.max())
    depth_range = depth_max - depth_min + 1e-6

    cx = float(np.mean(xs))
    cy = float(np.mean(ys))

    x_range = max(float(xs.max() - xs.min()), 1.0)
    y_range = max(float(ys.max() - ys.min()), 1.0)
    xy_range = max(x_range, y_range)

    depth_std = float(np.std(depth_values))
    depth_mean = float(np.mean(depth_values))
    depth_cv = depth_std / (depth_mean + 1e-6)
    if depth_cv < 0.05:
        z_scale_factor = 0.4
    elif depth_cv < 0.10:
        z_scale_factor = 0.7
    elif depth_cv < 0.20:
        z_scale_factor = 1.0
    else:
        z_scale_factor = 1.5

    z_scale = xy_range * z_scale_factor
    z_centroid, z_center_info = _estimate_arc_axis_z_centroid(
        xs=xs,
        ys=ys,
        depth_values=depth_values,
        cx=cx,
        cy=cy,
        depth_min=depth_min,
        depth_range=depth_range,
        z_scale=z_scale,
        min_points=96,
    )

    xs_f = xs.astype(np.float32)
    ys_f = ys.astype(np.float32)
    zs = (depth_values - depth_min) / depth_range * z_scale - z_centroid
    pts3d = np.stack([xs_f - cx, -(ys_f - cy), zs], axis=1).astype(np.float32)
    old_xy = np.stack([xs_f, ys_f], axis=1).astype(np.float32)

    num_pairs = min(len(comp_sam_pts), len(comp_targets_xy))
    if num_pairs == 0:
        debug_info.update({
            'intent_type': '3D_HYBRID_EMPTY',
            'sub_action': '3D_HYBRID_EMPTY',
            'anchors_img': np.zeros((0, 2), dtype=np.float32),
            'depth_filter_info': depth_filter_info,
            'sam_mask': filtered_mask_u8,
            'depth_map': depth_map,
        })
        return latents.clone(), comp_mask_start.clone(), "3D-Hybrid", None, np.zeros((0, 2), dtype=np.float32), debug_info

    pair_src = comp_sam_pts[:num_pairs]
    pair_tgt = comp_targets_xy[:num_pairs]

    # Hybrid 约定：
    # - 多控制点时：Phase1 刚性仅使用第1对点估计旋转
    # - Phase2 非刚性仍使用全部点对
    phase1_pair_count = 1 if num_pairs > 1 else num_pairs
    phase1_src = pair_src[:phase1_pair_count]
    phase1_tgt = pair_tgt[:phase1_pair_count]
    phase1_ref_global_idx = 0
    ref_idx = 0
    ref_h = phase1_src[ref_idx]
    ref_t = phase1_tgt[ref_idx]
    if num_pairs > 1:
        print(
            f"  [3D-Hybrid] Multi-point policy: "
            f"Phase1 rigid uses first pair only, Phase2 non-rigid uses all {num_pairs} pairs"
        )

    # Hybrid 锚点策略：
    # - 在 Phase1 旋转前先选锚点（语义位置稳定）
    # - Phase2 直接复用这些索引，并映射到旋转后点云
    phase1_handle_3d_for_anchor = []
    for i in range(phase1_pair_count):
        hx, hy = float(phase1_src[i][0]), float(phase1_src[i][1])
        hxi = int(np.clip(round(hx), 0, W_img - 1))
        hyi = int(np.clip(round(hy), 0, H_img - 1))
        h_depth = float(depth_map[hyi, hxi])
        hz = (h_depth - depth_min) / depth_range * z_scale - z_centroid
        phase1_handle_3d_for_anchor.append([hx - cx, -(hy - cy), hz])
    phase1_handle_3d_for_anchor = np.asarray(phase1_handle_3d_for_anchor, dtype=np.float32).reshape(-1, 3)

    pre_control_indices = _map_xy_controls_to_indices(old_xy, phase1_src.astype(np.float32))
    pre_front_axis_3d = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    pre_anchors_img, pre_anchor_select_info = _select_3d_hybrid_phase2_anchor_points(
        points_3d=pts3d,
        points_xy=old_xy,
        handle_points_3d=phase1_handle_3d_for_anchor,
        handle_points_xy=phase1_src.astype(np.float32),
        target_points_xy=phase1_tgt.astype(np.float32),
        front_axis_3d=pre_front_axis_3d,
        influence_ratio=influence_ratio,
        max_anchors=220,
        exclude_indices=pre_control_indices,
        anchor_strategy_3d=anchor_strategy_3d,
    )
    pre_anchor_indices = _map_xy_controls_to_indices(old_xy, pre_anchors_img)
    if pre_anchor_indices.size > 0:
        if pre_control_indices.size > 0:
            pre_anchor_indices = pre_anchor_indices[~np.isin(pre_anchor_indices, pre_control_indices)]
        pre_anchor_indices = np.unique(pre_anchor_indices.astype(np.int32))
        pre_anchor_indices = pre_anchor_indices[(pre_anchor_indices >= 0) & (pre_anchor_indices < old_xy.shape[0])]
    else:
        pre_anchor_indices = np.zeros((0,), dtype=np.int32)
    pre_anchor_select_info = dict(pre_anchor_select_info)
    pre_anchor_select_info.update({
        'phase2_anchor_selection_timing': 'before_phase1_rotation',
        'phase2_anchor_selection_space': 'pre_rotation_pointcloud',
        'phase2_anchor_pre_count': int(pre_anchor_indices.size),
    })

    # Phase1 直接用 3D 向量对齐来构造旋转矩阵，避免经验 yaw/pitch 过旋。
    ref_hx = int(np.clip(round(float(ref_h[0])), 0, W_img - 1))
    ref_hy = int(np.clip(round(float(ref_h[1])), 0, H_img - 1))
    ref_h_depth = float(depth_map[ref_hy, ref_hx])
    ref_h_z = (ref_h_depth - depth_min) / depth_range * z_scale - z_centroid
    ref_h_3d = np.array([float(ref_h[0] - cx), -float(ref_h[1] - cy), ref_h_z], dtype=np.float64)
    # 目标深度默认与 handle 一致，保证“同一参考平面”的刚性对齐。
    ref_t_3d = np.array([float(ref_t[0] - cx), -float(ref_t[1] - cy), ref_h_z], dtype=np.float64)

    R_direct, direct_info = _rotation_matrix_from_vec_to_vec(ref_h_3d, ref_t_3d)
    if (not np.isfinite(R_direct).all()) or (not bool(direct_info.get('ok', True))):
        R = np.eye(3, dtype=np.float32)
        rotation_axis = '3D-Direct'
        rotation_angle = 0.0
        rotation_yaw_deg, rotation_pitch_deg = 0.0, 0.0
        angle_debug_info = {
            'phase1_rotation_solver': 'identity_fallback',
            'phase1_rotation_from_vec_to_vec': direct_info,
            'handle_ref_3d': ref_h_3d.tolist(),
            'target_ref_3d': ref_t_3d.tolist(),
            'yaw_deg': 0.0,
            'pitch_deg': 0.0,
        }
        print(
            "  [3D-Hybrid] Phase1 direct rotation fallback: identity "
            f"(reason={direct_info.get('reason', 'unknown')})"
        )
    else:
        R = R_direct.astype(np.float32)
        rotation_axis = '3D-Direct'
        rotation_angle = float(direct_info.get('angle_deg', 0.0))
        rotation_yaw_deg, rotation_pitch_deg = _matrix_to_yaw_pitch_deg(R)
        h3_after = (R @ ref_h_3d.astype(np.float32)).astype(np.float32)
        align_err = float(np.linalg.norm(h3_after - ref_t_3d.astype(np.float32)))
        angle_debug_info = {
            'phase1_rotation_solver': 'vec_to_vec_min_rotation',
            'phase1_rotation_from_vec_to_vec': direct_info,
            'handle_ref_3d': ref_h_3d.tolist(),
            'target_ref_3d': ref_t_3d.tolist(),
            'handle_after_rotation_3d': h3_after.tolist(),
            'handle_target_alignment_error': align_err,
            'yaw_deg': float(rotation_yaw_deg),
            'pitch_deg': float(rotation_pitch_deg),
        }
        print(
            f"  [3D-Hybrid] Phase1 direct rotation: "
            f"angle={rotation_angle:.2f}deg, align_err={align_err:.3f}"
        )
    t_phase1 = np.zeros((3,), dtype=np.float32)

    # ===== Phase 1: 3D刚体变换 =====
    pts3d_rot = (R @ pts3d.T).T
    rot_x = np.clip(pts3d_rot[:, 0] + cx, 0, W_img - 1)
    rot_y = np.clip(-pts3d_rot[:, 1] + cy, 0, H_img - 1)
    rot_xy = np.stack([rot_x, rot_y], axis=1).astype(np.float32)

    handle_3d_orig = []
    handle_3d_rot = []
    rotated_handles_xy = []
    for i in range(num_pairs):
        hx, hy = float(pair_src[i][0]), float(pair_src[i][1])
        hxi = int(np.clip(round(hx), 0, W_img - 1))
        hyi = int(np.clip(round(hy), 0, H_img - 1))
        h_depth = float(depth_map[hyi, hxi])
        hz = (h_depth - depth_min) / depth_range * z_scale - z_centroid
        h3 = np.array([hx - cx, -(hy - cy), hz], dtype=np.float32)
        h3_rot = (R @ h3).astype(np.float32)
        handle_3d_orig.append(h3)
        handle_3d_rot.append(h3_rot)
        rotated_handles_xy.append([h3_rot[0] + cx, -h3_rot[1] + cy])

    handle_3d_orig = np.stack(handle_3d_orig, axis=0)
    handle_3d_rot = np.stack(handle_3d_rot, axis=0)
    rotated_handles_xy = np.asarray(rotated_handles_xy, dtype=np.float32)

    # 强约束：Phase1 刚性阶段仅锁定 phase1_pair_count 对控制点（多点时仅第一对）。
    for i in range(phase1_pair_count):
        sx, sy = float(pair_src[i][0]), float(pair_src[i][1])
        d2 = (old_xy[:, 0] - sx) ** 2 + (old_xy[:, 1] - sy) ** 2
        if d2.size == 0:
            continue
        nearest_idx = int(np.argmin(d2))
        pts3d_rot[nearest_idx] = handle_3d_rot[i]
        rot_xy[nearest_idx] = rotated_handles_xy[i]

    p1_new_z = pts3d_rot[:, 2].astype(np.float32)
    phase2_src_xy = rotated_handles_xy.astype(np.float32).copy()
    phase2_src_3d = handle_3d_rot.astype(np.float32).copy()
    phase2_src_indices = _map_xy_controls_to_indices(rot_xy, phase2_src_xy)
    print("  [3D-Hybrid] Phase2 control mode: direct rotated handles (no visibility matching)")

    need_image_outputs = bool(rotate_3d_processor.ENABLE_3D_DEBUG)
    latents_p1, comp_mask_p1, norm_grid_p1, m_p1_full, phase1_rgb, phase1_depth, rotated_mask_p1_u8, p1_proj_debug = (
        _projective_3d_warp_from_points(
            latents=latents,
            H_lat=H_lat,
            W_lat=W_lat,
            device=device,
            old_xy=old_xy,
            new_xy=rot_xy,
            new_z=p1_new_z,
            scale_x=scale_x,
            scale_y=scale_y,
            source_image_np=source_image_np,
            mask_start_full=comp_m_start_full,
            need_image_outputs=need_image_outputs,
            enable_subject_scope_fill=enable_subject_scope_fill,
        )
    )
    if not isinstance(m_p1_full, np.ndarray):
        m_p1_full = _latent_mask_to_fullres(comp_mask_p1, target_hw=(H_img, W_img))
    if not isinstance(rotated_mask_p1_u8, np.ndarray):
        rotated_mask_p1_u8 = (m_p1_full > 0.5).astype(np.uint8) * 255

    phase1_debug = {
        'component_id': component_id,
        'intent_type': '3D_RIGID_ROTATION',
        'sub_action': f"Yaw={rotation_yaw_deg:.1f}deg,Pitch={rotation_pitch_deg:.1f}deg",
        'deformation_space': '3D',
        'warp_backend': 'PROJECTIVE_ZBUFFER_3D',
        'comp_sam_pts': phase1_src.astype(np.float32),
        'comp_targets_xy': phase1_tgt.astype(np.float32),
        'rotated_handle_points_xy': rotated_handles_xy[:phase1_pair_count].astype(np.float32),
        'norm_grid': norm_grid_p1,
        'warped_latents': latents_p1.clone(),
        'mask_end': comp_mask_p1.clone(),
        'm_start_full': comp_m_start_full,
        'm_end_full': m_p1_full,
        'rotated_mask': rotated_mask_p1_u8,
        'rotated_rgb': phase1_rgb,
        'rotated_depth': phase1_depth,
        'subject_scope_mask_full': np.asarray((p1_proj_debug or {}).get('subject_scope_mask_full', np.zeros((H_img, W_img), dtype=np.float32)), dtype=np.float32),
        'subject_hole_mask_full': np.asarray((p1_proj_debug or {}).get('subject_hole_mask_full', np.zeros((H_img, W_img), dtype=np.float32)), dtype=np.float32),
        'subject_scope_mask_lat': np.asarray((p1_proj_debug or {}).get('subject_scope_mask_lat', np.zeros((H_lat, W_lat), dtype=np.float32)), dtype=np.float32),
        'subject_hole_mask_lat': np.asarray((p1_proj_debug or {}).get('subject_hole_mask_lat', np.zeros((H_lat, W_lat), dtype=np.float32)), dtype=np.float32),
        'sam_mask': filtered_mask_u8,
        'depth_map': depth_map,
        'depth_filter_info': depth_filter_info,
        'z_center_info': z_center_info,
        'z_scale_factor': float(z_scale_factor),
        'z_scale': float(z_scale),
        'z_centroid': float(z_centroid),
        'centroid': (int(round(cx)), int(round(cy))),
        'rotation_axis': rotation_axis,
        'rotation_angle_deg': float(rotation_angle),
        'rotation_yaw_deg': rotation_yaw_deg,
        'rotation_pitch_deg': rotation_pitch_deg,
        'phase1_translation_3d': t_phase1.tolist(),
        'phase1_rigid_pair_mode': 'first_only' if num_pairs > 1 else 'all',
        'phase1_rigid_pair_count': int(phase1_pair_count),
        'phase2_nonrigid_pair_count': int(num_pairs),
        'phase1_ref_global_idx': int(phase1_ref_global_idx),
        'angle_computation': angle_debug_info,
    }

    # ===== Phase 2: 旋转后点云上的3D非刚性 =====
    delta_3d = []
    target_depth_sampling_p2 = []
    phase2_active_mask = []
    control_indices_raw = np.asarray(phase2_src_indices, dtype=np.int32).reshape(-1)
    control_indices_valid = control_indices_raw[
        (control_indices_raw >= 0) & (control_indices_raw < rot_xy.shape[0])
    ]
    for i in range(num_pairs):
        tx, ty = float(pair_tgt[i][0]), float(pair_tgt[i][1])
        handle_z = float(phase2_src_3d[i][2])

        sampled_z = float(handle_z)
        sample_xy = (int(round(tx)), int(round(ty)))
        sample_dist = -1.0
        if rot_xy.shape[0] > 0:
            d2 = (rot_xy[:, 0] - tx) ** 2 + (rot_xy[:, 1] - ty) ** 2
            nearest = int(np.argmin(d2))
            sampled_z = float(pts3d_rot[nearest, 2])
            sample_xy = (
                int(round(float(rot_xy[nearest, 0]))),
                int(round(float(rot_xy[nearest, 1]))),
            )
            sample_dist = float(np.sqrt(float(d2[nearest])))

        # Hybrid phase2 不再用 z 门控决定是否执行，所有控制点都继续追目标。
        target_z = float(sampled_z)
        phase2_enabled = True
        phase2_active_mask.append(phase2_enabled)

        target_depth_sampling_p2.append({
            'pair_idx': int(i),
            'sample_xy': (int(sample_xy[0]), int(sample_xy[1])),
            'sample_dist_px': float(sample_dist),
            'mode': 'sample_target_z_no_gate',
            'rotated_handle_z': float(handle_z),
            'sampled_target_z': float(sampled_z),
            'final_target_z': float(target_z),
            'phase2_nonrigid_enabled': bool(phase2_enabled),
        })

        t3 = np.array([tx - cx, -(ty - cy), target_z], dtype=np.float32)
        delta_3d.append(t3 - phase2_src_3d[i])
    delta_3d = np.stack(delta_3d, axis=0)
    phase2_active_mask = np.asarray(phase2_active_mask, dtype=bool)
    phase2_active_count = int(np.sum(phase2_active_mask))
    print(f"  [3D-Hybrid] Phase2 active pairs: {phase2_active_count}/{num_pairs}")

    front_axis_3d, front_axis_info = _estimate_hybrid_phase2_front_axis_3d(
        points_3d=pts3d_rot,
        control_points_3d=phase2_src_3d[:num_pairs],
        control_points_xy=phase2_src_xy[:num_pairs],
        target_points_xy=pair_tgt[:num_pairs],
    )
    anchor_select_info = dict(pre_anchor_select_info)
    anchor_select_info['front_axis_info'] = front_axis_info
    anchor_select_info['phase2_anchor_mapping_mode'] = 'pre_selected_indices_rotate_with_phase1'

    influence_ratio = float(np.clip(influence_ratio, 0.0, 1.0))
    obj_diag_3d = float(np.sqrt(x_range ** 2 + y_range ** 2 + z_scale ** 2) + 1e-6)
    influence_radius = (0.12 + 0.88 * influence_ratio) * obj_diag_3d
    sigma = max(influence_radius * 0.45, 1e-3)
    anchor_indices = np.asarray(pre_anchor_indices, dtype=np.int32).reshape(-1)
    if anchor_indices.size > 0:
        if control_indices_valid.size > 0:
            anchor_indices = anchor_indices[~np.isin(anchor_indices, control_indices_valid)]
        anchor_indices = np.unique(anchor_indices.astype(np.int32))
        anchor_indices = anchor_indices[(anchor_indices >= 0) & (anchor_indices < rot_xy.shape[0])]
    if anchor_indices.size > 0:
        anchors_img = rot_xy[anchor_indices].astype(np.float32)
    else:
        anchors_img = np.zeros((0, 2), dtype=np.float32)
    anchor_pts3d = pts3d_rot[anchor_indices] if anchor_indices.size > 0 else np.zeros((0, 3), dtype=np.float32)
    target_3d_p2 = (phase2_src_3d + delta_3d).astype(np.float32)

    # Hybrid Phase2 目标：旋转后保持主体整体位置，优先做局部拉伸。
    # 因此禁用锚点残差跟随，避免锚点把全局平移传回主体。
    anchor_follow_gain = 0.0
    anchor_follow_delta = np.zeros_like(anchor_pts3d, dtype=np.float32)
    anchor_follow_info = {
        'enabled': False,
        'reason': 'disabled_for_local_stretch',
        'gain': float(anchor_follow_gain),
    }
    if anchor_pts3d.shape[0] > 0 and anchor_follow_gain > 1e-6:
        active_idx = np.where(phase2_active_mask[:num_pairs])[0].astype(np.int32)
        if active_idx.size == 0:
            active_idx = np.arange(num_pairs, dtype=np.int32)
        fit_src = phase2_src_3d[active_idx].astype(np.float64)
        fit_tgt = target_3d_p2[active_idx].astype(np.float64)
        fit_motion = np.linalg.norm((fit_tgt - fit_src)[:, :2], axis=1).astype(np.float64)
        R_res, t_res, fit_info = _fit_rigid_transform_kabsch_3d(
            src_pts=fit_src,
            tgt_pts=fit_tgt,
            weights=fit_motion,
        )
        if bool(fit_info.get('ok', False)):
            anchor_target_follow = (R_res @ anchor_pts3d.astype(np.float64).T).T + t_res[None, :]
            anchor_follow_delta = (anchor_target_follow - anchor_pts3d.astype(np.float64)).astype(np.float32)
            anchor_follow_delta *= float(anchor_follow_gain)
            anchor_follow_info = {
                'enabled': True,
                'reason': 'kabsch_residual_follow',
                'gain': float(anchor_follow_gain),
                'fit_info': fit_info,
                'translation_3d': np.asarray(t_res, dtype=np.float64).tolist(),
            }
        else:
            mean_shift = np.mean((target_3d_p2 - phase2_src_3d), axis=0).astype(np.float32)
            anchor_follow_delta = np.repeat(
                (mean_shift[None, :] * float(anchor_follow_gain)).astype(np.float32),
                anchor_pts3d.shape[0],
                axis=0,
            )
            anchor_follow_info = {
                'enabled': True,
                'reason': 'mean_shift_fallback',
                'gain': float(anchor_follow_gain),
                'fit_info': fit_info,
                'translation_3d': mean_shift.astype(np.float32).tolist(),
            }

    z_metric_weight = 1.35
    anchor_soft_strength = 0.0
    anchor_soft_total_mass = 0.0
    if anchor_pts3d.shape[0] > 0:
        handle_n = int(phase2_src_3d.shape[0])
        anchor_n = int(anchor_pts3d.shape[0])
        # 备份版锚点软约束强度（更弱，减少对拖拽质量的影响）
        anchor_total_ratio = float(np.clip(0.35 + 0.45 * (1.0 - influence_ratio), 0.20, 0.80))
        anchor_soft_total_mass = anchor_total_ratio * float(max(handle_n, 1))
        anchor_soft_strength = float(anchor_soft_total_mass / float(max(anchor_n, 1)))
        anchor_soft_strength = float(np.clip(anchor_soft_strength, 0.004, 0.22))
        constraint_pts3d = np.concatenate([phase2_src_3d, anchor_pts3d], axis=0)
        constraint_delta3d = np.concatenate([delta_3d, anchor_follow_delta], axis=0)
        constraint_w = np.concatenate(
            [
                np.ones((phase2_src_3d.shape[0],), dtype=np.float32),
                np.full((anchor_pts3d.shape[0],), anchor_soft_strength, dtype=np.float32),
            ],
            axis=0,
        )
    else:
        constraint_pts3d = phase2_src_3d
        constraint_delta3d = delta_3d
        constraint_w = np.ones((phase2_src_3d.shape[0],), dtype=np.float32)

    diff = pts3d_rot[:, None, :] - constraint_pts3d[None, :, :]
    diff_metric = diff.copy()
    diff_metric[:, :, 2] *= z_metric_weight
    dists = np.linalg.norm(diff_metric, axis=2)
    weights = np.exp(-(dists ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    weights *= constraint_w[None, :]
    sum_w = np.sum(weights, axis=1, keepdims=True) + 1e-6
    local_weights = weights / sum_w
    disp3d = local_weights @ constraint_delta3d

    diff_ctrl = pts3d_rot[:, None, :] - phase2_src_3d[None, :, :]
    diff_ctrl[:, :, 2] *= z_metric_weight
    min_ctrl_dist = np.min(np.linalg.norm(diff_ctrl, axis=2), axis=1)
    sigma_gate_scale = 0.69 if anchor_pts3d.shape[0] > 0 else 0.75
    sigma_gate = max(influence_radius * sigma_gate_scale, 1e-3)
    gate = np.exp(-(min_ctrl_dist ** 2) / (2.0 * sigma_gate ** 2)).astype(np.float32)
    disp3d *= gate[:, None]

    pts3d_final = pts3d_rot + disp3d
    phase2_point_delta = np.linalg.norm((pts3d_final - pts3d_rot).astype(np.float32), axis=1)
    phase2_point_delta_mean = float(np.mean(phase2_point_delta)) if phase2_point_delta.size > 0 else 0.0
    phase2_point_delta_max = float(np.max(phase2_point_delta)) if phase2_point_delta.size > 0 else 0.0
    if phase2_point_delta_max <= 1e-4:
        print("  [3D-Hybrid] Phase2 residual is near zero; rotated/projected views can look identical.")
    phase2_anchor_frame_shift = np.zeros((3,), dtype=np.float32)
    anchor_lock_gain = 0.0
    anchor_lock_snap = 0.0
    anchor_lock_info = {
        'enabled': False,
        'reason': 'disabled_use_backup_soft_anchor',
        'hard_locked_anchor_count': 0,
        'anchor_drift_mean_before': 0.0,
        'anchor_drift_mean_after': 0.0,
    }
    target_xy_p2 = pair_tgt[:num_pairs].copy()
    control_lock_radius_img = float(np.clip(6.5 + 9.0 * influence_ratio, 6.5, 16.0))
    runtime_lock_scale = float(max(np.sqrt(runtime_scale), 0.45))
    control_lock_radius_px = float(np.clip(control_lock_radius_img * runtime_lock_scale, 3.0, 16.0))
    control_lock_falloff_px = float(np.clip(control_lock_radius_px * 0.55, 1.5, 8.5))
    neighbor_scale = float(max(np.sqrt(runtime_scale), 0.50))
    min_neighbors_runtime = int(np.clip(round(12.0 * neighbor_scale), 4, 18))
    max_neighbors_runtime = int(np.clip(round(110.0 * neighbor_scale), max(min_neighbors_runtime, 24), 120))
    control_global_follow_gain = 0.0
    pts3d_final, control_lock_info = _enforce_control_targets_on_points_3d(
        pts3d_original=pts3d_rot,
        pts3d_deformed=pts3d_final,
        points_xy=rot_xy,
        handle_points_xy=phase2_src_xy[:num_pairs],
        target_points_xy=target_xy_p2,
        target_points_3d=target_3d_p2,
        control_indices=control_indices_raw,
        pair_active_mask=phase2_active_mask[:num_pairs],
        radius_px=control_lock_radius_px,
        min_neighbors=min_neighbors_runtime,
        max_neighbors=max_neighbors_runtime,
        falloff_px=control_lock_falloff_px,
        blend_strength=1.0,
        rigid_core_ratio=0.35,
        rigid_core_min_px=2.5,
        global_follow_gain=float(control_global_follow_gain),
    )
    anchor_guard_info = {
        'enabled': False,
        'reason': 'disabled_restore_minimal_hybrid',
        'guard_anchor_count': 0,
    }
    final_x = np.clip(pts3d_final[:, 0] + cx, 0, W_img - 1)
    final_y = np.clip(-pts3d_final[:, 1] + cy, 0, H_img - 1)
    final_xy = np.stack([final_x, final_y], axis=1).astype(np.float32)

    source_p2 = phase1_rgb if isinstance(phase1_rgb, np.ndarray) else source_image_np
    p2_start_mask = m_p1_full if isinstance(m_p1_full, np.ndarray) else comp_m_start_full
    p2_new_z = pts3d_final[:, 2].astype(np.float32)
    latents_final, comp_mask_final, norm_grid_p2, m_end_full, phase2_rgb, phase2_depth, rotated_mask_p2_u8, p2_proj_debug = (
        _projective_3d_warp_from_points(
            latents=latents_p1,
            H_lat=H_lat,
            W_lat=W_lat,
            device=device,
            old_xy=rot_xy,
            new_xy=final_xy,
            new_z=p2_new_z,
            scale_x=scale_x,
            scale_y=scale_y,
            source_image_np=source_p2,
            mask_start_full=p2_start_mask,
            need_image_outputs=need_image_outputs,
            enable_subject_scope_fill=enable_subject_scope_fill,
        )
    )

    mean_abs_dz_p2 = float(np.mean(np.abs(delta_3d[:, 2]))) if delta_3d.size > 0 else 0.0
    phase2_anchor_strategy_name = str(anchor_select_info.get('strategy_name', 'Centroid+HandleInvariant'))
    anchor_drift_before_p2 = float(anchor_lock_info.get('anchor_drift_mean_before', 0.0))
    anchor_drift_after_p2 = float(anchor_lock_info.get('anchor_drift_mean_after', 0.0))
    phase2_desc = (
        f"3D anchors={len(anchors_img)}, strategy={phase2_anchor_strategy_name}, "
        f"radius3d={influence_radius:.1f}, mean|dz|={mean_abs_dz_p2:.2f}, "
        f"active={phase2_active_count}/{num_pairs}, "
        f"anchorFollow={anchor_follow_gain:.2f}, "
        f"anchorFrameShiftXY=({phase2_anchor_frame_shift[0]:.2f},{phase2_anchor_frame_shift[1]:.2f}), "
        f"anchorDrift={anchor_drift_before_p2:.2f}->{anchor_drift_after_p2:.2f}"
    )
    phase2_debug = {
        'component_id': component_id,
        'intent_type': '3D_LOCAL_DEFORM',
        'sub_action': f"3D_LOCAL_DEFORM(R={influence_ratio:.2f})",
        'method_desc': phase2_desc,
        'deformation_space': '3D',
        'warp_backend': 'PROJECTIVE_ZBUFFER_3D',
        'anchors_img': anchors_img,
        'comp_sam_pts': phase2_src_xy.astype(np.float32),
        'comp_targets_xy': pair_tgt.astype(np.float32),
        'norm_grid': norm_grid_p2,
        'pivot_img': None,
        'warped_latents': latents_final.clone(),
        'mask_end': comp_mask_final.clone(),
        'm_start_full': m_p1_full,
        'm_end_full': m_end_full,
        'rotated_mask': rotated_mask_p2_u8,
        'rotated_rgb': phase2_rgb,
        'rotated_depth': phase2_depth,
        'subject_scope_mask_full': np.asarray((p2_proj_debug or {}).get('subject_scope_mask_full', np.zeros((H_img, W_img), dtype=np.float32)), dtype=np.float32),
        'subject_hole_mask_full': np.asarray((p2_proj_debug or {}).get('subject_hole_mask_full', np.zeros((H_img, W_img), dtype=np.float32)), dtype=np.float32),
        'subject_scope_mask_lat': np.asarray((p2_proj_debug or {}).get('subject_scope_mask_lat', np.zeros((H_lat, W_lat), dtype=np.float32)), dtype=np.float32),
        'subject_hole_mask_lat': np.asarray((p2_proj_debug or {}).get('subject_hole_mask_lat', np.zeros((H_lat, W_lat), dtype=np.float32)), dtype=np.float32),
        'sam_mask': (m_p1_full * 255).astype(np.uint8),
        'depth_map': phase1_depth if isinstance(phase1_depth, np.ndarray) else depth_map,
        'depth_filter_info': depth_filter_info,
        'z_center_info': z_center_info,
        'z_scale_factor': float(z_scale_factor),
        'z_scale': float(z_scale),
        'z_centroid': float(z_centroid),
        'target_depth_sampling': target_depth_sampling_p2,
        'phase2_target_depth_sampling_enabled': True,
        'phase2_target_depth_mode': 'sample_target_z_no_activation_gate',
        'phase2_nonrigid_active_pairs': int(phase2_active_count),
        'phase2_nonrigid_active_mask': phase2_active_mask.tolist(),
        'mean_abs_dz': mean_abs_dz_p2,
        'z_metric_weight': z_metric_weight,
        'sigma_gate_scale': float(sigma_gate_scale),
        'sigma_gate': float(sigma_gate),
        'influence_radius_3d': influence_radius,
        'influence_ratio': influence_ratio,
        'anchor_strategy': str(anchor_select_info.get('strategy', 'ROTATION_INVARIANT_CENTROID_HANDLE')),
        'anchor_select_info': anchor_select_info,
        'anchor_hard_constraint_count': 0,
        'anchor_soft_constraint_count': int(anchor_indices.size),
        'anchor_soft_strength': float(anchor_soft_strength),
        'anchor_soft_total_mass': float(anchor_soft_total_mass),
        'anchor_follow_gain': float(anchor_follow_gain),
        'anchor_follow_info': anchor_follow_info,
        'phase2_anchor_frame_shift_3d': phase2_anchor_frame_shift.astype(np.float32).tolist(),
        'anchor_lock_info': anchor_lock_info,
        'anchor_guard_info': anchor_guard_info,
        'anchor_lock_gain': float(anchor_lock_gain),
        'anchor_lock_snap': float(anchor_lock_snap),
        'runtime_scale': float(runtime_scale),
        'control_lock_info': control_lock_info,
        'control_global_follow_gain': float(control_global_follow_gain),
        'control_lock_radius_px': float(control_lock_radius_px),
        'control_lock_falloff_px': float(control_lock_falloff_px),
        'control_lock_neighbors': {
            'min': int(min_neighbors_runtime),
            'max': int(max_neighbors_runtime),
        },
        'centroid': (int(round(cx)), int(round(cy))),
        'rotation_axis': rotation_axis,
        'rotation_angle_deg': float(rotation_angle),
        'rotation_yaw_deg': rotation_yaw_deg,
        'rotation_pitch_deg': rotation_pitch_deg,
        'phase2_source_points_xy_raw': rotated_handles_xy.astype(np.float32),
        'phase2_source_indices': phase2_src_indices,
        'phase2_point_delta_mean': float(phase2_point_delta_mean),
        'phase2_point_delta_max': float(phase2_point_delta_max),
    }

    # 点云可视化：Hybrid 输出单张“三行”点云图（原始->刚性后->非刚性后）
    save_dir = rotate_3d_processor.ensure_debug_dir()
    if rotate_3d_processor.ENABLE_3D_DEBUG and save_dir is not None:
        comp_tag = int(component_id) + 1 if component_id is not None else 1
        if isinstance(source_image_np, np.ndarray) and source_image_np.shape[0] >= H_img and source_image_np.shape[1] >= W_img:
            colors_rgb = source_image_np[ys, xs].astype(np.float32) / 255.0
        else:
            z_norm = (zs - float(np.min(zs))) / (float(np.max(zs) - np.min(zs)) + 1e-6)
            colors_rgb = np.stack([z_norm, z_norm, z_norm], axis=1).astype(np.float32)
        extra_rot_p1, extra_cols_p1 = (None, None)
        extra_rot_p2, extra_cols_p2 = (None, None)
        try:
            phase1_handle_3d_vis = np.nan_to_num(handle_3d_orig, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            phase1_target_3d_vis = np.stack(
                [
                    pair_tgt[:, 0] - cx,
                    -(pair_tgt[:, 1] - cy),
                    handle_3d_orig[:, 2],
                ],
                axis=1,
            ).astype(np.float32)
            phase1_target_3d_vis = np.nan_to_num(phase1_target_3d_vis, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            phase1_rotated_handle_3d_vis = np.nan_to_num(handle_3d_rot, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            phase2_handle_3d_vis = np.nan_to_num(phase2_src_3d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            phase2_target_3d_vis = np.nan_to_num(target_3d_p2, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            phase2_rotated_handle_3d_vis = phase2_target_3d_vis.copy()
            for i, nearest_idx in enumerate(control_indices_raw):
                if i >= phase2_rotated_handle_3d_vis.shape[0]:
                    break
                nidx = int(nearest_idx)
                if 0 <= nidx < pts3d_final.shape[0]:
                    phase2_rotated_handle_3d_vis[i] = pts3d_final[nidx]
            phase1_anchor_3d_vis = (
                np.nan_to_num(pts3d[pre_anchor_indices], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                if pre_anchor_indices.size > 0
                else np.zeros((0, 3), dtype=np.float32)
            )
            phase2_anchor_3d_vis = (
                np.nan_to_num(anchor_pts3d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                if anchor_pts3d.shape[0] > 0
                else np.zeros((0, 3), dtype=np.float32)
            )
            phase2_anchor_final_3d_vis = (
                np.nan_to_num(pts3d_final[anchor_indices], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                if anchor_indices.size > 0
                else np.zeros((0, 3), dtype=np.float32)
            )

            # Hybrid 统一三行：原始 -> 刚性旋转后 -> 非刚性抵达后
            rotate_3d_processor.save_3d_point_cloud_visualization(
                pts3d, pts3d_rot, colors_rgb,
                rotation_axis, float(rotation_angle), save_dir,
                centroid_3d=[0.0, 0.0, 0.0],
                handle_3d=phase1_handle_3d_vis,
                target_3d=phase1_target_3d_vis,
                handle_rotated_3d=phase1_rotated_handle_3d_vis,
                anchor_points_3d=phase1_anchor_3d_vis,
                anchor_points_rotated_3d=phase2_anchor_3d_vis,
                third_points_3d=pts3d_final,
                handle_third_3d=phase2_rotated_handle_3d_vis,
                target_third_3d=phase2_target_3d_vis,
                third_row_title="Final-NonRigid",
                output_name=f"depth_05_3d_point_cloud_hybrid_comp{comp_tag}.png",
                yaw_deg=rotation_yaw_deg,
                pitch_deg=rotation_pitch_deg,
                extra_rotated_points_3d=extra_rot_p2,
                extra_colors_rgb=extra_cols_p2,
                image_hw=(H_img, W_img),
                image_center_xy=(cx, cy),
            )
            # 兼容老版：额外保存两阶段独立点云图（Phase1 / Phase2）
            rotate_3d_processor.save_3d_point_cloud_visualization(
                pts3d, pts3d_rot, colors_rgb,
                rotation_axis, float(rotation_angle), save_dir,
                centroid_3d=[0.0, 0.0, 0.0],
                handle_3d=phase1_handle_3d_vis,
                target_3d=phase1_target_3d_vis,
                handle_rotated_3d=phase1_rotated_handle_3d_vis,
                anchor_points_3d=phase1_anchor_3d_vis,
                anchor_points_rotated_3d=phase2_anchor_3d_vis,
                output_name=f"depth_05_3d_point_cloud_phase1_comp{comp_tag}.png",
                yaw_deg=rotation_yaw_deg,
                pitch_deg=rotation_pitch_deg,
                image_hw=(H_img, W_img),
                image_center_xy=(cx, cy),
            )
            rotate_3d_processor.save_3d_point_cloud_visualization(
                pts3d_rot, pts3d_final, colors_rgb,
                rotation_axis, 0.0, save_dir,
                centroid_3d=[0.0, 0.0, 0.0],
                handle_3d=phase2_handle_3d_vis,
                target_3d=phase2_target_3d_vis,
                handle_rotated_3d=phase2_rotated_handle_3d_vis,
                anchor_points_3d=phase2_anchor_3d_vis,
                anchor_points_rotated_3d=phase2_anchor_final_3d_vis,
                output_name=f"depth_06_3d_point_cloud_phase2_comp{comp_tag}.png",
                image_hw=(H_img, W_img),
                image_center_xy=(cx, cy),
            )
        except Exception as e:
            print(f"[3D-Hybrid] Point cloud vis failed: {e}")

    # 合成总网格（final <- original）
    norm_grid = norm_grid_p2
    if norm_grid_p1 is not None and norm_grid_p2 is not None:
        norm_grid = F.grid_sample(
            norm_grid_p1.permute(2, 0, 1).unsqueeze(0),
            norm_grid_p2.unsqueeze(0),
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        ).squeeze().permute(1, 2, 0)

    sub_action = f"Yaw={rotation_yaw_deg:.1f},Pitch={rotation_pitch_deg:.1f}+3D_LOCAL_DEFORM"
    debug_info.update({
        'intent_type': '3D_HYBRID_STAGED',
        'sub_action': sub_action,
        'method_desc': phase2_desc,
        'deformation_space': '3D',
        'warp_backend': 'PROJECTIVE_ZBUFFER_3D',
        'anchors_img': anchors_img,
        'anchor_hard_constraint_count': 0,
        'anchor_soft_constraint_count': int(anchor_indices.size),
        'anchor_soft_strength': float(anchor_soft_strength),
        'anchor_soft_total_mass': float(anchor_soft_total_mass),
        'anchor_follow_gain': float(anchor_follow_gain),
        'anchor_follow_info': anchor_follow_info,
        'phase2_anchor_frame_shift_3d': phase2_anchor_frame_shift.astype(np.float32).tolist(),
        'anchor_lock_info': anchor_lock_info,
        'anchor_guard_info': anchor_guard_info,
        'anchor_lock_gain': float(anchor_lock_gain),
        'anchor_lock_snap': float(anchor_lock_snap),
        'runtime_scale': float(runtime_scale),
        'control_lock_info': control_lock_info,
        'control_global_follow_gain': float(control_global_follow_gain),
        'control_lock_radius_px': float(control_lock_radius_px),
        'control_lock_falloff_px': float(control_lock_falloff_px),
        'control_lock_neighbors': {
            'min': int(min_neighbors_runtime),
            'max': int(max_neighbors_runtime),
        },
        'sigma_gate_scale': float(sigma_gate_scale),
        'sigma_gate': float(sigma_gate),
        'comp_sam_pts': pair_src.astype(np.float32),
        'comp_targets_xy': pair_tgt.astype(np.float32),
        'norm_grid': norm_grid,
        'warped_latents': latents_final.clone(),
        'mask_end': comp_mask_final.clone(),
        'm_end_full': m_end_full,
        'rotated_mask': rotated_mask_p2_u8,
        'rotated_rgb': phase2_rgb,
        'rotated_depth': phase2_depth,
        'sam_mask': filtered_mask_u8,
        'depth_map': depth_map,
        'depth_filter_info': depth_filter_info,
        'z_center_info': z_center_info,
        'z_scale_factor': float(z_scale_factor),
        'z_scale': float(z_scale),
        'z_centroid': float(z_centroid),
        'mean_abs_dz': mean_abs_dz_p2,
        'z_metric_weight': z_metric_weight,
        'influence_radius_3d': influence_radius,
        'influence_ratio': influence_ratio,
        'centroid': (int(round(cx)), int(round(cy))),
        'rotation_axis': rotation_axis,
        'rotation_angle_deg': float(rotation_angle),
        'rotation_yaw_deg': rotation_yaw_deg,
        'rotation_pitch_deg': rotation_pitch_deg,
        'rotation_ref_index': int(phase1_ref_global_idx),
        'angle_computation': angle_debug_info,
        'phase1_source_points_xy': phase1_src.astype(np.float32),
        'phase1_target_points_xy': phase1_tgt.astype(np.float32),
        'phase1_rigid_pair_mode': 'first_only' if num_pairs > 1 else 'all',
        'phase1_rigid_pair_count': int(phase1_pair_count),
        'phase1_translation_3d': t_phase1.tolist(),
        'phase2_nonrigid_pair_count': int(num_pairs),
        'phase2_source_points_xy': phase2_src_xy.astype(np.float32),
        'phase2_source_points_xy_raw': rotated_handles_xy.astype(np.float32),
        'phase2_source_indices': phase2_src_indices,
        'phase2_point_delta_mean': float(phase2_point_delta_mean),
        'phase2_point_delta_max': float(phase2_point_delta_max),
        'phase2_target_points_xy': pair_tgt.astype(np.float32),
        'phase1_3d': phase1_debug,
        'phase2_3d_nonrigid': phase2_debug,
        'phase2_nonrigid': phase2_debug,
    })

    print(f"  [3D-Hybrid] {sub_action}, {phase2_desc}")
    return latents_final, comp_mask_final, "3D-Hybrid", None, anchors_img, debug_info

# ==========================================
# 5. 混合变形 - 纯流程编排层 (修改版)
# ==========================================

def process_hybrid_deformation(
    comp_sam_pts, comp_targets_xy, mask_for_calc,
    src_xy, tgt_xy, H_lat, W_lat, device,
    latents, comp_mask_start, comp_m_start_full,
    scale_x, scale_y, rigid_ratio
):
    """
    【混合变形】两阶段编排器（重构版）
    Phase 1: 刚性旋转 (强制 ROTATION)
    Phase 2: 非刚性变形 (从旋转后位置到目标)
    
    返回两个阶段的完整信息，用于分别生成可视化
    """
    print(f"[Hybrid] Starting two-phase deformation...")

    # 计算图像分辨率
    H_img = _infer_image_size_from_latent_and_scale(H_lat, scale_y)
    W_img = _infer_image_size_from_latent_and_scale(W_lat, scale_x)

    # ============================================
    # 意图识别：单点 vs 多点
    # 单点: 完整两阶段 (旋转 + 非刚性)
    # 多点: 跳过旋转，直接非刚性 (防止多点导致旋转意图混乱)
    # ============================================
    num_points = src_xy.shape[0]
    skip_rotation = num_points > 1

    if skip_rotation:
        print(f"  [Intent] Multiple points ({num_points}): Skip rotation, direct to non-rigid")
    else:
        print(f"  [Intent] Single point: Full two-phase (rotation + non-rigid)")

    # ============================================
    # Phase 1: 刚性旋转（单点）或恒等变换（多点）
    # ============================================
    if skip_rotation:
        # 多点情况：Phase 1 保持不变（恒等变换）
        print(f"  [Phase 1] Identity (skipped rotation for multi-point)...")

        # 创建恒等 Grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H_lat, device=device),
            torch.linspace(-1, 1, W_lat, device=device),
            indexing='ij'
        )
        identity_grid = torch.stack([grid_x, grid_y], dim=2)

        # Phase 1 结果：不变
        latents_p1 = latents.clone()
        mask_p1 = comp_mask_start.clone() if isinstance(comp_mask_start, torch.Tensor) else comp_mask_start.copy()

        # 控制点不变
        rotated_src_xy = src_xy.clone()
        rotation_angle = 0.0

        # 计算 pivot（用于记录，但不实际使用）
        mask_tensor = torch.from_numpy(mask_for_calc).float().to(device)
        ys, xs = torch.where(mask_tensor > 0.5)
        if len(ys) > 0:
            pivot_y = ys.float().mean().item()
            pivot_x = xs.float().mean().item()
            pivot_img = np.array([pivot_x, pivot_y])
        else:
            pivot_img = np.array([W_img / 2, H_img / 2])

        # 构建 debug_p1（与正常 Phase 1 兼容）
        debug_p1 = {
            'norm_grid': identity_grid,
            'transformed_src_xy': rotated_src_xy,
            'rotation_angle': 0.0,
            'sub_action': 'Identity',
            'intent_type': 'IDENTITY',
            'method_desc': 'Multi-point skip rotation',
        }

        print(f"    ✓ Phase 1 complete (identity, no rotation)")

    else:
        # 单点情况：正常执行旋转
        print(f"  [Phase 1] Rigid Rotation (forced)...")

        latents_p1, mask_p1, action_p1, pivot_img, _, debug_p1 = process_rigid_deformation(
            comp_sam_pts=comp_sam_pts,
            comp_targets_xy=comp_targets_xy,
            mask_for_calc=mask_for_calc,
            src_xy=src_xy,
            tgt_xy=tgt_xy,
            H_lat=H_lat,
            W_lat=W_lat,
            device=device,
            latents=latents,
            comp_mask_start=comp_mask_start,
            comp_m_start_full=comp_m_start_full,
            scale_x=scale_x,
            scale_y=scale_y,
            force_mode="ROTATION"
        )

        # 提取旋转后的控制点
        rotated_src_xy = debug_p1['transformed_src_xy']
        rotation_angle = debug_p1.get('rotation_angle', 0)

        print(f"    Pivot: {pivot_img}, Angle: {rotation_angle:.2f}°")
    
    # ============================================
    # 关键修复：用高分辨率 grid 对原始高分辨率 mask 进行 warp
    # ============================================
    norm_grid_p1 = debug_p1['norm_grid']
    
    # 上采样 grid 到图像分辨率
    grid_p1_up = F.interpolate(
        norm_grid_p1.permute(2, 0, 1).unsqueeze(0),
        size=(H_img, W_img),
        mode='bilinear',
        align_corners=True
    ).permute(0, 2, 3, 1)
    
    # 用高分辨率 grid 对原始高分辨率 mask 进行 warp
    m_start_full_tensor = torch.from_numpy(comp_m_start_full).float().to(device).view(1, 1, H_img, W_img)
    m_p1_full_tensor = F.grid_sample(
        m_start_full_tensor,
        grid_p1_up,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    m_p1_full = m_p1_full_tensor.squeeze().cpu().numpy()
    m_p1_full = (m_p1_full > 0.5).astype(np.float32)  # 二值化
    
    print(f"    ✓ Phase 1 complete")
    
    # Refine Phase 1 结果
    latents_p1_refined = interpolation_fill_subject(latents_p1)
    
    # ============================================
    # Phase 2: 非刚性变形（从旋转后位置到目标）
    # ============================================
    print(f"  [Phase 2] Non-Rigid Deformation...")
    
    # 旋转后控制点的图像坐标
    rotated_sam_pts = rotated_src_xy.cpu().numpy() / np.array([scale_x, scale_y])
    
    # 使用高分辨率的旋转后 mask 进行意图识别
    m_p1_full_u8 = (m_p1_full * 255).astype(np.uint8)
    
    # 调用非刚性变形
    latents_p2, mask_p2, action_p2, _, anchors_img, debug_p2 = process_nonrigid_deformation(
        comp_sam_pts=rotated_sam_pts,
        comp_targets_xy=comp_targets_xy,
        mask_for_calc=m_p1_full_u8,  # 高分辨率 mask
        src_xy=rotated_src_xy,
        tgt_xy=tgt_xy,
        H_lat=H_lat,
        W_lat=W_lat,
        device=device,
        latents=latents_p1_refined,
        comp_mask_start=mask_p1,
        comp_m_start_full=m_p1_full,  # 高分辨率 mask
        scale_x=scale_x,
        scale_y=scale_y,
        rigid_ratio=rigid_ratio
    )
    
    phase2_intent = debug_p2.get('intent_type', 'Unknown')
    phase2_method = debug_p2.get('method_desc', '')
    print(f"    {phase2_intent}, {phase2_method}, Anchors: {len(anchors_img)}")
    
    # Refine Phase 2 结果
    latents_p2_refined = interpolation_fill_subject(latents_p2)
    
    # ============================================
    # 组合 debug_info
    # ============================================
    
    # Phase 1 的完整结果
    debug_p1['comp_sam_pts'] = comp_sam_pts
    debug_p1['comp_targets_xy'] = comp_targets_xy
    debug_p1['rotated_sam_pts'] = rotated_sam_pts
    debug_p1['m_end_full'] = m_p1_full  # 高分辨率的 Phase 1 结果 mask
    
    # Phase 2 的完整结果
    debug_p2['comp_sam_pts'] = rotated_sam_pts
    debug_p2['comp_targets_xy'] = comp_targets_xy
    debug_p2['m_start_full'] = m_p1_full  # Phase 2 的起点是 Phase 1 的终点
    
    # 组合两个 Grid
    norm_grid_p1 = debug_p1['norm_grid']
    norm_grid_p2 = debug_p2['norm_grid']
    
    combined_grid = F.grid_sample(
        norm_grid_p1.permute(2, 0, 1).unsqueeze(0),
        norm_grid_p2.unsqueeze(0),
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ).squeeze().permute(1, 2, 0)
    
    debug_info = {
        'phase1': debug_p1,
        'phase2': debug_p2,
        'combined_grid': combined_grid,
        'pivot_img': pivot_img,
        'anchors_img': anchors_img,
        'm_start_full': comp_m_start_full,
        'norm_grid': combined_grid,
    }
    
    sub_action = f"Rotation({rotation_angle:.1f}°)+{phase2_intent}"
    debug_info['sub_action'] = sub_action
    
    print(f"[Hybrid] ✓ Complete: {sub_action}")
    
    return latents_p2_refined, mask_p2, "Hybrid", pivot_img, anchors_img, debug_info


# ==========================================
# ==========================================
# 可视化函数 (重构版)
# ==========================================
# Visualization helpers moved to utils_drag.drag_visualization.

def process_drag_request(
    latents,
    source_image_np,
    user_drawn_mask,
    handle_points,
    target_points,
    drag_mode,
    rigid_ratio,
    device,
    visualize_drag=False,
    pointcloud_domain="auto",
    drag_plan=None,
    return_plan=False,
    hole_fill_mode=DEFAULT_HOLE_FILL_MODE,
    use_expanded_subject_fill=DEFAULT_EXPANDED_SUBJECT_FILL_ENABLED,
    expanded_subject_fill_px=DEFAULT_EXPANDED_SUBJECT_FILL_PX,
    use_drag_guided_prefill=DEFAULT_DRAG_GUIDED_PREFILL_ENABLED,
    mask_backend_mode=DEFAULT_MASK_BACKEND_MODE,
    anchor_strategy_3d=DEFAULT_THREE_D_ANCHOR_STRATEGY,
    enable_3d_subject_scope_fill=False,
    vae=None,
    clean_latents=None,
    alpha_prod_t=None,
):
    """
    【重构】支持多连通域同时拖拽
    刚性模式下每个连通域独立判断旋转/平移
    非刚性模式下智能生成锚点
    已移除深度图参数（点云逻辑残留）
    """
    hole_fill_mode = _normalize_hole_fill_mode(hole_fill_mode)
    mask_backend_mode = _normalize_mask_backend_mode(mask_backend_mode)
    use_expanded_subject_fill, expanded_subject_fill_px = _normalize_expanded_subject_fill(
        use_expanded_subject_fill=use_expanded_subject_fill,
        expanded_subject_fill_px=expanded_subject_fill_px,
    )
    use_drag_guided_prefill = _normalize_drag_guided_prefill(use_drag_guided_prefill)
    enable_3d_subject_scope_fill = bool(enable_3d_subject_scope_fill)

    if hole_fill_mode in {"sgf", "lama"}:
        print(f"[HoleFill SGF] inline mode ({hole_fill_mode})")
        print(f"[HoleFill SGF] drag_guided_prefill={'on' if use_drag_guided_prefill else 'off'}")

    B, C, H_lat, W_lat = latents.shape 
    if drag_plan is not None:
        replay_latents = _apply_drag_plan_to_latents(
            latents=latents,
            drag_plan=drag_plan,
            device=device,
            clean_latents=clean_latents,
            alpha_prod_t=alpha_prod_t,
        )
        if return_plan:
            return replay_latents, drag_plan
        return replay_latents

    canonical_mode, domain, base_mode = split_drag_mode(drag_mode)
    mode_is_3d = (domain == "3D")
    drag_mode = canonical_mode
    pointcloud_domain = str(pointcloud_domain or "auto").strip().lower()
    if pointcloud_domain not in {"auto", "latent", "image"}:
        pointcloud_domain = "auto"
    anchor_strategy_3d = _normalize_3d_anchor_strategy(anchor_strategy_3d)
    print(f"[Drag Mode] {canonical_mode} (domain={domain}, base={base_mode})")
    print(f"[Mask Backend] mode={mask_backend_mode}")
    if mode_is_3d:
        print(f"[3D Anchor] strategy={anchor_strategy_3d}")
    print(f"[Drag Processor] {__file__}")
    if mode_is_3d and (not bool(enable_3d_subject_scope_fill)):
        print("[Subject Fill][3D] disabled: skip component latent interpolation + post-compose subject-hole fill.")

    H_img, W_img = source_image_np.shape[:2]
    min_len = min(len(handle_points), len(target_points))
    if min_len == 0: 
        if return_plan:
            empty_plan = _build_drag_plan(
                drag_mode=drag_mode,
                m_start_full_overall=np.zeros((H_img, W_img), dtype=np.float32),
                m_pseudo_full_overall=np.zeros((H_img, W_img), dtype=np.float32),
                background_hole_full=np.zeros((H_img, W_img), dtype=np.float32),
                component_norm_grids=[],
                component_masks_start=[],
                component_masks_end=[],
                component_handles=[],
                component_targets=[],
                hole_fill_mode=hole_fill_mode,
                use_drag_guided_prefill=use_drag_guided_prefill,
                enable_subject_hole_fill=enable_3d_subject_scope_fill,
                subject_hole_mask_lat=np.zeros((H_lat, W_lat), dtype=np.float32),
            )
            return latents, empty_plan
        return latents
    
    handle_points = handle_points[:min_len]
    target_points = target_points[:min_len]

    # 坐标映射（align_corners=True 口径）
    scale_y = _coord_scale_image_to_latent(H_img, H_lat)
    scale_x = _coord_scale_image_to_latent(W_img, W_lat)
    all_sam_pts = []
    all_targets_xy = []
    for i in range(min_len):
        px = int(handle_points[i][1])
        py = int(handle_points[i][0])
        all_sam_pts.append([px, py])
        tx = int(target_points[i][1])
        ty = int(target_points[i][0])
        all_targets_xy.append([tx, ty])
    all_sam_pts = np.array(all_sam_pts)
    all_targets_xy = np.array(all_targets_xy)
    rotate_3d_processor.ENABLE_3D_DEBUG = bool(visualize_drag)
    # 初始化调试目录
    if visualize_drag:
        os.makedirs(MASK_DEBUG_ROOT, exist_ok=True)
        os.makedirs(DRAG_DEBUG_ROOT, exist_ok=True)
        cv2.imwrite(f"{MASK_DEBUG_ROOT}/00_input.jpg", cv2.cvtColor(source_image_np, cv2.COLOR_RGB2BGR))
    # Mask 准备
    if user_drawn_mask is not None:
        ht = user_drawn_mask.copy()
        if ht.max() <= 1.5: 
            ht = ht * 255
        hint_mask = ht.astype(np.uint8)
        _, hint_mask = cv2.threshold(hint_mask, 127, 255, cv2.THRESH_BINARY)
        if visualize_drag:
            cv2.imwrite(f"{MASK_DEBUG_ROOT}/00_mask.png", hint_mask)
    else:
        hint_mask = None
    # 连通域提取
    components_info = extract_mask_components(hint_mask, source_image_np, all_sam_pts)
    if len(components_info) == 0:
        components_info = [{
            'mask_u8': hint_mask if hint_mask is not None else np.ones((H_img, W_img), dtype=np.uint8) * 255,
            'handle_indices': list(range(min_len)),
            'hint_mask': hint_mask
        }]

    num_components = len(components_info)
    print(f"[Multi-Component] Found {num_components} separate components")
    # 确定调试子目录
    if drag_mode in SUPPORTED_CANONICAL_DRAG_MODES:
        sub_folder = drag_mode
    else:
        sub_folder = "2D-Hybrid"
    CURRENT_DEBUG_DIR = os.path.join(DRAG_DEBUG_ROOT, sub_folder) if visualize_drag else None
    if visualize_drag:
        os.makedirs(CURRENT_DEBUG_DIR, exist_ok=True)
    if mode_is_3d:
        rotate_3d_processor.ROTATE_3D_DEBUG_ROOT = os.path.join(DRAG_DEBUG_ROOT, sub_folder)
    # 批量 SAM 调用（sam_refined 模式）
    all_comp_sam_pts = []
    all_comp_hint_masks = []
    valid_comp_indices = []
    for comp_idx, comp_info in enumerate(components_info):
        comp_handle_indices = comp_info['handle_indices']
        if len(comp_handle_indices) == 0:
            continue
        
        comp_sam_pts = all_sam_pts[comp_handle_indices]
        all_comp_sam_pts.append(comp_sam_pts.astype(int))
        all_comp_hint_masks.append(comp_info['hint_mask'])
        valid_comp_indices.append(comp_idx)
    use_user_mask_backend = mask_backend_mode == "user_mask"
    if len(all_comp_sam_pts) > 0 and (not use_user_mask_backend):
        print(f"\n[SAM Batch] Processing {len(all_comp_sam_pts)} components...")
        all_comp_sam_masks = get_interactive_masks_batch(
            image=source_image_np,
            all_handle_points=all_comp_sam_pts,
            device=device,
            all_user_hint_masks=all_comp_hint_masks,
            debug_dir=MASK_DEBUG_ROOT if visualize_drag else None,
            enable_debug=visualize_drag
        )
        print(f"[SAM Batch] ✓ Generated {len(all_comp_sam_masks)} masks")
    elif use_user_mask_backend:
        print("\n[Mask Backend] User Mask mode: skip SAM batch.")
        all_comp_sam_masks = []
    else:
        all_comp_sam_masks = []
    # 结果容器 - 添加 sub_actions
    all_results = {
        'latents': [], 'masks_start': [], 'masks_end': [], 'grids': [],
        'handles': [], 'targets': [], 'pivots': [], 'anchors': [], 
        'actions': [], 'sub_actions': [], 'debug_infos': []
    }
    m_pseudo_full_overall = np.zeros((H_img, W_img), dtype=np.float32)
    m_start_full_overall = np.zeros((H_img, W_img), dtype=np.float32)
    # 逐连通域处理
    depth_map_cache = None
    comp_sam_mask_idx = 0
    for valid_idx, comp_idx in enumerate(valid_comp_indices):
        comp_info = components_info[comp_idx]
        print(f"\n--- Component {comp_idx + 1}/{num_components} ---")
        
        comp_handle_indices = comp_info['handle_indices']
        comp_sam_pts = all_sam_pts[comp_handle_indices]
        comp_targets_xy = all_targets_xy[comp_handle_indices]
        
        comp_hint_mask = comp_info['hint_mask']
        # SAM / User Mask
        if use_user_mask_backend:
            comp_user_mask = comp_hint_mask if comp_hint_mask is not None else comp_info.get("mask_u8")
            if comp_user_mask is None:
                comp_user_mask = np.ones((H_img, W_img), dtype=np.uint8) * 255
            comp_sam_mask_np = comp_user_mask.astype(np.uint8)
            if comp_sam_mask_np.max() <= 1:
                comp_sam_mask_np = (comp_sam_mask_np * 255).astype(np.uint8)
            _, comp_sam_mask_np = cv2.threshold(comp_sam_mask_np, 127, 255, cv2.THRESH_BINARY)
            comp_sam_mask_np = prune_inactive_masks(comp_sam_mask_np, comp_sam_pts)
        else:
            comp_sam_mask_uint8 = all_comp_sam_masks[comp_sam_mask_idx]
            comp_sam_mask_idx += 1
            comp_sam_mask_np = dilate_mask(comp_sam_mask_uint8)
            comp_sam_mask_np = prune_inactive_masks(comp_sam_mask_np, comp_sam_pts)
        comp_mask_start = None
        # 高分辨率 Mask
        comp_m_pseudo_full = comp_sam_mask_np.astype(np.float32) / 255.0
        if comp_hint_mask is not None:
            hint_mask_float = comp_hint_mask.astype(np.float32) / 255.0
            comp_m_start_full = comp_m_pseudo_full * hint_mask_float
            comp_m_start_full = (comp_m_start_full > 0.5).astype(np.float32)
            if comp_m_start_full.sum() < 10:
                comp_m_start_full = comp_m_pseudo_full.copy()
        else:
            comp_m_start_full = comp_m_pseudo_full.copy()
        m_pseudo_full_overall = np.maximum(m_pseudo_full_overall, comp_m_pseudo_full)
        m_start_full_overall = np.maximum(m_start_full_overall, comp_m_start_full)
        # 非刚性/后续几何计算继续使用主体 mask（SAM ∩ User）
        mask_for_calc = (comp_m_start_full * 255).astype(np.uint8)
        comp_m_start_lat = _downsample_image_mask_to_latent(
            comp_m_start_full,
            target_hw=(H_lat, W_lat),
            threshold=0.5,
        )
        comp_mask_start = torch.from_numpy(comp_m_start_lat).to(device).float()

        # 刚性意图识别统一使用主体 mask（SAM ∩ User），
        # 与前端显示保持一致，避免 user/sam 质心不一致造成歧义。
        rigid_intent_mask_for_calc = mask_for_calc
        
        # 控制点转换到 Latent
        src_xy = []
        tgt_xy = []
        for idx in comp_handle_indices:
            lx = handle_points[idx][1] * scale_x
            ly = handle_points[idx][0] * scale_y
            tx = target_points[idx][1] * scale_x
            ty = target_points[idx][0] * scale_y
            src_xy.append([lx, ly])
            tgt_xy.append([tx, ty])
        src_xy = torch.tensor(src_xy, device=device).float()
        tgt_xy = torch.tensor(tgt_xy, device=device).float()

        # 统一语义：值越大，影响范围越大。
        # 2D/3D 都直接使用该语义，不再做反向映射。
        influence_ratio = float(np.clip(rigid_ratio, 0.0, 1.0))
        influence_range_2d = influence_ratio

        # ============================================================
        # Grid 计算 - 2D/3D 统一模式分发
        # ============================================================
        result_targets_xy = comp_targets_xy
        result_handles_xy = comp_sam_pts
        runtime_geom_domain = pointcloud_domain
        if runtime_geom_domain == "auto":
            runtime_geom_domain = "image" if bool(visualize_drag) else "latent"

        # 2D 运行域切换（与 3D 语义保持一致）：
        # - latent: 变换计算在 latent 空间进行
        # - image : 变换计算在图像空间进行
        runtime_comp_sam_pts_2d = comp_sam_pts.astype(np.float32)
        runtime_comp_targets_xy_2d = comp_targets_xy.astype(np.float32)
        runtime_mask_for_calc_2d = mask_for_calc
        runtime_rigid_mask_for_calc_2d = rigid_intent_mask_for_calc
        runtime_src_xy_2d = src_xy
        runtime_tgt_xy_2d = tgt_xy
        runtime_comp_mask_start_2d = comp_mask_start
        runtime_comp_m_start_full_2d = comp_m_start_full
        runtime_scale_x_2d = float(scale_x)
        runtime_scale_y_2d = float(scale_y)
        if (not mode_is_3d) and runtime_geom_domain == "latent":
            # 关键：2D latent 下，意图识别/锚点规则仍使用 image 分辨率，
            # 只在最终求解与采样时落到 latent，避免低分辨率规则失真（过拉/欠拉）。
            runtime_mask_for_calc_2d = mask_for_calc
            runtime_rigid_mask_for_calc_2d = rigid_intent_mask_for_calc
            runtime_comp_m_start_full_2d = comp_m_start_full
            runtime_comp_mask_start_2d = comp_mask_start
            runtime_comp_sam_pts_2d = comp_sam_pts.astype(np.float32)
            runtime_comp_targets_xy_2d = comp_targets_xy.astype(np.float32)
            runtime_scale_x_2d = float(scale_x)
            runtime_scale_y_2d = float(scale_y)
            print(f"[2D] Runtime domain: LATENT execution (image-space intent/anchors, requested={pointcloud_domain})")
        elif not mode_is_3d:
            print(f"[2D] Runtime domain: IMAGE geometry (requested={pointcloud_domain})")

        def _refine_subject_latents_component(warped_latents_raw):
            # 3D SubjectScope 补洞开关关闭时，组件阶段不做这一步兜底补洞，
            # 否则会把开关差异提前抹平（看起来“开关无效”）。
            if mode_is_3d and (not bool(enable_3d_subject_scope_fill)):
                return warped_latents_raw
            return interpolation_fill_subject(warped_latents_raw)

        if mode_is_3d:
            comp_point_count = len(comp_handle_indices)
            if depth_map_cache is None:
                print("[3D] Estimating depth map...")
                depth_map_cache = estimate_depth(source_image_np, device=device)

            runtime_pointcloud_domain = runtime_geom_domain

            # 3D 运行域切换：
            # - latent: 纯 latent 点云（xy/mask/depth/控制点均在 latent 空间）
            # - image : 原图点云（便于可视化解读）
            runtime_comp_sam_pts = comp_sam_pts.astype(np.float32)
            runtime_comp_targets_xy = comp_targets_xy.astype(np.float32)
            runtime_mask_for_calc = mask_for_calc
            runtime_src_xy = src_xy
            runtime_tgt_xy = tgt_xy
            runtime_comp_mask_start = comp_mask_start
            runtime_comp_m_start_full = comp_m_start_full
            runtime_scale_x = float(scale_x)
            runtime_scale_y = float(scale_y)
            runtime_image_to_runtime_scale = 1.0
            runtime_depth_map = depth_map_cache
            runtime_source_image = source_image_np

            if runtime_pointcloud_domain == "latent":
                runtime_mask_for_calc = cv2.resize(
                    mask_for_calc.astype(np.uint8),
                    (W_lat, H_lat),
                    interpolation=cv2.INTER_NEAREST,
                )
                runtime_mask_for_calc = (runtime_mask_for_calc > 127).astype(np.uint8) * 255
                runtime_comp_m_start_full = runtime_mask_for_calc.astype(np.float32) / 255.0
                runtime_comp_mask_start = torch.from_numpy(runtime_comp_m_start_full).to(device).float()
                runtime_comp_sam_pts = runtime_src_xy.detach().cpu().numpy().astype(np.float32)
                runtime_comp_targets_xy = runtime_tgt_xy.detach().cpu().numpy().astype(np.float32)
                runtime_scale_x = 1.0
                runtime_scale_y = 1.0
                runtime_image_to_runtime_scale = float(np.clip(min(scale_x, scale_y), 1e-3, 1.0))
                runtime_depth_map = cv2.resize(
                    depth_map_cache.astype(np.float32),
                    (W_lat, H_lat),
                    interpolation=cv2.INTER_LINEAR,
                )
                runtime_source_image = None
                print(f"[3D] Runtime domain: LATENT point cloud (requested={pointcloud_domain})")
            else:
                print(f"[3D] Runtime domain: IMAGE point cloud (requested={pointcloud_domain})")

            if base_mode == "Non-Rigid":
                warped_latents_raw, comp_mask_end, _, pivot_img_np, current_anchors, debug_info_3dnr = process_3d_nonrigid_deformation(
                    comp_sam_pts=runtime_comp_sam_pts,
                    comp_targets_xy=runtime_comp_targets_xy,
                    mask_for_calc=runtime_mask_for_calc,
                    src_xy=runtime_src_xy,
                    tgt_xy=runtime_tgt_xy,
                    H_lat=H_lat,
                    W_lat=W_lat,
                    device=device,
                    latents=latents,
                    comp_mask_start=runtime_comp_mask_start,
                    comp_m_start_full=runtime_comp_m_start_full,
                    scale_x=runtime_scale_x,
                    scale_y=runtime_scale_y,
                    influence_ratio=influence_ratio,
                    depth_map=runtime_depth_map,
                    source_image_np=runtime_source_image,
                    component_id=comp_idx,
                    image_to_runtime_scale=runtime_image_to_runtime_scale,
                    anchor_strategy_3d=anchor_strategy_3d,
                    enable_subject_scope_fill=enable_3d_subject_scope_fill,
                )
                subject_latents_refined = _refine_subject_latents_component(warped_latents_raw)
                norm_grid = debug_info_3dnr.get('norm_grid')
                action = "3D-Non-Rigid"
                sub_action = debug_info_3dnr.get('sub_action', '3D_LOCAL_DEFORM')
                debug_info = debug_info_3dnr

            elif base_mode == "Hybrid":
                warped_latents_raw, comp_mask_end, _, pivot_img_np, current_anchors, debug_info_3dhy = process_3d_hybrid_unified_deformation(
                    comp_sam_pts=runtime_comp_sam_pts,
                    comp_targets_xy=runtime_comp_targets_xy,
                    mask_for_calc=runtime_mask_for_calc,
                    src_xy=runtime_src_xy,
                    tgt_xy=runtime_tgt_xy,
                    H_lat=H_lat,
                    W_lat=W_lat,
                    device=device,
                    latents=latents,
                    comp_mask_start=runtime_comp_mask_start,
                    comp_m_start_full=runtime_comp_m_start_full,
                    scale_x=runtime_scale_x,
                    scale_y=runtime_scale_y,
                    influence_ratio=influence_ratio,
                    depth_map=runtime_depth_map,
                    source_image_np=runtime_source_image,
                    component_id=comp_idx,
                    image_to_runtime_scale=runtime_image_to_runtime_scale,
                    anchor_strategy_3d=anchor_strategy_3d,
                    enable_subject_scope_fill=enable_3d_subject_scope_fill,
                )
                subject_latents_refined = _refine_subject_latents_component(warped_latents_raw)
                norm_grid = debug_info_3dhy.get('norm_grid')
                action = "3D-Hybrid"
                sub_action = debug_info_3dhy.get('sub_action', '3D_HYBRID_UNIFIED')
                debug_info = debug_info_3dhy
                print(f"  [3D-Hybrid] Staged 3D done: {sub_action}")

            elif comp_point_count > 1 and base_mode == "Rigid":
                print(f"  [3D] Multi-point ({comp_point_count}) in Rigid mode: keep pure 3D-Rigid (intent-aware)")
                warped_latents_raw, comp_mask_end, _, pivot_img_np, current_anchors, debug_info_3d = process_3d_rigid_unified_deformation(
                    comp_sam_pts=runtime_comp_sam_pts,
                    comp_targets_xy=runtime_comp_targets_xy,
                    mask_for_calc=runtime_mask_for_calc,
                    src_xy=runtime_src_xy,
                    tgt_xy=runtime_tgt_xy,
                    H_lat=H_lat,
                    W_lat=W_lat,
                    device=device,
                    latents=latents,
                    comp_mask_start=runtime_comp_mask_start,
                    comp_m_start_full=runtime_comp_m_start_full,
                    scale_x=runtime_scale_x,
                    scale_y=runtime_scale_y,
                    depth_map=runtime_depth_map,
                    source_image_np=runtime_source_image,
                    component_id=comp_idx,
                    enable_subject_scope_fill=enable_3d_subject_scope_fill,
                )

                subject_latents_refined = _refine_subject_latents_component(warped_latents_raw)
                norm_grid = debug_info_3d.get('norm_grid')
                axis_name = debug_info_3d.get('rotation_axis', 'Y')
                angle_deg = debug_info_3d.get('rotation_angle_deg', 0)
                yaw_deg = debug_info_3d.get('rotation_yaw_deg', angle_deg if axis_name == 'Y' else 0.0)
                pitch_deg = debug_info_3d.get('rotation_pitch_deg', angle_deg if axis_name == 'X' else 0.0)
                action = "3D-Rigid"
                sub_action = debug_info_3d.get('sub_action', f"Yaw={yaw_deg:.1f},Pitch={pitch_deg:.1f}")
                debug_info = dict(debug_info_3d)
                debug_info.update({
                    'is_3d_fallback': False,
                    'fallback_reason': 'disabled_keep_pure_rigid',
                    'point_count': int(comp_point_count),
                })
                print(f"  [3D-Rigid] {sub_action} (axis={axis_name}, angle={angle_deg:.2f}°, yaw={yaw_deg:.2f}°, pitch={pitch_deg:.2f}°)")

            else:
                # ========== 3D-Rigid ==========
                warped_latents_raw, comp_mask_end, _, pivot_img_np, current_anchors, debug_info_3d = process_3d_rigid_unified_deformation(
                    comp_sam_pts=runtime_comp_sam_pts,
                    comp_targets_xy=runtime_comp_targets_xy,
                    mask_for_calc=runtime_mask_for_calc,
                    src_xy=runtime_src_xy,
                    tgt_xy=runtime_tgt_xy,
                    H_lat=H_lat,
                    W_lat=W_lat,
                    device=device,
                    latents=latents,
                    comp_mask_start=runtime_comp_mask_start,
                    comp_m_start_full=runtime_comp_m_start_full,
                    scale_x=runtime_scale_x,
                    scale_y=runtime_scale_y,
                    depth_map=runtime_depth_map,
                    source_image_np=runtime_source_image,
                    component_id=comp_idx,
                    enable_subject_scope_fill=enable_3d_subject_scope_fill,
                )

                subject_latents_refined = _refine_subject_latents_component(warped_latents_raw)
                norm_grid = debug_info_3d.get('norm_grid')
                axis_name = debug_info_3d.get('rotation_axis', 'Y')
                angle_deg = debug_info_3d.get('rotation_angle_deg', 0)
                yaw_deg = debug_info_3d.get('rotation_yaw_deg', angle_deg if axis_name == 'Y' else 0.0)
                pitch_deg = debug_info_3d.get('rotation_pitch_deg', angle_deg if axis_name == 'X' else 0.0)
                action = "3D-Rigid"
                sub_action = debug_info_3d.get('sub_action', f"Yaw={yaw_deg:.1f},Pitch={pitch_deg:.1f}")
                debug_info = debug_info_3d
                print(f"  [3D-Rigid] {sub_action} (axis={axis_name}, angle={angle_deg:.2f}°, yaw={yaw_deg:.2f}°, pitch={pitch_deg:.2f}°)")

        elif drag_mode == '2D-Hybrid':
            # ========== 2D-Hybrid：调用流程编排器 ==========
            warped_latents_raw, comp_mask_end, action, pivot_img_np, current_anchors, debug_info = process_hybrid_deformation(
                comp_sam_pts=runtime_comp_sam_pts_2d,
                comp_targets_xy=runtime_comp_targets_xy_2d,
                mask_for_calc=runtime_rigid_mask_for_calc_2d,
                src_xy=runtime_src_xy_2d,
                tgt_xy=runtime_tgt_xy_2d,
                H_lat=H_lat,
                W_lat=W_lat,
                device=device,
                latents=latents,
                comp_mask_start=runtime_comp_mask_start_2d,
                comp_m_start_full=runtime_comp_m_start_full_2d,
                scale_x=runtime_scale_x_2d,
                scale_y=runtime_scale_y_2d,
                rigid_ratio=influence_range_2d
            )

            subject_latents_refined = _refine_subject_latents_component(warped_latents_raw)
            phase1_action = debug_info.get('phase1', {}).get('sub_action', 'Rotation')
            phase2_action = debug_info.get('phase2', {}).get('intent_type', 'Unknown')
            sub_action = f"{phase1_action}+{phase2_action}"
            action = "2D-Hybrid"

            norm_grid = debug_info.get('norm_grid')
            if norm_grid is None:
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(H_lat, device=device),
                    torch.arange(W_lat, device=device),
                    indexing='ij'
                )
                grid_abs = torch.stack([grid_x, grid_y], dim=2).float()
                norm_grid = torch.zeros_like(grid_abs)
                norm_grid[..., 0] = 2.0 * grid_abs[..., 0] / (W_lat - 1) - 1.0
                norm_grid[..., 1] = 2.0 * grid_abs[..., 1] / (H_lat - 1) - 1.0

        elif drag_mode == '2D-Rigid':
            # ========== 2D-Rigid ==========
            warped_latents_raw, comp_mask_end, action, pivot_img_np, current_anchors, debug_info = process_rigid_deformation(
                comp_sam_pts=runtime_comp_sam_pts_2d,
                comp_targets_xy=runtime_comp_targets_xy_2d,
                mask_for_calc=runtime_rigid_mask_for_calc_2d,
                src_xy=runtime_src_xy_2d,
                tgt_xy=runtime_tgt_xy_2d,
                H_lat=H_lat,
                W_lat=W_lat,
                device=device,
                latents=latents,
                comp_mask_start=runtime_comp_mask_start_2d,
                comp_m_start_full=runtime_comp_m_start_full_2d,
                scale_x=runtime_scale_x_2d,
                scale_y=runtime_scale_y_2d,
                force_mode=None
            )

            subject_latents_refined = _refine_subject_latents_component(warped_latents_raw)
            sub_action = debug_info.get('sub_action', 'Unknown')
            norm_grid = debug_info.get('norm_grid')
            action = "2D-Rigid"
            print(f"  [2D-Rigid] Sub-action: {sub_action}")

        else:
            # ========== 2D-Non-Rigid ==========
            warped_latents_raw, comp_mask_end, action, pivot_img_np, current_anchors, debug_info = process_nonrigid_deformation(
                comp_sam_pts=runtime_comp_sam_pts_2d,
                comp_targets_xy=runtime_comp_targets_xy_2d,
                mask_for_calc=runtime_mask_for_calc_2d,
                src_xy=runtime_src_xy_2d,
                tgt_xy=runtime_tgt_xy_2d,
                H_lat=H_lat,
                W_lat=W_lat,
                device=device,
                latents=latents,
                comp_mask_start=runtime_comp_mask_start_2d,
                comp_m_start_full=runtime_comp_m_start_full_2d,
                scale_x=runtime_scale_x_2d,
                scale_y=runtime_scale_y_2d,
                rigid_ratio=influence_range_2d
            )

            subject_latents_refined = _refine_subject_latents_component(warped_latents_raw)
            sub_action = debug_info.get('intent_type', 'Unknown')
            norm_grid = debug_info.get('norm_grid')
            action = "2D-Non-Rigid"

        # 在 latent 域运行时，把调试可视化坐标映射回图像域，避免前端错位/像素化
        if runtime_geom_domain == "latent":
            to_img_x = _coord_scale_latent_to_image(W_lat, W_img)
            to_img_y = _coord_scale_latent_to_image(H_lat, H_img)
            to_lat_x = _coord_scale_image_to_latent(W_img, W_lat)
            to_lat_y = _coord_scale_image_to_latent(H_img, H_lat)

            fg_u8_local = (np.asarray(comp_m_start_full, dtype=np.float32) > 0.5).astype(np.uint8)
            dist_to_fg_local = None
            if int(np.count_nonzero(fg_u8_local)) > 0:
                outside_local = (fg_u8_local == 0).astype(np.uint8)
                dist_to_fg_local = cv2.distanceTransform(outside_local, cv2.DIST_L2, 5)

            def _point_fit_score_local(arr_xy):
                if (arr_xy is None) or (arr_xy.shape[0] == 0) or (dist_to_fg_local is None):
                    return np.inf
                xs = np.clip(np.round(arr_xy[:, 0]).astype(np.int32), 0, W_img - 1)
                ys = np.clip(np.round(arr_xy[:, 1]).astype(np.int32), 0, H_img - 1)
                return float(np.mean(dist_to_fg_local[ys, xs]))

            def _to_image_points_if_needed(points):
                arr = _sanitize_xy_points(points)
                if arr.shape[0] == 0:
                    return arr
                arr_scaled = arr.astype(np.float32, copy=True)
                arr_scaled[:, 0] *= to_img_x
                arr_scaled[:, 1] *= to_img_y

                min_x = float(np.min(arr[:, 0])); max_x = float(np.max(arr[:, 0]))
                min_y = float(np.min(arr[:, 1])); max_y = float(np.max(arr[:, 1]))
                likely_lat = (
                    min_x >= -1.5 and min_y >= -1.5 and
                    max_x <= W_lat + 1.5 and max_y <= H_lat + 1.5
                )
                likely_img = (
                    min_x >= -1.5 and min_y >= -1.5 and
                    max_x <= W_img + 1.5 and max_y <= H_img + 1.5 and
                    ((max_x > W_lat + 1.5) or (max_y > H_lat + 1.5))
                )
                if likely_lat and (not likely_img):
                    return arr_scaled
                if likely_img and (not likely_lat):
                    return arr

                raw_score = _point_fit_score_local(arr)
                scaled_score = _point_fit_score_local(arr_scaled)
                if np.isfinite(raw_score) and np.isfinite(scaled_score):
                    return arr_scaled if scaled_score < raw_score else arr
                return arr

            if isinstance(current_anchors, np.ndarray) and current_anchors.ndim == 2 and current_anchors.shape[0] > 0:
                anchors_img = _to_image_points_if_needed(current_anchors).astype(np.float32)
                # latent 模式仅“可视化层”对齐到 latent 网格，避免显示成 image 连续坐标锚点。
                if (not mode_is_3d):
                    anchors_lat_vis = anchors_img.copy()
                    anchors_lat_vis[:, 0] = np.round(anchors_lat_vis[:, 0] * to_lat_x)
                    anchors_lat_vis[:, 1] = np.round(anchors_lat_vis[:, 1] * to_lat_y)
                    anchors_img[:, 0] = anchors_lat_vis[:, 0] * to_img_x
                    anchors_img[:, 1] = anchors_lat_vis[:, 1] * to_img_y
                current_anchors = anchors_img
            if pivot_img_np is not None:
                pivot_pts = _to_image_points_if_needed(pivot_img_np)
                if isinstance(pivot_pts, np.ndarray) and pivot_pts.shape[0] > 0:
                    pivot_img_np = [float(pivot_pts[0, 0]), float(pivot_pts[0, 1])]

        # 统一把 debug_info 的 m_end_full 规范为原图分辨率，避免后续合并/可视化维度不一致
        if isinstance(debug_info, dict):
            debug_info = dict(debug_info)
            if runtime_geom_domain == "latent":
                if mode_is_3d:
                    mask_start_vis = np.asarray(runtime_comp_m_start_full, dtype=np.float32)
                else:
                    mask_start_vis = np.asarray(runtime_comp_m_start_full_2d, dtype=np.float32)
            else:
                mask_start_vis = np.asarray(comp_m_start_full, dtype=np.float32)
            debug_info["_runtime_mask_start_vis"] = mask_start_vis

            if runtime_geom_domain == "latent":
                to_img_x = _coord_scale_latent_to_image(W_lat, W_img)
                to_img_y = _coord_scale_latent_to_image(H_lat, H_img)

                def _scale_points_to_image(points):
                    if points is None:
                        return None
                    pts_arr = _to_image_points_if_needed(points)
                    if not isinstance(pts_arr, np.ndarray):
                        return points
                    if pts_arr.shape[0] == 0:
                        return np.zeros((0, 2), dtype=np.float32)
                    src_ndim = points.ndim if torch.is_tensor(points) else np.asarray(points).ndim
                    if src_ndim == 1:
                        return np.asarray([float(pts_arr[0, 0]), float(pts_arr[0, 1])], dtype=np.float32)
                    return pts_arr

                def _scale_masks_to_image(mask_like):
                    if mask_like is None:
                        return None
                    return _latent_mask_to_fullres(mask_like, target_hw=(H_img, W_img))

                root_point_keys = (
                    'anchors_img',
                    'comp_sam_pts',
                    'comp_targets_xy',
                    'phase1_source_points_xy',
                    'phase1_target_points_xy',
                    'phase2_source_points_xy',
                    'phase2_source_points_xy_raw',
                    'phase2_target_points_xy',
                )
                for key in root_point_keys:
                    if key in debug_info:
                        debug_info[key] = _scale_points_to_image(debug_info.get(key))

                if 'pivot_img' in debug_info:
                    debug_info['pivot_img'] = _scale_points_to_image(debug_info.get('pivot_img'))

                for key in ('m_start_full',):
                    if key in debug_info:
                        debug_info[key] = _scale_masks_to_image(debug_info.get(key))

                for phase_key in ('phase1', 'phase2'):
                    phase_dbg = debug_info.get(phase_key)
                    if not isinstance(phase_dbg, dict):
                        continue
                    phase_dbg = dict(phase_dbg)
                    for key in ('comp_sam_pts', 'comp_targets_xy', 'rotated_sam_pts', 'anchors_img', 'pivot_img'):
                        if key in phase_dbg:
                            phase_dbg[key] = _scale_points_to_image(phase_dbg.get(key))
                    for key in ('m_start_full', 'm_end_full'):
                        if key in phase_dbg:
                            phase_dbg[key] = _scale_masks_to_image(phase_dbg.get(key))
                    debug_info[phase_key] = phase_dbg
            debug_info['runtime_geom_domain'] = str(runtime_geom_domain)
            debug_info['_points_already_image_space'] = True

            m_end_dbg_full = None
            m_end_dbg = debug_info.get('m_end_full')
            # 3D projective 后端优先使用显式 m_end_full，避免 norm_grid 的 identity 区把旧位置带回。
            if isinstance(m_end_dbg, np.ndarray):
                if m_end_dbg.shape[:2] != (H_img, W_img):
                    m_end_dbg_full = _latent_mask_to_fullres(m_end_dbg, target_hw=(H_img, W_img))
                else:
                    m_end_dbg_full = np.clip(m_end_dbg.astype(np.float32), 0.0, 1.0)

            # 若没有显式 m_end_full，再使用 norm_grid 在源图分辨率重建 end mask。
            if m_end_dbg_full is None and norm_grid is not None:
                try:
                    if torch.is_tensor(norm_grid):
                        ng = norm_grid.detach().to(device=device, dtype=torch.float32)
                    else:
                        ng = torch.from_numpy(np.asarray(norm_grid, dtype=np.float32)).to(device=device)
                    if ng.ndim == 4:
                        ng = ng[0]
                    if ng.ndim == 3 and ng.shape[-1] == 2:
                        grid_tensor = ng.permute(2, 0, 1).unsqueeze(0)
                        grid_up = F.interpolate(
                            grid_tensor,
                            size=(H_img, W_img),
                            mode='bilinear',
                            align_corners=True,
                        ).permute(0, 2, 3, 1)

                        m_start_src = comp_m_start_full
                        if (not isinstance(m_start_src, np.ndarray)) or (m_start_src.shape[:2] != (H_img, W_img)):
                            m_start_src = _latent_mask_to_fullres(m_start_src, target_hw=(H_img, W_img))
                        m_start_t = torch.from_numpy(np.asarray(m_start_src, dtype=np.float32)).to(device).view(1, 1, H_img, W_img)
                        m_end_t = F.grid_sample(m_start_t, grid_up, mode='nearest', align_corners=True)
                        m_end_dbg_full = np.clip(m_end_t.squeeze().detach().cpu().numpy(), 0.0, 1.0).astype(np.float32)
                except Exception:
                    m_end_dbg_full = None

            if m_end_dbg_full is None:
                m_end_dbg_full = _latent_mask_to_fullres(
                    m_end_dbg if m_end_dbg is not None else comp_mask_end,
                    target_hw=(H_img, W_img),
                )
            debug_info['m_end_full'] = m_end_dbg_full

        # ============================================================
        # 统一保存结果
        # ============================================================
        all_results['latents'].append(subject_latents_refined)
        all_results['masks_start'].append(comp_m_start_full)
        all_results['masks_end'].append(comp_mask_end)
        all_results['grids'].append(norm_grid)
        all_results['handles'].append(result_handles_xy)
        all_results['targets'].append(result_targets_xy)
        all_results['pivots'].append(pivot_img_np)
        all_results['anchors'].append(current_anchors)
        all_results['actions'].append(action)
        all_results['sub_actions'].append(sub_action)
        all_results['debug_infos'].append(debug_info)

    if hole_fill_mode in {"sgf", "lama"}:
        # ============================================================
        # Structure-Guided Fill inline：背景预填 + 主体贴回
        # ============================================================
        m_end_overall = _compute_overall_end_mask(
            all_results=all_results,
            H_img=H_img,
            W_img=W_img,
            device=device,
        )
        start_full_small = (m_start_full_overall > 0.5).astype(np.float32)

        # SGF: 背景补洞严格由初始主体区域(start mask)驱动。
        bg_hole_base_mask = start_full_small.copy()

        bg_hole_mask, bg_expand_info = _expand_background_fill_hole_mask(bg_hole_base_mask)
        if bool(bg_expand_info.get("enabled", False)):
            print(
                f"[Background Fill][SGF] hole expand: px={bg_expand_info.get('expand_px', 0)}, "
                f"k={bg_expand_info.get('kernel', 0)}, area={bg_expand_info.get('area_before', 0)}->{bg_expand_info.get('area_after', 0)}"
            )

        print("\n[Background Fill][SGF] Processing...")
        bg_clean_lat_for_plan = None
        lama_latents, lama_rgb, lama_clean_prior = (None, None, None)
        if hole_fill_mode == "lama":
            lama_latents, lama_rgb, lama_clean_prior = _fill_background_holes_lama_sgf(
                latents=latents,
                source_image_np=source_image_np,
                hole_mask_full=bg_hole_mask,
                forbidden_mask_full=m_pseudo_full_overall,
                device=device,
                vae=vae,
                clean_latents=clean_latents,
                alpha_prod_t=alpha_prod_t,
            )
        if lama_latents is not None:
            background_latents = lama_latents.to(dtype=latents.dtype)
            bg_clean_lat_for_plan = lama_clean_prior
            if use_drag_guided_prefill:
                print("[Background Fill][SGF/LaMa] drag-guided prefill bypassed.")
        elif use_drag_guided_prefill:
            guided_background, residual_hole_mask, pseudo_obj_mask, guided_prefill_px, residual_hole_n = (
                _apply_drag_guided_prefill_to_latents(
                    latents=latents,
                    background_hole_full=bg_hole_mask,
                    pseudo_subject_full=m_pseudo_full_overall,
                    component_masks_start=all_results.get("masks_start", []),
                    component_masks_end=all_results.get("masks_end", []),
                    component_handles=all_results.get("handles", []),
                    component_targets=all_results.get("targets", []),
                    device=device,
                )
            )
            if guided_prefill_px > 0:
                print(
                    f"[Background Fill][SGF] drag-guided prefill: filled={guided_prefill_px}, residual={residual_hole_n}"
                )
            background_latents = fill_background_holes(
                image_input=guided_background,
                hole_mask=residual_hole_mask,
                forbidden_mask=pseudo_obj_mask,
                device=device,
            ).to(dtype=latents.dtype)
        else:
            background_latents = fill_background_holes(
                image_input=latents,
                hole_mask=bg_hole_mask,
                forbidden_mask=m_pseudo_full_overall,
                device=device,
            ).to(dtype=latents.dtype)

        bg_filled_rgb = None
        if visualize_drag:
            if lama_rgb is not None:
                bg_filled_rgb = lama_rgb
            else:
                bg_filled_rgb = fill_background_holes(
                    image_input=source_image_np,
                    hole_mask=bg_hole_mask,
                    forbidden_mask=m_pseudo_full_overall,
                    device=device,
                )

        fill_vis_debug = {
            "subject_start_full": np.asarray(start_full_small, dtype=np.float32),
            "hole_total_full": np.asarray(bg_hole_mask, dtype=np.float32),
            "hole_inside_full": np.zeros((H_img, W_img), dtype=np.float32),
            "hole_outside_full": np.asarray(bg_hole_mask, dtype=np.float32),
            "hole_inside_full_vis": np.zeros((H_img, W_img), dtype=np.float32),
            "hole_outside_full_vis": np.asarray(bg_hole_mask, dtype=np.float32),
            "hole_visual_mode": str(hole_fill_mode),
            "components": [],
        }
        bg_hole_mask_for_vis = np.asarray(bg_hole_mask, dtype=np.float32)

        # 合成：主体按旧版 subject-scope 策略补洞后贴回
        print("\n[Composition][SGF] Merging...")
        final_latents = background_latents.clone()
        for idx, (warped_latent, mask_end) in enumerate(zip(all_results["latents"], all_results["masks_end"])):
            refined_latent = warped_latent
            if torch.is_tensor(mask_end):
                refined_mask = mask_end.to(device=device, dtype=latents.dtype)
            else:
                refined_mask = torch.from_numpy(np.asarray(mask_end, dtype=np.float32)).to(device=device, dtype=latents.dtype)

            m_new = refined_mask.unsqueeze(0).unsqueeze(0)
            final_latents = refined_latent * m_new + final_latents * (1 - m_new)

        m_end_overall = _compute_overall_end_mask(
            all_results=all_results,
            H_img=H_img,
            W_img=W_img,
            device=device,
        )

        subject_end_full = (np.asarray(m_end_overall, dtype=np.float32) > 0.5).astype(np.float32)
        subject_end_lat_u8 = _mask_to_binary_uint8(
            _downsample_image_mask_to_latent(
                subject_end_full,
                target_hw=(H_lat, W_lat),
                threshold=0.10,
            ),
            target_hw=(H_lat, W_lat),
        )
        subject_hole_full_debug = _collect_subject_hole_mask_from_results(
            all_results=all_results,
            target_hw=(H_img, W_img),
        )
        subject_hole_lat_debug = _collect_subject_hole_lat_mask_from_results(
            all_results=all_results,
            target_hw=(H_lat, W_lat),
        )
        subject_hole_lat_for_plan = np.zeros((H_lat, W_lat), dtype=np.float32)
        if bool(enable_3d_subject_scope_fill):
            _, subject_hole_full_end, _ = infer_subject_fill_scope(
                subject_end_full.astype(np.float32),
                close_ratio=0.10,
                max_close=35,
                max_fill_dist_ratio=0.18,
            )
            subject_hole_full = np.maximum(subject_hole_full_end, subject_hole_full_debug).astype(np.float32)
            subject_hole_full = np.logical_and(
                subject_hole_full > 0.5,
                subject_end_full > 0.5,
            ).astype(np.float32)
            hole_inside_lat = _downsample_image_mask_to_latent(
                subject_hole_full,
                target_hw=(H_lat, W_lat),
                threshold=0.15,
            )
            hole_inside_lat = np.logical_and(hole_inside_lat > 0.5, subject_end_lat_u8 > 0).astype(np.float32)
            hole_inside_lat = np.maximum(
                hole_inside_lat,
                np.logical_and(subject_hole_lat_debug > 0.5, subject_end_lat_u8 > 0).astype(np.float32),
            ).astype(np.float32)
            subject_hole_lat_for_plan = hole_inside_lat.astype(np.float32)
            donor_inside_lat_subject = np.logical_and(subject_end_lat_u8 > 0, hole_inside_lat <= 0.5).astype(np.float32)
            print(
                f"[Subject Fill][SGF] hole_inside_lat={int(np.sum(hole_inside_lat > 0.5))}, "
                f"hole_lat_debug={int(np.sum(subject_hole_lat_debug > 0.5))}, "
                f"donor_inside_lat={int(np.sum(donor_inside_lat_subject > 0.5))}"
            )
            if int(np.sum(hole_inside_lat > 0.5)) > 0:
                final_latents, _, residual_sub_1, info_sub_1 = fill_holes_with_bnni(
                    image_input=final_latents,
                    hole_mask=hole_inside_lat.astype(np.float32),
                    donor_mask=donor_inside_lat_subject.astype(np.float32),
                    device=device,
                    local_boundary_only=True,
                    return_debug=True,
                )
                print(
                    f"[Subject Fill][SGF] post-compose(pass1) "
                    f"filled={info_sub_1.get('filled',0)}, residual={info_sub_1.get('residual',0)}"
                )
                if int(np.sum(residual_sub_1 > 0.5)) > 0:
                    final_latents, _, _, info_sub_2 = fill_holes_with_bnni(
                        image_input=final_latents,
                        hole_mask=residual_sub_1.astype(np.float32),
                        donor_mask=donor_inside_lat_subject.astype(np.float32),
                        device=device,
                        local_boundary_only=False,
                        return_debug=True,
                    )
                    print(
                        f"[Subject Fill][SGF] post-compose(pass2) "
                        f"filled={info_sub_2.get('filled',0)}, residual={info_sub_2.get('residual',0)}"
                    )
        else:
            subject_hole_full = np.zeros((H_img, W_img), dtype=np.float32)
            print("[Subject Fill][SGF] disabled by enable_3d_subject_scope_fill=False")

        fill_vis_debug["subject_end_full"] = np.asarray(m_end_overall, dtype=np.float32)
        subject_hole_full_debug = np.asarray(subject_hole_full, dtype=np.float32)
        hole_outside_vis = np.logical_and(
            np.asarray(bg_hole_mask, dtype=np.float32) > 0.5,
            subject_hole_full_debug <= 0.5,
        ).astype(np.float32)
        fill_vis_debug["subject_hole_mask_full"] = np.asarray(subject_hole_full_debug, dtype=np.float32)
        fill_vis_debug["hole_inside_full"] = np.asarray(subject_hole_full_debug, dtype=np.float32)
        fill_vis_debug["hole_inside_full_vis"] = np.asarray(subject_hole_full_debug, dtype=np.float32)
        fill_vis_debug["hole_outside_full"] = np.asarray(hole_outside_vis, dtype=np.float32)
        fill_vis_debug["hole_outside_full_vis"] = np.asarray(hole_outside_vis, dtype=np.float32)

        if visualize_drag:
            print("\n[Visualization] Generating debug images...")
            try:
                from utils_drag.drag_visualization import (
                    _build_final_composite_rgb,
                    run_all_visualizations,
                )

                run_all_visualizations(
                    source_image_np,
                    all_results,
                    m_pseudo_full_overall,
                    start_full_small,
                    m_end_overall,
                    all_sam_pts,
                    all_targets_xy,
                    bg_filled_rgb,
                    drag_mode,
                    CURRENT_DEBUG_DIR,
                    device,
                    bg_hole_mask=bg_hole_mask_for_vis,
                    fill_vis_debug=fill_vis_debug,
                )
                try:
                    cv2.imwrite(
                        os.path.join(MASK_DEBUG_ROOT, "05_background_hole_mask.png"),
                        (np.clip(bg_hole_mask_for_vis, 0.0, 1.0) * 255).astype(np.uint8),
                    )
                    if isinstance(bg_filled_rgb, np.ndarray):
                        cv2.imwrite(
                            os.path.join(MASK_DEBUG_ROOT, "05_background_filled_rgb.jpg"),
                            cv2.cvtColor(np.clip(bg_filled_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                        )
                    final_composed_vis = _build_final_composite_rgb(
                        source_image_np=source_image_np,
                        all_results=all_results,
                        bg_filled_rgb=bg_filled_rgb,
                        device=device,
                    )
                    if isinstance(final_composed_vis, np.ndarray):
                        cv2.imwrite(
                            os.path.join(MASK_DEBUG_ROOT, "05_final_composed_rgb.jpg"),
                            cv2.cvtColor(final_composed_vis, cv2.COLOR_RGB2BGR),
                        )
                except Exception:
                    pass
                print("[Visualization] ✓ Complete")
            except Exception as e:
                print(f"[Warning] Visualization Error: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n[Done] Processed {len(all_results['latents'])} components")
        print(f"  Actions: {', '.join(all_results['actions'])}")
        print(f"  Sub-Actions: {', '.join(all_results['sub_actions'])}")

        drag_plan_out = _build_drag_plan(
            drag_mode=drag_mode,
            m_start_full_overall=start_full_small,
            m_pseudo_full_overall=m_pseudo_full_overall,
            background_hole_full=bg_hole_mask,
            background_hole_base_full=bg_hole_base_mask,
            component_norm_grids=all_results["grids"],
            component_masks_start=all_results["masks_start"],
            component_masks_end=all_results["masks_end"],
            component_handles=all_results["handles"],
            component_targets=all_results["targets"],
            hole_fill_mode=hole_fill_mode,
            use_drag_guided_prefill=use_drag_guided_prefill,
            enable_subject_hole_fill=enable_3d_subject_scope_fill,
            subject_hole_mask_lat=subject_hole_lat_for_plan,
            background_filled_clean_latent=bg_clean_lat_for_plan,
        )
        if return_plan:
            return final_latents, drag_plan_out
        return final_latents

    # 先计算全局 end mask，再定义背景补全区域（由 start mask 直接驱动）。
    m_end_overall = _compute_overall_end_mask(
        all_results=all_results,
        H_img=H_img,
        W_img=W_img,
        device=device,
    )
    start_full_small = (m_start_full_overall > 0.5).astype(np.float32)
    end_full_small = (m_end_overall > 0.5).astype(np.float32)

    # 背景补全区域：使用 start mask 驱动外部洞。
    # 对 sgf+* 模式，强制与 sgf 使用同一 expand 逻辑，确保外部补全一致。
    bg_hole_base_mask = start_full_small.copy()
    sgf_like_mode = hole_fill_mode in {"sgf+bnni"}
    if sgf_like_mode:
        bg_hole_mask, bg_expand_info = _expand_background_fill_hole_mask(bg_hole_base_mask)
        if bool(bg_expand_info.get("enabled", False)):
            print(
                f"[Mask Expand] sgf-aligned: px={bg_expand_info.get('expand_px', 0)}, "
                f"k={bg_expand_info.get('kernel', 0)}, area={bg_expand_info.get('area_before', 0)}->{bg_expand_info.get('area_after', 0)}"
            )
    else:
        if use_expanded_subject_fill and expanded_subject_fill_px > 0:
            k_full = _to_odd_kernel(2 * int(expanded_subject_fill_px) + 1)
            bg_hole_mask = _dilate_binary_mask(bg_hole_base_mask, k_full).astype(np.float32)
            print(
                f"[Mask Expand] enabled: fill_hole_from_start_mask, full_px={expanded_subject_fill_px}, k_full={k_full}"
            )
        else:
            bg_hole_mask = bg_hole_base_mask.copy()
            print("[Mask Expand] disabled: fill_hole_from_start_mask")

    # 轻度平滑 hole 边界，降低硬边补全痕迹
    def _stabilize_hole_mask(mask01):
        raw = (mask01 > 0.5).astype(np.float32)
        smoothed = _smooth_binary_mask_shape(
            raw,
            close_ratio=0.004,
            open_ratio=0.003,
            blur_ratio=0.003,
        )
        return np.logical_or(smoothed > 0.5, raw > 0.5).astype(np.float32)

    if not sgf_like_mode:
        # base: 仅用于主体内部空洞分区（不受扩张影响）
        bg_hole_base_mask = _stabilize_hole_mask(bg_hole_base_mask)
        # expanded/non-expanded: 仅用于背景补洞总区域
        bg_hole_mask = _stabilize_hole_mask(bg_hole_mask)

    # 背景填补：以扩张后的 start mask 为主
    # 分为主体内空洞 / 主体外空洞：
    # - inner: BNNI（主体 donor）
    # - outer: direct fill（当前 SGF 路径）
    print("\n[Background Fill] Processing (start-mask-driven, direct-fill + inner-BNNI)...")
    bg_hole_mask_for_vis = np.asarray(bg_hole_mask, dtype=np.float32)

    # 当前主体支持域：由 end mask（当前主体）推导，不再绑定 start。
    subject_full_u8 = _mask_to_binary_uint8(end_full_small, target_hw=(H_img, W_img))
    subject_lat_u8 = _mask_to_binary_uint8(
        _downsample_image_mask_to_latent(
            end_full_small,
            target_hw=(H_lat, W_lat),
            threshold=0.10,
        ),
        target_hw=(H_lat, W_lat),
    )
    bg_hole_full_u8 = _mask_to_binary_uint8(bg_hole_mask, target_hw=(H_img, W_img))
    bg_hole_lat_u8 = _mask_to_binary_uint8(bg_hole_mask, target_hw=(H_lat, W_lat))
    bg_hole_base_full_u8 = _mask_to_binary_uint8(bg_hole_base_mask, target_hw=(H_img, W_img))

    # 主体支持域：用于构造主体内补洞 donor。
    subject_full_clean = _clean_binary_mask_u8(
        subject_full_u8,
        close_k=0,
        open_k=0,
        min_area=max(8, int(round(0.0002 * H_img * W_img))),
        keep_largest_if_empty=True,
    )
    subject_lat_clean = _clean_binary_mask_u8(
        subject_lat_u8,
        close_k=0,
        open_k=0,
        min_area=max(2, int(round(0.003 * H_lat * W_lat))),
        keep_largest_if_empty=True,
    )

    # 主体补洞支持域使用当前主体（end）清洗后掩码。
    subject_full_fill = (subject_full_clean > 0).astype(np.float32)
    subject_lat_fill = (subject_lat_clean > 0).astype(np.float32)

    subject_full_fill_u8 = _mask_to_binary_uint8(subject_full_fill, target_hw=(H_img, W_img))
    subject_lat_fill_u8 = _mask_to_binary_uint8(subject_lat_fill, target_hw=(H_lat, W_lat))

    # 双通道：
    # 1) 背景 hole：由 start 驱动（SGF direct fill）
    # 2) 主体内部 hole：由当前主体(end/scope)推断（BNNI）
    _, subject_hole_full_end, _ = infer_subject_fill_scope(
        subject_full_fill_u8.astype(np.float32),
        close_ratio=0.10,
        max_close=35,
        max_fill_dist_ratio=0.18,
    )
    subject_hole_full_debug = _collect_subject_hole_mask_from_results(
        all_results=all_results,
        target_hw=(H_img, W_img),
    )
    subject_hole_lat_debug = _collect_subject_hole_lat_mask_from_results(
        all_results=all_results,
        target_hw=(H_lat, W_lat),
    )
    hole_inside_full = np.maximum(subject_hole_full_end, subject_hole_full_debug).astype(np.float32)
    hole_inside_full = np.logical_and(hole_inside_full > 0.5, subject_full_fill_u8 > 0).astype(np.float32)
    hole_outside_full = (bg_hole_full_u8 > 0).astype(np.float32)

    # 可视化分层与实际填充路径保持一致。
    hole_inside_full_vis = hole_inside_full.copy().astype(np.float32)
    hole_outside_full_vis = hole_outside_full.copy().astype(np.float32)

    # latent 分类：
    # - inside: 当前主体内部洞
    # - outside: start 驱动的背景洞
    hole_inside_lat = _downsample_image_mask_to_latent(
        hole_inside_full,
        target_hw=(H_lat, W_lat),
        threshold=0.15,
    )
    hole_inside_lat = np.logical_and(hole_inside_lat > 0.5, subject_lat_fill_u8 > 0).astype(np.float32)
    hole_inside_lat = np.maximum(
        hole_inside_lat,
        np.logical_and(subject_hole_lat_debug > 0.5, subject_lat_fill_u8 > 0).astype(np.float32),
    ).astype(np.float32)
    hole_outside_lat = (bg_hole_lat_u8 > 0).astype(np.float32)

    if use_expanded_subject_fill and expanded_subject_fill_px > 0:
        extra_outer = int(np.sum(np.logical_and(bg_hole_full_u8 > 0, bg_hole_base_full_u8 <= 0)))
        print(f"[Background Fill] mask-expand mode: keep inner-subject fill, extra_outer={extra_outer}")
    print(
        f"[Subject Hole] end_scope={int(np.sum(subject_hole_full_end > 0.5))}, "
        f"debug={int(np.sum(subject_hole_full_debug > 0.5))}, "
        f"debug_lat={int(np.sum(subject_hole_lat_debug > 0.5))}, "
        f"merged={int(np.sum(hole_inside_full > 0.5))}"
    )

    # donor 分类：
    # - sgf+bnni：inner hole 仅使用主体 donor
    # BNNI 默认 local_boundary_only=True，会优先在当前 hole 边界邻域取最近 donor。
    donor_inside_full_subject = np.logical_and(subject_full_fill_u8 > 0, hole_inside_full <= 0.5).astype(np.float32)
    donor_inside_lat_subject = np.logical_and(subject_lat_fill_u8 > 0, hole_inside_lat <= 0.5).astype(np.float32)
    donor_inner_full = donor_inside_full_subject
    donor_inner_lat = donor_inside_lat_subject

    fill_vis_debug = {
        "subject_start_full": np.asarray(start_full_small, dtype=np.float32),
        "hole_total_full_base": bg_hole_base_mask.astype(np.float32),
        "hole_total_full": bg_hole_mask.astype(np.float32),
        "hole_inside_full_vis": hole_inside_full_vis.astype(np.float32),
        "hole_outside_full_vis": hole_outside_full_vis.astype(np.float32),
        "hole_inside_full": hole_inside_full.astype(np.float32),
        "hole_outside_full": hole_outside_full.astype(np.float32),
        "hole_visual_mode": str(hole_fill_mode),
        "components": [],
    }

    # ===== latent 空间补全（用于最终合成）=====
    hole_total_lat = (bg_hole_lat_u8 > 0).astype(np.float32)
    donor_all_lat = (bg_hole_lat_u8 <= 0).astype(np.float32)

    if hole_fill_mode in {"bnni_all"}:
        print(f"[Background Fill] mode={hole_fill_mode} (global BNNI)")
        background_latents = latents.clone()
        filled_lat_total_u8 = np.zeros((H_lat, W_lat), dtype=np.uint8)
        residual_lat_u8 = hole_total_lat.astype(np.float32)

        if int(np.sum(hole_total_lat > 0.5)) > 0 and int(np.sum(donor_all_lat > 0.5)) > 0:
            background_latents, filled_all_1, residual_all_1, info_all_1 = fill_holes_with_bnni(
                image_input=background_latents,
                hole_mask=hole_total_lat.astype(np.float32),
                donor_mask=donor_all_lat.astype(np.float32),
                device=device,
                local_boundary_only=True,
                return_debug=True,
            )
            filled_lat_total_u8 = np.maximum(filled_lat_total_u8, (filled_all_1 > 0.5).astype(np.uint8))
            print(
                f"[Background Fill][Latent] global bnni(pass1) "
                f"filled={info_all_1.get('filled',0)}, residual={info_all_1.get('residual',0)}"
            )
            if int(np.sum(residual_all_1 > 0.5)) > 0:
                background_latents, filled_all_2, residual_all_2, info_all_2 = fill_holes_with_bnni(
                    image_input=background_latents,
                    hole_mask=residual_all_1.astype(np.float32),
                    donor_mask=donor_all_lat.astype(np.float32),
                    device=device,
                    local_boundary_only=False,
                    return_debug=True,
                )
                filled_lat_total_u8 = np.maximum(filled_lat_total_u8, (filled_all_2 > 0.5).astype(np.uint8))
                residual_lat_u8 = residual_all_2.astype(np.float32)
                print(
                    f"[Background Fill][Latent] global bnni(pass2) "
                    f"filled={info_all_2.get('filled',0)}, residual={info_all_2.get('residual',0)}"
                )
            else:
                residual_lat_u8 = residual_all_1.astype(np.float32)

        filled_lat_in_u8 = np.logical_and(filled_lat_total_u8 > 0, hole_inside_lat > 0.5).astype(np.uint8)
        filled_lat_out_u8 = np.logical_and(filled_lat_total_u8 > 0, hole_inside_lat <= 0.5).astype(np.uint8)
        residual_lat_u8 = np.logical_and(hole_total_lat > 0.5, filled_lat_total_u8 <= 0).astype(np.float32)
    else:
        # sgf+bnni：先补外洞（SGF），再补主体内洞（BNNI）
        donor_inner_lat = donor_inside_lat_subject
        donor_inner_full = donor_inside_full_subject
        print(f"[Background Fill] mode={hole_fill_mode} (inner donor: subject-only)")
        background_latents = fill_background_holes(
            image_input=latents,
            hole_mask=hole_outside_lat.astype(np.float32),
            forbidden_mask=m_pseudo_full_overall,
            device=device,
        )
        filled_lat_out_u8 = (hole_outside_lat > 0.5).astype(np.uint8)
        filled_lat_in_u8 = np.zeros((H_lat, W_lat), dtype=np.uint8)

        if int(np.sum(hole_inside_lat > 0.5)) > 0:
            background_latents, filled_in_lat_1, residual_in_lat_1, info_in_1 = fill_holes_with_bnni(
                image_input=background_latents,
                hole_mask=hole_inside_lat.astype(np.float32),
                donor_mask=donor_inner_lat,
                device=device,
                local_boundary_only=True,
                return_debug=True,
            )
            filled_lat_in_u8 = np.maximum(filled_lat_in_u8, (filled_in_lat_1 > 0.5).astype(np.uint8))
            print(
                f"[Background Fill][Latent] inner bnni(pass1) "
                f"filled={info_in_1.get('filled',0)}, residual={info_in_1.get('residual',0)}"
            )

            if int(np.sum(residual_in_lat_1 > 0.5)) > 0:
                background_latents, filled_in_lat_2, residual_in_lat_2, info_in_2 = fill_holes_with_bnni(
                    image_input=background_latents,
                    hole_mask=residual_in_lat_1.astype(np.float32),
                    donor_mask=donor_inner_lat,
                    device=device,
                    local_boundary_only=False,
                    return_debug=True,
                )
                filled_lat_in_u8 = np.maximum(filled_lat_in_u8, (filled_in_lat_2 > 0.5).astype(np.uint8))
                print(
                    f"[Background Fill][Latent] inner bnni(pass2) "
                    f"filled={info_in_2.get('filled',0)}, residual={info_in_2.get('residual',0)}"
                )

        residual_lat_u8 = np.logical_and(
            hole_inside_lat > 0.5,
            filled_lat_in_u8 <= 0,
        ).astype(np.float32)

    if torch.is_tensor(background_latents):
        background_latents = background_latents.to(dtype=latents.dtype)
    print(
        f"[Background Fill] latent inner={int(np.sum(filled_lat_in_u8 > 0))}, "
        f"outer={int(np.sum(filled_lat_out_u8 > 0))}, "
        f"residual={int(np.sum(residual_lat_u8 > 0.5))}"
    )

    # ===== RGB 空间补全（仅用于可视化）=====
    bg_filled_rgb = None
    bg_filled_rgb_outer = None
    if visualize_drag:
        try:
            # Step-1: 仅外空洞做 direct fill
            bg_filled_rgb_outer = fill_background_holes(
                image_input=source_image_np,
                hole_mask=hole_outside_full.astype(np.float32),
                forbidden_mask=m_pseudo_full_overall,
                device=device,
            )
            bg_filled_rgb = bg_filled_rgb_outer.copy()
            filled_img_in_u8 = np.zeros((H_img, W_img), dtype=np.uint8)
            filled_img_out_u8 = (hole_outside_full > 0.5).astype(np.uint8)

            hole_total_full_u8 = np.logical_or(bg_hole_full_u8 > 0, hole_inside_full > 0.5).astype(np.uint8)
            hole_in_full = (hole_inside_full > 0.5).astype(np.float32)
            hole_out_full = (bg_hole_full_u8 > 0).astype(np.float32)
            hole_in_full_vis = (hole_inside_full_vis > 0.5).astype(np.float32)
            hole_out_full_vis = (hole_outside_full_vis > 0.5).astype(np.float32)
            before_rgb = bg_filled_rgb_outer.copy()

            filled_in_full = np.zeros((H_img, W_img), dtype=np.float32)
            filled_out_full = hole_out_full.astype(np.float32)
            if int(np.sum(hole_in_full > 0.5)) > 0:
                bg_filled_rgb, filled_in_1, residual_in_1, _ = fill_holes_with_bnni(
                    image_input=bg_filled_rgb,
                    hole_mask=hole_in_full,
                    donor_mask=donor_inner_full,
                    device=device,
                    local_boundary_only=True,
                    return_debug=True,
                )
                filled_in_full = np.maximum(filled_in_full, filled_in_1.astype(np.float32))
                if int(np.sum(residual_in_1 > 0.5)) > 0:
                    bg_filled_rgb, filled_in_2, _, _ = fill_holes_with_bnni(
                        image_input=bg_filled_rgb,
                        hole_mask=residual_in_1.astype(np.float32),
                        donor_mask=donor_inner_full,
                        device=device,
                        local_boundary_only=False,
                        return_debug=True,
                    )
                    filled_in_full = np.maximum(filled_in_full, filled_in_2.astype(np.float32))

            filled_img_in_u8 = np.maximum(filled_img_in_u8, (filled_in_full > 0.5).astype(np.uint8))
            filled_img_out_u8 = np.maximum(filled_img_out_u8, (filled_out_full > 0.5).astype(np.uint8))

            fill_vis_debug["components"].append({
                "index": 0,
                "hole_total_full": hole_total_full_u8.astype(np.float32),
                "hole_in_full_vis": hole_in_full_vis.astype(np.float32),
                "hole_out_full_vis": hole_out_full_vis.astype(np.float32),
                "hole_in_full": hole_in_full.astype(np.float32),
                "hole_out_full": hole_out_full.astype(np.float32),
                "filled_in_full": filled_in_full.astype(np.float32),
                "filled_out_full": filled_out_full.astype(np.float32),
                "before_rgb": before_rgb,
                "after_rgb": bg_filled_rgb.copy(),
            })

            print(
                f"[Background Fill] rgb inner={int(np.sum(filled_img_in_u8 > 0))}, "
                f"outer={int(np.sum(filled_img_out_u8 > 0))}, "
                f"residual={0}"
            )
        except Exception as e:
            print(f"[Background Fill][RGB] failed: {e}")
            bg_filled_rgb = source_image_np.copy()
            bg_filled_rgb_outer = bg_filled_rgb.copy()
    # 合成
    print("\n[Composition] Merging...")
    final_latents = background_latents.clone()
    # 主体面积保持不变：最终贴回严格使用原始 end mask（mask小），不做扩张。
    for idx, (warped_latent, mask_end) in enumerate(zip(all_results['latents'], all_results['masks_end'])):
        refined_latent = warped_latent
        refined_mask = mask_end
        m_new = refined_mask.unsqueeze(0).unsqueeze(0)
        final_latents = refined_latent * m_new + final_latents * (1 - m_new)

    # sgf+bnni：在最终合成后再次执行主体内补洞，确保真正作用于最终结果。
    if hole_fill_mode == "sgf+bnni" and int(np.sum(hole_inside_lat > 0.5)) > 0:
        donor_post_lat = np.logical_and(subject_lat_fill_u8 > 0, hole_inside_lat <= 0.5).astype(np.float32)
        final_latents, filled_post_1, residual_post_1, info_post_1 = fill_holes_with_bnni(
            image_input=final_latents,
            hole_mask=hole_inside_lat.astype(np.float32),
            donor_mask=donor_post_lat.astype(np.float32),
            device=device,
            local_boundary_only=True,
            return_debug=True,
        )
        print(
            f"[Subject Fill] post-compose bnni(pass1) "
            f"filled={info_post_1.get('filled',0)}, residual={info_post_1.get('residual',0)}"
        )
        if int(np.sum(residual_post_1 > 0.5)) > 0:
            final_latents, filled_post_2, residual_post_2, info_post_2 = fill_holes_with_bnni(
                image_input=final_latents,
                hole_mask=residual_post_1.astype(np.float32),
                donor_mask=donor_post_lat.astype(np.float32),
                device=device,
                local_boundary_only=False,
                return_debug=True,
            )
            print(
                f"[Subject Fill] post-compose bnni(pass2) "
                f"filled={info_post_2.get('filled',0)}, residual={info_post_2.get('residual',0)}"
            )
            filled_post = np.maximum((filled_post_1 > 0.5).astype(np.uint8), (filled_post_2 > 0.5).astype(np.uint8))
            residual_post = residual_post_2.astype(np.float32)
        else:
            filled_post = (filled_post_1 > 0.5).astype(np.uint8)
            residual_post = residual_post_1.astype(np.float32)

        filled_lat_in_u8 = np.maximum(filled_lat_in_u8, filled_post)
        residual_lat_u8 = np.logical_and(hole_inside_lat > 0.5, filled_lat_in_u8 <= 0).astype(np.float32)
    # 计算最终合并 End Mask（更新过主体补洞后再计算一次）。
    m_end_overall = _compute_overall_end_mask(
        all_results=all_results,
        H_img=H_img,
        W_img=W_img,
        device=device,
    )
    if isinstance(fill_vis_debug, dict):
        fill_vis_debug["subject_end_full"] = np.asarray(m_end_overall, dtype=np.float32)
        fill_vis_debug["subject_hole_mask_full"] = np.asarray(hole_inside_full, dtype=np.float32)

    # 可视化 (调用统一入口)
    if visualize_drag:
        print("\n[Visualization] Generating debug images...")
        try:
            from utils_drag.drag_visualization import (
                _build_final_composite_rgb,
                run_all_visualizations,
            )

            # Hybrid 模式需要完整的 all_results（包含 debug_infos）
            vis_bg_rgb = bg_filled_rgb_outer if isinstance(bg_filled_rgb_outer, np.ndarray) else bg_filled_rgb
            run_all_visualizations(
                source_image_np, all_results, m_pseudo_full_overall, start_full_small,
                m_end_overall, all_sam_pts, all_targets_xy, vis_bg_rgb,
                drag_mode, CURRENT_DEBUG_DIR, device, bg_hole_mask=bg_hole_mask_for_vis, fill_vis_debug=fill_vis_debug
            )
            try:
                cv2.imwrite(
                    os.path.join(MASK_DEBUG_ROOT, "05_background_hole_mask.png"),
                    (np.clip(bg_hole_mask_for_vis, 0.0, 1.0) * 255).astype(np.uint8),
                )
                if isinstance(vis_bg_rgb, np.ndarray):
                    cv2.imwrite(
                        os.path.join(MASK_DEBUG_ROOT, "05_background_filled_rgb.jpg"),
                        cv2.cvtColor(np.clip(vis_bg_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                    )
                final_composed_vis = _build_final_composite_rgb(
                    source_image_np=source_image_np,
                    all_results=all_results,
                    bg_filled_rgb=vis_bg_rgb,
                    device=device,
                )
                if isinstance(final_composed_vis, np.ndarray):
                    cv2.imwrite(
                        os.path.join(MASK_DEBUG_ROOT, "05_final_composed_rgb.jpg"),
                        cv2.cvtColor(final_composed_vis, cv2.COLOR_RGB2BGR),
                    )
                for historical_name in (
                    "04_effect_mask.png",
                    "04_insert_mask.png",
                    "04_vacate_mask.png",
                    "04_trail_mask.png",
                ):
                    _ = historical_name  # 保留旧命名文件，不再删除
            except Exception:
                pass
            print("[Visualization] ✓ Complete")
        except Exception as e:
            print(f"[Warning] Visualization Error: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n[Done] Processed {len(all_results['latents'])} components")
    print(f"  Actions: {', '.join(all_results['actions'])}")
    print(f"  Sub-Actions: {', '.join(all_results['sub_actions'])}")
    
    drag_plan_out = _build_drag_plan(
        drag_mode=drag_mode,
        m_start_full_overall=start_full_small,
        m_pseudo_full_overall=m_pseudo_full_overall,
        background_hole_full=bg_hole_mask,
        background_hole_base_full=bg_hole_base_mask,
        component_norm_grids=all_results['grids'],
        component_masks_start=all_results['masks_start'],
        component_masks_end=all_results['masks_end'],
        component_handles=all_results['handles'],
        component_targets=all_results['targets'],
        hole_fill_mode=hole_fill_mode,
        use_drag_guided_prefill=use_drag_guided_prefill,
        enable_subject_hole_fill=enable_3d_subject_scope_fill,
        subject_hole_mask_lat=hole_inside_lat.astype(np.float32),
    )
    if return_plan:
        return final_latents, drag_plan_out
    return final_latents
