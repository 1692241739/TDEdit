# depth_estimator.py
# 深度估计模块 - 使用 Depth Anything V2

import os
import sys
import hashlib
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tdedit_paths import (
    DEPTH_V2_REPO, DEPTH_V2_CHECKPOINT_DIR, DEPTH_V1_REPO, DEPTH_V1_CHECKPOINT,
)

# 添加 Depth Anything V2 路径
DEPTH_ANYTHING_PATH = DEPTH_V2_REPO
DEPTH_CHECKPOINT_PATH = DEPTH_V2_CHECKPOINT_DIR
DEPTH_ANYTHING_V1_PATH = DEPTH_V1_REPO
DEPTH_ANYTHING_V1_CHECKPOINT = DEPTH_V1_CHECKPOINT

# ==========================================
# 全局深度模型管理
# ==========================================
_depth_model = None
_depth_device = None
_depth_backend = None


def _apply_registered_depth_perturbation(depth, image_np):
    """Apply an optional deterministic robustness perturbation.

    This path is disabled by default and is controlled only by the rebuttal
    experiment environment.  Perturbations act on the estimator output before
    the existing normalization, so the downstream geometry code is unchanged.
    """
    mode = os.environ.get("TDEDIT_DEPTH_PERTURBATION", "none").strip().lower()
    level = float(os.environ.get("TDEDIT_DEPTH_PERTURBATION_LEVEL", "0") or 0)
    depth = np.asarray(depth, dtype=np.float32)
    if mode in {"", "none", "off"}:
        return depth
    span = float(depth.max() - depth.min())
    if span <= 1e-12:
        return depth.copy()
    if mode == "gaussian":
        digest = hashlib.sha256(np.ascontiguousarray(image_np).tobytes()).digest()
        seed = int.from_bytes(digest[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        perturbed = depth + rng.normal(0.0, max(0.0, level) * span, size=depth.shape).astype(np.float32)
    elif mode == "smooth":
        sigma = max(0.0, level)
        perturbed = cv2.GaussianBlur(depth, (0, 0), sigmaX=sigma, sigmaY=sigma) if sigma > 0 else depth.copy()
    elif mode == "invert":
        # Controlled relative-order reversal: near/far ordering is globally
        # inverted while preserving the estimator's value range.
        perturbed = float(depth.min() + depth.max()) - depth
    else:
        raise ValueError(f"Unsupported TDEDIT_DEPTH_PERTURBATION={mode!r}")
    print(f"[DepthEstimator] registered perturbation mode={mode} level={level}")
    return np.asarray(perturbed, dtype=np.float32)

def get_depth_model(encoder='vitb', device=None):
    """
    获取或初始化深度估计模型（单例模式）

    参数:
        encoder: 模型规模 'vits', 'vitb', 'vitl'
        device: 计算设备

    返回:
        DepthAnythingV2 模型实例
    """
    global _depth_model, _depth_device, _depth_backend

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backend = os.environ.get("TDEDIT_DEPTH_BACKEND", "v2").strip().lower()
    if backend not in {"v1", "v2"}:
        raise ValueError(f"Unsupported TDEDIT_DEPTH_BACKEND={backend!r}; expected v1 or v2")

    if _depth_model is None or _depth_device != device or _depth_backend != backend:
        print(f"[DepthEstimator] Loading Depth Anything {backend.upper()} ({encoder})...")

        # 模型配置
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        config = model_configs.get(encoder, model_configs['vitb'])
        if backend == "v2":
            if DEPTH_ANYTHING_PATH not in sys.path:
                sys.path.insert(0, DEPTH_ANYTHING_PATH)
            try:
                from depth_anything_v2.dpt import DepthAnythingV2
            except ImportError as exc:
                raise ImportError(
                    "3D editing requires Depth Anything V2. Set TDEDIT_DEPTH_V2_REPO "
                    "to its checkout and TDEDIT_DEPTH_V2_CHECKPOINT_DIR to its weights."
                ) from exc
            _depth_model = DepthAnythingV2(**config)
            checkpoint_path = os.path.join(DEPTH_CHECKPOINT_PATH, f"depth_anything_v2_{encoder}.pth")
        else:
            if DEPTH_ANYTHING_V1_PATH not in sys.path:
                sys.path.insert(0, DEPTH_ANYTHING_V1_PATH)
            try:
                from depth_anything.dpt import DepthAnything
            except ImportError as exc:
                raise ImportError(
                    "The optional V1 depth backend requires Depth Anything V1. "
                    "Set TDEDIT_DEPTH_V1_REPO and TDEDIT_DEPTH_V1_CHECKPOINT."
                ) from exc

            # The official V1 implementation resolves its vendored DINOv2 hub
            # relative to the repository root.
            previous_cwd = os.getcwd()
            try:
                os.chdir(DEPTH_ANYTHING_V1_PATH)
                _depth_model = DepthAnything({**config, "localhub": True})
            finally:
                os.chdir(previous_cwd)
            checkpoint_path = DEPTH_ANYTHING_V1_CHECKPOINT

        # 加载权重
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Depth model checkpoint not found: {checkpoint_path}")

        _depth_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        _depth_model = _depth_model.to(device).eval()
        _depth_device = device
        _depth_backend = backend

        print(f"[DepthEstimator] Model loaded successfully on {device}")

    return _depth_model


def estimate_depth(image_np, device=None, input_size=518):
    """
    估计图像的深度图

    参数:
        image_np: numpy 数组，RGB 格式，形状 (H, W, 3)，值范围 [0, 255]
        device: 计算设备
        input_size: 输入图像大小

    返回:
        depth_map: numpy 数组，形状 (H, W)，深度值（越大越远）
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_depth_model(encoder='vitb', device=device)

    # 确保输入格式正确
    if image_np.dtype != np.uint8:
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)

    backend = os.environ.get("TDEDIT_DEPTH_BACKEND", "v2").strip().lower()
    if backend == "v2":
        # V2 official infer_image receives BGR input.
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        with torch.no_grad():
            depth = model.infer_image(image_bgr, input_size=input_size)
    else:
        from torchvision.transforms import Compose
        from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

        transform = Compose([
            Resize(
                width=input_size, height=input_size, resize_target=False,
                keep_aspect_ratio=True, ensure_multiple_of=14,
                resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        height, width = image_np.shape[:2]
        image = transform({'image': image_np.astype(np.float32) / 255.0})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(device)
        with torch.no_grad():
            depth_ts = model(image)
            depth_ts = F.interpolate(
                depth_ts[None], (height, width), mode='bilinear', align_corners=False
            )[0, 0]
        depth = depth_ts.float().cpu().numpy()

    return _apply_registered_depth_perturbation(depth, image_np)


def normalize_depth(depth_map, z_min=0, z_max=None, reverse=False):
    """
    将深度图归一化到指定范围

    参数:
        depth_map: numpy 数组，原始深度图
        z_min: 目标最小值
        z_max: 目标最大值（None 则使用图像最大维度）
        reverse: 是否反转深度（True 表示深度越大越近）

    返回:
        归一化后的深度图
    """
    height, width = depth_map.shape
    if z_max is None:
        z_max = max(height, width) - 1

    min_depth = depth_map.min()
    max_depth = depth_map.max()

    if max_depth == min_depth:
        return np.full_like(depth_map, z_min, dtype=np.float32)

    if reverse:
        normalized = z_max - (depth_map - min_depth) / (max_depth - min_depth) * (z_max - z_min)
    else:
        normalized = z_min + (depth_map - min_depth) / (max_depth - min_depth) * (z_max - z_min)

    return normalized.astype(np.float32)


def visualize_depth(depth_map, colormap='Spectral_r', save_path=None):
    """
    可视化深度图

    参数:
        depth_map: numpy 数组，深度图
        colormap: matplotlib 颜色映射名称
        save_path: 保存路径（可选）

    返回:
        彩色深度图 (H, W, 3) BGR 格式
    """
    import matplotlib

    # 归一化到 [0, 255]
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6) * 255.0
    depth_normalized = depth_normalized.astype(np.uint8)

    # 应用颜色映射
    cmap = matplotlib.colormaps.get_cmap(colormap)
    depth_colored = (cmap(depth_normalized)[:, :, :3] * 255).astype(np.uint8)
    depth_colored_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)

    if save_path:
        cv2.imwrite(save_path, depth_colored_bgr)
        print(f"[DepthEstimator] Depth visualization saved to {save_path}")

    return depth_colored_bgr


def get_depth_at_points(depth_map, points_yx):
    """
    获取指定点位置的深度值

    参数:
        depth_map: numpy 数组，深度图 (H, W)
        points_yx: numpy 数组，点坐标 [[y1, x1], [y2, x2], ...]

    返回:
        depths: numpy 数组，各点的深度值
    """
    height, width = depth_map.shape
    points_yx = np.asarray(points_yx)

    if points_yx.ndim == 1:
        points_yx = points_yx.reshape(1, 2)

    depths = np.zeros(len(points_yx), dtype=np.float32)

    for i, (y, x) in enumerate(points_yx):
        y_idx = int(round(y))
        x_idx = int(round(x))
        if 0 <= y_idx < height and 0 <= x_idx < width:
            depths[i] = depth_map[y_idx, x_idx]

    return depths
