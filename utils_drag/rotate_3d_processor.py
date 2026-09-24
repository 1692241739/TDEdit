# rotate_3d_processor.py
# 3D旋转处理器 - 基于深度图对连通域进行真实3D旋转
#
# 核心逻辑：
# - 水平拖拽（左右）→ 绕Y轴旋转（物体左右转头）
# - 垂直拖拽（上下）→ 绕X轴旋转（物体抬头低头）
# - 利用深度图构建3D点云，执行真正的3D旋转后投影回2D

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from scipy.ndimage import map_coordinates

from utils_drag.depth_estimator import estimate_depth, normalize_depth, visualize_depth, get_depth_at_points

# ==========================================
# 全局配置
# ==========================================
ROTATE_3D_DEBUG_ROOT = "debug_files/drag_process/3D-Rigid"
# 关闭后可显著加速 3D 拖拽（不会影响最终结果，只是不再保存调试可视化）
ENABLE_3D_DEBUG = True

# ==========================================
# 辅助函数
# ==========================================

def ensure_debug_dir():
    """确保调试目录存在"""
    if not ENABLE_3D_DEBUG:
        return None
    os.makedirs(ROTATE_3D_DEBUG_ROOT, exist_ok=True)
    return ROTATE_3D_DEBUG_ROOT


def _format_axis_value(v):
    v = float(v)
    if abs(v) >= 100 or abs(v - round(v)) < 0.05:
        return str(int(round(v)))
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _is_front_view(view_name):
    return str(view_name or "").strip().lower() == "front"


def _get_front_plane_limits(image_hw):
    H = int(image_hw[0])
    W = int(image_hw[1])
    return (-0.5, float(W) - 0.5), (float(H) - 0.5, -0.5)


FRONT_BG_RGBA = (0.92, 0.92, 0.92, 1.0)


def _compute_front_framed_limits(image_hw, point_sets, padding_ratio=0.10, min_pad_px=18.0):
    valid_x = []
    valid_y = []
    for item in point_sets:
        if item is None:
            continue
        px, py = item
        arr_x = np.asarray(px, dtype=np.float64).reshape(-1)
        arr_y = np.asarray(py, dtype=np.float64).reshape(-1)
        if arr_x.size == 0 or arr_y.size == 0 or arr_x.size != arr_y.size:
            continue
        mask = np.isfinite(arr_x) & np.isfinite(arr_y)
        if not np.any(mask):
            continue
        valid_x.append(arr_x[mask])
        valid_y.append(arr_y[mask])

    if not valid_x:
        return _get_front_plane_limits(image_hw)

    all_x = np.concatenate(valid_x, axis=0)
    all_y = np.concatenate(valid_y, axis=0)
    if all_x.size >= 64:
        x_min, x_max = np.percentile(all_x, [0.5, 99.5])
        y_min, y_max = np.percentile(all_y, [0.5, 99.5])
    else:
        x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
        y_min, y_max = float(np.min(all_y)), float(np.max(all_y))

    x_span = max(float(x_max - x_min), 1.0)
    y_span = max(float(y_max - y_min), 1.0)
    pad = max(max(x_span, y_span) * float(padding_ratio), float(min_pad_px))
    frame_size = max(x_span, y_span) + 2.0 * pad

    cx = 0.5 * (float(x_min) + float(x_max))
    cy = 0.5 * (float(y_min) + float(y_max))
    half = 0.5 * frame_size

    x_limits = (cx - half, cx + half)
    y_limits = (cy + half, cy - half)
    return x_limits, y_limits


def _setup_front_2d_axis(
    ax,
    x_limits,
    y_limits,
    title_text="",
    title_fontsize=18,
    tick_label_fontsize=11,
    title_pad=8,
    show_title=True,
):
    try:
        ax.set_xlim(list(x_limits))
        ax.set_ylim(list(y_limits))
        ax.set_aspect('equal', adjustable='box')
        ax.set_facecolor(FRONT_BG_RGBA)
        ax.margins(x=0.0, y=0.0)
        for side in ("left", "right", "top", "bottom"):
            ax.spines[side].set_visible(False)
            ax.spines[side].set_linewidth(0.0)
            ax.spines[side].set_color((0.0, 0.0, 0.0, 0.0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis='both', which='both', length=0, width=0)
        ax.set_axisbelow(True)
        if show_title:
            ax.set_title(title_text or "", fontsize=title_fontsize, pad=title_pad)
        else:
            ax.set_title("")
        return True
    except Exception as exc:
        print(f"[3DRotate] front 2d axes failed: {exc}")
        return False


def _draw_front_2d_background(ax, x_limits, y_limits):
    x0 = float(min(x_limits))
    x1 = float(max(x_limits))
    y0 = float(min(y_limits))
    y1 = float(max(y_limits))
    bg_rgba = np.empty((2, 2, 4), dtype=np.float32)
    bg_rgba[..., 0] = FRONT_BG_RGBA[0]
    bg_rgba[..., 1] = FRONT_BG_RGBA[1]
    bg_rgba[..., 2] = FRONT_BG_RGBA[2]
    bg_rgba[..., 3] = FRONT_BG_RGBA[3]
    ax.imshow(
        bg_rgba,
        extent=[x0, x1, y1, y0],
        interpolation='nearest',
        zorder=-2,
        aspect='auto',
    )


def _apply_standard_3d_axes(
    ax,
    title_text="",
    title_fontsize=18,
    tick_label_fontsize=12,
    title_pad=8,
    show_title=True,
):
    try:
        try:
            ax.set_proj_type('ortho')
        except Exception:
            pass
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        try:
            ax.patch.set_alpha(0.0)
        except Exception:
            pass
        for axis in (getattr(ax, "xaxis", None), getattr(ax, "yaxis", None), getattr(ax, "zaxis", None)):
            if axis is None:
                continue
            try:
                axis.pane.fill = True
                axis.pane.set_alpha(0.08)
            except Exception:
                pass
            try:
                axis.line.set_color((0.22, 0.22, 0.22, 0.58))
            except Exception:
                pass
            try:
                axis._axinfo["grid"]["linewidth"] = 0.7
                axis._axinfo["grid"]["color"] = (0.0, 0.0, 0.0, 0.12)
                axis._axinfo["tick"]["inward_factor"] = 0.20
                axis._axinfo["tick"]["outward_factor"] = 0.0
            except Exception:
                pass

        if show_title:
            ax.set_title(title_text or "", fontsize=title_fontsize, pad=title_pad)
        else:
            ax.set_title("")
        ax.tick_params(axis='x', labelsize=tick_label_fontsize, pad=-8)
        ax.tick_params(axis='y', labelsize=tick_label_fontsize, pad=-8)
        ax.tick_params(axis='z', labelsize=tick_label_fontsize, pad=-8)
    except Exception as exc:
        print(f"[3DRotate] standard 3d axes failed: {exc}")


def _apply_view_geometry(ax, view_name, image_hw=None, image_center_xy=None):
    try:
        ax.set_proj_type('ortho')
    except Exception:
        pass
    _set_axes_equal(ax)


def _paper_no_text_single_enabled():
    v = str(os.environ.get("TDEDIT_PAPER_NO_TEXT_SINGLE", "")).strip().lower()
    return v in {"1", "true", "yes", "on"}


def put_text_with_outline(img, text, pos, scale=0.6, color=(255, 255, 255), thickness=1):
    """带描边的文字"""
    if _paper_no_text_single_enabled():
        return
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def get_component_centroid(mask):
    """获取连通域的质心"""
    M = cv2.moments(mask.astype(np.uint8))
    if M["m00"] == 0:
        h, w = mask.shape
        return w // 2, h // 2
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy


def filter_background_by_depth(mask, depth_map, handle_points_yx, distance_threshold=None, save_dir=None, source_image_np=None):
    """
    【3D空间连通性过滤】基于3D点云的空间连通性过滤背景

    核心思想：
    1. 将mask中的每个像素投影到3D空间 (x, y, depth)
    2. 主体（如狗头）的点在3D空间中紧密相连
    3. 背景点虽然在2D上可能与主体相邻，但在3D空间中因深度差异而距离较远
    4. 从控制点出发，用3D距离做连通性扩散，只保留能连通到控制点的点

    参数:
        mask: SAM分割的mask (H, W)，0-255
        depth_map: 深度图 (H, W)
        handle_points_yx: 控制点列表 [[y, x], ...]
        distance_threshold: 3D空间中判定为"相邻"的距离阈值
        save_dir: 保存对比图的目录
        source_image_np: 源图像 (H, W, 3)

    返回:
        filtered_mask: 过滤后的mask
        filter_info: 过滤信息字典
    """
    H, W = mask.shape
    mask_binary = (mask > 127).astype(np.uint8)
    ys, xs = np.where(mask_binary > 0)

    if len(xs) == 0 or len(handle_points_yx) == 0:
        return mask.copy(), {'filtered': False, 'reason': 'empty_mask_or_no_handle'}

    # 1. 构建3D点云
    depth_values = depth_map[ys, xs]

    # 归一化深度到与xy相似的尺度
    depth_min = depth_values.min()
    depth_max = depth_values.max()
    depth_range = depth_max - depth_min + 1e-6

    # 计算xy范围用于归一化深度
    xy_range = max(xs.max() - xs.min(), ys.max() - ys.min())
    depth_scale = xy_range / depth_range if depth_range > 1e-6 else 1.0

    # 3D坐标: (x, y, scaled_depth)
    zs_scaled = (depth_values - depth_min) * depth_scale

    threshold_mode = "fixed"
    if distance_threshold is None:
        threshold_mode = "adaptive"
        # 在mask内估计相邻像素的深度跳变，动态设置3D连通阈值。
        mask_h = (mask_binary[:, 1:] > 0) & (mask_binary[:, :-1] > 0)
        mask_v = (mask_binary[1:, :] > 0) & (mask_binary[:-1, :] > 0)
        dz_h = np.abs(depth_map[:, 1:] - depth_map[:, :-1])[mask_h]
        dz_v = np.abs(depth_map[1:, :] - depth_map[:-1, :])[mask_v]
        if dz_h.size + dz_v.size > 0:
            dz_samples = np.concatenate([dz_h.reshape(-1), dz_v.reshape(-1)], axis=0)
            dz_robust = float(np.percentile(dz_samples, 75))
        else:
            dz_robust = float(depth_range / max(float(xy_range), 1.0))
        dz_scaled = dz_robust * depth_scale
        base_neighbor = np.sqrt(2.0 + dz_scaled * dz_scaled)
        distance_threshold = float(np.clip(2.2 * base_neighbor, 4.0, 35.0))
    else:
        distance_threshold = float(distance_threshold)

    print(f"[DepthFilter] Building 3D point cloud: {len(xs)} points")
    print(f"[DepthFilter] XY range: {xy_range:.1f}, Depth range: {depth_range:.3f}, Depth scale: {depth_scale:.1f}")
    print(f"[DepthFilter] distance_threshold={distance_threshold:.2f} ({threshold_mode})")

    # 2. 获取控制点的3D坐标
    handle_3d_coords = []
    for h_pt in handle_points_yx:
        hy, hx = int(h_pt[0]), int(h_pt[1])
        hy = np.clip(hy, 0, H - 1)
        hx = np.clip(hx, 0, W - 1)
        h_depth = depth_map[hy, hx]
        h_z_scaled = (h_depth - depth_min) * depth_scale
        handle_3d_coords.append((hx, hy, h_z_scaled))

    print(f"[DepthFilter] Handle points 3D: {handle_3d_coords}")

    # 3. 构建像素索引映射 (用于快速查找邻居)
    # pixel_to_idx: (y, x) -> index in point cloud
    pixel_to_idx = {}
    for idx, (y, x) in enumerate(zip(ys, xs)):
        pixel_to_idx[(y, x)] = idx

    # 4. 从控制点出发，进行3D连通性扩散
    # 使用BFS，只扩散到3D距离小于阈值的邻居
    visited = np.zeros(len(xs), dtype=bool)
    queue = deque()

    # 初始化：找到控制点对应的像素索引
    for hx, hy, hz in handle_3d_coords:
        if (hy, hx) in pixel_to_idx:
            idx = pixel_to_idx[(hy, hx)]
            if not visited[idx]:
                visited[idx] = True
                queue.append(int(idx))
        else:
            # 控制点不在mask内，找最近的点
            dists = (xs - hx)**2 + (ys - hy)**2
            nearest_idx = np.argmin(dists)
            if not visited[nearest_idx]:
                visited[nearest_idx] = True
                queue.append(int(nearest_idx))

    print(f"[DepthFilter] Starting BFS from {len(queue)} seed points")

    # 8邻域偏移
    neighbors_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # BFS扩散
    bfs_iterations = 0
    while queue:
        current_idx = queue.popleft()
        current_x = xs[current_idx]
        current_y = ys[current_idx]
        current_z = zs_scaled[current_idx]

        # 检查8邻域
        for dy, dx in neighbors_offsets:
            ny, nx = current_y + dy, current_x + dx
            if (ny, nx) in pixel_to_idx:
                neighbor_idx = pixel_to_idx[(ny, nx)]
                if not visited[neighbor_idx]:
                    # 计算3D距离
                    neighbor_z = zs_scaled[neighbor_idx]
                    dist_3d = np.sqrt(dx**2 + dy**2 + (current_z - neighbor_z)**2)

                    if dist_3d <= distance_threshold:
                        visited[neighbor_idx] = True
                        queue.append(int(neighbor_idx))

        bfs_iterations += 1

    num_connected = visited.sum()
    num_removed = len(xs) - num_connected

    print(f"[DepthFilter] BFS iterations: {bfs_iterations}")
    print(f"[DepthFilter] Connected to handles: {num_connected}/{len(xs)} pixels")
    print(f"[DepthFilter] Removed (not connected): {num_removed} pixels ({num_removed/len(xs)*100:.1f}%)")

    # 5. 更新mask
    filtered_mask = np.zeros_like(mask)
    connected_ys = ys[visited]
    connected_xs = xs[visited]
    filtered_mask[connected_ys, connected_xs] = 255

    # 6. 形态学后处理：填补小空洞
    kernel = np.ones((3, 3), np.uint8)
    filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)

    filter_info = {
        'filtered': True,
        'threshold_mode': threshold_mode,
        'distance_threshold': distance_threshold,
        'num_connected': int(num_connected),
        'num_removed': int(num_removed),
        'num_total': len(xs),
        'removal_ratio': num_removed / len(xs) if len(xs) > 0 else 0,
    }

    # 7. 保存3D点云对比图
    if save_dir is not None:
        vis_depth_filter_3d_comparison(mask, filtered_mask, depth_map, handle_points_yx,
                                        filter_info, save_dir, source_image_np)

    return filtered_mask, filter_info


def vis_depth_filter_3d_comparison(orig_mask, filtered_mask, depth_map, handle_points_yx,
                                    filter_info, save_dir, source_image_np=None,
                                    save_single_views=True):
    """
    【depth_00】3D点云过滤对比图（深度着色：近红远蓝，2x4布局）

    布局：
    - 第一行：过滤前 - 4个机位（正面、侧面、左前、右前）
    - 第二行：过滤后 - 4个机位
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.mplot3d import Axes3D

    H, W = orig_mask.shape

    # 获取原始mask的点
    orig_ys, orig_xs = np.where(orig_mask > 127)
    if len(orig_xs) == 0:
        return

    # 获取过滤后mask的点
    filtered_ys, filtered_xs = np.where(filtered_mask > 127)

    # 计算质心
    cx, cy = get_component_centroid(filtered_mask if len(filtered_xs) > 0 else orig_mask)

    # 获取深度值
    orig_depths = depth_map[orig_ys, orig_xs]
    depth_min = orig_depths.min()
    depth_max = orig_depths.max()
    depth_range = depth_max - depth_min + 1e-6

    # 计算XY范围
    x_min, x_max = orig_xs.min() - cx, orig_xs.max() - cx
    y_min, y_max = -(orig_ys.max() - cy), -(orig_ys.min() - cy)
    xy_range = max(x_max - x_min, y_max - y_min)

    # 动态计算z_scale
    depth_std = np.std(orig_depths)
    depth_mean = np.mean(orig_depths)
    depth_cv = depth_std / depth_mean if depth_mean > 1e-6 else 0

    if depth_cv < 0.05:
        z_scale_factor = 0.4
    elif depth_cv < 0.10:
        z_scale_factor = 0.7
    elif depth_cv < 0.20:
        z_scale_factor = 1.0
    else:
        z_scale_factor = 1.5

    z_scale = xy_range * z_scale_factor
    z_centroid = z_scale / 2.0

    # 构建3D点云
    orig_xs_3d = orig_xs.astype(np.float64) - cx
    orig_ys_3d = -(orig_ys.astype(np.float64) - cy)
    orig_zs_3d = (orig_depths - depth_min) / depth_range * z_scale - z_centroid

    # 判断哪些点被移除
    removed_mask_arr = (orig_mask > 127) & (filtered_mask < 127)
    is_removed = removed_mask_arr[orig_ys, orig_xs]

    # 【修改】使用深度colormap着色（与深度图一致，Spectral_r）
    import matplotlib
    depth_normalized = (orig_depths - depth_min) / depth_range
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    orig_colors = cmap(depth_normalized)[:, :3]  # 取RGB，去掉alpha

    # 坐标转换用于matplotlib显示
    plot_x = orig_xs_3d
    plot_y = orig_zs_3d
    plot_z = orig_ys_3d

    # 采样以提高性能
    n_points = len(orig_xs_3d)
    max_points = 15000
    if n_points > max_points:
        indices = np.random.choice(n_points, max_points, replace=False)
        plot_x = plot_x[indices]
        plot_y = plot_y[indices]
        plot_z = plot_z[indices]
        front_x_sampled = orig_xs[indices]
        front_y_sampled = orig_ys[indices]
        is_removed_sampled = is_removed[indices]
        colors_sampled = orig_colors[indices]
    else:
        front_x_sampled = orig_xs
        front_y_sampled = orig_ys
        is_removed_sampled = is_removed
        colors_sampled = orig_colors

    # 过滤前的颜色：保留点用深度着色，移除点用灰色标记
    colors_before = colors_sampled.copy()
    colors_before[is_removed_sampled] = [0.5, 0.5, 0.5]  # 灰色标记被移除的点

    # 保留点的数据
    kept_mask = ~is_removed_sampled
    kept_x = plot_x[kept_mask]
    kept_y = plot_y[kept_mask]
    kept_z = plot_z[kept_mask]
    kept_colors = colors_sampled[kept_mask]

    # 获取控制点的3D坐标
    handle_3d_list = []
    for h_pt in handle_points_yx:
        hy, hx = int(h_pt[0]), int(h_pt[1])
        hy = np.clip(hy, 0, H - 1)
        hx = np.clip(hx, 0, W - 1)
        h_depth = depth_map[hy, hx]
        h_x_3d = hx - cx
        h_y_3d = -(hy - cy)
        h_z_3d = (h_depth - depth_min) / depth_range * z_scale - z_centroid
        handle_3d_list.append([h_x_3d, h_z_3d, h_y_3d])

    # 统计信息
    removed_count = is_removed_sampled.sum()
    total_count = len(is_removed_sampled)
    kept_count = total_count - removed_count
    handle_front_x = np.array([int(np.clip(h_pt[1], 0, W - 1)) for h_pt in handle_points_yx], dtype=np.float64)
    handle_front_y = np.array([int(np.clip(h_pt[0], 0, H - 1)) for h_pt in handle_points_yx], dtype=np.float64)
    front_x_limits, front_y_limits = _compute_front_framed_limits(
        (H, W),
        point_sets=[
            (orig_xs, orig_ys),
            (filtered_xs, filtered_ys),
            (handle_front_x, handle_front_y),
        ],
        padding_ratio=0.10,
        min_pad_px=18.0,
    )

    def draw_handle_points(ax):
        """绘制控制点"""
        for idx, h_3d in enumerate(handle_3d_list):
            ax.scatter([h_3d[0]], [h_3d[1]], [h_3d[2]], c='red', s=800, marker='^',
                       edgecolors='black', linewidths=3,
                       label='Handle Point' if idx == 0 else None, zorder=10)

    def draw_handle_points_front(ax):
        for idx, (hx, hy) in enumerate(zip(handle_front_x, handle_front_y)):
            ax.scatter([hx], [hy], c='red', s=80, marker='^',
                       edgecolors='black', linewidths=1.4,
                       label='Handle Point' if idx == 0 else None, zorder=10)

    # 创建2x4的子图：第一行过滤前，第二行过滤后
    # 4个机位：正面、侧面、左前、右前
    fig = plt.figure(figsize=(32, 16))

    # 视角定义：
    # 正面: elev=0, azim=-90 (从正前方看)
    # 侧面: elev=0, azim=0 (从右侧看)
    # 左前: elev=20, azim=-120 (从左前方看)
    # 右前: elev=20, azim=-60 (从右前方看)
    views = [
        ('Front', 0, -90),
        ('Side', 0, 0),
        ('Left-Front', 20, 45),
        ('Right-Front', 20, 135),
    ]

    TITLE_FONTSIZE_FILTER = 24
    show_title_text = False

    def draw_single_view(ax, row_mode, view_idx, view_name, elev, azim):
        if _is_front_view(view_name):
            x_limits, y_limits = front_x_limits, front_y_limits
            axis_ready = _setup_front_2d_axis(
                ax,
                x_limits=x_limits,
                y_limits=y_limits,
                title_text="",
                title_fontsize=TITLE_FONTSIZE_FILTER,
                tick_label_fontsize=8,
                title_pad=8,
                show_title=False,
            )
            if axis_ready:
                _draw_front_2d_background(ax, x_limits, y_limits)
            if row_mode == "before":
                ax.scatter(front_x_sampled, front_y_sampled, c=colors_before, s=2, alpha=0.7)
                title_text = f'Before Filter - {view_name}\n(Gray = {removed_count} to remove)'
            else:
                if len(kept_x) > 0:
                    kept_front_x = front_x_sampled[kept_mask]
                    kept_front_y = front_y_sampled[kept_mask]
                    ax.scatter(kept_front_x, kept_front_y, c=kept_colors, s=2, alpha=0.7)
                title_text = f'After Filter - {view_name}\n({kept_count} kept)'
            draw_handle_points_front(ax)
            return

        if row_mode == "before":
            ax.scatter(plot_x, plot_y, plot_z, c=colors_before, s=2, alpha=0.7)
            title_text = f'Before Filter - {view_name}\n(Gray = {removed_count} to remove)'
        else:
            if len(kept_x) > 0:
                ax.scatter(kept_x, kept_y, kept_z, c=kept_colors, s=2, alpha=0.7)
            title_text = f'After Filter - {view_name}\n({kept_count} kept)'
        draw_handle_points(ax)
        ax.view_init(elev=elev, azim=azim)
        _apply_view_geometry(
            ax,
            view_name=view_name,
            image_hw=(H, W),
            image_center_xy=(cx, cy),
        )
        if view_idx >= 2:
            ax.invert_xaxis()
        _apply_standard_3d_axes(
            ax,
            title_text=title_text if show_title_text else "",
            title_fontsize=TITLE_FONTSIZE_FILTER,
            tick_label_fontsize=14,
            title_pad=12,
            show_title=show_title_text,
        )

    # 第一行：过滤前（灰色=待移除）
    for i, (view_name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 4, i + 1) if _is_front_view(view_name) else fig.add_subplot(2, 4, i + 1, projection='3d')
        draw_single_view(
            ax=ax,
            row_mode="before",
            view_idx=i,
            view_name=view_name,
            elev=elev,
            azim=azim,
        )

    # 第二行：过滤后（只显示保留的点）
    for i, (view_name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 4, i + 5) if _is_front_view(view_name) else fig.add_subplot(2, 4, i + 5, projection='3d')
        draw_single_view(
            ax=ax,
            row_mode="after",
            view_idx=i,
            view_name=view_name,
            elev=elev,
            azim=azim,
        )

    fig.patch.set_alpha(0.0)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.04, hspace=0.04)
    save_path = os.path.join(save_dir, "depth_00_filter_3d_comparison.png")
    plt.savefig(save_path, dpi=150, transparent=True)
    plt.close()
    print(f"[3DRotate] Saved: {save_path}")

    if save_single_views:
        base_name = "depth_00_filter_3d_comparison"
        row_entries = [("before", "before"), ("after", "after")]
        for row_mode, pose_tag in row_entries:
            for i, (view_name, elev, azim) in enumerate(views):
                fig_single = plt.figure(figsize=(5.12, 5.12), dpi=100)
                fig_single.patch.set_alpha(0.0)
                ax_single = fig_single.add_subplot(1, 1, 1) if _is_front_view(view_name) else fig_single.add_subplot(1, 1, 1, projection='3d')
                draw_single_view(
                    ax=ax_single,
                    row_mode=row_mode,
                    view_idx=i,
                    view_name=view_name,
                    elev=elev,
                    azim=azim,
                )
                safe_view_name = view_name.lower().replace(" ", "_").replace("-", "_")
                single_name = f"{base_name}__view_{pose_tag}_{i+1:02d}_{safe_view_name}.png"
                single_path = os.path.join(save_dir, single_name)
                ax_single.set_position([0.01, 0.01, 0.98, 0.98])
                plt.savefig(single_path, dpi=100, transparent=True, pad_inches=0.0)
                plt.close(fig_single)
        print("[3DRotate] Saved per-view files for: depth_00_filter_3d_comparison.png")


def create_depth_colormap(depth_map, colormap='Spectral_r'):
    """将深度图转换为彩色图像"""
    import matplotlib
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
    cmap = matplotlib.colormaps.get_cmap(colormap)
    depth_colored = (cmap(depth_normalized)[:, :, :3] * 255).astype(np.uint8)
    return depth_colored


# ==========================================
# 3D旋转核心函数
# ==========================================

def compute_rotation_axis_and_angle_3d(handle_pt_xy, target_pt_xy, centroid_xy,
                                        depth_map, z_scale, depth_min, depth_range,
                                        z_centroid):
    """
    【核心改进】将控制点和目标点投影到3D空间，计算真实的旋转轴和角度

    【修复】使用与 rotate_3d_with_depth 完全一致的坐标系：
    - X = pixel_x - center_x（直接像素偏移）
    - Y = -(pixel_y - center_y)（Y轴取反，向上为正）
    - Z = 归一化深度值 * z_scale - z_centroid

    流程:
    1. 将控制点(handle)投影到3D空间
    2. 将目标点(target)投影到3D空间（假设与控制点同深度）
    3. 计算质心到控制点的向量 V1
    4. 计算质心到目标点的向量 V2
    5. 根据拖拽方向确定旋转轴，计算两向量在对应平面上的夹角

    参数:
        handle_pt_xy: 控制点 [x, y]
        target_pt_xy: 目标点 [x, y]
        centroid_xy: 质心 [x, y]
        depth_map: 深度图 (H, W)
        z_scale: Z轴缩放因子（与rotate_3d_with_depth一致）
        depth_min: 深度最小值
        depth_range: 深度范围
        z_centroid: 质心的Z坐标

    返回:
        axis: 旋转轴 ('Y' 或 'X')
        angle_deg: 旋转角度（度）
        debug_info: 调试信息字典
    """
    H, W = depth_map.shape
    hx, hy = int(handle_pt_xy[0]), int(handle_pt_xy[1])
    tx, ty = int(target_pt_xy[0]), int(target_pt_xy[1])
    cx, cy = centroid_xy

    # 确保坐标在有效范围内
    hx = np.clip(hx, 0, W - 1)
    hy = np.clip(hy, 0, H - 1)
    tx = np.clip(tx, 0, W - 1)
    ty = np.clip(ty, 0, H - 1)

    # 1. 获取控制点的深度并转换到3D
    # 【修复】使用与 rotate_3d_with_depth 完全一致的坐标系
    depth_handle = depth_map[hy, hx]
    z_handle = (depth_handle - depth_min) / depth_range * z_scale

    # 控制点的3D坐标（相对于质心）
    # 【修复】直接使用像素偏移，不使用透视投影
    handle_3d_x = float(hx - cx)
    handle_3d_y = -float(hy - cy)  # Y轴取反，向上为正
    handle_3d_z = z_handle - z_centroid

    # 2. 目标点的3D坐标
    # 假设目标点与控制点在同一深度平面
    # 【修复】直接使用像素偏移
    target_3d_x = float(tx - cx)
    target_3d_y = -float(ty - cy)  # Y轴取反
    target_3d_z = z_handle - z_centroid  # 同深度

    # 3. 构建向量
    vec_handle = np.array([handle_3d_x, handle_3d_y, handle_3d_z])
    vec_target = np.array([target_3d_x, target_3d_y, target_3d_z])

    def _safe_unit(v, eps=1e-8):
        n = float(np.linalg.norm(v))
        if n < eps:
            return None
        return v / n

    a = _safe_unit(vec_handle)
    b = _safe_unit(vec_target)
    solver = "vec_to_vec_min_rotation"
    dot_val = 1.0
    if a is None or b is None:
        R = np.eye(3, dtype=np.float64)
        solver = "identity_zero_vector"
    else:
        dot_val = float(np.clip(np.dot(a, b), -1.0, 1.0))
        if dot_val > 1.0 - 1e-8:
            R = np.eye(3, dtype=np.float64)
            solver = "already_aligned"
        elif dot_val < -1.0 + 1e-8:
            basis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(np.dot(a, basis))) > 0.9:
                basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            axis_u = _safe_unit(np.cross(a, basis))
            if axis_u is None:
                axis_u = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            R = -np.eye(3, dtype=np.float64) + 2.0 * np.outer(axis_u, axis_u)
            solver = "opposite_direction_180deg"
        else:
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
            R = np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - dot_val) / (s * s + 1e-8))

    # 仅由 3D 几何旋转矩阵分解 yaw/pitch，不掺像素启发项。
    yaw_deg = float(np.degrees(np.arctan2(R[0, 2], R[0, 0])))
    pitch_deg = float(np.degrees(np.arcsin(np.clip(R[2, 1], -1.0, 1.0))))
    yaw_deg = float(np.clip(yaw_deg, -60.0, 60.0))
    pitch_deg = float(np.clip(pitch_deg, -45.0, 45.0))

    if abs(yaw_deg) >= abs(pitch_deg):
        axis = 'Y'
        angle_deg = yaw_deg
    else:
        axis = 'X'
        angle_deg = pitch_deg

    debug_info = {
        'handle_2d': (hx, hy),
        'target_2d': (tx, ty),
        'handle_3d': vec_handle.tolist(),
        'target_3d': vec_target.tolist(),
        'z_handle': float(z_handle),
        'z_scale': float(z_scale),
        'rotation_solver': solver,
        'rotation_dot': float(dot_val),
        'yaw_deg': float(yaw_deg),
        'pitch_deg': float(pitch_deg),
        'axis': axis,
        'angle_deg': float(angle_deg),
    }

    print(f"[3DRotate] 3D Angle Computation:")
    print(f"  Handle 2D: ({hx}, {hy}) -> 3D: ({handle_3d_x:.2f}, {handle_3d_y:.2f}, {handle_3d_z:.2f})")
    print(f"  Target 2D: ({tx}, {ty}) -> 3D: ({target_3d_x:.2f}, {target_3d_y:.2f}, {target_3d_z:.2f})")
    print(f"  Solver: {solver}, dot={dot_val:.4f}")
    print(f"  Candidate yaw/pitch: yaw={yaw_deg:.2f}°, pitch={pitch_deg:.2f}°")
    print(f"  Rotation: axis={axis}, angle={angle_deg:.2f} deg")

    return axis, angle_deg, debug_info


def compute_rotation_axis_and_angle(handle_pt_xy, target_pt_xy, centroid_xy):
    """
    旧版本的简化接口（保持向后兼容）
    实际使用时应该调用 compute_rotation_axis_and_angle_3d
    """
    dx = target_pt_xy[0] - handle_pt_xy[0]
    dy = target_pt_xy[1] - handle_pt_xy[1]

    if abs(dx) > abs(dy):
        axis = 'Y'
        angle_deg = dx * 0.3
        angle_deg = np.clip(angle_deg, -60, 60)
    else:
        axis = 'X'
        angle_deg = -dy * 0.3
        angle_deg = np.clip(angle_deg, -45, 45)

    return axis, angle_deg


def create_rotation_matrix(axis, angle_deg):
    """
    创建3D旋转矩阵

    参数:
        axis: 'X', 'Y', 或 'Z'
        angle_deg: 旋转角度（度）

    返回:
        R: 3x3 旋转矩阵
    """
    angle_rad = np.radians(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    if axis == 'X':
        R = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ], dtype=np.float64)
    elif axis == 'Y':
        R = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ], dtype=np.float64)
    else:  # Z
        R = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ], dtype=np.float64)

    return R


def create_rotation_matrix_yaw_pitch(yaw_deg=0.0, pitch_deg=0.0):
    """
    创建联合旋转矩阵（先 Yaw 再 Pitch）。

    约定:
    - yaw_deg: 绕 Y 轴（左右转头）
    - pitch_deg: 绕 X 轴（抬头/低头）
    """
    R_yaw = create_rotation_matrix('Y', float(yaw_deg))
    R_pitch = create_rotation_matrix('X', float(pitch_deg))
    return (R_pitch @ R_yaw).astype(np.float64)


def resolve_rotation_matrix(rotation_axis='Y', rotation_angle_deg=0.0,
                           yaw_deg=None, pitch_deg=None):
    """
    统一解析旋转矩阵:
    - 若提供 yaw/pitch，则使用双轴联合旋转
    - 否则退回单轴旋转
    """
    if yaw_deg is None and pitch_deg is None:
        return create_rotation_matrix(rotation_axis, rotation_angle_deg)
    use_yaw = 0.0 if yaw_deg is None else float(yaw_deg)
    use_pitch = 0.0 if pitch_deg is None else float(pitch_deg)
    return create_rotation_matrix_yaw_pitch(use_yaw, use_pitch)


def project_points_after_3d_rotation(
    points_yx,
    depth_map,
    centroid_xy,
    rotation_axis,
    rotation_angle_deg,
    depth_min,
    depth_range,
    z_scale,
    z_centroid,
    yaw_deg=None,
    pitch_deg=None,
):
    """
    将图像点([y, x])按当前3D旋转参数投影到旋转后的2D位置([x, y])。
    用于保证后续2D非刚性阶段与3D旋转后的主体处于同一坐标系。
    """
    if points_yx is None or len(points_yx) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    H, W = depth_map.shape
    cx, cy = centroid_xy
    R = resolve_rotation_matrix(
        rotation_axis=rotation_axis,
        rotation_angle_deg=rotation_angle_deg,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg
    )
    out_xy = []

    for pt in points_yx:
        py = int(np.clip(pt[0], 0, H - 1))
        px = int(np.clip(pt[1], 0, W - 1))

        depth_val = depth_map[py, px]
        z_val = (depth_val - depth_min) / (depth_range + 1e-6) * z_scale - z_centroid

        p3d = np.array([float(px - cx), -float(py - cy), float(z_val)], dtype=np.float64)
        p3d_rot = R @ p3d

        new_x = p3d_rot[0] + cx
        new_y = -p3d_rot[1] + cy
        out_xy.append([float(new_x), float(new_y)])

    return np.asarray(out_xy, dtype=np.float32)


def rotate_3d_with_depth(source_image_np, mask, depth_map, centroid_xy,
                         rotation_axis, rotation_angle_deg, device,
                         save_3d_vis=True, save_dir=None,
                         handle_point_xy=None, target_point_xy=None,
                         yaw_deg=None, pitch_deg=None):
    """
    使用深度图进行真实的3D旋转

    【关键修复】：
    1. X = pixel_x - center_x (直接像素偏移)
    2. Y = -(pixel_y - center_y) (Y轴取反，符合标准直角坐标系)
    3. Z = 归一化深度值，**比例与X/Y一致**（这是关键！）

    参数:
        source_image_np: 源图像 (H, W, 3)
        mask: 主体mask (H, W)，值为0或255
        depth_map: 深度图 (H, W)
        centroid_xy: 质心坐标 [x, y]（旋转中心）
        rotation_axis: 旋转轴 'X' 或 'Y'
        rotation_angle_deg: 旋转角度（度）
        device: 计算设备
        save_3d_vis: 是否保存3D可视化
        save_dir: 保存目录
        handle_point_xy: 控制点 [x, y]
        target_point_xy: 目标点 [x, y]

    返回:
        rotated_rgb: 旋转后的RGB图像
        rotated_depth: 旋转后的深度图
        rotated_mask: 旋转后的mask
    """
    H, W = source_image_np.shape[:2]
    obj_cx, obj_cy = centroid_xy  # 物体质心（旋转中心）

    if save_dir is None or not ENABLE_3D_DEBUG:
        save_3d_vis = False
        save_dir = None

    # 1. 清理mask - 只做轻微的形态学清理，不删除连通域
    # 注意：深度过滤已经在 process_3d_rotate_component 中完成
    mask_binary = (mask > 127).astype(np.uint8)

    # 轻微的形态学清理：只去除非常小的噪点（3x3核）
    kernel_clean = np.ones((3, 3), np.uint8)
    # 开运算去除小的噪点
    mask_cleaned = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel_clean)
    # 闭运算填补小空洞
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel_clean)

    # 【已移除】不再只保留最大连通域，因为深度过滤已经处理了背景
    # 多连通域的处理应该在上层 drag_processor 中完成

    # 使用清理后的mask
    ys, xs = np.where(mask_cleaned > 0)

    if len(xs) == 0:
        return source_image_np.copy(), depth_map.copy(), mask.copy()

    # 2. 计算X/Y的实际范围，用于确定Z的合理比例
    x_min, x_max = xs.min() - obj_cx, xs.max() - obj_cx
    y_min, y_max = -(ys.max() - obj_cy), -(ys.min() - obj_cy)  # 注意Y轴取反
    xy_range = max(x_max - x_min, y_max - y_min)

    # 3. 归一化深度图 - 【关键修复】根据深度特征动态计算Z比例
    depth_values = depth_map[ys, xs]
    depth_min = depth_values.min()
    depth_max = depth_values.max()
    depth_range = depth_max - depth_min + 1e-6

    # 【修复】根据深度分布动态估算物体厚度
    # 计算多个深度特征指标
    depth_std = np.std(depth_values)
    depth_mean = np.mean(depth_values)
    depth_normalized_std = depth_std / depth_range if depth_range > 1e-6 else 0

    # 额外指标：深度范围相对于物体尺寸的比例
    # 使用 coefficient of variation (CV)
    depth_cv = depth_std / depth_mean if depth_mean > 1e-6 else 0

    # 使用更宽松的阈值，默认给予更大的深度
    # 基于 CV 值来判断物体厚度类型
    if depth_cv < 0.05:
        # 非常扁平的物体
        z_scale_factor = 0.4
    elif depth_cv < 0.10:
        # 较扁平物体
        z_scale_factor = 0.7
    elif depth_cv < 0.20:
        # 中等厚度（如人脸）
        z_scale_factor = 1.0
    else:
        # 较厚物体
        z_scale_factor = 1.5

    z_scale = xy_range * z_scale_factor
    zs = (depth_values - depth_min) / depth_range * z_scale

    # 【修复】使用深度范围的中点作为质心Z坐标，而不是中位数
    # 这样质心位于物体的3D几何中心，而不是表面
    # 例如：对于球体，质心在球心而不是球面
    median_depth = np.median(depth_values)  # 仍保留用于调试输出
    z_centroid = z_scale / 2.0  # 深度范围的中点

    print(f"[3DRotate] Depth stats: min={depth_min:.2f}, max={depth_max:.2f}, "
          f"median={median_depth:.2f}, mean={depth_mean:.2f}, std={depth_std:.2f}")
    print(f"[3DRotate] Depth CV (std/mean)={depth_cv:.3f}, normalized_std={depth_normalized_std:.3f}")
    print(f"[3DRotate] XY range={xy_range:.1f}, z_scale_factor={z_scale_factor:.2f}, z_scale={z_scale:.1f}")
    print(f"[3DRotate] Centroid 2D: pixel=({obj_cx}, {obj_cy})")
    print(f"[3DRotate] Centroid 3D: X=0.0, Y=0.0, Z=0.0 (geometric center)")
    print(f"[3DRotate] z_centroid={z_centroid:.2f} (midpoint of depth range), "
          f"Z range after centering=[{-z_centroid:.2f}, {z_scale - z_centroid:.2f}]")

    # 4. 构建3D点云
    # X = pixel_x - center_x
    # Y = -(pixel_y - center_y)  <- 关键：Y轴取反！
    # Z = 深度值 - 旋转中心深度
    xs_3d = xs.astype(np.float64) - obj_cx
    ys_3d = -(ys.astype(np.float64) - obj_cy)  # Y轴取反
    zs_3d = zs - z_centroid  # 相对于旋转中心

    # 构建点云矩阵 (N, 3)
    points_3d = np.stack([xs_3d, ys_3d, zs_3d], axis=1)

    print(f"[3DRotate] Point cloud: X range=[{xs_3d.min():.1f}, {xs_3d.max():.1f}], "
          f"Y range=[{ys_3d.min():.1f}, {ys_3d.max():.1f}], Z range=[{zs_3d.min():.1f}, {zs_3d.max():.1f}]")

    # 5. 应用旋转矩阵（支持 Yaw+Pitch 联合旋转）
    R = resolve_rotation_matrix(
        rotation_axis=rotation_axis,
        rotation_angle_deg=rotation_angle_deg,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg
    )
    rotated_points = (R @ points_3d.T).T  # (N, 3)

    # 6. 转换回图像坐标
    new_xs_3d = rotated_points[:, 0]
    new_ys_3d = rotated_points[:, 1]
    new_zs_3d = rotated_points[:, 2]

    # 逆变换回像素坐标：
    # pixel_x = X + center_x
    # pixel_y = -Y + center_y  <- Y轴再取反
    new_xs = new_xs_3d + obj_cx
    new_ys = -new_ys_3d + obj_cy  # Y轴取反恢复
    new_zs = new_zs_3d + z_centroid

    print(f"[3DRotate] Projection stats: X range=[{new_xs.min():.1f}, {new_xs.max():.1f}], "
          f"Y range=[{new_ys.min():.1f}, {new_ys.max():.1f}]")

    # 7. 保存3D点云可视化
    if save_3d_vis and save_dir is not None:
        colors_rgb = source_image_np[ys, xs] / 255.0

        # 质心的3D坐标（应该在原点附近，因为我们是相对于质心建立的坐标系）
        centroid_3d = [0.0, 0.0, 0.0]  # 质心就是原点

        # 计算控制点和目标点的3D坐标
        handle_3d = None
        target_3d = None

        if handle_point_xy is not None:
            hx, hy = int(handle_point_xy[0]), int(handle_point_xy[1])
            hx = np.clip(hx, 0, W - 1)
            hy = np.clip(hy, 0, H - 1)
            # 获取控制点的深度
            h_depth = depth_map[hy, hx]
            h_z = (h_depth - depth_min) / depth_range * z_scale - z_centroid
            # 转换到3D坐标
            h_x_3d = hx - obj_cx
            h_y_3d = -(hy - obj_cy)  # Y轴取反
            handle_3d = [h_x_3d, h_y_3d, h_z]
            print(f"[3DRotate] Handle point: 2D=({hx}, {hy}) -> 3D=({h_x_3d:.1f}, {h_y_3d:.1f}, {h_z:.1f})")

        if target_point_xy is not None:
            tx, ty = int(target_point_xy[0]), int(target_point_xy[1])
            tx = np.clip(tx, 0, W - 1)
            ty = np.clip(ty, 0, H - 1)
            # 目标点假设与控制点同深度
            if handle_3d is not None:
                t_z = handle_3d[2]  # 使用控制点的深度
            else:
                t_depth = depth_map[ty, tx]
                t_z = (t_depth - depth_min) / depth_range * z_scale - z_centroid
            # 转换到3D坐标
            t_x_3d = tx - obj_cx
            t_y_3d = -(ty - obj_cy)  # Y轴取反
            target_3d = [t_x_3d, t_y_3d, t_z]
            print(f"[3DRotate] Target point: 2D=({tx}, {ty}) -> 3D=({t_x_3d:.1f}, {t_y_3d:.1f}, {t_z:.1f})")

        save_3d_point_cloud_visualization(
            points_3d, rotated_points, colors_rgb,
            rotation_axis, rotation_angle_deg, save_dir,
            centroid_3d=centroid_3d,
            handle_3d=handle_3d,
            target_3d=target_3d,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg
        )

    # 8. 创建输出图像（使用Z-buffer处理遮挡）
    rotated_rgb = np.zeros_like(source_image_np)
    rotated_depth = np.zeros((H, W), dtype=np.float32)
    rotated_mask = np.zeros((H, W), dtype=np.uint8)
    z_buffer = np.full((H, W), -np.inf, dtype=np.float32)  # Z越大越近（朝向相机）

    # 获取原始像素颜色
    colors = source_image_np[ys, xs]

    # 按深度排序（从远到近渲染，近的覆盖远的）
    # 在我们的坐标系中，Z越大表示越靠近相机
    depth_order = np.argsort(new_zs)  # 从远到近

    for idx in depth_order:
        new_x = int(round(new_xs[idx]))
        new_y = int(round(new_ys[idx]))

        if 0 <= new_x < W and 0 <= new_y < H:
            if new_zs[idx] > z_buffer[new_y, new_x]:
                z_buffer[new_y, new_x] = new_zs[idx]
                rotated_rgb[new_y, new_x] = colors[idx]
                rotated_depth[new_y, new_x] = new_zs[idx]
                rotated_mask[new_y, new_x] = 255

    # 7. 主体范围补洞
    try:
        from utils_drag.drag_processor import fill_subject_holes_within_scope

        rotated_rgb, refined_scope, scope_info = fill_subject_holes_within_scope(
            image_input=rotated_rgb,
            subject_mask=rotated_mask,
            radius=5,
            close_ratio=0.10,
            max_close=35,
            max_fill_dist_ratio=0.18,
        )
        hole_n = int(scope_info.get("hole_pixels", 0))
        if hole_n > 0:
            rotated_mask = np.maximum(rotated_mask, (refined_scope * 255.0).astype(np.uint8))
            print(f"[3DRotate] Subject-scope holes filled: {hole_n} pixels")
    except Exception as e:
        # 失败时退化到原先的闭运算 + inpaint
        print(f"[3DRotate] Subject-scope hole fill fallback: {e}")
        kernel_close = np.ones((7, 7), np.uint8)
        rotated_mask_closed = cv2.morphologyEx(rotated_mask, cv2.MORPH_CLOSE, kernel_close)
        internal_holes = (rotated_mask_closed > 0) & (rotated_mask == 0)
        print(f"[3DRotate] Internal holes to fill: {internal_holes.sum()} pixels")
        if internal_holes.sum() > 0:
            hole_mask_u8 = internal_holes.astype(np.uint8) * 255
            rotated_rgb_bgr = cv2.cvtColor(rotated_rgb, cv2.COLOR_RGB2BGR)
            inpainted_bgr = cv2.inpaint(rotated_rgb_bgr, hole_mask_u8, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
            rotated_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
            rotated_mask[internal_holes] = 255

    # 对边缘进行轻微模糊以减少锯齿（可选，仅针对主体边缘）
    edge_kernel = np.ones((3, 3), np.uint8)
    mask_eroded = cv2.erode(rotated_mask, edge_kernel, iterations=1)
    edge_pixels = (rotated_mask > 0) & (mask_eroded == 0)

    if edge_pixels.sum() > 0:
        # 对边缘区域应用轻微高斯模糊
        blurred_rgb = cv2.GaussianBlur(rotated_rgb, (3, 3), 0.5)
        rotated_rgb[edge_pixels] = blurred_rgb[edge_pixels]

    return rotated_rgb, rotated_depth, rotated_mask


# ==========================================
# 3D点云可视化函数
# ==========================================

def save_3d_point_cloud_visualization(points_3d, rotated_points, colors_rgb,
                                       rotation_axis, rotation_angle_deg, save_dir,
                                       centroid_3d=None, handle_3d=None, target_3d=None,
                                       handle_rotated_3d=None,
                                       anchor_points_3d=None,
                                       anchor_points_rotated_3d=None,
                                       output_name="depth_05_3d_point_cloud.png",
                                       yaw_deg=None, pitch_deg=None,
                                       extra_original_points_3d=None,
                                       extra_rotated_points_3d=None,
                                       extra_colors_rgb=None,
                                       third_points_3d=None,
                                       handle_third_3d=None,
                                       target_third_3d=None,
                                       third_row_title="Projected",
                                       save_single_views=True,
                                       image_hw=None,
                                       image_center_xy=None):
    """
    使用matplotlib保存真正的3D点云可视化

    【修复】：
    1. 视角调整：XY平面作为正面（像看图片一样），Z轴是深度方向
    2. 在matplotlib中，我们把 X->X, Y->Z(垂直), Z(depth)->Y 来实现正确视角
    3. 添加质心、控制点、目标点的显示

    参数:
        points_3d: 原始3D点云 (N, 3) [X, Y, Z] 其中X是水平，Y是垂直（向上），Z是深度
        rotated_points: 旋转后的3D点云 (N, 3)
        colors_rgb: 点的颜色 (N, 3)，范围[0,1]
        rotation_axis: 旋转轴
        rotation_angle_deg: 旋转角度
        save_dir: 保存目录
        centroid_3d: 质心的3D坐标 [X, Y, Z]（旋转中心，应该在原点附近）
        handle_3d: 控制点的3D坐标 [X, Y, Z]
        target_3d: 目标点的3D坐标 [X, Y, Z]
        handle_rotated_3d: 可选，旋转后控制点3D坐标（优先用于第二行标注）
        anchor_points_3d: 可选，原始姿态锚点3D坐标 (K, 3)
        anchor_points_rotated_3d: 可选，旋转/变形后锚点3D坐标 (K, 3)
        extra_original_points_3d: 可选，原始姿态下额外补全点云 (M, 3)
        extra_rotated_points_3d: 可选，旋转/变形后额外补全点云 (K, 3)
        extra_colors_rgb: 可选，额外点云颜色 (M,3) 或 (K,3)，范围[0,1]
        third_points_3d: 可选，第三行点云 (N, 3)，用于 Hybrid 的最终阶段展示
        handle_third_3d: 可选，第三行控制点3D坐标（用于3D-Non-Rigid扩展可视化）
        target_third_3d: 可选，第三行目标点3D坐标
        third_row_title: 第三行标题前缀
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    os.makedirs(save_dir, exist_ok=True)
    paper_no_text_single = _paper_no_text_single_enabled()

    # 为了性能，只采样部分点
    points_3d = np.asarray(points_3d, dtype=np.float32).reshape(-1, 3)
    rotated_points = np.asarray(rotated_points, dtype=np.float32).reshape(-1, 3)
    third_points_full = None
    if third_points_3d is not None:
        third_points_full = np.asarray(third_points_3d, dtype=np.float32).reshape(-1, 3)

    n_points = len(points_3d)
    max_points = 8000
    if n_points > max_points:
        indices = np.random.choice(n_points, max_points, replace=False)
        pts_orig = points_3d[indices]
        pts_rot = rotated_points[indices]
        colors = colors_rgb[indices]
        pts_third = third_points_full[indices] if (
            third_points_full is not None and third_points_full.shape[0] == n_points
        ) else None
    else:
        pts_orig = points_3d
        pts_rot = rotated_points
        colors = colors_rgb
        pts_third = third_points_full if (
            third_points_full is not None and third_points_full.shape[0] == n_points
        ) else None

    def prepare_extra_points(extra_points, extra_colors, max_points_extra=10000):
        if extra_points is None:
            return None, None
        pts = np.asarray(extra_points, dtype=np.float32).reshape(-1, 3)
        if pts.shape[0] == 0:
            return None, None

        cols = None
        if extra_colors is not None:
            c = np.asarray(extra_colors, dtype=np.float32).reshape(-1, 3)
            if c.shape[0] == pts.shape[0]:
                cols = np.clip(c, 0.0, 1.0)

        if pts.shape[0] > max_points_extra:
            keep_idx = np.random.choice(pts.shape[0], max_points_extra, replace=False)
            pts = pts[keep_idx]
            if cols is not None:
                cols = cols[keep_idx]
        return pts, cols

    extra_orig_pts, extra_orig_cols = prepare_extra_points(
        extra_original_points_3d, extra_colors_rgb
    )
    extra_rot_pts, extra_rot_cols = prepare_extra_points(
        extra_rotated_points_3d, extra_colors_rgb
    )

    # 坐标转换：为了在matplotlib中正确显示
    # 我们的坐标系: X(右), Y(上), Z(深度，朝向屏幕内)
    # matplotlib默认: X(右), Y(前), Z(上)
    # 转换: plot_x = X, plot_y = Z(深度), plot_z = Y(上)
    def convert_coords(pts):
        # 坐标转换：X, Y, Z -> plot_x, plot_y, plot_z
        # 不在这里取反，第3、4张图的镜像由 ax.invert_xaxis() 处理
        if pts.ndim == 1:
            return pts[0], pts[2], pts[1]  # 单个点
        return pts[:, 0], pts[:, 2], pts[:, 1]  # 多个点

    orig_x, orig_y, orig_z = convert_coords(pts_orig)
    rot_x, rot_y, rot_z = convert_coords(pts_rot)
    if pts_third is not None and np.asarray(pts_third).reshape(-1, 3).shape[0] > 0:
        third_x, third_y, third_z = convert_coords(np.asarray(pts_third, dtype=np.float32).reshape(-1, 3))
    else:
        third_x = third_y = third_z = None
    if extra_orig_pts is not None:
        ex_orig_x, ex_orig_y, ex_orig_z = convert_coords(extra_orig_pts)
    else:
        ex_orig_x = ex_orig_y = ex_orig_z = None
    if extra_rot_pts is not None:
        ex_rot_x, ex_rot_y, ex_rot_z = convert_coords(extra_rot_pts)
    else:
        ex_rot_x = ex_rot_y = ex_rot_z = None

    if anchor_points_3d is not None:
        anc_orig = np.asarray(anchor_points_3d, dtype=np.float32).reshape(-1, 3)
        if anc_orig.shape[0] > 0:
            anc_orig_x, anc_orig_y, anc_orig_z = convert_coords(anc_orig)
        else:
            anc_orig_x = anc_orig_y = anc_orig_z = None
    else:
        anc_orig_x = anc_orig_y = anc_orig_z = None

    if anchor_points_rotated_3d is not None:
        anc_rot = np.asarray(anchor_points_rotated_3d, dtype=np.float32).reshape(-1, 3)
        if anc_rot.shape[0] > 0:
            anc_rot_x, anc_rot_y, anc_rot_z = convert_coords(anc_rot)
        else:
            anc_rot_x = anc_rot_y = anc_rot_z = None
    else:
        anc_rot_x = anc_rot_y = anc_rot_z = None

    def _sanitize_points3d(pts):
        if pts is None:
            return np.zeros((0, 3), dtype=np.float64)
        arr_raw = np.asarray(pts, dtype=np.float64)
        if arr_raw.size < 3:
            return np.zeros((0, 3), dtype=np.float64)
        arr = arr_raw.reshape(-1, 3)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr.astype(np.float64)

    centroid_pts_3d = _sanitize_points3d(centroid_3d)
    if centroid_pts_3d.shape[0] == 0:
        centroid_pts_3d = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    handle_pts_3d = _sanitize_points3d(handle_3d)
    target_pts_3d = _sanitize_points3d(target_3d)
    handle_rot_pts_3d = _sanitize_points3d(handle_rotated_3d)
    handle_third_pts_3d = _sanitize_points3d(handle_third_3d)
    target_third_pts_3d = _sanitize_points3d(target_third_3d)

    if handle_rot_pts_3d.shape[0] == 0 and handle_pts_3d.shape[0] > 0:
        R = resolve_rotation_matrix(
            rotation_axis=rotation_axis,
            rotation_angle_deg=rotation_angle_deg,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg
        )
        handle_rot_pts_3d = (R @ handle_pts_3d.T).T
    elif handle_rot_pts_3d.shape[0] == 1 and handle_pts_3d.shape[0] > 1:
        handle_rot_pts_3d = np.repeat(handle_rot_pts_3d, handle_pts_3d.shape[0], axis=0)

    if target_third_pts_3d.shape[0] == 0 and target_pts_3d.shape[0] > 0:
        target_third_pts_3d = target_pts_3d.copy()

    cent_x, cent_y, cent_z = convert_coords(centroid_pts_3d[0])
    has_handle = handle_pts_3d.shape[0] > 0
    has_target = target_pts_3d.shape[0] > 0
    has_rotated_handle = handle_rot_pts_3d.shape[0] > 0
    if has_handle:
        handle_x, handle_y, handle_z = convert_coords(handle_pts_3d)
    if has_target:
        target_x, target_y, target_z = convert_coords(target_pts_3d)
    if has_rotated_handle:
        handle_rot_x, handle_rot_y, handle_rot_z = convert_coords(handle_rot_pts_3d)
    has_third_handle = handle_third_pts_3d.shape[0] > 0
    has_third_target = target_third_pts_3d.shape[0] > 0
    has_third_cloud = third_x is not None
    # Hybrid 需要稳定展示三阶段：只要给了第三阶段点云，就固定绘制第三行。
    has_third_row = has_third_cloud
    if has_third_handle:
        handle_third_x, handle_third_y, handle_third_z = convert_coords(handle_third_pts_3d)
    if has_third_target:
        target_third_x, target_third_y, target_third_z = convert_coords(target_third_pts_3d)

    front_view_enabled = image_hw is not None and image_center_xy is not None
    if front_view_enabled:
        front_h = int(image_hw[0])
        front_w = int(image_hw[1])
        front_cx = float(image_center_xy[0])
        front_cy = float(image_center_xy[1])

        def convert_front_coords(pts):
            arr = np.asarray(pts, dtype=np.float64)
            if arr.ndim == 1:
                return arr[0] + front_cx, front_cy - arr[1]
            arr = arr.reshape(-1, 3)
            return arr[:, 0] + front_cx, front_cy - arr[:, 1]

        orig_fx, orig_fy = convert_front_coords(pts_orig)
        rot_fx, rot_fy = convert_front_coords(pts_rot)
        third_fx = third_fy = None
        if pts_third is not None and np.asarray(pts_third).reshape(-1, 3).shape[0] > 0:
            third_fx, third_fy = convert_front_coords(np.asarray(pts_third, dtype=np.float32).reshape(-1, 3))
        ex_orig_fx = ex_orig_fy = None
        if extra_orig_pts is not None:
            ex_orig_fx, ex_orig_fy = convert_front_coords(extra_orig_pts)
        ex_rot_fx = ex_rot_fy = None
        if extra_rot_pts is not None:
            ex_rot_fx, ex_rot_fy = convert_front_coords(extra_rot_pts)
        anc_orig_fx = anc_orig_fy = None
        if anchor_points_3d is not None and anc_orig_x is not None:
            anc_orig_fx, anc_orig_fy = convert_front_coords(anc_orig)
        anc_rot_fx = anc_rot_fy = None
        if anchor_points_rotated_3d is not None and anc_rot_x is not None:
            anc_rot_fx, anc_rot_fy = convert_front_coords(anc_rot)
        cent_fx, cent_fy = convert_front_coords(centroid_pts_3d[0])
        if has_handle:
            handle_fx, handle_fy = convert_front_coords(handle_pts_3d)
        if has_target:
            target_fx, target_fy = convert_front_coords(target_pts_3d)
        if has_rotated_handle:
            handle_rot_fx, handle_rot_fy = convert_front_coords(handle_rot_pts_3d)
        if has_third_handle:
            handle_third_fx, handle_third_fy = convert_front_coords(handle_third_pts_3d)
        if has_third_target:
            target_third_fx, target_third_fy = convert_front_coords(target_third_pts_3d)
        front_x_limits, front_y_limits = _compute_front_framed_limits(
            (front_h, front_w),
            point_sets=[
                (orig_fx, orig_fy),
                (rot_fx, rot_fy),
                (third_fx, third_fy),
                (ex_orig_fx, ex_orig_fy),
                (ex_rot_fx, ex_rot_fy),
                (anc_orig_fx, anc_orig_fy),
                (anc_rot_fx, anc_rot_fy),
                ([cent_fx], [cent_fy]),
                (handle_fx, handle_fy) if has_handle else None,
                (target_fx, target_fy) if has_target else None,
                (handle_rot_fx, handle_rot_fy) if has_rotated_handle else None,
                (handle_third_fx, handle_third_fy) if has_third_handle else None,
                (target_third_fx, target_third_fy) if has_third_target else None,
            ],
            padding_ratio=0.10,
            min_pad_px=18.0,
        )

    # 缺失控制点/目标点时给出明确日志，便于定位“点云里看不到控制点和方向”的来源。
    handle_count = int(handle_pts_3d.shape[0])
    target_count = int(target_pts_3d.shape[0])
    rotated_handle_count = int(handle_rot_pts_3d.shape[0])
    third_handle_count = int(handle_third_pts_3d.shape[0])
    third_target_count = int(target_third_pts_3d.shape[0])
    if handle_count == 0 or target_count == 0:
        print(
            "[3DRotate][PointCloudVis] key points missing: "
            f"handle_count={handle_count}, target_count={target_count}, "
            f"rotated_handle_count={rotated_handle_count}, "
            f"third_handle_count={third_handle_count}, third_target_count={third_target_count}. "
            "Likely reason: upstream handle/target pairs are empty or filtered out."
        )

    # 统一可视化尺寸参数（点位缩小，图例字体与条目间距放大）
    POINT_SIZE_MAIN = 1.2
    POINT_SIZE_EXTRA = 1.2
    TITLE_FONTSIZE = 22
    ARROW_LINEWIDTH = 2.0
    ARROW_HEAD_RATIO = 0.22
    TITLE_PAD_DEFAULT = 8
    TITLE_PAD_PAPER_SINGLE = 2

    def draw_key_points(ax, row_mode="original"):
        """在ax上绘制关键点"""
        # 计算点大小比例（基于点云范围）
        x_range = pts_orig[:, 0].max() - pts_orig[:, 0].min()
        y_range = pts_orig[:, 1].max() - pts_orig[:, 1].min()
        z_range = pts_orig[:, 2].max() - pts_orig[:, 2].min()
        max_range = max(x_range, y_range, z_range)

        # 关键点大小与点云范围成比例
        key_point_size = max(60.0, min(170.0, max_range * 0.32))

        # 质心 - 绿色大球
        ax.scatter([cent_x], [cent_y], [cent_z], c='lime', s=key_point_size, marker='o',
                      edgecolors='black', linewidths=1.8, label='Origin (Centroid)', zorder=20, depthshade=False)

        if row_mode == "original":
            # 原始图：显示控制点和目标点
            if has_handle:
                ax.scatter(handle_x, handle_y, handle_z, c='red', s=key_point_size * 0.8, marker='^',
                          edgecolors='black', linewidths=1.8, label='Handle Point', zorder=21, depthshade=False)
            if has_target:
                ax.scatter(target_x, target_y, target_z, c='blue', s=key_point_size * 0.8, marker='s',
                          edgecolors='black', linewidths=1.8, label='Target Point', zorder=21, depthshade=False)
            # 画箭头从 handle 指向 target
            if has_handle and has_target:
                pair_n = int(min(handle_pts_3d.shape[0], target_pts_3d.shape[0]))
                for i in range(pair_n):
                    label = 'Drag Direction' if i == 0 else None
                    dx = float(target_x[i] - handle_x[i])
                    dy = float(target_y[i] - handle_y[i])
                    dz = float(target_z[i] - handle_z[i])
                    if (dx * dx + dy * dy + dz * dz) < 1e-8:
                        continue
                    ax.plot(
                        [handle_x[i], target_x[i]],
                        [handle_y[i], target_y[i]],
                        [handle_z[i], target_z[i]],
                        'yellow',
                        linewidth=ARROW_LINEWIDTH * 0.75,
                        alpha=0.75,
                        zorder=22,
                        label=label,
                    )
                    ax.quiver(
                        float(handle_x[i]),
                        float(handle_y[i]),
                        float(handle_z[i]),
                        dx,
                        dy,
                        dz,
                        color='yellow',
                        linewidth=ARROW_LINEWIDTH,
                        arrow_length_ratio=ARROW_HEAD_RATIO,
                        normalize=False,
                        zorder=23,
                    )
        else:
            # 旋转后/第三行：显示控制点与目标点
            if row_mode == "third":
                _has_h = has_third_handle
                _has_t = has_third_target
                _h_x, _h_y, _h_z = handle_third_x, handle_third_y, handle_third_z
                _t_x, _t_y, _t_z = target_third_x, target_third_y, target_third_z
                _h_pts = handle_third_pts_3d
                _t_pts = target_third_pts_3d
                _h_label = 'Projected Handle'
            else:
                _has_h = has_rotated_handle
                _has_t = has_target
                _h_x, _h_y, _h_z = handle_rot_x, handle_rot_y, handle_rot_z
                _t_x, _t_y, _t_z = target_x, target_y, target_z
                _h_pts = handle_rot_pts_3d
                _t_pts = target_pts_3d
                _h_label = 'Rotated Handle'

            if _has_t:
                ax.scatter(_t_x, _t_y, _t_z, c='blue', s=key_point_size * 0.8, marker='s',
                          edgecolors='black', linewidths=1.8, label='Target Point', zorder=21, depthshade=False)
            if _has_h:
                # 按用户要求：控制点统一使用红色，不使用橙色。
                ax.scatter(_h_x, _h_y, _h_z, c='red', s=key_point_size * 0.8, marker='^',
                          edgecolors='black', linewidths=1.8, label=_h_label,
                          zorder=24, depthshade=False)
            if _has_h and _has_t:
                pair_n = int(min(_h_pts.shape[0], _t_pts.shape[0]))
                overlap_eps = max(2.0, max_range * 0.012)
                for i in range(pair_n):
                    dx = float(_t_x[i] - _h_x[i])
                    dy = float(_t_y[i] - _h_y[i])
                    dz = float(_t_z[i] - _h_z[i])
                    dist = float(np.sqrt(dx * dx + dy * dy + dz * dz))
                    label = 'Drag Direction' if i == 0 else None
                    if dist <= overlap_eps:
                        # 控制点与目标点几乎重合时，使用“原始拖拽方向”回退箭头，避免看起来像没画方向。
                        if has_handle and i < handle_pts_3d.shape[0]:
                            fdx = float(target_x[i] - handle_x[i])
                            fdy = float(target_y[i] - handle_y[i])
                            fdz = float(target_z[i] - handle_z[i])
                            fnorm = float(np.sqrt(fdx * fdx + fdy * fdy + fdz * fdz))
                            if fnorm > 1e-8:
                                arrow_len = max(12.0, 0.08 * max_range)
                                ux, uy, uz = fdx / fnorm, fdy / fnorm, fdz / fnorm
                                sx = float(_t_x[i] - ux * arrow_len)
                                sy = float(_t_y[i] - uy * arrow_len)
                                sz = float(_t_z[i] - uz * arrow_len)
                                ax.plot(
                                    [sx, _t_x[i]],
                                    [sy, _t_y[i]],
                                    [sz, _t_z[i]],
                                    'yellow',
                                    linewidth=ARROW_LINEWIDTH * 0.9,
                                    alpha=0.9,
                                    zorder=22,
                                    label=label,
                                )
                                ax.quiver(
                                    sx,
                                    sy,
                                    sz,
                                    ux * arrow_len,
                                    uy * arrow_len,
                                    uz * arrow_len,
                                    color='yellow',
                                    linewidth=ARROW_LINEWIDTH,
                                    arrow_length_ratio=ARROW_HEAD_RATIO,
                                    normalize=False,
                                    zorder=23,
                                )
                        continue
                    ax.plot(
                        [_h_x[i], _t_x[i]],
                        [_h_y[i], _t_y[i]],
                        [_h_z[i], _t_z[i]],
                        'yellow',
                        linewidth=ARROW_LINEWIDTH * 0.75,
                        alpha=0.75,
                        zorder=22,
                        label=label,
                    )
                    ax.quiver(
                        float(_h_x[i]),
                        float(_h_y[i]),
                        float(_h_z[i]),
                        dx,
                        dy,
                        dz,
                        color='yellow',
                        linewidth=ARROW_LINEWIDTH,
                        arrow_length_ratio=ARROW_HEAD_RATIO,
                        normalize=False,
                        zorder=23,
                    )

    def draw_key_points_front(ax, row_mode="original"):
        x_range = pts_orig[:, 0].max() - pts_orig[:, 0].min()
        y_range = pts_orig[:, 1].max() - pts_orig[:, 1].min()
        z_range = pts_orig[:, 2].max() - pts_orig[:, 2].min()
        max_range = max(x_range, y_range, z_range)
        key_point_size = max(55.0, min(150.0, max_range * 0.30))

        ax.scatter([cent_fx], [cent_fy], c='lime', s=key_point_size, marker='o',
                   edgecolors='black', linewidths=1.2, zorder=20)

        if row_mode == "original":
            if has_handle:
                ax.scatter(handle_fx, handle_fy, c='red', s=key_point_size * 0.8, marker='^',
                           edgecolors='black', linewidths=1.2, zorder=21)
            if has_target:
                ax.scatter(target_fx, target_fy, c='blue', s=key_point_size * 0.8, marker='s',
                           edgecolors='black', linewidths=1.2, zorder=21)
            if has_handle and has_target:
                pair_n = int(min(handle_pts_3d.shape[0], target_pts_3d.shape[0]))
                for i in range(pair_n):
                    dx = float(target_fx[i] - handle_fx[i])
                    dy = float(target_fy[i] - handle_fy[i])
                    if (dx * dx + dy * dy) < 1e-8:
                        continue
                    ax.plot([handle_fx[i], target_fx[i]], [handle_fy[i], target_fy[i]],
                            color='yellow', linewidth=ARROW_LINEWIDTH * 0.85, alpha=0.85, zorder=22)
                    ax.quiver(
                        float(handle_fx[i]),
                        float(handle_fy[i]),
                        dx,
                        dy,
                        angles='xy',
                        scale_units='xy',
                        scale=1.0,
                        color='yellow',
                        width=0.004,
                        zorder=23,
                    )
        else:
            if row_mode == "third":
                _has_h = has_third_handle
                _has_t = has_third_target
                _h_x, _h_y = handle_third_fx, handle_third_fy
                _t_x, _t_y = target_third_fx, target_third_fy
                _h_pts = handle_third_pts_3d
                _t_pts = target_third_pts_3d
            else:
                _has_h = has_rotated_handle
                _has_t = has_target
                _h_x, _h_y = handle_rot_fx, handle_rot_fy
                _t_x, _t_y = target_fx, target_fy
                _h_pts = handle_rot_pts_3d
                _t_pts = target_pts_3d
            if _has_t:
                ax.scatter(_t_x, _t_y, c='blue', s=key_point_size * 0.8, marker='s',
                           edgecolors='black', linewidths=1.2, zorder=21)
            if _has_h:
                ax.scatter(_h_x, _h_y, c='red', s=key_point_size * 0.8, marker='^',
                           edgecolors='black', linewidths=1.2, zorder=24)
            if _has_h and _has_t:
                pair_n = int(min(_h_pts.shape[0], _t_pts.shape[0]))
                for i in range(pair_n):
                    dx = float(_t_x[i] - _h_x[i])
                    dy = float(_t_y[i] - _h_y[i])
                    if (dx * dx + dy * dy) < 1e-8:
                        continue
                    ax.plot([_h_x[i], _t_x[i]], [_h_y[i], _t_y[i]],
                            color='yellow', linewidth=ARROW_LINEWIDTH * 0.85, alpha=0.85, zorder=22)
                    ax.quiver(
                        float(_h_x[i]),
                        float(_h_y[i]),
                        dx,
                        dy,
                        angles='xy',
                        scale_units='xy',
                        scale=1.0,
                        color='yellow',
                        width=0.004,
                        zorder=23,
                    )

    def draw_single_view(ax, row_mode, view_idx, view_name, elev, azim, title_text):
        is_front_view = bool(front_view_enabled and _is_front_view(view_name))

        if is_front_view:
            x_limits, y_limits = front_x_limits, front_y_limits
            axis_ready = _setup_front_2d_axis(
                ax,
                x_limits=x_limits,
                y_limits=y_limits,
                title_text="",
                title_fontsize=TITLE_FONTSIZE,
                tick_label_fontsize=7,
                title_pad=2 if paper_no_text_single else 8,
                show_title=False,
            )
            if axis_ready:
                _draw_front_2d_background(ax, x_limits, y_limits)
            if row_mode == "original":
                ax.scatter(orig_fx, orig_fy, c=colors, s=POINT_SIZE_MAIN, alpha=0.72, zorder=1)
                if anc_orig_fx is not None:
                    ax.scatter(
                        anc_orig_fx,
                        anc_orig_fy,
                        c='#9ed9ff',
                        s=16,
                        alpha=0.95,
                        marker='o',
                        edgecolors='white',
                        linewidths=0.4,
                        zorder=2,
                    )
                if ex_orig_fx is not None:
                    if extra_orig_cols is not None:
                        ax.scatter(
                            ex_orig_fx,
                            ex_orig_fy,
                            c=extra_orig_cols,
                            s=POINT_SIZE_EXTRA,
                            alpha=0.35,
                            zorder=1,
                        )
                    else:
                        ax.scatter(
                            ex_orig_fx,
                            ex_orig_fy,
                            c='#66ffff',
                            s=POINT_SIZE_EXTRA,
                            alpha=0.35,
                            zorder=1,
                        )
                draw_key_points_front(ax, row_mode="original")
            else:
                if row_mode == "third" and has_third_cloud and third_fx is not None:
                    ax.scatter(third_fx, third_fy, c=colors, s=POINT_SIZE_MAIN, alpha=0.72, zorder=1)
                else:
                    ax.scatter(rot_fx, rot_fy, c=colors, s=POINT_SIZE_MAIN, alpha=0.72, zorder=1)
                if anc_rot_fx is not None:
                    ax.scatter(
                        anc_rot_fx,
                        anc_rot_fy,
                        c='#9ed9ff',
                        s=16,
                        alpha=0.95,
                        marker='o',
                        edgecolors='white',
                        linewidths=0.4,
                        zorder=2,
                    )
                elif anc_orig_fx is not None:
                    ax.scatter(
                        anc_orig_fx,
                        anc_orig_fy,
                        c='#9ed9ff',
                        s=16,
                        alpha=0.95,
                        marker='o',
                        edgecolors='white',
                        linewidths=0.4,
                        zorder=2,
                    )
                if ex_rot_fx is not None:
                    if extra_rot_cols is not None:
                        ax.scatter(
                            ex_rot_fx,
                            ex_rot_fy,
                            c=extra_rot_cols,
                            s=POINT_SIZE_EXTRA,
                            alpha=0.35,
                            zorder=1,
                        )
                    else:
                        ax.scatter(
                            ex_rot_fx,
                            ex_rot_fy,
                            c='#66ffff',
                            s=POINT_SIZE_EXTRA,
                            alpha=0.35,
                            zorder=1,
                        )
                draw_key_points_front(ax, row_mode=row_mode)

            return

        # mplot3d 默认按深度自动重排，可能把关键点/拖拽线压到点云后面。
        # 显式关闭后，按 zorder 保证控制点与 Drag Direction 可见。
        try:
            ax.computed_zorder = False
        except Exception:
            pass

        if row_mode == "original":
            ax.scatter(orig_x, orig_y, orig_z, c=colors, s=POINT_SIZE_MAIN, alpha=0.7, zorder=1)
            if anc_orig_x is not None:
                ax.scatter(
                    anc_orig_x, anc_orig_y, anc_orig_z,
                    c='#9ed9ff', s=16, alpha=0.95, marker='o',
                    edgecolors='white', linewidths=0.4,
                    label='Anchors',
                    zorder=2,
                )
            if ex_orig_x is not None:
                if extra_orig_cols is not None:
                    ax.scatter(
                        ex_orig_x, ex_orig_y, ex_orig_z,
                        c=extra_orig_cols, s=POINT_SIZE_EXTRA, alpha=0.35,
                        label='Completed Surface',
                        zorder=1,
                    )
                else:
                    ax.scatter(
                        ex_orig_x, ex_orig_y, ex_orig_z,
                        c='#66ffff', s=POINT_SIZE_EXTRA, alpha=0.35,
                        label='Completed Surface',
                        zorder=1,
                    )
            draw_key_points(ax, row_mode="original")
        else:
            if row_mode == "third" and has_third_cloud:
                ax.scatter(third_x, third_y, third_z, c=colors, s=POINT_SIZE_MAIN, alpha=0.7, zorder=1)
            else:
                ax.scatter(rot_x, rot_y, rot_z, c=colors, s=POINT_SIZE_MAIN, alpha=0.7, zorder=1)
            if anc_rot_x is not None:
                ax.scatter(
                    anc_rot_x, anc_rot_y, anc_rot_z,
                    c='#9ed9ff', s=16, alpha=0.95, marker='o',
                    edgecolors='white', linewidths=0.4,
                    label='Anchors',
                    zorder=2,
                )
            elif anc_orig_x is not None:
                ax.scatter(
                    anc_orig_x, anc_orig_y, anc_orig_z,
                    c='#9ed9ff', s=16, alpha=0.95, marker='o',
                    edgecolors='white', linewidths=0.4,
                    label='Anchors',
                    zorder=2,
                )
            if ex_rot_x is not None:
                if extra_rot_cols is not None:
                    ax.scatter(
                        ex_rot_x, ex_rot_y, ex_rot_z,
                        c=extra_rot_cols, s=POINT_SIZE_EXTRA, alpha=0.35,
                        label='Completed Surface',
                        zorder=1,
                    )
                else:
                    ax.scatter(
                        ex_rot_x, ex_rot_y, ex_rot_z,
                        c='#66ffff', s=POINT_SIZE_EXTRA, alpha=0.35,
                        label='Completed Surface',
                        zorder=1,
                    )
            draw_key_points(ax, row_mode=row_mode)

        title_pad = TITLE_PAD_PAPER_SINGLE if paper_no_text_single else TITLE_PAD_DEFAULT
        ax.view_init(elev=elev, azim=azim)
        _apply_view_geometry(
            ax,
            view_name=view_name,
            image_hw=image_hw,
            image_center_xy=image_center_xy,
        )
        if view_idx >= 2:
            ax.invert_xaxis()
        _apply_standard_3d_axes(
            ax,
            title_text=title_text if (not paper_no_text_single) else "",
            title_fontsize=TITLE_FONTSIZE,
            tick_label_fontsize=10,
            title_pad=title_pad,
            show_title=False,
        )

    # 创建子图：默认2行（原始/旋转后）；3D-Non-Rigid 可扩展为3行
    # 4个机位：正面、侧面、左前、右前
    n_rows = 3 if has_third_row else 2
    fig = plt.figure(figsize=(32, 24 if n_rows == 3 else 16))

    # 视角定义：
    # 正面: elev=0, azim=-90 (从正前方看)
    # 侧面: elev=0, azim=0 (从右侧看)
    # 左前: elev=20, azim=-120 (从左前方看)
    # 右前: elev=20, azim=-60 (从右前方看)
    views = [
        ('Front', 0, -90),
        ('Side', 0, 0),
        ('Left-Front', 20, 45),
        ('Right-Front', 20, 135),
    ]

    # 第一行：原始点云
    for i, (view_name, elev, azim) in enumerate(views):
        ax = (
            fig.add_subplot(n_rows, 4, i + 1)
            if (front_view_enabled and _is_front_view(view_name))
            else fig.add_subplot(n_rows, 4, i + 1, projection='3d')
        )
        draw_single_view(
            ax=ax,
            row_mode="original",
            view_idx=i,
            view_name=view_name,
            elev=elev,
            azim=azim,
            title_text=f'Original - {view_name}',
        )

    rot_title = None if paper_no_text_single else f"{rotation_axis}: {rotation_angle_deg:.1f}°"
    if yaw_deg is not None or pitch_deg is not None:
        # 按用户要求：可视化中不显示 Yaw/Pitch 数值标注。
        rot_title = None

    # 第二行：旋转后点云
    for i, (view_name, elev, azim) in enumerate(views):
        ax = (
            fig.add_subplot(n_rows, 4, i + 5)
            if (front_view_enabled and _is_front_view(view_name))
            else fig.add_subplot(n_rows, 4, i + 5, projection='3d')
        )
        draw_single_view(
            ax=ax,
            row_mode="rotated",
            view_idx=i,
            view_name=view_name,
            elev=elev,
            azim=azim,
            title_text=(
                f'Rotated - {view_name}'
                if not rot_title else f'Rotated - {view_name}\n({rot_title})'
            ),
        )

    # 第三行：可选（用于 Hybrid 展示“非刚性抵达后”阶段）
    if has_third_row:
        for i, (view_name, elev, azim) in enumerate(views):
            ax = (
                fig.add_subplot(n_rows, 4, i + 9)
                if (front_view_enabled and _is_front_view(view_name))
                else fig.add_subplot(n_rows, 4, i + 9, projection='3d')
            )
            draw_single_view(
                ax=ax,
                row_mode="third",
                view_idx=i,
                view_name=view_name,
                elev=elev,
                azim=azim,
                title_text=(
                    f'{third_row_title} - {view_name}'
                    if not rot_title else f'{third_row_title} - {view_name}\n({rot_title})'
                ),
            )

    fig.patch.set_alpha(0.0)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.04, hspace=0.04)
    save_path = os.path.join(save_dir, output_name)
    plt.savefig(save_path, dpi=150, transparent=True)
    plt.close()
    print(f"[3DRotate] Saved 3D point cloud visualization: {save_path}")

    if save_single_views:
        base_name, ext = os.path.splitext(output_name)
        ext = ext or ".png"
        row_entries = [("original", "original"), ("rotated", "rotated")]
        if has_third_row:
            row_entries.append(("third", "projected"))
        for row_mode, pose_tag in row_entries:
            for i, (view_name, elev, azim) in enumerate(views):
                is_front_single = bool(front_view_enabled and _is_front_view(view_name))
                fig_single = plt.figure(figsize=(5.12, 5.12), dpi=100)
                if is_front_single:
                    fig_single.patch.set_facecolor(FRONT_BG_RGBA)
                    fig_single.patch.set_alpha(1.0)
                else:
                    fig_single.patch.set_alpha(0.0)
                ax_single = (
                    fig_single.add_subplot(1, 1, 1)
                    if is_front_single
                    else fig_single.add_subplot(1, 1, 1, projection='3d')
                )
                if row_mode == "rotated":
                    title = (
                        f'Rotated - {view_name}'
                        if not rot_title else f'Rotated - {view_name}\n({rot_title})'
                    )
                elif row_mode == "third":
                    title = (
                        f'{third_row_title} - {view_name}'
                        if not rot_title else f'{third_row_title} - {view_name}\n({rot_title})'
                    )
                else:
                    title = f'Original - {view_name}'
                draw_single_view(
                    ax=ax_single,
                    row_mode=row_mode,
                    view_idx=i,
                    view_name=view_name,
                    elev=elev,
                    azim=azim,
                    title_text="",
                )
                safe_view_name = view_name.lower().replace(" ", "_").replace("-", "_")
                single_name = f"{base_name}__view_{pose_tag}_{i+1:02d}_{safe_view_name}{ext}"
                single_path = os.path.join(save_dir, single_name)
                if is_front_single:
                    ax_single.set_position([0.0, 0.0, 1.0, 1.0])
                else:
                    ax_single.set_position([0.01, 0.01, 0.98, 0.98])
                plt.savefig(single_path, dpi=100, transparent=(not is_front_single), pad_inches=0.0)
                plt.close(fig_single)
        print(f"[3DRotate] Saved per-view files for: {output_name}")

    # 打印点云范围信息用于调试
    print(f"[3DRotate] Point cloud ranges:")
    print(f"  Original: X=[{pts_orig[:,0].min():.1f}, {pts_orig[:,0].max():.1f}], "
          f"Y=[{pts_orig[:,1].min():.1f}, {pts_orig[:,1].max():.1f}], "
          f"Z=[{pts_orig[:,2].min():.1f}, {pts_orig[:,2].max():.1f}]")
    print(f"  Rotated:  X=[{pts_rot[:,0].min():.1f}, {pts_rot[:,0].max():.1f}], "
          f"Y=[{pts_rot[:,1].min():.1f}, {pts_rot[:,1].max():.1f}], "
          f"Z=[{pts_rot[:,2].min():.1f}, {pts_rot[:,2].max():.1f}]")
    if extra_orig_pts is not None:
        print(f"  Extra original points: {extra_orig_pts.shape[0]}")
    if extra_rot_pts is not None:
        print(f"  Extra rotated points: {extra_rot_pts.shape[0]}")
    if centroid_3d is not None:
        print(f"  Centroid 3D: [{centroid_3d[0]:.1f}, {centroid_3d[1]:.1f}, {centroid_3d[2]:.1f}]")
    if handle_pts_3d.shape[0] > 0:
        h0 = handle_pts_3d[0]
        print(f"  Handle 3D[0]: [{h0[0]:.1f}, {h0[1]:.1f}, {h0[2]:.1f}], count={handle_pts_3d.shape[0]}")
    if target_pts_3d.shape[0] > 0:
        t0 = target_pts_3d[0]
        print(f"  Target 3D[0]: [{t0[0]:.1f}, {t0[1]:.1f}, {t0[2]:.1f}], count={target_pts_3d.shape[0]}")


def _set_axes_equal(ax):
    """设置3D坐标轴等比例，确保物体不变形"""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])


# ==========================================
# 可视化函数
# ==========================================

def vis_3drotate_01_fill_scopes(source_image_np, m_start_full, m_pseudo_full, bg_filled_rgb, save_dir):
    """【3DRotate_01】Fill Scopes"""
    H, W = source_image_np.shape[:2]

    if m_start_full.ndim == 3:
        m_start_full = m_start_full[:, :, 0]
    if m_pseudo_full.ndim == 3:
        m_pseudo_full = m_pseudo_full[:, :, 0]

    vis_hole = np.zeros_like(source_image_np)
    vis_hole[m_start_full > 0.5] = [255, 255, 255]
    put_text_with_outline(vis_hole, "1. Hole (To Fill)", (10, 30), scale=0.5)

    vis_logic = source_image_np.copy()
    bool_pseudo = m_pseudo_full > 0.5
    bool_start = m_start_full > 0.5
    bool_bg = ~bool_pseudo
    vis_logic[bool_bg] = (vis_logic[bool_bg] * 0.3).astype(np.uint8)
    overlay = vis_logic.copy()
    overlay[bool_pseudo & (~bool_start)] = [255, 100, 100]
    overlay[bool_start] = [100, 255, 100]
    vis_logic = cv2.addWeighted(vis_logic, 0.5, overlay, 0.5, 0)
    put_text_with_outline(vis_logic, "2. Mask Logic", (10, 30), scale=0.5)

    vis_orig = source_image_np.copy()
    put_text_with_outline(vis_orig, "3. Original", (10, 30), scale=0.5)

    vis_bg = bg_filled_rgb.copy() if isinstance(bg_filled_rgb, np.ndarray) else source_image_np.copy()
    put_text_with_outline(vis_bg, "4. BG Filled", (10, 30), scale=0.5)

    combined = np.hstack([vis_hole, vis_logic, vis_orig, vis_bg])
    save_path = os.path.join(save_dir, "3DRotate_01_fill_scopes.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print(f"[3DRotate] Saved: {save_path}")


def vis_3drotate_02_masks_cutouts(source_image_np, sam_mask, m_start_full,
                                   all_handles, all_targets, save_dir):
    """【3DRotate_02】Masks & Cutouts"""
    H, W = source_image_np.shape[:2]

    if m_start_full.ndim == 3:
        m_start_full = m_start_full[:, :, 0]

    vis_mask = np.zeros_like(source_image_np)
    vis_mask[m_start_full > 0.5] = [255, 255, 255]
    put_text_with_outline(vis_mask, "1. User Mask", (10, 30), scale=0.5)

    vis_sam = source_image_np.copy()
    if sam_mask is not None:
        sam_overlay = vis_sam.copy()
        sam_overlay[sam_mask > 127] = [0, 255, 0]
        vis_sam = cv2.addWeighted(vis_sam, 0.6, sam_overlay, 0.4, 0)
    put_text_with_outline(vis_sam, "2. SAM Mask", (10, 30), scale=0.5)

    vis_points = source_image_np.copy()
    for h_pt, t_pt in zip(all_handles, all_targets):
        hx, hy = int(h_pt[1]), int(h_pt[0])
        tx, ty = int(t_pt[1]), int(t_pt[0])
        cv2.circle(vis_points, (hx, hy), 8, (255, 0, 0), -1)
        cv2.circle(vis_points, (tx, ty), 8, (0, 0, 255), -1)
        cv2.arrowedLine(vis_points, (hx, hy), (tx, ty), (255, 255, 255), 2, tipLength=0.2)
    put_text_with_outline(vis_points, "3. Control Points", (10, 30), scale=0.5)

    vis_cutout = source_image_np.copy()
    if sam_mask is not None:
        vis_cutout[sam_mask < 128] = 0
    else:
        vis_cutout[m_start_full < 0.5] = 0
    put_text_with_outline(vis_cutout, "4. Cutout (SAM)", (10, 30), scale=0.5)

    combined = np.hstack([vis_mask, vis_sam, vis_points, vis_cutout])
    save_path = os.path.join(save_dir, "3DRotate_02_masks_cutouts.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print(f"[3DRotate] Saved: {save_path}")


def vis_3drotate_03_trace_compare(source_image_np, warped_image_np, m_start_full, m_end_full,
                                   all_handles, all_targets, save_dir):
    """【3DRotate_03】Trace Compare"""
    H, W = source_image_np.shape[:2]

    if m_start_full.ndim == 3:
        m_start_full = m_start_full[:, :, 0]
    if m_end_full.ndim == 3:
        m_end_full = m_end_full[:, :, 0]

    vis_orig = source_image_np.copy()
    mask_bg = m_start_full < 0.5
    vis_orig[mask_bg] = (vis_orig[mask_bg] * 0.3).astype(np.uint8)
    put_text_with_outline(vis_orig, "1. Original Subject", (10, 30), scale=0.5)

    vis_warped = warped_image_np.copy()
    mask_bg_end = m_end_full < 0.5
    vis_warped[mask_bg_end] = (vis_warped[mask_bg_end] * 0.3).astype(np.uint8)
    put_text_with_outline(vis_warped, "2. Rotated Subject", (10, 30), scale=0.5)

    vis_overlay = source_image_np.copy().astype(np.float32)
    mask_orig_2d = m_start_full > 0.5
    red_overlay = np.zeros_like(vis_overlay)
    red_overlay[:, :] = [255, 0, 0]
    vis_overlay[mask_orig_2d] = vis_overlay[mask_orig_2d] * 0.5 + red_overlay[mask_orig_2d] * 0.5
    mask_new_2d = m_end_full > 0.5
    green_overlay = np.zeros_like(vis_overlay)
    green_overlay[:, :] = [0, 255, 0]
    vis_overlay[mask_new_2d] = vis_overlay[mask_new_2d] * 0.5 + green_overlay[mask_new_2d] * 0.5
    vis_overlay = vis_overlay.astype(np.uint8)
    put_text_with_outline(vis_overlay, "3. Before(R) vs After(G)", (10, 30), scale=0.5)

    vis_mask_change = np.zeros_like(source_image_np)
    vis_mask_change[m_start_full > 0.5] = [255, 0, 0]
    vis_mask_change[m_end_full > 0.5] = [0, 255, 0]
    overlap = (m_start_full > 0.5) & (m_end_full > 0.5)
    vis_mask_change[overlap] = [255, 255, 0]
    put_text_with_outline(vis_mask_change, "4. Mask: Orig(R)/New(G)", (10, 30), scale=0.5)

    combined = np.hstack([vis_orig, vis_warped, vis_overlay, vis_mask_change])
    save_path = os.path.join(save_dir, "3DRotate_03_trace_compare.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print(f"[3DRotate] Saved: {save_path}")


def vis_3drotate_04_final_preview(source_image_np, final_composed_rgb, bg_filled_rgb,
                                   all_handles, all_targets, save_dir):
    """【3DRotate_04】Final Preview"""
    H, W = source_image_np.shape[:2]

    vis_orig = source_image_np.copy()
    put_text_with_outline(vis_orig, "1. Original", (10, 30), scale=0.5)

    vis_bg = bg_filled_rgb.copy() if isinstance(bg_filled_rgb, np.ndarray) else source_image_np.copy()
    put_text_with_outline(vis_bg, "2. Background", (10, 30), scale=0.5)

    vis_result = final_composed_rgb.copy()
    put_text_with_outline(vis_result, "3. Final Result", (10, 30), scale=0.5)

    vis_with_pts = final_composed_rgb.copy()
    for h_pt, t_pt in zip(all_handles, all_targets):
        hx, hy = int(h_pt[1]), int(h_pt[0])
        tx, ty = int(t_pt[1]), int(t_pt[0])
        cv2.circle(vis_with_pts, (hx, hy), 6, (255, 0, 0), -1)
        cv2.circle(vis_with_pts, (tx, ty), 6, (0, 255, 0), -1)
        cv2.arrowedLine(vis_with_pts, (hx, hy), (tx, ty), (255, 255, 255), 2, tipLength=0.2)
    put_text_with_outline(vis_with_pts, "4. Result + Points", (10, 30), scale=0.5)

    combined = np.hstack([vis_orig, vis_bg, vis_result, vis_with_pts])
    save_path = os.path.join(save_dir, "3DRotate_04_final_preview.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print(f"[3DRotate] Saved: {save_path}")


def vis_depth_01_full_depth(source_image_np, depth_map, depth_colored, save_dir):
    """【depth_01】整体深度图"""
    H, W = source_image_np.shape[:2]

    vis_orig = source_image_np.copy()

    depth_norm = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6) * 255).astype(np.uint8)
    vis_depth_gray = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2RGB)

    vis_depth_color = depth_colored.copy()

    vis_overlay = cv2.addWeighted(source_image_np, 0.5, depth_colored, 0.5, 0)

    combined = np.hstack([vis_orig, vis_depth_gray, vis_depth_color, vis_overlay])
    save_path = os.path.join(save_dir, "depth_01_full_depth.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print(f"[3DRotate] Saved: {save_path}")


def vis_depth_02_component_depths(source_image_np, sam_mask, depth_map, centroid,
                                   rotation_axis, rotation_angle, save_dir,
                                   handle_points=None, target_points=None,
                                   yaw_deg=None, pitch_deg=None):
    """【depth_02】连通域深度信息 - 增强版，包含控制点和目标点"""
    H, W = source_image_np.shape[:2]
    depth_colored_full = create_depth_colormap(depth_map)
    cx, cy = centroid

    # 【修复】安全检查 handle_points 和 target_points
    has_points = (handle_points is not None and target_points is not None and
                  len(handle_points) > 0 and len(target_points) > 0)

    # 第一栏: 原图区域 + 所有关键点
    vis_region = source_image_np.copy()
    if sam_mask is not None:
        vis_region[sam_mask < 128] = (vis_region[sam_mask < 128] * 0.2).astype(np.uint8)
    # 绘制原点（质心）- 绿色大圆
    cv2.circle(vis_region, centroid, 10, (0, 255, 0), -1)
    cv2.circle(vis_region, centroid, 12, (255, 255, 255), 2)
    # 绘制控制点和目标点
    if has_points:
        for h_pt, t_pt in zip(handle_points, target_points):
            hx, hy = int(h_pt[0]), int(h_pt[1])  # [x,y] -> (x,y)
            tx, ty = int(t_pt[0]), int(t_pt[1])
            # 控制点 - 红色
            cv2.circle(vis_region, (hx, hy), 8, (255, 0, 0), -1)
            cv2.circle(vis_region, (hx, hy), 10, (255, 255, 255), 2)
            # 目标点 - 蓝色
            cv2.circle(vis_region, (tx, ty), 8, (0, 0, 255), -1)
            cv2.circle(vis_region, (tx, ty), 10, (255, 255, 255), 2)
            # 拖拽箭头 - 黄色
            cv2.arrowedLine(vis_region, (hx, hy), (tx, ty), (255, 255, 0), 3, tipLength=0.2)
    put_text_with_outline(vis_region, "1. Points on Image", (10, 30), scale=0.5)
    # 图例
    put_text_with_outline(vis_region, "Green=Origin", (10, H-60), scale=0.4, color=(0,255,0))
    put_text_with_outline(vis_region, "Red=Handle", (10, H-40), scale=0.4, color=(255,0,0))
    put_text_with_outline(vis_region, "Blue=Target", (10, H-20), scale=0.4, color=(0,0,255))

    # 第二栏: 深度图 + 关键点
    vis_depth_pts = depth_colored_full.copy()
    if sam_mask is not None:
        # 暗化非主体区域
        non_subject = sam_mask < 128
        vis_depth_pts[non_subject] = (vis_depth_pts[non_subject] * 0.3).astype(np.uint8)
    # 绘制原点
    cv2.circle(vis_depth_pts, centroid, 10, (0, 255, 0), -1)
    cv2.circle(vis_depth_pts, centroid, 12, (255, 255, 255), 2)
    # 绘制控制点和目标点
    if has_points:
        for h_pt, t_pt in zip(handle_points, target_points):
            hx, hy = int(h_pt[0]), int(h_pt[1])  # [x,y] -> (x,y)
            tx, ty = int(t_pt[0]), int(t_pt[1])
            cv2.circle(vis_depth_pts, (hx, hy), 8, (255, 0, 0), -1)
            cv2.circle(vis_depth_pts, (tx, ty), 8, (0, 0, 255), -1)
            cv2.arrowedLine(vis_depth_pts, (hx, hy), (tx, ty), (255, 255, 0), 3, tipLength=0.2)
    put_text_with_outline(vis_depth_pts, "2. Points on Depth", (10, 30), scale=0.5)

    # 第三栏: 3D坐标系可视化（在深度图上）
    vis_axis = depth_colored_full.copy()
    if sam_mask is not None:
        vis_axis[sam_mask < 128] = (vis_axis[sam_mask < 128] * 0.3).astype(np.uint8)
    # 绘制坐标轴
    axis_len = 80
    # 原点
    cv2.circle(vis_axis, (cx, cy), 6, (255, 255, 255), -1)
    # X轴 (红) - 图像右方向
    cv2.arrowedLine(vis_axis, (cx, cy), (cx + axis_len, cy), (255, 0, 0), 3, tipLength=0.15)
    if not _paper_no_text_single_enabled():
        cv2.putText(vis_axis, "X", (cx + axis_len + 5, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    # Y轴 (绿) - 图像下方向
    cv2.arrowedLine(vis_axis, (cx, cy), (cx, cy + axis_len), (0, 255, 0), 3, tipLength=0.15)
    if not _paper_no_text_single_enabled():
        cv2.putText(vis_axis, "Y", (cx + 5, cy + axis_len + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # Z轴 (蓝) - 深度方向，用斜线表示
    cv2.arrowedLine(vis_axis, (cx, cy), (cx - 40, cy - 40), (0, 100, 255), 3, tipLength=0.15)
    if not _paper_no_text_single_enabled():
        cv2.putText(vis_axis, "Z(depth)", (cx - 90, cy - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)
    # 标注旋转轴
    if yaw_deg is not None or pitch_deg is not None:
        cv2.arrowedLine(vis_axis, (cx, cy - 20), (cx, cy + 60), (0, 255, 255), 3, tipLength=0.1)
        cv2.arrowedLine(vis_axis, (cx - 20, cy), (cx + 60, cy), (0, 255, 255), 3, tipLength=0.1)
        put_text_with_outline(vis_axis, "3D Rotation", (10, H-20), scale=0.45, color=(0, 255, 255))
    elif rotation_axis == 'Y':
        cv2.arrowedLine(vis_axis, (cx, cy - 20), (cx, cy + 60), (0, 255, 255), 4, tipLength=0.1)
        put_text_with_outline(vis_axis, f"Rotate around Y: {rotation_angle:.1f}deg", (10, H-20), scale=0.45, color=(0,255,255))
    else:
        cv2.arrowedLine(vis_axis, (cx - 20, cy), (cx + 60, cy), (0, 255, 255), 4, tipLength=0.1)
        put_text_with_outline(vis_axis, f"Rotate around X: {rotation_angle:.1f}deg", (10, H-20), scale=0.45, color=(0,255,255))
    put_text_with_outline(vis_axis, "3. 3D Coordinate", (10, 30), scale=0.5)

    # 第四栏: 旋转信息详情
    vis_info = np.zeros_like(source_image_np)
    # 获取深度统计
    if sam_mask is not None:
        mask_depths = depth_map[sam_mask > 127]
        if len(mask_depths) > 0:
            depth_min_val = mask_depths.min()
            depth_max_val = mask_depths.max()
            depth_median = np.median(mask_depths)
        else:
            depth_min_val = depth_max_val = depth_median = 0
    else:
        depth_min_val = depth_map.min()
        depth_max_val = depth_map.max()
        depth_median = np.median(depth_map)

    angle_line = f"Rotation Angle: {rotation_angle:.1f} deg"
    if yaw_deg is not None or pitch_deg is not None:
        angle_line = "Rotation: Multi-axis"

    info_texts = [
        f"Origin (Centroid): ({cx}, {cy})",
        f"Rotation Axis: {rotation_axis}",
        angle_line,
        "",
        f"Depth Range: [{depth_min_val:.1f}, {depth_max_val:.1f}]",
        f"Median Depth: {depth_median:.1f}",
        "",
        "Axis Interpretation:",
        "  Y-axis rotation: Left/Right turn",
        "  X-axis rotation: Up/Down tilt",
        "",
        "Perspective Projection Applied"
    ]
    for j, txt in enumerate(info_texts):
        put_text_with_outline(vis_info, txt, (10, 40 + j * 22), scale=0.4)
    put_text_with_outline(vis_info, "4. Rotation Info", (10, 30), scale=0.5)

    combined = np.hstack([vis_region, vis_depth_pts, vis_axis, vis_info])
    save_path = os.path.join(save_dir, "depth_02_component_depths.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print(f"[3DRotate] Saved: {save_path}")


def vis_depth_03_rotated_depths(orig_depth_colored, rotated_depth_map, sam_mask, rotated_mask, save_dir):
    """【depth_03】旋转后的深度图（3栏：原始深度、旋转后深度、旋转后mask）"""
    H, W = orig_depth_colored.shape[:2]

    # 第一栏: 原始深度
    vis_orig = orig_depth_colored.copy()
    if sam_mask is not None:
        vis_orig[sam_mask < 128] = 0
    put_text_with_outline(vis_orig, "1. Original Depth", (10, 30), scale=0.5)

    # 第二栏: 旋转后深度
    vis_rotated = create_depth_colormap(rotated_depth_map)
    put_text_with_outline(vis_rotated, "2. Rotated Depth", (10, 30), scale=0.5)

    # 第三栏: 旋转后mask
    vis_mask = np.zeros((H, W, 3), dtype=np.uint8)
    vis_mask[rotated_mask > 127] = [0, 255, 0]
    put_text_with_outline(vis_mask, "3. Rotated Mask", (10, 30), scale=0.5)

    combined = np.hstack([vis_orig, vis_rotated, vis_mask])
    save_path = os.path.join(save_dir, "depth_03_rotated_depths.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print(f"[3DRotate] Saved: {save_path}")


def vis_depth_04_rgb_projection(source_image_np, sam_mask, rotated_rgb, final_composed, save_dir):
    """【depth_04】RGB空间旋转投影（3栏：原始RGB、旋转后RGB、合成结果）"""
    H, W = source_image_np.shape[:2]

    # 第一栏: 原始RGB（主体区域）
    vis_orig = source_image_np.copy()
    if sam_mask is not None:
        vis_orig[sam_mask < 128] = 0
    put_text_with_outline(vis_orig, "1. Original RGB", (10, 30), scale=0.5)

    # 第二栏: 旋转后RGB
    vis_rotated = rotated_rgb.copy()
    put_text_with_outline(vis_rotated, "2. Rotated RGB", (10, 30), scale=0.5)

    # 第三栏: 合成结果
    vis_composed = final_composed.copy()
    put_text_with_outline(vis_composed, "3. Final Composed", (10, 30), scale=0.5)

    combined = np.hstack([vis_orig, vis_rotated, vis_composed])
    stem = "depth_04_rgb_projection"
    save_path = os.path.join(save_dir, f"{stem}.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    # 额外保存原始 panel，供 paper 导出直接复用，避免后续对拼图二次裁切。
    panels = [vis_orig, vis_rotated, vis_composed]
    for idx, panel_rgb in enumerate(panels, start=1):
        panel_path = os.path.join(save_dir, f"{stem}__panel_{idx:02d}.jpg")
        cv2.imwrite(panel_path, cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2BGR))
    print(f"[3DRotate] Saved: {save_path}")


# ==========================================
# 主处理函数（集成到drag_processor流程）
# ==========================================

def process_3d_rotate_component(
    latents,
    source_image_np,
    sam_mask,
    handle_points_yx,
    target_points_yx,
    depth_map,
    device,
    component_id=0
):
    """
    对单个连通域进行3D旋转处理

    参数:
        latents: torch.Tensor (1, C, H_lat, W_lat)
        source_image_np: 源图像 (H, W, 3)
        sam_mask: SAM分割的主体mask (H, W)，0-255
        handle_points_yx: 控制点 [[y, x], ...]
        target_points_yx: 目标点 [[y, x], ...]
        depth_map: 深度图 (H, W)
        device: 计算设备
        component_id: 连通域ID

    返回:
        warped_latents, warped_mask, warped_rgb, debug_info
    """
    H_img, W_img = source_image_np.shape[:2]
    H_lat, W_lat = latents.shape[2], latents.shape[3]
    scale_x = W_lat / W_img
    scale_y = H_lat / H_img

    debug_info = {'component_id': component_id}

    # 【新增】深度过滤：基于3D空间连通性去除背景
    save_dir = ensure_debug_dir()
    if len(handle_points_yx) > 0:
        filtered_mask, filter_info = filter_background_by_depth(
            sam_mask, depth_map, handle_points_yx,
            save_dir=save_dir
        )
        debug_info['depth_filter_info'] = filter_info
        # 使用过滤后的mask
        sam_mask = filtered_mask

    # 1. 计算过滤后的mask信息
    mask_binary = (sam_mask > 127).astype(np.uint8)
    ys, xs = np.where(mask_binary > 0)

    if len(xs) == 0:
        # 没有有效像素
        mask_lat = cv2.resize(sam_mask.astype(np.float32) / 255.0, (W_lat, H_lat),
                              interpolation=cv2.INTER_NEAREST)
        debug_info['sam_mask'] = sam_mask
        debug_info['depth_map'] = depth_map
        debug_info['rotated_mask'] = sam_mask
        debug_info['rotated_rgb'] = source_image_np.copy()
        debug_info['m_end_full'] = sam_mask.astype(np.float32) / 255.0
        debug_info['rotated_handle_points_xy'] = np.zeros((0, 2), dtype=np.float32)
        debug_info['rotated_target_points_xy'] = np.zeros((0, 2), dtype=np.float32)
        return latents.clone(), torch.from_numpy(mask_lat).to(device), source_image_np.copy(), debug_info

    # 2. 【改进】计算3D质心（在过滤后的mask上）
    depth_values = depth_map[ys, xs]
    depth_min = depth_values.min()
    depth_max = depth_values.max()
    depth_range = depth_max - depth_min + 1e-6

    # 2D质心
    cx_2d = int(np.mean(xs))
    cy_2d = int(np.mean(ys))

    # 3D质心的深度分量（使用中位数更鲁棒）
    cz_depth = np.median(depth_values)

    # 计算XY范围和z_scale（与rotate_3d_with_depth保持一致）
    x_min, x_max = xs.min() - cx_2d, xs.max() - cx_2d
    y_min, y_max = -(ys.max() - cy_2d), -(ys.min() - cy_2d)
    xy_range = max(x_max - x_min, y_max - y_min)

    # 根据深度分布动态估算z_scale
    depth_std = np.std(depth_values)
    depth_mean = np.mean(depth_values)
    depth_cv = depth_std / depth_mean if depth_mean > 1e-6 else 0

    if depth_cv < 0.05:
        z_scale_factor = 0.4
    elif depth_cv < 0.10:
        z_scale_factor = 0.7
    elif depth_cv < 0.20:
        z_scale_factor = 1.0
    else:
        z_scale_factor = 1.5

    z_scale = xy_range * z_scale_factor
    z_centroid = z_scale / 2.0

    # 3D质心的z坐标（归一化后）
    cz_3d = (cz_depth - depth_min) / depth_range * z_scale - z_centroid

    # 使用2D质心作为旋转中心（cx, cy），但记录完整的3D质心信息
    cx, cy = cx_2d, cy_2d
    debug_info['centroid'] = (cx, cy)
    debug_info['centroid_3d'] = (cx_2d, cy_2d, cz_3d)
    debug_info['centroid_depth'] = cz_depth

    print(f"[3DRotate] 3D Centroid: 2D=({cx_2d}, {cy_2d}), depth={cz_depth:.3f}, z_3d={cz_3d:.2f}")

    # 3. 使用3D方法计算旋转轴和角度
    if len(handle_points_yx) > 0:
        h_pt = handle_points_yx[0]
        t_pt = target_points_yx[0]
        handle_xy = [h_pt[1], h_pt[0]]  # 转换为 [x, y]
        target_xy = [t_pt[1], t_pt[0]]

        rotation_axis, rotation_angle, angle_debug_info = compute_rotation_axis_and_angle_3d(
            handle_pt_xy=handle_xy,
            target_pt_xy=target_xy,
            centroid_xy=[cx, cy],
            depth_map=depth_map,
            z_scale=z_scale,
            depth_min=depth_min,
            depth_range=depth_range,
            z_centroid=z_centroid
        )
        debug_info['angle_computation'] = angle_debug_info
    else:
        rotation_axis, rotation_angle = 'Y', 0.0

    angle_debug_info = debug_info.get('angle_computation', {})
    rotation_yaw_deg = float(angle_debug_info.get('yaw_deg', rotation_angle if rotation_axis == 'Y' else 0.0))
    rotation_pitch_deg = float(angle_debug_info.get('pitch_deg', rotation_angle if rotation_axis == 'X' else 0.0))

    debug_info['rotation_axis'] = rotation_axis
    debug_info['rotation_angle_deg'] = rotation_angle
    debug_info['rotation_yaw_deg'] = rotation_yaw_deg
    debug_info['rotation_pitch_deg'] = rotation_pitch_deg

    # 记录控制点/目标点的旋转后2D投影，供后续Hybrid非刚性阶段对齐坐标系
    debug_info['rotated_handle_points_xy'] = project_points_after_3d_rotation(
        handle_points_yx, depth_map, (cx, cy), rotation_axis, rotation_angle,
        depth_min, depth_range, z_scale, z_centroid,
        yaw_deg=rotation_yaw_deg, pitch_deg=rotation_pitch_deg
    )
    debug_info['rotated_target_points_xy'] = project_points_after_3d_rotation(
        target_points_yx, depth_map, (cx, cy), rotation_axis, rotation_angle,
        depth_min, depth_range, z_scale, z_centroid,
        yaw_deg=rotation_yaw_deg, pitch_deg=rotation_pitch_deg
    )

    print(f"[3DRotate] Component {component_id}: centroid=({cx}, {cy}), "
          f"axis={rotation_axis}, angle={rotation_angle:.2f}°, "
          f"yaw={rotation_yaw_deg:.2f}°, pitch={rotation_pitch_deg:.2f}°")

    # 3. 执行3D旋转
    # 准备控制点和目标点的xy坐标
    handle_xy = None
    target_xy = None
    if len(handle_points_yx) > 0:
        h_pt = handle_points_yx[0]
        handle_xy = [h_pt[1], h_pt[0]]  # [y,x] -> [x,y]
    if len(target_points_yx) > 0:
        t_pt = target_points_yx[0]
        target_xy = [t_pt[1], t_pt[0]]  # [y,x] -> [x,y]

    rotated_rgb, rotated_depth, rotated_mask = rotate_3d_with_depth(
        source_image_np=source_image_np,
        mask=sam_mask,
        depth_map=depth_map,
        centroid_xy=(cx, cy),
        rotation_axis=rotation_axis,
        rotation_angle_deg=rotation_angle,
        device=device,
        save_dir=save_dir,
        handle_point_xy=handle_xy,
        target_point_xy=target_xy,
        yaw_deg=rotation_yaw_deg,
        pitch_deg=rotation_pitch_deg
    )

    debug_info['rotated_rgb'] = rotated_rgb
    debug_info['rotated_depth'] = rotated_depth
    debug_info['rotated_mask'] = rotated_mask
    debug_info['depth_map'] = depth_map
    debug_info['sam_mask'] = sam_mask

    # 4. 构建 latent 空间的变形网格
    # 获取原始和旋转后的mask像素位置
    orig_ys, orig_xs = np.where(sam_mask > 127)
    new_ys, new_xs = np.where(rotated_mask > 127)

    if len(orig_xs) == 0 or len(new_xs) == 0:
        # 没有有效像素，返回原始数据
        mask_lat = cv2.resize(sam_mask.astype(np.float32) / 255.0, (W_lat, H_lat),
                              interpolation=cv2.INTER_NEAREST)
        # 【修复】确保 debug_info 包含可视化所需的字段
        debug_info['sam_mask'] = sam_mask
        debug_info['depth_map'] = depth_map
        debug_info['m_end_full'] = rotated_mask.astype(np.float32) / 255.0
        return latents.clone(), torch.from_numpy(mask_lat).to(device), rotated_rgb, debug_info

    # 创建采样网格
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H_lat, device=device),
        torch.arange(W_lat, device=device),
        indexing='ij'
    )
    sample_x = grid_x.float()
    sample_y = grid_y.float()

    # 将旋转映射应用到latent空间
    # 需要建立从旋转后位置到原始位置的反向映射
    mask_lat = cv2.resize(sam_mask.astype(np.float32) / 255.0, (W_lat, H_lat),
                          interpolation=cv2.INTER_NEAREST)
    rotated_mask_lat_np = cv2.resize(rotated_mask.astype(np.float32) / 255.0, (W_lat, H_lat),
                                      interpolation=cv2.INTER_NEAREST)
    rotated_mask_lat_t = torch.from_numpy(rotated_mask_lat_np).to(device).float()

    # 由于3D旋转的复杂性，这里使用简化的方法：
    # 对latent空间应用相同的3D旋转变换
    lat_cx = int(cx * scale_x)
    lat_cy = int(cy * scale_y)

    # 获取latent空间中mask内的像素
    lat_mask_binary = mask_lat > 0.5
    lat_ys, lat_xs = np.where(lat_mask_binary)

    if len(lat_xs) > 0:
        # 构建3D点云（latent空间）
        lat_depth = cv2.resize(depth_map, (W_lat, H_lat), interpolation=cv2.INTER_LINEAR)
        lat_depth_values = lat_depth[lat_ys, lat_xs]

        depth_min = lat_depth_values.min()
        depth_max = lat_depth_values.max()
        depth_range = depth_max - depth_min + 1e-6

        # 【修复】使用与RGB空间一致的动态z_scale计算逻辑
        lat_x_min, lat_x_max = lat_xs.min() - lat_cx, lat_xs.max() - lat_cx
        lat_y_min, lat_y_max = -(lat_ys.max() - lat_cy), -(lat_ys.min() - lat_cy)
        lat_xy_range = max(lat_x_max - lat_x_min, lat_y_max - lat_y_min)

        lat_depth_std = np.std(lat_depth_values)
        lat_depth_mean = np.mean(lat_depth_values)
        lat_depth_cv = lat_depth_std / lat_depth_mean if lat_depth_mean > 1e-6 else 0

        if lat_depth_cv < 0.05:
            lat_z_scale_factor = 0.4
        elif lat_depth_cv < 0.10:
            lat_z_scale_factor = 0.7
        elif lat_depth_cv < 0.20:
            lat_z_scale_factor = 1.0
        else:
            lat_z_scale_factor = 1.5

        depth_scale = lat_xy_range * lat_z_scale_factor

        zs = (lat_depth_values - depth_min) / depth_range * depth_scale

        # 【修复】使用深度范围中点，与RGB空间一致
        z_centroid = depth_scale / 2.0

        # 转换为相对坐标（与 rotate_3d_with_depth 保持一致）
        # X = pixel_x - center_x
        # Y = -(pixel_y - center_y)  <- Y轴取反
        xs_centered = lat_xs - lat_cx
        ys_centered = -(lat_ys - lat_cy)  # Y轴取反，与RGB空间保持一致
        zs_centered = zs - z_centroid

        points_3d = np.stack([xs_centered, ys_centered, zs_centered], axis=1)

        # 应用旋转（与 RGB 空间一致，使用 Yaw+Pitch 联合旋转）
        R = resolve_rotation_matrix(
            rotation_axis=rotation_axis,
            rotation_angle_deg=rotation_angle,
            yaw_deg=rotation_yaw_deg,
            pitch_deg=rotation_pitch_deg
        )
        rotated_points = (R @ points_3d.T).T

        # 逆变换回像素坐标
        # pixel_x = X + center_x
        # pixel_y = -Y + center_y  <- Y轴再取反
        new_lat_xs = rotated_points[:, 0] + lat_cx
        new_lat_ys = -rotated_points[:, 1] + lat_cy  # Y轴取反恢复
        new_lat_zs = rotated_points[:, 2] + z_centroid

        # 建立反向映射（从新位置找原位置）
        z_buffer = torch.full((H_lat, W_lat), -np.inf, device=device)
        source_map_y = sample_y.clone()
        source_map_x = sample_x.clone()

        depth_order = np.argsort(new_lat_zs)
        for idx in depth_order:
            new_x = int(round(new_lat_xs[idx]))
            new_y = int(round(new_lat_ys[idx]))
            if 0 <= new_x < W_lat and 0 <= new_y < H_lat:
                if new_lat_zs[idx] > z_buffer[new_y, new_x].item():
                    z_buffer[new_y, new_x] = new_lat_zs[idx]
                    source_map_y[new_y, new_x] = lat_ys[idx]
                    source_map_x[new_y, new_x] = lat_xs[idx]

        # 更新采样网格 + 使用 z-buffer 生成 latent mask（与 RGB 遮挡一致）
        valid_mask = z_buffer > -np.inf
        sample_y[valid_mask] = source_map_y[valid_mask]
        sample_x[valid_mask] = source_map_x[valid_mask]

        rotated_mask_lat_t = valid_mask.float()
        rotated_mask_lat_np = rotated_mask_lat_t.detach().cpu().numpy()

    # 归一化到 [-1, 1]
    norm_grid = torch.stack([
        2.0 * sample_x / (W_lat - 1) - 1.0,
        2.0 * sample_y / (H_lat - 1) - 1.0
    ], dim=-1).unsqueeze(0)

    # 应用 grid_sample
    warped_latents = F.grid_sample(
        latents.float(),
        norm_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ).to(latents.dtype)

    # === Latent 内部空洞填充 ===
    try:
        # 基于主体范围自动识别空洞，并限制只用主体特征补全
        scale = min(H_lat, W_lat) / max(H_img, W_img)
        from utils_drag.drag_processor import fill_subject_holes_within_scope

        inpaint_radius = max(1, int(round(5 * scale)))
        warped_latents, refined_scope_lat, scope_info = fill_subject_holes_within_scope(
            image_input=warped_latents,
            subject_mask=rotated_mask_lat_np,
            device=device,
            radius=inpaint_radius,
            close_ratio=0.14,
            max_close=max(7, int(round(0.35 * min(H_lat, W_lat)))),
            max_fill_dist_ratio=0.20,
        )
        if isinstance(warped_latents, torch.Tensor):
            warped_latents = warped_latents.to(latents.dtype)
        hole_n = int(scope_info.get("hole_pixels", 0))
        if hole_n > 0:
            print(f"[3DRotate] Latent subject-scope holes filled: {hole_n} pixels")
            rotated_mask_lat_np = np.maximum(rotated_mask_lat_np, refined_scope_lat).astype(np.float32)
            rotated_mask_lat_t = torch.from_numpy(rotated_mask_lat_np).to(device).float()
    except Exception as e:
        print(f"[3DRotate] Latent hole fill failed: {e}")

    # 变形 mask
    warped_mask = rotated_mask_lat_t

    debug_info['norm_grid'] = norm_grid.squeeze(0)
    debug_info['m_end_full'] = rotated_mask.astype(np.float32) / 255.0

    return warped_latents, warped_mask, rotated_rgb, debug_info


def process_3d_rotate(
    latents,
    source_image_np,
    user_mask,
    sam_mask,
    handle_points_yx,
    target_points_yx,
    device,
    background_latents=None,
    bg_filled_rgb=None
):
    """
    3D旋转处理主入口

    参数:
        latents: 潜空间特征
        source_image_np: 源图像
        user_mask: 用户绘制的mask (0-255)
        sam_mask: SAM分割的主体mask (0-255)
        handle_points_yx: 控制点 [[y, x], ...]
        target_points_yx: 目标点 [[y, x], ...]
        device: 计算设备
        background_latents: 已填补的背景latents
        bg_filled_rgb: 已填补的背景RGB
    """
    save_dir = ensure_debug_dir()

    H_img, W_img = source_image_np.shape[:2]
    H_lat, W_lat = latents.shape[2], latents.shape[3]

    print(f"[3DRotate] Processing image: {W_img}x{H_img}, latent: {W_lat}x{H_lat}")

    # 1. 估计深度图
    print("[3DRotate] Estimating depth...")
    depth_map = estimate_depth(source_image_np, device=device)
    depth_colored = create_depth_colormap(depth_map)

    # 2. 准备mask
    m_start_full = (user_mask > 127).astype(np.float32)
    m_pseudo_full = (sam_mask > 127).astype(np.float32) if sam_mask is not None else m_start_full

    # 3. 处理3D旋转
    warped_latents, warped_mask, rotated_rgb, debug_info = process_3d_rotate_component(
        latents=latents,
        source_image_np=source_image_np,
        sam_mask=sam_mask if sam_mask is not None else user_mask,
        handle_points_yx=handle_points_yx,
        target_points_yx=target_points_yx,
        depth_map=depth_map,
        device=device,
        component_id=1
    )

    # 4. 合成到背景
    m_end_full = debug_info.get('m_end_full', m_start_full)
    rotated_mask_img = debug_info.get('rotated_mask', user_mask)

    # 【修复】确保背景是干净的（原始主体已被移除）
    if bg_filled_rgb is None:
        # 如果没有提供填补的背景，需要先移除原始主体区域
        # 使用 inpaint 填补原始主体区域
        from utils_drag.drag_processor import fill_background_holes
        # 使用 SAM mask 作为需要填补的区域
        inpaint_mask = sam_mask if sam_mask is not None else user_mask
        bg_filled_rgb = fill_background_holes(
            source_image_np.copy(),
            inpaint_mask,
            device=device,
            radius=5
        )

    # 【修复】合成逻辑：
    # 1. 首先确保背景中原始主体区域已被清除（使用原始mask）
    # 2. 然后将旋转后的主体贴到干净的背景上（使用旋转后的mask）

    # 创建合成用的mask
    m_start_binary = (sam_mask > 127 if sam_mask is not None else user_mask > 127).astype(np.float32)
    m_end_binary = m_end_full

    # 合成：背景 + 旋转后的主体
    # 在原始主体区域使用填补的背景，在旋转后主体区域使用旋转后的RGB
    mask_3ch = np.stack([m_end_binary] * 3, axis=2)
    final_composed_rgb = (rotated_rgb * mask_3ch + bg_filled_rgb * (1 - mask_3ch)).astype(np.uint8)

    # Latent合成
    if background_latents is not None:
        result_latents = background_latents.clone()
    else:
        result_latents = latents.clone()

    mask_binary = (warped_mask > 0.5).float()
    mask_4d = mask_binary.unsqueeze(0).unsqueeze(0).expand_as(result_latents)
    result_latents = result_latents * (1 - mask_4d) + warped_latents * mask_4d

    # 5. 生成可视化（可选）
    if ENABLE_3D_DEBUG and save_dir is not None:
        print("[3DRotate] Generating visualizations...")

        centroid = debug_info.get('centroid', (W_img // 2, H_img // 2))
        rotation_axis = debug_info.get('rotation_axis', 'Y')
        rotation_angle = debug_info.get('rotation_angle_deg', 0)
        rotation_yaw = debug_info.get('rotation_yaw_deg', None)
        rotation_pitch = debug_info.get('rotation_pitch_deg', None)

        # 4张主图
        vis_3drotate_01_fill_scopes(source_image_np, m_start_full, m_pseudo_full, bg_filled_rgb, save_dir)
        vis_3drotate_02_masks_cutouts(source_image_np, sam_mask, m_start_full,
                                       handle_points_yx, target_points_yx, save_dir)
        vis_3drotate_03_trace_compare(source_image_np, final_composed_rgb, m_start_full, m_end_full,
                                       handle_points_yx, target_points_yx, save_dir)
        vis_3drotate_04_final_preview(source_image_np, final_composed_rgb, bg_filled_rgb,
                                       handle_points_yx, target_points_yx, save_dir)

        # 4张深度图
        vis_depth_01_full_depth(source_image_np, depth_map, depth_colored, save_dir)
        vis_depth_02_component_depths(source_image_np, sam_mask, depth_map, centroid,
                                       rotation_axis, rotation_angle, save_dir,
                                       handle_points=handle_points_yx, target_points=target_points_yx,
                                       yaw_deg=rotation_yaw, pitch_deg=rotation_pitch)

        orig_depth_colored = depth_colored.copy()
        rotated_depth = debug_info.get('rotated_depth', depth_map)
        vis_depth_03_rotated_depths(orig_depth_colored, rotated_depth, sam_mask, rotated_mask_img, save_dir)
        vis_depth_04_rgb_projection(source_image_np, sam_mask, rotated_rgb, final_composed_rgb, save_dir)

        print(f"[3DRotate] Processing complete. Debug images saved to {save_dir}")
        print(f"  - 4 main images: 3DRotate_01 ~ 3DRotate_04")
        print(f"  - 4 depth images: depth_01 ~ depth_04")

    return result_latents, warped_mask, final_composed_rgb, debug_info
