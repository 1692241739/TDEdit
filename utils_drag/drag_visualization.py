"""Visualization helpers extracted from drag_processor."""

import glob
import os
import re
import shutil

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils_drag.drag_processor import (
    VIS_ARROW_COLOR,
    VIS_ANCHOR_COLOR,
    VIS_FONT_SCALE,
    VIS_HANDLE_COLOR,
    VIS_PIVOT_COLOR,
    VIS_TARGET_COLOR,
    _clean_binary_mask_u8,
    _coord_scale_latent_to_image,
    _latent_mask_to_fullres,
    _mask_to_binary_uint8,
    _mask_to_fullres_vis_float,
    _paper_no_text_single_enabled,
    _sanitize_xy_points,
    split_drag_mode,
)

__all__ = [
    "_build_final_composite_rgb",
    "_collect_subject_bnni_hole_mask_from_debug",
    "run_all_visualizations",
]


def put_text_with_outline(
    img,
    text,
    pos,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    scale=0.6,
    color=(255, 255, 255),
    thickness=2,
    outline_color=(0, 0, 0),
):
    if _paper_no_text_single_enabled():
        return img
    if not isinstance(img, np.ndarray):
        return img

    draw_img = img
    if (not draw_img.flags.c_contiguous) or (not draw_img.flags.writeable):
        draw_img = np.ascontiguousarray(draw_img)

    try:
        cv2.putText(draw_img, text, pos, font, scale, outline_color, thickness + 3, cv2.LINE_AA)
        cv2.putText(draw_img, text, pos, font, scale, color, thickness, cv2.LINE_AA)
    except cv2.error:
        return img

    if draw_img is not img:
        try:
            np.copyto(img, draw_img, casting="unsafe")
            return img
        except Exception:
            return draw_img
    return img


def draw_guides_on_image(img_bgr, handles, targets, px, py, title, anchors=None, action=None):
    handles_arr = _sanitize_xy_points(handles)
    targets_arr = _sanitize_xy_points(targets)
    anchors_arr = _sanitize_xy_points(anchors)

    pair_n = min(handles_arr.shape[0], targets_arr.shape[0])
    for i in range(pair_n):
        hx, hy = int(handles_arr[i][0]), int(handles_arr[i][1])
        tx, ty = int(targets_arr[i][0]), int(targets_arr[i][1])
        cv2.arrowedLine(img_bgr, (hx, hy), (tx, ty), VIS_ARROW_COLOR, 2, tipLength=0.2)
        cv2.circle(img_bgr, (hx, hy), 6, VIS_HANDLE_COLOR, -1)
        cv2.circle(img_bgr, (tx, ty), 6, VIS_TARGET_COLOR, -1)

    if px is not None and py is not None:
        cv2.circle(img_bgr, (px, py), 12, VIS_PIVOT_COLOR, 2)
        cv2.circle(img_bgr, (px, py), 6, VIS_PIVOT_COLOR, -1)
        put_text_with_outline(img_bgr, "P", (px + 10, py - 10), scale=VIS_FONT_SCALE, color=VIS_PIVOT_COLOR)

    if anchors_arr.shape[0] > 0:
        for apt in anchors_arr:
            cv2.circle(img_bgr, (int(apt[0]), int(apt[1])), 4, VIS_ANCHOR_COLOR, -1)

    put_text_with_outline(img_bgr, title, (10, 30), scale=VIS_FONT_SCALE)


def _resolve_hole_overlay_mode(hole_visual_mode):
    """
    统一 hole 可视化模式。
    当前策略：不再因为 paper/sgf 强制全红，优先按 inside/outside 分区显示。
    """
    mode_vis = str(hole_visual_mode or "").strip().lower()
    forced_by_paper = False
    use_all_red = False
    return mode_vis, use_all_red, forced_by_paper


def _build_ordered_visualization_list(save_dir, canonical_mode):
    """根据模式构建可视化文件顺序（返回[(src_name, normalized_name)]）。"""
    ordered = []
    added_norm = set()

    def add_map(items):
        """
        items: [(normalized_name, [candidate_src_name...]), ...]
        """
        for norm_name, candidates in items:
            if norm_name in added_norm:
                continue
            for src_name in candidates:
                if os.path.isfile(os.path.join(save_dir, src_name)):
                    ordered.append((src_name, norm_name))
                    added_norm.add(norm_name)
                    break

    def add_pattern(pattern):
        for p in sorted(glob.glob(os.path.join(save_dir, pattern))):
            src_name = os.path.basename(p)
            if src_name in added_norm:
                continue
            ordered.append((src_name, src_name))
            added_norm.add(src_name)

    if canonical_mode == "2D-Rigid":
        add_map([
            ("rigid_01_masks_cutouts.jpg", ["rigid_01_masks_cutouts.jpg", "rigid_02_masks_cutouts.jpg"]),
            ("rigid_02_trace_compare.jpg", ["rigid_02_trace_compare.jpg", "rigid_03_trace_compare.jpg"]),
            ("rigid_03_hole_masks.jpg", ["rigid_03_hole_masks.jpg"]),
            ("rigid_04_fill_process.jpg", ["rigid_04_fill_process.jpg"]),
            ("rigid_05_fill_scopes.jpg", ["rigid_05_fill_scopes.jpg", "rigid_01_fill_scopes.jpg"]),
            ("rigid_06_final_preview.jpg", ["rigid_06_final_preview.jpg", "rigid_04_final_preview.jpg"]),
            ("rigid_07_component_story.jpg", ["rigid_07_component_story.jpg", "rigid_05_component_story.jpg"]),
            ("rigid_08_final_composite.jpg", ["rigid_08_final_composite.jpg", "rigid_06_final_composite.jpg"]),
            ("rigid_09_background_chapter.jpg", ["rigid_09_background_chapter.jpg", "rigid_background_chapter.jpg"]),
        ])
        add_pattern("rigid_comp*_intent_story.jpg")
    elif canonical_mode == "2D-Non-Rigid":
        add_map([
            ("nonrigid_01_masks_cutouts.jpg", ["nonrigid_01_masks_cutouts.jpg", "nonrigid_02_masks_cutouts.jpg"]),
            ("nonrigid_02_trace_compare.jpg", ["nonrigid_02_trace_compare.jpg", "nonrigid_03_trace_compare.jpg"]),
            ("nonrigid_03_hole_masks.jpg", ["nonrigid_03_hole_masks.jpg"]),
            ("nonrigid_04_fill_process.jpg", ["nonrigid_04_fill_process.jpg"]),
            ("nonrigid_05_fill_scopes.jpg", ["nonrigid_05_fill_scopes.jpg", "nonrigid_01_fill_scopes.jpg"]),
            ("nonrigid_06_final_preview.jpg", ["nonrigid_06_final_preview.jpg", "nonrigid_04_final_preview.jpg"]),
            ("nonrigid_07_component_story.jpg", ["nonrigid_07_component_story.jpg", "nonrigid_05_component_story.jpg"]),
            ("nonrigid_08_final_composite.jpg", ["nonrigid_08_final_composite.jpg", "nonrigid_06_final_composite.jpg"]),
            ("nonrigid_09_background_chapter.jpg", ["nonrigid_09_background_chapter.jpg", "nonrigid_background_chapter.jpg"]),
        ])
        add_pattern("nonrigid_comp*_intent_story.jpg")
    elif canonical_mode == "2D-Hybrid":
        add_map([
            ("rigid_01_masks_cutouts.jpg", ["rigid_01_masks_cutouts.jpg", "rigid_02_masks_cutouts.jpg"]),
            ("rigid_02_trace_compare.jpg", ["rigid_02_trace_compare.jpg", "rigid_03_trace_compare.jpg"]),
            ("rigid_05_fill_scopes.jpg", ["rigid_05_fill_scopes.jpg", "rigid_01_fill_scopes.jpg"]),
            ("rigid_06_final_preview.jpg", ["rigid_06_final_preview.jpg", "rigid_04_final_preview.jpg"]),
            ("nonrigid_01_masks_cutouts.jpg", ["nonrigid_01_masks_cutouts.jpg", "nonrigid_02_masks_cutouts.jpg"]),
            ("nonrigid_02_trace_compare.jpg", ["nonrigid_02_trace_compare.jpg", "nonrigid_03_trace_compare.jpg"]),
            ("nonrigid_05_fill_scopes.jpg", ["nonrigid_05_fill_scopes.jpg", "nonrigid_01_fill_scopes.jpg"]),
            ("nonrigid_06_final_preview.jpg", ["nonrigid_06_final_preview.jpg", "nonrigid_04_final_preview.jpg"]),
            ("nonrigid_07_component_story.jpg", ["nonrigid_07_component_story.jpg", "nonrigid_05_component_story.jpg"]),
            ("nonrigid_08_final_composite.jpg", ["nonrigid_08_final_composite.jpg", "nonrigid_06_final_composite.jpg"]),
            ("nonrigid_09_background_chapter.jpg", ["nonrigid_09_background_chapter.jpg", "nonrigid_background_chapter.jpg"]),
        ])
        add_pattern("nonrigid_comp*_intent_story.jpg")
    elif canonical_mode == "3D-Rigid":
        add_map([
            ("3DRigid_01_masks_cutouts.jpg", ["3DRigid_01_masks_cutouts.jpg"]),
            ("3DRigid_02_trace_compare.jpg", ["3DRigid_02_trace_compare.jpg"]),
            ("3DRigid_03_hole_masks.jpg", ["3DRigid_03_hole_masks.jpg"]),
            ("3DRigid_04_fill_process.jpg", ["3DRigid_04_fill_process.jpg"]),
            ("3DRigid_05_fill_scopes.jpg", ["3DRigid_05_fill_scopes.jpg"]),
            ("3DRigid_06_final_preview.jpg", ["3DRigid_06_final_preview.jpg"]),
            ("depth_00_filter_3d_comparison.png", ["depth_00_filter_3d_comparison.png"]),
            ("depth_01_full_depth.jpg", ["depth_01_full_depth.jpg"]),
            ("depth_02_component_depths.jpg", ["depth_02_component_depths.jpg"]),
            ("depth_03_rotated_depths.jpg", ["depth_03_rotated_depths.jpg"]),
            ("depth_04_rgb_projection.jpg", ["depth_04_rgb_projection.jpg"]),
        ])
        add_pattern("depth_00_filter_3d_comparison__view_*.png")
        add_pattern("depth_05_3d_point_cloud*.png")
    elif canonical_mode == "3D-Non-Rigid":
        add_map([
            ("3DNonRigid_01_masks_cutouts.jpg", ["3DNonRigid_01_masks_cutouts.jpg"]),
            ("3DNonRigid_02_hole_masks.jpg", ["3DNonRigid_02_hole_masks.jpg", "3DNonRigid_03_hole_masks.jpg"]),
            ("3DNonRigid_03_fill_process.jpg", ["3DNonRigid_03_fill_process.jpg", "3DNonRigid_04_fill_process.jpg"]),
            ("3DNonRigid_04_trace_compare.jpg", ["3DNonRigid_04_trace_compare.jpg", "3DNonRigid_02_trace_compare.jpg"]),
            ("3DNonRigid_05_fill_scopes.jpg", ["3DNonRigid_05_fill_scopes.jpg"]),
            ("3DNonRigid_06_final_preview.jpg", ["3DNonRigid_06_final_preview.jpg"]),
            ("depth_00_filter_3d_comparison.png", ["depth_00_filter_3d_comparison.png"]),
            ("depth_01_full_depth.jpg", ["depth_01_full_depth.jpg"]),
            ("depth_02_component_depths.jpg", ["depth_02_component_depths.jpg"]),
            ("depth_03_rotated_depths.jpg", ["depth_03_rotated_depths.jpg"]),
            ("depth_04_rgb_projection.jpg", ["depth_04_rgb_projection.jpg"]),
        ])
        add_pattern("depth_00_filter_3d_comparison__view_*.png")
        add_pattern("depth_05_3d_point_cloud_nonrigid_comp*.png")
        add_pattern("depth_05_3d_point_cloud_nonrigid_compare_comp*.png")
    elif canonical_mode == "3D-Hybrid":
        add_map([
            ("nonrigid_01_masks_cutouts.jpg", ["nonrigid_01_masks_cutouts.jpg", "nonrigid_02_masks_cutouts.jpg"]),
            ("nonrigid_02_trace_compare.jpg", ["nonrigid_02_trace_compare.jpg", "nonrigid_03_trace_compare.jpg"]),
            ("nonrigid_05_fill_scopes.jpg", ["nonrigid_05_fill_scopes.jpg", "nonrigid_01_fill_scopes.jpg"]),
            ("nonrigid_06_final_preview.jpg", ["nonrigid_06_final_preview.jpg", "nonrigid_04_final_preview.jpg"]),
            ("depth_00_filter_3d_comparison.png", ["depth_00_filter_3d_comparison.png"]),
            ("nonrigid_08_final_composite.jpg", ["nonrigid_08_final_composite.jpg", "nonrigid_06_final_composite.jpg"]),
            ("nonrigid_09_background_chapter.jpg", ["nonrigid_09_background_chapter.jpg", "nonrigid_background_chapter.jpg"]),
        ])
        add_pattern("depth_00_filter_3d_comparison__view_*.png")
        # 3D-Hybrid：优先展示老版两阶段，再展示三行总览图。
        add_pattern("depth_05_3d_point_cloud_phase1_comp*.png")
        add_pattern("depth_06_3d_point_cloud_phase2_comp*.png")
        add_pattern("depth_05_3d_point_cloud_hybrid_comp*.png")

    return ordered


def _create_ordered_visualization_aliases(save_dir, canonical_mode):
    """
    在调试目录生成顺序化文件：
    01_xxx, 02_xxx, ...
    不删除文件夹，但会清理重复的非编号可视化文件，只保留编号结果。
    """
    if save_dir is None or not os.path.isdir(save_dir):
        return

    ordered = _build_ordered_visualization_list(save_dir, canonical_mode)
    if len(ordered) == 0:
        return

    def _is_vis_file(name):
        return name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))

    stale_patterns = ["step_*", "[0-9][0-9]_*"]
    for pat in stale_patterns:
        for old_alias in glob.glob(os.path.join(save_dir, pat)):
            if os.path.isfile(old_alias):
                try:
                    os.remove(old_alias)
                except Exception:
                    pass

    keep_aliases = []
    def _renumber_norm_name(name, order_idx):
        base, ext = os.path.splitext(str(name))
        # 深度图命名保留原语义编号（depth_00/01/...）
        if base.startswith("depth_"):
            return str(name)
        m = re.match(r"^(.*_)(\d{2})(_.*)$", base)
        if m is None:
            return str(name)
        return f"{m.group(1)}{int(order_idx):02d}{m.group(3)}{ext}"

    for idx, item in enumerate(ordered, start=1):
        if isinstance(item, tuple) and len(item) >= 2:
            src_name, norm_name = item[0], item[1]
        else:
            src_name = str(item)
            norm_name = src_name
        src_path = os.path.join(save_dir, src_name)
        if not os.path.isfile(src_path):
            continue
        norm_name_renum = _renumber_norm_name(norm_name, idx)
        alias_name = f"{idx:02d}_{norm_name_renum}"
        alias_path = os.path.join(save_dir, alias_name)
        try:
            shutil.copy2(src_path, alias_path)
            keep_aliases.append(alias_name)
            # 仅保留编号文件，原始文件在生成编号后删除，避免重复。
            if os.path.abspath(src_path) != os.path.abspath(alias_path) and os.path.isfile(src_path):
                try:
                    os.remove(src_path)
                except Exception:
                    pass
        except Exception as e:
            print(f"[Vis Alias] copy failed: {src_name} -> {alias_name}, err={e}")

    # 清理剩余非编号可视化文件（保留编号文件）
    for name in os.listdir(save_dir):
        path = os.path.join(save_dir, name)
        if (not os.path.isfile(path)) or (not _is_vis_file(name)):
            continue
        if "__panel_" in name.lower():
            continue
        if re.match(r"^\d{2}_", name):
            continue
        try:
            os.remove(path)
        except Exception:
            pass

    # 多连通域时的 depth_comp 子目录也补顺序名，避免人工查看时来回猜。
    for comp_dir in sorted(glob.glob(os.path.join(save_dir, "depth_comp*"))):
        if not os.path.isdir(comp_dir):
            continue
        for pat in stale_patterns:
            for old_alias in glob.glob(os.path.join(comp_dir, pat)):
                if os.path.isfile(old_alias):
                    try:
                        os.remove(old_alias)
                    except Exception:
                        pass
        comp_seq = [
            "depth_02_component_depths.jpg",
            "depth_03_rotated_depths.jpg",
            "depth_04_rgb_projection.jpg",
        ]
        comp_keep = []
        for cidx, src_name in enumerate(comp_seq, start=1):
            src_path = os.path.join(comp_dir, src_name)
            if not os.path.isfile(src_path):
                continue
            alias_name = f"{cidx:02d}_{src_name}"
            alias_path = os.path.join(comp_dir, alias_name)
            try:
                shutil.copy2(src_path, alias_path)
                comp_keep.append(alias_name)
                if os.path.abspath(src_path) != os.path.abspath(alias_path) and os.path.isfile(src_path):
                    try:
                        os.remove(src_path)
                    except Exception:
                        pass
            except Exception:
                pass
        for name in os.listdir(comp_dir):
            path = os.path.join(comp_dir, name)
            if (not os.path.isfile(path)) or (not _is_vis_file(name)):
                continue
            if "__panel_" in name.lower():
                continue
            if re.match(r"^\d{2}_", name):
                continue
            try:
                os.remove(path)
            except Exception:
                pass


def vis_01_fill_scopes(
    source_image_np,
    m_start_full_overall,
    m_pseudo_full_overall,
    bg_filled_rgb,
    save_dir,
    prefix="",
    m_end_full_overall=None,
    bg_hole_mask=None,
    hole_inside_mask=None,
    final_composed_rgb=None,
    all_results=None,
    device=None,
    hole_visual_mode=None,
):
    """
    【01】Fill Scopes (五栏): 原图+黑主体 -> 背景补全 -> 红色空洞 -> 叠加主体(不补内部) -> 内部空洞绿色
    """
    try:
        if source_image_np is None:
            return
        H, W = source_image_np.shape[:2]

        # 统一到图像分辨率（保留软边），避免 latent 掩码直接阈值导致坑洼。
        start_mask = _mask_to_fullres_vis_float(m_start_full_overall, target_hw=(H, W))
        if m_end_full_overall is None:
            end_mask = start_mask.copy()
        else:
            end_mask = _mask_to_fullres_vis_float(m_end_full_overall, target_hw=(H, W))

        # 与 trace_compare 对齐：优先使用组件渲染链路重建的 end mask（image 模式更平整）。
        end_mask_vis = _build_vis_end_mask_from_components(source_image_np, all_results, device)
        if isinstance(end_mask_vis, np.ndarray):
            end_mask = _mask_to_fullres_vis_float(end_mask_vis, target_hw=(H, W))

        hole_total = (
            np.logical_and(start_mask > 0.5, end_mask <= 0.5).astype(np.float32)
            if bg_hole_mask is None
            else _mask_to_fullres_vis_float(bg_hole_mask, target_hw=(H, W))
        )
        hole_total_u8 = _mask_to_binary_uint8(hole_total, target_hw=(H, W))
        hole_in_u8 = _mask_to_binary_uint8(hole_inside_mask, target_hw=(H, W))
        hole_in_u8 = np.logical_and(hole_in_u8 > 0, hole_total_u8 > 0).astype(np.uint8)
        subject_u8 = _mask_to_binary_uint8(end_mask, target_hw=(H, W))
        hole_out_u8 = np.logical_and(hole_total_u8 > 0, hole_in_u8 <= 0).astype(np.uint8)

        # Step1: 原图 + 透黑主体mask
        vis_step1 = _overlay_mask_tint_rgb(source_image_np, start_mask, color_rgb=[0, 0, 0], alpha=0.60)
        put_text_with_outline(vis_step1, "1) Source + Subject Mask (Black)", (10, 30), scale=0.5)

        # Step2: 背景补全（基底=背景补全图）
        vis_step2 = (
            np.clip(bg_filled_rgb, 0, 255).astype(np.uint8)
            if isinstance(bg_filled_rgb, np.ndarray)
            else source_image_np.copy()
        )
        put_text_with_outline(vis_step2, "2) Background Filled", (10, 30), scale=0.5)

        # Step3: 在背景图上叠加透红空洞mask
        vis_step3_bgr = _render_hole_overlay_panel(
            base_rgb=vis_step2,
            hole_in_mask=np.zeros_like(hole_total_u8, dtype=np.float32),
            hole_out_mask=hole_total_u8.astype(np.float32),
            title="3) Background + Red Hole Mask",
            blacken_hole=False,
        )
        vis_step3 = cv2.cvtColor(vis_step3_bgr, cv2.COLOR_BGR2RGB)

        # Step4: 在 Step3 上叠加“无内部补全”主体（内部洞保留红色）
        vis_step4 = vis_step3.copy()
        comp_rgb = None
        if isinstance(final_composed_rgb, np.ndarray):
            comp_rgb = np.clip(final_composed_rgb, 0, 255).astype(np.uint8)
        elif isinstance(all_results, dict):
            comp_built = _build_final_composite_rgb(
                source_image_np=source_image_np,
                all_results=all_results,
                bg_filled_rgb=vis_step2,
                device=device,
            )
            if isinstance(comp_built, np.ndarray):
                comp_rgb = np.clip(comp_built, 0, 255).astype(np.uint8)
        if isinstance(comp_rgb, np.ndarray):
            paste_mask = np.logical_and(subject_u8 > 0, hole_in_u8 <= 0)
            vis_step4[paste_mask] = comp_rgb[paste_mask]
        put_text_with_outline(vis_step4, "4) Overlay Subject (No Inner Fill)", (10, 30), scale=0.5)

        # Step5: 主体内部空洞透绿（其余空洞保持红色）
        vis_step5_bgr = _render_hole_overlay_panel(
            base_rgb=vis_step4,
            hole_in_mask=hole_in_u8.astype(np.float32),
            hole_out_mask=np.zeros_like(hole_out_u8, dtype=np.float32),
            title="5) Inner Holes in Green",
            blacken_hole=False,
        )
        vis_step5 = cv2.cvtColor(vis_step5_bgr, cv2.COLOR_BGR2RGB)

        # 合并并保存（五步流程）
        vis = np.hstack([vis_step1, vis_step2, vis_step3, vis_step4, vis_step5])
        stem = f"{prefix}_01_fill_scopes" if prefix else "01_fill_scopes"
        fname = f"{stem}.jpg"
        cv2.imwrite(os.path.join(save_dir, fname), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        # 额外保存原始 panel，供 paper 导出直接复用，避免后续对拼图二次裁切。
        panel_images = [vis_step1, vis_step2, vis_step3, vis_step4, vis_step5]
        for idx, panel_rgb in enumerate(panel_images, start=1):
            panel_name = f"{stem}__panel_{idx:02d}.jpg"
            cv2.imwrite(os.path.join(save_dir, panel_name), cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"[Vis Error] 01_fill_scopes: {e}")

def vis_02_masks_cutouts(source_image_np, all_results, m_start_full_overall, all_sam_pts, all_targets_xy, save_dir, device, prefix=""):
    """
    【02】Masks + Cutouts + Deformation (三栏 x N行)
    每行: Mask (纯mask) | Before Cutout (带标注) | After Deformation (带标注)
    最后一行: 合并Mask | 合并Before | 合并After
    """
    try:
        num_components = len(all_results['masks_start'])
        vis_rows = []
        H, W = source_image_np.shape[:2]

        for comp_idx in range(num_components):
            comp_data = _build_component_warp_visual_data(source_image_np, all_results, comp_idx, device)
            handles = comp_data['handles']
            targets = comp_data['targets']
            pv_x = comp_data['pv_x']
            pv_y = comp_data['pv_y']
            valid_anchors = comp_data['anchors']
            action = comp_data['action']
            sub_action = comp_data['sub_action']
            m_u8 = comp_data['m_start_u8']
            m_end_hires = comp_data['m_end_u8']
            warped_rgb_np = comp_data['warped_rgb_np']

            display_action = sub_action if sub_action else action

            # 左栏: Mask
            col1 = cv2.cvtColor(m_u8, cv2.COLOR_GRAY2BGR)
            put_text_with_outline(col1, f"Comp{comp_idx+1} Mask", (10, 30), scale=0.5)
            
            # 中栏: Before Cutout
            cutout = source_image_np.copy().astype(np.float32)
            cutout[m_u8 < 127] *= 0.3
            col2 = cv2.cvtColor(cutout.astype(np.uint8), cv2.COLOR_RGB2BGR)
            draw_guides_on_image(col2, handles, targets, pv_x, pv_y, f"Before [{display_action}]", valid_anchors, action)
            
            # 右栏: After Deformation
            cutout_after = warped_rgb_np.copy().astype(np.float32)
            cutout_after[m_end_hires < 127] *= 0.3
            col3 = cv2.cvtColor(cutout_after.astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            put_text_with_outline(col3, f"After [{display_action}]", (10, 30), scale=0.5, color=(0, 255, 0))
            vis_rows.append(np.hstack([col1, col2, col3]))
        
        # ==========================================
        # 最后一行: 合并结果
        # ==========================================
        m_overall_u8 = (m_start_full_overall * 255).astype(np.uint8)
        
        # 左: 合并Mask (无标注)
        col1 = cv2.cvtColor(m_overall_u8, cv2.COLOR_GRAY2BGR)
        put_text_with_outline(col1, "Merged Mask", (10, 30), scale=0.5)
        
        # 中: 合并Before Cutout (带标注)
        cutout_merged = source_image_np.copy().astype(np.float32)
        cutout_merged[m_overall_u8 < 127] *= 0.3
        col2 = cv2.cvtColor(cutout_merged.astype(np.uint8), cv2.COLOR_RGB2BGR)
        draw_guides_on_image(col2, all_sam_pts, all_targets_xy, None, None, "Merged Before", None, None)
        
        # 右: 合并After Deformation
        merged_after = source_image_np.copy().astype(np.float32)
        merged_mask_after = np.zeros((H, W), dtype=np.float32)
        
        for comp_idx in range(num_components):
            comp_data = _build_component_warp_visual_data(source_image_np, all_results, comp_idx, device)
            warped_rgb_np = comp_data['warped_rgb_np'].astype(np.float32)
            m_end_np = (comp_data['m_end_u8'].astype(np.float32) / 255.0)

            mask_3ch = np.stack([m_end_np] * 3, axis=2)
            merged_after = warped_rgb_np * mask_3ch + merged_after * (1 - mask_3ch)
            merged_mask_after = np.maximum(merged_mask_after, m_end_np)
        
        merged_after[merged_mask_after < 0.5] *= 0.3
        col3 = cv2.cvtColor(merged_after.astype(np.uint8), cv2.COLOR_RGB2BGR)
        put_text_with_outline(col3, "Merged After", (10, 30), scale=0.5, color=(0, 255, 0))
        
        vis_rows.append(np.hstack([col1, col2, col3]))
        
        # 保存
        fname = f"{prefix}_02_masks_cutouts.jpg" if prefix else "02_masks_cutouts.jpg"
        cv2.imwrite(os.path.join(save_dir, fname), np.vstack(vis_rows))
        
    except Exception as e:
        print(f"[Vis Error] 02_masks_cutouts: {e}")
        import traceback
        traceback.print_exc()

def _render_component_with_norm_grid(source_image_np, m_start_full, norm_grid, device, render_domain="image"):
    """
    统一可视化渲染逻辑：
    - 输入 source 图 + 起始 mask + latent norm_grid
    - 输出 image 分辨率的 warped RGB 与 warped mask
    """
    H_img, W_img = source_image_np.shape[:2]
    domain = str(render_domain or "image").strip().lower()

    # 可视化统一走图像分辨率渲染，避免 latent 分支“先降采样再放大”造成发糊。
    m_start = _mask_to_fullres_vis_float(m_start_full, target_hw=(H_img, W_img))
    m_start_u8 = np.clip(m_start * 255.0, 0, 255).astype(np.uint8)

    if norm_grid is None:
        return source_image_np.copy().astype(np.uint8), m_start_u8

    img_tensor = torch.from_numpy(source_image_np).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    if torch.is_tensor(norm_grid):
        ng = norm_grid.detach().to(device=device, dtype=torch.float32)
    else:
        ng = torch.from_numpy(np.asarray(norm_grid, dtype=np.float32)).to(device=device)
    if ng.ndim == 4:
        ng = ng[0]
    if ng.ndim != 3 or ng.shape[-1] != 2:
        return source_image_np.copy().astype(np.uint8), m_start_u8
    grid_tensor = ng.permute(2, 0, 1).unsqueeze(0)
    grid_up = F.interpolate(grid_tensor, size=(H_img, W_img), mode='bilinear', align_corners=True)
    grid_up = grid_up.permute(0, 2, 3, 1)

    warped_rgb = F.grid_sample(img_tensor, grid_up, mode='bilinear', padding_mode='border', align_corners=True)
    warped_rgb_np = np.clip(
        warped_rgb.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255.0,
        0,
        255,
    ).astype(np.uint8)

    # latent 模式保留硬边形态；image 模式保留软边。
    mask_mode = 'nearest' if domain == "latent" else 'bilinear'
    m_start_t = torch.from_numpy(m_start).float().to(device).view(1, 1, H_img, W_img)
    m_end_t = F.grid_sample(m_start_t, grid_up, mode=mask_mode, align_corners=True)
    m_end_u8 = np.clip(m_end_t.squeeze().detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    return warped_rgb_np, m_end_u8

def _build_component_warp_visual_data(source_image_np, all_results, comp_idx, device):
    """
    为单个连通域构建可视化所需数据：
    - m_start_u8 / m_end_u8
    - warped_rgb_np
    - handles/targets/pivot/anchors/action/sub_action
    """
    H_img, W_img = source_image_np.shape[:2]

    handles = all_results['handles'][comp_idx]
    targets = all_results['targets'][comp_idx]
    pivot = all_results['pivots'][comp_idx]
    anchors = all_results['anchors'][comp_idx]
    action = all_results['actions'][comp_idx]
    sub_action = all_results.get('sub_actions', [None] * len(all_results['actions']))[comp_idx]
    debug_info = all_results.get('debug_infos', [{}] * len(all_results['actions']))[comp_idx]
    m_start_raw = all_results['masks_start'][comp_idx]
    norm_grid = all_results['grids'][comp_idx]

    runtime_geom_domain = ""
    if isinstance(debug_info, dict):
        runtime_geom_domain = str(debug_info.get('runtime_geom_domain', '')).strip().lower()

    m_start_vis_src = m_start_raw
    if runtime_geom_domain != "image" and isinstance(debug_info, dict) and ("_runtime_mask_start_vis" in debug_info):
        vis_src = debug_info.get("_runtime_mask_start_vis")
        if isinstance(vis_src, np.ndarray) and vis_src.shape[:2] == (H_img, W_img):
            m_start_vis_src = vis_src

    m_start = _mask_to_fullres_vis_float(m_start_vis_src, target_hw=(H_img, W_img))
    m_start_u8 = np.clip(m_start * 255.0, 0, 255).astype(np.uint8)

    gh, gw = None, None
    if torch.is_tensor(norm_grid):
        gh, gw = int(norm_grid.shape[0]), int(norm_grid.shape[1])
    elif isinstance(norm_grid, np.ndarray) and norm_grid.ndim >= 2:
        gh, gw = int(norm_grid.shape[0]), int(norm_grid.shape[1])
    elif torch.is_tensor(all_results.get('masks_end', [None])[comp_idx]):
        me = all_results['masks_end'][comp_idx]
        if me.ndim >= 2:
            gh, gw = int(me.shape[-2]), int(me.shape[-1])

    sx = _coord_scale_latent_to_image(gw, W_img) if (gw is not None) else 1.0
    sy = _coord_scale_latent_to_image(gh, H_img) if (gh is not None) else 1.0
    need_scale = (gw is not None and gh is not None and (W_img != gw or H_img != gh))
    dist_to_fg = None
    fg_u8 = (m_start_u8 > 127).astype(np.uint8)
    if int(np.count_nonzero(fg_u8)) > 0:
        outside = (fg_u8 == 0).astype(np.uint8)
        dist_to_fg = cv2.distanceTransform(outside, cv2.DIST_L2, 5)

    def _point_fit_score(arr_xy):
        if (dist_to_fg is None) or (arr_xy is None) or (arr_xy.shape[0] == 0):
            return np.inf
        xs = np.clip(np.round(arr_xy[:, 0]).astype(np.int32), 0, W_img - 1)
        ys = np.clip(np.round(arr_xy[:, 1]).astype(np.int32), 0, H_img - 1)
        return float(np.mean(dist_to_fg[ys, xs]))

    points_already_image_space = bool(
        isinstance(debug_info, dict) and debug_info.get("_points_already_image_space", False)
    )

    def _normalize_points_to_source(points):
        arr = _sanitize_xy_points(points)
        if arr.shape[0] == 0:
            return arr
        # 明确标记为 image 坐标时，不再做坐标空间猜测，避免“左上角小坐标”误判为 latent 坐标。
        if points_already_image_space:
            return arr
        if not need_scale:
            return arr

        arr_scaled = arr.astype(np.float32, copy=True)
        arr_scaled[:, 0] *= sx
        arr_scaled[:, 1] *= sy

        max_x = float(np.max(arr[:, 0]))
        max_y = float(np.max(arr[:, 1]))
        min_x = float(np.min(arr[:, 0]))
        min_y = float(np.min(arr[:, 1]))
        likely_grid_space = (
            (gw is not None) and (gh is not None) and
            min_x >= -1.5 and min_y >= -1.5 and
            max_x <= gw + 1.5 and max_y <= gh + 1.5
        )
        likely_image_space = (
            min_x >= -1.5 and min_y >= -1.5 and
            max_x <= W_img + 1.5 and max_y <= H_img + 1.5 and
            ((gw is None) or (max_x > gw + 1.5) or (max_y > gh + 1.5))
        )

        if runtime_geom_domain == "latent":
            # latent 域下可能已提前被映射到 image（避免二次缩放导致锚点飞出画布）
            if likely_grid_space and not likely_image_space:
                return arr_scaled
            if likely_image_space and not likely_grid_space:
                return arr
            # 不确定时退回贴合度选择
            raw_score = _point_fit_score(arr)
            scaled_score = _point_fit_score(arr_scaled)
            if np.isfinite(raw_score) and np.isfinite(scaled_score):
                return arr_scaled if scaled_score < raw_score else arr
            return arr_scaled

        raw_score = _point_fit_score(arr)
        scaled_score = _point_fit_score(arr_scaled)
        if runtime_geom_domain == "image":
            # image 域一般无需缩放；仅当缩放后明显更贴合主体时才纠正。
            if np.isfinite(raw_score) and np.isfinite(scaled_score) and (scaled_score + 1.0 < raw_score):
                return arr_scaled
            return arr

        # 域未知：优先选择与主体 mask 更贴合的一组坐标。
        if np.isfinite(raw_score) and np.isfinite(scaled_score):
            return arr_scaled if scaled_score < raw_score else arr
        return arr_scaled

    handles = _normalize_points_to_source(handles)
    targets = _normalize_points_to_source(targets)

    valid_anchors = None
    if anchors is not None:
        anchors_arr = _normalize_points_to_source(anchors)
        if anchors_arr.ndim == 2 and anchors_arr.shape[0] > 0 and anchors_arr.shape[1] >= 2:
            valid_anchors = anchors_arr.astype(np.float32)

    pv_x, pv_y = None, None
    if pivot is not None:
        pivot_arr = _normalize_points_to_source(pivot)
        if pivot_arr.shape[0] > 0:
            pv_x = int(round(float(pivot_arr[0, 0])))
            pv_y = int(round(float(pivot_arr[0, 1])))

    # 默认可视化走 norm_grid 重建（2D 分支更一致）。
    warped_rgb_np, m_end_u8 = _render_component_with_norm_grid(
        source_image_np=source_image_np,
        m_start_full=m_start_vis_src,
        norm_grid=norm_grid,
        device=device,
        render_domain=runtime_geom_domain,
    )

    if isinstance(debug_info, dict):
        # 3D projective 场景优先使用显式 rotated_rgb/m_end_full，避免 identity 区混入旧位置。
        prefer_projective_visual = str(debug_info.get("warp_backend", "")).strip().upper() == "PROJECTIVE_ZBUFFER_3D"

        m_end_full = debug_info.get("m_end_full")
        if m_end_full is not None and (prefer_projective_visual or norm_grid is None):
            m_end_full = _latent_mask_to_fullres(m_end_full, target_hw=(H_img, W_img))
            m_end_u8 = (np.clip(m_end_full, 0.0, 1.0) * 255).astype(np.uint8)

        rotated_rgb = debug_info.get("rotated_rgb")
        if isinstance(rotated_rgb, np.ndarray) and (prefer_projective_visual or norm_grid is None):
            if rotated_rgb.shape[:2] != (H_img, W_img):
                rotated_rgb = cv2.resize(rotated_rgb, (W_img, H_img), interpolation=cv2.INTER_LINEAR)
            warped_rgb_np = np.clip(rotated_rgb, 0, 255).astype(np.uint8)
    else:
        if norm_grid is None:
            warped_rgb_np = source_image_np.copy().astype(np.uint8)
            m_end_u8 = m_start_u8.copy()

    return {
        'handles': handles,
        'targets': targets,
        'pv_x': pv_x,
        'pv_y': pv_y,
        'anchors': valid_anchors,
        'action': action,
        'sub_action': sub_action,
        'm_start_u8': m_start_u8,
        'm_end_u8': m_end_u8,
        'warped_rgb_np': warped_rgb_np
    }

def vis_05_component_intent_story(source_image_np, all_results, bg_filled_rgb, save_dir, device, prefix=""):
    """
    【05】每连通域三联图（单图）
    顺序：意图识别(带锚点) | 颜色变化 | 变形后效果
    背景单独输出为一张独立章节图。
    同时输出总汇总图（按连通域纵向拼接）。
    """
    try:
        num_components = len(all_results['latents'])
        H, W = source_image_np.shape[:2]
        all_rows = []

        # 背景单独保存（独立章节）
        bg_only = cv2.cvtColor(bg_filled_rgb.copy(), cv2.COLOR_RGB2BGR)
        put_text_with_outline(bg_only, "BG Filled", (10, 30), scale=0.6)
        bg_name = f"{prefix}_background_chapter.jpg" if prefix else "background_chapter.jpg"
        cv2.imwrite(os.path.join(save_dir, bg_name), bg_only)

        for comp_idx in range(num_components):
            comp_data = _build_component_warp_visual_data(source_image_np, all_results, comp_idx, device)
            handles = comp_data['handles']
            targets = comp_data['targets']
            pv_x = comp_data['pv_x']
            pv_y = comp_data['pv_y']
            valid_anchors = comp_data['anchors']
            action = comp_data['action']
            sub_action = comp_data['sub_action']
            m_start_u8 = comp_data['m_start_u8']
            m_end_u8 = comp_data['m_end_u8']
            warped_rgb_np = comp_data['warped_rgb_np']

            display_action = sub_action if sub_action else action

            # 1) 意图识别（nonrigid_02 中间栏风格，带锚点）
            before_cutout = source_image_np.copy().astype(np.float32)
            before_cutout[m_start_u8 < 127] *= 0.3
            panel_intent = cv2.cvtColor(before_cutout.astype(np.uint8), cv2.COLOR_RGB2BGR)
            draw_guides_on_image(panel_intent, handles, targets, pv_x, pv_y, f"Intent: {display_action}", valid_anchors, action)

            # 2) 颜色变化（nonrigid_03 的带色变化）
            mask_diff_clean = np.zeros((H, W, 3), dtype=np.uint8)
            mask_diff_clean[m_start_u8 > 127, 2] = 255  # Start: Red
            mask_diff_clean[m_end_u8 > 127, 1] = 255    # End: Green
            panel_change = cv2.cvtColor((source_image_np.astype(np.float32) * 0.4).astype(np.uint8), cv2.COLOR_RGB2BGR)
            panel_change = cv2.addWeighted(panel_change, 1.0, mask_diff_clean, 0.6, 0)
            for i in range(len(handles)):
                hx, hy = int(handles[i][0]), int(handles[i][1])
                tx, ty = int(targets[i][0]), int(targets[i][1])
                cv2.arrowedLine(panel_change, (hx, hy), (tx, ty), VIS_ARROW_COLOR, 2, tipLength=0.15)
                cv2.circle(panel_change, (hx, hy), 5, VIS_HANDLE_COLOR, -1)
                cv2.circle(panel_change, (tx, ty), 5, VIS_TARGET_COLOR, -1)
            put_text_with_outline(panel_change, "Trace Change", (10, 30), scale=0.55)

            # 3) 变形后效果（nonrigid_02 第三栏风格）
            after_cutout = warped_rgb_np.copy().astype(np.float32)
            after_cutout[m_end_u8 < 127] *= 0.3
            panel_after = cv2.cvtColor(after_cutout.astype(np.uint8), cv2.COLOR_RGB2BGR)
            put_text_with_outline(panel_after, "After Deformation", (10, 30), scale=0.55, color=(0, 255, 0))

            row = np.hstack([panel_intent, panel_change, panel_after])
            all_rows.append(row)

            # 每连通域单独文件
            per_comp_name = f"{prefix}_comp{comp_idx + 1:02d}_intent_story.jpg" if prefix else f"comp{comp_idx + 1:02d}_intent_story.jpg"
            cv2.imwrite(os.path.join(save_dir, per_comp_name), row)

        if len(all_rows) > 0:
            summary = np.vstack(all_rows)
            fname = f"{prefix}_05_component_story.jpg" if prefix else "05_component_story.jpg"
            cv2.imwrite(os.path.join(save_dir, fname), summary)
    except Exception as e:
        print(f"[Vis Error] 05_component_intent_story: {e}")
        import traceback
        traceback.print_exc()


def _build_final_composite_rgb(source_image_np, all_results, bg_filled_rgb, device):
    """
    生成最终拼接图（RGB），供多处可视化复用。
    """
    if not isinstance(source_image_np, np.ndarray):
        return None
    if isinstance(bg_filled_rgb, np.ndarray):
        merged = bg_filled_rgb.copy().astype(np.float32)
    else:
        merged = source_image_np.copy().astype(np.float32)

    num_components = len(all_results.get("latents", []))
    for comp_idx in range(num_components):
        comp_data = _build_component_warp_visual_data(source_image_np, all_results, comp_idx, device)
        warped_rgb_np = comp_data["warped_rgb_np"].astype(np.float32)
        m_end_u8 = comp_data["m_end_u8"]
        mask_3ch = np.stack([m_end_u8 / 255.0] * 3, axis=2)
        merged = warped_rgb_np * mask_3ch + merged * (1 - mask_3ch)
    return np.clip(merged, 0, 255).astype(np.uint8)


def vis_06_final_composite(source_image_np, all_results, bg_filled_rgb, save_dir, device, prefix=""):
    """
    【06】总结果图：背景 + 全部连通域拼接后的最终图
    对齐 nonrigid_04_final_preview.jpg 最后一行第一栏的语义。
    """
    try:
        merged = _build_final_composite_rgb(
            source_image_np=source_image_np,
            all_results=all_results,
            bg_filled_rgb=bg_filled_rgb,
            device=device,
        )
        if merged is None:
            return
        out_bgr = cv2.cvtColor(merged, cv2.COLOR_RGB2BGR)
        if prefix != "nonrigid":
            put_text_with_outline(out_bgr, "Final Composite", (10, 30), scale=0.6, color=(0, 255, 0))

        fname = f"{prefix}_06_final_composite.jpg" if prefix else "06_final_composite.jpg"
        cv2.imwrite(os.path.join(save_dir, fname), out_bgr)
    except Exception as e:
        print(f"[Vis Error] 06_final_composite: {e}")
        import traceback
        traceback.print_exc()

def _build_vis_end_mask_from_components(source_image_np, all_results, device):
    """
    从组件可视化结果合成 end mask（与 vis_02/vis_05 同源），用于避免 trace_compare 的分辨率不一致。
    """
    if not isinstance(all_results, dict):
        return None
    if device is None or not isinstance(source_image_np, np.ndarray):
        return None

    H_img, W_img = source_image_np.shape[:2]
    comp_num = len(all_results.get("latents", []))
    if comp_num <= 0:
        return None

    merged = np.zeros((H_img, W_img), dtype=np.float32)
    valid_count = 0
    for comp_idx in range(comp_num):
        try:
            comp_data = _build_component_warp_visual_data(source_image_np, all_results, comp_idx, device)
            m_end_u8 = comp_data.get("m_end_u8")
            if not isinstance(m_end_u8, np.ndarray):
                continue
            if m_end_u8.shape[:2] != (H_img, W_img):
                m_end_u8 = cv2.resize(
                    m_end_u8.astype(np.uint8),
                    (W_img, H_img),
                    interpolation=cv2.INTER_LINEAR,
                )
            merged = np.maximum(merged, np.clip(m_end_u8.astype(np.float32) / 255.0, 0.0, 1.0))
            valid_count += 1
        except Exception:
            continue

    if valid_count <= 0:
        return None
    return merged.astype(np.float32)


def vis_03_trace_compare(
    source_image_np,
    m_start_full_overall,
    m_end_overall,
    all_sam_pts,
    all_targets_xy,
    save_dir,
    prefix="",
    all_results=None,
    device=None,
):
    """
    【03】Trace Compare (2x2布局)
    """
    try:
        H, W = source_image_np.shape[:2]
        m_start_vis = _mask_to_fullres_vis_float(m_start_full_overall, target_hw=(H, W))

        # 关键：优先使用组件可视化链路生成的 end mask，保持与 vis_02/vis_05 一致的分辨率与边缘形态。
        m_end_vis = _build_vis_end_mask_from_components(source_image_np, all_results, device)
        if not isinstance(m_end_vis, np.ndarray):
            m_end_vis = _mask_to_fullres_vis_float(m_end_overall, target_hw=(H, W))

        m_start_u8 = np.clip(m_start_vis * 255.0, 0, 255).astype(np.uint8)
        m_end_u8 = np.clip(m_end_vis * 255.0, 0, 255).astype(np.uint8)
        
        # 先创建纯净的 mask_diff（不带文字）
        mask_diff_clean = np.zeros((H, W, 3), dtype=np.uint8)
        mask_diff_clean[m_start_u8 > 127, 2] = 255
        mask_diff_clean[m_end_u8 > 127, 1] = 255
        
        # 左上: Mask差异
        mask_diff = mask_diff_clean.copy()
        put_text_with_outline(mask_diff, "Mask Diff (R=Start, G=End)", (10, 30), scale=VIS_FONT_SCALE)
        
        # 右上: Overlay + Trace
        bg_dim = (source_image_np.astype(np.float32) * 0.4).astype(np.uint8)
        overlay = cv2.cvtColor(bg_dim, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(overlay, 1.0, mask_diff_clean, 0.6, 0)
        
        # ✅ 使用全局颜色变量
        for i in range(len(all_sam_pts)):
            hx, hy = int(all_sam_pts[i][0]), int(all_sam_pts[i][1])
            tx, ty = int(all_targets_xy[i][0]), int(all_targets_xy[i][1])
            cv2.arrowedLine(overlay, (hx, hy), (tx, ty), VIS_ARROW_COLOR, 2, tipLength=0.15)
            cv2.circle(overlay, (hx, hy), 5, VIS_HANDLE_COLOR, -1)
            cv2.circle(overlay, (tx, ty), 5, VIS_TARGET_COLOR, -1)
        
        put_text_with_outline(overlay, "Trace Overlay", (10, 30), scale=VIS_FONT_SCALE)
        
        top_row = np.hstack([mask_diff, overlay])
        
        # 左下/右下
        cutout_before = source_image_np.copy()
        cutout_before[m_start_u8 < 127] = 0
        cutout_before = cv2.cvtColor(cutout_before, cv2.COLOR_RGB2BGR)
        put_text_with_outline(cutout_before, "Before Cutout", (10, 30), scale=VIS_FONT_SCALE)
        
        # After Cutout: 显示“拖拽后的主体纹理”，而不是 source 的剪影。
        cutout_after = np.zeros_like(source_image_np, dtype=np.float32)
        composed_mask = np.zeros((H, W), dtype=np.float32)
        if isinstance(all_results, dict) and device is not None:
            comp_num = len(all_results.get("latents", []))
            for comp_idx in range(comp_num):
                comp_data = _build_component_warp_visual_data(source_image_np, all_results, comp_idx, device)
                warped_rgb_np = comp_data.get("warped_rgb_np")
                m_comp_u8 = comp_data.get("m_end_u8")
                if not isinstance(warped_rgb_np, np.ndarray) or not isinstance(m_comp_u8, np.ndarray):
                    continue
                m_comp = (m_comp_u8.astype(np.float32) / 255.0)
                if m_comp.shape[:2] != (H, W):
                    m_comp = cv2.resize(m_comp, (W, H), interpolation=cv2.INTER_LINEAR)
                if warped_rgb_np.shape[:2] != (H, W):
                    warped_rgb_np = cv2.resize(warped_rgb_np, (W, H), interpolation=cv2.INTER_LINEAR)
                m3 = np.stack([np.clip(m_comp, 0.0, 1.0)] * 3, axis=2)
                cutout_after = warped_rgb_np.astype(np.float32) * m3 + cutout_after * (1.0 - m3)
                composed_mask = np.maximum(composed_mask, np.clip(m_comp, 0.0, 1.0))
        if int(np.sum(composed_mask > 0.5)) <= 0:
            fallback = source_image_np.copy().astype(np.float32)
            fallback[m_end_u8 < 127] = 0
            cutout_after = fallback
        else:
            cutout_after[composed_mask <= 0.5] = 0
        cutout_after = cv2.cvtColor(np.clip(cutout_after, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        put_text_with_outline(cutout_after, "After Cutout (Warped Subject)", (10, 30), scale=VIS_FONT_SCALE)
        
        bottom_row = np.hstack([cutout_before, cutout_after])
        
        final = np.vstack([top_row, bottom_row])
        fname = f"{prefix}_03_trace_compare.jpg" if prefix else "03_trace_compare.jpg"
        cv2.imwrite(os.path.join(save_dir, fname), final)
    except Exception as e:
        print(f"[Vis Error] 03_trace_compare: {e}")

def vis_04_final_preview(source_image_np, all_results, bg_filled_rgb, all_sam_pts, all_targets_xy, save_dir, device, prefix=""):
    """
    【04】Final Preview（无锚点/无控制点）
    仅展示每个连通域合成结果与最终合成结果，避免调试元素干扰质量判断。
    """
    try:
        num_components = len(all_results['latents'])
        vis_rows = []

        for comp_idx in range(num_components):
            comp_data = _build_component_warp_visual_data(source_image_np, all_results, comp_idx, device)
            action = comp_data['action']
            sub_action = comp_data['sub_action']
            m_end_np = comp_data['m_end_u8']
            warped_rgb_np = comp_data['warped_rgb_np']
            display_action = sub_action if sub_action else action

            mask_3ch = np.stack([m_end_np / 255.0] * 3, axis=2)
            comp_result = (warped_rgb_np * mask_3ch + bg_filled_rgb * (1 - mask_3ch)).astype(np.uint8)
            panel = cv2.cvtColor(comp_result.copy(), cv2.COLOR_RGB2BGR)
            put_text_with_outline(panel, f"Comp{comp_idx+1} [{display_action}]", (10, 30), scale=0.55)
            vis_rows.append(panel)

        merged = _build_final_composite_rgb(
            source_image_np=source_image_np,
            all_results=all_results,
            bg_filled_rgb=bg_filled_rgb,
            device=device,
        )
        if merged is None:
            return
        all_sub_actions = all_results.get('sub_actions', all_results['actions'])
        actions_str = ", ".join([s for s in all_sub_actions if s])
        final_panel = cv2.cvtColor(merged.copy(), cv2.COLOR_RGB2BGR)
        put_text_with_outline(final_panel, f"FINAL [{actions_str}]", (10, 30), scale=0.6, color=(0, 255, 0))
        vis_rows.append(final_panel)

        fname = f"{prefix}_04_final_preview.jpg" if prefix else "04_final_preview.jpg"
        cv2.imwrite(os.path.join(save_dir, fname), np.vstack(vis_rows))
    except Exception as e:
        print(f"[Vis Error] 04_final_preview: {e}")
        import traceback
        traceback.print_exc()


def _render_hole_overlay_panel(base_rgb, hole_in_mask, hole_out_mask, title="", blacken_hole=False):
    """
    空洞覆盖图：
    - 空洞底先压黑
    - 主体外空洞：红色半透明 + 白边
    - 主体内空洞：淡绿色半透明 + 白边
    """
    img = np.clip(base_rgb, 0, 255).astype(np.uint8).copy()
    H, W = img.shape[:2]
    min_area = 0
    h_in_u8 = _clean_binary_mask_u8(
        _mask_to_binary_uint8(hole_in_mask, target_hw=(H, W)),
        close_k=0,
        open_k=0,
        min_area=min_area,
    )
    h_out_u8 = _clean_binary_mask_u8(
        _mask_to_binary_uint8(hole_out_mask, target_hw=(H, W)),
        close_k=0,
        open_k=0,
        min_area=min_area,
    )
    h_in = h_in_u8 > 0
    h_out = h_out_u8 > 0
    h_any = h_in | h_out

    if bool(blacken_hole) and np.any(h_any):
        img[h_any] = 0

    # 软边 alpha，减少锯齿感；风格对齐 subject_scope 的半透明叠加
    alpha_out = cv2.GaussianBlur(h_out.astype(np.float32), (5, 5), 0)
    alpha_in = cv2.GaussianBlur(h_in.astype(np.float32), (5, 5), 0)
    alpha_out = np.clip(alpha_out, 0.0, 1.0)
    alpha_in = np.clip(alpha_in, 0.0, 1.0)

    if np.any(h_out):
        c_out = np.zeros_like(img, dtype=np.uint8)
        c_out[h_out] = [255, 60, 60]  # RGB, outer red
        img = (
            img.astype(np.float32) * (1.0 - 0.72 * alpha_out[..., None]) +
            c_out.astype(np.float32) * (0.72 * alpha_out[..., None])
        ).astype(np.uint8)
    if np.any(h_in):
        c_in = np.zeros_like(img, dtype=np.uint8)
        c_in[h_in] = [160, 255, 160]  # RGB, inner light green
        img = (
            img.astype(np.float32) * (1.0 - 0.72 * alpha_in[..., None]) +
            c_in.astype(np.float32) * (0.72 * alpha_in[..., None])
        ).astype(np.uint8)

    out_bgr = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if title:
        put_text_with_outline(out_bgr, title, (10, 30), scale=0.55)
    return out_bgr


def _overlay_mask_tint_rgb(base_rgb, mask, color_rgb, alpha=0.55):
    """在 RGB 图上叠加半透明颜色遮罩（带软边）。"""
    img = np.clip(base_rgb, 0, 255).astype(np.uint8).copy()
    H, W = img.shape[:2]
    m_u8 = _mask_to_binary_uint8(mask, target_hw=(H, W))
    m_bool = m_u8 > 0
    if not np.any(m_bool):
        return img
    a = cv2.GaussianBlur(m_bool.astype(np.float32), (5, 5), 0)
    a = np.clip(a, 0.0, 1.0) * float(alpha)
    c = np.zeros_like(img, dtype=np.uint8)
    c[m_bool] = np.array(color_rgb, dtype=np.uint8)
    out = (
        img.astype(np.float32) * (1.0 - a[..., None]) +
        c.astype(np.float32) * a[..., None]
    ).astype(np.uint8)
    return out


def _collect_subject_bnni_hole_mask_from_debug(all_results, target_hw):
    """
    汇总所有组件 debug_info 里的“主体内部空洞(补前)”mask。
    该 mask 对应主体补洞/BNNI 路径，而非背景 hole split。
    """
    h, w = int(target_hw[0]), int(target_hw[1])
    merged = np.zeros((h, w), dtype=np.float32)
    if not isinstance(all_results, dict):
        return merged
    for dbg in all_results.get("debug_infos", []):
        if not isinstance(dbg, dict):
            continue
        cand = dbg.get("subject_hole_mask_full")
        if isinstance(cand, np.ndarray):
            c_u8 = _mask_to_binary_uint8(cand, target_hw=(h, w))
            merged = np.maximum(merged, (c_u8 > 0).astype(np.float32))
        # Hybrid 分阶段 debug
        for k in ("phase1", "phase2", "phase1_3d", "phase2_3d_nonrigid", "phase2_nonrigid"):
            sub = dbg.get(k)
            if not isinstance(sub, dict):
                continue
            cand2 = sub.get("subject_hole_mask_full")
            if isinstance(cand2, np.ndarray):
                c_u8 = _mask_to_binary_uint8(cand2, target_hw=(h, w))
                merged = np.maximum(merged, (c_u8 > 0).astype(np.float32))
    return merged


def vis_03_hole_masks(source_image_np, all_results, bg_filled_rgb, fill_vis_debug, save_dir, device, prefix=""):
    """
    【03】空洞掩码流程图：透黑总空洞 -> 背景补全 -> 叠加主体 -> 外红内绿分区
    """
    try:
        if not isinstance(fill_vis_debug, dict):
            return
        final_rgb = _build_final_composite_rgb(
            source_image_np=source_image_np,
            all_results=all_results,
            bg_filled_rgb=bg_filled_rgb,
            device=device,
        )
        if not isinstance(final_rgb, np.ndarray):
            final_rgb = source_image_np

        mode_vis, use_all_red, forced_by_paper = _resolve_hole_overlay_mode(
            fill_vis_debug.get("hole_visual_mode", "")
        )
        hole_total = fill_vis_debug.get("hole_total_full")
        hole_total_u8 = _mask_to_binary_uint8(hole_total, target_hw=final_rgb.shape[:2])
        if use_all_red:
            hole_in_u8 = np.zeros_like(hole_total_u8, dtype=np.uint8)
            hole_out_u8 = (hole_total_u8 > 0).astype(np.uint8)
            if forced_by_paper and mode_vis != "sgf":
                panel4_title = "4) Paper: All Holes in Red"
            else:
                panel4_title = "4) SGF: All Holes in Red"
        else:
            hole_in_pref = fill_vis_debug.get("subject_hole_mask_full")
            if not isinstance(hole_in_pref, np.ndarray):
                hole_in_pref = _collect_subject_bnni_hole_mask_from_debug(all_results, final_rgb.shape[:2])
            if int(np.sum(hole_in_pref > 0.5)) <= 0:
                hole_in_pref = fill_vis_debug.get("hole_inside_full")
            hole_in_u8 = _mask_to_binary_uint8(hole_in_pref, target_hw=final_rgb.shape[:2])
            hole_in_u8 = np.logical_and(hole_in_u8 > 0, hole_total_u8 > 0).astype(np.uint8)
            hole_out_u8 = np.logical_and(hole_total_u8 > 0, hole_in_u8 <= 0).astype(np.uint8)
            panel4_title = "4) Outside Red / Inside Green"

        panel1 = _overlay_mask_tint_rgb(source_image_np, hole_total, color_rgb=[0, 0, 0], alpha=0.60)
        put_text_with_outline(panel1, "1) Full Hole Region (Translucent Black)", (10, 30), scale=0.55)

        panel2 = (
            np.clip(bg_filled_rgb, 0, 255).astype(np.uint8)
            if isinstance(bg_filled_rgb, np.ndarray)
            else source_image_np.copy()
        )
        put_text_with_outline(panel2, "2) Background Filled", (10, 30), scale=0.55)

        panel3 = final_rgb.copy()
        put_text_with_outline(panel3, "3) Overlay Dragged Subject", (10, 30), scale=0.55)

        panel4_bgr = _render_hole_overlay_panel(
            base_rgb=final_rgb,
            hole_in_mask=hole_in_u8.astype(np.float32),
            hole_out_mask=hole_out_u8.astype(np.float32),
            title=panel4_title,
            blacken_hole=False,
        )
        panel4 = cv2.cvtColor(panel4_bgr, cv2.COLOR_BGR2RGB)

        vis = np.hstack([panel1, panel2, panel3, panel4])
        fname = f"{prefix}_03_hole_masks.jpg" if prefix else "03_hole_masks.jpg"
        cv2.imwrite(os.path.join(save_dir, fname), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"[Vis Error] 03_hole_masks: {e}")


def vis_04_fill_process(source_image_np, fill_vis_debug, save_dir, prefix="", base_rgb=None, all_results=None):
    """
    【04】补全过程：每连通域四步（透黑总空洞 -> 背景补全 -> 叠加主体 -> 外红内绿分区）
    """
    try:
        if not isinstance(fill_vis_debug, dict):
            return
        comps = fill_vis_debug.get("components", [])
        if len(comps) == 0:
            return

        rows = []
        if isinstance(base_rgb, np.ndarray):
            base_panel_rgb = np.clip(base_rgb, 0, 255).astype(np.uint8)
        else:
            base_panel_rgb = np.clip(source_image_np, 0, 255).astype(np.uint8)

        hole_total_global_u8 = _mask_to_binary_uint8(fill_vis_debug.get("hole_total_full"), target_hw=base_panel_rgb.shape[:2])
        subject_start_global_u8 = _mask_to_binary_uint8(
            fill_vis_debug.get("subject_start_full"),
            target_hw=base_panel_rgb.shape[:2],
        )
        masks_start_list = []
        if isinstance(all_results, dict):
            masks_start_list = all_results.get("masks_start", []) or []
        mode_vis, use_all_red, forced_by_paper = _resolve_hole_overlay_mode(
            fill_vis_debug.get("hole_visual_mode", "")
        )
        for c in comps:
            comp_idx = int(c.get("index", 0))
            cid = comp_idx + 1

            comp_total_u8 = _mask_to_binary_uint8(c.get("hole_total_full"), target_hw=base_panel_rgb.shape[:2])
            if len(comps) == 1:
                comp_total_u8 = hole_total_global_u8.copy()

            hole_total_u8 = _mask_to_binary_uint8(comp_total_u8.astype(np.float32), target_hw=base_panel_rgb.shape[:2])
            if use_all_red:
                hole_in_u8 = np.zeros_like(hole_total_u8, dtype=np.uint8)
                hole_out_u8 = (hole_total_u8 > 0).astype(np.uint8)
                if forced_by_paper and mode_vis != "sgf":
                    p4_title = f"Comp{cid} 4) Paper: All Holes Red"
                else:
                    p4_title = f"Comp{cid} 4) SGF: All Holes Red"
            else:
                comp_in_raw = fill_vis_debug.get("subject_hole_mask_full")
                if (not isinstance(comp_in_raw, np.ndarray)) or int(np.sum(comp_in_raw > 0.5)) <= 0:
                    comp_in_raw = c.get("hole_in_full")
                hole_in_u8 = _mask_to_binary_uint8(comp_in_raw, target_hw=base_panel_rgb.shape[:2])
                hole_in_u8 = np.logical_and(hole_in_u8 > 0, hole_total_u8 > 0).astype(np.uint8)
                hole_out_u8 = np.logical_and(hole_total_u8 > 0, hole_in_u8 <= 0).astype(np.uint8)
                p4_title = f"Comp{cid} 4) Outside Red / Inside Green"

            bg_stage = c.get("before_rgb")
            if not isinstance(bg_stage, np.ndarray):
                bg_stage = source_image_np
            bg_stage = np.clip(bg_stage, 0, 255).astype(np.uint8)

            comp_final = np.clip(base_panel_rgb, 0, 255).astype(np.uint8)

            comp_start_u8 = None
            if comp_idx < len(masks_start_list):
                comp_start_full = _latent_mask_to_fullres(
                    masks_start_list[comp_idx],
                    target_hw=base_panel_rgb.shape[:2],
                )
                comp_start_u8 = _mask_to_binary_uint8(comp_start_full, target_hw=base_panel_rgb.shape[:2])
            if (comp_start_u8 is None) or (int(np.sum(comp_start_u8 > 0)) <= 0):
                comp_start_u8 = subject_start_global_u8.copy()
            if int(np.sum(comp_start_u8 > 0)) <= 0:
                comp_start_u8 = _mask_to_binary_uint8(
                    fill_vis_debug.get("subject_end_full"),
                    target_hw=base_panel_rgb.shape[:2],
                )
            if int(np.sum(comp_start_u8 > 0)) <= 0:
                comp_start_u8 = comp_total_u8.copy()

            p1 = _overlay_mask_tint_rgb(
                source_image_np,
                comp_start_u8.astype(np.float32),
                color_rgb=[0, 0, 0],
                alpha=0.60,
            )
            put_text_with_outline(p1, f"Comp{cid} 1) Subject Start Mask (Black)", (10, 30), scale=0.55)

            p2 = bg_stage.copy()
            put_text_with_outline(p2, f"Comp{cid} 2) Background Filled", (10, 30), scale=0.55)

            p3 = comp_final.copy()
            put_text_with_outline(p3, f"Comp{cid} 3) Overlay Dragged Subject", (10, 30), scale=0.55)

            p4_bgr = _render_hole_overlay_panel(
                base_rgb=comp_final,
                hole_in_mask=hole_in_u8.astype(np.float32),
                hole_out_mask=hole_out_u8.astype(np.float32),
                title=p4_title,
                blacken_hole=False,
            )
            p4 = cv2.cvtColor(p4_bgr, cv2.COLOR_BGR2RGB)

            rows.append(np.hstack([p1, p2, p3, p4]))

        vis = np.vstack(rows)
        fname = f"{prefix}_04_fill_process.jpg" if prefix else "04_fill_process.jpg"
        cv2.imwrite(os.path.join(save_dir, fname), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"[Vis Error] 04_fill_process: {e}")

def run_hybrid_visualizations(source_image_np, all_results, debug_info, 
                               m_pseudo_full_overall, m_start_full_overall,
                               all_sam_pts, all_targets_xy, bg_filled_rgb,
                               save_dir, device, fill_vis_debug=None):
    """
    Hybrid 模式专用可视化 - 生成 8 张图
    rigid_01~04: Phase 1 (刚性旋转) 的结果
    nonrigid_01~04: Phase 2 (非刚性变形) 的结果
    """
    print("\n[Hybrid Visualization] Generating 8 debug images...")
    
    H_img, W_img = source_image_np.shape[:2]
    num_components = len(all_results['debug_infos'])
    
    # ============================================
    # 提取 Phase 1 结果
    # ============================================
    phase1_results = {
        'latents': [],
        'masks_start': [],
        'masks_end': [],
        'grids': [],
        'handles': [],
        'targets': [],
        'pivots': [],
        'anchors': [],
        'actions': [],
        'sub_actions': []
    }
    
    # Phase 1 的 end mask（所有组件合并）
    m_end_p1_overall = np.zeros((H_img, W_img), dtype=np.float32)
    
    for comp_idx in range(num_components):
        comp_debug = all_results['debug_infos'][comp_idx]
        
        if 'phase1' in comp_debug:
            p1 = comp_debug['phase1']
            m_start_full = _latent_mask_to_fullres(
                all_results['masks_start'][comp_idx],
                target_hw=(H_img, W_img),
            )
            phase1_results['latents'].append(p1.get('warped_latents'))
            phase1_results['masks_start'].append(m_start_full)
            phase1_results['masks_end'].append(p1.get('mask_end'))
            phase1_results['grids'].append(p1.get('norm_grid'))
            phase1_results['handles'].append(p1.get('comp_sam_pts', all_results['handles'][comp_idx]))
            # Phase 1 的目标是旋转后的位置
            rotated_pts = p1.get('rotated_sam_pts')
            if rotated_pts is not None:
                phase1_results['targets'].append(rotated_pts)
            else:
                phase1_results['targets'].append(all_results['targets'][comp_idx])
            phase1_results['pivots'].append(p1.get('pivot_img'))
            phase1_results['anchors'].append(np.zeros((0, 2), dtype=np.float32))  # 刚性无锚点
            phase1_results['actions'].append('Rigid')
            phase1_results['sub_actions'].append(p1.get('sub_action', 'Rotation'))
            
            # 累积 Phase 1 的 end mask
            norm_grid = p1.get('norm_grid')
            if norm_grid is not None:
                grid_tensor = norm_grid.permute(2, 0, 1).unsqueeze(0)
                grid_up = F.interpolate(grid_tensor, size=(H_img, W_img), mode='bilinear', align_corners=True)
                grid_up = grid_up.permute(0, 2, 3, 1)
                m_start_t = torch.from_numpy(m_start_full).float().to(device).view(1, 1, H_img, W_img)
                m_end_t = F.grid_sample(m_start_t, grid_up, mode='nearest', align_corners=True)
                m_end_p1_overall = np.maximum(m_end_p1_overall, m_end_t.squeeze().cpu().numpy())
        else:
            # 非 Hybrid 组件，跳过
            pass
    
    # ============================================
    # 提取 Phase 2 结果
    # ============================================
    phase2_results = {
        'latents': [],
        'masks_start': [],
        'masks_end': [],
        'grids': [],
        'handles': [],
        'targets': [],
        'pivots': [],
        'anchors': [],
        'actions': [],
        'sub_actions': []
    }
    
    # Phase 2 使用 Phase 1 的输出作为起点
    m_start_p2_overall = m_end_p1_overall.copy()
    m_end_p2_overall = np.zeros((H_img, W_img), dtype=np.float32)
    
    for comp_idx in range(num_components):
        comp_debug = all_results['debug_infos'][comp_idx]
        
        if 'phase2' in comp_debug:
            p2 = comp_debug['phase2']
            m_start_full = _latent_mask_to_fullres(
                p2.get('m_start_full', m_end_p1_overall),
                target_hw=(H_img, W_img),
            )
            phase2_results['latents'].append(p2.get('warped_latents'))
            phase2_results['masks_start'].append(m_start_full)
            phase2_results['masks_end'].append(p2.get('mask_end'))
            phase2_results['grids'].append(p2.get('norm_grid'))
            # Phase 2 的起点是旋转后的位置
            phase2_results['handles'].append(p2.get('comp_sam_pts', all_results['handles'][comp_idx]))
            phase2_results['targets'].append(all_results['targets'][comp_idx])
            phase2_results['pivots'].append(None)  # 非刚性无支点
            phase2_results['anchors'].append(p2.get('anchors_img', np.zeros((0, 2), dtype=np.float32)))
            phase2_results['actions'].append('Non-Rigid')
            phase2_results['sub_actions'].append(p2.get('intent_type', 'Unknown'))
            
            # 累积 Phase 2 的 end mask
            norm_grid = p2.get('norm_grid')
            if norm_grid is not None:
                grid_tensor = norm_grid.permute(2, 0, 1).unsqueeze(0)
                grid_up = F.interpolate(grid_tensor, size=(H_img, W_img), mode='bilinear', align_corners=True)
                grid_up = grid_up.permute(0, 2, 3, 1)
                m_start_t = torch.from_numpy(m_start_full).float().to(device).view(1, 1, H_img, W_img)
                m_end_t = F.grid_sample(m_start_t, grid_up, mode='nearest', align_corners=True)
                m_end_p2_overall = np.maximum(m_end_p2_overall, m_end_t.squeeze().cpu().numpy())
    
    # ============================================
    # 生成 Phase 1 可视化（rigid_01 ~ rigid_04）
    # ============================================
    if len(phase1_results['grids']) > 0 and phase1_results['grids'][0] is not None:
        # Phase 1 的起点和终点
        all_handles_p1 = np.vstack(phase1_results['handles']) if phase1_results['handles'] else all_sam_pts
        all_targets_p1 = np.vstack(phase1_results['targets']) if phase1_results['targets'] else all_targets_xy
        
        vis_01_fill_scopes(source_image_np, m_start_full_overall, m_pseudo_full_overall, 
                          bg_filled_rgb, save_dir, prefix="rigid",
                          hole_inside_mask=(fill_vis_debug or {}).get("subject_hole_mask_full", (fill_vis_debug or {}).get("hole_inside_full")) if isinstance(fill_vis_debug, dict) else None,
                          hole_visual_mode=(fill_vis_debug or {}).get("hole_visual_mode") if isinstance(fill_vis_debug, dict) else None,
                          all_results=phase1_results, device=device)
        
        vis_02_masks_cutouts(source_image_np, phase1_results, m_start_full_overall,
                            all_handles_p1, all_targets_p1, save_dir, device, prefix="rigid")
        
        vis_03_trace_compare(
            source_image_np,
            m_start_full_overall,
            m_end_p1_overall,
            all_handles_p1,
            all_targets_p1,
            save_dir,
            prefix="rigid",
            all_results=phase1_results,
            device=device,
        )
        
        vis_04_final_preview(source_image_np, phase1_results, bg_filled_rgb,
                            all_handles_p1, all_targets_p1, save_dir, device, prefix="rigid")
        
        print(f"  [rigid] ✓ rigid_01 ~ rigid_04")
    
    # ============================================
    # 生成 Phase 2 可视化（nonrigid_01 ~ nonrigid_04）
    # ============================================
    if len(phase2_results['grids']) > 0 and phase2_results['grids'][0] is not None:
        # Phase 2 的起点（旋转后）和终点（目标）
        all_handles_p2 = np.vstack(phase2_results['handles']) if phase2_results['handles'] else all_sam_pts
        all_targets_p2 = np.vstack(phase2_results['targets']) if phase2_results['targets'] else all_targets_xy

        # ============================================
        # 关键修复：生成 Phase 1 旋转后的中间状态图像
        # Phase 2 的源图 = 背景 + 旋转后的主体
        # ============================================
        img_tensor = torch.from_numpy(source_image_np).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        source_p2 = bg_filled_rgb.copy().astype(np.float32)  # 从填充后的背景开始

        for comp_idx in range(len(phase1_results['grids'])):
            grid_p1 = phase1_results['grids'][comp_idx]
            if grid_p1 is not None:
                # 上采样 grid 到图像分辨率
                grid_tensor = grid_p1.permute(2, 0, 1).unsqueeze(0)
                grid_up = F.interpolate(grid_tensor, size=(H_img, W_img), mode='bilinear', align_corners=True)
                grid_up = grid_up.permute(0, 2, 3, 1)

                # 对原图进行旋转变换
                warped_rgb = F.grid_sample(img_tensor, grid_up, mode='bilinear', padding_mode='border', align_corners=True)
                warped_rgb_np = (warped_rgb.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.float32)

                # 获取旋转后的 mask
                m_start = phase1_results['masks_start'][comp_idx]
                m_start_t = torch.from_numpy(m_start).float().to(device).view(1, 1, H_img, W_img)
                m_end_t = F.grid_sample(m_start_t, grid_up, mode='nearest', align_corners=True)
                m_end_np = m_end_t.squeeze().cpu().numpy()

                # 合成：旋转后的主体 + 背景
                mask_3ch = np.stack([m_end_np] * 3, axis=2)
                source_p2 = warped_rgb_np * mask_3ch + source_p2 * (1 - mask_3ch)

        source_p2 = source_p2.astype(np.uint8)

        # Phase 2 使用旋转后的图像作为源图，背景仍然是同一个 bg_filled_rgb
        vis_01_fill_scopes(source_p2, m_start_p2_overall, m_end_p1_overall,
                          bg_filled_rgb, save_dir, prefix="nonrigid",
                          hole_inside_mask=(fill_vis_debug or {}).get("subject_hole_mask_full", (fill_vis_debug or {}).get("hole_inside_full")) if isinstance(fill_vis_debug, dict) else None,
                          hole_visual_mode=(fill_vis_debug or {}).get("hole_visual_mode") if isinstance(fill_vis_debug, dict) else None,
                          all_results=phase2_results, device=device)

        vis_02_masks_cutouts(source_p2, phase2_results, m_start_p2_overall,
                            all_handles_p2, all_targets_p2, save_dir, device, prefix="nonrigid")

        vis_03_trace_compare(
            source_p2,
            m_start_p2_overall,
            m_end_p2_overall,
            all_handles_p2,
            all_targets_p2,
            save_dir,
            prefix="nonrigid",
            all_results=phase2_results,
            device=device,
        )

        vis_04_final_preview(source_p2, phase2_results, bg_filled_rgb,
                            all_handles_p2, all_targets_p2, save_dir, device, prefix="nonrigid")
        vis_05_component_intent_story(source_p2, phase2_results, bg_filled_rgb,
                                     save_dir, device, prefix="nonrigid")
        vis_06_final_composite(source_p2, phase2_results, bg_filled_rgb,
                              save_dir, device, prefix="nonrigid")

        print(f"  [nonrigid] ✓ nonrigid_01 ~ nonrigid_06 (+ per-component stories)")
    
    print("[Hybrid Visualization] ✓ Complete (8 images)")


def run_3d_hybrid_visualizations(source_image_np, all_results,
                                 m_pseudo_full_overall, m_start_full_overall,
                                 all_sam_pts, all_targets_xy, bg_filled_rgb,
                                 save_dir, device, fill_vis_debug=None):
    """
    3D-Hybrid 专用两阶段可视化
    - Phase1: 3D-Rigid（旋转）
    - Phase2: 3D-Non-Rigid（同空间细化）
    """
    print("\n[3D-Hybrid Visualization] Generating two-stage debug images...")
    # 3D-Hybrid 不再展示“intent story”前端可视化，先清理历史遗留文件避免混淆
    for pat in ("nonrigid_*_intent_story.jpg", "nonrigid_05_component_story.jpg"):
        for p in glob.glob(os.path.join(save_dir, pat)):
            try:
                os.remove(p)
            except Exception:
                pass

    H_img, W_img = source_image_np.shape[:2]
    num_components = len(all_results.get('debug_infos', []))

    def _norm_grid_hw(norm_grid):
        if torch.is_tensor(norm_grid) and norm_grid.ndim >= 2:
            return int(norm_grid.shape[0]), int(norm_grid.shape[1])
        if isinstance(norm_grid, np.ndarray) and norm_grid.ndim >= 2:
            return int(norm_grid.shape[0]), int(norm_grid.shape[1])
        return None

    def _to_image_points(points, norm_grid):
        arr = _sanitize_xy_points(points)
        if arr.shape[0] == 0:
            return arr
        hw = _norm_grid_hw(norm_grid)
        if hw is None:
            return arr
        gh, gw = hw
        max_x = float(np.max(arr[:, 0]))
        max_y = float(np.max(arr[:, 1]))
        min_x = float(np.min(arr[:, 0]))
        min_y = float(np.min(arr[:, 1]))
        # 坐标若落在 grid 尺度范围内，按 latent->image 比例映射到原图可视化坐标。
        likely_grid_space = (min_x >= -1.5 and min_y >= -1.5 and max_x <= gw + 1.5 and max_y <= gh + 1.5)
        if likely_grid_space and (W_img > gw or H_img > gh):
            out = arr.copy()
            out[:, 0] *= _coord_scale_latent_to_image(gw, W_img)
            out[:, 1] *= _coord_scale_latent_to_image(gh, H_img)
            return out
        return arr

    phase1_results = {
        'latents': [],
        'masks_start': [],
        'masks_end': [],
        'grids': [],
        'handles': [],
        'targets': [],
        'pivots': [],
        'anchors': [],
        'actions': [],
        'sub_actions': [],
        'debug_infos': []
    }
    m_end_p1_overall = np.zeros((H_img, W_img), dtype=np.float32)

    for comp_idx in range(num_components):
        comp_debug = all_results['debug_infos'][comp_idx]
        p1 = comp_debug.get('phase1_3d')
        if p1 is None:
            continue

        norm_grid_p1 = p1.get('norm_grid')
        sam_mask = p1.get('sam_mask')
        if isinstance(sam_mask, np.ndarray):
            m_start_full = _latent_mask_to_fullres(
                sam_mask.astype(np.float32) / 255.0,
                target_hw=(H_img, W_img),
            )
        else:
            m_start_full = _latent_mask_to_fullres(
                all_results['masks_start'][comp_idx],
                target_hw=(H_img, W_img),
            )

        handles_p1 = _to_image_points(
            p1.get('comp_sam_pts', comp_debug.get('phase1_source_points_xy', all_results['handles'][comp_idx])),
            norm_grid_p1,
        )
        targets_p1 = _to_image_points(
            p1.get('rotated_handle_points_xy', comp_debug.get('phase1_target_points_xy', all_results['targets'][comp_idx])),
            norm_grid_p1,
        )
        if targets_p1.shape[0] != handles_p1.shape[0]:
            targets_p1 = handles_p1.copy()

        centroid = p1.get('centroid')
        pivot = np.array([centroid[0], centroid[1]], dtype=np.float32) if centroid is not None else None
        axis = p1.get('rotation_axis', 'Y')
        angle = p1.get('rotation_angle_deg', 0.0)
        yaw = p1.get('rotation_yaw_deg', angle if axis == 'Y' else 0.0)
        pitch = p1.get('rotation_pitch_deg', angle if axis == 'X' else 0.0)

        phase1_results['latents'].append(p1.get('warped_latents'))
        phase1_results['masks_start'].append(m_start_full)
        phase1_results['masks_end'].append(p1.get('mask_end'))
        phase1_results['grids'].append(norm_grid_p1)
        phase1_results['handles'].append(handles_p1)
        phase1_results['targets'].append(targets_p1)
        phase1_results['pivots'].append(pivot)
        phase1_results['anchors'].append(np.zeros((0, 2), dtype=np.float32))
        phase1_results['actions'].append('3D-Rigid')
        phase1_results['sub_actions'].append(f"Yaw={yaw:.1f},Pitch={pitch:.1f}")

        m_end_full = p1.get('m_end_full')
        if isinstance(m_end_full, np.ndarray):
            m_end_full = _latent_mask_to_fullres(m_end_full, target_hw=(H_img, W_img))
            m_end_p1_overall = np.maximum(m_end_p1_overall, m_end_full.astype(np.float32))
        else:
            m_end_full = None

        p1_vis = dict(p1)
        p1_vis['m_end_full'] = m_end_full
        if isinstance(comp_debug, dict):
            if 'runtime_geom_domain' in comp_debug:
                p1_vis['runtime_geom_domain'] = comp_debug.get('runtime_geom_domain')
            if '_points_already_image_space' in comp_debug:
                p1_vis['_points_already_image_space'] = bool(comp_debug.get('_points_already_image_space', False))
            mask_start_vis = comp_debug.get('_runtime_mask_start_vis')
            if isinstance(mask_start_vis, np.ndarray):
                p1_vis['_runtime_mask_start_vis'] = mask_start_vis
        phase1_results['debug_infos'].append(p1_vis)

    if len(phase1_results['grids']) > 0:
        print("  [3D-Hybrid/Phase1] skip rigid_* visuals (point-cloud only)")

    phase2_results = {
        'latents': [],
        'masks_start': [],
        'masks_end': [],
        'grids': [],
        'handles': [],
        'targets': [],
        'pivots': [],
        'anchors': [],
        'actions': [],
        'sub_actions': [],
        'debug_infos': []
    }
    m_start_p2_overall = m_end_p1_overall.copy() if np.any(m_end_p1_overall > 0.5) else m_start_full_overall.copy()
    m_end_p2_overall = np.zeros((H_img, W_img), dtype=np.float32)

    if len(phase1_results['debug_infos']) == 0:
        source_p2 = source_image_np.copy()
    else:
        source_p2 = bg_filled_rgb.copy().astype(np.float32)
        for p1_idx, p1 in enumerate(phase1_results['debug_infos']):
            rotated_rgb = p1.get('rotated_rgb')
            m_end_full = p1.get('m_end_full')
            if not isinstance(rotated_rgb, np.ndarray):
                p1_domain = str(p1.get('runtime_geom_domain', 'image')).strip().lower()
                p1_start = p1.get('_runtime_mask_start_vis', phase1_results['masks_start'][p1_idx])
                fallback_rgb, fallback_m_end_u8 = _render_component_with_norm_grid(
                    source_image_np=source_image_np,
                    m_start_full=p1_start,
                    norm_grid=phase1_results['grids'][p1_idx],
                    device=device,
                    render_domain=p1_domain,
                )
                rotated_rgb = fallback_rgb
                if not isinstance(m_end_full, np.ndarray):
                    m_end_full = (fallback_m_end_u8.astype(np.float32) / 255.0)
            if isinstance(rotated_rgb, np.ndarray):
                if rotated_rgb.shape[:2] != (H_img, W_img):
                    rotated_rgb = cv2.resize(rotated_rgb, (W_img, H_img), interpolation=cv2.INTER_LINEAR)
                rotated_rgb = np.clip(rotated_rgb, 0, 255).astype(np.float32)
            if isinstance(m_end_full, np.ndarray):
                m_end_full = _latent_mask_to_fullres(m_end_full, target_hw=(H_img, W_img))
                mask_3ch = np.stack([m_end_full] * 3, axis=2)
                source_p2 = rotated_rgb * mask_3ch + source_p2 * (1 - mask_3ch)
        source_p2 = source_p2.astype(np.uint8)

    for comp_idx in range(num_components):
        comp_debug = all_results['debug_infos'][comp_idx]
        p2 = comp_debug.get('phase2_3d_nonrigid')
        if p2 is None:
            p2 = comp_debug.get('phase2_nonrigid')
        if p2 is None and comp_debug.get('is_3d_fallback', False):
            p2 = comp_debug
        if p2 is None:
            continue

        norm_grid_p2 = p2.get('norm_grid')
        m_start_full = p2.get('m_start_full')
        if not isinstance(m_start_full, np.ndarray):
            m_start_full = m_start_p2_overall
        m_start_full = _latent_mask_to_fullres(m_start_full, target_hw=(H_img, W_img))

        default_handles = all_results['handles'][comp_idx] if comp_idx < len(all_results.get('handles', [])) else np.zeros((0, 2), dtype=np.float32)
        default_targets = all_results['targets'][comp_idx] if comp_idx < len(all_results.get('targets', [])) else np.zeros((0, 2), dtype=np.float32)

        phase2_handles = _to_image_points(
            p2.get('comp_sam_pts', comp_debug.get('phase2_source_points_xy', default_handles)),
            norm_grid_p2,
        )
        phase2_targets = _to_image_points(default_targets, norm_grid_p2)
        if phase2_targets.shape[0] != phase2_handles.shape[0]:
            phase2_targets = phase2_handles.copy()

        phase2_results['latents'].append(p2.get('warped_latents'))
        phase2_results['masks_start'].append(m_start_full)
        phase2_results['masks_end'].append(p2.get('mask_end'))
        phase2_results['grids'].append(norm_grid_p2)
        phase2_results['handles'].append(phase2_handles)
        phase2_results['targets'].append(phase2_targets)
        phase2_results['pivots'].append(None)
        phase2_results['anchors'].append(_to_image_points(p2.get('anchors_img', np.zeros((0, 2), dtype=np.float32)), norm_grid_p2))
        phase2_results['actions'].append('3D-Non-Rigid')
        phase2_results['sub_actions'].append('3D-Rigid+3D-Non-Rigid')
        p2_vis = dict(p2)
        if isinstance(comp_debug, dict):
            if 'runtime_geom_domain' in comp_debug:
                p2_vis['runtime_geom_domain'] = comp_debug.get('runtime_geom_domain')
            if '_points_already_image_space' in comp_debug:
                p2_vis['_points_already_image_space'] = bool(comp_debug.get('_points_already_image_space', False))
            mask_start_vis = comp_debug.get('_runtime_mask_start_vis')
            if isinstance(mask_start_vis, np.ndarray):
                p2_vis['_runtime_mask_start_vis'] = mask_start_vis
        phase2_results['debug_infos'].append(p2_vis)

        m_end_full = p2.get('m_end_full')
        if isinstance(m_end_full, np.ndarray):
            m_end_full = _latent_mask_to_fullres(m_end_full, target_hw=(H_img, W_img))
            m_end_p2_overall = np.maximum(m_end_p2_overall, m_end_full.astype(np.float32))
        else:
            if norm_grid_p2 is not None:
                grid_tensor = norm_grid_p2.permute(2, 0, 1).unsqueeze(0)
                grid_up = F.interpolate(grid_tensor, size=(H_img, W_img), mode='bilinear', align_corners=True)
                grid_up = grid_up.permute(0, 2, 3, 1)
                m_start_t = torch.from_numpy(m_start_full).float().to(device).view(1, 1, H_img, W_img)
                m_end_t = F.grid_sample(m_start_t, grid_up, mode='nearest', align_corners=True)
                m_end_p2_overall = np.maximum(m_end_p2_overall, m_end_t.squeeze().cpu().numpy())

    if len(phase2_results['grids']) > 0:
        all_handles_p2 = np.vstack(phase2_results['handles']) if phase2_results['handles'] else all_sam_pts
        all_targets_p2 = np.vstack(phase2_results['targets']) if phase2_results['targets'] else all_targets_xy

        vis_01_fill_scopes(source_p2, m_start_p2_overall, m_start_p2_overall,
                          bg_filled_rgb, save_dir, prefix="nonrigid",
                          hole_inside_mask=(fill_vis_debug or {}).get("subject_hole_mask_full", (fill_vis_debug or {}).get("hole_inside_full")) if isinstance(fill_vis_debug, dict) else None,
                          hole_visual_mode=(fill_vis_debug or {}).get("hole_visual_mode") if isinstance(fill_vis_debug, dict) else None,
                          all_results=phase2_results, device=device)
        vis_02_masks_cutouts(source_p2, phase2_results, m_start_p2_overall,
                            all_handles_p2, all_targets_p2, save_dir, device, prefix="nonrigid")
        vis_03_trace_compare(
            source_p2,
            m_start_p2_overall,
            m_end_p2_overall,
            all_handles_p2,
            all_targets_p2,
            save_dir,
            prefix="nonrigid",
            all_results=phase2_results,
            device=device,
        )
        vis_04_final_preview(source_p2, phase2_results, bg_filled_rgb,
                            all_handles_p2, all_targets_p2, save_dir, device, prefix="nonrigid")
        vis_06_final_composite(source_p2, phase2_results, bg_filled_rgb,
                              save_dir, device, prefix="nonrigid")
        print("  [3D-Hybrid/Phase2] ✓ nonrigid_01 ~ nonrigid_06 (no intent-story)")

    print("[3D-Hybrid Visualization] ✓ Complete")

def run_all_visualizations(source_image_np, all_results, m_pseudo_full_overall, m_start_full_overall,
                           m_end_overall, all_sam_pts, all_targets_xy, bg_filled_rgb,
                           drag_mode, save_dir, device, bg_hole_mask=None, fill_vis_debug=None):
    """
    统一可视化入口
    - 2D-Rigid/2D-Non-Rigid: 生成 6 张图（含论文版 story/composite）
    - 2D-Hybrid: 生成 8 张图（调用专用函数）
    - 3D-Non-Rigid: 生成 4 张图 + 深度图可视化 depth_01~04
    - 3D-Hybrid: 两阶段可视化（Phase1 3D-Rigid + Phase2 3D-Non-Rigid）
    - 3D-Rigid: 生成 4 张图 + 深度图可视化 depth_01~04
    """
    print("\n[Visualization] Generating debug images...")
    canonical_mode, domain, base_mode = split_drag_mode(drag_mode)
    is_3d = (domain == "3D")

    def _copy_if_exists(src_name, dst_name):
        src_path = os.path.join(save_dir, src_name)
        dst_path = os.path.join(save_dir, dst_name)
        if os.path.isfile(src_path):
            try:
                shutil.copy2(src_path, dst_path)
            except Exception:
                pass

    try:
        if canonical_mode == '3D-Hybrid':
            run_3d_hybrid_visualizations(
                source_image_np, all_results,
                m_pseudo_full_overall, m_start_full_overall,
                all_sam_pts, all_targets_xy, bg_filled_rgb,
                save_dir, device, fill_vis_debug=fill_vis_debug
            )
            _create_ordered_visualization_aliases(save_dir, canonical_mode)
            return
        if is_3d:
            # 3D 系列模式：复用通用可视化函数 + 深度图可视化
            prefix = canonical_mode.replace("-", "")
            final_composed_preview = _build_final_composite_rgb(
                source_image_np=source_image_np,
                all_results=all_results,
                bg_filled_rgb=bg_filled_rgb,
                device=device,
            )

            # 【重构】复用通用可视化函数，支持多连通域
            vis_01_fill_scopes(
                source_image_np,
                m_start_full_overall,
                m_pseudo_full_overall,
                bg_filled_rgb,
                save_dir,
                prefix,
                m_end_full_overall=m_end_overall,
                bg_hole_mask=bg_hole_mask,
                hole_inside_mask=(fill_vis_debug or {}).get("subject_hole_mask_full", (fill_vis_debug or {}).get("hole_inside_full")) if isinstance(fill_vis_debug, dict) else None,
                hole_visual_mode=(fill_vis_debug or {}).get("hole_visual_mode") if isinstance(fill_vis_debug, dict) else None,
                final_composed_rgb=final_composed_preview,
                all_results=all_results,
                device=device,
            )
            vis_02_masks_cutouts(source_image_np, all_results, m_start_full_overall,
                                all_sam_pts, all_targets_xy, save_dir, device, prefix)
            vis_03_trace_compare(
                source_image_np,
                m_start_full_overall,
                m_end_overall,
                all_sam_pts,
                all_targets_xy,
                save_dir,
                prefix,
                all_results=all_results,
                device=device,
            )
            vis_04_final_preview(source_image_np, all_results, bg_filled_rgb,
                                all_sam_pts, all_targets_xy, save_dir, device, prefix)
            vis_03_hole_masks(source_image_np, all_results, bg_filled_rgb, fill_vis_debug, save_dir, device, prefix)
            vis_04_fill_process(
                source_image_np,
                fill_vis_debug,
                save_dir,
                prefix,
                base_rgb=final_composed_preview,
                all_results=all_results,
            )

            # 3D-Non-Rigid/3D-Rigid: 同步生成“语义序号”文件名，避免前缀序号与后缀序号错位。
            if canonical_mode in ("3D-Non-Rigid", "3D-Rigid"):
                _copy_if_exists(f"{prefix}_02_masks_cutouts.jpg", f"{prefix}_01_masks_cutouts.jpg")
                _copy_if_exists(f"{prefix}_03_trace_compare.jpg", f"{prefix}_02_trace_compare.jpg")
                _copy_if_exists(f"{prefix}_01_fill_scopes.jpg", f"{prefix}_05_fill_scopes.jpg")
                _copy_if_exists(f"{prefix}_04_final_preview.jpg", f"{prefix}_06_final_preview.jpg")

            # 【额外】深度图可视化
            from utils_drag.rotate_3d_processor import (
                vis_depth_01_full_depth, vis_depth_02_component_depths,
                vis_depth_03_rotated_depths, vis_depth_04_rgb_projection,
                create_depth_colormap as colorize_depth
            )

            debug_infos = all_results.get('debug_infos', [{}])
            num_components = len(debug_infos)

            # 遍历所有连通域生成深度可视化
            for comp_idx, debug_info in enumerate(debug_infos):
                depth_map = debug_info.get('depth_map')
                sam_mask = debug_info.get('sam_mask')
                rotated_mask = debug_info.get('rotated_mask')
                rotated_rgb = debug_info.get('rotated_rgb')
                rotation_axis = debug_info.get('rotation_axis', 'Y')
                rotation_angle = debug_info.get('rotation_angle_deg', 0)
                centroid = debug_info.get('centroid')

                if depth_map is None:
                    print(f"  [3D] Comp{comp_idx+1}: No depth_map, skipping depth visualization")
                    continue

                H_src, W_src = source_image_np.shape[:2]

                depth_map_vis = np.asarray(depth_map, dtype=np.float32)
                if depth_map_vis.shape[:2] != (H_src, W_src):
                    src_h_raw, src_w_raw = depth_map_vis.shape[:2]
                    depth_map_vis = cv2.resize(
                        depth_map_vis,
                        (W_src, H_src),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    if centroid is not None:
                        sx = float(W_src) / float(max(src_w_raw, 1))
                        sy = float(H_src) / float(max(src_h_raw, 1))
                        centroid = (
                            int(round(float(centroid[0]) * sx)),
                            int(round(float(centroid[1]) * sy)),
                        )

                sam_mask_vis = sam_mask
                if isinstance(sam_mask_vis, np.ndarray):
                    sam_mask_vis = sam_mask_vis.astype(np.uint8)
                    if sam_mask_vis.max() <= 1:
                        sam_mask_vis = (sam_mask_vis * 255).astype(np.uint8)
                    if sam_mask_vis.shape[:2] != (H_src, W_src):
                        sam_mask_vis = cv2.resize(
                            sam_mask_vis,
                            (W_src, H_src),
                            interpolation=cv2.INTER_NEAREST,
                        )
                elif comp_idx < len(all_results.get('masks_start', [])):
                    sam_mask_full = _latent_mask_to_fullres(
                        all_results['masks_start'][comp_idx],
                        target_hw=(H_src, W_src),
                    )
                    sam_mask_vis = (sam_mask_full * 255).astype(np.uint8)

                rotated_mask_vis = rotated_mask
                if isinstance(rotated_mask_vis, np.ndarray):
                    rotated_mask_vis = rotated_mask_vis.astype(np.uint8)
                    if rotated_mask_vis.max() <= 1:
                        rotated_mask_vis = (rotated_mask_vis * 255).astype(np.uint8)
                    if rotated_mask_vis.shape[:2] != (H_src, W_src):
                        rotated_mask_vis = cv2.resize(
                            rotated_mask_vis,
                            (W_src, H_src),
                            interpolation=cv2.INTER_NEAREST,
                        )

                rotated_depth_vis = debug_info.get('rotated_depth', depth_map_vis)
                if isinstance(rotated_depth_vis, np.ndarray):
                    rotated_depth_vis = np.asarray(rotated_depth_vis, dtype=np.float32)
                    if rotated_depth_vis.shape[:2] != (H_src, W_src):
                        rotated_depth_vis = cv2.resize(
                            rotated_depth_vis,
                            (W_src, H_src),
                            interpolation=cv2.INTER_LINEAR,
                        )
                else:
                    rotated_depth_vis = depth_map_vis

                rotated_rgb_vis = rotated_rgb
                if isinstance(rotated_rgb_vis, np.ndarray):
                    if rotated_rgb_vis.shape[:2] != (H_src, W_src):
                        rotated_rgb_vis = cv2.resize(
                            rotated_rgb_vis,
                            (W_src, H_src),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    rotated_rgb_vis = np.clip(rotated_rgb_vis, 0, 255).astype(np.uint8)
                elif comp_idx < len(all_results.get('grids', [])):
                    fallback_grid = all_results['grids'][comp_idx]
                    fallback_dbg = (
                        all_results['debug_infos'][comp_idx]
                        if comp_idx < len(all_results.get('debug_infos', []))
                        else {}
                    )
                    fallback_domain = str(fallback_dbg.get('runtime_geom_domain', 'image')).strip().lower()
                    fallback_start = (
                        fallback_dbg.get('_runtime_mask_start_vis')
                        if isinstance(fallback_dbg, dict) and ('_runtime_mask_start_vis' in fallback_dbg)
                        else (
                            all_results['masks_start'][comp_idx]
                            if comp_idx < len(all_results.get('masks_start', []))
                            else np.zeros((H_src, W_src), dtype=np.float32)
                        )
                    )
                    fallback_rgb, fallback_m_end_u8 = _render_component_with_norm_grid(
                        source_image_np=source_image_np,
                        m_start_full=fallback_start,
                        norm_grid=fallback_grid,
                        device=device,
                        render_domain=fallback_domain,
                    )
                    rotated_rgb_vis = fallback_rgb
                    if rotated_mask_vis is None:
                        rotated_mask_vis = fallback_m_end_u8
                # 不再为每个连通域创建子目录，统一写到同一目录并覆盖同名文件
                comp_save_dir = save_dir

                depth_colored = colorize_depth(depth_map_vis)

                # depth_01: 整体深度图（只生成一次）
                if comp_idx == 0:
                    vis_depth_01_full_depth(source_image_np, depth_map_vis, depth_colored, save_dir)

                # depth_02: 连通域深度信息
                if sam_mask_vis is not None and centroid is not None:
                    # 获取当前连通域的控制点
                    comp_handles = all_results['handles'][comp_idx] if comp_idx < len(all_results['handles']) else []
                    comp_targets = all_results['targets'][comp_idx] if comp_idx < len(all_results['targets']) else []
                    vis_depth_02_component_depths(source_image_np, sam_mask_vis, depth_map_vis, centroid,
                                                  rotation_axis, rotation_angle, comp_save_dir,
                                                  comp_handles, comp_targets,
                                                  yaw_deg=debug_info.get('rotation_yaw_deg'),
                                                  pitch_deg=debug_info.get('rotation_pitch_deg'))

                # depth_03: 旋转后深度
                if rotated_mask_vis is not None:
                    vis_depth_03_rotated_depths(
                        depth_colored,
                        rotated_depth_vis,
                        sam_mask_vis,
                        rotated_mask_vis,
                        comp_save_dir,
                    )

                # depth_04: RGB投影
                if rotated_rgb_vis is not None and sam_mask_vis is not None:
                    # 合成最终结果
                    m_end_full = debug_info.get('m_end_full')
                    if m_end_full is not None:
                        m_end_full_vis = np.asarray(m_end_full, dtype=np.float32)
                        if m_end_full_vis.shape[:2] != (H_src, W_src):
                            m_end_full_vis = cv2.resize(
                                m_end_full_vis,
                                (W_src, H_src),
                                interpolation=cv2.INTER_LINEAR,
                            )
                        m_end_full_vis = np.clip(m_end_full_vis, 0.0, 1.0)
                        mask_3ch = np.stack([m_end_full_vis] * 3, axis=2)
                        final_composed = (rotated_rgb_vis * mask_3ch + bg_filled_rgb * (1 - mask_3ch)).astype(np.uint8)
                    else:
                        final_composed = rotated_rgb_vis
                    vis_depth_04_rgb_projection(
                        source_image_np,
                        sam_mask_vis,
                        rotated_rgb_vis,
                        final_composed,
                        comp_save_dir,
                    )

            print(f"  [{canonical_mode}] ✓ {prefix}_01~04 + depth_01~04")
            _create_ordered_visualization_aliases(save_dir, canonical_mode)
            return
        elif canonical_mode == '2D-Hybrid':
            # Hybrid 模式：调用专用可视化函数
            run_hybrid_visualizations(
                source_image_np, all_results, all_results.get('debug_infos', [{}])[0],
                m_pseudo_full_overall, m_start_full_overall,
                all_sam_pts, all_targets_xy, bg_filled_rgb,
                save_dir, device, fill_vis_debug=fill_vis_debug
            )
            _create_ordered_visualization_aliases(save_dir, canonical_mode)
        else:
            # 2D-Rigid / 2D-Non-Rigid 模式：生成 6 张图
            if canonical_mode == '2D-Rigid':
                prefix = "rigid"
            else:
                prefix = "nonrigid"

            final_composed_preview = _build_final_composite_rgb(
                source_image_np=source_image_np,
                all_results=all_results,
                bg_filled_rgb=bg_filled_rgb,
                device=device,
            )
            vis_01_fill_scopes(
                source_image_np,
                m_start_full_overall,
                m_pseudo_full_overall,
                bg_filled_rgb,
                save_dir,
                prefix,
                m_end_full_overall=m_end_overall,
                bg_hole_mask=bg_hole_mask,
                hole_inside_mask=(fill_vis_debug or {}).get("subject_hole_mask_full", (fill_vis_debug or {}).get("hole_inside_full")) if isinstance(fill_vis_debug, dict) else None,
                hole_visual_mode=(fill_vis_debug or {}).get("hole_visual_mode") if isinstance(fill_vis_debug, dict) else None,
                final_composed_rgb=final_composed_preview,
                all_results=all_results,
                device=device,
            )

            vis_02_masks_cutouts(source_image_np, all_results, m_start_full_overall,
                                all_sam_pts, all_targets_xy, save_dir, device, prefix)

            vis_03_trace_compare(
                source_image_np,
                m_start_full_overall,
                m_end_overall,
                all_sam_pts,
                all_targets_xy,
                save_dir,
                prefix,
                all_results=all_results,
                device=device,
            )

            vis_04_final_preview(source_image_np, all_results, bg_filled_rgb,
                                all_sam_pts, all_targets_xy, save_dir, device, prefix)
            vis_03_hole_masks(source_image_np, all_results, bg_filled_rgb, fill_vis_debug, save_dir, device, prefix)
            vis_04_fill_process(
                source_image_np,
                fill_vis_debug,
                save_dir,
                prefix,
                base_rgb=final_composed_preview,
                all_results=all_results,
            )
            vis_05_component_intent_story(source_image_np, all_results, bg_filled_rgb,
                                         save_dir, device, prefix)
            vis_06_final_composite(source_image_np, all_results, bg_filled_rgb,
                                  save_dir, device, prefix)

            print(f"  [{prefix}] ✓ {prefix}_01 ~ {prefix}_06 (+ per-component stories)")
            _create_ordered_visualization_aliases(save_dir, canonical_mode)
        
        print("[Visualization] ✓ Complete")
    except Exception as e:
        print(f"[Warning] Visualization Error: {e}")
        import traceback
        traceback.print_exc()


# ==========================================
# 6. 主流程 - 简化版
# ==========================================
