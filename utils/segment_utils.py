# segment_utils.py
import torch
import numpy as np
import os
import sys
import cv2
import shutil
from utils import MODEL_INIT_LOCK
from tdedit_paths import SAM2_CHECKPOINT, SAM2_CONFIG

# ==========================================
# 1. 路径与配置
# ==========================================
SAM2_CONFIG_LOCAL = SAM2_CONFIG
DEFAULT_MASK_DEBUG_DIR = "debug_files/mask_process"
PAPER_NO_TEXT_SINGLE_ENV = "TDEDIT_PAPER_NO_TEXT_SINGLE"

# ==========================================
# 2. 辅助函数
# ==========================================
def put_text_with_outline(img, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.6, color=(255, 255, 255), thickness=2, outline_color=(0, 0, 0)):
    """在图像上绘制带黑色描边的文字（支持叠加）"""
    if _paper_no_text_single_enabled():
        return
    cv2.putText(img, text, pos, font, scale, outline_color, thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def _paper_no_text_single_enabled():
    value = str(os.environ.get(PAPER_NO_TEXT_SINGLE_ENV, "")).strip().lower()
    return value in {"1", "true", "yes", "on"}

def _module_has_meta_tensors(module):
    for tensor in module.parameters():
        if getattr(tensor, "is_meta", False):
            return True
    for tensor in module.buffers():
        if getattr(tensor, "is_meta", False):
            return True
    return False


def manual_load_checkpoint(model, ckpt_path, force_assign=False):
    if ckpt_path is not None:
        print(f"[SAM2] Loading checkpoint from {ckpt_path}...")
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        load_kwargs = {"strict": False}
        use_assign = force_assign or _module_has_meta_tensors(model)
        if use_assign:
            # 关键兜底：在 meta 参数上用 assign=True，避免 copy no-op。
            load_kwargs["assign"] = True
            print("[SAM2] Meta tensors detected, loading with assign=True.")

        missing_keys, unexpected_keys = model.load_state_dict(sd, **load_kwargs)
        if missing_keys:
            print(f"[SAM2 Info] Missing {len(missing_keys)} keys.")
        if unexpected_keys:
            print(f"[SAM2 Info] Ignored {len(unexpected_keys)} unexpected keys.")

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0: return 0.0
    return intersection / union

# ==========================================
# 3. SAM2 包装类
# ==========================================
class SAM2Wrapper:
    def __init__(self, device):
        self.device = device
        self.predictor = None
        self.is_initialized = False

    def initialize(self):
        if self.is_initialized:
            return

        with MODEL_INIT_LOCK:
            if self.is_initialized:
                return
            # SAM is optional for text-only editing; import it only on use.
            try:
                from hydra.utils import instantiate
                from omegaconf import OmegaConf
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except ImportError as exc:
                raise ImportError(
                    "SAM-refined editing requires SAM2, hydra-core and omegaconf. "
                    "Install the optional SAM2 dependencies and configure "
                    "TDEDIT_SAM2_CHECKPOINT and TDEDIT_SAM2_CONFIG."
                ) from exc
            if not os.path.exists(SAM2_CHECKPOINT):
                raise FileNotFoundError(f"SAM checkpoint not found: {SAM2_CHECKPOINT}; set TDEDIT_SAM2_CHECKPOINT")
            if not os.path.exists(SAM2_CONFIG_LOCAL):
                raise FileNotFoundError(f"SAM config not found: {SAM2_CONFIG_LOCAL}; set TDEDIT_SAM2_CONFIG")
            try:
                cfg = OmegaConf.load(SAM2_CONFIG_LOCAL)
                model = instantiate(cfg.model, _recursive_=True)
                manual_load_checkpoint(model, SAM2_CHECKPOINT)

                if _module_has_meta_tensors(model):
                    # 双重兜底：若仍含 meta 参数，先在 CPU 分配空参数后再装载。
                    print("[SAM2] Meta tensors remained after load, retry via to_empty(cpu).")
                    model = model.to_empty(device=torch.device("cpu"))
                    manual_load_checkpoint(model, SAM2_CHECKPOINT, force_assign=True)

                model = model.to(self.device)
                model.eval()
                self.predictor = SAM2ImagePredictor(model)
                self.is_initialized = True
            except Exception as e:
                print(f"[SAM2] Init Failed: {e}")
                raise e

    def get_object_mask(self, image_np, points, user_hint_mask=None, debug_dir=None, show_hidden_points=True, comp_id=None):
        """
        Args:
            image_np: RGB 图像
            points: 用户点击的点列表 [[x1, y1], [x2, y2], ...]
            user_hint_mask: 用户提供的 hint mask (可选)
            debug_dir: 调试输出目录
            show_hidden_points: 是否显示自动生成的隐藏点
            comp_id: 连通域 ID (用于多连通域场景的文件命名)
        """
        if not self.is_initialized: self.initialize()
        if debug_dir: os.makedirs(debug_dir, exist_ok=True)

        if image_np.dtype != np.uint8:
            image_np = (image_np * 255).astype(np.uint8) if image_np.max() <= 1.0 else image_np.astype(np.uint8)

        self.predictor.set_image(image_np)
        
        # ==========================================================
        # 1. 准备点数据 & 视觉核心采样 (Visual Core Squared Sampling)
        # ==========================================================
        input_points_list = list(points)
        input_labels_list = [1] * len(points)
        num_original_points = len(points)

        if user_hint_mask is not None:
            mask_uint8 = (user_hint_mask > 127).astype(np.uint8) if user_hint_mask.max() > 1 else (user_hint_mask > 0.5).astype(np.uint8)
            
            dist_map = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
            prob_map = dist_map ** 2
            
            total_weight = prob_map.sum()
            if total_weight > 0:
                flat_prob = prob_map.flatten() / total_weight
                num_pixels = prob_map.size
                num_samples = 5
                
                np.random.seed(42) 
                sampled_indices = np.random.choice(
                    num_pixels, 
                    size=num_samples, 
                    replace=False, 
                    p=flat_prob
                )
                
                sampled_coords = np.unravel_index(sampled_indices, dist_map.shape)
                y_coords = sampled_coords[0]
                x_coords = sampled_coords[1]
                
                comp_label = f"Comp {comp_id + 1}" if comp_id is not None else "Single"
                print(f"[SAM2 {comp_label}] Added {len(x_coords)} hidden points based on Visual Core (Squared).")
                
                for i in range(len(x_coords)):
                    input_points_list.append([x_coords[i], y_coords[i]])
                    input_labels_list.append(1)

        final_input_points = np.array(input_points_list)
        final_input_labels = np.array(input_labels_list)

        # ==========================================================
        # 2. Predict
        # ==========================================================
        masks, scores, logits = self.predictor.predict(
            point_coords=final_input_points,
            point_labels=final_input_labels,
            multimask_output=True 
        )
        
        # ==========================================================
        # 3. Rank (IoU + Coverage)
        # ==========================================================
        candidates = []
        user_hint_bool = None
        user_hint_area = 0
        
        if user_hint_mask is not None and user_hint_mask.sum() > 0:
            user_hint_bool = user_hint_mask > (0.5 if user_hint_mask.max() <= 1.0 else 127)
            user_hint_area = user_hint_bool.sum()
        
        for i in range(3):
            m = masks[i] > 0
            area = np.sum(m)
            
            iou = 0.0
            coverage = 0.0
            
            if user_hint_bool is not None:
                intersection = np.logical_and(m, user_hint_bool).sum()
                union = np.logical_or(m, user_hint_bool).sum()
                
                if union > 0: iou = intersection / union
                if user_hint_area > 0: coverage = intersection / user_hint_area
            else:
                iou = scores[i] 

            candidates.append({
                "index": i, "mask": m, "score": scores[i], "area": area, 
                "iou": iou, "coverage": coverage
            })

        if user_hint_bool is not None:
            def calculate_score(cand):
                return cand["iou"] * 0.8 + cand["coverage"] * 0.2
            candidates.sort(key=calculate_score, reverse=True)
            
            comp_label = f"Comp {comp_id + 1}" if comp_id is not None else "Single"
            print(f"[SAM2 {comp_label} Rank] Results (Sorted by Hybrid Score):")
            for cand in candidates:
                score = calculate_score(cand)
                print(f"  Idx={cand['index']} | Score={score:.3f} | Cov={cand['coverage']:.2f} | IoU={cand['iou']:.2f}")
        else:
            candidates.sort(key=lambda x: x["score"], reverse=True)

        # ==========================================================
        # 4. Visualize Candidates
        # ==========================================================
        if debug_dir:
            summary_cols = []
            
            for rank, cand in enumerate(candidates):
                m_vis = (cand['mask'] > 0).astype(np.uint8)
                
                # --- A. Comparison View ---
                if user_hint_bool is not None:
                    top_view = np.zeros((m_vis.shape[0], m_vis.shape[1], 3), dtype=np.uint8)
                    top_view[..., 0] = user_hint_bool.astype(np.uint8) * 255  # Blue: Hint
                    top_view[..., 2] = m_vis * 255                             # Red: SAM
                    top_view[..., 1] = np.logical_and(user_hint_bool, m_vis).astype(np.uint8) * 255  # Green: Overlap
                else:
                    top_view = cv2.cvtColor(m_vis * 255, cv2.COLOR_GRAY2BGR)

                # --- B. Overlay View ---
                overlay = image_np.copy()
                green_mask = np.zeros_like(overlay)
                green_mask[m_vis > 0] = [0, 255, 0]
                
                mask_indices = m_vis > 0
                overlay[mask_indices] = cv2.addWeighted(overlay[mask_indices], 0.6, green_mask[mask_indices], 0.4, 0)
                
                for i, pt in enumerate(final_input_points):
                    pt_coords = (int(pt[0]), int(pt[1]))
                    if i < num_original_points:
                        cv2.circle(overlay, pt_coords, 5, (0, 0, 255), -1)
                    else:
                        if show_hidden_points:
                            cv2.circle(overlay, pt_coords, 4, (64, 64, 64), -1)
                
                overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

                # --- C. 添加文字信息（在保存之前）---
                info_txt = f"R{rank+1} C:{cand['coverage']:.2f} I:{cand['iou']:.2f}"
                put_text_with_outline(top_view, info_txt, (10, 30), scale=0.8)
                put_text_with_outline(overlay_bgr, info_txt, (10, 30), scale=0.8)
                
                # --- D. 保存单独文件（带文字）---
                cv2.imwrite(os.path.join(debug_dir, f"03_sam_rank{rank+1}_comparison.jpg"), top_view)
                cv2.imwrite(os.path.join(debug_dir, f"03_sam_rank{rank+1}_overlay.jpg"), overlay_bgr)

                # --- E. Summary ---
                col_combined = np.vstack([top_view, overlay_bgr])
                summary_cols.append(col_combined)

            # --- F. Summary ---
            if len(summary_cols) > 0:
                try:
                    summary_final = np.hstack(summary_cols)
                    if summary_final.shape[1] > 2000:
                        scale = 2000 / summary_final.shape[1]
                        summary_final = cv2.resize(summary_final, None, fx=scale, fy=scale)
                    cv2.imwrite(os.path.join(debug_dir, f"03_sam_all_candidates_summary.jpg"), summary_final)
                except Exception as e:
                    print(f"[SAM2] Summary generation failed: {e}")

        # ==========================================================
        # 5. Final Selection
        # ==========================================================
        best_cand = candidates[0]
        result_mask_uint8 = best_cand['mask'].astype(np.uint8) * 255
        
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, f"04_final_selected_mask.png"), result_mask_uint8)
            cutout = image_np.copy()
            cutout[result_mask_uint8 == 0] = 0
            cv2.imwrite(os.path.join(debug_dir, f"05_final_selected_cutout.jpg"), cv2.cvtColor(cutout, cv2.COLOR_RGB2BGR))

        return result_mask_uint8


# ==========================================
# 4. 全局接口（支持多连通域）
# ==========================================
sam2_wrapper = None


def preload_sam2(device=None):
    """
    预加载 SAM2 模型，供前端启动阶段调用。
    """
    global sam2_wrapper
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with MODEL_INIT_LOCK:
        if sam2_wrapper is None:
            sam2_wrapper = SAM2Wrapper(device)
    sam2_wrapper.initialize()
    return sam2_wrapper

def get_interactive_mask(image, handle_points, device, user_hint_mask=None, debug_dir=None, comp_id=None, enable_debug=True):
    """
    为单个连通域获取 SAM 分割结果
    
    Args:
        image: RGB 图像
        handle_points: 控制点列表
        device: torch device
        user_hint_mask: 可选的 hint mask
        debug_dir: 调试输出目录
        comp_id: 连通域 ID (用于多连通域场景，仅用于日志输出)
    """
    global sam2_wrapper
    if sam2_wrapper is None:
        sam2_wrapper = preload_sam2(device=device)
    
    if enable_debug:
        target_dir = debug_dir if debug_dir else DEFAULT_MASK_DEBUG_DIR
    else:
        target_dir = None
    return sam2_wrapper.get_object_mask(
        image, 
        handle_points, 
        user_hint_mask, 
        target_dir,
        show_hidden_points=True,
        comp_id=comp_id
    )


def get_interactive_masks_batch(image, all_handle_points, device, all_user_hint_masks=None, debug_dir=None, enable_debug=True):
    """
    批量处理多个连通域，分别调用 SAM2
    """
    num_components = len(all_handle_points)
    all_masks = []
    all_visualizations = []
    
    for comp_idx in range(num_components):
        handle_points = all_handle_points[comp_idx]
        user_hint_mask = all_user_hint_masks[comp_idx] if all_user_hint_masks else None
        
        print(f"\n[SAM2 Batch] Processing Component {comp_idx + 1}/{num_components}...")
        
        temp_comp_dir = os.path.join(debug_dir, f"temp_comp{comp_idx}") if (enable_debug and debug_dir) else None
        if temp_comp_dir:
            os.makedirs(temp_comp_dir, exist_ok=True)
        
        mask = get_interactive_mask(
            image=image,
            handle_points=handle_points,
            device=device,
            user_hint_mask=user_hint_mask,
            debug_dir=temp_comp_dir,
            enable_debug=enable_debug,
            comp_id=comp_idx
        )
        
        all_masks.append(mask)
        
        if temp_comp_dir:
            comp_vis = {
                'comp_idx': comp_idx,
                'rank1_comparison': os.path.join(temp_comp_dir, "03_sam_rank1_comparison.jpg"),
                'rank2_comparison': os.path.join(temp_comp_dir, "03_sam_rank2_comparison.jpg"),
                'rank3_comparison': os.path.join(temp_comp_dir, "03_sam_rank3_comparison.jpg"),
                'rank1_overlay': os.path.join(temp_comp_dir, "03_sam_rank1_overlay.jpg"),
                'rank2_overlay': os.path.join(temp_comp_dir, "03_sam_rank2_overlay.jpg"),
                'rank3_overlay': os.path.join(temp_comp_dir, "03_sam_rank3_overlay.jpg"),
                'final_mask': os.path.join(temp_comp_dir, "04_final_selected_mask.png"),
                'final_cutout': os.path.join(temp_comp_dir, "05_final_selected_cutout.jpg")
            }
            all_visualizations.append(comp_vis)
    
    # ==========================================================
    # 生成汇总可视化
    # 1) 03_sam_summary.jpg: 分章版（Mask / Subject / 2x3 Analysis）
    # 2) 03_sam_summary_overview.jpg: 原4列总览版
    # 3) 每个连通域单独输出（mask/subject/analysis/summary）
    # ==========================================================
    if enable_debug and debug_dir and len(all_visualizations) > 0:
        print(f"\n[SAM2 Batch] Generating summary visualization...")
        
        FONT_SCALE = 0.6
        MAX_OUTPUT_WIDTH = 3000
        
        try:
            def read_bgr_image(path):
                if not path or not os.path.exists(path):
                    return None
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    return None
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                return img

            def add_title_bar(panel, title, height=30, bg=50):
                bar = np.ones((height, panel.shape[1], 3), dtype=np.uint8) * bg
                put_text_with_outline(bar, title, (10, 22), scale=FONT_SCALE, color=(255, 255, 255))
                return np.vstack([bar, panel])

            def stack_vertical(panels):
                if len(panels) == 0:
                    return None
                max_width = max(p.shape[1] for p in panels)
                padded = []
                for p in panels:
                    if p.shape[1] < max_width:
                        pad = np.zeros((p.shape[0], max_width - p.shape[1], 3), dtype=np.uint8)
                        p = np.hstack([p, pad])
                    padded.append(p)
                return np.vstack(padded)

            def normalize_panel(img, h, w, fallback_text):
                if img is None:
                    canvas = np.zeros((h, w, 3), dtype=np.uint8)
                    put_text_with_outline(canvas, fallback_text, (10, 50), scale=FONT_SCALE)
                    return canvas
                if img.shape[0] != h or img.shape[1] != w:
                    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                return img

            def save_with_width_limit(path, img):
                out = img
                if out is None:
                    return
                if out.shape[1] > MAX_OUTPUT_WIDTH:
                    scale = MAX_OUTPUT_WIDTH / out.shape[1]
                    out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                cv2.imwrite(path, out)

            overview_rows = []
            chapter_masks = []
            chapter_subjects = []
            chapter_analysis = []

            for vis_info in all_visualizations:
                comp_idx = vis_info['comp_idx']
                comp_no = comp_idx + 1

                mask_img = read_bgr_image(vis_info['final_mask'])
                cutout_img = read_bgr_image(vis_info['final_cutout'])
                rank_comp_imgs = [read_bgr_image(vis_info.get(f"rank{r}_comparison")) for r in range(1, 4)]
                rank_over_imgs = [read_bgr_image(vis_info.get(f"rank{r}_overlay")) for r in range(1, 4)]

                ref_imgs = [mask_img, cutout_img] + rank_comp_imgs + rank_over_imgs
                base_img = next((im for im in ref_imgs if im is not None), None)
                if base_img is None:
                    h, w = 400, 300
                else:
                    h, w = base_img.shape[:2]

                mask_panel = normalize_panel(mask_img, h, w, "Final Mask N/A")
                cutout_panel = normalize_panel(cutout_img, h, w, "Final Subject N/A")
                put_text_with_outline(mask_panel, "Final Mask", (10, 30), scale=FONT_SCALE)
                put_text_with_outline(cutout_panel, "Final Subject", (10, 30), scale=FONT_SCALE)

                comp_top = []
                comp_bottom = []
                for rank in range(3):
                    comp_panel = normalize_panel(rank_comp_imgs[rank], h, w, f"Rank{rank + 1} Compare N/A")
                    over_panel = normalize_panel(rank_over_imgs[rank], h, w, f"Rank{rank + 1} Overlay N/A")
                    comp_top.append(comp_panel)
                    comp_bottom.append(over_panel)

                analysis_panel = np.vstack([np.hstack(comp_top), np.hstack(comp_bottom)])

                # 旧版总览：每个连通域 4 列（第1列为 mask+subject）
                overview_cols = [np.vstack([mask_panel, cutout_panel])]
                for rank in range(3):
                    overview_cols.append(np.vstack([comp_top[rank], comp_bottom[rank]]))
                overview_row = np.hstack(overview_cols)
                overview_rows.append(overview_row)

                # 每连通域单独输出
                comp_mask_labeled = mask_panel
                comp_subject_labeled = cutout_panel
                comp_analysis_labeled = analysis_panel
                comp_summary = stack_vertical([comp_mask_labeled, comp_subject_labeled, comp_analysis_labeled])

                save_with_width_limit(os.path.join(debug_dir, f"03_sam_comp{comp_no:02d}_mask.jpg"), comp_mask_labeled)
                save_with_width_limit(os.path.join(debug_dir, f"03_sam_comp{comp_no:02d}_subject.jpg"), comp_subject_labeled)
                save_with_width_limit(os.path.join(debug_dir, f"03_sam_comp{comp_no:02d}_analysis.jpg"), comp_analysis_labeled)
                save_with_width_limit(os.path.join(debug_dir, f"03_sam_comp{comp_no:02d}_summary.jpg"), comp_summary)

                chapter_masks.append(mask_panel)
                chapter_subjects.append(cutout_panel)
                chapter_analysis.append(analysis_panel)

            # 旧版总览输出
            overview_summary = stack_vertical(overview_rows)
            save_with_width_limit(os.path.join(debug_dir, "03_sam_summary_overview.jpg"), overview_summary)
            if overview_summary is not None:
                print(f"  ✓ Saved: 03_sam_summary_overview.jpg")

            # 分章输出
            mask_summary = stack_vertical(chapter_masks)
            subject_summary = stack_vertical(chapter_subjects)
            analysis_summary = stack_vertical(chapter_analysis)

            if mask_summary is not None:
                save_with_width_limit(os.path.join(debug_dir, "03_sam_summary_mask.jpg"), mask_summary)
                print(f"  ✓ Saved: 03_sam_summary_mask.jpg")
            if subject_summary is not None:
                save_with_width_limit(os.path.join(debug_dir, "03_sam_summary_subject.jpg"), subject_summary)
                print(f"  ✓ Saved: 03_sam_summary_subject.jpg")
            if analysis_summary is not None:
                save_with_width_limit(os.path.join(debug_dir, "03_sam_summary_analysis.jpg"), analysis_summary)
                print(f"  ✓ Saved: 03_sam_summary_analysis.jpg")

            chapter_blocks = []
            if mask_summary is not None:
                chapter_blocks.append(add_title_bar(mask_summary, "Chapter 1: Final Masks", height=34, bg=70))
            if subject_summary is not None:
                chapter_blocks.append(add_title_bar(subject_summary, "Chapter 2: Masked Subjects", height=34, bg=70))
            if analysis_summary is not None:
                chapter_blocks.append(add_title_bar(analysis_summary, "Chapter 3: Analysis Process (2x3)", height=34, bg=70))

            final_summary = stack_vertical(chapter_blocks)
            save_with_width_limit(os.path.join(debug_dir, "03_sam_summary.jpg"), final_summary)
            if final_summary is not None:
                print(f"  ✓ Saved: 03_sam_summary.jpg")

            # 清理临时目录
            for vis_info in all_visualizations:
                temp_dir = os.path.dirname(vis_info['final_mask'])
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

            print(f"[SAM2 Batch] ✓ Summary visualization complete!")
            
        except Exception as e:
            print(f"[SAM2 Batch] Warning: Summary visualization failed: {e}")
            import traceback
            traceback.print_exc()
    
    return all_masks
