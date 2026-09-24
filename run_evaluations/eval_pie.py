import os, json, argparse, torch, numpy as np
import glob
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PIE_PAPER_SCALE_FACTORS = {
    # 论文/表格口径：
    # Structure Distance × 1e3, LPIPS × 1e3, MSE × 1e4, SSIM × 1e2
    "Structure Distance": 1000.0,
    "LPIPS": 1000.0,
    "MSE": 10000.0,
    "SSIM": 100.0,
}

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
        return dict(source)
    if view_mode == "user_study":
        if isinstance(user_study, dict) and len(user_study) > 0:
            return dict(user_merged)
        return dict(source)
    if view_mode in {"modified", "effective"}:
        if isinstance(modified, dict) and len(modified) > 0:
            return dict(merged)
        return dict(source)
    return dict(source)


def _decode_rle_mask(encoded_mask, image_shape=(512, 512)):
    """
    兼容 PIE-Bench 原始 JSON 的 RLE mask（[start, len, start, len, ...]）。
    """
    if not isinstance(encoded_mask, (list, tuple)) or len(encoded_mask) < 2:
        return np.zeros(image_shape, dtype=np.float32)
    if len(encoded_mask) % 2 != 0:
        encoded_mask = encoded_mask[:-1]

    h, w = image_shape
    length = h * w
    mask_flat = np.zeros((length,), dtype=np.float32)
    for i in range(0, len(encoded_mask), 2):
        start = int(encoded_mask[i])
        run = int(encoded_mask[i + 1])
        if run <= 0 or start >= length:
            continue
        if start < 0:
            run += start
            start = 0
        if run <= 0:
            continue
        end = min(start + run, length)
        mask_flat[start:end] = 1.0
    return mask_flat.reshape(h, w)


def _apply_mask_boundary_fix(mask_np):
    """
    与 PIE 原评估保持一致：边界全设为编辑区域，规避标注边界误差。
    """
    mask_np = np.array(mask_np, dtype=np.float32)
    if mask_np.ndim != 2:
        return mask_np
    if mask_np.shape[0] > 0:
        mask_np[0, :] = 1.0
        mask_np[-1, :] = 1.0
    if mask_np.shape[1] > 0:
        mask_np[:, 0] = 1.0
        mask_np[:, -1] = 1.0
    return mask_np


def _load_eval_mask(item, data_root):
    """
    优先读取 mask 图片；若不存在则回退到 JSON 内嵌 RLE（字段: mask）。
    """
    mask_rel = item.get("pie_mask_path", "") or item.get("mask_path", "")
    if mask_rel:
        mask_path = os.path.join(data_root, mask_rel)
        if os.path.exists(mask_path):
            mask_pil = Image.open(mask_path).convert("L").resize((512, 512), Image.NEAREST)
            mask_np = (np.array(mask_pil) > 127).astype(np.float32)
            return _apply_mask_boundary_fix(mask_np)

    if isinstance(item.get("mask"), (list, tuple)):
        mask_np = _decode_rle_mask(item.get("mask"), image_shape=(512, 512))
        return _apply_mask_boundary_fix(mask_np)

    return np.zeros((512, 512), dtype=np.float32)


def _load_target_eval_image(tgt_path):
    """
    与 PnP 口径一致：若目标图非方形，取右下角 512x512 再缩放到 512。
    """
    tgt_pil = Image.open(tgt_path).convert("RGB")
    if tgt_pil.size[0] != tgt_pil.size[1]:
        right, bottom = tgt_pil.size
        left = max(0, right - 512)
        top = max(0, bottom - 512)
        tgt_pil = tgt_pil.crop((left, top, right, bottom))
    return tgt_pil.resize((512, 512), Image.BILINEAR)


def _apply_pie_paper_scales(metric_dict):
    out = dict(metric_dict)
    for key, factor in PIE_PAPER_SCALE_FACTORS.items():
        val = out.get(key, None)
        if val is None:
            continue
        if isinstance(val, (int, float)) and np.isfinite(val):
            out[key] = float(val * factor)
    return out


def _empty_eval_stats():
    return {
        "files_seen": 0,
        "mapping_hits": 0,
        "mapping_miss": 0,
        "empty_image_path": 0,
        "missing_source_image": 0,
        "missing_result_image": 0,
        "eval_success": 0,
        "eval_error": 0,
    }


def _merge_eval_stats(all_stats, shard_stats):
    for key in all_stats.keys():
        all_stats[key] += int(shard_stats.get(key, 0))
    return all_stats


def _load_editing_seconds(exp_root, img_files):
    """
    从实验目录读取生成阶段记录的单图编辑耗时，返回:
    (mean_sec, matched_count, available_count, source_path)
    """
    candidate_paths = sorted(glob.glob(os.path.join(exp_root, "editing_times*.json")))
    if not candidate_paths:
        fallback = os.path.join(exp_root, "editing_times.json")
        return None, 0, 0, fallback

    time_map = {}
    used_paths = []
    for src_path in candidate_paths:
        if not os.path.isfile(src_path):
            continue
        try:
            with open(src_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue

        raw_map = {}
        if isinstance(payload, dict):
            for cand in (
                payload.get("image_times_sec"),
                payload.get("image_times"),
                payload.get("per_image_sec"),
                payload,
            ):
                if isinstance(cand, dict):
                    raw_map = cand
                    break
        if not raw_map:
            continue

        local_map = {}
        for k, v in raw_map.items():
            try:
                sec = float(v)
            except Exception:
                continue
            if np.isfinite(sec):
                local_map[str(k)] = sec
        if not local_map:
            continue

        # 后加载文件可覆盖同 id，便于修复重跑样本。
        time_map.update(local_map)
        used_paths.append(src_path)

    if not time_map:
        fallback = candidate_paths[0]
        return None, 0, 0, fallback

    stems = {os.path.splitext(os.path.basename(name))[0] for name in (img_files or [])}
    matched_vals = [time_map[s] for s in stems if s in time_map]
    if not matched_vals:
        matched_vals = list(time_map.values())
    source_desc = ",".join(used_paths) if used_paths else candidate_paths[0]
    return float(np.mean(matched_vals)), int(len(matched_vals)), int(len(time_map)), source_desc


def _load_run_hyperparams(res_path_norm, args):
    """
    从实验目录读取本次运行超参数，并补充评估侧参数。
    返回可直接写入 JSON 的字典。
    """
    base_name = os.path.basename(res_path_norm)
    exp_root = os.path.dirname(res_path_norm) if base_name == "results" else res_path_norm
    cfg_path = os.path.join(exp_root, "experiment_config.json")

    run_cfg = {}
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                run_cfg = json.load(f)
        except Exception:
            run_cfg = {}

    eval_args = {
        "annotation_view": args.annotation_view,
        "seed": int(args.seed),
        "metrics_as_percent": bool(args.metrics_as_percent),
        "paper_scaled_output": True,
        "device": [int(v) for v in args.device],
    }
    code_project_root = str(run_cfg.get("code_project_root") or Path(__file__).resolve().parents[1])
    code_project_name = str(run_cfg.get("code_project_name") or os.path.basename(code_project_root))
    code_entry_script = str(run_cfg.get("code_entry_script") or "")

    return {
        "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": exp_root,
        "experiment_config_path": cfg_path,
        "experiment_config": run_cfg,
        "code_project_name": code_project_name,
        "code_project_root": code_project_root,
        "code_entry_script": code_entry_script,
        "code_eval_script": str(Path(__file__).resolve()),
        "evaluation": eval_args,
    }


def evaluate_shard(gpu_id, img_list, img_dir, data_root, mapping, annotation_view, return_dict):
    device = f"cuda:{gpu_id}"
    try:
        from run_evaluations.metrics_utils_piebench import MetricsCalculator
        calc = MetricsCalculator(device)
    except Exception as e:
        print(f"GPU {gpu_id} 模型加载失败: {e}")
        return

    shard_results = []
    shard_stats = _empty_eval_stats()
    
    # 路径检查标志
    check_first = True

    for fname in tqdm(img_list, desc=f"GPU {gpu_id}"):
        shard_stats["files_seen"] += 1
        img_id = os.path.splitext(fname)[0]
        if img_id not in mapping:
            shard_stats["mapping_miss"] += 1
            continue
        shard_stats["mapping_hits"] += 1
        
        item = _resolve_annotation_entry(mapping[img_id], view_mode=annotation_view)
        image_rel = item.get("image_path", "")
        if not image_rel:
            shard_stats["empty_image_path"] += 1
            continue

        src_path = os.path.join(data_root, image_rel)
        tgt_path = os.path.join(img_dir, fname)
        if check_first and gpu_id == 0:
            print(f"\n[路径检查] 目标文件: {tgt_path} (存在: {os.path.exists(tgt_path)})")
            check_first = False

        if not os.path.exists(src_path):
            shard_stats["missing_source_image"] += 1
            continue
        if not os.path.exists(tgt_path):
            shard_stats["missing_result_image"] += 1
            continue
        
        try:
            src_pil = Image.open(src_path).convert("RGB").resize((512, 512), Image.BILINEAR)
            tgt_pil = _load_target_eval_image(tgt_path)
            mask_np = _load_eval_mask(item, data_root)
            
            # 计算全套指标
            target_prompt = str(item.get("target_prompt", "")).replace("[","").replace("]","")
            res = calc.calculate_pie_metrics(
                src_pil, tgt_pil, mask_np, 
                target_prompt
            )
            shard_results.append(res)
            shard_stats["eval_success"] += 1
        except Exception:
            shard_stats["eval_error"] += 1
            continue

    return_dict[gpu_id] = {"results": shard_results, "stats": shard_stats}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dirs', nargs='+', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--mapping_file', required=True)
    parser.add_argument('--mode', default="Text")
    parser.add_argument('--output_json', default="metrics_piebench.json")
    parser.add_argument('--device', nargs='+', type=int, default=[0])
    parser.add_argument(
        '--annotation_view',
        choices=['effective', 'source', 'modified', 'user_study'],
        default='modified',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--metrics-as-percent', action='store_true', help='兼容参数：保留但已默认按论文倍率写入输出。')
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)
    with open(args.mapping_file, 'r') as f: mapping = json.load(f)

    for res_path in args.result_dirs:
        # --- 核心修改部分：提取更具辨识度的实验名称 ---
        res_path_norm = os.path.normpath(res_path)
        base_name = os.path.basename(res_path_norm)
        
        # 如果路径以 "results" 结尾，则取其父目录名（如 PIE-Bench-20251226_114417）
        if base_name == "results":
            exp_name = os.path.basename(os.path.dirname(res_path_norm))
            exp_root = os.path.dirname(res_path_norm)
        else:
            exp_name = base_name
            exp_root = res_path_norm

        # 确定图片读取目录
        img_dir = res_path_norm if base_name == "results" else os.path.join(res_path_norm, "results")
        if not os.path.exists(img_dir):
            img_dir = res_path_norm # 回退方案：如果不存在results子目录，直接用当前目录
            
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png','.jpg'))]
        
        if not img_files:
            print(f"警告: {img_dir} 中未发现图片。")
            continue

        print(f"\n开始评估实验: {exp_name}")
        eval_start = datetime.now()
        shards = np.array_split(img_files, len(args.device))
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []

        for i, gpu_id in enumerate(args.device):
            p = mp.Process(
                target=evaluate_shard,
                args=(
                    gpu_id,
                    shards[i].tolist(),
                    img_dir,
                    args.data_root,
                    mapping,
                    args.annotation_view,
                    return_dict,
                ),
            )
            p.start(); processes.append(p)
        for p in processes: p.join()

        all_res = []
        all_stats = _empty_eval_stats()
        for v in return_dict.values():
            if isinstance(v, dict):
                all_res.extend(v.get("results", []))
                all_stats = _merge_eval_stats(all_stats, v.get("stats", {}))
            elif isinstance(v, list):
                # 兼容旧格式
                all_res.extend(v)
        eval_end = datetime.now()
        eval_duration_sec = float((eval_end - eval_start).total_seconds())
        
        if all_res:
            # 计算平均值
            keys = list(all_res[0].keys())
            avg_raw = {}
            for k in keys:
                vals = [r[k] for r in all_res if k in r and not np.isnan(r[k])]
                avg_raw[k] = float(np.mean(vals)) if vals else 0.0
            editing_sec, editing_cnt, editing_total, editing_src = _load_editing_seconds(exp_root, img_files)
            if editing_sec is None:
                editing_sec = 0.0
            avg_raw["Editing(s)"] = float(editing_sec)
            avg = _apply_pie_paper_scales(avg_raw)
            # 保证 Editing(s) 是最后一个指标
            avg["Editing(s)"] = float(editing_sec)
            run_hparams = _load_run_hyperparams(res_path_norm, args)
            run_hparams["evaluation_started_at"] = eval_start.strftime("%Y-%m-%d %H:%M:%S")
            run_hparams["evaluation_finished_at"] = eval_end.strftime("%Y-%m-%d %H:%M:%S")
            run_hparams["evaluation_duration_sec"] = eval_duration_sec
            
            # 写入汇总 JSON
            db = {}
            if os.path.exists(args.output_json):
                try: 
                    with open(args.output_json, 'r') as f:
                        db = json.load(f)
                except: db = {}
            
            if args.mode not in db: db[args.mode] = {}
            
            # 这里记录的就是类似 "PIE-Bench-20251226_114417" 的 Key
            entry = dict(avg)
            entry["_raw_metrics"] = avg_raw
            entry["_scale_factors"] = dict(PIE_PAPER_SCALE_FACTORS)
            entry["_eval_stats"] = {
                "images_in_result_dir": int(len(img_files)),
                "images_evaluated": int(len(all_res)),
                "coverage_ratio": float(len(all_res) / max(1, len(img_files))),
                "editing_time_source": editing_src,
                "editing_time_matched": int(editing_cnt),
                "editing_time_available": int(editing_total),
                **all_stats,
            }
            entry["_hyperparams"] = run_hparams
            db[args.mode][exp_name] = entry
            
            with open(args.output_json, 'w') as f:
                json.dump(db, f, indent=4, ensure_ascii=False)

            # 打印结果
            print(f"\n✅ 实验汇总: {exp_name} ({len(all_res)} 张图)")
            print("   [论文口径] Distance×1e3, LPIPS×1e3, MSE×1e4, SSIM×1e2（已写入 JSON）。")
            print(
                f"   [覆盖统计] result图数={len(img_files)}, 评估成功={len(all_res)}, "
                f"coverage={len(all_res)/max(1,len(img_files)):.3f}, 异常={all_stats['eval_error']}"
            )
            print(f"   [速度] Editing(s)={avg['Editing(s)']:.4f} (matched={editing_cnt}, source={editing_src})")
            distance_col = "Distance×1e3"
            lpips_col = "LPIPS×1e3"
            mse_col = "MSE×1e4"
            ssim_col = "SSIM×1e2"
            print("-" * 135)
            print(f"{distance_col:^18} | {'PSNR':^10} | {lpips_col:^10} | {mse_col:^10} | {ssim_col:^10} | {'CLIP-W':^15} | {'CLIP-E':^15}")
            print("-" * 135)
            print(f"{avg['Structure Distance']:^18.2f} | {avg['PSNR']:^10.2f} | {avg['LPIPS']:^10.2f} | {avg['MSE']:^10.2f} | {avg['SSIM']:^10.2f} | {avg['CLIP Similarity Whole']:^15.2f} | {avg['CLIP Similarity Edited']:^15.2f}")
            print("-" * 135 + "\n")

if __name__ == "__main__":
    main()
