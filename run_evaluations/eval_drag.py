import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image
from torchvision.transforms import PILToTensor
from tqdm import tqdm

# Allow both `python -m run_evaluations.eval_drag` and direct script execution.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tdedit_paths import EVAL_SD_PATH, PROJECT_ROOT, configured_path

DRAG_PERCENT_KEYS = {"LPIPS", "1-LPIPS", "CLIP_Sim"}


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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _set_torch_seed(seed: int) -> None:
    """
    仅设置 torch RNG，用于每样本固定 DIFT 随机噪声。
    """
    seed = int(seed) & 0x7FFFFFFF
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _build_md_sample_seed(base_seed: int, image_id: str, branch: str) -> int:
    """
    为每个样本/分支构造稳定 seed，避免多进程分片导致 MD 漂移。
    """
    key = f"{int(base_seed)}::{image_id}::{branch}".encode("utf-8")
    # 使用稳定哈希，不依赖 Python 进程 hash 随机化。
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % 2147483647


def _convert_drag_metrics_to_percent(metric_dict):
    out = dict(metric_dict)
    for key in DRAG_PERCENT_KEYS:
        val = out.get(key, None)
        if val is None:
            continue
        if isinstance(val, (int, float)) and np.isfinite(val):
            out[key] = float(val * 100.0)
    return out


def _empty_eval_stats():
    return {
        "files_seen": 0,
        "mapping_hits": 0,
        "mapping_miss": 0,
        "empty_image_path": 0,
        "missing_source_image": 0,
        "missing_result_image": 0,
        "sim_success": 0,
        "points_missing": 0,
        "point_pairs_used": 0,
        "point_distance_count": 0,
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
    src_path = os.path.join(exp_root, "editing_times.json")
    if not os.path.isfile(src_path):
        return None, 0, 0, src_path

    try:
        with open(src_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None, 0, 0, src_path

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

    time_map = {}
    for k, v in raw_map.items():
        try:
            sec = float(v)
        except Exception:
            continue
        if np.isfinite(sec):
            time_map[str(k)] = sec

    if not time_map:
        return None, 0, 0, src_path

    stems = {os.path.splitext(os.path.basename(name))[0] for name in (img_files or [])}
    matched_vals = [time_map[s] for s in stems if s in time_map]
    if not matched_vals:
        matched_vals = list(time_map.values())

    return float(np.mean(matched_vals)), int(len(matched_vals)), int(len(time_map)), src_path


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
        "records_one_minus_lpips": True,
        "md_impl": args.md_impl,
        "device": [int(v) for v in args.device],
    }
    code_project_root = str(Path(__file__).resolve().parents[1])
    code_project_name = os.path.basename(code_project_root)

    return {
        "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": exp_root,
        "experiment_config_path": cfg_path,
        "experiment_config": run_cfg,
        "code_project_name": code_project_name,
        "code_project_root": code_project_root,
        "code_eval_script": str(Path(__file__).resolve()),
        "evaluation": eval_args,
    }


def _build_md_featurizer(sd_path: str, md_impl: str, device: str = "cuda"):
    """
    MD 特征提取实现：
    - fastdrag: 优先对齐 FastDrag 原版 dift_sd.py
    - tdedit: 使用当前文件内 SDFeaturizer 兼容实现
    - auto: 先 fastdrag，失败回退 tdedit
    """
    impl = (md_impl or "auto").strip().lower()
    if impl not in {"auto", "fastdrag", "tdedit"}:
        raise ValueError(f"Unsupported md_impl: {md_impl}")

    def _build_fastdrag_impl():
        fastdrag_eval_dir = Path(configured_path(
            "TDEDIT_FASTDRAG_EVAL_DIR",
            PROJECT_ROOT / "third_party" / "FastDrag" / "drag_bench_evaluation",
        ))
        if not fastdrag_eval_dir.is_dir():
            raise FileNotFoundError(f"FastDrag evaluation dir not found: {fastdrag_eval_dir}")
        fastdrag_eval_dir_s = str(fastdrag_eval_dir)
        if fastdrag_eval_dir_s not in sys.path:
            sys.path.insert(0, fastdrag_eval_dir_s)
        from dift_sd import SDFeaturizer as FastDragSDFeaturizer

        return FastDragSDFeaturizer(sd_id=sd_path)

    if impl == "fastdrag":
        return _build_fastdrag_impl(), "fastdrag"
    if impl == "tdedit":
        return SDFeaturizer(sd_path, device), "tdedit"

    # auto
    try:
        return _build_fastdrag_impl(), "fastdrag"
    except Exception as e:
        print(f"[eval_drag] fastdrag md_impl unavailable, fallback to tdedit: {e}")
        return SDFeaturizer(sd_path, device), "tdedit"


# --- 核心组件：MyUNet (对齐原始代码的动态尺寸对齐逻辑) ---
class MyUNet(UNet2DConditionModel):
    def forward(self, sample, timestep, up_ft_indices, encoder_hidden_states, **kwargs):
        dtype = sample.dtype
        # 计算是否需要动态尺寸对齐
        default_overall_up_factor = 2**self.num_upsamplers
        forward_upsample_size = any(s % default_overall_up_factor != 0 for s in sample.shape[-2:])

        # 1. Time embedding
        t_emb = self.time_embedding(
            self.time_proj(timestep.expand(sample.shape[0])).to(dtype=dtype)
        )
        # 2. Pre-process
        sample = self.conv_in(sample)
        down_block_res_samples = (sample,)
        # 3. Down blocks
        for block in self.down_blocks:
            if hasattr(block, "has_cross_attention") and block.has_cross_attention:
                sample, res = block(sample, t_emb, encoder_hidden_states=encoder_hidden_states)
            else:
                sample, res = block(sample, t_emb)
            down_block_res_samples += res
        # 4. Mid block
        if self.mid_block:
            sample = self.mid_block(sample, t_emb, encoder_hidden_states=encoder_hidden_states)
        # 5. Up blocks
        up_ft = {}
        for i, block in enumerate(self.up_blocks):
            if i > max(up_ft_indices): break
            
            res_samples = down_block_res_samples[-len(block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(block.resnets)]
            
            # 动态计算 upsample_size (解决 23 vs 24 拼接报错的关键)
            upsample_size = None
            if forward_upsample_size and i < len(self.up_blocks) - 1:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(block, "has_cross_attention") and block.has_cross_attention:
                sample = block(
                    hidden_states=sample,
                    res_hidden_states_tuple=res_samples,
                    temb=t_emb,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size
                )
            else:
                sample = block(
                    hidden_states=sample,
                    res_hidden_states_tuple=res_samples,
                    temb=t_emb,
                    upsample_size=upsample_size
                )
            if i in up_ft_indices:
                up_ft[i] = sample.detach()
        return {'up_ft': up_ft}

class SDFeaturizer:
    def __init__(self, sd_id, device):
        self.device = device
        unet = MyUNet.from_pretrained(sd_id, subfolder="unet")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            sd_id, unet=unet, safety_checker=None, torch_dtype=torch.float16
        ).to(device)
        self.pipe.unet.to(torch.float16)
        self.pipe.vae.to(torch.float16)
        self.pipe.scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")

    @torch.no_grad()
    def forward(self, img_pil, prompt, t=261, up_idx=1, seed=None):
        # 必须是 8 的倍数以适配 VAE
        W, H = img_pil.size
        new_W, new_H = (W // 8) * 8, (H // 8) * 8
        if new_W != W or new_H != H:
            img_pil = img_pil.resize((new_W, new_H), Image.BILINEAR)

        img_ts = (PILToTensor()(img_pil).unsqueeze(0).to(self.device).to(torch.float16) / 255.0 - 0.5) * 2
        img_ts = img_ts.repeat(8, 1, 1, 1) # ensemble_size=8
        
        prompt_outputs = self.pipe.encode_prompt(prompt, self.device, 1, False)
        emb = prompt_outputs[0].to(torch.float16).repeat(8, 1, 1)
        
        latents = self.pipe.vae.encode(img_ts).latent_dist.sample() * self.pipe.vae.config.scaling_factor
        t_ts = torch.tensor([t], device=self.device).long()
        if seed is None:
            noise = torch.randn_like(latents)
        else:
            generator = torch.Generator(device=latents.device)
            generator.manual_seed(int(seed))
            noise = torch.randn(
                latents.shape,
                device=latents.device,
                dtype=latents.dtype,
                generator=generator,
            )
        noisy = self.pipe.scheduler.add_noise(latents, noise, t_ts)
        
        out = self.pipe.unet(noisy, t_ts, [up_idx], encoder_hidden_states=emb)
        return out['up_ft'][up_idx].mean(0, keepdim=True)


def _split_pairs(points: List[List[float]]) -> List[Tuple[List[float], List[float]]]:
    """
    当前数据逻辑：控制点全部按 [handle, target, handle, target, ...] 成对保存。
    """
    if not isinstance(points, list) or len(points) < 2:
        return []

    # 偶数点才是完整配对；若意外出现奇数，丢弃最后一个孤立点
    usable = len(points) if len(points) % 2 == 0 else len(points) - 1
    pairs = []
    for i in range(0, usable, 2):
        hp = points[i]
        tp = points[i + 1]
        if (
            isinstance(hp, (list, tuple)) and len(hp) >= 2 and
            isinstance(tp, (list, tuple)) and len(tp) >= 2
        ):
            pairs.append((hp, tp))
    return pairs


def drag_shard_worker(
    gpu_id,
    img_list,
    img_dir,
    data_root,
    mapping,
    sd_path,
    md_impl,
    annotation_view,
    seed,
    return_dict,
):
    device = f"cuda:{gpu_id}"
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    _set_seed(int(seed))
    try:
        featurizer, md_impl_used = _build_md_featurizer(sd_path, md_impl, device=device)
        from run_evaluations.metrics_utils_dragbench import MetricsCalculator
        calc = MetricsCalculator(device)
    except Exception as e:
        print(f"GPU {gpu_id} 加载失败: {e}"); return

    cos_sim = torch.nn.CosineSimilarity(dim=1)
    shard_point_dists: List[float] = []
    shard_lpips: List[float] = []
    shard_clip: List[float] = []
    shard_per_image = {}
    shard_stats = _empty_eval_stats()

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
        if not os.path.exists(src_path):
            shard_stats["missing_source_image"] += 1
            continue
        if not os.path.exists(tgt_path):
            shard_stats["missing_result_image"] += 1
            continue

        try:
            src_pil = Image.open(src_path).convert("RGB")
            W, H = src_pil.size
            tgt_pil = Image.open(tgt_path).convert("RGB").resize((W, H), Image.BILINEAR)
            
            # 1) Similarity（与 DragDiffusion 一致，按图像全局平均）
            lpips_v, clip_v = calc.calculate_drag_similarity(src_pil, tgt_pil)
            shard_lpips.append(lpips_v)
            shard_clip.append(clip_v)
            shard_stats["sim_success"] += 1
            
            # 2) Mean Distance（按点全局平均，与 DragDiffusion 一致）
            pairs = _split_pairs(item.get("points", []))
            if not pairs:
                shard_stats["points_missing"] += 1
                continue

            prompt = str(item.get("source_prompt", "") or item.get("target_prompt", ""))
            prompt = prompt.replace("[", "").replace("]", "").strip()
            src_seed = _build_md_sample_seed(seed, img_id, "source")
            tgt_seed = _build_md_sample_seed(seed, img_id, "target")

            if md_impl_used == "fastdrag":
                src_tensor = (PILToTensor()(src_pil) / 255.0 - 0.5) * 2
                tgt_tensor = (PILToTensor()(tgt_pil) / 255.0 - 0.5) * 2
                _set_torch_seed(src_seed)
                ft_s = F.interpolate(
                    featurizer.forward(
                        src_tensor,
                        prompt=prompt,
                        t=261,
                        up_ft_index=1,
                        ensemble_size=8,
                    ),
                    (H, W),
                    mode="bilinear",
                )
                _set_torch_seed(tgt_seed)
                ft_t = F.interpolate(
                    featurizer.forward(
                        tgt_tensor,
                        prompt=prompt,
                        t=261,
                        up_ft_index=1,
                        ensemble_size=8,
                    ),
                    (H, W),
                    mode="bilinear",
                )
            else:
                ft_s = F.interpolate(
                    featurizer.forward(src_pil, prompt, seed=src_seed),
                    (H, W),
                    mode="bilinear",
                )
                ft_t = F.interpolate(
                    featurizer.forward(tgt_pil, prompt, seed=tgt_seed),
                    (H, W),
                    mode="bilinear",
                )

            valid_pair_count = 0
            image_point_dists: List[float] = []
            for hp_xy, tp_xy in pairs:
                hp_rc = [int(hp_xy[1]), int(hp_xy[0])]
                tp_rc = torch.tensor([float(tp_xy[1]), float(tp_xy[0])], dtype=torch.float32)

                if hp_rc[0] < 0 or hp_rc[1] < 0 or hp_rc[0] >= H or hp_rc[1] >= W:
                    continue
                src_vec = ft_s[0, :, hp_rc[0], hp_rc[1]].view(1, -1, 1, 1)
                sim_map = cos_sim(src_vec, ft_t).cpu().numpy()[0]
                max_rc = np.unravel_index(sim_map.argmax(), sim_map.shape)
                
                dist = torch.norm(tp_rc - torch.tensor(max_rc, dtype=torch.float32)).item()
                shard_point_dists.append(dist)
                image_point_dists.append(float(dist))
                valid_pair_count += 1
                shard_stats["point_distance_count"] += 1

            shard_stats["point_pairs_used"] += valid_pair_count
            if image_point_dists:
                shard_per_image[str(img_id)] = {
                    "MD": float(np.mean(image_point_dists)),
                    "point_distances": image_point_dists,
                    "LPIPS": float(lpips_v),
                    "1-LPIPS": float(1.0 - lpips_v),
                    "CLIP_Sim": float(clip_v),
                    "point_pair_count": int(valid_pair_count),
                }
        except Exception:
            shard_stats["eval_error"] += 1
            continue

    return_dict[gpu_id] = {
        "point_dists": shard_point_dists,
        "lpips": shard_lpips,
        "clip": shard_clip,
        "per_image": shard_per_image,
        "stats": shard_stats,
        "md_impl": md_impl_used,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dirs', nargs='+', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--mapping_file', required=True)
    parser.add_argument('--mode', default="Drag") 
    parser.add_argument('--output_json', default="metrics_dragbench.json")
    parser.add_argument('--sd_path', default=EVAL_SD_PATH)
    parser.add_argument('--device', nargs='+', type=int, default=[0])
    parser.add_argument(
        '--annotation_view',
        choices=['effective', 'source', 'modified', 'user_study'],
        default='modified',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--metrics-as-percent', action='store_true')
    parser.add_argument('--md-impl', choices=['auto', 'fastdrag', 'tdedit'], default='auto')
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)
    with open(args.mapping_file, 'r') as f: mapping = json.load(f)

    for res_path in args.result_dirs:
        res_path_norm = os.path.normpath(res_path)
        base_name = os.path.basename(res_path_norm)
        exp_name = os.path.basename(os.path.dirname(res_path_norm)) if base_name == "results" else base_name
        exp_root = os.path.dirname(res_path_norm) if base_name == "results" else res_path_norm
        img_dir = res_path_norm if base_name == "results" else os.path.join(res_path_norm, "results")
        
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg'))]
        if not img_files: continue

        print(f"\n🚀 评估实验: {exp_name} ({len(img_files)} imgs)")
        eval_start = datetime.now()
        shards = np.array_split(img_files, len(args.device))
        manager = mp.Manager(); return_dict = manager.dict(); processes = []

        for i, gpu_id in enumerate(args.device):
            p = mp.Process(
                target=drag_shard_worker,
                args=(
                    gpu_id,
                    shards[i].tolist(),
                    img_dir,
                    args.data_root,
                    mapping,
                    args.sd_path,
                    args.md_impl,
                    args.annotation_view,
                    args.seed,
                    return_dict,
                ),
            )
            p.start(); processes.append(p)
        for p in processes: p.join()

        all_point_dists: List[float] = []
        all_lpips: List[float] = []
        all_clip: List[float] = []
        all_per_image = {}
        all_stats = _empty_eval_stats()
        md_impl_used = None
        for shard_data in return_dict.values():
            if not isinstance(shard_data, dict):
                continue
            all_point_dists.extend(shard_data.get("point_dists", []))
            all_lpips.extend(shard_data.get("lpips", []))
            all_clip.extend(shard_data.get("clip", []))
            all_per_image.update(shard_data.get("per_image", {}))
            all_stats = _merge_eval_stats(all_stats, shard_data.get("stats", {}))
            if md_impl_used is None and isinstance(shard_data.get("md_impl"), str):
                md_impl_used = shard_data.get("md_impl")
        eval_end = datetime.now()
        eval_duration_sec = float((eval_end - eval_start).total_seconds())
        if md_impl_used is None:
            md_impl_used = args.md_impl

        if not all_lpips or not all_clip:
            print(f"⚠️ {exp_name} 没有有效样本，跳过写入。")
            continue
        if not all_point_dists:
            print(f"⚠️ {exp_name} 没有有效点对，无法计算 MD，跳过写入。")
            continue

        avg_md = float(np.mean(all_point_dists))
        avg_lpips = float(np.mean(all_lpips))
        avg_clip = float(np.mean(all_clip))
        output_metrics_raw = {
            "MD": avg_md,
            "LPIPS": avg_lpips,
            "1-LPIPS": float(1.0 - avg_lpips),
            "CLIP_Sim": avg_clip,
        }
        editing_sec, editing_cnt, editing_total, editing_src = _load_editing_seconds(exp_root, img_files)
        if editing_sec is None:
            editing_sec = 0.0
        # 保证 Editing(s) 是最后一个指标
        output_metrics_raw["Editing(s)"] = float(editing_sec)
        output_metrics = dict(output_metrics_raw)
        if args.metrics_as_percent:
            output_metrics = _convert_drag_metrics_to_percent(output_metrics)
        run_hparams = _load_run_hyperparams(res_path_norm, args)
        run_hparams["evaluation_started_at"] = eval_start.strftime("%Y-%m-%d %H:%M:%S")
        run_hparams["evaluation_finished_at"] = eval_end.strftime("%Y-%m-%d %H:%M:%S")
        run_hparams["evaluation_duration_sec"] = eval_duration_sec
        run_hparams["md_impl_used"] = md_impl_used

        db = {}
        if os.path.exists(args.output_json):
            try:
                with open(args.output_json, 'r') as f:
                    db = json.load(f)
            except Exception:
                db = {}
        if args.mode not in db:
            db[args.mode] = {}
        entry = dict(output_metrics)
        entry["_raw_metrics"] = output_metrics_raw
        entry["_eval_stats"] = {
            "images_in_result_dir": int(len(img_files)),
            "images_with_similarity": int(len(all_lpips)),
            "point_distance_count": int(len(all_point_dists)),
            "coverage_similarity_ratio": float(len(all_lpips) / max(1, len(img_files))),
            "editing_time_source": editing_src,
            "editing_time_matched": int(editing_cnt),
            "editing_time_available": int(editing_total),
            **all_stats,
        }
        entry["_per_image"] = dict(sorted(all_per_image.items(), key=lambda kv: kv[0]))
        entry["_hyperparams"] = run_hparams
        db[args.mode][exp_name] = entry
        with open(args.output_json, 'w') as f:
            json.dump(db, f, indent=4, ensure_ascii=False)
        if args.metrics_as_percent:
            print("   [百分比模式] LPIPS/1-LPIPS/CLIP 指标已转换到 0-100。")
        print(f"   [MD实现] {md_impl_used}")
        print(
            f"   [覆盖统计] result图数={len(img_files)}, similarity样本={len(all_lpips)}, "
            f"point-distance数={len(all_point_dists)}, 异常={all_stats['eval_error']}"
        )
        print(f"   [速度] Editing(s)={output_metrics['Editing(s)']:.4f} (matched={editing_cnt}, source={editing_src})")
        print(
            f"✅ {exp_name} | MD: {output_metrics['MD']:.2f} | "
            f"1-LPIPS: {output_metrics['1-LPIPS']:.4f} | "
            f"LPIPS: {output_metrics['LPIPS']:.4f} | CLIP: {output_metrics['CLIP_Sim']:.4f} | "
            f"Editing(s): {output_metrics['Editing(s)']:.4f}"
        )

if __name__ == "__main__":
    main()
