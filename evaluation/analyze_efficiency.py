#!/usr/bin/env python3
"""Aggregate synchronized, matched-sample efficiency measurements."""

import argparse
import json
from pathlib import Path

import numpy as np


def mean_ci(values, seed=20260825, n_boot=10000):
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    draws = arr[idx].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)),
        "ci95": [float(v) for v in np.quantile(draws, [0.025, 0.975])],
        "n": int(len(arr)),
    }


def get_values(payload, ids, field="image_times_sec"):
    table = payload.get(field, {})
    missing = [sample_id for sample_id in ids if sample_id not in table]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} samples from {field}: {missing[:3]}")
    return np.asarray([float(table[sample_id]) for sample_id in ids], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tdedit", required=True)
    parser.add_argument("--fastdrag-steady", required=True)
    parser.add_argument("--fastdrag-cold")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    td = json.loads(Path(args.tdedit).read_text(encoding="utf-8"))
    fd = json.loads(Path(args.fastdrag_steady).read_text(encoding="utf-8"))
    fd_cold = (json.loads(Path(args.fastdrag_cold).read_text(encoding="utf-8"))
               if args.fastdrag_cold else None)
    ids = [row["sample_id"] for row in manifest["samples"] if row["role"] == "measured"]
    meta = {row["sample_id"]: row for row in manifest["samples"]}

    td_t = get_values(td, ids)
    fd_t = get_values(fd, ids)
    fd_cold_t = get_values(fd_cold, ids) if fd_cold is not None else None
    td_mem = get_values(td, ids, "peak_memory_allocated_bytes") / (1024 ** 3)
    fd_mem = get_values(fd, ids, "peak_memory_allocated_bytes") / (1024 ** 3)

    rng = np.random.default_rng(20260825)
    idx = rng.integers(0, len(ids), size=(10000, len(ids)))
    speed_draws = fd_t[idx].mean(axis=1) / td_t[idx].mean(axis=1)
    speedup = {
        "ratio_of_mean_latencies": float(fd_t.mean() / td_t.mean()),
        "ci95": [float(v) for v in np.quantile(speed_draws, [0.025, 0.975])],
        "positive_means_tdedit_faster": True,
    }

    load_times = td.get("summary", {}).get("model_load_seconds_by_worker", [])
    td_first_warmup = next(row["sample_id"] for row in manifest["samples"] if row["role"] == "warmup")
    td_cold_start = (float(load_times[0]) if load_times else 0.0) + float(td["image_times_sec"][td_first_warmup])
    stage_table = td.get("stage_times_sec", {})
    stage_names = ["preparation_sec", "geometry_anchor_sec", "denoising_sec", "decode_postprocess_sec", "pipeline_total_sec"]
    stage_breakdown = {}
    for family in ("2D", "3D"):
        family_ids = [sample_id for sample_id in ids if str(meta[sample_id]["drag_type"]).startswith(family)]
        stage_breakdown[family] = {
            stage: mean_ci([stage_table[sample_id][stage] for sample_id in family_ids])
            for stage in stage_names
        }
    result = {
        "protocol": {
            "gpu": "NVIDIA GeForce RTX 4090",
            "resolution": "512x512",
            "batch_size": 1,
            "precision": "FP16",
            "timing": "CUDA synchronization at both boundaries",
            "native_steps": {"TDEdit": 17, "FastDrag": 10},
            "warmup": "first 2D and first 3D cases excluded",
            "measured_images": len(ids),
            "paired_samples": True,
        },
        "steady_state_latency_seconds": {
            "TDEdit": mean_ci(td_t),
            "FastDrag": mean_ci(fd_t),
            "FastDrag_over_TDEdit_speedup": speedup,
        },
        "peak_allocated_memory_gib": {
            "TDEdit": mean_ci(td_mem),
            "FastDrag": mean_ci(fd_mem),
            "TDEdit_max": float(td_mem.max()),
            "FastDrag_max": float(fd_mem.max()),
        },
        "cold_start_seconds": {
            "TDEdit_model_load_plus_first_edit": td_cold_start,
            "TDEdit_model_load_only": float(load_times[0]) if load_times else None,
            "FastDrag_original_reload_each_edit": mean_ci(fd_cold_t) if fd_cold_t is not None else None,
        },
        "TDEdit_stage_breakdown_seconds": stage_breakdown,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    td_s = result["steady_state_latency_seconds"]["TDEdit"]
    fd_s = result["steady_state_latency_seconds"]["FastDrag"]
    ratio = result["steady_state_latency_seconds"]["FastDrag_over_TDEdit_speedup"]
    lines = [
        "# Fair efficiency benchmark", "",
        "All latency values use matched 512×512 inputs, batch size 1, FP16, one RTX 4090, and CUDA synchronization.", "",
        "| Method | Native steps | Steady latency, s (95% CI) | Peak allocated memory, GiB (max) |",
        "|---|---:|---:|---:|",
        f"| TDEdit | 17 | {td_s['mean']:.3f} [{td_s['ci95'][0]:.3f}, {td_s['ci95'][1]:.3f}] | {result['peak_allocated_memory_gib']['TDEdit_max']:.2f} |",
        f"| FastDrag | 10 | {fd_s['mean']:.3f} [{fd_s['ci95'][0]:.3f}, {fd_s['ci95'][1]:.3f}] | {result['peak_allocated_memory_gib']['FastDrag_max']:.2f} |",
        "",
        f"FastDrag/TDEdit ratio of mean steady-state latency: **{ratio['ratio_of_mean_latencies']:.3f}×** "
        f"(95% CI {ratio['ci95'][0]:.3f}–{ratio['ci95'][1]:.3f}).",
    ]
    Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
