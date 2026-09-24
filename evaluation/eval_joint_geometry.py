#!/usr/bin/env python3
"""Per-case DIFT point-localization evaluation for joint editing."""

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import PILToTensor

def resolve(record):
    item = dict(record.get("source", {}))
    item.update(record.get("modified", {}))
    return item


def bootstrap_ci(values, seed=20260825, repetitions=10000):
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0:
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions)
    for start in range(0, repetitions, 500):
        n = min(500, repetitions - start)
        means[start:start + n] = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return {"mean": float(x.mean()), "ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))], "n": len(x)}


def feature(featurizer, image, prompt, seed, h, w):
    return F.interpolate(
        featurizer.forward(image, prompt, seed=seed),
        (h, w),
        mode="bilinear",
        align_corners=False,
    )


def source_cache_key(case_id, source_path, prompt, seed, sd_path, w, h, pairs):
    stat = source_path.stat()
    payload = {
        "case_id": case_id,
        "source_path": str(source_path.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "prompt": prompt,
        "feature_seed": seed,
        "sd_path": str(Path(sd_path).resolve()),
        "width": w,
        "height": h,
        "handles": [[int(handle[0]), int(handle[1])] for handle, _ in pairs],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def load_or_compute_source_vectors(cache_dir, cache_key, featurizer, source, prompt, seed, h, w, pairs):
    cache_path = cache_dir / f"{cache_key}.pt"
    if cache_path.is_file():
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            vectors = payload["vectors"]
            if vectors.ndim == 2 and vectors.shape[0] == len(pairs):
                return vectors.float(), True
        except Exception:
            pass

    source_ft = feature(featurizer, source, prompt, seed, h, w)
    vectors = []
    for handle, _ in pairs:
        hx, hy = int(handle[0]), int(handle[1])
        if not (0 <= hx < w and 0 <= hy < h):
            vectors.append(torch.zeros(source_ft.shape[1], dtype=torch.float32))
        else:
            vectors.append(source_ft[0, :, hy, hx].detach().cpu().float())
    vectors = torch.stack(vectors, dim=0)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(f".tmp.{os.getpid()}")
    torch.save({"vectors": vectors}, temporary)
    os.replace(temporary, cache_path)
    return vectors, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sd-path", required=True, help="Local Stable Diffusion 2.1 checkpoint directory")
    parser.add_argument("--eval-dir", type=Path, default=Path(__file__).resolve().parents[1] / "run_evaluations")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source-cache-dir",
        type=Path,
        default=Path(".cache/dift_source_handle_vectors"),
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.eval_dir.resolve()))
    from eval_drag import SDFeaturizer, _build_md_sample_seed, _set_seed, _split_pairs

    torch.cuda.set_device(torch.device(args.device))
    _set_seed(args.seed)
    featurizer = SDFeaturizer(args.sd_path, args.device)
    cosine = torch.nn.CosineSimilarity(dim=1)
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    result_images = args.result_dir / "results" if (args.result_dir / "results").is_dir() else args.result_dir
    cases, errors = [], []
    source_cache_hits = 0
    source_cache_misses = 0

    for case_id, record in mapping.items():
        item = resolve(record)
        source_path = args.data_root / item["image_path"]
        result_path = result_images / f"{case_id}.png"
        pairs = _split_pairs(item.get("points", []))
        if not source_path.is_file() or not result_path.is_file() or not pairs:
            errors.append({"case_id": case_id, "reason": "missing image/result/pairs"})
            continue
        try:
            source = Image.open(source_path).convert("RGB")
            w, h = source.size
            edited = Image.open(result_path).convert("RGB").resize((w, h), Image.Resampling.BILINEAR)
            prompt = str(item.get("source_prompt", "") or item.get("target_prompt", "")).replace("[", "").replace("]", "").strip()
            source_seed = _build_md_sample_seed(args.seed, case_id, "source")
            edited_seed = _build_md_sample_seed(args.seed, case_id, "target")
            cache_key = source_cache_key(case_id, source_path, prompt, source_seed, args.sd_path, w, h, pairs)
            source_vectors, cache_hit = load_or_compute_source_vectors(
                args.source_cache_dir, cache_key, featurizer, source, prompt, source_seed, h, w, pairs
            )
            source_cache_hits += int(cache_hit)
            source_cache_misses += int(not cache_hit)
            edited_ft = feature(featurizer, edited, prompt, edited_seed, h, w)
            distances = []
            predictions = []
            peak_zscores = []
            peak_margins = []
            border_hits = []
            for pair_index, (handle, target) in enumerate(pairs):
                hx, hy = int(handle[0]), int(handle[1])
                tx, ty = float(target[0]), float(target[1])
                if not (0 <= hx < w and 0 <= hy < h):
                    continue
                source_vec = source_vectors[pair_index].to(
                    device=edited_ft.device, dtype=edited_ft.dtype
                ).view(1, -1, 1, 1)
                similarity = cosine(source_vec, edited_ft).cpu().numpy()[0]
                pred_y, pred_x = np.unravel_index(similarity.argmax(), similarity.shape)
                peak = float(similarity[pred_y, pred_x])
                suppressed = similarity.copy()
                y0, y1 = max(0, pred_y - 5), min(h, pred_y + 6)
                x0, x1 = max(0, pred_x - 5), min(w, pred_x + 6)
                suppressed[y0:y1, x0:x1] = -np.inf
                second_peak = float(np.max(suppressed))
                peak_margin = peak - second_peak
                peak_z = (peak - float(np.mean(similarity))) / max(float(np.std(similarity)), 1e-8)
                border_hit = bool(pred_x in {0, w - 1} or pred_y in {0, h - 1})
                distance = math.hypot(tx - pred_x, ty - pred_y)
                distances.append(distance)
                peak_zscores.append(peak_z)
                peak_margins.append(peak_margin)
                border_hits.append(border_hit)
                predictions.append({
                    "handle_xy": [hx, hy], "target_xy": [tx, ty],
                    "predicted_xy": [int(pred_x), int(pred_y)], "distance_px": distance,
                    "peak_cosine": peak, "second_peak_outside_5px": second_peak,
                    "peak_margin": peak_margin, "peak_zscore": peak_z,
                    "border_hit": border_hit,
                })
            if not distances:
                raise RuntimeError("no valid pairs")
            mean_px = float(np.mean(distances))
            cases.append({
                "case_id": case_id,
                "category": item.get("category", "unknown"),
                "drag_type": item.get("drag_type", "unknown"),
                "point_count": len(distances),
                "mean_distance_px": mean_px,
                "mean_distance_normalized": mean_px / math.hypot(h, w),
                "mean_peak_zscore": float(np.mean(peak_zscores)),
                "mean_peak_margin": float(np.mean(peak_margins)),
                "any_border_hit": bool(any(border_hits)),
                "points": predictions,
            })
        except Exception as exc:
            errors.append({"case_id": case_id, "reason": f"{type(exc).__name__}: {exc}"})

    thresholds = [.025, .05, .075, .10]
    summary = {
        "mean_distance_px": bootstrap_ci([c["mean_distance_px"] for c in cases]),
        "mean_distance_normalized": bootstrap_ci([c["mean_distance_normalized"] for c in cases]),
        "mean_peak_zscore": bootstrap_ci([c["mean_peak_zscore"] for c in cases]),
        "mean_peak_margin": bootstrap_ci([c["mean_peak_margin"] for c in cases]),
        "border_hit_rate": bootstrap_ci([float(c["any_border_hit"]) for c in cases]),
        "success_rate_by_normalized_threshold": {
            str(t): bootstrap_ci([float(c["mean_distance_normalized"] <= t) for c in cases]) for t in thresholds
        },
    }
    strata = defaultdict(list)
    for case in cases:
        strata[f"category::{case['category']}"] .append(case)
        strata[f"drag_type::{case['drag_type']}"] .append(case)
    stratified = {
        name: {
            "mean_distance_px": bootstrap_ci([c["mean_distance_px"] for c in vals], repetitions=2000),
            "mean_distance_normalized": bootstrap_ci([c["mean_distance_normalized"] for c in vals], repetitions=2000),
        }
        for name, vals in sorted(strata.items())
    }
    payload = {
        "protocol": {
            "feature": "DIFT from Stable Diffusion 2.1, up block 1, t=261, ensemble=8",
            "matching": "global cosine argmax from each source handle feature",
            "confidence_diagnostics": "peak z-score, margin to the best match outside an 11x11 neighborhood, and border-hit rate; no case is excluded using these diagnostics",
            "case_aggregation": "mean over annotated point pairs; then equal-weight mean over cases",
            "normalization": "Euclidean pixel distance divided by image diagonal",
            "primary_success_threshold": 0.05,
            "threshold_sensitivity": thresholds,
            "uncertainty": "case-level nonparametric bootstrap, 10000 repetitions",
            "source_handle_cache": str(args.source_cache_dir),
            "source_handle_cache_key": "SHA-256 over source path/stat, prompt, feature seed, SD path, raster size, and handles; cached vectors are lossless float32",
        },
        "source_cache": {"hits": source_cache_hits, "misses": source_cache_misses},
        "coverage": {"mapping": len(mapping), "evaluated": len(cases), "errors": len(errors)},
        "summary": summary,
        "stratified": stratified,
        "cases": cases,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"coverage": payload["coverage"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
