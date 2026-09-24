#!/usr/bin/env python3
"""Paired bootstrap analysis for input-mask perturbations."""

import argparse
import json
from pathlib import Path

import numpy as np


def per_case(path):
    runs = json.loads(Path(path).read_text(encoding="utf-8"))["Drag"]
    if len(runs) != 1:
        raise RuntimeError(f"Expected one run in {path}, got {len(runs)}")
    return next(iter(runs.values()))["_per_image"]


def stat(values, seed=20260825, draws=10000):
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = a[rng.integers(0, len(a), size=(draws, len(a)))].mean(1)
    return {"mean": float(a.mean()), "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])], "n": int(len(a))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--erode5", required=True)
    ap.add_argument("--dilate5", required=True)
    ap.add_argument("--shift10", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--perturbation-summary", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    clean_all = per_case(args.clean)
    subset = set(json.loads(Path(args.subset).read_text(encoding="utf-8")))
    clean = {k: v for k, v in clean_all.items() if k in subset}
    if not subset or set(clean) != subset:
        raise ValueError("Clean run must cover every requested subset case")
    variants = {name: per_case(getattr(args, name)) for name in ("erode5", "dilate5", "shift10")}
    result = {"protocol": {"paired_cases": len(clean), "bootstrap_samples": 10000, "selection_uses_outputs": False}, "mask_perturbations": json.loads(Path(args.perturbation_summary).read_text()), "clean": {}, "variants": {}}
    metrics = ("MD", "1-LPIPS", "CLIP_Sim")
    for metric in metrics:
        result["clean"][metric] = stat([clean[i][metric] for i in sorted(clean)])
    for name, rows in variants.items():
        ids = sorted(set(clean) & set(rows))
        if len(ids) != len(clean):
            raise RuntimeError(f"{name} coverage {len(ids)}/{len(clean)}")
        result["variants"][name] = {}
        for metric in metrics:
            vals = np.asarray([rows[i][metric] for i in ids])
            base = np.asarray([clean[i][metric] for i in ids])
            result["variants"][name][metric] = stat(vals)
            result["variants"][name][f"perturbed_minus_clean_{metric}"] = stat(vals - base)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
