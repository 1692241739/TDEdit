#!/usr/bin/env python3
"""Analyze held-out 1.5x/2.0x displacement stress tests."""

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ["joint_success", "continuous_joint_utility", "mean_distance_normalized", "clip_directional", "outside_mse"]


def stat(values, seed=20260825, n=10000):
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for start in range(0, n, 500):
        end = min(n, start + 500)
        idx = rng.integers(0, len(arr), size=(end - start, len(arr)))
        means[start:end] = arr[idx].mean(axis=1)
    return {"mean": float(arr.mean()), "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])], "n": len(arr)}


def load(root, label):
    data = json.loads((root / label / "joint_score.json").read_text(encoding="utf-8"))
    return {row["case_id"]: row for row in data["cases"]}, data["primary"]


def compare(a, b, metric):
    if not a or set(a) != set(b):
        raise ValueError("Extreme-test variants must contain identical nonempty case IDs")
    ids = sorted(set(a) & set(b))
    av = np.asarray([float(a[i][metric]) for i in ids])
    bv = np.asarray([float(b[i][metric]) for i in ids])
    return {"A": stat(av), "B": stat(bv), "A_minus_B": stat(av - bv)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics-root", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--markdown", required=True)
    args = p.parse_args()
    root = Path(args.metrics_root)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))["selected"]["configuration"]
    cases = {}; primary = {}
    for label in ("default_1p5", "selected_1p5", "default_2p0", "selected_2p0"):
        cases[label], primary[label] = load(root, label)
    result = {"selected_configuration": selection, "primary": primary, "selected_minus_default": {}, "scale_2p0_minus_1p5": {}}
    for scale in ("1p5", "2p0"):
        result["selected_minus_default"][scale] = {
            metric: compare(cases[f"selected_{scale}"], cases[f"default_{scale}"], metric)
            for metric in METRICS
        }
    for config in ("default", "selected"):
        result["scale_2p0_minus_1p5"][config] = {
            metric: compare(cases[f"{config}_2p0"], cases[f"{config}_1p5"], metric)
            for metric in METRICS
        }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Extreme-displacement stress test", "", f"Selected schedule: `{selection}`. Calibration and test cases are disjoint.", "", "| Scale | Config | Joint success | Utility | Norm. MD | Directional CLIP | Outside MSE |", "|---|---|---:|---:|---:|---:|---:|"]
    for scale in ("1p5", "2p0"):
        for config in ("default", "selected"):
            label = f"{config}_{scale}"; row = cases[label]
            vals = {m: np.mean([float(x[m]) for x in row.values()]) for m in METRICS}
            lines.append(f"| {scale.replace('p','.')}x | {config} | {vals['joint_success']:.3f} | {vals['continuous_joint_utility']:.3f} | {vals['mean_distance_normalized']:.4f} | {vals['clip_directional']:.4f} | {vals['outside_mse']:.5f} |")
    Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
