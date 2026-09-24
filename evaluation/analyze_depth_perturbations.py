#!/usr/bin/env python3
"""Pair controlled depth perturbations against the unperturbed V2 run."""

import argparse
import json
from pathlib import Path

import numpy as np


def stat(values, seed=20260825, n=10000):
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for start in range(0, n, 500):
        end = min(n, start + 500)
        idx = rng.integers(0, len(arr), size=(end - start, len(arr)))
        means[start:end] = arr[idx].mean(axis=1)
    return {"mean": float(arr.mean()), "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])], "n": len(arr)}


def sole_entry(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))["Drag"]
    if len(data) != 1:
        raise RuntimeError(f"Expected one run in {path}, got {len(data)}")
    return next(iter(data.values()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clean-combined", required=True)
    p.add_argument("--perturb-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--markdown", required=True)
    p.add_argument("--clean-run", help="Explicit run key under Drag (recommended)")
    args = p.parse_args()
    combined = json.loads(Path(args.clean_combined).read_text(encoding="utf-8"))["Drag"]
    if args.clean_run:
        clean_entry = combined[args.clean_run]
    else:
        matches = [entry for entry in combined.values() if "drag_fixed_influence_05" in str(entry.get("_hyperparams", {}).get("output_dir", ""))]
        if len(matches) != 1:
            raise ValueError("Use --clean-run to identify exactly one clean V2 run")
        clean_entry = matches[0]
    clean = clean_entry["_per_image"]
    labels = ["gaussian_005", "gaussian_010", "smooth_4", "smooth_8", "order_invert"]
    result = {"clean_summary": {m: clean_entry[m] for m in ("MD", "CLIP_Sim", "1-LPIPS")}, "perturbations": {}}
    for label in labels:
        entry = sole_entry(Path(args.perturb_root) / f"{label}.json")
        pert = entry["_per_image"]
        if not clean or set(clean) != set(pert):
            raise ValueError(f"{label}: perturbed and clean case IDs must match")
        ids = sorted(set(clean) & set(pert))
        row = {"summary": {m: entry[m] for m in ("MD", "CLIP_Sim", "1-LPIPS")}, "paired": {}}
        for metric in ("MD", "CLIP_Sim", "1-LPIPS"):
            c = np.asarray([clean[i][metric] for i in ids]); v = np.asarray([pert[i][metric] for i in ids])
            row["paired"][metric] = {"perturbed": stat(v), "clean": stat(c), "perturbed_minus_clean": stat(v - c)}
        result["perturbations"][label] = row
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Controlled depth perturbations", "", "All 36 annotated 3D cases; each perturbation is paired with the same unperturbed V2 output.", "", "| Perturbation | MD | ΔMD (95% CI) | 1-LPIPS | CLIP similarity |", "|---|---:|---:|---:|---:|"]
    for label in labels:
        row = result["perturbations"][label]["paired"]; d = row["MD"]["perturbed_minus_clean"]
        lines.append(f"| {label} | {row['MD']['perturbed']['mean']:.3f} | {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] | {row['1-LPIPS']['perturbed']['mean']:.4f} | {row['CLIP_Sim']['perturbed']['mean']:.4f} |")
    Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
