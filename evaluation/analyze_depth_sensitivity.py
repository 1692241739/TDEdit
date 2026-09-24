#!/usr/bin/env python3
"""Paired Depth Anything V1/V2 sensitivity analysis on all 3D cases."""

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--markdown", required=True)
    p.add_argument("--disagreement")
    p.add_argument("--v1-run", help="Explicit key under Drag for the V1 run")
    p.add_argument("--v2-run", help="Explicit key under Drag for the V2 run")
    args = p.parse_args()
    runs = json.loads(Path(args.metrics).read_text(encoding="utf-8"))["Drag"]
    backends = {}
    summaries = {}
    for name, entry in runs.items():
        output_dir = str(entry.get("_hyperparams", {}).get("output_dir", ""))
        if args.v1_run or args.v2_run:
            if not (args.v1_run and args.v2_run) or args.v1_run == args.v2_run:
                raise ValueError("Provide distinct --v1-run and --v2-run together")
            backend = "V1" if name == args.v1_run else "V2" if name == args.v2_run else None
        else:
            backend = "V1" if "depth_v1_sensitivity" in output_dir else "V2" if "drag_fixed_influence_05" in output_dir else None
        if backend:
            if backend in backends:
                raise ValueError(f"Ambiguous {backend} runs; use explicit run keys")
            backends[backend] = entry["_per_image"]
            summaries[backend] = {key: entry[key] for key in ("MD", "LPIPS", "1-LPIPS", "CLIP_Sim")}
    if set(backends) != {"V1", "V2"}:
        raise RuntimeError(f"Expected V1 and V2, found {sorted(backends)}")
    ids = sorted(set(backends["V1"]) & set(backends["V2"]))
    if not ids or set(backends["V1"]) != set(backends["V2"]):
        raise ValueError("V1 and V2 runs must have identical nonempty case IDs")
    result = {"protocol": {"paired_cases": len(ids), "subset": "all annotated 3D cases", "bootstrap_samples": 10000}, "aggregate": summaries, "paired": {}}
    for metric in ("MD", "CLIP_Sim", "1-LPIPS"):
        v1 = np.asarray([backends["V1"][i][metric] for i in ids])
        v2 = np.asarray([backends["V2"][i][metric] for i in ids])
        result["paired"][metric] = {"V1": stat(v1), "V2": stat(v2), "V1_minus_V2": stat(v1 - v2)}
    if args.disagreement:
        disagreement = json.loads(Path(args.disagreement).read_text(encoding="utf-8"))
        bins = {r["case_id"]: r["disagreement_tertile"] for r in disagreement["cases"]}
        result["by_input_depth_disagreement"] = {}
        for label in ("low", "medium", "high"):
            group = [i for i in ids if bins.get(i) == label]
            result["by_input_depth_disagreement"][label] = {"n": len(group)}
            for metric in ("MD", "CLIP_Sim", "1-LPIPS"):
                delta = [backends["V1"][i][metric] - backends["V2"][i][metric] for i in group]
                result["by_input_depth_disagreement"][label][f"V1_minus_V2_{metric}"] = stat(delta)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Depth-backend sensitivity", "", "All annotated 3D cases, paired by image; 10,000 bootstrap resamples.", "", "| Metric | Depth Anything V1 | Depth Anything V2 | V1 − V2 (95% CI) |", "|---|---:|---:|---:|"]
    for metric in ("MD", "CLIP_Sim", "1-LPIPS"):
        row = result["paired"][metric]; d = row["V1_minus_V2"]
        lines.append(f"| {metric} | {row['V1']['mean']:.4f} | {row['V2']['mean']:.4f} | {d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] |")
    if "by_input_depth_disagreement" in result:
        lines.extend(["", "## Stratified by input-only V1/V2 disagreement", "", "| Tertile | N | ΔMD, V1−V2 (95% CI) |", "|---|---:|---:|"])
        for label, row in result["by_input_depth_disagreement"].items():
            d = row["V1_minus_V2_MD"]
            lines.append(f"| {label} | {row['n']} | {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] |")
    Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
