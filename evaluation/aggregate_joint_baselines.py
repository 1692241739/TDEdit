#!/usr/bin/env python3
"""Aggregate seeds within cases, then compare methods by paired case bootstrap."""

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = [
    "geometry_success",
    "semantic_success",
    "preservation_success",
    "joint_success",
    "joint_plus_preservation",
    "continuous_joint_utility",
    "mean_distance_normalized",
    "clip_directional",
    "outside_mse",
]


def load_cases(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [row["case_id"] for row in payload["cases"]]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"Empty or duplicate case IDs in {path}")
    return {row["case_id"]: row for row in payload["cases"]}


def aggregate_seeds(paths, expected_cases=None):
    if not paths:
        raise ValueError("At least one seed file is required")
    seeds = [load_cases(path) for path in paths]
    if any(set(seed) != set(seeds[0]) for seed in seeds):
        raise ValueError("Seed case IDs differ; incomplete intersections are not permitted")
    if expected_cases is not None:
        bad = [(str(path), len(seed)) for path, seed in zip(paths, seeds) if len(seed) != expected_cases]
        if bad:
            raise RuntimeError(f"Incomplete method/seed coverage; expected {expected_cases}: {bad}")
    ids = sorted(set.intersection(*(set(seed) for seed in seeds)))
    if expected_cases is not None and len(ids) != expected_cases:
        raise RuntimeError(
            f"Seed case-ID intersection has {len(ids)} cases; expected {expected_cases}"
        )
    output = {}
    for case_id in ids:
        output[case_id] = {
            metric: float(np.mean([float(seed[case_id][metric]) for seed in seeds]))
            for metric in METRICS
        }
    return output


def stat(values, seed=20260825, n=10000):
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for start in range(0, n, 500):
        end = min(n, start + 500)
        idx = rng.integers(0, len(arr), size=(end - start, len(arr)))
        means[start:end] = arr[idx].mean(axis=1)
    return {"mean": float(arr.mean()), "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])], "n": len(arr)}


def sign_flip_p(differences, seed=20260825, n=100000):
    d = np.asarray(differences, dtype=np.float64)
    observed = abs(float(d.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, n, 1000):
        end = min(n, start + 1000)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(end - start, len(d)))
        exceed += int(np.sum(np.abs((signs * d).mean(axis=1)) >= observed))
    return float((exceed + 1) / (n + 1))


def holm(p_values):
    ordered = sorted(p_values, key=p_values.get)
    m = len(ordered); adjusted = {}; running = 0.0
    for rank, key in enumerate(ordered):
        value = min(1.0, (m - rank) * p_values[key])
        running = max(running, value); adjusted[key] = running
    return adjusted


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--markdown", required=True)
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    # Paths in the config are relative to that file, not the caller's cwd.
    for info in config["methods"].values():
        info["joint_score_paths"] = [
            str((Path(args.config).resolve().parent / path).resolve())
            for path in info["joint_score_paths"]
        ]
    expected_cases = config.get("expected_cases")
    methods = {
        name: aggregate_seeds(info["joint_score_paths"], expected_cases=expected_cases)
        for name, info in config["methods"].items()
    }
    reference_name = config.get("reference", "TDEdit")
    reference = methods[reference_name]
    if any(set(cases) != set(reference) for cases in methods.values()):
        raise ValueError("Method case IDs differ; comparisons require identical cases")
    summary = {
        name: {metric: stat([row[metric] for row in cases.values()]) for metric in METRICS}
        for name, cases in methods.items()
    }
    comparisons = {}; raw_p = {}
    for name, cases in methods.items():
        if name == reference_name:
            continue
        ids = sorted(set(reference) & set(cases))
        comparisons[name] = {}
        for metric in METRICS:
            delta = np.asarray([reference[i][metric] - cases[i][metric] for i in ids])
            comparisons[name][metric] = {"TDEdit_minus_baseline": stat(delta), "raw_sign_flip_p": sign_flip_p(delta)}
            if metric in {"joint_success", "continuous_joint_utility"}:
                raw_p[f"{name}:{metric}"] = comparisons[name][metric]["raw_sign_flip_p"]
    adjusted = holm(raw_p)
    for key, value in adjusted.items():
        name, metric = key.split(":", 1); comparisons[name][metric]["holm_adjusted_p"] = value
    result = {"protocol": {"seed_aggregation": "mean within case, then bootstrap cases", "bootstrap_samples": 10000, "paired_sign_flip_samples": 100000, "multiplicity": "Holm across pre-specified success/utility baseline comparisons"}, "coverage": {name: len(cases) for name, cases in methods.items()}, "method_metadata": config["methods"], "summary": summary, "paired_vs_TDEdit": comparisons}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Joint protocol: methods and sequential controls", "", "Seeds are averaged within each case; 95% CIs bootstrap cases. Unsupported channels remain partial-control references.", "", "| Method | Control status | Seeds | Cases | Geometry success | Semantic success | Preservation success | Joint success | Three-way diagnostic | Utility | Norm. MD | Directional CLIP | Outside MSE |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in summary.items():
        meta = config["methods"][name]
        lines.append(f"| {name} | {meta['control_status']} | {len(meta['joint_score_paths'])} | {len(methods[name])} | {row['geometry_success']['mean']:.3f} | {row['semantic_success']['mean']:.3f} | {row['preservation_success']['mean']:.3f} | {row['joint_success']['mean']:.3f} | {row['joint_plus_preservation']['mean']:.3f} | {row['continuous_joint_utility']['mean']:.3f} | {row['mean_distance_normalized']['mean']:.4f} | {row['clip_directional']['mean']:.4f} | {row['outside_mse']['mean']:.5f} |")
    Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
