#!/usr/bin/env python3
"""Input-defined performance strata for drag-only and joint test outputs."""

import argparse
import json
from pathlib import Path

import numpy as np


def stat(values, seed=20260825, n=10000):
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for start in range(0, n, 500):
        end = min(n, start + 500)
        idx = rng.integers(0, len(arr), size=(end - start, len(arr)))
        means[start:end] = arr[idx].mean(axis=1)
    return {"mean": float(arr.mean()), "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])], "n": len(arr)}


def sole_drag(path):
    runs = json.loads(Path(path).read_text(encoding="utf-8"))["Drag"]
    if len(runs) != 1:
        raise RuntimeError(f"Expected one drag run, got {len(runs)}")
    return next(iter(runs.values()))["_per_image"]


def grouped(rows, metadata, ids, field, levels, metrics):
    result = {}
    for level in levels:
        selected = [case_id for case_id in ids if metadata[case_id][field] == level and case_id in rows]
        result[level] = {metric: stat([float(rows[i][metric]) for i in selected]) for metric in metrics}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--natural-extreme", required=True)
    p.add_argument("--drag-metrics", required=True)
    p.add_argument("--joint-score", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--markdown", required=True)
    args = p.parse_args()
    metadata = json.loads(Path(args.manifest).read_text(encoding="utf-8"))["cases"]
    extreme_ids = set(json.loads(Path(args.natural_extreme).read_text(encoding="utf-8")))
    drag = sole_drag(args.drag_metrics)
    joint_payload = json.loads(Path(args.joint_score).read_text(encoding="utf-8"))
    joint = {row["case_id"]: row for row in joint_payload["cases"]}
    definitions = {
        "displacement": ("displacement_stratum", ["small", "medium", "large"]),
        "object_size": ("object_size_stratum", ["small", "medium", "large"]),
        "boundary_complexity": ("complexity_stratum", ["low", "medium", "high"]),
    }
    result = {"protocol": {"bins_use_inputs_only": True, "cut_points": "empirical tertiles fixed before output evaluation", "bootstrap_samples": 10000}, "drag_only": {}, "joint": {}}
    for name, (field, levels) in definitions.items():
        result["drag_only"][name] = grouped(drag, metadata, list(drag), field, levels, ["MD", "1-LPIPS", "CLIP_Sim"])
        result["joint"][name] = grouped(joint, metadata, list(joint), field, levels, ["mean_distance_normalized", "clip_directional", "outside_mse", "joint_success", "continuous_joint_utility"])
    for task, rows, metrics in (
        ("drag_only", drag, ["MD", "1-LPIPS", "CLIP_Sim"]),
        ("joint", joint, ["mean_distance_normalized", "clip_directional", "outside_mse", "joint_success", "continuous_joint_utility"]),
    ):
        result[task]["natural_top20pct_displacement"] = {
            "extreme": {m: stat([float(row[m]) for case_id, row in rows.items() if case_id in extreme_ids]) for m in metrics},
            "remainder": {m: stat([float(row[m]) for case_id, row in rows.items() if case_id not in extreme_ids]) for m in metrics},
        }
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Input-defined stratified performance", "", "Tertile cut points were fixed from masks and point annotations before inspecting outputs. CIs use 10,000 image-bootstrap samples.", ""]
    for task, metrics in (("drag_only", ["MD", "1-LPIPS"]), ("joint", ["mean_distance_normalized", "clip_directional", "joint_success"])):
        lines.extend([f"## {task}", ""])
        for name in definitions:
            levels = definitions[name][1]
            lines.append("| Stratum | N | " + " | ".join(metrics) + " |")
            lines.append("|---|---:|" + "---:|" * len(metrics))
            for level in levels:
                row = result[task][name][level]
                lines.append(f"| {name}: {level} | {row[metrics[0]]['n']} | " + " | ".join(f"{row[m]['mean']:.4f}" for m in metrics) + " |")
            lines.append("")
    Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
