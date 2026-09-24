#!/usr/bin/env python3
"""Analyze forced-mode robustness with paired, per-image comparisons."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


MODES = [
    "2D-Rigid", "2D-Non-Rigid", "2D-Hybrid",
    "3D-Rigid", "3D-Non-Rigid", "3D-Hybrid",
]


def ci(values, seed=20260825, n_boot=10000):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, 500):
        end = min(n_boot, start + 500)
        idx = rng.integers(0, arr.size, size=(end - start, arr.size))
        means[start:end] = arr[idx].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
        "n": int(arr.size),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()

    raw_metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))["Drag"]
    mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    true_mode = {
        sample_id: str((entry.get("modified") or entry.get("source") or {}).get("drag_type", ""))
        for sample_id, entry in mapping.items()
    }

    by_forced = {}
    run_summary = {}
    for run_name, entry in raw_metrics.items():
        cfg = entry.get("_hyperparams", {}).get("experiment_config", {})
        forced = str(cfg.get("drag_type", ""))
        if forced not in MODES:
            continue
        per_image = entry.get("_per_image", {})
        by_forced[forced] = per_image
        run_summary[forced] = {
            key: float(entry[key]) for key in ("MD", "LPIPS", "1-LPIPS", "CLIP_Sim", "Editing(s)")
        }
        run_summary[forced]["generated_and_evaluated"] = int(len(per_image))
        run_summary[forced]["requested"] = int(len(mapping))
        run_summary[forced]["coverage"] = float(len(per_image) / max(1, len(mapping)))

    missing = sorted(set(MODES) - set(by_forced))
    if missing:
        raise RuntimeError(f"Missing forced modes: {missing}")

    matrix = {}
    for actual in MODES:
        ids = [sample_id for sample_id, mode in true_mode.items() if mode == actual]
        if not ids:
            continue
        matrix[actual] = {}
        for forced in MODES:
            rows = [by_forced[forced][sample_id] for sample_id in ids if sample_id in by_forced[forced]]
            matrix[actual][forced] = {
                "n": len(rows),
                "MD": float(np.mean([row["MD"] for row in rows])) if rows else None,
                "CLIP_Sim": float(np.mean([row["CLIP_Sim"] for row in rows])) if rows else None,
                "1-LPIPS": float(np.mean([row["1-LPIPS"] for row in rows])) if rows else None,
            }

    matched_rows = []
    mismatch_average_rows = []
    oracle_rows = []
    per_type_deltas = defaultdict(lambda: {"MD": [], "CLIP_Sim": [], "1-LPIPS": []})
    complete_ids = set.intersection(*(set(rows) for rows in by_forced.values()))
    for sample_id, actual in true_mode.items():
        if sample_id not in complete_ids or actual not in by_forced:
            continue
        matched = by_forced[actual][sample_id]
        alternatives = [
            by_forced[forced][sample_id] for forced in MODES
            if forced != actual and sample_id in by_forced[forced]
        ]
        if not alternatives:
            continue
        mismatch_avg = {
            metric: float(np.mean([row[metric] for row in alternatives]))
            for metric in ("MD", "CLIP_Sim", "1-LPIPS")
        }
        oracle = min(
            (by_forced[forced][sample_id] for forced in MODES if sample_id in by_forced[forced]),
            key=lambda row: row["MD"],
        )
        matched_rows.append(matched)
        mismatch_average_rows.append(mismatch_avg)
        oracle_rows.append(oracle)
        for metric in ("MD", "CLIP_Sim", "1-LPIPS"):
            per_type_deltas[actual][metric].append(float(mismatch_avg[metric] - matched[metric]))

    paired = {}
    for metric in ("MD", "CLIP_Sim", "1-LPIPS"):
        matched_values = [row[metric] for row in matched_rows]
        mismatch_values = [row[metric] for row in mismatch_average_rows]
        oracle_values = [row[metric] for row in oracle_rows]
        paired[metric] = {
            "matched": ci(matched_values),
            "average_mismatched": ci(mismatch_values),
            "mismatched_minus_matched": ci(np.asarray(mismatch_values) - np.asarray(matched_values)),
            "oracle_forced_mode": ci(oracle_values),
        }

    output = {
        "protocol": {
            "design": "Each of 205 images is edited under every forced mode; comparisons are paired by image.",
            "complete_case_policy": "Metric deltas use only cases generated successfully under all six forced modes; coverage/OOM failures are reported separately and never silently discarded.",
            "complete_paired_cases": int(len(complete_ids)),
            "bootstrap_samples": 10000,
            "bootstrap_unit": "image",
            "selection_uses_outputs": False,
        },
        "run_summary_all_205": run_summary,
        "missing_output_ids_by_forced_mode": {
            forced: sorted(set(mapping) - set(rows)) for forced, rows in by_forced.items()
        },
        "true_by_forced_matrix": matrix,
        "paired_sensitivity": paired,
        "per_true_type_mismatched_minus_matched": {
            actual: {metric: ci(values) for metric, values in metrics.items()}
            for actual, metrics in per_type_deltas.items()
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Mode-mismatch sensitivity analysis", "",
        "Each image was edited under all six forced modes; uncertainty is a 10,000-sample paired image bootstrap.", "",
        "| Metric | Matched | Avg. mismatched | Mismatch − matched (95% CI) | Oracle forced mode |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in ("MD", "CLIP_Sim", "1-LPIPS"):
        row = paired[metric]
        delta = row["mismatched_minus_matched"]
        lines.append(
            f"| {metric} | {row['matched']['mean']:.4f} | {row['average_mismatched']['mean']:.4f} | "
            f"{delta['mean']:.4f} [{delta['ci95'][0]:.4f}, {delta['ci95'][1]:.4f}] | "
            f"{row['oracle_forced_mode']['mean']:.4f} |"
        )
    lines.extend(["", "## Mean Distance by true and forced mode", ""])
    present = list(matrix)
    lines.append("| True \\ Forced | " + " | ".join(MODES) + " |")
    lines.append("|---|" + "---:|" * len(MODES))
    for actual in present:
        cells = [
            "—" if matrix[actual][forced]["MD"] is None else f"{matrix[actual][forced]['MD']:.2f}"
            for forced in MODES
        ]
        lines.append(f"| {actual} | " + " | ".join(cells) + " |")
    Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
