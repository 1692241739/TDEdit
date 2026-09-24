#!/usr/bin/env python3
"""Paired factorial contrasts for the joint-edit ablation."""

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = [
    "joint_success",
    "continuous_joint_utility",
    "mean_distance_normalized",
    "clip_directional",
    "outside_mse",
]


def load(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["case_id"]: row for row in payload["cases"]}


def stat(values, seed=20260825, repetitions=10000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 500):
        end = min(repetitions, start + 500)
        indices = rng.integers(0, len(values), size=(end - start, len(values)))
        means[start:end] = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
        "n": int(len(values)),
    }


def contrast(rows, coefficients, metric, ids):
    return stat([
        sum(coefficient * float(rows[label][case_id][metric]) for label, coefficient in coefficients.items())
        for case_id in ids
    ])


def family(rows, definitions, ids):
    return {
        name: {metric: contrast(rows, coefficients, metric, ids) for metric in METRICS}
        for name, coefficients in definitions.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-cases", type=int, default=163)
    args = parser.parse_args()
    labels = [
        "full", "no_lqm", "no_rkm", "no_lqm_no_rkm",
        "appearance_anchor_only", "geometry_anchor_only", "neither_anchor",
    ]
    rows = {label: load(args.metrics_root / label / "joint_score.json") for label in labels}
    if any(set(value) != set(rows["full"]) for value in rows.values()):
        raise RuntimeError("Ablation case IDs differ; partial intersections are not silently analyzed")
    ids = sorted(set.intersection(*(set(value) for value in rows.values())))
    if len(ids) != args.expected_cases:
        raise RuntimeError(f"Expected {args.expected_cases} paired cases, found {len(ids)}")

    lqm_rki = {
        "factorial_interaction_full-no_lqm-no_rki+neither": {
            "full": 1, "no_lqm": -1, "no_rkm": -1, "no_lqm_no_rkm": 1,
        },
        "lqm_effect_with_rki": {"full": 1, "no_lqm": -1},
        "lqm_effect_without_rki": {"no_rkm": 1, "no_lqm_no_rkm": -1},
        "rki_effect_with_lqm": {"full": 1, "no_rkm": -1},
        "rki_effect_without_lqm": {"no_lqm": 1, "no_lqm_no_rkm": -1},
    }
    anchors = {
        "factorial_interaction_full-appearance_only-geometry_only+neither": {
            "full": 1, "appearance_anchor_only": -1,
            "geometry_anchor_only": -1, "neither_anchor": 1,
        },
        "geometry_effect_with_appearance": {"full": 1, "appearance_anchor_only": -1},
        "geometry_effect_without_appearance": {"geometry_anchor_only": 1, "neither_anchor": -1},
        "appearance_effect_with_geometry": {"full": 1, "geometry_anchor_only": -1},
        "appearance_effect_without_geometry": {"appearance_anchor_only": 1, "neither_anchor": -1},
    }
    payload = {
        "protocol": {
            "paired_cases": len(ids),
            "bootstrap_samples": 10000,
            "factorial_contrast": "A+B interaction = both - A-only - B-only + neither",
        },
        "lqm_rki": family(rows, lqm_rki, ids),
        "anchors": family(rows, anchors, ids),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
