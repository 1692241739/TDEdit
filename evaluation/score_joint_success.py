#!/usr/bin/env python3
"""Join semantic/preservation and geometry records into joint success metrics."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


PRIMARY = {"geometry": 0.05, "semantic_directional": 0.0, "outside_mse": 0.01}


def bootstrap(values, seed=20260825, repetitions=10000):
    x = np.asarray(values, dtype=np.float64)
    if not len(x):
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions)
    for start in range(0, repetitions, 500):
        n = min(500, repetitions - start)
        means[start:start + n] = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return {"mean": float(x.mean()), "ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))], "n": len(x)}


def success(case, g=0.05, s=0.0):
    """Primary simultaneous-control outcome: geometry AND semantics.

    Preservation is intentionally not a universal gate because some target
    prompts request global changes (e.g. day-to-night or season changes).
    """
    return case["mean_distance_normalized"] <= g and case["clip_directional"] >= s


def success_with_preservation(case, g=0.05, s=0.0, p=0.01):
    return success(case, g, s) and case["outside_mse"] <= p


def continuous_utility(case):
    geometry = math.exp(-case["mean_distance_normalized"] / PRIMARY["geometry"])
    semantic = 1.0 / (1.0 + math.exp(-case["clip_directional"] / 0.05))
    values = [max(v, 1e-8) for v in (geometry, semantic)]
    return 2.0 / sum(1.0 / v for v in values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantics", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260825)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow incomplete evaluator coverage. Disabled by default to prevent silent case dropping.",
    )
    args = parser.parse_args()

    sem_payload = json.loads(args.semantics.read_text(encoding="utf-8"))
    geo_payload = json.loads(args.geometry.read_text(encoding="utf-8"))
    for name, payload in (("semantics", sem_payload), ("geometry", geo_payload)):
        ids = [row["case_id"] for row in payload["cases"]]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError(f"{name} must contain nonempty, unique case IDs")
    sem = {x["case_id"]: x for x in sem_payload["cases"]}
    geo = {x["case_id"]: x for x in geo_payload["cases"]}
    if not args.allow_partial:
        for label, payload, rows in (("semantics", sem_payload, sem), ("geometry", geo_payload, geo)):
            coverage = payload.get("coverage", {})
            mapping_n = coverage.get("mapping")
            if mapping_n is not None and len(rows) != int(mapping_n):
                raise RuntimeError(
                    f"Incomplete {label} coverage: evaluated {len(rows)} of {mapping_n}; "
                    "refusing to score a silently reduced subset"
                )
        if sem.keys() != geo.keys():
            raise RuntimeError(
                f"Semantic and geometry case IDs differ: semantics={len(sem)}, geometry={len(geo)}"
            )
    ids = sorted(sem.keys() & geo.keys())
    cases = []
    for case_id in ids:
        case = dict(sem[case_id])
        case.update({
            "point_count": geo[case_id]["point_count"],
            "mean_distance_px": geo[case_id]["mean_distance_px"],
            "mean_distance_normalized": geo[case_id]["mean_distance_normalized"],
        })
        for key in ("mean_distance_normalized", "clip_directional", "outside_mse"):
            value = case[key]
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"{case_id}: {key} must be finite; do not replace missing data by zero")
        if case["mean_distance_normalized"] < 0 or case["outside_mse"] < 0:
            raise ValueError(f"{case_id}: distances and MSE must be nonnegative")
        case["geometry_success"] = case["mean_distance_normalized"] <= PRIMARY["geometry"]
        case["semantic_success"] = case["clip_directional"] >= PRIMARY["semantic_directional"]
        case["preservation_success"] = case["outside_mse"] <= PRIMARY["outside_mse"]
        case["joint_success"] = success(case)
        case["joint_plus_preservation"] = success_with_preservation(case)
        case["continuous_joint_utility"] = continuous_utility(case)
        cases.append(case)

    primary = {
        "geometry_success_rate": bootstrap([c["geometry_success"] for c in cases], args.bootstrap_seed),
        "semantic_success_rate": bootstrap([c["semantic_success"] for c in cases], args.bootstrap_seed),
        "preservation_success_rate": bootstrap([c["preservation_success"] for c in cases], args.bootstrap_seed),
        "joint_success_rate": bootstrap([c["joint_success"] for c in cases], args.bootstrap_seed),
        "joint_plus_preservation_rate_diagnostic": bootstrap([c["joint_plus_preservation"] for c in cases], args.bootstrap_seed),
        "continuous_joint_utility": bootstrap([c["continuous_joint_utility"] for c in cases], args.bootstrap_seed),
    }
    sensitivity = {"joint_geometry_and_semantics": {}, "joint_plus_preservation_diagnostic": {}}
    for g in [.025, .05, .075]:
        for s in [-.05, 0.0, .05]:
            key = f"g<={g:.3f}|direction>={s:.2f}"
            sensitivity["joint_geometry_and_semantics"][key] = bootstrap(
                [success(c, g, s) for c in cases], args.bootstrap_seed, 2000
            )
            for p in [.005, .01, .02]:
                diagnostic_key = f"g<={g:.3f}|direction>={s:.2f}|outside_mse<={p:.3f}"
                sensitivity["joint_plus_preservation_diagnostic"][diagnostic_key] = bootstrap(
                    [success_with_preservation(c, g, s, p) for c in cases], args.bootstrap_seed, 2000
                )

    payload = {
        "protocol": {
            "primary_thresholds": PRIMARY,
            "joint_success": "geometry and semantic thresholds must both be satisfied on the same case",
            "preservation": "reported separately; the three-way diagnostic is not a primary endpoint because some prompts intentionally change the global scene",
            "unit_of_analysis": "case (point distances averaged within case)",
            "configuration_selection": "maximize calibration joint-success rate; break ties by continuous joint utility; never inspect test outputs",
            "continuous_utility": "harmonic mean of exp(-normalized MD/0.05) and sigmoid(CLIP direction/0.05)",
        },
        "coverage": {
            "semantics": len(sem), "geometry": len(geo), "joined": len(cases),
            "semantics_only": sorted(sem.keys() - geo.keys()),
            "geometry_only": sorted(geo.keys() - sem.keys()),
        },
        "primary": primary,
        "threshold_sensitivity": sensitivity,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"coverage": payload["coverage"], "primary": primary}, indent=2))


if __name__ == "__main__":
    main()
