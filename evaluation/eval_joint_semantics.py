#!/usr/bin/env python3
"""Per-case semantic compliance and outside-region preservation for joint edits.

This evaluator intentionally keeps geometry separate; DIFT point localization is
computed by eval_joint_geometry.py.  The common join key is the DragBench case id.
"""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def resolve(record):
    item = dict(record.get("source", {}))
    item.update(record.get("modified", {}))
    return item


def make_edit_region(mask, points, dilation_fraction=0.03):
    h, w = mask.shape
    binary = (mask > 127).astype(np.uint8)
    union = binary.copy()
    usable = len(points) - len(points) % 2
    for i in range(0, usable, 2):
        hx, hy = points[i][:2]
        tx, ty = points[i + 1][:2]
        matrix = np.float32([[1, 0, float(tx) - float(hx)], [0, 1, float(ty) - float(hy)]])
        shifted = cv2.warpAffine(binary, matrix, (w, h), flags=cv2.INTER_NEAREST)
        union = np.maximum(union, shifted)
    radius = max(1, int(round(dilation_fraction * math.hypot(h, w))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(union, kernel).astype(bool), radius


def crop_region(image, region, expansion_fraction=0.10):
    ys, xs = np.where(region)
    if len(xs) == 0:
        return image
    h, w = region.shape
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad = int(round(expansion_fraction * max(x1 - x0, y1 - y0)))
    return image.crop((max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)))


def bootstrap_ci(values, seed=20260825, repetitions=10000):
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0:
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 500):
        n = min(500, repetitions - start)
        idx = rng.integers(0, len(x), size=(n, len(x)))
        means[start:start + n] = x[idx].mean(axis=1)
    return {
        "mean": float(x.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "n": int(len(x)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-model", type=Path, required=True, help="Local CLIP ViT-L/14 checkpoint directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-seed", type=int, default=20260825)
    args = parser.parse_args()

    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    result_images = args.result_dir / "results" if (args.result_dir / "results").is_dir() else args.result_dir
    cases, errors = [], []
    for case_id, record in mapping.items():
        item = resolve(record)
        source_path = args.data_root / item["image_path"]
        result_path = result_images / f"{case_id}.png"
        mask_path = args.data_root / item["mask_path"]
        if not source_path.is_file() or not result_path.is_file() or not mask_path.is_file():
            errors.append({"case_id": case_id, "reason": "missing source, result, or mask"})
            continue
        source = Image.open(source_path).convert("RGB")
        edited = Image.open(result_path).convert("RGB").resize(source.size, Image.Resampling.BILINEAR)
        mask = np.asarray(Image.open(mask_path).convert("L").resize(source.size, Image.Resampling.NEAREST))
        region, dilation_radius = make_edit_region(mask, item.get("points", []))
        outside = ~region
        source_np = np.asarray(source).astype(np.float32) / 255.0
        edited_np = np.asarray(edited).astype(np.float32) / 255.0
        diff = edited_np - source_np
        outside_values = diff[outside]
        outside_mse = float(np.mean(outside_values ** 2)) if outside_values.size else None
        outside_mae = float(np.mean(np.abs(outside_values))) if outside_values.size else None
        cases.append({
            "case_id": case_id,
            "category": item.get("category", "unknown"),
            "drag_type": item.get("drag_type", "unknown"),
            "source_prompt": item.get("source_prompt", ""),
            "target_prompt": item.get("target_prompt", ""),
            "source_image": source,
            "edited_image": edited,
            "edited_crop": crop_region(edited, region),
            "outside_mse": outside_mse,
            "outside_mae": outside_mae,
            "edit_region_fraction": float(region.mean()),
            "dilation_radius_px": dilation_radius,
        })

    model = CLIPModel.from_pretrained(str(args.clip_model)).to(args.device).eval()
    processor = CLIPProcessor.from_pretrained(str(args.clip_model))

    for start in range(0, len(cases), args.batch_size):
        batch = cases[start:start + args.batch_size]
        texts = [x["source_prompt"] for x in batch] + [x["target_prompt"] for x in batch]
        text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        whole_inputs = processor(images=[x["source_image"] for x in batch] + [x["edited_image"] for x in batch], return_tensors="pt")
        crop_inputs = processor(images=[x["edited_crop"] for x in batch], return_tensors="pt")
        text_inputs = {k: v.to(args.device) for k, v in text_inputs.items()}
        whole_inputs = {k: v.to(args.device) for k, v in whole_inputs.items()}
        crop_inputs = {k: v.to(args.device) for k, v in crop_inputs.items()}
        with torch.inference_mode():
            text = model.get_text_features(**text_inputs)
            whole = model.get_image_features(**whole_inputs)
            crop = model.get_image_features(**crop_inputs)
            text = text / text.norm(dim=-1, keepdim=True)
            whole = whole / whole.norm(dim=-1, keepdim=True)
            crop = crop / crop.norm(dim=-1, keepdim=True)
        n = len(batch)
        source_text, target_text = text[:n], text[n:]
        source_image, edited_image = whole[:n], whole[n:]
        for i, case in enumerate(batch):
            whole_source = torch.dot(edited_image[i], source_text[i]).item()
            whole_target = torch.dot(edited_image[i], target_text[i]).item()
            crop_source = torch.dot(crop[i], source_text[i]).item()
            crop_target = torch.dot(crop[i], target_text[i]).item()
            image_delta = edited_image[i] - source_image[i]
            text_delta = target_text[i] - source_text[i]
            directional = torch.dot(
                image_delta / image_delta.norm().clamp_min(1e-8),
                text_delta / text_delta.norm().clamp_min(1e-8),
            ).item()
            case.update({
                "clip_whole_source": whole_source,
                "clip_whole_target": whole_target,
                "clip_whole_margin": whole_target - whole_source,
                "clip_crop_source": crop_source,
                "clip_crop_target": crop_target,
                "clip_crop_margin": crop_target - crop_source,
                "clip_directional": directional,
            })
            for key in ("source_image", "edited_image", "edited_crop"):
                case.pop(key, None)

    metric_keys = [
        "clip_whole_target", "clip_whole_margin", "clip_crop_target",
        "clip_crop_margin", "clip_directional", "outside_mse", "outside_mae",
        "edit_region_fraction",
    ]
    summary = {k: bootstrap_ci([c[k] for c in cases if c[k] is not None], args.bootstrap_seed) for k in metric_keys}
    strata = defaultdict(list)
    for case in cases:
        strata[f"category::{case['category']}"] .append(case)
        strata[f"drag_type::{case['drag_type']}"] .append(case)
    stratified = {
        name: {k: bootstrap_ci([c[k] for c in vals if c[k] is not None], args.bootstrap_seed, 2000) for k in metric_keys}
        for name, vals in sorted(strata.items())
    }
    payload = {
        "protocol": {
            "clip_model": str(args.clip_model),
            "clip_model_sha256_config": hashlib.sha256((args.clip_model / "config.json").read_bytes()).hexdigest(),
            "edit_region": "union(source mask, one translated mask per handle-target pair), dilated by 3% image diagonal",
            "crop": "tight edit-region box expanded by 10%",
            "outside_preservation": "pixel MSE/MAE over complement of edit region",
            "uncertainty": "case-level nonparametric bootstrap, 10000 repetitions",
        },
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
