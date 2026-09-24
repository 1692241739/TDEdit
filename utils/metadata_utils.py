import json
import os
import pickle

import numpy as np


def load_metadata(filename, meta_data_type, save_path):
    """
    Load metadata from json / pkl files.
    """
    base_filename = os.path.splitext(filename)[0]
    json_path = os.path.join(save_path, "json", f"{base_filename}.json")
    pkl_path = os.path.join(save_path, "pkl", f"{base_filename}.pkl")

    source_prompt = None
    target_prompt = None
    mask = None
    points = None
    drag_type = None
    rigid_type = None
    origin_point_distance = 20.0
    shield_distance = 30.0
    reletive_distance = 0.0
    influence_ratio = 1.0
    influence_range = 0.7
    non_influence_range = 0.7
    cross_replace_steps = 0.7
    self_replace_steps = 0.7

    if meta_data_type in ("json", "both") and os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            source_prompt = data.get("source_prompt", source_prompt)
            target_prompt = data.get("target_prompt", target_prompt)
            raw_mask = data.get("mask")
            mask = np.array(raw_mask, dtype=np.uint8) if raw_mask is not None else mask
            points = data.get("points", points)
            drag_type = data.get("drag_type", drag_type)
            rigid_type = data.get("rigid_type", rigid_type)
            origin_point_distance = data.get("origin_point_distance", origin_point_distance)
            shield_distance = data.get("shield_distance", shield_distance)
            reletive_distance = data.get("reletive_distance", reletive_distance)
            influence_ratio = data.get("influence_ratio", influence_ratio)
            influence_range = data.get("influence_range", influence_range)
            non_influence_range = data.get("non_influence_range", non_influence_range)
            cross_replace_steps = data.get("cross_replace_steps", cross_replace_steps)
            self_replace_steps = data.get("self_replace_steps", self_replace_steps)
        except Exception as e:
            print(f"Load JSON metadata failed: {e}")

    need_fallback = (
        source_prompt is None
        or target_prompt is None
        or mask is None
        or points is None
        or drag_type is None
    )
    if meta_data_type in ("pkl", "both") and need_fallback and os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            if source_prompt is None:
                source_prompt = data.get("source_prompt")
            if target_prompt is None:
                target_prompt = data.get("target_prompt")
            if mask is None:
                raw_mask = data.get("mask")
                mask = np.array(raw_mask, dtype=np.uint8) if raw_mask is not None else None
            if points is None:
                points = data.get("points")
            if drag_type is None:
                drag_type = data.get("drag_type")
            if rigid_type is None:
                rigid_type = data.get("rigid_type", rigid_type)
            origin_point_distance = data.get("origin_point_distance", origin_point_distance)
            shield_distance = data.get("shield_distance", shield_distance)
            reletive_distance = data.get("reletive_distance", reletive_distance)
            influence_ratio = data.get("influence_ratio", influence_ratio)
            influence_range = data.get("influence_range", influence_range)
            non_influence_range = data.get("non_influence_range", non_influence_range)
            cross_replace_steps = data.get("cross_replace_steps", cross_replace_steps)
            self_replace_steps = data.get("self_replace_steps", self_replace_steps)
        except Exception as e:
            print(f"Load PKL metadata failed: {e}")

    return (
        source_prompt,
        target_prompt,
        mask,
        points,
        drag_type,
        rigid_type,
        origin_point_distance,
        shield_distance,
        reletive_distance,
        influence_ratio,
        influence_range,
        non_influence_range,
        cross_replace_steps,
        self_replace_steps,
    )
