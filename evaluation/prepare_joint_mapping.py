#!/usr/bin/env python3
"""Resolve the frozen joint protocol against locally supplied, verified assets.

No assets are downloaded or inferred. In particular, a source/official mask
cannot silently replace the author's coarse mask hint.
"""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _legacy_digest(stream)


def _legacy_digest(stream):
    sha = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        sha.update(block)
    return sha.hexdigest()


def checked_asset(root, relative, expected):
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Asset escapes its declared root: {relative}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Required local asset missing: {path}")
    if digest(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True, help="External local protocol JSON; annotations are not bundled with the code")
    parser.add_argument("--image-root", type=Path, required=True, help="Dataset root containing images/<case>.png")
    parser.add_argument("--hint-root", type=Path, required=True, help="Input supplement root containing masks/author_hint/<case>.png")
    parser.add_argument("--split", choices=("calibration", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    calibration, test = protocol["calibration_ids"], protocol["test_ids"]
    if len(calibration) != 41 or len(test) != 163 or set(calibration) & set(test):
        raise ValueError("Expected the frozen, disjoint 41/163 split")
    selected = protocol[f"{args.split}_ids"]
    output = {}
    for case_id in selected:
        row = protocol["cases"][case_id]
        hint = row["views"]["author_hint"]
        image_path = checked_asset(args.image_root, row["image_path_in_original_dataset"], row["image_sha256"])
        hint_path = checked_asset(args.hint_root, hint["mask_path_in_package"], hint["mask_sha256"])
        points = hint["points"]
        if not points or len(points) % 2 or any(len(point) != 2 for point in points):
            raise ValueError(f"{case_id}: expected alternating H,T pairs in (x,y) pixels")
        item = {key: hint[key] for key in ("category", "points", "source_prompt", "target_prompt", "drag_type", "influence_range")}
        item.update(image_path=image_path, mask_path=hint_path)
        output[case_id] = {"source": dict(item), "modified": dict(item)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(output)} verified author-view cases to {args.output}")
    print("The mapping contains coarse hints, not final SAM masks. Use the normal runtime SAM2 refinement.")


if __name__ == "__main__":
    main()
