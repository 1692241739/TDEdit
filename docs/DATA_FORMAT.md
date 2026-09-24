# Input data and annotations

The mapping is a JSON object keyed by case identifier. Images and masks are referenced relative to `--data-root`. Use your own or properly licensed images; this package does not redistribute benchmark photographs.

An illustrative flat record is:

```json
{
  "example_001": {
    "image_path": "images/example_001.png",
    "mask_path": "masks/example_001.png",
    "source_prompt": "a red toy car",
    "target_prompt": "a blue toy car",
    "points": [[120, 180], [160, 180]],
    "drag_type": "2D-Rigid"
  }
}
```

The example contains schema values only, not an included paper case. Use the exact supported drag-mode spelling shown by the runner/code when creating a new annotation. Coordinates are alternating image-space `[x,y]` pairs: `[H1,T1,H2,T2,...]`, with the origin at the upper-left. H is the source feature; T is its intended destination. Coordinates and mask dimensions must match the original image. Do not convert to `[row,column]` or normalized coordinates unless explicitly required by a separate metric adapter.

## Author hints are refined, not substituted

`mask_path` identifies the author-drawn binary hint: white is selected and black is unselected. The normal drag/joint path refines the selected region through SAM before geometric editing. An input hint, a SAM-refined inference mask, and a visualization overlay are distinct artifacts. Use the original author point pairs and hints for reported paper cases; official benchmark annotations are not interchangeable.

The historical batch reader understands nested `source`, `modified`, and `user_study` records. `modified` merges author corrections over `source`; `user_study` is a separate view. For new public-release inputs, prefer a flat, fully resolved mapping to make provenance unambiguous. Missing masks, prompts, or point pairs must be corrected before evaluation, not silently replaced with another dataset's controls.

For text-only mode no drag pairs are needed. For pure drag mode the source and target text are identical; the batch path forces unchanged text. For joint mode both the text change and drag controls are part of the requested edit. Do not infer new target prompts from a filename or from a visually plausible result.

Keep a case manifest identifying the source image, annotation version, selected view, prompts, points, hint hash, refined-mask hash, and model/configuration hashes. Keep survey answers and participant identifiers out of code releases.
