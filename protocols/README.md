# External input protocol — no annotation data included

The actual protocol JSON is NOT included in this code-only repository.
It must be supplied as a separate local input, and must not be committed.
The paper protocol records 41 calibration IDs and 163 disjoint held-out IDs (split seed
20260825), revision-specific prompts, alternating handle/target points in
`(x,y)` pixel coordinates, user-selected modes, coarse author-mask hint paths,
and SHA-256 fingerprints. Case IDs are dataset IDs, not participant identifiers.

This is an **author-annotated research protocol**, not an official DragBench
joint-editing benchmark. `views.source` and `views.author_hint` must not be
silently interchanged. Some target prompts and editing controls were authored
or revised for these experiments. The source view is retained for provenance,
not as a fallback when author annotations are unavailable.

No source/result images, mask PNGs (coarse or post-SAM), annotation JSONs,
case lists, per-case prompts/points/fingerprints, checkpoints, raw human
responses, participant IDs, or machine-environment dumps are included here.
The historical field `mask_path_in_package` refers to the **separate input
supplement**, not to files bundled in this repository. Obtain the images under
their original dataset terms and supply the authors' coarse-hint supplement
locally. Missing hints should stop the run, not trigger official-mask fallback.

Verify and prepare a local mapping:

```bash
python evaluation/prepare_joint_mapping.py \
  --protocol /path/to/local/joint_protocol.json \
  --image-root /path/to/dragbench \
  --hint-root /path/to/joint-input-supplement \
  --split test --output outputs/joint_test.mapping.json
```

This verifies every input image and author-hint hash and writes absolute local
paths to a runtime `source`/`modified` mapping. Treat that generated mapping as
a local artifact; it need not be committed. Repeat with `--split calibration`
for calibration inputs. The helper does not tune parameters, infer annotations,
run generation, or create final masks.

Author hints are refined by the normal runtime SAM2 path. Refinement normally
intersects a selected SAM candidate with the hint, with the code's near-empty
fallback. Do not bypass refinement or label hints as final SAM masks. Exact
segmentation depends on the stated checkpoint and runtime implementation.

The primary 163-case comparison aggregates seeds 0, 42, and 123 within each case
for TDEdit and CLIPDrag, whereas other controls have one seed. The released
metric utilities do not remove this unequal seed coverage or make a partial
control equivalent to a native joint method. See `configs/paper_primary.json`
and `evaluation/README.md` for scopes and endpoints.
