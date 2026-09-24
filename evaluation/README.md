# Evaluation and statistical analysis

These are the revision's evaluation/analysis scripts, adapted for portable
paths and fail-fast case matching. They do not download private data, publish
responses, regenerate all external baselines, or claim an end-to-end
reproduction without the separately obtained inputs/checkpoints. The pure
analysis commands require Python and NumPy; image evaluators also require the
repository's Torch, torchvision, Pillow, OpenCV, Transformers and Diffusers
environment. Run commands from the repository root.

## Paper settings versus software defaults

`configs/paper_primary.json` records the verified paper settings, **not a
direct runner configuration**. Use the documented release runner for generation.
Do not substitute defaults from historical UI/batch scripts for paper settings.

| Task | Configured steps | Strength | Executed updates | Source/target guidance | Seed |
|---|---:|---:|---:|---|---|
| Drag-only | 17 | 0.75 | 12 | 1.0 / 1.0 | 42 |
| Joint | 15 | 0.70 | 10 | 1.0 / 1.5 | 0, 42, 123 |
| Text-only | 12 | 1.0 | 12 | 1.0 / 2.1 | 0 |

Text-only cross/self replacement ratios are 0.8/0.7. Joint and drag use the
author annotations and fixed influence ratio 0.5. The independent extreme
stress protocol uses target guidance 2.0 and must not be presented as the
primary joint protocol. Configured steps differ from executed updates because
strength truncates the trajectory. The final factorial analysis contains 163
cases; the original exploratory utility had a 60-case default, changed here to
163. Historical `no_rkm` folder labels refer to the RKI ablation.

Image synthesis uses **LCM Dreamshaper v7**; Stable Diffusion 2.1 is only the
DIFT evaluator checkpoint. Text-mode RKI/RDM/LQM flags are **effectively off**.
Although the historical high-level launcher/canonical metadata requested RKI
and RDM, its text path overrides these requests before calling the core. The
archived `experiment_config.json` and runtime logs confirm the disabled flags
and reference-branch output. Recorded text RDM maximum/start 0.35/0.6 are inactive
parameters, not evidence that RDM ran. The release metadata distinguishes the
historical request from the effective configuration.

## Build a verified mapping

Follow `protocols/README.md` and pass an external local `--protocol` explicitly.
Source/result images, masks, annotation JSONs and case lists are omitted
from this code-only repository. A mapping is an object keyed by dataset `case_id`, with
`source` and `modified` objects; `modified` overrides `source`. Required effective
fields are `image_path`, `mask_path`, `points` (alternating H,T in x,y pixels),
`source_prompt`, `target_prompt`; `category` and `drag_type` enable strata.
`prepare_joint_mapping.py` emits author-view entries in both slots so no
source-view fallback can occur. Use normal SAM2 refinement for generation;
the metric edit-region mask is the fixed input mask, not an output-chosen mask.

## Compute joint metrics

Generated PNGs must be named `<case_id>.png`, directly in the result directory
or in its `results/` subdirectory. The two image evaluators preserve the original
metric definitions. They output errors and coverage; the scorer rejects missing
coverage by default. Image evaluation requires GPU/model assets and is not part
of the lightweight CPU smoke test.

```bash
python evaluation/eval_joint_semantics.py \
  --result-dir outputs/tdedit/seed42 --data-root . \
  --mapping outputs/joint_test.mapping.json \
  --clip-model /path/to/clip-vit-large-patch14 --device cuda:0 \
  --output outputs/tdedit/seed42/semantics.json

python evaluation/eval_joint_geometry.py \
  --result-dir outputs/tdedit/seed42 --data-root . \
  --mapping outputs/joint_test.mapping.json \
  --sd-path /path/to/stable-diffusion-2-1 --device cuda:0 --seed 42 \
  --output outputs/tdedit/seed42/geometry.json

python evaluation/score_joint_success.py \
  --semantics outputs/tdedit/seed42/semantics.json \
  --geometry outputs/tdedit/seed42/geometry.json \
  --output outputs/tdedit/seed42/joint_score.json
```

The DIFT feature seed is independent of the generation seed. The geometry
evaluator uses the packaged `run_evaluations/eval_drag.py` helpers; `--eval-dir`
can explicitly select a compatible directory. Its default local feature cache
is `.cache/dift_source_handle_vectors`. Inputs must be trusted image files.

### Metric schemas and units

- Both metric JSONs have `cases: [...]`, unique `case_id`, and optional
  `coverage.mapping` indicating requested count.
- Geometry rows need `point_count`, `mean_distance_px`, and
  `mean_distance_normalized` (pixel MD divided by image diagonal).
- Semantics rows need `clip_directional` and `outside_mse`. MSE uses RGB values
  in [0,1] outside the union of the source mask and translated masks, dilated
  by 3% of the image diagonal. Do not convert MSE to percent.
- Primary joint success is normalized MD <= 0.05 **and** directional CLIP >= 0.
  Outside MSE <= 0.01 is a separately reported three-way diagnostic, not the
  universal primary gate, since some prompts intentionally change background.
- Utility is the harmonic mean of exp(-MD/0.05) and
  sigmoid(directional CLIP/0.05). The scorer also reports threshold sensitivity.
- Missing/nonfinite metrics are rejected, not replaced with zero. The optional
  `--allow-partial` scorer flag is diagnostic only and changes the analyzed set.

## Aggregate methods and seeds

Edit the example paths to your generated metrics. Paths inside the config are
resolved relative to the config file. Never call unreproduced result paths a
completed baseline run. Methods/seeds must contain identical case IDs.

```bash
python evaluation/aggregate_joint_baselines.py \
  --config configs/joint_methods.example.json \
  --output outputs/joint_comparison.json --markdown outputs/joint_comparison.md

python evaluation/analyze_factorial_interactions.py \
  --metrics-root outputs/ablations --expected-cases 163 \
  --output outputs/factorial_interactions.json
```

Ablation folders: `full`, `no_lqm`, `no_rkm`, `no_lqm_no_rkm`,
`appearance_anchor_only`, `geometry_anchor_only`, `neither_anchor`; each contains
`joint_score.json`. Seeds are averaged within case before the case-level
bootstrap (10,000 draws). Comparisons use paired sign-flip tests (100,000 draws)
and Holm correction for the pre-specified success/utility comparisons. Cases,
not points or seeds, are the unit of analysis. See the scripts for exact tests.

## Other analyses

Each command exposes the complete CLI with `--help`. No paths to the authors'
machine are required. Inputs below are metrics, not raw participant responses.

| Script | Inputs and expected layout |
|---|---|
| `analyze_mode_mismatch.py` | `--metrics` Drag-evaluator JSON with six runs; each `_hyperparams.experiment_config.drag_type` is one of 2D/3D-Rigid/Non-Rigid/Hybrid. `--mapping` maps IDs to author `drag_type`. Outputs `--output`, `--markdown`. |
| `analyze_depth_sensitivity.py` | `--metrics` Drag JSON with V1/V2 runs; set `--v1-run` and `--v2-run` explicitly. Optional `--disagreement` has `cases:[{case_id,disagreement_tertile}]`. Outputs JSON/Markdown. |
| `analyze_depth_perturbations.py` | `--clean-combined` Drag JSON, `--clean-run` explicit key; `--perturb-root` contains `gaussian_005.json`, `gaussian_010.json`, `smooth_4.json`, `smooth_8.json`, `order_invert.json` (one Drag run each). Outputs JSON/Markdown. |
| `analyze_mask_sensitivity.py` | `--clean`, `--erode5`, `--dilate5`, `--shift10`: one-run Drag JSONs; `--subset`: case-ID array/object; `--perturbation-summary`: perturbation metadata; `--output`. |
| `analyze_stratified_performance.py` | `--manifest` has `cases:{id:{displacement_stratum,object_size_stratum,complexity_stratum}}`; `--natural-extreme`: ID array/object; `--drag-metrics`: one-run Drag JSON; `--joint-score`: scorer JSON; outputs JSON/Markdown. Strata are fixed from inputs, not results. |
| `analyze_efficiency.py` | `--manifest` has `samples:[{sample_id,role,drag_type}]` with measured/warmup roles; `--tdedit`, `--fastdrag-steady` contain ID-keyed `image_times_sec`, `peak_memory_allocated_bytes`; TDEdit additionally has `stage_times_sec` and optional `summary.model_load_seconds_by_worker`; optional `--fastdrag-cold`; outputs JSON/Markdown. |
| `analyze_extreme_stress.py` | `--metrics-root` has `default_1p5`, `selected_1p5`, `default_2p0`, `selected_2p0`, each with `joint_score.json`; `--selection` has `selected.configuration` chosen only on calibration cases; outputs JSON/Markdown. |

Drag-evaluator JSON shape is `{"Drag":{"run-key":{"_per_image":{"id":{...}},
"_hyperparams":{...},"MD":...,"LPIPS":...,"1-LPIPS":...,"CLIP_Sim":...}}}`.
The drag evaluator stores LPIPS/IF/CLIP in percent units; the sensitivity
scripts preserve those units, while joint directional CLIP is unscaled.
Efficiency metadata describes the paper's RTX 4090, 512x512, FP16, batch-one,
CUDA-synchronized protocol, not automatic detection of the current hardware.
Its “17 steps” denotes configured drag steps (12 executed updates); FastDrag
uses its native 10-step setting. Do not apply that report unchanged to another
hardware/timing protocol. Mode, depth, and extreme analyses likewise target
their frozen paper protocols, not arbitrary datasets without adapting labels.

No participant-level questionnaire files or scripts that operate on private
responses are included. Code presence and CPU tests do not claim all GPU
experiments or external baselines were rerun for this release.
