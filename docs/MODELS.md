# Models and external code

All weights are downloaded separately under their providers' terms. Model downloads are not performed silently by the release wrapper.

| Component | Expected artifact | Official source |
| --- | --- | --- |
| Editing backbone | Complete `LCM_Dreamshaper_v7` Diffusers model directory | [LCM Dreamshaper v7](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7) |
| Mask refinement | `sam2.1_hiera_large.pt`, SAM 2 implementation, frozen author configuration | [SAM 2](https://github.com/facebookresearch/sam2) |
| Default relative depth | Depth Anything V2 source and `depth_anything_v2_vitb.pth` | [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) |
| Optional V1 sensitivity | Depth Anything V1 source and `depth_anything_vitb14.pth` | [Depth Anything V1](https://github.com/LiheYoung/Depth-Anything) |

The editing model and the metric model are different: **LCM Dreamshaper v7 performs editing; Stable Diffusion 2.1 is used by the separate DIFT metric pipeline.** Do not substitute the metric checkpoint for the editing backbone. See the evaluation instructions before downloading metric weights.

## Paths

Run `python run_release.py --help` for supported model flags. Set the editing directory with `--model-path` or `TDEDIT_MODEL_PATH`. Provide SAM checkpoint/configuration and the depth source/checkpoint directory explicitly when not using the documented checkout-relative defaults.

The frozen `configs/sam2_hiera_l.yaml` preserves the authors' configuration, whose SHA-256 is:

```text
0ee9f2037fd98489212bb0b78a56f33ac82eb177cf5e52c88be3714cb84eea26
```

Its filename is historical. Do not replace it based only on its name: the standard upstream SAM 2.1 example uses a differently named configuration. The stored author configuration also differs in object-pointer/video-memory options (for example, `add_tpos_enc_to_obj_ptrs` is false). Those settings are preserved rather than silently upgraded during packaging. Configuration and checkpoint matching must be validated together. Retain warnings about missing/unexpected checkpoint keys in the run log and investigate them before claiming equivalence.

Optional V1 sensitivity runs use `TDEDIT_DEPTH_BACKEND=v1`, `TDEDIT_DEPTH_V1_REPO`, and `TDEDIT_DEPTH_V1_CHECKPOINT`; obtain the upstream V1 source, including its local DINOv2 hub files. This is a separate experimental condition. Metric paths include `TDEDIT_EVAL_SD_PATH` (DIFT) and `TDEDIT_CLIP_MODEL_PATH` (CLIP); neither changes the editing backbone.

## Terms and reproducibility

The upstream Depth Anything V2 documentation assigns **CC-BY-NC-4.0** to the Base/Large/Giant weights, including the default ViT-B used here; its Small weights have different terms. Replacing Base with Small changes the experimental condition. See the upstream model notices before any non-research use.

Record model repository/revision, local checkpoint SHA-256, config SHA-256, and external-code revision with every run. Do not upload your checkpoint cache, private authentication settings, or model downloads to this repository. The release does not grant rights to third-party weights or datasets.
