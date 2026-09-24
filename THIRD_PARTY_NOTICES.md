# Third-party notices

This inventory documents dependencies and visible source notices; it does not relicense them or replace upstream terms.

| Component | Relationship | Source / license notice |
| --- | --- | --- |
| Hugging Face Diffusers | Installed runtime; custom-derived modules in `utils/unet_drag/` retain copyright headers | [Diffusers](https://github.com/huggingface/diffusers), Apache-2.0 |
| Hugging Face PEFT | Installed runtime dependency, version 0.13.2 | [PEFT](https://github.com/huggingface/peft), Apache-2.0 |
| Meta SAM 2 | Separately installed runtime; frozen author configuration retained for reproduction | [SAM 2](https://github.com/facebookresearch/sam2), upstream license applies |
| Depth Anything V2 | Separately supplied code and ViT-B weights | [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2); ViT-B weights CC-BY-NC-4.0 |
| Depth Anything V1 | Optional sensitivity-test dependency | [Depth Anything](https://github.com/LiheYoung/Depth-Anything), upstream terms apply |
| LCM Dreamshaper v7 | Separately downloaded editing backbone | [Model card](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7); model and underlying-model terms apply |
| InfEdit | Matching alignment functions and matching/modified attention utilities; core classes also require provenance review | [InfEdit](https://github.com/sled-group/InfEdit), [CC-BY-NC-ND-4.0 license](https://github.com/sled-group/InfEdit/blob/main/LICENSE.txt) |
| Prompt-to-Prompt | Earlier upstream for some alignment/attention routines; not an assumed license for later InfEdit additions | [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt), [Apache-2.0 license](https://github.com/google/prompt-to-prompt/blob/main/LICENSE) |

The historical backup's unused vendored Diffusers/PEFT trees are not installed by this release. Their original licenses remain in the private source backup. Apache notices in active customized UNet files are preserved; the license is included as [DIFFUSERS_LICENSE](docs/licenses/DIFFUSERS_LICENSE). The frozen SAM configuration carries [SAM2_LICENSE](configs/SAM2_LICENSE). These notices must accompany redistribution of the respective components.

Source comparison against the official InfEdit implementation identifies exact syntax-tree matches for all 11 effective top-level definitions in `utils_text/seq_aligner.py`; `utils_text/ptp_utils.py` contains five exact top-level matches and a modified `register_attention_control`. Shared/modified classes in `tdedit_core.py` also need permission and provenance review. Acknowledgement alone is not permission to redistribute derivatives under an incompatible license. See [LICENSE_STATUS.md](LICENSE_STATUS.md) before publishing.

No pretrained weight, dataset, or survey-data rights are conveyed by a future license for original TDEdit code.
