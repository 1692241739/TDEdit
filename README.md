# TDEdit

Research implementation for text editing, drag editing, and joint text-and-drag editing. This code-only snapshot packages the authors' evaluated implementation with configurable paths and a reproducible command-line entry point. It does not train a new model.

**Release status:** a code-only snapshot is publicly accessible at [1692241739/TDEdit](https://github.com/1692241739/TDEdit). The frozen implementation commit is [`7b9089d67f5bfa721c173d779df590a9b0b2a1a6`](https://github.com/1692241739/TDEdit/tree/7b9089d67f5bfa721c173d779df590a9b0b2a1a6). The project license and third-party permissions remain pending: public access does not establish an open-source license or permission to redistribute adapted components. See [license status](LICENSE_STATUS.md).

**No research data is included:** no dataset images, masks, sample annotations,
case lists, per-case prompts/coordinates/fingerprints, experiment results,
questionnaire records or pretrained weights. Only code, configuration,
documentation and synthetic test/schema examples are supplied.

## Quick start

1. Create an isolated Python environment following [installation](docs/INSTALL.md).
2. Download the editing, segmentation, and depth models from their original providers; see [models](docs/MODELS.md). We do not redistribute weights.
3. Prepare an input mapping with your own image, prompts, handle/target pairs, and author-drawn mask hints; see [data format](docs/DATA_FORMAT.md).
4. Inspect the command before starting GPU inference:

```bash
python run_release.py --mode joint \
  --data-root /path/to/data --mapping /path/to/mapping.json \
  --output outputs/joint --device 0 \
  --model-path /path/to/LCM_Dreamshaper_v7 --limit 1 --dry-run
```

Remove `--dry-run` only after the dependencies, models, and inputs have been validated. Run `python run_release.py --help` for the available model-path options. Use `--mode drag` for unchanged-text drag editing or `--mode text` for text-only editing. The wrapper calls `run_batch.py` with explicit settings; it does not silently invoke a separate baseline or metric pipeline.

Drag and joint inputs use the **author annotations followed by SAM refinement**, not an automatic substitution of official DragBench point pairs or masks. The displayed original hint is not the refined inference mask. Keep both inputs and the refined outputs when auditing an experiment.

## Contents

| Path | Purpose |
| --- | --- |
| `run_release.py` | Portable command-line entry point |
| `run_batch.py`, `tdedit_core.py` | Batch inference and editing pipeline |
| `utils_drag/`, `utils_text/`, `utils/` | Geometry, depth, segmentation, attention, and custom UNet modules |
| `configs/` | Explicit paper settings and segmentation configuration |
| `evaluation/` | Evaluation commands and their data requirements |
| `tests/`, `scripts/check_release.py` | Release-level validation |
| `docs/` | Installation, models, input format, and reproducibility notes |

The historical GUI and launcher are retained for provenance; use `run_release.py` as the documented entry point. The GUI is a local research tool, not a production web service. If using it locally, pass `--host 127.0.0.1 --no-share`; the historical host default can expose it to the network. Do not expose its server publicly without a separate security review.

## Reproduction scope

See [validation actually completed](docs/VALIDATION.md), [reproducibility](docs/REPRODUCIBILITY.md) and [evaluation](evaluation/README.md). A successful import or dry run does not establish GPU equivalence or reproduce a paper table. Reproducing a reported result also requires the same case list, author annotations, model files, settings, software, and metric definitions. No participant records, private survey data, model weights, or benchmark photographs are included in this code package.

## 中文说明

先安装依赖并单独下载模型，再准备自己的图像、提示词、手工点对和 mask hint。先运行上面的 `--dry-run` 检查配置，确认后再推理。拖拽和联合编辑按作者标注经 SAM 精炼的流程处理，不能把官方点对／mask 当作等价输入。代码仓库目前可以公开访问，但不包含实验数据或模型权重，也不表示全部论文实验已在新环境重跑。项目许可证及第三方改编代码的授权仍待解决，不将“公开可访问”等同于“授权完善的开源发布”。
