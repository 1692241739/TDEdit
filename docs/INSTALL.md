# Installation

## Recorded environment versus validated release

The preserved timing-run dependency manifest records PyTorch 2.9.1, TorchVision 0.24.1, Diffusers 0.31.0, PEFT 0.13.2, Transformers 4.46.1, Accelerate 1.0.1, and NumPy 1.26.4. The inspected inference environment uses Python 3.11.0. These observations guide the requirements; they are not evidence of a fresh installation or a complete benchmark rerun.

The old backup also contains separate Diffusers and PEFT source trees. They are **not** installed by these instructions: the inspected runtime resolves the installed packages, and the unused PEFT source reports an older 0.3.0 development version. Active customized UNet code is retained in `utils/unet_drag/`.

## Isolated environment

Use Linux and an NVIDIA GPU with sufficient memory. Do not change a working experiment environment in place.

```bash
conda create -n tdedit-release python=3.11
conda activate tdedit-release
python -m pip install --upgrade pip
python -m pip install -r requirements-torch.txt
python -m pip install -r requirements.txt
```

Select the appropriate official PyTorch CUDA wheel index when needed. The two pinned torch packages must remain a compatible pair; package versions alone do not specify the driver or CUDA runtime. Record both for timings.

Install [SAM 2](https://github.com/facebookresearch/sam2) from its upstream repository, including its dependencies and matching checkpoint/configuration. Consult its installation instructions for CUDA-extension support. Keep an exact checkout revision in your experiment record; unpinned upstream `main` is not a reproducibility identifier.

Download the [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) source separately and supply its directory using `--depth-v2-repo`; do not replace the imported implementation with a different Transformers API. V1 is an optional sensitivity-test backend, not the default. See [model setup](MODELS.md).

```bash
python -m pip check
python run_release.py --help
python -m unittest discover -s tests
python scripts/check_release.py
```

Read the reported failures instead of assuming a passed CLI test implies complete installation. `--dry-run` validates the invocation without loading a diffusion model. Actual GPU inference and benchmark metrics require the external data and weights.

## Evaluation environment caveat

The historical environment is not a clean lockfile: its `scikit-image==0.25.2` installation requests `Pillow>=10.1`, while the recorded Pillow is 9.5.0. Its installed Ninja also reports a platform mismatch. The inference requirements therefore do not copy every historical package or install Ninja/scikit-image automatically. Resolve optional metric dependencies in a separate evaluation environment and record the exact versions; a changed dependency environment must not be represented as a bit-identical rerun. See `evaluation/README.md` for metrics and inputs.

Only package/version allowlists belong in a public environment record. Do not publish complete shell-environment dumps, access tokens, private hostnames, or account paths.
