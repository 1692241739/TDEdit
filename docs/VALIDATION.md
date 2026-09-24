# Validation record — 2026-09-24

Status: **publicly accessible code-only snapshot; license/permissions pending**.
Repository: [1692241739/TDEdit](https://github.com/1692241739/TDEdit).
Frozen implementation commit: `7b9089d67f5bfa721c173d779df590a9b0b2a1a6`.
GPU/statistical checks below were performed on r1 before the code-only
packaging changes. No inference algorithms changed in the packaging pass.
Actual input protocols, sample identifiers and output fingerprints remain local.
The later visibility/status update does not add a new inference or benchmark test.

## Completed checks

- `python -B scripts/check_release.py`: source/JSON syntax and static release
  hygiene checks passed. It reports locations, not possible credential values.
- `python -B -m unittest discover -s tests -v`: 13 tests passed. These include
  exact primary settings, explicit seed overrides, author-view selection,
  missing/invalid inputs, non-mutating dry runs, preservation of existing output
  directories, and rejection of a zero-exit worker with missing output images.
- The five historical generation/evaluation entry points returned help
  successfully in the existing author environment. All 13 packaged evaluation
  scripts passed syntax and `--help` checks.
- Protocol: 204 cases, disjoint 41-case calibration and 163-case held-out sets.
  All 163 local source-image and author-hint hashes matched the protocol.
- Recomputed joint success and factorial statistics from the archived 163-case
  metric inputs; both outputs exactly matched the archived statistics. These
  checks did not rerun the image-level metrics or baseline generation.
- Two GPU smoke rounds on an allocated RTX 4090: text, drag and joint modes
  each completed one case per round (six runs total). Every run returned zero,
  recorded one completed timing entry and produced a valid 512-by-512 PNG.
  Drag/joint logs confirmed runtime SAM refinement and Depth Anything V2.
  Round two used the final wrapper's isolated working directory and exact
  text-mode inactive settings; output hashes matched round one for all modes.

The smoke input exercised a 3D non-rigid drag, not all geometric modes or the
complete benchmark. Its identity, annotations and output hashes are retained
in the private audit, not uploaded with this code-only repository.

## Scope and known limitations

- Tested in the existing Python 3.11 author environment, not a newly installed
  clean environment. See `INSTALL.md` for recorded dependency conflicts.
- No full benchmark rerun, multi-GPU throughput validation or complete external
  baseline-generation reproduction was performed in this packaging pass.
- Author hints and checkpoints are required separately. This package does not
  include benchmark photos, refined masks, weights or participant data.
- Legacy in-run diagnostic branch filenames are shared between workers; do not
  use those diagnostic files as a multi-GPU per-case audit. Use the uniquely
  keyed final results and per-run records; single-device execution is the
  validated path for this candidate.
- Source scans do not replace a security review or licensing clearance. Public
  access has been enabled, but the project license and permissions identified
  in `LICENSE_STATUS.md` remain unresolved.

The private original backup and candidate archives are separately hashed.
The upload manifest contains source-code file hashes only, not data hashes.
