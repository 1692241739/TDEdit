# Change log

## 2026-09-24 — r2, code-only private upload candidate

- Prepared for the author's private TDEdit repository; license still pending.
- Excluded the actual joint protocol and all per-case annotations, split lists
  and dataset fingerprints. The earlier r1 archive remains locally preserved.
- Require an explicit external `--protocol` when preparing a paper mapping.
- Removed smoke-case identifiers/output hashes from the uploaded documentation.
- Added code-only guardrails and refreshed the upload manifest. No editing
  algorithm or numerical experiment settings were changed.

## 2026-09-24 — r1, private release candidate

- Created a separate candidate from the evaluated author implementation after a
  full hashed backup. The original source and existing experiment results are
  not edited by this packaging pass.
- Added portable model/data/output paths and lazy SAM/depth imports. Preserved
  the author SAM configuration instead of replacing it with a different default.
- Added `run_release.py` with explicit text/drag/joint experiment settings,
  author-input validation, protected output folders and output-coverage checks.
  Legacy debug output is isolated inside each run directory.
- Packaged 13 evaluation entry points; r1's locally bundled frozen 41/163 joint
  protocol is excluded from the subsequent code-only upload.
  Images, mask binaries, checkpoints and participant records remain separate.
- Removed unused vendored Diffusers/PEFT directories from the candidate (not
  from the original backup); use the recorded installed packages instead.
- Added dependency, model, input-format, provenance and reproducibility notes;
  static checks and standard-library regression tests.
- Corrected candidate metadata before packaging: LCM Dreamshaper v7 generates
  images, SD 2.1 evaluates DIFT; text-mode drag couplings are effectively off.
- Project license remains pending. Third-party permission checks remain open.
  This version has **not** been publicly uploaded or designated open source.

See `docs/VALIDATION.md` for the actual tests and their limited scope. A local
version and a successful smoke test do not imply full benchmark reproduction.
