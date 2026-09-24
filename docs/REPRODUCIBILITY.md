# Reproducibility and release boundaries

## What this candidate preserves

The release is derived from the evaluated author implementation. Portable paths and the release entry point are packaging changes, not a new editing method. The private backup is kept separately; it must not be committed with the release. Parameter metadata is in `configs/paper_primary.json`; use the explicit release command rather than assuming the historical GUI/launcher defaults reproduce a paper condition.

In particular, text-only, pure-drag, primary joint, and extreme-condition calibration runs have different settings. The historical launcher defaults are not the experiment registry. A result obtained with a retuned guidance or scheduling value belongs to that condition and must not replace a fixed-default result.

## Validation levels

1. **Static checks:** source syntax, required files, portable paths, configuration construction, and release hygiene.
2. **Runtime checks:** package imports, model/configuration loading, and one-case inference on allocated hardware.
3. **Reproduction:** the exact evaluation case list and annotations, all required model files, frozen parameters/seeds, metric definitions, and aggregation.

Passing an earlier level does not imply a later one. The release record must state which commands actually ran, their exit status, and any missing prerequisite. An existing experiment log is evidence for the original run, not proof that the repackaged code has completed a clean-room run.

## Per-run record

Record code version/hash; case list and mapping hashes; input image/hint hashes; author point pairs; SAM configuration and resulting refined masks; model names and checkpoint hashes; seed and every overridden flag; Python/package versions; GPU, driver, and CUDA information; actual outputs and metric commands. For timing experiments also retain warm-up count, synchronization, batch size, precision, and loading/I/O exclusions.

The code-only package does not include participant records, private API credentials, benchmark images, masks, annotation JSONs, case lists, per-case fingerprints, experiment outputs or pretrained weights. Their absence is intentional. Supply the input protocol externally using `--protocol`; evaluation scripts alone do not reproduce the reported tables without those inputs. Consult [evaluation](../evaluation/README.md).

## Publishing gate

Before labeling this a public release: select a repository with confirmed authorization; resolve the project-code license and third-party provenance; run the release checks; verify at least representative inference and its inputs; inspect the exact upload manifest; then record the public commit/tag. Only a verified accessible URL may be inserted into the manuscript and response letter. A local archive is not a GitHub publication.
