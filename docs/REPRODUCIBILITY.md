# Reproducibility and release boundaries

## What this code-only snapshot preserves

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

## Public access and remaining release requirements

The code-only repository is publicly accessible at [1692241739/TDEdit](https://github.com/1692241739/TDEdit), with frozen implementation commit `7b9089d67f5bfa721c173d779df590a9b0b2a1a6`. Later status-documentation updates do not change that implementation. Research inputs, case-level annotations, participant data, results and weights remain excluded.

Public access does not resolve the project-code license or permissions for InfEdit-derived components; consult `LICENSE_STATUS.md`. Resolve those permissions before describing the repository as a cleared open-source release. The completed checks and their limits remain in `VALIDATION.md`; the visibility change is not evidence of a clean-environment installation or a full benchmark rerun. Manuscript and response-letter wording must distinguish publicly accessible code from complete reproduction materials and licensing clearance.
