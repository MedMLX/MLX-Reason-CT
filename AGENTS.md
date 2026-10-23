# MLX-Reason-CT

- Project: MLX-Reason-CT; package/CLI: `mlx-reason-ct`.
- Repository: `MedMLX/MLX-Reason-CT`; public code repository; retain separate model-material licensing.
- Python 3.13 or 3.14; pinned dependencies in `pyproject.toml` and `uv.lock`.
- Supported platform: macOS arm64 with MLX on Metal only, FP32 by default;
  explicit BF16 arithmetic profiles retain the pinned source boundaries.
  Host preprocessing, conversion and bundle verification are not inference backends.
  No runtime fallback. Test references are recorded on darwin-arm64.
- Port additions: Apache-2.0, Copyright (c) 2026 Joseph Sandoval.
  Preserve upstream OpenMDW-1.1, Apache-2.0, MIT and BSD notices.
- Keep the package independent of application registries and project/run frameworks.
- Preserve pinned checkpoint, tokenizer, template, geometry and numerical contracts.
- Run `make env` only when preparing or updating the environment.
- Use only synthetic CT fixtures; generated reports require human review.
- Keep weights and local outputs outside Git. Do not add hosted CI.
- Use the existing environment or temporary directories outside the checkout for
  release checks; do not retain copied checkouts or per-release environments.
- Keep unique measurements, settings, source/checkpoint revisions and test fixtures.
  Do not generate source/publication inventories, audit maps or duplicate receipts;
  document contracts in SOURCE.md and measurements in docs/verification.md.

## Source documentation

For new or changed owned source code, document actual behavior and non-obvious
rationale in docstrings and inline comments, including scientific/data contracts;
verify claims against the implementation, and preserve native documentation
conventions. Avoid boilerplate and unrelated, upstream, or generated
documentation rewrites.

## Node runtime for development

- Use a patched host Node for Pyright. The `pyright[nodejs]` extra was rejected
  because its wheel currently installs an older Node and can undo runtime
  security updates during environment synchronization.

## Validation scope

Choose the smallest check that covers the changed behavior. `make qa` runs
static checks in the existing environment; it does not run model tests or builds.
For behavior changes, run the affected tests with `make test TESTS=path/to/test.py`
(and `PYTEST_ARGS` for pytest options). Full suites, numerical parity, model
exports, benchmarks, live checks and package builds are explicit tasks, used
when the change or the user requires them. Documentation, file moves and metadata
edits do not require full QA. Do not synchronize environments, run sibling gates,
or repeat passed checks without a new relevant change. Mandatory full QA after
every edit was rejected because it adds unrelated work and delays feedback.
