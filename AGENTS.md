# MLX-Reason-CT

- Project: MLX-Reason-CT; package/CLI: `mlx-reason-ct`.
- Repository: `sandovaljoseph/MLX-Reason-CT`; public code repository; retain separate model-material licensing.
- Preset: base. Python 3.12; pinned dependencies in `pyproject.toml` and `uv.lock`.
- Inference: macOS Apple Silicon Metal only, FP32 only. CPU preprocessing,
  conversion and bundle verification are portable. No runtime fallback.
- Port additions: Apache-2.0, Copyright (c) 2026 Joseph Sandoval.
  Preserve upstream OpenMDW-1.1, Apache-2.0, MIT and BSD notices.
- Keep the package independent of application registries and project/run frameworks.
- Preserve pinned checkpoint, tokenizer, template, geometry and numerical contracts.
- Run `make env`, `make qa`, and `make build` for package changes.
- Use only synthetic CT fixtures; generated reports require human review.
- Keep weights and local outputs outside Git. Do not add hosted CI.
