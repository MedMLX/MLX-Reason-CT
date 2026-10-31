# mlx-reason-ct

Native MLX port of NVIDIA NV-Reason-CT for 3D CT report generation and
question answering.

## Rules

- Allow `Any` at untyped third-party library boundaries; keep package interfaces typed.

- Model execution runs in MLX on Metal (macOS arm64). There is no PyTorch,
  MPS, NumPy or CPU fallback; host I/O and preprocessing may use CPU
  libraries.
- Unsupported checkpoints, options and inputs raise a clear error. Do not add
  compatibility shims or silent fallbacks.
- Never commit model weights, medical images, patient data or generated
  outputs.
- Keep upstream notices in THIRD_PARTY_NOTICES.md; model weights retain their
  publisher terms.
- FP32 by default; explicit BF16 profiles keep the pinned source boundaries.
- Preserve the pinned checkpoint, tokenizer, chat template and geometry
  contracts. Use only synthetic CT fixtures; generated reports require human
  review.
- NVIDIA model materials are OpenMDW-1.1; port contributions are Apache-2.0.

## Checks

- `make env` creates the locked development environment.
- `make qa` runs Ruff and Pyright; it does not run models.
- `make test TESTS=tests/test_name.py` runs selected tests; numerical changes
  need the affected comparison against the upstream reference.
- There is no hosted CI; local checks are authoritative.
