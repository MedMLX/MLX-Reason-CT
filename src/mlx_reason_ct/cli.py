"""Standalone bundle download, verification and offline inference."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import cast

from mlx_reason_ct.api import PRECISION_PROFILES, generate_report
from mlx_reason_ct.mlx_weights import read_manifest


def main() -> int:
    """Run the selected download, verify or report subcommand; return the exit status."""
    parser = argparse.ArgumentParser(prog="mlx-reason-ct", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download the FP32 bundle")
    download.add_argument(
        "--repo", default="MedMLX/MLX-Reason-CT", help="Hub repository (default: %(default)s)"
    )
    download.add_argument("--revision", required=True, help="Pinned Hub commit")
    download.add_argument("--model-dir", type=Path, required=True)
    verify = commands.add_parser(
        "verify", help="Check manifest metadata and required file presence"
    )
    verify.add_argument("--model-dir", type=Path, required=True)
    report = commands.add_parser("report", help="Generate a report from one local HU NIfTI CT")
    report.add_argument("--input", type=Path, required=True)
    report.add_argument("--model-dir", type=Path, required=True)
    report.add_argument("--output-dir", type=Path, required=True)
    report.add_argument(
        "--anatomy-region",
        choices=["chest", "abdomen", "none"],
        default="chest",
        help="'none' uses the pinned processor's own centered crop, with no "
        "chest/abdomen lung detection",
    )
    report.add_argument(
        "--prompt", help="Question or instruction (default: structured report for the region)"
    )
    report.add_argument("--enable-thinking", action="store_true")
    report.add_argument("--max-new-tokens", type=int, default=512)
    report.add_argument("--precision", choices=tuple(PRECISION_PROFILES), default="float32")
    report.add_argument("--overwrite", action="store_true", help="Replace existing report files")
    report.add_argument(
        "--history",
        type=Path,
        help="JSON file holding a list of already-answered [question, answer] "
        "turns, oldest first, to continue a conversation about the same CT",
    )
    args = parser.parse_args()
    if args.command == "download":
        hub = import_module("huggingface_hub")
        hub.snapshot_download(args.repo, revision=args.revision, local_dir=args.model_dir)
        read_manifest(args.model_dir)
        print(args.model_dir)
    elif args.command == "verify":
        manifest = read_manifest(args.model_dir)
        print(
            json.dumps(
                {
                    "verified": True,
                    "revision": manifest["revision"],
                    "dtype": manifest["dtype"],
                    "shards": len(manifest["shards"]),
                }
            )
        )
    else:
        history: Sequence[tuple[str, str]] | None = None
        if args.history is not None:
            # Preserve untrusted JSON for the API's history validator. Stringifying
            # entries would turn malformed values into plausible conversation text.
            history = cast(
                Sequence[tuple[str, str]],
                json.loads(args.history.read_text(encoding="utf-8")),
            )
        result = generate_report(
            args.input,
            args.output_dir,
            model_dir=args.model_dir,
            prompt=args.prompt,
            anatomy_region=None if args.anatomy_region == "none" else args.anatomy_region,
            enable_thinking=args.enable_thinking,
            max_new_tokens=args.max_new_tokens,
            precision=args.precision,
            history=history,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, indent=2))
    return 0
