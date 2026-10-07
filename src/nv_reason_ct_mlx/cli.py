"""Standalone bundle download, verification, conversion and offline inference."""

from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path

from nv_reason_ct_mlx.api import generate_report
from nv_reason_ct_mlx.mlx_weights import convert_checkpoint, read_manifest


def main() -> int:
    parser = argparse.ArgumentParser(prog="nv-reason-ct-mlx", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download the FP32 bundle")
    download.add_argument("--repo", default="josand/NV-Reason-CT-MLX")
    download.add_argument("--revision", required=True, help="Pinned Hub commit")
    download.add_argument("--model-dir", type=Path, required=True)
    verify = commands.add_parser("verify", help="Verify every bundled file and FP32 shard")
    verify.add_argument("--model-dir", type=Path, required=True)
    convert = commands.add_parser("convert", help="Convert a pinned upstream checkpoint offline")
    convert.add_argument("--source-dir", type=Path, required=True)
    convert.add_argument("--output-dir", type=Path, required=True)
    report = commands.add_parser("report", help="Generate a report from one local HU NIfTI CT")
    report.add_argument("--input", type=Path, required=True)
    report.add_argument("--model-dir", type=Path, required=True)
    report.add_argument("--output-dir", type=Path, required=True)
    report.add_argument("--anatomy-region", choices=["chest", "abdomen"], default="chest")
    report.add_argument(
        "--prompt", help="Question or instruction (default: structured report for the region)"
    )
    report.add_argument("--enable-thinking", action="store_true")
    report.add_argument("--max-new-tokens", type=int, default=512)
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
    elif args.command == "convert":
        print(convert_checkpoint(args.source_dir, args.output_dir))
    else:
        result = generate_report(
            args.input,
            args.output_dir,
            model_dir=args.model_dir,
            prompt=args.prompt,
            anatomy_region=args.anatomy_region,
            enable_thinking=args.enable_thinking,
            max_new_tokens=args.max_new_tokens,
        )
        print(json.dumps(result, indent=2))
    return 0
