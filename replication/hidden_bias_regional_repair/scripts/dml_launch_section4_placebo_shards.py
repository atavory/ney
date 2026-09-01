#!/usr/bin/env python3
"""Launch Section 4 placebo-region shards by wrapping the public launcher."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--launcher-source", required=True, type=Path)
    parser.add_argument(
        "--analysis-region",
        default="estimated_residual_highp_matched_lowp_supported",
        choices=[
            "estimated_residual_lowp_supported",
            "estimated_residual_highp_supported",
            "estimated_residual_highp_matched_lowp_supported",
        ],
    )
    known, forwarded = parser.parse_known_args()
    return known, forwarded


def load_launcher(path: Path):
    spec = importlib.util.spec_from_file_location("section4_placebo_launcher", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load launcher: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def forwarded_value(args: list[str], name: str) -> str | None:
    try:
        return args[args.index(name) + 1]
    except (ValueError, IndexError):
        return None


def replace_analysis_region(command: tuple[str, ...], analysis_region: str) -> tuple[str, ...]:
    command_parts = list(command)
    try:
        region_index = command_parts.index("--analysis-region") + 1
    except ValueError as exc:
        raise SystemExit("launcher command did not contain --analysis-region") from exc
    command_parts[region_index] = analysis_region
    return tuple(command_parts)


def main() -> None:
    known, forwarded = parse_args()
    launcher = load_launcher(known.launcher_source.resolve())
    original_build_jobs = launcher.build_jobs

    def build_jobs(args):
        jobs = original_build_jobs(args)
        return [
            replace(job, command=replace_analysis_region(job.command, known.analysis_region))
            for job in jobs
        ]

    launcher.build_jobs = build_jobs
    sys.argv = [sys.argv[0], *forwarded]
    launcher.main()

    run_dir_value = forwarded_value(forwarded, "--run-dir")
    if run_dir_value is None:
        return
    provenance_path = Path(run_dir_value).resolve() / "provenance.json"
    if not provenance_path.exists():
        return
    provenance = json.loads(provenance_path.read_text())
    provenance["analysis_region"] = known.analysis_region
    provenance["placebo_launcher"] = str(Path(__file__).resolve())
    provenance["placebo_launcher_sha256"] = launcher.sha256(Path(__file__).resolve())
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
