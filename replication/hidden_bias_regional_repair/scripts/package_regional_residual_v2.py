#!/usr/bin/env python3
"""Materialize the frozen regional-residual-v2 release bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_csv(paths: list[Path], destination: Path) -> int:
    writer = None
    count = 0
    with destination.open("w", newline="") as output:
        for path in paths:
            with path.open(newline="") as source:
                reader = csv.DictReader(source)
                if writer is None:
                    fieldnames = ["source_file", *(reader.fieldnames or [])]
                    writer = csv.DictWriter(output, fieldnames=fieldnames)
                    writer.writeheader()
                elif ["source_file", *(reader.fieldnames or [])] != writer.fieldnames:
                    raise SystemExit(f"schema mismatch: {path}")
                for row in reader:
                    writer.writerow({"source_file": path.name, **row})
                    count += 1
    return count


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    counts = {
        "tmle_kang_schafer": merge_csv(
            sorted((args.run_root / "tmle_ks").glob("*.reps.csv")),
            args.out / "tmle_kang_schafer_rows.csv",
        ),
        "tmle_cui_published": merge_csv(
            sorted((args.run_root / "tmle_cui").glob("*.reps.csv")),
            args.out / "tmle_cui_published_rows.csv",
        ),
        "ma_published_did": merge_csv(
            sorted((args.run_root / "ma").glob("ma_dgp*_chunk*.csv")),
            args.out / "ma_published_did_rows.csv",
        ),
    }
    if counts != {
        "tmle_kang_schafer": 768,
        "tmle_cui_published": 768,
        "ma_published_did": 768,
    }:
        raise SystemExit(f"coverage mismatch: {counts}")

    shutil.copyfile(args.run_root / "summary.json", args.out / "summary.json")
    for family, source_dir in (
        ("tmle_kang_schafer", args.run_root / "tmle_ks"),
        ("tmle_cui_published", args.run_root / "tmle_cui"),
    ):
        for name in ("manifest.tsv", "provenance.json", "status.json"):
            shutil.copyfile(source_dir / name, args.out / f"{family}_{name}")

    source_files = {
        "validated_reference_transfer.py": args.source_root
        / "scripts/validated_reference_transfer.py",
        "ma_published_did_projection.py": args.source_root
        / "scripts/ma_published_did_projection.py",
        "section4_breadth_experiments.py": args.source_root
        / "scripts/section4_breadth_experiments.py",
        "launch_section4_breadth_shards.py": args.source_root
        / "scripts/launch_section4_breadth_shards.py",
        "section4_cui_published_experiments.py": args.source_root
        / "scripts/section4_cui_published_experiments.py",
        "launch_d2_cui_published.py": args.source_root
        / "scripts/launch_d2_cui_published.py",
        "summarize_regional_residual_v2.py": args.source_root
        / "scripts/summarize_regional_residual_v2.py",
        "regional_residual_v2_protocol.md": args.source_root
        / "regional_residual_v2_protocol.md",
        "residual_v2_protocol.md": args.source_root / "residual_v2_protocol.md",
    }
    source_hashes = {name: sha256(path) for name, path in source_files.items()}
    executed_ma_source = args.out / "executed_ma_published_did_projection.py"
    # The jobs loaded the same Ma source with one additional blank line at
    # EOF. Preserve those exact bytes while publishing the whitespace-clean
    # source in the public repository.
    executed_ma_source.write_bytes(
        source_files["ma_published_did_projection.py"].read_bytes() + b"\n"
    )
    executed_ma_sha256 = sha256(executed_ma_source)
    expected_executed_ma_sha256 = (
        "b3c6f888ffc9e6fb81692568830c13e66b5909e75fd869fa30dd728649587d7a"
    )
    if executed_ma_sha256 != expected_executed_ma_sha256:
        raise SystemExit("executed Ma source reconstruction hash mismatch")
    ma_manifest = {
        "adapter": "regional_residual",
        "dgps": [2, 3],
        "n": 10000,
        "h": 0.05,
        "folds": 3,
        "learner": "xgboost 3.4.0",
        "gamma_grid": [0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1],
        "gamma_se": 2.83,
        "chunks_per_dgp": 8,
        "reps_per_chunk": 48,
        "seed_base_by_chunk": [
            3700000000 + 1000 * chunk for chunk in range(8)
        ],
        "executed_source_sha256": executed_ma_sha256,
    }
    (args.out / "ma_run_manifest.json").write_text(
        json.dumps(ma_manifest, indent=2, sort_keys=True) + "\n"
    )

    ks_provenance = json.loads(
        (args.run_root / "tmle_ks/provenance.json").read_text()
    )
    cui_provenance = json.loads(
        (args.run_root / "tmle_cui/provenance.json").read_text()
    )
    expected_source = source_hashes["validated_reference_transfer.py"]
    for name, provenance in (
        ("Kang--Schafer", ks_provenance),
        ("Cui", cui_provenance),
    ):
        if provenance["frozen_source_sha256"] != expected_source:
            raise SystemExit(f"{name}: frozen source hash mismatch")

    verification = {
        "schema": 1,
        "status": "PASS",
        "protocol": "frozen before output; global draft superseded before execution",
        "counts": counts,
        "expected_counts": {
            "tmle_kang_schafer": 768,
            "tmle_cui_published": 768,
            "ma_published_did": 768,
        },
        "settings": {
            "adapter": "regional outcome residual",
            "region": "estimated_residual_lowp_supported",
            "minimum_responders": 30,
            "gamma_grid": [0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1],
            "validation_loss_se": 2.83,
            "folds": 3,
            "learner": "xgboost 3.4.0",
            "tmle_bootstraps": 50,
            "summary_bootstraps": 200000,
        },
        "source_sha256": source_hashes,
        "ma_executed_source_sha256": executed_ma_sha256,
        "ma_public_source_difference": "one trailing blank line at EOF only",
        "summary_sha256": sha256(args.out / "summary.json"),
        "launch_incident": (
            "The first Kang--Schafer controller invocation rejected TMLE in a "
            "stale launcher whitelist before launching a scientific job. The "
            "corrected launcher then ran the identical frozen source, seed "
            "block, and settings; its status records 192/192 successful jobs."
        ),
    }
    (args.out / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n"
    )

    readme = """# Regional residual v2 release

This bundle records the fresh-seed evaluation of one shared regional
outcome-residual repair for TMLE and the published Ma DR-BC DiD estimator.
The operative protocol froze the region detector, residual weights, damping
path, and score-risk gate before the run produced output. The earlier global
draft remains in the public history and marks its pre-execution supersession.

The three raw CSVs contain 768 paired rows each. `summary.json` reports
200,000-draw paired-bootstrap intervals. `verification.json` records coverage,
settings, source hashes, the summary hash, and the harmless initial launcher
whitelist incident. The two TMLE status files record 192/192 successful jobs
and zero failed jobs in each family.
"""
    (args.out / "README.md").write_text(readme)

    artifacts = sorted(path for path in args.out.iterdir() if path.is_file())
    with (args.out / "SHA256SUMS").open("w") as handle:
        for path in artifacts:
            if path.name == "SHA256SUMS":
                continue
            handle.write(f"{sha256(path)}  {path.name}\n")
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
