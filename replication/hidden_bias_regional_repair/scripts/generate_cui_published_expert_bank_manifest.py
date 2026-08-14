#!/usr/bin/env python3
"""Generate the 5 experts x 8 Cui cells x 96 full-plus-CV bank."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


DESIGNS = ("cui_published_scenario1", "cui_published_scenario2")
SAMPLE_SIZES = (250, 500, 1000, 2000)
METHODS = ("aipw", "tmle", "ctmle", "cui_selective_ml", "ma_dr_bc")
CHUNKS = 24
REPLICATIONS = 4
SPLIT_SALT = "cui-published-expert-bank-split-v1-20260814"
BASE_SEED = 2_100_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    builder = args.builder.resolve()
    source_hash = sha256(source)
    builder_hash = sha256(builder)
    datasets = []
    for cell_index, (design, n) in enumerate(
        (design, n) for design in DESIGNS for n in SAMPLE_SIZES
    ):
        for chunk in range(CHUNKS):
            base_seed = BASE_SEED + (cell_index * CHUNKS + chunk + 1) * 100_000
            for replication in range(REPLICATIONS):
                identity = (
                    f"cui_published|{design}|n={n}|strength=1.0|"
                    f"chunk={chunk:02d}|rep={replication}|base_seed={base_seed}"
                )
                datasets.append((design, n, chunk, replication, base_seed, identity))
    by_cell: dict[tuple[str, int], list[str]] = {}
    for design, n, _, _, _, identity in datasets:
        by_cell.setdefault((design, n), []).append(identity)
    assignments = {}
    for cell, identities in by_cell.items():
        if len(identities) != 96:
            raise ValueError(f"cell {cell} has {len(identities)} datasets")
        ranked = sorted(
            identities,
            key=lambda identity: hashlib.sha256(
                f"{SPLIT_SALT}|{identity}".encode()
            ).hexdigest(),
        )
        for rank, identity in enumerate(ranked):
            assignments[identity] = "development" if rank < 48 else "validation"
    split_payload = {
        "schema": 1,
        "salt": SPLIT_SALT,
        "rule": "Within each Cui design x n cell, SHA256-rank 96 dataset realizations; first 48 development, last 48 validation; all five experts share assignment.",
        "assignments": assignments,
    }
    args.split_json.parent.mkdir(parents=True, exist_ok=True)
    args.split_json.write_text(json.dumps(split_payload, indent=2, sort_keys=True) + "\n")
    split_hash = sha256(args.split_json)
    rows = []
    for design, n, chunk, replication, base_seed, dataset_identity in datasets:
        for method in METHODS:
            entry_identity = f"{dataset_identity}|method={method}"
            owner = (
                "dml"
                if int(hashlib.sha256(entry_identity.encode()).hexdigest()[:8], 16) % 2 == 0
                else "dml2"
            )
            resample_seed = 1_000_000_000 + (
                int(
                    hashlib.sha256(f"resample|{entry_identity}".encode()).hexdigest()[:8],
                    16,
                )
                % 2_000_000_000
            )
            dataset_sha = hashlib.sha256(dataset_identity.encode()).hexdigest()
            output = (
                args.output_root
                / owner
                / assignments[dataset_identity]
                / "cui_published"
                / dataset_sha[:20]
                / method
            )
            command = [
                # Preserve the environment entry point. Resolving this symlink
                # escapes the venv and silently selects a Python without the
                # frozen XGBoost installation.
                str(args.python.absolute()),
                str(builder),
                "--frozen-source", str(source),
                "--output", str(output),
                "--method", method,
                "--design", design,
                "--mar-design", "box",
                "--n", str(n),
                "--epsilon", "0.05",
                "--strength", "1.0",
                "--base-seed", str(base_seed),
                "--replication", str(replication),
                "--resample-seed", str(resample_seed),
                "--repeated-crossfits", "20",
                "--delete-blocks", "0",
                "--bootstraps", "0",
                "--inner-folds", "3",
                "--learner", "xgboost",
                "--propensity-learner", "xgboost",
                "--propensity-mode", "estimated",
                "--tau-grid", "0.05",
                "--repair-mode", "if_residual",
                "--candidate-grid", "0.0", "0.25", "0.5", "1.0",
                "--validation-risk", "balanced_mse",
                "--validation-loss-se", "1.0",
                "--validation-region-weight", "-1",
                "--shrink-c", "2.0",
                "--selector", "obsval",
                "--lepski-c", "4",
                "--analysis-region", "estimated_residual_lowp_supported",
                "--region-quantile", "0.10",
                "--region-min-observed", "30",
                "--region-kappa-floor", "0",
                "--region-selector-ablation", "legacy",
                "--region-detector-c", "4.0",
            ]
            rows.append(
                {
                    "entry_identity": entry_identity,
                    "dataset_identity": dataset_identity,
                    "dataset_identity_sha256": dataset_sha,
                    "split": assignments[dataset_identity],
                    "owner": owner,
                    "group": "cui_published",
                    "design": design,
                    "method": method,
                    "n": n,
                    "strength": 1.0,
                    "chunk": chunk,
                    "replication": replication,
                    "base_seed": base_seed,
                    "source_sha256": source_hash,
                    "builder_sha256": builder_hash,
                    "split_sha256": split_hash,
                    "output": str(output),
                    "command_json": json.dumps(command, separators=(",", ":")),
                }
            )
    if len(rows) != 3_840 or len({row["entry_identity"] for row in rows}) != 3_840:
        raise ValueError("Cui bank must have exactly 3,840 unique entries")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["entry_identity"]))
    print(
        json.dumps(
            {
                "entries": len(rows),
                "dataset_realizations": len(assignments),
                "cells": len(by_cell),
                "owners": {
                    owner: sum(row["owner"] == owner for row in rows)
                    for owner in ("dml", "dml2")
                },
                "splits": {
                    split: sum(row["split"] == split for row in rows)
                    for split in ("development", "validation")
                },
                "manifest_sha256": sha256(args.manifest),
                "split_sha256": split_hash,
                "source_sha256": source_hash,
                "builder_sha256": builder_hash,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
