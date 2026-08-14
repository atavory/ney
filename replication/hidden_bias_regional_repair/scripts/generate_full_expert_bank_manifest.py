#!/usr/bin/env python3
"""Generate the complete 5 x 34 x 96 frozen fitted-value bank."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


SPLIT_SALT = "full-expert-bank-dataset-split-v1-20260814"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def option(command, name, *, multiple=False, default=None):
    if name not in command:
        return default
    start = command.index(name) + 1
    if not multiple:
        return command[start]
    values = []
    for value in command[start:]:
        if value.startswith("--"):
            break
        values.append(value)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-manifest", type=Path, action="append", required=True)
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
    baseline_hashes = {str(path.resolve()): sha256(path) for path in args.baseline_manifest}
    expanded = []
    for baseline in args.baseline_manifest:
        with baseline.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                command = json.loads(row["command_json"])
                for replication in range(4):
                    dataset_identity = (
                        f"{row['group']}|{row['design']}|n={row['n']}|"
                        f"strength={row['strength']}|chunk={int(row['chunk']):02d}|"
                        f"rep={replication}|base_seed={row['seed']}"
                    )
                    expanded.append((row, command, replication, dataset_identity))
    # Each dataset realization must occur once per method and exactly five times.
    identities = sorted({item[3] for item in expanded})
    by_cell = {}
    for identity in identities:
        parts = identity.split("|")
        cell = "|".join(parts[:4])
        by_cell.setdefault(cell, []).append(identity)
    assignments = {}
    for cell, values in by_cell.items():
        if len(values) != 96:
            raise ValueError(f"cell {cell} has {len(values)} dataset realizations, expected 96")
        ranked = sorted(
            values,
            key=lambda identity: hashlib.sha256(
                f"{SPLIT_SALT}|{identity}".encode()
            ).hexdigest(),
        )
        for rank, identity in enumerate(ranked):
            assignments[identity] = "development" if rank < 48 else "validation"
    split_payload = {
        "schema": 1,
        "salt": SPLIT_SALT,
        "rule": "Within each of 34 method-free cells, SHA256-rank 96 dataset realizations; first 48 development, last 48 validation. All five methods share the assignment.",
        "assignments": assignments,
    }
    args.split_json.parent.mkdir(parents=True, exist_ok=True)
    args.split_json.write_text(json.dumps(split_payload, indent=2, sort_keys=True) + "\n")
    split_hash = sha256(args.split_json)
    rows = []
    for row, original, replication, dataset_identity in expanded:
        entry_identity = f"{dataset_identity}|method={row['method']}"
        owner_bucket = int(hashlib.sha256(entry_identity.encode()).hexdigest()[:8], 16) % 7
        owner = "dml" if owner_bucket < 4 else "dml2"
        resample_seed = 1_000_000_000 + (
            int(hashlib.sha256(f"resample|{entry_identity}".encode()).hexdigest()[:8], 16)
            % 2_000_000_000
        )
        safe_identity = hashlib.sha256(dataset_identity.encode()).hexdigest()[:20]
        output = (
            args.output_root
            / owner
            / assignments[dataset_identity]
            / row["group"]
            / safe_identity
            / row["method"]
        )
        build_command = [
            str(args.python.absolute()), str(builder),
            "--frozen-source", str(source),
            "--output", str(output),
            "--method", row["method"],
            "--design", row["design"],
            "--mar-design", option(original, "--mar-design", default="box"),
            "--n", row["n"],
            "--epsilon", option(original, "--epsilon", default="0.05"),
            "--strength", row["strength"],
            "--base-seed", row["seed"],
            "--replication", str(replication),
            "--resample-seed", str(resample_seed),
            "--repeated-crossfits", "20",
            "--delete-blocks", "0",
            "--bootstraps", "0",
            "--inner-folds", option(original, "--folds", default="3"),
            "--learner", option(original, "--learner", default="xgboost"),
            "--propensity-learner", option(original, "--propensity-learner", default="xgboost"),
            "--propensity-mode", option(original, "--propensity-mode", multiple=True, default=["estimated"])[0],
            "--tau-grid", *option(original, "--tau-grid", multiple=True, default=["0.05"]),
            "--repair-mode", option(original, "--repair-mode", default="if_residual"),
            "--candidate-grid", *option(original, "--region-damp-grid", multiple=True, default=["0", ".25", ".5", "1"]),
            "--validation-risk", option(original, "--validation-risk", default="balanced_mse"),
            "--validation-loss-se", option(original, "--validation-loss-se", default="1"),
            "--validation-region-weight", option(original, "--validation-region-weight", default="-1"),
            "--shrink-c", option(original, "--c", default="2"),
            "--selector", option(original, "--selector", default="obsval"),
            "--lepski-c", option(original, "--lepski-c", default="4"),
            "--analysis-region", option(original, "--analysis-region", default="estimated_residual_lowp_supported"),
            "--region-quantile", option(original, "--region-quantile", default="0.10"),
            "--region-min-observed", option(original, "--region-min-observed", default="30"),
            "--region-kappa-floor", option(original, "--region-kappa-floor", default="0"),
            "--region-selector-ablation", option(original, "--region-selector-ablation", default="legacy"),
            "--region-detector-c", option(original, "--region-detector-c", default="4"),
        ]
        rows.append(
            {
                "entry_identity": entry_identity,
                "dataset_identity": dataset_identity,
                "dataset_identity_sha256": hashlib.sha256(dataset_identity.encode()).hexdigest(),
                "split": assignments[dataset_identity],
                "owner": owner,
                "group": row["group"],
                "design": row["design"],
                "method": row["method"],
                "n": row["n"],
                "strength": row["strength"],
                "chunk": row["chunk"],
                "replication": replication,
                "base_seed": row["seed"],
                "source_sha256": source_hash,
                "builder_sha256": builder_hash,
                "split_sha256": split_hash,
                "output": str(output),
                "command_json": json.dumps(build_command, separators=(",", ":")),
            }
        )
    if len(rows) != 16_320 or len({row["entry_identity"] for row in rows}) != 16_320:
        raise ValueError("full bank must contain exactly 16,320 unique entries")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["entry_identity"]))
    print(
        json.dumps(
            {
                "entries": len(rows),
                "datasets": len(assignments),
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
                "baseline_manifest_sha256": baseline_hashes,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
