#!/usr/bin/env python3
"""Generate the locked AIPW/KS versus TMLE/KS expert-bank pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


DESIGNS = ("kang_schafer_cc", "kang_schafer_ci", "kang_schafer_ic", "kang_schafer_ii")
SPLIT_SALT = "expert-bank-direction-pilot-v1-20260814"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
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
    for design in DESIGNS:
        for n in (200, 1000):
            stratum = []
            for replication in range(12):
                identity = f"{design}|n={n}|strength=0|rep={replication:03d}"
                rank_hash = hashlib.sha256(f"{SPLIT_SALT}|{identity}".encode()).hexdigest()
                stratum.append((rank_hash, identity, design, n, replication))
            stratum.sort()
            for rank, values in enumerate(stratum):
                rank_hash, identity, design, n, replication = values
                datasets.append(
                    {
                        "dataset_identity": identity,
                        "dataset_identity_sha256": hashlib.sha256(identity.encode()).hexdigest(),
                        "split_rank_sha256": rank_hash,
                        "split": "development" if rank < 6 else "validation",
                        "design": design,
                        "n": n,
                        "replication": replication,
                    }
                )
    split_payload = {
        "schema": 1,
        "salt": SPLIT_SALT,
        "rule": "Within each of 8 KS strata, rank 12 dataset identities by SHA256(salt|identity); first 6 development, last 6 validation. Methods share the dataset split.",
        "datasets": sorted(datasets, key=lambda row: row["dataset_identity"]),
    }
    args.split_json.parent.mkdir(parents=True, exist_ok=True)
    args.split_json.write_text(json.dumps(split_payload, indent=2, sort_keys=True) + "\n")
    split_hash = sha256(args.split_json)
    rows = []
    for dataset in datasets:
        cell_number = DESIGNS.index(dataset["design"]) * 2 + (dataset["n"] == 1000)
        base_seed = 3_000_000_000 + cell_number * 20_000_000
        resample_seed = 3_300_000_000 + cell_number * 100_000 + dataset["replication"]
        for method in ("aipw", "tmle"):
            output = (
                args.output_root
                / dataset["split"]
                / dataset["dataset_identity"].replace("|", "__").replace("=", "")
                / method
            )
            command = [
                # Preserve the venv launcher path. Resolving this symlink would
                # invoke the base interpreter without the frozen xgboost env.
                str(args.python.absolute()),
                str(builder),
                "--frozen-source", str(source),
                "--output", str(output),
                "--method", method,
                "--design", dataset["design"],
                "--n", str(dataset["n"]),
                "--strength", "0",
                "--base-seed", str(base_seed),
                "--replication", str(dataset["replication"]),
                "--resample-seed", str(resample_seed),
                "--repeated-crossfits", "20",
                "--delete-blocks", "10",
                "--bootstraps", "0",
                "--inner-folds", "3",
                "--learner", "xgboost",
                "--propensity-learner", "xgboost",
                "--propensity-mode", "estimated",
                "--tau-grid", "0.05",
                "--repair-mode", "if_residual",
                "--candidate-grid", "0", "0.25", "0.5", "1",
                "--validation-risk", "balanced_mse",
                "--validation-loss-se", "1",
                "--shrink-c", "2",
            ]
            rows.append(
                {
                    **dataset,
                    "method": method,
                    "source_sha256": source_hash,
                    "builder_sha256": builder_hash,
                    "split_sha256": split_hash,
                    "output": str(output),
                    "command_json": json.dumps(command, separators=(",", ":")),
                }
            )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["dataset_identity"], row["method"])))
    print(
        json.dumps(
            {
                "jobs": len(rows),
                "datasets": len(datasets),
                "development_datasets": sum(row["split"] == "development" for row in datasets),
                "validation_datasets": sum(row["split"] == "validation" for row in datasets),
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
