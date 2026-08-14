#!/usr/bin/env python3
"""Evaluate the single preregistered truth-free direction statistic."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
from statistics import mean

from frozen_expert_bank import direction_reproducibility, load_entry


def finite_or_none(value):
    return value if not isinstance(value, float) or math.isfinite(value) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "validation"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle, delimiter="\t")
            if row["split"] == args.split
        ]
    paired = {}
    details = []
    for row in rows:
        entry = load_entry(Path(row["output"]))
        diagnostic = direction_reproducibility(entry)
        record = {
            "dataset_identity": row["dataset_identity"],
            "design": row["design"],
            "n": int(row["n"]),
            "method": row["method"],
            **{key: finite_or_none(value) for key, value in asdict(diagnostic).items()},
        }
        details.append(record)
        paired.setdefault(row["dataset_identity"], {})[row["method"]] = record
    if any(set(value) != {"aipw", "tmle"} for value in paired.values()):
        raise ValueError("every dataset must have exactly paired AIPW and TMLE entries")
    comparisons = []
    for identity, methods in sorted(paired.items()):
        aipw = methods["aipw"]["sign_reproducibility"]
        tmle = methods["tmle"]["sign_reproducibility"]
        concordance = 1.0 if aipw > tmle else 0.5 if aipw == tmle else 0.0
        comparisons.append(
            {
                "dataset_identity": identity,
                "design": methods["aipw"]["design"],
                "n": methods["aipw"]["n"],
                "aipw_S": aipw,
                "tmle_S": tmle,
                "concordance": concordance,
            }
        )
    overall = mean(row["concordance"] for row in comparisons)
    strata = {}
    for row in comparisons:
        key = f"{row['design']}|n={row['n']}"
        strata.setdefault(key, []).append(row["concordance"])
    output = {
        "split": args.split,
        "dataset_count": len(comparisons),
        "primary_statistic": "paired concordance of sign_reproducibility",
        "acceptance_threshold": 0.70,
        "paired_concordance": overall,
        "passes": overall >= 0.70,
        "stratum_concordance": {
            key: mean(values) for key, values in sorted(strata.items())
        },
        "details": sorted(details, key=lambda row: (row["dataset_identity"], row["method"])),
        "paired_details": comparisons,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key not in {"details", "paired_details"}}, sort_keys=True))


if __name__ == "__main__":
    main()
