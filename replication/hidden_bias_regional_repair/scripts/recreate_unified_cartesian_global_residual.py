#!/usr/bin/env python3
"""Recreate the unified Cartesian global-residual summaries.

This is intentionally stdlib-only.  The original public aggregation script used
NumPy, but the historical execution virtualenv is not reliably available on the
submission host.  The point estimates are exact functions of the archived
replication rows; bootstrap intervals are deterministic under the seed recorded
in the output verification file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable


METHODS = ("aipw", "tmle", "ctmle", "cui_selective_ml", "ma_dr_bc")
GROUPS = ("kang_schafer", "alignment", "real", "anchor")
CONFIG_KEYS = (
    "frozen_source_sha256",
    "wrapper_sha256",
    "repair_mode",
    "region_damp_grid",
    "validation_loss_se",
    "shrink_c",
    "seed_base",
    "chunks",
    "reps_per_chunk",
    "bootstraps",
    "xgboost_version",
)
EXPECTED_FULL_MANIFEST_SHA256 = (
    "65720b1fce2a24d55872ab9e008cf7ef62945b30b791272f1fdfe65280e2287f"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        type=Path,
        help="Extracted run directory containing *.reps.csv and provenance.json.",
    )
    parser.add_argument("--full-manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--expected-full-manifest-sha256",
        default=EXPECTED_FULL_MANIFEST_SHA256,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scientific_manifest(path: Path) -> tuple[list[dict[str, str]], str]:
    fields = ("group", "design", "method", "n", "strength", "chunk", "seed")
    with path.open(newline="") as handle:
        rows = [
            {field: row[field] for field in fields}
            for row in csv.DictReader(handle, delimiter="\t")
        ]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return rows, hashlib.sha256(payload).hexdigest()


def read_provenance(run_dirs: Iterable[Path]) -> list[dict[str, object]]:
    records = []
    for run_dir in run_dirs:
        path = run_dir / "provenance.json"
        if not path.exists():
            raise SystemExit(f"missing provenance file: {path}")
        records.append(json.loads(path.read_text()))
    if not records:
        raise SystemExit("at least one run directory is required")
    for key in CONFIG_KEYS:
        values = [record.get(key) for record in records]
        if any(value != values[0] for value in values[1:]):
            raise SystemExit(f"provenance mismatch for {key}: {values}")
    frozen = records[0]
    if frozen["repair_mode"] != "if_residual":
        raise SystemExit("run is not the global residual repair mode")
    if float(frozen["validation_loss_se"]) != 1.0:
        raise SystemExit("run does not use the frozen 1-SE gate")
    if float(frozen["shrink_c"]) != 2.0:
        raise SystemExit("run does not use c=2 shrinkage")
    if [float(value) for value in frozen["region_damp_grid"]] != [0.0, 0.25, 0.5, 1.0]:
        raise SystemExit("run does not use the frozen candidate grid")
    if int(frozen["bootstraps"]) != 0:
        raise SystemExit("run rows are not from the frozen no-bootstrap baseline")
    return records


def as_float(value: str) -> float:
    try:
        return float(value)
    except ValueError as error:
        raise SystemExit(f"cannot parse float value {value!r}") from error


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * pct / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def gain_from_squares(reference_sq: list[float], repaired_sq: list[float]) -> float:
    ref_mse = mean(reference_sq)
    if ref_mse == 0.0:
        raise SystemExit("zero reference MSE encountered")
    return 1.0 - mean(repaired_sq) / ref_mse


def bootstrap_draws(
    reference_sq: list[float],
    repaired_sq: list[float],
    rng: random.Random,
    count: int,
) -> list[float]:
    draws: list[float] = []
    n = len(reference_sq)
    for _ in range(count):
        ref_total = 0.0
        repaired_total = 0.0
        for _ in range(n):
            index = rng.randrange(n)
            ref_total += reference_sq[index]
            repaired_total += repaired_sq[index]
        draws.append(1.0 - repaired_total / ref_total)
    return draws


def summarize_cell(
    rows: list[dict[str, str]],
    rng: random.Random,
    draws: int,
) -> tuple[dict[str, object], list[float]]:
    reference_sq = [as_float(row["ref_error"]) ** 2 for row in rows]
    proposal_sq = [as_float(row["rt_error"]) ** 2 for row in rows]
    repaired_sq = [as_float(row["shrink_error"]) ** 2 for row in rows]
    weights = [as_float(row["weight"]) for row in rows]
    selected = [as_float(row["selected_region_damp"]) for row in rows]
    active_indices = [index for index, weight in enumerate(weights) if weight > 0.0]
    cell_draws = bootstrap_draws(reference_sq, repaired_sq, rng, draws)
    sorted_draws = sorted(cell_draws)
    if active_indices:
        active_ref = [reference_sq[index] for index in active_indices]
        active_repaired = [repaired_sq[index] for index in active_indices]
        conditional_gain = gain_from_squares(active_ref, active_repaired)
        conditional_harm = mean(
            [
                1.0 if repaired_sq[index] > reference_sq[index] else 0.0
                for index in active_indices
            ]
        )
    else:
        conditional_gain = None
        conditional_harm = None
    return (
        {
            "reps": len(rows),
            "reference_mse": mean(reference_sq),
            "proposal_mse": mean(proposal_sq),
            "repaired_mse": mean(repaired_sq),
            "proposal_gain": gain_from_squares(reference_sq, proposal_sq),
            "repaired_gain": gain_from_squares(reference_sq, repaired_sq),
            "repaired_gain_ci_low": percentile(sorted_draws, 2.5),
            "repaired_gain_ci_high": percentile(sorted_draws, 97.5),
            "path_activation": mean([1.0 if value > 0.0 else 0.0 for value in selected]),
            "final_activation": mean([1.0 if value > 0.0 else 0.0 for value in weights]),
            "conditional_gain_when_active": conditional_gain,
            "conditional_harm_when_active": conditional_harm,
            "unconditional_harm": mean(
                [
                    1.0 if repaired > reference else 0.0
                    for reference, repaired in zip(reference_sq, repaired_sq)
                ]
            ),
            "mean_shrink_weight": mean(weights),
        },
        cell_draws,
    )


def format_optional(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_optional(row.get(key, "")) for key in fieldnames})


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows, manifest_sha = scientific_manifest(args.full_manifest)
    if manifest_sha != args.expected_full_manifest_sha256:
        raise SystemExit(
            "full manifest SHA mismatch: "
            f"observed {manifest_sha}, expected {args.expected_full_manifest_sha256}"
        )
    if len(manifest_rows) != 4080:
        raise SystemExit(f"full manifest has {len(manifest_rows)} rows, expected 4080")
    expected_cells = {
        (
            row["group"],
            row["design"],
            row["method"],
            int(row["n"]),
            float(row["strength"]),
        )
        for row in manifest_rows
    }
    if len(expected_cells) != 170:
        raise SystemExit(f"full manifest has {len(expected_cells)} cells, expected 170")
    cell_lookup = {
        (design, method, n, strength): group
        for group, design, method, n, strength in expected_cells
    }

    provenance = read_provenance(args.run_dir)
    grouped: dict[tuple[str, str, str, int, float], list[dict[str, str]]] = defaultdict(list)
    reps_files: list[Path] = []
    rows_seen = 0
    for run_dir in args.run_dir:
        for path in sorted(run_dir.glob("*.reps.csv")):
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 4:
                raise SystemExit(f"{path}: expected 4 replication rows, observed {len(rows)}")
            reps_files.append(path)
            for row in rows:
                method = row["reference_method"]
                design = row["design"]
                n = int(row["n"])
                strength = float(row["strength"])
                key_without_group = (design, method, n, strength)
                if key_without_group not in cell_lookup:
                    raise SystemExit(f"row does not belong to the full manifest: {key_without_group}")
                if row["repair_mode"] != "if_residual":
                    raise SystemExit(f"{path}: row repair_mode is not if_residual")
                if float(row["validation_loss_se"]) != 1.0:
                    raise SystemExit(f"{path}: row validation_loss_se is not 1.0")
                if row["region_damp_grid"] != "0.0|0.25|0.5|1.0":
                    raise SystemExit(f"{path}: row has wrong candidate grid")
                grouped[(cell_lookup[key_without_group], design, method, n, strength)].append(row)
                rows_seen += 1
    if len(reps_files) != 4080:
        raise SystemExit(f"observed {len(reps_files)} shard files, expected 4080")
    if set(grouped) != expected_cells:
        missing = sorted(expected_cells - set(grouped))
        extra = sorted(set(grouped) - expected_cells)
        raise SystemExit(f"cell mismatch: missing={missing[:5]} extra={extra[:5]}")
    bad_cells = {key: len(rows) for key, rows in grouped.items() if len(rows) != 96}
    if bad_cells:
        raise SystemExit(f"incomplete cells: {bad_cells}")
    if rows_seen != 16320:
        raise SystemExit(f"observed {rows_seen} replication rows, expected 16320")

    raw_hash_payload = []
    for path in reps_files:
        matching_run_dir = next(run_dir for run_dir in args.run_dir if path.is_relative_to(run_dir))
        raw_hash_payload.append(
            {
                "run_dir": matching_run_dir.name,
                "relative_path": str(path.relative_to(matching_run_dir)),
                "sha256": sha256_file(path),
            }
        )
    raw_hash = hashlib.sha256(
        json.dumps(raw_hash_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    rng = random.Random(args.seed)
    cell_rows: list[dict[str, object]] = []
    cell_draws: dict[tuple[str, str, str, int, float], list[float]] = {}
    for key in sorted(grouped):
        group, design, method, n, strength = key
        summary, draws = summarize_cell(grouped[key], rng, args.draws)
        cell_draws[key] = draws
        cell_rows.append(
            {
                "group": group,
                "design": design,
                "method": method,
                "n": n,
                "strength": strength,
                **summary,
            }
        )

    family_rows: list[dict[str, object]] = []
    summary_methods: dict[str, object] = {}
    for method in METHODS:
        method_cell_rows = [row for row in cell_rows if row["method"] == method]
        method_draw_lists = [
            cell_draws[(row["group"], row["design"], row["method"], int(row["n"]), float(row["strength"]))]
            for row in method_cell_rows
        ]
        method_family_draws = [
            mean([cell_draw[index] for cell_draw in method_draw_lists])
            for index in range(args.draws)
        ]
        method_family_sorted = sorted(method_family_draws)
        method_summary = {
            "method": method,
            "family": "all",
            "cells": len(method_cell_rows),
            "reps": sum(int(row["reps"]) for row in method_cell_rows),
            "equal_cell_gain": mean([float(row["repaired_gain"]) for row in method_cell_rows]),
            "equal_cell_gain_ci_low": percentile(method_family_sorted, 2.5),
            "equal_cell_gain_ci_high": percentile(method_family_sorted, 97.5),
            "path_activation": mean([float(row["path_activation"]) for row in method_cell_rows]),
            "final_activation": mean([float(row["final_activation"]) for row in method_cell_rows]),
            "unconditional_harm": mean([float(row["unconditional_harm"]) for row in method_cell_rows]),
        }
        family_rows.append(method_summary)
        families: dict[str, object] = {}
        for group in GROUPS:
            group_cell_rows = [
                row
                for row in method_cell_rows
                if row["group"] == group
            ]
            group_draw_lists = [
                cell_draws[(row["group"], row["design"], row["method"], int(row["n"]), float(row["strength"]))]
                for row in group_cell_rows
            ]
            group_family_draws = [
                mean([cell_draw[index] for cell_draw in group_draw_lists])
                for index in range(args.draws)
            ]
            group_family_sorted = sorted(group_family_draws)
            group_summary = {
                "method": method,
                "family": group,
                "cells": len(group_cell_rows),
                "reps": sum(int(row["reps"]) for row in group_cell_rows),
                "equal_cell_gain": mean(
                    [float(row["repaired_gain"]) for row in group_cell_rows]
                ),
                "equal_cell_gain_ci_low": percentile(group_family_sorted, 2.5),
                "equal_cell_gain_ci_high": percentile(group_family_sorted, 97.5),
                "path_activation": mean(
                    [float(row["path_activation"]) for row in group_cell_rows]
                ),
                "final_activation": mean(
                    [float(row["final_activation"]) for row in group_cell_rows]
                ),
                "unconditional_harm": mean(
                    [float(row["unconditional_harm"]) for row in group_cell_rows]
                ),
            }
            family_rows.append(group_summary)
            families[group] = group_summary
        summary_methods[method] = {**method_summary, "families": families}

    verification = {
        "status": "PASS",
        "draws": args.draws,
        "seed": args.seed,
        "full_manifest": str(args.full_manifest),
        "full_manifest_sha256": manifest_sha,
        "run_dirs": [str(path) for path in args.run_dir],
        "run_provenance": provenance,
        "reps_files": len(reps_files),
        "replication_rows": rows_seen,
        "cells": len(cell_rows),
        "raw_reps_manifest_sha256": raw_hash,
        "config": {key: provenance[0][key] for key in CONFIG_KEYS},
    }
    summary = {
        "verification": verification,
        "methods": summary_methods,
    }

    write_csv(
        args.out_dir / "cell_summary.csv",
        cell_rows,
        [
            "group",
            "design",
            "method",
            "n",
            "strength",
            "reps",
            "reference_mse",
            "proposal_mse",
            "repaired_mse",
            "proposal_gain",
            "repaired_gain",
            "repaired_gain_ci_low",
            "repaired_gain_ci_high",
            "path_activation",
            "final_activation",
            "conditional_gain_when_active",
            "conditional_harm_when_active",
            "unconditional_harm",
            "mean_shrink_weight",
        ],
    )
    write_csv(
        args.out_dir / "family_summary.csv",
        family_rows,
        [
            "method",
            "family",
            "cells",
            "reps",
            "equal_cell_gain",
            "equal_cell_gain_ci_low",
            "equal_cell_gain_ci_high",
            "path_activation",
            "final_activation",
            "unconditional_harm",
        ],
    )
    (args.out_dir / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    for path in ("cell_summary.csv", "family_summary.csv", "verification.json", "summary.json"):
        print(f"{path}: {sha256_file(args.out_dir / path)}")


if __name__ == "__main__":
    main()
