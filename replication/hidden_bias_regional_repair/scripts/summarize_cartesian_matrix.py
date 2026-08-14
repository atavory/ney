#!/usr/bin/env python3
"""Fail-closed aggregation of the complete unified expert-by-dataset matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METHODS = ("aipw", "tmle", "ctmle", "cui_selective_ml", "ma_dr_bc")
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


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--full-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


def scientific_manifest(path: Path) -> tuple[list[dict[str, str]], str]:
    fields = ("group", "design", "method", "n", "strength", "chunk", "seed")
    with path.open(newline="") as handle:
        rows = [{field: row[field] for field in fields} for row in csv.DictReader(handle, delimiter="\t")]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return rows, hashlib.sha256(payload).hexdigest()


def paired_draws(
    reference: np.ndarray,
    repaired: np.ndarray,
    rng: np.random.Generator,
    count: int,
) -> np.ndarray:
    indices = rng.integers(0, len(reference), size=(count, len(reference)))
    return 1.0 - np.mean(repaired[indices] ** 2, axis=1) / np.mean(
        reference[indices] ** 2, axis=1
    )


def gain(reference: np.ndarray, repaired: np.ndarray) -> float:
    return 1.0 - float(np.mean(repaired**2)) / float(np.mean(reference**2))


def summarize_cell(
    rows: list[dict[str, str]], rng: np.random.Generator, count: int
) -> tuple[dict[str, object], np.ndarray]:
    reference = np.asarray([float(row["ref_error"]) for row in rows])
    proposal = np.asarray([float(row["rt_error"]) for row in rows])
    repaired = np.asarray([float(row["shrink_error"]) for row in rows])
    weights = np.asarray([float(row["weight"]) for row in rows])
    selected = np.asarray([float(row["selected_region_damp"]) for row in rows])
    active = weights > 0.0
    draws = paired_draws(reference, repaired, rng, count)
    conditional_gain = gain(reference[active], repaired[active]) if np.any(active) else None
    conditional_harm = (
        float(np.mean(repaired[active] ** 2 > reference[active] ** 2))
        if np.any(active)
        else None
    )
    return (
        {
            "reps": len(rows),
            "reference_mse": float(np.mean(reference**2)),
            "proposal_mse": float(np.mean(proposal**2)),
            "repaired_mse": float(np.mean(repaired**2)),
            "proposal_gain": gain(reference, proposal),
            "repaired_gain": gain(reference, repaired),
            "repaired_gain_ci": np.percentile(draws, [2.5, 97.5]).tolist(),
            "path_activation": float(np.mean(selected > 0.0)),
            "final_activation": float(np.mean(active)),
            "conditional_gain_when_active": conditional_gain,
            "conditional_harm_when_active": conditional_harm,
            "unconditional_harm": float(np.mean(repaired**2 > reference**2)),
            "mean_shrink_weight": float(np.mean(weights)),
            "shrink_weight_quantiles": np.percentile(
                weights, [0, 25, 50, 75, 100]
            ).tolist(),
        },
        draws,
    )


def main() -> None:
    args = arguments()
    expected_jobs, full_hash = scientific_manifest(args.full_manifest)
    if len(expected_jobs) != 4080:
        raise SystemExit(f"full manifest has {len(expected_jobs)} jobs, expected 4080")
    expected_cells = {
        (row["group"], row["design"], row["method"], int(row["n"]), float(row["strength"]))
        for row in expected_jobs
    }
    if len(expected_cells) != 170:
        raise SystemExit(f"full manifest has {len(expected_cells)} cells, expected 170")

    provenance = [json.loads((directory / "provenance.json").read_text()) for directory in args.run_dir]
    for key in CONFIG_KEYS:
        values = [record.get(key) for record in provenance]
        if any(value != values[0] for value in values[1:]):
            raise SystemExit(f"provenance mismatch for {key}: {values}")
    frozen = provenance[0]
    if frozen["repair_mode"] != "if_residual":
        raise SystemExit("matrix is not global residual")
    if float(frozen["validation_loss_se"]) != 1.0 or float(frozen["shrink_c"]) != 2.0:
        raise SystemExit("matrix does not use SE=1,c=2")
    if [float(value) for value in frozen["region_damp_grid"]] != [0.0, 0.25, 0.5, 1.0]:
        raise SystemExit("matrix uses the wrong candidate grid")

    design_group = {(group, design, n, strength): group for group, design, _, n, strength in expected_cells}
    grouped: dict[tuple[str, str, str, int, float], list[dict[str, str]]] = defaultdict(list)
    file_count = 0
    for directory in args.run_dir:
        for path in sorted(directory.glob("*.reps.csv")):
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 4:
                raise SystemExit(f"{path}: expected 4 rows, observed {len(rows)}")
            file_count += 1
            for row in rows:
                method = row["reference_method"]
                design = row["design"]
                n = int(row["n"])
                strength = float(row["strength"])
                matches = [group for group, expected_design, expected_method, expected_n, expected_strength in expected_cells if (expected_design, expected_method, expected_n, expected_strength) == (design, method, n, strength)]
                if len(matches) != 1:
                    raise SystemExit(f"row does not map uniquely to expected matrix: {method}/{design}/{n}/{strength}")
                if row["repair_mode"] != "if_residual" or float(row["validation_loss_se"]) != 1.0:
                    raise SystemExit("replication row has wrong repair configuration")
                grouped[(matches[0], design, method, n, strength)].append(row)
    if file_count != 4080:
        raise SystemExit(f"observed {file_count} shard files, expected 4080")
    if set(grouped) != expected_cells:
        missing = sorted(expected_cells - set(grouped))
        extra = sorted(set(grouped) - expected_cells)
        raise SystemExit(f"cell mismatch: missing={missing[:5]} extra={extra[:5]}")
    if any(len(rows) != 96 for rows in grouped.values()):
        bad = {key: len(rows) for key, rows in grouped.items() if len(rows) != 96}
        raise SystemExit(f"incomplete cells: {bad}")

    rng = np.random.default_rng(args.seed)
    cells: dict[str, object] = {}
    method_draws: dict[str, list[np.ndarray]] = defaultdict(list)
    family_draws: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for key, rows in sorted(grouped.items()):
        group, design, method, n, strength = key
        label = f"{group}/{design}/{method}/n={n}/strength={strength:g}"
        cells[label], cell_draws = summarize_cell(rows, rng, args.draws)
        method_draws[method].append(cell_draws)
        family_draws[(method, group)].append(cell_draws)

    methods: dict[str, object] = {}
    for method in METHODS:
        selected_cells = [value for label, value in cells.items() if f"/{method}/" in label]
        stacked = np.mean(np.stack(method_draws[method]), axis=0)
        families = {}
        for group in ("kang_schafer", "alignment", "real", "anchor"):
            family = np.mean(np.stack(family_draws[(method, group)]), axis=0)
            family_cells = [
                value
                for label, value in cells.items()
                if label.startswith(f"{group}/") and f"/{method}/" in label
            ]
            families[group] = {
                "cells": len(family_cells),
                "equal_cell_gain": float(
                    np.mean([value["repaired_gain"] for value in family_cells])
                ),
                "equal_cell_gain_ci": np.percentile(family, [2.5, 97.5]).tolist(),
            }
        methods[method] = {
            "cells": len(selected_cells),
            "equal_cell_gain": float(
                np.mean([value["repaired_gain"] for value in selected_cells])
            ),
            "equal_cell_gain_ci": np.percentile(stacked, [2.5, 97.5]).tolist(),
            "families": families,
        }

    result = {
        "scientific_manifest_sha256": full_hash,
        "jobs": file_count,
        "cells": cells,
        "methods": methods,
        "provenance": {key: frozen[key] for key in CONFIG_KEYS},
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
