#!/usr/bin/env python3
"""Fail-closed summary for the shared global-residual SE/c experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def gain(reference: np.ndarray, repaired: np.ndarray) -> float:
    return 1.0 - float(np.mean(repaired**2)) / float(np.mean(reference**2))


def draws(
    reference: np.ndarray,
    repaired: np.ndarray,
    rng: np.random.Generator,
    count: int,
) -> np.ndarray:
    index = rng.integers(0, len(reference), size=(count, len(reference)))
    return 1.0 - np.mean(repaired[index] ** 2, axis=1) / np.mean(
        reference[index] ** 2, axis=1
    )


def endpoint(
    rows: list[dict[str, str]], rng: np.random.Generator, count: int
) -> tuple[dict[str, object], np.ndarray]:
    reference = np.asarray([float(row["ref_error"]) for row in rows])
    proposal_key = "proposal_error" if "proposal_error" in rows[0] else "rt_error"
    final_key = "repair_error" if "repair_error" in rows[0] else "shrink_error"
    proposal = np.asarray([float(row[proposal_key]) for row in rows])
    final = np.asarray([float(row[final_key]) for row in rows])
    final_draws = draws(reference, final, rng, count)
    weight_key = "shrink_weight" if "shrink_weight" in rows[0] else "weight"
    selected_key = (
        "selected_gamma" if "selected_gamma" in rows[0] else "selected_region_damp"
    )
    return (
        {
            "reps": len(rows),
            "reference_mse": float(np.mean(reference**2)),
            "proposal_mse": float(np.mean(proposal**2)),
            "repaired_mse": float(np.mean(final**2)),
            "proposal_gain": gain(reference, proposal),
            "final_gain": gain(reference, final),
            "final_gain_ci": np.percentile(final_draws, [2.5, 97.5]).tolist(),
            "path_activation": float(
                np.mean([float(row[selected_key]) > 0.0 for row in rows])
            ),
            "final_activation": float(
                np.mean([float(row[weight_key]) > 0.0 for row in rows])
            ),
            "mean_shrink_weight": float(
                np.mean([float(row[weight_key]) for row in rows])
            ),
            "proposal_harm": float(np.mean(proposal**2 > reference**2)),
            "final_harm": float(np.mean(final**2 > reference**2)),
        },
        final_draws,
    )


def summarize_ma(
    directory: Path, rng: np.random.Generator, count: int
) -> dict[str, object]:
    rows = read_rows(sorted(directory.glob("ma_global_residual_dgp*_chunk*.csv")))
    if len(rows) != 768:
        raise SystemExit(f"Ma coverage: expected 768 rows, observed {len(rows)}")
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    for row in rows:
        if row["adapter"] != "global_residual":
            raise SystemExit("Ma used a non-global-residual construction")
        if float(row["se_threshold"]) != 1.0 or float(row["shrink_c"]) != 2.0:
            raise SystemExit("Ma used the wrong shared SE/c parameters")
        key = (int(row["dgp"]), int(row["seed"]))
        if key in seen:
            raise SystemExit(f"duplicate Ma DGP/seed: {key}")
        seen.add(key)
        grouped[key[0]].append(row)
    if set(grouped) != {2, 3} or any(len(rows) != 384 for rows in grouped.values()):
        raise SystemExit("Ma DGP coverage is incomplete")
    cells: dict[str, object] = {}
    family_draws = []
    for dgp, cell_rows in sorted(grouped.items()):
        cells[str(dgp)], cell_draws = endpoint(cell_rows, rng, count)
        family_draws.append(cell_draws)
    family = np.mean(np.stack(family_draws), axis=0)
    return {
        "cells": cells,
        "equal_cell_final_gain": float(
            np.mean([cell["final_gain"] for cell in cells.values()])
        ),
        "equal_cell_final_gain_ci": np.percentile(family, [2.5, 97.5]).tolist(),
    }


def summarize_ks(
    directory: Path, rng: np.random.Generator, count: int
) -> dict[str, object]:
    rows = read_rows(sorted(directory.glob("*.reps.csv")))
    expected = 2 * 8 * 96
    if len(rows) != expected:
        raise SystemExit(f"KS coverage: expected {expected} rows, observed {len(rows)}")
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["reference_method"] not in {"aipw", "tmle"}:
            raise SystemExit("KS includes an unexpected expert")
        if row["repair_mode"] != "if_residual":
            raise SystemExit("KS used a non-global-residual construction")
        if float(row["validation_loss_se"]) != 1.0:
            raise SystemExit("KS used the wrong SE threshold")
        grouped[(row["reference_method"], row["design"], int(row["n"]))].append(row)
    if len(grouped) != 16 or any(len(rows) != 96 for rows in grouped.values()):
        raise SystemExit("KS cell coverage is incomplete")
    result: dict[str, object] = {}
    for method in ("aipw", "tmle"):
        cells: dict[str, object] = {}
        family_draws = []
        for (_, design, n), cell_rows in sorted(grouped.items()):
            if cell_rows[0]["reference_method"] != method:
                continue
            cells[f"{design}:n={n}"], cell_draws = endpoint(cell_rows, rng, count)
            family_draws.append(cell_draws)
        if len(cells) != 8:
            raise SystemExit(f"{method} has {len(cells)} cells instead of 8")
        family = np.mean(np.stack(family_draws), axis=0)
        result[method] = {
            "cells": cells,
            "equal_cell_final_gain": float(
                np.mean([cell["final_gain"] for cell in cells.values()])
            ),
            "equal_cell_final_gain_ci": np.percentile(
                family, [2.5, 97.5]
            ).tolist(),
            "pooled_reference_mse": float(
                np.mean([cell["reference_mse"] for cell in cells.values()])
            ),
            "pooled_repaired_mse": float(
                np.mean([cell["repaired_mse"] for cell in cells.values()])
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ma-dir", required=True, type=Path)
    parser.add_argument("--ks-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    result = {
        "rule": {
            "scope": "global",
            "construction": "residual",
            "se_threshold": 1.0,
            "shrink_c": 2.0,
        },
        "ma": summarize_ma(
            args.ma_dir, np.random.default_rng(args.seed), args.draws
        ),
        "kang_schafer": summarize_ks(
            args.ks_dir, np.random.default_rng(args.seed + 1), args.draws
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
