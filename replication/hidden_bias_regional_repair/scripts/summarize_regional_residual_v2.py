#!/usr/bin/env python3
"""Fail-closed summary of the frozen regional-residual-v2 evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def gain(reference: np.ndarray, repaired: np.ndarray) -> float:
    denominator = float(np.mean(reference * reference))
    return 1.0 - float(np.mean(repaired * repaired)) / denominator


def paired_draws(
    reference: np.ndarray,
    repaired: np.ndarray,
    rng: np.random.Generator,
    count: int,
) -> np.ndarray:
    indices = rng.integers(0, len(reference), size=(count, len(reference)))
    ref_mse = np.mean(reference[indices] ** 2, axis=1)
    repaired_mse = np.mean(repaired[indices] ** 2, axis=1)
    return 1.0 - repaired_mse / ref_mse


def summarize_tmle(
    run_root: Path, rng: np.random.Generator, draws: int
) -> dict[str, object]:
    result: dict[str, object] = {}
    for family, directory in (
        ("kang_schafer", run_root / "tmle_ks"),
        ("cui_published", run_root / "tmle_cui"),
    ):
        rows = read_rows(sorted(directory.glob("*.reps.csv")))
        if len(rows) != 768:
            raise SystemExit(f"{family}: expected 768 rows, observed {len(rows)}")
        grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            if row["repair_mode"] != "regional_if_residual":
                raise SystemExit(f"{family}: unexpected repair mode")
            if float(row["validation_loss_se"]) != 2.83:
                raise SystemExit(f"{family}: unexpected validation threshold")
            grouped[(row["design"], int(row["n"]))].append(row)
        expected_cells = 8
        if len(grouped) != expected_cells or any(len(v) != 96 for v in grouped.values()):
            raise SystemExit(f"{family}: incomplete 96-replication cells")
        cells: dict[str, object] = {}
        family_endpoint_draws = []
        family_final_draws = []
        for (design, n), cell_rows in sorted(grouped.items()):
            ref = np.asarray([float(row["ref_error"]) for row in cell_rows])
            endpoint = np.asarray([float(row["rt_error"]) for row in cell_rows])
            final = np.asarray([float(row["shrink_error"]) for row in cell_rows])
            endpoint_boot = paired_draws(ref, endpoint, rng, draws)
            final_boot = paired_draws(ref, final, rng, draws)
            family_endpoint_draws.append(endpoint_boot)
            family_final_draws.append(final_boot)
            cells[f"{design}:n={n}"] = {
                "reps": len(ref),
                "endpoint_gain": gain(ref, endpoint),
                "endpoint_gain_ci": np.percentile(endpoint_boot, [2.5, 97.5]).tolist(),
                "final_gain": gain(ref, final),
                "final_gain_ci": np.percentile(final_boot, [2.5, 97.5]).tolist(),
                "path_activation": float(
                    np.mean([float(row["selected_region_damp"]) > 0 for row in cell_rows])
                ),
                "final_activation": float(
                    np.mean([float(row["weight"]) > 0 for row in cell_rows])
                ),
                "endpoint_harm": float(np.mean(endpoint**2 > ref**2)),
                "final_harm": float(np.mean(final**2 > ref**2)),
                "mean_region_mass": float(
                    np.mean([float(row["analysis_region_mass"]) for row in cell_rows])
                ),
            }
        endpoint_family = np.mean(np.stack(family_endpoint_draws), axis=0)
        final_family = np.mean(np.stack(family_final_draws), axis=0)
        result[family] = {
            "cells": cells,
            "equal_cell_endpoint_gain": float(
                np.mean([cell["endpoint_gain"] for cell in cells.values()])
            ),
            "equal_cell_endpoint_gain_ci": np.percentile(
                endpoint_family, [2.5, 97.5]
            ).tolist(),
            "equal_cell_final_gain": float(
                np.mean([cell["final_gain"] for cell in cells.values()])
            ),
            "equal_cell_final_gain_ci": np.percentile(
                final_family, [2.5, 97.5]
            ).tolist(),
        }
    return result


def summarize_ma(
    run_root: Path, rng: np.random.Generator, draws: int
) -> dict[str, object]:
    rows = read_rows(sorted((run_root / "ma").glob("ma_dgp*_chunk*.csv")))
    if len(rows) != 768:
        raise SystemExit(f"Ma: expected 768 rows, observed {len(rows)}")
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    seeds: set[tuple[int, int]] = set()
    for row in rows:
        if row["adapter"] != "regional_residual":
            raise SystemExit("Ma: unexpected adapter")
        key = (int(row["dgp"]), int(row["seed"]))
        if key in seeds:
            raise SystemExit(f"Ma: duplicate seed {key}")
        seeds.add(key)
        grouped[key[0]].append(row)
    if set(grouped) != {2, 3} or any(len(v) != 384 for v in grouped.values()):
        raise SystemExit("Ma: incomplete 384-replication DGP cells")
    cells: dict[str, object] = {}
    family_draws = []
    for dgp, cell_rows in sorted(grouped.items()):
        ref = np.asarray([float(row["ref_error"]) for row in cell_rows])
        repaired = np.asarray([float(row["repair_error"]) for row in cell_rows])
        bootstrap = paired_draws(ref, repaired, rng, draws)
        family_draws.append(bootstrap)
        cells[str(dgp)] = {
            "reps": len(ref),
            "gain": gain(ref, repaired),
            "gain_ci": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
            "activation": float(
                np.mean([float(row["selected_gamma"]) > 0 for row in cell_rows])
            ),
            "harm": float(np.mean(repaired**2 > ref**2)),
            "mean_region_mass": float(
                np.mean([float(row["region_mass"]) for row in cell_rows])
            ),
        }
    family = np.mean(np.stack(family_draws), axis=0)
    return {
        "cells": cells,
        "equal_dgp_gain": float(np.mean([cell["gain"] for cell in cells.values()])),
        "equal_dgp_gain_ci": np.percentile(family, [2.5, 97.5]).tolist(),
    }


def main() -> None:
    args = arguments()
    result = {
        "protocol": "regional_residual_v2_protocol.md",
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seeds": {"ma": args.seed, "tmle": args.seed + 1},
        # Independent streams make each family's interval invariant to report
        # ordering and to the presence of other families in the bundle.
        "tmle": summarize_tmle(
            args.run_root,
            np.random.default_rng(args.seed + 1),
            args.bootstrap_draws,
        ),
        "ma": summarize_ma(
            args.run_root,
            np.random.default_rng(args.seed),
            args.bootstrap_draws,
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
