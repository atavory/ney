#!/usr/bin/env python3
"""Fail-closed aggregation for the Ma published-DiD projection experiment.

The reported estimand is the relative ATT MSE gain, computed from pooled paired
replication errors.  The family estimand gives DGP 2 and DGP 3 equal weight.
Confidence intervals use a stratified paired bootstrap: resample replication
pairs within each DGP, compute each pooled-MSE gain, then average the gains.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


REQUIRED_COLUMNS = {"dgp", "seed", "ref_error", "repair_error", "selected_gamma"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expected-reps-per-dgp", type=int, default=384)
    parser.add_argument("--nboot", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def load(run_dir: Path, expected: int) -> dict[int, dict[str, np.ndarray]]:
    raw: dict[int, dict[str, list[float]]] = {
        2: {"ref": [], "repair": [], "gamma": []},
        3: {"ref": [], "repair": [], "gamma": []},
    }
    seen: set[tuple[int, int]] = set()
    paths = sorted(run_dir.glob("shard_*.csv"))
    if not paths:
        raise SystemExit(f"no shard_*.csv files in {run_dir}")
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
            if missing:
                raise SystemExit(f"{path}: missing columns {sorted(missing)}")
            for line, row in enumerate(reader, start=2):
                try:
                    dgp = int(row["dgp"])
                    seed = int(row["seed"])
                    ref = float(row["ref_error"])
                    repair = float(row["repair_error"])
                    gamma = float(row["selected_gamma"])
                except ValueError as error:
                    raise SystemExit(f"{path}:{line}: invalid numeric value: {error}") from error
                if dgp not in raw:
                    raise SystemExit(f"{path}:{line}: unexpected DGP {dgp}")
                if not all(math.isfinite(value) for value in (ref, repair, gamma)):
                    raise SystemExit(f"{path}:{line}: non-finite value")
                key = (dgp, seed)
                if key in seen:
                    raise SystemExit(f"duplicate DGP/seed pair: {key}")
                seen.add(key)
                raw[dgp]["ref"].append(ref)
                raw[dgp]["repair"].append(repair)
                raw[dgp]["gamma"].append(gamma)
    counts = {dgp: len(values["ref"]) for dgp, values in raw.items()}
    if counts != {2: expected, 3: expected}:
        raise SystemExit(f"coverage mismatch: expected {expected}/DGP, observed {counts}")
    return {
        dgp: {key: np.asarray(value, dtype=float) for key, value in values.items()}
        for dgp, values in raw.items()
    }


def gain(ref2: np.ndarray, repair2: np.ndarray) -> float:
    denominator = float(ref2.mean())
    if denominator <= 0:
        raise SystemExit("non-positive reference MSE")
    return 1.0 - float(repair2.mean()) / denominator


def main() -> None:
    args = arguments()
    by_dgp = load(args.run_dir, args.expected_reps_per_dgp)
    rng = np.random.default_rng(args.seed)
    cells: dict[str, dict[str, float | int]] = {}
    cell_draws: dict[int, np.ndarray] = {}
    for dgp, values in sorted(by_dgp.items()):
        ref2 = values["ref"] ** 2
        repair2 = values["repair"] ** 2
        indices = rng.integers(0, len(ref2), size=(args.nboot, len(ref2)))
        ref_draw = ref2[indices].mean(axis=1)
        repair_draw = repair2[indices].mean(axis=1)
        draws = 1.0 - repair_draw / ref_draw
        cell_draws[dgp] = draws
        cells[str(dgp)] = {
            "reps": len(ref2),
            "ref_mse": float(ref2.mean()),
            "repair_mse": float(repair2.mean()),
            "gain": gain(ref2, repair2),
            "gain_lo": float(np.percentile(draws, 2.5)),
            "gain_hi": float(np.percentile(draws, 97.5)),
            "activation": float(np.mean(values["gamma"] > 0)),
            "harm": float(np.mean(repair2 > ref2)),
        }
    family_draws = np.mean(np.stack([cell_draws[2], cell_draws[3]]), axis=0)
    family_gain = float(np.mean([cells["2"]["gain"], cells["3"]["gain"]]))
    result = {
        "run_dir": str(args.run_dir),
        "estimand": "equal-DGP mean of pooled relative ATT-MSE gains",
        "bootstrap": "stratified paired percentile bootstrap",
        "nboot": args.nboot,
        "bootstrap_seed": args.seed,
        "cells": cells,
        "family": {
            "gain": family_gain,
            "gain_lo": float(np.percentile(family_draws, 2.5)),
            "gain_hi": float(np.percentile(family_draws, 97.5)),
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
