#!/usr/bin/env python3
"""Reference nuisance error by response bin for each expert family."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


_VRT = None
_NUISANCE = None


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


METHODS = ("aipw", "ctmle", "cui_selective_ml", "ma_dr_bc", "tmle")
PRIMARY_METHODS = ("aipw", "ctmle", "cui_selective_ml", "ma_dr_bc")


def _init_worker(
    source_script: str, breadth_script: str, nuisance_script: str, support_data: str
) -> None:
    global _VRT, _NUISANCE
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ["USHMOO_SUPPORT_DATA"] = support_data
    os.environ["DML_SUPPORT_DATA"] = support_data
    script_dir = str(Path(source_script).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    vrt = _load_module("dml_reference_vrt_worker", Path(source_script))
    breadth = _load_module("dml_reference_breadth_worker", Path(breadth_script))
    breadth._install_adapter(vrt)
    _VRT = vrt
    _NUISANCE = _load_module("dml_reference_nuisance_worker", Path(nuisance_script))


def _safe_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def _safe_sd(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.std()) if len(arr) else float("nan")


def _rank_bins(score: np.ndarray, bins: int) -> np.ndarray:
    order = np.argsort(score, kind="mergesort")
    labels = np.empty(len(score), dtype=int)
    labels[order] = np.minimum((np.arange(len(score)) * bins) // len(score), bins - 1)
    return labels


def _task(args_tuple: tuple[Any, int, str, int, str, str, int, int]) -> list[dict[str, Any]]:
    setting, rep, method, seed0, learner, propensity_learner, folds, bins = args_tuple
    vrt = _VRT
    nuisance = _NUISANCE
    if vrt is None or nuisance is None:
        raise RuntimeError("worker was not initialized")
    seed = nuisance._design_seed(vrt, setting, rep, seed0)
    raw = vrt.make_data(
        setting.n,
        setting.epsilon,
        setting.strength,
        setting.design,
        seed,
        setting.mar_design,
    )
    x, y, response, region, true_pi, theta, mu = raw
    tau_grid = (
        (0.05,)
        if method in {"tmle", "ma_dr_bc"}
        else (0.05, 0.10, 0.25, 0.50)
    )
    fit = vrt._crossfit_selected(
        (x, y, response, region, true_pi, theta, mu),
        method,
        "estimated",
        learner,
        propensity_learner,
        "if_residual",
        tau_grid,
        folds,
        seed + 17,
        (0.0,),
        -1.0,
        1.0,
        "obsval",
        4.0,
    )
    ref_outcome = np.asarray(fit["ref_outcome"], dtype=float)
    selected_p = np.asarray(fit["selected_p"], dtype=float)
    labels = _rank_bins(true_pi, bins)
    mu_sd = max(float(np.std(mu)), 1e-12)
    rows = []
    for bin_index in range(bins):
        mask = labels == bin_index
        outcome_rmse = float(np.sqrt(np.mean((ref_outcome[mask] - mu[mask]) ** 2)))
        rows.append(
            {
                "dataset": setting.dataset,
                "design": setting.design,
                "setting": setting.setting,
                "n": setting.n,
                "strength": setting.strength,
                "method": method,
                "role": "primary" if method in PRIMARY_METHODS else "diagnostic",
                "rep": rep + 1,
                "bin_by": "true_pi",
                "bin": bin_index + 1,
                "bins": bins,
                "mean_true_pi": float(np.mean(true_pi[mask])),
                "response_rate": float(np.mean(response[mask])),
                "outcome_rmse": outcome_rmse,
                "outcome_nrmse": outcome_rmse / mu_sd,
                "mean_selected_p": float(np.mean(selected_p[mask])),
                "floor_fraction": float(np.mean(selected_p[mask] <= 0.050000001)),
                "selected_tau": float(fit["selected_tau"])
                if math.isfinite(float(fit["selected_tau"]))
                else float("nan"),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    metrics = [
        "mean_true_pi",
        "response_rate",
        "outcome_rmse",
        "outcome_nrmse",
        "mean_selected_p",
        "floor_fraction",
        "selected_tau",
    ]
    out = []
    for key_values, group in sorted(groups.items()):
        row = dict(zip(keys, key_values))
        row["rep_bins"] = len(group)
        for metric in metrics:
            values = [float(item[metric]) for item in group]
            row[f"mean_{metric}"] = _safe_mean(values)
            row[f"sd_{metric}"] = _safe_sd(values)
        out.append(row)
    return out


def _aggregate_low_high(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["method"], row["role"])].append(row)
    metrics = [
        "low_mean_true_pi",
        "high_mean_true_pi",
        "low_response_rate",
        "high_response_rate",
        "low_outcome_nrmse",
        "high_outcome_nrmse",
        "low_high_outcome_nrmse_ratio",
        "low_mean_selected_p",
        "high_mean_selected_p",
        "low_floor_fraction",
        "high_floor_fraction",
        "mean_selected_tau",
    ]
    out = []
    for key_values, group in sorted(groups.items()):
        row = dict(zip(("dataset", "method", "role"), key_values))
        row["settings"] = len(group)
        for metric in metrics:
            row[f"mean_{metric}"] = _safe_mean(
                [float(item[metric]) for item in group]
            )
        out.append(row)
    return out


def _low_high(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[
        tuple[str, str, str, str], dict[int, list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[
            (row["dataset"], row["setting"], row["method"], row["role"])
        ][int(row["bin"])].append(row)
    out = []
    for (dataset, setting, method, role), bins in sorted(groups.items()):
        low = bins[min(bins)]
        high = bins[max(bins)]

        def avg(group: list[dict[str, Any]], key: str) -> float:
            return _safe_mean([float(row[key]) for row in group])

        low_nrmse = avg(low, "outcome_nrmse")
        high_nrmse = avg(high, "outcome_nrmse")
        out.append(
            {
                "dataset": dataset,
                "setting": setting,
                "method": method,
                "role": role,
                "low_mean_true_pi": avg(low, "mean_true_pi"),
                "high_mean_true_pi": avg(high, "mean_true_pi"),
                "low_response_rate": avg(low, "response_rate"),
                "high_response_rate": avg(high, "response_rate"),
                "low_outcome_nrmse": low_nrmse,
                "high_outcome_nrmse": high_nrmse,
                "low_high_outcome_nrmse_ratio": (
                    low_nrmse / high_nrmse if high_nrmse > 0.0 else float("nan")
                ),
                "low_mean_selected_p": avg(low, "mean_selected_p"),
                "high_mean_selected_p": avg(high, "mean_selected_p"),
                "low_floor_fraction": avg(low, "floor_fraction"),
                "high_floor_fraction": avg(high, "floor_fraction"),
                "mean_selected_tau": avg(low + high, "selected_tau"),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = Path(__file__).resolve().parent
    parser.add_argument("--source-script", type=Path, default=default_dir / "validated_reference_transfer.py")
    parser.add_argument("--breadth-script", type=Path, default=default_dir / "section4_breadth_experiments.py")
    parser.add_argument(
        "--nuisance-script",
        type=Path,
        default=default_dir / "dml_nuisance_cv_accuracy_by_response.py",
    )
    parser.add_argument("--support-data", type=Path, default=Path("/tmp/dml_real_benchmark_support_data"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=16)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=14700000)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--learner", default="xgboost")
    parser.add_argument("--propensity-learner", default="xgboost")
    args = parser.parse_args()
    nuisance = _load_module("dml_reference_nuisance_main", args.nuisance_script)
    settings = nuisance._settings()
    tasks = [
        (
            setting,
            rep,
            method,
            args.seed,
            args.learner,
            args.propensity_learner,
            args.folds,
            args.bins,
        )
        for setting in settings
        for rep in range(args.reps)
        for method in METHODS
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=args.jobs,
        initializer=_init_worker,
        initargs=(
            str(args.source_script.resolve()),
            str(args.breadth_script.resolve()),
            str(args.nuisance_script.resolve()),
            str(args.support_data.resolve()),
        ),
    ) as pool:
        futures = [pool.submit(_task, task) for task in tasks]
        completed = 0
        for future in as_completed(futures):
            rows.extend(future.result())
            completed += 1
            if completed % max(1, args.jobs) == 0 or completed == len(tasks):
                print(f"completed {completed}/{len(tasks)} setting-reps", flush=True)
    rows.sort(
        key=lambda row: (
            row["dataset"],
            row["setting"],
            row["method"],
            row["rep"],
            row["bin"],
        )
    )
    _write_csv(args.out_dir / "reference_error_per_bin_reps.csv", rows)
    _write_csv(
        args.out_dir / "reference_error_bin_summary.csv",
        _aggregate(
            rows,
            (
                "dataset",
                "design",
                "setting",
                "n",
                "strength",
                "method",
                "role",
                "bin",
            ),
        ),
    )
    low_high = _low_high(rows)
    _write_csv(args.out_dir / "reference_error_low_high_summary.csv", low_high)
    _write_csv(
        args.out_dir / "reference_error_dataset_summary.csv",
        _aggregate_low_high(low_high),
    )
    print(f"wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
