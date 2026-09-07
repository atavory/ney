#!/usr/bin/env python3
"""Replay gamma selection using held-out weighted residual loss.

This diagnostic keeps the fitted residual direction and upstream fits fixed,
then replaces centered score-loss gamma selection by the same response-weighted
residual loss used to construct the direction.  It writes a separate support
bundle and does not modify the paper-facing release.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


METHODS = ("aipw", "tmle", "ctmle", "cui_selective_ml", "ma_dr_bc")
PRIMARY_METHODS = ("aipw", "ctmle", "cui_selective_ml", "ma_dr_bc")
GATE = None
RHO = None


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _init_worker(
    gate_script: str,
    rho_script: str,
    source_script: str,
    breadth_script: str,
    support_data: str,
) -> None:
    global GATE, RHO
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    gate = _load_module("dml_weighted_residual_gate_helper", Path(gate_script))
    gate.METHODS = METHODS
    gate._init_worker(rho_script, source_script, breadth_script, support_data)
    GATE = gate
    RHO = gate.RHO


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _gain(reference: list[float], repaired: list[float]) -> float:
    ref_mse = _mean([value * value for value in reference])
    repaired_mse = _mean([value * value for value in repaired])
    return 1.0 - repaired_mse / ref_mse if ref_mse > 0.0 else float("nan")


def _bootstrap_ci(
    reference: list[float],
    repaired: list[float],
    seed: int,
    draws_count: int = 5000,
) -> tuple[float, float]:
    if draws_count <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    ref = np.asarray(reference, dtype=float)
    repaired = np.asarray(repaired, dtype=float)
    draws = []
    for _ in range(draws_count):
        index = rng.integers(0, len(ref), len(ref))
        ref_mse = float(np.mean(ref[index] ** 2))
        draws.append(1.0 - float(np.mean(repaired[index] ** 2)) / ref_mse)
    lo, hi = np.percentile(np.asarray(draws), [2.5, 97.5])
    return float(lo), float(hi)


def _bootstrap_diff_ci(
    reference: list[float],
    current: list[float],
    residual_selected: list[float],
    seed: int,
    draws_count: int = 5000,
) -> tuple[float, float]:
    if draws_count <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    res = np.asarray(residual_selected, dtype=float)
    draws = []
    for _ in range(draws_count):
        index = rng.integers(0, len(ref), len(ref))
        ref_mse = float(np.mean(ref[index] ** 2))
        current_gain = 1.0 - float(np.mean(cur[index] ** 2)) / ref_mse
        residual_gain = 1.0 - float(np.mean(res[index] ** 2)) / ref_mse
        draws.append(residual_gain - current_gain)
    lo, hi = np.percentile(np.asarray(draws), [2.5, 97.5])
    return float(lo), float(hi)


def _weighted_residual_losses(
    y: np.ndarray,
    response: np.ndarray,
    p: np.ndarray,
    outcome: np.ndarray,
) -> np.ndarray:
    observed = np.asarray(response, dtype=bool)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0)
    weight = (1.0 - p[observed]) / (p[observed] * p[observed])
    residual = np.asarray(y, dtype=float)[observed] - np.asarray(outcome, dtype=float)[observed]
    return weight * residual * residual


def _select_by_weighted_residual_loss(
    row: Any,
    candidates: dict[str, Any],
    y: np.ndarray,
    response: np.ndarray,
) -> dict[str, Any]:
    grid = tuple(float(value) for value in row.region_damp_grid)
    losses = {}
    loss_vectors = {}
    for gamma in grid:
        vector = _weighted_residual_losses(
            y,
            response,
            candidates["p"],
            candidates["outcomes"][gamma],
        )
        loss_vectors[gamma] = vector
        losses[gamma] = float(np.mean(vector)) if len(vector) else float("inf")

    base = loss_vectors[0.0]
    eligible = [0.0]
    for gamma in grid:
        if gamma == 0.0:
            continue
        improvement = base - loss_vectors[gamma]
        if len(improvement) < 2:
            continue
        se = float(np.std(improvement, ddof=1) / math.sqrt(len(improvement)))
        if float(np.mean(improvement)) > float(row.validation_loss_se) * se:
            eligible.append(gamma)
    selected = min(eligible, key=lambda gamma: (losses[gamma], gamma))
    current_gamma = float(row.selected_gamma)
    return {
        "residual_gamma": selected,
        "residual_error": candidates["errors"][selected],
        "residual_loss": losses[selected],
        "base_residual_loss": losses[0.0],
        "current_residual_loss": losses[current_gamma],
        "current_replay_error": candidates["errors"][current_gamma],
        "eligible_count": len(eligible),
        "gamma_changed": abs(selected - current_gamma) > 1e-12,
    }


def _process_sample(rows: list[Any]) -> list[dict[str, Any]]:
    if GATE is None or RHO is None:
        raise RuntimeError("worker not initialized")
    vrt = RHO._VRT
    first = rows[0]
    seed = RHO._internal_seed(vrt, first)
    x, y, response, region, true_pi, theta, _mu = vrt.make_data(
        first.n,
        first.epsilon,
        first.strength,
        first.design,
        seed,
        first.mar_design,
    )
    out = []
    for row in rows:
        candidates = GATE._candidate_arrays(row, x, y, response, region, true_pi, theta, seed)
        selected = _select_by_weighted_residual_loss(row, candidates, y, response)
        out.append(
            {
                "source": row.source,
                "dataset": RHO._dataset(row.design),
                "setting": RHO._setting_label(row.design, row.n, row.strength),
                "group": row.group,
                "design": row.design,
                "method": row.method,
                "n": row.n,
                "strength": row.strength,
                "seed0": row.seed0,
                "rep": row.rep,
                "ref_error": row.ref_error,
                "current_error": row.selected_error,
                "current_gamma": row.selected_gamma,
                "current_replay_error": selected["current_replay_error"],
                "residual_error": selected["residual_error"],
                "residual_gamma": selected["residual_gamma"],
                "base_residual_loss": selected["base_residual_loss"],
                "current_residual_loss": selected["current_residual_loss"],
                "selected_residual_loss": selected["residual_loss"],
                "eligible_count": selected["eligible_count"],
                "gamma_changed": selected["gamma_changed"],
                "grid": "|".join(f"{float(value):g}" for value in row.region_damp_grid),
            }
        )
    return out


def _aggregate(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
    nboot: int = 5000,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out = []
    for key_values, group in sorted(groups.items()):
        ref = [float(row["ref_error"]) for row in group]
        current = [float(row["current_error"]) for row in group]
        residual = [float(row["residual_error"]) for row in group]
        seed = int(hashlib.sha256(json.dumps(key_values, sort_keys=True).encode()).hexdigest()[:8], 16)
        lo, hi = _bootstrap_ci(ref, residual, seed, nboot)
        diff_lo, diff_hi = _bootstrap_diff_ci(ref, current, residual, seed + 1, nboot)
        current_gain = _gain(ref, current)
        residual_gain = _gain(ref, residual)
        row = dict(zip(keys, key_values))
        row.update(
            {
                "reps": len(group),
                "ref_mse": _mean([value * value for value in ref]),
                "current_mse": _mean([value * value for value in current]),
                "residual_mse": _mean([value * value for value in residual]),
                "current_gain": current_gain,
                "residual_gain": residual_gain,
                "residual_gain_ci_low": lo,
                "residual_gain_ci_high": hi,
                "residual_minus_current_gain": residual_gain - current_gain,
                "residual_minus_current_gain_ci_low": diff_lo,
                "residual_minus_current_gain_ci_high": diff_hi,
                "current_activation": _mean(
                    [1.0 if float(item["current_gamma"]) > 0.0 else 0.0 for item in group]
                ),
                "residual_activation": _mean(
                    [1.0 if float(item["residual_gamma"]) > 0.0 else 0.0 for item in group]
                ),
                "gamma_change_rate": _mean([1.0 if item["gamma_changed"] else 0.0 for item in group]),
                "current_harm_rate": _mean(
                    [
                        1.0
                        if float(item["current_error"]) ** 2 > float(item["ref_error"]) ** 2
                        else 0.0
                        for item in group
                    ]
                ),
                "residual_harm_rate": _mean(
                    [
                        1.0
                        if float(item["residual_error"]) ** 2 > float(item["ref_error"]) ** 2
                        else 0.0
                        for item in group
                    ]
                ),
                "mean_base_residual_loss": _mean(
                    [float(item["base_residual_loss"]) for item in group]
                ),
                "mean_current_residual_loss": _mean(
                    [float(item["current_residual_loss"]) for item in group]
                ),
                "mean_selected_residual_loss": _mean(
                    [float(item["selected_residual_loss"]) for item in group]
                ),
            }
        )
        row["significance_class"] = (
            "win" if lo > 0.0 else "loss" if hi < 0.0 else "crosses_zero"
        )
        out.append(row)
    return out


def _primary_summary(rows: list[dict[str, Any]], nboot: int = 5000) -> list[dict[str, Any]]:
    group = [row for row in rows if row["method"] in PRIMARY_METHODS]
    ref = [float(row["ref_error"]) for row in group]
    current = [float(row["current_error"]) for row in group]
    residual = [float(row["residual_error"]) for row in group]
    seed = 20260903
    lo, hi = _bootstrap_ci(ref, residual, seed, nboot)
    diff_lo, diff_hi = _bootstrap_diff_ci(ref, current, residual, seed + 1, nboot)
    current_gain = _gain(ref, current)
    residual_gain = _gain(ref, residual)
    return [
        {
            "summary": "primary_equal_replication",
            "methods": "|".join(PRIMARY_METHODS),
            "reps": len(group),
            "current_gain": current_gain,
            "residual_gain": residual_gain,
            "residual_gain_ci_low": lo,
            "residual_gain_ci_high": hi,
            "residual_minus_current_gain": residual_gain - current_gain,
            "residual_minus_current_gain_ci_low": diff_lo,
            "residual_minus_current_gain_ci_high": diff_hi,
            "current_activation": _mean(
                [1.0 if float(item["current_gamma"]) > 0.0 else 0.0 for item in group]
            ),
            "residual_activation": _mean(
                [1.0 if float(item["residual_gamma"]) > 0.0 else 0.0 for item in group]
            ),
            "gamma_change_rate": _mean([1.0 if item["gamma_changed"] else 0.0 for item in group]),
        }
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def _write_results(
    path: Path,
    method_rows: list[dict[str, Any]],
    primary_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Weighted-Residual Gamma Selection Ablation",
        "",
        "This diagnostic keeps the fitted residual direction and upstream fits",
        "fixed, then selects gamma by held-out responder weighted residual",
        "loss instead of centered score loss.  The gamma grid and one-SE",
        "stand-down rule are otherwise unchanged.",
        "",
        "## Method Readout",
        "",
    ]
    for row in sorted(method_rows, key=lambda item: item["method"]):
        lines.append(
            "- {method}: current {current}, residual-selected {new} [{lo}, {hi}], "
            "delta {delta} [{delta_lo}, {delta_hi}], activation {new_act:.1f}% vs "
            "{cur_act:.1f}%, harm {harm:.1f}% vs {cur_harm:.1f}%, gamma changed "
            "{changed:.1f}%".format(
                method=row["method"],
                current=_pct(float(row["current_gain"])),
                new=_pct(float(row["residual_gain"])),
                lo=_pct(float(row["residual_gain_ci_low"])),
                hi=_pct(float(row["residual_gain_ci_high"])),
                delta=_pct(float(row["residual_minus_current_gain"])),
                delta_lo=_pct(float(row["residual_minus_current_gain_ci_low"])),
                delta_hi=_pct(float(row["residual_minus_current_gain_ci_high"])),
                new_act=100.0 * float(row["residual_activation"]),
                cur_act=100.0 * float(row["current_activation"]),
                harm=100.0 * float(row["residual_harm_rate"]),
                cur_harm=100.0 * float(row["current_harm_rate"]),
                changed=100.0 * float(row["gamma_change_rate"]),
            )
        )
    if primary_rows:
        primary = primary_rows[0]
        lines.extend(
            [
                "",
                "## Primary Average",
                "",
                "- Primary methods: current {current}, residual-selected {new} "
                "[{lo}, {hi}], delta {delta} [{delta_lo}, {delta_hi}], "
                "gamma changed {changed:.1f}%.".format(
                    current=_pct(float(primary["current_gain"])),
                    new=_pct(float(primary["residual_gain"])),
                    lo=_pct(float(primary["residual_gain_ci_low"])),
                    hi=_pct(float(primary["residual_gain_ci_high"])),
                    delta=_pct(float(primary["residual_minus_current_gain"])),
                    delta_lo=_pct(float(primary["residual_minus_current_gain_ci_low"])),
                    delta_hi=_pct(float(primary["residual_minus_current_gain_ci_high"])),
                    changed=100.0 * float(primary["gamma_change_rate"]),
                ),
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--gate-script",
        type=Path,
        default=Path("scripts/dml_universal_leverage_gate_diagnostic.py"),
    )
    parser.add_argument(
        "--rho-script",
        type=Path,
        default=Path("scripts/dml_mse_gain_by_rho.py"),
    )
    parser.add_argument(
        "--source-script",
        type=Path,
        default=Path(
            "/tmp/ney_global_recreate_20260830/replication/hidden_bias_regional_repair/"
            "source_snapshots/20260814_unified_cartesian_v3/validated_reference_transfer.py"
        ),
    )
    parser.add_argument(
        "--breadth-script",
        type=Path,
        default=Path(
            "/tmp/ney_global_recreate_20260830/replication/hidden_bias_regional_repair/"
            "scripts/section4_breadth_experiments.py"
        ),
    )
    parser.add_argument("--support-data", type=Path, default=Path("/tmp/dml_real_benchmark_support_data"))
    parser.add_argument("--aug14-run-dir", type=Path, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("support_csv/dml_weighted_residual_gamma_selection_ablation_20260903"),
    )
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--write-replication-rows", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    gate_script = args.gate_script if args.gate_script.is_absolute() else data_root / args.gate_script
    rho_script = args.rho_script if args.rho_script.is_absolute() else data_root / args.rho_script
    gate = _load_module("dml_weighted_residual_gate_loader", gate_script)
    gate.METHODS = METHODS
    rho = _load_module("dml_weighted_residual_rho_loader", rho_script)
    rho.PRIMARY_METHODS = METHODS
    aug14 = args.aug14_run_dir
    if aug14 is None:
        latest = Path("/tmp/dml_rho_extract_latest.txt")
        if not latest.exists():
            raise SystemExit("pass --aug14-run-dir or create /tmp/dml_rho_extract_latest.txt")
        aug14 = Path(latest.read_text().strip()) / "dml" / "cartesian_dml_ks_alignment_v3"

    rows = []
    rows.extend(rho._read_extracted_aug14_rows(aug14))
    rows.extend(
        rho._read_release_rows(
            data_root / "support_csv/dml_real_benchmark_expansion_20260831",
            "real_benchmark_ihdp_acic2016",
        )
    )
    rows.extend(
        rho._read_release_rows(
            data_root / "support_csv/dml_real_benchmark_acic2017_20260831",
            "real_benchmark_acic2017",
        )
    )
    rows.extend(
        rho._read_release_rows(
            data_root / "support_csv/dml_real_benchmark_twins_20260831",
            "real_benchmark_twins",
        )
    )

    sample_groups: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
    for row in rows:
        key = (
            row.design,
            row.n,
            row.epsilon,
            row.strength,
            row.mar_design,
            row.seed0,
            row.rep,
            row.propensity_mode,
            row.learner,
            row.propensity_learner,
            row.folds,
        )
        sample_groups[key].append(row)

    replication_rows: list[dict[str, Any]] = []
    tasks = list(sample_groups.values())
    with ProcessPoolExecutor(
        max_workers=args.jobs,
        initializer=_init_worker,
        initargs=(
            str(gate_script.resolve()),
            str(rho_script.resolve()),
            str(args.source_script.resolve()),
            str(args.breadth_script.resolve()),
            str(args.support_data.resolve()),
        ),
    ) as pool:
        futures = [pool.submit(_process_sample, task) for task in tasks]
        completed = 0
        for future in as_completed(futures):
            replication_rows.extend(future.result())
            completed += 1
            if completed % max(1, args.jobs * 4) == 0 or completed == len(tasks):
                print(f"completed {completed}/{len(tasks)} samples", flush=True)

    replication_rows.sort(
        key=lambda row: (
            row["dataset"],
            row["setting"],
            row["method"],
            row["seed0"],
            row["rep"],
        )
    )
    setting_rows = _aggregate(
        replication_rows,
        ("dataset", "setting", "design", "method", "n", "strength"),
        nboot=0,
    )
    dataset_method_rows = _aggregate(
        replication_rows,
        ("dataset", "method"),
        nboot=0,
    )
    method_rows = _aggregate(
        replication_rows,
        ("method",),
        nboot=5000,
    )
    primary_rows = _primary_summary(replication_rows, nboot=5000)
    out_dir = args.out_dir if args.out_dir.is_absolute() else data_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.write_replication_rows:
        _write_csv(out_dir / "weighted_residual_gamma_replication_rows.csv", replication_rows)
    _write_csv(out_dir / "weighted_residual_gamma_setting_summary.csv", setting_rows)
    _write_csv(out_dir / "weighted_residual_gamma_dataset_method_summary.csv", dataset_method_rows)
    _write_csv(out_dir / "weighted_residual_gamma_method_summary.csv", method_rows)
    _write_csv(out_dir / "weighted_residual_gamma_primary_summary.csv", primary_rows)
    _write_results(out_dir / "RESULTS.md", method_rows, primary_rows)
    provenance = {
        "source_script": str(args.source_script.resolve()),
        "source_script_sha256": _sha256(args.source_script.resolve()),
        "breadth_script": str(args.breadth_script.resolve()),
        "breadth_script_sha256": _sha256(args.breadth_script.resolve()),
        "rho_script": str(rho_script.resolve()),
        "rho_script_sha256": _sha256(rho_script.resolve()),
        "gate_script": str(gate_script.resolve()),
        "gate_script_sha256": _sha256(gate_script.resolve()),
        "support_data": str(args.support_data.resolve()),
        "aug14_run_dir": str(aug14.resolve()),
        "methods": list(METHODS),
        "primary_methods": list(PRIMARY_METHODS),
        "rows": len(replication_rows),
        "sample_groups": len(sample_groups),
        "replication_rows_written": bool(args.write_replication_rows),
        "diagnostic": "gamma replay selected by held-out weighted residual loss",
        "selection_loss": "mean over responders of (1 - p_hat) / p_hat^2 * (Y - m_gamma)^2",
        "selection_gate": "one-SE improvement over gamma=0, then minimum weighted residual loss among eligible candidates",
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    checksum_rows = []
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            checksum_rows.append(f"{_sha256(path)}  {path.name}")
    (out_dir / "SHA256SUMS").write_text("\n".join(checksum_rows) + "\n")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
