#!/usr/bin/env python3
"""Add gamma=0.75 to the weighted-residual gamma selector replay."""

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
AUGMENTED_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
WR = None
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _init_worker(
    weighted_script: str,
    gate_script: str,
    rho_script: str,
    source_script: str,
    breadth_script: str,
    support_data: str,
) -> None:
    global WR, GATE, RHO
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    wr = _load_module("dml_wr_augmented_worker", Path(weighted_script))
    wr._init_worker(gate_script, rho_script, source_script, breadth_script, support_data)
    WR = wr
    GATE = wr.GATE
    RHO = wr.RHO


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
    augmented: list[float],
    seed: int,
    draws_count: int = 5000,
) -> tuple[float, float]:
    if draws_count <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    aug = np.asarray(augmented, dtype=float)
    draws = []
    for _ in range(draws_count):
        index = rng.integers(0, len(ref), len(ref))
        ref_mse = float(np.mean(ref[index] ** 2))
        current_gain = 1.0 - float(np.mean(cur[index] ** 2)) / ref_mse
        augmented_gain = 1.0 - float(np.mean(aug[index] ** 2)) / ref_mse
        draws.append(augmented_gain - current_gain)
    lo, hi = np.percentile(np.asarray(draws), [2.5, 97.5])
    return float(lo), float(hi)


def _load_selection_map(path: Path) -> dict[tuple[str, str, int, float, int, int], dict[str, float]]:
    out = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["design"],
                row["method"],
                int(row["n"]),
                float(row["strength"]),
                int(row["seed0"]),
                int(row["rep"]),
            )
            out[key] = {
                "residual_error": float(row["residual_error"]),
                "residual_gamma": float(row["residual_gamma"]),
            }
    return out


def _selection_key(row: Any) -> tuple[str, str, int, float, int, int]:
    return (
        row.design,
        row.method,
        int(row.n),
        float(row.strength),
        int(row.seed0),
        int(row.rep),
    )


def _outcome_at_gamma(candidates: dict[str, Any], gamma: float) -> np.ndarray:
    outcomes = candidates["outcomes"]
    if gamma in outcomes:
        return np.asarray(outcomes[gamma], dtype=float)
    return np.asarray(outcomes[0.0], dtype=float) + gamma * (
        np.asarray(outcomes[1.0], dtype=float) - np.asarray(outcomes[0.0], dtype=float)
    )


def _error_at_gamma(candidates: dict[str, Any], gamma: float) -> float:
    errors = candidates["errors"]
    if gamma in errors:
        return float(errors[gamma])
    return float(errors[0.0]) + gamma * (float(errors[1.0]) - float(errors[0.0]))


def _weighted_losses(y: np.ndarray, response: np.ndarray, p: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    observed = np.asarray(response, dtype=bool)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0)
    residual = np.asarray(y, dtype=float)[observed] - np.asarray(outcome, dtype=float)[observed]
    weight = (1.0 - p[observed]) / (p[observed] * p[observed])
    return weight * residual * residual


def _select_augmented(row: Any, candidates: dict[str, Any], y: np.ndarray, response: np.ndarray) -> dict[str, Any]:
    losses = {}
    loss_vectors = {}
    for gamma in AUGMENTED_GRID:
        outcome = _outcome_at_gamma(candidates, gamma)
        vector = _weighted_losses(y, response, candidates["p"], outcome)
        losses[gamma] = float(np.mean(vector)) if len(vector) else float("inf")
        loss_vectors[gamma] = vector
    base = loss_vectors[0.0]
    eligible = [0.0]
    for gamma in AUGMENTED_GRID:
        if gamma == 0.0:
            continue
        improvement = base - loss_vectors[gamma]
        if len(improvement) < 2:
            continue
        se = float(np.std(improvement, ddof=1) / math.sqrt(len(improvement)))
        if float(np.mean(improvement)) > float(row.validation_loss_se) * se:
            eligible.append(gamma)
    selected = min(eligible, key=lambda gamma: (losses[gamma], gamma))
    return {
        "augmented_gamma": selected,
        "augmented_error": _error_at_gamma(candidates, selected),
        "augmented_residual_loss": losses[selected],
        "candidate_075_error": _error_at_gamma(candidates, 0.75),
        "candidate_075_residual_loss": losses[0.75],
        "selected_is_075": abs(selected - 0.75) <= 1e-12,
        "eligible_count": len(eligible),
    }


def _process_sample(rows: list[Any], selection_map: dict[tuple[str, str, int, float, int, int], dict[str, float]]) -> list[dict[str, Any]]:
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
        key = _selection_key(row)
        current = selection_map[key]
        candidates = GATE._candidate_arrays(row, x, y, response, region, true_pi, theta, seed)
        augmented = _select_augmented(row, candidates, y, response)
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
                "current_error": current["residual_error"],
                "current_gamma": current["residual_gamma"],
                "augmented_error": augmented["augmented_error"],
                "augmented_gamma": augmented["augmented_gamma"],
                "candidate_075_error": augmented["candidate_075_error"],
                "candidate_075_residual_loss": augmented["candidate_075_residual_loss"],
                "augmented_residual_loss": augmented["augmented_residual_loss"],
                "augmented_selected_is_075": augmented["selected_is_075"],
                "eligible_count": augmented["eligible_count"],
                "gamma_changed": abs(augmented["augmented_gamma"] - current["residual_gamma"]) > 1e-12,
            }
        )
    return out


def _aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...], nboot: int = 5000) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out = []
    for key_values, group in sorted(groups.items()):
        ref = [float(row["ref_error"]) for row in group]
        current = [float(row["current_error"]) for row in group]
        augmented = [float(row["augmented_error"]) for row in group]
        seed = int(hashlib.sha256(json.dumps(key_values, sort_keys=True).encode()).hexdigest()[:8], 16)
        lo, hi = _bootstrap_ci(ref, augmented, seed, nboot)
        diff_lo, diff_hi = _bootstrap_diff_ci(ref, current, augmented, seed + 1, nboot)
        current_gain = _gain(ref, current)
        augmented_gain = _gain(ref, augmented)
        row = dict(zip(keys, key_values))
        row.update(
            {
                "reps": len(group),
                "ref_mse": _mean([value * value for value in ref]),
                "current_mse": _mean([value * value for value in current]),
                "augmented_mse": _mean([value * value for value in augmented]),
                "current_gain": current_gain,
                "augmented_gain": augmented_gain,
                "augmented_gain_ci_low": lo,
                "augmented_gain_ci_high": hi,
                "augmented_minus_current_gain": augmented_gain - current_gain,
                "augmented_minus_current_gain_ci_low": diff_lo,
                "augmented_minus_current_gain_ci_high": diff_hi,
                "current_activation": _mean([1.0 if float(item["current_gamma"]) > 0.0 else 0.0 for item in group]),
                "augmented_activation": _mean([1.0 if float(item["augmented_gamma"]) > 0.0 else 0.0 for item in group]),
                "gamma_change_rate": _mean([1.0 if item["gamma_changed"] else 0.0 for item in group]),
                "selected_075_rate": _mean([1.0 if item["augmented_selected_is_075"] else 0.0 for item in group]),
                "current_harm_rate": _mean([1.0 if float(item["current_error"]) ** 2 > float(item["ref_error"]) ** 2 else 0.0 for item in group]),
                "augmented_harm_rate": _mean([1.0 if float(item["augmented_error"]) ** 2 > float(item["ref_error"]) ** 2 else 0.0 for item in group]),
            }
        )
        row["significance_class"] = "win" if lo > 0.0 else "loss" if hi < 0.0 else "crosses_zero"
        out.append(row)
    return out


def _primary_summary(rows: list[dict[str, Any]], nboot: int = 5000) -> list[dict[str, Any]]:
    group = [row for row in rows if row["method"] in PRIMARY_METHODS]
    ref = [float(row["ref_error"]) for row in group]
    current = [float(row["current_error"]) for row in group]
    augmented = [float(row["augmented_error"]) for row in group]
    lo, hi = _bootstrap_ci(ref, augmented, 20260903, nboot)
    diff_lo, diff_hi = _bootstrap_diff_ci(ref, current, augmented, 20260904, nboot)
    current_gain = _gain(ref, current)
    augmented_gain = _gain(ref, augmented)
    return [
        {
            "summary": "primary_equal_replication",
            "methods": "|".join(PRIMARY_METHODS),
            "reps": len(group),
            "current_gain": current_gain,
            "augmented_gain": augmented_gain,
            "augmented_gain_ci_low": lo,
            "augmented_gain_ci_high": hi,
            "augmented_minus_current_gain": augmented_gain - current_gain,
            "augmented_minus_current_gain_ci_low": diff_lo,
            "augmented_minus_current_gain_ci_high": diff_hi,
            "current_activation": _mean([1.0 if float(item["current_gamma"]) > 0.0 else 0.0 for item in group]),
            "augmented_activation": _mean([1.0 if float(item["augmented_gamma"]) > 0.0 else 0.0 for item in group]),
            "gamma_change_rate": _mean([1.0 if item["gamma_changed"] else 0.0 for item in group]),
            "selected_075_rate": _mean([1.0 if item["augmented_selected_is_075"] else 0.0 for item in group]),
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
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
            extrasaction="ignore",
        )
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
        "# Weighted-Residual Selector Augmented Gamma Grid Ablation",
        "",
        "This diagnostic keeps the same fitted residual directions and selects",
        "gamma by held-out weighted residual loss after adding gamma=0.75 to",
        "the candidate grid.  The current comparator is the four-point",
        "weighted-residual selector, not the older centered-score selector.",
        "",
        "## Method Readout",
        "",
    ]
    for row in sorted(method_rows, key=lambda item: item["method"]):
        lines.append(
            "- {method}: current {current}, augmented {new} [{lo}, {hi}], "
            "delta {delta} [{delta_lo}, {delta_hi}], activation {new_act:.1f}% vs "
            "{cur_act:.1f}%, harm {harm:.1f}% vs {cur_harm:.1f}%, gamma changed "
            "{changed:.1f}%, selected 0.75 {sel075:.1f}%".format(
                method=row["method"],
                current=_pct(float(row["current_gain"])),
                new=_pct(float(row["augmented_gain"])),
                lo=_pct(float(row["augmented_gain_ci_low"])),
                hi=_pct(float(row["augmented_gain_ci_high"])),
                delta=_pct(float(row["augmented_minus_current_gain"])),
                delta_lo=_pct(float(row["augmented_minus_current_gain_ci_low"])),
                delta_hi=_pct(float(row["augmented_minus_current_gain_ci_high"])),
                new_act=100.0 * float(row["augmented_activation"]),
                cur_act=100.0 * float(row["current_activation"]),
                harm=100.0 * float(row["augmented_harm_rate"]),
                cur_harm=100.0 * float(row["current_harm_rate"]),
                changed=100.0 * float(row["gamma_change_rate"]),
                sel075=100.0 * float(row["selected_075_rate"]),
            )
        )
    if primary_rows:
        primary = primary_rows[0]
        lines.extend(
            [
                "",
                "## Primary Average",
                "",
                "- Primary methods: current {current}, augmented {new} [{lo}, {hi}], "
                "delta {delta} [{delta_lo}, {delta_hi}], selected 0.75 "
                "{sel075:.1f}%.".format(
                    current=_pct(float(primary["current_gain"])),
                    new=_pct(float(primary["augmented_gain"])),
                    lo=_pct(float(primary["augmented_gain_ci_low"])),
                    hi=_pct(float(primary["augmented_gain_ci_high"])),
                    delta=_pct(float(primary["augmented_minus_current_gain"])),
                    delta_lo=_pct(float(primary["augmented_minus_current_gain_ci_low"])),
                    delta_hi=_pct(float(primary["augmented_minus_current_gain_ci_high"])),
                    sel075=100.0 * float(primary["selected_075_rate"]),
                ),
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def _load_rows(data_root: Path, rho: Any, aug14: Path) -> list[Any]:
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
    return rows


def _resolve(data_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else data_root / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--weighted-script",
        type=Path,
        default=Path("scripts/dml_weighted_residual_gamma_selection_ablation.py"),
    )
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
        "--selection-overlay",
        type=Path,
        default=Path(
            "support_csv/dml_weighted_residual_gamma_selection_ablation_20260903/"
            "weighted_residual_gamma_replication_rows.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("support_csv/dml_weighted_residual_augmented_gamma_grid_ablation_20260903"),
    )
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--write-replication-rows", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    rho_script = _resolve(data_root, args.rho_script)
    weighted_script = _resolve(data_root, args.weighted_script)
    gate_script = _resolve(data_root, args.gate_script)
    selection_overlay = _resolve(data_root, args.selection_overlay)
    rho = _load_module("dml_wr_augmented_rho_loader", rho_script)
    rho.PRIMARY_METHODS = METHODS

    aug14 = args.aug14_run_dir
    if aug14 is None:
        latest = Path("/tmp/dml_rho_extract_latest.txt")
        if not latest.exists():
            raise SystemExit("pass --aug14-run-dir or create /tmp/dml_rho_extract_latest.txt")
        aug14 = Path(latest.read_text().strip()) / "dml" / "cartesian_dml_ks_alignment_v3"

    rows = _load_rows(data_root, rho, aug14)
    selection_map = _load_selection_map(selection_overlay)
    missing = [_selection_key(row) for row in rows if _selection_key(row) not in selection_map]
    if missing:
        raise SystemExit(f"missing weighted-residual selection keys: {missing[:5]}")

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
            str(weighted_script.resolve()),
            str(gate_script.resolve()),
            str(rho_script.resolve()),
            str(args.source_script.resolve()),
            str(args.breadth_script.resolve()),
            str(args.support_data.resolve()),
        ),
    ) as pool:
        futures = [pool.submit(_process_sample, task, selection_map) for task in tasks]
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
    dataset_method_rows = _aggregate(replication_rows, ("dataset", "method"), nboot=0)
    method_rows = _aggregate(replication_rows, ("method",), nboot=5000)
    primary_rows = _primary_summary(replication_rows, nboot=5000)

    out_dir = args.out_dir if args.out_dir.is_absolute() else data_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.write_replication_rows:
        _write_csv(out_dir / "weighted_residual_augmented_gamma_replication_rows.csv", replication_rows)
    _write_csv(out_dir / "weighted_residual_augmented_gamma_setting_summary.csv", setting_rows)
    _write_csv(out_dir / "weighted_residual_augmented_gamma_dataset_method_summary.csv", dataset_method_rows)
    _write_csv(out_dir / "weighted_residual_augmented_gamma_method_summary.csv", method_rows)
    _write_csv(out_dir / "weighted_residual_augmented_gamma_primary_summary.csv", primary_rows)
    _write_results(out_dir / "RESULTS.md", method_rows, primary_rows)
    provenance = {
        "source_script": str(args.source_script.resolve()),
        "source_script_sha256": _sha256(args.source_script.resolve()),
        "breadth_script": str(args.breadth_script.resolve()),
        "breadth_script_sha256": _sha256(args.breadth_script.resolve()),
        "weighted_script": str(weighted_script.resolve()),
        "weighted_script_sha256": _sha256(weighted_script.resolve()),
        "rho_script": str(rho_script.resolve()),
        "rho_script_sha256": _sha256(rho_script.resolve()),
        "gate_script": str(gate_script.resolve()),
        "gate_script_sha256": _sha256(gate_script.resolve()),
        "selection_overlay": str(selection_overlay.resolve()),
        "selection_overlay_sha256": _sha256(selection_overlay.resolve()),
        "support_data": str(args.support_data.resolve()),
        "aug14_run_dir": str(aug14.resolve()),
        "methods": list(METHODS),
        "primary_methods": list(PRIMARY_METHODS),
        "rows": len(replication_rows),
        "sample_groups": len(sample_groups),
        "augmented_grid": list(AUGMENTED_GRID),
        "diagnostic": "gamma=0.75 challenger under held-out weighted-residual selection",
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
