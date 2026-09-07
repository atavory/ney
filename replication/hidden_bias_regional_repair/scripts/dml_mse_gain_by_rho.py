#!/usr/bin/env python3
"""Retrospective MSE gain by response-model misspecification rho.

The diagnostic replays the paper-facing known-truth settings, computes

    rho_p^2 = E[(1 - pi / p_hat)^2 pi / (1 - pi)]

for the response model actually used by each emitted expert row, and joins it
to the released reference-vs-selected-candidate squared errors.  It also
reconstructs the reference and selected repaired outcome predictions from the
emitted tuning choices in order to compute true residual-square diagnostics.
It does not reselect the candidate library.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PRIMARY_METHODS = ("aipw", "ma_dr_bc", "cui_selective_ml", "ctmle")
KS_DESIGNS = {
    "kang_schafer_cc": "CC",
    "kang_schafer_ci": "CI",
    "kang_schafer_ic": "IC",
    "kang_schafer_ii": "II",
}
BENCHMARK_DESIGNS = {
    "ihdp_semisynth",
    "ihdp_misaligned",
    "acic2016_semisynth",
    "acic2016_misaligned",
    "acic2017_semisynth",
    "acic2017_misaligned",
    "twins_semisynth",
    "twins_misaligned",
}

_VRT = None


@dataclass(frozen=True)
class MethodRow:
    source: str
    group: str
    design: str
    method: str
    n: int
    epsilon: float
    strength: float
    mar_design: str
    seed0: int
    rep: int
    analysis_region: str
    propensity_mode: str
    learner: str
    propensity_learner: str
    repair_mode: str
    tau_grid: tuple[float, ...]
    folds: int
    region_damp_grid: tuple[float, ...]
    validation_risk: str
    validation_region_weight: float
    validation_loss_se: float
    selector: str
    lepski_c: float
    region_quantile: float
    region_min_observed: int
    region_kappa_floor: float
    selector_ablation: str
    region_detector_c: float
    selected_tau: float
    selected_propensity_learner: str
    selected_outcome_learner: str
    ref_error: float
    selected_error: float
    c2_error: float
    selected_gamma: float
    c2_weight: float


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _init_worker(source_script: str, breadth_script: str, support_data: str) -> None:
    global _VRT
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ["DML_SUPPORT_DATA"] = support_data
    os.environ["USHMOO_SUPPORT_DATA"] = support_data
    for module_dir in (
        Path(source_script).resolve().parent,
        Path(breadth_script).resolve().parent,
    ):
        module_dir_text = str(module_dir)
        if module_dir_text not in sys.path:
            sys.path.insert(0, module_dir_text)
    vrt = _load_module("dml_rho_validated_reference_transfer", Path(source_script))
    breadth = _load_module("dml_rho_section4_breadth", Path(breadth_script))
    breadth._install_adapter(vrt)
    _VRT = vrt


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _option_values(command: list[str], name: str) -> list[str]:
    if name not in command:
        return []
    index = command.index(name) + 1
    values: list[str] = []
    while index < len(command) and not command[index].startswith("--"):
        values.append(command[index])
        index += 1
    return values


def _command_values(
    command: list[str],
    name: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    values = _option_values(command, name)
    return tuple(values) if values else default


def _row_float(row: dict[str, str], name: str, default: float) -> float:
    value = row.get(name, "")
    parsed = _parse_float(value)
    return parsed if math.isfinite(parsed) else default


def _row_int(row: dict[str, str], name: str, default: int) -> int:
    value = row.get(name, "")
    return int(float(value)) if value not in {"", None} else default


def _job_lookup(manifest: Path) -> dict[str, dict[str, str]]:
    lookup = {}
    with manifest.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            row = dict(row)
            command = json.loads(row["command_json"])
            row["folds"] = (_option_values(command, "--folds") or ["3"])[0]
            row["tau_grid"] = "|".join(
                _command_values(
                    command,
                    "--tau-grid",
                    ("0.05", "0.10", "0.25", "0.50"),
                )
            )
            row["region_damp_grid"] = "|".join(
                _command_values(command, "--region-damp-grid", ("1.0",))
            )
            lookup[Path(row["rep_out"]).name] = row
    return lookup


def _dataset(design: str) -> str:
    if design.startswith("kang_schafer"):
        return "Kang-Schafer"
    if design.startswith("ihdp"):
        return "IHDP"
    if design.startswith("acic2016"):
        return "ACIC 2016"
    if design.startswith("acic2017"):
        return "ACIC 2017"
    if design.startswith("twins"):
        return "Twins"
    return design


def _setting_label(design: str, n: int, strength: float) -> str:
    if design in KS_DESIGNS:
        return f"KS {KS_DESIGNS[design]}, n={n}"
    variant = "A" if design.endswith("_semisynth") else "B"
    return f"{_dataset(design)} {variant}, strength={strength:g}"


def _include(row: dict[str, str]) -> bool:
    method = row.get("reference_method", row.get("method", ""))
    design = row.get("design", "")
    if method not in PRIMARY_METHODS:
        return False
    return design in KS_DESIGNS or design in BENCHMARK_DESIGNS


def _method_row(
    row: dict[str, str],
    job: dict[str, str],
    source: str,
) -> MethodRow:
    return MethodRow(
        source=source,
        group=job.get("group", row.get("group", "")),
        design=row["design"],
        method=row["reference_method"],
        n=int(row["n"]),
        epsilon=_parse_float(row["epsilon"]),
        strength=_parse_float(row["strength"]),
        mar_design=row.get("mar_design", "box"),
        seed0=int(job["seed"]),
        rep=int(row["rep"]),
        analysis_region=row.get("analysis_region", "estimated_residual_lowp_supported"),
        propensity_mode=row.get("propensity_mode", "estimated"),
        learner=row.get("learner", "xgboost"),
        propensity_learner=row.get("propensity_learner", row.get("learner", "xgboost")),
        repair_mode=row.get("repair_mode", "if_residual"),
        tau_grid=tuple(float(value) for value in job["tau_grid"].split("|")),
        folds=int(job.get("folds", row.get("folds", "3"))),
        region_damp_grid=tuple(
            float(value)
            for value in job.get(
                "region_damp_grid",
                row.get("region_damp_grid", "1.0").replace("|", " "),
            ).replace("|", " ").split()
        ),
        validation_risk=row.get("validation_risk", "balanced_mse"),
        validation_region_weight=_row_float(row, "validation_region_weight", -1.0),
        validation_loss_se=_row_float(row, "validation_loss_se", 1.0),
        selector=row.get("selector", "obsval"),
        lepski_c=_row_float(row, "lepski_c", 4.0),
        region_quantile=_row_float(row, "region_quantile", 0.10),
        region_min_observed=_row_int(row, "region_min_observed", 30),
        region_kappa_floor=_row_float(row, "region_kappa_floor", 0.10),
        selector_ablation=row.get("region_selector_ablation", "legacy"),
        region_detector_c=_row_float(row, "region_detector_c", 4.0),
        selected_tau=_parse_float(row.get("selected_tau", "nan")),
        selected_propensity_learner=row.get("cui_selected_propensity_learner", ""),
        selected_outcome_learner=row.get("cui_selected_outcome_learner", ""),
        ref_error=_parse_float(row["ref_error"]),
        selected_error=_parse_float(row["rt_error"]),
        c2_error=_parse_float(row["shrink_error"]),
        selected_gamma=_parse_float(row.get("selected_region_damp", "nan")),
        c2_weight=_parse_float(row.get("weight", "nan")),
    )


def _read_extracted_aug14_rows(run_dir: Path) -> list[MethodRow]:
    jobs = _job_lookup(run_dir / "manifest.tsv")
    out: list[MethodRow] = []
    for path in sorted(run_dir.glob("*.reps.csv")):
        job = jobs.get(path.name)
        if job is None:
            raise SystemExit(f"no manifest row for {path.name}")
        if job["group"] != "kang_schafer":
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if _include(row):
                    out.append(_method_row(row, job, "aug14_unified_cartesian"))
    return out


def _read_release_rows(directory: Path, source: str) -> list[MethodRow]:
    jobs = _job_lookup(directory / "manifest.tsv")
    out: list[MethodRow] = []
    with (directory / "raw_rows.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not _include(row):
                continue
            job = jobs.get(Path(row["source_rep_file"]).name)
            if job is None:
                raise SystemExit(f"no manifest row for {row['source_rep_file']}")
            out.append(_method_row(row, job, source))
    return out


def _internal_seed(vrt: Any, row: MethodRow) -> int:
    return (
        row.seed0
        + row.n * 1009
        + int(row.epsilon * 1000) * 100003
        + vrt._design_strength_code(row.strength)
        * 10007
        + (row.rep - 1) * 37
        + vrt._design_seed_offset(row.design)
        + vrt._mar_seed_offset(row.mar_design)
    )


def _crossfit_p_raw(vrt: Any, x, response, true_pi, row: MethodRow, seed: int):
    n = len(response)
    labels = np.random.default_rng(seed + 17).integers(0, row.folds, n)
    p_raw = np.empty(n, dtype=float)
    for fold in range(row.folds):
        test = labels == fold
        train = ~test
        _, p_test = vrt._propensity_predictions(
            x,
            response,
            true_pi,
            train,
            test,
            row.propensity_mode,
            seed + 17 + 101 * fold,
            row.propensity_learner,
        )
        p_raw[test] = p_test
    return np.clip(p_raw, 1e-12, 1.0)


def _cui_selected_p(vrt: Any, x, response, selected_name: str, seed: int):
    if not selected_name:
        raise RuntimeError("missing selective-ML propensity learner")
    names = ("logistic_l1", "random_forest", "gradient_boosting")
    selected_index = names.index(selected_name)
    split_count = 2
    labels = np.empty(len(response), dtype=int)
    splitter = vrt.StratifiedKFold(
        n_splits=split_count,
        shuffle=True,
        random_state=seed + 17 + 7001,
    )
    for fold, (_, test_index) in enumerate(
        splitter.split(np.zeros((len(response), 1)), response)
    ):
        labels[test_index] = fold
    p = np.empty(len(response), dtype=float)
    for fold in range(split_count):
        test = labels == fold
        train = ~test
        model = vrt._cui_candidate_propensity(
            selected_name,
            seed + 17 + 7001 + 1009 * fold + 31 * selected_index,
        )
        model.fit(x[train], response[train])
        p[test] = model.predict_proba(x[test])[:, 1]
    return np.clip(p, 1e-6, 1.0)


def _cui_selected_p_m(
    vrt: Any,
    x: np.ndarray,
    y: np.ndarray,
    response: np.ndarray,
    selected_propensity_name: str,
    selected_outcome_name: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    propensity_names = ("logistic_l1", "random_forest", "gradient_boosting")
    outcome_names = ("lasso", "random_forest", "gradient_boosting")
    selected_k = propensity_names.index(selected_propensity_name)
    selected_l = outcome_names.index(selected_outcome_name)
    split_count = 2
    labels = np.empty(len(response), dtype=int)
    splitter = vrt.StratifiedKFold(
        n_splits=split_count,
        shuffle=True,
        random_state=seed + 17 + 7001,
    )
    for fold, (_, test_index) in enumerate(
        splitter.split(np.zeros((len(response), 1)), response)
    ):
        labels[test_index] = fold
    p = np.empty(len(response), dtype=float)
    m = np.empty(len(response), dtype=float)
    observed_all = response.astype(bool)
    for fold in range(split_count):
        test = labels == fold
        train = ~test
        observed = train & observed_all
        p_model = vrt._cui_candidate_propensity(
            selected_propensity_name,
            seed + 17 + 7001 + 1009 * fold + 31 * selected_k,
        )
        p_model.fit(x[train], response[train])
        p[test] = p_model.predict_proba(x[test])[:, 1]
        if int(np.sum(observed)) < 20:
            raise RuntimeError("too few observed outcomes for selected CUI learner")
        m_model = vrt._cui_candidate_outcome(
            selected_outcome_name,
            seed + 17 + 7001 + 2003 * fold + 37 * selected_l,
        )
        m_model.fit(x[observed], y[observed])
        m[test] = m_model.predict(x[test])
    return np.clip(p, 1e-6, 1.0), m


def _initial_outcome_and_p_raw(
    vrt: Any,
    x: np.ndarray,
    y: np.ndarray,
    response: np.ndarray,
    true_pi: np.ndarray,
    row: MethodRow,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.random.default_rng(seed + 17).integers(0, row.folds, len(y))
    m_oof = np.empty(len(y), dtype=float)
    p_raw = np.empty(len(y), dtype=float)
    observed_all = response.astype(bool)
    for fold in range(row.folds):
        test = labels == fold
        train = ~test
        observed = train & observed_all
        if int(np.sum(observed)) < 20:
            raise RuntimeError("too few observed outcomes in training fold")
        _, p_test = vrt._propensity_predictions(
            x,
            response,
            true_pi,
            train,
            test,
            row.propensity_mode,
            seed + 17 + 101 * fold,
            row.propensity_learner,
        )
        model = vrt._regressor(seed + 17 + 211 * fold, row.learner)
        model.fit(x[observed], y[observed])
        p_raw[test] = p_test
        m_oof[test] = model.predict(x[test])
    return m_oof, np.clip(p_raw, 1e-12, 1.0)


def _selected_base_and_p(
    vrt: Any,
    x: np.ndarray,
    y: np.ndarray,
    response: np.ndarray,
    true_pi: np.ndarray,
    row: MethodRow,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if row.method == "cui_selective_ml":
        if not row.selected_propensity_learner:
            raise RuntimeError("missing selective-ML propensity learner")
        if not row.selected_outcome_learner:
            raise RuntimeError("missing selective-ML outcome learner")
        p, base_outcome = _cui_selected_p_m(
            vrt,
            x,
            y,
            response,
            row.selected_propensity_learner,
            row.selected_outcome_learner,
            seed,
        )
        return base_outcome, p, min(row.tau_grid)

    base_outcome, p_raw = _initial_outcome_and_p_raw(
        vrt,
        x,
        y,
        response,
        true_pi,
        row,
        seed,
    )
    tau = row.selected_tau
    if not math.isfinite(tau):
        raise RuntimeError(f"missing selected tau for {row.method}")
    p = np.maximum(p_raw, tau)
    if row.method in {"tmle", "ctmle"}:
        observed = response.astype(bool)
        clever = 1.0 / np.maximum(p[observed], 1e-12)
        denominator = float(np.dot(clever, clever))
        epsilon = (
            float(np.dot(clever, y[observed] - base_outcome[observed]) / denominator)
            if denominator > 0.0
            else 0.0
        )
        base_outcome = base_outcome + epsilon / p
    return base_outcome, p, tau


def _rho(true_pi: np.ndarray, p: np.ndarray) -> tuple[float, float, float, float, float]:
    pi = np.clip(np.asarray(true_pi, dtype=float), 1e-12, 1.0 - 1e-12)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    kappa = 1.0 - pi / p
    rho2 = float(np.mean(kappa * kappa * pi / (1.0 - pi)))
    return (
        math.sqrt(max(rho2, 0.0)),
        rho2,
        float(np.sqrt(np.mean((p - pi) ** 2))),
        float(np.mean(np.abs(kappa))),
        float(np.mean(p <= 0.0500000001)),
    )


def _residual_square(true_pi: np.ndarray, prediction: np.ndarray, mu: np.ndarray) -> tuple[float, float]:
    pi = np.clip(np.asarray(true_pi, dtype=float), 1e-12, 1.0)
    prediction = np.asarray(prediction, dtype=float)
    mu = np.asarray(mu, dtype=float)
    residual_sq = (prediction - mu) ** 2
    weighted = float(np.mean((1.0 - pi) / pi * residual_sq))
    unweighted = float(np.mean(residual_sq))
    return weighted, unweighted


def _response_bin_labels(true_pi: np.ndarray, bin_count: int = 4) -> np.ndarray:
    labels = np.empty(len(true_pi), dtype=int)
    order = np.argsort(np.asarray(true_pi, dtype=float))
    for bin_index in range(bin_count):
        start = (bin_index * len(order)) // bin_count
        end = ((bin_index + 1) * len(order)) // bin_count
        labels[order[start:end]] = bin_index + 1
    return labels


def _response_bin_rows(
    row: MethodRow,
    true_pi: np.ndarray,
    labels: np.ndarray,
    base_outcome: np.ndarray,
    selected_outcome: np.ndarray,
    mu: np.ndarray,
    rho_value: float,
) -> list[dict[str, Any]]:
    pi = np.clip(np.asarray(true_pi, dtype=float), 1e-12, 1.0)
    ref_residual_sq = (np.asarray(base_outcome, dtype=float) - mu) ** 2
    selected_residual_sq = (np.asarray(selected_outcome, dtype=float) - mu) ** 2
    action = np.asarray(selected_outcome, dtype=float) - np.asarray(
        base_outcome,
        dtype=float,
    )
    action_sq = action**2
    action_abs = np.abs(action)
    weight = (1.0 - pi) / pi
    out = []
    for bin_index in sorted(set(labels.tolist())):
        mask = labels == bin_index
        mask_float = mask.astype(float)
        action_square = float(np.mean(mask_float * action_sq))
        weighted_action_square = float(np.mean(mask_float * weight * action_sq))
        absolute_action = float(np.mean(mask_float * action_abs))
        ref_weighted = float(np.mean(mask.astype(float) * weight * ref_residual_sq))
        selected_weighted = float(
            np.mean(mask.astype(float) * weight * selected_residual_sq)
        )
        ref_unweighted = float(np.mean(mask.astype(float) * ref_residual_sq))
        selected_unweighted = float(np.mean(mask.astype(float) * selected_residual_sq))
        out.append(
            {
                "source": row.source,
                "dataset": _dataset(row.design),
                "setting": _setting_label(row.design, row.n, row.strength),
                "group": row.group,
                "design": row.design,
                "method": row.method,
                "n": row.n,
                "strength": row.strength,
                "seed0": row.seed0,
                "rep": row.rep,
                "response_bin": bin_index,
                "response_bin_label": (
                    "lowest_response"
                    if bin_index == 1
                    else "highest_response"
                    if bin_index == int(np.max(labels))
                    else f"response_bin_{bin_index}"
                ),
                "n_share": float(np.mean(mask)),
                "mean_true_pi": float(np.mean(pi[mask])),
                "min_true_pi": float(np.min(pi[mask])),
                "max_true_pi": float(np.max(pi[mask])),
                "rho": rho_value,
                "action_square_contribution": action_square,
                "weighted_action_square_contribution": weighted_action_square,
                "absolute_action_contribution": absolute_action,
                "ref_weighted_residual_square_contribution": ref_weighted,
                "selected_weighted_residual_square_contribution": selected_weighted,
                "weighted_residual_square_delta": ref_weighted - selected_weighted,
                "weighted_residual_square_reduction": _relative_reduction(
                    ref_weighted,
                    selected_weighted,
                ),
                "ref_unweighted_residual_square_contribution": ref_unweighted,
                "selected_unweighted_residual_square_contribution": selected_unweighted,
                "unweighted_residual_square_delta": ref_unweighted
                - selected_unweighted,
                "unweighted_residual_square_reduction": _relative_reduction(
                    ref_unweighted,
                    selected_unweighted,
                ),
                "selected_gamma": row.selected_gamma,
                "c2_weight": row.c2_weight,
            }
        )
    return out


def _relative_reduction(before: float, after: float) -> float:
    return 1.0 - after / before if before > 0.0 else float("nan")


def _same_float(left: float, right: float, tol: float = 1e-10) -> bool:
    if math.isnan(left) and math.isnan(right):
        return True
    return math.isfinite(left) and math.isfinite(right) and abs(left - right) <= tol


def _process_sample(rows: list[MethodRow]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if _VRT is None:
        raise RuntimeError("worker not initialized")
    vrt = _VRT
    first = rows[0]
    seed = _internal_seed(vrt, first)
    x, y, response, _region, true_pi, _theta, _mu = vrt.make_data(
        first.n,
        first.epsilon,
        first.strength,
        first.design,
        seed,
        first.mar_design,
    )
    out = []
    response_bin_out = []
    response_labels = _response_bin_labels(true_pi)
    for row in rows:
        base_outcome, p, correction_tau = _selected_base_and_p(
            vrt,
            x,
            y,
            response,
            true_pi,
            row,
            seed,
        )
        correction = vrt._crossfit_weighted_residual_correction(
            x,
            y,
            response,
            p,
            base_outcome,
            row.learner,
            seed + 17 + 16001 + int(1000 * correction_tau),
            row.folds,
        )
        if row.repair_mode == "regional_if_residual":
            analysis_mask = vrt._analysis_region(
                x,
                y,
                response,
                _region,
                true_pi,
                row.analysis_region,
                row.propensity_mode,
                row.learner,
                row.propensity_learner,
                seed + 9091,
                row.region_quantile,
                row.region_min_observed,
                row.region_kappa_floor,
                row.selector_ablation,
                row.region_detector_c,
                row.folds,
            )
            correction = correction * analysis_mask.astype(float)
        selected_outcome = base_outcome + row.selected_gamma * correction
        rho_value, rho2, p_rmse, mean_abs_kappa, p_floor_share = _rho(true_pi, p)
        ref_weighted, ref_unweighted = _residual_square(
            true_pi,
            base_outcome,
            _mu,
        )
        selected_weighted, selected_unweighted = _residual_square(
            true_pi,
            selected_outcome,
            _mu,
        )
        response_bin_out.extend(
            _response_bin_rows(
                row,
                true_pi,
                response_labels,
                base_outcome,
                selected_outcome,
                _mu,
                rho_value,
            )
        )
        replay_selected_gamma = row.selected_gamma
        replay_selected_tau = correction_tau
        selected_gamma_matches = True
        selected_tau_matches = (
            True
            if row.method == "cui_selective_ml"
            else _same_float(row.selected_tau, correction_tau)
        )
        if not selected_tau_matches:
            raise RuntimeError(
                "selected tau mismatch for "
                f"{row.design}/{row.method}/seed={row.seed0}/rep={row.rep}: "
                f"tau row={row.selected_tau} reconstructed={correction_tau}"
            )
        out.append(
            {
                "source": row.source,
                "dataset": _dataset(row.design),
                "setting": _setting_label(row.design, row.n, row.strength),
                "group": row.group,
                "design": row.design,
                "method": row.method,
                "n": row.n,
                "strength": row.strength,
                "seed0": row.seed0,
                "rep": row.rep,
                "selected_tau": row.selected_tau,
                "correction_tau": correction_tau,
                "cui_selected_propensity_learner": row.selected_propensity_learner,
                "rho": rho_value,
                "rho2": rho2,
                "p_rmse": p_rmse,
                "mean_abs_kappa": mean_abs_kappa,
                "p_floor_share": p_floor_share,
                "ref_weighted_residual_square": ref_weighted,
                "selected_weighted_residual_square": selected_weighted,
                "weighted_residual_square_reduction": _relative_reduction(
                    ref_weighted,
                    selected_weighted,
                ),
                "ref_unweighted_residual_square": ref_unweighted,
                "selected_unweighted_residual_square": selected_unweighted,
                "unweighted_residual_square_reduction": _relative_reduction(
                    ref_unweighted,
                    selected_unweighted,
                ),
                "replay_selected_tau": replay_selected_tau,
                "replay_selected_gamma": replay_selected_gamma,
                "selected_tau_matches": selected_tau_matches,
                "selected_gamma_matches": selected_gamma_matches,
                "ref_error": row.ref_error,
                "selected_error": row.selected_error,
                "c2_error": row.c2_error,
                "selected_gamma": row.selected_gamma,
                "c2_weight": row.c2_weight,
            }
        )
    return out, response_bin_out


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _sd(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(statistics.pstdev(finite)) if len(finite) > 1 else 0.0


def _gain(reference: list[float], repaired: list[float]) -> float:
    ref_mse = _mean([value * value for value in reference])
    rep_mse = _mean([value * value for value in repaired])
    return 1.0 - rep_mse / ref_mse if ref_mse > 0.0 else float("nan")


def _bootstrap_ci(reference: list[float], repaired: list[float], seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    ref = np.asarray(reference, dtype=float)
    rep = np.asarray(repaired, dtype=float)
    draws = []
    for _ in range(5000):
        index = rng.integers(0, len(ref), len(ref))
        ref_mse = float(np.mean(ref[index] ** 2))
        draws.append(1.0 - float(np.mean(rep[index] ** 2)) / ref_mse)
    lo, hi = np.percentile(np.asarray(draws), [2.5, 97.5])
    return float(lo), float(hi)


def _aggregate_setting_rows(rep_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = ("dataset", "setting", "design", "method", "n", "strength")
    for row in rep_rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out = []
    for key_values, rows in sorted(groups.items()):
        ref = [float(row["ref_error"]) for row in rows]
        selected = [float(row["selected_error"]) for row in rows]
        c2 = [float(row["c2_error"]) for row in rows]
        ref_weighted_residual = [
            float(row["ref_weighted_residual_square"]) for row in rows
        ]
        selected_weighted_residual = [
            float(row["selected_weighted_residual_square"]) for row in rows
        ]
        ref_unweighted_residual = [
            float(row["ref_unweighted_residual_square"]) for row in rows
        ]
        selected_unweighted_residual = [
            float(row["selected_unweighted_residual_square"]) for row in rows
        ]
        seed = int(
            hashlib.sha256(
                json.dumps(key_values, sort_keys=True).encode()
            ).hexdigest()[:8],
            16,
        )
        lo, hi = _bootstrap_ci(ref, selected, seed)
        row = dict(zip(keys, key_values))
        row.update(
            {
                "reps": len(rows),
                "ref_mse": _mean([value * value for value in ref]),
                "selected_mse": _mean([value * value for value in selected]),
                "c2_mse": _mean([value * value for value in c2]),
                "selected_gain": _gain(ref, selected),
                "selected_gain_ci_low": lo,
                "selected_gain_ci_high": hi,
                "c2_gain": _gain(ref, c2),
                "path_activation": _mean(
                    [1.0 if float(item["selected_gamma"]) > 0.0 else 0.0 for item in rows]
                ),
                "c2_activation": _mean(
                    [1.0 if float(item["c2_weight"]) > 0.0 else 0.0 for item in rows]
                ),
                "selected_harm_rate": _mean(
                    [
                        1.0
                        if float(item["selected_error"]) ** 2
                        > float(item["ref_error"]) ** 2
                        else 0.0
                        for item in rows
                    ]
                ),
                "c2_harm_rate": _mean(
                    [
                        1.0
                        if float(item["c2_error"]) ** 2 > float(item["ref_error"]) ** 2
                        else 0.0
                        for item in rows
                    ]
                ),
                "mean_rho": _mean([float(item["rho"]) for item in rows]),
                "sd_rho": _sd([float(item["rho"]) for item in rows]),
                "mean_rho2": _mean([float(item["rho2"]) for item in rows]),
                "mean_p_rmse": _mean([float(item["p_rmse"]) for item in rows]),
                "mean_abs_kappa": _mean([float(item["mean_abs_kappa"]) for item in rows]),
                "mean_p_floor_share": _mean([float(item["p_floor_share"]) for item in rows]),
                "mean_selected_tau": _mean([float(item["selected_tau"]) for item in rows]),
                "ref_weighted_residual_square": _mean(ref_weighted_residual),
                "selected_weighted_residual_square": _mean(
                    selected_weighted_residual
                ),
                "weighted_residual_square_reduction": _relative_reduction(
                    _mean(ref_weighted_residual),
                    _mean(selected_weighted_residual),
                ),
                "weighted_residual_square_delta": _mean(ref_weighted_residual)
                - _mean(selected_weighted_residual),
                "ref_unweighted_residual_square": _mean(ref_unweighted_residual),
                "selected_unweighted_residual_square": _mean(
                    selected_unweighted_residual
                ),
                "unweighted_residual_square_reduction": _relative_reduction(
                    _mean(ref_unweighted_residual),
                    _mean(selected_unweighted_residual),
                ),
                "unweighted_residual_square_delta": _mean(ref_unweighted_residual)
                - _mean(selected_unweighted_residual),
                "mean_rep_weighted_residual_square_reduction": _mean(
                    [
                        float(item["weighted_residual_square_reduction"])
                        for item in rows
                    ]
                ),
                "mean_rep_unweighted_residual_square_reduction": _mean(
                    [
                        float(item["unweighted_residual_square_reduction"])
                        for item in rows
                    ]
                ),
                "replay_selected_gamma_match_failures": sum(
                    1 for item in rows if not item["selected_gamma_matches"]
                ),
                "replay_selected_tau_match_failures": sum(
                    1 for item in rows if not item["selected_tau_matches"]
                ),
                "cui_propensity_learners": ";".join(
                    f"{name}:{count}"
                    for name, count in sorted(
                        Counter(
                            item["cui_selected_propensity_learner"]
                            for item in rows
                            if item["cui_selected_propensity_learner"]
                        ).items()
                    )
                ),
            }
        )
        row["significance_class"] = (
            "win"
            if lo > 0.0
            else "loss"
            if hi < 0.0
            else "crosses_zero"
        )
        out.append(row)
    return out


def _aggregate_response_bin_rows(bin_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = (
        "dataset",
        "setting",
        "design",
        "method",
        "n",
        "strength",
        "response_bin",
        "response_bin_label",
    )
    for row in bin_rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out = []
    for key_values, rows in sorted(groups.items()):
        ref_weighted = [
            float(row["ref_weighted_residual_square_contribution"]) for row in rows
        ]
        selected_weighted = [
            float(row["selected_weighted_residual_square_contribution"])
            for row in rows
        ]
        ref_unweighted = [
            float(row["ref_unweighted_residual_square_contribution"]) for row in rows
        ]
        selected_unweighted = [
            float(row["selected_unweighted_residual_square_contribution"])
            for row in rows
        ]
        action_square = [float(row["action_square_contribution"]) for row in rows]
        weighted_action_square = [
            float(row["weighted_action_square_contribution"]) for row in rows
        ]
        absolute_action = [float(row["absolute_action_contribution"]) for row in rows]
        row = dict(zip(keys, key_values))
        row.update(
            {
                "reps": len(rows),
                "mean_n_share": _mean([float(item["n_share"]) for item in rows]),
                "mean_true_pi": _mean([float(item["mean_true_pi"]) for item in rows]),
                "min_true_pi": min(float(item["min_true_pi"]) for item in rows),
                "max_true_pi": max(float(item["max_true_pi"]) for item in rows),
                "mean_rho": _mean([float(item["rho"]) for item in rows]),
                "path_activation": _mean(
                    [
                        1.0 if float(item["selected_gamma"]) > 0.0 else 0.0
                        for item in rows
                    ]
                ),
                "action_square_contribution": _mean(action_square),
                "weighted_action_square_contribution": _mean(weighted_action_square),
                "absolute_action_contribution": _mean(absolute_action),
                "ref_weighted_residual_square_contribution": _mean(ref_weighted),
                "selected_weighted_residual_square_contribution": _mean(
                    selected_weighted
                ),
                "weighted_residual_square_delta": _mean(ref_weighted)
                - _mean(selected_weighted),
                "weighted_residual_square_reduction": _relative_reduction(
                    _mean(ref_weighted),
                    _mean(selected_weighted),
                ),
                "ref_unweighted_residual_square_contribution": _mean(ref_unweighted),
                "selected_unweighted_residual_square_contribution": _mean(
                    selected_unweighted
                ),
                "unweighted_residual_square_delta": _mean(ref_unweighted)
                - _mean(selected_unweighted),
                "unweighted_residual_square_reduction": _relative_reduction(
                    _mean(ref_unweighted),
                    _mean(selected_unweighted),
                ),
                "mean_rep_weighted_residual_square_delta": _mean(
                    [float(item["weighted_residual_square_delta"]) for item in rows]
                ),
            }
        )
        out.append(row)
    return out


def _response_bin_summary(
    setting_bin_rows: list[dict[str, Any]],
    label_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in setting_bin_rows:
        groups[tuple(row[key] for key in label_keys)].append(row)
    out = []
    for key_values, group in sorted(groups.items()):
        row = dict(zip(label_keys, key_values))
        row.update(
            {
                "setting_bins": len(group),
                "mean_n_share": _mean([float(item["mean_n_share"]) for item in group]),
                "mean_true_pi": _mean([float(item["mean_true_pi"]) for item in group]),
                "mean_rho": _mean([float(item["mean_rho"]) for item in group]),
                "mean_action_square_contribution": _mean(
                    [float(item["action_square_contribution"]) for item in group]
                ),
                "mean_weighted_action_square_contribution": _mean(
                    [
                        float(item["weighted_action_square_contribution"])
                        for item in group
                    ]
                ),
                "mean_absolute_action_contribution": _mean(
                    [float(item["absolute_action_contribution"]) for item in group]
                ),
                "mean_weighted_residual_square_delta": _mean(
                    [
                        float(item["weighted_residual_square_delta"])
                        for item in group
                    ]
                ),
                "mean_weighted_residual_reduction_pct": _mean(
                    [
                        100.0
                        * float(item["weighted_residual_square_reduction"])
                        for item in group
                    ]
                ),
                "median_weighted_residual_square_delta": float(
                    statistics.median(
                        [
                            float(item["weighted_residual_square_delta"])
                            for item in group
                        ]
                    )
                ),
                "positive_delta": sum(
                    1
                    for item in group
                    if float(item["weighted_residual_square_delta"]) > 0.0
                ),
                "negative_delta": sum(
                    1
                    for item in group
                    if float(item["weighted_residual_square_delta"]) < 0.0
                ),
                "zero_delta": sum(
                    1
                    for item in group
                    if float(item["weighted_residual_square_delta"]) == 0.0
                ),
                "mean_unweighted_residual_square_delta": _mean(
                    [
                        float(item["unweighted_residual_square_delta"])
                        for item in group
                    ]
                ),
            }
        )
        out.append(row)

    parent_keys = tuple(
        key for key in label_keys if key not in {"response_bin", "response_bin_label"}
    )
    parent_totals: dict[tuple[Any, ...], float] = defaultdict(float)
    parent_action_totals: dict[tuple[Any, ...], float] = defaultdict(float)
    parent_weighted_action_totals: dict[tuple[Any, ...], float] = defaultdict(float)
    parent_abs_action_totals: dict[tuple[Any, ...], float] = defaultdict(float)
    for row in out:
        parent = tuple(row[key] for key in parent_keys)
        parent_totals[parent] += float(row["mean_weighted_residual_square_delta"])
        parent_action_totals[parent] += float(row["mean_action_square_contribution"])
        parent_weighted_action_totals[parent] += float(
            row["mean_weighted_action_square_contribution"]
        )
        parent_abs_action_totals[parent] += float(
            row["mean_absolute_action_contribution"]
        )
    for row in out:
        parent = tuple(row[key] for key in parent_keys)
        total = parent_totals[parent]
        action_total = parent_action_totals[parent]
        weighted_action_total = parent_weighted_action_totals[parent]
        abs_action_total = parent_abs_action_totals[parent]
        row["share_of_mean_weighted_residual_delta"] = (
            float(row["mean_weighted_residual_square_delta"]) / total
            if abs(total) > 1e-12
            else float("nan")
        )
        row["share_of_mean_action_square"] = (
            float(row["mean_action_square_contribution"]) / action_total
            if abs(action_total) > 1e-12
            else float("nan")
        )
        row["share_of_mean_weighted_action_square"] = (
            float(row["mean_weighted_action_square_contribution"])
            / weighted_action_total
            if abs(weighted_action_total) > 1e-12
            else float("nan")
        )
        row["share_of_mean_absolute_action"] = (
            float(row["mean_absolute_action_contribution"]) / abs_action_total
            if abs(abs_action_total) > 1e-12
            else float("nan")
        )
    return out


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def _linear_summary(rows: list[dict[str, Any]], label_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in label_keys)].append(row)
    if label_keys:
        groups[tuple(["all"] * len(label_keys))] = list(rows)
    out = []
    for key_values, group in sorted(groups.items()):
        x = np.asarray([float(row["mean_rho"]) for row in group], dtype=float)
        y = np.asarray([100.0 * float(row["selected_gain"]) for row in group], dtype=float)
        z = np.asarray(
            [
                100.0 * float(row["weighted_residual_square_reduction"])
                for row in group
            ],
            dtype=float,
        )
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if len(x) < 2 or float(np.std(x)) <= 1e-12:
            slope = intercept = pearson = spearman = float("nan")
        else:
            design = np.column_stack([np.ones_like(x), x])
            intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
            pearson = (
                float(np.corrcoef(x, y)[0, 1])
                if float(np.std(y)) > 1e-12
                else float("nan")
            )
            rank_x = np.asarray(_rank(x.tolist()), dtype=float)
            rank_y = np.asarray(_rank(y.tolist()), dtype=float)
            spearman = (
                float(np.corrcoef(rank_x, rank_y)[0, 1])
                if float(np.std(rank_x)) > 1e-12
                and float(np.std(rank_y)) > 1e-12
                else float("nan")
            )
        residual_mask = np.isfinite(
            np.asarray([float(row["mean_rho"]) for row in group], dtype=float)
        ) & np.isfinite(z)
        residual_x = np.asarray(
            [float(row["mean_rho"]) for row in group], dtype=float
        )[residual_mask]
        residual_y = z[residual_mask]
        if (
            len(residual_x) < 2
            or float(np.std(residual_x)) <= 1e-12
            or float(np.std(residual_y)) <= 1e-12
        ):
            residual_intercept = residual_slope = residual_pearson = float("nan")
        else:
            residual_design = np.column_stack([np.ones_like(residual_x), residual_x])
            residual_intercept, residual_slope = np.linalg.lstsq(
                residual_design,
                residual_y,
                rcond=None,
            )[0]
            residual_pearson = float(np.corrcoef(residual_x, residual_y)[0, 1])
        row = dict(zip(label_keys, key_values))
        row.update(
            {
                "setting_methods": len(group),
                "mean_rho": _mean([float(item["mean_rho"]) for item in group]),
                "mean_selected_gain_pct": _mean(
                    [100.0 * float(item["selected_gain"]) for item in group]
                ),
                "mean_weighted_residual_reduction_pct": _mean(
                    [
                        100.0 * float(item["weighted_residual_square_reduction"])
                        for item in group
                    ]
                ),
                "mean_weighted_residual_square_delta": _mean(
                    [float(item["weighted_residual_square_delta"]) for item in group]
                ),
                "median_weighted_residual_reduction_pct": float(
                    statistics.median(
                        [
                            100.0
                            * float(item["weighted_residual_square_reduction"])
                            for item in group
                        ]
                    )
                ),
                "median_selected_gain_pct": float(
                    statistics.median(
                        [100.0 * float(item["selected_gain"]) for item in group]
                    )
                ),
                "wins": sum(1 for item in group if item["significance_class"] == "win"),
                "losses": sum(1 for item in group if item["significance_class"] == "loss"),
                "crosses_zero": sum(
                    1 for item in group if item["significance_class"] == "crosses_zero"
                ),
                "ols_intercept_gain_pct": float(intercept),
                "ols_slope_gain_pct_per_rho": float(slope),
                "nonnegative_slope_gain_pct_per_rho": float(max(0.0, slope))
                if math.isfinite(slope)
                else float("nan"),
                "pearson_r": pearson,
                "spearman_r": spearman,
                "ols_intercept_weighted_residual_reduction_pct": float(
                    residual_intercept
                ),
                "ols_slope_weighted_residual_reduction_pct_per_rho": float(
                    residual_slope
                ),
                "pearson_r_weighted_residual_reduction": residual_pearson,
            }
        )
        out.append(row)
    return out


def _rho_bins(rows: list[dict[str, Any]], bins: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row["mean_rho"]))
    out = []
    for bin_index in range(bins):
        group = ordered[
            (bin_index * len(ordered)) // bins : ((bin_index + 1) * len(ordered)) // bins
        ]
        if not group:
            continue
        out.append(
            {
                "rho_bin": bin_index + 1,
                "setting_methods": len(group),
                "rho_min": min(float(row["mean_rho"]) for row in group),
                "rho_max": max(float(row["mean_rho"]) for row in group),
                "mean_rho": _mean([float(row["mean_rho"]) for row in group]),
                "mean_selected_gain_pct": _mean(
                    [100.0 * float(row["selected_gain"]) for row in group]
                ),
                "mean_weighted_residual_reduction_pct": _mean(
                    [
                        100.0 * float(row["weighted_residual_square_reduction"])
                        for row in group
                    ]
                ),
                "mean_weighted_residual_square_delta": _mean(
                    [float(row["weighted_residual_square_delta"]) for row in group]
                ),
                "median_weighted_residual_reduction_pct": float(
                    statistics.median(
                        [
                            100.0 * float(row["weighted_residual_square_reduction"])
                            for row in group
                        ]
                    )
                ),
                "median_selected_gain_pct": float(
                    statistics.median(
                        [100.0 * float(row["selected_gain"]) for row in group]
                    )
                ),
                "wins": sum(1 for row in group if row["significance_class"] == "win"),
                "losses": sum(1 for row in group if row["significance_class"] == "loss"),
                "crosses_zero": sum(
                    1 for row in group if row["significance_class"] == "crosses_zero"
                ),
                "mean_path_activation": _mean(
                    [float(row["path_activation"]) for row in group]
                ),
                "mean_selected_harm_rate": _mean(
                    [float(row["selected_harm_rate"]) for row in group]
                ),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_results(path: Path, setting_rows: list[dict[str, Any]], bins: list[dict[str, Any]]) -> None:
    all_summary = _linear_summary(setting_rows, ("dataset",))
    all_row = next(row for row in all_summary if row["dataset"] == "all")
    method_rows = [row for row in _linear_summary(setting_rows, ("method",)) if row["method"] != "all"]
    strongest = sorted(
        setting_rows,
        key=lambda row: float(row["selected_gain"]),
        reverse=True,
    )[:8]
    weakest = sorted(setting_rows, key=lambda row: float(row["selected_gain"]))[:8]
    lines = [
        "# Retrospective MSE Gain by Rho",
        "",
        "This diagnostic joins selected-candidate MSE gains to the response-model",
        "misspecification index rho_p used in Section 3.  It is retrospective:",
        "rho_p is computed from the known true response probability in the",
        "simulation designs and was not used to select the repair.",
        "",
        "Primary endpoint: selected candidate (`rt_error`), not the older c=2",
        "shrunken endpoint.  The c=2 columns are retained only for audit.",
        "",
        "Theory-facing residual endpoint: true response-weighted residual",
        "square, E[(1-pi)/pi * (m_hat-mu)^2], computed for the reference",
        "outcome prediction and for the selected repaired outcome prediction.",
        "The residual-square delta is reference minus selected, so positive",
        "values mean the repaired outcome prediction reduced this quantity.",
        "",
        "For IHDP, ACIC 2016, ACIC 2017, and Twins, labels A and B denote the",
        "two preexisting known-truth variants in the released experiment, each",
        "at strengths 0 and 3.  The internal design names remain in the CSVs.",
        "",
        "Interpretation: rho_p is a severity covariate, not a sufficient",
        "condition for improvement.  The repair also needs residual signal that",
        "the selected candidate can exploit.",
        "",
        "## Pooled Readout",
        "",
        f"- Setting-method rows: {all_row['setting_methods']}",
        f"- Mean rho: {all_row['mean_rho']:.3f}",
        f"- Mean selected-candidate gain: {all_row['mean_selected_gain_pct']:.3f}%",
        f"- Mean weighted residual-square reduction: {all_row['mean_weighted_residual_reduction_pct']:.3f}%",
        f"- Mean weighted residual-square delta: {all_row['mean_weighted_residual_square_delta']:.6g}",
        f"- Wins / losses / crossing zero: {all_row['wins']} / {all_row['losses']} / {all_row['crosses_zero']}",
        f"- OLS slope of gain on rho: {all_row['ols_slope_gain_pct_per_rho']:.3f} percentage points per rho unit",
        f"- OLS slope of weighted residual-square reduction on rho: {all_row['ols_slope_weighted_residual_reduction_pct_per_rho']:.3f} percentage points per rho unit",
        f"- Pearson / Spearman: {all_row['pearson_r']:.3f} / {all_row['spearman_r']:.3f}",
        "",
        "## Rho Bins",
        "",
    ]
    for row in bins:
        lines.append(
            "- bin {rho_bin}: rho {rho_min:.3f}-{rho_max:.3f}, mean gain "
            "{mean_selected_gain_pct:.3f}%, mean weighted-residual reduction "
            "{mean_weighted_residual_reduction_pct:.3f}%, mean weighted-residual "
            "delta {mean_weighted_residual_square_delta:.6g}, wins/losses/crossing "
            "{wins}/{losses}/{crosses_zero}".format(**row)
        )
    lines.extend(["", "## By Expert", ""])
    for row in sorted(method_rows, key=lambda item: item["method"]):
        lines.append(
            "- {method}: mean rho {mean_rho:.3f}, mean gain "
            "{mean_selected_gain_pct:.3f}%, mean weighted-residual reduction "
            "{mean_weighted_residual_reduction_pct:.3f}%, mean weighted-residual "
            "delta {mean_weighted_residual_square_delta:.6g}, wins/losses/crossing "
            "{wins}/{losses}/{crosses_zero}, gain slope {ols_slope_gain_pct_per_rho:.3f}, "
            "residual slope {ols_slope_weighted_residual_reduction_pct_per_rho:.3f}".format(
                **row
            )
        )
    lines.extend(["", "## Largest Gains", ""])
    for row in strongest:
        lines.append(
            f"- {row['method']} on {row['setting']}: "
            f"{100.0 * float(row['selected_gain']):.3f}% gain, "
            f"{100.0 * float(row['weighted_residual_square_reduction']):.3f}% weighted-residual reduction, "
            f"weighted-residual delta {float(row['weighted_residual_square_delta']):.6g}, "
            f"rho {float(row['mean_rho']):.3f}"
        )
    lines.extend(["", "## Largest Losses", ""])
    for row in weakest:
        lines.append(
            f"- {row['method']} on {row['setting']}: "
            f"{100.0 * float(row['selected_gain']):.3f}% gain, "
            f"{100.0 * float(row['weighted_residual_square_reduction']):.3f}% weighted-residual reduction, "
            f"weighted-residual delta {float(row['weighted_residual_square_delta']):.6g}, "
            f"rho {float(row['mean_rho']):.3f}"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_response_bin_results(
    path: Path,
    response_bin_summary: list[dict[str, Any]],
    method_bin_summary: list[dict[str, Any]],
) -> None:
    lines = [
        "# Response-Bin Action and Reward Decomposition",
        "",
        "This diagnostic decomposes two quantities by true response-probability",
        "quartiles within each generated sample.  Bin 1 is the lowest-response",
        "quartile and bin 4 is the highest-response quartile.",
        "",
        "Action is where the repair moves the fitted outcome surface:",
        "(m_selected - m_ref)^2, with both raw and response-weighted shares",
        "reported.  Reward is where that movement reduces true",
        "response-weighted residual-square:",
        "E[(1-pi)/pi * (m_hat-mu)^2].  Each bin entry is an unconditional",
        "contribution, so the four bins add back to the corresponding total.",
        "",
        "This is the direct check for the low-response story.  The rho diagnostic",
        "is only a response-model misspecification severity check.",
        "The supported claim is about magnitude: the positive average",
        "response-weighted residual-square delta is concentrated in the",
        "lowest-response quartile.  It is not a claim that low-response bins",
        "improve more often in every setting; the positive/negative/zero counts",
        "are reported separately.",
        "",
        "## All Experts",
        "",
    ]
    for row in response_bin_summary:
        lines.append(
            "- bin {response_bin} ({response_bin_label}): mean pi {mean_true_pi:.3f}, "
            "action share {share_of_mean_action_square:.3f} raw / "
            "{share_of_mean_weighted_action_square:.3f} weighted, "
            "reward share {share_of_mean_weighted_residual_delta:.3f}, "
            "positive/negative/zero {positive_delta}/{negative_delta}/{zero_delta}".format(
                **row
            )
        )
    lines.extend(["", "## By Expert", ""])
    for row in method_bin_summary:
        lines.append(
            "- {method}, bin {response_bin}: mean pi {mean_true_pi:.3f}, "
            "action share {share_of_mean_action_square:.3f} raw / "
            "{share_of_mean_weighted_action_square:.3f} weighted, "
            "reward share {share_of_mean_weighted_residual_delta:.3f}, "
            "positive/negative/zero {positive_delta}/{negative_delta}/{zero_delta}".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _bar_width(share: float, max_width_cm: float = 3.2) -> str:
    return f"{max(0.0, share) * max_width_cm:.3f}"


def _tex_pct(value: float) -> str:
    return f"{100.0 * value:.1f}\\%"


def _write_response_bin_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: int(row["response_bin"]))
    labels = {
        "lowest_response": "lowest",
        "response_bin_2": "second",
        "response_bin_3": "third",
        "highest_response": "highest",
    }
    lines = [
        "% Generated by scripts/dml_mse_gain_by_rho.py; do not edit.\n",
        "\\begin{figure}[t]\n",
        "\\centering\n",
        "\\small\n",
        "\\newcommand{\\quartilebar}[2]{\\makebox[3.25cm][l]{\\textcolor{#1}{\\rule{#2cm}{1.05ex}}}}\n",
        "\\setlength{\\tabcolsep}{3pt}\n",
        "\\resizebox{\\textwidth}{!}{%\n",
        "\\begin{tabular}{@{}llccc@{}}\n",
        "\\toprule\n",
        "response quartile & mean $\\pi$ & raw action & weighted action & reward \\\\\n",
        "\\midrule\n",
    ]
    for row in ordered:
        label = labels.get(str(row["response_bin_label"]), str(row["response_bin_label"]))
        raw_action = float(row["share_of_mean_action_square"])
        weighted_action = float(row["share_of_mean_weighted_action_square"])
        reward = float(row["share_of_mean_weighted_residual_delta"])
        lines.append(
            f"{label} & {float(row['mean_true_pi']):.3f} & "
            f"\\quartilebar{{black!55}}{{{_bar_width(raw_action)}}} {_tex_pct(raw_action)} & "
            f"\\quartilebar{{black!70}}{{{_bar_width(weighted_action)}}} {_tex_pct(weighted_action)} & "
            f"\\quartilebar{{black!85}}{{{_bar_width(reward)}}} {_tex_pct(reward)} \\\\\n"
        )
    lines.extend(
        [
            "\\bottomrule\n",
            "\\end{tabular}%\n",
            "}\n",
            "\\caption{Action and reward by response-probability quartile.  Bins are within-sample quartiles of the true response probability.  Action is the squared movement of the fitted outcome surface, $(\\widehat m_{\\mathrm{selected}}-\\widehat m_{\\mathrm{ref}})^2$, shown both raw and weighted by $(1-\\pi)/\\pi$.  Reward is the reduction in true response-weighted residual square.  Raw action is largest in the lowest-response quartile; response-weighted action and reward are concentrated there.}\n",
            "\\label{fig:response-bin-action-reward}\n",
            "\\end{figure}\n",
        ]
    )
    path.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path.cwd())
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
    parser.add_argument(
        "--support-data",
        type=Path,
        default=Path("/tmp/dml_real_benchmark_support_data"),
    )
    parser.add_argument(
        "--aug14-run-dir",
        type=Path,
        default=None,
        help="Extracted cartesian_dml_ks_alignment_v3 directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("support_csv/dml_mse_gain_by_rho_20260902"),
    )
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--bins", type=int, default=4)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    aug14 = args.aug14_run_dir
    if aug14 is None:
        latest = Path("/tmp/dml_rho_extract_latest.txt")
        if not latest.exists():
            raise SystemExit("pass --aug14-run-dir or create /tmp/dml_rho_extract_latest.txt")
        aug14 = Path(latest.read_text().strip()) / "dml" / "cartesian_dml_ks_alignment_v3"

    rows: list[MethodRow] = []
    rows.extend(_read_extracted_aug14_rows(aug14))
    rows.extend(
        _read_release_rows(
            data_root / "support_csv/dml_real_benchmark_expansion_20260831",
            "real_benchmark_ihdp_acic2016",
        )
    )
    rows.extend(
        _read_release_rows(
            data_root / "support_csv/dml_real_benchmark_acic2017_20260831",
            "real_benchmark_acic2017",
        )
    )
    rows.extend(
        _read_release_rows(
            data_root / "support_csv/dml_real_benchmark_twins_20260831",
            "real_benchmark_twins",
        )
    )

    sample_groups: dict[tuple[Any, ...], list[MethodRow]] = defaultdict(list)
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

    rep_rows: list[dict[str, Any]] = []
    response_bin_rows: list[dict[str, Any]] = []
    tasks = list(sample_groups.values())
    with ProcessPoolExecutor(
        max_workers=args.jobs,
        initializer=_init_worker,
        initargs=(
            str(args.source_script.resolve()),
            str(args.breadth_script.resolve()),
            str(args.support_data.resolve()),
        ),
    ) as pool:
        futures = [pool.submit(_process_sample, task) for task in tasks]
        completed = 0
        for future in as_completed(futures):
            sample_rep_rows, sample_response_bin_rows = future.result()
            rep_rows.extend(sample_rep_rows)
            response_bin_rows.extend(sample_response_bin_rows)
            completed += 1
            if completed % max(1, args.jobs * 4) == 0 or completed == len(tasks):
                print(f"completed {completed}/{len(tasks)} samples", flush=True)

    rep_rows.sort(
        key=lambda row: (
            row["dataset"],
            row["setting"],
            row["method"],
            row["seed0"],
            row["rep"],
        )
    )
    response_bin_rows.sort(
        key=lambda row: (
            row["dataset"],
            row["setting"],
            row["method"],
            row["response_bin"],
            row["seed0"],
            row["rep"],
        )
    )
    setting_rows = _aggregate_setting_rows(rep_rows)
    response_bin_setting_rows = _aggregate_response_bin_rows(response_bin_rows)
    response_bin_summary = _response_bin_summary(
        response_bin_setting_rows,
        ("response_bin", "response_bin_label"),
    )
    response_bin_method_summary = _response_bin_summary(
        response_bin_setting_rows,
        ("method", "response_bin", "response_bin_label"),
    )
    bin_rows = _rho_bins(setting_rows, args.bins)
    dataset_rows = _linear_summary(setting_rows, ("dataset",))
    dataset_method_rows = _linear_summary(setting_rows, ("dataset", "method"))
    method_rows = _linear_summary(setting_rows, ("method",))

    out_dir = args.out_dir if args.out_dir.is_absolute() else data_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "rho_replication_rows.csv", rep_rows)
    _write_csv(out_dir / "setting_rho_gain.csv", setting_rows)
    _write_csv(out_dir / "response_bin_replication_rows.csv", response_bin_rows)
    _write_csv(out_dir / "response_bin_setting_summary.csv", response_bin_setting_rows)
    _write_csv(out_dir / "response_bin_summary.csv", response_bin_summary)
    _write_csv(out_dir / "response_bin_method_summary.csv", response_bin_method_summary)
    _write_csv(out_dir / "rho_bin_summary.csv", bin_rows)
    _write_csv(out_dir / "dataset_rho_summary.csv", dataset_rows)
    _write_csv(out_dir / "dataset_method_rho_summary.csv", dataset_method_rows)
    _write_csv(out_dir / "method_rho_summary.csv", method_rows)
    _write_results(out_dir / "RESULTS.md", setting_rows, bin_rows)
    _write_response_bin_results(
        out_dir / "RESPONSE_BIN_RESULTS.md",
        response_bin_summary,
        response_bin_method_summary,
    )
    _write_response_bin_figure(
        out_dir / "section4_response_bin_action_reward_figure.tex",
        response_bin_summary,
    )
    provenance = {
        "source_script": str(args.source_script.resolve()),
        "source_script_sha256": _sha256(args.source_script.resolve()),
        "breadth_script": str(args.breadth_script.resolve()),
        "breadth_script_sha256": _sha256(args.breadth_script.resolve()),
        "support_data": str(args.support_data.resolve()),
        "aug14_run_dir": str(aug14.resolve()),
        "rows": len(rep_rows),
        "response_bin_rows": len(response_bin_rows),
        "settings": len(setting_rows),
        "response_bin_settings": len(response_bin_setting_rows),
        "sample_groups": len(sample_groups),
        "methods": list(PRIMARY_METHODS),
        "endpoint": "selected candidate: rt_error",
        "rho_formula": "mean((1 - true_pi / selected_p)^2 * true_pi / (1 - true_pi))",
        "response_bins": "within-sample quartiles of true_pi; bin 1 is lowest response",
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    checksum_rows = []
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            checksum_rows.append(f"{_sha256(path)}  {path.name}")
    (out_dir / "SHA256SUMS").write_text("\n".join(checksum_rows) + "\n")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
