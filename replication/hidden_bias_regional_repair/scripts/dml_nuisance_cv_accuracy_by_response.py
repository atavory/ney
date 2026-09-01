#!/usr/bin/env python3
"""Cross-fitted nuisance accuracy by response-propensity bin.

This diagnostic regenerates the Section 4 benchmark settings and compares the
cross-fitted response model p_hat with the cross-fitted outcome model m_hat as
response probability varies.  Cross-fitting uses folds stratified by true pi
so each held-out fold has comparable response-propensity support.  It writes
replication-level bin summaries plus aggregates by setting and dataset.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_VRT = None


@dataclass(frozen=True)
class Setting:
    dataset: str
    design: str
    setting: str
    n: int
    epsilon: float
    strength: float
    mar_design: str = "box"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _init_worker(
    source_script: str,
    breadth_script: str,
    support_data: str,
) -> None:
    global _VRT
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ["USHMOO_SUPPORT_DATA"] = support_data
    os.environ["DML_SUPPORT_DATA"] = support_data
    script_dir = str(Path(source_script).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    vrt = _load_module("dml_validated_reference_transfer", Path(source_script))
    breadth = _load_module("dml_section4_breadth_experiments", Path(breadth_script))
    breadth._install_adapter(vrt)
    _VRT = vrt


def _design_seed(vrt: Any, setting: Setting, rep: int, seed0: int) -> int:
    return (
        seed0
        + setting.n * 1009
        + int(setting.epsilon * 1000) * 100003
        + vrt._design_strength_code(setting.strength) * 10007
        + rep * 37
        + vrt._design_seed_offset(setting.design)
        + vrt._mar_seed_offset(setting.mar_design)
    )


def _rank_bins(score: np.ndarray, bins: int) -> np.ndarray:
    order = np.argsort(score, kind="mergesort")
    labels = np.empty(len(score), dtype=int)
    labels[order] = np.minimum((np.arange(len(score)) * bins) // len(score), bins - 1)
    return labels


def _pi_stratified_fold_ids(
    true_pi: np.ndarray,
    folds: int,
    seed: int,
    strata: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    strata_labels = _rank_bins(true_pi, strata)
    fold_ids = np.empty(len(true_pi), dtype=int)
    for stratum in range(strata):
        indices = np.flatnonzero(strata_labels == stratum)
        rng.shuffle(indices)
        fold_ids[indices] = np.arange(len(indices)) % folds
    return fold_ids


def _crossfit_response_score(
    vrt: Any,
    x: np.ndarray,
    response: np.ndarray,
    learner: str,
    fold_ids: np.ndarray,
    seed: int,
) -> np.ndarray:
    score = np.empty(len(response), dtype=float)
    for fold in range(int(fold_ids.max()) + 1):
        test = fold_ids == fold
        train = ~test
        if len(np.unique(response[train])) < 2:
            score[test] = float(np.mean(response[train]))
            continue
        model = vrt._classifier(seed + 101 * (fold + 1), learner)
        model.fit(x[train], response[train].astype(bool))
        score[test] = model.predict_proba(x[test])[:, 1]
    return np.clip(score, 1e-6, 1.0 - 1e-6)


def _crossfit_outcome_prediction(
    vrt: Any,
    x: np.ndarray,
    y: np.ndarray,
    response: np.ndarray,
    learner: str,
    fold_ids: np.ndarray,
    seed: int,
) -> np.ndarray:
    prediction = np.empty(len(y), dtype=float)
    observed_all = response.astype(bool)
    global_mean = float(np.mean(y[observed_all])) if np.any(observed_all) else 0.0
    for fold in range(int(fold_ids.max()) + 1):
        test = fold_ids == fold
        observed = (fold_ids != fold) & observed_all
        if observed.sum() < 20:
            prediction[test] = global_mean
            continue
        model = vrt._regressor(seed + 211 * fold, learner)
        model.fit(x[observed], y[observed])
        prediction[test] = model.predict(x[test])
    return prediction


def _safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def _safe_sd(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.std()) if len(values) else float("nan")


def _safe_rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values * values))) if len(values) else float("nan")


def _skill(mse: float, null_mse: float) -> float:
    if not math.isfinite(mse) or not math.isfinite(null_mse) or null_mse <= 0.0:
        return float("nan")
    return 1.0 - mse / null_mse


def _row_for_bin(
    setting: Setting,
    rep: int,
    seed: int,
    bin_by: str,
    bin_index: int,
    bins: int,
    mask: np.ndarray,
    true_pi: np.ndarray,
    pi_hat: np.ndarray,
    response: np.ndarray,
    y: np.ndarray,
    mu: np.ndarray,
    m_hat: np.ndarray,
) -> dict[str, Any]:
    observed = mask & response.astype(bool)
    p_label_mse = _safe_mean((pi_hat[mask] - response[mask]) ** 2)
    response_rate = _safe_mean(response[mask])
    p_null_mse = _safe_mean((response[mask] - response_rate) ** 2)
    p_true_rmse = _safe_rmse(pi_hat[mask] - true_pi[mask])
    p_label_rmse = _safe_rmse(pi_hat[mask] - response[mask])
    m_true_rmse = _safe_rmse(m_hat[mask] - mu[mask])
    m_observed_mse = _safe_mean((m_hat[observed] - y[observed]) ** 2)
    observed_y_mean = _safe_mean(y[observed])
    m_null_mse = _safe_mean((y[observed] - observed_y_mean) ** 2)
    m_observed_rmse = _safe_rmse(m_hat[observed] - y[observed])
    pi_sd = max(float(np.std(true_pi)), 1e-12)
    mu_sd = max(float(np.std(mu)), 1e-12)
    return {
        "dataset": setting.dataset,
        "design": setting.design,
        "setting": setting.setting,
        "n": setting.n,
        "epsilon": setting.epsilon,
        "strength": setting.strength,
        "rep": rep + 1,
        "seed": seed,
        "bin_by": bin_by,
        "bin": bin_index + 1,
        "bins": bins,
        "units": int(mask.sum()),
        "responders": int(observed.sum()),
        "response_rate": response_rate,
        "mean_true_pi": _safe_mean(true_pi[mask]),
        "mean_pi_hat": _safe_mean(pi_hat[mask]),
        "p_label_rmse": p_label_rmse,
        "p_label_skill": _skill(p_label_mse, p_null_mse),
        "p_true_rmse": p_true_rmse,
        "p_true_nrmse": p_true_rmse / pi_sd,
        "p_calibration_error": _safe_mean(pi_hat[mask] - response[mask]),
        "m_observed_rmse": m_observed_rmse,
        "m_observed_skill": _skill(m_observed_mse, m_null_mse),
        "m_true_rmse": m_true_rmse,
        "m_true_nrmse": m_true_rmse / mu_sd,
        "m_observed_label_fraction": float(observed.sum()) / float(mask.sum()),
    }


def _run_setting_rep(args_tuple: tuple[Setting, int, int, str, str, int, int]) -> list[dict[str, Any]]:
    setting, rep, seed0, learner, propensity_learner, folds, bins = args_tuple
    if _VRT is None:
        raise RuntimeError("worker was not initialized")
    vrt = _VRT
    seed = _design_seed(vrt, setting, rep, seed0)
    x, y, response, _region, true_pi, _theta, mu = vrt.make_data(
        setting.n,
        setting.epsilon,
        setting.strength,
        setting.design,
        seed,
        setting.mar_design,
    )
    fold_ids = _pi_stratified_fold_ids(
        true_pi,
        folds,
        seed + 9091,
        max(bins, 10),
    )
    pi_hat = _crossfit_response_score(
        vrt,
        x,
        response,
        propensity_learner,
        fold_ids,
        seed + 9091,
    )
    m_hat = _crossfit_outcome_prediction(
        vrt,
        x,
        y,
        response,
        learner,
        fold_ids,
        seed + 17,
    )
    rows: list[dict[str, Any]] = []
    for bin_by, score in (("pi_hat", pi_hat), ("true_pi", true_pi)):
        labels = _rank_bins(np.asarray(score, dtype=float), bins)
        for bin_index in range(bins):
            mask = labels == bin_index
            rows.append(
                _row_for_bin(
                    setting,
                    rep,
                    seed,
                    bin_by,
                    bin_index,
                    bins,
                    mask,
                    true_pi,
                    pi_hat,
                    response,
                    y,
                    mu,
                    m_hat,
                )
            )
    return rows


def _settings() -> list[Setting]:
    settings: list[Setting] = []
    ks_labels = {
        "cc": "outcome correct, response correct",
        "ci": "outcome correct, response incorrect",
        "ic": "outcome incorrect, response correct",
        "ii": "outcome incorrect, response incorrect",
    }
    for suffix, label in ks_labels.items():
        for n in (200, 1000):
            settings.append(
                Setting(
                    dataset="Kang-Schafer",
                    design=f"kang_schafer_{suffix}",
                    setting=f"{label}; n={n}",
                    n=n,
                    epsilon=0.05,
                    strength=0.0,
                )
            )
    for dataset, prefix in (
        ("IHDP", "ihdp"),
        ("ACIC 2016", "acic2016"),
        ("ACIC 2017", "acic2017"),
        ("Twins", "twins"),
    ):
        for variant, label in (
            ("semisynth", "low-response outcome shift"),
            ("misaligned", "off-response outcome shift"),
        ):
            for strength in (0.0, 3.0):
                settings.append(
                    Setting(
                        dataset=dataset,
                        design=f"{prefix}_{variant}",
                        setting=f"{label}; strength={strength:g}",
                        n=3000,
                        epsilon=0.05,
                        strength=strength,
                    )
                )
    return settings


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    metric_names = [
        "units",
        "responders",
        "response_rate",
        "mean_true_pi",
        "mean_pi_hat",
        "p_label_rmse",
        "p_label_skill",
        "p_true_rmse",
        "p_true_nrmse",
        "p_calibration_error",
        "m_observed_rmse",
        "m_observed_skill",
        "m_true_rmse",
        "m_true_nrmse",
        "m_observed_label_fraction",
    ]
    out = []
    for key_values, group in sorted(groups.items()):
        row = dict(zip(keys, key_values))
        row["rep_bins"] = len(group)
        for name in metric_names:
            values = np.asarray([float(item[name]) for item in group], dtype=float)
            row[f"mean_{name}"] = _safe_mean(values)
            row[f"sd_{name}"] = _safe_sd(values)
        out.append(row)
    return out


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2 or float(np.std(x[mask])) <= 1e-12:
        return float("nan")
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def _dependence_from_bin_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("dataset", "design", "setting", "n", "strength", "bin_by")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out = []
    for key_values, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: int(row["bin"]))
        low = ordered[0]
        high = ordered[-1]
        x = np.asarray([float(row["mean_mean_true_pi"]) for row in ordered])
        p = np.asarray([float(row["mean_p_true_nrmse"]) for row in ordered])
        m = np.asarray([float(row["mean_m_true_nrmse"]) for row in ordered])
        p_skill = np.asarray([float(row["mean_p_label_skill"]) for row in ordered])
        m_skill = np.asarray([float(row["mean_m_observed_skill"]) for row in ordered])
        p_low = float(low["mean_p_true_nrmse"])
        p_high = float(high["mean_p_true_nrmse"])
        m_low = float(low["mean_m_true_nrmse"])
        m_high = float(high["mean_m_true_nrmse"])
        row = dict(zip(keys, key_values))
        row.update(
            {
                "low_mean_true_pi": float(low["mean_mean_true_pi"]),
                "high_mean_true_pi": float(high["mean_mean_true_pi"]),
                "low_response_rate": float(low["mean_response_rate"]),
                "high_response_rate": float(high["mean_response_rate"]),
                "low_p_true_nrmse": p_low,
                "high_p_true_nrmse": p_high,
                "low_m_true_nrmse": m_low,
                "high_m_true_nrmse": m_high,
                "low_high_ratio_p_true_nrmse": (
                    p_low / p_high if p_high > 0.0 else float("nan")
                ),
                "low_high_ratio_m_true_nrmse": (
                    m_low / m_high if m_high > 0.0 else float("nan")
                ),
                "low_response_excess_p_true_nrmse": p_low - p_high,
                "low_response_excess_m_true_nrmse": m_low - m_high,
                "low_response_excess_gap_m_minus_p": (m_low - m_high)
                - (p_low - p_high),
                "slope_p_true_nrmse_vs_true_pi": _linear_slope(x, p),
                "slope_m_true_nrmse_vs_true_pi": _linear_slope(x, m),
                "slope_gap_m_minus_p": _linear_slope(x, m) - _linear_slope(x, p),
                "slope_p_label_skill_vs_true_pi": _linear_slope(x, p_skill),
                "slope_m_observed_skill_vs_true_pi": _linear_slope(x, m_skill),
            }
        )
        out.append(row)
    return out


def _aggregate_dependence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("dataset", "bin_by")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    metrics = [
        "low_mean_true_pi",
        "high_mean_true_pi",
        "low_response_rate",
        "high_response_rate",
        "low_p_true_nrmse",
        "high_p_true_nrmse",
        "low_m_true_nrmse",
        "high_m_true_nrmse",
        "low_high_ratio_p_true_nrmse",
        "low_high_ratio_m_true_nrmse",
        "low_response_excess_p_true_nrmse",
        "low_response_excess_m_true_nrmse",
        "low_response_excess_gap_m_minus_p",
        "slope_p_true_nrmse_vs_true_pi",
        "slope_m_true_nrmse_vs_true_pi",
        "slope_gap_m_minus_p",
        "slope_p_label_skill_vs_true_pi",
        "slope_m_observed_skill_vs_true_pi",
    ]
    out = []
    for key_values, group in sorted(groups.items()):
        row = dict(zip(keys, key_values))
        row["settings"] = len(group)
        for metric in metrics:
            values = np.asarray([float(item[metric]) for item in group], dtype=float)
            row[f"mean_{metric}"] = _safe_mean(values)
        out.append(row)
    return out


def _normalized_error_fit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["bin_by"] != "true_pi":
            continue
        groups[
            (
                row["dataset"],
                row["design"],
                row["setting"],
                row["n"],
                row["strength"],
            )
        ].append(row)
    out = []
    for key_values, group in sorted(groups.items()):
        pi = np.asarray([float(row["mean_true_pi"]) for row in group], dtype=float)
        units = np.asarray([float(row["units"]) for row in group], dtype=float)
        p_mse = np.asarray([float(row["p_true_rmse"]) ** 2 for row in group])
        m_mse = np.asarray([float(row["m_true_rmse"]) ** 2 for row in group])
        p_mean = float(np.average(p_mse, weights=units))
        m_mean = float(np.average(m_mse, weights=units))
        scarcity = 1.0 - pi
        design = np.column_stack([np.ones_like(scarcity), scarcity])

        def wls(y: np.ndarray) -> tuple[float, float, float]:
            if not np.isfinite(y).all() or float(np.std(y)) <= 1e-12:
                return float("nan"), float("nan"), float("nan")
            root_w = np.sqrt(units)
            beta = np.linalg.lstsq(design * root_w[:, None], y * root_w, rcond=None)[0]
            fitted = design @ beta
            mean_y = float(np.average(y, weights=units))
            ss_res = float(np.sum(units * (y - fitted) ** 2))
            ss_tot = float(np.sum(units * (y - mean_y) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else float("nan")
            return float(beta[0]), float(beta[1]), r2

        p_intercept, p_slope, p_r2 = wls(p_mse / p_mean)
        m_intercept, m_slope, m_r2 = wls(m_mse / m_mean)
        row = dict(zip(("dataset", "design", "setting", "n", "strength"), key_values))
        row.update(
            {
                "fit": "normalized_squared_true_error_on_response_scarcity",
                "formula": "error_over_mean = intercept + scarcity_slope * (1 - true_pi)",
                "p_intercept": p_intercept,
                "p_scarcity_slope": p_slope,
                "p_r2": p_r2,
                "m_intercept": m_intercept,
                "m_scarcity_slope": m_slope,
                "m_r2": m_r2,
                "slope_gap_m_minus_p": m_slope - p_slope,
            }
        )
        out.append(row)
    return out


def _aggregate_fit_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["dataset"]].append(row)
    groups["all"] = list(rows)
    groups["non_Kang_Schafer"] = [
        row for row in rows if row["dataset"] != "Kang-Schafer"
    ]
    metrics = [
        "p_scarcity_slope",
        "m_scarcity_slope",
        "slope_gap_m_minus_p",
        "p_r2",
        "m_r2",
    ]
    out = []
    for dataset, group in sorted(groups.items()):
        row = {"dataset": dataset, "settings": len(group)}
        for metric in metrics:
            values = np.asarray([float(item[metric]) for item in group], dtype=float)
            row[f"mean_{metric}"] = _safe_mean(values)
        out.append(row)
    return out


def _low_high(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed: dict[tuple[str, str, str], dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        keyed[(row["dataset"], row["setting"], row["bin_by"])][int(row["bin"])].append(
            row
        )
    out = []
    for (dataset, setting, bin_by), by_bin in sorted(keyed.items()):
        low = by_bin[min(by_bin)]
        high = by_bin[max(by_bin)]
        def mean_metric(group: list[dict[str, Any]], name: str) -> float:
            return _safe_mean(np.asarray([float(row[name]) for row in group]))

        out.append(
            {
                "dataset": dataset,
                "setting": setting,
                "bin_by": bin_by,
                "low_mean_true_pi": mean_metric(low, "mean_true_pi"),
                "high_mean_true_pi": mean_metric(high, "mean_true_pi"),
                "low_response_rate": mean_metric(low, "response_rate"),
                "high_response_rate": mean_metric(high, "response_rate"),
                "low_responders": mean_metric(low, "responders"),
                "high_responders": mean_metric(high, "responders"),
                "low_p_true_rmse": mean_metric(low, "p_true_rmse"),
                "high_p_true_rmse": mean_metric(high, "p_true_rmse"),
                "low_m_true_rmse": mean_metric(low, "m_true_rmse"),
                "high_m_true_rmse": mean_metric(high, "m_true_rmse"),
                "low_p_true_nrmse": mean_metric(low, "p_true_nrmse"),
                "high_p_true_nrmse": mean_metric(high, "p_true_nrmse"),
                "low_m_true_nrmse": mean_metric(low, "m_true_nrmse"),
                "high_m_true_nrmse": mean_metric(high, "m_true_nrmse"),
                "low_p_label_skill": mean_metric(low, "p_label_skill"),
                "high_p_label_skill": mean_metric(high, "p_label_skill"),
                "low_m_observed_skill": mean_metric(low, "m_observed_skill"),
                "high_m_observed_skill": mean_metric(high, "m_observed_skill"),
                "low_m_observed_rmse": mean_metric(low, "m_observed_rmse"),
                "high_m_observed_rmse": mean_metric(high, "m_observed_rmse"),
            }
        )
    return out


def _write_readme(path: Path, args: argparse.Namespace, settings: list[Setting]) -> None:
    path.write_text(
        "\n".join(
            [
                "# Cross-fitted nuisance accuracy by response",
                "",
                "This diagnostic regenerates the benchmark settings and fits the",
                "cross-fitted response score p_hat and outcome prediction m_hat.",
                "Cross-fitting uses the same folds for p_hat and m_hat, stratified",
                "by rank bins of true pi so the held-out folds have comparable",
                "response-propensity support.  True pi is used here only because",
                "these are known-truth simulations.",
                "Rows are binned both by p_hat, which is observable in applications,",
                "and by true pi, which is available only in these known-truth",
                "simulations.",
                "",
                "The response model is scored against the observed response label",
                "and, as a simulation diagnostic, against true pi.  The outcome",
                "model is scored against observed Y among responders and, as a",
                "simulation diagnostic, against true mu on all units in the bin.",
                "The p and m losses live on different scales; the normalized RMSE",
                "columns divide by the setting-level standard deviation of true pi",
                "or true mu and are mainly for within-setting shape checks.",
                "",
                f"reps: {args.reps}",
                f"bins: {args.bins}",
                f"folds: {args.folds}",
                f"learner: {args.learner}",
                f"propensity_learner: {args.propensity_learner}",
                f"support_data: {args.support_data}",
                f"settings: {len(settings)}",
                "",
                "Output files:",
                "",
                "- per_bin_reps.csv: one row per setting, replication, binning rule, and bin.",
                "- bin_summary.csv: averages over replications for each setting and bin.",
                "- dataset_summary.csv: averages over settings and replications by dataset and bin.",
                "- low_high_summary.csv: lowest-bin versus highest-bin summaries by setting.",
                "- dependence_summary.csv: per-setting low/high ratios and slopes versus response probability.",
                "- dataset_dependence_summary.csv: dataset-level averages of the dependence diagnostics.",
                "- normalized_error_fit_summary.csv: per-setting fits of normalized squared true error against response scarcity.",
                "- dataset_normalized_error_fit_summary.csv: dataset-level averages of the normalized error fits.",
                "",
            ]
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--source-script",
        type=Path,
        default=default_dir / "validated_reference_transfer.py",
    )
    parser.add_argument(
        "--breadth-script",
        type=Path,
        default=default_dir / "section4_breadth_experiments.py",
    )
    parser.add_argument(
        "--support-data",
        type=Path,
        default=Path("/tmp/dml_real_benchmark_support_data"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=16)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=14700000)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--learner", default="xgboost")
    parser.add_argument("--propensity-learner", default="xgboost")
    args = parser.parse_args()
    settings = _settings()
    tasks = [
        (
            setting,
            rep,
            args.seed,
            args.learner,
            args.propensity_learner,
            args.folds,
            args.bins,
        )
        for setting in settings
        for rep in range(args.reps)
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_readme(args.out_dir / "README.md", args, settings)
    rows: list[dict[str, Any]] = []
    completed = 0
    with ProcessPoolExecutor(
        max_workers=args.jobs,
        initializer=_init_worker,
        initargs=(
            str(args.source_script.resolve()),
            str(args.breadth_script.resolve()),
            str(args.support_data.resolve()),
        ),
    ) as pool:
        futures = [pool.submit(_run_setting_rep, task) for task in tasks]
        for future in as_completed(futures):
            rows.extend(future.result())
            completed += 1
            if completed % max(1, args.jobs) == 0 or completed == len(tasks):
                print(f"completed {completed}/{len(tasks)} setting-reps", flush=True)
    rows.sort(
        key=lambda row: (
            row["dataset"],
            row["setting"],
            row["rep"],
            row["bin_by"],
            row["bin"],
        )
    )
    _write_csv(args.out_dir / "per_bin_reps.csv", rows)
    bin_summary = _aggregate(
        rows,
        ("dataset", "design", "setting", "n", "strength", "bin_by", "bin"),
    )
    dependence_summary = _dependence_from_bin_summary(bin_summary)
    normalized_fit_summary = _normalized_error_fit(rows)
    _write_csv(args.out_dir / "bin_summary.csv", bin_summary)
    _write_csv(
        args.out_dir / "dataset_summary.csv",
        _aggregate(rows, ("dataset", "bin_by", "bin")),
    )
    _write_csv(args.out_dir / "low_high_summary.csv", _low_high(rows))
    _write_csv(args.out_dir / "dependence_summary.csv", dependence_summary)
    _write_csv(
        args.out_dir / "dataset_dependence_summary.csv",
        _aggregate_dependence(dependence_summary),
    )
    _write_csv(
        args.out_dir / "normalized_error_fit_summary.csv", normalized_fit_summary
    )
    _write_csv(
        args.out_dir / "dataset_normalized_error_fit_summary.csv",
        _aggregate_fit_summary(normalized_fit_summary),
    )
    print(f"wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
