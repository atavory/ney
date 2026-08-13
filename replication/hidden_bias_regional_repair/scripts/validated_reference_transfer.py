#!/usr/bin/env python3
"""Region-targeted repair experiment.

The default reference is standard C-TMLE: initial outcome fit + single clever
covariate 1/p, with the propensity floor tau selected by observed-outcome
validation loss. The same driver can also report the AIPW score reference, the
in-driver GL-risk path selector, or a Cui--Tchetgen-style global DR-risk
selector over the same propensity-floor grid.

Repair = add a REGION-SPECIFIC targeting fluctuation with clever covariate
1_G/p, holding the SAME selected tau / propensity. This solves the extra
regional score  sum_obs 1_G (1/p)(Y - m*) = 0  that the global targeting leaves
unsolved. Unlike reweighting the outcome model, a new targeting direction is
NOT re-absorbed by the global targeting step, so the same-propensity contrast
D = theta_ref - theta_rt is a genuine observable signal of C-TMLE's residual
regional bias. The positive-part gate then shrinks the C-TMLE reference toward
the region-targeted repair only when that contrast clears its own noise.

Decisive readout per cell:
  ref_bias    : reference regional bias (should be nonzero)
  repair_bias : region-targeted bias (is it SMALLER? -> there was residual to fix)
  m_snr       : |E delta| / sqrt(E vd)  (does the contrast clear the ~0.69 floor?)
  gain_shrink : MSE reduction of the gated estimator over the reference
Run with: fbpython ushmoo_ctmle_region_targeted.py --out ...
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None
    XGBRegressor = None


SUPPORT_DATA = Path(
    os.environ.get(
        "USHMOO_SUPPORT_DATA",
        "/home/atavory/.overleaf_git_clone/69edff47028a983c95b7fcc2/support/data",
    )
)


def _stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_stamp()}] {message}", flush=True)


def _rank01(values):
    order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
    denom = max(1, len(values) - 1)
    return order.astype(float) / float(denom)


def _standardize_columns(x):
    x = np.asarray(x, dtype=float)
    center = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (x - center) / scale


def _col(x, index):
    return x[:, min(index, x.shape[1] - 1)]


def _sigmoid(values):
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def _diabetes_population():
    data = load_diabetes()
    x = data.data.astype(float)
    target = data.target.astype(float)
    y_real = (target - float(target.mean())) / float(target.std())
    score_region = 1.2 * x[:, 2] - 0.8 * x[:, 8] + 0.5 * x[:, 0] - 0.3 * x[:, 6]
    score_signal = 0.7 * x[:, 2] + 0.9 * x[:, 3] - 0.4 * x[:, 8] + 0.2 * x[:, 1]
    response_region = score_region <= np.quantile(score_region, 0.15)
    signal_region = score_signal <= np.quantile(score_signal, 0.18)
    outside_rank = _rank01(0.6 * x[:, 3] + 0.8 * x[:, 8] - 0.4 * x[:, 6])
    base = (
        0.65 * y_real
        + 0.35 * np.sin(2.0 * np.pi * _rank01(x[:, 2]))
        + 0.25 * (x[:, 3] / max(float(x[:, 3].std()), 1e-12)) ** 2
    )
    base = (base - float(base.mean())) / max(float(base.std()), 1e-12)
    return x, y_real, base, response_region, signal_region, outside_rank


def _make_diabetes_data(
    n: int, epsilon: float, strength: float, design: str, seed: int
):
    rng = np.random.default_rng(seed)
    x_pop, y_real, base, response_region, signal_region, outside_rank = (
        _diabetes_population()
    )
    idx = rng.integers(0, len(y_real), n)
    x = x_pop[idx]
    region = response_region[idx]
    outside = np.clip(0.68 + 0.22 * outside_rank[idx], 0.05, 0.95)
    true_pi = np.where(region, epsilon, outside)
    response = rng.binomial(1, true_pi)

    if design == "diabetes_real":
        mu_pop = y_real
        y = y_real[idx]
        theta = float(mu_pop.mean())
        return x, y, response, region, true_pi, theta, y

    if design == "diabetes_semisynth":
        signal_pop = response_region.astype(float) * (
            0.65 + 0.70 * _rank01(x_pop[:, 2])
        )
    elif design == "diabetes_misaligned":
        signal_pop = signal_region.astype(float) * (0.65 + 0.70 * _rank01(x_pop[:, 3]))
    else:
        raise ValueError(f"unknown diabetes design: {design}")

    mu_pop = base + 0.35 * strength * signal_pop
    y = mu_pop[idx] + rng.standard_normal(n)
    theta = float(mu_pop.mean())
    return x, y, response, region, true_pi, theta, mu_pop[idx]


def _ihdp_population(seed: int):
    train_path = SUPPORT_DATA / "ihdp" / "ihdp_npci_1-100.train.npz"
    test_path = SUPPORT_DATA / "ihdp" / "ihdp_npci_1-100.test.npz"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"IHDP data not found under {SUPPORT_DATA / 'ihdp'}")
    train = np.load(train_path)
    test = np.load(test_path)
    surface = int(seed) % 100
    x = np.vstack([train["x"][:, :, surface], test["x"][:, :, surface]])
    mu0 = np.concatenate([train["mu0"][:, surface], test["mu0"][:, surface]])
    mu1 = np.concatenate([train["mu1"][:, surface], test["mu1"][:, surface]])
    x = _standardize_columns(x)
    base = 0.70 * _standardize_columns(mu0.reshape(-1, 1))[:, 0]
    base += 0.30 * _standardize_columns((mu1 - mu0).reshape(-1, 1))[:, 0]
    score_region = (
        1.0 * _col(x, 0) - 0.7 * _col(x, 1) + 0.6 * _col(x, 5) - 0.4 * _col(x, 8)
    )
    score_signal = 0.8 * _col(x, 2) + 0.5 * _col(x, 6) - 0.6 * _col(x, 10)
    response_region = score_region <= np.quantile(score_region, 0.15)
    signal_region = score_signal <= np.quantile(score_signal, 0.18)
    outside_rank = _rank01(0.5 * _col(x, 3) + 0.6 * _col(x, 7))
    return x, base, response_region, signal_region, outside_rank


def _make_ihdp_data(n: int, epsilon: float, strength: float, design: str, seed: int):
    rng = np.random.default_rng(seed)
    x_pop, base, response_region, signal_region, outside_rank = _ihdp_population(seed)
    idx = rng.integers(0, len(base), n)
    x = x_pop[idx]
    region = response_region[idx]
    outside = np.clip(0.68 + 0.22 * outside_rank[idx], 0.05, 0.95)
    true_pi = np.where(region, epsilon, outside)
    response = rng.binomial(1, true_pi)
    if design == "ihdp_semisynth":
        signal_pop = response_region.astype(float) * (
            0.70 + 0.50 * _rank01(_col(x_pop, 2))
        )
    elif design == "ihdp_misaligned":
        signal_pop = signal_region.astype(float) * (
            0.70 + 0.50 * _rank01(_col(x_pop, 3))
        )
    else:
        raise ValueError(f"unknown IHDP design: {design}")
    mu_pop = base + 0.35 * strength * signal_pop
    y = mu_pop[idx] + rng.standard_normal(n)
    theta = float(mu_pop.mean())
    return x, y, response, region, true_pi, theta, mu_pop[idx]


def _acic2016_population():
    path = SUPPORT_DATA / "acic" / "acic2016_x.csv"
    if not path.exists():
        raise FileNotFoundError(f"ACIC 2016 covariates not found at {path}")
    x = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    matrix = np.column_stack([x[name] for name in x.dtype.names])
    matrix = _standardize_columns(np.nan_to_num(matrix))
    base = (
        0.35 * _col(matrix, 0)
        - 0.25 * _col(matrix, 4)
        + 0.20 * _col(matrix, 11) * _col(matrix, 12)
        + 0.25 * np.sin(_col(matrix, 20))
    )
    base = _standardize_columns(base.reshape(-1, 1))[:, 0]
    score_region = (
        0.8 * _col(matrix, 1)
        - 0.7 * _col(matrix, 5)
        + 0.6 * _col(matrix, 16)
        - 0.4 * _col(matrix, 28)
    )
    score_signal = (
        0.7 * _col(matrix, 2) + 0.6 * _col(matrix, 8) - 0.5 * _col(matrix, 24)
    )
    response_region = score_region <= np.quantile(score_region, 0.15)
    signal_region = score_signal <= np.quantile(score_signal, 0.18)
    outside_rank = _rank01(0.5 * _col(matrix, 3) + 0.4 * _col(matrix, 17))
    return matrix, base, response_region, signal_region, outside_rank


def _make_acic2016_data(
    n: int, epsilon: float, strength: float, design: str, seed: int
):
    rng = np.random.default_rng(seed)
    x_pop, base, response_region, signal_region, outside_rank = _acic2016_population()
    idx = rng.integers(0, len(base), n)
    x = x_pop[idx]
    region = response_region[idx]
    outside = np.clip(0.68 + 0.22 * outside_rank[idx], 0.05, 0.95)
    true_pi = np.where(region, epsilon, outside)
    response = rng.binomial(1, true_pi)
    if design == "acic2016_semisynth":
        signal_pop = response_region.astype(float) * (
            0.70 + 0.50 * _rank01(_col(x_pop, 2))
        )
    elif design == "acic2016_misaligned":
        signal_pop = signal_region.astype(float) * (
            0.70 + 0.50 * _rank01(_col(x_pop, 6))
        )
    else:
        raise ValueError(f"unknown ACIC 2016 design: {design}")
    mu_pop = base + 0.35 * strength * signal_pop
    y = mu_pop[idx] + rng.standard_normal(n)
    theta = float(mu_pop.mean())
    return x, y, response, region, true_pi, theta, mu_pop[idx]


def _region_signal(region, values):
    signal = np.zeros(len(region))
    if not np.any(region):
        return signal
    inside = np.maximum(values[region], 0.0)
    scale = float(np.mean(inside))
    if scale <= 0.0:
        signal[region] = 1.0
    else:
        signal[region] = inside / scale
    return signal


def make_data(
    n: int,
    epsilon: float,
    strength: float,
    design: str,
    seed: int,
    mar_design: str,
):
    if design.startswith("diabetes_"):
        if mar_design != "box":
            raise ValueError("diabetes designs use the built-in response pattern")
        return _make_diabetes_data(n, epsilon, strength, design, seed)
    if design.startswith("ihdp_"):
        if mar_design != "box":
            raise ValueError("IHDP designs use the built-in response pattern")
        return _make_ihdp_data(n, epsilon, strength, design, seed)
    if design.startswith("acic2016_"):
        if mar_design != "box":
            raise ValueError("ACIC 2016 designs use the built-in response pattern")
        return _make_acic2016_data(n, epsilon, strength, design, seed)

    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, (n, 5))
    if mar_design == "box":
        region = (x[:, 0] < 0.2) & (x[:, 1] < 0.5)
        outside = np.clip(0.72 + 0.18 * x[:, 3], 0.05, 0.95)
        true_pi = np.where(region, epsilon, outside)
    elif mar_design == "smooth_tail":
        linear = 2.8 - 5.5 * x[:, 0] - 4.0 * x[:, 1] + 0.75 * x[:, 2]
        true_pi = epsilon + (0.95 - epsilon) * _sigmoid(linear)
        true_pi = np.clip(true_pi, epsilon, 0.95)
        region = true_pi <= float(np.quantile(true_pi, 0.10))
    elif mar_design == "nonlinear_mar":
        raw_pi = (
            0.5
            + 0.4 * np.sin(2.0 * np.pi * x[:, 0] * x[:, 1])
            - 0.3 * (x[:, 2] > x[:, 3]).astype(float)
        )
        true_pi = np.clip(raw_pi, max(epsilon, 0.03), 0.95)
        region = true_pi <= float(np.quantile(true_pi, 0.10))
    elif mar_design == "two_stratum_flip":
        g1 = (x[:, 0] < 0.2) & (x[:, 1] < 0.25)
        g2 = (x[:, 0] < 0.2) & (x[:, 1] >= 0.25) & (x[:, 1] < 0.5)
        region = g1
        outside = np.clip(0.72 + 0.18 * x[:, 3], 0.05, 0.95)
        true_pi = np.where(g1 | g2, epsilon, outside)
    else:
        raise ValueError(f"unknown mar_design: {mar_design}")
    if design == "flat":
        base = np.full(n, 0.3)
    else:
        base = 0.3 + 0.4 * (x[:, 0] - 0.5) + 0.3 * np.sin(2.0 * np.pi * x[:, 1])
    patterns = {
        "flat": np.zeros(n),
        "smooth": region * (1.0 - x[:, 0] / 0.2),
        "pockets": region * (0.5 + x[:, 2]),
        "oscillatory": region * (1.0 + np.sin(4.0 * np.pi * x[:, 2])),
        "regional_shift": region.astype(float),
        "regional_ramp": _region_signal(region, 0.25 + x[:, 2]),
        "regional_bump": _region_signal(
            region,
            np.exp(-20.0 * ((x[:, 2] - 0.5) ** 2 + (x[:, 3] - 0.5) ** 2)),
        ),
    }
    if mar_design == "two_stratum_flip":
        g1 = (x[:, 0] < 0.2) & (x[:, 1] < 0.25)
        g2 = (x[:, 0] < 0.2) & (x[:, 1] >= 0.25) & (x[:, 1] < 0.5)
        mu = base + strength * (g1.astype(float) - g2.astype(float))
    else:
        mu = base + strength * patterns[design]
    y = mu + rng.standard_normal(n)
    response = rng.binomial(1, true_pi)
    theta = float(np.mean(mu))
    return x, y, response, region, true_pi, theta, mu


def _estimated_response_score(
    x,
    response,
    true_pi,
    mode,
    learner,
    seed,
    folds=3,
):
    if mode == "true":
        return true_pi.copy()
    if len(np.unique(response)) < 2:
        return true_pi.copy()
    response_bool = response.astype(bool)
    class_counts = np.bincount(response_bool.astype(int), minlength=2)
    n_splits = min(max(2, folds), int(class_counts.min()))
    score = np.empty(len(response), dtype=float)
    if n_splits >= 2:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        for fold, (train, test) in enumerate(splitter.split(x, response_bool)):
            if len(np.unique(response_bool[train])) < 2:
                score[test] = float(np.mean(response_bool[train]))
                continue
            model = _classifier(seed + 101 * (fold + 1), learner)
            model.fit(x[train], response_bool[train])
            score[test] = model.predict_proba(x[test])[:, 1]
    else:
        model = _classifier(seed, learner)
        model.fit(x, response_bool)
        score = model.predict_proba(x)[:, 1]
    return score


def _low_score_region(score, response, region_quantile, min_observed=0):
    if min_observed > 0:
        order = np.argsort(score, kind="mergesort")
        min_count = min(
            len(score),
            max(1, int(math.ceil(region_quantile * len(score)))),
        )
        observed = np.cumsum(response[order].astype(bool))
        hits = np.flatnonzero(observed >= min_observed)
        count = max(min_count, int(hits[0]) + 1) if len(hits) else len(score)
        mask = np.zeros(len(score), dtype=bool)
        mask[order[:count]] = True
        return mask
    cutoff = float(np.quantile(score, region_quantile))
    return score <= cutoff


def _estimated_lowp_region(
    x,
    response,
    true_pi,
    mode,
    propensity_learner,
    seed,
    region_quantile,
    min_observed=0,
    folds=3,
):
    score = _estimated_response_score(
        x,
        response,
        true_pi,
        mode,
        propensity_learner,
        seed,
        folds,
    )
    return _low_score_region(score, response, region_quantile, min_observed)


def _oof_outcome_prediction(x, y, response, learner, seed, folds):
    labels = np.random.default_rng(seed).integers(0, folds, len(y))
    prediction = np.empty(len(y), dtype=float)
    global_mean = float(np.mean(y[response.astype(bool)])) if np.any(response) else 0.0
    for fold in range(folds):
        test = labels == fold
        observed = (labels != fold) & response.astype(bool)
        if observed.sum() < 20:
            prediction[test] = global_mean
            continue
        model = _regressor(seed + 211 * fold, learner)
        model.fit(x[observed], y[observed])
        prediction[test] = model.predict(x[test])
    return prediction


def _variance_of_full_sample_mean(sum_z, sum_z_sq, n):
    if n <= 1:
        return float("nan")
    numerator = sum_z_sq - (sum_z * sum_z) / float(n)
    return max(0.0, numerator) / float(n - 1) / float(n)


def _estimated_residual_lowp_region(
    x,
    y,
    response,
    true_pi,
    mode,
    learner,
    propensity_learner,
    seed,
    region_quantile,
    min_observed,
    selector_ablation,
    region_detector_c,
    folds,
):
    standdown_ablations = {
        "empty_standdown",
        "crossfit_rank_empty_standdown",
    }
    response_score = _estimated_response_score(
        x,
        response,
        true_pi,
        mode,
        propensity_learner,
        seed,
        folds,
    )
    lowp = _low_score_region(response_score, response, region_quantile, min_observed)
    observed = response.astype(bool)
    observed_lowp = observed & lowp
    empty = np.zeros(len(y), dtype=bool)
    if observed_lowp.sum() < max(3, min_observed):
        return empty if selector_ablation in standdown_ablations else lowp

    m0 = _oof_outcome_prediction(x, y, response, learner, seed + 7001, folds)
    weighted_residual = np.full(len(y), np.nan)
    weighted_residual[observed] = (y[observed] - m0[observed]) / np.maximum(
        response_score[observed], 0.02
    )
    direction = float(np.mean(weighted_residual[observed_lowp]))
    if direction == 0.0 or not np.isfinite(direction):
        return empty if selector_ablation in standdown_ablations else lowp

    direction_sign = float(np.sign(direction))
    if selector_ablation == "crossfit_rank_empty_standdown":
        # The legacy detector fits the residual-ranking model and evaluates
        # candidate prefixes on the same residuals. Flexible learners can then
        # certify their own in-sample overfit. Give every observed unit a rank
        # predicted by a model that did not train on that unit's residual.
        target = np.zeros(len(y), dtype=float)
        target[observed] = direction_sign * weighted_residual[observed]
        full_model = _regressor(seed + 9001, learner)
        full_model.fit(x[observed], target[observed])
        rank_signal = full_model.predict(x)
        labels = np.random.default_rng(seed + 9002).integers(0, folds, len(y))
        for fold in range(folds):
            test = observed & (labels == fold)
            train = observed & (labels != fold)
            if not np.any(test):
                continue
            if int(np.sum(train)) < 20:
                rank_signal[test] = (
                    float(np.mean(target[train])) if np.any(train) else 0.0
                )
                continue
            residual_model = _regressor(seed + 9101 + 211 * fold, learner)
            residual_model.fit(x[train], target[train])
            rank_signal[test] = residual_model.predict(x[test])
        rank_signs = (1.0,)
    elif selector_ablation == "raw_rank_only":
        rank_signal = np.zeros(len(y), dtype=float)
        rank_signal[observed] = direction_sign * weighted_residual[observed]
        rank_signs = (1.0,)
    elif selector_ablation == "both_signs":
        residual_model = _regressor(seed + 9001, learner)
        residual_model.fit(x[observed], weighted_residual[observed])
        rank_signal = residual_model.predict(x)
        rank_signs = (1.0, -1.0)
    else:
        residual_model = _regressor(seed + 9001, learner)
        residual_model.fit(
            x[observed],
            direction_sign * weighted_residual[observed],
        )
        rank_signal = residual_model.predict(x)
        rank_signs = (1.0,)

    inside = np.flatnonzero(lowp)
    if len(inside) == 0:
        return empty if selector_ablation in standdown_ablations else lowp
    best_mask = empty if selector_ablation in standdown_ablations else lowp
    best_score = 0.0
    for rank_sign in rank_signs:
        order = inside[np.argsort(-(rank_sign * rank_signal[inside]), kind="mergesort")]
        if selector_ablation == "all_prefixes":
            candidate_counts = range(1, len(order) + 1)
        else:
            candidate_counts = (
                max(1, int(math.ceil(frac * len(order))))
                for frac in (0.25, 0.50, 0.75, 1.00)
            )
        for count in candidate_counts:
            mask = np.zeros(len(lowp), dtype=bool)
            mask[order[:count]] = True
            if int(np.sum(mask & observed)) < max(3, min_observed):
                continue
            score_sign = (
                rank_sign if selector_ablation == "both_signs" else direction_sign
            )
            if selector_ablation == "whole_sample_score":
                values = np.zeros(len(y), dtype=float)
                values[observed] = score_sign * weighted_residual[observed]
                sum_z = float(np.sum(values[mask]))
                sum_z_sq = float(np.sum(values[mask] * values[mask]))
                mean_val = sum_z / float(len(y))
                var_val = _variance_of_full_sample_mean(sum_z, sum_z_sq, len(y))
            else:
                vals = score_sign * weighted_residual[mask & observed]
                if len(vals) < 2:
                    continue
                mean_val = float(np.mean(vals))
                var_val = float(np.var(vals, ddof=1) / len(vals))
            if not np.isfinite(var_val):
                continue
            score = mean_val * mean_val - region_detector_c * var_val
            if score > best_score:
                best_score = score
                best_mask = mask
    return best_mask


def _estimated_kappa_residual_lowp_region(
    x,
    y,
    response,
    true_pi,
    mode,
    learner,
    propensity_learner,
    seed,
    region_quantile,
    min_observed,
    region_kappa_floor,
    folds,
):
    response_score = _estimated_response_score(
        x,
        response,
        true_pi,
        mode,
        propensity_learner,
        seed,
        folds,
    )
    lowp = _low_score_region(response_score, response, region_quantile, min_observed)
    observed = response.astype(bool)
    observed_lowp = observed & lowp
    if observed_lowp.sum() < max(3, min_observed):
        return np.zeros(len(y), dtype=bool)

    m0 = _oof_outcome_prediction(x, y, response, learner, seed + 7001, folds)
    pi_hat = np.maximum(response_score, 0.02)
    clipped_p = np.maximum(response_score, region_kappa_floor)
    kappa_hat = 1.0 - response_score / clipped_p
    weighted_residual = np.full(len(y), np.nan)
    weighted_residual[observed] = (
        kappa_hat[observed] * (m0[observed] - y[observed]) / pi_hat[observed]
    )
    direction = float(np.mean(weighted_residual[observed_lowp]))
    if direction == 0.0 or not np.isfinite(direction):
        return np.zeros(len(y), dtype=bool)

    residual_model = _regressor(seed + 9001, learner)
    residual_model.fit(
        x[observed],
        np.sign(direction) * weighted_residual[observed],
    )
    aligned_signal = residual_model.predict(x)

    inside = np.flatnonzero(lowp)
    if len(inside) == 0:
        return np.zeros(len(y), dtype=bool)
    order = inside[np.argsort(-aligned_signal[inside], kind="mergesort")]
    candidate_fracs = (0.25, 0.50, 0.75, 1.00)
    best_mask = np.zeros(len(lowp), dtype=bool)
    best_score = 0.0
    for frac in candidate_fracs:
        count = max(1, int(math.ceil(frac * len(order))))
        mask = np.zeros(len(lowp), dtype=bool)
        mask[order[:count]] = True
        if int(np.sum(mask & observed)) < max(3, min_observed):
            continue
        vals = np.sign(direction) * weighted_residual[mask & observed]
        if len(vals) < 2:
            continue
        mean_val = float(np.mean(vals))
        var_val = float(np.var(vals, ddof=1) / len(vals))
        score = mean_val * mean_val - 4.0 * var_val
        if score > best_score:
            best_score = score
            best_mask = mask
    return best_mask


def _analysis_region(
    x,
    y,
    response,
    response_region,
    true_pi,
    variant: str,
    mode,
    learner,
    propensity_learner,
    seed,
    region_quantile,
    region_min_observed,
    region_kappa_floor,
    selector_ablation,
    region_detector_c,
    folds,
):
    if variant == "true":
        return response_region
    if variant == "true_lowp":
        return true_pi <= float(np.quantile(true_pi, region_quantile))
    if variant in {"estimated_lowp", "estimated_lowp_supported"}:
        return _estimated_lowp_region(
            x,
            response,
            true_pi,
            mode,
            propensity_learner,
            seed,
            region_quantile,
            region_min_observed if variant == "estimated_lowp_supported" else 0,
            folds,
        )
    if variant == "estimated_residual_lowp_supported":
        return _estimated_residual_lowp_region(
            x,
            y,
            response,
            true_pi,
            mode,
            learner,
            propensity_learner,
            seed,
            region_quantile,
            region_min_observed,
            selector_ablation,
            region_detector_c,
            folds,
        )
    if variant == "estimated_kappa_residual_lowp_supported":
        return _estimated_kappa_residual_lowp_region(
            x,
            y,
            response,
            true_pi,
            mode,
            learner,
            propensity_learner,
            seed,
            region_quantile,
            region_min_observed,
            region_kappa_floor,
            folds,
        )
    if variant == "flip_g1":
        return (x[:, 0] < 0.2) & (x[:, 1] < 0.25)
    if variant == "flip_g2":
        return (x[:, 0] < 0.2) & (x[:, 1] >= 0.25) & (x[:, 1] < 0.5)
    if variant == "flip_both":
        return (x[:, 0] < 0.2) & (x[:, 1] < 0.5)
    x0 = _rank01(x[:, 0])
    x1 = _rank01(x[:, 1])
    if variant == "shrink":
        if not np.any(response_region):
            return response_region
        cutoff = float(np.quantile(x0[response_region], 0.5))
        return response_region & (x0 <= cutoff)
    if variant == "expand":
        return response_region | ((x0 < 0.30) & (x1 < 0.65))
    if variant == "shift":
        return (x0 >= 0.20) & (x0 < 0.40) & (x1 < 0.50)
    if variant == "wrong":
        return (x0 > 0.70) & (x1 > 0.50)
    raise ValueError(f"unknown analysis_region: {variant}")


def _design_seed_offset(design: str) -> int:
    known = {
        "flat": 0,
        "smooth": 1,
        "pockets": 2,
        "oscillatory": 3,
        "diabetes_real": 11,
        "diabetes_semisynth": 12,
        "diabetes_misaligned": 13,
        "ihdp_semisynth": 21,
        "ihdp_misaligned": 22,
        "acic2016_semisynth": 31,
        "acic2016_misaligned": 32,
    }
    if design in known:
        return known[design] * 10_000_000
    return sum((i + 1) * ord(ch) for i, ch in enumerate(design)) * 1000


def _mar_seed_offset(mar_design: str) -> int:
    known = {
        "box": 0,
        "smooth_tail": 101,
        "nonlinear_mar": 202,
        "two_stratum_flip": 404,
    }
    return known.get(mar_design, 303) * 10_000_000


def _design_strength_code(strength: float) -> int:
    return int(round(strength * 100))


def _regressor(seed: int, learner: str):
    if learner == "xgboost":
        if XGBRegressor is None:
            raise RuntimeError("xgboost learner requested but xgboost is unavailable")
        return XGBRegressor(random_state=seed, n_jobs=1, verbosity=0)
    return HistGradientBoostingRegressor(
        max_iter=60,
        learning_rate=0.10,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=seed,
    )


def _classifier(seed: int, learner: str):
    if learner == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost learner requested but xgboost is unavailable")
        return XGBClassifier(random_state=seed, n_jobs=1, verbosity=0)
    return HistGradientBoostingClassifier(
        max_iter=60,
        learning_rate=0.10,
        max_leaf_nodes=12,
        min_samples_leaf=40,
        l2_regularization=2.0,
        random_state=seed,
    )


def _propensity_predictions(x, response, true_pi, train, test, mode, seed, learner):
    if mode == "true":
        return true_pi[train].copy(), true_pi[test].copy()
    model = _classifier(seed, learner)
    model.fit(x[train], response[train])
    return model.predict_proba(x[train])[:, 1], model.predict_proba(x[test])[:, 1]


def _region_damp() -> float:
    return float(os.environ.get("USHMOO_REGION_DAMP", "1.0"))


def _validation_risk() -> str:
    return os.environ.get("USHMOO_VALIDATION_RISK", "balanced_mse")


def _targets(
    y_obs,
    p_obs,
    m_obs,
    region_obs,
    p_test,
    m_test,
    region_test,
    region_damp,
    target_global=True,
):
    """Return (ref_star_test, region_targeted_star_test) at a given tau/propensity."""
    h_obs = 1.0 / p_obs
    if target_global:
        denom = float(np.dot(h_obs, h_obs))
        eps = 0.0 if denom <= 0.0 else float(np.dot(h_obs, y_obs - m_obs) / denom)
        ref_obs = m_obs + eps / p_obs
        ref_test = m_test + eps / p_test
    else:
        ref_obs = m_obs
        ref_test = m_test
    if region_damp == 0.0:
        return ref_test, ref_test
    hg_obs = h_obs * region_obs.astype(float)
    if float(np.dot(hg_obs, hg_obs)) > 0.0:
        e2 = float(np.dot(hg_obs, y_obs - ref_obs) / np.dot(hg_obs, hg_obs))
        e2 *= region_damp
        rt_test = ref_test + e2 * region_test.astype(float) / p_test
    else:
        rt_test = ref_test.copy()
    return ref_test, rt_test


def _region_balance_weight(region, validation_region_weight):
    if validation_region_weight >= 0.0:
        return float(validation_region_weight)
    q = float(np.mean(region))
    if q <= 0.0 or q >= 1.0:
        return 0.0
    return max(0.0, (1.0 - q) / q - 1.0)


def _weighted_loss(y, prediction, region, validation_region_weight):
    lam = _region_balance_weight(region, validation_region_weight)
    weights = 1.0 + lam * region.astype(float)
    return float(np.average((y - prediction) ** 2, weights=weights))


def _observed_validation_weights(p, region, validation_region_weight):
    """Weights for a loss evaluated only on observed outcomes."""
    if _validation_risk() == "aipw_variance":
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0)
        return (1.0 - p) / (p * p)
    return 1.0 + _region_balance_weight(
        region, validation_region_weight
    ) * region.astype(float)


def _mean_variance_of_mean(values):
    if len(values) < 2:
        return float(np.mean(values)) if len(values) else 0.0, float("inf")
    return float(np.mean(values)), float(np.var(values, ddof=1) / len(values))


def _aipw_score(y, response, p, m):
    return m + response.astype(float) * (y - m) / p


def _fit_with_sample_weight(model, x, y, sample_weight):
    """Route weights through a sklearn Pipeline when an adapter wraps a learner."""
    steps = getattr(model, "steps", None)
    if steps:
        final_name = steps[-1][0]
        model.fit(
            x,
            y,
            **{f"{final_name}__sample_weight": sample_weight},
        )
    else:
        model.fit(x, y, sample_weight=sample_weight)


def _crossfit_weighted_residual_correction(
    x, y, response, p, base_prediction, learner, seed, folds
):
    """Learn an honest residual correction for the MAR variance risk."""
    labels = np.random.default_rng(seed).integers(0, folds, len(y))
    observed = response.astype(bool)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0)
    correction = np.zeros(len(y), dtype=float)
    for fold in range(folds):
        test = labels == fold
        train = observed & (labels != fold)
        if int(np.sum(train)) < 20:
            continue
        # On responders, these sample weights identify
        # E[(1-pi)/pi * {m-m_star}^2].
        sample_weight = (1.0 - p[train]) / (p[train] * p[train])
        if not np.any(sample_weight > 0.0):
            continue
        residual_model = _regressor(seed + 401 * fold, learner)
        _fit_with_sample_weight(
            residual_model,
            x[train],
            y[train] - base_prediction[train],
            sample_weight,
        )
        correction[test] = residual_model.predict(x[test])
    return correction


def _crossfit_influence_projection(x, response, p, base_score, learner, seed, folds):
    """Learn the variance-optimal MAR control variate for any expert score.

    For ``a = 1 - R / pi(X)``, every correction ``a * g(X)`` has population
    mean zero under MAR.  The conditional least-squares projection

        g*(X) = -E[a * phi_0 | X] / E[a**2 | X]

    minimizes ``Var(phi_0 + a * g(X))``.  We fit the equivalent weighted
    regression of ``-phi_0 / a`` on X with weights ``a**2`` on independent
    folds.  Centering ``base_score`` is harmless because E[a | X] = 0 and
    improves numerical stability.
    """
    labels = np.random.default_rng(seed).integers(0, folds, len(base_score))
    response = np.asarray(response, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    score = np.asarray(base_score, dtype=float)
    a = 1.0 - response / p
    correction = np.zeros(len(score), dtype=float)
    for fold in range(folds):
        test = labels == fold
        train = labels != fold
        usable = train & (np.abs(a) > 1e-8) & np.isfinite(score)
        if int(np.sum(usable)) < 20:
            continue
        centered_score = score - float(np.mean(score[usable]))
        pseudo_outcome = -centered_score[usable] / a[usable]
        sample_weight = a[usable] * a[usable]
        projection_model = _regressor(seed + 503 * fold, learner)
        _fit_with_sample_weight(
            projection_model,
            x[usable],
            pseudo_outcome,
            sample_weight,
        )
        correction[test] = projection_model.predict(x[test])
    return correction


def _shifted_legendre_derivative_at_zero(degree, order):
    """Derivatives at zero of P_j(2a-1), j=0,...,degree."""
    derivatives = np.zeros(degree + 1, dtype=float)
    for j in range(degree + 1):
        if j < order:
            continue
        basis = np.polynomial.legendre.Legendre.basis(j).deriv(order)
        derivatives[j] = (2.0**order) * float(basis(-1.0))
    return derivatives


def _ma_dr_bc_reference(
    y,
    response,
    p,
    m,
    trim_h=0.05,
    correction_order=1,
    sieve_degree=3,
):
    """Ma--Sant'Anna--Sasaki--Ura DR-BC for the MAR outcome mean.

    This is equation (3.1) of Ma, Sant'Anna, Sasaki, and Ura (2026),
    specialized to

        theta = E[m(X)] + E[B/A],
        A = p(X), B = R * (Y - m(X)).

    The conditional mean xi(a)=E[B|A=a] is estimated with their shifted
    Legendre sieve.  The returned observation-level values average to the
    same-target bias-corrected trimmed DR estimator.
    """
    if not (0.0 <= trim_h < 1.0):
        raise ValueError("DR-BC trimming threshold must lie in [0, 1)")
    if correction_order < 1:
        raise ValueError("DR-BC correction order must be positive")
    if sieve_degree < correction_order:
        raise ValueError("DR-BC sieve degree must cover the correction order")

    y = np.asarray(y, dtype=float)
    response = np.asarray(response, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    m = np.asarray(m, dtype=float)
    if not (len(y) == len(response) == len(p) == len(m)):
        raise ValueError("DR-BC inputs must have equal length")

    b = response * (y - m)
    shifted = 2.0 * p - 1.0
    design = np.polynomial.legendre.legvander(shifted, sieve_degree)
    coefficients, _, rank, _ = np.linalg.lstsq(design, b, rcond=None)
    if rank < sieve_degree + 1:
        raise RuntimeError("DR-BC shifted-Legendre sieve is rank deficient")

    trimmed = p < trim_h
    untrimmed_ratio = np.where(trimmed, 0.0, b / p)
    correction = np.zeros(len(p), dtype=float)
    derivatives = {}
    for order in range(1, correction_order + 1):
        basis_derivative = _shifted_legendre_derivative_at_zero(sieve_degree, order)
        xi_derivative = float(np.dot(coefficients, basis_derivative))
        derivatives[order] = xi_derivative
        correction += (
            trimmed.astype(float)
            * p ** (order - 1)
            * xi_derivative
            / math.factorial(order)
        )

    values = m + untrimmed_ratio + correction
    return values, {
        "trim_h": float(trim_h),
        "correction_order": int(correction_order),
        "sieve_degree": int(sieve_degree),
        "trimmed_fraction": float(np.mean(trimmed)),
        "xi_derivative_1": float(derivatives.get(1, float("nan"))),
        "bias_correction_mean": float(np.mean(correction)),
    }


def _crossfit_ma_dr_bc_score(
    y,
    response,
    p,
    m,
    seed,
    folds,
    trim_h=0.05,
    correction_order=1,
    sieve_degree=3,
):
    """Honest DR-BC score used only to learn and validate a repair.

    The published full-sample DR-BC estimator remains the reference endpoint.
    Here each observation's boundary-correction coefficient is estimated on
    other folds, preventing the repair projection from learning or validating
    on a score whose sieve coefficient used that observation.
    """
    y = np.asarray(y, dtype=float)
    response = np.asarray(response, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    m = np.asarray(m, dtype=float)
    labels = np.random.default_rng(seed).integers(0, folds, len(y))
    values = np.empty(len(y), dtype=float)
    b = response * (y - m)
    shifted = 2.0 * p - 1.0
    full_design = np.polynomial.legendre.legvander(shifted, sieve_degree)
    for fold in range(folds):
        test = labels == fold
        train = ~test
        coefficients, _, rank, _ = np.linalg.lstsq(
            full_design[train], b[train], rcond=None
        )
        if rank < sieve_degree + 1:
            raise RuntimeError("cross-fitted DR-BC sieve is rank deficient")
        trimmed = p[test] < trim_h
        untrimmed_ratio = np.where(trimmed, 0.0, b[test] / p[test])
        correction = np.zeros(int(np.sum(test)), dtype=float)
        for order in range(1, correction_order + 1):
            basis_derivative = _shifted_legendre_derivative_at_zero(sieve_degree, order)
            xi_derivative = float(np.dot(coefficients, basis_derivative))
            correction += (
                trimmed.astype(float)
                * p[test] ** (order - 1)
                * xi_derivative
                / math.factorial(order)
            )
        values[test] = m[test] + untrimmed_ratio + correction
    return values


def _cui_candidate_propensity(name, seed):
    if name == "logistic_l1":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                l1_ratio=1.0,
                solver="saga",
                C=1.0,
                max_iter=2000,
                random_state=seed,
            ),
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=20,
            n_jobs=1,
            random_state=seed,
        )
    if name == "gradient_boosting":
        return _classifier(seed, "histgb")
    raise ValueError(f"unknown Cui propensity candidate: {name}")


def _cui_candidate_outcome(name, seed):
    if name == "lasso":
        return make_pipeline(
            StandardScaler(),
            LassoCV(cv=3, max_iter=5000, n_jobs=1, random_state=seed),
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=20,
            n_jobs=1,
            random_state=seed,
        )
    if name == "gradient_boosting":
        return _regressor(seed, "histgb")
    raise ValueError(f"unknown Cui outcome candidate: {name}")


def _cui_mixed_minimax(psi_by_split):
    """Algorithm 1 mixed-minimax selector from Cui--Tchetgen Tchetgen."""
    psi = np.asarray(psi_by_split, dtype=float)
    if psi.ndim != 3:
        raise ValueError("Cui split estimates must have shape (S, K, L)")
    _, propensity_count, outcome_count = psi.shape
    risks = np.empty((propensity_count, outcome_count), dtype=float)
    for k0 in range(propensity_count):
        propensity_spread = max(
            float(np.mean((psi[:, k0, l1] - psi[:, k0, l2]) ** 2))
            for l1 in range(outcome_count)
            for l2 in range(outcome_count)
        )
        for l0 in range(outcome_count):
            outcome_spread = max(
                float(np.mean((psi[:, k1, l0] - psi[:, k2, l0]) ** 2))
                for k1 in range(propensity_count)
                for k2 in range(propensity_count)
            )
            risks[k0, l0] = propensity_spread + outcome_spread
    selected = min(
        np.ndindex(risks.shape),
        key=lambda pair: (risks[pair], pair[0], pair[1]),
    )
    return selected, risks


def _cui_selective_ml_reference(x, y, response, seed):
    """Two-fold Algorithm 1 selective ML for the MAR outcome mean."""
    propensity_names = ("logistic_l1", "random_forest", "gradient_boosting")
    outcome_names = ("lasso", "random_forest", "gradient_boosting")
    split_count = 2
    labels = np.empty(len(y), dtype=int)
    splitter = StratifiedKFold(
        n_splits=split_count,
        shuffle=True,
        random_state=seed,
    )
    for fold, (_, test_index) in enumerate(
        splitter.split(np.zeros((len(y), 1)), response)
    ):
        labels[test_index] = fold

    psi = np.empty(
        (split_count, len(propensity_names), len(outcome_names)), dtype=float
    )
    p_oof = {name: np.empty(len(y)) for name in propensity_names}
    m_oof = {name: np.empty(len(y)) for name in outcome_names}
    for fold in range(split_count):
        test = labels == fold
        train = ~test
        observed = train & (response == 1)
        if observed.sum() < 20:
            raise RuntimeError("too few observed outcomes for Cui learner library")
        for k, name in enumerate(propensity_names):
            model = _cui_candidate_propensity(name, seed + 1009 * fold + 31 * k)
            model.fit(x[train], response[train])
            p_oof[name][test] = np.clip(model.predict_proba(x[test])[:, 1], 1e-6, 1.0)
        for l, name in enumerate(outcome_names):
            model = _cui_candidate_outcome(name, seed + 2003 * fold + 37 * l)
            model.fit(x[observed], y[observed])
            m_oof[name][test] = model.predict(x[test])
        for k, propensity_name in enumerate(propensity_names):
            p_test = p_oof[propensity_name][test]
            for l, outcome_name in enumerate(outcome_names):
                m_test = m_oof[outcome_name][test]
                score = _aipw_score(y[test], response[test], p_test, m_test)
                psi[fold, k, l] = float(np.mean(score))

    (selected_k, selected_l), risks = _cui_mixed_minimax(psi)
    selected_propensity = propensity_names[selected_k]
    selected_outcome = outcome_names[selected_l]
    estimate = float(np.mean(psi[:, selected_k, selected_l]))
    return {
        "ref": np.full(len(y), estimate, dtype=float),
        "selected_p": p_oof[selected_propensity],
        "selected_m": m_oof[selected_outcome],
        "selected_propensity_learner": selected_propensity,
        "selected_outcome_learner": selected_outcome,
        "pseudo_risk": float(risks[selected_k, selected_l]),
        "split_estimates": psi[:, selected_k, selected_l].copy(),
    }


def _gl_path_stats(ref, rt_by_gamma, region_damp_grid, lepski_c):
    gammas = tuple(sorted({float(g) for g in region_damp_grid} | {0.0}))
    values = {0.0: ref}
    for gamma in gammas:
        if gamma != 0.0:
            values[gamma] = rt_by_gamma[gamma]
    stats = {}
    for gamma in gammas:
        candidate = values[gamma]
        variance = _mean_variance_of_mean(candidate)[1]
        bias_proxy = 0.0
        for higher_gamma in gammas:
            if higher_gamma <= gamma:
                continue
            contrast = values[higher_gamma] - candidate
            mean_contrast, contrast_variance = _mean_variance_of_mean(contrast)
            bias_proxy = max(
                bias_proxy,
                mean_contrast * mean_contrast - lepski_c * contrast_variance,
            )
        bias_proxy = max(0.0, bias_proxy)
        stats[gamma] = {
            "variance": variance,
            "bias_proxy": bias_proxy,
            "risk": variance + bias_proxy,
        }
    return stats


def _global_dr_risk_stats(ref_values, tau_grid, lepski_c):
    taus = tuple(sorted(float(t) for t in tau_grid))
    stats = {}
    for tau in taus:
        candidate = ref_values[tau]
        variance = _mean_variance_of_mean(candidate)[1]
        bias_proxy = 0.0
        for other_tau in taus:
            if other_tau == tau:
                continue
            contrast = ref_values[other_tau] - candidate
            mean_contrast, contrast_variance = _mean_variance_of_mean(contrast)
            bias_proxy = max(
                bias_proxy,
                mean_contrast * mean_contrast - lepski_c * contrast_variance,
            )
        bias_proxy = max(0.0, bias_proxy)
        stats[tau] = {
            "variance": variance,
            "bias_proxy": bias_proxy,
            "risk": variance + bias_proxy,
        }
    return stats


def _select_global_dr_risk(ref_values, tau_grid, lepski_c):
    dr_stats = _global_dr_risk_stats(ref_values, tau_grid, lepski_c)
    selected_tau = min(
        dr_stats,
        key=lambda t: (
            dr_stats[t]["risk"],
            dr_stats[t]["bias_proxy"],
            dr_stats[t]["variance"],
            t,
        ),
    )
    return selected_tau, dr_stats


def _select_first_lepski(ref, rt_by_gamma, region_damp_grid, lepski_c):
    selected_damp = 0.0 if 0.0 in region_damp_grid else min(region_damp_grid)
    for region_damp in sorted(g for g in region_damp_grid if g != 0.0):
        diff = rt_by_gamma[region_damp] - ref
        if len(diff) < 2:
            continue
        mean_diff, var_diff = _mean_variance_of_mean(diff)
        if var_diff > 0.0 and mean_diff * mean_diff > lepski_c * var_diff:
            return region_damp
    return selected_damp


def _select_gl_risk(ref, rt_by_gamma, region_damp_grid, lepski_c):
    gl_stats = _gl_path_stats(ref, rt_by_gamma, region_damp_grid, lepski_c)
    selected_damp = min(
        gl_stats,
        key=lambda g: (
            gl_stats[g]["risk"],
            gl_stats[g]["bias_proxy"],
            gl_stats[g]["variance"],
            g,
        ),
    )
    return selected_damp, gl_stats


def _select_observed_validation_damp(
    region_damp_grid,
    mean_damp_losses,
    damp_improvements,
    selected,
    validation_loss_se,
):
    if validation_loss_se < 0.0 or 0.0 not in region_damp_grid:
        return min(region_damp_grid, key=lambda g: (mean_damp_losses[g], g))
    eligible = [0.0]
    for region_damp in region_damp_grid:
        if region_damp == 0.0:
            continue
        improvement = np.asarray(damp_improvements[(selected, region_damp)])
        if len(improvement) < 2:
            continue
        mean_improvement, variance = _mean_variance_of_mean(improvement)
        se_improvement = math.sqrt(variance)
        if mean_improvement > validation_loss_se * se_improvement:
            eligible.append(region_damp)
    return min(eligible, key=lambda g: (mean_damp_losses[g], g))


def _crossfit_selected(
    data,
    reference_method,
    mode,
    learner,
    propensity_learner,
    repair_mode,
    tau_grid,
    folds,
    seed,
    region_damp_grid,
    validation_region_weight,
    validation_loss_se,
    selector,
    lepski_c,
):
    if reference_method not in {
        "aipw",
        "tmle",
        "ctmle",
        "ma_dr_bc",
        "cui_selective_ml",
        "glrisk",
        "glrisk_reference",
        "cui_tchetgen",
    }:
        raise ValueError(f"unknown reference_method: {reference_method}")
    x, y, response, region, true_pi, _ = data[:6]
    n = len(y)
    labels = np.random.default_rng(seed).integers(0, folds, n)
    tau_grid = tuple(float(t) for t in tau_grid)
    region_damp_grid = tuple(float(g) for g in region_damp_grid)
    ref_values = {t: np.empty(n) for t in tau_grid}
    p_values = {t: np.empty(n) for t in tau_grid}
    ref_outcome_values = {t: np.empty(n) for t in tau_grid}
    initial_outcome_values = np.empty(n)
    p_raw_values = np.empty(n)
    rt_values = {(t, g): np.empty(n) for t in tau_grid for g in region_damp_grid}
    rt_outcome_values = {
        (t, g): np.empty(n) for t in tau_grid for g in region_damp_grid
    }
    losses = {t: [] for t in tau_grid}
    damp_losses = {(t, g): [] for t in tau_grid for g in region_damp_grid}
    damp_improvements = {(t, g): [] for t in tau_grid for g in region_damp_grid}
    candidate_kind = {
        (t, g): ("reference" if g == 0.0 else "regional")
        for t in tau_grid
        for g in region_damp_grid
    }
    for fold in range(folds):
        test = labels == fold
        train = ~test
        observed = train & (response == 1)
        test_observed = test & (response == 1)
        if observed.sum() < 20:
            raise RuntimeError("too few observed outcomes in training fold")
        p_train_raw, p_test_raw = _propensity_predictions(
            x,
            response,
            true_pi,
            train,
            test,
            mode,
            seed + 101 * fold,
            propensity_learner,
        )
        observed_in_train = response[train] == 1
        model = _regressor(seed + 211 * fold, learner)
        model.fit(x[observed], y[observed])
        m_obs = model.predict(x[observed])
        m_test = model.predict(x[test])
        initial_outcome_values[test] = m_test
        p_raw_values[test] = p_test_raw
        reweighted_obs = {}
        reweighted_test = {}
        if repair_mode == "reweight":
            for region_damp in region_damp_grid:
                if region_damp == 0.0:
                    reweighted_obs[region_damp] = m_obs
                    reweighted_test[region_damp] = m_test
                    continue
                weights = 1.0 + region_damp * region[observed].astype(float)
                repair = _regressor(
                    seed + 307 * fold + int(1000 * region_damp), learner
                )
                _fit_with_sample_weight(repair, x[observed], y[observed], weights)
                reweighted_obs[region_damp] = repair.predict(x[observed])
                reweighted_test[region_damp] = repair.predict(x[test])
        elif repair_mode not in {
            "targeting",
            "if_residual",
            "regional_if_residual",
            "if_projection",
            "if_library",
        }:
            raise ValueError(f"unknown repair_mode: {repair_mode}")
        for tau in tau_grid:
            p_train = np.maximum(p_train_raw, tau)
            p_test = np.maximum(p_test_raw, tau)
            p_obs = p_train[observed_in_train]
            ref_test, _ = _targets(
                y[observed],
                p_obs,
                m_obs,
                region[observed],
                p_test,
                m_test,
                region[test],
                0.0,
            )
            if reference_method == "aipw":
                ref_value = _aipw_score(y[test], response[test], p_test, m_test)
                ref_outcome = m_test
            else:
                ref_value = ref_test
                ref_outcome = ref_test
            ref_values[tau][test] = ref_value
            p_values[tau][test] = p_test
            ref_outcome_values[tau][test] = ref_outcome
            if test_observed.any():
                pos = np.flatnonzero(test_observed[test])
                # For psi_m = m(X) + R/pi(X)*{Y-m(X)}, the m-dependent part
                # of Var(psi_m) is E[(1-pi)/pi*{m-m_star}^2].  Under MAR its
                # observable loss is R*(1-pi)/pi^2*{Y-m}^2.  This helper also
                # supplies the historical balanced-MSE weights for the
                # explicit comparison arm.
                validation_weight = _observed_validation_weights(
                    p_test[pos],
                    region[test_observed],
                    validation_region_weight,
                )
                baseline_loss = (
                    validation_weight * (y[test_observed] - ref_outcome[pos]) ** 2
                )
                losses[tau].append(
                    float(np.mean((y[test_observed] - ref_outcome[pos]) ** 2))
                )
            for region_damp in region_damp_grid:
                if repair_mode in {"targeting", "if_library"}:
                    _, rt_test = _targets(
                        y[observed],
                        p_obs,
                        m_obs,
                        region[observed],
                        p_test,
                        m_test,
                        region[test],
                        region_damp,
                        target_global=reference_method != "aipw",
                    )
                    rt_prediction = rt_test
                elif repair_mode == "reweight":
                    rt_test, _ = _targets(
                        y[observed],
                        p_obs,
                        reweighted_obs[region_damp],
                        region[observed],
                        p_test,
                        reweighted_test[region_damp],
                        region[test],
                        0.0,
                    )
                    rt_prediction = reweighted_test[region_damp]
                else:
                    # Placeholder.  Once the reference-specific OOF outcome
                    # expert is finalized below, the influence-function
                    # residual path overwrites every candidate array.
                    rt_test = ref_outcome.copy()
                    rt_prediction = ref_outcome.copy()
                regional_increment = rt_prediction - ref_outcome
                rt_value = ref_value + regional_increment
                rt_outcome = rt_prediction
                if not np.allclose(
                    rt_value - ref_value,
                    regional_increment,
                    rtol=1e-12,
                    atol=1e-12,
                ):
                    raise AssertionError("candidate endpoint is not additive")
                rt_values[(tau, region_damp)][test] = rt_value
                rt_outcome_values[(tau, region_damp)][test] = rt_outcome
                if test_observed.any():
                    pos = np.flatnonzero(test_observed[test])
                    candidate_loss = validation_weight * (
                        (y[test_observed] - rt_test[pos]) ** 2
                    )
                    damp_losses[(tau, region_damp)].extend(candidate_loss.tolist())
                    damp_improvements[(tau, region_damp)].extend(
                        (baseline_loss - candidate_loss).tolist()
                    )
    targeting_moment = float("nan")
    targeting_score_gap = float("nan")
    ma_dr_bc_stats = None
    ma_projection_score = None
    cui_selective_stats = None
    canonical_plugin_methods = {"tmle", "ctmle", "cui_tchetgen"}
    targeting_moments = {}
    targeting_score_gaps = {}
    if reference_method == "tmle" and len(tau_grid) != 1:
        raise ValueError("tmle requires one explicit propensity floor")
    if reference_method in canonical_plugin_methods:
        # The nuisance regressions above are cross-fitted, but their fold-specific
        # fluctuations were historically fit on training responders.  Re-solve the
        # fluctuation on the pooled OOF predictions so every emitted plug-in
        # reference satisfies its empirical targeting equation.
        observed = response == 1
        m_oof = initial_outcome_values
        for tau in tau_grid:
            p_oof = np.maximum(p_values[tau], 1e-12)
            clever = 1.0 / p_oof[observed]
            denominator = float(np.dot(clever, clever))
            if denominator <= 0.0:
                raise RuntimeError(
                    f"{reference_method} fluctuation has no observed support"
                )
            epsilon = float(np.dot(clever, y[observed] - m_oof[observed]) / denominator)
            targeted = m_oof + epsilon / p_oof
            moment = float(np.mean(response.astype(float) * (y - targeted) / p_oof))
            moment_scale = max(
                1.0,
                float(np.mean(np.abs(response.astype(float) * (y - targeted) / p_oof))),
            )
            if abs(moment) > 1e-10 * moment_scale:
                raise AssertionError(
                    f"pooled OOF {reference_method} targeting moment is not zero"
                )
            targeted_plugin = float(np.mean(targeted))
            targeted_aipw = float(np.mean(_aipw_score(y, response, p_oof, targeted)))
            score_gap = targeted_plugin - targeted_aipw
            if abs(score_gap) > 1e-10 * max(1.0, abs(targeted_plugin)):
                raise AssertionError(
                    f"{reference_method} plug-in does not equal targeted AIPW"
                )
            targeting_moments[tau] = moment
            targeting_score_gaps[tau] = score_gap

            previous_ref = ref_values[tau]
            previous_ref_outcome = ref_outcome_values[tau]
            ref_values[tau] = targeted.copy()
            ref_outcome_values[tau] = targeted.copy()
            if reference_method == "tmle":
                losses[tau] = [float(np.mean((y[observed] - targeted[observed]) ** 2))]
            validation_weight = _observed_validation_weights(
                p_oof[observed], region[observed], validation_region_weight
            )
            baseline_loss = validation_weight * (y[observed] - targeted[observed]) ** 2
            if repair_mode in {"targeting", "if_library"}:
                regional_clever = region[observed].astype(float) / p_oof[observed]
                regional_denominator = float(np.dot(regional_clever, regional_clever))
                regional_alpha = (
                    float(
                        np.dot(
                            regional_clever,
                            y[observed] - targeted[observed],
                        )
                        / regional_denominator
                    )
                    if regional_denominator > 0.0
                    else 0.0
                )
            for region_damp in region_damp_grid:
                if repair_mode in {"targeting", "if_library"}:
                    regional_increment = (
                        region_damp * regional_alpha * region.astype(float) / p_oof
                    )
                    candidate = targeted + regional_increment
                    rt_values[(tau, region_damp)] = candidate
                    rt_outcome_values[(tau, region_damp)] = candidate.copy()
                else:
                    endpoint_increment = rt_values[(tau, region_damp)] - previous_ref
                    outcome_increment = (
                        rt_outcome_values[(tau, region_damp)] - previous_ref_outcome
                    )
                    rt_values[(tau, region_damp)] = targeted + endpoint_increment
                    rt_outcome_values[(tau, region_damp)] = targeted + outcome_increment
                    candidate = rt_outcome_values[(tau, region_damp)]
                candidate_loss = (
                    validation_weight * (y[observed] - candidate[observed]) ** 2
                )
                damp_losses[(tau, region_damp)] = candidate_loss.tolist()
                damp_improvements[(tau, region_damp)] = (
                    baseline_loss - candidate_loss
                ).tolist()
    if reference_method == "ma_dr_bc":
        if len(tau_grid) != 1:
            raise ValueError("ma_dr_bc requires one explicit trimming threshold")
        trim_h = min(tau_grid)
        p_raw = np.clip(p_raw_values, 1e-12, 1.0)
        m_oof = initial_outcome_values
        dr_bc_values, ma_dr_bc_stats = _ma_dr_bc_reference(
            y,
            response,
            p_raw,
            m_oof,
            trim_h=trim_h,
            correction_order=1,
            sieve_degree=3,
        )
        ma_projection_score = _crossfit_ma_dr_bc_score(
            y,
            response,
            p_raw,
            m_oof,
            seed + 11003,
            folds,
            trim_h=trim_h,
            correction_order=1,
            sieve_degree=3,
        )
        p_repair = np.maximum(p_raw, trim_h)
        p_values[trim_h] = p_repair
        ref_values[trim_h] = dr_bc_values
        ref_outcome_values[trim_h] = m_oof.copy()
        observed = response == 1
        losses[trim_h] = [float(np.mean((y[observed] - m_oof[observed]) ** 2))]
        validation_weight = _observed_validation_weights(
            p_repair[observed], region[observed], validation_region_weight
        )
        baseline_loss = validation_weight * (y[observed] - m_oof[observed]) ** 2
        regional_clever = region[observed].astype(float) / p_repair[observed]
        regional_denominator = float(np.dot(regional_clever, regional_clever))
        regional_alpha = (
            float(
                np.dot(regional_clever, y[observed] - m_oof[observed])
                / regional_denominator
            )
            if regional_denominator > 0.0
            else 0.0
        )
        for region_damp in region_damp_grid:
            regional_increment = (
                region_damp * regional_alpha * region.astype(float) / p_repair
            )
            candidate_value = dr_bc_values + regional_increment
            candidate_outcome = m_oof + regional_increment
            if not np.allclose(
                candidate_value - dr_bc_values,
                regional_increment,
                rtol=1e-12,
                atol=1e-12,
            ):
                raise AssertionError("DR-BC candidate endpoint is not additive")
            rt_values[(trim_h, region_damp)] = candidate_value
            rt_outcome_values[(trim_h, region_damp)] = candidate_outcome
            candidate_loss = (
                validation_weight * (y[observed] - candidate_outcome[observed]) ** 2
            )
            damp_losses[(trim_h, region_damp)] = candidate_loss.tolist()
            damp_improvements[(trim_h, region_damp)] = (
                baseline_loss - candidate_loss
            ).tolist()
    if reference_method == "cui_selective_ml":
        if mode != "estimated":
            raise ValueError("cui_selective_ml requires estimated nuisances")
        selected_key = min(tau_grid)
        cui_selective_stats = _cui_selective_ml_reference(x, y, response, seed + 7001)
        selected_p = np.asarray(cui_selective_stats["selected_p"], dtype=float)
        selected_m = np.asarray(cui_selective_stats["selected_m"], dtype=float)
        selected_ref = np.asarray(cui_selective_stats["ref"], dtype=float)
        p_values[selected_key] = selected_p
        ref_values[selected_key] = selected_ref
        ref_outcome_values[selected_key] = selected_m
        observed = response == 1
        losses[selected_key] = [
            float(np.mean((y[observed] - selected_m[observed]) ** 2))
        ]
        validation_weight = _observed_validation_weights(
            selected_p[observed], region[observed], validation_region_weight
        )
        baseline_loss = validation_weight * (y[observed] - selected_m[observed]) ** 2
        regional_clever = region[observed].astype(float) / selected_p[observed]
        regional_denominator = float(np.dot(regional_clever, regional_clever))
        regional_alpha = (
            float(
                np.dot(regional_clever, y[observed] - selected_m[observed])
                / regional_denominator
            )
            if regional_denominator > 0.0
            else 0.0
        )
        for region_damp in region_damp_grid:
            regional_increment = (
                region_damp * regional_alpha * region.astype(float) / selected_p
            )
            candidate_value = selected_ref + regional_increment
            candidate_outcome = selected_m + regional_increment
            if not np.allclose(
                candidate_value - selected_ref,
                regional_increment,
                rtol=1e-12,
                atol=1e-12,
            ):
                raise AssertionError("Cui candidate endpoint is not additive")
            rt_values[(selected_key, region_damp)] = candidate_value
            rt_outcome_values[(selected_key, region_damp)] = candidate_outcome
            candidate_loss = (
                validation_weight * (y[observed] - candidate_outcome[observed]) ** 2
            )
            damp_losses[(selected_key, region_damp)] = candidate_loss.tolist()
            damp_improvements[(selected_key, region_damp)] = (
                baseline_loss - candidate_loss
            ).tolist()
    if repair_mode == "if_projection":
        projection_taus = (
            (min(tau_grid),)
            if reference_method in {"tmle", "ma_dr_bc", "cui_selective_ml"}
            else tau_grid
        )
        for tau in projection_taus:
            p_oof = np.clip(np.asarray(p_values[tau], dtype=float), 1e-6, 1.0)
            base_score = np.asarray(ref_values[tau], dtype=float)
            selection_score = (
                np.asarray(ma_projection_score, dtype=float)
                if reference_method == "ma_dr_bc"
                else base_score
            )
            base_outcome = np.asarray(ref_outcome_values[tau], dtype=float)
            projection = _crossfit_influence_projection(
                x,
                response,
                p_oof,
                selection_score,
                learner,
                seed + 14009 + int(1000 * tau),
                folds,
            )
            control = (1.0 - response.astype(float) / p_oof) * projection
            # All candidates preserve the same population target.  Use the
            # reference score mean as their common finite-sample center; giving
            # each candidate its own center would hide a dangerous nonzero
            # correction mean.
            common_center = float(np.mean(selection_score))
            baseline_centered = selection_score - common_center
            baseline_loss = baseline_centered * baseline_centered
            losses[tau] = [float(np.mean(baseline_loss))]
            for region_damp in region_damp_grid:
                candidate_value = base_score + region_damp * control
                candidate_selection_score = selection_score + region_damp * control
                candidate_centered = candidate_selection_score - common_center
                candidate_loss = candidate_centered * candidate_centered
                rt_values[(tau, region_damp)] = candidate_value
                # This generic control-variate repair changes the expert score,
                # not a uniquely defined outcome regression.
                rt_outcome_values[(tau, region_damp)] = base_outcome.copy()
                damp_losses[(tau, region_damp)] = candidate_loss.tolist()
                damp_improvements[(tau, region_damp)] = (
                    baseline_loss - candidate_loss
                ).tolist()
                candidate_kind[(tau, region_damp)] = (
                    "reference" if region_damp == 0.0 else "if_projection"
                )
    if repair_mode in {"if_residual", "regional_if_residual"}:
        residual_taus = (
            (min(tau_grid),)
            if reference_method in {"tmle", "ma_dr_bc", "cui_selective_ml"}
            else tau_grid
        )
        for tau in residual_taus:
            p_oof = np.clip(np.asarray(p_values[tau], dtype=float), 1e-6, 1.0)
            base_outcome = np.asarray(ref_outcome_values[tau], dtype=float)
            base_endpoint = np.asarray(ref_values[tau], dtype=float)
            correction = _crossfit_weighted_residual_correction(
                x,
                y,
                response,
                p_oof,
                base_outcome,
                learner,
                seed + 16001 + int(1000 * tau),
                folds,
            )
            # The two residual rules are intentionally identical except for
            # this support mask.  Both learn the same honest direction, use
            # the same score-risk gate, and traverse the same damping path.
            # This makes global-versus-regional a complete protocol-level
            # comparison rather than an estimator-specific adapter choice.
            if repair_mode == "regional_if_residual":
                correction = correction * region.astype(float)

            if reference_method == "ma_dr_bc":
                trim_h = float(tau)
                p_endpoint = np.clip(np.asarray(p_raw_values), 1e-12, 1.0)
                base_selection_score = np.asarray(ma_projection_score, dtype=float)
            else:
                base_selection_score = _aipw_score(y, response, p_oof, base_outcome)
            common_center = float(np.mean(base_selection_score))
            baseline_loss = (base_selection_score - common_center) ** 2
            losses[tau] = [float(np.mean(baseline_loss))]

            for region_damp in region_damp_grid:
                candidate_outcome = base_outcome + region_damp * correction
                if reference_method == "ma_dr_bc":
                    candidate_endpoint, _ = _ma_dr_bc_reference(
                        y,
                        response,
                        p_endpoint,
                        candidate_outcome,
                        trim_h=trim_h,
                        correction_order=1,
                        sieve_degree=3,
                    )
                    candidate_selection_score = _crossfit_ma_dr_bc_score(
                        y,
                        response,
                        p_endpoint,
                        candidate_outcome,
                        seed + 11003,
                        folds,
                        trim_h=trim_h,
                        correction_order=1,
                        sieve_degree=3,
                    )
                else:
                    candidate_aipw = _aipw_score(y, response, p_oof, candidate_outcome)
                    candidate_endpoint = (
                        base_endpoint + candidate_aipw - base_selection_score
                    )
                    candidate_selection_score = candidate_aipw
                candidate_loss = (candidate_selection_score - common_center) ** 2
                rt_values[(tau, region_damp)] = candidate_endpoint
                rt_outcome_values[(tau, region_damp)] = candidate_outcome
                damp_losses[(tau, region_damp)] = candidate_loss.tolist()
                damp_improvements[(tau, region_damp)] = (
                    baseline_loss - candidate_loss
                ).tolist()
                candidate_kind[(tau, region_damp)] = (
                    "reference" if region_damp == 0.0 else repair_mode
                )
    if repair_mode == "if_library":
        residual_taus = (
            (min(tau_grid),)
            if reference_method in {"tmle", "ma_dr_bc", "cui_selective_ml"}
            else tau_grid
        )
        observed = response == 1
        for tau in residual_taus:
            p_oof = np.clip(np.asarray(p_values[tau], dtype=float), 1e-6, 1.0)
            base_outcome = np.asarray(ref_outcome_values[tau], dtype=float)
            base_score = np.asarray(ref_values[tau], dtype=float)
            correction = _crossfit_weighted_residual_correction(
                x,
                y,
                response,
                p_oof,
                base_outcome,
                learner,
                seed + 12001 + int(1000 * tau),
                folds,
            )
            validation_weight = _observed_validation_weights(
                p_oof[observed], region[observed], validation_region_weight
            )
            baseline_loss = (
                validation_weight * (y[observed] - base_outcome[observed]) ** 2
            )
            losses[tau] = [float(np.mean(baseline_loss))]
            base_aipw = _aipw_score(y, response, p_oof, base_outcome)
            for region_damp in region_damp_grid:
                regional_value = np.asarray(
                    rt_values[(tau, region_damp)], dtype=float
                ).copy()
                regional_outcome = np.asarray(
                    rt_outcome_values[(tau, region_damp)], dtype=float
                ).copy()
                regional_loss = np.asarray(damp_losses[(tau, region_damp)], dtype=float)
                candidate_outcome = base_outcome + region_damp * correction
                score_contrast = (
                    _aipw_score(y, response, p_oof, candidate_outcome) - base_aipw
                )
                candidate_value = base_score + score_contrast
                rt_values[(tau, region_damp)] = candidate_value
                rt_outcome_values[(tau, region_damp)] = candidate_outcome
                candidate_loss = (
                    validation_weight * (y[observed] - candidate_outcome[observed]) ** 2
                )
                regional_is_eligible = False
                if (
                    len(regional_loss) == len(candidate_loss)
                    and len(candidate_loss) >= 2
                ):
                    # The influence-residual correction is the theory-backed
                    # default.  Charge the optional regional candidate for the
                    # extra library choice: it must beat the generic candidate
                    # by the same one-SE margin used against the reference.
                    regional_improvement = candidate_loss - regional_loss
                    mean_improvement, variance = _mean_variance_of_mean(
                        regional_improvement
                    )
                    regional_is_eligible = mean_improvement > (
                        validation_loss_se * math.sqrt(variance)
                    )
                if regional_is_eligible:
                    candidate_value = regional_value
                    candidate_outcome = regional_outcome
                    candidate_loss = regional_loss
                    candidate_kind[(tau, region_damp)] = (
                        "reference" if region_damp == 0.0 else "regional"
                    )
                else:
                    candidate_kind[(tau, region_damp)] = (
                        "reference" if region_damp == 0.0 else "if_residual"
                    )
                damp_losses[(tau, region_damp)] = candidate_loss.tolist()
                damp_improvements[(tau, region_damp)] = (
                    baseline_loss - candidate_loss
                ).tolist()
    mean_losses = {
        t: (float(np.mean(v)) if v else float("inf")) for t, v in losses.items()
    }
    global_dr_stats = None
    if reference_method in {"tmle", "ma_dr_bc", "cui_selective_ml"}:
        selected = min(tau_grid)
    elif reference_method == "cui_tchetgen":
        selected, global_dr_stats = _select_global_dr_risk(
            ref_values,
            tau_grid,
            lepski_c,
        )
    else:
        selected = min(tau_grid, key=lambda t: (mean_losses[t], t))
    if reference_method in canonical_plugin_methods:
        targeting_moment = targeting_moments[selected]
        targeting_score_gap = targeting_score_gaps[selected]
    mean_damp_losses = {
        g: (
            float(np.mean(damp_losses[(selected, g)]))
            if damp_losses[(selected, g)]
            else float("inf")
        )
        for g in region_damp_grid
    }
    rt_by_gamma = {g: rt_values[(selected, g)] for g in region_damp_grid}
    ref = ref_values[selected]
    gl_stats = None
    path_selector = (
        "glrisk" if reference_method in {"glrisk", "glrisk_reference"} else selector
    )
    if path_selector == "lepski":
        selected_damp = _select_first_lepski(
            ref,
            rt_by_gamma,
            region_damp_grid,
            lepski_c,
        )
    elif path_selector == "glrisk":
        selected_damp, gl_stats = _select_gl_risk(
            ref,
            rt_by_gamma,
            region_damp_grid,
            lepski_c,
        )
    elif path_selector == "obsval":
        selected_damp = _select_observed_validation_damp(
            region_damp_grid,
            mean_damp_losses,
            damp_improvements,
            selected,
            validation_loss_se,
        )
    else:
        raise ValueError(f"unknown selector: {path_selector}")
    selected_value = (
        ref_values[selected]
        if selected_damp == 0.0
        else rt_values[(selected, selected_damp)]
    )
    selected_outcome = (
        ref_outcome_values[selected]
        if selected_damp == 0.0
        else rt_outcome_values[(selected, selected_damp)]
    )
    returned_ref = ref_values[selected]
    returned_ref_outcome = ref_outcome_values[selected]
    returned_rt = selected_value
    returned_rt_outcome = selected_outcome
    if reference_method == "glrisk_reference":
        if 1.0 not in region_damp_grid:
            raise ValueError("glrisk_reference requires gamma=1 in region_damp_grid")
        returned_ref = selected_value
        returned_ref_outcome = selected_outcome
        returned_rt = rt_values[(selected, 1.0)]
        returned_rt_outcome = rt_outcome_values[(selected, 1.0)]
        full_increment = rt_values[(selected, 1.0)] - ref_values[selected]
        remaining_increment = returned_rt - returned_ref
        if not np.allclose(
            remaining_increment,
            (1.0 - selected_damp) * full_increment,
            rtol=1e-10,
            atol=1e-10,
        ):
            raise AssertionError(
                "GL reference does not leave the expected path remainder"
            )
        if selected_damp != min(
            gl_stats,
            key=lambda g: (
                gl_stats[g]["risk"],
                gl_stats[g]["bias_proxy"],
                gl_stats[g]["variance"],
                g,
            ),
        ):
            raise AssertionError("emitted GL reference does not minimize GL risk")
    return {
        "ref": returned_ref,
        "rt": returned_rt,
        "selected_p": p_values[selected],
        "ref_outcome": returned_ref_outcome,
        "rt_outcome": returned_rt_outcome,
        "selected_tau": (
            float("nan") if reference_method == "cui_selective_ml" else selected
        ),
        "selected_region_damp": selected_damp,
        "selected_repair_kind": candidate_kind[(selected, selected_damp)],
        "targeting_moment": targeting_moment,
        "targeting_score_gap": targeting_score_gap,
        "ma_trimmed_fraction": (
            ma_dr_bc_stats["trimmed_fraction"]
            if ma_dr_bc_stats is not None
            else float("nan")
        ),
        "ma_xi_derivative_1": (
            ma_dr_bc_stats["xi_derivative_1"]
            if ma_dr_bc_stats is not None
            else float("nan")
        ),
        "ma_bias_correction_mean": (
            ma_dr_bc_stats["bias_correction_mean"]
            if ma_dr_bc_stats is not None
            else float("nan")
        ),
        "cui_selected_propensity_learner": (
            cui_selective_stats["selected_propensity_learner"]
            if cui_selective_stats is not None
            else ""
        ),
        "cui_selected_outcome_learner": (
            cui_selective_stats["selected_outcome_learner"]
            if cui_selective_stats is not None
            else ""
        ),
        "cui_pseudo_risk": (
            cui_selective_stats["pseudo_risk"]
            if cui_selective_stats is not None
            else float("nan")
        ),
        "gl_original_bias_proxy": (
            gl_stats[0.0]["bias_proxy"] if gl_stats is not None else float("nan")
        ),
        "gl_selected_bias_proxy": (
            gl_stats[selected_damp]["bias_proxy"]
            if gl_stats is not None
            else float("nan")
        ),
        "gl_selected_variance_proxy": (
            gl_stats[selected_damp]["variance"]
            if gl_stats is not None
            else float("nan")
        ),
        "gl_selected_risk_proxy": (
            gl_stats[selected_damp]["risk"] if gl_stats is not None else float("nan")
        ),
        "global_dr_selected_bias_proxy": (
            global_dr_stats[selected]["bias_proxy"]
            if global_dr_stats is not None
            else float("nan")
        ),
        "global_dr_selected_variance_proxy": (
            global_dr_stats[selected]["variance"]
            if global_dr_stats is not None
            else float("nan")
        ),
        "global_dr_selected_risk_proxy": (
            global_dr_stats[selected]["risk"]
            if global_dr_stats is not None
            else float("nan")
        ),
    }


def _bootstrap_cov(
    data,
    analysis_region,
    reference_method,
    mode,
    learner,
    propensity_learner,
    repair_mode,
    tau_grid,
    folds,
    seed,
    bootstraps,
    region_damp_grid,
    validation_region_weight,
    validation_loss_se,
    selector,
    lepski_c,
    region_quantile,
    region_min_observed,
    region_kappa_floor,
    selector_ablation,
    region_detector_c,
):
    rng = np.random.default_rng(seed)
    n = len(data[0])
    means = []
    for boot in range(bootstraps):
        idx = rng.integers(0, n, n)
        resampled_raw = tuple(v[idx] if isinstance(v, np.ndarray) else v for v in data)
        x, y, response, response_region, true_pi, theta, mu = resampled_raw
        analysis_mask = _analysis_region(
            x,
            y,
            response,
            response_region,
            true_pi,
            analysis_region,
            mode,
            learner,
            propensity_learner,
            seed + 2003 * (boot + 1),
            region_quantile,
            region_min_observed,
            region_kappa_floor,
            selector_ablation,
            region_detector_c,
            folds,
        )
        resampled = (
            x,
            y,
            response,
            analysis_mask,
            true_pi,
            theta,
            mu,
        )
        try:
            fit = _crossfit_selected(
                resampled,
                reference_method,
                mode,
                learner,
                propensity_learner,
                repair_mode,
                tau_grid,
                folds,
                seed + 1009 * (boot + 1),
                region_damp_grid,
                validation_region_weight,
                validation_loss_se,
                selector,
                lepski_c,
            )
        except Exception:
            continue
        means.append([float(fit["ref"].mean()), float(fit["rt"].mean())])
    if len(means) < max(3, bootstraps // 2):
        return None
    return np.cov(np.asarray(means), rowvar=False, ddof=1)


def _one_rep(
    design,
    mar_design,
    n,
    epsilon,
    strength,
    analysis_region,
    reference_method,
    mode,
    learner,
    propensity_learner,
    repair_mode,
    rep,
    c,
    tau_grid,
    folds,
    bootstraps,
    seed0,
    region_damp_grid,
    validation_region_weight,
    validation_loss_se,
    selector,
    lepski_c,
    region_quantile,
    region_min_observed,
    region_kappa_floor,
    selector_ablation,
    region_detector_c,
):
    seed = (
        seed0
        + n * 1009
        + int(epsilon * 1000) * 100003
        + _design_strength_code(strength) * 10007
        + rep * 37
        + _design_seed_offset(design)
        + _mar_seed_offset(mar_design)
    )
    raw_data = make_data(n, epsilon, strength, design, seed, mar_design)
    x, y, response, region, true_pi, theta, mu = raw_data
    analysis_mask = _analysis_region(
        x,
        y,
        response,
        region,
        true_pi,
        analysis_region,
        mode,
        learner,
        propensity_learner,
        seed + 9091,
        region_quantile,
        region_min_observed,
        region_kappa_floor,
        selector_ablation,
        region_detector_c,
        folds,
    )
    overlap = float(np.sum(analysis_mask & region))
    analysis_mass = float(np.mean(analysis_mask))
    true_region_mass = float(np.mean(region))
    region_overlap_precision = (
        overlap / float(np.sum(analysis_mask))
        if np.any(analysis_mask)
        else float("nan")
    )
    region_overlap_recall = (
        overlap / float(np.sum(region)) if np.any(region) else float("nan")
    )
    data = (
        x,
        y,
        response,
        analysis_mask,
        true_pi,
        theta,
        mu,
    )
    theta = data[5]
    fit = _crossfit_selected(
        data,
        reference_method,
        mode,
        learner,
        propensity_learner,
        repair_mode,
        tau_grid,
        folds,
        seed + 17,
        region_damp_grid,
        validation_region_weight,
        validation_loss_se,
        selector,
        lepski_c,
    )
    cov = _bootstrap_cov(
        raw_data,
        analysis_region,
        reference_method,
        mode,
        learner,
        propensity_learner,
        repair_mode,
        tau_grid,
        folds,
        seed + 1701,
        bootstraps,
        region_damp_grid,
        validation_region_weight,
        validation_loss_se,
        selector,
        lepski_c,
        region_quantile,
        region_min_observed,
        region_kappa_floor,
        selector_ablation,
        region_detector_c,
    )
    ref_est = float(fit["ref"].mean())
    rt_est = float(fit["rt"].mean())
    delta = rt_est - ref_est
    if cov is None:
        vd = gamma = float("nan")
    else:
        vd = float(cov[1, 1] + cov[0, 0] - 2.0 * cov[0, 1])
        gamma = float(cov[0, 1] - cov[0, 0])
    weight = (
        max(0.0, 1.0 - c * max(vd, 0.0) / (delta * delta))
        if delta and np.isfinite(vd)
        else 0.0
    )
    ref_error = ref_est - theta
    selected_p = np.maximum(np.asarray(fit["selected_p"], dtype=float), 1e-12)
    kappa = 1.0 - true_pi / selected_p
    ref_outcome = np.asarray(fit["ref_outcome"], dtype=float)
    rt_outcome = np.asarray(fit["rt_outcome"], dtype=float)

    def weighted_bias(mask, prediction):
        return float(np.mean(mask.astype(float) * kappa * (prediction - mu)))

    def underfit_rmse(mask, prediction):
        if not np.any(mask):
            return float("nan")
        return float(np.sqrt(np.mean((prediction[mask] - mu[mask]) ** 2)))

    analysis_observed = int(np.sum(response.astype(bool) & analysis_mask))
    true_region_observed = int(np.sum(response.astype(bool) & region))
    analysis_bias_ref = weighted_bias(analysis_mask, ref_outcome)
    analysis_bias_rt = weighted_bias(analysis_mask, rt_outcome)
    true_region_bias_ref = weighted_bias(region, ref_outcome)
    true_region_bias_rt = weighted_bias(region, rt_outcome)
    return {
        "ref_error": ref_error,
        "rt_error": rt_est - theta,
        "shrink_error": ref_error + weight * delta,
        "delta": delta,
        "vd": vd,
        "gamma": gamma,
        "weight": weight,
        "selected_tau": float(fit["selected_tau"]),
        "selected_region_damp": float(fit["selected_region_damp"]),
        "selected_repair_kind": fit["selected_repair_kind"],
        "targeting_moment": float(fit["targeting_moment"]),
        "targeting_score_gap": float(fit["targeting_score_gap"]),
        "ma_trimmed_fraction": float(fit["ma_trimmed_fraction"]),
        "ma_xi_derivative_1": float(fit["ma_xi_derivative_1"]),
        "ma_bias_correction_mean": float(fit["ma_bias_correction_mean"]),
        "cui_selected_propensity_learner": fit["cui_selected_propensity_learner"],
        "cui_selected_outcome_learner": fit["cui_selected_outcome_learner"],
        "cui_pseudo_risk": float(fit["cui_pseudo_risk"]),
        "gl_original_bias_proxy": float(fit["gl_original_bias_proxy"]),
        "gl_selected_bias_proxy": float(fit["gl_selected_bias_proxy"]),
        "gl_selected_variance_proxy": float(fit["gl_selected_variance_proxy"]),
        "gl_selected_risk_proxy": float(fit["gl_selected_risk_proxy"]),
        "global_dr_selected_bias_proxy": float(fit["global_dr_selected_bias_proxy"]),
        "global_dr_selected_variance_proxy": float(
            fit["global_dr_selected_variance_proxy"]
        ),
        "global_dr_selected_risk_proxy": float(fit["global_dr_selected_risk_proxy"]),
        "analysis_region_mass": analysis_mass,
        "true_region_mass": true_region_mass,
        "region_overlap_precision": region_overlap_precision,
        "region_overlap_recall": region_overlap_recall,
        "analysis_observed": analysis_observed,
        "true_region_observed": true_region_observed,
        "analysis_response_rate": (
            float(np.mean(response[analysis_mask]))
            if np.any(analysis_mask)
            else float("nan")
        ),
        "true_region_response_rate": (
            float(np.mean(response[region])) if np.any(region) else float("nan")
        ),
        "analysis_bias_ref": analysis_bias_ref,
        "analysis_bias_repair": analysis_bias_rt,
        "analysis_abs_bias_reduction": abs(analysis_bias_ref) - abs(analysis_bias_rt),
        "true_region_bias_ref": true_region_bias_ref,
        "true_region_bias_repair": true_region_bias_rt,
        "true_region_abs_bias_reduction": abs(true_region_bias_ref)
        - abs(true_region_bias_rt),
        "analysis_underfit_rmse_ref": underfit_rmse(analysis_mask, ref_outcome),
        "analysis_underfit_rmse_repair": underfit_rmse(analysis_mask, rt_outcome),
        "true_region_underfit_rmse_ref": underfit_rmse(region, ref_outcome),
        "true_region_underfit_rmse_repair": underfit_rmse(region, rt_outcome),
    }


def _finite_mean(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def _append_csv_row(path: Path, row) -> None:
    write_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def run_cell(cell, reps, progress_every, rep_log_path):
    (
        design,
        mar_design,
        n,
        epsilon,
        strength,
        analysis_region,
        reference_method,
        mode,
        learner,
        propensity_learner,
        repair_mode,
        c,
        tau_grid,
        folds,
        bootstraps,
        seed,
        region_damp_grid,
        validation_region_weight,
        validation_loss_se,
        selector,
        lepski_c,
        region_quantile,
        region_min_observed,
        region_kappa_floor,
        selector_ablation,
        region_detector_c,
    ) = cell
    effective_selector = (
        "glrisk" if reference_method in {"glrisk", "glrisk_reference"} else selector
    )
    label = (
        f"reference_method={reference_method} learner={learner} "
        f"propensity_learner={propensity_learner} "
        f"repair_mode={repair_mode} selector={effective_selector} "
        f"selector_ablation={selector_ablation} region_detector_c={region_detector_c} "
        f"mar={mar_design} design={design} region={analysis_region} n={n} eps={epsilon} "
        f"strength={strength} mode={mode} kappa_floor={region_kappa_floor}"
    )
    _log(f"CELL_START {label} reps={reps} bootstraps={bootstraps} folds={folds}")
    start = time.monotonic()
    recs = []
    for rep in range(reps):
        record = _one_rep(
            design,
            mar_design,
            n,
            epsilon,
            strength,
            analysis_region,
            reference_method,
            mode,
            learner,
            propensity_learner,
            repair_mode,
            rep,
            c,
            tau_grid,
            folds,
            bootstraps,
            seed,
            region_damp_grid,
            validation_region_weight,
            validation_loss_se,
            effective_selector,
            lepski_c,
            region_quantile,
            region_min_observed,
            region_kappa_floor,
            selector_ablation,
            region_detector_c,
        )
        recs.append(record)
        elapsed_s = time.monotonic() - start
        _append_csv_row(
            rep_log_path,
            {
                "design": design,
                "mar_design": mar_design,
                "n": n,
                "epsilon": epsilon,
                "strength": strength,
                "analysis_region": analysis_region,
                "reference_method": reference_method,
                "propensity_mode": mode,
                "learner": learner,
                "propensity_learner": propensity_learner,
                "repair_mode": repair_mode,
                "region_damp": _region_damp(),
                "region_damp_grid": "|".join(str(g) for g in region_damp_grid),
                "validation_risk": _validation_risk(),
                "validation_region_weight": validation_region_weight,
                "validation_loss_se": validation_loss_se,
                "selector": effective_selector,
                "lepski_c": lepski_c,
                "region_quantile": region_quantile,
                "region_min_observed": region_min_observed,
                "region_kappa_floor": region_kappa_floor,
                "region_selector_ablation": selector_ablation,
                "region_detector_c": region_detector_c,
                "reps": reps,
                "bootstraps": bootstraps,
                "rep": rep + 1,
                "elapsed_s": elapsed_s,
                **record,
            },
        )
        _log(
            f"REP_WRITTEN {label} rep={rep + 1}/{reps} "
            f"weight={record['weight']:.3g} gamma={record['selected_region_damp']:.3g} "
            f"tau={record['selected_tau']:.3g} "
            f"elapsed_s={elapsed_s:.1f}"
        )
        if (rep + 1) % progress_every == 0 or rep + 1 == reps:
            _log(
                f"REP_DONE {label} rep={rep + 1}/{reps} elapsed_s={time.monotonic() - start:.1f}"
            )
    ref = np.array([r["ref_error"] for r in recs])
    rt = np.array([r["rt_error"] for r in recs])
    shrink = np.array([r["shrink_error"] for r in recs])
    delta = np.array([r["delta"] for r in recs])
    vd = np.array([r["vd"] for r in recs])
    gamma = np.array([r["gamma"] for r in recs])
    weights = np.array([r["weight"] for r in recs])
    tau = np.array([r["selected_tau"] for r in recs])
    selected_region_damp = np.array([r["selected_region_damp"] for r in recs])
    gl_original_bias_proxy = np.array([r["gl_original_bias_proxy"] for r in recs])
    gl_selected_bias_proxy = np.array([r["gl_selected_bias_proxy"] for r in recs])
    gl_selected_variance_proxy = np.array(
        [r["gl_selected_variance_proxy"] for r in recs]
    )
    gl_selected_risk_proxy = np.array([r["gl_selected_risk_proxy"] for r in recs])
    global_dr_selected_bias_proxy = np.array(
        [r["global_dr_selected_bias_proxy"] for r in recs]
    )
    global_dr_selected_variance_proxy = np.array(
        [r["global_dr_selected_variance_proxy"] for r in recs]
    )
    global_dr_selected_risk_proxy = np.array(
        [r["global_dr_selected_risk_proxy"] for r in recs]
    )
    ma_trimmed_fraction = np.array([r["ma_trimmed_fraction"] for r in recs])
    ma_xi_derivative_1 = np.array([r["ma_xi_derivative_1"] for r in recs])
    ma_bias_correction_mean = np.array([r["ma_bias_correction_mean"] for r in recs])
    analysis_region_mass = np.array([r["analysis_region_mass"] for r in recs])
    true_region_mass = np.array([r["true_region_mass"] for r in recs])
    region_overlap_precision = np.array([r["region_overlap_precision"] for r in recs])
    region_overlap_recall = np.array([r["region_overlap_recall"] for r in recs])
    analysis_observed = np.array([r["analysis_observed"] for r in recs])
    true_region_observed = np.array([r["true_region_observed"] for r in recs])
    analysis_response_rate = np.array([r["analysis_response_rate"] for r in recs])
    true_region_response_rate = np.array([r["true_region_response_rate"] for r in recs])
    analysis_bias_ref = np.array([r["analysis_bias_ref"] for r in recs])
    analysis_bias_repair = np.array([r["analysis_bias_repair"] for r in recs])
    analysis_abs_bias_reduction = np.array(
        [r["analysis_abs_bias_reduction"] for r in recs]
    )
    true_region_bias_ref = np.array([r["true_region_bias_ref"] for r in recs])
    true_region_bias_repair = np.array([r["true_region_bias_repair"] for r in recs])
    true_region_abs_bias_reduction = np.array(
        [r["true_region_abs_bias_reduction"] for r in recs]
    )
    analysis_underfit_rmse_ref = np.array(
        [r["analysis_underfit_rmse_ref"] for r in recs]
    )
    analysis_underfit_rmse_repair = np.array(
        [r["analysis_underfit_rmse_repair"] for r in recs]
    )
    true_region_underfit_rmse_ref = np.array(
        [r["true_region_underfit_rmse_ref"] for r in recs]
    )
    true_region_underfit_rmse_repair = np.array(
        [r["true_region_underfit_rmse_repair"] for r in recs]
    )
    ref_mse = float(np.mean(ref**2))
    rt_mse = float(np.mean(rt**2))
    shrink_mse = float(np.mean(shrink**2))
    ref_bias = float(ref.mean())
    repair_bias = float(rt.mean())
    mean_delta = float(delta.mean())
    mean_vd = _finite_mean(vd)
    m_snr = (
        abs(mean_delta) / math.sqrt(mean_vd)
        if mean_vd == mean_vd and mean_vd > 0
        else float("nan")
    )
    row = {
        "design": design,
        "mar_design": mar_design,
        "n": n,
        "epsilon": epsilon,
        "strength": strength,
        "analysis_region": analysis_region,
        "reference_method": reference_method,
        "propensity_mode": mode,
        "learner": learner,
        "propensity_learner": propensity_learner,
        "repair_mode": repair_mode,
        "region_damp": _region_damp(),
        "region_damp_grid": "|".join(str(g) for g in region_damp_grid),
        "validation_risk": _validation_risk(),
        "validation_region_weight": validation_region_weight,
        "validation_loss_se": validation_loss_se,
        "selector": effective_selector,
        "lepski_c": lepski_c,
        "region_quantile": region_quantile,
        "region_min_observed": region_min_observed,
        "region_kappa_floor": region_kappa_floor,
        "reps": len(ref),
        "bootstraps": bootstraps,
        "mse_ctmle_ref": ref_mse,
        "mse_ref": ref_mse,
        "mse_region_targeted": rt_mse,
        "mse_shrink": shrink_mse,
        "rmse_ctmle_ref": math.sqrt(ref_mse),
        "rmse_ref": math.sqrt(ref_mse),
        "rmse_region_targeted": math.sqrt(rt_mse),
        "rmse_shrink": math.sqrt(shrink_mse),
        "gain_region_targeted": (
            (ref_mse - rt_mse) / ref_mse if ref_mse > 0 else float("nan")
        ),
        "gain_shrink": (
            (ref_mse - shrink_mse) / ref_mse if ref_mse > 0 else float("nan")
        ),
        "harm_region_targeted": float(np.mean(rt**2 > ref**2)),
        "harm_shrink": float(np.mean(shrink**2 > ref**2)),
        "activation": float(np.mean(weights > 0.0)),
        "ref_bias": ref_bias,
        "repair_bias": repair_bias,
        "abs_bias_ratio": (
            abs(repair_bias) / abs(ref_bias) if abs(ref_bias) > 1e-9 else float("nan")
        ),
        "mean_delta": mean_delta,
        "mean_vd": mean_vd,
        "mean_gamma": _finite_mean(gamma),
        "m_snr": m_snr,
        "mean_selected_tau": float(tau.mean()),
        "mean_selected_region_damp": float(selected_region_damp.mean()),
        "frac_damp_zero": float(np.mean(selected_region_damp == 0.0)),
        "frac_damp_max": float(np.mean(selected_region_damp == max(region_damp_grid))),
        "mean_gl_original_bias_proxy": _finite_mean(gl_original_bias_proxy),
        "mean_gl_selected_bias_proxy": _finite_mean(gl_selected_bias_proxy),
        "mean_gl_selected_variance_proxy": _finite_mean(gl_selected_variance_proxy),
        "mean_gl_selected_risk_proxy": _finite_mean(gl_selected_risk_proxy),
        "mean_global_dr_selected_bias_proxy": _finite_mean(
            global_dr_selected_bias_proxy
        ),
        "mean_global_dr_selected_variance_proxy": _finite_mean(
            global_dr_selected_variance_proxy
        ),
        "mean_global_dr_selected_risk_proxy": _finite_mean(
            global_dr_selected_risk_proxy
        ),
        "mean_ma_trimmed_fraction": _finite_mean(ma_trimmed_fraction),
        "mean_ma_xi_derivative_1": _finite_mean(ma_xi_derivative_1),
        "mean_ma_bias_correction_mean": _finite_mean(ma_bias_correction_mean),
        "mean_analysis_region_mass": _finite_mean(analysis_region_mass),
        "mean_true_region_mass": _finite_mean(true_region_mass),
        "mean_region_overlap_precision": _finite_mean(region_overlap_precision),
        "mean_region_overlap_recall": _finite_mean(region_overlap_recall),
        "mean_analysis_observed": _finite_mean(analysis_observed),
        "mean_true_region_observed": _finite_mean(true_region_observed),
        "mean_analysis_response_rate": _finite_mean(analysis_response_rate),
        "mean_true_region_response_rate": _finite_mean(true_region_response_rate),
        "mean_analysis_bias_ref": _finite_mean(analysis_bias_ref),
        "mean_analysis_bias_repair": _finite_mean(analysis_bias_repair),
        "mean_analysis_abs_bias_reduction": _finite_mean(analysis_abs_bias_reduction),
        "mean_true_region_bias_ref": _finite_mean(true_region_bias_ref),
        "mean_true_region_bias_repair": _finite_mean(true_region_bias_repair),
        "mean_true_region_abs_bias_reduction": _finite_mean(
            true_region_abs_bias_reduction
        ),
        "mean_analysis_underfit_rmse_ref": _finite_mean(analysis_underfit_rmse_ref),
        "mean_analysis_underfit_rmse_repair": _finite_mean(
            analysis_underfit_rmse_repair
        ),
        "mean_true_region_underfit_rmse_ref": _finite_mean(
            true_region_underfit_rmse_ref
        ),
        "mean_true_region_underfit_rmse_repair": _finite_mean(
            true_region_underfit_rmse_repair
        ),
    }
    _log(
        f"CELL_DONE {label} ref_bias={ref_bias:+.4f} repair_bias={repair_bias:+.4f} "
        f"m_snr={m_snr:.2f} gain_shrink={row['gain_shrink']:+.4f} activation={row['activation']:.2f}"
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--bootstraps", type=int, default=6)
    ap.add_argument("--n", type=int, nargs="+", default=[3000])
    ap.add_argument("--epsilon", type=float, nargs="+", default=[0.05])
    ap.add_argument("--strength", type=float, nargs="+", default=[0.0, 3.0])
    ap.add_argument("--design", nargs="+", default=["smooth", "pockets", "oscillatory"])
    ap.add_argument(
        "--mar-design",
        choices=["box", "smooth_tail", "nonlinear_mar", "two_stratum_flip"],
        default="box",
        help="Response mechanism used for synthetic designs.",
    )
    ap.add_argument(
        "--analysis-region",
        nargs="+",
        choices=[
            "true",
            "true_lowp",
            "estimated_lowp",
            "estimated_lowp_supported",
            "estimated_residual_lowp_supported",
            "estimated_kappa_residual_lowp_supported",
            "shrink",
            "expand",
            "shift",
            "wrong",
            "flip_g1",
            "flip_g2",
            "flip_both",
        ],
        default=["true"],
        help=(
            "Region supplied to the repair. Response probabilities are still "
            "generated from the true low-response region."
        ),
    )
    ap.add_argument(
        "--region-quantile",
        type=float,
        default=0.10,
        help="Low-propensity quantile used by true_lowp and estimated_lowp regions.",
    )
    ap.add_argument(
        "--region-min-observed",
        type=int,
        default=0,
        help=(
            "Minimum responders required for estimated_lowp_supported. The "
            "selected low-propensity prefix expands until this count is reached."
        ),
    )
    ap.add_argument(
        "--region-kappa-floor",
        type=float,
        default=0.10,
        help=(
            "Propensity floor used only by estimated_kappa_residual_lowp_supported "
            "to form kappa_hat=1-pi_hat/max(pi_hat, floor)."
        ),
    )
    ap.add_argument(
        "--region-selector-ablation",
        choices=[
            "legacy",
            "raw_rank_only",
            "whole_sample_score",
            "all_prefixes",
            "empty_standdown",
            "crossfit_rank_empty_standdown",
            "both_signs",
        ],
        default="legacy",
        help=(
            "Selector-mechanism ablation for estimated_residual_lowp_supported. "
            "legacy matches the current learned residual-score selector."
        ),
    )
    ap.add_argument(
        "--region-detector-c",
        type=float,
        default=4.0,
        help="Detector penalty used in the residual-region score.",
    )
    ap.add_argument(
        "--reference-method",
        choices=[
            "aipw",
            "tmle",
            "ctmle",
            "ma_dr_bc",
            "cui_selective_ml",
            "glrisk",
            "glrisk_reference",
            "cui_tchetgen",
        ],
        default="ctmle",
        help=(
            "Reference estimator for the comparison. tmle uses the standard "
            "global targeting fluctuation at the smallest pre-specified "
            "propensity floor without collaborative floor selection; "
            "ma_dr_bc uses the Ma--Sant'Anna--Sasaki--Ura same-target "
            "bias-corrected trimmed DR estimator with fixed h from the "
            "singleton tau grid, k=1, and shifted-Legendre K=3; "
            "cui_selective_ml is the published two-fold mixed-minimax "
            "selector over L1-linear, random-forest, and gradient-boosting "
            "propensity/outcome learner libraries; "
            "glrisk is "
            "the historical "
            "C-TMLE repair-path selector ablation; glrisk_reference promotes "
            "the GL-selected path point to the reference and repairs its "
            "remaining regional increment; "
            "cui_tchetgen selects the global C-TMLE reference by a DR-risk "
            "criterion over the propensity-floor grid."
        ),
    )
    ap.add_argument(
        "--propensity-mode",
        nargs="+",
        choices=["true", "estimated"],
        default=["estimated"],
    )
    ap.add_argument(
        "--learner",
        choices=["histgb", "xgboost"],
        default="histgb",
        help="Learner used for the initial outcome fit and regional repair.",
    )
    ap.add_argument(
        "--propensity-learner",
        choices=["histgb", "xgboost"],
        default=None,
        help=(
            "Learner used for estimated propensities and estimated-region "
            "construction. Defaults to --learner."
        ),
    )
    ap.add_argument(
        "--repair-mode",
        choices=[
            "targeting",
            "reweight",
            "if_residual",
            "regional_if_residual",
            "if_projection",
            "if_library",
        ],
        default="targeting",
        help=(
            "Candidate construction: add a regional targeting fluctuation, "
            "refit with regional weights, learn a cross-fitted residual "
            "correction globally or only inside the supported region, or "
            "learn the generic mean-zero MAR influence-score projection."
        ),
    )
    ap.add_argument(
        "--tau-grid", type=float, nargs="+", default=[0.05, 0.10, 0.25, 0.50]
    )
    ap.add_argument("--c", type=float, default=4.0)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=14700000)
    ap.add_argument("--progress-every", type=int, default=4)
    ap.add_argument("--region-damp", type=float, default=1.0)
    ap.add_argument("--region-damp-grid", type=float, nargs="+")
    ap.add_argument(
        "--validation-risk",
        choices=["balanced_mse", "aipw_variance"],
        default="balanced_mse",
        help=(
            "Loss used to choose the regional damping path. aipw_variance "
            "uses the MAR-identifiable outcome-risk term that determines the "
            "one-step influence-function variance."
        ),
    )
    ap.add_argument(
        "--validation-loss-se",
        type=float,
        default=1.0,
        help=(
            "One-SE margin for moving gamma away from zero; negative uses the "
            "minimum validation loss without the margin."
        ),
    )
    ap.add_argument(
        "--validation-region-weight",
        type=float,
        default=-1.0,
        help="If negative, balance the validation loss between G and its complement.",
    )
    ap.add_argument(
        "--selector",
        choices=["obsval", "lepski", "glrisk"],
        default="obsval",
        help=(
            "Rule for choosing the regional damping value: observed-outcome validation "
            "loss, the first Lepski-detectable contrast, or a full GL risk proxy."
        ),
    )
    ap.add_argument(
        "--lepski-c",
        type=float,
        default=4.0,
        help="Detection threshold for --selector lepski.",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rep-out", type=Path)
    args = ap.parse_args()
    os.environ["USHMOO_REGION_DAMP"] = str(args.region_damp)
    os.environ["USHMOO_VALIDATION_RISK"] = args.validation_risk
    region_damp_grid = (
        tuple(float(g) for g in args.region_damp_grid)
        if args.region_damp_grid
        else (float(args.region_damp),)
    )
    propensity_learner = args.propensity_learner or args.learner
    cells = [
        (
            d,
            args.mar_design,
            n,
            e,
            s,
            ar,
            args.reference_method,
            m,
            args.learner,
            propensity_learner,
            args.repair_mode,
            args.c,
            tuple(args.tau_grid),
            args.folds,
            args.bootstraps,
            args.seed,
            region_damp_grid,
            args.validation_region_weight,
            args.validation_loss_se,
            args.selector,
            args.lepski_c,
            args.region_quantile,
            args.region_min_observed,
            args.region_kappa_floor,
            args.region_selector_ablation,
            args.region_detector_c,
        )
        for d in args.design
        for n in args.n
        for e in args.epsilon
        for s in args.strength
        for ar in args.analysis_region
        for m in args.propensity_mode
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rep_log_path = args.rep_out or args.out.with_suffix(args.out.suffix + ".reps.csv")
    rep_log_path.parent.mkdir(parents=True, exist_ok=True)
    rep_log_path.write_text("")
    _log(
        f"RUN_START cells={len(cells)} reps={args.reps} out={args.out} rep_out={rep_log_path}"
    )
    with args.out.open("w", newline="") as handle:
        writer = None
        for index, cell in enumerate(cells, start=1):
            _log(f"CELL_QUEUE {index}/{len(cells)}")
            row = run_cell(cell, args.reps, max(1, args.progress_every), rep_log_path)
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            _log(
                f"ROW_WRITTEN {index}/{len(cells)} {row['design']} strength={row['strength']}"
            )
    _log(f"RUN_DONE cells={len(cells)} out={args.out}")


if __name__ == "__main__":
    main()
