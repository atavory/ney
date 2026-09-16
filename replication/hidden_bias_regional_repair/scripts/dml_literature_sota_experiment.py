#!/usr/bin/env python3
"""Run the locked Zhao/Kallus missing-mean SOTA comparison.

This adapts published DGP ingredients to a superpopulation missing-outcome
mean.  It does not reproduce the source papers' estimands or estimators.
Upstream expert construction and residual-path fitting are delegated unchanged
to ``validated_reference_transfer.py``.  The emitted path is selected by the
paper's current paired one-SE response-weighted residual-loss rule.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


CELLS = (
    "zhao24a_vanilla_y0",
    "zhao24a_vanilla_y1",
    "kallus23a_lognormal_y1_n200",
    "kallus23a_lognormal_y1_n800",
    "kallus23a_lognormal_y1_n3200",
)
METHODS = ("aipw", "cui_selective_ml", "ma_dr_bc", "ctmle")
GAMMAS = (0.0, 0.25, 0.5, 1.0)
TAU_BY_METHOD = {
    "aipw": (0.05,),
    "cui_selective_ml": (0.05,),
    "ma_dr_bc": (0.05,),
    "ctmle": (0.05, 0.10, 0.25, 0.50),
}
CELL_OFFSETS = {
    "zhao24a_vanilla_y0": 0,
    "zhao24a_vanilla_y1": 1_000_000,
    "kallus23a_lognormal_y1_n200": 2_000_000,
    "kallus23a_lognormal_y1_n800": 3_000_000,
    "kallus23a_lognormal_y1_n3200": 4_000_000,
}
METHOD_FIT_COUNTS = {
    "aipw": {"xgboost": 9, "native_library": 0, "floor_candidates": 1},
    "ma_dr_bc": {"xgboost": 9, "native_library": 0, "floor_candidates": 1},
    "ctmle": {"xgboost": 9, "native_library": 0, "floor_candidates": 4},
    "cui_selective_ml": {"xgboost": 9, "native_library": 12, "floor_candidates": 0},
}


def logistic(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def zhao_problem(cell: str, seed: int) -> tuple[np.ndarray, ...]:
    n = 2000
    rng = np.random.default_rng(seed)
    uniforms = rng.uniform(0.0, 1.0, size=(n, 2))
    normals = rng.multivariate_normal(
        mean=np.zeros(2), covariance=np.asarray([[1.0, 0.3], [0.3, 1.0]]), size=n
    )
    x = np.column_stack([uniforms, normals])
    p_treated = logistic(0.35 - 0.3 * np.sum(x, axis=1))
    treatment = rng.binomial(1, p_treated)
    noise = rng.normal(size=n)
    mu0 = 20.0 * (
        1.0 + x[:, 0] - x[:, 1] + x[:, 2] ** 2 + np.exp(x[:, 1])
    )
    tau = 25.0 * (
        3.0 - 5.0 * x[:, 0] + 2.0 * x[:, 1] - 3.0 * x[:, 2] + x[:, 3]
    )
    if cell.endswith("y1"):
        y = mu0 + tau + 20.0 * noise
        response = treatment
        true_pi = p_treated
        mu = mu0 + tau
        theta = 20.0 * (1.0 + math.e) + 37.5
    else:
        y = mu0 + 20.0 * noise
        response = 1 - treatment
        true_pi = 1.0 - p_treated
        mu = mu0
        theta = 20.0 * (1.0 + math.e)
    region = true_pi <= float(np.quantile(true_pi, 0.10))
    return x, y, response, region, true_pi, float(theta), mu


def kallus_problem(cell: str, seed: int) -> tuple[np.ndarray, ...]:
    n = int(cell.rsplit("n", 1)[1])
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=(n, 10))
    true_pi = logistic(6.0 * (x[:, 0] - 0.5))
    response = rng.binomial(1, true_pi)
    log_mean = x[:, 0] + x[:, 1]
    y = rng.lognormal(mean=log_mean, sigma=0.2)
    mu = np.exp(log_mean + 0.5 * 0.2**2)
    theta = math.exp(0.5 * 0.2**2) * (math.e - 1.0) ** 2
    region = true_pi <= float(np.quantile(true_pi, 0.10))
    return x, y, response, region, true_pi, float(theta), mu


def make_problem(cell: str, seed: int) -> tuple[np.ndarray, ...]:
    if cell.startswith("zhao24a_"):
        return zhao_problem(cell, seed)
    if cell.startswith("kallus23a_"):
        return kallus_problem(cell, seed)
    raise ValueError(cell)


def load_upstream(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("literature_sota_upstream", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import upstream source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def mean_and_se(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return float(np.mean(values)), float("inf")
    return float(np.mean(values)), float(np.std(values, ddof=1) / math.sqrt(len(values)))


def select_weighted_residual_candidate(
    y: np.ndarray,
    response: np.ndarray,
    p: np.ndarray,
    gammas: np.ndarray,
    outcomes: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    if tuple(float(value) for value in gammas) != GAMMAS:
        raise ValueError(f"candidate grid mismatch: {gammas}")
    observed = response.astype(bool)
    if int(np.sum(observed)) < 2:
        raise RuntimeError("fewer than two respondents")
    p_observed = np.clip(p[observed], 1e-6, 1.0)
    weight = (1.0 - p_observed) / (p_observed * p_observed)
    losses = []
    improvements = []
    baseline_terms = weight * (y[observed] - outcomes[0, observed]) ** 2
    for index, _gamma in enumerate(gammas):
        terms = weight * (y[observed] - outcomes[index, observed]) ** 2
        losses.append(float(np.mean(terms)))
        improvements.append(baseline_terms - terms)
    eligible = [0]
    for index in range(1, len(gammas)):
        improvement, se = mean_and_se(improvements[index])
        if improvement > se:
            eligible.append(index)
    selected_index = min(eligible, key=lambda index: (losses[index], gammas[index]))
    selected_gamma = float(gammas[selected_index])
    gain = (
        100.0 * (1.0 - losses[selected_index] / losses[0])
        if losses[0] > 0.0 else float("nan")
    )
    return {
        "index": selected_index,
        "gamma": selected_gamma,
        "estimate": float(np.mean(scores[selected_index])),
        "score": scores[selected_index],
        "outcome": outcomes[selected_index],
        "weighted_residual_loss_ref": losses[0],
        "weighted_residual_loss_repaired": losses[selected_index],
        "weighted_residual_gain_pct": gain,
    }


def run_one(
    upstream: Any,
    cell: str,
    method: str,
    rep: int,
    seed_base: int,
) -> dict[str, Any]:
    seed = seed_base + CELL_OFFSETS[cell] + rep
    data = make_problem(cell, seed)
    x, y, response, _region, true_pi, theta, _mu = data
    started = time.monotonic()
    fit = upstream._crossfit_selected(
        data=data,
        reference_method=method,
        mode="estimated",
        learner="xgboost",
        propensity_learner="xgboost",
        repair_mode="if_residual",
        tau_grid=TAU_BY_METHOD[method],
        folds=3,
        seed=seed + 17,
        region_damp_grid=GAMMAS,
        validation_region_weight=-1.0,
        validation_loss_se=1.0,
        selector="obsval",
        lepski_c=4.0,
    )
    elapsed = time.monotonic() - started
    gammas = np.asarray(fit["candidate_values"], dtype=float)
    scores = np.asarray(fit["candidate_scores"], dtype=float)
    outcomes = np.asarray(fit["candidate_outcomes"], dtype=float)
    p = np.asarray(fit["selected_p"], dtype=float)
    ref_score = np.asarray(fit["ref"], dtype=float)
    ref_outcome = np.asarray(fit["ref_outcome"], dtype=float)
    if not np.allclose(scores[0], ref_score, rtol=1e-10, atol=1e-10):
        raise AssertionError("gamma-zero score does not equal upstream reference")
    if not np.allclose(outcomes[0], ref_outcome, rtol=1e-10, atol=1e-10):
        raise AssertionError("gamma-zero outcome does not equal upstream reference")
    selected = select_weighted_residual_candidate(
        y, response, p, gammas, outcomes, scores
    )
    ref_estimate = float(np.mean(ref_score))
    repaired_estimate = float(selected["estimate"])
    fit_counts = METHOD_FIT_COUNTS[method]
    return {
        "cell": cell,
        "method": method,
        "rep": rep,
        "seed": seed,
        "n": len(y),
        "status": "ok",
        "error": "",
        "respondents": int(np.sum(response)),
        "mean_true_response_prob": float(np.mean(true_pi)),
        "min_true_response_prob": float(np.min(true_pi)),
        "target": theta,
        "ref_estimate": ref_estimate,
        "repaired_estimate": repaired_estimate,
        "ref_error": ref_estimate - theta,
        "repaired_error": repaired_estimate - theta,
        "ref_average_score_variance": float(np.var(ref_score, ddof=1) / len(y)),
        "repaired_average_score_variance": float(np.var(selected["score"], ddof=1) / len(y)),
        "selected_gamma": selected["gamma"],
        "active": int(selected["gamma"] != 0.0),
        "weighted_residual_loss_ref": selected["weighted_residual_loss_ref"],
        "weighted_residual_loss_repaired": selected["weighted_residual_loss_repaired"],
        "weighted_residual_gain_pct": selected["weighted_residual_gain_pct"],
        "selected_tau": fit["selected_tau"],
        "selected_repair_kind": "reference" if selected["gamma"] == 0.0 else "if_residual",
        "cui_selected_propensity_learner": fit["cui_selected_propensity_learner"],
        "cui_selected_outcome_learner": fit["cui_selected_outcome_learner"],
        "ma_trimmed_fraction": fit["ma_trimmed_fraction"],
        "elapsed_seconds": elapsed,
        "nominal_xgboost_fits": fit_counts["xgboost"],
        "nominal_native_library_fits": fit_counts["native_library"],
        "floor_candidates": fit_counts["floor_candidates"],
    }


def failed_row(cell: str, method: str, rep: int, seed: int, exc: Exception) -> dict[str, Any]:
    return {
        "cell": cell, "method": method, "rep": rep, "seed": seed,
        "n": 2000 if cell.startswith("zhao") else int(cell.rsplit("n", 1)[1]),
        "status": "failed", "error": f"{type(exc).__name__}: {exc}",
        "respondents": "", "mean_true_response_prob": "", "min_true_response_prob": "",
        "target": "", "ref_estimate": "", "repaired_estimate": "", "ref_error": "",
        "repaired_error": "", "ref_average_score_variance": "",
        "repaired_average_score_variance": "", "selected_gamma": "", "active": "",
        "weighted_residual_loss_ref": "", "weighted_residual_loss_repaired": "",
        "weighted_residual_gain_pct": "", "selected_tau": "", "selected_repair_kind": "",
        "cui_selected_propensity_learner": "", "cui_selected_outcome_learner": "",
        "ma_trimmed_fraction": "", "elapsed_seconds": "", "nominal_xgboost_fits": "",
        "nominal_native_library_fits": "", "floor_candidates": "",
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=CELLS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--rep-start", type=int, required=True)
    parser.add_argument("--rep-stop", type=int, required=True)
    parser.add_argument("--seed-base", type=int, default=202609160)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--upstream-source",
        type=Path,
        default=Path(__file__).with_name("validated_reference_transfer.py"),
    )
    args = parser.parse_args()
    if not 0 <= args.rep_start < args.rep_stop <= 200:
        raise ValueError("replication range must lie in [0, 200]")
    import xgboost
    if xgboost.__version__ != "3.4.0":
        raise RuntimeError(f"expected xgboost 3.4.0, found {xgboost.__version__}")
    os.environ["USHMOO_VALIDATION_RISK"] = "aipw_variance"
    upstream = load_upstream(args.upstream_source.resolve())
    rows: list[dict[str, Any]] = []
    for rep in range(args.rep_start, args.rep_stop):
        seed = args.seed_base + CELL_OFFSETS[args.cell] + rep
        try:
            row = run_one(upstream, args.cell, args.method, rep, args.seed_base)
        except Exception as exc:
            row = failed_row(args.cell, args.method, rep, seed, exc)
        rows.append(row)
        print(json.dumps({"cell": args.cell, "method": args.method, "rep": rep, "status": row["status"]}), flush=True)
    write_rows(args.output, rows)
    metadata = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "xgboost": xgboost.__version__,
        "cell": args.cell,
        "method": args.method,
        "rep_start": args.rep_start,
        "rep_stop": args.rep_stop,
        "seed_base": args.seed_base,
        "upstream_source": str(args.upstream_source.resolve()),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
