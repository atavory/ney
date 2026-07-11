#!/usr/bin/env python3
"""Self-contained regional-repair companion simulations.

The script intentionally uses only NumPy and pandas. It is not a mirror of the
older R/SuperLearner workflow cited in the manuscript; it is the public,
rerunnable companion for the visible low-response regional-repair mechanism.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RunConfig:
    n: int
    reps: int
    budget_reps: int
    seed: int
    tail_fraction: float
    low_response: float
    high_response: float
    degree: int
    noise_sd: float
    guard_margin: float
    min_guard_observed: int
    output_dir: Path


@dataclass(frozen=True)
class FitResult:
    estimate: float
    se: float
    selected_regional: bool
    n_region_observed_validation: int


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Run regional-repair companion simulations."
    )
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--reps", type=int, default=80)
    parser.add_argument("--budget-reps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--tail-fraction", type=float, default=0.15)
    parser.add_argument("--low-response", type=float, default=0.25)
    parser.add_argument("--high-response", type=float, default=0.85)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--noise-sd", type=float, default=1.0)
    parser.add_argument("--guard-margin", type=float, default=0.01)
    parser.add_argument("--min-guard-observed", type=int, default=12)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer replications for a fast code-path check.",
    )
    args = parser.parse_args()
    reps = 12 if args.quick else args.reps
    budget_reps = 16 if args.quick else args.budget_reps
    return RunConfig(
        n=args.n,
        reps=reps,
        budget_reps=budget_reps,
        seed=args.seed,
        tail_fraction=args.tail_fraction,
        low_response=args.low_response,
        high_response=args.high_response,
        degree=args.degree,
        noise_sd=args.noise_sd,
        guard_margin=args.guard_margin,
        min_guard_observed=args.min_guard_observed,
        output_dir=args.output_dir,
    )


def region_indicator(x: np.ndarray, tail_fraction: float) -> np.ndarray:
    return x <= tail_fraction


def response_probability(x: np.ndarray, cfg: RunConfig) -> np.ndarray:
    region = region_indicator(x, cfg.tail_fraction)
    return np.where(region, cfg.low_response, cfg.high_response)


def outcome_mean(x: np.ndarray, design: str, tail_fraction: float) -> np.ndarray:
    base = 0.35 * np.sin(2.0 * np.pi * x) + 0.25 * (x - 0.5)
    region = region_indicator(x, tail_fraction)
    tail_coordinate = np.zeros_like(x)
    tail_coordinate[region] = 1.0 - x[region] / tail_fraction

    if design == "hetero":
        local = 1.25 * tail_coordinate
    elif design == "bump":
        local = 0.95 * np.exp(-0.5 * ((x - tail_fraction) / 0.045) ** 2)
    elif design == "reverse":
        local = -0.80 * tail_coordinate + 0.30 * np.exp(
            -0.5 * ((x - tail_fraction) / 0.06) ** 2
        )
    elif design == "homo":
        local = np.zeros_like(x)
    elif design == "regional_bump":
        local = 1.70 * tail_coordinate + 0.45 * np.exp(
            -0.5 * ((x - tail_fraction) / 0.05) ** 2
        )
    else:
        raise ValueError(f"unknown design: {design}")

    return base + local


def truth(design: str, cfg: RunConfig) -> float:
    grid = np.linspace(0.0, 1.0, 20001)
    return float(outcome_mean(grid, design, cfg.tail_fraction).mean())


def simulate_data(
    rng: np.random.Generator, n: int, design: str, cfg: RunConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = rng.uniform(size=n)
    pi = response_probability(x, cfg)
    observed = rng.binomial(1, pi).astype(bool)
    y = outcome_mean(x, design, cfg.tail_fraction) + rng.normal(
        scale=cfg.noise_sd, size=n
    )
    return x, pi, observed, y


def basis(
    x: np.ndarray, degree: int, regional: bool, tail_fraction: float
) -> np.ndarray:
    columns = [np.ones_like(x)]
    columns.extend(x**power for power in range(1, degree + 1))
    if regional:
        region = region_indicator(x, tail_fraction).astype(float)
        columns.extend(region * x**power for power in range(0, degree + 1))
    return np.column_stack(columns)


def fit_outcome(
    x: np.ndarray,
    y: np.ndarray,
    observed: np.ndarray,
    cfg: RunConfig,
    regional: bool,
) -> np.ndarray:
    design = basis(x[observed], cfg.degree, regional, cfg.tail_fraction)
    target = y[observed]
    ridge = 1.0e-8 * np.eye(design.shape[1])
    return np.linalg.solve(design.T @ design + ridge, design.T @ target)


def predict_outcome(
    x: np.ndarray, beta: np.ndarray, cfg: RunConfig, regional: bool
) -> np.ndarray:
    return basis(x, cfg.degree, regional, cfg.tail_fraction) @ beta


def aipw_estimate(
    x: np.ndarray,
    pi: np.ndarray,
    observed: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    cfg: RunConfig,
    regional: bool,
) -> tuple[float, float]:
    m = predict_outcome(x, beta, cfg, regional)
    score = m + observed.astype(float) / pi * (y - m)
    return float(score.mean()), float(score.std(ddof=1) / np.sqrt(len(score)))


def guarded_fit(
    rng: np.random.Generator,
    x: np.ndarray,
    pi: np.ndarray,
    observed: np.ndarray,
    y: np.ndarray,
    cfg: RunConfig,
) -> FitResult:
    validation = rng.uniform(size=len(x)) < 0.35
    training = ~validation
    validation_region_observed = (
        region_indicator(x, cfg.tail_fraction) & validation & observed
    )
    n_region_observed = int(validation_region_observed.sum())

    beta_global_train = fit_outcome(
        x[training], y[training], observed[training], cfg, regional=False
    )
    beta_regional_train = fit_outcome(
        x[training], y[training], observed[training], cfg, regional=True
    )

    choose_regional = False
    if n_region_observed >= cfg.min_guard_observed:
        y_validation = y[validation_region_observed]
        pred_global = predict_outcome(
            x[validation_region_observed], beta_global_train, cfg, regional=False
        )
        pred_regional = predict_outcome(
            x[validation_region_observed], beta_regional_train, cfg, regional=True
        )
        loss_global = float(np.mean((y_validation - pred_global) ** 2))
        loss_regional = float(np.mean((y_validation - pred_regional) ** 2))
        choose_regional = loss_regional + cfg.guard_margin < loss_global

    beta = fit_outcome(x, y, observed, cfg, regional=choose_regional)
    estimate, se = aipw_estimate(
        x, pi, observed, y, beta, cfg, regional=choose_regional
    )
    return FitResult(estimate, se, choose_regional, n_region_observed)


def run_single(
    rng: np.random.Generator, design: str, n: int, cfg: RunConfig
) -> list[dict[str, object]]:
    x, pi, observed, y = simulate_data(rng, n, design, cfg)
    theta = truth(design, cfg)

    beta_global = fit_outcome(x, y, observed, cfg, regional=False)
    beta_regional = fit_outcome(x, y, observed, cfg, regional=True)
    est_global, se_global = aipw_estimate(
        x, pi, observed, y, beta_global, cfg, False
    )
    est_regional, se_regional = aipw_estimate(
        x, pi, observed, y, beta_regional, cfg, True
    )
    guarded = guarded_fit(rng, x, pi, observed, y, cfg)

    rows: list[dict[str, object]] = []
    for method, estimate, se, selected in [
        ("global_target", est_global, se_global, False),
        ("regional_target", est_regional, se_regional, True),
        ("guarded_repair", guarded.estimate, guarded.se, guarded.selected_regional),
    ]:
        rows.append(
            {
                "design": design,
                "method": method,
                "estimate": estimate,
                "truth": theta,
                "error": estimate - theta,
                "covered": abs(estimate - theta) <= 1.96 * se,
                "selected_regional": selected,
                "n_region_observed": int(
                    (region_indicator(x, cfg.tail_fraction) & observed).sum()
                ),
            }
        )
    return rows


def summarize_replications(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    grouped = df.groupby(["design", "method"], as_index=False)
    return grouped.agg(
        rmse=("error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        bias=("error", "mean"),
        coverage=("covered", "mean"),
        guard_rate=("selected_regional", "mean"),
        mean_region_observed=("n_region_observed", "mean"),
        reps=("error", "size"),
    )


def run_regional_summary(cfg: RunConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, object]] = []
    for design in ["hetero", "bump", "reverse", "homo"]:
        for _ in range(cfg.reps):
            rows.extend(run_single(rng, design, cfg.n, cfg))
    return summarize_replications(rows)


def run_budget_summary(cfg: RunConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 100_000)
    rows: list[dict[str, object]] = []
    sample_sizes = [600, 1200, 2400, 4800]
    for n in sample_sizes:
        for _ in range(cfg.budget_reps):
            replicated = run_single(rng, "regional_bump", n, cfg)
            by_method = {row["method"]: row for row in replicated}
            reference_error = float(by_method["global_target"]["error"])
            for method in ["global_target", "regional_target", "guarded_repair"]:
                error = float(by_method[method]["error"])
                rows.append(
                    {
                        "n": n,
                        "method": method,
                        "error": error,
                        "n_region_observed": by_method[method][
                            "n_region_observed"
                        ],
                        "harm_vs_reference": error**2 > reference_error**2,
                    }
                )
    df = pd.DataFrame(rows)
    grouped = df.groupby(["n", "method"], as_index=False)
    return grouped.agg(
        rmse=("error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        mean_region_observed=("n_region_observed", "mean"),
        harm_rate=("harm_vs_reference", "mean"),
        reps=("error", "size"),
    )


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    regional = run_regional_summary(cfg)
    budget = run_budget_summary(cfg)

    regional_path = cfg.output_dir / "regional_repair_summary.csv"
    budget_path = cfg.output_dir / "observed_outcome_budget.csv"
    regional.to_csv(regional_path, index=False)
    budget.to_csv(budget_path, index=False)

    print(f"wrote {regional_path}")
    print(regional.to_string(index=False))
    print(f"wrote {budget_path}")
    print(budget.to_string(index=False))


if __name__ == "__main__":
    main()
