#!/usr/bin/env python3
"""Same-sample selection-penalty experiment for current Algorithm 1.

This finite-support diagnostic runs the current full-data cross-fitted
weighted-residual selector.  Each replication uses all observations in the
cross-fitted estimator, so selection and final estimation reuse the same sample
as in the efficient Algorithm 1 implementation.

The output reports both the actual same-sample Monte Carlo MSE and the
fresh-candidate risk of the selected gamma.  Their difference is the empirical
same-sample selection penalty that the current theorem draft isolates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

from dml_honest_split_selected_gamma_experiment import (
    GAMMA_GRID,
    BaseScenario,
    FittedPath,
    SampleAgg,
    _draw_atom,
    _fit_path,
    _scenarios,
    _theta,
)


def _new_accumulator(width: int) -> list[list[float]]:
    return [
        [0 for _ in range(width)],
        [0 for _ in range(width)],
        [0.0 for _ in range(width)],
        [0.0 for _ in range(width)],
    ]


def _add_observation(
    accumulator: list[list[float]],
    index: int,
    response: float,
    y_value: float,
) -> None:
    counts, response_counts, response_sum_y, response_sum_y2 = accumulator
    counts[index] += 1
    if response:
        response_counts[index] += 1
        response_sum_y[index] += y_value
        response_sum_y2[index] += y_value * y_value


def _to_agg(accumulator: list[list[float]]) -> SampleAgg:
    counts, response_counts, response_sum_y, response_sum_y2 = accumulator
    return SampleAgg(
        n=int(sum(counts)),
        counts=tuple(int(x) for x in counts),
        response_counts=tuple(int(x) for x in response_counts),
        response_sum_y=tuple(float(x) for x in response_sum_y),
        response_sum_y2=tuple(float(x) for x in response_sum_y2),
    )


def _combine_aggs(aggs: list[SampleAgg]) -> SampleAgg:
    width = len(aggs[0].counts)
    return SampleAgg(
        n=sum(agg.n for agg in aggs),
        counts=tuple(sum(agg.counts[i] for agg in aggs) for i in range(width)),
        response_counts=tuple(sum(agg.response_counts[i] for agg in aggs) for i in range(width)),
        response_sum_y=tuple(sum(agg.response_sum_y[i] for agg in aggs) for i in range(width)),
        response_sum_y2=tuple(sum(agg.response_sum_y2[i] for agg in aggs) for i in range(width)),
    )


def _draw_fold_aggs(
    scenario: BaseScenario,
    n_total: int,
    folds: int,
    rng: random.Random,
) -> list[SampleAgg]:
    cumulative = []
    acc = 0.0
    for value in scenario.q:
        acc += value
        cumulative.append(acc)
    accumulators = [_new_accumulator(len(scenario.q)) for _ in range(folds)]
    for row in range(n_total):
        index = _draw_atom(scenario, rng, cumulative)
        response = 1.0 if rng.random() < scenario.pi[index] else 0.0
        y_value = 0.0
        if response:
            y_value = scenario.mu[index] + scenario.sigma[index] * rng.gauss(0.0, 1.0)
        _add_observation(accumulators[row % folds], index, response, y_value)
    return [_to_agg(accumulator) for accumulator in accumulators]


def _fit_crossfit_paths(
    scenario: BaseScenario,
    fold_aggs: list[SampleAgg],
) -> list[FittedPath]:
    paths = []
    for fold_index in range(len(fold_aggs)):
        training = _combine_aggs(
            [agg for index, agg in enumerate(fold_aggs) if index != fold_index]
        )
        paths.append(_fit_path(scenario, training))
    return paths


def _score_sum(path: FittedPath, sample: SampleAgg, gamma: float) -> float:
    scenario = path.scenario
    total = 0.0
    for index, count in enumerate(sample.counts):
        p_i = scenario.p[index]
        m_i = path.m0[index] + gamma * path.h[index]
        response_count = sample.response_counts[index]
        total += count * m_i
        total += (sample.response_sum_y[index] - response_count * m_i) / p_i
    return total


def _estimate_crossfit(
    paths: list[FittedPath],
    fold_aggs: list[SampleAgg],
    gamma: float,
) -> float:
    total_n = sum(agg.n for agg in fold_aggs)
    total = 0.0
    for path, fold_agg in zip(paths, fold_aggs):
        total += _score_sum(path, fold_agg, gamma)
    return total / total_n


def _gain_sums(path: FittedPath, sample: SampleAgg) -> dict[float, dict[str, float]]:
    scenario = path.scenario
    sums = {gamma: {"gain": 0.0, "gain2": 0.0} for gamma in GAMMA_GRID}
    for index, response_count in enumerate(sample.response_counts):
        if response_count == 0:
            continue
        p_i = scenario.p[index]
        weight = (1.0 - p_i) / (p_i * p_i)
        m0_i = path.m0[index]
        h_i = path.h[index]
        residual_sum = sample.response_sum_y[index] - response_count * m0_i
        residual2_sum = (
            sample.response_sum_y2[index]
            - 2.0 * m0_i * sample.response_sum_y[index]
            + response_count * m0_i * m0_i
        )
        for gamma in GAMMA_GRID:
            linear = 2.0 * gamma * h_i
            constant = gamma * gamma * h_i * h_i
            gain_sum = weight * (linear * residual_sum - constant * response_count)
            gain2_sum = weight * weight * (
                linear * linear * residual2_sum
                - 2.0 * linear * constant * residual_sum
                + constant * constant * response_count
            )
            sums[gamma]["gain"] += gain_sum
            sums[gamma]["gain2"] += gain2_sum
    return sums


def _select_gamma_crossfit(
    paths: list[FittedPath],
    fold_aggs: list[SampleAgg],
) -> float:
    total_n = sum(agg.n for agg in fold_aggs)
    totals = {gamma: {"gain": 0.0, "gain2": 0.0} for gamma in GAMMA_GRID}
    for path, fold_agg in zip(paths, fold_aggs):
        fold_sums = _gain_sums(path, fold_agg)
        for gamma in GAMMA_GRID:
            totals[gamma]["gain"] += fold_sums[gamma]["gain"]
            totals[gamma]["gain2"] += fold_sums[gamma]["gain2"]
    gains = {}
    eligible = [0.0]
    for gamma in GAMMA_GRID:
        mean_gain = totals[gamma]["gain"] / total_n
        gains[gamma] = mean_gain
        if gamma == 0.0:
            se = 0.0
        else:
            variance = max(
                0.0,
                (totals[gamma]["gain2"] - total_n * mean_gain * mean_gain)
                / (total_n - 1),
            )
            se = math.sqrt(variance / total_n)
        if gamma != 0.0 and mean_gain > se:
            eligible.append(gamma)
    return max(eligible, key=lambda gamma: gains[gamma])


def _bias(path: FittedPath, gamma: float) -> float:
    scenario = path.scenario
    terms = []
    for p_i, pi_i, m0_i, h_i, mu_i in zip(
        scenario.p,
        scenario.pi,
        path.m0,
        path.h,
        scenario.mu,
    ):
        m_i = m0_i + gamma * h_i
        terms.append((p_i - pi_i) / p_i * (m_i - mu_i))
    return sum(q_i * value for q_i, value in zip(scenario.q, terms))


def _variance(path: FittedPath, gamma: float) -> float:
    scenario = path.scenario
    theta = _theta(scenario)
    cond_var = []
    cond_mean = []
    for pi_i, p_i, m0_i, h_i, mu_i, sigma_i in zip(
        scenario.pi,
        scenario.p,
        path.m0,
        path.h,
        scenario.mu,
        scenario.sigma,
    ):
        m_i = m0_i + gamma * h_i
        error_i = m_i - mu_i
        cond_var.append(
            pi_i / (p_i * p_i) * sigma_i * sigma_i
            + pi_i * (1.0 - pi_i) / (p_i * p_i) * error_i * error_i
        )
        cond_mean.append(mu_i + (p_i - pi_i) / p_i * error_i)
    mean_cond = sum(q_i * value for q_i, value in zip(scenario.q, cond_mean))
    second = sum(q_i * value * value for q_i, value in zip(scenario.q, cond_mean))
    return sum(q_i * value for q_i, value in zip(scenario.q, cond_var)) + second - theta * theta


def _crossfit_fresh_risk(
    paths: list[FittedPath],
    gamma: float,
    n_total: int,
) -> float:
    mean_bias = sum(_bias(path, gamma) for path in paths) / len(paths)
    mean_variance = sum(_variance(path, gamma) for path in paths) / len(paths)
    return mean_bias * mean_bias + mean_variance / n_total


def _one_run(
    scenario: BaseScenario,
    n_total: int,
    folds: int,
    rng: random.Random,
) -> dict[str, float]:
    fold_aggs = _draw_fold_aggs(scenario, n_total, folds, rng)
    paths = _fit_crossfit_paths(scenario, fold_aggs)
    selected_gamma = _select_gamma_crossfit(paths, fold_aggs)
    theta = _theta(scenario)
    baseline_estimate = _estimate_crossfit(paths, fold_aggs, 0.0)
    selected_estimate = _estimate_crossfit(paths, fold_aggs, selected_gamma)
    fresh_risks = {gamma: _crossfit_fresh_risk(paths, gamma, n_total) for gamma in GAMMA_GRID}
    oracle_gamma = min(GAMMA_GRID, key=lambda gamma: fresh_risks[gamma])
    return {
        "selected_gamma": selected_gamma,
        "oracle_gamma": oracle_gamma,
        "baseline_sq_error": (baseline_estimate - theta) ** 2,
        "selected_sq_error": (selected_estimate - theta) ** 2,
        "baseline_fresh_risk": fresh_risks[0.0],
        "selected_fresh_risk": fresh_risks[selected_gamma],
        "oracle_fresh_risk": fresh_risks[oracle_gamma],
        "same_sample_penalty": (selected_estimate - theta) ** 2 - fresh_risks[selected_gamma],
        "selection_regret": fresh_risks[selected_gamma] - fresh_risks[oracle_gamma],
        "fresh_harm": float(fresh_risks[selected_gamma] > fresh_risks[0.0]),
        "same_sample_harm": float((selected_estimate - theta) ** 2 > (baseline_estimate - theta) ** 2),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    return ordered[int(q * (len(ordered) - 1))]


def _summarize_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    baseline_mse = _mean([row["baseline_sq_error"] for row in rows])
    selected_mse = _mean([row["selected_sq_error"] for row in rows])
    baseline_risk = _mean([row["baseline_fresh_risk"] for row in rows])
    selected_risk = _mean([row["selected_fresh_risk"] for row in rows])
    oracle_risk = _mean([row["oracle_fresh_risk"] for row in rows])
    penalties = [row["same_sample_penalty"] for row in rows]
    abs_penalties = [abs(value) for value in penalties]
    return {
        "reps": len(rows),
        "baseline_mse": baseline_mse,
        "selected_mse": selected_mse,
        "same_sample_mse_gain_pct": 100.0 * (baseline_mse - selected_mse) / baseline_mse,
        "baseline_fresh_risk": baseline_risk,
        "selected_fresh_risk": selected_risk,
        "fresh_risk_gain_pct": 100.0 * (baseline_risk - selected_risk) / baseline_risk,
        "oracle_fresh_risk": oracle_risk,
        "oracle_fresh_risk_gain_pct": 100.0 * (baseline_risk - oracle_risk) / baseline_risk,
        "mean_selection_regret": _mean([row["selection_regret"] for row in rows]),
        "mean_same_sample_penalty": _mean(penalties),
        "median_same_sample_penalty": _quantile(penalties, 0.50),
        "p90_abs_same_sample_penalty": _quantile(abs_penalties, 0.90),
        "same_sample_penalty_over_selected_risk": _mean(penalties) / selected_risk,
        "fresh_harm_share": _mean([row["fresh_harm"] for row in rows]),
        "same_sample_harm_share": _mean([row["same_sample_harm"] for row in rows]),
        **{
            f"selected_share_gamma_{gamma:g}": sum(1.0 for row in rows if row["selected_gamma"] == gamma) / len(rows)
            for gamma in GAMMA_GRID
        },
        **{
            f"oracle_share_gamma_{gamma:g}": sum(1.0 for row in rows if row["oracle_gamma"] == gamma) / len(rows)
            for gamma in GAMMA_GRID
        },
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--small-reps", type=int, default=4000)
    parser.add_argument("--medium-reps", type=int, default=1500)
    parser.add_argument("--large-reps", type=int, default=400)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--write-replication-rows", action="store_true")
    args = parser.parse_args()

    n_grid = ((2000, args.small_reps), (8000, args.medium_reps), (32000, args.large_reps))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    summary_rows: list[dict[str, object]] = []
    replication_rows: list[dict[str, object]] = []
    for scenario in _scenarios():
        for n_total, reps in n_grid:
            print(f"running {scenario.name} n={n_total} reps={reps}", flush=True)
            rows = []
            for rep in range(reps):
                row = _one_run(scenario, n_total, args.folds, rng)
                rows.append(row)
                if args.write_replication_rows:
                    replication_rows.append(
                        {
                            "scenario": scenario.name,
                            "n_total": n_total,
                            "folds": args.folds,
                            "rep": rep,
                            **row,
                        }
                    )
            summary_rows.append(
                {
                    "scenario": scenario.name,
                    "n_total": n_total,
                    "folds": args.folds,
                    **_summarize_rows(rows),
                }
            )
    _write_csv(out_dir / "current_algorithm_selection_penalty_summary.csv", summary_rows)
    if args.write_replication_rows:
        _write_csv(out_dir / "current_algorithm_selection_penalty_replication_rows.csv", replication_rows)

    lines = [
        "# Current Algorithm 1 Selection-Penalty Experiment",
        "",
        "This non-overwriting experiment runs the current full-data cross-fitted",
        "weighted-residual selector.  It compares the actual same-sample returned",
        "estimator with the gamma-zero baseline and with the selected candidate's",
        "fresh-sample risk.",
        "",
        "## Summary",
        "",
    ]
    for row in summary_rows:
        lines.append(
            "- {scenario}, n={n}, reps={reps}: same-sample MSE gain {same_gain:.3f}%, "
            "fresh-risk gain {fresh_gain:.3f}%, oracle fresh-risk gain {oracle_gain:.3f}%, "
            "mean same-sample penalty {penalty:.6g} ({penalty_ratio:.3f}x selected fresh risk), "
            "fresh harm {fresh_harm:.3f}, same-sample harm {same_harm:.3f}, "
            "selected shares gamma0={g0:.3f}, gamma0.25={g025:.3f}, gamma0.5={g05:.3f}, "
            "gamma1={g1:.3f}.".format(
                scenario=row["scenario"],
                n=row["n_total"],
                reps=row["reps"],
                same_gain=row["same_sample_mse_gain_pct"],
                fresh_gain=row["fresh_risk_gain_pct"],
                oracle_gain=row["oracle_fresh_risk_gain_pct"],
                penalty=row["mean_same_sample_penalty"],
                penalty_ratio=row["same_sample_penalty_over_selected_risk"],
                fresh_harm=row["fresh_harm_share"],
                same_harm=row["same_sample_harm_share"],
                g0=row["selected_share_gamma_0"],
                g025=row["selected_share_gamma_0.25"],
                g05=row["selected_share_gamma_0.5"],
                g1=row["selected_share_gamma_1"],
            )
        )
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")
    provenance = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "dependency": str((Path(__file__).resolve().parent / "dml_honest_split_selected_gamma_experiment.py")),
        "dependency_sha256": _sha256(Path(__file__).resolve().parent / "dml_honest_split_selected_gamma_experiment.py"),
        "seed": args.seed,
        "folds": args.folds,
        "gamma_grid": list(GAMMA_GRID),
        "n_grid": [
            {"n_total": n_total, "reps": reps}
            for n_total, reps in n_grid
        ],
        "write_replication_rows": args.write_replication_rows,
        "scope": "finite-support current Algorithm 1 same-sample selection-penalty diagnostic; not a Section 4 benchmark run",
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
