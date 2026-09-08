#!/usr/bin/env python3
"""Honest train/validation/estimation experiment for selected gamma.

This script implements the theorem-compatible experiment design:

1. train data fit the reference outcome model and residual direction;
2. validation data select gamma by the one-SE weighted-residual rule; and
3. an untouched estimation sample computes the returned AIPW average.

The output is intentionally separate from the paper-facing Section 4 bundles.
It is a numerical check that the algorithm/experiment pairing is coherent when
selection and estimation use independent samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


GAMMA_GRID = (0.0, 0.25, 0.5, 1.0)


@dataclass(frozen=True)
class BaseScenario:
    name: str
    q: tuple[float, ...]
    z: tuple[float, ...]
    mu: tuple[float, ...]
    sigma: tuple[float, ...]
    pi: tuple[float, ...]
    p: tuple[float, ...]


@dataclass(frozen=True)
class FittedPath:
    scenario: BaseScenario
    m0: tuple[float, ...]
    h: tuple[float, ...]


@dataclass(frozen=True)
class SampleAgg:
    n: int
    counts: tuple[int, ...]
    response_counts: tuple[int, ...]
    response_sum_y: tuple[float, ...]
    response_sum_y2: tuple[float, ...]


def _normalize(values: Iterable[float]) -> tuple[float, ...]:
    items = tuple(float(value) for value in values)
    total = sum(items)
    return tuple(value / total for value in items)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _scenarios() -> list[BaseScenario]:
    q = _normalize((0.07, 0.10, 0.13, 0.08, 0.16, 0.11, 0.14, 0.21))
    z = (-1.7, -1.1, -0.55, -0.05, 0.35, 0.85, 1.25, 1.75)
    pi = (0.16, 0.22, 0.34, 0.46, 0.58, 0.70, 0.82, 0.28)
    sigma = (0.8, 1.1, 0.7, 1.3, 0.8, 1.0, 1.2, 0.9)
    mu_clear = tuple(0.35 + 0.8 * x - 0.55 * x * x + 0.35 * math.sin(2.0 * x) for x in z)
    mu_near = tuple(0.20 + 0.70 * x + 0.08 * x * x + 0.06 * math.sin(2.0 * x) for x in z)
    mu_tail = tuple(0.15 + 0.50 * x - 0.75 * x * x + 0.55 * (x < -0.8) for x in z)
    p_miss = tuple(_clip(0.82 * p_i + 0.06 + 0.06 * math.sin(1.5 * x), 0.12, 0.88) for p_i, x in zip(pi, z))
    p_clip = tuple(max(p_i, 0.30) for p_i in pi)
    return [
        BaseScenario("correct_response_clear", q, z, mu_clear, sigma, pi, pi),
        BaseScenario("correct_response_near_tie", q, z, mu_near, sigma, pi, pi),
        BaseScenario("misspecified_response_tradeoff", q, z, mu_clear, sigma, pi, p_miss),
        BaseScenario("clipped_response_tail", q, z, mu_tail, sigma, pi, p_clip),
    ]


def _dot(q: Sequence[float], values: Iterable[float]) -> float:
    return sum(q_i * value for q_i, value in zip(q, values))


def _basis_reference(z: float) -> tuple[float, ...]:
    return (1.0, z)


def _basis_residual(z: float) -> tuple[float, ...]:
    return (1.0, z, z * z, math.sin(2.0 * z))


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    aug = [row[:] + [rhs_i] for row, rhs_i in zip(matrix, rhs)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-12:
            continue
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        for item in range(col, n + 1):
            aug[col][item] /= scale
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0.0:
                continue
            for item in range(col, n + 1):
                aug[row][item] -= factor * aug[col][item]
    return [aug[row][n] for row in range(n)]


def _weighted_least_squares(
    rows: Sequence[tuple[tuple[float, ...], float, float]],
    width: int,
    ridge: float = 1e-6,
) -> tuple[float, ...]:
    xtx = [[0.0 for _ in range(width)] for _ in range(width)]
    xty = [0.0 for _ in range(width)]
    for basis, y_value, weight in rows:
        if weight <= 0.0:
            continue
        for i in range(width):
            xty[i] += weight * basis[i] * y_value
            for j in range(width):
                xtx[i][j] += weight * basis[i] * basis[j]
    for i in range(width):
        xtx[i][i] += ridge
    return tuple(_solve_linear_system(xtx, xty))


def _predict(beta: Sequence[float], basis: Sequence[float]) -> float:
    return sum(b_i * x_i for b_i, x_i in zip(beta, basis))


def _draw_atom(scenario: BaseScenario, rng: random.Random, cumulative: Sequence[float]) -> int:
    u = rng.random()
    for index, cutoff in enumerate(cumulative):
        if u <= cutoff:
            return index
    return len(cumulative) - 1


def _draw_sample(scenario: BaseScenario, n: int, rng: random.Random) -> SampleAgg:
    cumulative = []
    acc = 0.0
    for value in scenario.q:
        acc += value
        cumulative.append(acc)
    counts = [0 for _ in scenario.q]
    response_counts = [0 for _ in scenario.q]
    response_sum_y = [0.0 for _ in scenario.q]
    response_sum_y2 = [0.0 for _ in scenario.q]
    for _ in range(n):
        index = _draw_atom(scenario, rng, cumulative)
        counts[index] += 1
        response = 1.0 if rng.random() < scenario.pi[index] else 0.0
        if response:
            y_value = scenario.mu[index] + scenario.sigma[index] * rng.gauss(0.0, 1.0)
            response_counts[index] += 1
            response_sum_y[index] += y_value
            response_sum_y2[index] += y_value * y_value
    return SampleAgg(
        n=n,
        counts=tuple(counts),
        response_counts=tuple(response_counts),
        response_sum_y=tuple(response_sum_y),
        response_sum_y2=tuple(response_sum_y2),
    )


def _fit_path(scenario: BaseScenario, train: SampleAgg) -> FittedPath:
    reference_rows = []
    for index, response_count in enumerate(train.response_counts):
        if response_count:
            mean_y = train.response_sum_y[index] / response_count
            reference_rows.append((_basis_reference(scenario.z[index]), mean_y, float(response_count)))
    beta_m0 = _weighted_least_squares(reference_rows, len(_basis_reference(0.0)))
    m0 = tuple(_predict(beta_m0, _basis_reference(z_i)) for z_i in scenario.z)
    residual_rows = []
    for index, response_count in enumerate(train.response_counts):
        if response_count:
            mean_y = train.response_sum_y[index] / response_count
            p_i = scenario.p[index]
            weight = response_count * (1.0 - p_i) / (p_i * p_i)
            residual_rows.append((_basis_residual(scenario.z[index]), mean_y - m0[index], weight))
    beta_h = _weighted_least_squares(residual_rows, len(_basis_residual(0.0)))
    h = tuple(_predict(beta_h, _basis_residual(z_i)) for z_i in scenario.z)
    return FittedPath(scenario, m0, h)


def _m(path: FittedPath, gamma: float) -> tuple[float, ...]:
    return tuple(m0_i + gamma * h_i for m0_i, h_i in zip(path.m0, path.h))


def _theta(scenario: BaseScenario) -> float:
    return _dot(scenario.q, scenario.mu)


def _weighted_l2(path: FittedPath, values: Iterable[float]) -> float:
    scenario = path.scenario
    terms = []
    for pi_i, p_i, value in zip(scenario.pi, scenario.p, values):
        terms.append(pi_i * (1.0 - p_i) / (p_i * p_i) * value * value)
    return _dot(scenario.q, terms)


def _bias(path: FittedPath, gamma: float) -> float:
    scenario = path.scenario
    m = _m(path, gamma)
    terms = []
    for p_i, pi_i, m_i, mu_i in zip(scenario.p, scenario.pi, m, scenario.mu):
        terms.append((p_i - pi_i) / p_i * (m_i - mu_i))
    return _dot(scenario.q, terms)


def _variance(path: FittedPath, gamma: float) -> float:
    scenario = path.scenario
    m = _m(path, gamma)
    theta = _theta(scenario)
    cond_var = []
    cond_mean = []
    for pi_i, p_i, m_i, mu_i, sigma_i in zip(
        scenario.pi, scenario.p, m, scenario.mu, scenario.sigma
    ):
        e_i = m_i - mu_i
        cond_var.append(
            pi_i / (p_i * p_i) * sigma_i * sigma_i
            + pi_i * (1.0 - pi_i) / (p_i * p_i) * e_i * e_i
        )
        cond_mean.append(mu_i + (p_i - pi_i) / p_i * e_i)
    return _dot(scenario.q, cond_var) + _dot(scenario.q, (x * x for x in cond_mean)) - theta * theta


def _risk(path: FittedPath, gamma: float, n_est: int) -> float:
    bias = _bias(path, gamma)
    return bias * bias + _variance(path, gamma) / float(n_est)


def _population_gain(path: FittedPath, gamma: float) -> float:
    scenario = path.scenario
    e0 = tuple(m0_i - mu_i for m0_i, mu_i in zip(path.m0, scenario.mu))
    eg = tuple(m_i - mu_i for m_i, mu_i in zip(_m(path, gamma), scenario.mu))
    terms = []
    for pi_i, p_i, e0_i, eg_i in zip(scenario.pi, scenario.p, e0, eg):
        weight = pi_i * (1.0 - p_i) / (p_i * p_i)
        terms.append(weight * (e0_i * e0_i - eg_i * eg_i))
    return _dot(scenario.q, terms)


def _bbar(path: FittedPath) -> float:
    scenario = path.scenario
    xi = tuple(p_i - pi_i for p_i, pi_i in zip(scenario.p, scenario.pi))
    delta = max(abs(value) for value in xi)
    if delta == 0.0:
        return 0.0
    eta = min(
        min(scenario.pi),
        min(scenario.p),
        1.0 - max(scenario.pi),
        1.0 - max(scenario.p),
    )
    mu_mean = _theta(scenario)
    s_mu = math.sqrt(max(0.0, _dot(scenario.q, ((x - mu_mean) ** 2 for x in scenario.mu))))
    out = 0.0
    for gamma in GAMMA_GRID:
        total = 0.0
        for model in (path.m0, _m(path, gamma)):
            err_l2 = math.sqrt(_dot(scenario.q, ((m_i - mu_i) ** 2 for m_i, mu_i in zip(model, scenario.mu))))
            total += (
                2.0 * s_mu * delta / eta * err_l2
                + (delta + delta * delta) / (eta * eta) * err_l2 * err_l2
            )
        out = max(out, total)
    return out


def _rho2(path: FittedPath) -> float:
    scenario = path.scenario
    terms = []
    for p_i, pi_i in zip(scenario.p, scenario.pi):
        xi = p_i - pi_i
        terms.append(xi * xi / (pi_i * (1.0 - p_i)))
    return _dot(scenario.q, terms)


def _select_gamma(
    path: FittedPath,
    validation: SampleAgg,
) -> tuple[float, float, float]:
    scenario = path.scenario
    sums = {gamma: {"gain": 0.0, "gain2": 0.0} for gamma in GAMMA_GRID}
    for index, response_count in enumerate(validation.response_counts):
        if response_count == 0:
            continue
        p_i = scenario.p[index]
        weight = (1.0 - p_i) / (p_i * p_i)
        m0_i = path.m0[index]
        h_i = path.h[index]
        resid_sum = validation.response_sum_y[index] - response_count * m0_i
        resid2_sum = (
            validation.response_sum_y2[index]
            - 2.0 * m0_i * validation.response_sum_y[index]
            + response_count * m0_i * m0_i
        )
        for gamma in GAMMA_GRID:
            linear = 2.0 * gamma * h_i
            constant = gamma * gamma * h_i * h_i
            gain_sum = weight * (linear * resid_sum - constant * response_count)
            gain2_sum = weight * weight * (
                linear * linear * resid2_sum
                - 2.0 * linear * constant * resid_sum
                + constant * constant * response_count
            )
            sums[gamma]["gain"] += gain_sum
            sums[gamma]["gain2"] += gain2_sum
    n_val = validation.n
    gains = {}
    max_se = 0.0
    eligible = [0.0]
    for gamma in GAMMA_GRID:
        mean_gain = sums[gamma]["gain"] / n_val
        gains[gamma] = mean_gain
        if gamma == 0.0:
            se = 0.0
        else:
            variance = max(0.0, (sums[gamma]["gain2"] - n_val * mean_gain * mean_gain) / (n_val - 1))
            se = math.sqrt(variance / n_val)
        max_se = max(max_se, se)
        if gamma != 0.0 and mean_gain > se:
            eligible.append(gamma)
    selected = max(eligible, key=lambda gamma: gains[gamma])
    delta_n = max(abs(gains[gamma] - _population_gain(path, gamma)) for gamma in GAMMA_GRID)
    return selected, delta_n, max_se


def _estimate(
    path: FittedPath,
    sample: SampleAgg,
    gamma: float,
) -> float:
    scenario = path.scenario
    total = 0.0
    for index, count in enumerate(sample.counts):
        p_i = scenario.p[index]
        m_i = path.m0[index] + gamma * path.h[index]
        response_count = sample.response_counts[index]
        total += count * m_i
        total += (sample.response_sum_y[index] - response_count * m_i) / p_i
    return total / sample.n


def _one_run(
    scenario: BaseScenario,
    n_train: int,
    n_val: int,
    n_est: int,
    rng: random.Random,
) -> dict[str, float]:
    train = _draw_sample(scenario, n_train, rng)
    validation = _draw_sample(scenario, n_val, rng)
    estimation = _draw_sample(scenario, n_est, rng)
    path = _fit_path(scenario, train)
    selected_gamma, delta_n, max_se = _select_gamma(path, validation)
    theta = _theta(scenario)
    baseline_estimate = _estimate(path, estimation, 0.0)
    selected_estimate = _estimate(path, estimation, selected_gamma)
    reused_validation_estimate = _estimate(path, validation, selected_gamma)
    risks = {gamma: _risk(path, gamma, n_est) for gamma in GAMMA_GRID}
    oracle_gamma = min(GAMMA_GRID, key=lambda gamma: risks[gamma])
    bbar = _bbar(path)
    rho2 = _rho2(path)
    e0 = _weighted_l2(path, (m0_i - mu_i for m0_i, mu_i in zip(path.m0, scenario.mu)))
    d1 = _weighted_l2(path, path.h)
    response_cost_scale = 2.0 * rho2 * math.sqrt(d1) * (math.sqrt(e0) + math.sqrt(d1))
    bound = (
        (2.0 * bbar + 2.0 * delta_n + max_se) / n_est
        + response_cost_scale * abs(selected_gamma - oracle_gamma)
    )
    rhs = risks[oracle_gamma] + bound
    return {
        "selected_gamma": selected_gamma,
        "oracle_gamma": oracle_gamma,
        "baseline_sq_error": (baseline_estimate - theta) ** 2,
        "selected_sq_error": (selected_estimate - theta) ** 2,
        "reused_validation_sq_error": (reused_validation_estimate - theta) ** 2,
        "selected_fresh_risk": risks[selected_gamma],
        "baseline_fresh_risk": risks[0.0],
        "oracle_fresh_risk": risks[oracle_gamma],
        "theorem_rhs": rhs,
        "theorem_bound": bound,
        "fresh_violation": float(risks[selected_gamma] > rhs + 1e-12),
        "estimation_sq_error_over_rhs": float((selected_estimate - theta) ** 2 > rhs + 1e-12),
    }


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items)


def _summarize_rows(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    selected_mse = _mean(row["selected_sq_error"] for row in rows)
    baseline_mse = _mean(row["baseline_sq_error"] for row in rows)
    selected_risk = _mean(row["selected_fresh_risk"] for row in rows)
    baseline_risk = _mean(row["baseline_fresh_risk"] for row in rows)
    oracle_risk = _mean(row["oracle_fresh_risk"] for row in rows)
    reused_mse = _mean(row["reused_validation_sq_error"] for row in rows)
    return {
        "reps": len(rows),
        "selected_mse": selected_mse,
        "baseline_mse": baseline_mse,
        "mse_gain_pct": 100.0 * (baseline_mse - selected_mse) / baseline_mse,
        "selected_fresh_risk": selected_risk,
        "baseline_fresh_risk": baseline_risk,
        "fresh_risk_gain_pct": 100.0 * (baseline_risk - selected_risk) / baseline_risk,
        "oracle_fresh_risk": oracle_risk,
        "oracle_gain_pct": 100.0 * (baseline_risk - oracle_risk) / baseline_risk,
        "mean_theorem_rhs": _mean(row["theorem_rhs"] for row in rows),
        "fresh_violations": int(sum(row["fresh_violation"] for row in rows)),
        "estimation_sq_error_over_rhs_count": int(sum(row["estimation_sq_error_over_rhs"] for row in rows)),
        "reused_validation_mse": reused_mse,
        "reused_to_honest_ratio": reused_mse / selected_mse if selected_mse else math.nan,
        **{
            f"selected_share_gamma_{gamma:g}": sum(1.0 for row in rows if row["selected_gamma"] == gamma) / len(rows)
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
    parser.add_argument("--small-reps", type=int, default=3000)
    parser.add_argument("--medium-reps", type=int, default=1200)
    parser.add_argument("--large-reps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--write-replication-rows", action="store_true")
    args = parser.parse_args()

    split_grid = (
        (1000, 500, 500, args.small_reps),
        (4000, 2000, 2000, args.medium_reps),
        (16000, 8000, 8000, args.large_reps),
    )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    summary_rows: list[dict[str, object]] = []
    replication_rows: list[dict[str, object]] = []
    for scenario in _scenarios():
        for n_train, n_val, n_est, reps in split_grid:
            print(
                f"running {scenario.name} train/validation/estimation="
                f"{n_train}/{n_val}/{n_est} reps={reps}",
                flush=True,
            )
            rows = []
            for rep in range(reps):
                row = _one_run(scenario, n_train, n_val, n_est, rng)
                rows.append(row)
                if args.write_replication_rows:
                    replication_rows.append(
                        {
                            "scenario": scenario.name,
                            "n_train": n_train,
                            "n_validation": n_val,
                            "n_estimation": n_est,
                            "rep": rep,
                            **row,
                        }
                    )
            summary = _summarize_rows(rows)
            summary_rows.append(
                {
                    "scenario": scenario.name,
                    "n_train": n_train,
                    "n_validation": n_val,
                    "n_estimation": n_est,
                    **summary,
                }
            )
    _write_csv(out_dir / "honest_split_selected_gamma_summary.csv", summary_rows)
    if args.write_replication_rows:
        _write_csv(out_dir / "honest_split_selected_gamma_replication_rows.csv", replication_rows)

    lines = [
        "# Honest-Split Selected-Gamma Experiment",
        "",
        "This non-overwriting experiment uses a train/validation/estimation split.",
        "The reference outcome model and residual direction are fit on training",
        "data, gamma is selected on validation data by the one-SE weighted-residual",
        "rule, and the returned AIPW estimate is computed on an untouched",
        "estimation sample.",
        "",
        "## Summary",
        "",
    ]
    for row in summary_rows:
        lines.append(
            "- {scenario}, train/validation/estimation={n_train}/{n_val}/{n_est}, "
            "reps={reps}: MSE gain {gain:.3f}%, fresh-risk gain {risk_gain:.3f}%, "
            "oracle fresh-risk gain {oracle_gain:.3f}%, fresh violations "
            "{violations}/{reps}, estimation squared-error over RHS {est_over}/{reps}, "
            "selected shares gamma0={g0:.3f}, gamma0.25={g025:.3f}, "
            "gamma0.5={g05:.3f}, gamma1={g1:.3f}.".format(
                scenario=row["scenario"],
                n_train=row["n_train"],
                n_val=row["n_validation"],
                n_est=row["n_estimation"],
                reps=row["reps"],
                gain=row["mse_gain_pct"],
                risk_gain=row["fresh_risk_gain_pct"],
                oracle_gain=row["oracle_gain_pct"],
                violations=row["fresh_violations"],
                est_over=row["estimation_sq_error_over_rhs_count"],
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
        "seed": args.seed,
        "gamma_grid": list(GAMMA_GRID),
        "splits": [
            {"n_train": n_train, "n_validation": n_val, "n_estimation": n_est, "reps": reps}
            for n_train, n_val, n_est, reps in split_grid
        ],
        "write_replication_rows": args.write_replication_rows,
        "scope": "finite-support honest split selected-gamma experiment; not a Section 4 benchmark run",
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
