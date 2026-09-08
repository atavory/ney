#!/usr/bin/env python3
"""Finite-support checks for the selected-gamma oracle theorem.

The script simulates the exact one-standard-error weighted-residual selector
used in the current Algorithm 1 notation.  It reports two distinct quantities:

1. the theorem object, the fresh-sample candidate risk
   b(m_gamma,p)^2 + V(m_gamma,p) / n after gamma has been selected; and
2. the same-sample squared error of the returned sample average.

The second quantity is diagnostic only.  It is included to keep the theorem
scope explicit and prevent confusing fresh-candidate risk with realized
same-sample error.
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
from typing import Iterable


GAMMA_GRID = (0.0, 0.25, 0.5, 1.0)


@dataclass(frozen=True)
class Scenario:
    name: str
    q: tuple[float, ...]
    mu: tuple[float, ...]
    sigma: tuple[float, ...]
    pi: tuple[float, ...]
    p: tuple[float, ...]
    m0: tuple[float, ...]
    h: tuple[float, ...]


def _normalize(values: Iterable[float]) -> tuple[float, ...]:
    items = tuple(float(value) for value in values)
    total = sum(items)
    return tuple(value / total for value in items)


def _scenarios() -> list[Scenario]:
    q = _normalize((0.07, 0.10, 0.13, 0.08, 0.16, 0.11, 0.14, 0.21))
    mu = (-1.2, -0.5, 0.25, 1.1, -0.9, 0.7, 1.6, -1.5)
    sigma = (0.8, 1.1, 0.6, 1.4, 0.7, 1.0, 1.3, 0.9)
    pi = (0.18, 0.24, 0.36, 0.48, 0.62, 0.74, 0.82, 0.30)
    err = (1.4, -0.8, 0.6, -1.2, 0.35, -0.5, 0.85, -0.75)
    m0 = tuple(mu_i + err_i for mu_i, err_i in zip(mu, err))
    h_clear = tuple(
        -0.70 * err_i + tweak
        for err_i, tweak in zip(err, (0.20, 0.05, -0.10, 0.12, -0.08, 0.04, -0.15, 0.10))
    )
    h_screen = tuple(
        -0.30 * err_i + 0.50 * tweak
        for err_i, tweak in zip(err, (0.8, -1.1, 0.2, 0.9, -0.6, 1.2, -0.3, -0.7))
    )
    p_miss = (0.26, 0.19, 0.42, 0.56, 0.54, 0.66, 0.78, 0.38)
    h_miss = tuple(
        -0.45 * err_i + tweak
        for err_i, tweak in zip(err, (-0.10, 0.16, -0.04, 0.18, 0.05, -0.14, 0.10, -0.08))
    )
    return [
        Scenario("correct_clear", q, mu, sigma, pi, pi, m0, h_clear),
        Scenario("correct_near_tie", q, mu, sigma, pi, pi, m0, h_screen),
        Scenario("misspecified_tradeoff", q, mu, sigma, pi, p_miss, m0, h_miss),
    ]


def _dot(q: tuple[float, ...], values: Iterable[float]) -> float:
    return sum(q_i * value for q_i, value in zip(q, values))


def _weighted_l2(scenario: Scenario, values: Iterable[float]) -> float:
    terms = []
    for pi_i, p_i, value in zip(scenario.pi, scenario.p, values):
        terms.append(pi_i * (1.0 - p_i) / (p_i * p_i) * value * value)
    return _dot(scenario.q, terms)


def _theta(scenario: Scenario) -> float:
    return _dot(scenario.q, scenario.mu)


def _m(scenario: Scenario, gamma: float) -> tuple[float, ...]:
    return tuple(m0_i + gamma * h_i for m0_i, h_i in zip(scenario.m0, scenario.h))


def _bias(scenario: Scenario, gamma: float) -> float:
    m = _m(scenario, gamma)
    terms = []
    for p_i, pi_i, m_i, mu_i in zip(scenario.p, scenario.pi, m, scenario.mu):
        terms.append((p_i - pi_i) / p_i * (m_i - mu_i))
    return _dot(scenario.q, terms)


def _variance(scenario: Scenario, gamma: float) -> float:
    m = _m(scenario, gamma)
    theta = _theta(scenario)
    cond_var = []
    cond_mean = []
    for pi_i, p_i, m_i, mu_i, sigma_i in zip(
        scenario.pi, scenario.p, m, scenario.mu, scenario.sigma
    ):
        e_i = m_i - mu_i
        cond_var.append(pi_i / (p_i * p_i) * sigma_i * sigma_i + pi_i * (1.0 - pi_i) / (p_i * p_i) * e_i * e_i)
        cond_mean.append(mu_i + (p_i - pi_i) / p_i * e_i)
    return _dot(scenario.q, cond_var) + _dot(scenario.q, (x * x for x in cond_mean)) - theta * theta


def _risk(scenario: Scenario, gamma: float, n: int) -> float:
    b = _bias(scenario, gamma)
    return b * b + _variance(scenario, gamma) / float(n)


def _population_gain(scenario: Scenario, gamma: float) -> float:
    e0 = tuple(m0_i - mu_i for m0_i, mu_i in zip(scenario.m0, scenario.mu))
    eg = tuple(m_i - mu_i for m_i, mu_i in zip(_m(scenario, gamma), scenario.mu))
    terms = []
    for pi_i, p_i, e0_i, eg_i in zip(scenario.pi, scenario.p, e0, eg):
        weight = pi_i * (1.0 - p_i) / (p_i * p_i)
        terms.append(weight * (e0_i * e0_i - eg_i * eg_i))
    return _dot(scenario.q, terms)


def _bbar(scenario: Scenario) -> float:
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
        for model in (scenario.m0, _m(scenario, gamma)):
            err_l2 = math.sqrt(_dot(scenario.q, ((m_i - mu_i) ** 2 for m_i, mu_i in zip(model, scenario.mu))))
            total += (
                2.0 * s_mu * delta / eta * err_l2
                + (delta + delta * delta) / (eta * eta) * err_l2 * err_l2
            )
        out = max(out, total)
    return out


def _rho2(scenario: Scenario) -> float:
    terms = []
    for p_i, pi_i in zip(scenario.p, scenario.pi):
        xi = p_i - pi_i
        terms.append(xi * xi / (pi_i * (1.0 - p_i)))
    return _dot(scenario.q, terms)


def _draw_atom(scenario: Scenario, rng: random.Random, cumulative: tuple[float, ...]) -> int:
    u = rng.random()
    for index, cutoff in enumerate(cumulative):
        if u <= cutoff:
            return index
    return len(cumulative) - 1


def _one_run(scenario: Scenario, n: int, rng: random.Random) -> dict[str, float]:
    cumulative = []
    acc = 0.0
    for value in scenario.q:
        acc += value
        cumulative.append(acc)
    theta = _theta(scenario)
    ea = _dot(
        scenario.q,
        (
            pi_i * (1.0 - p_i) / (p_i * p_i) * (mu_i - m0_i) * h_i
            for pi_i, p_i, mu_i, m0_i, h_i in zip(
                scenario.pi, scenario.p, scenario.mu, scenario.m0, scenario.h
            )
        ),
    )
    eb = _weighted_l2(scenario, scenario.h)
    sums = {gamma: {"gain": 0.0, "gain2": 0.0, "score": 0.0} for gamma in GAMMA_GRID}
    sum_a = 0.0
    sum_b = 0.0
    for _ in range(n):
        index = _draw_atom(scenario, rng, tuple(cumulative))
        p_i = scenario.p[index]
        response = 1.0 if rng.random() < scenario.pi[index] else 0.0
        y_i = scenario.mu[index] + scenario.sigma[index] * rng.gauss(0.0, 1.0)
        h_i = scenario.h[index]
        m0_i = scenario.m0[index]
        weight = response * (1.0 - p_i) / (p_i * p_i)
        a_i = weight * (y_i - m0_i) * h_i
        b_i = weight * h_i * h_i
        sum_a += a_i
        sum_b += b_i
        for gamma in GAMMA_GRID:
            gain_i = 2.0 * gamma * a_i - gamma * gamma * b_i
            m_i = m0_i + gamma * h_i
            score_i = m_i + response / p_i * (y_i - m_i)
            sums[gamma]["gain"] += gain_i
            sums[gamma]["gain2"] += gain_i * gain_i
            sums[gamma]["score"] += score_i
    eligible = [0.0]
    gains = {}
    max_se = 0.0
    for gamma in GAMMA_GRID:
        mean_gain = sums[gamma]["gain"] / n
        gains[gamma] = mean_gain
        if gamma == 0.0:
            se = 0.0
        else:
            variance = max(0.0, (sums[gamma]["gain2"] - n * mean_gain * mean_gain) / (n - 1))
            se = math.sqrt(variance / n)
        max_se = max(max_se, se)
        if gamma != 0.0 and mean_gain > se:
            eligible.append(gamma)
    selected_gamma = max(eligible, key=lambda gamma: gains[gamma])
    selected_estimate = sums[selected_gamma]["score"] / n
    delta_n = 2.0 * abs(sum_a / n - ea) + abs(sum_b / n - eb)
    risks = {gamma: _risk(scenario, gamma, n) for gamma in GAMMA_GRID}
    oracle_gamma = min(GAMMA_GRID, key=lambda gamma: risks[gamma])
    return {
        "selected_gamma": selected_gamma,
        "oracle_gamma": oracle_gamma,
        "fresh_risk": risks[selected_gamma],
        "oracle_risk": risks[oracle_gamma],
        "same_sample_sq_error": (selected_estimate - theta) ** 2,
        "delta_n": delta_n,
        "max_se": max_se,
    }


def _summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    count = len(ordered)
    return {
        "mean": sum(ordered) / count,
        "median": ordered[count // 2],
        "p90": ordered[int(0.90 * (count - 1))],
        "max": ordered[-1],
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
    parser.add_argument("--small-reps", type=int, default=6000)
    parser.add_argument("--medium-reps", type=int, default=3000)
    parser.add_argument("--large-reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--write-replication-rows", action="store_true")
    args = parser.parse_args()

    n_reps = ((500, args.small_reps), (2000, args.medium_reps), (8000, args.large_reps))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    summary_rows: list[dict[str, object]] = []
    replication_rows: list[dict[str, object]] = []
    for scenario in _scenarios():
        bbar = _bbar(scenario)
        rho2 = _rho2(scenario)
        e0 = _weighted_l2(scenario, (m_i - mu_i for m_i, mu_i in zip(scenario.m0, scenario.mu)))
        d1 = _weighted_l2(scenario, scenario.h)
        response_cost_scale = 2.0 * rho2 * math.sqrt(d1) * (math.sqrt(e0) + math.sqrt(d1))
        for n, reps in n_reps:
            fresh_violations = 0
            same_sample_over_rhs = 0
            selected_counts = {gamma: 0 for gamma in GAMMA_GRID}
            regret_values = []
            bound_values = []
            same_values = []
            fresh_values = []
            rhs_values = []
            for rep in range(reps):
                row = _one_run(scenario, n, rng)
                selected_counts[row["selected_gamma"]] += 1
                regret = row["fresh_risk"] - row["oracle_risk"]
                bound = (
                    (2.0 * bbar + 2.0 * row["delta_n"] + row["max_se"]) / n
                    + response_cost_scale * abs(row["selected_gamma"] - row["oracle_gamma"])
                )
                rhs = row["oracle_risk"] + bound
                fresh_violations += int(row["fresh_risk"] > rhs + 1e-12)
                same_sample_over_rhs += int(row["same_sample_sq_error"] > rhs + 1e-12)
                regret_values.append(regret)
                bound_values.append(bound)
                same_values.append(row["same_sample_sq_error"])
                fresh_values.append(row["fresh_risk"])
                rhs_values.append(rhs)
                if args.write_replication_rows:
                    replication_rows.append(
                        {
                            "scenario": scenario.name,
                            "n": n,
                            "rep": rep,
                            "selected_gamma": row["selected_gamma"],
                            "oracle_gamma": row["oracle_gamma"],
                            "fresh_risk": row["fresh_risk"],
                            "oracle_risk": row["oracle_risk"],
                            "same_sample_sq_error": row["same_sample_sq_error"],
                            "theorem_rhs": rhs,
                            "fresh_violation": row["fresh_risk"] > rhs + 1e-12,
                            "same_sample_over_rhs": row["same_sample_sq_error"] > rhs + 1e-12,
                        }
                    )
            regret_summary = _summarize(regret_values)
            bound_summary = _summarize(bound_values)
            summary_rows.append(
                {
                    "scenario": scenario.name,
                    "n": n,
                    "reps": reps,
                    "bbar": bbar,
                    "rho2": rho2,
                    "response_cost_scale": response_cost_scale,
                    "fresh_risk_violations": fresh_violations,
                    "same_sample_over_rhs_count": same_sample_over_rhs,
                    "mean_fresh_risk": sum(fresh_values) / reps,
                    "mean_same_sample_sq_error": sum(same_values) / reps,
                    "mean_theorem_rhs": sum(rhs_values) / reps,
                    "same_over_fresh_ratio": (sum(same_values) / reps) / (sum(fresh_values) / reps),
                    "mean_regret": regret_summary["mean"],
                    "median_regret": regret_summary["median"],
                    "p90_regret": regret_summary["p90"],
                    "max_regret": regret_summary["max"],
                    "mean_bound": bound_summary["mean"],
                    "max_bound": bound_summary["max"],
                    **{f"selected_share_gamma_{gamma:g}": selected_counts[gamma] / reps for gamma in GAMMA_GRID},
                }
            )
    _write_csv(out_dir / "selected_gamma_theorem_summary.csv", summary_rows)
    if args.write_replication_rows:
        _write_csv(out_dir / "selected_gamma_theorem_replication_rows.csv", replication_rows)
    lines = [
        "# Selected-Gamma Theorem Simulation",
        "",
        "This finite-support simulation checks the one-standard-error weighted-residual",
        "selector against the selected-candidate oracle inequality.  The theorem",
        "object is fresh-candidate risk after selection.  The same-sample squared",
        "error is reported only as a diagnostic.",
        "",
        "## Summary",
        "",
    ]
    for row in summary_rows:
        lines.append(
            "- {scenario}, n={n}: fresh violations {fresh}/{reps}; same-sample over RHS "
            "{same}/{reps}; selected shares gamma0={g0:.3f}, gamma0.25={g025:.3f}, "
            "gamma0.5={g05:.3f}, gamma1={g1:.3f}; mean fresh risk {fresh_risk:.6g}; "
            "mean same-sample squared error {same_risk:.6g}; mean RHS {rhs:.6g}.".format(
                scenario=row["scenario"],
                n=row["n"],
                fresh=row["fresh_risk_violations"],
                same=row["same_sample_over_rhs_count"],
                reps=row["reps"],
                g0=row["selected_share_gamma_0"],
                g025=row["selected_share_gamma_0.25"],
                g05=row["selected_share_gamma_0.5"],
                g1=row["selected_share_gamma_1"],
                fresh_risk=row["mean_fresh_risk"],
                same_risk=row["mean_same_sample_sq_error"],
                rhs=row["mean_theorem_rhs"],
            )
        )
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")
    provenance = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "seed": args.seed,
        "gamma_grid": list(GAMMA_GRID),
        "small_reps": args.small_reps,
        "medium_reps": args.medium_reps,
        "large_reps": args.large_reps,
        "write_replication_rows": args.write_replication_rows,
        "scope": "finite-support theorem simulation; not a Section 4 benchmark run",
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
