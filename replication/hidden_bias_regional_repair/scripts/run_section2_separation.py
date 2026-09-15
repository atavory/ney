#!/usr/bin/env python3
"""Generate raw rows for the Section 2 selection-separation experiment.

The data-generating process is the finite three-atom construction in the
paper's known-target lower bound.  Population risks are computed by explicit
enumeration of the observed-data state space, not from the displayed risk
contrast in the paper.  Both selectors are evaluated on the same realization:

* ``hidden_mse`` is the equal-prior likelihood-ratio test between P1 and P2,
  and hence the optimal observed-data selector for the MSE ordering;
* ``score_variance`` minimizes the empirical variance of the two AIPW scores.

One invocation writes one immutable raw-replication CSV.  A separate verifier
in the data archive checks coverage, provenance, all derived fields, and
regenerates summaries and figures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "section2-separation-v1"
DEFAULT_N_VALUES = (64, 128, 256, 512, 1024, 2048, 4096)


def _parse_n_values(text: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in text.split(",") if item.strip())
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("n-values must be unique and nonempty")
    return values


def _config_id(q: float, eps: float, a: float, b_y: float,
               p_star: float, n_values: tuple[int, ...]) -> str:
    payload = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "q": q,
            "epsilon": eps,
            "a": a,
            "b_y": b_y,
            "p_star": p_star,
            "n_values": n_values,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class Construction:
    """Exact finite-support construction used by Theorem 2.1."""

    def __init__(self, *, q: float, eps: float, a: float, b_y: float,
                 p_star: float, n: int) -> None:
        if not 0.0 < q <= 0.5:
            raise ValueError("q must lie in (0, 1/2]")
        if not 0.0 < eps <= 1.0:
            raise ValueError("epsilon must lie in (0, 1]")
        if not 0.0 < p_star <= 1.0 or p_star == eps:
            raise ValueError("p_star must lie in (0,1] and differ from epsilon")
        if a <= 0.0 or b_y <= 0.0:
            raise ValueError("a and B_Y must be positive")
        if n <= 2 or n * q * eps < 0.25:
            raise ValueError("need n>2 and n*q*epsilon>=1/4")
        self.q = q
        self.eps = eps
        self.a = a
        self.b_y = b_y
        self.p_star = p_star
        self.n = n
        self.gamma = b_y / (4.0 * math.sqrt(n * q * eps))
        self.kappa = 1.0 - eps / p_star
        if self.gamma > b_y / 2.0 + 1e-15:
            raise ValueError("outcome law invalid: gamma exceeds B_Y/2")

    def mu(self, distribution: int, atom: int) -> float:
        sign = 1.0 if distribution == 1 else -1.0
        return (0.0, sign * self.gamma, -sign * self.gamma)[atom]

    def observed_states(self, distribution: int) -> tuple[np.ndarray, ...]:
        """Return probabilities, scores, and log-LR sufficient statistic.

        There are eight positive-probability observed-data states: two outcome
        signs for each responding atom, plus one nonresponse state for each
        low-response atom.  Latent outcomes are integrated out when R=0.
        """
        masses = (1.0 - 2.0 * self.q, self.q, self.q)
        pis = (1.0, self.eps, self.eps)
        ps = (1.0, self.p_star, self.eps)
        m1s = (0.0, self.a, 0.0)
        m2s = (0.0, -self.a, 0.0)
        probs: list[float] = []
        t1s: list[float] = []
        t2s: list[float] = []
        lrt: list[int] = []
        for atom in range(3):
            mu = self.mu(distribution, atom)
            p_plus = 0.5 + mu / (2.0 * self.b_y)
            for y, py in ((self.b_y, p_plus), (-self.b_y, 1.0 - p_plus)):
                probs.append(masses[atom] * pis[atom] * py)
                t1s.append(m1s[atom] + (y - m1s[atom]) / ps[atom])
                t2s.append(m2s[atom] + (y - m2s[atom]) / ps[atom])
                if atom == 1:
                    lrt.append(1 if y > 0 else -1)
                elif atom == 2:
                    lrt.append(-1 if y > 0 else 1)
                else:
                    lrt.append(0)
            if pis[atom] < 1.0:
                probs.append(masses[atom] * (1.0 - pis[atom]))
                t1s.append(m1s[atom])
                t2s.append(m2s[atom])
                lrt.append(0)
        probability = np.asarray(probs, dtype=np.float64)
        probability /= probability.sum()
        return (
            probability,
            np.asarray(t1s, dtype=np.float64),
            np.asarray(t2s, dtype=np.float64),
            np.asarray(lrt, dtype=np.int64),
        )

    def population_risk(self, distribution: int, expert: int) -> tuple[float, float, float]:
        """Return MSE, average variance, and bias by state enumeration."""
        probability, t1, t2, _ = self.observed_states(distribution)
        score = t1 if expert == 1 else t2
        mean = float(probability @ score)
        second = float(probability @ (score * score))
        theta = sum(
            mass * self.mu(distribution, atom)
            for atom, mass in enumerate((1.0 - 2.0 * self.q, self.q, self.q))
        )
        bias = mean - theta
        variance = second - mean * mean
        return bias * bias + variance / self.n, variance / self.n, bias


FIELDS = (
    "schema_version", "config_id", "seed", "rep", "n", "distribution",
    "q", "epsilon", "a", "b_y", "p_star", "kappa", "gamma", "nqepsilon",
    "theta", "population_mse_1", "population_mse_2", "population_avar_1",
    "population_avar_2", "population_bias_1", "population_bias_2",
    "lrt_stat", "lrt_tie_coin", "hidden_mse_selector", "score_variance_selector",
    "vhat_1", "vhat_2", "hidden_mse_wrong", "score_variance_wrong",
    "hidden_mse_regret", "score_variance_regret",
)


def generate(args: argparse.Namespace) -> None:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    config_id = _config_id(args.q, args.epsilon, args.a, args.b_y,
                           args.p_star, args.n_values)
    root_seed = np.random.SeedSequence(args.seed)
    cell_seeds = root_seed.spawn(len(args.n_values) * 2)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        cell_index = 0
        for n in args.n_values:
            construction = Construction(
                q=args.q,
                eps=args.epsilon,
                a=args.a,
                b_y=args.b_y,
                p_star=args.p_star,
                n=n,
            )
            risks = {
                (r, j): construction.population_risk(r, j)
                for r in (1, 2) for j in (1, 2)
            }
            for distribution in (1, 2):
                rng = np.random.default_rng(cell_seeds[cell_index])
                cell_index += 1
                probability, t1, t2, lrt_weight = construction.observed_states(distribution)
                counts = rng.multinomial(n, probability, size=args.reps)
                lrt_stat = counts @ lrt_weight
                tie_coin = rng.integers(1, 3, size=args.reps)
                hidden_selector = np.where(lrt_stat > 0, 1,
                                           np.where(lrt_stat < 0, 2, tie_coin))
                sum_t1 = counts @ t1
                sum_t2 = counts @ t2
                vhat1 = counts @ (t1 * t1) / n - (sum_t1 / n) ** 2
                vhat2 = counts @ (t2 * t2) / n - (sum_t2 / n) ** 2
                variance_selector = np.where(vhat1 <= vhat2, 1, 2)

                mse1, avar1, bias1 = risks[(distribution, 1)]
                mse2, avar2, bias2 = risks[(distribution, 2)]
                best_mse = 1 if mse1 < mse2 else 2
                best_avar = 1 if avar1 < avar2 else 2
                mse_gap = abs(mse1 - mse2)
                avar_gap = abs(avar1 - avar2)
                hidden_wrong = hidden_selector != best_mse
                variance_wrong = variance_selector != best_avar

                for rep in range(args.reps):
                    writer.writerow({
                        "schema_version": SCHEMA_VERSION,
                        "config_id": config_id,
                        "seed": args.seed,
                        "rep": rep,
                        "n": n,
                        "distribution": distribution,
                        "q": format(args.q, ".17g"),
                        "epsilon": format(args.epsilon, ".17g"),
                        "a": format(args.a, ".17g"),
                        "b_y": format(args.b_y, ".17g"),
                        "p_star": format(args.p_star, ".17g"),
                        "kappa": format(construction.kappa, ".17g"),
                        "gamma": format(construction.gamma, ".17g"),
                        "nqepsilon": format(n * args.q * args.epsilon, ".17g"),
                        "theta": "0",
                        "population_mse_1": format(mse1, ".17g"),
                        "population_mse_2": format(mse2, ".17g"),
                        "population_avar_1": format(avar1, ".17g"),
                        "population_avar_2": format(avar2, ".17g"),
                        "population_bias_1": format(bias1, ".17g"),
                        "population_bias_2": format(bias2, ".17g"),
                        "lrt_stat": int(lrt_stat[rep]),
                        "lrt_tie_coin": int(tie_coin[rep]),
                        "hidden_mse_selector": int(hidden_selector[rep]),
                        "score_variance_selector": int(variance_selector[rep]),
                        "vhat_1": format(float(vhat1[rep]), ".17g"),
                        "vhat_2": format(float(vhat2[rep]), ".17g"),
                        "hidden_mse_wrong": int(hidden_wrong[rep]),
                        "score_variance_wrong": int(variance_wrong[rep]),
                        "hidden_mse_regret": format(mse_gap * hidden_wrong[rep], ".17g"),
                        "score_variance_regret": format(avar_gap * variance_wrong[rep], ".17g"),
                    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--n-values", type=_parse_n_values,
                        default=DEFAULT_N_VALUES)
    parser.add_argument("--q", type=float, default=0.25)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--b-y", type=float, default=1.0)
    parser.add_argument("--p-star", type=float, default=0.50)
    args = parser.parse_args()
    if args.reps <= 0:
        parser.error("reps must be positive")
    generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
