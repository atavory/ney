#!/usr/bin/env python3
from __future__ import annotations

"""
IV stage ablation on Card (1995) and 401(k) data.

Card (1995): Z=nearc4 (college proximity), D=educ, Y=lwage
  Compliance is heterogeneous: proximity binds for low-SES (black, south)
  but not high-SES. corr(nearc4,educ) = 0.189 black vs 0.109 white.

401(k): Z=e401 (eligibility), D=p401 (participation), Y=net_tfa
  Compliance ranges 0.63-0.79 by income quintile.

Stage ablation tests:
  unwt: standard 2SLS
  y_fit_only: weight outcome fit by estimated gamma(X)^2
  h_star_only: use score-aligned optimal instrument
  y_fit_h_star: both

Gamma(X) = E[D|X,Z=1] - E[D|X,Z=0] estimated by cross-fitted regression.
"""

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

os.environ.setdefault("OMP_NUM_THREADS", "1")

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

RESULT_FIELDS = [
    "experiment",
    "N",
    "learner",
    "capacity_param",
    "seed",
    "variant",
    "theta_est",
    "gamma_ratio",
    "first_stage_f",
]


@dataclass(frozen=True)
class Variant:
    name: str
    weight_y_fit: bool
    use_optimal_instrument: bool


VARIANTS = (
    Variant("unwt", False, False),
    Variant("y_fit_only", True, False),
    Variant("h_star_only", False, True),
    Variant("y_fit_h_star", True, True),
)


def load_card() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(os.path.join(DATA, "data_card.csv"))
    y = df["lwage"].values.astype(float)
    d = df["educ"].values.astype(float)
    z = df["nearc4"].values.astype(float)
    x_cols = [c for c in ["black", "smsa", "south", "exper", "married"] if c in df.columns]
    x = df[x_cols].values.astype(float)
    mask = np.isfinite(y) & np.isfinite(d) & np.isfinite(z) & np.all(np.isfinite(x), axis=1)
    return x[mask], y[mask], d[mask], z[mask]


def load_401k_iv() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(os.path.join(DATA, "data_401k.csv"))
    y = df["net_tfa"].values.astype(float)
    d = df["p401"].values.astype(float)
    z = df["e401"].values.astype(float)
    x = df[["age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown"]].values.astype(float)
    return x, y, d, z


def make_features(x: np.ndarray, degree: int) -> np.ndarray:
    x_std = StandardScaler().fit_transform(x)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    return StandardScaler().fit_transform(poly.fit_transform(x_std))


def estimate_compliance(
    x_poly: np.ndarray,
    d: np.ndarray,
    z: np.ndarray,
    seed: int,
) -> np.ndarray:
    gamma = np.zeros(len(d))
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed + 5555)
    for tr, te in splitter.split(x_poly):
        xz = np.column_stack([x_poly, z.reshape(-1, 1)])
        m = Ridge(alpha=1.0)
        m.fit(xz[tr], d[tr])
        pred_z1 = m.predict(np.column_stack([x_poly[te], np.ones((len(te), 1))]))
        pred_z0 = m.predict(np.column_stack([x_poly[te], np.zeros((len(te), 1))]))
        gamma[te] = pred_z1 - pred_z0
    return np.maximum(np.abs(gamma), 1e-6)


def first_stage_f(d: np.ndarray, z: np.ndarray) -> float:
    n = len(d)
    z1 = z == 1
    z0 = z == 0
    n1, n0 = z1.sum(), z0.sum()
    if n1 < 2 or n0 < 2:
        return 0.0
    d1_mean = d[z1].mean()
    d0_mean = d[z0].mean()
    pooled_var = (np.sum((d[z1] - d1_mean) ** 2) + np.sum((d[z0] - d0_mean) ** 2)) / (n - 2)
    if pooled_var < 1e-10:
        return 0.0
    return float((d1_mean - d0_mean) ** 2 / (pooled_var * (1.0 / n1 + 1.0 / n0)))


def fit_iv(
    x_poly: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    z: np.ndarray,
    gamma: np.ndarray,
    variant: Variant,
    seed: int,
) -> float:
    n = len(y)
    y_resid = np.zeros(n)
    d_resid = np.zeros(n)
    h_star = np.zeros(n)
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    sensitivity = gamma ** 2

    for tr, te in splitter.split(x_poly):
        y_w = sensitivity[tr] if variant.weight_y_fit else None
        y_model = Ridge(alpha=1.0)
        y_model.fit(x_poly[tr], y[tr], sample_weight=y_w)
        y_resid[te] = y[te] - y_model.predict(x_poly[te])

        d_model = Ridge(alpha=1.0)
        d_model.fit(x_poly[tr], d[tr])
        d_resid[te] = d[te] - d_model.predict(x_poly[te])

        if variant.use_optimal_instrument:
            gamma_model = Ridge(alpha=1.0)
            gamma_model.fit(x_poly[tr], gamma[tr])
            h_star[te] = gamma_model.predict(x_poly[te]) * z[te]
        else:
            fs_model = Ridge(alpha=1.0)
            fs_model.fit(np.column_stack([x_poly[tr], z[tr]]), d[tr])
            h_star[te] = fs_model.predict(np.column_stack([x_poly[te], z[te]]))

    instrument = h_star - np.mean(h_star)
    numerator = np.sum(instrument * y_resid)
    denominator = np.sum(instrument * d_resid)
    if abs(denominator) < 1e-10:
        return float("nan")
    return float(numerator / denominator)


def run_task(
    args: tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, Variant, int],
) -> dict[str, object]:
    experiment, x, y, d, z, degree, variant, seed = args
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.choice(n, size=n, replace=True)
    x_b, y_b, d_b, z_b = x[idx], y[idx], d[idx], z[idx]
    x_poly = make_features(x_b, degree)
    gamma = estimate_compliance(x_poly, d_b, z_b, seed)
    theta_hat = fit_iv(x_poly, y_b, d_b, z_b, gamma, variant, seed)
    return {
        "experiment": experiment,
        "N": n,
        "learner": "poly",
        "capacity_param": str(degree),
        "seed": seed,
        "variant": variant.name,
        "theta_est": theta_hat,
        "gamma_ratio": float(np.max(gamma) / np.min(gamma)),
        "first_stage_f": first_stage_f(d_b, z_b),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degrees", type=str, default="1,2,3")
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output",
        type=str,
        default="results/real_iv_card_401k_v1.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    degrees = tuple(int(s) for s in args.degrees.split(","))

    x_card, y_card, d_card, z_card = load_card()
    x_401k, y_401k, d_401k, z_401k = load_401k_iv()
    print(f"Card: n={len(y_card)}, p={x_card.shape[1]}")
    print(f"401k IV: n={len(y_401k)}, p={x_401k.shape[1]}")

    tasks = []
    for degree in degrees:
        for variant in VARIANTS:
            for seed in range(args.seeds):
                tasks.append(("card", x_card, y_card, d_card, z_card, degree, variant, seed))
                tasks.append(("401k_iv", x_401k, y_401k, d_401k, z_401k, degree, variant, seed))

    print(f"Running {len(tasks)} tasks")
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, row in enumerate(executor.map(run_task, tasks), start=1):
            rows.append(row)
            if index % 500 == 0:
                print(f"  completed {index}/{len(tasks)}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
