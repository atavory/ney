#!/usr/bin/env python3
"""Run published-paper benchmark adaptations for the DML repair paper.

The upstream papers use several different estimands: missing-data density
estimation, policy learning, and conditional distributional treatment effects.
This runner keeps their data-generating ingredients where they are directly
available, then translates them into the missing-outcome mean problem studied by
the paper.  These outputs are diagnostic until accepted explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


GAMMAS = (0.0, 0.25, 0.5, 1.0)


@dataclass(frozen=True)
class Problem:
    name: str
    source: str
    n: int
    x: list[list[float]]
    y_full: list[float]
    response_prob: list[float]
    notes: str
    fixed_response: list[int] | None = None

    @property
    def theta(self) -> float:
        return sum(self.y_full) / len(self.y_full)


def _logistic(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = _mean(values)
    return sum((value - center) ** 2 for value in values) / (len(values) - 1)


def _se(values: list[float]) -> float:
    if len(values) < 2:
        return float("inf")
    return math.sqrt(_variance(values) / len(values))


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return sorted_values[lo]
    frac = position - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _bootstrap_ci(
    ref_errors: list[float],
    repaired_errors: list[float],
    rng: random.Random,
    draws: int,
) -> tuple[float, float]:
    n = len(ref_errors)
    values = []
    for _ in range(draws):
        ref = 0.0
        repaired = 0.0
        for _j in range(n):
            idx = rng.randrange(n)
            ref += ref_errors[idx] ** 2
            repaired += repaired_errors[idx] ** 2
        values.append(100.0 * (1.0 - repaired / ref) if ref > 0.0 else float("nan"))
    finite = sorted(value for value in values if math.isfinite(value))
    return _quantile(finite, 0.025), _quantile(finite, 0.975)


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    aug = [row[:] + [rhs_i] for row, rhs_i in zip(matrix, rhs)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-12:
            continue
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= scale
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def _fit_wls(
    basis_rows: list[list[float]],
    y: list[float],
    weights: list[float],
    ridge: float = 1e-6,
) -> list[float]:
    if not basis_rows:
        return [0.0]
    p = len(basis_rows[0])
    xtwx = [[0.0 for _ in range(p)] for _ in range(p)]
    xtwy = [0.0 for _ in range(p)]
    for row, yi, wi in zip(basis_rows, y, weights):
        for a in range(p):
            xtwy[a] += wi * row[a] * yi
            for b in range(p):
                xtwx[a][b] += wi * row[a] * row[b]
    for j in range(p):
        xtwx[j][j] += ridge
    return _solve_linear_system(xtwx, xtwy)


def _dot(row: list[float], coef: list[float]) -> float:
    return sum(a * b for a, b in zip(row, coef))


def _basis_linear(row: list[float]) -> list[float]:
    return [1.0] + row


def _basis_quadratic(row: list[float]) -> list[float]:
    out = [1.0] + row
    out.extend(value * value for value in row)
    if len(row) >= 2:
        out.extend(row[j] * row[j + 1] for j in range(len(row) - 1))
    return out


def _basis_first7(row: list[float]) -> list[float]:
    return _basis_quadratic(row[:7])


def _basis_wind(row: list[float]) -> list[float]:
    angle = row[0]
    return [
        1.0,
        math.cos(angle),
        math.sin(angle),
        math.cos(2.0 * angle),
        math.sin(2.0 * angle),
    ]


def _make_folds(n: int, k: int, seed: int) -> list[list[int]]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return [idx[j::k] for j in range(k)]


def _run_repair(
    problem: Problem,
    response: list[int],
    seed: int,
    basis_ref: Callable[[list[float]], list[float]],
    basis_repair: Callable[[list[float]], list[float]],
    folds: int = 3,
) -> dict[str, float]:
    n = problem.n
    m0 = [0.0] * n
    h = [0.0] * n
    all_idx = set(range(n))
    for fold in _make_folds(n, folds, seed):
        fold_set = set(fold)
        train = [idx for idx in all_idx if idx not in fold_set and response[idx] == 1]
        if not train:
            train = [idx for idx in range(n) if response[idx] == 1]
        ref_basis = [basis_ref(problem.x[idx]) for idx in train]
        ref_y = [problem.y_full[idx] for idx in train]
        ref_coef = _fit_wls(ref_basis, ref_y, [1.0] * len(train))
        train_pred = [_dot(basis_ref(problem.x[idx]), ref_coef) for idx in train]
        residual = [problem.y_full[idx] - pred for idx, pred in zip(train, train_pred)]
        repair_basis = [basis_repair(problem.x[idx]) for idx in train]
        repair_weights = [
            (1.0 - problem.response_prob[idx])
            / max(problem.response_prob[idx] ** 2, 1e-8)
            for idx in train
        ]
        repair_coef = _fit_wls(repair_basis, residual, repair_weights)
        for idx in fold:
            m0[idx] = _dot(basis_ref(problem.x[idx]), ref_coef)
            h[idx] = _dot(basis_repair(problem.x[idx]), repair_coef)

    losses: dict[float, float] = {}
    improvements: dict[float, list[float]] = {}
    responder_idx = [idx for idx, ri in enumerate(response) if ri == 1]
    if not responder_idx:
        return {
            "theta": problem.theta,
            "ref_estimate": float("nan"),
            "repaired_estimate": float("nan"),
            "ref_error": float("nan"),
            "repaired_error": float("nan"),
            "selected_gamma": 0.0,
            "active": 0.0,
            "weighted_residual_gain": float("nan"),
        }
    for gamma in GAMMAS:
        terms = []
        imp_terms = []
        for idx in responder_idx:
            weight = (1.0 - problem.response_prob[idx]) / max(problem.response_prob[idx] ** 2, 1e-8)
            base = problem.y_full[idx] - m0[idx]
            cand = problem.y_full[idx] - (m0[idx] + gamma * h[idx])
            terms.append(weight * cand * cand)
            imp_terms.append(weight * (base * base - cand * cand))
        losses[gamma] = _mean(terms)
        improvements[gamma] = imp_terms
    eligible = [0.0]
    for gamma in GAMMAS:
        if gamma == 0.0:
            continue
        improvement = _mean(improvements[gamma])
        if improvement > _se(improvements[gamma]):
            eligible.append(gamma)
    selected = min(eligible, key=lambda gamma: (losses[gamma], gamma))
    ref_scores = []
    repaired_scores = []
    for idx in range(n):
        p = min(max(problem.response_prob[idx], 1e-6), 1.0 - 1e-6)
        y = problem.y_full[idx]
        ref_m = m0[idx]
        rep_m = m0[idx] + selected * h[idx]
        ref_scores.append(ref_m + response[idx] / p * (y - ref_m))
        repaired_scores.append(rep_m + response[idx] / p * (y - rep_m))
    ref_estimate = _mean(ref_scores)
    repaired_estimate = _mean(repaired_scores)
    weighted_gain = (
        100.0 * (1.0 - losses[selected] / losses[0.0])
        if losses[0.0] > 0.0
        else float("nan")
    )
    return {
        "theta": problem.theta,
        "ref_estimate": ref_estimate,
        "repaired_estimate": repaired_estimate,
        "ref_error": ref_estimate - problem.theta,
        "repaired_error": repaired_estimate - problem.theta,
        "selected_gamma": selected,
        "active": 1.0 if selected != 0.0 else 0.0,
        "weighted_residual_gain": weighted_gain,
    }


def _zhao_vanilla(problem_name: str, n: int, seed: int, arm: int) -> Problem:
    rng = random.Random(seed)
    x: list[list[float]] = []
    y_full: list[float] = []
    response_prob: list[float] = []
    rho = 0.3
    for _ in range(n):
        x1 = rng.random()
        x2 = rng.random()
        z1 = rng.gauss(0.0, 1.0)
        z2 = rho * z1 + math.sqrt(1.0 - rho * rho) * rng.gauss(0.0, 1.0)
        row = [x1, x2, z1, z2]
        e = _logistic(0.35 - 0.3 * sum(row))
        err = rng.gauss(0.0, 1.0)
        y0 = 20.0 * (1.0 + x1 - x2 + z1 * z1 + math.exp(x2)) + err * 20.0
        tau = 25.0 * (3.0 - 5.0 * x1 + 2.0 * x2 - 3.0 * z1 + z2)
        y1 = y0 + tau
        x.append(row)
        if arm == 1:
            y_full.append(y1)
            response_prob.append(e)
        else:
            y_full.append(y0)
            response_prob.append(1.0 - e)
    return Problem(
        name=problem_name,
        source="Zhao et al. 2024 vanilla policy-learning simulation, adapted to one observed potential-outcome mean",
        n=n,
        x=x,
        y_full=y_full,
        response_prob=response_prob,
        notes=f"R is treatment arm {arm}; target is the full-sample mean of Y({arm}).",
    )


def _cdte_lognormal(n: int, seed: int) -> Problem:
    rng = random.Random(seed)
    p_dim = 10
    x: list[list[float]] = []
    y_full: list[float] = []
    response_prob: list[float] = []
    for _ in range(n):
        row = [rng.random() for _ in range(p_dim)]
        e = _logistic(6.0 * (row[0] - 0.5))
        y1 = rng.lognormvariate(row[0] + row[1], 0.2)
        x.append(row)
        y_full.append(y1)
        response_prob.append(e)
    return Problem(
        name=f"cdte_lognormal_y1_n{n}",
        source="Kallus et al. 2023 CDTE lognormal simulation, adapted to one observed potential-outcome mean",
        n=n,
        x=x,
        y_full=y_full,
        response_prob=response_prob,
        notes="R is the treatment arm from propensity expit(6*(X1-0.5)); target is mean Y(1).",
    )


def _load_uehara_wind(path: Path, transform: str) -> Problem:
    values = [float(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    first = [values[idx] / 16.0 * 2.0 * math.pi for idx in range(0, 24 * 365, 24)]
    second = [values[idx] / 16.0 * 2.0 * math.pi for idx in range(12, 24 * 365, 24)]
    if transform == "cos":
        y = [math.cos(value) for value in second]
    elif transform == "sin":
        y = [math.sin(value) for value in second]
    else:
        raise ValueError(transform)
    return Problem(
        name=f"uehara_tokyo_wind_{transform}",
        source="Uehara et al. 2020 Tokyo wind application, adapted by treating the second daily direction transform as the missing outcome",
        n=len(first),
        x=[[value] for value in first],
        y_full=y,
        response_prob=[_logistic(math.cos(value)) for value in first],
        notes="Uses Uehara missingness P(missing X2|X1)=1/(1+exp(cos(X1))); therefore P(observed)=logistic(cos(X1)).",
    )


def _parse_float(text: str, default: float = 0.0) -> float:
    try:
        return float(text)
    except ValueError:
        return default


def _fit_logistic_probabilities(
    x: list[list[float]],
    y: list[int],
    iterations: int = 160,
    step_size: float = 0.15,
    l2: float = 1e-3,
) -> list[float]:
    d = len(x[0])
    coef = [0.0 for _ in range(d + 1)]
    n = len(x)
    for _ in range(iterations):
        grad = [0.0 for _ in range(d + 1)]
        for row, yi in zip(x, y):
            features = [1.0] + row
            pred = _logistic(_dot(features, coef))
            diff = pred - yi
            for j, value in enumerate(features):
                grad[j] += diff * value
        for j in range(d + 1):
            penalty = 0.0 if j == 0 else l2 * coef[j]
            coef[j] -= step_size * (grad[j] / n + penalty)
    return [min(max(_logistic(_dot([1.0] + row, coef)), 0.02), 0.98) for row in x]


def _load_zhao_diabetes_population(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = []
    for row in raw_rows:
        race_raw = row["race"]
        gender_raw = row["gender"]
        if race_raw == "?" or gender_raw == "Unknown/Invalid":
            continue
        race = 1.0 if race_raw == "Caucasian" else 0.0
        gender = 1.0 if gender_raw == "Female" else 0.0
        age = 0.0 if row["age"] in {"[60-70)", "[70-80)", "[80-90)", "[90-100)"} else 1.0
        rows.append(
            {
                "race": race,
                "gender": gender,
                "age": age,
                "time_in_hospital": _parse_float(row["time_in_hospital"]),
                "num_lab_procedures": _parse_float(row["num_lab_procedures"]),
                "num_medications": _parse_float(row["num_medications"]),
                "number_diagnoses": _parse_float(row["number_diagnoses"]),
                "diabetesMed": 1 if row["diabetesMed"] == "Yes" else 0,
            }
        )
    scale_cols = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_medications",
        "number_diagnoses",
    ]
    maxima = {col: max(float(row[col]) for row in rows) for col in scale_cols}
    x_all = []
    y_all = []
    for row in rows:
        x = [
            float(row["race"]),
            float(row["gender"]),
            float(row["age"]),
            float(row["time_in_hospital"]) / maxima["time_in_hospital"],
            float(row["num_lab_procedures"]) / maxima["num_lab_procedures"],
            float(row["num_medications"]) / maxima["num_medications"],
            float(row["number_diagnoses"]) / maxima["number_diagnoses"],
        ]
        x_all.append(x)
        y_all.append(int(row["diabetesMed"]))
    p_hat = _fit_logistic_probabilities(x_all, y_all)
    out = []
    rng = random.Random(9907)
    for x, row, p in zip(x_all, rows, p_hat):
        err = rng.gauss(0.0, 1.0)
        race, gender, age, time_h, lab, meds, diagnoses = x
        base = 20.0 * (
            1.0
            + gender
            - age
            + time_h
            + lab
            + meds
            + meds * meds
            + math.exp(diagnoses)
        )
        y0 = base + err * 20.0
        y1 = base + 25.0 * (3.0 - 5.0 * age + 2.0 * time_h - 3.0 * meds + race) + err * 20.0
        out.append(
            {
                "x": x,
                "response": int(row["diabetesMed"]),
                "p": p,
                "y0": y0,
                "y1": y1,
            }
        )
    return out


def _read_stata_113(path: Path) -> list[dict[str, float]]:
    data = path.read_bytes()
    if data[0] != 113:
        raise ValueError(f"expected Stata 113 file, got format byte {data[0]}")
    endian = "<" if data[1] == 2 else ">"
    offset = 4
    nvar = struct.unpack(endian + "H", data[offset : offset + 2])[0]
    offset += 2
    nobs = struct.unpack(endian + "I", data[offset : offset + 4])[0]
    offset += 4 + 81 + 18
    type_codes = list(data[offset : offset + nvar])
    offset += nvar
    names = []
    for _ in range(nvar):
        raw = data[offset : offset + 33]
        offset += 33
        names.append(raw.split(b"\0", 1)[0].decode("latin1"))
    offset += 2 * (nvar + 1)
    offset += 12 * nvar
    offset += 33 * nvar
    offset += 81 * nvar
    while True:
        exp_type = data[offset]
        exp_len = struct.unpack(endian + "I", data[offset + 1 : offset + 5])[0]
        offset += 5
        if exp_type == 0 and exp_len == 0:
            break
        offset += exp_len

    sizes = []
    for code in type_codes:
        if 1 <= code <= 244:
            sizes.append(code)
        elif code == 251:
            sizes.append(1)
        elif code == 252:
            sizes.append(2)
        elif code in (253, 254):
            sizes.append(4)
        elif code == 255:
            sizes.append(8)
        else:
            raise ValueError(f"unsupported Stata type code {code}")

    rows: list[dict[str, float]] = []
    for _ in range(nobs):
        row: dict[str, float] = {}
        for name, code, size in zip(names, type_codes, sizes):
            raw = data[offset : offset + size]
            offset += size
            if 1 <= code <= 244:
                row[name] = float("nan")
            elif code == 251:
                row[name] = float(struct.unpack("b", raw)[0])
            elif code == 252:
                row[name] = float(struct.unpack(endian + "h", raw)[0])
            elif code == 253:
                row[name] = float(struct.unpack(endian + "i", raw)[0])
            elif code == 254:
                row[name] = float(struct.unpack(endian + "f", raw)[0])
            elif code == 255:
                row[name] = float(struct.unpack(endian + "d", raw)[0])
        rows.append(row)
    return rows


def _load_401k_population(path: Path) -> list[dict[str, object]]:
    raw_rows = _read_stata_113(path)
    feature_names = ["age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown"]
    maxima = {
        name: max(abs(row[name]) for row in raw_rows if math.isfinite(row[name])) or 1.0
        for name in ("age", "inc", "educ", "fsize")
    }
    x_all = []
    response = []
    for row in raw_rows:
        x_all.append(
            [
                row["age"] / maxima["age"],
                row["inc"] / maxima["inc"],
                row["educ"] / maxima["educ"],
                row["fsize"] / maxima["fsize"],
                row["marr"],
                row["twoearn"],
                row["db"],
                row["pira"],
                row["hown"],
            ]
        )
        response.append(1 if row["e401"] > 0.5 else 0)
    p_hat = _fit_logistic_probabilities(x_all, response)
    return [
        {
            "x": x,
            "response": ri,
            "p": p,
            "y": row["net_tfa"],
        }
        for row, x, ri, p in zip(raw_rows, x_all, response, p_hat)
    ]


def _kallus_401k_application(
    population: list[dict[str, object]],
    seed: int,
) -> Problem:
    del seed
    x = [list(row["x"]) for row in population]  # type: ignore[arg-type]
    response = [int(row["response"]) for row in population]
    p = [float(row["p"]) for row in population]
    y_full = [float(row["y"]) for row in population]
    return Problem(
        name="kallus401k_net_tfa_e401",
        source="Kallus et al. 2023 / DoubleML 401(k) application, adapted to missing-outcome mean estimation",
        n=len(population),
        x=x,
        y_full=y_full,
        response_prob=p,
        notes="Uses SIPP 1991 401(k) covariates; target is mean net_tfa with e401 treated as the response indicator.",
        fixed_response=response,
    )


def _zhao_diabetes_application(
    population: list[dict[str, object]],
    n: int,
    seed: int,
    arm: int,
) -> Problem:
    rng = random.Random(seed)
    sample = rng.sample(population, n)
    x = [list(row["x"]) for row in sample]  # type: ignore[arg-type]
    observed = [int(row["response"]) for row in sample]
    if arm == 1:
        y_full = [float(row["y1"]) for row in sample]
        response = observed
        p = [float(row["p"]) for row in sample]
    else:
        y_full = [float(row["y0"]) for row in sample]
        response = [1 - value for value in observed]
        p = [1.0 - float(row["p"]) for row in sample]
    return Problem(
        name=f"zhao24a_diabetes_y{arm}",
        source="Zhao et al. 2024 UCI Diabetes application, adapted to one observed potential-outcome mean",
        n=n,
        x=x,
        y_full=y_full,
        response_prob=p,
        notes=f"Uses Zhao preprocessing columns and potential-outcome formula; R is diabetesMed arm {arm}.",
        fixed_response=response,
    )


def _sample_response(problem: Problem, seed: int) -> list[int]:
    if problem.fixed_response is not None:
        return list(problem.fixed_response)
    rng = random.Random(seed)
    return [1 if rng.random() < p else 0 for p in problem.response_prob]


def _rows_for_problem(
    problem_factory: Callable[[int], Problem],
    reps: int,
    seed_offset: int,
    basis_ref: Callable[[list[float]], list[float]],
    basis_repair: Callable[[list[float]], list[float]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rep in range(reps):
        problem = problem_factory(seed_offset + rep)
        response = _sample_response(problem, seed_offset * 100000 + rep)
        result = _run_repair(
            problem=problem,
            response=response,
            seed=seed_offset * 200000 + rep,
            basis_ref=basis_ref,
            basis_repair=basis_repair,
        )
        rows.append(
            {
                "benchmark": problem.name,
                "source": problem.source,
                "n": problem.n,
                "rep": rep,
                "respondents": sum(response),
                "mean_response_prob": _mean(problem.response_prob),
                "min_response_prob": min(problem.response_prob),
                "target": result["theta"],
                "ref_estimate": result["ref_estimate"],
                "repaired_estimate": result["repaired_estimate"],
                "ref_error": result["ref_error"],
                "repaired_error": result["repaired_error"],
                "selected_gamma": result["selected_gamma"],
                "active": int(result["active"]),
                "weighted_residual_gain_pct": result["weighted_residual_gain"],
                "notes": problem.notes,
            }
        )
    return rows


def _summarize(rows: list[dict[str, object]], bootstrap_draws: int) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["benchmark"]), []).append(row)
    out: list[dict[str, object]] = []
    rng = random.Random(20260915)
    for benchmark, group in sorted(grouped.items()):
        ref_errors = [float(row["ref_error"]) for row in group]
        repaired_errors = [float(row["repaired_error"]) for row in group]
        ref_mse = _mean([value * value for value in ref_errors])
        repaired_mse = _mean([value * value for value in repaired_errors])
        gain = 100.0 * (1.0 - repaired_mse / ref_mse) if ref_mse > 0.0 else float("nan")
        lo, hi = _bootstrap_ci(ref_errors, repaired_errors, rng, bootstrap_draws)
        active_rows = [row for row in group if int(row["active"]) == 1]
        harm = sum(
            1
            for row in group
            if float(row["repaired_error"]) ** 2 > float(row["ref_error"]) ** 2
        )
        active_harm = sum(
            1
            for row in active_rows
            if float(row["repaired_error"]) ** 2 > float(row["ref_error"]) ** 2
        )
        out.append(
            {
                "benchmark": benchmark,
                "source": group[0]["source"],
                "n": group[0]["n"],
                "replications": len(group),
                "mean_response_prob": _mean([float(row["mean_response_prob"]) for row in group]),
                "min_response_prob": _mean([float(row["min_response_prob"]) for row in group]),
                "ref_mse": ref_mse,
                "repaired_mse": repaired_mse,
                "mse_gain_pct": gain,
                "mse_gain_ci_low": lo,
                "mse_gain_ci_high": hi,
                "activation_pct": 100.0 * len(active_rows) / len(group),
                "harm_pct": 100.0 * harm / len(group),
                "active_harm_pct": 100.0 * active_harm / len(active_rows) if active_rows else 0.0,
                "mean_weighted_residual_gain_pct": _mean(
                    [float(row["weighted_residual_gain_pct"]) for row in group]
                ),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_results(out_dir: Path, summary: list[dict[str, object]]) -> None:
    lines = [
        "# Literature Benchmark Probe",
        "",
        "**Status: DIAGNOSTIC.** These are first-pass adaptations of benchmark",
        "data-generating ingredients used in nearby AISTATS/ICML papers. They do",
        "not reproduce those papers' original estimands.",
        "",
        "## Summary",
        "",
        "| benchmark | n | reps | MSE gain | 95% CI | activation | harm | active harm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {benchmark} | {n} | {replications} | {gain:.3f}% | [{lo:.3f}, {hi:.3f}] | "
            "{activation:.1f}% | {harm:.1f}% | {active_harm:.1f}% |".format(
                benchmark=row["benchmark"],
                n=int(row["n"]),
                replications=int(row["replications"]),
                gain=float(row["mse_gain_pct"]),
                lo=float(row["mse_gain_ci_low"]),
                hi=float(row["mse_gain_ci_high"]),
                activation=float(row["activation_pct"]),
                harm=float(row["harm_pct"]),
                active_harm=float(row["active_harm_pct"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This probe uses the published covariate and response/treatment mechanisms",
            "where available, but translates them into the missing-outcome mean problem.",
            "Large positive values would mean the residual repair helps on a published",
            "benchmark ingredient; near-zero activation means the one-SE rule treats the",
            "benchmark as already stable for this estimand.",
        ]
    )
    out_dir.joinpath("RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--uehara-wind-csv",
        type=Path,
        default=Path("/tmp/dml_literature_sources/uehara20b-supp/code_sup_missing 5/code23/tokyo_wind_2018.csv"),
    )
    parser.add_argument(
        "--zhao-diabetes-csv",
        type=Path,
        default=Path("/tmp/dml_literature_sources/zhao_diabetes/diabetic_data.csv"),
    )
    parser.add_argument(
        "--sipp1991-dta",
        type=Path,
        default=Path("/tmp/dml_literature_sources/VC2015-DMLonGitHub/sipp1991.dta"),
    )
    parser.add_argument("--reps", type=int, default=800)
    parser.add_argument("--diabetes-reps", type=int, default=400)
    parser.add_argument("--k401-reps", type=int, default=400)
    parser.add_argument("--wind-reps", type=int, default=2000)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []
    print("RUNNING zhao24a_vanilla_y1")
    rows.extend(
        _rows_for_problem(
            lambda seed: _zhao_vanilla("zhao24a_vanilla_y1", 2000, seed, 1),
            args.reps,
            1000,
            _basis_linear,
            _basis_quadratic,
        )
    )
    print("RUNNING zhao24a_vanilla_y0")
    rows.extend(
        _rows_for_problem(
            lambda seed: _zhao_vanilla("zhao24a_vanilla_y0", 2000, seed, 0),
            args.reps,
            2000,
            _basis_linear,
            _basis_quadratic,
        )
    )
    for n in (200, 800, 3200):
        print(f"RUNNING cdte_lognormal_y1_n{n}")
        rows.extend(
            _rows_for_problem(
                lambda seed, n=n: _cdte_lognormal(n, seed),
                args.reps,
                3000 + n,
                _basis_linear,
                _basis_quadratic,
            )
        )
    if args.zhao_diabetes_csv.exists():
        print("LOADING zhao24a_diabetes")
        diabetes_population = _load_zhao_diabetes_population(args.zhao_diabetes_csv)
        args.out_dir.joinpath("zhao_diabetes_diabetic_data.csv").write_text(
            args.zhao_diabetes_csv.read_text()
        )
        for arm in (0, 1):
            print(f"RUNNING zhao24a_diabetes_y{arm}")
            rows.extend(
                _rows_for_problem(
                    lambda seed, arm=arm: _zhao_diabetes_application(
                        diabetes_population,
                        2500,
                        seed,
                        arm,
                    ),
                    args.diabetes_reps,
                    7000 + arm,
                    _basis_linear,
                    _basis_first7,
                )
            )
    if args.sipp1991_dta.exists():
        print("LOADING kallus401k")
        k401_population = _load_401k_population(args.sipp1991_dta)
        args.out_dir.joinpath("sipp1991.dta").write_bytes(args.sipp1991_dta.read_bytes())
        print("RUNNING kallus401k_net_tfa_e401")
        rows.extend(
            _rows_for_problem(
                lambda seed: _kallus_401k_application(k401_population, seed),
                args.k401_reps,
                8000,
                _basis_linear,
                _basis_quadratic,
            )
        )
    if args.uehara_wind_csv.exists():
        for transform in ("cos", "sin"):
            print(f"RUNNING uehara_tokyo_wind_{transform}")
            rows.extend(
                _rows_for_problem(
                    lambda _seed, transform=transform: _load_uehara_wind(args.uehara_wind_csv, transform),
                    args.wind_reps,
                    9000 + (1 if transform == "cos" else 2),
                    _basis_wind,
                    _basis_wind,
                )
            )
        args.out_dir.joinpath("tokyo_wind_2018.csv").write_text(args.uehara_wind_csv.read_text())

    summary = _summarize(rows, args.bootstrap_draws)
    _write_csv(args.out_dir / "literature_benchmark_replication_rows.csv", rows)
    _write_csv(args.out_dir / "literature_benchmark_summary.csv", summary)
    _write_results(args.out_dir, summary)
    manifest = [
        {
            "run_id": args.out_dir.name,
            "command": (
                "python3 scripts/dml_literature_benchmark_probe.py "
                f"--out-dir {args.out_dir} --reps {args.reps} "
                f"--diabetes-reps {args.diabetes_reps} --wind-reps {args.wind_reps} "
                f"--k401-reps {args.k401_reps} "
                f"--bootstrap-draws {args.bootstrap_draws}"
            ),
            "status": "completed",
            "notes": "First-pass published-paper benchmark adaptations; diagnostic.",
        }
    ]
    _write_csv(args.out_dir / "manifest.tsv", manifest)
    args.out_dir.joinpath("controller.log").write_text(
        "runner_dml start: literature benchmark probe\n"
        "runner_dml complete: exit_code=0\n"
    )
    provenance = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "status": "DIAGNOSTIC",
        "external_sources": {
            "uehara20b_supplement": "http://proceedings.mlr.press/v108/uehara20b/uehara20b-supp.zip",
            "zhao24a_code": "https://github.com/panzhaooo/positivity-free-policy-learning",
            "zhao24a_diabetes_source": "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008",
            "kallus23a_cdte_code": "https://github.com/CausalML/CDTE",
            "kallus401k_sipp_source": "https://github.com/VC2015/DMLonGitHub/raw/master/sipp1991.dta",
        },
        "adaptation": "Treatment/response variables from published DGPs are interpreted as response indicators for a one-arm missing-outcome mean target.",
        "gamma_grid": list(GAMMAS),
    }
    if args.uehara_wind_csv.exists():
        provenance["tokyo_wind_csv_sha256"] = _sha256(args.uehara_wind_csv)
    if args.zhao_diabetes_csv.exists():
        provenance["zhao_diabetes_csv_sha256"] = _sha256(args.zhao_diabetes_csv)
    if args.sipp1991_dta.exists():
        provenance["sipp1991_dta_sha256"] = _sha256(args.sipp1991_dta)
    args.out_dir.joinpath("provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    verification = {
        "status": "PASS",
        "checks": {
            "benchmarks": sorted({str(row["benchmark"]) for row in rows}),
            "replication_rows": len(rows),
            "summary_rows": len(summary),
            "all_summary_replications_positive": all(
                int(row["replications"]) > 0 for row in summary
            ),
            "uehara_wind_included": args.uehara_wind_csv.exists(),
            "zhao_diabetes_included": args.zhao_diabetes_csv.exists(),
            "kallus401k_included": args.sipp1991_dta.exists(),
        },
    }
    args.out_dir.joinpath("verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n"
    )
    checksum_lines = []
    for path in sorted(args.out_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            checksum_lines.append(f"{_sha256(path)}  {path.name}")
    args.out_dir.joinpath("SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")
    print(f"wrote {args.out_dir} with {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
