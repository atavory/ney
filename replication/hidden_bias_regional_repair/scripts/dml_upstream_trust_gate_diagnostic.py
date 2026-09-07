#!/usr/bin/env python3
"""Retrospective upstream-trust stand-down diagnostics for the repair path."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


METHODS = ("aipw", "tmle", "ctmle", "cui_selective_ml", "ma_dr_bc")
FLOOR_SENS_TAUS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50)
FLOOR_SENS_MAXS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, math.inf)
ESS_MINS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20)
FLOOR_SHARE_MAXS = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 1.0)
MAX_WEIGHT_SHARE_MAXS = (0.02, 0.05, 0.10, 0.20, 0.50, 1.0)
SCORE_TAIL_MAXS = (4.0, 6.0, 8.0, 10.0, 12.0, 16.0, math.inf)
GATE = None


@dataclass(frozen=True)
class Rule:
    gate_id: str
    floor_sens_max: float = math.inf
    residual_ess_min: float = 0.0
    floor_share_max: float = 1.0
    max_weight_share_max: float = 1.0
    score_tail_max: float = math.inf
    veto_floor_sens_max: float = -math.inf
    veto_floor_share_min: float = math.inf
    veto_residual_ess_max: float = math.inf
    veto_score_tail_min: float = -math.inf


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _init_worker(
    gate_script: str,
    rho_script: str,
    source_script: str,
    breadth_script: str,
    support_data: str,
) -> None:
    global GATE
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    gate = _load_module("dml_upstream_trust_gate_helper", Path(gate_script))
    gate.METHODS = METHODS
    gate._init_worker(rho_script, source_script, breadth_script, support_data)
    GATE = gate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric_key(value: float) -> str:
    return "inf" if not math.isfinite(value) else f"{value:g}"


def _rules() -> tuple[Rule, ...]:
    out: list[Rule] = []
    for value in FLOOR_SENS_MAXS:
        out.append(Rule(gate_id=f"floor_sens_le_{_metric_key(value)}", floor_sens_max=value))
    for value in ESS_MINS:
        out.append(Rule(gate_id=f"resid_ess_ge_{_metric_key(value)}", residual_ess_min=value))
    for value in FLOOR_SHARE_MAXS:
        out.append(Rule(gate_id=f"floor_share_le_{_metric_key(value)}", floor_share_max=value))
    for value in MAX_WEIGHT_SHARE_MAXS:
        out.append(
            Rule(
                gate_id=f"max_resid_weight_share_le_{_metric_key(value)}",
                max_weight_share_max=value,
            )
        )
    for value in SCORE_TAIL_MAXS:
        out.append(Rule(gate_id=f"score_tail_le_{_metric_key(value)}", score_tail_max=value))
    for floor_sens in FLOOR_SENS_MAXS:
        if not math.isfinite(floor_sens):
            continue
        for ess_min in (0.0, 0.01, 0.05, 0.10):
            out.append(
                Rule(
                    gate_id=(
                        f"floor_sens_le_{_metric_key(floor_sens)}"
                        f"_resid_ess_ge_{_metric_key(ess_min)}"
                    ),
                    floor_sens_max=floor_sens,
                    residual_ess_min=ess_min,
                )
            )
    for floor_sens in FLOOR_SENS_MAXS:
        if not math.isfinite(floor_sens):
            continue
        for floor_share in (0.02, 0.05, 0.10, 0.20, 1.0):
            out.append(
                Rule(
                    gate_id=(
                        f"floor_sens_le_{_metric_key(floor_sens)}"
                        f"_floor_share_le_{_metric_key(floor_share)}"
                    ),
                    floor_sens_max=floor_sens,
                    floor_share_max=floor_share,
                )
            )
    for floor_sens in (0.5, 1.0, 2.0, 5.0, math.inf):
        for tail in (6.0, 8.0, 10.0, 12.0, 16.0, math.inf):
            out.append(
                Rule(
                    gate_id=(
                        f"floor_sens_le_{_metric_key(floor_sens)}"
                        f"_score_tail_le_{_metric_key(tail)}"
                    ),
                    floor_sens_max=floor_sens,
                    score_tail_max=tail,
                )
            )
    for floor_sens in (0.25, 0.5, 0.75, 1.0, 1.25):
        for floor_share in (0.05, 0.10, 0.15, 0.20):
            out.append(
                Rule(
                    gate_id=(
                        f"veto_stable_floor_sens_le_{_metric_key(floor_sens)}"
                        f"_floor_share_ge_{_metric_key(floor_share)}"
                    ),
                    veto_floor_sens_max=floor_sens,
                    veto_floor_share_min=floor_share,
                )
            )
            for score_tail in (8.0, 10.0, 12.0, 14.0):
                out.append(
                    Rule(
                        gate_id=(
                            f"veto_stable_floor_sens_le_{_metric_key(floor_sens)}"
                            f"_floor_share_ge_{_metric_key(floor_share)}"
                            f"_score_tail_ge_{_metric_key(score_tail)}"
                        ),
                        veto_floor_sens_max=floor_sens,
                        veto_floor_share_min=floor_share,
                        veto_score_tail_min=score_tail,
                    )
                )
            for residual_ess in (0.05, 0.10, 0.20):
                out.append(
                    Rule(
                        gate_id=(
                            f"veto_stable_floor_sens_le_{_metric_key(floor_sens)}"
                            f"_floor_share_ge_{_metric_key(floor_share)}"
                            f"_resid_ess_le_{_metric_key(residual_ess)}"
                        ),
                        veto_floor_sens_max=floor_sens,
                        veto_floor_share_min=floor_share,
                        veto_residual_ess_max=residual_ess,
                    )
                )
    dedup = {}
    for rule in out:
        dedup[rule.gate_id] = rule
    return tuple(dedup.values())


def _ess_share(weights: np.ndarray, denominator: int) -> float:
    weights = np.asarray(weights, dtype=float)
    weights = weights[np.isfinite(weights) & (weights > 0.0)]
    if denominator <= 0:
        return 0.0
    if len(weights) == 0:
        return 0.0
    sum_w = float(np.sum(weights))
    sum_w2 = float(np.sum(weights * weights))
    if sum_w2 <= 0.0:
        return 0.0
    return (sum_w * sum_w / sum_w2) / float(denominator)


def _max_share(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    weights = weights[np.isfinite(weights) & (weights > 0.0)]
    if len(weights) == 0:
        return 0.0
    total = float(np.sum(weights))
    return float(np.max(weights) / total) if total > 0.0 else 0.0


def _tail_ratio(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    centered = values - float(np.mean(values))
    rms = float(np.sqrt(np.mean(centered * centered)))
    if rms <= 0.0:
        return 0.0
    return float(np.max(np.abs(centered)) / rms)


def _targeted_outcome(vrt: Any, y, response, p, m) -> np.ndarray:
    observed = response.astype(bool)
    clever = 1.0 / np.maximum(p[observed], 1e-12)
    denominator = float(np.dot(clever, clever))
    epsilon = (
        float(np.dot(clever, y[observed] - m[observed]) / denominator)
        if denominator > 0.0
        else 0.0
    )
    return m + epsilon / np.maximum(p, 1e-12)


def _ma_base_score(vrt: Any, y, response, p_raw, m, tau: float, seed: int, folds: int):
    endpoint, _ = vrt._ma_dr_bc_reference(
        y,
        response,
        p_raw,
        m,
        trim_h=tau,
        correction_order=1,
        sieve_degree=3,
    )
    score = vrt._crossfit_ma_dr_bc_score(
        y,
        response,
        p_raw,
        m,
        seed + 11003,
        folds,
        trim_h=tau,
        correction_order=1,
        sieve_degree=3,
    )
    return endpoint, score


def _floor_sensitivity(
    row: Any,
    x: np.ndarray,
    y: np.ndarray,
    response: np.ndarray,
    true_pi: np.ndarray,
    seed: int,
    base_score: np.ndarray,
    selected_tau: float,
) -> float:
    if GATE is None or GATE.RHO is None:
        raise RuntimeError("worker not initialized")
    if row.method == "cui_selective_ml":
        return 0.0
    vrt = GATE.RHO._VRT
    m, p_raw = GATE.RHO._initial_outcome_and_p_raw(
        vrt,
        x,
        y,
        response,
        true_pi,
        row,
        seed,
    )
    p_raw = np.clip(p_raw, 1e-12, 1.0)
    endpoints = {}
    for tau in FLOOR_SENS_TAUS:
        tau = float(tau)
        if row.method == "ma_dr_bc":
            endpoint, _ = vrt._ma_dr_bc_reference(
                y,
                response,
                p_raw,
                m,
                trim_h=tau,
                correction_order=1,
                sieve_degree=3,
            )
            endpoints[tau] = float(np.mean(endpoint))
            continue
        p = np.maximum(p_raw, tau)
        if row.method in {"tmle", "ctmle"}:
            endpoint = float(np.mean(_targeted_outcome(vrt, y, response, p, m)))
        else:
            endpoint = float(np.mean(vrt._aipw_score(y, response, p, m)))
        endpoints[tau] = endpoint
    selected = min(endpoints, key=lambda tau: abs(tau - selected_tau))
    reference = endpoints[selected]
    se = math.sqrt(float(np.var(np.asarray(base_score, dtype=float), ddof=1)) / len(base_score))
    if se <= 0.0 or not math.isfinite(se):
        return math.inf
    return max(abs(value - reference) for value in endpoints.values()) / se


def _upstream_metrics(row: Any, x, y, response, true_pi, theta, seed: int) -> dict[str, float]:
    if GATE is None or GATE.RHO is None:
        raise RuntimeError("worker not initialized")
    vrt = GATE.RHO._VRT
    base_outcome, p, correction_tau = GATE.RHO._selected_base_and_p(
        vrt,
        x,
        y,
        response,
        true_pi,
        row,
        seed,
    )
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0)
    if row.method == "ma_dr_bc":
        m_oof, p_raw = GATE.RHO._initial_outcome_and_p_raw(
            vrt,
            x,
            y,
            response,
            true_pi,
            row,
            seed,
        )
        _endpoint, base_score = _ma_base_score(
            vrt,
            y,
            response,
            np.clip(p_raw, 1e-12, 1.0),
            m_oof,
            correction_tau,
            seed,
            row.folds,
        )
    else:
        base_score = vrt._aipw_score(y, response, p, base_outcome)
    observed = response.astype(bool)
    residual_weights = np.zeros(len(response), dtype=float)
    residual_weights[observed] = (1.0 - p[observed]) / (p[observed] * p[observed])
    ipw_weights = np.zeros(len(response), dtype=float)
    ipw_weights[observed] = 1.0 / p[observed]
    floor_tol = max(1e-8, 1e-8 * abs(correction_tau))
    floor_sens = _floor_sensitivity(
        row,
        x,
        y,
        response,
        true_pi,
        seed,
        np.asarray(base_score, dtype=float),
        correction_tau,
    )
    return {
        "selected_tau": correction_tau,
        "floor_share": float(np.mean(p <= correction_tau + floor_tol)),
        "observed_floor_share": (
            float(np.mean(p[observed] <= correction_tau + floor_tol))
            if np.any(observed)
            else 0.0
        ),
        "residual_weight_ess_share": _ess_share(residual_weights, len(response)),
        "ipw_weight_ess_share": _ess_share(ipw_weights, len(response)),
        "max_residual_weight_share": _max_share(residual_weights),
        "max_ipw_weight_share": _max_share(ipw_weights),
        "score_tail_ratio": _tail_ratio(np.asarray(base_score, dtype=float)),
        "floor_sensitivity_se": floor_sens,
    }


def _passes(rule: Rule, metrics: dict[str, float]) -> bool:
    threshold_pass = (
        metrics["floor_sensitivity_se"] <= rule.floor_sens_max
        and metrics["residual_weight_ess_share"] >= rule.residual_ess_min
        and metrics["floor_share"] <= rule.floor_share_max
        and metrics["max_residual_weight_share"] <= rule.max_weight_share_max
        and metrics["score_tail_ratio"] <= rule.score_tail_max
    )
    vetoed = (
        metrics["floor_sensitivity_se"] <= rule.veto_floor_sens_max
        and metrics["floor_share"] >= rule.veto_floor_share_min
        and metrics["residual_weight_ess_share"] <= rule.veto_residual_ess_max
        and metrics["score_tail_ratio"] >= rule.veto_score_tail_min
    )
    return threshold_pass and not vetoed


def _process_sample(rows: list[Any], rules: tuple[Rule, ...]) -> list[dict[str, Any]]:
    if GATE is None or GATE.RHO is None:
        raise RuntimeError("worker not initialized")
    vrt = GATE.RHO._VRT
    first = rows[0]
    seed = GATE.RHO._internal_seed(vrt, first)
    x, y, response, _region, true_pi, theta, _mu = vrt.make_data(
        first.n,
        first.epsilon,
        first.strength,
        first.design,
        seed,
        first.mar_design,
    )
    out = []
    for row in rows:
        metrics = _upstream_metrics(row, x, y, response, true_pi, theta, seed)
        for rule in rules:
            passed = _passes(rule, metrics)
            out.append(
                {
                    "gate_id": rule.gate_id,
                    "floor_sens_max": rule.floor_sens_max,
                    "residual_ess_min": rule.residual_ess_min,
                    "floor_share_max": rule.floor_share_max,
                    "max_weight_share_max": rule.max_weight_share_max,
                    "score_tail_max": rule.score_tail_max,
                    "veto_floor_sens_max": rule.veto_floor_sens_max,
                    "veto_floor_share_min": rule.veto_floor_share_min,
                    "veto_residual_ess_max": rule.veto_residual_ess_max,
                    "veto_score_tail_min": rule.veto_score_tail_min,
                    "source": row.source,
                    "dataset": GATE.RHO._dataset(row.design),
                    "setting": GATE.RHO._setting_label(row.design, row.n, row.strength),
                    "group": row.group,
                    "design": row.design,
                    "method": row.method,
                    "n": row.n,
                    "strength": row.strength,
                    "seed0": row.seed0,
                    "rep": row.rep,
                    "ref_error": row.ref_error,
                    "current_error": row.selected_error,
                    "current_replay_error": row.selected_error,
                    "gated_error": row.selected_error if passed else row.ref_error,
                    "current_gamma": row.selected_gamma,
                    "gated_gamma": row.selected_gamma if passed else 0.0,
                    "gamma_changed": (not passed and float(row.selected_gamma) != 0.0),
                    "current_passed_gate": passed,
                    "allowed_count": len(row.region_damp_grid) if passed else 1,
                    "current_max_ratio": metrics["score_tail_ratio"],
                    "current_ess_share": metrics["residual_weight_ess_share"],
                    **metrics,
                }
            )
    return out


def _objective_summary(method_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_gate: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in method_rows:
        by_gate[row["gate_id"]][row["method"]] = row
    out = []
    for gate_id, rows in sorted(by_gate.items()):
        if not all(method in rows for method in METHODS):
            continue
        aipw = rows["aipw"]
        selective = rows["cui_selective_ml"]
        ctmle = rows["ctmle"]
        tmle = rows["tmle"]
        ma = rows["ma_dr_bc"]
        row = {
            "gate_id": gate_id,
            "floor_sens_max": tmle["floor_sens_max"],
            "residual_ess_min": tmle["residual_ess_min"],
            "floor_share_max": tmle["floor_share_max"],
            "max_weight_share_max": tmle["max_weight_share_max"],
            "score_tail_max": tmle["score_tail_max"],
            "veto_floor_sens_max": tmle["veto_floor_sens_max"],
            "veto_floor_share_min": tmle["veto_floor_share_min"],
            "veto_residual_ess_max": tmle["veto_residual_ess_max"],
            "veto_score_tail_min": tmle["veto_score_tail_min"],
            "aipw_current_gain": aipw["current_gain"],
            "aipw_gated_gain": aipw["gated_gain"],
            "aipw_retention": (
                float(aipw["gated_gain"]) / float(aipw["current_gain"])
                if float(aipw["current_gain"]) > 0.0
                else float("nan")
            ),
            "selective_current_gain": selective["current_gain"],
            "selective_gated_gain": selective["gated_gain"],
            "selective_retention": (
                float(selective["gated_gain"]) / float(selective["current_gain"])
                if float(selective["current_gain"]) > 0.0
                else float("nan")
            ),
            "ctmle_gated_gain": ctmle["gated_gain"],
            "ma_current_gain": ma["current_gain"],
            "ma_gated_gain": ma["gated_gain"],
            "ma_retention": (
                float(ma["gated_gain"]) / float(ma["current_gain"])
                if float(ma["current_gain"]) > 0.0
                else float("nan")
            ),
            "tmle_current_gain": tmle["current_gain"],
            "tmle_gated_gain": tmle["gated_gain"],
            "tmle_gated_ci_low": tmle["gated_gain_ci_low"],
            "tmle_gated_ci_high": tmle["gated_gain_ci_high"],
            "tmle_recovery": float(tmle["gated_gain"]) - float(tmle["current_gain"]),
            "tmle_activation": tmle["gated_activation"],
            "aipw_activation": aipw["gated_activation"],
            "selective_activation": selective["gated_activation"],
            "ma_activation": ma["gated_activation"],
        }
        row["passes_basic_goal"] = (
            float(row["aipw_retention"]) >= 0.80
            and float(row["selective_retention"]) >= 0.80
            and float(row["ma_retention"]) >= 0.80
            and float(row["ctmle_gated_gain"]) >= -0.005
            and float(row["tmle_gated_ci_low"]) > -0.05
        )
        row["objective_score"] = (
            4.0 * float(row["tmle_recovery"])
            + float(row["aipw_retention"])
            + float(row["selective_retention"])
            + float(row["ma_retention"])
            + max(-2.0, min(2.0, 20.0 * float(row["tmle_gated_gain"])))
            + max(-1.0, min(1.0, 10.0 * float(row["ctmle_gated_gain"])))
        )
        out.append(row)
    out.sort(key=lambda row: (not row["passes_basic_goal"], -float(row["objective_score"])))
    return out


def _pct(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def _write_results(
    path: Path,
    objective_rows: list[dict[str, Any]],
    method_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
) -> None:
    by_method_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        by_method_metric[row["method"]].append(row)
    method_by_gate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in method_rows:
        method_by_gate[row["gate_id"]].append(row)
    lines = [
        "# Upstream Trust Gate Diagnostic",
        "",
        "This retrospective diagnostic forces the repair to stand down when an",
        "estimator-agnostic upstream diagnostic fails.  It does not inspect the",
        "method name and it does not refit the repair path.",
        "",
        "## Upstream Metrics",
        "",
    ]
    for method in sorted(by_method_metric):
        rows = by_method_metric[method]
        values = {
            key: float(np.mean([float(row[key]) for row in rows]))
            for key in (
                "floor_sensitivity_se",
                "residual_weight_ess_share",
                "floor_share",
                "max_residual_weight_share",
                "score_tail_ratio",
            )
        }
        lines.append(
            "- {method}: floor sensitivity {floor_sens:.3f} SE, residual ESS "
            "{ess:.3f}, floor share {floor_share:.3f}, max residual-weight share "
            "{max_share:.3f}, score tail {tail:.3f}".format(
                method=method,
                floor_sens=values["floor_sensitivity_se"],
                ess=values["residual_weight_ess_share"],
                floor_share=values["floor_share"],
                max_share=values["max_residual_weight_share"],
                tail=values["score_tail_ratio"],
            )
        )
    lines.extend(["", "## Best Rules", ""])
    for row in objective_rows[:15]:
        lines.append(
            "- {gate_id}: AIPW {aipw}, selective {selective}, C-TMLE {ctmle}, "
            "fixed-floor TMLE {tmle} [{tmle_lo}, {tmle_hi}], Ma {ma}, "
            "AIPW/selective/Ma retention {aipw_ret:.3f}/{sel_ret:.3f}/{ma_ret:.3f}, "
            "TMLE activation {tmle_act:.1f}%, pass={passed}".format(
                gate_id=row["gate_id"],
                aipw=_pct(float(row["aipw_gated_gain"])),
                selective=_pct(float(row["selective_gated_gain"])),
                ctmle=_pct(float(row["ctmle_gated_gain"])),
                tmle=_pct(float(row["tmle_gated_gain"])),
                tmle_lo=_pct(float(row["tmle_gated_ci_low"])),
                tmle_hi=_pct(float(row["tmle_gated_ci_high"])),
                ma=_pct(float(row["ma_gated_gain"])),
                aipw_ret=float(row["aipw_retention"]),
                sel_ret=float(row["selective_retention"]),
                ma_ret=float(row["ma_retention"]),
                tmle_act=100.0 * float(row["tmle_activation"]),
                passed=row["passes_basic_goal"],
            )
        )
    if objective_rows:
        best = objective_rows[0]
        lines.extend(["", f"## Method Readout for {best['gate_id']}", ""])
        for row in sorted(method_by_gate[best["gate_id"]], key=lambda item: item["method"]):
            lines.append(
                "- {method}: current {current}, gated {gated} [{lo}, {hi}], "
                "activation {activation:.1f}% vs {current_activation:.1f}%, "
                "harm {harm:.1f}% vs {current_harm:.1f}%".format(
                    method=row["method"],
                    current=_pct(float(row["current_gain"])),
                    gated=_pct(float(row["gated_gain"])),
                    lo=_pct(float(row["gated_gain_ci_low"])),
                    hi=_pct(float(row["gated_gain_ci_high"])),
                    activation=100.0 * float(row["gated_activation"]),
                    current_activation=100.0 * float(row["current_activation"]),
                    harm=100.0 * float(row["gated_harm_rate"]),
                    current_harm=100.0 * float(row["current_harm_rate"]),
                )
            )
    path.write_text("\n".join(lines) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], writer_module: Any) -> None:
    writer_module._write_csv(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--gate-script",
        type=Path,
        default=Path("scripts/dml_universal_leverage_gate_diagnostic.py"),
    )
    parser.add_argument(
        "--rho-script",
        type=Path,
        default=Path("scripts/dml_mse_gain_by_rho.py"),
    )
    parser.add_argument(
        "--source-script",
        type=Path,
        default=Path(
            "/tmp/ney_global_recreate_20260830/replication/hidden_bias_regional_repair/"
            "source_snapshots/20260814_unified_cartesian_v3/validated_reference_transfer.py"
        ),
    )
    parser.add_argument(
        "--breadth-script",
        type=Path,
        default=Path(
            "/tmp/ney_global_recreate_20260830/replication/hidden_bias_regional_repair/"
            "scripts/section4_breadth_experiments.py"
        ),
    )
    parser.add_argument("--support-data", type=Path, default=Path("/tmp/dml_real_benchmark_support_data"))
    parser.add_argument("--aug14-run-dir", type=Path, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("support_csv/dml_upstream_trust_gate_diagnostic_20260902"),
    )
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--write-replication-rows", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    gate_script = args.gate_script if args.gate_script.is_absolute() else data_root / args.gate_script
    rho_script = args.rho_script if args.rho_script.is_absolute() else data_root / args.rho_script
    gate = _load_module("dml_upstream_trust_gate_loader", gate_script)
    gate.METHODS = METHODS
    rho = _load_module("dml_upstream_trust_rho_loader", rho_script)
    rho.PRIMARY_METHODS = METHODS
    aug14 = args.aug14_run_dir
    if aug14 is None:
        latest = Path("/tmp/dml_rho_extract_latest.txt")
        if not latest.exists():
            raise SystemExit("pass --aug14-run-dir or create /tmp/dml_rho_extract_latest.txt")
        aug14 = Path(latest.read_text().strip()) / "dml" / "cartesian_dml_ks_alignment_v3"

    rows = []
    rows.extend(rho._read_extracted_aug14_rows(aug14))
    rows.extend(
        rho._read_release_rows(
            data_root / "support_csv/dml_real_benchmark_expansion_20260831",
            "real_benchmark_ihdp_acic2016",
        )
    )
    rows.extend(
        rho._read_release_rows(
            data_root / "support_csv/dml_real_benchmark_acic2017_20260831",
            "real_benchmark_acic2017",
        )
    )
    rows.extend(
        rho._read_release_rows(
            data_root / "support_csv/dml_real_benchmark_twins_20260831",
            "real_benchmark_twins",
        )
    )

    sample_groups: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
    for row in rows:
        key = (
            row.design,
            row.n,
            row.epsilon,
            row.strength,
            row.mar_design,
            row.seed0,
            row.rep,
            row.propensity_mode,
            row.learner,
            row.propensity_learner,
            row.folds,
        )
        sample_groups[key].append(row)

    rules = _rules()
    replication_rows: list[dict[str, Any]] = []
    tasks = list(sample_groups.values())
    with ProcessPoolExecutor(
        max_workers=args.jobs,
        initializer=_init_worker,
        initargs=(
            str(gate_script.resolve()),
            str(rho_script.resolve()),
            str(args.source_script.resolve()),
            str(args.breadth_script.resolve()),
            str(args.support_data.resolve()),
        ),
    ) as pool:
        futures = [pool.submit(_process_sample, task, rules) for task in tasks]
        completed = 0
        for future in as_completed(futures):
            replication_rows.extend(future.result())
            completed += 1
            if completed % max(1, args.jobs * 4) == 0 or completed == len(tasks):
                print(f"completed {completed}/{len(tasks)} samples", flush=True)

    replication_rows.sort(
        key=lambda row: (
            row["gate_id"],
            row["dataset"],
            row["setting"],
            row["method"],
            row["seed0"],
            row["rep"],
        )
    )
    metric_keys = (
        "source",
        "dataset",
        "setting",
        "group",
        "design",
        "method",
        "n",
        "strength",
        "seed0",
        "rep",
        "selected_tau",
        "floor_share",
        "observed_floor_share",
        "residual_weight_ess_share",
        "ipw_weight_ess_share",
        "max_residual_weight_share",
        "max_ipw_weight_share",
        "score_tail_ratio",
        "floor_sensitivity_se",
    )
    seen_metric = set()
    metric_rows = []
    for row in replication_rows:
        key = tuple(row[name] for name in metric_keys[:10])
        if key in seen_metric:
            continue
        seen_metric.add(key)
        metric_rows.append({name: row[name] for name in metric_keys})

    setting_rows = gate._aggregate(
        replication_rows,
        (
            "gate_id",
            "floor_sens_max",
            "residual_ess_min",
            "floor_share_max",
            "max_weight_share_max",
            "score_tail_max",
            "veto_floor_sens_max",
            "veto_floor_share_min",
            "veto_residual_ess_max",
            "veto_score_tail_min",
            "dataset",
            "setting",
            "design",
            "method",
            "n",
            "strength",
        ),
        nboot=0,
    )
    dataset_method_rows = gate._aggregate(
        replication_rows,
        (
            "gate_id",
            "floor_sens_max",
            "residual_ess_min",
            "floor_share_max",
            "max_weight_share_max",
            "score_tail_max",
            "veto_floor_sens_max",
            "veto_floor_share_min",
            "veto_residual_ess_max",
            "veto_score_tail_min",
            "dataset",
            "method",
        ),
        nboot=0,
    )
    method_rows = gate._aggregate(
        replication_rows,
        (
            "gate_id",
            "floor_sens_max",
            "residual_ess_min",
            "floor_share_max",
            "max_weight_share_max",
            "score_tail_max",
            "veto_floor_sens_max",
            "veto_floor_share_min",
            "veto_residual_ess_max",
            "veto_score_tail_min",
            "method",
        ),
        nboot=5000,
    )
    objective_rows = _objective_summary(method_rows)

    out_dir = args.out_dir if args.out_dir.is_absolute() else data_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.write_replication_rows:
        _write_csv(out_dir / "gate_replication_rows.csv", replication_rows, gate)
    _write_csv(out_dir / "upstream_metric_rows.csv", metric_rows, gate)
    _write_csv(out_dir / "gate_setting_summary.csv", setting_rows, gate)
    _write_csv(out_dir / "gate_dataset_method_summary.csv", dataset_method_rows, gate)
    _write_csv(out_dir / "gate_method_summary.csv", method_rows, gate)
    _write_csv(out_dir / "gate_objective_summary.csv", objective_rows, gate)
    _write_results(out_dir / "RESULTS.md", objective_rows, method_rows, metric_rows)
    provenance = {
        "source_script": str(args.source_script.resolve()),
        "source_script_sha256": _sha256(args.source_script.resolve()),
        "breadth_script": str(args.breadth_script.resolve()),
        "breadth_script_sha256": _sha256(args.breadth_script.resolve()),
        "rho_script": str(rho_script.resolve()),
        "rho_script_sha256": _sha256(rho_script.resolve()),
        "gate_script": str(gate_script.resolve()),
        "gate_script_sha256": _sha256(gate_script.resolve()),
        "support_data": str(args.support_data.resolve()),
        "aug14_run_dir": str(aug14.resolve()),
        "methods": list(METHODS),
        "floor_sensitivity_taus": list(FLOOR_SENS_TAUS),
        "rules": [rule.__dict__ for rule in rules],
        "rows": len(replication_rows),
        "replication_rows_written": bool(args.write_replication_rows),
        "sample_groups": len(sample_groups),
        "gate": "repair stands down when upstream diagnostic thresholds fail",
        "endpoint": "selected candidate: rt_error if upstream trust passes, reference error otherwise",
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
