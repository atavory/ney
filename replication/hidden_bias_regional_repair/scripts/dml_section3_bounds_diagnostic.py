#!/usr/bin/env python3
"""Numerical diagnostics for the Section 3.2--3.3 repair bounds.

This script replays the paper-facing benchmark samples, reconstructs the
selected weighted-residual candidate path, and computes the quantities in
Subsections 3.2 and 3.3 from the known simulation truth.  It writes a separate
diagnostic bundle and does not modify any Section 4 result bundle.

The diagnostics are retrospective.  They use true pi and mu from the simulation
designs to evaluate the bound terms; those quantities are not used by the
repair selector.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DATA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_SCRIPT = Path(
    "/tmp/dml_section3_verify_extract/replication/hidden_bias_regional_repair/"
    "scripts/validated_reference_transfer.py"
)
DEFAULT_BREADTH_SCRIPT = Path(
    "/tmp/dml_section3_verify_extract/replication/hidden_bias_regional_repair/"
    "scripts/section4_breadth_experiments.py"
)
DEFAULT_SUPPORT_DATA = Path("/tmp/dml_real_benchmark_support_data")
DEFAULT_AUG14_RUN_DIR = Path("/tmp/dml_section3_verify_extract/cartesian_dml_ks_alignment_v3")
DEFAULT_RHO_SCRIPT = DATA_ROOT / "scripts/dml_weighted_residual_rho_helper.py"
DEFAULT_GATE_SCRIPT = DATA_ROOT / "scripts/dml_universal_leverage_gate_diagnostic.py"
PRIMARY_METHODS = ("aipw", "ctmle", "cui_selective_ml", "ma_dr_bc")
ALL_METHODS = PRIMARY_METHODS + ("tmle",)
SIGMA2 = 1.0

GATE = None


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
    gate = _load_module("dml_section3_bounds_gate", Path(gate_script))
    gate.METHODS = ALL_METHODS
    gate._init_worker(rho_script, source_script, breadth_script, support_data)
    GATE = gate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _quantile(values: list[float], probability: float) -> float:
    vals = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    index = probability * (len(vals) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return vals[lower]
    weight = index - lower
    return vals[lower] * (1.0 - weight) + vals[upper] * weight


def _emp_mean(values: np.ndarray) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def _emp_l2(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def _emp_var(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.mean(values * values) - np.mean(values) ** 2)


def _score_variance(
    mu: np.ndarray,
    true_pi: np.ndarray,
    p: np.ndarray,
    m: np.ndarray,
) -> float:
    e = np.asarray(m, dtype=float) - np.asarray(mu, dtype=float)
    pi = np.asarray(true_pi, dtype=float)
    p = np.asarray(p, dtype=float)
    conditional_variance = pi / (p * p) * SIGMA2 + pi * (1.0 - pi) / (p * p) * e * e
    conditional_mean = np.asarray(mu, dtype=float) + (p - pi) / p * e
    return _emp_mean(conditional_variance) + _emp_var(conditional_mean)


def _bound_quantities(
    true_pi: np.ndarray,
    mu: np.ndarray,
    p: np.ndarray,
    m0: np.ndarray,
    mg: np.ndarray,
    n: int,
) -> dict[str, float]:
    pi = np.clip(np.asarray(true_pi, dtype=float), 1e-12, 1.0 - 1e-12)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    mu = np.asarray(mu, dtype=float)
    m0 = np.asarray(m0, dtype=float)
    mg = np.asarray(mg, dtype=float)
    d = mg - m0
    xi = p - pi
    e0 = m0 - mu
    eg = mg - mu
    w = pi * (1.0 - p) / (p * p)

    q0 = _emp_mean(w * (SIGMA2 + e0 * e0))
    qg = _emp_mean(w * (SIGMA2 + eg * eg))
    e0_weighted = _emp_mean(w * e0 * e0)
    eg_weighted = _emp_mean(w * eg * eg)
    step_size = _emp_mean(w * d * d)
    rho2 = _emp_mean(xi * xi / (pi * (1.0 - p)))
    b0 = _emp_mean(xi / p * e0)
    bg = _emp_mean(xi / p * eg)
    a0 = _score_variance(mu, pi, p, m0) / float(n)
    ag = _score_variance(mu, pi, p, mg) / float(n)

    eta = min(
        float(np.min(pi)),
        float(np.min(p)),
        1.0 - float(np.max(pi)),
        1.0 - float(np.max(p)),
    )
    delta = float(np.max(np.abs(xi)))
    s_mu = math.sqrt(max(0.0, _emp_var(mu)))
    l2_e0 = _emp_l2(e0)
    l2_eg = _emp_l2(eg)
    l2_d = _emp_l2(d)

    if eta <= 0.0:
        b_global = b_local = b_min = float("inf")
    else:
        b_global = sum(
            2.0 * s_mu * delta / eta * value
            + (delta * delta / (eta * eta) + delta / (eta * eta)) * value * value
            for value in (l2_e0, l2_eg)
        )
        b_local = (
            2.0 * (s_mu + delta / eta * l2_e0) * delta / eta * l2_d
            + delta * delta / (eta * eta) * l2_d * l2_d
            + delta / (eta * eta) * (2.0 * l2_e0 * l2_d + l2_d * l2_d)
        )
        b_min = min(b_global, b_local)

    gain = q0 - qg
    variance_rhs = a0 - (gain - b_min) / float(n)
    variance_slack = variance_rhs - ag
    bias_abs_bound = rho2 * eg_weighted
    bias_path_bound = b0 * b0 + rho2 * (
        2.0 * math.sqrt(max(0.0, e0_weighted * step_size)) + step_size
    )
    bias_min_bound = min(bias_abs_bound, bias_path_bound)
    return {
        "n_eval": float(n),
        "eta": eta,
        "delta_inf": delta,
        "rho2": rho2,
        "q0": q0,
        "qg": qg,
        "G_xi": gain,
        "E_xi_m0": e0_weighted,
        "E_xi_mg": eg_weighted,
        "D_xi_gamma": step_size,
        "bias0": b0,
        "biasg": bg,
        "bias0_sq": b0 * b0,
        "biasg_sq": bg * bg,
        "A0": a0,
        "Ag": ag,
        "variance_reduction": a0 - ag,
        "B_global": b_global,
        "B_local": b_local,
        "B_min": b_min,
        "variance_bound_rhs": variance_rhs,
        "variance_bound_slack": variance_slack,
        "variance_bound_violation": float(variance_slack < -1e-10),
        "variance_certifies_improvement": float(gain > b_min),
        "bias_abs_bound": bias_abs_bound,
        "bias_path_bound": bias_path_bound,
        "bias_min_bound": bias_min_bound,
        "bias_abs_slack": bias_abs_bound - bg * bg,
        "bias_path_slack": bias_path_bound - bg * bg,
        "bias_min_slack": bias_min_bound - bg * bg,
        "bias_bound_violation": float(bias_min_bound - bg * bg < -1e-10),
        "bias_abs_term_tighter": float(bias_abs_bound <= bias_path_bound),
        "bias_path_term_tighter": float(bias_path_bound < bias_abs_bound),
    }


def _process_sample(rows: list[Any]) -> list[dict[str, Any]]:
    if GATE is None or GATE.RHO is None:
        raise RuntimeError("worker not initialized")
    vrt = GATE.RHO._VRT
    first = rows[0]
    seed = GATE.RHO._internal_seed(vrt, first)
    x, y, response, region, true_pi, theta, mu = vrt.make_data(
        first.n,
        first.epsilon,
        first.strength,
        first.design,
        seed,
        first.mar_design,
    )
    out = []
    for row in rows:
        if row.method not in ALL_METHODS:
            continue
        candidates = GATE._candidate_arrays(row, x, y, response, region, true_pi, theta, seed)
        gamma = float(row.selected_gamma)
        if gamma not in candidates["outcomes"]:
            raise RuntimeError(f"missing selected gamma {gamma} for {row}")
        base_outcome = np.asarray(candidates["base_outcome"], dtype=float)
        selected_outcome = np.asarray(candidates["outcomes"][gamma], dtype=float)
        p = np.asarray(candidates["p"], dtype=float)
        quantities = _bound_quantities(true_pi, mu, p, base_outcome, selected_outcome, int(row.n))
        ref_mse = float(row.ref_error) ** 2
        selected_mse = float(row.selected_error) ** 2
        out.append(
            {
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
                "selected_gamma": gamma,
                "ref_error": row.ref_error,
                "selected_error": row.selected_error,
                "mse_gain": 1.0 - selected_mse / ref_mse if ref_mse > 0.0 else float("nan"),
                **quantities,
            }
        )
    return out


def _read_rows(args: argparse.Namespace) -> list[Any]:
    rho = _load_module("dml_section3_bounds_rho_reader", args.rho_script)
    methods = tuple(args.method_filter) if args.method_filter else PRIMARY_METHODS
    unknown_methods = set(methods).difference(ALL_METHODS)
    if unknown_methods:
        raise RuntimeError(f"unknown method filter: {sorted(unknown_methods)}")
    rho.PRIMARY_METHODS = methods
    if hasattr(rho, "_BASE"):
        rho._BASE.PRIMARY_METHODS = methods
    rows: list[Any] = []
    rows.extend(rho._read_extracted_aug14_rows(args.aug14_run_dir))
    rows.extend(
        rho._read_release_rows(
            args.data_root / "support_csv/dml_real_benchmark_expansion_20260831",
            "real_benchmark_ihdp_acic2016",
        )
    )
    rows.extend(
        rho._read_release_rows(
            args.data_root / "support_csv/dml_real_benchmark_acic2017_20260831",
            "real_benchmark_acic2017",
        )
    )
    rows.extend(
        rho._read_release_rows(
            args.data_root / "support_csv/dml_real_benchmark_twins_20260831",
            "real_benchmark_twins",
        )
    )
    rows = [row for row in rows if row.method in set(methods)]
    if args.source_filter:
        sources = set(args.source_filter)
        rows = [row for row in rows if row.source in sources]
    if args.design_filter:
        designs = set(args.design_filter)
        rows = [row for row in rows if row.design in designs]
    if args.max_rows:
        rows = rows[: args.max_rows]
    return rows


def _sample_key(row: Any) -> tuple[Any, ...]:
    return (
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


def _aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out = []
    for key_values, group in sorted(groups.items()):
        active_group = [
            row for row in group if abs(float(row["selected_gamma"])) > 1e-12
        ]
        result = dict(zip(keys, key_values))
        result.update(
            {
                "rows": len(group),
                "active_share": _mean(
                    abs(float(row["selected_gamma"])) > 1e-12 for row in group
                ),
                "mean_mse_gain": _mean(row["mse_gain"] for row in group),
                "harm_share": _mean(
                    float(row["selected_error"]) ** 2 > float(row["ref_error"]) ** 2
                    for row in group
                ),
                "active_harm_share": _mean(
                    float(row["selected_error"]) ** 2 > float(row["ref_error"]) ** 2
                    for row in active_group
                ),
                "mean_G_xi": _mean(row["G_xi"] for row in group),
                "positive_G_xi_share": _mean(row["G_xi"] > 0.0 for row in group),
                "active_positive_G_xi_share": _mean(
                    row["G_xi"] > 0.0 for row in active_group
                ),
                "mean_rho2": _mean(row["rho2"] for row in group),
                "mean_bias0_sq": _mean(row["bias0_sq"] for row in group),
                "mean_biasg_sq": _mean(row["biasg_sq"] for row in group),
                "mean_bias_sq_change": _mean(
                    row["bias0_sq"] - row["biasg_sq"] for row in group
                ),
                "positive_bias_sq_change_share": _mean(
                    row["bias0_sq"] - row["biasg_sq"] > 0.0 for row in group
                ),
                "active_positive_bias_sq_change_share": _mean(
                    row["bias0_sq"] - row["biasg_sq"] > 0.0 for row in active_group
                ),
                "mean_variance_reduction": _mean(
                    row["variance_reduction"] for row in group
                ),
                "positive_variance_reduction_share": _mean(
                    row["variance_reduction"] > 0.0 for row in group
                ),
                "active_positive_variance_reduction_share": _mean(
                    row["variance_reduction"] > 0.0 for row in active_group
                ),
                "variance_violations": int(
                    sum(int(row["variance_bound_violation"]) for row in group)
                ),
                "bias_violations": int(sum(int(row["bias_bound_violation"]) for row in group)),
                "variance_certified": int(
                    sum(int(row["variance_certifies_improvement"]) for row in group)
                ),
                "variance_certified_share": _mean(
                    row["variance_certifies_improvement"] for row in group
                ),
                "abs_term_tighter_share": _mean(
                    row["bias_abs_term_tighter"] for row in group
                ),
                "path_term_tighter_share": _mean(
                    row["bias_path_term_tighter"] for row in group
                ),
                "min_variance_slack": min(float(row["variance_bound_slack"]) for row in group),
                "p05_variance_slack": _quantile(
                    [float(row["variance_bound_slack"]) for row in group], 0.05
                ),
                "median_variance_slack": _quantile(
                    [float(row["variance_bound_slack"]) for row in group], 0.50
                ),
                "min_bias_slack": min(float(row["bias_min_slack"]) for row in group),
                "p05_bias_slack": _quantile(
                    [float(row["bias_min_slack"]) for row in group], 0.05
                ),
                "median_bias_slack": _quantile(
                    [float(row["bias_min_slack"]) for row in group], 0.50
                ),
            }
        )
        out.append(result)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            converted: dict[str, Any] = {}
            for key, value in row.items():
                try:
                    converted[key] = float(value)
                except ValueError:
                    converted[key] = value
            rows.append(converted)
    return rows


def _write_results(path: Path, overall: dict[str, Any], by_method: list[dict[str, Any]]) -> None:
    lines = [
        "# Section 3 Bounds Diagnostic",
        "",
        "This retrospective diagnostic evaluates the numerical quantities in",
        "Subsections 3.2 and 3.3 on the paper-facing benchmark fits. It uses true",
        "`pi` and `mu` from the simulation designs and is not a selection rule.",
        "",
        "The variance and bias inequalities are checked on the empirical",
        "distribution of the generated covariates, using the additive unit noise",
        "variance in the benchmark data generators.",
        "",
        "## Overall",
        "",
        f"- Rows: {overall['rows']}",
        f"- Nonzero selected-gamma share: {overall['active_share']:.3f}",
        f"- Harm share: {overall['harm_share']:.3f}",
        f"- Active-move harm share: {overall['active_harm_share']:.3f}",
        f"- Variance-bound violations: {overall['variance_violations']}",
        f"- Squared-bias-bound violations: {overall['bias_violations']}",
        f"- Variance sufficient-condition share: {overall['variance_certified_share']:.3f}",
        f"- Positive weighted-risk-improvement share: {overall['positive_G_xi_share']:.3f}",
        f"- Active positive weighted-risk-improvement share: {overall['active_positive_G_xi_share']:.3f}",
        f"- Positive average-variance-reduction share: {overall['positive_variance_reduction_share']:.3f}",
        f"- Active positive average-variance-reduction share: {overall['active_positive_variance_reduction_share']:.3f}",
        f"- Positive hidden-squared-bias-reduction share: {overall['positive_bias_sq_change_share']:.3f}",
        f"- Active positive hidden-squared-bias-reduction share: {overall['active_positive_bias_sq_change_share']:.3f}",
        f"- Mean hidden squared-bias change: {overall['mean_bias_sq_change']:.6g}",
        f"- Mean average-variance reduction: {overall['mean_variance_reduction']:.6g}",
        "",
        "## By Expert",
        "",
    ]
    for row in by_method:
        lines.append(
            "- {method}: rows {rows}, variance violations {variance_violations}, "
            "bias violations {bias_violations}, variance certified {variance_certified_share:.3f}, "
            "active {active_share:.3f}, active harm {active_harm_share:.3f}, "
            "active positive weighted-risk improvement {active_positive_G:.3f}, "
            "active positive variance reduction {active_positive_variance:.3f}, "
            "active positive squared-bias reduction {active_positive_bias:.3f}".format(
                method=row["method"],
                rows=row["rows"],
                variance_violations=row["variance_violations"],
                bias_violations=row["bias_violations"],
                variance_certified_share=row["variance_certified_share"],
                active_share=row["active_share"],
                active_harm_share=row["active_harm_share"],
                active_positive_G=row["active_positive_G_xi_share"],
                active_positive_variance=row["active_positive_variance_reduction_share"],
                active_positive_bias=row["active_positive_bias_sq_change_share"],
            )
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--source-script", type=Path, default=DEFAULT_SOURCE_SCRIPT)
    parser.add_argument("--breadth-script", type=Path, default=DEFAULT_BREADTH_SCRIPT)
    parser.add_argument("--support-data", type=Path, default=DEFAULT_SUPPORT_DATA)
    parser.add_argument("--aug14-run-dir", type=Path, default=DEFAULT_AUG14_RUN_DIR)
    parser.add_argument("--rho-script", type=Path, default=DEFAULT_RHO_SCRIPT)
    parser.add_argument("--gate-script", type=Path, default=DEFAULT_GATE_SCRIPT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_ROOT / "support_csv/dml_section3_bounds_diagnostic_20260906",
    )
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--source-filter", action="append", default=[])
    parser.add_argument("--design-filter", action="append", default=[])
    parser.add_argument("--method-filter", action="append", default=[])
    parser.add_argument("--reuse-replication-rows", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="l1_ratio parameter is only used when penalty is 'elasticnet'",
        category=UserWarning,
    )
    args = parse_args()
    if args.reuse_replication_rows is None:
        rows = _read_rows(args)
        groups: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
        for row in rows:
            groups[_sample_key(row)].append(row)

        rep_rows: list[dict[str, Any]] = []
        tasks = list(groups.values())
        with ProcessPoolExecutor(
            max_workers=args.jobs,
            initializer=_init_worker,
            initargs=(
                str(args.gate_script.resolve()),
                str(args.rho_script.resolve()),
                str(args.source_script.resolve()),
                str(args.breadth_script.resolve()),
                str(args.support_data.resolve()),
            ),
        ) as pool:
            futures = [pool.submit(_process_sample, task) for task in tasks]
            completed = 0
            for future in as_completed(futures):
                rep_rows.extend(future.result())
                completed += 1
                if completed % max(1, args.jobs * 4) == 0 or completed == len(tasks):
                    print(f"completed {completed}/{len(tasks)} samples", flush=True)
    else:
        rep_rows = _read_csv_rows(args.reuse_replication_rows)
        tasks = []

    rep_rows.sort(
        key=lambda row: (
            row["dataset"],
            row["setting"],
            row["method"],
            int(row["seed0"]),
            int(row["rep"]),
        )
    )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "section3_bounds_replication_rows.csv", rep_rows)
    by_method = _aggregate(rep_rows, ("method",))
    by_dataset = _aggregate(rep_rows, ("dataset",))
    by_dataset_method = _aggregate(rep_rows, ("dataset", "method"))
    overall = _aggregate(rep_rows, tuple())[0]
    _write_csv(out_dir / "section3_bounds_method_summary.csv", by_method)
    _write_csv(out_dir / "section3_bounds_dataset_summary.csv", by_dataset)
    _write_csv(out_dir / "section3_bounds_dataset_method_summary.csv", by_dataset_method)
    _write_csv(out_dir / "section3_bounds_overall_summary.csv", [overall])
    _write_results(out_dir / "RESULTS.md", overall, by_method)

    provenance = {
        "diagnostic": "Section 3.2 average-variance and Section 3.3 squared-bias bound replay",
        "rows": len(rep_rows),
        "sample_groups": len(tasks),
        "methods": sorted({str(row["method"]) for row in rep_rows}),
        "sigma2": SIGMA2,
        "source_script": str(args.source_script),
        "source_script_sha256": _sha256(args.source_script),
        "breadth_script": str(args.breadth_script),
        "breadth_script_sha256": _sha256(args.breadth_script),
        "rho_script": str(args.rho_script),
        "rho_script_sha256": _sha256(args.rho_script),
        "gate_script": str(args.gate_script),
        "gate_script_sha256": _sha256(args.gate_script),
        "support_data": str(args.support_data),
        "aug14_run_dir": str(args.aug14_run_dir),
        "reuse_replication_rows": (
            str(args.reuse_replication_rows) if args.reuse_replication_rows else None
        ),
        "retrospective_truth": "uses true pi and mu from simulation designs",
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    sums = []
    for path in sorted(out_dir.iterdir()):
        if path.name == "SHA256SUMS" or not path.is_file():
            continue
        sums.append(f"{_sha256(path)}  {path.name}")
    (out_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n")


if __name__ == "__main__":
    main()
