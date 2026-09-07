#!/usr/bin/env python3
"""High-response placebo ablation under weighted-residual gamma selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DATA_ROOT = Path(__file__).resolve().parents[1]
WR = None
RHO = None
HIGH = None


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--low-run-dir",
        action="append",
        required=True,
        type=Path,
        help="Raw low-response run directory containing manifest.tsv and *.reps.csv.",
    )
    parser.add_argument(
        "--high-run-dir",
        action="append",
        required=True,
        type=Path,
        help="Raw high-response run directory containing manifest.tsv and *.reps.csv.",
    )
    parser.add_argument(
        "--weighted-script",
        type=Path,
        default=Path("scripts/dml_weighted_residual_gamma_selection_ablation.py"),
    )
    parser.add_argument(
        "--rho-script",
        type=Path,
        default=Path("scripts/dml_mse_gain_by_rho.py"),
    )
    parser.add_argument(
        "--gate-script",
        type=Path,
        default=Path("scripts/dml_universal_leverage_gate_diagnostic.py"),
    )
    parser.add_argument(
        "--high-summary-script",
        type=Path,
        default=Path("scripts/summarize_high_response_placebo_ablation.py"),
    )
    parser.add_argument(
        "--source-script",
        type=Path,
        default=Path(
            "/tmp/ney_global_recreate_20260830/replication/hidden_bias_regional_repair/"
            "scripts/validated_reference_transfer_canonical_region_if_library.py"
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
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("support_csv/dml_weighted_residual_high_response_placebo_ablation_20260903"),
    )
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def _resolve(data_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else data_root / path


def _read_raw_dirs(
    paths: list[Path],
    rho: Any,
    high: Any,
    source_label: str,
) -> list[Any]:
    rows = []
    for run_dir in paths:
        jobs = rho._job_lookup(run_dir / "manifest.tsv")
        for path in sorted(run_dir.glob("*.reps.csv")):
            job = jobs.get(path.name)
            if job is None:
                raise SystemExit(f"no manifest row for {path.name}")
            with path.open(newline="") as handle:
                for raw in csv.DictReader(handle):
                    method = raw.get("reference_method", raw.get("method", ""))
                    if method not in high.PRIMARY_METHODS:
                        continue
                    setting = high.design_setting(
                        raw["design"],
                        int(raw["n"]),
                        float(raw["strength"]),
                    )
                    if setting is None:
                        continue
                    rows.append(rho._method_row(raw, job, source_label))
    return rows


def _init_worker(
    weighted_script: str,
    gate_script: str,
    rho_script: str,
    source_script: str,
    breadth_script: str,
    support_data: str,
) -> None:
    global WR, RHO
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    wr = _load_module("dml_wr_high_response_worker", Path(weighted_script))
    wr._init_worker(gate_script, rho_script, source_script, breadth_script, support_data)
    WR = wr
    RHO = wr.RHO


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


def _process_sample(rows: list[Any]) -> list[dict[str, Any]]:
    if WR is None or RHO is None:
        raise RuntimeError("worker not initialized")
    vrt = RHO._VRT
    first = rows[0]
    seed = RHO._internal_seed(vrt, first)
    x, y, response, region, true_pi, theta, _mu = vrt.make_data(
        first.n,
        first.epsilon,
        first.strength,
        first.design,
        seed,
        first.mar_design,
    )
    out = []
    for row in rows:
        candidates = WR.GATE._candidate_arrays(row, x, y, response, region, true_pi, theta, seed)
        selected = WR._select_by_weighted_residual_loss(row, candidates, y, response)
        out.append(
            {
                "source": row.source,
                "design": row.design,
                "reference_method": row.method,
                "method": row.method,
                "n": row.n,
                "strength": row.strength,
                "seed0": row.seed0,
                "rep": row.rep,
                "ref_error": row.ref_error,
                "rt_error": selected["residual_error"],
                "selected_region_damp": selected["residual_gamma"],
                "score_selected_error": row.selected_error,
                "score_selected_gamma": row.selected_gamma,
                "base_residual_loss": selected["base_residual_loss"],
                "score_selected_residual_loss": selected["current_residual_loss"],
                "weighted_residual_selected_loss": selected["residual_loss"],
                "eligible_count": selected["eligible_count"],
                "gamma_changed": selected["gamma_changed"],
                "pair_id": "",
            }
        )
    return out


def _replay(
    rows: list[Any],
    args: argparse.Namespace,
    data_root: Path,
    label: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
    for row in rows:
        groups[_sample_key(row)].append(row)
    tasks = list(groups.values())
    replayed: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=args.jobs,
        initializer=_init_worker,
        initargs=(
            str(_resolve(data_root, args.weighted_script).resolve()),
            str(_resolve(data_root, args.gate_script).resolve()),
            str(_resolve(data_root, args.rho_script).resolve()),
            str(args.source_script.resolve()),
            str(args.breadth_script.resolve()),
            str(args.support_data.resolve()),
        ),
    ) as pool:
        futures = [pool.submit(_process_sample, task) for task in tasks]
        completed = 0
        for future in as_completed(futures):
            replayed.extend(future.result())
            completed += 1
            if completed % max(1, args.jobs * 4) == 0 or completed == len(tasks):
                print(f"{label}: completed {completed}/{len(tasks)} samples", flush=True)
    for row in replayed:
        key = HIGH.canonical_key(row)
        if key is None:
            raise RuntimeError(f"unexpected setting row after replay: {row}")
        row["pair_id"] = (
            f"{key}|seed={int(row['seed0'])}|rep={int(row['rep'])}"
        )
    replayed.sort(
        key=lambda row: (
            row["source"],
            row["design"],
            row["reference_method"],
            int(row["seed0"]),
            int(row["rep"]),
        )
    )
    return replayed


def _group_for_summary(rows: list[dict[str, Any]], high: Any) -> dict[tuple[str, str, str, int, float], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = high.canonical_key(row)
        if key is not None:
            grouped[key].append(row)
    return grouped


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    global HIGH
    rho = _load_module("dml_wr_high_response_rho_loader", _resolve(data_root, args.rho_script))
    high = _load_module("dml_wr_high_response_summary_loader", _resolve(data_root, args.high_summary_script))
    HIGH = high

    low_source_rows = _read_raw_dirs(args.low_run_dir, rho, high, "weighted_residual_low_response")
    high_source_rows = _read_raw_dirs(args.high_run_dir, rho, high, "weighted_residual_high_response")
    low_replayed = _replay(low_source_rows, args, data_root, "low")
    high_replayed = _replay(high_source_rows, args, data_root, "high")

    low_grouped = _group_for_summary(low_replayed, high)
    high_grouped = _group_for_summary(high_replayed, high)
    if set(low_grouped) != set(high_grouped):
        missing_high = sorted(set(low_grouped) - set(high_grouped))[:10]
        missing_low = sorted(set(high_grouped) - set(low_grouped))[:10]
        raise SystemExit(
            f"low/high setting keys differ: missing_high={missing_high} "
            f"missing_low={missing_low}"
        )
    if len(low_grouped) != 24 * len(high.PRIMARY_METHODS):
        raise SystemExit(
            f"expected 96 low/high setting-expert groups, found {len(low_grouped)}"
        )

    rng = high.random.Random(args.seed)
    setting_rows: list[dict[str, object]] = []
    setting_draws: dict[tuple[str, str, str, int, float], list[tuple[float, float, float]]] = {}
    for key in sorted(low_grouped):
        row, draws = high.summarize_setting(
            key,
            low_grouped[key],
            high_grouped[key],
            rng,
            args.draws,
        )
        setting_rows.append(row)
        setting_draws[key] = draws
    summary_rows = [
        high.summarize_method(method, setting_rows, setting_draws, args.draws)
        for method in high.PRIMARY_METHODS
    ]

    out_dir = args.out_dir if args.out_dir.is_absolute() else data_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    setting_fields = [
        "group",
        "setting",
        "method",
        "n",
        "strength",
        "reps",
        "low_response_gain",
        "low_response_gain_ci_low",
        "low_response_gain_ci_high",
        "high_response_gain",
        "high_response_gain_ci_low",
        "high_response_gain_ci_high",
        "low_minus_high_gain",
        "low_minus_high_gain_ci_low",
        "low_minus_high_gain_ci_high",
        "low_final_activation",
        "high_final_activation",
        "low_harm",
        "high_harm",
    ]
    summary_fields = [
        "method",
        "settings",
        "low_response_gain",
        "low_response_gain_ci_low",
        "low_response_gain_ci_high",
        "high_response_gain",
        "high_response_gain_ci_low",
        "high_response_gain_ci_high",
        "low_minus_high_gain",
        "low_minus_high_gain_ci_low",
        "low_minus_high_gain_ci_high",
        "low_positive_settings",
        "high_positive_settings",
        "low_final_activation",
        "high_final_activation",
        "low_harm",
        "high_harm",
    ]
    high.write_csv(out_dir / "weighted_residual_high_response_placebo_replication_rows.csv", low_replayed + high_replayed, list(low_replayed[0]))
    high.write_csv(out_dir / "high_response_placebo_cell_summary.csv", setting_rows, setting_fields)
    high.write_csv(out_dir / "high_response_placebo_summary.csv", summary_rows, summary_fields)
    high.write_latex_table(
        out_dir / "section4_weighted_residual_high_response_placebo_ablation_table.tex",
        summary_rows,
    )
    text = [
        "# Weighted-Residual Selector High-Response Placebo Ablation",
        "",
        "This bundle replays the existing low-response and high-response placebo",
        "ablation runs, but replaces the original score-selected gamma by",
        "held-out weighted-residual gamma selection.  It does not refit the",
        "upstream nuisances or overwrite the earlier placebo bundle.",
        "",
        "## Summary",
        "",
    ]
    for row in summary_rows:
        text.append(
            "- {method}: low-response {low:.3f}% [{lo:.3f}, {hi:.3f}], "
            "high-response {high_gain:.3f}% [{high_lo:.3f}, {high_hi:.3f}], "
            "low-minus-high {diff:.3f}% [{diff_lo:.3f}, {diff_hi:.3f}], "
            "activation low/high {low_act:.1f}%/{high_act:.1f}%.".format(
                method=high.METHOD_LABELS[str(row["method"])],
                low=100.0 * float(row["low_response_gain"]),
                lo=100.0 * float(row["low_response_gain_ci_low"]),
                hi=100.0 * float(row["low_response_gain_ci_high"]),
                high_gain=100.0 * float(row["high_response_gain"]),
                high_lo=100.0 * float(row["high_response_gain_ci_low"]),
                high_hi=100.0 * float(row["high_response_gain_ci_high"]),
                diff=100.0 * float(row["low_minus_high_gain"]),
                diff_lo=100.0 * float(row["low_minus_high_gain_ci_low"]),
                diff_hi=100.0 * float(row["low_minus_high_gain_ci_high"]),
                low_act=100.0 * float(row["low_final_activation"]),
                high_act=100.0 * float(row["high_final_activation"]),
            )
        )
    (out_dir / "RESULTS.md").write_text("\n".join(text) + "\n")
    verification = {
        "status": "PASS",
        "draws": args.draws,
        "seed": args.seed,
        "settings": 24,
        "methods": list(high.PRIMARY_METHODS),
        "setting_expert_groups": len(setting_rows),
        "low_source_rows": len(low_source_rows),
        "high_source_rows": len(high_source_rows),
        "low_replayed_rows": len(low_replayed),
        "high_replayed_rows": len(high_replayed),
        "high_run_dirs": [str(path) for path in args.high_run_dir],
        "low_run_dirs": [str(path) for path in args.low_run_dir],
        "weighted_script": str(_resolve(data_root, args.weighted_script).resolve()),
        "weighted_script_sha256": _sha256(_resolve(data_root, args.weighted_script).resolve()),
        "rho_script": str(_resolve(data_root, args.rho_script).resolve()),
        "rho_script_sha256": _sha256(_resolve(data_root, args.rho_script).resolve()),
        "gate_script": str(_resolve(data_root, args.gate_script).resolve()),
        "gate_script_sha256": _sha256(_resolve(data_root, args.gate_script).resolve()),
        "source_script": str(args.source_script.resolve()),
        "source_script_sha256": _sha256(args.source_script.resolve()),
        "breadth_script": str(args.breadth_script.resolve()),
        "breadth_script_sha256": _sha256(args.breadth_script.resolve()),
        "support_data": str(args.support_data.resolve()),
    }
    (out_dir / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n"
    )
    checksum_rows = []
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            checksum_rows.append(f"{_sha256(path)}  {path.name}")
    (out_dir / "SHA256SUMS").write_text("\n".join(checksum_rows) + "\n")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
