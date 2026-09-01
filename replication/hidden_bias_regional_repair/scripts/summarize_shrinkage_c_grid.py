#!/usr/bin/env python3
"""Summarize a post-selection shrinkage-c grid from Section 4 row data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from recreate_unified_cartesian_global_residual import mean, percentile


DATA_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_METHODS = ("aipw", "cui_selective_ml", "ma_dr_bc", "ctmle")
METHOD_LABELS = {
    "aipw": "AIPW",
    "ctmle": "C-TMLE",
    "cui_selective_ml": "selective ML",
    "ma_dr_bc": "Ma DR-BC",
}
BENCHMARK_DATASETS = (
    (
        "ihdp",
        "IHDP",
        "support_csv/dml_real_benchmark_expansion_20260831/raw_rows.csv",
        ("ihdp_semisynth", "ihdp_misaligned"),
    ),
    (
        "acic2016",
        "ACIC 2016",
        "support_csv/dml_real_benchmark_expansion_20260831/raw_rows.csv",
        ("acic2016_semisynth", "acic2016_misaligned"),
    ),
    (
        "acic2017",
        "ACIC 2017",
        "support_csv/dml_real_benchmark_acic2017_20260831/raw_rows.csv",
        ("acic2017_semisynth", "acic2017_misaligned"),
    ),
    (
        "twins",
        "Twins",
        "support_csv/dml_real_benchmark_twins_20260831/raw_rows.csv",
        ("twins_semisynth", "twins_misaligned"),
    ),
)
DEFAULT_C_GRID = tuple([i / 4 for i in range(0, 21)] + [6.0, 8.0, 10.0])
HARM_THRESHOLDS = (0.01, 0.02, 0.03, 0.05, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ks-run-dir",
        required=True,
        type=Path,
        help="Extracted Aug. 14 KS/alignment run directory containing *.reps.csv.",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--c-grid",
        default=",".join(f"{value:g}" for value in DEFAULT_C_GRID),
        help="Comma-separated nonnegative c values.",
    )
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args()


def parse_c_grid(raw: str) -> list[float]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value < 0:
            raise SystemExit("c-grid values must be nonnegative")
        values.append(value)
    if not values:
        raise SystemExit("empty c-grid")
    return sorted(set(values))


def as_float(value: str) -> float:
    if value in ("", "NA", "NaN", "nan"):
        return math.nan
    return float(value)


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in fields})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shrink_weight(row: dict[str, str], c_value: float) -> float:
    delta = as_float(row["delta"])
    variance = as_float(row["vd"])
    selected = as_float(row["selected_region_damp"])
    if selected == 0.0 or abs(delta) <= 1e-15:
        return 0.0
    return max(0.0, 1.0 - c_value * variance / (delta * delta))


def error_at_c(row: dict[str, str], c_value: float) -> float:
    return as_float(row["ref_error"]) + shrink_weight(row, c_value) * as_float(row["delta"])


def gain(reference_sq: list[float], target_sq: list[float]) -> float:
    reference_mse = mean(reference_sq)
    if reference_mse == 0.0:
        raise SystemExit("zero reference MSE encountered")
    return 1.0 - mean(target_sq) / reference_mse


def paired_gain_draws(
    reference_sq: list[float],
    target_by_c: dict[float, list[float]],
    rng: np.random.Generator,
    draws: int,
) -> dict[float, np.ndarray]:
    reference = np.asarray(reference_sq, dtype=float)
    n = len(reference)
    indices = rng.integers(0, n, size=(draws, n))
    ref_total = reference[indices].sum(axis=1)
    out: dict[float, np.ndarray] = {}
    for c_value, target_sq in target_by_c.items():
        target = np.asarray(target_sq, dtype=float)
        out[c_value] = 1.0 - target[indices].sum(axis=1) / ref_total
    return out


def summarize_setting(
    setting: tuple[str, str, str, int, float],
    rows: list[dict[str, str]],
    c_grid: Iterable[float],
    rng: np.random.Generator,
    draws: int,
) -> tuple[list[dict[str, object]], dict[float, np.ndarray]]:
    dataset, setting_id, method, n, strength = setting
    reference_sq = [as_float(row["ref_error"]) ** 2 for row in rows]
    target_sq = {
        c_value: [error_at_c(row, c_value) ** 2 for row in rows]
        for c_value in c_grid
    }
    gain_draws = paired_gain_draws(reference_sq, target_sq, rng, draws)
    result_rows = []
    for c_value in c_grid:
        weights = [shrink_weight(row, c_value) for row in rows]
        target = target_sq[c_value]
        ordered_draws = sorted(float(value) for value in gain_draws[c_value])
        result_rows.append(
            {
                "dataset": dataset,
                "setting": setting_id,
                "method": method,
                "n": n,
                "strength": strength,
                "c": c_value,
                "reps": len(rows),
                "reference_mse": mean(reference_sq),
                "repaired_mse": mean(target),
                "gain": gain(reference_sq, target),
                "gain_ci_low": percentile(ordered_draws, 2.5),
                "gain_ci_high": percentile(ordered_draws, 97.5),
                "activation": mean([1.0 if weight > 0.0 else 0.0 for weight in weights]),
                "harm": mean(
                    [
                        1.0 if target_value > reference_value else 0.0
                        for reference_value, target_value in zip(reference_sq, target)
                    ]
                ),
                "mean_weight": mean(weights),
            }
        )
    return result_rows, gain_draws


def summarize_group(
    name: str,
    rows: list[dict[str, object]],
    draws_by_setting: dict[tuple[str, str, str, int, float], dict[float, np.ndarray]],
    c_grid: Iterable[float],
    draws: int,
) -> list[dict[str, object]]:
    out = []
    settings_by_c: dict[float, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        settings_by_c[float(row["c"])].append(row)
    for c_value in c_grid:
        c_rows = settings_by_c[c_value]
        by_setting = [
            draws_by_setting[
                (
                    str(row["dataset"]),
                    str(row["setting"]),
                    str(row["method"]),
                    int(row["n"]),
                    float(row["strength"]),
                )
            ][c_value]
            for row in c_rows
        ]
        if not by_setting:
            raise SystemExit(f"empty summary group for {name}")
        group_draws = np.mean(np.vstack(by_setting), axis=0)
        if len(group_draws) != draws:
            raise SystemExit("unexpected draw count")
        ordered = sorted(float(value) for value in group_draws)
        out.append(
            {
                "summary": name,
                "method": c_rows[0]["method"],
                "c": c_value,
                "settings": len(c_rows),
                "reps": sum(int(row["reps"]) for row in c_rows),
                "gain": mean([float(row["gain"]) for row in c_rows]),
                "gain_ci_low": percentile(ordered, 2.5),
                "gain_ci_high": percentile(ordered, 97.5),
                "activation": mean([float(row["activation"]) for row in c_rows]),
                "harm": mean([float(row["harm"]) for row in c_rows]),
                "mean_weight": mean([float(row["mean_weight"]) for row in c_rows]),
            }
        )
    return out


def load_ks_rows(ks_run_dir: Path) -> dict[tuple[str, str, str, int, float], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, str, int, float], list[dict[str, str]]] = defaultdict(list)
    for path in sorted(ks_run_dir.glob("*.reps.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                method = row["reference_method"]
                design = row["design"]
                if method not in PRIMARY_METHODS or not design.startswith("kang_schafer_"):
                    continue
                setting = f"{design.removeprefix('kang_schafer_')}_n{int(row['n'])}"
                key = ("kang_schafer", setting, method, int(row["n"]), float(row["strength"]))
                grouped[key].append(row)
    return grouped


def load_benchmark_rows() -> dict[tuple[str, str, str, int, float], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, str, int, float], list[dict[str, str]]] = defaultdict(list)
    for dataset, _label, raw_path, designs in BENCHMARK_DATASETS:
        allowed = set(designs)
        with (DATA_ROOT / raw_path).open(newline="") as handle:
            for row in csv.DictReader(handle):
                method = row["reference_method"]
                design = row["design"]
                if method not in PRIMARY_METHODS or design not in allowed:
                    continue
                design_suffix = "A" if design.endswith("_semisynth") else "B"
                setting = f"{design_suffix}_s{float(row['strength']):g}"
                key = (dataset, setting, method, int(row["n"]), float(row["strength"]))
                grouped[key].append(row)
    return grouped


def write_readme(
    path: Path,
    best_rows: list[dict[str, object]],
    frontier_rows: list[dict[str, object]],
    c_grid: list[float],
) -> None:
    lines = [
        "# Shrinkage-c Grid",
        "",
        "This bundle re-evaluates the final scalar shrinkage constant using the",
        "saved Section 4 replication rows.  It does not refit nuisance models or",
        "reselect the damping candidate; it changes only the final weight",
        "`max(0, 1 - c * vd / delta^2)` applied to the selected movement.",
        "",
        "The benchmark target is the same 24-setting table used in the current",
        "manuscript: eight Kang--Schafer settings and four settings each for",
        "IHDP, ACIC 2016, ACIC 2017, and Twins, over AIPW, selective ML,",
        "Ma DR-BC, and C-TMLE.",
        "",
        f"Grid: {', '.join(f'{value:g}' for value in c_grid)}.",
        "",
        "Best equal-setting benchmark gains on this grid:",
        "",
        "| expert | best c | gain | interval | harm | activation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in best_rows:
        lines.append(
            f"| {METHOD_LABELS[str(row['method'])]} | {float(row['c']):g} | "
            f"{100 * float(row['gain']):.3f}% | "
            f"[{100 * float(row['gain_ci_low']):.3f}, "
            f"{100 * float(row['gain_ci_high']):.3f}] | "
            f"{100 * float(row['harm']):.3f}% | "
            f"{100 * float(row['activation']):.3f}% |"
        )
    lines.extend(
        [
            "",
            "Best gains subject to harm no greater than 5%:",
            "",
            "| expert | c | gain | harm | activation |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in frontier_rows:
        if abs(float(row["harm_threshold"]) - 0.05) > 1e-12:
            continue
        lines.append(
            f"| {METHOD_LABELS[str(row['method'])]} | {float(row['c']):g} | "
            f"{100 * float(row['gain']):.3f}% | "
            f"{100 * float(row['harm']):.3f}% | "
            f"{100 * float(row['activation']):.3f}% |"
        )
    lines.extend(
        [
            "",
            "Files:",
            "",
            "- `c_grid_setting_summary.csv`: setting-level summaries.",
            "- `c_grid_dataset_summary.csv`: dataset-level summaries.",
            "- `c_grid_benchmark_summary.csv`: 24-setting summaries by expert and c.",
            "- `c_grid_best_by_expert.csv`: best c by equal-setting benchmark gain.",
            "- `c_grid_best_under_harm.csv`: best c by expert under harm caps.",
            "- `verification.json`: row counts, grid values, and source hashes.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    c_grid = parse_c_grid(args.c_grid)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_groups = load_ks_rows(args.ks_run_dir)
    raw_groups.update(load_benchmark_rows())
    expected = 24 * len(PRIMARY_METHODS)
    if len(raw_groups) != expected:
        raise SystemExit(f"expected {expected} method/settings, saw {len(raw_groups)}")

    rng = np.random.default_rng(args.seed)
    setting_rows: list[dict[str, object]] = []
    draws_by_setting: dict[tuple[str, str, str, int, float], dict[float, np.ndarray]] = {}
    for key in sorted(raw_groups):
        rows, draws = summarize_setting(key, raw_groups[key], c_grid, rng, args.draws)
        setting_rows.extend(rows)
        draws_by_setting[key] = draws

    dataset_rows: list[dict[str, object]] = []
    benchmark_rows: list[dict[str, object]] = []
    for method in PRIMARY_METHODS:
        method_rows = [row for row in setting_rows if row["method"] == method]
        benchmark_rows.extend(
            summarize_group("benchmark", method_rows, draws_by_setting, c_grid, args.draws)
        )
        for dataset in ("kang_schafer", "ihdp", "acic2016", "acic2017", "twins"):
            dataset_rows.extend(
                summarize_group(
                    dataset,
                    [row for row in method_rows if row["dataset"] == dataset],
                    draws_by_setting,
                    c_grid,
                    args.draws,
                )
            )

    best_rows = []
    frontier_rows = []
    for method in PRIMARY_METHODS:
        candidates = [row for row in benchmark_rows if row["method"] == method]
        best_rows.append(max(candidates, key=lambda row: (float(row["gain"]), -float(row["c"]))))
        for threshold in HARM_THRESHOLDS:
            admissible = [
                row for row in candidates if float(row["harm"]) <= threshold
            ]
            if not admissible:
                continue
            best = max(admissible, key=lambda row: (float(row["gain"]), -float(row["c"])))
            frontier_rows.append({"harm_threshold": threshold, **best})

    setting_fields = [
        "dataset",
        "setting",
        "method",
        "n",
        "strength",
        "c",
        "reps",
        "reference_mse",
        "repaired_mse",
        "gain",
        "gain_ci_low",
        "gain_ci_high",
        "activation",
        "harm",
        "mean_weight",
    ]
    summary_fields = [
        "summary",
        "method",
        "c",
        "settings",
        "reps",
        "gain",
        "gain_ci_low",
        "gain_ci_high",
        "activation",
        "harm",
        "mean_weight",
    ]
    best_fields = [
        "method",
        "c",
        "settings",
        "reps",
        "gain",
        "gain_ci_low",
        "gain_ci_high",
        "activation",
        "harm",
        "mean_weight",
    ]
    frontier_fields = ["harm_threshold", *best_fields]
    write_csv(args.out_dir / "c_grid_setting_summary.csv", setting_rows, setting_fields)
    write_csv(args.out_dir / "c_grid_dataset_summary.csv", dataset_rows, summary_fields)
    write_csv(args.out_dir / "c_grid_benchmark_summary.csv", benchmark_rows, summary_fields)
    write_csv(args.out_dir / "c_grid_best_by_expert.csv", best_rows, best_fields)
    write_csv(args.out_dir / "c_grid_best_under_harm.csv", frontier_rows, frontier_fields)
    write_readme(args.out_dir / "README.md", best_rows, frontier_rows, c_grid)

    output_files = [
        "README.md",
        "c_grid_setting_summary.csv",
        "c_grid_dataset_summary.csv",
        "c_grid_benchmark_summary.csv",
        "c_grid_best_by_expert.csv",
        "c_grid_best_under_harm.csv",
    ]
    with (args.out_dir / "SHA256SUMS").open("w") as handle:
        for name in output_files:
            handle.write(f"{sha256_file(args.out_dir / name)}  {name}\n")
    verification = {
        "status": "PASS",
        "draws": args.draws,
        "seed": args.seed,
        "c_grid": c_grid,
        "ks_run_dir": str(args.ks_run_dir),
        "settings": expected,
        "setting_rows": len(setting_rows),
        "benchmark_rows": len(benchmark_rows),
        "source_rows": sum(len(rows) for rows in raw_groups.values()),
        "output_sha256": {
            name: sha256_file(args.out_dir / name)
            for name in [*output_files, "SHA256SUMS"]
        },
    }
    (args.out_dir / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
