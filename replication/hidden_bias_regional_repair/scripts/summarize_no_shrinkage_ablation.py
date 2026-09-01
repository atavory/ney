#!/usr/bin/env python3
"""Summarize the Section 4 no-shrinkage ablation from raw replication rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from recreate_unified_cartesian_global_residual import (
    CONFIG_KEYS,
    EXPECTED_FULL_MANIFEST_SHA256,
    as_float,
    mean,
    percentile,
    read_provenance,
    scientific_manifest,
)


METHODS = ("aipw", "tmle", "ctmle", "cui_selective_ml", "ma_dr_bc")
PRIMARY_METHODS = ("aipw", "cui_selective_ml", "ma_dr_bc", "ctmle")
GROUPS = ("kang_schafer", "alignment", "real", "anchor")
SUMMARY_FAMILIES = (
    ("primary", ("kang_schafer", "real")),
    ("all", GROUPS),
    ("kang_schafer", ("kang_schafer",)),
    ("alignment", ("alignment",)),
    ("real", ("real",)),
    ("anchor", ("anchor",)),
)
METHOD_LABELS = {
    "aipw": "AIPW",
    "tmle": "fixed-floor TMLE",
    "ctmle": "C-TMLE",
    "cui_selective_ml": "selective ML",
    "ma_dr_bc": "Ma DR-BC",
}
BENCHMARK_DATASETS = (
    (
        "ihdp",
        "IHDP",
        "support_csv/dml_real_benchmark_expansion_20260831/raw_rows.csv",
        (("ihdp_semisynth", "A"), ("ihdp_misaligned", "B")),
    ),
    (
        "acic2016",
        "ACIC 2016",
        "support_csv/dml_real_benchmark_expansion_20260831/raw_rows.csv",
        (("acic2016_semisynth", "A"), ("acic2016_misaligned", "B")),
    ),
    (
        "acic2017",
        "ACIC 2017",
        "support_csv/dml_real_benchmark_acic2017_20260831/raw_rows.csv",
        (("acic2017_semisynth", "A"), ("acic2017_misaligned", "B")),
    ),
    (
        "twins",
        "Twins",
        "support_csv/dml_real_benchmark_twins_20260831/raw_rows.csv",
        (("twins_semisynth", "A"), ("twins_misaligned", "B")),
    ),
)
DATA_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        type=Path,
        help="Extracted run directory containing *.reps.csv and provenance.json.",
    )
    parser.add_argument("--full-manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gain(reference_sq: list[float], target_sq: list[float]) -> float:
    reference_mse = mean(reference_sq)
    if reference_mse == 0.0:
        raise SystemExit("zero reference MSE encountered")
    return 1.0 - mean(target_sq) / reference_mse


def paired_draws(
    reference_sq: list[float],
    no_shrink_sq: list[float],
    c2_sq: list[float],
    rng: random.Random,
    count: int,
) -> list[tuple[float, float, float]]:
    draws = []
    n = len(reference_sq)
    for _ in range(count):
        ref_total = 0.0
        no_total = 0.0
        c2_total = 0.0
        for _ in range(n):
            index = rng.randrange(n)
            ref_total += reference_sq[index]
            no_total += no_shrink_sq[index]
            c2_total += c2_sq[index]
        no_gain = 1.0 - no_total / ref_total
        c2_gain = 1.0 - c2_total / ref_total
        draws.append((no_gain, c2_gain, c2_gain - no_gain))
    return draws


def interval(draws: list[tuple[float, float, float]], index: int) -> tuple[float, float]:
    values = sorted(draw[index] for draw in draws)
    return percentile(values, 2.5), percentile(values, 97.5)


def summarize_cell(
    rows: list[dict[str, str]],
    rng: random.Random,
    draws: int,
) -> tuple[dict[str, object], list[tuple[float, float, float]]]:
    reference_sq = [as_float(row["ref_error"]) ** 2 for row in rows]
    no_shrink_sq = [as_float(row["rt_error"]) ** 2 for row in rows]
    c2_sq = [as_float(row["shrink_error"]) ** 2 for row in rows]
    weights = [as_float(row["weight"]) for row in rows]
    selected = [as_float(row["selected_region_damp"]) for row in rows]
    cell_draws = paired_draws(reference_sq, no_shrink_sq, c2_sq, rng, draws)
    no_lo, no_hi = interval(cell_draws, 0)
    c2_lo, c2_hi = interval(cell_draws, 1)
    diff_lo, diff_hi = interval(cell_draws, 2)
    no_gain = gain(reference_sq, no_shrink_sq)
    c2_gain = gain(reference_sq, c2_sq)
    return (
        {
            "reps": len(rows),
            "reference_mse": mean(reference_sq),
            "no_shrinkage_mse": mean(no_shrink_sq),
            "c2_shrinkage_mse": mean(c2_sq),
            "no_shrinkage_gain": no_gain,
            "no_shrinkage_gain_ci_low": no_lo,
            "no_shrinkage_gain_ci_high": no_hi,
            "c2_shrinkage_gain": c2_gain,
            "c2_shrinkage_gain_ci_low": c2_lo,
            "c2_shrinkage_gain_ci_high": c2_hi,
            "c2_minus_no_shrinkage_gain": c2_gain - no_gain,
            "c2_minus_no_shrinkage_gain_ci_low": diff_lo,
            "c2_minus_no_shrinkage_gain_ci_high": diff_hi,
            "path_activation": mean([1.0 if value > 0.0 else 0.0 for value in selected]),
            "final_activation": mean([1.0 if value > 0.0 else 0.0 for value in weights]),
            "no_shrinkage_harm": mean(
                [
                    1.0 if target > reference else 0.0
                    for reference, target in zip(reference_sq, no_shrink_sq)
                ]
            ),
            "c2_shrinkage_harm": mean(
                [
                    1.0 if target > reference else 0.0
                    for reference, target in zip(reference_sq, c2_sq)
                ]
            ),
            "mean_shrink_weight": mean(weights),
        },
        cell_draws,
    )


def format_optional(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_optional(row.get(key, "")) for key in fieldnames})


def group_draws(
    cell_rows: list[dict[str, object]],
    cell_draws: dict[tuple[str, str, str, int, float], list[tuple[float, float, float]]],
    draws: int,
) -> list[tuple[float, float, float]]:
    draw_lists = [
        cell_draws[
            (
                str(row["group"]),
                str(row["design"]),
                str(row["method"]),
                int(row["n"]),
                float(row["strength"]),
            )
        ]
        for row in cell_rows
    ]
    return [
        (
            mean([draw[index][0] for draw in draw_lists]),
            mean([draw[index][1] for draw in draw_lists]),
            mean([draw[index][2] for draw in draw_lists]),
        )
        for index in range(draws)
    ]


def benchmark_group_draws(
    cell_rows: list[dict[str, object]],
    cell_draws: dict[tuple[str, str, str, int, float], list[tuple[float, float, float]]],
    draws: int,
) -> list[tuple[float, float, float]]:
    draw_lists = [
        cell_draws[
            (
                str(row["group"]),
                str(row["design"]),
                str(row["method"]),
                int(row["n"]),
                float(row["strength"]),
            )
        ]
        for row in cell_rows
    ]
    return [
        (
            mean([draw[index][0] for draw in draw_lists]),
            mean([draw[index][1] for draw in draw_lists]),
            mean([draw[index][2] for draw in draw_lists]),
        )
        for index in range(draws)
    ]


def summarize_group(
    method: str,
    family: str,
    rows: list[dict[str, object]],
    draws: list[tuple[float, float, float]],
) -> dict[str, object]:
    no_lo, no_hi = interval(draws, 0)
    c2_lo, c2_hi = interval(draws, 1)
    diff_lo, diff_hi = interval(draws, 2)
    no_gain = mean([float(row["no_shrinkage_gain"]) for row in rows])
    c2_gain = mean([float(row["c2_shrinkage_gain"]) for row in rows])
    return {
        "method": method,
        "family": family,
        "cells": len(rows),
        "reps": sum(int(row["reps"]) for row in rows),
        "no_shrinkage_gain": no_gain,
        "no_shrinkage_gain_ci_low": no_lo,
        "no_shrinkage_gain_ci_high": no_hi,
        "c2_shrinkage_gain": c2_gain,
        "c2_shrinkage_gain_ci_low": c2_lo,
        "c2_shrinkage_gain_ci_high": c2_hi,
        "c2_minus_no_shrinkage_gain": c2_gain - no_gain,
        "c2_minus_no_shrinkage_gain_ci_low": diff_lo,
        "c2_minus_no_shrinkage_gain_ci_high": diff_hi,
        "path_activation": mean([float(row["path_activation"]) for row in rows]),
        "final_activation": mean([float(row["final_activation"]) for row in rows]),
        "no_shrinkage_harm": mean([float(row["no_shrinkage_harm"]) for row in rows]),
        "c2_shrinkage_harm": mean([float(row["c2_shrinkage_harm"]) for row in rows]),
        "mean_shrink_weight": mean([float(row["mean_shrink_weight"]) for row in rows]),
    }


def read_benchmark_cell_rows(
    rng: random.Random,
    draws: int,
) -> tuple[
    list[dict[str, object]],
    dict[tuple[str, str, str, int, float], list[tuple[float, float, float]]],
]:
    raw_groups: dict[tuple[str, str, str, int, float], list[dict[str, str]]] = defaultdict(list)
    for dataset_key, _label, raw_path, design_pairs in BENCHMARK_DATASETS:
        design_labels = dict(design_pairs)
        with (DATA_ROOT / raw_path).open(newline="") as handle:
            for row in csv.DictReader(handle):
                method = row["reference_method"]
                design = row["design"]
                if method not in PRIMARY_METHODS or design not in design_labels:
                    continue
                key = (
                    dataset_key,
                    design_labels[design],
                    method,
                    int(row["n"]),
                    float(row["strength"]),
                )
                raw_groups[key].append(row)

    expected = len(BENCHMARK_DATASETS) * 2 * 2 * len(PRIMARY_METHODS)
    if len(raw_groups) != expected:
        raise SystemExit(
            f"benchmark no-shrinkage groups differ: {len(raw_groups)} != {expected}"
        )

    cell_rows: list[dict[str, object]] = []
    cell_draw_lookup: dict[tuple[str, str, str, int, float], list[tuple[float, float, float]]] = {}
    for key in sorted(raw_groups):
        group, design, method, n, strength = key
        row, cell_draws = summarize_cell(raw_groups[key], rng, draws)
        cell_draw_lookup[key] = cell_draws
        cell_rows.append(
            {
                "group": group,
                "design": design,
                "method": method,
                "n": n,
                "strength": strength,
                **row,
            }
        )
    return cell_rows, cell_draw_lookup


def summarize_benchmark(
    old_cell_rows: list[dict[str, object]],
    old_cell_draw_lookup: dict[
        tuple[str, str, str, int, float], list[tuple[float, float, float]]
    ],
    benchmark_cell_rows: list[dict[str, object]],
    benchmark_cell_draw_lookup: dict[
        tuple[str, str, str, int, float], list[tuple[float, float, float]]
    ],
    draws: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in PRIMARY_METHODS:
        ks_rows = [
            row
            for row in old_cell_rows
            if row["group"] == "kang_schafer" and row["method"] == method
        ]
        extra_rows = [row for row in benchmark_cell_rows if row["method"] == method]
        method_rows = [*ks_rows, *extra_rows]
        if len(method_rows) != 24:
            raise SystemExit(
                f"benchmark no-shrinkage row count for {method}: "
                f"{len(method_rows)} != 24"
            )
        draw_lookup = {**old_cell_draw_lookup, **benchmark_cell_draw_lookup}
        rows.append(
            summarize_group(
                method,
                "benchmark",
                method_rows,
                benchmark_group_draws(method_rows, draw_lookup, draws),
            )
        )
    return rows


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.3f}"


def table_value(row: dict[str, object], prefix: str) -> str:
    return (
        f"{fmt_pct(float(row[f'{prefix}_gain']))} "
        f"[{fmt_pct(float(row[f'{prefix}_gain_ci_low']))}, "
        f"{fmt_pct(float(row[f'{prefix}_gain_ci_high']))}]"
    )


def write_latex_table(path: Path, benchmark_rows: list[dict[str, object]]) -> None:
    primary_rows = {
        str(row["method"]): row
        for row in benchmark_rows
        if row["family"] == "benchmark"
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Final shrinkage ablation on the 24 benchmark settings in Table~\ref{tab:unified-global-residual-families}.  We compare the selected, unshrunk candidate with the final \(c=2\) plug-in contrast-shrinkage rule used in the primary run. Values are equal-setting percent MSE gain with paired percentile intervals.}",
        r"\label{tab:no-shrinkage-ablation}",
        r"\begin{tabular}{@{}lrr@{}}",
        r"\toprule",
        r"expert & no shrinkage & \(c=2\) shrinkage \\",
        r"\midrule",
    ]
    for method in PRIMARY_METHODS:
        row = primary_rows[method]
        lines.append(
            f"{METHOD_LABELS[method]} & "
            f"{table_value(row, 'no_shrinkage')} & "
            f"{table_value(row, 'c2_shrinkage')} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows, manifest_sha = scientific_manifest(args.full_manifest)
    if manifest_sha != EXPECTED_FULL_MANIFEST_SHA256:
        raise SystemExit(
            "full manifest SHA mismatch: "
            f"observed {manifest_sha}, expected {EXPECTED_FULL_MANIFEST_SHA256}"
        )
    expected_cells = {
        (
            row["group"],
            row["design"],
            row["method"],
            int(row["n"]),
            float(row["strength"]),
        )
        for row in manifest_rows
    }
    if len(manifest_rows) != 4080 or len(expected_cells) != 170:
        raise SystemExit("unexpected manifest shape")
    cell_lookup = {
        (design, method, n, strength): group
        for group, design, method, n, strength in expected_cells
    }

    provenance = read_provenance(args.run_dir)
    grouped: dict[tuple[str, str, str, int, float], list[dict[str, str]]] = defaultdict(list)
    reps_files: list[Path] = []
    rows_seen = 0
    for run_dir in args.run_dir:
        for path in sorted(run_dir.glob("*.reps.csv")):
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 4:
                raise SystemExit(f"{path}: expected 4 replication rows, observed {len(rows)}")
            reps_files.append(path)
            for row in rows:
                key_without_group = (
                    row["design"],
                    row["reference_method"],
                    int(row["n"]),
                    float(row["strength"]),
                )
                if key_without_group not in cell_lookup:
                    raise SystemExit(f"row does not belong to the full manifest: {key_without_group}")
                if row["repair_mode"] != "if_residual":
                    raise SystemExit(f"{path}: row repair_mode is not if_residual")
                grouped[(cell_lookup[key_without_group], *key_without_group)].append(row)
                rows_seen += 1
    if len(reps_files) != 4080 or rows_seen != 16320:
        raise SystemExit(
            f"unexpected raw shape: shard_files={len(reps_files)} rows={rows_seen}"
        )
    bad_cells = {key: len(rows) for key, rows in grouped.items() if len(rows) != 96}
    if bad_cells:
        raise SystemExit(f"incomplete cells: {bad_cells}")

    rng = random.Random(args.seed)
    cell_rows: list[dict[str, object]] = []
    cell_draw_lookup: dict[tuple[str, str, str, int, float], list[tuple[float, float, float]]] = {}
    for key in sorted(grouped):
        group, design, method, n, strength = key
        row, draws = summarize_cell(grouped[key], rng, args.draws)
        cell_draw_lookup[key] = draws
        cell_rows.append(
            {
                "group": group,
                "design": design,
                "method": method,
                "n": n,
                "strength": strength,
                **row,
            }
        )

    family_rows: list[dict[str, object]] = []
    for method in METHODS:
        method_cell_rows = [row for row in cell_rows if row["method"] == method]
        for family, groups in SUMMARY_FAMILIES:
            family_cell_rows = [
                row
                for row in method_cell_rows
                if row["group"] in groups
            ]
            family_rows.append(
                summarize_group(
                    method,
                    family,
                    family_cell_rows,
                    group_draws(family_cell_rows, cell_draw_lookup, args.draws),
                )
            )

    benchmark_cell_rows, benchmark_cell_draw_lookup = read_benchmark_cell_rows(
        rng, args.draws
    )
    benchmark_rows = summarize_benchmark(
        cell_rows,
        cell_draw_lookup,
        benchmark_cell_rows,
        benchmark_cell_draw_lookup,
        args.draws,
    )

    fields = [
        "method",
        "family",
        "cells",
        "reps",
        "no_shrinkage_gain",
        "no_shrinkage_gain_ci_low",
        "no_shrinkage_gain_ci_high",
        "c2_shrinkage_gain",
        "c2_shrinkage_gain_ci_low",
        "c2_shrinkage_gain_ci_high",
        "c2_minus_no_shrinkage_gain",
        "c2_minus_no_shrinkage_gain_ci_low",
        "c2_minus_no_shrinkage_gain_ci_high",
        "path_activation",
        "final_activation",
        "no_shrinkage_harm",
        "c2_shrinkage_harm",
        "mean_shrink_weight",
    ]
    write_csv(args.out_dir / "no_shrinkage_family_summary.csv", family_rows, fields)
    write_csv(args.out_dir / "no_shrinkage_benchmark_summary.csv", benchmark_rows, fields)
    write_csv(
        args.out_dir / "no_shrinkage_cell_summary.csv",
        cell_rows,
        [
            "group",
            "design",
            "method",
            "n",
            "strength",
            "reps",
            "reference_mse",
            "no_shrinkage_mse",
            "c2_shrinkage_mse",
            *fields[4:],
        ],
    )
    write_latex_table(
        args.out_dir / "section4_no_shrinkage_ablation_table.tex",
        benchmark_rows,
    )
    verification = {
        "status": "PASS",
        "draws": args.draws,
        "seed": args.seed,
        "full_manifest": args.full_manifest.name,
        "full_manifest_sha256": manifest_sha,
        "run_dirs": [path.name for path in args.run_dir],
        "run_provenance": provenance,
        "reps_files": len(reps_files),
        "replication_rows": rows_seen,
        "cells": len(cell_rows),
        "benchmark_cells": 24 * len(PRIMARY_METHODS),
        "config": {key: provenance[0][key] for key in CONFIG_KEYS},
    }
    (args.out_dir / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
