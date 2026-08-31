#!/usr/bin/env python3
"""Summarize the low-response versus high-response repair-region ablation."""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path


STRENGTHS = ("0", "3", "5", "8")
REGIONS = ("low_response", "high_response_placebo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def as_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return math.nan


def strength_label(value: str) -> str:
    return str(int(float(value)))


def gain(rows: list[dict[str, str]]) -> float:
    reference = sum(as_float(row["ref_error"]) ** 2 for row in rows)
    repaired = sum(as_float(row["shrink_error"]) ** 2 for row in rows)
    if reference <= 0.0:
        return math.nan
    return 1.0 - repaired / reference


def mean(rows: list[dict[str, str]], column: str) -> float:
    values = [as_float(row[column]) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return sum(values) / len(values) if values else math.nan


def metric_row(rows: list[dict[str, str]]) -> dict[str, float | int]:
    reference_sq = [as_float(row["ref_error"]) ** 2 for row in rows]
    proposal_sq = [as_float(row["rt_error"]) ** 2 for row in rows]
    repaired_sq = [as_float(row["shrink_error"]) ** 2 for row in rows]
    reference = sum(reference_sq)
    proposal = sum(proposal_sq)
    repaired = sum(repaired_sq)
    return {
        "reps": len(rows),
        "reference_mse": reference / len(rows),
        "proposal_mse": proposal / len(rows),
        "repaired_mse": repaired / len(rows),
        "proposal_gain": 1.0 - proposal / reference,
        "repaired_gain": 1.0 - repaired / reference,
        "activation": sum(as_float(row["weight"]) > 0.0 for row in rows) / len(rows),
        "path_activation": (
            sum(as_float(row["selected_region_damp"]) > 0.0 for row in rows)
            / len(rows)
        ),
        "harm": sum(
            repaired_value > reference_value
            for reference_value, repaired_value in zip(reference_sq, repaired_sq)
        )
        / len(rows),
        "mean_response_rate": mean(rows, "analysis_response_rate"),
        "mean_region_mass": mean(rows, "analysis_region_mass"),
        "mean_overlap_precision": mean(rows, "region_overlap_precision"),
        "mean_overlap_recall": mean(rows, "region_overlap_recall"),
        "mean_true_region_abs_bias_reduction": mean(
            rows, "true_region_abs_bias_reduction"
        ),
        "mean_analysis_abs_bias_reduction": mean(
            rows, "analysis_abs_bias_reduction"
        ),
    }


def percentile(sorted_values: list[float], pct: float) -> float:
    position = (len(sorted_values) - 1) * pct / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def gain_interval(
    rows: list[dict[str, str]], rng: random.Random, draws: int
) -> tuple[float, float]:
    values = []
    n = len(rows)
    for _ in range(draws):
        values.append(gain([rows[rng.randrange(n)] for _ in range(n)]))
    values.sort()
    return percentile(values, 2.5), percentile(values, 97.5)


def paired_rows(
    rows: list[dict[str, str]], strengths: set[str]
) -> list[tuple[dict[str, str], dict[str, str]]]:
    low = {}
    high = {}
    for row in rows:
        if strength_label(row["strength"]) not in strengths:
            continue
        target = low if row["repair_region_label"] == "low_response" else high
        target[row["paired_id"]] = row
    return [(low[key], high[key]) for key in sorted(set(low) & set(high))]


def paired_difference_interval(
    pairs: list[tuple[dict[str, str], dict[str, str]]],
    rng: random.Random,
    draws: int,
) -> tuple[float, float, float]:
    low = [pair[0] for pair in pairs]
    high = [pair[1] for pair in pairs]
    observed = gain(low) - gain(high)
    values = []
    n = len(pairs)
    for _ in range(draws):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        values.append(gain([pair[0] for pair in sample]) - gain([pair[1] for pair in sample]))
    values.sort()
    return observed, percentile(values, 2.5), percentile(values, 97.5)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.3f}"


def table_value(row: dict[str, object]) -> str:
    return (
        f"{fmt_pct(float(row['repaired_gain']))} "
        f"[{fmt_pct(float(row['repaired_gain_ci_low']))}, "
        f"{fmt_pct(float(row['repaired_gain_ci_high']))}]"
    )


def paired_table_value(row: dict[str, object]) -> str:
    return (
        f"{fmt_pct(float(row['paired_low_minus_high_gain']))} "
        f"[{fmt_pct(float(row['paired_low_minus_high_gain_ci_low']))}, "
        f"{fmt_pct(float(row['paired_low_minus_high_gain_ci_high']))}]"
    )


def write_latex_table(
    path: Path,
    by_strength: list[dict[str, object]],
    overall: list[dict[str, object]],
) -> None:
    rows_by_key = {
        (str(row.get("group", row.get("strength"))), row["repair_region"]): row
        for row in [*by_strength, *overall]
    }
    signal_low = rows_by_key[("signal_strengths", "low_response")]
    signal_high = rows_by_key[("signal_strengths", "high_response_placebo")]
    null_low = rows_by_key[("0", "low_response")]
    null_high = rows_by_key[("0", "high_response_placebo")]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Repair-region placement control.  We hold the C-TMLE reference, data law, \(\Gamma=\{0,0.25,0.5,1\}\) in \eqref{eq:global-residual-candidates}, one-standard-error gate, and \(c=2\) shrinkage fixed while moving the analysis region from the low-response box to a disjoint high-response placebo. Values are percent MSE gain with paired percentile intervals.}",
        r"\label{tab:low-high-response-ablation}",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"strength set & low-response repair & high-response placebo & low minus high \\",
        r"\midrule",
        (
            "signal strengths 3, 5, 8 & "
            f"{table_value(signal_low)} & "
            f"{table_value(signal_high)} & "
            f"{paired_table_value(signal_low)} \\\\"
        ),
        (
            "null strength 0 & "
            f"{table_value(null_low)} & "
            f"{table_value(null_high)} & "
            f"{paired_table_value(null_low)} \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with args.replications.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 768:
        raise SystemExit(f"expected 768 replications, found {len(rows)}")

    rng = random.Random(args.seed)
    by_group: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        by_group.setdefault(
            (strength_label(row["strength"]), row["repair_region_label"]), []
        ).append(row)

    summary = []
    for strength in STRENGTHS:
        diff, diff_lo, diff_hi = paired_difference_interval(
            paired_rows(rows, {strength}), rng, args.draws
        )
        for region in REGIONS:
            group = by_group[(strength, region)]
            lo, hi = gain_interval(group, rng, args.draws)
            summary.append(
                {
                    "strength": strength,
                    "repair_region": region,
                    **metric_row(group),
                    "repaired_gain_ci_low": lo,
                    "repaired_gain_ci_high": hi,
                    "paired_low_minus_high_gain": diff,
                    "paired_low_minus_high_gain_ci_low": diff_lo,
                    "paired_low_minus_high_gain_ci_high": diff_hi,
                }
            )

    fields = [
        "strength",
        "repair_region",
        "reps",
        "reference_mse",
        "proposal_mse",
        "repaired_mse",
        "proposal_gain",
        "repaired_gain",
        "repaired_gain_ci_low",
        "repaired_gain_ci_high",
        "paired_low_minus_high_gain",
        "paired_low_minus_high_gain_ci_low",
        "paired_low_minus_high_gain_ci_high",
        "activation",
        "path_activation",
        "harm",
        "mean_response_rate",
        "mean_region_mass",
        "mean_overlap_precision",
        "mean_overlap_recall",
        "mean_true_region_abs_bias_reduction",
        "mean_analysis_abs_bias_reduction",
    ]
    write_csv(args.out_dir / "summary_by_strength.csv", summary, fields)

    overall = []
    for label, strengths in [
        ("all_strengths", {"0", "3", "5", "8"}),
        ("signal_strengths", {"3", "5", "8"}),
    ]:
        diff, diff_lo, diff_hi = paired_difference_interval(
            paired_rows(rows, strengths), rng, args.draws
        )
        for region in REGIONS:
            group = [
                row
                for row in rows
                if strength_label(row["strength"]) in strengths
                and row["repair_region_label"] == region
            ]
            lo, hi = gain_interval(group, rng, args.draws)
            overall.append(
                {
                    "group": label,
                    "repair_region": region,
                    **metric_row(group),
                    "repaired_gain_ci_low": lo,
                    "repaired_gain_ci_high": hi,
                    "paired_low_minus_high_gain": diff,
                    "paired_low_minus_high_gain_ci_low": diff_lo,
                    "paired_low_minus_high_gain_ci_high": diff_hi,
                }
            )
    write_csv(
        args.out_dir / "summary_overall.csv",
        overall,
        ["group", "repair_region", *fields[2:]],
    )
    write_latex_table(
        args.out_dir / "section4_low_high_response_ablation_table.tex",
        summary,
        overall,
    )


if __name__ == "__main__":
    main()
