#!/usr/bin/env python3
"""Audit same-sample versus fixed-candidate risk with explicit aggregation.

The paper's estimand gives every benchmark setting equal weight.  This script
therefore computes each setting-by-method gain first and averages those gains.
It also emits the pooled ratio-of-means as a separately labelled sensitivity
calculation so that the two estimands cannot be confused.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


PRIMARY_METHODS = ("aipw", "ctmle", "cui_selective_ml", "ma_dr_bc")
CELL_KEYS = ("dataset", "setting", "method")
EXPECTED_SETTINGS = 24
EXPECTED_REPS_PER_CELL = 96
CLUSTER_BOOTSTRAP_DRAWS = 20_000
CLUSTER_BOOTSTRAP_SEED = 20260909


def _float(row: dict[str, object], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}: {row[key]}")
    return value


def _read_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    identities: set[tuple[object, ...]] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "source", "dataset", "setting", "method", "n", "strength",
                "seed0", "rep", "selected_gamma", "ref_error",
                "selected_error", "bias0_sq", "biasg_sq", "A0", "Ag",
            }
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}")
            for row in reader:
                ref_error_raw = _float(row, "ref_error")
                selected_error_raw = _float(row, "selected_error")
                ref_error = ref_error_raw * ref_error_raw
                selected_error = selected_error_raw * selected_error_raw
                ref_fresh_risk = _float(row, "bias0_sq") + _float(row, "A0")
                selected_fresh_risk = _float(row, "biasg_sq") + _float(row, "Ag")
                parsed = {
                        "source_file": str(path),
                        "source": row["source"],
                        "dataset": row["dataset"],
                        "setting": row["setting"],
                        "method": row["method"],
                        "n": int(row["n"]),
                        "strength": row["strength"],
                        "seed0": int(row["seed0"]),
                        "rep": int(row["rep"]),
                        "selected_gamma": _float(row, "selected_gamma"),
                        "ref_error_raw": ref_error_raw,
                        "selected_error_raw": selected_error_raw,
                        "ref_sq_error": ref_error,
                        "selected_sq_error": selected_error,
                        "ref_fresh_risk": ref_fresh_risk,
                        "selected_fresh_risk": selected_fresh_risk,
                        "same_sample_delta": selected_error - ref_error,
                        "fresh_risk_delta": selected_fresh_risk - ref_fresh_risk,
                        "selection_penalty_delta": (selected_error - ref_error)
                        - (selected_fresh_risk - ref_fresh_risk),
                        "active": _float(row, "selected_gamma") != 0.0,
                    }
                identity = tuple(
                    parsed[key]
                    for key in (
                        "source", "dataset", "setting", "method", "n",
                        "strength", "seed0", "rep",
                    )
                )
                if identity in identities:
                    raise ValueError(f"duplicate replication identity: {identity}")
                identities.add(identity)
                rows.append(parsed)
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[int(q * (len(ordered) - 1))]


def _linear_quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return (
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def _summarize_cell(group: list[dict[str, object]]) -> dict[str, object]:
    ref_error = [_float(row, "ref_sq_error") for row in group]  # type: ignore[arg-type]
    selected_error = [_float(row, "selected_sq_error") for row in group]  # type: ignore[arg-type]
    ref_fresh = [_float(row, "ref_fresh_risk") for row in group]  # type: ignore[arg-type]
    selected_fresh = [_float(row, "selected_fresh_risk") for row in group]  # type: ignore[arg-type]
    penalties = [_float(row, "selection_penalty_delta") for row in group]  # type: ignore[arg-type]
    active = [row for row in group if bool(row["active"])]
    same_gain = 100.0 * (_mean(ref_error) - _mean(selected_error)) / _mean(ref_error)
    fresh_gain = 100.0 * (_mean(ref_fresh) - _mean(selected_fresh)) / _mean(ref_fresh)
    return {
        "rows": len(group),
        "active_rows": len(active),
        "active_same_sample_harm_rows": sum(
            1
            for row in active
            if _float(row, "selected_sq_error") > _float(row, "ref_sq_error")
        ),
        "active_fresh_risk_harm_rows": sum(
            1
            for row in active
            if _float(row, "selected_fresh_risk") > _float(row, "ref_fresh_risk")
        ),
        "active_share": len(active) / len(group),
        "same_sample_mse_gain_pct": same_gain,
        "fresh_risk_gain_pct": fresh_gain,
        "gain_gap_same_minus_fresh_pct": same_gain - fresh_gain,
        "mean_selection_penalty_delta": _mean(penalties),
        "median_selection_penalty_delta": _quantile(penalties, 0.50),
        "p90_abs_selection_penalty_delta": _quantile([abs(x) for x in penalties], 0.90),
        "same_sample_harm_share": sum(1 for row in group if _float(row, "selected_sq_error") > _float(row, "ref_sq_error")) / len(group),  # type: ignore[arg-type]
        "fresh_risk_harm_share": sum(1 for row in group if _float(row, "selected_fresh_risk") > _float(row, "ref_fresh_risk")) / len(group),  # type: ignore[arg-type]
        "active_same_sample_harm_share": (
            sum(1 for row in active if _float(row, "selected_sq_error") > _float(row, "ref_sq_error")) / len(active)  # type: ignore[arg-type]
            if active
            else ""
        ),
        "active_fresh_risk_harm_share": (
            sum(1 for row in active if _float(row, "selected_fresh_risk") > _float(row, "ref_fresh_risk")) / len(active)  # type: ignore[arg-type]
            if active
            else ""
        ),
    }


def _mean_summaries(summaries: list[dict[str, object]]) -> dict[str, object]:
    """Average setting-level estimands, giving every cell exactly one vote."""
    if not summaries:
        raise ValueError("cannot aggregate an empty list of setting summaries")
    numeric_fields = (
        "active_share",
        "same_sample_mse_gain_pct",
        "fresh_risk_gain_pct",
        "gain_gap_same_minus_fresh_pct",
        "same_sample_harm_share",
        "fresh_risk_harm_share",
    )
    result: dict[str, object] = {
        "aggregation": "equal_setting_method_cells",
        "cells": len(summaries),
        "rows": sum(int(row["rows"]) for row in summaries),
        "active_rows": sum(int(row["active_rows"]) for row in summaries),
        "active_same_sample_harm_rows": sum(
            int(row["active_same_sample_harm_rows"]) for row in summaries
        ),
        "active_fresh_risk_harm_rows": sum(
            int(row["active_fresh_risk_harm_rows"]) for row in summaries
        ),
    }
    for field in numeric_fields:
        result[field] = _mean([_float(row, field) for row in summaries])

    for field in ("active_same_sample_harm_share", "active_fresh_risk_harm_share"):
        values = [float(row[field]) for row in summaries if row[field] != ""]
        result[field] = _mean(values) if values else ""
    gaps = [_float(row, "gain_gap_same_minus_fresh_pct") for row in summaries]
    result["mean_abs_gain_gap_pct"] = _mean([abs(value) for value in gaps])
    result["min_gain_gap_pct"] = min(gaps)
    result["max_gain_gap_pct"] = max(gaps)
    return result


def _cluster_bootstrap_gap(
    cell_summaries: list[dict[str, object]], seed: int
) -> dict[str, float | int]:
    """Bootstrap the mean gain gap by resampling the 24 settings as clusters."""
    grouped: dict[tuple[object, object], list[float]] = defaultdict(list)
    for row in cell_summaries:
        grouped[(row["dataset"], row["setting"])].append(
            _float(row, "gain_gap_same_minus_fresh_pct")
        )
    cluster_values = [
        _mean(values) for _key, values in sorted(grouped.items())
    ]
    if len(cluster_values) != EXPECTED_SETTINGS:
        raise ValueError(
            f"expected {EXPECTED_SETTINGS} bootstrap clusters, got {len(cluster_values)}"
        )
    rng = random.Random(seed)
    draws = [
        _mean(
            [
                cluster_values[rng.randrange(len(cluster_values))]
                for _ in cluster_values
            ]
        )
        for _ in range(CLUSTER_BOOTSTRAP_DRAWS)
    ]
    return {
        "gain_gap_ci_low": _linear_quantile(draws, 0.025),
        "gain_gap_ci_high": _linear_quantile(draws, 0.975),
        "gain_gap_bootstrap_se": statistics.stdev(draws),
        "gain_gap_bootstrap_clusters": len(cluster_values),
        "gain_gap_bootstrap_draws": CLUSTER_BOOTSTRAP_DRAWS,
        "gain_gap_bootstrap_seed": seed,
    }


def _group_rows(
    rows: list[dict[str, object]], keys: tuple[str, ...]
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out = []
    for key_values, group in sorted(groups.items()):
        summary = {key: value for key, value in zip(keys, key_values)}
        summary.update(_summarize_cell(group))
        out.append(summary)
    return out


def _equal_cell_summary(
    cell_rows: list[dict[str, object]],
    keys: tuple[str, ...],
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in cell_rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out = []
    for key_values, group in sorted(groups.items()):
        summary = {key: value for key, value in zip(keys, key_values)}
        summary.update(_mean_summaries(group))
        out.append(summary)
    return out


def _validate_coverage(
    primary_rows: list[dict[str, object]],
    tmle_rows: list[dict[str, object]],
) -> dict[str, object]:
    primary_by_method = {
        method: {
            (row["dataset"], row["setting"])
            for row in primary_rows
            if row["method"] == method
        }
        for method in PRIMARY_METHODS
    }
    reference_settings = primary_by_method[PRIMARY_METHODS[0]]
    if len(reference_settings) != EXPECTED_SETTINGS:
        raise ValueError(
            f"expected {EXPECTED_SETTINGS} primary settings, got {len(reference_settings)}"
        )
    for method, settings in primary_by_method.items():
        if settings != reference_settings:
            raise ValueError(f"primary setting coverage differs for {method}")
    tmle_settings = {(row["dataset"], row["setting"]) for row in tmle_rows}
    if tmle_settings != reference_settings:
        raise ValueError("fixed-floor TMLE setting coverage differs from primary coverage")

    all_rows = primary_rows + tmle_rows
    counts: dict[tuple[object, ...], int] = defaultdict(int)
    for row in all_rows:
        counts[tuple(row[key] for key in CELL_KEYS)] += 1
    unexpected = {key: count for key, count in counts.items() if count != EXPECTED_REPS_PER_CELL}
    if unexpected:
        raise ValueError(
            f"expected {EXPECTED_REPS_PER_CELL} replications per setting-method cell; "
            f"found {unexpected}"
        )
    return {
        "settings": len(reference_settings),
        "primary_methods": list(PRIMARY_METHODS),
        "primary_cells": len(primary_by_method) * len(reference_settings),
        "fixed_floor_tmle_cells": len(tmle_settings),
        "replications_per_cell": EXPECTED_REPS_PER_CELL,
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
    parser.add_argument("--primary-rows", type=Path, required=True)
    parser.add_argument("--tmle-rows", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=False)
    rows = _read_rows([args.primary_rows, args.tmle_rows])
    primary_rows = [row for row in rows if row["method"] in PRIMARY_METHODS]
    tmle_rows = [row for row in rows if row["method"] == "tmle"]
    if len(primary_rows) + len(tmle_rows) != len(rows):
        unexpected = sorted(
            {str(row["method"]) for row in rows}
            - set(PRIMARY_METHODS)
            - {"tmle"}
        )
        raise ValueError(f"unexpected methods: {unexpected}")
    coverage = _validate_coverage(primary_rows, tmle_rows)

    setting_method_rows = _group_rows(rows, CELL_KEYS)
    primary_cells = [row for row in setting_method_rows if row["method"] in PRIMARY_METHODS]
    tmle_cells = [row for row in setting_method_rows if row["method"] == "tmle"]

    equal_overall = [
        {"scope": "primary", **_mean_summaries(primary_cells)},
        {"scope": "fixed_floor_tmle", **_mean_summaries(tmle_cells)},
    ]
    equal_overall[0].update(
        _cluster_bootstrap_gap(primary_cells, CLUSTER_BOOTSTRAP_SEED)
    )
    equal_overall[1].update(
        _cluster_bootstrap_gap(tmle_cells, CLUSTER_BOOTSTRAP_SEED + 1)
    )
    primary_dataset_rows = [
        {"scope": "primary", **row}
        for row in _equal_cell_summary(primary_cells, ("dataset",))
    ]
    tmle_dataset_rows = [
        {"scope": "fixed_floor_tmle", **row}
        for row in _equal_cell_summary(tmle_cells, ("dataset",))
    ]
    pooled_overall = [
        {
            "scope": "primary",
            "aggregation": "pooled_ratio_of_means_sensitivity",
            "cells": len(primary_cells),
            **_summarize_cell(primary_rows),
        },
        {
            "scope": "fixed_floor_tmle",
            "aggregation": "pooled_ratio_of_means_sensitivity",
            "cells": len(tmle_cells),
            **_summarize_cell(tmle_rows),
        },
    ]
    tmle_ks_cells = [row for row in tmle_cells if row["dataset"] == "Kang-Schafer"]
    tmle_ks_active = sum(int(row["active_rows"]) for row in tmle_ks_cells)
    tmle_active = sum(int(row["active_rows"]) for row in tmle_cells)

    tables = {
        "same_sample_penalty_equal_setting_summary.csv": equal_overall,
        "same_sample_penalty_pooled_sensitivity.csv": pooled_overall,
        "same_sample_penalty_setting_method_summary.csv": setting_method_rows,
        "same_sample_penalty_method_summary.csv": _equal_cell_summary(
            setting_method_rows, ("method",)
        ),
        "same_sample_penalty_dataset_summary.csv": (
            primary_dataset_rows + tmle_dataset_rows
        ),
        "same_sample_penalty_dataset_method_summary.csv": _equal_cell_summary(
            setting_method_rows, ("dataset", "method")
        ),
    }
    for name, table_rows in tables.items():
        _write_csv(out_dir / name, table_rows)

    primary = equal_overall[0]
    tmle = equal_overall[1]
    pooled_primary = pooled_overall[0]
    pooled_tmle = pooled_overall[1]
    lines = [
        "# Section 4 Same-Sample Selection Penalty",
        "",
        "This derived diagnostic reads the Section 3 bound replay rows for the",
        "paper-facing benchmark fits.  It compares actual same-sample squared",
        "error with the fresh-risk decomposition `bias^2 + A` for the same",
        "selected gamma.  The paper's primary estimand first computes each",
        "setting-by-method gain and then averages those gains with equal weight.",
        "The pooled ratio-of-means appears only as a labelled sensitivity check.",
        "",
        "## Equal-setting result",
        "",
        "- Primary experts: same-sample MSE gain {same:.3f}%, fresh-risk gain {fresh:.3f}%, "
        "gain gap {gap:.3f} percentage points [{lo:.3f}, {hi:.3f}], active share {active:.3f}, same-sample harm {harm:.3f}, "
        "fresh-risk harm {fresh_harm:.3f}.".format(
            same=primary["same_sample_mse_gain_pct"],
            fresh=primary["fresh_risk_gain_pct"],
            gap=primary["gain_gap_same_minus_fresh_pct"],
            lo=primary["gain_gap_ci_low"],
            hi=primary["gain_gap_ci_high"],
            active=primary["active_share"],
            harm=primary["same_sample_harm_share"],
            fresh_harm=primary["fresh_risk_harm_share"],
        ),
        "- Fixed-floor TMLE: same-sample MSE gain {same:.3f}%, fresh-risk gain {fresh:.3f}%, "
        "gain gap {gap:.3f} percentage points [{lo:.3f}, {hi:.3f}], active share {active:.3f}, same-sample harm {harm:.3f}, "
        "fresh-risk harm {fresh_harm:.3f}.".format(
            same=tmle["same_sample_mse_gain_pct"],
            fresh=tmle["fresh_risk_gain_pct"],
            gap=tmle["gain_gap_same_minus_fresh_pct"],
            lo=tmle["gain_gap_ci_low"],
            hi=tmle["gain_gap_ci_high"],
            active=tmle["active_share"],
            harm=tmle["same_sample_harm_share"],
            fresh_harm=tmle["fresh_risk_harm_share"],
        ),
        "- Across the 96 primary setting-by-method cells, the mean absolute gain gap is "
        "{mean_abs:.3f} percentage points and the range is [{minimum:.3f}, {maximum:.3f}].".format(
            mean_abs=primary["mean_abs_gain_gap_pct"],
            minimum=primary["min_gain_gap_pct"],
            maximum=primary["max_gain_gap_pct"],
        ),
        "- Kang--Schafer contributes {ks_active} of {all_active} fixed-floor TMLE active "
        "moves ({share:.1f}%).".format(
            ks_active=tmle_ks_active,
            all_active=tmle_active,
            share=100.0 * tmle_ks_active / tmle_active,
        ),
        "",
        "## Pooled sensitivity (not the paper estimand)",
        "",
        "- Primary experts: same-sample MSE gain {same:.3f}%, fixed-candidate risk gain "
        "{fresh:.3f}%, gap {gap:.3f} percentage points.".format(
            same=pooled_primary["same_sample_mse_gain_pct"],
            fresh=pooled_primary["fresh_risk_gain_pct"],
            gap=pooled_primary["gain_gap_same_minus_fresh_pct"],
        ),
        "- Fixed-floor TMLE: same-sample MSE gain {same:.3f}%, fixed-candidate risk gain "
        "{fresh:.3f}%, gap {gap:.3f} percentage points.".format(
            same=pooled_tmle["same_sample_mse_gain_pct"],
            fresh=pooled_tmle["fresh_risk_gain_pct"],
            gap=pooled_tmle["gain_gap_same_minus_fresh_pct"],
        ),
        "",
        "## By Method",
        "",
    ]
    for row in tables["same_sample_penalty_method_summary.csv"]:
        lines.append(
            "- {method}: same-sample MSE gain {same:.3f}%, fresh-risk gain {fresh:.3f}%, "
            "gain gap {gap:.3f} percentage points, active {active:.3f}, same-sample harm {harm:.3f}, "
            "fresh-risk harm {fresh_harm:.3f}.".format(
                method=row["method"],
                same=row["same_sample_mse_gain_pct"],
                fresh=row["fresh_risk_gain_pct"],
                gap=row["gain_gap_same_minus_fresh_pct"],
                active=row["active_share"],
                harm=row["same_sample_harm_share"],
                fresh_harm=row["fresh_risk_harm_share"],
            )
        )
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")

    provenance = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "primary_rows": str(args.primary_rows.resolve()),
        "primary_rows_sha256": _sha256(args.primary_rows),
        "tmle_rows": str(args.tmle_rows.resolve()),
        "tmle_rows_sha256": _sha256(args.tmle_rows),
        "scope": "derived same-sample selection penalty audit from Section 3 bound replay rows",
        "primary_aggregation": (
            "within each dataset-setting-method cell, compute "
            "100*(mean(ref_error^2)-mean(selected_error^2))/mean(ref_error^2); "
            "then average cell gains with equal weight"
        ),
        "fixed_candidate_aggregation": (
            "within each dataset-setting-method cell, compute "
            "100*(mean(bias0_sq+A0)-mean(biasg_sq+Ag))/mean(bias0_sq+A0); "
            "then average cell gains with equal weight"
        ),
        "coverage": coverage,
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    verification = {
        "status": "PASS",
        "checks": {
            "required_input_columns": True,
            "unique_replication_identities": True,
            "identical_24_setting_coverage": True,
            "replications_per_setting_method_cell": EXPECTED_REPS_PER_CELL,
            "primary_setting_method_cells": coverage["primary_cells"],
            "fixed_floor_tmle_setting_method_cells": coverage[
                "fixed_floor_tmle_cells"
            ],
            "primary_estimand": "equal_setting_method_cells",
            "pooled_ratio_is_sensitivity_only": True,
            "setting_cluster_bootstrap_draws": CLUSTER_BOOTSTRAP_DRAWS,
            "setting_cluster_bootstrap_seed_primary": CLUSTER_BOOTSTRAP_SEED,
            "setting_cluster_bootstrap_seed_fixed_floor_tmle": (
                CLUSTER_BOOTSTRAP_SEED + 1
            ),
        },
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
