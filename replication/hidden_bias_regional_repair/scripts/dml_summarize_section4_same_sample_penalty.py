#!/usr/bin/env python3
"""Summarize same-sample versus fresh-risk behavior from Section 3 diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


PRIMARY_METHODS = {"aipw", "ctmle", "cui_selective_ml", "ma_dr_bc"}


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _read_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                ref_error_raw = _float(row, "ref_error")
                selected_error_raw = _float(row, "selected_error")
                ref_error = ref_error_raw * ref_error_raw
                selected_error = selected_error_raw * selected_error_raw
                ref_fresh_risk = _float(row, "bias0_sq") + _float(row, "A0")
                selected_fresh_risk = _float(row, "biasg_sq") + _float(row, "Ag")
                rows.append(
                    {
                        "source_file": str(path),
                        "dataset": row["dataset"],
                        "setting": row["setting"],
                        "method": row["method"],
                        "n": int(row["n"]),
                        "strength": row["strength"],
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
                )
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[int(q * (len(ordered) - 1))]


def _summarize(group: list[dict[str, object]]) -> dict[str, object]:
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


def _group_rows(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out = []
    for key_values, group in sorted(groups.items()):
        summary = {key: value for key, value in zip(keys, key_values)}
        summary.update(_summarize(group))
        out.append(summary)
    return out


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

    tables = {
        "same_sample_penalty_overall_summary.csv": [
            {"scope": "primary", **_summarize(primary_rows)},
            {"scope": "fixed_floor_tmle", **_summarize(tmle_rows)},
            {"scope": "all", **_summarize(rows)},
        ],
        "same_sample_penalty_method_summary.csv": _group_rows(rows, ("method",)),
        "same_sample_penalty_dataset_summary.csv": _group_rows(rows, ("dataset",)),
        "same_sample_penalty_dataset_method_summary.csv": _group_rows(rows, ("dataset", "method")),
    }
    for name, table_rows in tables.items():
        _write_csv(out_dir / name, table_rows)

    primary = tables["same_sample_penalty_overall_summary.csv"][0]
    tmle = tables["same_sample_penalty_overall_summary.csv"][1]
    lines = [
        "# Section 4 Same-Sample Selection Penalty",
        "",
        "This derived diagnostic reads the Section 3 bound replay rows for the",
        "paper-facing benchmark fits.  It compares actual same-sample squared",
        "error with the fresh-risk decomposition `bias^2 + A` for the same",
        "selected gamma.",
        "",
        "## Overall",
        "",
        "- Primary experts: same-sample MSE gain {same:.3f}%, fresh-risk gain {fresh:.3f}%, "
        "gain gap {gap:.3f} percentage points, active share {active:.3f}, same-sample harm {harm:.3f}, "
        "fresh-risk harm {fresh_harm:.3f}.".format(
            same=primary["same_sample_mse_gain_pct"],
            fresh=primary["fresh_risk_gain_pct"],
            gap=primary["gain_gap_same_minus_fresh_pct"],
            active=primary["active_share"],
            harm=primary["same_sample_harm_share"],
            fresh_harm=primary["fresh_risk_harm_share"],
        ),
        "- Fixed-floor TMLE: same-sample MSE gain {same:.3f}%, fresh-risk gain {fresh:.3f}%, "
        "gain gap {gap:.3f} percentage points, active share {active:.3f}, same-sample harm {harm:.3f}, "
        "fresh-risk harm {fresh_harm:.3f}.".format(
            same=tmle["same_sample_mse_gain_pct"],
            fresh=tmle["fresh_risk_gain_pct"],
            gap=tmle["gain_gap_same_minus_fresh_pct"],
            active=tmle["active_share"],
            harm=tmle["same_sample_harm_share"],
            fresh_harm=tmle["fresh_risk_harm_share"],
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
        "primary_rows": str(args.primary_rows),
        "primary_rows_sha256": _sha256(args.primary_rows),
        "tmle_rows": str(args.tmle_rows),
        "tmle_rows_sha256": _sha256(args.tmle_rows),
        "scope": "derived same-sample selection penalty summary from Section 3 bound replay rows",
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
