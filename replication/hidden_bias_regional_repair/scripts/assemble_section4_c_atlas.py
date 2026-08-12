#!/usr/bin/env python3
"""Assemble the complete Section 4 c-curve atlas from certified paired rows.

The script makes the source policy explicit, validates 96 replications in
every native cell, reconstructs only the positive-part shrinkage rule, and
writes both native-cell and equal-cell curves.  It never fits an estimator or
selects c from the observed errors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


C_GRID = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
PRIMARY_C = 2.0
NBOOT = 20_000
EXPECTED_REPS = 96


@dataclass(frozen=True)
class Source:
    name: str
    relative_path: str


SOURCES = (
    Source("core", "dml_section4_confirmatory_20260810_v1/section4_raw_rows.csv"),
    Source("cui_published", "dml_section4_cui_published_projection_20260811_v1/raw_rows.csv"),
    Source("aligned_real", "dml_section4_aligned_real_20260811_v1/raw_rows.csv"),
    Source("real_safety", "dml_section4_wider_partial_real_20260810_v1/raw_rows.csv"),
    Source("d0_anchor", "dml_section4_region_local_anchor_20260811_v1/raw_rows.csv"),
    Source("ks_tmle_aipw", "dml_section4_ks_tmle_aipw_20260811_v1/raw_rows.csv"),
    Source("tmle_projection", "dml_section4_tmle_projection_20260811_v1/raw_rows.csv"),
)


PANEL_META = {
    "ks_ctmle": ("natural", "Kang--Schafer", "ctmle", "C-TMLE"),
    "ks_cui": ("natural", "Kang--Schafer", "cui_selective_ml", "selective ML"),
    "ks_aipw": ("natural", "Kang--Schafer", "aipw", "AIPW"),
    "ks_tmle": ("natural", "Kang--Schafer", "tmle", "plain TMLE (projection)"),
    "cui_s1_aipw": ("natural", "Cui scenario 1", "aipw", "AIPW"),
    "cui_s1_tmle": ("natural", "Cui scenario 1", "tmle", "plain TMLE (projection)"),
    "cui_s2_aipw": ("natural", "Cui scenario 2", "aipw", "AIPW"),
    "cui_s2_tmle": ("natural", "Cui scenario 2", "tmle", "plain TMLE (projection)"),
    "real_digits_ctmle": ("natural", "digits, wider/partial", "ctmle", "C-TMLE"),
    "real_digits_cui": ("natural", "digits, wider/partial", "cui_selective_ml", "selective ML"),
    "real_breast_ctmle": ("natural", "breast cancer, wider/partial", "ctmle", "C-TMLE"),
    "real_breast_cui": ("natural", "breast cancer, wider/partial", "cui_selective_ml", "selective ML"),
    "d0_ctmle": ("emphasized", "aligned anchor", "ctmle", "C-TMLE"),
    "d0_cui": ("emphasized", "aligned anchor", "cui_selective_ml", "selective ML"),
    "placement_ctmle": ("emphasized", "placement stress", "ctmle", "C-TMLE"),
    "placement_cui": ("emphasized", "placement stress", "cui_selective_ml", "selective ML"),
    "aligned_digits_ctmle": ("emphasized", "aligned digits", "ctmle", "C-TMLE"),
    "aligned_digits_cui": ("emphasized", "aligned digits", "cui_selective_ml", "selective ML"),
    "aligned_breast_ctmle": ("emphasized", "aligned breast cancer", "ctmle", "C-TMLE"),
    "aligned_breast_cui": ("emphasized", "aligned breast cancer", "cui_selective_ml", "selective ML"),
    "ks_tmle_residual": ("internal", "Kang--Schafer", "tmle", "plain TMLE (residual)"),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--nboot", type=int, default=NBOOT)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(base: int, *parts: object) -> int:
    payload = "|".join(map(str, (base,) + parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def finite(raw: str, source: str, field: str) -> float:
    try:
        value = float(raw)
    except ValueError as error:
        raise SystemExit(f"{source}: invalid {field}={raw!r}") from error
    if not math.isfinite(value):
        raise SystemExit(f"{source}: non-finite {field}={raw!r}")
    return value


def panel_for(source: str, row: dict[str, str]) -> str | None:
    method, design = row["reference_method"], row["design"]
    if source == "core" and design.startswith("kang_schafer_"):
        return {"ctmle": "ks_ctmle", "cui_selective_ml": "ks_cui"}.get(method)
    if source == "core" and design.startswith("alignment_"):
        return {"ctmle": "placement_ctmle", "cui_selective_ml": "placement_cui"}.get(method)
    if source == "ks_tmle_aipw" and design.startswith("kang_schafer_"):
        return {"aipw": "ks_aipw", "tmle": "ks_tmle_residual"}.get(method)
    if source == "tmle_projection" and method == "tmle":
        if design.startswith("kang_schafer_"):
            return "ks_tmle"
        if design.startswith("cui_published_"):
            return f"cui_s{design[-1]}_tmle"
    if source == "cui_published" and method == "aipw" and design.startswith("cui_published_"):
        return f"cui_s{design[-1]}_aipw"
    if source == "real_safety":
        pool = "digits" if "digits" in design else "breast"
        return {"ctmle": f"real_{pool}_ctmle", "cui_selective_ml": f"real_{pool}_cui"}.get(method)
    if source == "d0_anchor":
        return {"ctmle": "d0_ctmle", "cui_selective_ml": "d0_cui"}.get(method)
    if source == "aligned_real":
        pool = "digits" if "digits" in design else "breast"
        return {"ctmle": f"aligned_{pool}_ctmle", "cui_selective_ml": f"aligned_{pool}_cui"}.get(method)
    return None


def native_label(design: str, n: int, strength: float) -> str:
    if design.startswith("kang_schafer_"):
        return f"{design.rsplit('_', 1)[1].upper()}, n={n}"
    if design.startswith("cui_published_"):
        return f"scenario {design[-1]}, n={n}"
    if design == "regional_shift":
        return f"s={strength:g}"
    if design.startswith("alignment_"):
        return f"{design.split('_', 1)[1]}, s={strength:g}"
    if design.startswith("real_"):
        pool = "digits" if "digits" in design else "breast cancer"
        return f"{pool}, s={strength:g}"
    return f"{design}, n={n}, s={strength:g}"


def use_in_summary(panel: str, strength: float) -> bool:
    # The emphasized efficacy summaries separate signal from the s=0 null,
    # matching the manuscript-facing estimand. Native null curves remain in
    # the cell-level output and are drawn in the figure.
    if panel in {
        "d0_ctmle", "d0_cui", "aligned_digits_ctmle", "aligned_digits_cui",
        "aligned_breast_ctmle", "aligned_breast_cui",
    }:
        return strength > 0
    return True


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = arguments()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    support = args.data_root / "support_csv"
    source_paths = {source.name: support / source.relative_path for source in SOURCES}
    for name, path in source_paths.items():
        if not path.is_file():
            raise SystemExit(f"missing source {name}: {path}")

    # panel -> native identity -> paired rows (ref, delta, nonnegative vd)
    cells: dict[str, dict[tuple[str, int, float], list[tuple[float, float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    source_rows: dict[str, int] = defaultdict(int)
    for source_name, path in source_paths.items():
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"design", "n", "strength", "reference_method", "ref_error", "rt_error", "delta", "vd"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise SystemExit(f"{path}: missing columns {sorted(missing)}")
            for line, row in enumerate(reader, start=2):
                panel = panel_for(source_name, row)
                if panel is None:
                    continue
                source = f"{path}:{line}"
                n = int(row["n"])
                strength = finite(row["strength"], source, "strength")
                ref = finite(row["ref_error"], source, "ref_error")
                rt = finite(row["rt_error"], source, "rt_error")
                delta = finite(row["delta"], source, "delta")
                vd = max(0.0, finite(row["vd"], source, "vd"))
                if abs(rt - ref - delta) > 1e-10:
                    raise SystemExit(f"{source}: rt_error != ref_error + delta")
                cells[panel][(row["design"], n, strength)].append((ref, delta, vd))
                source_rows[source_name] += 1

    if set(cells) != set(PANEL_META):
        raise SystemExit(f"panel coverage mismatch: observed={sorted(cells)}")
    for panel, native in cells.items():
        bad = {key: len(rows) for key, rows in native.items() if len(rows) != EXPECTED_REPS}
        if bad:
            raise SystemExit(f"{panel}: expected {EXPECTED_REPS} reps/native cell; bad={bad}")

    cell_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for panel, native in sorted(cells.items()):
        block, dataset, method, method_label = PANEL_META[panel]
        computed: dict[tuple[str, int, float, float], tuple[float, float, float, np.ndarray]] = {}
        for (design, n, strength), values in sorted(native.items()):
            array = np.asarray(values, dtype=float)
            ref, delta, vd = array.T
            ref2 = ref * ref
            for c_value in C_GRID:
                ratio = np.divide(
                    c_value * vd,
                    delta * delta,
                    out=np.full_like(delta, np.inf),
                    where=delta != 0,
                )
                weight = np.maximum(0.0, 1.0 - ratio)
                repaired = ref + weight * delta
                repaired2 = repaired * repaired
                gain = 1.0 - float(repaired2.mean()) / float(ref2.mean())
                activation = float(np.mean(weight > 0))
                harm = float(np.mean(repaired2 > ref2))
                rng = np.random.default_rng(stable_seed(args.seed, panel, design, n, strength, c_value))
                indices = rng.integers(0, len(ref), size=(args.nboot, len(ref)))
                draws = 1.0 - repaired2[indices].mean(axis=1) / ref2[indices].mean(axis=1)
                computed[(design, n, strength, c_value)] = (gain, activation, harm, draws)
                cell_rows.append(
                    {
                        "block": block,
                        "panel": panel,
                        "dataset": dataset,
                        "method": method,
                        "method_label": method_label,
                        "design": design,
                        "n": n,
                        "strength": f"{strength:g}",
                        "native_cell": native_label(design, n, strength),
                        "c": f"{c_value:g}",
                        "reps": len(ref),
                        "relative_mse_reduction": f"{gain:.17g}",
                        "ci_lo": f"{float(np.percentile(draws, 2.5)):.17g}",
                        "ci_hi": f"{float(np.percentile(draws, 97.5)):.17g}",
                        "activation": f"{activation:.17g}",
                        "harm": f"{harm:.17g}",
                    }
                )
        for c_value in C_GRID:
            selected = [
                values
                for (design, n, strength, c), values in computed.items()
                if c == c_value and use_in_summary(panel, strength)
            ]
            gains = [value[0] for value in selected]
            family_draws = np.mean(np.stack([value[3] for value in selected]), axis=0)
            summary_rows.append(
                {
                    "block": block,
                    "panel": panel,
                    "dataset": dataset,
                    "method": method,
                    "method_label": method_label,
                    "c": f"{c_value:g}",
                    "cells": len(selected),
                    "equal_cell_relative_mse_reduction": f"{float(np.mean(gains)):.17g}",
                    "ci_lo": f"{float(np.percentile(family_draws, 2.5)):.17g}",
                    "ci_hi": f"{float(np.percentile(family_draws, 97.5)):.17g}",
                    "mean_activation": f"{float(np.mean([value[1] for value in selected])):.17g}",
                    "mean_harm_rate": f"{float(np.mean([value[2] for value in selected])):.17g}",
                }
            )

    write_csv(args.out_dir / "cell_c_curves.csv", cell_rows)
    write_csv(args.out_dir / "panel_c_curves.csv", summary_rows)
    source_map = [
        {
            "source": source.name,
            "relative_path": str(source_paths[source.name].relative_to(args.data_root)),
            "sha256": sha256(source_paths[source.name]),
            "selected_rows": source_rows[source.name],
        }
        for source in SOURCES
    ]
    write_csv(args.out_dir / "source_map.csv", source_map)
    (args.out_dir / "README.md").write_text(
        "# Complete Section 4 c-curve data\n\n"
        "Generated by `scripts/assemble_section4_c_atlas.py` from the certified "
        "paired-row bundles listed in `source_map.csv`. `cell_c_curves.csv` retains "
        "every native design/sample-size/intensity cell. `panel_c_curves.csv` gives "
        "the equal-cell curve used as the heavy line in each figure panel. The "
        "prespecified grid is `0,1,2,3,4,5,6,8`; `c=2` remains primary. No value of "
        "c is selected from observed MSE. Ma's DiD projection uses a distinct "
        "pre-registered gamma path and is therefore not falsely represented as a "
        "c-indexed estimator.\n",
        encoding="utf-8",
    )
    verification = {
        "status": "COMPLETE",
        "c_grid": list(C_GRID),
        "primary_c": PRIMARY_C,
        "bootstrap_draws": args.nboot,
        "panels": len(PANEL_META),
        "natural_panels": sum(meta[0] == "natural" for meta in PANEL_META.values()),
        "emphasized_panels": sum(meta[0] == "emphasized" for meta in PANEL_META.values()),
        "internal_panels": sum(meta[0] == "internal" for meta in PANEL_META.values()),
        "native_cells": sum(len(value) for value in cells.values()),
        "cell_curve_rows": len(cell_rows),
        "panel_curve_rows": len(summary_rows),
        "script_sha256": sha256(Path(__file__)),
    }
    (args.out_dir / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files = sorted(path for path in args.out_dir.iterdir() if path.name != "SHA256SUMS")
    (args.out_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files), encoding="utf-8"
    )
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
