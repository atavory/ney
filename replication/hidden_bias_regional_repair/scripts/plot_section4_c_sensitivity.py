#!/usr/bin/env python3
"""Render the canonical Section 4 c-sensitivity figure from the release CSV.

This is a pure rendering step: it validates and plots the already-aggregated
Section 4 sensitivity table. It never recomputes results or selects c.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


C_GRID = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
PRIMARY_C = 2.0
METHODS = ("ctmle", "cui_selective_ml")
FAMILIES = ("kang_schafer", "alignment")
METHOD_LABEL = {
    "ctmle": "C-TMLE",
    "cui_selective_ml": "selective ML",
}
METHOD_COLOR = {
    "ctmle": "#0072B2",
    "cui_selective_ml": "#D55E00",
}
FAMILY_LABEL = {
    "kang_schafer": "natural",
    "alignment": "emphasized",
}
FAMILY_STYLE = {
    "kang_schafer": ("-", "o"),
    "alignment": ("--", "s"),
}
REQUIRED_COLUMNS = {
    "method",
    "family",
    "c",
    "cells",
    "equal_cell_relative_mse_reduction",
    "ci_lo",
    "ci_hi",
    "mean_harm_rate",
    "mean_activation",
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def parse_float(raw: str, field: str, row_number: int) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise SystemExit(f"row {row_number}: invalid {field}={raw!r}") from exc
    if not math.isfinite(value):
        raise SystemExit(f"row {row_number}: non-finite {field}={raw!r}")
    return value


def load_rows(path: Path) -> dict[tuple[str, str, float], dict[str, float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise SystemExit(f"missing columns: {sorted(missing)}")
        rows: dict[tuple[str, str, float], dict[str, float]] = {}
        for row_number, raw in enumerate(reader, start=2):
            method = raw["method"]
            family = raw["family"]
            c_value = parse_float(raw["c"], "c", row_number)
            if method not in METHODS or family not in FAMILIES:
                raise SystemExit(
                    f"row {row_number}: unexpected method/family={method}/{family}"
                )
            if c_value not in C_GRID:
                raise SystemExit(f"row {row_number}: unexpected c={c_value:g}")
            key = (method, family, c_value)
            if key in rows:
                raise SystemExit(f"row {row_number}: duplicate key={key}")
            values = {
                "cells": parse_float(raw["cells"], "cells", row_number),
                "gain": parse_float(
                    raw["equal_cell_relative_mse_reduction"], "gain", row_number
                ),
                "lo": parse_float(raw["ci_lo"], "ci_lo", row_number),
                "hi": parse_float(raw["ci_hi"], "ci_hi", row_number),
                "harm": parse_float(raw["mean_harm_rate"], "harm", row_number),
                "activation": parse_float(
                    raw["mean_activation"], "activation", row_number
                ),
            }
            if values["cells"] <= 0:
                raise SystemExit(f"row {row_number}: cells must be positive")
            if not values["lo"] <= values["gain"] <= values["hi"]:
                raise SystemExit(f"row {row_number}: CI does not contain gain")
            for field in ("harm", "activation"):
                if not 0 <= values[field] <= 1:
                    raise SystemExit(f"row {row_number}: {field} outside [0,1]")
            rows[key] = values
    expected = {
        (method, family, c_value)
        for method in METHODS
        for family in FAMILIES
        for c_value in C_GRID
    }
    if set(rows) != expected:
        missing = sorted(expected - set(rows))
        extra = sorted(set(rows) - expected)
        raise SystemExit(f"incomplete sensitivity grid: missing={missing}; extra={extra}")
    return rows


def series(
    rows: dict[tuple[str, str, float], dict[str, float]],
    method: str,
    family: str,
    field: str,
    scale: float = 1.0,
) -> list[float]:
    return [rows[(method, family, c_value)][field] * scale for c_value in C_GRID]


def style_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.axvline(PRIMARY_C, color="0.35", linewidth=1.0, linestyle=":", zorder=0)
    ax.axhline(0, color="0.25", linewidth=0.8, zorder=0)
    ax.set_xticks(C_GRID)
    ax.set_xlabel("shrinkage constant $c$")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="0.88", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_gain(
    ax: plt.Axes,
    rows: dict[tuple[str, str, float], dict[str, float]],
    family: str,
    title: str,
) -> None:
    for method in METHODS:
        gains = series(rows, method, family, "gain", 100)
        lows = series(rows, method, family, "lo", 100)
        highs = series(rows, method, family, "hi", 100)
        color = METHOD_COLOR[method]
        ax.fill_between(C_GRID, lows, highs, color=color, alpha=0.14, linewidth=0)
        ax.plot(
            C_GRID,
            gains,
            color=color,
            marker="o",
            linewidth=2.0,
            markersize=4.0,
            label=METHOD_LABEL[method],
        )
    style_axis(ax, "MSE gain vs. reference (%)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)


def plot_diagnostic(
    ax: plt.Axes,
    rows: dict[tuple[str, str, float], dict[str, float]],
    field: str,
    title: str,
) -> None:
    for family in FAMILIES:
        linestyle, marker = FAMILY_STYLE[family]
        for method in METHODS:
            ax.plot(
                C_GRID,
                series(rows, method, family, field, 100),
                color=METHOD_COLOR[method],
                linestyle=linestyle,
                marker=marker,
                linewidth=1.8,
                markersize=3.8,
                label=f"{METHOD_LABEL[method]}, {FAMILY_LABEL[family]}",
            )
    style_axis(ax, f"{field} rate (%)")
    ax.set_ylim(bottom=-0.5)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=6.8, ncol=2)


def render(input_path: Path, out_dir: Path) -> None:
    rows = load_rows(input_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "section4_c_sensitivity.pdf"
    png_path = out_dir / "section4_c_sensitivity.png"
    provenance_path = out_dir / "provenance.json"
    readme_path = out_dir / "README.md"
    checksums_path = out_dir / "SHA256SUMS"

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 8,
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    plot_gain(
        axes[0, 0],
        rows,
        "kang_schafer",
        "(a) Natural benchmark: Kang–Schafer",
    )
    plot_gain(
        axes[0, 1],
        rows,
        "alignment",
        "(b) Emphasized alignment design",
    )
    plot_diagnostic(axes[1, 0], rows, "activation", "(c) Activation")
    plot_diagnostic(axes[1, 1], rows, "harm", "(d) Individual harm")
    axes[0, 0].text(
        PRIMARY_C,
        axes[0, 0].get_ylim()[1],
        " primary",
        color="0.35",
        fontsize=7,
        ha="left",
        va="top",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={"Creator": "plot_section4_c_sensitivity.py", "CreationDate": None},
    )
    fig.savefig(
        png_path,
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "plot_section4_c_sensitivity.py"},
    )
    plt.close(fig)

    readme_path.write_text(
        "# Section 4 c-sensitivity figure\n\n"
        "This bundle is generated deterministically by "
        "`scripts/plot_section4_c_sensitivity.py` from the canonical "
        "`section4_c_sensitivity.csv`. It is a rendering step only: the script "
        "does not recompute estimates or select c.\n\n"
        f"- Input SHA-256: `{sha256(input_path)}`\n"
        f"- Rows: `{len(rows)}`\n"
        f"- Prespecified grid: `{list(C_GRID)}`\n"
        f"- Primary value: `c={PRIMARY_C:g}`\n",
        encoding="utf-8",
    )
    provenance = {
        "schema": 1,
        "status": "COMPLETE",
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "input_rows": len(rows),
        "methods": list(METHODS),
        "families": list(FAMILIES),
        "c_grid": list(C_GRID),
        "primary_c": PRIMARY_C,
        "script_sha256": sha256(Path(__file__)),
        "outputs": {
            pdf_path.name: sha256(pdf_path),
            png_path.name: sha256(png_path),
        },
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_rows = [
        f"{sha256(path)}  {path.name}"
        for path in (readme_path, pdf_path, png_path, provenance_path)
    ]
    checksums_path.write_text("\n".join(checksum_rows) + "\n", encoding="utf-8")
    print(
        f"wrote {pdf_path}\nwrote {png_path}\nwrote {provenance_path}"
        f"\nvalidated rows={len(rows)} c_grid={list(C_GRID)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    render(args.input, args.out_dir)


if __name__ == "__main__":
    main()
