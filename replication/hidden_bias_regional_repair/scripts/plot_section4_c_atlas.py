#!/usr/bin/env python3
"""Render the complete natural, emphasized, and internal Section 4 c atlases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.size": 7.2,
    "axes.titlesize": 7.8,
    "axes.labelsize": 7.2,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 5.5,
})


C_GRID = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
PRIMARY_C = 2.0
NATURAL = (
    "ks_ctmle", "ks_cui", "ks_aipw", "ks_tmle",
    "cui_s1_aipw", "cui_s1_tmle", "cui_s2_aipw", "cui_s2_tmle",
    "real_digits_ctmle", "real_digits_cui", "real_breast_ctmle", "real_breast_cui",
)
EMPHASIZED = (
    "d0_ctmle", "d0_cui", "placement_ctmle", "placement_cui",
    "aligned_digits_ctmle", "aligned_digits_cui",
    "aligned_breast_ctmle", "aligned_breast_cui",
)
SAFETY_ACTIVE = ("cui_s2_aipw", "real_digits_cui", "real_breast_cui")
TITLE = {
    "ks_ctmle": "Kang–Schafer × C-TMLE",
    "ks_cui": "Kang–Schafer × selective ML",
    "ks_aipw": "Kang–Schafer × AIPW",
    "ks_tmle": "Kang–Schafer × plain TMLE",
    "cui_s1_aipw": "Cui scenario 1 × AIPW",
    "cui_s1_tmle": "Cui scenario 1 × plain TMLE",
    "cui_s2_aipw": "Cui scenario 2 × AIPW",
    "cui_s2_tmle": "Cui scenario 2 × plain TMLE",
    "real_digits_ctmle": "wider/partial digits × C-TMLE",
    "real_digits_cui": "wider/partial digits × selective ML",
    "real_breast_ctmle": "wider/partial breast cancer × C-TMLE",
    "real_breast_cui": "wider/partial breast cancer × selective ML",
    "d0_ctmle": "aligned anchor × C-TMLE",
    "d0_cui": "aligned anchor × selective ML",
    "placement_ctmle": "placement stress × C-TMLE",
    "placement_cui": "placement stress × selective ML",
    "aligned_digits_ctmle": "aligned digits × C-TMLE",
    "aligned_digits_cui": "aligned digits × selective ML",
    "aligned_breast_ctmle": "aligned breast cancer × C-TMLE",
    "aligned_breast_cui": "aligned breast cancer × selective ML",
    "ks_tmle_residual": "Kang–Schafer × residual TMLE",
}
SHORT = {
    "ks_ctmle": "KS/C-TMLE", "ks_cui": "KS/selective ML",
    "ks_aipw": "KS/AIPW", "ks_tmle": "KS/TMLE",
    "cui_s1_aipw": "Cui1/AIPW", "cui_s1_tmle": "Cui1/TMLE",
    "cui_s2_aipw": "Cui2/AIPW", "cui_s2_tmle": "Cui2/TMLE",
    "real_digits_ctmle": "digits/C-TMLE", "real_digits_cui": "digits/selective ML",
    "real_breast_ctmle": "breast/C-TMLE", "real_breast_cui": "breast/selective ML",
    "d0_ctmle": "anchor/C-TMLE", "d0_cui": "anchor/selective ML",
    "placement_ctmle": "placement/C-TMLE", "placement_cui": "placement/selective ML",
    "aligned_digits_ctmle": "aligned digits/C-TMLE",
    "aligned_digits_cui": "aligned digits/selective ML",
    "aligned_breast_ctmle": "aligned breast/C-TMLE",
    "aligned_breast_cui": "aligned breast/selective ML",
    "ks_tmle_residual": "KS/TMLE residual",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"empty input: {path}")
    return rows


def index_summary(rows: list[dict[str, str]]) -> dict[str, dict[float, dict[str, float]]]:
    result: dict[str, dict[float, dict[str, float]]] = defaultdict(dict)
    for row in rows:
        c_value = float(row["c"])
        result[row["panel"]][c_value] = {
            "gain": 100 * float(row["equal_cell_relative_mse_reduction"]),
            "lo": 100 * float(row["ci_lo"]),
            "hi": 100 * float(row["ci_hi"]),
            "activation": 100 * float(row["mean_activation"]),
            "harm": 100 * float(row["mean_harm_rate"]),
            "gross_harm": 100 * float(row["equal_cell_gross_harm_mse"]),
            "gross_benefit": 100 * float(row["equal_cell_gross_benefit_mse"]),
        }
    return result


def index_cells(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[float, float]]]:
    result: dict[str, dict[str, dict[float, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        result[row["panel"]][row["native_cell"]][float(row["c"])] = (
            100 * float(row["relative_mse_reduction"])
        )
    return result


def base_axis(
    ax: plt.Axes,
    ylabel: str | None = None,
    xlabel: str | None = None,
) -> None:
    ax.axhline(0, color="0.25", linewidth=0.8, zorder=0)
    ax.axvline(PRIMARY_C, color="0.35", linestyle=":", linewidth=1.0, zorder=0)
    ax.set_xticks(C_GRID)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="0.9", linewidth=0.65)
    ax.spines[["top", "right"]].set_visible(False)


def gain_panel(
    ax: plt.Axes,
    panel: str,
    summary: dict[str, dict[float, dict[str, float]]],
    cells: dict[str, dict[str, dict[float, float]]],
    letter: str,
    title: str | None = None,
) -> None:
    for label, values in sorted(cells[panel].items()):
        ax.plot(C_GRID, [values[c] for c in C_GRID], color="0.67", linewidth=0.65, alpha=0.65)
    curve = summary[panel]
    gain = [curve[c]["gain"] for c in C_GRID]
    lo = [curve[c]["lo"] for c in C_GRID]
    hi = [curve[c]["hi"] for c in C_GRID]
    ax.fill_between(C_GRID, lo, hi, color="#0072B2", alpha=0.16, linewidth=0)
    ax.plot(C_GRID, gain, color="#0072B2", marker="o", markersize=3.4, linewidth=2.0)
    base_axis(ax)
    ax.set_title(f"({letter}) {title or TITLE[panel]}", loc="left", fontweight="bold")
    ax.text(0.98, 0.94, f"{len(cells[panel])} native cells", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.2, color="0.35")


def render_gain_atlas(
    panels: tuple[str, ...],
    shape: tuple[int, int],
    summary: dict[str, dict[float, dict[str, float]]],
    cells: dict[str, dict[str, dict[float, float]]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(
        *shape, figsize=(7.2, 1.55 * shape[0] + 0.2), constrained_layout=True
    )
    flat = list(axes.flat)
    for index, panel in enumerate(panels):
        gain_panel(flat[index], panel, summary, cells, chr(ord("a") + index))
    for ax in flat[len(panels):]:
        ax.axis("off")
    fig.supxlabel("shrinkage constant $c$", fontsize=7.2)
    fig.supylabel("MSE gain (%)", fontsize=7.2)
    fig.savefig(path, bbox_inches="tight", metadata={"Creator": "plot_section4_c_atlas.py", "CreationDate": None})
    fig.savefig(
        path.with_suffix(".png"), dpi=180, bbox_inches="tight",
        metadata={"Software": "plot_section4_c_atlas.py"},
    )
    plt.close(fig)


def render_emphasized_atlas(
    summary: dict[str, dict[float, dict[str, float]]],
    cells: dict[str, dict[str, dict[float, float]]],
    path: Path,
) -> None:
    """Arrange estimators by row and designs by column for direct comparison."""
    matrix = (
        ("d0_ctmle", "placement_ctmle", "aligned_digits_ctmle", "aligned_breast_ctmle"),
        ("d0_cui", "placement_cui", "aligned_digits_cui", "aligned_breast_cui"),
    )
    titles = ("aligned anchor", "placement", "digits", "breast cancer")
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.25), constrained_layout=True)
    for row, panels in enumerate(matrix):
        for column, panel in enumerate(panels):
            index = row * 4 + column
            gain_panel(
                axes[row, column], panel, summary, cells,
                chr(ord("a") + index), titles[column],
            )
    axes[0, 0].set_ylabel("C-TMLE\nMSE gain (%)")
    axes[1, 0].set_ylabel("selective ML\nMSE gain (%)")
    fig.supxlabel("shrinkage constant $c$", fontsize=7.2)
    fig.savefig(
        path, bbox_inches="tight",
        metadata={"Creator": "plot_section4_c_atlas.py", "CreationDate": None},
    )
    fig.savefig(
        path.with_suffix(".png"), dpi=180, bbox_inches="tight",
        metadata={"Software": "plot_section4_c_atlas.py"},
    )
    plt.close(fig)


def render_safety_atlas(
    summary: dict[str, dict[float, dict[str, float]]],
    path: Path,
) -> None:
    """Show what the safeguard does in natural low-opportunity settings.

    At c=0 the positive-part rule applies every available candidate in full.
    The harmful and helpful tails are kept separate so that cancellation in
    average MSE cannot masquerade as safety.  Increasing c suppresses noisy
    candidates; the first two panels show harm prevented and harm remaining,
    while the third records the useful MSE reduction retained.
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.95), constrained_layout=True)
    colors = ("#009E73", "#D55E00", "#0072B2")
    labels = {
        "cui_s2_aipw": "Cui scenario 2 / AIPW",
        "real_digits_cui": "digits / selective ML",
        "real_breast_cui": "breast cancer / selective ML",
    }
    for panel, color in zip(SAFETY_ACTIVE, colors):
        curve = summary[panel]
        baseline_harm = curve[0.0]["gross_harm"]
        residual_harm = [curve[c]["gross_harm"] for c in C_GRID]
        prevented_harm = [baseline_harm - value for value in residual_harm]
        retained_benefit = [curve[c]["gross_benefit"] for c in C_GRID]
        for ax, values in zip(axes, (prevented_harm, residual_harm, retained_benefit)):
            ax.plot(
                C_GRID, values, color=color, linewidth=1.8, marker="o",
                markersize=3.0, label=labels[panel],
            )
    titles = (
        "(a) Harm prevented",
        "(b) Harm remaining",
        "(c) Benefit retained",
    )
    for ax, title in zip(axes, titles):
        base_axis(ax)
        ax.set_ylim(bottom=-0.05)
        ax.set_title(title, loc="left", fontweight="bold")
    fig.supxlabel("shrinkage constant $c$", fontsize=7.2)
    fig.supylabel("% of reference MSE", fontsize=7.2)
    axes[1].text(
        0.98, 0.95,
        "TMLE and C-TMLE:\nno admissible candidate",
        transform=axes[1].transAxes, ha="right", va="top", fontsize=5.2,
        color="0.35",
    )
    axes[2].legend(frameon=False, fontsize=5.0, loc="upper right")
    fig.savefig(
        path, bbox_inches="tight",
        metadata={"Creator": "plot_section4_c_atlas.py", "CreationDate": None},
    )
    fig.savefig(
        path.with_suffix(".png"), dpi=180, bbox_inches="tight",
        metadata={"Software": "plot_section4_c_atlas.py"},
    )
    plt.close(fig)


def diagnostic_lines(
    ax: plt.Axes,
    panels: tuple[str, ...],
    summary: dict[str, dict[float, dict[str, float]]],
    field: str,
    title: str,
) -> None:
    colors = plt.get_cmap("tab10")
    for index, panel in enumerate(panels):
        ax.plot(C_GRID, [summary[panel][c][field] for c in C_GRID],
                color=colors(index % 10), linewidth=1.35, marker="o", markersize=2.4,
                label=SHORT[panel])
    base_axis(ax, f"{field} rate (%)")
    ax.set_ylim(bottom=-0.5)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=5.8, ncol=2)


def render_internal(summary: dict[str, dict[float, dict[str, float]]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 3.75), constrained_layout=True)
    diagnostic_lines(axes[0, 0], NATURAL, summary, "activation", "(a) Natural: activation")
    diagnostic_lines(axes[0, 1], NATURAL, summary, "harm", "(b) Natural: individual harm")
    diagnostic_lines(axes[1, 0], EMPHASIZED, summary, "activation", "(c) Emphasized: activation")
    residual, projection = summary["ks_tmle_residual"], summary["ks_tmle"]
    for curve, label, color in (
        (residual, "residual adapter", "#D55E00"),
        (projection, "target-preserving projection", "#0072B2"),
    ):
        axes[1, 1].plot(C_GRID, [curve[c]["gain"] for c in C_GRID], color=color,
                        marker="o", markersize=3, linewidth=1.8, label=label)
    base_axis(axes[1, 1], "MSE gain (%)")
    axes[1, 1].set_title("(d) Plain TMLE adapter ablation", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False, fontsize=5.8)
    fig.supxlabel("shrinkage constant $c$", fontsize=7.2)
    fig.savefig(path, bbox_inches="tight", metadata={"Creator": "plot_section4_c_atlas.py", "CreationDate": None})
    fig.savefig(
        path.with_suffix(".png"), dpi=180, bbox_inches="tight",
        metadata={"Software": "plot_section4_c_atlas.py"},
    )
    plt.close(fig)


def main() -> None:
    args = arguments()
    panel_path = args.data_dir / "panel_c_curves.csv"
    cell_path = args.data_dir / "cell_c_curves.csv"
    summary = index_summary(load(panel_path))
    cells = index_cells(load(cell_path))
    expected = set(NATURAL) | set(EMPHASIZED) | {"ks_tmle_residual"}
    if set(summary) != expected or set(cells) != expected:
        raise SystemExit("incomplete atlas panel coverage")
    if any(set(values) != set(C_GRID) for values in summary.values()):
        raise SystemExit("incomplete c grid")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "section4_c_natural_efficacy.pdf": (NATURAL[:6], (2, 3)),
    }
    for name, (panels, shape) in outputs.items():
        render_gain_atlas(panels, shape, summary, cells, args.out_dir / name)
    render_safety_atlas(summary, args.out_dir / "section4_c_natural_safety.pdf")
    render_emphasized_atlas(
        summary, cells, args.out_dir / "section4_c_emphasized.pdf"
    )
    render_internal(summary, args.out_dir / "section4_c_internal.pdf")
    provenance = {
        "status": "COMPLETE",
        "panel_csv_sha256": sha256(panel_path),
        "cell_csv_sha256": sha256(cell_path),
        "script_sha256": sha256(Path(__file__)),
        "natural_panels": list(NATURAL),
        "emphasized_panels": list(EMPHASIZED),
        "natural_safety_panels": list(SAFETY_ACTIVE),
        "natural_exact_standdown_panels": [
            "cui_s2_tmle", "real_digits_ctmle", "real_breast_ctmle"
        ],
        "internal_diagnostic": "activation, harm, and TMLE adapter ablation",
        "outputs": {
            path.name: sha256(path)
            for path in sorted(args.out_dir.iterdir())
            if path.suffix in {".pdf", ".png"}
        },
    }
    (args.out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "README.md").write_text(
        "# Section 4 c-curve atlases\n\n"
        "The efficacy atlases contain one curve for every reported missing-outcome "
        "dataset/design by upstream-estimator combination. Thin gray lines are "
        "native cells; the heavy blue line is the equal-cell target and its paired-"
        "bootstrap interval. Natural part II is purpose-built for safety: it "
        "separates harmful-tail MSE prevented, harmful-tail MSE remaining, and "
        "helpful-tail MSE retained instead of presenting stand-down as failed "
        "efficacy. The natural efficacy curves use a compact 2-by-3 layout, and all "
        "eight emphasized curves use one 2-by-4 estimator-by-design matrix. The "
        "internal atlas reports activation, "
        "individual harm, and the plain-TMLE adapter ablation. The dotted line is "
        "the frozen primary `c=2`.\n",
        encoding="utf-8",
    )
    files = sorted(path for path in args.out_dir.iterdir() if path.name != "SHA256SUMS")
    (args.out_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files), encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
