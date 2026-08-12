#!/usr/bin/env python3
"""Fail-closed verification of the Section 4 release against the manuscript."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--paper-root", required=True, type=Path)
    args = parser.parse_args()
    release = args.data_root / "support_csv/dml_section4_release_20260812_v1"
    atlas_data = args.data_root / "support_csv/dml_section4_c_atlas_20260812_v1"
    figure_release = (
        args.data_root / "support_csv/dml_section4_c_atlas_figures_20260812_v1"
    )

    for line in (release / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = release / name
        if sha256(path) != expected:
            raise SystemExit(f"release checksum mismatch: {path}")

    generated = {
        "section4_values.tex": args.paper_root / "sections/generated/section4_values.tex",
        "section4_sensitivity_rows.tex": args.paper_root / "sections/generated/section4_sensitivity_rows.tex",
        "section4_natural_table.tex": args.paper_root / "sections/generated/section4_natural_table.tex",
        "section4_emphasized_table.tex": args.paper_root / "sections/generated/section4_emphasized_table.tex",
    }
    for name, paper_path in generated.items():
        if (release / name).read_bytes() != paper_path.read_bytes():
            raise SystemExit(f"paper generated file differs from release: {paper_path}")

    for line in (figure_release / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = figure_release / name
        if sha256(path) != expected:
            raise SystemExit(f"figure-release checksum mismatch: {path}")
    for line in (atlas_data / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = atlas_data / name
        if sha256(path) != expected:
            raise SystemExit(f"atlas-data checksum mismatch: {path}")
    atlas_verification = json.loads((atlas_data / "verification.json").read_text())
    expected_counts = {
        "status": "COMPLETE", "panels": 21, "natural_panels": 12,
        "emphasized_panels": 8, "internal_panels": 1,
        "native_cells": 108, "cell_curve_rows": 864, "panel_curve_rows": 168,
    }
    for key, expected in expected_counts.items():
        if atlas_verification.get(key) != expected:
            raise SystemExit(
                f"unexpected atlas verification {key}: "
                f"{atlas_verification.get(key)!r} != {expected!r}"
            )
    if atlas_verification.get("primary_c") != 2.0:
        raise SystemExit("unexpected atlas primary c")

    figure_paper = args.paper_root / "figures/section4_c_atlas_20260812_v1"
    figure_files = (
        "README.md", "SHA256SUMS", "provenance.json",
        "section4_c_natural_1.pdf", "section4_c_natural_1.png",
        "section4_c_natural_2.pdf", "section4_c_natural_2.png",
        "section4_c_emphasized_1.pdf", "section4_c_emphasized_1.png",
        "section4_c_emphasized_2.pdf", "section4_c_emphasized_2.png",
        "section4_c_internal.pdf", "section4_c_internal.png",
    )
    for name in figure_files:
        if (figure_release / name).read_bytes() != (figure_paper / name).read_bytes():
            raise SystemExit(f"paper figure differs from release: {figure_paper / name}")
    figure_provenance = json.loads((figure_release / "provenance.json").read_text())
    if figure_provenance.get("status") != "COMPLETE":
        raise SystemExit("figure provenance is not COMPLETE")
    if len(figure_provenance.get("natural_panels", [])) != 12:
        raise SystemExit("unexpected natural atlas panel count")
    if len(figure_provenance.get("emphasized_panels", [])) != 8:
        raise SystemExit("unexpected emphasized atlas panel count")
    if figure_provenance.get("panel_csv_sha256") != sha256(atlas_data / "panel_c_curves.csv"):
        raise SystemExit("figure input hash does not match panel curves")
    if figure_provenance.get("cell_csv_sha256") != sha256(atlas_data / "cell_c_curves.csv"):
        raise SystemExit("figure input hash does not match cell curves")

    manuscript = (args.paper_root / "sections/experiments_rule_quality.tex").read_text()
    definitions = set(re.findall(r"\\newcommand\{\\(SFour[A-Za-z]+)\}", generated["section4_values.tex"].read_text()))
    generated_tables = "\n".join(
        generated[name].read_text()
        for name in ("section4_natural_table.tex", "section4_emphasized_table.tex")
    )
    uses = set(re.findall(r"\\(SFour[A-Za-z]+)", manuscript + "\n" + generated_tables))
    undefined = uses - definitions
    if undefined:
        raise SystemExit(f"undefined generated Section 4 macros: {sorted(undefined)}")
    figure_includes = (
        "section4_c_natural_1.pdf", "section4_c_natural_2.pdf",
        "section4_c_emphasized_1.pdf", "section4_c_emphasized_2.pdf",
        "section4_c_internal.pdf",
    )
    for name in figure_includes:
        figure_include = (
            "\\includegraphics[width=\\textwidth]"
            f"{{figures/section4_c_atlas_20260812_v1/{name}}}"
        )
        if figure_include not in manuscript:
            raise SystemExit(f"manuscript does not include verified atlas figure: {name}")
    for name in ("section4_natural_table", "section4_emphasized_table"):
        if f"\\input{{sections/generated/{name}}}" not in manuscript:
            raise SystemExit(f"manuscript does not input generated table: {name}")
    if "\\begin{table}" in manuscript:
        raise SystemExit("Section 4 contains a hand-maintained table environment")
    if "section4_c_sensitivity_20260812_v1" in manuscript:
        raise SystemExit("manuscript still includes superseded aggregate c figure")
    if "\\input{sections/generated/section4_sensitivity_rows}" in manuscript:
        raise SystemExit("manuscript still inputs the superseded c-sensitivity table")

    bundles = {}
    with (release / "source_map.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            bundles[row["bundle"]] = args.data_root / row["relative_path"]
            if row["verification_sha256"]:
                actual = sha256(bundles[row["bundle"]] / "verification.json")
                if actual != row["verification_sha256"]:
                    raise SystemExit(f"source-map verification hash mismatch: {row['bundle']}")
    with (release / "paper_values.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        source = bundles[row["source_bundle"]] / row["source_file"]
        if sha256(source) != row["source_sha256"]:
            raise SystemExit(f"paper-value source hash mismatch: {row['key']}")
    print(
        f"VERIFIED release_files={len((release / 'SHA256SUMS').read_text().splitlines())} "
        f"paper_values={len(rows)} macros_used={len(uses)} source_bundles={len(bundles)}"
        f" figure_files={len(figure_files)}"
    )


if __name__ == "__main__":
    main()
