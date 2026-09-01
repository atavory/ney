#!/usr/bin/env python3
"""Verify the unified global-residual Section 4 release against the manuscript."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import tempfile
from pathlib import Path


RELEASE_DIR = "support_csv/dml_unified_cartesian_global_residual_20260814"
GENERATED_FILES = (
    "section4_values.tex",
    "section4_unified_overview_table.tex",
    "section4_unified_family_table.tex",
    "section4_unified_summary_table.tex",
    "section4_synthetic_diagnostic_table.tex",
    "section4_fixed_floor_tmle_diagnostic_table.tex",
)
AUXILIARY_GENERATED = {
    "support_csv/dml_high_response_placebo_ablation_20260831": (
        "section4_high_response_placebo_ablation_table.tex",
    ),
}
BENCHMARK_RELEASES = {
    "support_csv/dml_real_benchmark_expansion_20260831": {
        "jobs": 1440,
        "rows": 5760,
        "expected_c": 2.0,
        "seed": 20260831,
    },
    "support_csv/dml_real_benchmark_acic2017_20260831": {
        "jobs": 480,
        "rows": 1920,
        "expected_c": 2.0,
        "seed": 20260831,
    },
    "support_csv/dml_real_benchmark_twins_20260831": {
        "jobs": 480,
        "rows": 1920,
        "expected_c": 2.0,
        "seed": 20260831,
    },
}
EXPECTED_CONFIG = {
    "repair_mode": "if_residual",
    "validation_loss_se": 1.0,
    "shrink_c": 2.0,
    "bootstraps": 0,
    "frozen_source_sha256": (
        "98987b31cf7c883d4776996ae7b28f7f1b9fe134d6da323e95250f00232842ce"
    ),
}
EXPECTED_GRID = [0.0, 0.25, 0.5, 1.0]


def load_assembler(script_path: Path):
    spec = importlib.util.spec_from_file_location("section4_assembler", script_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load assembler: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_generated(data_root: Path, paper_root: Path) -> None:
    assembler = load_assembler(
        data_root / "scripts" / "assemble_section4_unified_global_residual.py"
    )
    release = data_root / RELEASE_DIR
    rows = assembler.read_selected_family_rows()
    cell_rows = assembler.read_cell_rows(release / "cell_summary.csv")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        assembler.write_values(out_dir / "section4_values.tex", rows)
        assembler.write_overview(out_dir / "section4_unified_overview_table.tex", rows)
        assembler.write_family_table(
            out_dir / "section4_unified_family_table.tex", cell_rows
        )
        assembler.write_benchmark_summary_table(
            out_dir / "section4_unified_summary_table.tex", cell_rows
        )
        assembler.write_synthetic_diagnostic_table(
            out_dir / "section4_synthetic_diagnostic_table.tex", rows
        )
        assembler.write_tmle_diagnostic_table(
            out_dir / "section4_fixed_floor_tmle_diagnostic_table.tex", cell_rows
        )
        for name in GENERATED_FILES:
            paper_path = paper_root / "sections" / "generated" / name
            if (out_dir / name).read_bytes() != paper_path.read_bytes():
                raise SystemExit(f"paper generated file differs from release: {paper_path}")


def verify_checksums(release: Path) -> int:
    checksum_path = release / "SHA256SUMS"
    if not checksum_path.exists():
        raise SystemExit(f"missing checksum file: {checksum_path}")
    count = 0
    for line in checksum_path.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = release / name
        if not path.exists():
            raise SystemExit(f"checksum target missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise SystemExit(f"checksum mismatch: {path}")
        count += 1
    return count


def compare_auxiliary_generated(data_root: Path, paper_root: Path) -> int:
    count = 0
    for release_name, generated_files in AUXILIARY_GENERATED.items():
        release = data_root / release_name
        if not release.exists():
            raise SystemExit(f"missing auxiliary Section 4 bundle: {release}")
        verify_checksums(release)
        verification = json.loads((release / "verification.json").read_text())
        if verification.get("status") != "PASS":
            raise SystemExit(f"auxiliary verification is not PASS: {release}")
        for name in generated_files:
            paper_path = paper_root / "sections" / "generated" / name
            release_path = release / name
            if release_path.read_bytes() != paper_path.read_bytes():
                raise SystemExit(
                    f"paper auxiliary generated file differs from release: {paper_path}"
                )
            count += 1
    return count


def verify_benchmark_releases(data_root: Path) -> int:
    for release_name, expected_values in BENCHMARK_RELEASES.items():
        release = data_root / release_name
        if not release.exists():
            raise SystemExit(f"missing benchmark bundle: {release}")
        verify_checksums(release)
        verification = json.loads((release / "verification.json").read_text())
        if verification.get("status") != "COMPLETE":
            raise SystemExit(f"benchmark verification is not COMPLETE: {release}")
        for key, expected in expected_values.items():
            if verification.get(key) != expected:
                raise SystemExit(
                    f"unexpected benchmark verification {key}: "
                    f"{verification.get(key)!r} != {expected!r}"
                )
    return len(BENCHMARK_RELEASES)


def verify_release(data_root: Path) -> dict:
    release = data_root / RELEASE_DIR
    verification = json.loads((release / "verification.json").read_text())
    if verification.get("status") != "PASS":
        raise SystemExit("unified release verification is not PASS")
    expected_counts = {
        "reps_files": 4080,
        "replication_rows": 16320,
        "cells": 170,
        "draws": 20000,
        "seed": 20260814,
        "full_manifest_sha256": (
            "65720b1fce2a24d55872ab9e008cf7ef62945b30b791272f1fdfe65280e2287f"
        ),
        "raw_reps_manifest_sha256": (
            "08d0e7f95d71773fe54eb137107e73c9f0346955247432a8ebb0e0dd1d195e92"
        ),
    }
    for key, expected in expected_counts.items():
        if verification.get(key) != expected:
            raise SystemExit(
                f"unexpected unified verification {key}: "
                f"{verification.get(key)!r} != {expected!r}"
            )
    config = verification.get("config", {})
    for key, expected in EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            raise SystemExit(
                f"unexpected unified config {key}: {config.get(key)!r} != {expected!r}"
            )
    if config.get("region_damp_grid") != EXPECTED_GRID:
        raise SystemExit("unexpected unified damping grid")
    for provenance in verification.get("run_provenance", []):
        for key, expected in EXPECTED_CONFIG.items():
            if provenance.get(key) != expected:
                raise SystemExit(
                    f"unexpected run provenance {key}: "
                    f"{provenance.get(key)!r} != {expected!r}"
                )
        if provenance.get("validation_risk") != "balanced_mse":
            raise SystemExit("unexpected run provenance validation_risk")
        if provenance.get("region_damp_grid") != EXPECTED_GRID:
            raise SystemExit("unexpected run provenance damping grid")
    return verification


def verify_manuscript(paper_root: Path) -> tuple[int, int]:
    section4 = (paper_root / "sections" / "experiments_rule_quality.tex").read_text()
    appendix = (paper_root / "appendices" / "empirical_checks.tex").read_text()
    values = (paper_root / "sections/generated/section4_values.tex").read_text()
    overview = (
        paper_root / "sections/generated/section4_unified_overview_table.tex"
    ).read_text()
    family = (
        paper_root / "sections/generated/section4_unified_family_table.tex"
    ).read_text()
    synthetic = (
        paper_root / "sections/generated/section4_synthetic_diagnostic_table.tex"
    ).read_text()
    diagnostic = (
        paper_root / "sections/generated/section4_fixed_floor_tmle_diagnostic_table.tex"
    ).read_text()

    required_inputs = (
        "sections/generated/section4_unified_family_table",
        "sections/generated/section4_unified_summary_table",
        "sections/generated/section4_high_response_placebo_ablation_table",
    )
    for name in required_inputs:
        if f"\\input{{{name}}}" not in section4:
            raise SystemExit(f"Section 4 does not input generated table: {name}")
    if (
        "\\input{sections/generated/section4_fixed_floor_tmle_diagnostic_table}"
        not in appendix
    ):
        raise SystemExit("appendix does not input fixed-floor TMLE diagnostic table")
    if "\\input{sections/generated/section4_unified_overview_table}" in section4:
        raise SystemExit("Section 4 inputs the aggregate overview table")
    if "\\begin{table}" in section4:
        raise SystemExit("Section 4 contains a hand-maintained table environment")
    paper_text = section4 + "\n" + appendix
    forbidden_paper_tokens = (
        "support_csv",
        "Manifold",
        "manifold",
        "SHA-256",
        "sha256",
        "\\path{",
    )
    for token in forbidden_paper_tokens:
        if token in paper_text:
            raise SystemExit(f"paper text contains data-provenance token: {token}")
    if "fixed-floor TMLE &" in overview or "fixed-floor TMLE &" in family:
        raise SystemExit("primary Section 4 tables still include fixed-floor TMLE")
    if "fixed-floor TMLE" not in diagnostic:
        raise SystemExit("diagnostic table does not identify fixed-floor TMLE")
    if "primary TMLE comparator" not in section4 + "\n" + appendix:
        raise SystemExit("manuscript does not identify C-TMLE as primary comparator")
    manuscript_text = paper_text
    normalized_manuscript_text = re.sub(r"\s+", " ", manuscript_text)
    required_phrases = (
        "The selected candidate is the returned estimate.",
        "The selected candidate is the reported estimate.",
    )
    for phrase in required_phrases:
        if phrase not in normalized_manuscript_text:
            raise SystemExit(f"manuscript does not record selected-candidate rule: {phrase}")
    forbidden_method_tokens = (
        "section4_no_shrinkage_ablation_table",
        "no-shrinkage",
        "no shrinkage",
        "shrinkage",
        "shrunk",
        "shrink",
        "c=2",
        "c=0",
        "\\(c=2\\)",
        "\\(c=0\\)",
        "\\texttt{bootstraps}=0",
        "plug-in selected score-contrast variance",
    )
    for token in forbidden_method_tokens:
        if token in manuscript_text:
            raise SystemExit(f"manuscript still contains removed scalar-damping token: {token}")
    banned_words = ("deliberately", "frozen")
    for tex_path in paper_root.rglob("*.tex"):
        text = tex_path.read_text(errors="replace")
        lower_text = text.lower()
        for word in banned_words:
            if re.search(rf"\b{re.escape(word)}\b", lower_text):
                raise SystemExit(
                    f"paper text contains banned wording {word!r}: {tex_path}"
                )
        for token in forbidden_method_tokens:
            if token.lower() in lower_text:
                raise SystemExit(
                    f"paper text contains removed scalar-damping token {token!r}: "
                    f"{tex_path}"
                )

    definitions = set(re.findall(r"\\newcommand\{\\(SFour[A-Za-z]+)\}", values))
    generated_text = "\n".join((overview, family, synthetic, diagnostic))
    uses = set(re.findall(r"\\(SFour[A-Za-z]+)", section4 + "\n" + generated_text))
    undefined = uses - definitions
    if undefined:
        raise SystemExit(f"undefined generated Section 4 macros: {sorted(undefined)}")
    return len(definitions), len(uses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--paper-root", required=True, type=Path)
    args = parser.parse_args()

    verification = verify_release(args.data_root)
    compare_generated(args.data_root, args.paper_root)
    auxiliary_files = compare_auxiliary_generated(args.data_root, args.paper_root)
    benchmark_releases = verify_benchmark_releases(args.data_root)
    definitions, uses = verify_manuscript(args.paper_root)
    print(
        "VERIFIED "
        f"release={RELEASE_DIR} "
        f"benchmark_releases={benchmark_releases} "
        f"reps_files={verification['reps_files']} "
        f"replication_rows={verification['replication_rows']} "
        f"expert_settings={verification['cells']} "
        f"generated_files={len(GENERATED_FILES)} "
        f"auxiliary_generated_files={auxiliary_files} "
        f"macros_defined={definitions} "
        f"macros_used={uses}"
    )


if __name__ == "__main__":
    main()
