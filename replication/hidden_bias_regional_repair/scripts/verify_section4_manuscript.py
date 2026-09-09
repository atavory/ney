#!/usr/bin/env python3
"""Verify the current EJS Section 4 artifacts against the manuscript."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


PRIMARY_RELEASE = "support_csv/dml_weighted_residual_gamma_selection_ablation_20260903"
CURRENT_GENERATED = (
    (
        PRIMARY_RELEASE,
        "section4_unified_family_table.tex",
        "section4_unified_family_table.tex",
    ),
    (
        PRIMARY_RELEASE,
        "section4_unified_summary_table.tex",
        "section4_unified_summary_table.tex",
    ),
    (
        PRIMARY_RELEASE,
        "section4_fixed_floor_tmle_diagnostic_table.tex",
        "section4_fixed_floor_tmle_diagnostic_table.tex",
    ),
    (
        "support_csv/dml_weighted_residual_high_response_placebo_ablation_20260903",
        "section4_weighted_residual_high_response_placebo_ablation_table.tex",
        "section4_high_response_placebo_ablation_table.tex",
    ),
    (
        "support_csv/dml_weighted_residual_mse_gain_by_rho_20260903_v2",
        "section4_response_bin_action_reward_figure.tex",
        "section4_response_bin_action_reward_figure.tex",
    ),
)
REQUIRED_RELEASES = (
    PRIMARY_RELEASE,
    "support_csv/dml_weighted_residual_mse_gain_by_rho_20260903_v2",
    "support_csv/dml_weighted_residual_high_response_placebo_ablation_20260903",
    "support_csv/dml_weighted_residual_augmented_gamma_grid_ablation_20260903",
    "support_csv/dml_weighted_residual_upstream_trust_gate_diagnostic_20260903_v2",
    "support_csv/dml_section3_bounds_diagnostic_20260908_theorem_fit_public_v1",
    "support_csv/dml_section3_bounds_diagnostic_fixed_floor_tmle_20260908_theorem_fit_public_v1",
    "support_csv/dml_selected_gamma_theorem_simulation_20260908_v2",
    "support_csv/dml_honest_split_selected_gamma_20260908_v2",
    "support_csv/dml_current_algorithm_selection_penalty_20260908_v1",
    "support_csv/dml_section4_same_sample_selection_penalty_20260908_v2",
)
PRIMARY_METHODS = (
    ("aipw", "AIPW", "aipw"),
    ("cui_selective_ml", "selective ML", "selective_ml"),
    ("ma_dr_bc", "Ma DR-BC", "ma_dr_bc"),
    ("ctmle", "C-TMLE", "c_tmle"),
)
EXPECTED_SUMMARY = {
    "aipw": (8.95, 20, 4, 11),
    "cui_selective_ml": (5.19, 16, 8, 8),
    "ma_dr_bc": (7.66, 18, 6, 7),
    "ctmle": (4.98, 9, 0, 8),
}


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def verify_primary_release(data_root: Path) -> dict[str, object]:
    release = data_root / PRIMARY_RELEASE
    if not release.exists():
        raise SystemExit(f"missing primary Section 4 release: {release}")
    checksum_count = verify_checksums(release)

    primary_rows = read_csv(release / "weighted_residual_gamma_primary_table.csv")
    tmle_rows = read_csv(release / "weighted_residual_gamma_fixed_floor_tmle_table.csv")
    if len(primary_rows) != 24:
        raise SystemExit(f"expected 24 primary benchmark rows, found {len(primary_rows)}")
    if len(tmle_rows) != 24:
        raise SystemExit(f"expected 24 fixed-floor TMLE rows, found {len(tmle_rows)}")

    for method, _label, stem in PRIMARY_METHODS:
        gains = [float(row[f"{stem}_gain_pct"]) for row in primary_rows]
        positive = sum(gain > 0.0 for gain in gains)
        negative = sum(gain < 0.0 for gain in gains)
        interval_above = sum(float(row[f"{stem}_lo_pct"]) > 0.0 for row in primary_rows)
        summary = (round(sum(gains) / len(gains), 2), positive, negative, interval_above)
        expected = EXPECTED_SUMMARY[method]
        if summary != expected:
            raise SystemExit(
                f"unexpected primary summary for {method}: {summary!r} != {expected!r}"
            )

    provenance = json.loads((release / "provenance.json").read_text())
    if provenance.get("diagnostic") != "gamma replay selected by held-out weighted residual loss":
        raise SystemExit("primary release does not identify weighted-residual selection")
    if provenance.get("primary_table_rows") != 24:
        raise SystemExit("primary release provenance does not record 24 table rows")
    if provenance.get("primary_table_methods") != [
        method for method, _label, _stem in PRIMARY_METHODS
    ]:
        raise SystemExit("primary release provenance has unexpected primary method order")
    return {"checksum_count": checksum_count, "primary_rows": len(primary_rows)}


def verify_required_releases(data_root: Path) -> int:
    count = 0
    for release_name in REQUIRED_RELEASES:
        release = data_root / release_name
        if not release.exists():
            raise SystemExit(f"missing required release: {release}")
        verify_checksums(release)
        verification_path = release / "verification.json"
        if verification_path.exists():
            verification = json.loads(verification_path.read_text())
            if verification.get("status") not in {"PASS", "COMPLETE"}:
                raise SystemExit(f"release verification is not passing: {release}")
        count += 1
    return count


def compare_generated(data_root: Path, paper_root: Path) -> int:
    count = 0
    for release_name, release_file, paper_file in CURRENT_GENERATED:
        release_path = data_root / release_name / release_file
        paper_path = paper_root / "sections" / "generated" / paper_file
        if not release_path.exists():
            raise SystemExit(f"missing generated release file: {release_path}")
        if not paper_path.exists():
            raise SystemExit(f"missing generated paper file: {paper_path}")
        if release_path.read_bytes() != paper_path.read_bytes():
            raise SystemExit(f"paper generated file differs from release: {paper_path}")
        count += 1
    return count


def verify_manuscript(paper_root: Path) -> int:
    section4 = "\n".join(
        (paper_root / path).read_text()
        for path in (
            "sections/experiments.tex",
            "sections/experiments/settings.tex",
            "sections/experiments/results.tex",
            "sections/experiments/diagnostics.tex",
        )
    )
    appendix = (paper_root / "appendices" / "empirical_checks.tex").read_text()
    generated_family = (
        paper_root / "sections/generated/section4_unified_family_table.tex"
    ).read_text()
    generated_summary = (
        paper_root / "sections/generated/section4_unified_summary_table.tex"
    ).read_text()

    required_section4_inputs = (
        "sections/generated/section4_unified_family_table",
        "sections/generated/section4_unified_summary_table",
        "sections/generated/section4_response_bin_action_reward_figure",
    )
    for name in required_section4_inputs:
        if f"\\input{{{name}}}" not in section4:
            raise SystemExit(f"Section 4 does not input generated artifact: {name}")
    if (
        "\\input{sections/generated/section4_fixed_floor_tmle_diagnostic_table}"
        not in appendix
    ):
        raise SystemExit("appendix does not input fixed-floor TMLE diagnostic table")
    forbidden_inputs = (
        "sections/generated/section4_values",
        "sections/generated/section4_unified_overview_table",
        "sections/generated/section4_synthetic_diagnostic_table",
        "sections/generated/section4_no_shrinkage_ablation_table",
    )
    for name in forbidden_inputs:
        if f"\\input{{{name}}}" in section4 + "\n" + appendix:
            raise SystemExit(f"manuscript still inputs obsolete artifact: {name}")

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
    if "fixed-floor TMLE &" in generated_family or "fixed-floor TMLE &" in generated_summary:
        raise SystemExit("primary Section 4 tables include fixed-floor TMLE")
    if "primary TMLE comparator" not in paper_text:
        raise SystemExit("manuscript does not identify C-TMLE as primary comparator")

    normalized_manuscript_text = re.sub(r"\s+", " ", paper_text)
    required_phrases = (
        "The selected candidate is the returned estimate.",
        "The selected candidate is the reported estimate",
        "same cross-fitted scores used for selection",
        "cross-fitted response-weighted residual loss",
    )
    for phrase in required_phrases:
        if phrase not in normalized_manuscript_text:
            raise SystemExit(f"manuscript does not record current rule: {phrase}")

    forbidden_method_tokens = (
        "section4_no_shrinkage_ablation_table",
        "no-shrinkage",
        "no shrinkage",
        "shrinkage",
        "shrunk",
        "c=2",
        "c=0",
        "\\(c=2\\)",
        "\\(c=0\\)",
        "\\texttt{bootstraps}=0",
        "plug-in selected score-contrast variance",
    )
    banned_words = ("deliberately", "frozen")
    checked = 0
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
        checked += 1
    return checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--paper-root", required=True, type=Path)
    args = parser.parse_args()

    primary = verify_primary_release(args.data_root)
    required_releases = verify_required_releases(args.data_root)
    generated_files = compare_generated(args.data_root, args.paper_root)
    tex_files = verify_manuscript(args.paper_root)
    print(
        "VERIFIED "
        f"primary_release={PRIMARY_RELEASE} "
        f"primary_rows={primary['primary_rows']} "
        f"checksum_files={primary['checksum_count']} "
        f"required_releases={required_releases} "
        f"generated_files={generated_files} "
        f"tex_files_checked={tex_files}"
    )


if __name__ == "__main__":
    main()
