#!/usr/bin/env python3
"""Fail-closed verification of the unified Section 4 source against the paper."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


RELEASE = "dml_unified_cartesian_global_residual_20260814"
EXPECTED_MANIFEST_SHA256 = (
    "65720b1fce2a24d55872ab9e008cf7ef62945b30b791272f1fdfe65280e2287f"
)
EXPECTED_RAW_REPS_SHA256 = (
    "08d0e7f95d71773fe54eb137107e73c9f0346955247432a8ebb0e0dd1d195e92"
)
EXPECTED_SOURCE_SHA256 = (
    "98987b31cf7c883d4776996ae7b28f7f1b9fe134d6da323e95250f00232842ce"
)
EXPECTED_WRAPPER_SHA256 = (
    "b1b08b9fc32b03e969f2f24ba7816a850de12336bbf6a335f092238715ccb332"
)
EXPECTED_GENERATED = (
    "section4_values.tex",
    "section4_unified_overview_table.tex",
    "section4_unified_family_table.tex",
    "section4_fixed_floor_tmle_diagnostic_table.tex",
)
EXPECTED_METHODS = {
    "aipw",
    "tmle",
    "ctmle",
    "cui_selective_ml",
    "ma_dr_bc",
}
EXPECTED_GROUPS = {"kang_schafer", "alignment", "real", "anchor"}
EXPECTED_FAMILIES = EXPECTED_GROUPS | {"all"}
EXPECTED_ALL_ROWS = {
    "aipw": {
        "gain": 0.0279144540494,
        "lo": 0.0195689399394,
        "hi": 0.0352814499466,
    },
    "tmle": {
        "gain": -0.00410353920309,
        "lo": -0.0120329864572,
        "hi": 0.00234623933187,
    },
    "ctmle": {
        "gain": 0.000283792467922,
        "lo": -0.000402899786557,
        "hi": 0.00118516548806,
    },
    "cui_selective_ml": {
        "gain": 0.0172335211175,
        "lo": 0.011839800464,
        "hi": 0.0225203999264,
    },
    "ma_dr_bc": {
        "gain": 0.0299899878989,
        "lo": 0.0233726249994,
        "hi": 0.0365564675682,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_close(label: str, observed: str, expected: float) -> None:
    if abs(float(observed) - expected) > 5e-13:
        raise SystemExit(f"{label} differs: {observed} != {expected}")


def verify_checksums(release: Path) -> int:
    checksum_path = release / "SHA256SUMS"
    if not checksum_path.exists():
        raise SystemExit(f"missing release checksum file: {checksum_path}")
    count = 0
    for line in checksum_path.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = release / name
        if not path.exists():
            raise SystemExit(f"checksum target missing: {path}")
        observed = sha256(path)
        if observed != expected:
            raise SystemExit(f"release checksum mismatch: {path}")
        count += 1
    return count


def verify_reconstruction_json(path: Path) -> dict[str, object]:
    verification = json.loads(path.read_text())
    if verification.get("status") != "PASS":
        raise SystemExit("unified reconstruction status is not PASS")
    expected_top_level = {
        "cells": 170,
        "replication_rows": 16320,
        "reps_files": 4080,
        "draws": 20000,
        "seed": 20260814,
        "full_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "raw_reps_manifest_sha256": EXPECTED_RAW_REPS_SHA256,
    }
    for key, expected in expected_top_level.items():
        if verification.get(key) != expected:
            raise SystemExit(
                f"unexpected unified verification {key}: "
                f"{verification.get(key)!r} != {expected!r}"
            )
    config = verification.get("config")
    if not isinstance(config, dict):
        raise SystemExit("unified verification lacks config object")
    expected_config = {
        "repair_mode": "if_residual",
        "validation_loss_se": 1.0,
        "shrink_c": 2.0,
        "region_damp_grid": [0.0, 0.25, 0.5, 1.0],
        "bootstraps": 0,
        "chunks": 24,
        "reps_per_chunk": 4,
        "seed_base": 1800000000,
        "xgboost_version": "3.4.0",
        "frozen_source_sha256": EXPECTED_SOURCE_SHA256,
        "wrapper_sha256": EXPECTED_WRAPPER_SHA256,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise SystemExit(
                f"unexpected unified config {key}: "
                f"{config.get(key)!r} != {expected!r}"
            )
    return verification


def verify_family_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 25:
        raise SystemExit(f"family summary has {len(rows)} rows, expected 25")
    keyed = {(row["method"], row["family"]): row for row in rows}
    expected_keys = {
        (method, family)
        for method in EXPECTED_METHODS
        for family in EXPECTED_FAMILIES
    }
    observed_keys = set(keyed)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise SystemExit(f"family summary keys differ: missing={missing}, extra={extra}")
    for method, expected in EXPECTED_ALL_ROWS.items():
        row = keyed[(method, "all")]
        if int(row["cells"]) != 34:
            raise SystemExit(f"{method} all-family cell count is not 34")
        if int(row["reps"]) != 3264:
            raise SystemExit(f"{method} all-family replication count is not 3264")
        require_close(f"{method} all gain", row["equal_cell_gain"], expected["gain"])
        require_close(
            f"{method} all lower CI",
            row["equal_cell_gain_ci_low"],
            expected["lo"],
        )
        require_close(
            f"{method} all upper CI",
            row["equal_cell_gain_ci_high"],
            expected["hi"],
        )
    return rows


def verify_cell_summary(path: Path) -> int:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 170:
        raise SystemExit(f"cell summary has {len(rows)} rows, expected 170")
    if sum(int(row["reps"]) for row in rows) != 16320:
        raise SystemExit("cell summary replication count is not 16,320")
    if {row["method"] for row in rows} != EXPECTED_METHODS:
        raise SystemExit("cell summary method set differs from the unified protocol")
    if {row["group"] for row in rows} != EXPECTED_GROUPS:
        raise SystemExit("cell summary group set differs from the unified protocol")
    return len(rows)


def verify_generated_outputs(data_root: Path, release: Path, paper_root: Path) -> None:
    script = data_root / "scripts/assemble_section4_unified_global_residual.py"
    if not script.exists():
        raise SystemExit(f"missing unified assembler script: {script}")
    with tempfile.TemporaryDirectory(prefix="section4-unified-") as tmp:
        rebuilt = Path(tmp)
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--summary",
                str(release / "family_summary.csv"),
                "--out-dir",
                str(rebuilt),
            ],
            check=True,
        )
        for name in EXPECTED_GENERATED:
            rebuilt_path = rebuilt / name
            release_path = release / name
            paper_path = paper_root / "sections/generated" / name
            if rebuilt_path.read_bytes() != release_path.read_bytes():
                raise SystemExit(f"release generated file is stale: {release_path}")
            if release_path.read_bytes() != paper_path.read_bytes():
                raise SystemExit(f"paper generated file differs from release: {paper_path}")


def verify_manuscript(paper_root: Path, release: Path) -> int:
    manuscript = (paper_root / "sections/experiments_rule_quality.tex").read_text()
    appendix = (paper_root / "appendices/empirical_checks.tex").read_text()
    if RELEASE not in manuscript:
        raise SystemExit(f"manuscript does not name {RELEASE}")
    required_inputs = (
        "section4_values",
        "section4_unified_overview_table",
        "section4_unified_family_table",
    )
    for name in required_inputs:
        if f"\\input{{sections/generated/{name}}}" not in manuscript:
            raise SystemExit(f"manuscript does not input generated file: {name}")
    diagnostic_input = "section4_fixed_floor_tmle_diagnostic_table"
    if f"\\input{{sections/generated/{diagnostic_input}}}" not in appendix:
        raise SystemExit("appendix does not input fixed-floor TMLE diagnostic table")
    forbidden = (
        "section4_natural_table",
        "section4_emphasized_table",
        "section4_sensitivity_rows",
        "section4_c_atlas_20260812_v3",
        "dml_section4_release_20260812_v1",
        "score-projection adapter",
    )
    for token in forbidden:
        if token in manuscript:
            raise SystemExit(f"manuscript still references superseded Section 4 source: {token}")
    if "\\begin{table}" in manuscript:
        raise SystemExit("Section 4 contains a hand-maintained table environment")

    values = (release / "section4_values.tex").read_text()
    overview = (release / "section4_unified_overview_table.tex").read_text()
    family = (release / "section4_unified_family_table.tex").read_text()
    diagnostic = (
        release / "section4_fixed_floor_tmle_diagnostic_table.tex"
    ).read_text()
    if "fixed-floor TMLE &" in overview or "fixed-floor TMLE &" in family:
        raise SystemExit("primary generated tables still include fixed-floor TMLE")
    if "fixed-floor TMLE" not in diagnostic:
        raise SystemExit("diagnostic table does not identify fixed-floor TMLE")
    tables = "\n".join((overview, family, diagnostic))
    definitions = set(re.findall(r"\\newcommand\{\\(SFourUnified[A-Za-z]+)\}", values))
    uses = set(re.findall(r"\\(SFourUnified[A-Za-z]+)", manuscript + "\n" + tables))
    undefined = uses - definitions
    if undefined:
        raise SystemExit(f"undefined generated Section 4 macros: {sorted(undefined)}")
    return len(uses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--paper-root", required=True, type=Path)
    args = parser.parse_args()

    release = args.data_root / "support_csv" / RELEASE
    if not release.exists():
        raise SystemExit(f"missing unified Section 4 release: {release}")
    if not (args.data_root / "scripts/recreate_unified_cartesian_global_residual.py").exists():
        raise SystemExit("missing unified reconstruction script")

    release_files = verify_checksums(release)
    verify_reconstruction_json(release / "verification.json")
    family_rows = verify_family_summary(release / "family_summary.csv")
    cell_rows = verify_cell_summary(release / "cell_summary.csv")
    verify_generated_outputs(args.data_root, release, args.paper_root)
    macros_used = verify_manuscript(args.paper_root, release)

    print(
        f"VERIFIED unified_release_files={release_files} "
        f"family_rows={len(family_rows)} cell_rows={cell_rows} "
        f"macros_used={macros_used}"
    )


if __name__ == "__main__":
    main()
