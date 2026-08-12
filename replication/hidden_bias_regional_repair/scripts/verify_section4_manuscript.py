#!/usr/bin/env python3
"""Fail-closed verification of the Section 4 release against the manuscript."""

from __future__ import annotations

import argparse
import csv
import hashlib
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

    for line in (release / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = release / name
        if sha256(path) != expected:
            raise SystemExit(f"release checksum mismatch: {path}")

    generated = {
        "section4_values.tex": args.paper_root / "sections/generated/section4_values.tex",
        "section4_sensitivity_rows.tex": args.paper_root / "sections/generated/section4_sensitivity_rows.tex",
    }
    for name, paper_path in generated.items():
        if (release / name).read_bytes() != paper_path.read_bytes():
            raise SystemExit(f"paper generated file differs from release: {paper_path}")

    manuscript = (args.paper_root / "sections/experiments_rule_quality.tex").read_text()
    definitions = set(re.findall(r"\\newcommand\{\\(SFour[A-Za-z]+)\}", generated["section4_values.tex"].read_text()))
    uses = set(re.findall(r"\\(SFour[A-Za-z]+)", manuscript))
    undefined = uses - definitions
    if undefined:
        raise SystemExit(f"undefined generated Section 4 macros: {sorted(undefined)}")
    if "\\input{sections/generated/section4_sensitivity_rows}" not in manuscript:
        raise SystemExit("manuscript does not input generated c-sensitivity rows")

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
    )


if __name__ == "__main__":
    main()
