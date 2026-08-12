#!/usr/bin/env python3
"""Assemble the authoritative Section 4 release and manuscript value table.

All numerical values are selected from committed, manifest-bound summaries.
The script records the exact source row for every manuscript-facing value and
fails if expected coverage or exact-stand-down identities do not hold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"empty source CSV: {path}")
    return rows


def unique(path: Path, **selector: str) -> dict[str, str]:
    matches = [
        row
        for row in csv_rows(path)
        if all(row.get(key) == value for key, value in selector.items())
    ]
    if len(matches) != 1:
        raise SystemExit(f"{path}: selector {selector} matched {len(matches)} rows")
    return matches[0]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def macro_stem(key: str) -> str:
    words = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
             "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    parts = []
    for token in key.split("_"):
        token = re.sub(r"\d", lambda match: words[match.group()], token)
        parts.append(token[:1].upper() + token[1:])
    stem = "SFour" + "".join(parts)
    if not stem.isalpha():
        raise SystemExit(f"invalid generated TeX macro stem: {stem}")
    return stem


def main() -> None:
    args = arguments()
    support = args.data_root / "support_csv"
    bundles = {
        "core": support / "dml_section4_confirmatory_20260810_v1",
        "cui_published": support / "dml_section4_cui_published_projection_20260811_v1",
        "aligned_real": support / "dml_section4_aligned_real_20260811_v1",
        "real_safety": support / "dml_section4_wider_partial_real_20260810_v1",
        "d0_anchor": support / "dml_section4_region_local_anchor_20260811_v1",
        "ks_tmle_aipw": support / "dml_section4_ks_tmle_aipw_20260811_v1",
        "tmle_projection": support / "dml_section4_tmle_projection_20260811_v1",
    }
    for name, bundle in bundles.items():
        if not bundle.is_dir():
            raise SystemExit(f"missing bundle {name}: {bundle}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if any(args.out_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.out_dir}")

    source_map: list[dict[str, object]] = []
    expected_portable = {
        "real_safety": (288, 1152),
        "d0_anchor": (192, 768),
        "ks_tmle_aipw": (384, 1536),
        "tmle_projection": (576, 2304),
    }
    for name, bundle in bundles.items():
        verification = bundle / "verification.json"
        jobs = rows = status = "legacy-certified"
        if verification.is_file():
            record = json.loads(verification.read_text())
            status = record.get("status", record.get("bundle_status", "unknown"))
            jobs, rows = record.get("jobs", ""), record.get("rows", "")
            if name in expected_portable:
                expected_jobs, expected_rows = expected_portable[name]
                if status != "COMPLETE" or (jobs, rows) != (expected_jobs, expected_rows):
                    raise SystemExit(
                        f"{name}: expected COMPLETE/{expected_jobs}/{expected_rows}, "
                        f"observed {status}/{jobs}/{rows}"
                    )
        source_map.append(
            {
                "bundle": name,
                "relative_path": str(bundle.relative_to(args.data_root)),
                "status": status,
                "jobs": jobs,
                "rows": rows,
                "verification_sha256": sha256(verification) if verification.is_file() else "",
            }
        )

    values: list[dict[str, object]] = []

    def add(
        key: str,
        block: str,
        design: str,
        method: str,
        cells: int,
        gain: float,
        lo: float,
        hi: float,
        bundle: str,
        file: str,
        selector: str,
    ) -> None:
        values.append(
            {
                "key": key,
                "block": block,
                "design": design,
                "method": method,
                "cells": cells,
                "gain": f"{gain:.17g}",
                "ci_lo": f"{lo:.17g}",
                "ci_hi": f"{hi:.17g}",
                "gain_pct": f"{100 * gain:.3f}",
                "ci_lo_pct": f"{100 * lo:.3f}",
                "ci_hi_pct": f"{100 * hi:.3f}",
                "source_bundle": bundle,
                "source_file": file,
                "source_selector": selector,
                "source_sha256": sha256(bundles[bundle] / file),
            }
        )

    ma_path = bundles["core"] / "ma_xgboost_summary.json"
    ma = json.loads(ma_path.read_text())
    for dgp in ("2", "3"):
        row = ma["cells"][dgp]
        add(
            f"ma_dgp{dgp}", "natural", f"Ma DGP {dgp}", "ma_dr_bc", 1,
            row["gain"], row["gain_lo"], row["gain_hi"], "core",
            ma_path.name, f"cells.{dgp}",
        )
    row = ma["family"]
    add(
        "ma_family", "natural", "Ma family", "ma_dr_bc", 2,
        row["gain"], row["gain_lo"], row["gain_hi"], "core",
        ma_path.name, "family",
    )

    sensitivity = bundles["core"] / "section4_c_sensitivity.csv"
    for method, key in (("ctmle", "ks_ctmle"), ("cui_selective_ml", "ks_cui")):
        row = unique(sensitivity, method=method, family="kang_schafer", c="2")
        add(
            key, "natural", "Kang--Schafer", method, int(row["cells"]),
            float(row["equal_cell_relative_mse_reduction"]), float(row["ci_lo"]),
            float(row["ci_hi"]), "core", sensitivity.name,
            f"method={method};family=kang_schafer;c=2",
        )

    ks_extra = bundles["ks_tmle_aipw"] / "family_summary.csv"
    for method, key in (("aipw", "ks_aipw"), ("tmle", "ks_tmle_residual_negative")):
        row = unique(ks_extra, method=method, family="kang_schafer")
        add(
            key, "natural" if method == "aipw" else "diagnostic",
            "Kang--Schafer", method, int(row["cells"]),
            float(row["equal_cell_relative_mse_reduction"]), float(row["ci_lo"]),
            float(row["ci_hi"]), "ks_tmle_aipw", ks_extra.name,
            f"method={method};family=kang_schafer",
        )

    cui_pub = bundles["cui_published"] / "family_summary.csv"
    row = unique(cui_pub, method="aipw", family="suite")
    add(
        "cui_scenarios_aipw", "natural", "Cui scenarios", "aipw", int(row["cells"]),
        float(row["equal_cell_relative_mse_reduction"]), float(row["ci_lo"]),
        float(row["ci_hi"]), "cui_published", cui_pub.name,
        "method=aipw;family=suite",
    )

    projection = bundles["tmle_projection"] / "family_summary.csv"
    projection_rows = [row for row in csv_rows(projection) if row["method"] == "tmle"]
    if sum(int(row["cells"]) for row in projection_rows) != 16 or any(
        any(float(row[field]) != 0 for field in ("equal_cell_relative_mse_reduction", "ci_lo", "ci_hi"))
        for row in projection_rows
    ):
        raise SystemExit("plain-TMLE projection is not exact stand-down in all 16 cells")
    add(
        "tmle_projection_16", "natural", "Kang--Schafer + Cui scenarios", "tmle",
        16, 0, 0, 0, "tmle_projection", projection.name, "method=tmle;all families",
    )

    real = bundles["real_safety"] / "family_summary.csv"
    for method, key in (("ctmle", "real_safety_ctmle"), ("cui_selective_ml", "real_safety_cui")):
        row = unique(real, method=method, family="real")
        add(
            key, "natural", "real covariates, wider/partial", method, int(row["cells"]),
            float(row["equal_cell_relative_mse_reduction"]), float(row["ci_lo"]),
            float(row["ci_hi"]), "real_safety", real.name,
            f"method={method};family=real",
        )

    d0 = bundles["d0_anchor"] / "family_summary.csv"
    for method, key in (("ctmle", "d0_signal_ctmle"), ("cui_selective_ml", "d0_signal_cui")):
        row = unique(d0, method=method, family="d0_signal")
        add(
            key, "emphasized", "aligned anchor, signal", method, int(row["cells"]),
            float(row["equal_cell_relative_mse_reduction"]), float(row["ci_lo"]),
            float(row["ci_hi"]), "d0_anchor", d0.name,
            f"method={method};family=d0_signal",
        )
    row = unique(d0, method="ctmle", family="d0_null")
    add(
        "d0_null_ctmle", "emphasized", "aligned anchor, null", "ctmle", 1,
        float(row["equal_cell_relative_mse_reduction"]), float(row["ci_lo"]),
        float(row["ci_hi"]), "d0_anchor", d0.name, "method=ctmle;family=d0_null",
    )

    core_family = bundles["core"] / "section4_family_summary.csv"
    aligned = unique(core_family, method="cui_selective_ml", family="alignment_aligned")
    add(
        "placement_aligned_cui", "emphasized", "placement stress, aligned",
        "cui_selective_ml", int(aligned["cells"]),
        float(aligned["equal_cell_relative_mse_reduction"]), float(aligned["ci_lo"]),
        float(aligned["ci_hi"]), "core", core_family.name,
        "method=cui_selective_ml;family=alignment_aligned",
    )
    zero_groups = {
        "placement_all_ctmle": ("ctmle", ["alignment_aligned", "alignment_partial", "alignment_disjoint", "alignment_null"], 10),
        "placement_other_cui": ("cui_selective_ml", ["alignment_partial", "alignment_disjoint", "alignment_null"], 7),
    }
    for key, (method, families, cells) in zero_groups.items():
        rows = [unique(core_family, method=method, family=family) for family in families]
        if sum(int(row["cells"]) for row in rows) != cells or any(
            any(float(row[field]) != 0 for field in ("equal_cell_relative_mse_reduction", "ci_lo", "ci_hi"))
            for row in rows
        ):
            raise SystemExit(f"{key}: expected exact stand-down")
        add(
            key, "emphasized", "placement stress", method, cells, 0, 0, 0,
            "core", core_family.name, f"method={method};families={'+'.join(families)}",
        )

    aligned_real = bundles["aligned_real"] / "bootstrap_summary.csv"
    for method in ("ctmle", "cui_selective_ml"):
        for family, suffix in (("nonzero_strengths", "signal"), ("strength0", "null")):
            row = unique(
                aligned_real, error_column="shrink_error", reference_method=method, family=family
            )
            add(
                f"aligned_real_{suffix}_{method}", "emphasized",
                f"aligned real covariates, {suffix}", method, int(row["cells"]),
                float(row["equal_cell_gain_pct"]) / 100,
                float(row["ci95_low_pct"]) / 100,
                float(row["ci95_high_pct"]) / 100,
                "aligned_real", aligned_real.name,
                f"error_column=shrink_error;reference_method={method};family={family}",
            )

    write_csv(args.out_dir / "source_map.csv", source_map)
    write_csv(args.out_dir / "paper_values.csv", values)
    shutil_target = args.out_dir / "section4_c_sensitivity.csv"
    shutil_target.write_bytes(sensitivity.read_bytes())
    tex_lines = ["% Generated by scripts/assemble_section4_release.py; do not edit.\n"]
    for row in values:
        stem = macro_stem(str(row["key"]))
        for suffix, field in (
            ("Cells", "cells"), ("Gain", "gain_pct"),
            ("Lo", "ci_lo_pct"), ("Hi", "ci_hi_pct"),
        ):
            tex_lines.append(f"\\newcommand{{\\{stem}{suffix}}}{{{row[field]}}}\n")
        for suffix, field in (("GainTwo", "gain"), ("LoTwo", "ci_lo"), ("HiTwo", "ci_hi")):
            tex_lines.append(
                f"\\newcommand{{\\{stem}{suffix}}}{{{100 * float(row[field]):.2f}}}\n"
            )
    (args.out_dir / "section4_values.tex").write_text("".join(tex_lines))
    sensitivity_rows = csv_rows(sensitivity)
    sensitivity_tex = ["% Generated by scripts/assemble_section4_release.py; do not edit.\n"]
    for method, label in (("ctmle", "C-TMLE"), ("cui_selective_ml", "selective ML")):
        for c_value in ("0", "1", "2", "3", "4", "5", "6", "8"):
            row = unique(sensitivity, method=method, family="kang_schafer", c=c_value)
            sensitivity_tex.append(
                f"{label} & {c_value} & "
                f"${100 * float(row['equal_cell_relative_mse_reduction']):.2f}\\% "
                f"[{100 * float(row['ci_lo']):.2f},{100 * float(row['ci_hi']):.2f}]$ & "
                f"{float(row['mean_harm_rate']):.3f} & "
                f"{float(row['mean_activation']):.3f} \\\\\n"
            )
    (args.out_dir / "section4_sensitivity_rows.tex").write_text("".join(sensitivity_tex))
    (args.out_dir / "README.md").write_text(
        "# Authoritative Section 4 release\n\n"
        "This is the single entry point for the submission-facing Section 4 evidence. "
        "`source_map.csv` enumerates the component bundles and their verification hashes. "
        "`paper_values.csv` gives every manuscript-facing value, its exact source selector, "
        "and the source-file SHA-256. The generated TeX files are copied verbatim into the "
        "paper repository and must not be edited by hand.\n\n"
        "Rebuild into an empty directory with:\n\n"
        "```sh\npython3 scripts/assemble_section4_release.py --data-root . --out-dir REBUILT\n```\n"
    )
    verification = {
        "status": "COMPLETE",
        "source_bundles": len(bundles),
        "paper_values": len(values),
        "outputs": {
            name: sha256(args.out_dir / name)
            for name in (
                "source_map.csv", "paper_values.csv", "section4_c_sensitivity.csv",
                "section4_values.tex",
                "section4_sensitivity_rows.tex",
                "README.md",
            )
        },
    }
    (args.out_dir / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n"
    )
    files = sorted(path for path in args.out_dir.iterdir() if path.name != "SHA256SUMS")
    (args.out_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files)
    )
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
