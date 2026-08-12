#!/usr/bin/env python3
"""Materialize a portable, fail-closed Section 4 evidence bundle.

This is the only supported importer for completed Section 4 runs.  It accepts
one or more run directories and/or immutable ``.tar.zst`` archives, verifies
manifest coverage and per-job replication rows, and writes portable raw,
cell, and family CSVs.  No estimator is rerun and no result is entered by
hand.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", action="append", type=Path, default=[])
    parser.add_argument("--archive", action="append", type=Path, default=[])
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--expected-jobs", required=True, type=int)
    parser.add_argument("--reps-per-job", type=int, default=4)
    parser.add_argument("--expected-c", type=float, default=2.0)
    parser.add_argument("--nboot", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: str, source: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise SystemExit(f"non-finite value at {source}: {value!r}")
    return result


def safe_extract(archive: Path, destination: Path) -> None:
    if archive.name.endswith(".tar.zst"):
        listing = subprocess.run(
            ["tar", "--zstd", "-tf", str(archive)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        root = destination.resolve()
        for name in listing:
            target = (destination / name).resolve()
            if root not in target.parents and target != root:
                raise SystemExit(f"unsafe archive member: {name}")
        subprocess.run(
            ["tar", "--zstd", "-xf", str(archive), "-C", str(destination)],
            check=True,
        )
        return
    with tarfile.open(archive, "r:*") as handle:
        root = destination.resolve()
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise SystemExit(f"unsafe archive member: {member.name}")
        handle.extractall(destination)


def find_manifests(root: Path) -> list[Path]:
    preferred = sorted(root.rglob("manifest.tsv"))
    if preferred:
        return preferred
    return sorted(root.rglob("*manifest.resolved.tsv"))


def command_value(command: list[str], name: str) -> str | None:
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError):
        return None


def family_for(row: dict[str, str]) -> str:
    design = row["design"]
    strength = float(row["strength"])
    if design.startswith("kang_schafer_"):
        return "kang_schafer"
    if design.startswith("real_"):
        return "real"
    if design == "regional_shift":
        return "d0_null" if strength == 0 else "d0_signal"
    if design.startswith("alignment_"):
        return "alignment"
    return design


def bootstrap_gain(
    cells: list[tuple[np.ndarray, np.ndarray]], nboot: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.empty(nboot)
    for draw in range(nboot):
        gains = []
        for reference, repaired in cells:
            indices = rng.integers(0, len(reference), len(reference))
            ref_mse = np.mean(reference[indices] ** 2)
            rep_mse = np.mean(repaired[indices] ** 2)
            gains.append(1.0 - rep_mse / ref_mse)
        draws[draw] = np.mean(gains)
    return tuple(float(x) for x in np.quantile(draws, (0.025, 0.975)))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    if not rows:
        raise SystemExit(f"refusing to write empty CSV: {path}")
    fieldnames = fields or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = arguments()
    if not args.source_dir and not args.archive:
        raise SystemExit("provide --source-dir and/or --archive")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="section4_release_") as temporary:
        temp = Path(temporary)
        roots = [path.resolve() for path in args.source_dir]
        archive_hashes: dict[str, str] = {}
        for number, archive in enumerate(args.archive, start=1):
            archive = archive.resolve()
            extracted = temp / f"archive_{number}"
            extracted.mkdir()
            safe_extract(archive, extracted)
            roots.append(extracted)
            copied = args.out_dir / archive.name
            shutil.copy2(archive, copied)
            archive_hashes[copied.name] = sha256(copied)

        manifest_paths = [manifest for root in roots for manifest in find_manifests(root)]
        if not manifest_paths:
            raise SystemExit("no manifest found")
        rep_candidates: dict[str, list[Path]] = defaultdict(list)
        for root in roots:
            for path in root.rglob("*.reps.csv"):
                if path.stat().st_size:
                    rep_candidates[path.name].append(path)

        portable_manifest: list[dict[str, object]] = []
        raw_rows: list[dict[str, object]] = []
        raw_fields: list[str] = []
        seen_jobs: set[str] = set()
        for manifest_number, manifest in enumerate(manifest_paths, start=1):
            with manifest.open(newline="") as handle:
                jobs = list(csv.DictReader(handle, delimiter="\t"))
            for job_number, job in enumerate(jobs, start=1):
                original_index = job.get("index", str(job_number))
                job_id = f"m{manifest_number:02d}-j{int(original_index):04d}"
                if job_id in seen_jobs:
                    raise SystemExit(f"duplicate release job id: {job_id}")
                seen_jobs.add(job_id)
                rep_name = Path(job["rep_out"]).name
                candidates = rep_candidates.get(rep_name, [])
                if len(candidates) != 1:
                    raise SystemExit(
                        f"{job_id}: expected one {rep_name}, found {len(candidates)}"
                    )
                rep_path = candidates[0]
                command = json.loads(job.get("command_json", "[]"))
                command_c = command_value(command, "--c")
                command_reps = command_value(command, "--reps")
                if command_c is not None and float(command_c) != args.expected_c:
                    raise SystemExit(f"{job_id}: c={command_c}, expected {args.expected_c}")
                if command_reps is not None and int(command_reps) != args.reps_per_job:
                    raise SystemExit(
                        f"{job_id}: reps={command_reps}, expected {args.reps_per_job}"
                    )
                with rep_path.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                if len(rows) != args.reps_per_job:
                    raise SystemExit(
                        f"{job_id}: expected {args.reps_per_job} rows, found {len(rows)}"
                    )
                if sorted(int(row["rep"]) for row in rows) != list(
                    range(1, args.reps_per_job + 1)
                ):
                    raise SystemExit(f"{job_id}: invalid replication identifiers")
                for line_number, row in enumerate(rows, start=2):
                    source = f"{rep_path}:{line_number}"
                    for key in ("ref_error", "shrink_error"):
                        finite(row[key], source)
                    checks = {
                        "design": job["design"],
                        "reference_method": job["method"],
                        "n": job["n"],
                    }
                    for key, expected in checks.items():
                        if row[key] != expected:
                            raise SystemExit(
                                f"{source}: {key}={row[key]!r}, expected {expected!r}"
                            )
                    if float(row["strength"]) != float(job["strength"]):
                        raise SystemExit(f"{source}: strength disagrees with manifest")
                    if row.get("region_damp_grid") != "0.0|0.25|0.5|1.0":
                        raise SystemExit(f"{source}: wrong damping grid")
                    prefix = {
                        "release_job_id": job_id,
                        "source_manifest": manifest.name,
                        "source_rep_file": rep_path.name,
                    }
                    merged = {**prefix, **row}
                    raw_rows.append(merged)
                    for field in merged:
                        if field not in raw_fields:
                            raw_fields.append(field)
                portable_manifest.append(
                    {
                        "release_job_id": job_id,
                        "source_manifest": manifest.name,
                        "source_index": original_index,
                        "group": job.get("group", family_for(rows[0])),
                        "design": job["design"],
                        "method": job["method"],
                        "n": job["n"],
                        "strength": job["strength"],
                        "chunk": job.get("chunk", ""),
                        "seed": job.get("seed", ""),
                        "source_rep_file": rep_path.name,
                        "source_rep_sha256": sha256(rep_path),
                        "shrink_c": command_c or f"{args.expected_c:g}",
                    }
                )

        if len(portable_manifest) != args.expected_jobs:
            raise SystemExit(
                f"expected {args.expected_jobs} jobs, found {len(portable_manifest)}"
            )
        expected_rows = args.expected_jobs * args.reps_per_job
        if len(raw_rows) != expected_rows:
            raise SystemExit(f"expected {expected_rows} rows, found {len(raw_rows)}")

        write_csv(args.out_dir / "release_manifest.csv", portable_manifest)
        write_csv(args.out_dir / "raw_rows.csv", raw_rows, raw_fields)

        grouped: dict[tuple[str, str, str, int, float], list[dict[str, object]]] = defaultdict(list)
        for row in raw_rows:
            key = (
                str(row["reference_method"]),
                family_for(row),
                str(row["design"]),
                int(float(str(row["n"]))),
                float(str(row["strength"])),
            )
            grouped[key].append(row)
        cell_rows: list[dict[str, object]] = []
        arrays: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
        for (method, family, design, n, strength), rows in sorted(grouped.items()):
            reference = np.asarray([float(row["ref_error"]) for row in rows])
            repaired = np.asarray([float(row["shrink_error"]) for row in rows])
            ref_mse = float(np.mean(reference**2))
            rep_mse = float(np.mean(repaired**2))
            arrays[(method, family)].append((reference, repaired))
            cell_rows.append(
                {
                    "method": method,
                    "family": family,
                    "design": design,
                    "n": n,
                    "strength": f"{strength:g}",
                    "reps": len(rows),
                    "ref_mse": f"{ref_mse:.17g}",
                    "repaired_mse": f"{rep_mse:.17g}",
                    "relative_mse_reduction": f"{1 - rep_mse / ref_mse:.17g}",
                    "harm_rate": f"{float(np.mean(repaired**2 > reference**2)):.17g}",
                }
            )
        write_csv(args.out_dir / "cell_summary.csv", cell_rows)

        family_rows: list[dict[str, object]] = []
        for family_number, ((method, family), cells) in enumerate(sorted(arrays.items())):
            gains = [1.0 - np.mean(rep**2) / np.mean(ref**2) for ref, rep in cells]
            lo, hi = bootstrap_gain(cells, args.nboot, args.seed + family_number)
            family_rows.append(
                {
                    "method": method,
                    "family": family,
                    "cells": len(cells),
                    "reps_per_cell_min": min(len(ref) for ref, _ in cells),
                    "reps_per_cell_max": max(len(ref) for ref, _ in cells),
                    "equal_cell_relative_mse_reduction": f"{float(np.mean(gains)):.17g}",
                    "ci_lo": f"{lo:.17g}",
                    "ci_hi": f"{hi:.17g}",
                }
            )
        write_csv(args.out_dir / "family_summary.csv", family_rows)

        verification = {
            "status": "COMPLETE",
            "jobs": len(portable_manifest),
            "rows": len(raw_rows),
            "reps_per_job": args.reps_per_job,
            "expected_c": args.expected_c,
            "nboot": args.nboot,
            "seed": args.seed,
            "archive_sha256": archive_hashes,
            "outputs": {
                name: sha256(args.out_dir / name)
                for name in ("release_manifest.csv", "raw_rows.csv", "cell_summary.csv", "family_summary.csv")
            },
        }
        (args.out_dir / "verification.json").write_text(
            json.dumps(verification, indent=2, sort_keys=True) + "\n"
        )
        files = sorted(path for path in args.out_dir.iterdir() if path.name != "SHA256SUMS")
        (args.out_dir / "SHA256SUMS").write_text(
            "".join(f"{sha256(path)}  {path.name}\n" for path in files if path.is_file())
        )
        print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
