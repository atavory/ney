#!/usr/bin/env python3
"""Launch the locked Zhao/Kallus four-family comparison in deterministic chunks."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


CELLS = (
    "zhao24a_vanilla_y0",
    "zhao24a_vanilla_y1",
    "kallus23a_lognormal_y1_n200",
    "kallus23a_lognormal_y1_n800",
    "kallus23a_lognormal_y1_n3200",
)
METHODS = ("aipw", "cui_selective_ml", "ma_dr_bc", "ctmle")


@dataclass(frozen=True)
class Job:
    index: int
    cell: str
    method: str
    rep_start: int
    rep_stop: int
    output: str
    log: str
    command_json: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_jobs(args: argparse.Namespace) -> list[Job]:
    jobs: list[Job] = []
    runner = args.runner.resolve()
    upstream = args.upstream_source.resolve()
    for cell in CELLS:
        for method in METHODS:
            for rep_start in range(0, args.reps, args.chunk_size):
                rep_stop = min(args.reps, rep_start + args.chunk_size)
                stem = f"{cell}__{method}__r{rep_start:03d}_{rep_stop:03d}"
                output = args.run_dir / "chunks" / f"{stem}.csv"
                log = args.run_dir / "logs" / f"{stem}.log"
                command = (
                    str(args.python.absolute()), str(runner),
                    "--cell", cell, "--method", method,
                    "--rep-start", str(rep_start), "--rep-stop", str(rep_stop),
                    "--seed-base", str(args.seed_base),
                    "--upstream-source", str(upstream),
                    "--output", str(output),
                )
                jobs.append(Job(
                    index=len(jobs), cell=cell, method=method,
                    rep_start=rep_start, rep_stop=rep_stop,
                    output=str(output), log=str(log),
                    command_json=json.dumps(command),
                ))
    return jobs


def run_job(job: Job) -> dict[str, object]:
    command = tuple(json.loads(job.command_json))
    output = Path(job.output)
    log = Path(job.log)
    if output.exists():
        with output.open() as handle:
            rows = max(0, sum(1 for _ in handle) - 1)
        if rows == job.rep_stop - job.rep_start:
            return {"index": job.index, "returncode": 0, "status": "already_complete", "seconds": 0.0}
        return {"index": job.index, "returncode": 98, "status": "partial_output", "seconds": 0.0}
    environment = dict(os.environ)
    environment.update({
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    })
    started = time.monotonic()
    with log.open("x") as handle:
        handle.write(json.dumps({"command": command}) + "\n")
        handle.flush()
        result = subprocess.run(command, env=environment, stdout=handle, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started
    return {
        "index": job.index, "returncode": result.returncode,
        "status": "completed" if result.returncode == 0 else "failed",
        "seconds": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument(
        "--runner", type=Path,
        default=Path(__file__).with_name("dml_literature_sota_experiment.py"),
    )
    parser.add_argument(
        "--upstream-source", type=Path,
        default=Path(__file__).with_name("validated_reference_transfer.py"),
    )
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=202609160)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    if args.reps != 200:
        raise ValueError("locked design requires exactly 200 replications")
    if args.chunk_size <= 0 or args.reps % args.chunk_size:
        raise ValueError("chunk size must divide 200")
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / "chunks").mkdir()
    (args.run_dir / "logs").mkdir()
    jobs = build_jobs(args)
    write_tsv(args.run_dir / "manifest.tsv", [asdict(job) for job in jobs])
    versions = subprocess.check_output(
        [str(args.python.absolute()), "-c", "import json,numpy,sklearn,xgboost; print(json.dumps({'numpy':numpy.__version__,'sklearn':sklearn.__version__,'xgboost':xgboost.__version__},sort_keys=True))"],
        text=True,
        env=dict(os.environ),
    ).strip()
    provenance = {
        "status": "IN_PROGRESS" if not args.manifest_only else "MANIFEST_ONLY",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cells": list(CELLS), "methods": list(METHODS),
        "excluded_methods": ["tmle"],
        "reps_per_cell_method": args.reps,
        "chunk_size": args.chunk_size, "workers": args.workers,
        "seed_base": args.seed_base,
        "gamma_grid": [0.0, 0.25, 0.5, 1.0],
        "folds": 3, "validation_loss_se": 1.0,
        "response_model": "estimated", "learner": "xgboost",
        "aipw_floor": 0.05, "ma_floor": 0.05,
        "ctmle_floor_grid": [0.05, 0.10, 0.25, 0.50],
        "scalar_shrink": False,
        "python": str(args.python.absolute()), "versions": json.loads(versions),
        "runner": str(args.runner.resolve()), "runner_sha256": sha256(args.runner.resolve()),
        "launcher_sha256": sha256(Path(__file__).resolve()),
        "upstream_source": str(args.upstream_source.resolve()),
        "upstream_source_sha256": sha256(args.upstream_source.resolve()),
    }
    (args.run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    if args.manifest_only:
        return 0
    controller = args.run_dir / "controller.log"
    with controller.open("x") as handle:
        handle.write(f"start {provenance['created_utc']} jobs={len(jobs)} workers={args.workers}\n")
    results: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_job, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            with controller.open("a") as handle:
                handle.write(json.dumps(result, sort_keys=True) + "\n")
    results.sort(key=lambda row: int(row["index"]))
    write_tsv(args.run_dir / "job_status.tsv", results)
    failed = [row for row in results if int(row["returncode"]) != 0]
    provenance["status"] = "FAILED" if failed else "RAW_COMPLETE"
    provenance["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    provenance["failed_jobs"] = len(failed)
    (args.run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
