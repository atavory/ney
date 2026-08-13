#!/usr/bin/env python3
"""Deterministic launcher for preregistered D2 Cui-published DGP."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import datetime as dt
import hashlib
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


DESIGNS = ("cui_published_scenario1", "cui_published_scenario2")
NS = (250, 500, 1000, 2000)


@dataclass(frozen=True)
class Job:
    index: int
    group: str
    design: str
    method: str
    n: int
    strength: float
    chunk: int
    seed: int
    out: Path
    rep_out: Path
    log: Path
    command: tuple[str, ...]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_count(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open() as handle:
        return max(0, sum(1 for _ in handle) - 1)


def build_jobs(args) -> list[Job]:
    jobs: list[Job] = []
    index = 0
    for design in DESIGNS:
        for n in NS:
            for method in args.methods:
                for chunk in range(args.chunks):
                    index += 1
                    seed = args.seed_base + index * 100_000
                    if seed >= 2**32:
                        raise ValueError(f"seed exceeds uint32 range: {seed}")
                    stem = (
                        f"{args.owner}_d2_{design}_{method}_n{n}_"
                        f"chunk{chunk:02d}"
                    )
                    out = args.run_dir / f"{stem}.csv"
                    rep_out = args.run_dir / f"{stem}.reps.csv"
                    log = args.run_dir / f"{stem}.log"
                    fixed_floor = (
                        ("--tau-grid", "0.05")
                        if method in {"tmle", "aipw"}
                        else ()
                    )
                    repair_mode = (
                        args.repair_mode
                        if method in {"tmle", "aipw"}
                        else "if_residual"
                    )
                    command = (
                        str(args.python),
                        str(args.wrapper),
                        "--frozen-source",
                        str(args.frozen_source),
                        "--reps",
                        str(args.reps_per_chunk),
                        "--bootstraps",
                        str(args.bootstraps),
                        "--n",
                        str(n),
                        "--epsilon",
                        "0.05",
                        "--strength",
                        "1.0",
                        "--design",
                        design,
                        "--mar-design",
                        "box",
                        "--analysis-region",
                        "estimated_residual_lowp_supported",
                        "--region-quantile",
                        "0.10",
                        "--region-min-observed",
                        "30",
                        "--reference-method",
                        method,
                        *fixed_floor,
                        "--propensity-mode",
                        "estimated",
                        "--learner",
                        args.learner,
                        "--propensity-learner",
                        args.learner,
                        "--repair-mode",
                        repair_mode,
                        "--region-damp-grid",
                        *(str(value) for value in args.region_damp_grid),
                        "--validation-risk",
                        "aipw_variance",
                        "--validation-loss-se",
                        str(args.validation_loss_se),
                        "--c",
                        "2.0",
                        "--folds",
                        "3",
                        "--selector",
                        "obsval",
                        "--lepski-c",
                        "4",
                        "--region-detector-c",
                        str(args.region_detector_c),
                        "--region-selector-ablation",
                        "crossfit_rank_empty_standdown",
                        "--seed",
                        str(seed),
                        "--progress-every",
                        "1",
                        "--out",
                        str(out),
                        "--rep-out",
                        str(rep_out),
                    )
                    jobs.append(
                        Job(
                            index=index,
                            group="cui_published",
                            design=design,
                            method=method,
                            n=n,
                            strength=1.0,
                            chunk=chunk,
                            seed=seed,
                            out=out,
                            rep_out=rep_out,
                            log=log,
                            command=command,
                        )
                    )
    return jobs


def write_manifest(path: Path, jobs: list[Job]) -> None:
    fields = [
        "index",
        "group",
        "design",
        "method",
        "n",
        "strength",
        "chunk",
        "seed",
        "out",
        "rep_out",
        "log",
        "command_json",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    "index": job.index,
                    "group": job.group,
                    "design": job.design,
                    "method": job.method,
                    "n": job.n,
                    "strength": job.strength,
                    "chunk": job.chunk,
                    "seed": job.seed,
                    "out": job.out,
                    "rep_out": job.rep_out,
                    "log": job.log,
                    "command_json": json.dumps(job.command),
                }
            )


def run_job(job: Job, reps_per_chunk: int):
    if row_count(job.rep_out) >= reps_per_chunk:
        return job, 0, "already_complete", 0.0
    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    started = time.monotonic()
    with job.log.open("w") as handle:
        handle.write(json.dumps({"command": job.command}) + "\n")
        handle.flush()
        result = subprocess.run(
            job.command,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.monotonic() - started
    if result.returncode == 0 and row_count(job.rep_out) < reps_per_chunk:
        return job, 97, "short_rep_file", elapsed
    return job, result.returncode, "ran", elapsed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--frozen-source", required=True, type=Path)
    parser.add_argument("--wrapper", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--owner", default="dml2")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["aipw", "tmle", "ctmle", "cui_selective_ml"],
        default=["ctmle", "cui_selective_ml"],
    )
    parser.add_argument("--learner", choices=["xgboost"], default="xgboost")
    parser.add_argument("--chunks", type=int, default=24)
    parser.add_argument("--reps-per-chunk", type=int, default=4)
    parser.add_argument("--bootstraps", type=int, default=50)
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--seed-base", type=int, default=2_050_000_000)
    parser.add_argument("--region-detector-c", type=float, default=1.0)
    parser.add_argument(
        "--repair-mode",
        choices=("if_projection", "regional_if_residual"),
        default="if_projection",
    )
    parser.add_argument("--validation-loss-se", type=float, default=1.0)
    parser.add_argument(
        "--region-damp-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 1.0],
    )
    parser.add_argument("--manifest-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    args.frozen_source = args.frozen_source.resolve()
    args.wrapper = args.wrapper.resolve()
    args.python = args.python.expanduser().absolute()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    xgboost_version = subprocess.check_output(
        [str(args.python), "-c", "import xgboost; print(xgboost.__version__)"],
        text=True,
    ).strip()
    if xgboost_version != "3.4.0":
        raise SystemExit(f"expected xgboost 3.4.0, got {xgboost_version!r}")
    jobs = build_jobs(args)
    manifest = args.run_dir / "manifest.tsv"
    write_manifest(manifest, jobs)
    provenance = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "design": "D2 Cui--Tchetgen published DGP",
        "paper": "Cui and Tchetgen Tchetgen, arXiv:1911.02029v6, Section 7",
        "mapping": "E[Y(1)] is run as a MAR mean with R=A, pi(X)=Pr(A=1|X), Y=Y(1), theta=E[Y(1)].",
        "x_distribution": "iid Uniform(0,1)^5",
        "propensity_logit": "(1,-1,1,-1,1)' f(X)",
        "outcome_mu": "E[Y|A,X]=2*(1+1'f(X)+1'f(X)*A+A); target uses A=1",
        "scenarios": {
            "cui_published_scenario1": "f_j(x)=1/(1+exp(-20*(x_j-0.5)))",
            "cui_published_scenario2": "f_j(x)=x_j^2",
        },
        "methods": args.methods,
        "n_values": NS,
        "chunks": args.chunks,
        "reps_per_chunk": args.reps_per_chunk,
        "bootstraps": args.bootstraps,
        "max_workers": args.max_workers,
        "seed_base": args.seed_base,
        "learner": args.learner,
        "xgboost_version": xgboost_version,
        "region_selector_ablation": "crossfit_rank_empty_standdown",
        "region_detector_c": args.region_detector_c,
        "region_damp_grid": args.region_damp_grid,
        "repair_mode": args.repair_mode,
        "validation_risk": "aipw_variance",
        "validation_loss_se": args.validation_loss_se,
        "shrink_c": 2.0,
        "frozen_source": str(args.frozen_source),
        "frozen_source_sha256": sha256(args.frozen_source),
        "wrapper": str(args.wrapper),
        "wrapper_sha256": sha256(args.wrapper),
        "launcher": str(Path(__file__).resolve()),
        "launcher_sha256": sha256(Path(__file__).resolve()),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "manifest_only": args.manifest_only,
    }
    (args.run_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    if args.manifest_only:
        print(f"wrote {len(jobs)} jobs to {manifest}")
        return
    status = {
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "total_jobs": len(jobs),
        "completed_jobs": 0,
        "failed_jobs": 0,
        "results": [],
    }
    lock = threading.Lock()
    events_path = args.run_dir / "events.jsonl"
    status_path = args.run_dir / "status.json"

    def write_status() -> None:
        status["updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        tmp = status_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        tmp.replace(status_path)

    def event(kind: str, **payload) -> None:
        record = {
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "kind": kind,
            **payload,
        }
        with events_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def record(result) -> None:
        job, returncode, disposition, elapsed = result
        with lock:
            if returncode == 0:
                status["completed_jobs"] += 1
            else:
                status["failed_jobs"] += 1
            payload = {
                "index": job.index,
                "design": job.design,
                "method": job.method,
                "n": job.n,
                "chunk": job.chunk,
                "seed": job.seed,
                "returncode": int(returncode),
                "disposition": disposition,
                "elapsed_seconds": float(elapsed),
                "out": str(job.out),
                "rep_out": str(job.rep_out),
                "log": str(job.log),
            }
            status["results"].append(payload)
            event("job_end", **payload)
            write_status()

    write_status()
    event("runner_start", total_jobs=len(jobs), max_workers=args.max_workers)
    with futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        pending = []
        for job in jobs:
            event(
                "job_start",
                index=job.index,
                design=job.design,
                method=job.method,
                n=job.n,
                chunk=job.chunk,
                seed=job.seed,
            )
            pending.append(executor.submit(run_job, job, args.reps_per_chunk))
        for future in futures.as_completed(pending):
            record(future.result())
    status["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_status()
    event(
        "runner_exit",
        completed_jobs=status["completed_jobs"],
        failed_jobs=status["failed_jobs"],
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    if status["failed_jobs"] or status["completed_jobs"] != len(jobs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
