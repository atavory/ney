#!/usr/bin/env python3
"""Launch paired global/regional residual shards for published Ma DGPs."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import datetime as dt
import hashlib
import json
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Job:
    index: int
    adapter: str
    dgp: int
    chunk: int
    seed_argument: int
    out: Path
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


def build_jobs(args: argparse.Namespace) -> list[Job]:
    jobs: list[Job] = []
    for adapter in args.adapters:
        for dgp in args.dgps:
            for chunk in range(args.chunks):
                index = len(jobs) + 1
                # ma_published_did_projection.py adds 100000*dgp+rep. The
                # argument below depends only on chunk, never on adapter.
                seed_argument = args.seed_base + chunk * args.reps_per_chunk
                stem = f"ma_{adapter}_dgp{dgp}_chunk{chunk:02d}"
                out = args.run_dir / f"{stem}.csv"
                log = args.run_dir / f"{stem}.log"
                command = (
                    str(args.python),
                    str(args.source),
                    "--reps",
                    str(args.reps_per_chunk),
                    "--n",
                    str(args.n),
                    "--dgp",
                    str(dgp),
                    "--folds",
                    "3",
                    "--adapter",
                    adapter,
                    "--projection-learner",
                    "xgboost",
                    "--gamma-se",
                    "2.83",
                    "--gammas",
                    "0",
                    ".01",
                    ".025",
                    ".05",
                    ".1",
                    ".25",
                    ".5",
                    "1",
                    "--seed",
                    str(seed_argument),
                    "--out",
                    str(out),
                )
                jobs.append(
                    Job(index, adapter, dgp, chunk, seed_argument, out, log, command)
                )
    return jobs


def write_manifest(path: Path, jobs: list[Job]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "index",
                "adapter",
                "dgp",
                "chunk",
                "seed_argument",
                "out",
                "log",
                "command",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    "index": job.index,
                    "adapter": job.adapter,
                    "dgp": job.dgp,
                    "chunk": job.chunk,
                    "seed_argument": job.seed_argument,
                    "out": job.out,
                    "log": job.log,
                    "command": json.dumps(job.command),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument(
        "--adapters",
        nargs="+",
        choices=("global_residual", "regional_residual"),
        default=("global_residual", "regional_residual"),
    )
    parser.add_argument("--dgps", nargs="+", type=int, choices=(2, 3), default=(2, 3))
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--reps-per-chunk", type=int, default=48)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--seed-base", type=int, default=4_100_000_000)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    args.source = args.source.resolve()
    args.python = args.python.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args)
    manifest = args.run_dir / "manifest.tsv"
    write_manifest(manifest, jobs)
    provenance = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "launcher": str(Path(__file__).resolve()),
        "launcher_sha256": sha256(Path(__file__).resolve()),
        "manifest_sha256": sha256(manifest),
        "adapters": list(args.adapters),
        "dgps": list(args.dgps),
        "chunks": args.chunks,
        "reps_per_chunk": args.reps_per_chunk,
        "n": args.n,
        "seed_base": args.seed_base,
        "max_workers": args.max_workers,
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

    def write_status() -> None:
        status["updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        (args.run_dir / "status.json").write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n"
        )

    def run(job: Job) -> dict[str, object]:
        if row_count(job.out) == args.reps_per_chunk:
            return {
                "index": job.index,
                "adapter": job.adapter,
                "dgp": job.dgp,
                "chunk": job.chunk,
                "returncode": 0,
                "disposition": "already_complete",
            }
        started = time.monotonic()
        with job.log.open("w") as handle:
            completed = subprocess.run(job.command, stdout=handle, stderr=subprocess.STDOUT)
        return {
            "index": job.index,
            "adapter": job.adapter,
            "dgp": job.dgp,
            "chunk": job.chunk,
            "returncode": completed.returncode,
            "disposition": "ran",
            "elapsed_seconds": time.monotonic() - started,
            "out": str(job.out),
            "log": str(job.log),
        }

    write_status()
    with futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {executor.submit(run, job): job for job in jobs}
        for future in futures.as_completed(future_map):
            result = future.result()
            with lock:
                status["results"].append(result)
                status["completed_jobs"] += 1
                status["failed_jobs"] += int(result["returncode"] != 0)
                write_status()
    status["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_status()
    print(json.dumps(status, indent=2, sort_keys=True))
    if status["failed_jobs"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
