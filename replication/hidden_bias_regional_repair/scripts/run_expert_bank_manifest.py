#!/usr/bin/env python3
"""Resumable parallel runner for a frozen expert-bank manifest."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import threading
from datetime import datetime, timezone

from frozen_expert_bank import verify_entry


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--split", choices=["development", "validation", "all"], required=True
    )
    parser.add_argument("--owner", choices=["dml", "dml2"])
    parser.add_argument(
        "--method",
        choices=["aipw", "tmle", "ctmle", "cui_selective_ml", "ma_dr_bc"],
    )
    parser.add_argument("--max-workers", type=int, default=24)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle, delimiter="\t")
            if (args.split == "all" or row["split"] == args.split)
            and (args.owner is None or row.get("owner") == args.owner)
            and (args.method is None or row["method"] == args.method)
        ]
    args.status.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    state = {
        "split": args.split,
        "owner": args.owner or "all",
        "method": args.method or "all",
        "total_jobs": len(rows),
        "completed_jobs": 0,
        "already_complete_jobs": 0,
        "active_jobs": 0,
        "failed_jobs": 0,
        "started_utc": utcnow(),
        "updated_utc": utcnow(),
        "failures": [],
    }

    def publish():
        state["updated_utc"] = utcnow()
        temporary = args.status.with_suffix(args.status.suffix + ".tmp")
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, args.status)

    def run(row):
        output = Path(row["output"])
        if output.exists():
            verify_entry(output)
            return "already_complete", ""
        command = json.loads(row["command_json"])
        label = f"{row['dataset_identity_sha256'][:12]}_{row['method']}"
        log = args.log_dir / f"{label}.log"
        environment = dict(os.environ)
        environment.update(
            {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        with lock:
            state["active_jobs"] += 1
            publish()
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=environment,
            )
            log.write_text(result.stdout)
            if result.returncode != 0:
                return "failed", f"returncode={result.returncode}; log={log}"
            verify_entry(output)
            return "completed", ""
        except BaseException as error:
            return "failed", f"{type(error).__name__}: {error}; log={log}"
        finally:
            with lock:
                state["active_jobs"] -= 1
                publish()

    publish()
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(run, row): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            disposition, detail = future.result()
            with lock:
                if disposition == "already_complete":
                    state["already_complete_jobs"] += 1
                    state["completed_jobs"] += 1
                elif disposition == "completed":
                    state["completed_jobs"] += 1
                else:
                    state["failed_jobs"] += 1
                    state["failures"].append(
                        {
                            "dataset_identity": row["dataset_identity"],
                            "method": row["method"],
                            "detail": detail,
                        }
                    )
                publish()
    if state["failed_jobs"]:
        raise SystemExit(f"{state['failed_jobs']} expert-bank jobs failed")
    print(json.dumps(state, sort_keys=True))


if __name__ == "__main__":
    main()
