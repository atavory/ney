#!/usr/bin/env python3
"""Run the traceable learned-detector, sequential-target reference transfer."""

import concurrent.futures as cf
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "support/csv/ushmoo_validated_reference_transfer_20260808"
EXPERIMENT_BINARY = Path(os.environ["USHMOO_EXPERIMENT_BINARY"]).resolve()
EXPERIMENT_SOURCE = Path(os.environ["USHMOO_EXPERIMENT_SOURCE"]).resolve()
METHODS = {
    "ctmle": "ctmle",
    "global_dr_risk_proxy": "cui_tchetgen",
    "aipw": "aipw",
}
STRENGTHS = (0, 3, 5, 8)
CHUNKS_PER_STRENGTH = 24
REPS_PER_CHUNK = 4
BOOTSTRAPS = 50
MAX_WORKERS = int(os.environ.get("USHMOO_MAX_WORKERS", "48"))
HEARTBEAT_SECONDS = 60


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jobs():
    for label, method in METHODS.items():
        for strength in STRENGTHS:
            for chunk in range(CHUNKS_PER_STRENGTH):
                seed = 2026080100 + strength * 100000 + chunk * 1000
                stem = f"{label}_s{strength}_chunk{chunk}"
                out = RUN_DIR / f"{stem}.csv"
                rep_out = RUN_DIR / f"{stem}.reps.csv"
                log = RUN_DIR / f"{stem}.log"
                cmd = [
                    str(EXPERIMENT_BINARY),
                    "--reps", str(REPS_PER_CHUNK),
                    "--bootstraps", str(BOOTSTRAPS),
                    "--n", "3000",
                    "--epsilon", "0.05",
                    "--strength", str(strength),
                    "--design", "regional_shift",
                    "--mar-design", "nonlinear_mar",
                    "--analysis-region", "estimated_residual_lowp_supported",
                    "--region-quantile", "0.2",
                    "--region-min-observed", "30",
                    "--reference-method", method,
                    "--propensity-mode", "estimated",
                    "--learner", "xgboost",
                    "--propensity-learner", "xgboost",
                    "--repair-mode", "targeting",
                    "--c", "2",
                    "--folds", "3",
                    "--selector", "obsval",
                    "--lepski-c", "4",
                    "--region-detector-c", "4",
                    "--region-selector-ablation", "legacy",
                    "--seed", str(seed),
                    "--progress-every", "1",
                    "--out", str(out),
                    "--rep-out", str(rep_out),
                ]
                yield label, strength, chunk, seed, out, rep_out, log, cmd


def row_count(path):
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open() as handle:
        return max(0, sum(1 for _ in handle) - 1)


def complete(rep_out):
    return row_count(rep_out) >= REPS_PER_CHUNK


def run(job):
    label, strength, chunk, seed, out, rep_out, log, cmd = job
    if complete(rep_out):
        return label, strength, chunk, 0, "skipped"
    env = dict(os.environ)
    env.update({key: "1" for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )})
    started = time.monotonic()
    with log.open("w") as handle:
        result = subprocess.run(
            cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT
        )
    return (
        label, strength, chunk, result.returncode,
        f"elapsed_s={time.monotonic() - started:.1f}",
    )


def write_provenance(all_jobs):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(EXPERIMENT_SOURCE, RUN_DIR / "experiment_source_snapshot.py")
    provenance = {
        "created": dt.datetime.now().isoformat(),
        "binary": str(EXPERIMENT_BINARY),
        "binary_sha256": sha256(EXPERIMENT_BINARY),
        "source": str(EXPERIMENT_SOURCE),
        "source_sha256": sha256(EXPERIMENT_SOURCE),
        "source_snapshot_sha256": sha256(RUN_DIR / "experiment_source_snapshot.py"),
        "paper_git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "methods": METHODS,
        "note": (
            "AIPW uses its untargeted outcome regression as the reference and "
            "adds only the selected regional targeting direction. The other "
            "arms use global targeting followed by the same regional direction."
        ),
    }
    (RUN_DIR / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    with (RUN_DIR / "manifest.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("label", "strength", "chunk", "seed", "command"))
        for label, strength, chunk, seed, *_, cmd in all_jobs:
            writer.writerow((label, strength, chunk, seed, " ".join(cmd)))


def main():
    all_jobs = list(jobs())
    write_provenance(all_jobs)
    lock_handle = (RUN_DIR / "controller.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"controller already running for {RUN_DIR}")
    lock_handle.write(str(os.getpid()))
    lock_handle.flush()

    failed = []
    state = {"completed": 0, "active": 0, "started": time.monotonic()}
    state_lock = threading.Lock()
    stop_heartbeat = threading.Event()
    total_reps = len(all_jobs) * REPS_PER_CHUNK

    def heartbeat():
        while not stop_heartbeat.wait(HEARTBEAT_SECONDS):
            with state_lock:
                completed = state["completed"]
                active = state["active"]
                elapsed = time.monotonic() - state["started"]
            written = sum(row_count(job[5]) for job in all_jobs)
            rate = written / elapsed if elapsed > 0 else 0.0
            eta = (total_reps - written) / rate if rate > 0 else float("inf")
            print(
                f"{dt.datetime.now().isoformat()} HEARTBEAT chunks={completed}/"
                f"{len(all_jobs)} reps={written}/{total_reps} active={active} "
                f"failed={len(failed)} elapsed_min={elapsed / 60:.1f} "
                f"eta_min={eta / 60:.1f}",
                flush=True,
            )

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()

    def tracked_run(job):
        with state_lock:
            state["active"] += 1
        try:
            return run(job)
        finally:
            with state_lock:
                state["active"] -= 1

    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(tracked_run, job) for job in all_jobs]
        for index, future in enumerate(cf.as_completed(futures), 1):
            result = future.result()
            with state_lock:
                state["completed"] = index
            print(
                f"{dt.datetime.now().isoformat()} {index}/{len(all_jobs)} {result}",
                flush=True,
            )
            if result[3] != 0:
                failed.append(result)

    stop_heartbeat.set()
    heartbeat_thread.join(timeout=2)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
