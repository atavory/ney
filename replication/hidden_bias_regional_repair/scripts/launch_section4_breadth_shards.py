#!/usr/bin/env python3
"""Launch resumable Section-4 breadth shards from a deterministic manifest."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import datetime as datetime_module
import hashlib
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


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


def maximum_internal_seed(
    seed: int,
    design: str,
    mar_design: str,
    n: int,
    strength: float,
    reps_per_chunk: int,
    bootstraps: int,
) -> int:
    """Conservative upper bound for sklearn/XGBoost random-state seeds.

    The scientific runner adds design, MAR, replication, nuisance-fit, and
    whole-procedure-bootstrap offsets to the manifest seed.  Checking only
    the manifest seed allowed the nonlinear-MAR anchor to overflow uint32.
    Keep this execution guard synchronized with validated_reference_transfer.
    """
    known_design_offsets = {
        "flat": 0,
        "smooth": 1,
        "pockets": 2,
        "oscillatory": 3,
        "diabetes_real": 11,
        "diabetes_semisynth": 12,
        "diabetes_misaligned": 13,
        "ihdp_semisynth": 21,
        "ihdp_misaligned": 22,
        "acic2016_semisynth": 31,
        "acic2016_misaligned": 32,
        "acic2017_semisynth": 33,
        "acic2017_misaligned": 34,
    }
    if design in known_design_offsets:
        design_offset = known_design_offsets[design] * 10_000_000
    else:
        design_offset = sum(
            (index + 1) * ord(character)
            for index, character in enumerate(design)
        ) * 1000
    mar_offset = {
        "box": 0,
        "smooth_tail": 101,
        "nonlinear_mar": 202,
        "two_stratum_flip": 404,
    }.get(mar_design, 303) * 10_000_000
    one_rep_seed = (
        seed
        + n * 1009
        + 50 * 100003
        + int(round(strength * 100)) * 10007
        + max(0, reps_per_chunk - 1) * 37
        + design_offset
        + mar_offset
    )
    # The largest nested path is the whole-procedure bootstrap, followed by
    # response-region and nuisance-fit offsets.  One million is a strict
    # conservative cushion over all current nested offsets.
    return one_rep_seed + 1_000_000


def design_cells(groups: set[str]):
    if "kang_schafer" in groups:
        for design in (
            "kang_schafer_cc",
            "kang_schafer_ci",
            "kang_schafer_ic",
            "kang_schafer_ii",
        ):
            for n in (200, 1000):
                yield "kang_schafer", design, n, 0.0
    if "alignment" in groups:
        designs = (
            "alignment_aligned",
            "alignment_partial",
            "alignment_disjoint",
        )
        # Strength zero is one shared null, not three differently seeded
        # copies bearing alignment labels.
        yield "alignment", "alignment_aligned", 3000, 0.0
        for design in designs:
            for strength in (3.0, 5.0, 8.0):
                yield "alignment", design, 3000, strength
    if "real" in groups:
        for design in (
            "real_digits_misaligned",
            "real_breast_cancer_misaligned",
            "real_diabetes_misaligned",
            "real_wine_misaligned",
            "real_digits_aligned",
            "real_breast_cancer_aligned",
            "real_diabetes_aligned",
            "real_wine_aligned",
        ):
            for strength in (0.0, 1.0, 2.0):
                yield "real", design, 6000, strength
    if "anchor" in groups:
        for strength in (0.0, 3.0, 5.0, 8.0):
            yield "anchor", "regional_shift", 3000, strength
    if "real_benchmark" in groups:
        for design in (
            "diabetes_semisynth",
            "diabetes_misaligned",
            "ihdp_semisynth",
            "ihdp_misaligned",
            "acic2016_semisynth",
            "acic2016_misaligned",
            "acic2017_semisynth",
            "acic2017_misaligned",
        ):
            for strength in (0.0, 3.0):
                yield "real_benchmark", design, 3000, strength


def build_jobs(args) -> list[Job]:
    jobs: list[Job] = []
    index = 0
    for cell_index, (group, design, n, strength) in enumerate(
        design_cells({"kang_schafer", "alignment", "real", "anchor", "real_benchmark"})
    ):
        # Seed identity is invariant to how the matrix is partitioned across
        # hosts: enumerate the full frozen cell universe, then filter work.
        if group not in set(args.groups):
            continue
        if args.design_filter and design not in set(args.design_filter):
            continue
        if args.strength_filter and strength not in set(args.strength_filter):
            continue
        for method in args.methods:
            for chunk in range(args.chunks):
                index += 1
                # Every estimator and residual scope sees the same generated
                # sample.  The seed depends only on the design cell and chunk,
                # never on method order or repair rule.
                seed = args.seed_base + (cell_index * args.chunks + chunk + 1) * 100_000
                mar_design = "nonlinear_mar" if design == "regional_shift" else "box"
                internal_seed = maximum_internal_seed(
                    seed,
                    design,
                    mar_design,
                    n,
                    strength,
                    args.reps_per_chunk,
                    args.bootstraps,
                )
                if internal_seed >= 2**32:
                    raise ValueError(
                        "derived seed exceeds uint32 range: "
                        f"manifest={seed} maximum_internal={internal_seed} "
                        f"design={design} strength={strength}"
                    )
                stem = (
                    f"{args.owner}_{group}_{design}_{method}_n{n}_"
                    f"s{strength:g}_chunk{chunk:02d}"
                )
                out = args.run_dir / f"{stem}.csv"
                rep_out = args.run_dir / f"{stem}.reps.csv"
                log = args.run_dir / f"{stem}.log"
                fixed_floor = (
                    ("--tau-grid", "0.05")
                    if method in {"tmle", "aipw", "ma_dr_bc"}
                    else ()
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
                    str(strength),
                    "--design",
                    design,
                    "--mar-design",
                    mar_design,
                    "--analysis-region",
                    "estimated_residual_lowp_supported",
                    "--region-quantile",
                    "0.20" if design == "regional_shift" else "0.10",
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
                    args.repair_mode,
                    "--region-damp-grid",
                    *(str(value) for value in args.region_damp_grid),
                    "--validation-risk",
                    args.validation_risk,
                    "--validation-loss-se",
                    str(args.validation_loss_se),
                    "--c",
                    str(args.shrink_c),
                    "--folds",
                    "3",
                    "--selector",
                    "obsval",
                    "--lepski-c",
                    "4",
                    "--region-detector-c",
                    str(args.region_detector_c),
                    "--region-selector-ablation",
                    args.region_selector_ablation,
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
                        index,
                        group,
                        design,
                        method,
                        n,
                        strength,
                        chunk,
                        seed,
                        out,
                        rep_out,
                        log,
                        command,
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
                    **{field: getattr(job, field) for field in fields[:-1]},
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
    parser.add_argument(
        "--wrapper",
        type=Path,
        default=Path(__file__).with_name("section4_breadth_experiments.py"),
    )
    parser.add_argument(
        "--python", type=Path, default=Path(os.environ.get("PYTHON", "python3"))
    )
    parser.add_argument("--owner", choices=["dml", "dml2"], required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["aipw", "tmle", "ctmle", "cui_selective_ml", "ma_dr_bc"],
        required=True,
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["kang_schafer", "alignment", "real", "anchor", "real_benchmark"],
        required=True,
    )
    parser.add_argument(
        "--learner",
        choices=["xgboost"],
        default="xgboost",
        help="Frozen confirmatory learner; HistGB is intentionally rejected.",
    )
    parser.add_argument("--chunks", type=int, default=24)
    parser.add_argument("--reps-per-chunk", type=int, default=4)
    parser.add_argument("--bootstraps", type=int, default=50)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--seed-base", type=int, default=1_260_810_000)
    parser.add_argument("--design-filter", nargs="+")
    parser.add_argument("--strength-filter", type=float, nargs="+")
    parser.add_argument("--shrink-c", type=float, default=2.0)
    parser.add_argument(
        "--repair-mode",
        choices=[
            "targeting",
            "if_residual",
            "regional_if_residual",
            "if_projection",
            "regional_if_projection",
            "if_library",
        ],
        default="targeting",
    )
    parser.add_argument(
        "--region-selector-ablation",
        choices=[
            "legacy",
            "raw_rank_only",
            "whole_sample_score",
            "all_prefixes",
            "empty_standdown",
            "crossfit_rank_empty_standdown",
            "both_signs",
        ],
        default="legacy",
    )
    parser.add_argument("--region-detector-c", type=float, default=4.0)
    parser.add_argument(
        "--region-damp-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 1.0],
        help="Observed-validation regional path; must include zero stand-down.",
    )
    parser.add_argument(
        "--validation-risk",
        choices=["balanced_mse", "aipw_variance"],
        default="balanced_mse",
    )
    parser.add_argument("--validation-loss-se", type=float, default=1.0)
    parser.add_argument(
        "--source-commit", default="1f30548b050e6fbd190db3270bbb8334516b483c"
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Write the deterministic manifest and provenance without launching jobs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    args.frozen_source = args.frozen_source.resolve()
    args.wrapper = args.wrapper.resolve()
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
        "created_utc": datetime_module.datetime.now(
            datetime_module.timezone.utc
        ).isoformat(),
        "owner": args.owner,
        "groups": args.groups,
        "methods": args.methods,
        "chunks": args.chunks,
        "reps_per_chunk": args.reps_per_chunk,
        "bootstraps": args.bootstraps,
        "max_workers": args.max_workers,
        "seed_base": args.seed_base,
        "design_filter": args.design_filter,
        "strength_filter": args.strength_filter,
        "shrink_c": args.shrink_c,
        "repair_mode": args.repair_mode,
        "learner": args.learner,
        "xgboost_version": xgboost_version,
        "python": str(args.python.resolve()),
        "region_selector_ablation": args.region_selector_ablation,
        "region_detector_c": args.region_detector_c,
        "region_damp_grid": args.region_damp_grid,
        "validation_risk": args.validation_risk,
        "validation_loss_se": args.validation_loss_se,
        "frozen_source": str(args.frozen_source),
        "frozen_source_sha256": sha256(args.frozen_source),
        "wrapper": str(args.wrapper),
        "wrapper_sha256": sha256(args.wrapper),
        "launcher_sha256": sha256(Path(__file__).resolve()),
        "manifest_sha256": sha256(manifest),
        "source_commit": args.source_commit,
        "manifest_only": args.manifest_only,
    }
    (args.run_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    if args.manifest_only:
        print(f"wrote {len(jobs)} jobs to {manifest}")
        return
    lock = threading.Lock()
    status = {
        "started_utc": datetime_module.datetime.now(
            datetime_module.timezone.utc
        ).isoformat(),
        "total_jobs": len(jobs),
        "completed_jobs": 0,
        "failed_jobs": 0,
        "results": [],
    }

    def record(result) -> None:
        job, returncode, disposition, elapsed = result
        with lock:
            status["completed_jobs"] += int(returncode == 0)
            status["failed_jobs"] += int(returncode != 0)
            status["results"].append(
                {
                    "index": job.index,
                    "design": job.design,
                    "method": job.method,
                    "chunk": job.chunk,
                    "returncode": returncode,
                    "disposition": disposition,
                    "elapsed_seconds": round(elapsed, 3),
                }
            )
            status["updated_utc"] = datetime_module.datetime.now(
                datetime_module.timezone.utc
            ).isoformat()
            (args.run_dir / "status.json").write_text(
                json.dumps(status, indent=2, sort_keys=True) + "\n"
            )

    with futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        pending = [executor.submit(run_job, job, args.reps_per_chunk) for job in jobs]
        for completed in futures.as_completed(pending):
            record(completed.result())
    if status["failed_jobs"]:
        raise SystemExit(f"{status['failed_jobs']} jobs failed")


if __name__ == "__main__":
    main()
