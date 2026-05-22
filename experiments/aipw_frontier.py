#!/usr/bin/env python3
from __future__ import annotations

"""
Fixed-regime AIPW capacity frontier.

This reruns the confirmed AIPW tail regimes across sample sizes. It is meant
to replace search-driven storytelling with fixed-regime capacity curves.
"""

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from real_tail_aipw import Dataset, LEARNERS, LearnerSpec, load_dataset, run_cell, subsample

os.environ.setdefault("OMP_NUM_THREADS", "1")

RESULT_FIELDS = [
    "regime",
    "dataset",
    "population_size",
    "response_rate",
    "tail_fraction",
    "tail_effect",
    "feature_count",
    "learner",
    "pi_source",
    "seed",
    "variant",
    "estimate",
    "true_value",
    "bias",
    "abs_bias",
    "n_resp",
    "tail_resp_rate",
    "body_resp_rate",
    "tail_share_resp",
    "fit_deff",
    "corr_risk",
    "tail_rmse",
    "body_rmse",
]


@dataclass(frozen=True)
class Regime:
    name: str
    dataset: str
    response_rate: float
    tail_fraction: float
    tail_effect: float
    feature_count: int
    learner: str
    pi_source: str


REGIMES = {
    "acs_top2": Regime("acs_top2", "acs", 0.1811, 0.1442, 5.23, 2, "hgb2_50", "estimated"),
    "brfss_top3": Regime("brfss_top3", "brfss", 0.1913, 0.1391, 6.56, 4, "hgb2_50", "estimated"),
    "ces_top4": Regime("ces_top4", "ces", 0.1602, 0.1329, 5.67, 6, "hgb2_50", "oracle"),
    "brfss_top5": Regime("brfss_top5", "brfss", 0.1623, 0.0697, 7.47, 6, "hgb2_50", "oracle"),
    "gss_top1": Regime("gss_top1", "gss", 0.0744, 0.0248, 7.56, 3, "ridge2", "oracle"),
}


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_csv_strs(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def run_task(
    task: tuple[Regime, Dataset, LearnerSpec, int, float],
) -> list[dict[str, object]]:
    regime, dataset, learner, seed, pi_floor = task
    rows = run_cell(
        dataset,
        regime.response_rate,
        regime.tail_fraction,
        regime.tail_effect,
        regime.feature_count,
        learner,
        regime.pi_source,
        seed,
        pi_floor,
    )
    for row in rows:
        row["regime"] = regime.name
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regimes", type=str, default="acs_top2,brfss_top3,ces_top4,brfss_top5")
    parser.add_argument("--sample-sizes", type=str, default="2000,5000,10000,30000")
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--executor", type=str, choices=("process", "thread"), default="process")
    parser.add_argument("--pi-floor", type=float, default=0.01)
    parser.add_argument("--population-seed", type=int, default=20260519)
    parser.add_argument("--output", type=str, default="results/aipw_confirmed_frontier_v1.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    regimes = tuple(REGIMES[name] for name in parse_csv_strs(args.regimes))
    sample_sizes = parse_csv_ints(args.sample_sizes)
    full_datasets = {name: load_dataset(name) for name in sorted({r.dataset for r in regimes})}
    sized: dict[tuple[str, int], Dataset] = {}
    for dataset_name, dataset in full_datasets.items():
        for size in sample_sizes:
            sized[(dataset_name, size)] = subsample(dataset, size, args.population_seed + size + len(dataset_name))
            print(f"{dataset_name}: requested={size}, n={len(sized[(dataset_name, size)].x)}")
    tasks: list[tuple[Regime, Dataset, LearnerSpec, int, float]] = []
    for regime in regimes:
        learner = LEARNERS[regime.learner]
        for size in sample_sizes:
            dataset = sized[(regime.dataset, size)]
            for seed in range(args.seeds):
                tasks.append((regime, dataset, learner, seed, args.pi_floor))
    print(f"Running {len(tasks)} fixed-regime cells")
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    row_count = 0
    skipped_count = 0
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        handle.flush()
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for index, rows in enumerate(executor.map(run_task, tasks), start=1):
                if rows:
                    writer.writerows(rows)
                    row_count += len(rows)
                else:
                    skipped_count += 1
                if index % 200 == 0:
                    handle.flush()
                    os.fsync(handle.fileno())
                    print(f"  completed {index}/{len(tasks)}; skipped={skipped_count}")
        handle.flush()
        os.fsync(handle.fileno())
    print(f"Wrote {row_count} rows to {args.output}; skipped_cells={skipped_count}")


if __name__ == "__main__":
    main()
