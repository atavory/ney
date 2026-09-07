#!/usr/bin/env python3
"""Compatibility helper exposing weighted-residual selected candidates.

Several follow-up diagnostics were written against ``dml_mse_gain_by_rho.py``:
they load the released rows and then read ``selected_gamma``/``selected_error``
from each ``MethodRow``.  This helper reuses that code but replaces those two
fields by the held-out weighted-residual gamma-selection replay, leaving the
original support bundles untouched.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any


DATA_ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = DATA_ROOT / "scripts/dml_mse_gain_by_rho.py"
DEFAULT_SELECTION_PATH = (
    DATA_ROOT
    / "support_csv/dml_weighted_residual_gamma_selection_ablation_20260903"
    / "weighted_residual_gamma_replication_rows.csv"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_BASE = _load_module("dml_weighted_residual_rho_base", BASE_PATH)
_VRT = None
PRIMARY_METHODS = _BASE.PRIMARY_METHODS
KS_DESIGNS = _BASE.KS_DESIGNS
BENCHMARK_DESIGNS = _BASE.BENCHMARK_DESIGNS
MethodRow = _BASE.MethodRow


def __getattr__(name: str) -> Any:
    return getattr(_BASE, name)


def _selection_key_from_row(row: Any) -> tuple[str, str, int, float, int, int]:
    return (
        row.design,
        row.method,
        int(row.n),
        float(row.strength),
        int(row.seed0),
        int(row.rep),
    )


def _selection_key_from_csv(row: dict[str, str]) -> tuple[str, str, int, float, int, int]:
    return (
        row["design"],
        row["method"],
        int(row["n"]),
        float(row["strength"]),
        int(row["seed0"]),
        int(row["rep"]),
    )


def _selection_map(path: Path = DEFAULT_SELECTION_PATH) -> dict[tuple[str, str, int, float, int, int], dict[str, float]]:
    out: dict[tuple[str, str, int, float, int, int], dict[str, float]] = {}
    with path.open(newline="") as handle:
        for csv_row in csv.DictReader(handle):
            key = _selection_key_from_csv(csv_row)
            if key in out:
                raise RuntimeError(f"duplicate weighted-residual selection row: {key}")
            out[key] = {
                "selected_error": float(csv_row["residual_error"]),
                "selected_gamma": float(csv_row["residual_gamma"]),
            }
    return out


def _overlay(rows: list[Any]) -> list[Any]:
    selections = _selection_map()
    out = []
    missing = []
    for row in rows:
        key = _selection_key_from_row(row)
        selected = selections.get(key)
        if selected is None:
            missing.append(key)
            continue
        gamma = selected["selected_gamma"]
        error = selected["selected_error"]
        out.append(
            replace(
                row,
                selected_error=error,
                selected_gamma=gamma,
                c2_error=error,
                c2_weight=1.0 if math.isfinite(gamma) and gamma > 0.0 else 0.0,
            )
        )
    if missing:
        raise RuntimeError(
            "weighted-residual selection rows are missing keys: "
            + ", ".join(str(item) for item in missing[:5])
        )
    return out


def _read_extracted_aug14_rows(run_dir: Path) -> list[Any]:
    _BASE.PRIMARY_METHODS = PRIMARY_METHODS
    return _overlay(_BASE._read_extracted_aug14_rows(run_dir))


def _read_release_rows(directory: Path, source: str) -> list[Any]:
    _BASE.PRIMARY_METHODS = PRIMARY_METHODS
    return _overlay(_BASE._read_release_rows(directory, source))


def _init_worker(source_script: str, breadth_script: str, support_data: str) -> None:
    global _VRT
    _BASE._init_worker(source_script, breadth_script, support_data)
    _VRT = _BASE._VRT


def _process_sample(rows: list[Any]):
    return _BASE._process_sample(rows)


def _main() -> None:
    args = _BASE.parse_args() if hasattr(_BASE, "parse_args") else None
    if args is not None:
        raise RuntimeError("unexpected parser hook in base script")


def main() -> None:
    import argparse
    import json
    import hashlib
    from collections import defaultdict

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--source-script",
        type=Path,
        default=Path(
            "/tmp/ney_global_recreate_20260830/replication/hidden_bias_regional_repair/"
            "source_snapshots/20260814_unified_cartesian_v3/validated_reference_transfer.py"
        ),
    )
    parser.add_argument(
        "--breadth-script",
        type=Path,
        default=Path(
            "/tmp/ney_global_recreate_20260830/replication/hidden_bias_regional_repair/"
            "scripts/section4_breadth_experiments.py"
        ),
    )
    parser.add_argument(
        "--support-data",
        type=Path,
        default=Path("/tmp/dml_real_benchmark_support_data"),
    )
    parser.add_argument("--aug14-run-dir", type=Path, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("support_csv/dml_weighted_residual_mse_gain_by_rho_20260903"),
    )
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--bins", type=int, default=4)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    aug14 = args.aug14_run_dir
    if aug14 is None:
        latest = Path("/tmp/dml_rho_extract_latest.txt")
        if not latest.exists():
            raise SystemExit("pass --aug14-run-dir or create /tmp/dml_rho_extract_latest.txt")
        aug14 = Path(latest.read_text().strip()) / "dml" / "cartesian_dml_ks_alignment_v3"

    rows: list[Any] = []
    rows.extend(_read_extracted_aug14_rows(aug14))
    rows.extend(
        _read_release_rows(
            data_root / "support_csv/dml_real_benchmark_expansion_20260831",
            "real_benchmark_ihdp_acic2016",
        )
    )
    rows.extend(
        _read_release_rows(
            data_root / "support_csv/dml_real_benchmark_acic2017_20260831",
            "real_benchmark_acic2017",
        )
    )
    rows.extend(
        _read_release_rows(
            data_root / "support_csv/dml_real_benchmark_twins_20260831",
            "real_benchmark_twins",
        )
    )

    sample_groups: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
    for row in rows:
        key = (
            row.design,
            row.n,
            row.epsilon,
            row.strength,
            row.mar_design,
            row.seed0,
            row.rep,
            row.propensity_mode,
            row.learner,
            row.propensity_learner,
            row.folds,
        )
        sample_groups[key].append(row)

    rep_rows: list[dict[str, Any]] = []
    response_bin_rows: list[dict[str, Any]] = []
    tasks = list(sample_groups.values())
    with ProcessPoolExecutor(
        max_workers=args.jobs,
        initializer=_init_worker,
        initargs=(
            str(args.source_script.resolve()),
            str(args.breadth_script.resolve()),
            str(args.support_data.resolve()),
        ),
    ) as pool:
        futures = [pool.submit(_process_sample, task) for task in tasks]
        completed = 0
        for future in as_completed(futures):
            sample_rep_rows, sample_response_bin_rows = future.result()
            rep_rows.extend(sample_rep_rows)
            response_bin_rows.extend(sample_response_bin_rows)
            completed += 1
            if completed % max(1, args.jobs * 4) == 0 or completed == len(tasks):
                print(f"completed {completed}/{len(tasks)} samples", flush=True)

    rep_rows.sort(
        key=lambda row: (
            row["dataset"],
            row["setting"],
            row["method"],
            row["seed0"],
            row["rep"],
        )
    )
    response_bin_rows.sort(
        key=lambda row: (
            row["dataset"],
            row["setting"],
            row["method"],
            row["response_bin"],
            row["seed0"],
            row["rep"],
        )
    )
    setting_rows = _BASE._aggregate_setting_rows(rep_rows)
    response_bin_setting_rows = _BASE._aggregate_response_bin_rows(response_bin_rows)
    response_bin_summary = _BASE._response_bin_summary(
        response_bin_setting_rows,
        ("response_bin", "response_bin_label"),
    )
    response_bin_method_summary = _BASE._response_bin_summary(
        response_bin_setting_rows,
        ("method", "response_bin", "response_bin_label"),
    )
    bin_rows = _BASE._rho_bins(setting_rows, args.bins)
    dataset_rows = _BASE._linear_summary(setting_rows, ("dataset",))
    dataset_method_rows = _BASE._linear_summary(setting_rows, ("dataset", "method"))
    method_rows = _BASE._linear_summary(setting_rows, ("method",))

    out_dir = args.out_dir if args.out_dir.is_absolute() else data_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _BASE._write_csv(out_dir / "rho_replication_rows.csv", rep_rows)
    _BASE._write_csv(out_dir / "setting_rho_gain.csv", setting_rows)
    _BASE._write_csv(out_dir / "response_bin_replication_rows.csv", response_bin_rows)
    _BASE._write_csv(out_dir / "response_bin_setting_summary.csv", response_bin_setting_rows)
    _BASE._write_csv(out_dir / "response_bin_summary.csv", response_bin_summary)
    _BASE._write_csv(out_dir / "response_bin_method_summary.csv", response_bin_method_summary)
    _BASE._write_csv(out_dir / "rho_bin_summary.csv", bin_rows)
    _BASE._write_csv(out_dir / "dataset_rho_summary.csv", dataset_rows)
    _BASE._write_csv(out_dir / "dataset_method_rho_summary.csv", dataset_method_rows)
    _BASE._write_csv(out_dir / "method_rho_summary.csv", method_rows)
    _BASE._write_results(out_dir / "RESULTS.md", setting_rows, bin_rows)
    _BASE._write_response_bin_results(
        out_dir / "RESPONSE_BIN_RESULTS.md",
        response_bin_summary,
        response_bin_method_summary,
    )
    _BASE._write_response_bin_figure(
        out_dir / "section4_response_bin_action_reward_figure.tex",
        response_bin_summary,
    )

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    provenance = {
        "source_script": str(args.source_script.resolve()),
        "source_script_sha256": sha256(args.source_script.resolve()),
        "breadth_script": str(args.breadth_script.resolve()),
        "breadth_script_sha256": sha256(args.breadth_script.resolve()),
        "base_rho_script": str(BASE_PATH.resolve()),
        "base_rho_script_sha256": sha256(BASE_PATH.resolve()),
        "selection_overlay": str(DEFAULT_SELECTION_PATH.resolve()),
        "selection_overlay_sha256": sha256(DEFAULT_SELECTION_PATH.resolve()),
        "support_data": str(args.support_data.resolve()),
        "aug14_run_dir": str(aug14.resolve()),
        "rows": len(rep_rows),
        "response_bin_rows": len(response_bin_rows),
        "settings": len(setting_rows),
        "response_bin_settings": len(response_bin_setting_rows),
        "sample_groups": len(sample_groups),
        "methods": list(PRIMARY_METHODS),
        "endpoint": "weighted-residual selected candidate",
        "rho_formula": "mean((1 - true_pi / selected_p)^2 * true_pi / (1 - true_pi))",
        "response_bins": "within-sample quartiles of true_pi; bin 1 is lowest response",
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    text = (out_dir / "RESULTS.md").read_text()
    text = text.replace(
        "# Retrospective MSE Gain by Rho",
        "# Weighted-Residual Selector MSE Gain by Rho",
        1,
    )
    text = text.replace(
        "Primary endpoint: selected candidate (`rt_error`), not the older c=2\n"
        "shrunken endpoint.  The c=2 columns are retained only for audit.",
        "Primary endpoint: the gamma selected by held-out weighted residual loss.  "
        "The original score-selected endpoint is not used in this bundle.",
        1,
    )
    (out_dir / "RESULTS.md").write_text(text)
    text = (out_dir / "RESPONSE_BIN_RESULTS.md").read_text()
    text = text.replace(
        "# Response-Bin Action and Reward Decomposition",
        "# Weighted-Residual Selector Response-Bin Action and Reward Decomposition",
        1,
    )
    (out_dir / "RESPONSE_BIN_RESULTS.md").write_text(text)
    checksum_rows = []
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            checksum_rows.append(f"{sha256(path)}  {path.name}")
    (out_dir / "SHA256SUMS").write_text("\n".join(checksum_rows) + "\n")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
