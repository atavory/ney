#!/usr/bin/env python3
"""Build one algorithm x dataset realization for the frozen expert bank."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import platform

import joblib
import numpy as np
import sklearn

import section4_breadth_experiments as breadth
from frozen_expert_bank import ExpertFitSnapshot, FrozenExpertBankEntry, write_entry


def _derived_dataset_seed(module, args) -> int:
    return int(
        args.base_seed
        + args.n * 1009
        + int(args.epsilon * 1000) * 100003
        + module._design_strength_code(args.strength) * 10007
        + args.replication * 37
        + module._design_seed_offset(args.design)
        + module._mar_seed_offset(args.mar_design)
    )


def _fit_snapshot(module, data, source_index, role, seed, args):
    fit = module._crossfit_selected(
        data,
        args.method,
        args.propensity_mode,
        args.learner,
        args.propensity_learner,
        args.repair_mode,
        tuple(args.tau_grid),
        args.inner_folds,
        seed,
        tuple(args.candidate_grid),
        args.validation_region_weight,
        args.validation_loss_se,
        args.selector,
        args.lepski_c,
    )
    reference_score = np.asarray(fit["ref"], dtype=float)
    proposal_score = np.asarray(fit["rt"], dtype=float)
    reference_estimate = float(np.mean(reference_score))
    proposal_estimate = float(np.mean(proposal_score))
    contrast = proposal_score - reference_score
    delta = proposal_estimate - reference_estimate
    contrast_variance = float(np.var(contrast, ddof=1) / len(contrast))
    weight = (
        max(0.0, 1.0 - args.shrink_c * contrast_variance / (delta * delta))
        if delta != 0.0 and np.isfinite(contrast_variance)
        else 0.0
    )
    return ExpertFitSnapshot(
        role=role,
        seed=int(seed),
        reference_score=reference_score,
        reference_outcome=np.asarray(fit["ref_outcome"], dtype=float),
        initial_outcome=np.asarray(fit["initial_outcome"], dtype=float),
        propensity_prediction=np.asarray(fit["selected_p"], dtype=float),
        fold_ids=np.asarray(fit["fold_ids"], dtype=np.int16),
        candidate_values=np.asarray(fit["candidate_values"], dtype=float),
        candidate_scores=np.asarray(fit["candidate_scores"], dtype=float),
        candidate_outcomes=np.asarray(fit["candidate_outcomes"], dtype=float),
        source_index=np.asarray(source_index, dtype=np.int64),
        evaluation_index=np.asarray(source_index, dtype=np.int64),
        selected_candidate=float(fit["selected_region_damp"]),
        repair_weight=float(weight),
        diagnostics={
            "selected_tau": (
                float(fit["selected_tau"])
                if np.isfinite(float(fit["selected_tau"]))
                else None
            ),
            "selected_candidate": float(fit["selected_region_damp"]),
            "selected_repair_kind": str(fit["selected_repair_kind"]),
            "contrast_variance": contrast_variance,
            "raw_delta": delta,
            "sample_size": len(reference_score),
        },
    )


def _subset(data, index):
    x, y, response, region, true_pi, theta, mu = data
    return (
        x[index],
        y[index],
        response[index],
        region[index],
        true_pi[index],
        theta,
        mu[index],
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(args) -> tuple[FrozenExpertBankEntry, dict[str, np.ndarray]]:
    module = breadth._load_frozen(args.frozen_source.resolve())
    breadth._install_adapter(module)
    os.environ["USHMOO_VALIDATION_RISK"] = args.validation_risk
    dataset_seed = _derived_dataset_seed(module, args)
    raw = module.make_data(
        args.n,
        args.epsilon,
        args.strength,
        args.design,
        dataset_seed,
        args.mar_design,
    )
    x, y, response, true_region, true_pi, theta, mu = raw
    analysis_mask = module._analysis_region(
        x,
        y,
        response,
        true_region,
        true_pi,
        args.analysis_region,
        args.propensity_mode,
        args.learner,
        args.propensity_learner,
        dataset_seed + 9091,
        args.region_quantile,
        args.region_min_observed,
        args.region_kappa_floor,
        args.region_selector_ablation,
        args.region_detector_c,
        args.inner_folds,
    )
    data = (x, y, response, analysis_mask, true_pi, theta, mu)
    identity = np.arange(len(x), dtype=np.int64)
    full_fit = _fit_snapshot(
        module, data, identity, "full", dataset_seed + 17, args
    )
    refits = []
    for repeat in range(args.repeated_crossfits):
        refits.append(
            _fit_snapshot(
                module,
                data,
                identity,
                "repeated_crossfit",
                dataset_seed + 1_000_003 * (repeat + 1),
                args,
            )
        )
    if args.delete_blocks:
        permutation = np.random.default_rng(args.resample_seed).permutation(len(x))
        blocks = np.array_split(permutation, args.delete_blocks)
        for block_number, block in enumerate(blocks):
            keep = np.setdiff1d(identity, block, assume_unique=True)
            refits.append(
                _fit_snapshot(
                    module,
                    _subset(data, keep),
                    keep,
                    f"delete_block_{block_number}",
                    dataset_seed + 2_000_003 + 1009 * block_number,
                    args,
                )
            )
    rng = np.random.default_rng(args.resample_seed + 1)
    for bootstrap in range(args.bootstraps):
        sample = rng.integers(0, len(x), size=len(x), dtype=np.int64)
        refits.append(
            _fit_snapshot(
                module,
                _subset(data, sample),
                sample,
                "bootstrap",
                dataset_seed + 3_000_003 + 1009 * bootstrap,
                args,
            )
        )
    try:
        import xgboost

        xgboost_version = xgboost.__version__
    except ImportError:
        xgboost_version = "unavailable"
    return FrozenExpertBankEntry(
        algorithm=args.method,
        dataset_id=(
            f"{args.design}__n{args.n}__s{args.strength:g}"
            f"__rep{args.replication:03d}"
        ),
        dataset_seed=dataset_seed,
        source_sha256=_sha256(args.frozen_source.resolve()),
        observed_data={
            "x": np.asarray(x),
            "y": np.asarray(y),
            "response": np.asarray(response),
            "analysis_mask": np.asarray(analysis_mask),
        },
        fit_specification={
            "method": args.method,
            "repair_mode": args.repair_mode,
            "candidate_grid": list(args.candidate_grid),
            "validation_risk": args.validation_risk,
            "validation_loss_se": args.validation_loss_se,
            "shrink_c": args.shrink_c,
            "inner_folds": args.inner_folds,
            "repeated_crossfits": args.repeated_crossfits,
            "delete_blocks": args.delete_blocks,
            "bootstraps": args.bootstraps,
            "resample_seed": args.resample_seed,
            "learner": args.learner,
            "propensity_learner": args.propensity_learner,
            "propensity_mode": args.propensity_mode,
            "tau_grid": list(args.tau_grid),
        },
        full_fit=full_fit,
        refits=tuple(refits),
        software={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "joblib": joblib.__version__,
            "xgboost": xgboost_version,
        },
    ), {
        "true_region": np.asarray(true_region),
        "true_propensity": np.asarray(true_pi),
        "theta": np.asarray([theta], dtype=float),
        "conditional_mean": np.asarray(mu),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=["aipw", "tmle", "ctmle", "cui_selective_ml", "ma_dr_bc"],
        required=True,
    )
    parser.add_argument("--design", default="kang_schafer_cc")
    parser.add_argument("--mar-design", default="box")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--strength", type=float, default=0.0)
    parser.add_argument("--base-seed", type=int, default=3_200_100_000)
    parser.add_argument("--replication", type=int, default=0)
    parser.add_argument("--resample-seed", type=int, default=3_300_100_000)
    parser.add_argument("--repeated-crossfits", type=int, default=12)
    parser.add_argument("--delete-blocks", type=int, default=10)
    parser.add_argument("--bootstraps", type=int, default=0)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--learner", choices=["histgb", "xgboost"], default="xgboost")
    parser.add_argument(
        "--propensity-learner", choices=["histgb", "xgboost"], default="xgboost"
    )
    parser.add_argument("--propensity-mode", choices=["true", "estimated"], default="estimated")
    parser.add_argument("--tau-grid", type=float, nargs="+", default=[0.05])
    parser.add_argument("--repair-mode", default="if_residual")
    parser.add_argument("--candidate-grid", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--validation-risk", choices=["balanced_mse", "aipw_variance"], default="balanced_mse")
    parser.add_argument("--validation-loss-se", type=float, default=1.0)
    parser.add_argument("--validation-region-weight", type=float, default=-1.0)
    parser.add_argument("--shrink-c", type=float, default=2.0)
    parser.add_argument("--selector", default="obsval")
    parser.add_argument("--lepski-c", type=float, default=4.0)
    parser.add_argument("--analysis-region", default="estimated_residual_lowp_supported")
    parser.add_argument("--region-quantile", type=float, default=0.10)
    parser.add_argument("--region-min-observed", type=int, default=30)
    parser.add_argument("--region-kappa-floor", type=float, default=0.0)
    parser.add_argument("--region-selector-ablation", default="legacy")
    parser.add_argument("--region-detector-c", type=float, default=4.0)
    return parser.parse_args()


def main():
    args = parse_args()
    entry, truth = build(args)
    content_sha256 = write_entry(entry, args.output, evaluation_truth=truth)
    print(f"EXPERT_BANK_ENTRY_WRITTEN {args.output} sha256={content_sha256}")


if __name__ == "__main__":
    main()
