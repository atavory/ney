#!/usr/bin/env python3
"""Development experiment: influence projection around published Ma DR-BC DiD.

This implements the current arXiv:2304.08974 simulation design and the
two-period/two-group DR-BC ATT estimator.  It is a development program, not a
confirmatory result generator.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor

from unified_expert_repair import (
    ExpertEvaluation,
    FunctionalExpert,
    RepairParameters,
    RepairProposal,
    repair_expert,
    select_candidate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=12)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument(
        "--dgp", type=int, choices=[1, 2, 3, 4], nargs="+", default=[2, 3]
    )
    parser.add_argument("--h", type=float, default=0.05)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument(
        "--adapter",
        choices=(
            "projection",
            "global_projection",
            "regional_projection",
            "global_residual",
            "regional_residual",
        ),
        help="Deprecated combined spelling; prefer --scope and --construction.",
    )
    parser.add_argument("--scope", choices=("global", "regional"))
    parser.add_argument("--construction", choices=("residual", "projection"))
    parser.add_argument(
        "--projection-learner",
        choices=("linear", "xgboost"),
        default="linear",
    )
    parser.add_argument(
        "--gamma-se",
        type=float,
        default=2.83,
        help=(
            "One-sided Bonferroni threshold for 3 held-out folds times "
            "7 nonzero gamma candidates (Phi^-1(1-.05/21)=2.82)."
        ),
    )
    parser.add_argument("--shrink-c", type=float, default=2.0)
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=[0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1],
    )
    parser.add_argument("--seed", type=int, default=1_108_202_026)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def estimated_supported_region(
    x: np.ndarray,
    dy: np.ndarray,
    response: np.ndarray,
    folds: int,
    seed: int,
    quantile: float = 0.10,
    min_responders: int = 30,
    detector_c: float = 1.0,
) -> np.ndarray:
    """Frozen residual-rank detector inside the estimated low-response tail."""
    labels = np.empty(len(response), dtype=int)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (_, test_index) in enumerate(splitter.split(x, response.astype(int))):
        labels[test_index] = fold
    response_score = np.empty(len(response))
    outcome_prediction = np.empty(len(response))
    global_mean = float(np.mean(dy[response.astype(bool)]))
    for fold in range(folds):
        test = labels == fold
        train = ~test
        classifier = XGBClassifier(
            random_state=seed + 101 * fold, n_jobs=1, verbosity=0
        )
        classifier.fit(x[train], response[train].astype(int))
        response_score[test] = classifier.predict_proba(x[test])[:, 1]
        observed_train = train & response.astype(bool)
        if int(np.sum(observed_train)) < 20:
            outcome_prediction[test] = global_mean
        else:
            model = XGBRegressor(random_state=seed + 211 * fold, n_jobs=1, verbosity=0)
            model.fit(x[observed_train], dy[observed_train])
            outcome_prediction[test] = model.predict(x[test])

    order = np.argsort(response_score, kind="mergesort")
    minimum_count = max(1, int(math.ceil(quantile * len(response))))
    observed_cumulative = np.cumsum(response[order].astype(bool))
    hits = np.flatnonzero(observed_cumulative >= min_responders)
    low_count = max(minimum_count, int(hits[0]) + 1) if len(hits) else len(response)
    low_response = np.zeros(len(response), dtype=bool)
    low_response[order[:low_count]] = True
    observed = response.astype(bool)
    if int(np.sum(low_response & observed)) < min_responders:
        return np.zeros(len(response), dtype=bool)

    weighted_residual = np.full(len(response), np.nan)
    weighted_residual[observed] = (
        dy[observed] - outcome_prediction[observed]
    ) / np.maximum(response_score[observed], 0.02)
    direction = float(np.mean(weighted_residual[low_response & observed]))
    if not np.isfinite(direction) or direction == 0.0:
        return np.zeros(len(response), dtype=bool)
    signed = np.sign(direction) * weighted_residual

    rank_signal = np.empty(len(response))
    for fold in range(folds):
        test = labels == fold
        train = (~test) & observed
        if int(np.sum(train)) < 20:
            rank_signal[test] = float(np.nanmean(signed[train]))
            continue
        model = XGBRegressor(random_state=seed + 401 * fold, n_jobs=1, verbosity=0)
        model.fit(x[train], signed[train])
        rank_signal[test] = model.predict(x[test])

    inside = np.flatnonzero(low_response)
    ranked = inside[np.argsort(-rank_signal[inside], kind="mergesort")]
    best = np.zeros(len(response), dtype=bool)
    best_score = 0.0
    for fraction in (0.25, 0.50, 0.75, 1.00):
        count = max(1, int(math.ceil(fraction * len(ranked))))
        candidate = np.zeros(len(response), dtype=bool)
        candidate[ranked[:count]] = True
        values = signed[candidate & observed]
        if len(values) < min_responders:
            continue
        variance = float(np.var(values, ddof=1) / len(values))
        score = float(np.mean(values)) ** 2 - detector_c * variance
        if score > best_score:
            best_score = score
            best = candidate
    return best


def make_data(n: int, dgp: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_t(df=10, size=(n, 4))
    sigma1 = math.sqrt(10.0 / 8.0)
    sigma2 = math.sqrt(2.0 * (6.25 - 1.25**2))
    sigma3 = math.sqrt(78.125)
    z = np.column_stack(
        [
            x[:, 0] / sigma1,
            (x[:, 0] ** 2 - x[:, 1] ** 2) / sigma2,
            x[:, 2] ** 3 / sigma3,
            x[:, 3] ** 3 / sigma3,
        ]
    )
    if dgp == 1:
        w_ps, w_reg = z, z
    elif dgp == 2:
        w_ps, w_reg = x, z
    elif dgp == 3:
        w_ps, w_reg = z, x
    else:
        w_ps, w_reg = x, x
    f_reg = 1.0 + np.sum(w_reg, axis=1)
    f_ps = 1.5 + np.sum(w_ps, axis=1)
    true_p = 1.0 / (1.0 + np.exp(-np.clip(f_ps, -35.0, 35.0)))
    d = rng.binomial(1, true_p, size=n).astype(float)
    # The shared unit effect cancels from the panel change but is generated to
    # match the paper's DGP exactly.
    upsilon = rng.normal(d * f_reg, 1.0, size=n)
    eps0 = rng.normal(size=n)
    eps10 = rng.normal(size=n)
    eps11 = rng.normal(size=n)
    y0 = f_reg + upsilon + eps0
    y1 = 2.0 * f_reg + upsilon + np.where(d == 1.0, eps11, eps10)
    return {"x": x, "z": z, "d": d, "dy": y1 - y0, "true_p": true_p}


def shifted_legendre(a: np.ndarray, degree: int = 3) -> np.ndarray:
    return np.polynomial.legendre.legvander(2.0 * a - 1.0, degree)


def shifted_legendre_derivative_zero(degree: int = 3) -> np.ndarray:
    result = np.zeros(degree + 1)
    for j in range(1, degree + 1):
        result[j] = 2.0 * float(np.polynomial.legendre.Legendre.basis(j).deriv(1)(-1.0))
    return result


def dr_bc_moment(
    a: np.ndarray,
    b: np.ndarray,
    h: float,
    fixed_trimmed: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Ma alpha-hat and plug-in omega score for k=1, K=3."""
    a = np.clip(np.asarray(a, dtype=float), 1e-8, 1.0)
    b = np.asarray(b, dtype=float)
    q = shifted_legendre(a)
    gram = q.T @ q / len(a)
    gram_inv = np.linalg.pinv(gram)
    beta = gram_inv @ (q.T @ b / len(a))
    derivative = shifted_legendre_derivative_zero()
    xi_prime = float(derivative @ beta)
    fitted = q @ beta
    # A finite-difference perturbation of an estimated propensity can move an
    # individual observation across the hard trimming threshold.  That is not
    # a consistent estimate of the derivative of the *population* moment: it
    # produces isolated O(step^-1) jumps.  Ma et al.'s first-stage derivative
    # is smooth under their DR boundary condition.  During that derivative we
    # therefore condition on the base-sample trim set; ordinary point and IF
    # evaluation continues to use the observed set.
    trimmed = a < h if fixed_trimmed is None else fixed_trimmed
    ratio = np.where(trimmed, 0.0, b / a)
    alpha = float(np.mean(ratio) + np.mean(trimmed) * xi_prime)
    psi = (q @ gram_inv.T @ derivative) * (b - fitted)
    omega = ratio + trimmed.astype(float) * xi_prime + float(np.mean(trimmed)) * psi
    return alpha, omega


def _ma_theta_direct_score(
    d: np.ndarray,
    dy: np.ndarray,
    p: np.ndarray,
    m: np.ndarray,
    h: float,
    fixed_trimmed: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    residual = dy - m
    q0 = 1.0 - p
    b2 = p * (1.0 - d) * residual
    b4 = p * (1.0 - d)
    alpha2, omega2 = dr_bc_moment(q0, b2, h, fixed_trimmed)
    alpha4, omega4 = dr_bc_moment(q0, b4, h, fixed_trimmed)
    dbar = float(np.mean(d))
    u = d * residual
    ubar = float(np.mean(u))
    treated_term = ubar / dbar
    theta = treated_term - alpha2 / alpha4
    treated_if = (u - ubar) / dbar - (ubar / (dbar * dbar)) * (d - dbar)
    phi = (
        treated_if
        - (omega2 - alpha2) / alpha4
        + alpha2 * (omega4 - alpha4) / (alpha4 * alpha4)
    )
    return float(theta), phi


def ma_reference_score(
    data: dict[str, np.ndarray], h: float
) -> dict[str, np.ndarray | float]:
    z, d, dy = data["z"], data["d"], data["dy"]
    design = np.column_stack([np.ones(len(z)), z])
    p_model = LogisticRegression(C=1e10, max_iter=2000, solver="lbfgs")
    p_model.fit(z, d.astype(int))
    beta_p = np.r_[p_model.intercept_, p_model.coef_.ravel()]
    p = np.clip(p_model.predict_proba(z)[:, 1], 1e-6, 1.0 - 1e-6)
    controls = d == 0.0
    m_model = LinearRegression().fit(z[controls], dy[controls])
    beta_m = np.r_[m_model.intercept_, m_model.coef_.ravel()]
    m = design @ beta_m
    theta, direct_phi = _ma_theta_direct_score(d, dy, p, m, h)

    # Ma et al.'s omega includes the first-stage IF.  Compute its contribution
    # by the delta method: numerical derivative of the full DR-BC functional
    # times the exact logit/OLS estimating-equation influence functions.
    base_trimmed = (1.0 - p) < h

    def theta_at(candidate_p: np.ndarray, candidate_m: np.ndarray) -> float:
        return _ma_theta_direct_score(
            d,
            dy,
            candidate_p,
            candidate_m,
            h,
            fixed_trimmed=base_trimmed,
        )[0]

    step = 1e-5
    grad_p = np.empty(len(beta_p))
    grad_m = np.empty(len(beta_m))
    for j in range(len(beta_p)):
        plus, minus = beta_p.copy(), beta_p.copy()
        plus[j] += step
        minus[j] -= step
        p_plus = 1.0 / (1.0 + np.exp(-np.clip(design @ plus, -35.0, 35.0)))
        p_minus = 1.0 / (1.0 + np.exp(-np.clip(design @ minus, -35.0, 35.0)))
        grad_p[j] = (theta_at(p_plus, m) - theta_at(p_minus, m)) / (2.0 * step)
    for j in range(len(beta_m)):
        plus, minus = beta_m.copy(), beta_m.copy()
        plus[j] += step
        minus[j] -= step
        grad_m[j] = (theta_at(p, design @ plus) - theta_at(p, design @ minus)) / (
            2.0 * step
        )

    p_hessian = (design.T * (p * (1.0 - p))) @ design / len(d)
    p_if = (design * (d - p)[:, None]) @ np.linalg.pinv(p_hessian).T
    residual = dy - m
    m_hessian = (design.T * (1.0 - d)) @ design / len(d)
    m_if = (design * ((1.0 - d) * residual)[:, None]) @ np.linalg.pinv(m_hessian).T
    first_stage_phi = p_if @ grad_p + m_if @ grad_m
    phi = direct_phi + first_stage_phi
    return {
        "theta": float(theta),
        "score": theta + phi,
        "p": p,
        "m": m,
        "trimmed_fraction": float(np.mean((1.0 - p) < h)),
        "propensity_gradient_norm": float(np.linalg.norm(grad_p)),
        "outcome_gradient_norm": float(np.linalg.norm(grad_m)),
        "score_max_abs": float(np.max(np.abs(theta + phi))),
    }


def honest_influence_projection(
    z: np.ndarray,
    d: np.ndarray,
    p: np.ndarray,
    score: np.ndarray,
    support: np.ndarray,
    folds: int,
    seed: int,
    gammas: list[float],
    gamma_se: float,
    projection_learner: str,
) -> tuple[np.ndarray, np.ndarray, list[float], float, float]:
    if folds < 3:
        raise ValueError("honest projection requires at least three folds")
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, folds, len(d))
    # Comparison outcomes are the missing counterfactual arm for ATT.  The
    # inverse residual a0=1-(1-D)/(1-p) is conditionally mean-zero but
    # unbounded under weak overlap.  Multiplying by the known X-function
    # (1-p) gives the equivalent, bounded control a=(1-p)*a0=D-p.  It remains
    # exactly conditionally mean-zero and spans the same projection class
    # after rescaling g, without rare inverse-overlap endpoint explosions.
    a = d - p
    applied_control = np.zeros(len(d))
    selected_gammas: list[float] = []
    for fold in range(folds):
        test = labels == fold
        validation = labels == ((fold + 1) % folds)
        projection_train = ~(test | validation)
        center = float(np.mean(score[projection_train]))
        usable = projection_train & (np.abs(a) > 1e-8)
        target = -(score[usable] - center) / a[usable]
        weight = a[usable] ** 2
        if projection_learner == "linear":
            # The published DGP3 outcome defect is linear in observed raw X.
            # This is the disclosed DGP-specific projection class.
            model = LinearRegression()
        else:
            # Exact suite-default flexible learner used by the breadth runner.
            model = XGBRegressor(
                random_state=seed + 101 * fold,
                n_jobs=1,
                verbosity=0,
            )
        model.fit(z[usable], target, sample_weight=weight)
        # Bound g(X), not a*g(X), using projection-training predictions only.
        # This prevents tree extrapolation under extreme overlap weights while
        # preserving E[a*g(X)|X]=0.
        training_prediction = model.predict(z[projection_train])
        lower, upper = np.quantile(training_prediction, [0.01, 0.99])
        validation_prediction = np.clip(model.predict(z[validation]), lower, upper)
        test_prediction = np.clip(model.predict(z[test]), lower, upper)
        validation_control = (
            a[validation] * validation_prediction * support[validation]
        )
        test_control = a[test] * test_prediction * support[test]
        validation_center = float(np.mean(score[validation]))
        candidate_scores = [
            score[validation] + gamma * validation_control for gamma in gammas
        ]
        selected_index = select_candidate(
            score[validation],
            candidate_scores,
            gammas,
            gamma_se,
            center=validation_center,
        )
        selected = gammas[selected_index]
        selected_gammas.append(float(selected))
        applied_control[test] = selected * test_control
    common_center = float(np.mean(score))
    ref_risk = float(np.mean((score - common_center) ** 2))
    candidate_score = score + applied_control
    repaired_risk = float(np.mean((candidate_score - common_center) ** 2))
    return applied_control, candidate_score, selected_gammas, ref_risk, repaired_risk


def honest_outcome_residual(
    x: np.ndarray,
    d: np.ndarray,
    dy: np.ndarray,
    p: np.ndarray,
    m: np.ndarray,
    score: np.ndarray,
    support: np.ndarray,
    h: float,
    folds: int,
    seed: int,
    gammas: list[float],
    gamma_se: float,
) -> tuple[np.ndarray, np.ndarray, list[float], float, float]:
    """Cross-fit a supported outcome residual and gate by full score risk."""
    if folds < 3:
        raise ValueError("honest residual selection requires at least three folds")
    labels = np.random.default_rng(seed).integers(0, folds, len(d))
    response = 1.0 - d
    applied_residual = np.zeros(len(d))
    selected_gammas: list[float] = []
    for fold in range(folds):
        test = labels == fold
        validation = labels == ((fold + 1) % folds)
        train = ~(test | validation)
        observed_train = train & response.astype(bool)
        if int(np.sum(observed_train)) < 20 or not np.any(support):
            selected_gammas.append(0.0)
            continue
        response_probability = np.clip(1.0 - p, 1e-6, 1.0)
        weights = (1.0 - response_probability[observed_train]) / response_probability[
            observed_train
        ] ** 2
        model = XGBRegressor(random_state=seed + 503 * fold, n_jobs=1, verbosity=0)
        model.fit(
            x[observed_train],
            dy[observed_train] - m[observed_train],
            sample_weight=weights,
        )
        validation_residual = model.predict(x[validation]) * support[validation]
        test_residual = model.predict(x[test]) * support[test]

        _, base_validation_phi = _ma_theta_direct_score(
            d[validation], dy[validation], p[validation], m[validation], h
        )
        base_validation_theta, _ = _ma_theta_direct_score(
            d[validation], dy[validation], p[validation], m[validation], h
        )
        base_direct_score = base_validation_theta + base_validation_phi
        common_center = float(np.mean(score[validation]))
        candidate_scores = []
        for gamma in gammas:
            candidate_theta, candidate_phi = _ma_theta_direct_score(
                d[validation],
                dy[validation],
                p[validation],
                m[validation] + gamma * validation_residual,
                h,
            )
            candidate_direct_score = candidate_theta + candidate_phi
            candidate_score = (
                score[validation] + candidate_direct_score - base_direct_score
            )
            candidate_scores.append(candidate_score)
        selected_index = select_candidate(
            score[validation],
            candidate_scores,
            gammas,
            gamma_se,
            center=common_center,
        )
        selected = gammas[selected_index]
        selected_gammas.append(float(selected))
        applied_residual[test] = selected * test_residual

    base_theta, base_phi = _ma_theta_direct_score(d, dy, p, m, h)
    candidate_theta, candidate_phi = _ma_theta_direct_score(
        d, dy, p, m + applied_residual, h
    )
    base_direct_score = base_theta + base_phi
    candidate_direct_score = candidate_theta + candidate_phi
    candidate_score = score + candidate_direct_score - base_direct_score
    common_center = float(np.mean(score))
    ref_risk = float(np.mean((score - common_center) ** 2))
    repaired_risk = float(np.mean((candidate_score - common_center) ** 2))
    return applied_residual, candidate_score, selected_gammas, ref_risk, repaired_risk


def main() -> None:
    args = parse_args()
    if 0.0 not in args.gammas:
        raise SystemExit("--gammas must contain zero")
    if args.adapter:
        if args.scope or args.construction:
            raise SystemExit("use either --adapter or --scope/--construction, not both")
        legacy = {
            "projection": ("global", "projection"),
            "global_projection": ("global", "projection"),
            "regional_projection": ("regional", "projection"),
            "global_residual": ("global", "residual"),
            "regional_residual": ("regional", "residual"),
        }
        scope, construction = legacy[args.adapter]
    else:
        if not args.scope or not args.construction:
            raise SystemExit("--scope and --construction are both required")
        scope, construction = args.scope, args.construction
    parameters = RepairParameters(
        scope=scope,
        construction=construction,
        se_threshold=args.gamma_se,
        shrink_c=args.shrink_c,
        candidate_grid=tuple(args.gammas),
    )
    adapter = f"{scope}_{construction}"
    rows: list[dict[str, object]] = []
    for dgp in args.dgp:
        for rep in range(args.reps):
            seed = args.seed + 100_000 * dgp + rep
            data = make_data(args.n, dgp, seed)
            # Ma's published simulation explicitly gives the researcher raw X
            # and makes the reference nuisance fits use transformed Z.  The
            # repair must see the observed raw covariates; feeding it only Z
            # would force it to inherit the reference's designed outcome-model
            # misspecification in DGP 3.
            region = (
                estimated_supported_region(
                    data["x"],
                    data["dy"],
                    1.0 - data["d"],
                    args.folds,
                    seed + 10_003,
                )
                if scope == "regional"
                else np.zeros(len(data["d"]), dtype=bool)
            )
            support = (
                region if scope == "regional" else np.ones(len(region), dtype=bool)
            )
            diagnostics: dict[str, object] = {}

            def evaluate(config: RepairParameters) -> ExpertEvaluation:
                reference = ma_reference_score(data, args.h)
                score = np.asarray(reference["score"], dtype=float)
                ref = float(reference["theta"])
                if config.construction == "projection":
                    (
                        applied,
                        candidate_score,
                        selected,
                        ref_risk,
                        proposal_risk,
                    ) = honest_influence_projection(
                        data["x"],
                        data["d"],
                        np.asarray(reference["p"], dtype=float),
                        score,
                        support,
                        args.folds,
                        seed + 20_003,
                        list(config.candidate_grid),
                        config.se_threshold,
                        args.projection_learner,
                    )
                    proposal_estimate = ref + float(np.mean(applied))
                else:
                    (
                        applied,
                        candidate_score,
                        selected,
                        ref_risk,
                        proposal_risk,
                    ) = honest_outcome_residual(
                        data["x"],
                        data["d"],
                        data["dy"],
                        np.asarray(reference["p"], dtype=float),
                        np.asarray(reference["m"], dtype=float),
                        score,
                        support,
                        args.h,
                        args.folds,
                        seed + 20_003,
                        list(config.candidate_grid),
                        config.se_threshold,
                    )
                    base_direct = _ma_theta_direct_score(
                        data["d"],
                        data["dy"],
                        np.asarray(reference["p"]),
                        np.asarray(reference["m"]),
                        args.h,
                    )[0]
                    candidate_direct = _ma_theta_direct_score(
                        data["d"],
                        data["dy"],
                        np.asarray(reference["p"]),
                        np.asarray(reference["m"]) + applied,
                        args.h,
                    )[0]
                    proposal_estimate = ref + candidate_direct - base_direct
                diagnostics.update(
                    reference=reference,
                    ref=ref,
                    applied=applied,
                    selected=selected,
                    ref_risk=ref_risk,
                    proposal_risk=proposal_risk,
                )
                return ExpertEvaluation(
                    reference_estimate=ref,
                    reference_score=score,
                    proposal=RepairProposal(
                        estimate=proposal_estimate,
                        score=candidate_score,
                        selected_values=tuple(selected),
                    ),
                )

            result = repair_expert(
                FunctionalExpert(evaluate),
                parameters,
            )
            reference = diagnostics["reference"]
            ref = float(diagnostics["ref"])
            repaired = result.estimate
            selected_gammas = result.selected_values
            applied_control = np.asarray(diagnostics["applied"], dtype=float)
            ref_risk = result.reference_risk
            repaired_risk = result.repaired_risk
            gamma = float(np.mean(selected_gammas))
            rows.append(
                {
                    "dgp": dgp,
                    "rep": rep,
                    "seed": seed,
                    "n": args.n,
                    "h": args.h,
                    "adapter": adapter,
                    "repair_scope": scope,
                    "repair_construction": construction,
                    "se_threshold": args.gamma_se,
                    "shrink_c": args.shrink_c,
                    "projection_learner": args.projection_learner,
                    "ref_error": ref,
                    "repair_error": repaired,
                    "proposal_error": result.proposal_estimate,
                    "delta": repaired - ref,
                    "proposal_delta": result.proposal_estimate - ref,
                    "contrast_variance": result.contrast_variance,
                    "shrink_weight": result.weight,
                    "selected_gamma": gamma,
                    "fold_gammas": ";".join(f"{value:g}" for value in selected_gammas),
                    "ref_score_variance": ref_risk,
                    "selected_score_variance": repaired_risk,
                    "score_variance_gain": (ref_risk - repaired_risk) / ref_risk,
                    "trimmed_fraction": reference["trimmed_fraction"],
                    "propensity_gradient_norm": reference["propensity_gradient_norm"],
                    "outcome_gradient_norm": reference["outcome_gradient_norm"],
                    "score_max_abs": reference["score_max_abs"],
                    "mean_control": float(np.mean(applied_control)),
                    "region_mass": float(np.mean(region)),
                    "region_responders": int(np.sum(region & (data["d"] == 0.0))),
                    "repair_support_mass": float(np.mean(support)),
                    "repair_support_responders": (
                        int(np.sum(support & (data["d"] == 0.0)))
                        if construction == "residual"
                        else int(np.sum(support))
                    ),
                }
            )
            print(
                f"dgp={dgp} rep={rep + 1}/{args.reps} gamma={gamma:g} "
                f"ref={ref:+.5f} repair={repaired:+.5f} "
                f"if_gain={(ref_risk - repaired_risk) / ref_risk:+.3%}",
                flush=True,
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
