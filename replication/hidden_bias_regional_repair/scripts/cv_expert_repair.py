#!/usr/bin/env python3
"""One observable-only repair API for a full expert and its CV refits.

The public entry point is exactly
``repair(expert, cv_experts, dataset, params)``.  It receives no algorithm
name, dataset name, or simulation truth.  The same function is therefore used
for AIPW, TMLE, C-TMLE, Cui, and Ma on every dataset.

This module is analysis scaffolding for the completed frozen-value atlas.  Its
parameters are explicit and hashable; no default setting is asserted to be the
final paper rule.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from frozen_expert_bank import ExpertFitSnapshot, FrozenExpertBankEntry


@dataclass(frozen=True)
class ObservedDataset:
    """Only the observed dataset arrays available to the repair function."""

    x: np.ndarray
    y: np.ndarray
    response: np.ndarray
    analysis_mask: np.ndarray | None = None

    def validate(self) -> None:
        x = np.asarray(self.x)
        y = np.asarray(self.y)
        response = np.asarray(self.response)
        if x.ndim != 2 or y.shape != (len(x),) or response.shape != (len(x),):
            raise ValueError("dataset x, y, and response do not align")
        if np.any((response != 0) & (response != 1)):
            raise ValueError("dataset response must be binary")
        if self.analysis_mask is not None and np.asarray(self.analysis_mask).shape != (
            len(x),
        ):
            raise ValueError("dataset analysis_mask does not align")


def dataset_from_entry(entry: FrozenExpertBankEntry) -> ObservedDataset:
    """Construct the observable dataset argument from one frozen bank entry."""

    return ObservedDataset(
        x=np.asarray(entry.observed_data["x"]),
        y=np.asarray(entry.observed_data["y"]),
        response=np.asarray(entry.observed_data["response"]),
        analysis_mask=(
            np.asarray(entry.observed_data["analysis_mask"])
            if "analysis_mask" in entry.observed_data
            else None
        ),
    )


@dataclass(frozen=True)
class RepairParameters:
    """Every choice made by the uniform full-plus-CV repair function."""

    candidate_grid: tuple[float, ...]
    se_threshold: float
    shrink_c: float
    minimum_cv_direction_reproducibility: float = 0.0
    minimum_cv_full_direction_agreement: float = 0.0
    move_bound_kappa: float | None = None
    zero_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if not self.candidate_grid or 0.0 not in self.candidate_grid:
            raise ValueError("candidate_grid must contain zero")
        if len(set(self.candidate_grid)) != len(self.candidate_grid):
            raise ValueError("candidate_grid contains duplicates")
        for name in ("se_threshold", "shrink_c", "zero_tolerance"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "minimum_cv_direction_reproducibility",
            "minimum_cv_full_direction_agreement",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.move_bound_kappa is not None and (
            not math.isfinite(self.move_bound_kappa)
            or self.move_bound_kappa < 0.0
        ):
            raise ValueError("move_bound_kappa must be finite and non-negative")


@dataclass(frozen=True)
class CandidateObservableProfile:
    """Truth-free behavior of one candidate across full and CV experts."""

    value: float
    full_move: float
    full_mean_improvement: float
    full_improvement_se: float
    cv_moves: tuple[float, ...]
    cv_mean_improvements: tuple[float, ...]
    cv_direction_reproducibility: float
    cv_full_direction_agreement: float


@dataclass(frozen=True)
class RepairResult:
    reference_estimate: float
    estimate: float
    selected_candidate: float
    proposal_estimate: float
    weight: float
    unconstrained_weight: float
    contrast_variance: float
    reference_se: float
    move_bound_applied: bool
    profiles: tuple[CandidateObservableProfile, ...]


def _candidate_index(snapshot: ExpertFitSnapshot, value: float) -> int:
    matches = np.flatnonzero(
        np.isclose(np.asarray(snapshot.candidate_values, dtype=float), value, atol=1e-12)
    )
    if len(matches) != 1:
        raise ValueError(f"candidate {value} is absent or duplicated")
    return int(matches[0])


def _validate_snapshot(
    snapshot: ExpertFitSnapshot,
    grid: tuple[float, ...],
    observed_n: int,
) -> None:
    values = tuple(float(value) for value in snapshot.candidate_values)
    if values != grid:
        raise ValueError(f"candidate grid mismatch: expected {grid}, got {values}")
    reference = np.asarray(snapshot.reference_score, dtype=float)
    candidates = np.asarray(snapshot.candidate_scores, dtype=float)
    if reference.ndim != 1 or len(reference) < 2:
        raise ValueError("reference score must be a vector with at least two rows")
    if len(reference) != observed_n:
        raise ValueError("expert score length does not match the observed dataset")
    if candidates.shape != (len(grid), len(reference)):
        raise ValueError("candidate scores do not align with the reference score")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidates)):
        raise ValueError("scores must be finite")


def _profile_candidate(
    expert: ExpertFitSnapshot,
    cv_experts: Sequence[ExpertFitSnapshot],
    value: float,
    zero_tolerance: float,
) -> CandidateObservableProfile:
    def measurements(snapshot: ExpertFitSnapshot) -> tuple[float, float, float]:
        reference = np.asarray(snapshot.reference_score, dtype=float)
        candidate = np.asarray(
            snapshot.candidate_scores[_candidate_index(snapshot, value)], dtype=float
        )
        center = float(np.mean(reference))
        improvement = (reference - center) ** 2 - (candidate - center) ** 2
        move = float(np.mean(candidate) - np.mean(reference))
        se = float(np.std(improvement, ddof=1) / math.sqrt(len(improvement)))
        return move, float(np.mean(improvement)), se

    full_move, full_improvement, full_se = measurements(expert)
    cv_values = tuple(measurements(snapshot) for snapshot in cv_experts)
    cv_moves = np.asarray([item[0] for item in cv_values], dtype=float)
    signs = np.where(np.abs(cv_moves) > zero_tolerance, np.sign(cv_moves), 0.0)
    reproducibility = float(abs(np.sum(signs)) / len(signs))
    if abs(full_move) > zero_tolerance:
        agreement = float(np.mean(signs == np.sign(full_move)))
    else:
        agreement = float(np.mean(signs == 0.0))
    return CandidateObservableProfile(
        value=float(value),
        full_move=full_move,
        full_mean_improvement=full_improvement,
        full_improvement_se=full_se,
        cv_moves=tuple(float(item) for item in cv_moves),
        cv_mean_improvements=tuple(float(item[1]) for item in cv_values),
        cv_direction_reproducibility=reproducibility,
        cv_full_direction_agreement=agreement,
    )


def characterize(
    expert: ExpertFitSnapshot,
    cv_experts: Sequence[ExpertFitSnapshot],
    dataset: ObservedDataset,
    params: RepairParameters,
) -> tuple[CandidateObservableProfile, ...]:
    """Return every observable candidate profile without selecting a winner."""

    if not cv_experts:
        raise ValueError("at least one CV expert is required")
    dataset.validate()
    observed_n = len(np.asarray(dataset.x))
    _validate_snapshot(expert, params.candidate_grid, observed_n)
    for cv_expert in cv_experts:
        _validate_snapshot(cv_expert, params.candidate_grid, observed_n)
    return tuple(
        _profile_candidate(expert, cv_experts, value, params.zero_tolerance)
        for value in params.candidate_grid
    )


def repair(
    expert: ExpertFitSnapshot,
    cv_experts: Sequence[ExpertFitSnapshot],
    dataset: ObservedDataset,
    params: RepairParameters,
) -> RepairResult:
    """Repair one expert using the same observable full-plus-CV rule for all.

    A nonzero candidate must pass the full-fit paired-improvement gate and both
    configured CV direction requirements.  Among eligible candidates, the
    function chooses the largest full-fit mean score improvement, then applies
    the common positive-part shrinkage and optional move bound.
    """

    profiles = characterize(expert, cv_experts, dataset, params)
    eligible = [profile for profile in profiles if profile.value == 0.0]
    for profile in profiles:
        if profile.value == 0.0:
            continue
        clears_full_gate = (
            profile.full_mean_improvement
            > params.se_threshold * profile.full_improvement_se
        )
        clears_cv = (
            profile.cv_direction_reproducibility
            >= params.minimum_cv_direction_reproducibility
            and profile.cv_full_direction_agreement
            >= params.minimum_cv_full_direction_agreement
        )
        if clears_full_gate and clears_cv:
            eligible.append(profile)
    selected = max(
        eligible,
        key=lambda profile: (profile.full_mean_improvement, -profile.value),
    )
    reference_score = np.asarray(expert.reference_score, dtype=float)
    candidate_score = np.asarray(
        expert.candidate_scores[_candidate_index(expert, selected.value)], dtype=float
    )
    reference_estimate = float(np.mean(reference_score))
    proposal_estimate = float(np.mean(candidate_score))
    delta = proposal_estimate - reference_estimate
    contrast = candidate_score - reference_score
    contrast_variance = float(np.var(contrast, ddof=1) / len(contrast))
    reference_se = float(
        math.sqrt(float(np.var(reference_score, ddof=1)) / len(reference_score))
    )
    if selected.value == 0.0 or abs(delta) <= params.zero_tolerance:
        unconstrained_weight = 0.0
    else:
        unconstrained_weight = max(
            0.0,
            1.0 - params.shrink_c * contrast_variance / (delta * delta),
        )
    weight = unconstrained_weight
    move_bound_applied = False
    if params.move_bound_kappa is not None and abs(delta) > params.zero_tolerance:
        maximum_weight = params.move_bound_kappa * reference_se / abs(delta)
        if weight > maximum_weight:
            weight = maximum_weight
            move_bound_applied = True
    return RepairResult(
        reference_estimate=reference_estimate,
        estimate=float(reference_estimate + weight * delta),
        selected_candidate=selected.value,
        proposal_estimate=proposal_estimate,
        weight=float(weight),
        unconstrained_weight=float(unconstrained_weight),
        contrast_variance=contrast_variance,
        reference_se=reference_se,
        move_bound_applied=move_bound_applied,
        profiles=profiles,
    )
