#!/usr/bin/env python3
"""Expert-agnostic selection and shrinkage for reference repair.

The public entry point is :func:`repair_expert`.  It knows nothing about Ma,
C-TMLE, Cui, AIPW, or TMLE.  An expert supplies its estimate, influence score,
and a proposal factory; all experimental choices live in ``RepairParameters``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Literal, Mapping, Protocol, Sequence

import numpy as np


Scope = Literal["global", "regional"]
Construction = Literal["residual", "projection"]


@dataclass(frozen=True)
class TrustPrecondition:
    """Estimator-agnostic acceptable range for one expert diagnostic."""

    diagnostic: str
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if not self.diagnostic:
            raise ValueError("trust diagnostic name must not be empty")
        if self.minimum is None and self.maximum is None:
            raise ValueError("trust precondition must have a minimum or maximum")
        if self.minimum is not None and not math.isfinite(self.minimum):
            raise ValueError("trust diagnostic minimum must be finite")
        if self.maximum is not None and not math.isfinite(self.maximum):
            raise ValueError("trust diagnostic maximum must be finite")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("trust diagnostic minimum exceeds maximum")


@dataclass(frozen=True)
class RepairParameters:
    scope: Scope
    construction: Construction
    se_threshold: float
    shrink_c: float
    candidate_grid: tuple[float, ...]
    move_bound_kappa: float | None = None
    trust_preconditions: tuple[TrustPrecondition, ...] = ()

    def __post_init__(self) -> None:
        if self.scope not in ("global", "regional"):
            raise ValueError(f"unknown repair scope: {self.scope}")
        if self.construction not in ("residual", "projection"):
            raise ValueError(f"unknown repair construction: {self.construction}")
        if not math.isfinite(self.se_threshold) or self.se_threshold < 0.0:
            raise ValueError("se_threshold must be finite and non-negative")
        if not math.isfinite(self.shrink_c) or self.shrink_c < 0.0:
            raise ValueError("shrink_c must be finite and non-negative")
        if not self.candidate_grid or 0.0 not in self.candidate_grid:
            raise ValueError("candidate_grid must contain the stand-down value zero")
        if len(set(self.candidate_grid)) != len(self.candidate_grid):
            raise ValueError("candidate_grid must not contain duplicates")
        if self.move_bound_kappa is not None and (
            not math.isfinite(self.move_bound_kappa) or self.move_bound_kappa < 0.0
        ):
            raise ValueError("move_bound_kappa must be finite and non-negative")
        diagnostic_names = tuple(rule.diagnostic for rule in self.trust_preconditions)
        if len(set(diagnostic_names)) != len(diagnostic_names):
            raise ValueError("trust_preconditions must not repeat diagnostics")


@dataclass(frozen=True)
class RepairProposal:
    estimate: float
    score: np.ndarray
    selected_values: tuple[float, ...]


@dataclass(frozen=True)
class ExpertEvaluation:
    """Reference and selected proposal produced under one parameter object."""

    reference_estimate: float
    reference_score: np.ndarray
    proposal: RepairProposal
    diagnostics: Mapping[str, float] = field(default_factory=dict)


class Expert(Protocol):
    """Only interface consumed by the universal repair function."""

    def evaluate(self, parameters: RepairParameters) -> ExpertEvaluation:
        """Build the reference and proposal using the supplied parameters."""


@dataclass(frozen=True)
class FunctionalExpert:
    """Convenience expert whose complete evaluation is supplied by a closure."""

    evaluation_factory: Callable[[RepairParameters], ExpertEvaluation]

    def evaluate(self, parameters: RepairParameters) -> ExpertEvaluation:
        return self.evaluation_factory(parameters)


@dataclass(frozen=True)
class RepairResult:
    reference_estimate: float
    estimate: float
    proposal_estimate: float
    weight: float
    unconstrained_weight: float
    contrast_variance: float
    reference_se: float
    trust_passed: bool
    move_bound_applied: bool
    selected_values: tuple[float, ...]
    reference_risk: float
    proposal_risk: float
    repaired_risk: float


def _passes_trust_preconditions(
    diagnostics: Mapping[str, float],
    preconditions: Sequence[TrustPrecondition],
) -> bool:
    """Return whether generic diagnostics satisfy every configured rule."""

    for rule in preconditions:
        try:
            value = float(diagnostics[rule.diagnostic])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isfinite(value):
            return False
        if rule.minimum is not None and value < rule.minimum:
            return False
        if rule.maximum is not None and value > rule.maximum:
            return False
    return True


def select_candidate(
    reference_score: np.ndarray,
    candidate_scores: Sequence[np.ndarray],
    candidate_values: Sequence[float],
    se_threshold: float,
    *,
    center: float | None = None,
) -> int:
    """Choose the lowest-risk eligible candidate with a common SE gate.

    Candidate value zero is the mandatory stand-down candidate.  A nonzero
    candidate is eligible only when its paired mean loss improvement exceeds
    ``se_threshold`` standard errors.  No estimator identity enters the rule.
    """

    reference = np.asarray(reference_score, dtype=float)
    if reference.ndim != 1 or len(reference) < 2:
        raise ValueError("reference_score must be a one-dimensional sample")
    if len(candidate_scores) != len(candidate_values):
        raise ValueError("candidate score/value lengths differ")
    try:
        list(candidate_values).index(0.0)
    except ValueError as error:
        raise ValueError("candidate_values must contain zero") from error
    common_center = float(np.mean(reference)) if center is None else float(center)
    baseline_loss = (reference - common_center) ** 2
    mean_losses: dict[float, float] = {}
    improvements: dict[float, np.ndarray] = {}
    for index, raw_candidate in enumerate(candidate_scores):
        candidate = np.asarray(raw_candidate, dtype=float)
        if candidate.shape != reference.shape:
            raise ValueError("candidate score shape differs from reference")
        loss = (candidate - common_center) ** 2
        value = float(candidate_values[index])
        mean_losses[value] = float(np.mean(loss))
        improvements[value] = baseline_loss - loss
    selected = select_candidate_from_improvements(
        tuple(float(value) for value in candidate_values),
        mean_losses,
        improvements,
        se_threshold,
    )
    return list(float(value) for value in candidate_values).index(selected)


def select_candidate_from_improvements(
    candidate_values: Sequence[float],
    mean_losses: Mapping[float, float],
    paired_improvements: Mapping[float, np.ndarray],
    se_threshold: float,
) -> float:
    """Shared held-out gate used by every expert and construction."""

    values = tuple(float(value) for value in candidate_values)
    if 0.0 not in values:
        raise ValueError("candidate_values must contain zero")
    eligible = [0.0]
    for value in values:
        if value == 0.0:
            continue
        improvement = np.asarray(paired_improvements[value], dtype=float)
        if len(improvement) < 2:
            continue
        se = float(np.std(improvement, ddof=1) / math.sqrt(len(improvement)))
        if float(np.mean(improvement)) > se_threshold * se:
            eligible.append(value)
    return min(eligible, key=lambda value: (float(mean_losses[value]), value))


def repair_expert(expert: Expert, parameters: RepairParameters) -> RepairResult:
    """Repair any expert with one common selection-plus-shrinkage rule."""

    evaluation = expert.evaluate(parameters)
    reference_score = np.asarray(evaluation.reference_score, dtype=float)
    proposal = evaluation.proposal
    proposal_score = np.asarray(proposal.score, dtype=float)
    if reference_score.ndim != 1 or proposal_score.shape != reference_score.shape:
        raise ValueError("expert and proposal scores must be equal-length vectors")
    if len(reference_score) < 2:
        raise ValueError("at least two score observations are required")
    if not np.all(np.isfinite(reference_score)) or not np.all(np.isfinite(proposal_score)):
        raise ValueError("scores must be finite")

    delta = float(proposal.estimate - evaluation.reference_estimate)
    contrast = proposal_score - reference_score
    contrast_variance = float(np.var(contrast, ddof=1) / len(contrast))
    reference_se = float(
        math.sqrt(float(np.var(reference_score, ddof=1)) / len(reference_score))
    )
    unconstrained_weight = (
        max(0.0, 1.0 - parameters.shrink_c * contrast_variance / (delta * delta))
        if delta != 0.0 and math.isfinite(contrast_variance)
        else 0.0
    )
    trust_passed = _passes_trust_preconditions(
        evaluation.diagnostics,
        parameters.trust_preconditions,
    )
    weight = unconstrained_weight if trust_passed else 0.0
    move_bound_applied = False
    if parameters.move_bound_kappa is not None and delta != 0.0:
        maximum_weight = parameters.move_bound_kappa * reference_se / abs(delta)
        if weight > maximum_weight:
            weight = maximum_weight
            move_bound_applied = True
    repaired_score = reference_score + weight * contrast
    center = float(np.mean(reference_score))
    risk = lambda values: float(np.mean((values - center) ** 2))
    return RepairResult(
        reference_estimate=float(evaluation.reference_estimate),
        estimate=float(evaluation.reference_estimate + weight * delta),
        proposal_estimate=float(proposal.estimate),
        weight=float(weight),
        unconstrained_weight=float(unconstrained_weight),
        contrast_variance=contrast_variance,
        reference_se=reference_se,
        trust_passed=trust_passed,
        move_bound_applied=move_bound_applied,
        selected_values=proposal.selected_values,
        reference_risk=risk(reference_score),
        proposal_risk=risk(proposal_score),
        repaired_risk=risk(repaired_score),
    )
