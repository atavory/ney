from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cv_expert_repair import (  # noqa: E402
    ObservedDataset,
    RepairParameters,
    characterize,
    repair,
)
from frozen_expert_bank import ExpertFitSnapshot  # noqa: E402


def snapshot(move: float, *, role: str = "full", reverse: bool = False):
    reference = np.array([-2.0, -1.0, 1.0, 2.0])
    helpful = reference * 0.5 + move
    if reverse:
        helpful = reference * 1.5 - move
    candidates = np.stack([reference, helpful])
    n = len(reference)
    return ExpertFitSnapshot(
        role=role,
        seed=1,
        reference_score=reference,
        reference_outcome=np.zeros(n),
        initial_outcome=np.zeros(n),
        propensity_prediction=np.full(n, 0.5),
        fold_ids=np.arange(n) % 2,
        candidate_values=np.array([0.0, 1.0]),
        candidate_scores=candidates,
        candidate_outcomes=np.zeros_like(candidates),
        source_index=np.arange(n),
        evaluation_index=np.arange(n),
        selected_candidate=0.0,
        repair_weight=0.0,
    )


def params(**overrides):
    values = dict(
        candidate_grid=(0.0, 1.0),
        se_threshold=0.0,
        shrink_c=0.0,
    )
    values.update(overrides)
    return RepairParameters(**values)


def dataset(n: int = 4) -> ObservedDataset:
    return ObservedDataset(
        x=np.zeros((n, 2)),
        y=np.zeros(n),
        response=np.ones(n),
        analysis_mask=np.ones(n, dtype=bool),
    )


def test_public_api_uses_full_and_cv_experts() -> None:
    full = snapshot(0.3)
    cvs = [snapshot(0.2, role="repeated_crossfit") for _ in range(20)]
    result = repair(
        full,
        cvs,
        dataset(),
        params(minimum_cv_direction_reproducibility=0.9),
    )
    assert result.selected_candidate == 1.0
    assert result.weight == 1.0
    assert len(result.profiles[1].cv_moves) == 20


def test_unstable_cv_directions_force_standdown() -> None:
    full = snapshot(0.3)
    cvs = [
        snapshot(0.2 if number % 2 == 0 else -0.2, role="repeated_crossfit")
        for number in range(20)
    ]
    result = repair(
        full,
        cvs,
        dataset(),
        params(minimum_cv_direction_reproducibility=0.9),
    )
    assert result.selected_candidate == 0.0
    assert result.estimate == result.reference_estimate


def test_characterization_returns_every_candidate_without_truth() -> None:
    profiles = characterize(
        snapshot(0.3),
        [snapshot(0.2, role="repeated_crossfit")],
        dataset(),
        params(),
    )
    assert [profile.value for profile in profiles] == [0.0, 1.0]


def test_candidate_grid_mismatch_fails_closed() -> None:
    config = RepairParameters(
        candidate_grid=(0.0, 0.5, 1.0),
        se_threshold=1.0,
        shrink_c=2.0,
    )
    with pytest.raises(ValueError, match="candidate grid mismatch"):
        repair(snapshot(0.3), [snapshot(0.2)], dataset(), config)


def test_dataset_is_an_explicit_validated_argument() -> None:
    with pytest.raises(ValueError, match="does not match the observed dataset"):
        repair(snapshot(0.3), [snapshot(0.2)], dataset(5), params())
