from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from unified_expert_repair import (  # noqa: E402
    ExpertEvaluation,
    FunctionalExpert,
    RepairParameters,
    RepairProposal,
    TrustPrecondition,
    repair_expert,
    select_candidate,
)


def parameters(**overrides: object) -> RepairParameters:
    values = {
        "scope": "global",
        "construction": "residual",
        "se_threshold": 1.0,
        "shrink_c": 2.0,
        "candidate_grid": (0.0, 0.5, 1.0),
    }
    values.update(overrides)
    return RepairParameters(**values)


def test_single_entry_point_passes_all_choices_to_expert() -> None:
    seen: list[RepairParameters] = []
    score = np.array([-2.0, -1.0, 1.0, 2.0])

    def evaluation(config: RepairParameters) -> ExpertEvaluation:
        seen.append(config)
        return ExpertEvaluation(
            0.0,
            score,
            RepairProposal(0.5, score * 0.5, (0.5, 0.5, 0.5)),
        )

    config = parameters(scope="regional", construction="projection")
    repair_expert(FunctionalExpert(evaluation), config)
    assert seen == [config]


def test_positive_part_weight_uses_shared_c() -> None:
    score = np.array([-2.0, -1.0, 1.0, 2.0])
    proposal_score = score + np.array([-0.4, 0.0, 0.0, 0.4])
    expert = FunctionalExpert(
        lambda _: ExpertEvaluation(
            0.0,
            score,
            RepairProposal(0.5, proposal_score, (1.0,)),
        )
    )
    result0 = repair_expert(expert, parameters(shrink_c=0.0))
    result2 = repair_expert(expert, parameters(shrink_c=2.0))
    assert result0.weight == 1.0
    assert 0.0 < result2.weight < 1.0
    assert result2.estimate == pytest.approx(result2.weight * 0.5)


def test_move_bound_caps_final_move_in_reference_standard_errors() -> None:
    score = np.array([-1.0, 1.0])
    expert = FunctionalExpert(
        lambda _: ExpertEvaluation(
            2.0,
            score,
            RepairProposal(12.0, score, (1.0,)),
        )
    )

    unbounded = repair_expert(expert, parameters(shrink_c=0.0))
    bounded = repair_expert(
        expert,
        parameters(shrink_c=0.0, move_bound_kappa=0.25),
    )

    assert unbounded.estimate == 12.0
    assert bounded.reference_se == pytest.approx(1.0)
    assert abs(bounded.estimate - bounded.reference_estimate) == pytest.approx(0.25)
    assert bounded.weight == pytest.approx(0.025)
    assert bounded.unconstrained_weight == 1.0
    assert bounded.move_bound_applied


def test_move_bound_is_disabled_by_default() -> None:
    assert parameters().move_bound_kappa is None


def test_trust_precondition_stands_down_on_generic_diagnostic() -> None:
    score = np.array([-2.0, -1.0, 1.0, 2.0])

    def expert_with_ess(effective_sample_size: float) -> FunctionalExpert:
        return FunctionalExpert(
            lambda _: ExpertEvaluation(
                0.0,
                score,
                RepairProposal(0.5, score, (1.0,)),
                diagnostics={"weight_effective_sample_size": effective_sample_size},
            )
        )

    config = parameters(
        shrink_c=0.0,
        trust_preconditions=(
            TrustPrecondition("weight_effective_sample_size", minimum=20.0),
        ),
    )
    failed = repair_expert(expert_with_ess(10.0), config)
    passed = repair_expert(expert_with_ess(30.0), config)

    assert not failed.trust_passed
    assert failed.unconstrained_weight == 1.0
    assert failed.weight == 0.0
    assert failed.estimate == failed.reference_estimate
    assert passed.trust_passed
    assert passed.weight == 1.0


@pytest.mark.parametrize("diagnostics", [{}, {"improvement_kurtosis": float("nan")}])
def test_enabled_trust_precondition_fails_closed_on_unusable_diagnostic(
    diagnostics: dict[str, float],
) -> None:
    score = np.array([-1.0, 1.0])
    expert = FunctionalExpert(
        lambda _: ExpertEvaluation(
            0.0,
            score,
            RepairProposal(1.0, score, (1.0,)),
            diagnostics=diagnostics,
        )
    )
    config = parameters(
        shrink_c=0.0,
        trust_preconditions=(
            TrustPrecondition("improvement_kurtosis", maximum=10.0),
        ),
    )

    result = repair_expert(expert, config)

    assert not result.trust_passed
    assert result.weight == 0.0


def test_trust_precondition_is_disabled_by_default() -> None:
    assert parameters().trust_preconditions == ()


def test_se_gate_stands_down_when_improvement_is_not_clear() -> None:
    reference = np.array([-1.0, 1.0, -1.0, 1.0])
    candidates = [reference, np.array([-1.1, 0.9, -0.9, 1.1])]
    assert select_candidate(reference, candidates, [0.0, 1.0], 1.0) == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("se_threshold", -1.0),
        ("shrink_c", -1.0),
        ("move_bound_kappa", -1.0),
    ],
)
def test_invalid_shared_knobs_fail(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        parameters(**{field: value})


def test_invalid_trust_preconditions_fail() -> None:
    with pytest.raises(ValueError):
        TrustPrecondition("weight_effective_sample_size")
    with pytest.raises(ValueError):
        TrustPrecondition("improvement_kurtosis", minimum=2.0, maximum=1.0)
    with pytest.raises(ValueError):
        parameters(
            trust_preconditions=(
                TrustPrecondition("weight_effective_sample_size", minimum=10.0),
                TrustPrecondition("weight_effective_sample_size", maximum=100.0),
            )
        )
