#!/usr/bin/env python3
"""Non-executable, content-addressed artifacts for whole-pipeline expert refits."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = 2


def _array(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if result.dtype.hasobject:
        raise ValueError(f"{name} must not contain Python objects")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


@dataclass(frozen=True)
class ExpertFitSnapshot:
    """Observable fitted values from one complete pipeline perturbation."""

    role: str
    seed: int
    reference_score: np.ndarray
    reference_outcome: np.ndarray
    initial_outcome: np.ndarray
    propensity_prediction: np.ndarray
    fold_ids: np.ndarray
    candidate_values: np.ndarray
    candidate_scores: np.ndarray
    candidate_outcomes: np.ndarray
    source_index: np.ndarray
    evaluation_index: np.ndarray
    selected_candidate: float
    repair_weight: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def reference_estimate(self) -> float:
        return float(np.mean(self.reference_score))

    def candidate_index(self, value: float) -> int:
        matches = np.flatnonzero(np.isclose(self.candidate_values, value, atol=1e-12))
        if len(matches) != 1:
            raise ValueError(f"candidate {value} is absent or duplicated")
        return int(matches[0])

    @property
    def selected_score(self) -> np.ndarray:
        return np.asarray(self.candidate_scores[self.candidate_index(self.selected_candidate)])

    @property
    def selected_estimate(self) -> float:
        return float(np.mean(self.selected_score))

    @property
    def move(self) -> float:
        return float(self.repair_weight * (self.selected_estimate - self.reference_estimate))

    @property
    def repaired_score(self) -> np.ndarray:
        return np.asarray(self.reference_score) + self.repair_weight * (
            self.selected_score - np.asarray(self.reference_score)
        )

    def validate(self, observed_n: int) -> None:
        n = len(np.asarray(self.reference_score))
        g = len(np.asarray(self.candidate_values))
        if n < 2 or g < 1:
            raise ValueError("snapshot needs at least two observations and one candidate")
        for name in (
            "reference_score",
            "reference_outcome",
            "initial_outcome",
            "propensity_prediction",
            "fold_ids",
            "source_index",
            "evaluation_index",
        ):
            _array(name, getattr(self, name), (n,))
        _array("candidate_values", self.candidate_values, (g,))
        _array("candidate_scores", self.candidate_scores, (g, n))
        _array("candidate_outcomes", self.candidate_outcomes, (g, n))
        propensity = np.asarray(self.propensity_prediction)
        if np.any((propensity <= 0.0) | (propensity > 1.0)):
            raise ValueError("propensity predictions must lie in (0, 1]")
        for index_name in ("source_index", "evaluation_index"):
            index = np.asarray(getattr(self, index_name))
            if np.any(index < 0) or np.any(index >= observed_n):
                raise ValueError(f"{index_name} points outside the frozen dataset")
        self.candidate_index(self.selected_candidate)
        if not np.isfinite(self.repair_weight) or not 0.0 <= self.repair_weight <= 1.0:
            raise ValueError("repair_weight must lie in [0, 1]")


@dataclass(frozen=True)
class FrozenExpertBankEntry:
    algorithm: str
    dataset_id: str
    dataset_seed: int
    source_sha256: str
    observed_data: Mapping[str, np.ndarray]
    fit_specification: Mapping[str, Any]
    full_fit: ExpertFitSnapshot
    refits: tuple[ExpertFitSnapshot, ...]
    software: Mapping[str, str]
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported expert-bank schema {self.schema_version}")
        if len(self.source_sha256) != 64:
            raise ValueError("source_sha256 must be a complete SHA-256 digest")
        required = {"x", "y", "response"}
        if required - set(self.observed_data):
            raise ValueError("observed_data must include x, y, and response")
        x = np.asarray(self.observed_data["x"])
        y = np.asarray(self.observed_data["y"])
        response = np.asarray(self.observed_data["response"])
        if x.ndim != 2 or y.shape != (len(x),) or response.shape != (len(x),):
            raise ValueError("observed x/y/response shapes do not align")
        if np.any((response != 0) & (response != 1)):
            raise ValueError("response must be binary")
        for name, value in self.observed_data.items():
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise ValueError(f"observed array {name} contains Python objects")
        self.full_fit.validate(len(x))
        if not self.refits:
            raise ValueError("at least one whole-pipeline refit is required")
        for snapshot in self.refits:
            snapshot.validate(len(x))


@dataclass(frozen=True)
class DirectionReproducibility:
    refit_count: int
    nonzero_count: int
    median_move: float
    sign_reproducibility: float
    pairwise_sign_agreement: float
    score_direction_correlation: float
    deletion_sign_flip_rate: float
    maximum_deletion_relative_change: float


def direction_reproducibility(
    entry: FrozenExpertBankEntry, *, zero_tolerance: float = 1e-12
) -> DirectionReproducibility:
    """Primary truth-free aiming diagnostic; never reads the truth file."""

    entry.validate()
    primary_refits = tuple(
        fit for fit in entry.refits if fit.role == "repeated_crossfit"
    )
    if not primary_refits:
        raise ValueError("primary direction statistic requires repeated_crossfit refits")
    moves = np.asarray([fit.move for fit in primary_refits])
    active_moves = moves[np.abs(moves) > zero_tolerance]
    if len(active_moves):
        signs = np.sign(active_moves)
        # Zeros count as abstentions. This prevents one isolated nonzero refit
        # from receiving a perfect reproducibility score.
        reproducibility = float(abs(np.sum(signs)) / len(moves))
        if len(signs) > 1:
            upper = np.triu_indices(len(signs), 1)
            pairwise = float(np.mean((signs[:, None] == signs[None, :])[upper]))
        else:
            pairwise = 1.0
    else:
        reproducibility = 0.0
        pairwise = 0.0
    full_direction = entry.full_fit.repaired_score - entry.full_fit.reference_score
    correlations = []
    for fit in primary_refits:
        direction = fit.repaired_score - fit.reference_score
        if direction.shape != full_direction.shape:
            continue
        if np.std(direction) <= zero_tolerance or np.std(full_direction) <= zero_tolerance:
            continue
        correlations.append(float(np.corrcoef(full_direction, direction)[0, 1]))
    deletions = [fit for fit in entry.refits if fit.role.startswith("delete_")]
    full_move = entry.full_fit.move
    if deletions:
        deletion_moves = np.asarray([fit.move for fit in deletions])
        if abs(full_move) > zero_tolerance:
            deletion_flips = float(np.mean(np.sign(deletion_moves) != np.sign(full_move)))
            maximum_change = float(
                np.max(np.abs(deletion_moves - full_move)) / abs(full_move)
            )
        else:
            deletion_flips = float(np.mean(np.abs(deletion_moves) > zero_tolerance))
            maximum_change = float("inf") if np.any(np.abs(deletion_moves) > zero_tolerance) else 0.0
    else:
        deletion_flips = float("nan")
        maximum_change = float("nan")
    return DirectionReproducibility(
        refit_count=len(primary_refits),
        nonzero_count=len(active_moves),
        median_move=float(np.median(moves)),
        sign_reproducibility=reproducibility,
        pairwise_sign_agreement=pairwise,
        score_direction_correlation=(
            float(np.median(correlations)) if correlations else float("nan")
        ),
        deletion_sign_flip_rate=deletion_flips,
        maximum_deletion_relative_change=maximum_change,
    )


FIT_ARRAY_NAMES = (
    "reference_score",
    "reference_outcome",
    "initial_outcome",
    "propensity_prediction",
    "fold_ids",
    "candidate_values",
    "candidate_scores",
    "candidate_outcomes",
    "source_index",
    "evaluation_index",
)


def _canonical_arrays(entry: FrozenExpertBankEntry) -> dict[str, np.ndarray]:
    arrays = {f"data__{key}": np.asarray(value) for key, value in entry.observed_data.items()}
    for number, fit in enumerate((entry.full_fit, *entry.refits)):
        for name in FIT_ARRAY_NAMES:
            arrays[f"fit_{number:04d}__{name}"] = np.asarray(getattr(fit, name))
    return arrays


def _fit_metadata(fit: ExpertFitSnapshot, prefix: str) -> dict[str, Any]:
    return {
        "prefix": prefix,
        "role": fit.role,
        "seed": fit.seed,
        "selected_candidate": fit.selected_candidate,
        "repair_weight": fit.repair_weight,
        "diagnostics": dict(fit.diagnostics),
    }


def _base_manifest(entry: FrozenExpertBankEntry) -> dict[str, Any]:
    return {
        "schema_version": entry.schema_version,
        "algorithm": entry.algorithm,
        "dataset_id": entry.dataset_id,
        "dataset_seed": entry.dataset_seed,
        "source_sha256": entry.source_sha256,
        "fit_specification": dict(entry.fit_specification),
        "software": dict(entry.software),
        "fits": [
            _fit_metadata(fit, f"fit_{number:04d}")
            for number, fit in enumerate((entry.full_fit, *entry.refits))
        ],
        "storage": "JSON manifest + compressed NPZ arrays; no pickle or executable model",
    }


def _update_array_digest(digest, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode())
    digest.update(array.dtype.str.encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes(order="C"))


def _scientific_digest(
    manifest: Mapping[str, Any],
    observable_arrays: Mapping[str, np.ndarray],
    truth_arrays: Mapping[str, np.ndarray],
) -> str:
    digest = hashlib.sha256()
    base = {key: value for key, value in manifest.items() if key != "content_sha256"}
    digest.update(json.dumps(base, sort_keys=True, separators=(",", ":")).encode())
    for namespace, arrays in (("observable", observable_arrays), ("truth", truth_arrays)):
        for name in sorted(arrays):
            _update_array_digest(digest, f"{namespace}::{name}", np.asarray(arrays[name]))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_entry(
    entry: FrozenExpertBankEntry,
    destination: Path,
    *,
    evaluation_truth: Mapping[str, np.ndarray] | None = None,
) -> str:
    entry.validate()
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    observable = _canonical_arrays(entry)
    truth = {key: np.asarray(value) for key, value in (evaluation_truth or {}).items()}
    manifest = _base_manifest(entry)
    content_sha256 = _scientific_digest(manifest, observable, truth)
    manifest["content_sha256"] = content_sha256
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        np.savez_compressed(temporary / "observable_arrays.npz", **observable)
        if truth:
            np.savez_compressed(temporary / "SEALED_EVALUATION_TRUTH.npz", **truth)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        files = sorted(path for path in temporary.iterdir() if path.is_file())
        (temporary / "SHA256SUMS").write_text(
            "".join(f"{_file_sha256(path)}  {path.name}\n" for path in files)
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return content_sha256


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def verify_entry(destination: Path) -> str:
    destination = Path(destination)
    for line in (destination / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        actual = _file_sha256(destination / name)
        if actual != expected:
            raise ValueError(f"file hash mismatch for {name}")
    manifest = json.loads((destination / "manifest.json").read_text())
    observable = _read_npz(destination / "observable_arrays.npz")
    truth = _read_npz(destination / "SEALED_EVALUATION_TRUTH.npz")
    actual = _scientific_digest(manifest, observable, truth)
    if actual != manifest["content_sha256"]:
        raise ValueError("scientific content hash mismatch")
    return actual


def _snapshot_from_storage(meta: Mapping[str, Any], arrays: Mapping[str, np.ndarray]):
    prefix = meta["prefix"]
    values = {name: arrays[f"{prefix}__{name}"] for name in FIT_ARRAY_NAMES}
    return ExpertFitSnapshot(
        role=meta["role"],
        seed=int(meta["seed"]),
        selected_candidate=float(meta["selected_candidate"]),
        repair_weight=float(meta["repair_weight"]),
        diagnostics=meta.get("diagnostics", {}),
        **values,
    )


def load_entry(destination: Path) -> FrozenExpertBankEntry:
    """Load inspectable numeric data only; no arbitrary code is executed."""

    destination = Path(destination)
    verify_entry(destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    arrays = _read_npz(destination / "observable_arrays.npz")
    observed = {
        name.removeprefix("data__"): value
        for name, value in arrays.items()
        if name.startswith("data__")
    }
    fits = tuple(_snapshot_from_storage(meta, arrays) for meta in manifest["fits"])
    entry = FrozenExpertBankEntry(
        algorithm=manifest["algorithm"],
        dataset_id=manifest["dataset_id"],
        dataset_seed=int(manifest["dataset_seed"]),
        source_sha256=manifest["source_sha256"],
        observed_data=observed,
        fit_specification=manifest["fit_specification"],
        full_fit=fits[0],
        refits=fits[1:],
        software=manifest["software"],
        schema_version=int(manifest["schema_version"]),
    )
    entry.validate()
    return entry
