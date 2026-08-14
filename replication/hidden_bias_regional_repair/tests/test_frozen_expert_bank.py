import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from frozen_expert_bank import (  # noqa: E402
    ExpertFitSnapshot,
    FrozenExpertBankEntry,
    direction_reproducibility,
    load_entry,
    write_entry,
)


def snapshot(role, seed, move, direction):
    reference = np.asarray([0.2, -0.1, 0.4, -0.2])
    direction = np.asarray(direction)
    proposal = reference + direction - np.mean(direction) + move
    return ExpertFitSnapshot(
        role=role,
        seed=seed,
        reference_score=reference,
        reference_outcome=np.zeros(4),
        initial_outcome=np.zeros(4),
        propensity_prediction=np.full(4, 0.5),
        fold_ids=np.asarray([0, 1, 0, 1]),
        candidate_values=np.asarray([0.0, 1.0]),
        candidate_scores=np.stack([reference, proposal]),
        candidate_outcomes=np.stack([np.zeros(4), np.ones(4)]),
        source_index=np.arange(4),
        evaluation_index=np.arange(4),
        selected_candidate=1.0,
        repair_weight=1.0,
    )


class FrozenExpertBankTest(unittest.TestCase):
    def entry(self):
        direction = np.asarray([0.1, -0.2, 0.2, -0.1])
        return FrozenExpertBankEntry(
            algorithm="aipw",
            dataset_id="pilot",
            dataset_seed=7,
            source_sha256="a" * 64,
            observed_data={
                "x": np.arange(8).reshape(4, 2),
                "y": np.arange(4, dtype=float),
                "response": np.asarray([1, 0, 1, 1]),
            },
            fit_specification={"folds": 2},
            full_fit=snapshot("full", 10, 0.2, direction),
            refits=(
                snapshot("repeated_crossfit", 11, 0.3, direction * 0.9),
                snapshot("delete_block_0", 12, 0.1, direction * 1.1),
            ),
            software={"numpy": np.__version__},
        )

    def test_round_trip_and_truth_separation(self):
        with tempfile.TemporaryDirectory() as raw:
            destination = Path(raw) / "entry"
            write_entry(
                self.entry(),
                destination,
                evaluation_truth={"theta": np.asarray([1.5])},
            )
            loaded = load_entry(destination)
            self.assertEqual(loaded.algorithm, "aipw")
            self.assertAlmostEqual(loaded.full_fit.move, 0.2)
            self.assertNotIn("theta", loaded.observed_data)
            self.assertTrue((destination / "SEALED_EVALUATION_TRUTH.npz").exists())
            self.assertFalse((destination / "entry.pkl").exists())

    def test_direction_reproducibility_is_truth_free(self):
        result = direction_reproducibility(self.entry())
        self.assertEqual(result.nonzero_count, 1)
        self.assertEqual(result.sign_reproducibility, 1.0)
        self.assertEqual(result.pairwise_sign_agreement, 1.0)
        self.assertEqual(result.deletion_sign_flip_rate, 0.0)

    def test_hash_failure_is_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            destination = Path(raw) / "entry"
            write_entry(self.entry(), destination)
            with (destination / "manifest.json").open("a") as handle:
                handle.write("tampered")
            with self.assertRaises(ValueError):
                load_entry(destination)


if __name__ == "__main__":
    unittest.main()
