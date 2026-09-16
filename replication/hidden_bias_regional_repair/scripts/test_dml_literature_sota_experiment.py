#!/usr/bin/env python3
"""Regression tests for the locked literature-DGP SOTA driver."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LiteratureSotaTest(unittest.TestCase):
    def test_selective_logistic_is_effectively_l1(self) -> None:
        runner = load("literature_sota_runner", HERE / "dml_literature_sota_experiment.py")
        upstream = load("literature_sota_upstream_test", HERE / "validated_reference_transfer.py")
        runner.install_corrected_selective_logistic(upstream)
        candidate = upstream._cui_candidate_propensity("logistic_l1", 123)
        params = candidate.steps[-1][1].get_params()
        self.assertEqual(params["solver"], "saga")
        self.assertEqual(params["penalty"], "elasticnet")
        self.assertEqual(params["l1_ratio"], 1.0)
        self.assertEqual(params["C"], 1.0)
        self.assertEqual(params["max_iter"], 2000)

    def test_plain_tmle_is_not_in_locked_methods(self) -> None:
        runner = load("literature_sota_runner_methods", HERE / "dml_literature_sota_experiment.py")
        self.assertEqual(
            runner.METHODS,
            ("aipw", "cui_selective_ml", "ma_dr_bc", "ctmle"),
        )
        self.assertNotIn("tmle", runner.METHODS)

    def test_all_locked_dgps_construct_with_analytic_truth(self) -> None:
        runner = load("literature_sota_runner_dgps", HERE / "dml_literature_sota_experiment.py")
        for cell in runner.CELLS:
            data = runner.make_problem(cell, 202609160 + runner.CELL_OFFSETS[cell])
            x, y, response, region, true_pi, theta, mu = data
            expected_n = 2000 if cell.startswith("zhao") else int(cell.rsplit("n", 1)[1])
            self.assertEqual(x.shape[0], expected_n)
            self.assertEqual(len(y), expected_n)
            self.assertEqual(len(response), expected_n)
            self.assertEqual(len(region), expected_n)
            self.assertEqual(len(true_pi), expected_n)
            self.assertEqual(len(mu), expected_n)
            self.assertTrue(0.0 < theta < 200.0)
            self.assertTrue(((true_pi > 0.0) & (true_pi < 1.0)).all())


if __name__ == "__main__":
    unittest.main()
