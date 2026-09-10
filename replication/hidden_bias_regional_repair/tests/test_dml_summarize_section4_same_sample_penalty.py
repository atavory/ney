#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "dml_summarize_section4_same_sample_penalty.py"
)
SPEC = importlib.util.spec_from_file_location("same_sample_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(ref: float, selected: float) -> dict[str, object]:
    return {
        "ref_sq_error": ref,
        "selected_sq_error": selected,
        "ref_fresh_risk": ref,
        "selected_fresh_risk": selected,
        "selection_penalty_delta": 0.0,
        "active": False,
    }


class EqualSettingAggregationTest(unittest.TestCase):
    def test_equal_setting_average_is_not_pooled_ratio_of_means(self) -> None:
        high_scale = MODULE._summarize_cell([_row(100.0, 80.0)])
        low_scale = MODULE._summarize_cell([_row(1.0, 2.0)])

        equal_setting = MODULE._mean_summaries([high_scale, low_scale])
        pooled = MODULE._summarize_cell(
            [_row(100.0, 80.0), _row(1.0, 2.0)]
        )

        self.assertTrue(
            math.isclose(equal_setting["same_sample_mse_gain_pct"], -40.0)
        )
        self.assertTrue(
            math.isclose(
                pooled["same_sample_mse_gain_pct"],
                100.0 * (101.0 - 82.0) / 101.0,
            )
        )
        self.assertNotEqual(
            equal_setting["same_sample_mse_gain_pct"],
            pooled["same_sample_mse_gain_pct"],
        )

    def test_setting_cluster_bootstrap_preserves_constant_gap(self) -> None:
        cells = [
            {
                "dataset": "benchmark",
                "setting": f"setting-{index}",
                "gain_gap_same_minus_fresh_pct": 2.0,
            }
            for index in range(MODULE.EXPECTED_SETTINGS)
        ]

        result = MODULE._cluster_bootstrap_gap(cells, seed=17)

        self.assertEqual(2.0, result["gain_gap_ci_low"])
        self.assertEqual(2.0, result["gain_gap_ci_high"])
        self.assertEqual(0.0, result["gain_gap_bootstrap_se"])
        self.assertEqual(
            MODULE.EXPECTED_SETTINGS,
            result["gain_gap_bootstrap_clusters"],
        )


if __name__ == "__main__":
    unittest.main()
