import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import validated_reference_transfer as vrt  # noqa: E402
import ma_published_did_projection as ma_did  # noqa: E402


class RegionalResidualTest(unittest.TestCase):
    def synthetic_data(self, empty_region: bool):
        rng = np.random.default_rng(912_731)
        n = 360
        x = rng.normal(size=(n, 5))
        p = np.clip(0.35 + 0.20 * (x[:, 0] > 0), 0.10, 0.90)
        response = rng.binomial(1, p)
        mu = x[:, 0] - 0.5 * x[:, 1]
        y = mu + rng.normal(size=n)
        region = np.zeros(n, dtype=bool)
        if not empty_region:
            region = x[:, 0] < np.quantile(x[:, 0], 0.20)
        return x, y, response, region, p, float(np.mean(mu)), mu

    def test_empty_region_is_exact_standdown(self):
        fit = vrt._crossfit_selected(
            self.synthetic_data(empty_region=True),
            "tmle",
            "true",
            "histgb",
            "histgb",
            "regional_if_residual",
            (0.05,),
            3,
            731_009,
            (0.0, 0.25, 0.5, 1.0),
            -1.0,
            2.83,
            "obsval",
            4.0,
        )
        np.testing.assert_array_equal(fit["ref"], fit["rt"])
        np.testing.assert_array_equal(fit["ref_outcome"], fit["rt_outcome"])
        self.assertEqual(fit["selected_region_damp"], 0.0)

    def test_gamma_zero_endpoint_identity_and_regional_support(self):
        data = self.synthetic_data(empty_region=False)
        x, y, response, region, p, _, _ = data
        base = np.zeros(len(y))
        correction = vrt._crossfit_weighted_residual_correction(
            x, y, response, p, base, "histgb", 811_003, 3
        )
        regional = correction * region.astype(float)
        np.testing.assert_array_equal(regional[~region], np.zeros(np.sum(~region)))
        candidate = base + 0.0 * regional
        np.testing.assert_array_equal(candidate, base)

    def test_ma_boundary_corrected_candidate_is_not_aipw_shortcut(self):
        rng = np.random.default_rng(88_031)
        n = 500
        y = rng.normal(size=n)
        response = rng.binomial(1, 0.25, size=n)
        p = np.linspace(0.01, 0.90, n)
        base = rng.normal(scale=0.2, size=n)
        changed = base.copy()
        changed[p < 0.08] += 0.5
        base_value, _ = vrt._ma_dr_bc_reference(y, response, p, base)
        candidate_value, _ = vrt._ma_dr_bc_reference(y, response, p, changed)
        aipw_shortcut = base_value + (
            vrt._aipw_score(y, response, np.maximum(p, 0.05), changed)
            - vrt._aipw_score(y, response, np.maximum(p, 0.05), base)
        )
        self.assertFalse(np.allclose(candidate_value, aipw_shortcut))

    def test_ma_empty_region_is_exact_standdown(self):
        data = ma_did.make_data(500, 2, 717_991)
        reference = ma_did.ma_reference_score(data, 0.05)
        residual, candidate_score, gammas, ref_risk, repaired_risk = (
            ma_did.honest_outcome_residual(
                data["x"],
                data["d"],
                data["dy"],
                np.asarray(reference["p"]),
                np.asarray(reference["m"]),
                np.asarray(reference["score"]),
                np.zeros(500, dtype=bool),
                0.05,
                3,
                818_301,
                [0.0, 0.25, 0.5, 1.0],
                2.83,
            )
        )
        np.testing.assert_array_equal(residual, np.zeros(500))
        np.testing.assert_allclose(
            candidate_score, np.asarray(reference["score"]), atol=1e-12, rtol=0.0
        )
        self.assertEqual(gammas, [0.0, 0.0, 0.0])
        self.assertEqual(ref_risk, repaired_risk)


if __name__ == "__main__":
    unittest.main()
