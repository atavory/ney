"""
Weighted-outcome DR-AIPW for population mean under nonresponse.

Standard DR-AIPW but the outcome model is trained with clipped
propensity weights to target population risk instead of respondent risk.

Weight modes:
  power=0               DR         -- standard (unweighted outcome)
  stabilize=True        Stab_DR(c) -- outcome weighted by min(1/e_hat, c)
  stabilize=False       SA_DR(c)   -- outcome weighted by min(1/e_hat^power, c)
                                      (power=2 = score-aligned, Algorithm 1)

The score-aligned exponent (power=2) matches the AIPW sensitivity factor
a(X)^2 = 1/pi(X)^2 from Table 1 of the paper.

Clip level selection:
  Fixed grid: c in {3, 5, 10, 20, None}
  EIF-variance selector: pick c minimizing IF variance
  One-SE selector: smallest c within 1 SE of minimum IF variance
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import KFold


def _make_weights(
    ps_train: np.ndarray,
    clip: float | None,
    power: int,
    stabilize: bool,
) -> np.ndarray | None:
    """Build fitting-loss weights from propensity scores.

    Parameters
    ----------
    ps_train : array of propensity scores in (0, 1)
    clip : maximum weight (None = unweighted)
    power : exponent on 1/pi when stabilize=False. 0 = unweighted, 2 = aligned.
    stabilize : if True, use 1/pi (stabilized variant, Proposition 1 Remark);
        if False, use 1/pi^power (fully aligned, Algorithm 1 Step 2b).
    """
    if clip is None or power == 0:
        return None
    if stabilize:
        # Stabilized variant: w = min(1/pi, c)  (Proposition 1, Remark)
        raw = 1.0 / ps_train
    else:
        # Fully aligned: w = min(1/pi^2, c)  (Algorithm 1, Step 2b)
        raw = (1.0 / ps_train) ** power
    w = np.clip(raw, 0.0, clip)
    w = w / w.mean()
    return w


def _weight_diagnostics(
    ps_train: np.ndarray,
    clip: float | None,
    power: int,
    stabilize: bool,
) -> dict:
    if clip is None or power == 0:
        return {
            "frac_clipped": 0.0,
            "mean_w": 1.0,
            "max_w": 1.0,
            "ess": float(len(ps_train)),
        }
    if stabilize:
        # Stabilized variant: w = min(1/pi, c)
        raw = 1.0 / ps_train
    else:
        # Fully aligned: w = min(1/pi^power, c)
        raw = (1.0 / ps_train) ** power
    clipped = np.clip(raw, 0.0, clip)
    clipped = clipped / clipped.mean()
    frac = float(np.mean(raw > clip))
    ess = float(len(clipped) / (1 + np.var(clipped)))
    return {
        "frac_clipped": frac,
        "mean_w": float(clipped.mean()),
        "max_w": float(clipped.max()),
        "ess": ess,
    }


class WeightedDR:
    """Weighted-outcome DR-AIPW.

    Parameters
    ----------
    pop_X : ndarray (N, p)
        Population covariates.
    n_folds : int
        Number of cross-fitting folds.
    outcome_type : 'auto', 'binary', 'continuous'
        Outcome type. Auto-detects from y.
    clip : float or None
        Max weight for outcome training. None = unweighted (standard DR).
    power : int
        Exponent on 1/pi for fitting weights.
        0 = unweighted, 1 = stabilized, 2 = score-aligned.
    stabilize : bool
        If True, use stabilized variant (1/pi). If False, use fully
        aligned weights (1/pi^power).
    max_iter : int
        HGB iterations.
    ps_clip : tuple
        Propensity score clipping bounds (lower, upper).
    """

    def __init__(
        self,
        pop_X: np.ndarray,
        n_folds: int = 5,
        outcome_type: str = "auto",
        clip: float | None = 10.0,
        power: int = 2,
        stabilize: bool = False,
        max_iter: int = 200,
        ps_clip: tuple[float, float] = (0.025, 0.975),
    ):
        self.pop_X = pop_X
        self.n_folds = n_folds
        self.outcome_type = outcome_type
        self.clip = clip
        self.power = power
        self.stabilize = stabilize
        self.max_iter = max_iter
        self.ps_clip = ps_clip

        self.estimate_: float | None = None
        self.if_values_: np.ndarray | None = None
        self.if_variance_: float | None = None
        self.weight_diags_: list[dict] = []

    def _is_binary(self, y: np.ndarray) -> bool:
        if self.outcome_type == "binary":
            return True
        if self.outcome_type == "continuous":
            return False
        return len(np.unique(y)) <= 2

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 0) -> WeightedDR:
        """Fit the DR-AIPW estimator.

        Parameters
        ----------
        X : ndarray (n, p)
            Respondent covariates.
        y : ndarray (n,)
            Respondent outcomes.
        seed : int
            Random seed for cross-fitting splits.
        """
        n = len(X)
        n_pop = len(self.pop_X)
        binary = self._is_binary(y)
        ps_lo, ps_hi = self.ps_clip

        X_all = np.vstack([X, self.pop_X]).astype(float)
        r_all = np.concatenate([np.ones(n), np.zeros(n_pop)])

        ps_oof = np.full(n, 0.5)
        mu_oof = np.full(n, float(y.mean()))
        mu_pop = np.zeros(n_pop)
        self.weight_diags_ = []

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=seed)

        for train_idx, val_idx in kf.split(X):
            t_mask = np.ones(len(X_all), dtype=bool)
            t_mask[val_idx] = False

            # Response model (propensity)
            ps_m = HistGradientBoostingClassifier(
                max_iter=self.max_iter,
                max_depth=4,
                random_state=seed,
            )
            ps_m.fit(X_all[t_mask], r_all[t_mask])
            ps_oof[val_idx] = np.clip(
                ps_m.predict_proba(X[val_idx].astype(float))[:, 1],
                ps_lo,
                ps_hi,
            )

            # Outcome model weights
            ps_train = np.clip(
                ps_m.predict_proba(X[train_idx].astype(float))[:, 1],
                ps_lo,
                ps_hi,
            )
            sw = _make_weights(ps_train, self.clip, self.power, self.stabilize)
            self.weight_diags_.append(
                _weight_diagnostics(ps_train, self.clip, self.power, self.stabilize)
            )

            # Outcome model
            if binary:
                om = HistGradientBoostingClassifier(
                    max_iter=self.max_iter,
                    max_depth=4,
                    random_state=seed,
                )
                om.fit(
                    X[train_idx].astype(float), y[train_idx], sample_weight=sw
                )
                mu_oof[val_idx] = om.predict_proba(
                    X[val_idx].astype(float)
                )[:, 1]
                mu_pop += (
                    om.predict_proba(self.pop_X.astype(float))[:, 1]
                    / self.n_folds
                )
            else:
                om = HistGradientBoostingRegressor(
                    max_iter=self.max_iter,
                    max_depth=4,
                    random_state=seed,
                )
                om.fit(
                    X[train_idx].astype(float), y[train_idx], sample_weight=sw
                )
                mu_oof[val_idx] = om.predict(X[val_idx].astype(float))
                mu_pop += (
                    om.predict(self.pop_X.astype(float)) / self.n_folds
                )

        # AIPW estimate: prediction term + augmented correction
        #   theta = (1/N) sum mu_hat(X_i) + (1/n) sum [(1-pi)/pi] (Y - mu_hat)
        w = (1 - ps_oof) / ps_oof
        w *= n / w.sum()
        prediction_term = float(mu_pop.mean())
        correction = w * (y - mu_oof)
        self.estimate_ = prediction_term + float(np.mean(correction))

        # Influence function values for variance estimation
        self.if_values_ = mu_oof + w * (y - mu_oof) - self.estimate_
        self.if_variance_ = float(np.var(self.if_values_))

        return self

    def predict_population_mean(self) -> float:
        if self.estimate_ is None:
            raise ValueError("Call fit() first")
        return self.estimate_


def select_clip_eif(
    X,
    y,
    pop_X,
    outcome_type="auto",
    clips=(3, 5, 10, 20, None),
    n_folds=5,
    power=2,
    stabilize=False,
    seed=0,
):
    """Select clip level by minimizing EIF variance."""
    best_clip = None
    best_var = float("inf")
    results = {}

    for c in clips:
        dr = WeightedDR(
            pop_X=pop_X,
            n_folds=n_folds,
            outcome_type=outcome_type,
            clip=c,
            power=power,
            stabilize=stabilize,
        )
        dr.fit(X, y, seed=seed)
        results[c] = {
            "estimate": dr.estimate_,
            "if_var": dr.if_variance_,
            "diags": dr.weight_diags_,
        }
        if dr.if_variance_ < best_var:
            best_var = dr.if_variance_
            best_clip = c

    return best_clip, results


def select_clip_one_se(
    X,
    y,
    pop_X,
    outcome_type="auto",
    clips=(3, 5, 10, 20, None),
    n_folds=5,
    power=2,
    stabilize=False,
    seed=0,
):
    """Select smallest clip within 1 SE of minimum EIF variance."""
    _, results = select_clip_eif(
        X, y, pop_X, outcome_type, clips, n_folds, power, stabilize, seed
    )

    vars_by_clip = {c: r["if_var"] for c, r in results.items()}
    min_var = min(vars_by_clip.values())
    n = len(y)
    se_threshold = min_var + np.sqrt(2 * min_var**2 / n)

    ordered = sorted(
        clips, key=lambda c: c if c is not None else float("inf")
    )
    for c in ordered:
        if vars_by_clip[c] <= se_threshold:
            return c, results

    return min(vars_by_clip, key=vars_by_clip.get), results
