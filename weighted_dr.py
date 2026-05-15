"""
Weighted-outcome DR-AIPW.

Standard DR-AIPW but the outcome model is trained with clipped
propensity weights to target population risk instead of respondent risk.

Three variants:
  DR         — standard (unweighted outcome)
  Wt_DR(c)   — outcome weighted by min(1/e_hat^2, c)  (default, score-aligned)
  Stab_DR(c) — outcome weighted by min(p_bar/e_hat, c)  (stabilized)

When squared=False, uses 1/e_hat instead of 1/e_hat^2 (stabilized variant).

Clip level selection:
  Fixed grid: c in {3, 5, 10, 20, inf}
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


def _make_weights(ps_train: np.ndarray, clip: float | None, stabilize: bool, squared: bool = True) -> np.ndarray | None:
    if clip is None:
        return None
    if squared:
        raw = 1.0 / ps_train ** 2
    else:
        raw = 1.0 / ps_train
    if stabilize:
        raw = raw * ps_train.mean()
    w = np.clip(raw, 0.0, clip)
    w = w / w.mean()
    return w


def _weight_diagnostics(ps_train: np.ndarray, clip: float | None, stabilize: bool, squared: bool = True) -> dict:
    if clip is None:
        return {"frac_clipped": 0.0, "mean_w": 1.0, "max_w": 1.0, "ess": float(len(ps_train))}
    if squared:
        raw = 1.0 / ps_train ** 2
    else:
        raw = 1.0 / ps_train
    if stabilize:
        raw = raw * ps_train.mean()
    clipped = np.clip(raw, 0.0, clip)
    clipped = clipped / clipped.mean()
    frac = float(np.mean(raw > clip))
    ess = float(len(clipped) / (1 + np.var(clipped)))
    return {"frac_clipped": frac, "mean_w": float(clipped.mean()), "max_w": float(clipped.max()), "ess": ess}


class WeightedDR:
    """Weighted-outcome DR-AIPW.

    Parameters
    ----------
    pop_X : ndarray (N, p)
    n_folds : int
    outcome_type : 'auto', 'binary', 'continuous'
    clip : float or None
        Max weight for outcome training. None = unweighted (standard DR).
    stabilize : bool
        Use stabilized weights p_bar/e instead of 1/e^2.
    squared : bool
        When True (default), use 1/e^2 weights (score-aligned).
        When False, use 1/e weights (stabilized variant).
    max_iter : int
        HGB iterations.
    """

    def __init__(
        self,
        pop_X: np.ndarray,
        n_folds: int = 5,
        outcome_type: str = "auto",
        clip: float | None = None,
        stabilize: bool = False,
        squared: bool = True,
        max_iter: int = 200,
    ):
        self.pop_X = pop_X
        self.n_folds = n_folds
        self.outcome_type = outcome_type
        self.clip = clip
        self.stabilize = stabilize
        self.squared = squared
        self.max_iter = max_iter

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

    def fit(self, X: np.ndarray, y: np.ndarray) -> WeightedDR:
        n = len(X)
        n_pop = len(self.pop_X)
        binary = self._is_binary(y)

        X_all = np.vstack([X, self.pop_X]).astype(float)
        r_all = np.concatenate([np.ones(n), np.zeros(n_pop)])

        ps_oof = np.full(n, 0.5)
        mu_oof = np.full(n, float(y.mean()))
        mu_pop = np.zeros(n_pop)
        self.weight_diags_ = []

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X):
            t_mask = np.ones(len(X_all), dtype=bool)
            t_mask[val_idx] = False

            # Response model
            ps_m = HistGradientBoostingClassifier(
                max_iter=self.max_iter, max_depth=4, random_state=42,
            )
            ps_m.fit(X_all[t_mask], r_all[t_mask])
            ps_oof[val_idx] = np.clip(
                ps_m.predict_proba(X[val_idx].astype(float))[:, 1], 0.025, 0.975,
            )

            # Outcome model weights
            ps_train = np.clip(
                ps_m.predict_proba(X[train_idx].astype(float))[:, 1], 0.025, 0.975,
            )
            sw = _make_weights(ps_train, self.clip, self.stabilize, self.squared)
            self.weight_diags_.append(_weight_diagnostics(ps_train, self.clip, self.stabilize, self.squared))

            # Outcome model
            if binary:
                om = HistGradientBoostingClassifier(
                    max_iter=self.max_iter, max_depth=4, random_state=42,
                )
                om.fit(X[train_idx].astype(float), y[train_idx], sample_weight=sw)
                mu_oof[val_idx] = om.predict_proba(X[val_idx].astype(float))[:, 1]
                mu_pop += om.predict_proba(self.pop_X.astype(float))[:, 1] / self.n_folds
            else:
                om = HistGradientBoostingRegressor(
                    max_iter=self.max_iter, max_depth=4, random_state=42,
                )
                om.fit(X[train_idx].astype(float), y[train_idx], sample_weight=sw)
                mu_oof[val_idx] = om.predict(X[val_idx].astype(float))
                mu_pop += om.predict(self.pop_X.astype(float)) / self.n_folds

        # AIPW estimate
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


def select_clip_eif(X, y, pop_X, outcome_type="auto", clips=(3, 5, 10, 20, None),
                    n_folds=5, stabilize=False):
    """Select clip level by minimizing EIF variance."""
    best_clip = None
    best_var = float("inf")
    results = {}

    for c in clips:
        dr = WeightedDR(pop_X=pop_X, n_folds=n_folds, outcome_type=outcome_type,
                        clip=c, stabilize=stabilize)
        dr.fit(X, y)
        results[c] = {"estimate": dr.estimate_, "if_var": dr.if_variance_,
                       "diags": dr.weight_diags_}
        if dr.if_variance_ < best_var:
            best_var = dr.if_variance_
            best_clip = c

    return best_clip, results


def select_clip_one_se(X, y, pop_X, outcome_type="auto", clips=(3, 5, 10, 20, None),
                       n_folds=5, stabilize=False):
    """Select smallest clip within 1 SE of minimum EIF variance."""
    _, results = select_clip_eif(X, y, pop_X, outcome_type, clips, n_folds, stabilize)

    vars_by_clip = {c: r["if_var"] for c, r in results.items()}
    min_var = min(vars_by_clip.values())
    n = len(y)
    se_threshold = min_var + np.sqrt(2 * min_var**2 / n)

    ordered = sorted(clips, key=lambda c: c if c is not None else float("inf"))
    for c in ordered:
        if vars_by_clip[c] <= se_threshold:
            return c, results

    return min(vars_by_clip, key=vars_by_clip.get), results
