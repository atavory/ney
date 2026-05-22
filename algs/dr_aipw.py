"""
Doubly Robust AIPW estimator for population mean under nonresponse.

Cross-fitted propensity and outcome models with one-step augmented correction.
This is the unweighted baseline (no score alignment).
Supports both binary and continuous outcomes.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import KFold


class DRAIPW:
    """DR-AIPW for population mean under nonresponse.

    Parameters
    ----------
    pop_X : ndarray (N, p)
        Population covariates.
    n_folds : int
        Number of cross-fitting folds.
    max_iter : int
        Max iterations for nuisance HGB models.
    outcome_type : str
        'auto', 'binary', or 'continuous'. Auto-detects from y.
    ps_clip : tuple
        Propensity score clipping bounds (lower, upper).
    """

    def __init__(
        self,
        pop_X: np.ndarray,
        n_folds: int = 5,
        max_iter: int = 200,
        outcome_type: str = "auto",
        ps_clip: tuple[float, float] = (0.025, 0.975),
    ):
        self.pop_X = pop_X
        self.n_folds = n_folds
        self.max_iter = max_iter
        self.outcome_type = outcome_type
        self.ps_clip = ps_clip
        self.estimate_: float | None = None
        self.if_values_: np.ndarray | None = None
        self.if_variance_: float | None = None

    def _is_binary(self, y: np.ndarray) -> bool:
        if self.outcome_type == "binary":
            return True
        if self.outcome_type == "continuous":
            return False
        unique = np.unique(y)
        return len(unique) <= 2

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 0) -> DRAIPW:
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
        binary = self._is_binary(y)
        ps_lo, ps_hi = self.ps_clip
        X_all = np.vstack([X, self.pop_X]).astype(float)
        r_all = np.concatenate([np.ones(n), np.zeros(len(self.pop_X))])

        ps_oof = np.full(n, 0.5)
        mu_oof = np.full(n, float(y.mean()))
        mu_pop = np.zeros(len(self.pop_X))
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=seed)

        for train_idx, val_idx in kf.split(X):
            t_mask = np.ones(len(X_all), dtype=bool)
            t_mask[val_idx] = False

            ps_m = HistGradientBoostingClassifier(
                max_iter=self.max_iter, max_depth=4, random_state=seed
            )
            ps_m.fit(X_all[t_mask], r_all[t_mask])
            ps_oof[val_idx] = np.clip(
                ps_m.predict_proba(X[val_idx].astype(float))[:, 1],
                ps_lo,
                ps_hi,
            )

            if binary:
                out_m = HistGradientBoostingClassifier(
                    max_iter=self.max_iter, max_depth=4, random_state=seed
                )
                out_m.fit(X[train_idx].astype(float), y[train_idx])
                mu_oof[val_idx] = out_m.predict_proba(
                    X[val_idx].astype(float)
                )[:, 1]
                mu_pop += (
                    out_m.predict_proba(self.pop_X.astype(float))[:, 1]
                    / self.n_folds
                )
            else:
                out_m = HistGradientBoostingRegressor(
                    max_iter=self.max_iter, max_depth=4, random_state=seed
                )
                out_m.fit(X[train_idx].astype(float), y[train_idx])
                mu_oof[val_idx] = out_m.predict(X[val_idx].astype(float))
                mu_pop += (
                    out_m.predict(self.pop_X.astype(float)) / self.n_folds
                )

        # AIPW estimate: prediction term + augmented correction
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
