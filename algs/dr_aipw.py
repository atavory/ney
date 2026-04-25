"""
Doubly Robust AIPW estimator for population mean under nonresponse.

Cross-fitted propensity and outcome models with one-step augmented correction.
This is a baseline comparator, not an NML method.
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
    """

    def __init__(
        self,
        pop_X: np.ndarray,
        n_folds: int = 5,
        max_iter: int = 200,
        outcome_type: str = "auto",
    ):
        self.pop_X = pop_X
        self.n_folds = n_folds
        self.max_iter = max_iter
        self.outcome_type = outcome_type
        self.estimate_: float | None = None
        self.weights_: np.ndarray | None = None

    def _is_binary(self, y: np.ndarray) -> bool:
        if self.outcome_type == "binary":
            return True
        if self.outcome_type == "continuous":
            return False
        unique = np.unique(y)
        return len(unique) <= 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> DRAIPW:
        n = len(X)
        binary = self._is_binary(y)
        X_all = np.vstack([X, self.pop_X])
        r_all = np.concatenate([np.ones(n), np.zeros(len(self.pop_X))])

        ps_oof = np.full(n, 0.5)
        mu_oof = np.full(n, float(y.mean()))
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X):
            t_mask = np.ones(len(X_all), dtype=bool)
            t_mask[val_idx] = False

            ps_m = HistGradientBoostingClassifier(
                max_iter=self.max_iter, max_depth=4, random_state=42
            )
            ps_m.fit(X_all[t_mask], r_all[t_mask])
            ps_oof[val_idx] = np.clip(
                ps_m.predict_proba(X[val_idx])[:, 1], 0.01, 0.99
            )

            if binary:
                out_m = HistGradientBoostingClassifier(
                    max_iter=self.max_iter, max_depth=4, random_state=42
                )
                out_m.fit(X[train_idx], y[train_idx])
                mu_oof[val_idx] = out_m.predict_proba(X[val_idx])[:, 1]
            else:
                out_m = HistGradientBoostingRegressor(
                    max_iter=self.max_iter, max_depth=4, random_state=42
                )
                out_m.fit(X[train_idx], y[train_idx])
                mu_oof[val_idx] = out_m.predict(X[val_idx])

        if binary:
            out_full = HistGradientBoostingClassifier(
                max_iter=self.max_iter, max_depth=4, random_state=42
            )
            out_full.fit(X, y)
            mu_pop = float(out_full.predict_proba(self.pop_X)[:, 1].mean())
        else:
            out_full = HistGradientBoostingRegressor(
                max_iter=self.max_iter, max_depth=4, random_state=42
            )
            out_full.fit(X, y)
            mu_pop = float(out_full.predict(self.pop_X).mean())

        w = (1 - ps_oof) / ps_oof
        w *= n / w.sum()
        self.weights_ = w
        self.estimate_ = mu_pop + float(np.mean(w * (y - mu_oof)))
        return self

    def predict_population_mean(self) -> float:
        if self.estimate_ is None:
            raise ValueError("Call fit() first")
        return self.estimate_

    def predict_weights(self) -> np.ndarray:
        if self.weights_ is None:
            raise ValueError("Call fit() first")
        return self.weights_.copy()
