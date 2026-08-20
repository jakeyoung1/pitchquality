"""Expected-whiff models and honest evaluation.

Three models are fit so that every claim has something to be measured against:

``baseline``   pitch-type mean whiff rate from the training season. Beating
               this is the minimum bar for the model to have earned its
               complexity.
``stuff``      physical release characteristics only — no location, no count.
               This is the grade that travels with the pitcher.
``full``       stuff plus location and count. Upper bound on what the feature
               set can explain; the gap to ``stuff`` is roughly what command
               and sequencing are worth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .features import CATEGORICAL, LOCATION_FEATURES, STUFF_FEATURES

RANDOM_STATE = 17


@dataclass
class Evaluation:
    """Held-out performance for a single model."""

    name: str
    auc: float
    log_loss: float
    brier: float
    n: int

    def row(self) -> dict:
        return {
            "model": self.name,
            "auc": round(self.auc, 4),
            "log_loss": round(self.log_loss, 4),
            "brier": round(self.brier, 5),
            "n_pitches": self.n,
        }


def _design(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Assemble the model matrix, encoding categoricals for HistGBM."""
    X = df[feature_names + CATEGORICAL].copy()
    X["pitch_type"] = X["pitch_type"].astype("category")
    X["same_hand"] = X["same_hand"].astype("category")
    return X


def feature_set(include_location: bool) -> list[str]:
    return STUFF_FEATURES + (LOCATION_FEATURES if include_location else [])


def fit(train: pd.DataFrame, include_location: bool) -> tuple:
    """Fit a gradient-boosted whiff model. Returns (model, feature_names)."""
    names = feature_set(include_location)
    model = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=25,
        categorical_features="from_dtype",
        random_state=RANDOM_STATE,
    )
    model.fit(_design(train, names), train["is_whiff"].astype(int))
    return model, names


def predict(model, df: pd.DataFrame, names: list[str]) -> np.ndarray:
    return model.predict_proba(_design(df, names))[:, 1]


def baseline_rates(train: pd.DataFrame) -> pd.Series:
    """Training whiff rate per pitch type — the model to beat."""
    return train.groupby("pitch_type", observed=True)["is_whiff"].mean()


def baseline_predict(rates: pd.Series, df: pd.DataFrame) -> np.ndarray:
    overall = float(rates.mean())
    return df["pitch_type"].map(rates).fillna(overall).to_numpy(dtype=float)


def evaluate(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> Evaluation:
    return Evaluation(
        name=name,
        auc=roc_auc_score(y_true, y_prob),
        log_loss=log_loss(y_true, y_prob),
        brier=brier_score_loss(y_true, y_prob),
        n=len(y_true),
    )


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed whiff rate by probability decile.

    A model that discriminates well but is miscalibrated is useless for any
    decision expressed in rates, so this gets reported alongside AUC.
    """
    df = pd.DataFrame({"p": y_prob, "y": y_true})
    df["bin"] = pd.qcut(df["p"], bins, labels=False, duplicates="drop")
    out = (
        df.groupby("bin")
        .agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
        .reset_index(drop=True)
    )
    out["gap"] = out["observed"] - out["predicted"]
    return out
