"""Pitcher-level Stuff+ style grades, and the test of whether they mean anything.

Producing an index is easy. The question that decides whether it belongs in
front of a decision-maker is whether a grade computed on past pitches predicts
*future* results better than past results do. That test lives here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Scale chosen to match the convention analysts already read fluently:
# 100 is league average, each 10 points is one standard deviation.
INDEX_MEAN = 100.0
INDEX_SD = 10.0


def scale_index(values: pd.Series) -> pd.Series:
    """Map raw predicted-whiff probabilities onto the familiar 100/10 scale."""
    mu, sigma = values.mean(), values.std(ddof=0)
    if not np.isfinite(sigma) or sigma == 0:
        return pd.Series(INDEX_MEAN, index=values.index)
    return INDEX_MEAN + INDEX_SD * (values - mu) / sigma


def pitcher_grades(df: pd.DataFrame, prob_col: str, min_swings: int = 100) -> pd.DataFrame:
    """Aggregate pitch-level predictions into one grade per pitcher."""
    g = (
        df.groupby(["pitcher", "player_name"], observed=True)
        .agg(
            swings=("is_whiff", "size"),
            actual_whiff_rate=("is_whiff", "mean"),
            expected_whiff_rate=(prob_col, "mean"),
        )
        .reset_index()
    )
    g = g[g["swings"] >= min_swings].copy()
    g["stuff_plus"] = scale_index(g["expected_whiff_rate"]).round(1)
    g["whiff_over_expected"] = (g["actual_whiff_rate"] - g["expected_whiff_rate"]).round(4)
    g["actual_whiff_rate"] = g["actual_whiff_rate"].round(4)
    g["expected_whiff_rate"] = g["expected_whiff_rate"].round(4)
    return g.sort_values("stuff_plus", ascending=False).reset_index(drop=True)


def split_halves(df: pd.DataFrame, date_col: str = "game_date") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a season at its median game date."""
    dates = pd.to_datetime(df[date_col])
    cut = dates.median()
    return df[dates <= cut].copy(), df[dates > cut].copy()


def predictive_validity(
    df: pd.DataFrame,
    prob_col: str,
    min_swings: int = 100,
) -> pd.DataFrame:
    """Does first-half stuff predict second-half whiff rate better than first-half whiff rate?

    This is the only result that justifies the model existing. A metric that
    merely describes what already happened is a box score; a metric that
    forecasts what happens next is a decision tool. Both candidate predictors
    are measured against the same held-out second half, on the same pitchers.
    """
    first, second = split_halves(df)
    a = pitcher_grades(first, prob_col, min_swings)[
        ["pitcher", "player_name", "stuff_plus", "expected_whiff_rate", "actual_whiff_rate"]
    ].rename(
        columns={
            "expected_whiff_rate": "h1_expected",
            "actual_whiff_rate": "h1_actual",
            "stuff_plus": "h1_stuff_plus",
        }
    )
    b = pitcher_grades(second, prob_col, min_swings)[
        ["pitcher", "actual_whiff_rate"]
    ].rename(columns={"actual_whiff_rate": "h2_actual"})

    joined = a.merge(b, on="pitcher", how="inner")
    if len(joined) < 3:
        return pd.DataFrame()

    rows = [
        {
            "predictor": "first-half expected whiff (model)",
            "r": joined["h1_expected"].corr(joined["h2_actual"]),
        },
        {
            "predictor": "first-half actual whiff (outcome)",
            "r": joined["h1_actual"].corr(joined["h2_actual"]),
        },
    ]
    out = pd.DataFrame(rows)
    out["r_squared"] = (out["r"] ** 2).round(4)
    out["r"] = out["r"].round(4)
    out["n_pitchers"] = len(joined)
    return out


def validity_by_threshold(
    df: pd.DataFrame,
    prob_col: str,
    thresholds: tuple[int, ...] = (25, 50, 75, 100, 150, 200, 300),
) -> pd.DataFrame:
    """Compare model vs outcome as a predictor across sample-size cutoffs.

    Whiff rate is one of the faster-stabilizing pitcher statistics, so given
    enough swings a pitcher's own past whiff rate is a strong forecast and hard
    to beat. A stuff model earns its keep in the thin-sample regime — a
    prospect with 30 swings of data, a reliever just called up, a trade
    deadline target with three weeks of a new pitch. Sweeping the threshold
    shows where, if anywhere, the model is the better tool.
    """
    rows = []
    for k in thresholds:
        # The second half must clear a fixed bar so the *target* is equally
        # reliable at every threshold; only the predictor's sample varies.
        first, second = split_halves(df)
        a = pitcher_grades(first, prob_col, min_swings=k)[
            ["pitcher", "expected_whiff_rate", "actual_whiff_rate"]
        ].rename(columns={"expected_whiff_rate": "h1_expected", "actual_whiff_rate": "h1_actual"})
        b = pitcher_grades(second, prob_col, min_swings=150)[
            ["pitcher", "actual_whiff_rate"]
        ].rename(columns={"actual_whiff_rate": "h2_actual"})
        j = a.merge(b, on="pitcher", how="inner")
        if len(j) < 10:
            continue
        r_model = j["h1_expected"].corr(j["h2_actual"])
        r_outcome = j["h1_actual"].corr(j["h2_actual"])
        rows.append({
            "min_swings_h1": k,
            "n_pitchers": len(j),
            "r_model": round(r_model, 4),
            "r_outcome": round(r_outcome, 4),
            "model_advantage": round(r_model - r_outcome, 4),
        })
    return pd.DataFrame(rows)


def blend_validity(df: pd.DataFrame, prob_col: str, min_swings: int = 100) -> pd.DataFrame:
    """Does model + outcome together beat either alone?

    If the model carries information the outcome does not, a simple average of
    the two standardized predictors should forecast better than the outcome by
    itself. This is the weaker but more realistic claim: not that stuff
    replaces results, but that it adds something on top of them.
    """
    first, second = split_halves(df)
    a = pitcher_grades(first, prob_col, min_swings)[
        ["pitcher", "expected_whiff_rate", "actual_whiff_rate"]
    ].rename(columns={"expected_whiff_rate": "h1_expected", "actual_whiff_rate": "h1_actual"})
    b = pitcher_grades(second, prob_col, min_swings)[
        ["pitcher", "actual_whiff_rate"]
    ].rename(columns={"actual_whiff_rate": "h2_actual"})
    j = a.merge(b, on="pitcher", how="inner")
    if len(j) < 10:
        return pd.DataFrame()

    z = lambda s: (s - s.mean()) / s.std(ddof=0)
    j["blend"] = 0.5 * z(j["h1_expected"]) + 0.5 * z(j["h1_actual"])

    rows = [
        ("model alone", j["h1_expected"].corr(j["h2_actual"])),
        ("outcome alone", j["h1_actual"].corr(j["h2_actual"])),
        ("50/50 blend", j["blend"].corr(j["h2_actual"])),
    ]
    out = pd.DataFrame(rows, columns=["predictor", "r"])
    out["r_squared"] = (out["r"] ** 2).round(4)
    out["r"] = out["r"].round(4)
    out["n_pitchers"] = len(j)
    return out
