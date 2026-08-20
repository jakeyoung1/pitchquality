"""Run the full pipeline: load -> fit on 2025 -> evaluate on 2026 -> report."""
from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

from . import features, grades, model, plots

DATA = Path("data")
REPORTS = Path("reports")
TRAIN_SEASON = "2025"
TEST_SEASON = "2026"


def load(season: str) -> pd.DataFrame:
    return features.build(pd.read_parquet(DATA / f"statcast_{season}.parquet"))


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    print(f"loading {TRAIN_SEASON} (train) and {TEST_SEASON} (holdout)...", flush=True)
    train, test = load(TRAIN_SEASON), load(TEST_SEASON)
    print(f"  train swings: {len(train):,}   whiff rate: {train['is_whiff'].mean():.4f}")
    print(f"  test  swings: {len(test):,}   whiff rate: {test['is_whiff'].mean():.4f}")

    y_test = test["is_whiff"].astype(int).to_numpy()
    evals = []

    rates = model.baseline_rates(train)
    evals.append(model.evaluate("baseline (pitch-type mean)", y_test, model.baseline_predict(rates, test)))

    print("fitting stuff model (physical release characteristics only)...", flush=True)
    m_stuff, n_stuff = model.fit(train, include_location=False)
    p_stuff = model.predict(m_stuff, test, n_stuff)
    evals.append(model.evaluate("stuff (no location)", y_test, p_stuff))

    print("fitting full model (stuff + location + count)...", flush=True)
    m_full, n_full = model.fit(train, include_location=True)
    p_full = model.predict(m_full, test, n_full)
    evals.append(model.evaluate("full (stuff + location + count)", y_test, p_full))

    results = pd.DataFrame([e.row() for e in evals])
    print("\n=== HELD-OUT PERFORMANCE (train 2025 -> test 2026) ===")
    print(results.to_string(index=False))
    results.to_csv(REPORTS / "model_performance.csv", index=False)

    calib = model.calibration_table(y_test, p_full)
    print("\n=== CALIBRATION (full model, deciles) ===")
    print(calib.to_string(index=False))
    calib.to_csv(REPORTS / "calibration.csv", index=False)

    test = test.assign(p_stuff=p_stuff, p_full=p_full)

    g = grades.pitcher_grades(test, "p_stuff", min_swings=100)
    g.to_csv(REPORTS / "pitcher_stuff_grades.csv", index=False)
    print(f"\n=== TOP 15 BY STUFF+ ({TEST_SEASON}, min 100 swings, n={len(g)}) ===")
    print(g.head(15)[["player_name", "swings", "stuff_plus", "expected_whiff_rate", "actual_whiff_rate"]].to_string(index=False))

    print("\n=== MOST WHIFFS OVER EXPECTED (deception / sequencing beyond raw stuff) ===")
    over = g.sort_values("whiff_over_expected", ascending=False)
    print(over.head(10)[["player_name", "swings", "stuff_plus", "expected_whiff_rate", "actual_whiff_rate", "whiff_over_expected"]].to_string(index=False))

    pv = grades.predictive_validity(test, "p_stuff", min_swings=100)
    if not pv.empty:
        print(f"\n=== PREDICTIVE VALIDITY ({TEST_SEASON} first half -> second half) ===")
        print(pv.to_string(index=False))
        pv.to_csv(REPORTS / "predictive_validity.csv", index=False)

    sweep = grades.validity_by_threshold(test, "p_stuff")
    if not sweep.empty:
        print("\n=== MODEL vs OUTCOME BY SAMPLE SIZE (h2 target fixed at 150+ swings) ===")
        print(sweep.to_string(index=False))
        sweep.to_csv(REPORTS / "validity_by_threshold.csv", index=False)

    blend = grades.blend_validity(test, "p_stuff", min_swings=100)
    if not blend.empty:
        print("\n=== DOES THE MODEL ADD ANYTHING ON TOP OF OUTCOMES? ===")
        print(blend.to_string(index=False))
        blend.to_csv(REPORTS / "blend_validity.csv", index=False)

    print("\nwriting figures...", flush=True)
    plots.calibration(calib, REPORTS / "calibration.png")
    plots.whiff_by_velo(test, REPORTS / "whiff_by_velocity.png")
    plots.stuff_vs_outcome(g, REPORTS / "stuff_vs_outcome.png")
    plots.movement_map(test, REPORTS / "movement_map.png")
    print(f"done -> {REPORTS}/")


if __name__ == "__main__":
    main()
