"""Run the full pipeline: load -> fit on 2025 -> evaluate on 2026 -> report."""
from __future__ import annotations

import warnings
import argparse
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

from . import artifacts, features, grades, model, plots

DATA = Path("data")
REPORTS = Path("reports")
TRAIN_SEASON = "2025"
TEST_SEASON = "2026"


def load(season: str) -> pd.DataFrame:
    return features.build(pd.read_parquet(DATA / f"statcast_{season}.parquet"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-model", type=Path,
                        help="Save this one-time baseline fit; refuses to overwrite an artifact")
    args = parser.parse_args()
    if args.save_model and (args.save_model.exists() or args.save_model.with_suffix(".json").exists()):
        parser.error("Frozen model already exists; scoring uses python -m pitchquality.live")
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

    if args.save_model:
        train_stuff = model.predict(m_stuff, train, n_stuff)
        train_full = model.predict(m_full, train, n_full)
        y_train = train["is_whiff"].astype(int).to_numpy()
        artifacts.save(args.save_model, (m_stuff, n_stuff), (m_full, n_full), {
            "train_season": TRAIN_SEASON, "test_season": TEST_SEASON,
            "train_start": "2025-03-27", "train_end": "2025-09-28",
            "test_start": "2026-03-26", "test_end": "2026-08-19",
            "train_source_rows": len(pd.read_parquet(DATA / "statcast_2025.parquet")),
            "test_source_rows": len(pd.read_parquet(DATA / "statcast_2026.parquet")),
            "train_swings": len(train), "test_swings": len(test),
            "train_auc": model.evaluate("full train", y_train, train_full).auc,
            "test_auc": evals[2].auc,
            "stuff_train_auc": model.evaluate("stuff train", y_train, train_stuff).auc,
            "stuff_test_auc": evals[1].auc,
            "stuff_reference_mean": float(train_stuff.mean()),
            "stuff_reference_std": float(train_stuff.std(ddof=0)),
        })
        print(f"saved frozen models -> {args.save_model}", flush=True)

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
