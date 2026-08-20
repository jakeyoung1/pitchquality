"""Tests for the Stuff+ index and the predictive-validity machinery."""
import numpy as np
import pandas as pd
import pytest

from pitchquality import grades


def synth(n_pitchers=40, swings=200, seed=3):
    """Pitchers with a stable latent skill, observed with binomial noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_pitchers):
        skill = rng.uniform(0.12, 0.38)
        for i in range(swings):
            # First half = Apr/May, second half = Jul/Aug.
            date = "2026-04-15" if i < swings // 2 else "2026-08-01"
            rows.append({
                "pitcher": pid,
                "player_name": f"Arm {pid:02d}",
                "game_date": date,
                "is_whiff": bool(rng.random() < skill),
                "p_stuff": skill + rng.normal(0, 0.01),
            })
    return pd.DataFrame(rows)


class TestScaleIndex:
    def test_index_is_centered_at_100_with_sd_10(self):
        idx = grades.scale_index(pd.Series([0.1, 0.2, 0.3, 0.4, 0.5]))
        assert idx.mean() == pytest.approx(100.0)
        assert idx.std(ddof=0) == pytest.approx(10.0)

    def test_constant_input_does_not_divide_by_zero(self):
        idx = grades.scale_index(pd.Series([0.25] * 6))
        assert (idx == 100.0).all()

    def test_ordering_is_preserved(self):
        raw = pd.Series([0.4, 0.1, 0.25])
        assert list(grades.scale_index(raw).rank()) == list(raw.rank())


class TestPitcherGrades:
    def test_min_swings_filter_applies(self):
        df = synth(n_pitchers=3, swings=40)
        assert len(grades.pitcher_grades(df, "p_stuff", min_swings=10)) == 3
        assert grades.pitcher_grades(df, "p_stuff", min_swings=500).empty

    def test_whiff_over_expected_is_actual_minus_expected(self):
        g = grades.pitcher_grades(synth(n_pitchers=5), "p_stuff", min_swings=10)
        recomputed = (g["actual_whiff_rate"] - g["expected_whiff_rate"]).round(4)
        assert np.allclose(g["whiff_over_expected"], recomputed, atol=1e-4)

    def test_output_is_sorted_by_stuff_plus_descending(self):
        g = grades.pitcher_grades(synth(), "p_stuff", min_swings=10)
        assert g["stuff_plus"].is_monotonic_decreasing


class TestSplitHalves:
    def test_halves_partition_the_data_without_overlap(self):
        df = synth(n_pitchers=4, swings=100)
        a, b = grades.split_halves(df)
        assert len(a) + len(b) == len(df)
        assert pd.to_datetime(a["game_date"]).max() <= pd.to_datetime(b["game_date"]).min()


class TestValidity:
    def test_a_near_perfect_predictor_scores_high(self):
        # p_stuff is latent skill plus tiny noise, so it should track
        # second-half results closely on synthetic data.
        pv = grades.predictive_validity(synth(swings=400), "p_stuff", min_swings=50)
        assert not pv.empty
        model_r = pv.loc[pv["predictor"].str.contains("model"), "r"].iloc[0]
        assert model_r > 0.8

    def test_returns_empty_when_too_few_pitchers_clear_the_bar(self):
        assert grades.predictive_validity(synth(n_pitchers=2), "p_stuff", min_swings=10_000).empty

    def test_threshold_sweep_reports_one_row_per_usable_cutoff(self):
        sweep = grades.validity_by_threshold(synth(swings=400), "p_stuff", thresholds=(50, 100))
        assert list(sweep["min_swings_h1"]) == [50, 100]
        assert (sweep["n_pitchers"] > 0).all()

    def test_blend_reports_all_three_predictors(self):
        b = grades.blend_validity(synth(swings=400), "p_stuff", min_swings=50)
        assert list(b["predictor"]) == ["model alone", "outcome alone", "50/50 blend"]
