"""Model tests. These guard the claims the writeup makes, not just the plumbing.

The expensive fits are shared through module-scoped fixtures — the point is to
check contracts and directional behavior, not to re-train once per assertion.
"""
import numpy as np
import pandas as pd
import pytest

from pitchquality import features, model


def synth(n=4000, seed=11):
    """Swings whose whiff probability is driven by velocity within pitch type.

    Built this way on purpose: a per-pitch-type baseline cannot see the signal,
    so any AUC the stuff model earns here has to come from the physics columns.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        pid = i % 20
        hand = "L" if pid % 4 == 0 else "R"
        ptype = ["FF", "SI", "SL", "CH"][i % 4]
        velo = float(rng.uniform(86.0, 101.0))
        # Whiff odds rise with velocity; the baseline model has no access to it.
        p = np.clip((velo - 86.0) / 15.0 * 0.55 + 0.05, 0.0, 1.0)
        rows.append({
            "pitch_type": ptype,
            "description": "swinging_strike" if rng.random() < p else "foul",
            "p_throws": hand,
            "stand": "R" if i % 3 else "L",
            "pitcher": pid,
            "batter": 500 + (i % 37),
            "player_name": f"Arm {pid:02d}",
            "game_date": "2026-05-01",
            "release_speed": velo,
            "release_spin_rate": float(rng.normal(2300, 150)),
            "release_extension": float(rng.normal(6.5, 0.3)),
            "release_pos_x": float(rng.normal(-1.8, 0.4)),
            "release_pos_z": float(rng.normal(6.0, 0.3)),
            "pfx_x": float(rng.normal(0.8, 0.4)),
            "pfx_z": float(rng.normal(1.4, 0.5)),
            "plate_x": float(rng.normal(0.0, 0.7)),
            "plate_z": float(rng.normal(2.4, 0.7)),
            "spin_axis": float(rng.uniform(0, 360)),
            "arm_angle": float(rng.normal(42, 8)),
            "balls": int(rng.integers(0, 4)),
            "strikes": int(rng.integers(0, 3)),
            "vx0": float(rng.normal(5.0, 2.0)),
            "vy0": -float(velo) * 1.45,
            "vz0": float(rng.normal(-4.0, 1.5)),
            "ax": float(rng.normal(-10.0, 3.0)),
            "ay": float(rng.normal(28.0, 2.0)),
            "az": float(rng.normal(-14.0, 3.0)),
        })
    return features.build(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def train():
    return synth(seed=11)


@pytest.fixture(scope="module")
def test(train):
    return synth(seed=29)


@pytest.fixture(scope="module")
def stuff_fit(train):
    return model.fit(train, include_location=False)


class TestFeatureSet:
    def test_stuff_model_cannot_see_location_or_count(self):
        # This is the whole premise of the stuff grade — if location leaks in,
        # the number stops being a property of the pitcher.
        names = model.feature_set(include_location=False)
        for leaked in features.LOCATION_FEATURES:
            assert leaked not in names

    def test_full_model_adds_location_on_top_of_stuff(self):
        stuff = model.feature_set(include_location=False)
        full = model.feature_set(include_location=True)
        assert full[:len(stuff)] == stuff
        assert set(full) - set(stuff) == set(features.LOCATION_FEATURES)


class TestDesignMatrix:
    def test_categoricals_are_encoded_for_histgbm(self, train):
        # HistGBM only splits natively on category dtype; plain ints would be
        # treated as ordered and the model would learn a nonsense ordering.
        X = model._design(train, model.feature_set(False))
        assert str(X["pitch_type"].dtype) == "category"
        assert str(X["same_hand"].dtype) == "category"

    def test_columns_are_exactly_the_named_features_plus_categoricals(self, train):
        names = model.feature_set(True)
        X = model._design(train, names)
        assert list(X.columns) == names + features.CATEGORICAL

    def test_design_does_not_mutate_the_caller_frame(self, train):
        before = str(train["same_hand"].dtype)
        model._design(train, model.feature_set(False))
        assert str(train["same_hand"].dtype) == before


class TestBaseline:
    def test_rates_are_per_pitch_type_training_means(self, train):
        rates = model.baseline_rates(train)
        expected = train[train["pitch_type"] == "FF"]["is_whiff"].mean()
        assert rates["FF"] == pytest.approx(expected)

    def test_unseen_pitch_type_falls_back_instead_of_going_nan(self, train):
        rates = model.baseline_rates(train)
        unseen = train.head(3).copy()
        unseen["pitch_type"] = "SV"  # never thrown in the training frame
        pred = model.baseline_predict(rates, unseen)
        assert not np.isnan(pred).any()
        assert pred == pytest.approx(float(rates.mean()))

    def test_predictions_are_probabilities(self, train, test):
        pred = model.baseline_predict(model.baseline_rates(train), test)
        assert len(pred) == len(test)
        assert ((pred >= 0.0) & (pred <= 1.0)).all()


class TestFitAndPredict:
    def test_predictions_are_probabilities_of_the_whiff_class(self, stuff_fit, test):
        fitted, names = stuff_fit
        p = model.predict(fitted, test, names)
        assert len(p) == len(test)
        assert ((p >= 0.0) & (p <= 1.0)).all()
        # Column 1 must be P(whiff); if the classes were swapped the mean
        # prediction would sit near 1 - the observed rate.
        assert p.mean() == pytest.approx(test["is_whiff"].mean(), abs=0.06)

    def test_fitting_is_deterministic(self, train, test, stuff_fit):
        fitted, names = stuff_fit
        again, names2 = model.fit(train, include_location=False)
        assert names == names2
        np.testing.assert_allclose(
            model.predict(fitted, test, names),
            model.predict(again, test, names2),
        )

    def test_stuff_model_beats_the_pitch_type_baseline(self, train, test, stuff_fit):
        fitted, names = stuff_fit
        base = model.evaluate(
            "baseline", test["is_whiff"].astype(int),
            model.baseline_predict(model.baseline_rates(train), test),
        )
        stuff = model.evaluate(
            "stuff", test["is_whiff"].astype(int),
            model.predict(fitted, test, names),
        )
        # Earning its complexity is the minimum bar stated in the module docstring.
        assert stuff.auc > base.auc
        assert stuff.log_loss < base.log_loss

    def test_model_recovers_the_planted_velocity_signal(self, stuff_fit, test):
        fitted, names = stuff_fit
        p = model.predict(fitted, test, names)
        hard = p[test["release_speed"] > 98.0].mean()
        soft = p[test["release_speed"] < 88.0].mean()
        assert hard > soft


class TestEvaluate:
    def test_perfect_separation_scores_auc_one(self):
        y = np.array([0, 0, 1, 1])
        ev = model.evaluate("perfect", y, np.array([0.01, 0.02, 0.98, 0.99]))
        assert ev.auc == pytest.approx(1.0)
        assert ev.n == 4

    def test_row_is_rounded_for_reporting(self):
        ev = model.Evaluation(name="stuff", auc=0.7671234, log_loss=0.612345,
                              brier=0.1234567, n=100)
        row = ev.row()
        assert row == {"model": "stuff", "auc": 0.7671, "log_loss": 0.6123,
                       "brier": 0.12346, "n_pitches": 100}


class TestCalibration:
    def test_gap_is_observed_minus_predicted(self):
        rng = np.random.default_rng(5)
        p = rng.uniform(0.05, 0.6, 2000)
        y = (rng.random(2000) < p).astype(int)
        tab = model.calibration_table(y, p)
        np.testing.assert_allclose(tab["gap"], tab["observed"] - tab["predicted"])

    def test_every_row_is_accounted_for(self):
        rng = np.random.default_rng(6)
        p = rng.uniform(0.05, 0.6, 1500)
        y = (rng.random(1500) < p).astype(int)
        assert model.calibration_table(y, p)["n"].sum() == 1500

    def test_bins_are_ordered_by_predicted_probability(self):
        rng = np.random.default_rng(7)
        p = rng.uniform(0.05, 0.6, 2000)
        y = (rng.random(2000) < p).astype(int)
        tab = model.calibration_table(y, p)
        assert tab["predicted"].is_monotonic_increasing

    def test_a_well_calibrated_model_shows_small_gaps(self):
        # Probabilities generated from the truth, so any large gap here would
        # mean the table itself is wrong rather than the model.
        rng = np.random.default_rng(8)
        p = rng.uniform(0.05, 0.6, 20000)
        y = (rng.random(20000) < p).astype(int)
        assert model.calibration_table(y, p)["gap"].abs().max() < 0.03

    def test_degenerate_probabilities_do_not_raise(self):
        # qcut on a constant column has no unique edges; duplicates="drop"
        # is what keeps a single-value prediction from crashing the report.
        y = np.array([0, 1, 0, 1, 1, 0])
        tab = model.calibration_table(y, np.full(6, 0.25))
        assert len(tab) >= 1
        assert tab["n"].sum() == 6
