"""Live contracts: unchanged scoring, atomic storage, replay, and no-fit refresh."""
from datetime import date
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest

from pitchquality import artifacts, features, fetch, live, model
from test_features import frame


class Estimator:
    def __init__(self, base):
        self.base = base

    def predict_proba(self, design):
        p = self.base + (design["release_speed"].to_numpy() - 95) * 0.01
        return np.column_stack([1 - p, p])


@pytest.fixture
def raw():
    df = frame([
        {"description": "swinging_strike", "release_speed": 96},
        {"description": "foul_tip", "release_speed": 94},
        {"description": "ball"},
        {"pitch_type": "KN"},
        {"release_speed": np.nan},
    ])
    df["game_pk"] = 100
    df["at_bat_number"] = 1
    df["pitch_number"] = range(1, len(df) + 1)
    df["game_type"] = "R"
    df["game_date"] = "2026-08-20"
    return df


@pytest.fixture
def bundle():
    return {"stuff": (Estimator(0.2), model.feature_set(False)),
            "full": (Estimator(0.3), model.feature_set(True))}


@pytest.fixture
def metadata():
    return {"model_version": "fixture-2025", "train_auc": 0.8, "test_auc": 0.77,
            "stuff_train_auc": 0.79, "stuff_test_auc": 0.76,
            "stuff_reference_mean": 0.2, "stuff_reference_std": 0.1,
            "last_fetch_date": "2026-08-20"}


def test_live_matches_the_original_feature_and_predict_paths(raw, bundle, metadata, monkeypatch):
    monkeypatch.setattr(model, "fit", lambda *a, **kw: pytest.fail("live must never fit"))
    expected = features.build(raw)
    scored = live.score(raw, bundle, metadata)
    assert scored["pitch_number"].tolist() == [1, 2]
    assert scored["actual_whiff"].tolist() == [1, 0]
    np.testing.assert_array_equal(scored["expected_whiff"],
                                  model.predict(bundle["full"][0], expected, bundle["full"][1]))
    np.testing.assert_allclose(scored["stuff_score"], [101.0, 99.0])
    assert raw.shape[0] == 5  # Caller retains takes and other excluded inputs.


def test_missing_identity_fails_instead_of_collapsing_pitches(raw, bundle, metadata):
    raw.loc[0, "pitch_number"] = np.nan
    with pytest.raises(ValueError, match="identity"):
        live.score(raw, bundle, metadata)


def test_replay_is_idempotent_and_replaces_corrected_or_excluded_rows(tmp_path, raw, bundle, metadata):
    db = tmp_path / "pitchquality.db"
    live.persist(db, live.score(raw, bundle, metadata), metadata)
    live.persist(db, live.score(raw, bundle, metadata), metadata)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM pitches").fetchone()[0] == 2
        assert conn.execute("SELECT pitches, actual_whiff, expected_whiff FROM leaderboard").fetchone() == pytest.approx((2, 0.5, 0.3))
    raw.loc[0, "description"] = "ball"
    live.persist(db, live.score(raw, bundle, metadata), metadata)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT pitch_number, actual_whiff FROM pitches").fetchall() == [(2, 0)]
        assert conn.execute("SELECT pitches, whiff_over_expected FROM leaderboard").fetchone() == pytest.approx((1, -0.29))


def test_failed_write_rolls_back_pitches_leaderboard_and_watermark(tmp_path, raw, bundle, metadata):
    db = tmp_path / "pitchquality.db"
    scored = live.score(raw, bundle, metadata)
    live.persist(db, scored, metadata)
    invalid = scored.copy()
    invalid.loc[1, "expected_whiff"] = 1.5
    with pytest.raises(sqlite3.IntegrityError):
        live.persist(db, invalid, {**metadata, "last_fetch_date": "2026-08-21"})
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM pitches").fetchone()[0] == 2
        assert conn.execute("SELECT pitches FROM leaderboard").fetchone()[0] == 2
    assert live.read_meta(db)["last_fetch_date"] == "2026-08-20"


def test_cannot_mix_model_versions(tmp_path, raw, bundle, metadata):
    db = tmp_path / "pitchquality.db"
    scored = live.score(raw, bundle, metadata)
    live.persist(db, scored, metadata)
    with pytest.raises(ValueError, match="different frozen model"):
        live.persist(db, scored, {**metadata, "model_version": "other"})
    assert live.read_meta(db)["model_version"] == metadata["model_version"]


def test_offseason_noop_needs_no_network_or_model(tmp_path, raw, bundle, metadata, monkeypatch):
    db = tmp_path / "pitchquality.db"
    live.persist(db, live.score(raw, bundle, metadata), {**metadata, "last_fetch_date": "2026-10-28"})
    before = db.read_bytes()
    monkeypatch.setattr(live, "fetch_cached", lambda *a, **kw: pytest.fail("no fetch in offseason"))
    monkeypatch.setattr(artifacts, "load", lambda *a: pytest.fail("no model load in offseason"))
    result = live.run(db, today=date(2026, 11, 9))
    assert result["status"] == "no-op"
    assert db.read_bytes() == before


def test_full_season_refresh_and_failed_fetch_preserve_watermark(tmp_path, raw, bundle, metadata, monkeypatch):
    db = tmp_path / "pitchquality.db"
    baseline = raw.copy()
    baseline["game_date"] = "2026-08-19"
    baseline["game_pk"] = 99
    calls = []
    def fake_fetch(start, end, path, **kwargs):
        calls.append((start, end))
        return baseline if start == "2026-03-26" else raw
    monkeypatch.setattr(live, "fetch_cached", fake_fetch)
    monkeypatch.setattr(artifacts, "load", lambda path: (bundle, metadata))
    monkeypatch.setattr(model, "fit", lambda *a, **kw: pytest.fail("refresh must not fit"))
    result = live.run(db, today=date(2026, 8, 21))
    assert calls == [("2026-03-26", "2026-08-19"), ("2026-08-20", "2026-08-20")]
    assert result["rows_scored"] == 4
    assert result["backfill_rows_scored"] == 2
    assert result["last_fetch_date"] == "2026-08-20"
    before = db.read_bytes()
    def fail_fetch(*args, **kwargs):
        raise ConnectionError("upstream failure")
    monkeypatch.setattr(live, "fetch_cached", fail_fetch)
    with pytest.raises(ConnectionError):
        live.run(db, today=date(2026, 8, 22))
    assert db.read_bytes() == before


def test_backfill_uses_season_end_and_can_replay_completed_dates(tmp_path, raw, bundle, metadata, monkeypatch):
    db = tmp_path / "pitchquality.db"
    baseline = raw.copy()
    baseline["game_date"] = "2026-08-19"
    baseline["game_pk"] = 99
    calls = []
    def fake_fetch(start, end, path, **kwargs):
        calls.append((start, end, kwargs.get("force")))
        return baseline if start == "2026-03-26" else raw
    monkeypatch.setattr(live, "fetch_cached", fake_fetch)
    monkeypatch.setattr(artifacts, "load", lambda path: (bundle, metadata))
    result = live.run(db, today=date(2026, 11, 9), backfill=True)
    assert calls[-1] == ("2026-08-20", "2026-10-28", True)
    assert result["last_fetch_date"] == "2026-10-28"
    calls.clear()
    live.run(db, today=date(2026, 11, 9), backfill=True)
    assert len(calls) == 2


def test_empty_response_cannot_advance_watermark(tmp_path, bundle, metadata, monkeypatch):
    db = tmp_path / "pitchquality.db"
    monkeypatch.setattr(artifacts, "load", lambda path: (bundle, metadata))
    monkeypatch.setattr(live, "fetch_cached", lambda *a, **kw: pd.DataFrame())
    with pytest.raises(ValueError, match="watermark"):
        live.run(db, today=date(2026, 8, 21))
    assert not db.exists()


def test_fetch_errors_are_not_cached_and_successes_are_reused(tmp_path, raw, monkeypatch):
    path = tmp_path / "cached.parquet"
    def fail(**kwargs):
        raise ConnectionError("upstream failure")
    monkeypatch.setattr(fetch, "statcast", fail)
    with pytest.raises(ConnectionError):
        fetch.fetch_cached("2026-08-20", "2026-08-20", path)
    assert not path.exists()
    monkeypatch.setattr(fetch, "statcast", lambda **kw: raw)
    fetched = fetch.fetch_cached("2026-08-20", "2026-08-20", path)
    monkeypatch.setattr(fetch, "statcast", fail)
    pd.testing.assert_frame_equal(fetch.fetch_cached("2026-08-20", "2026-08-20", path), fetched)


def test_shipped_database_matches_frozen_model():
    root = Path(__file__).resolve().parents[1]
    bundle, manifest = artifacts.load(root / "models/whiff-2025.joblib")
    db = root / "data/pitchquality.db"
    metadata = live.read_meta(db)
    assert metadata["model_version"] == manifest["model_version"]
    assert float(metadata["train_auc"]) == manifest["train_auc"]
    assert float(metadata["test_auc"]) == manifest["test_auc"]
    with sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT COUNT(*) FROM pitches").fetchone()[0] == int(metadata["rows_scored"])
        assert conn.execute("SELECT SUM(pitches) FROM leaderboard").fetchone()[0] == int(metadata["rows_scored"])
        assert conn.execute("SELECT MAX(game_date) FROM pitches").fetchone()[0] == "2026-10-28"
    assert set(bundle) == {"stuff", "full"}
