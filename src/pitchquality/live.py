"""Score the 2026 season with frozen models and atomically refresh SQLite."""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

from . import artifacts, features, model
from .fetch import WINDOWS, fetch_cached

SEASON = 2026
SEASON_START = date(2026, 3, 26)
BACKFILL_START = date(2026, 8, 20)
# Last final game in MLB's 2026 schedule, including the postseason.
SEASON_END = date(2026, 10, 28)
IDENTITY = ["game_pk", "at_bat_number", "pitch_number"]
COLUMNS = IDENTITY + ["game_date", "pitcher", "player_name", "pitch_type", "stuff_score",
                       "expected_whiff", "actual_whiff", "stuff_whiff", "season", "game_type"]


def read_meta(db: Path) -> dict[str, str]:
    if not db.exists():
        return {}
    with sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True) as conn:
        return dict(conn.execute("SELECT key, value FROM meta"))


def score(raw: pd.DataFrame, bundle: dict, meta: dict) -> pd.DataFrame:
    """Reuse the original feature/predict paths on a whole-season window.

    Rebuilding the whole season preserves the original fastball-reference
    behavior. Scoring daily chunks separately would change differentials.
    """
    if raw[IDENTITY + ["game_date", "pitcher", "game_type"]].isna().any().any():
        raise ValueError("Statcast pitch identity/date/pitcher/game_type is missing")
    raw = raw.drop_duplicates(IDENTITY, keep="last")
    frame = features.build(raw)
    if frame.empty:
        return pd.DataFrame(columns=COLUMNS)
    stuff_estimator, stuff_names = bundle["stuff"]
    full_estimator, full_names = bundle["full"]
    stuff = model.predict(stuff_estimator, frame, stuff_names)
    full = model.predict(full_estimator, frame, full_names)
    if not (np.isfinite(stuff).all() and np.isfinite(full).all()):
        raise ValueError("Model returned non-finite probabilities")
    std = meta["stuff_reference_std"]
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid frozen Stuff score reference")
    frame["stuff_score"] = 100.0 + 10.0 * (stuff - meta["stuff_reference_mean"]) / std
    frame["stuff_whiff"] = stuff
    frame["expected_whiff"] = full
    frame["actual_whiff"] = frame["is_whiff"].astype(int)
    dates = pd.to_datetime(frame["game_date"])
    frame["game_date"] = dates.dt.strftime("%Y-%m-%d")
    frame["season"] = dates.dt.year.astype(int)
    frame["player_name"] = frame["player_name"].fillna("")
    for col in IDENTITY + ["pitcher"]:
        frame[col] = frame[col].astype("int64")
    return frame[COLUMNS].sort_values(IDENTITY).reset_index(drop=True)


def persist(db: Path, scored: pd.DataFrame, metadata: dict) -> None:
    """Replace the season and its aggregates in one transaction, including meta.

    A repeated run replaces the same stable pitch identities; a correction
    that makes an old pitch ineligible also removes the stale scored row.
    """
    if scored.empty:
        raise ValueError("Refusing to replace a season with no scored swings")
    if set(scored["season"]) != {SEASON}:
        raise ValueError("Only the 2026 season can be written by this pipeline")
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.executescript(files("pitchquality").joinpath("schema.sql").read_text(encoding="utf-8"))
        with conn:
            old_version = conn.execute("SELECT value FROM meta WHERE key='model_version'").fetchone()
            if old_version and old_version[0] != metadata["model_version"]:
                raise ValueError("Database belongs to a different frozen model")
            conn.execute("DELETE FROM pitches WHERE season=?", (SEASON,))
            placeholders = ",".join("?" for _ in COLUMNS)
            conn.executemany(f"INSERT INTO pitches ({','.join(COLUMNS)}) VALUES ({placeholders})",
                             scored[COLUMNS].itertuples(index=False, name=None))
            conn.execute("DELETE FROM leaderboard WHERE season=?", (SEASON,))
            conn.execute("""
                INSERT INTO leaderboard
                SELECT pitcher, season, MAX(player_name), COUNT(*), AVG(stuff_score),
                       AVG(expected_whiff), AVG(actual_whiff), AVG(actual_whiff)-AVG(expected_whiff)
                FROM pitches WHERE season=? GROUP BY pitcher, season
            """, (SEASON,))
            conn.executemany("INSERT INTO meta(key,value) VALUES (?,?) "
                             "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                             [(k, str(v)) for k, v in metadata.items()])


def run(db: Path = Path("data/pitchquality.db"),
        model_path: Path = Path("models/whiff-2025.joblib"), *,
        today: date | None = None, backfill: bool = False) -> dict:
    """Fetch, score, and commit only on success; the offseason does no work.

    --backfill explicitly replays completed dates, including source corrections.
    Regular refresh stops at yesterday and never extends the 2026 season.
    """
    today = today or datetime.now(timezone.utc).date()
    end = min(SEASON_END, today - timedelta(days=1))
    if backfill:
        if today <= SEASON_END:
            raise ValueError("Full-season backfill requires the 2026 season to be complete")
        end = SEASON_END
    previous = read_meta(db)
    if end < SEASON_START or (not backfill and previous.get("last_fetch_date", "") >= end.isoformat()):
        return {"status": "no-op", "last_fetch_date": previous.get("last_fetch_date"), "rows_scored": 0}
    # No fallback to training if the artifact is absent, corrupt, or incompatible.
    bundle, model_meta = artifacts.load(model_path)
    if previous.get("model_version", model_meta["model_version"]) != model_meta["model_version"]:
        raise ValueError("Database belongs to a different frozen model")
    data = db.parent
    baseline_end = date.fromisoformat(WINDOWS["2026"][1])
    if end <= baseline_end:
        raw = fetch_cached(SEASON_START.isoformat(), end.isoformat(),
                           data / f"statcast_live_{end}.parquet", force=backfill)
    else:
        baseline = fetch_cached(*WINDOWS["2026"], data / "statcast_2026.parquet", force=backfill)
        tail = fetch_cached(BACKFILL_START.isoformat(), end.isoformat(),
                            data / f"statcast_backfill_{end}.parquet", force=backfill)
        raw = pd.concat([baseline, tail], ignore_index=True)
    if raw.empty:
        raise ValueError("No Statcast data returned for the season; watermark was not advanced")
    if not set(IDENTITY + ["game_type"]).issubset(raw.columns):
        raise ValueError("Old cache lacks pitch identities/game_type; remove cache and fetch again")
    dates = pd.to_datetime(raw["game_date"])
    if not dates.between(pd.Timestamp(SEASON_START), pd.Timestamp(end)).all():
        raise ValueError("Statcast response contains dates outside the requested season window")
    raw = raw.drop_duplicates(IDENTITY, keep="last")
    scored = score(raw, bundle, model_meta)
    metrics = {key: model_meta[key] for key in (
        "model_version", "train_auc", "test_auc", "stuff_train_auc", "stuff_test_auc",
        "stuff_reference_mean", "stuff_reference_std")}
    summary = {**metrics, "schema_version": 1, "score_scope": "eligible swings from features.build",
               "season": SEASON, "season_start": SEASON_START.isoformat(),
               "season_end": SEASON_END.isoformat(), "backfill_start": BACKFILL_START.isoformat(),
               "last_fetch_date": end.isoformat(), "last_pitch_date": scored["game_date"].max(),
               "source_rows": len(raw), "rows_scored": len(scored),
               "backfill_source_rows": int((pd.to_datetime(raw["game_date"]).dt.date >= BACKFILL_START).sum()),
               "backfill_rows_scored": int((scored["game_date"] >= BACKFILL_START.isoformat()).sum())}
    persist(db, scored, summary)
    return {"status": "updated", **summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/pitchquality.db"))
    parser.add_argument("--model", type=Path, default=Path("models/whiff-2025.joblib"))
    parser.add_argument("--backfill", action="store_true", help="Explicitly replay the complete 2026 season")
    args = parser.parse_args()
    result = run(args.db, args.model, backfill=args.backfill)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["status"] == "updated":
        reports = Path("reports")
        reports.mkdir(exist_ok=True)
        (reports / "live_pipeline.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
