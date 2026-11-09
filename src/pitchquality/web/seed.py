"""Build a contract-shaped ``data/pitchquality.db`` from the committed reports.

The data pipeline (relay issue #132) owns the real database. Until it lands,
this seed lets the API and dashboard run against real numbers: the 2026
pitcher grades and held-out AUCs already committed under ``reports/``.

Nothing is made up. Each value is copied from a report, and anything the
reports do not hold is left empty:

- ``leaderboard`` gets one row per pitcher in ``pitcher_stuff_grades.csv``.
- ``pitches`` is created with the contract columns and left with zero rows,
  because the repo commits no pitch-level data.
- ``meta.train_auc`` is NULL. The reports record held-out AUCs only.

Run ``python -m pitchquality.web.seed``. With ``--if-missing`` it leaves an
existing database alone, so a pipeline-built file is never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

from .. import __version__
from . import facts
from .db import REPO_ROOT, default_db_path

SEASON = 2026

SCHEMA = """
CREATE TABLE pitches (
    game_date      TEXT    NOT NULL,
    pitcher        INTEGER NOT NULL,
    pitch_type     TEXT    NOT NULL,
    stuff_score    REAL,
    expected_whiff REAL,
    actual_whiff   INTEGER
);
CREATE INDEX idx_pitches_pitcher ON pitches (pitcher, game_date);

CREATE TABLE leaderboard (
    pitcher             INTEGER NOT NULL,
    season              INTEGER NOT NULL,
    pitches             INTEGER NOT NULL,
    avg_stuff           REAL,
    expected_whiff      REAL,
    actual_whiff        REAL,
    whiff_over_expected REAL,
    player_name         TEXT,
    PRIMARY KEY (pitcher, season)
);

CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _auc(performance: list[dict[str, str]], prefix: str) -> str:
    for row in performance:
        if row["model"].startswith(prefix):
            return row["auc"]
    raise ValueError(f"no '{prefix}' row in model_performance.csv")


def build(db_path: Path, reports_dir: Path) -> int:
    """Write the database and return the number of leaderboard rows."""
    grades = _read_csv(reports_dir / "pitcher_stuff_grades.csv")
    performance = _read_csv(reports_dir / "model_performance.csv")

    leaderboard = [
        (
            int(g["pitcher"]),
            SEASON,
            # The model scores swings only, so the count of scored pitches is
            # the swing count. meta.sample_unit records that.
            int(g["swings"]),
            float(g["stuff_plus"]),
            float(g["expected_whiff_rate"]),
            float(g["actual_whiff_rate"]),
            float(g["whiff_over_expected"]),
            g["player_name"],
        )
        for g in grades
    ]
    meta = [
        # Contract keys.
        ("model_version", __version__),
        ("last_fetch_date", facts.REPORT_LAST_GAME_DATE),
        ("train_auc", None),
        ("test_auc", _auc(performance, "full")),
        # Extensions the dashboard reads when present.
        ("test_auc_stuff", _auc(performance, "stuff")),
        ("first_game_date", facts.REPORT_FIRST_GAME_DATE),
        ("sample_unit", "swings"),
        ("stuff_scale", "stuff_plus"),
        ("source", "seed:reports"),
    ]

    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".db.tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    try:
        con.executescript(SCHEMA)
        con.executemany("INSERT INTO leaderboard VALUES (?, ?, ?, ?, ?, ?, ?, ?)", leaderboard)
        con.executemany("INSERT INTO meta VALUES (?, ?)", meta)
        con.commit()
    finally:
        con.close()
    tmp.replace(db_path)
    return len(leaderboard)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=default_db_path())
    parser.add_argument("--reports", type=Path, default=REPO_ROOT / "reports")
    parser.add_argument(
        "--if-missing",
        action="store_true",
        help="do nothing when the database file already exists",
    )
    args = parser.parse_args(argv)

    if args.if_missing and args.db.is_file():
        print(f"{args.db} exists, leaving it alone")
        return 0
    n = build(args.db, args.reports)
    print(f"seeded {args.db} from {args.reports}: {n} leaderboard rows, 0 pitch rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
