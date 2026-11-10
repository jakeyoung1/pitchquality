"""Read-only access to ``data/pitchquality.db``.

The file is written by the live pipeline (``pitchquality.live``) and its schema
is ``src/pitchquality/schema.sql``, version 1:

- ``pitches``: one row per scored swing. ``expected_whiff`` is the full model's
  probability, ``stuff_whiff`` the stuff model's, ``stuff_score`` the stuff
  probability on a 100/10 scale frozen to the 2025 training swings.
- ``leaderboard``: one row per pitcher and season, the means of those columns.
  ``pitches`` counts scored swings. No sample minimum is applied in storage.
- ``meta``: TEXT key/value rows.

Storage keeps every pitcher, so ranking is done here: a pitcher qualifies for a
rank with at least ``facts.QUALIFIED_MIN_SWINGS`` swings.

Nothing here writes. The connection is opened with ``mode=ro``.
"""
from __future__ import annotations

import os
import sqlite3
import statistics
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from . import facts

REPO_ROOT = Path(__file__).resolve().parents[3]

LEADERBOARD_COLUMNS = (
    "pitcher",
    "season",
    "player_name",
    "pitches",
    "avg_stuff",
    "expected_whiff",
    "actual_whiff",
    "whiff_over_expected",
)
PITCH_COLUMNS = (
    "game_date",
    "pitcher",
    "pitch_type",
    "stuff_score",
    "expected_whiff",
    "actual_whiff",
    "stuff_whiff",
    "season",
    "game_type",
)

_AGGREGATES = (
    "COUNT(*) AS pitches, AVG(stuff_score) AS avg_stuff, AVG(stuff_whiff) AS stuff_whiff, "
    "AVG(expected_whiff) AS expected_whiff, AVG(actual_whiff) AS actual_whiff"
)


class DatabaseUnavailable(RuntimeError):
    """The database file is missing or does not match the schema."""


def default_db_path() -> Path:
    return Path(os.environ.get("PITCHQUALITY_DB") or REPO_ROOT / "data" / "pitchquality.db")


def display_name(raw: str) -> str:
    """Statcast stores names as ``Last, First``; show ``First Last``."""
    last, sep, first = raw.partition(",")
    if not sep or not first.strip():
        return raw.strip()
    return f"{first.strip()} {last.strip()}"


def fold(text: str) -> str:
    """Lowercase and strip accents so a search for ``munoz`` finds ``Muñoz``."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


@dataclass(frozen=True)
class Snapshot:
    """``leaderboard`` and ``meta``, loaded once per version of the file."""

    meta: dict[str, str]
    rows: tuple[dict, ...]
    seasons: tuple[int, ...]

    def season_rows(self, season) -> list[dict]:
        return [r for r in self.rows if str(r["season"]) == str(season)]

    def qualified(self, season) -> list[dict]:
        """Ranked pitchers for a season, best stuff first."""
        rows = [r for r in self.season_rows(season) if r["qualified"]]
        return sorted(rows, key=lambda r: r["rank"])

    def find(self, pitcher_id: str) -> list[dict]:
        """Every season line for one pitcher, newest season first."""
        hits = [r for r in self.rows if r["id"] == pitcher_id]
        return sorted(hits, key=lambda r: r["season"], reverse=True)


class Database:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else default_db_path()
        self._stamp: tuple[int, int] | None = None
        self._snapshot: Snapshot | None = None

    def _connect(self) -> sqlite3.Connection:
        if not self.path.is_file():
            raise DatabaseUnavailable(f"{self.path.name} is not in this deploy")
        uri = f"file:{quote(self.path.resolve().as_posix())}?mode=ro"
        try:
            con = sqlite3.connect(uri, uri=True, check_same_thread=False)
        except sqlite3.Error as exc:  # pragma: no cover - depends on the OS
            raise DatabaseUnavailable(str(exc)) from exc
        con.row_factory = sqlite3.Row
        return con

    @staticmethod
    def _require(con: sqlite3.Connection, table: str, columns: tuple[str, ...]) -> None:
        have = {row["name"] for row in con.execute(f"PRAGMA table_info({table})")}
        missing = [c for c in columns if c not in have]
        if missing:
            raise DatabaseUnavailable(
                f"{table} table is missing schema columns: " + ", ".join(missing)
            )

    def snapshot(self) -> Snapshot:
        """Load ``leaderboard`` and ``meta``, reusing the copy in memory until
        the file on disk changes."""
        if not self.path.is_file():
            raise DatabaseUnavailable(f"{self.path.name} is not in this deploy")
        stat = self.path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        if self._snapshot is not None and stamp == self._stamp:
            return self._snapshot

        con = self._connect()
        try:
            self._require(con, "leaderboard", LEADERBOARD_COLUMNS)
            self._require(con, "pitches", PITCH_COLUMNS)
            self._require(con, "meta", ("key", "value"))
            cur = con.execute(f"SELECT {', '.join(LEADERBOARD_COLUMNS)} FROM leaderboard")
            rows = [self._shape(dict(r)) for r in cur]
            meta = {r["key"]: r["value"] for r in con.execute("SELECT key, value FROM meta")}
        except sqlite3.Error as exc:
            raise DatabaseUnavailable(str(exc)) from exc
        finally:
            con.close()

        self._rank(rows)
        seasons = tuple(sorted({r["season"] for r in rows}, reverse=True))
        self._snapshot = Snapshot(meta, tuple(rows), seasons)
        self._stamp = stamp
        return self._snapshot

    @staticmethod
    def _shape(row: dict) -> dict:
        raw_name = row.pop("player_name")
        row["id"] = str(row["pitcher"])
        row["name"] = display_name(raw_name)
        row["search"] = fold(f"{row['name']} {raw_name} {row['id']}")
        row["qualified"] = row["pitches"] >= facts.QUALIFIED_MIN_SWINGS
        row["rank"] = row["woe_rank"] = row["stuff_z"] = None
        return row

    @staticmethod
    def _rank(rows: list[dict]) -> None:
        """Rank the qualified pitchers of each season, by stuff score and by
        whiffs over expected, and place each stuff score among its peers."""
        by_season: dict[int, list[dict]] = {}
        for r in rows:
            by_season.setdefault(r["season"], []).append(r)
        for group in by_season.values():
            qualified = [r for r in group if r["qualified"]]
            for key, field in (("rank", "avg_stuff"), ("woe_rank", "whiff_over_expected")):
                ordered = sorted(qualified, key=lambda r: (-r[field], r["name"], r["pitcher"]))
                for i, r in enumerate(ordered, start=1):
                    r[key] = i
            scores = [r["avg_stuff"] for r in qualified]
            spread = statistics.pstdev(scores) if len(scores) > 1 else 0.0
            center = statistics.fmean(scores) if scores else 0.0
            for r in qualified:
                r["stuff_z"] = (r["avg_stuff"] - center) / spread if spread else 0.0
            for r in group:
                r["season_qualified"] = len(qualified)
                r["season_size"] = len(group)

    def pitch_breakdown(self, pitcher: int, season: int) -> dict:
        """Aggregates of the ``pitches`` table for one pitcher's season: the
        whole season, then by pitch type, month, and game type."""
        where = "FROM pitches WHERE pitcher = ? AND season = ?"
        args = (pitcher, season)
        con = self._connect()
        try:
            total = dict(con.execute(f"SELECT {_AGGREGATES} {where}", args).fetchone())
            by_type = [
                dict(r)
                for r in con.execute(
                    f"SELECT pitch_type, {_AGGREGATES} {where} "
                    "GROUP BY pitch_type ORDER BY pitches DESC, pitch_type",
                    args,
                )
            ]
            by_month = [
                dict(r)
                for r in con.execute(
                    f"SELECT substr(game_date, 1, 7) AS month, {_AGGREGATES} {where} "
                    "GROUP BY month ORDER BY month",
                    args,
                )
            ]
            by_stage = [
                dict(r)
                for r in con.execute(
                    f"SELECT game_type, MIN(game_date) AS first_date, {_AGGREGATES} {where} "
                    "GROUP BY game_type ORDER BY first_date",
                    args,
                )
            ]
        except sqlite3.Error as exc:
            raise DatabaseUnavailable(str(exc)) from exc
        finally:
            con.close()

        count = total["pitches"] or 0
        for r in by_type:
            r["share"] = r["pitches"] / count
        for r in by_stage:
            r.pop("first_date")
        for r in [total, *by_type, *by_month, *by_stage]:
            if r["pitches"]:
                r["whiff_over_expected"] = r["actual_whiff"] - r["expected_whiff"]
            else:
                r["whiff_over_expected"] = None
        return {"total": total, "by_type": by_type, "by_month": by_month, "by_stage": by_stage}
