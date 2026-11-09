"""Read-only access to ``data/pitchquality.db``.

The storage contract (relay issue #132) fixes a minimum set of columns:

- ``pitches``: game_date, pitcher, pitch_type, stuff_score, expected_whiff,
  actual_whiff
- ``leaderboard``: pitcher, season, pitches, avg_stuff, expected_whiff,
  actual_whiff, whiff_over_expected
- ``meta``: key/value rows, at least model_version, last_fetch_date,
  train_auc, test_auc

The pipeline may add columns. This module reads the minimum set by name and
picks up one optional extension, ``leaderboard.player_name``, when it exists.
Nothing here writes: the connection is opened with ``mode=ro``.
"""
from __future__ import annotations

import os
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[3]

LEADERBOARD_COLUMNS = (
    "pitcher",
    "season",
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
)


class DatabaseUnavailable(RuntimeError):
    """The database file is missing or does not match the storage contract."""


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
    """Everything the pages need from ``leaderboard`` and ``meta``, loaded once
    per version of the database file."""

    meta: dict[str, str | None]
    rows: tuple[dict, ...]
    seasons: tuple[int | str, ...]
    has_pitch_rows: bool

    def season_rows(self, season) -> list[dict]:
        return [r for r in self.rows if str(r["season"]) == str(season)]

    def find(self, pitcher_id: str) -> list[dict]:
        """Every season line for one pitcher, newest season first."""
        hits = [r for r in self.rows if r["id"] == pitcher_id]
        return sorted(hits, key=lambda r: str(r["season"]), reverse=True)


class Database:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else default_db_path()
        self._stamp: tuple[int, int] | None = None
        self._snapshot: Snapshot | None = None

    def _connect(self) -> sqlite3.Connection:
        if not self.path.is_file():
            raise DatabaseUnavailable(f"{self.path.name} has not been built yet")
        uri = f"file:{quote(self.path.resolve().as_posix())}?mode=ro"
        try:
            con = sqlite3.connect(uri, uri=True, check_same_thread=False)
        except sqlite3.Error as exc:  # pragma: no cover - depends on the OS
            raise DatabaseUnavailable(str(exc)) from exc
        con.row_factory = sqlite3.Row
        return con

    @staticmethod
    def _columns(con: sqlite3.Connection, table: str) -> set[str]:
        return {row["name"] for row in con.execute(f"PRAGMA table_info({table})")}

    def snapshot(self) -> Snapshot:
        """Load ``leaderboard`` and ``meta``, reusing the copy in memory until
        the file on disk changes."""
        if not self.path.is_file():
            raise DatabaseUnavailable(f"{self.path.name} has not been built yet")
        stat = self.path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        if self._snapshot is not None and stamp == self._stamp:
            return self._snapshot

        con = self._connect()
        try:
            have = self._columns(con, "leaderboard")
            missing = [c for c in LEADERBOARD_COLUMNS if c not in have]
            if missing:
                raise DatabaseUnavailable(
                    "leaderboard table is missing contract columns: " + ", ".join(missing)
                )
            name_col = "player_name" if "player_name" in have else "NULL"
            cur = con.execute(
                f"SELECT {', '.join(LEADERBOARD_COLUMNS)}, {name_col} AS player_name "
                "FROM leaderboard"
            )
            rows = [self._shape(dict(r)) for r in cur]

            if not self._columns(con, "meta") >= {"key", "value"}:
                raise DatabaseUnavailable("meta table is missing key/value columns")
            meta = {r["key"]: r["value"] for r in con.execute("SELECT key, value FROM meta")}

            pitch_cols = self._columns(con, "pitches")
            has_pitch_rows = bool(
                pitch_cols >= set(PITCH_COLUMNS)
                and con.execute("SELECT 1 FROM pitches LIMIT 1").fetchone()
            )
        except sqlite3.Error as exc:
            raise DatabaseUnavailable(str(exc)) from exc
        finally:
            con.close()

        self._rank(rows)
        seasons = tuple(sorted({r["season"] for r in rows}, key=str, reverse=True))
        self._snapshot = Snapshot(meta, tuple(rows), seasons, has_pitch_rows)
        self._stamp = stamp
        return self._snapshot

    @staticmethod
    def _shape(row: dict) -> dict:
        raw_name = row.pop("player_name")
        row["id"] = str(row["pitcher"])
        row["name"] = display_name(raw_name) if raw_name else row["id"]
        row["search"] = fold(f"{row['name']} {raw_name or ''} {row['id']}")
        return row

    @staticmethod
    def _rank(rows: list[dict]) -> None:
        """Rank within each season: by stuff grade, and by whiffs over expected."""
        by_season: dict[str, list[dict]] = {}
        for r in rows:
            by_season.setdefault(str(r["season"]), []).append(r)
        for group in by_season.values():
            for key, field in (("rank", "avg_stuff"), ("woe_rank", "whiff_over_expected")):
                ordered = sorted(
                    group,
                    key=lambda r: (r[field] is None, -(r[field] or 0.0), r["name"]),
                )
                for i, r in enumerate(ordered, start=1):
                    r[key] = i
            for r in group:
                r["season_size"] = len(group)

    def pitch_breakdown(self, pitcher, season) -> dict[str, list[dict]]:
        """Per-pitch-type and per-month aggregates for one pitcher's season.

        Returns empty lists when the ``pitches`` table has nothing for this
        pitcher, which is the case until the pipeline writes pitch rows.
        """
        con = self._connect()
        try:
            if not self._columns(con, "pitches") >= set(PITCH_COLUMNS):
                return {"by_type": [], "by_month": []}
            where = "pitcher = ? AND substr(game_date, 1, 4) = ?"
            args = (pitcher, str(season))
            agg = (
                "COUNT(*) AS pitches, AVG(stuff_score) AS avg_stuff, "
                "AVG(expected_whiff) AS expected_whiff, AVG(actual_whiff) AS actual_whiff"
            )
            by_type = [
                dict(r)
                for r in con.execute(
                    f"SELECT pitch_type, {agg} FROM pitches WHERE {where} "
                    "GROUP BY pitch_type ORDER BY pitches DESC, pitch_type",
                    args,
                )
            ]
            by_month = [
                dict(r)
                for r in con.execute(
                    f"SELECT substr(game_date, 1, 7) AS month, {agg} FROM pitches "
                    f"WHERE {where} GROUP BY month ORDER BY month",
                    args,
                )
            ]
        except sqlite3.Error as exc:
            raise DatabaseUnavailable(str(exc)) from exc
        finally:
            con.close()

        total = sum(r["pitches"] for r in by_type) or 1
        for r in by_type:
            r["share"] = r["pitches"] / total
        for r in by_type + by_month:
            if r["actual_whiff"] is not None and r["expected_whiff"] is not None:
                r["whiff_over_expected"] = r["actual_whiff"] - r["expected_whiff"]
            else:
                r["whiff_over_expected"] = None
        return {"by_type": by_type, "by_month": by_month}
