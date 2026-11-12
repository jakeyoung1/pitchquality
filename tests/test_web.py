"""Tests for the live API and dashboard (``pitchquality.web``).

Two databases are used. ``REAL_DB`` is the pipeline's committed
``data/pitchquality.db``: those tests check every page and endpoint against
plain SQL on the same file. ``tiny_db`` is a few hand-written rows in a
temporary file created from ``src/pitchquality/schema.sql``, for exact
arithmetic and edge cases. The hand-written rows never leave this file.
"""
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pitchquality.web import facts, reports
from pitchquality.web.app import (
    DEFAULT_MIN,
    create_app,
    fmt_auc,
    fmt_date,
    fmt_rate,
    fmt_signed,
    fmt_stuff,
)
from pitchquality.web.db import (
    LEADERBOARD_COLUMNS,
    PITCH_COLUMNS,
    Database,
    DatabaseUnavailable,
    display_name,
    fold,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REAL_DB = ROOT / "data" / "pitchquality.db"
SCHEMA = (ROOT / "src" / "pitchquality" / "schema.sql").read_text(encoding="utf-8")
FLOOR = facts.QUALIFIED_MIN_SWINGS


def sql(query: str, *args):
    con = sqlite3.connect(f"file:{REAL_DB.as_posix()}?mode=ro", uri=True)
    try:
        return con.execute(query, args).fetchall()
    finally:
        con.close()


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(REAL_DB, REPORTS))


@pytest.fixture(scope="module")
def meta() -> dict:
    return dict(sql("SELECT key, value FROM meta"))


def pitch(n, date, pitcher, name, kind, stuff_whiff, expected, whiff, game_type="R"):
    """One hand-written pitches row. The stuff score uses made-up reference
    values of mean 0.25 and standard deviation 0.20."""
    score = 100 + 10 * (stuff_whiff - 0.25) / 0.20
    return (1, n, 1, date, pitcher, name, kind, score, expected, whiff, stuff_whiff, int(date[:4]), game_type)


TINY_PITCHES = [
    pitch(1, "2026-04-02", 11, "Able, Al", "FF", 0.45, 0.40, 1),
    pitch(2, "2026-04-02", 11, "Able, Al", "FF", 0.25, 0.20, 0),
    pitch(3, "2026-04-09", 11, "Able, Al", "FF", 0.35, 0.30, 1),
    pitch(4, "2026-05-01", 11, "Able, Al", "SL", 0.15, 0.30, 0),
    pitch(5, "2026-10-02", 11, "Able, Al", "XX", 0.25, 0.30, 0, "F"),
    pitch(6, "2026-05-01", 22, "Baker, Bo", "CH", 0.20, 0.20, 0),
    pitch(7, "2025-06-01", 11, "Able, Al", "CU", 0.25, 0.25, 0),
]


def write_tiny(path: Path) -> None:
    """Build pitches, then derive leaderboard rows the way the pipeline does."""
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO pitches VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", TINY_PITCHES)
    con.execute(
        "INSERT INTO leaderboard SELECT pitcher, season, player_name, COUNT(*), AVG(stuff_score), "
        "AVG(expected_whiff), AVG(actual_whiff), AVG(actual_whiff) - AVG(expected_whiff) "
        "FROM pitches GROUP BY pitcher, season, player_name"
    )
    con.executemany(
        "INSERT INTO meta VALUES (?, ?)",
        [
            ("model_version", "test-model"),
            ("schema_version", "1"),
            ("last_fetch_date", "2026-10-02"),
            ("last_pitch_date", "2026-10-02"),
            ("train_auc", "0.81"),
            ("test_auc", "0.79"),
            ("stuff_train_auc", "0.78"),
            ("stuff_test_auc", "0.77"),
            ("rows_scored", "7"),
        ],
    )
    con.commit()
    con.close()


@pytest.fixture()
def tiny_db(tmp_path) -> Path:
    path = tmp_path / "pitchquality.db"
    write_tiny(path)
    return path


class TestSchema:
    def test_the_columns_the_site_reads_are_in_schema_sql(self):
        con = sqlite3.connect(":memory:")
        con.executescript(SCHEMA)
        cols = lambda t: {r[1] for r in con.execute(f"PRAGMA table_info({t})")}
        assert set(LEADERBOARD_COLUMNS) <= cols("leaderboard")
        assert set(PITCH_COLUMNS) <= cols("pitches")
        assert cols("meta") == {"key", "value"}

    def test_the_committed_database_has_those_columns(self):
        cols = lambda t: {r[1] for r in sql(f"PRAGMA table_info({t})")}
        assert set(LEADERBOARD_COLUMNS) <= cols("leaderboard")
        assert set(PITCH_COLUMNS) <= cols("pitches")

    def test_the_seed_is_gone(self):
        assert not (ROOT / "src" / "pitchquality" / "web" / "seed.py").exists()
        assert "seed" not in (ROOT / "render.yaml").read_text(encoding="utf-8")


class TestFactsMatchTheRepo:
    """Numbers quoted in prose must still appear in the file they came from."""

    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_vaa_fix_number_is_in_the_readme(self):
        assert facts.STUFF_AUC_BEFORE_VAA == "0.652"
        assert ".652 → .767 AUC" in self.readme

    def test_data_table_is_in_the_readme(self):
        for row in facts.DATA_ROWS:
            for key in ("pitches", "swings", "whiff_rate"):
                assert row[key] in self.readme

    def test_qualifying_floor_is_the_research_threshold(self):
        cli = (ROOT / "src" / "pitchquality" / "cli.py").read_text(encoding="utf-8")
        assert f'pitcher_grades(test, "p_stuff", min_swings={FLOOR})' in cli
        assert DEFAULT_MIN == FLOOR == 100

    def test_sample_size_guidance_is_in_the_readme(self):
        assert f"below roughly {facts.MODEL_LEADS_BELOW_SWINGS} swings" in self.readme
        assert f"with {facts.OUTCOME_LEADS_FROM_SWINGS} swings in hand" in self.readme

    def test_pitch_type_names_cover_the_modeled_types(self):
        source = (ROOT / "src" / "pitchquality" / "features.py").read_text(encoding="utf-8")
        block = source.split("MODELED_PITCH_TYPES = {")[1].split("}")[0]
        modeled = {code.strip().strip('"') for code in block.split(",") if code.strip()}
        assert modeled == set(facts.PITCH_TYPE_NAMES)
        assert {r[0] for r in sql("SELECT DISTINCT pitch_type FROM pitches")} <= modeled

    def test_game_type_names_cover_the_database(self):
        assert {r[0] for r in sql("SELECT DISTINCT game_type FROM pitches")} <= set(facts.GAME_TYPE_NAMES)


class TestHealth:
    def test_reports_the_real_database(self, client, meta):
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["leaderboard_rows"] == sql("SELECT COUNT(*) FROM leaderboard")[0][0]
        assert body["last_fetch_date"] == meta["last_fetch_date"]
        assert body["last_pitch_date"] == sql("SELECT MAX(game_date) FROM pitches")[0][0]
        assert body["model_version"] == meta["model_version"]
        assert body["schema_version"] == "1"

    def test_503_when_the_database_is_missing(self, tmp_path):
        c = TestClient(create_app(tmp_path / "nope.db", REPORTS))
        assert c.get("/api/health").status_code == 503
        assert c.get("/api/leaderboard").status_code == 503
        page = c.get("/")
        assert page.status_code == 503
        assert "Database unavailable" in page.text

    def test_503_when_a_schema_column_is_missing(self, tmp_path):
        path = tmp_path / "old.db"
        con = sqlite3.connect(path)
        con.executescript(SCHEMA.replace("    player_name TEXT NOT NULL,\n    pitches", "    pitches"))
        con.close()
        response = TestClient(create_app(path, REPORTS)).get("/api/health")
        assert response.status_code == 503
        assert "player_name" in response.json()["detail"]


class TestLeaderboardApi:
    def test_default_board_is_qualified_pitchers_in_stuff_order(self, client):
        body = client.get("/api/leaderboard", params={"limit": 1000}).json()
        expected = sql(
            "SELECT pitcher, pitches, avg_stuff, expected_whiff, actual_whiff, whiff_over_expected "
            "FROM leaderboard WHERE pitches >= ? ORDER BY avg_stuff DESC",
            FLOOR,
        )
        assert body["total"] == len(expected) > 0
        assert body["min_pitches"] == FLOOR
        got = [
            (r["pitcher"], r["pitches"], r["avg_stuff"], r["expected_whiff"], r["actual_whiff"],
             r["whiff_over_expected"])
            for r in body["rows"]
        ]
        assert got == expected
        assert [r["rank"] for r in body["rows"]] == list(range(1, len(expected) + 1))
        assert all(r["qualified"] for r in body["rows"])

    def test_tiny_samples_do_not_lead_the_default_board(self, client):
        raw_top = sql("SELECT pitcher, pitches FROM leaderboard ORDER BY avg_stuff DESC LIMIT 1")[0]
        assert raw_top[1] < FLOOR  # storage applies no floor, so the raw top is a tiny sample
        top = client.get("/api/leaderboard", params={"limit": 1}).json()["rows"][0]
        assert top["pitcher"] != raw_top[0]
        assert top["pitches"] >= FLOOR

    def test_min_pitches_zero_returns_everyone_and_leaves_them_unranked(self, client):
        body = client.get("/api/leaderboard", params={"min_pitches": 0, "limit": 1000}).json()
        assert body["total"] == sql("SELECT COUNT(*) FROM leaderboard")[0][0]
        unranked = [r for r in body["rows"] if not r["qualified"]]
        assert unranked and all(r["rank"] is None and r["pitches"] < FLOOR for r in unranked)

    def test_names_come_from_player_name(self, client):
        raw = dict(sql("SELECT pitcher, player_name FROM leaderboard"))
        body = client.get("/api/leaderboard", params={"min_pitches": 0, "limit": 1000}).json()
        assert all(r["name"] == display_name(raw[r["pitcher"]]) for r in body["rows"])

    def test_search_ignores_accents_and_case(self, client):
        body = client.get("/api/leaderboard", params={"q": "MUNOZ, andres"}).json()
        assert [r["name"] for r in body["rows"]] == ["Andrés Muñoz"]

    def test_search_by_mlbam_id(self, client):
        body = client.get("/api/leaderboard", params={"q": "695243"}).json()
        assert [r["pitcher"] for r in body["rows"]] == [695243]

    def test_sort_and_order(self, client):
        top = client.get("/api/leaderboard", params={"sort": "whiff_over_expected", "limit": 1}).json()
        best = sql(
            "SELECT pitcher FROM leaderboard WHERE pitches >= ? ORDER BY whiff_over_expected DESC LIMIT 1",
            FLOOR,
        )[0][0]
        assert top["rows"][0]["pitcher"] == best
        most = client.get("/api/leaderboard", params={"sort": "pitches", "limit": 1}).json()
        assert most["rows"][0]["pitches"] == sql("SELECT MAX(pitches) FROM leaderboard")[0][0]
        low = client.get(
            "/api/leaderboard", params={"sort": "pitches", "order": "asc", "min_pitches": 0, "limit": 1}
        ).json()
        assert low["rows"][0]["pitches"] == sql("SELECT MIN(pitches) FROM leaderboard")[0][0]

    def test_paging_covers_the_board_once(self, client):
        seen = []
        for offset in range(0, 1000, 200):
            rows = client.get("/api/leaderboard", params={"limit": 200, "offset": offset}).json()["rows"]
            seen += [r["pitcher"] for r in rows]
        assert len(seen) == len(set(seen)) == sql(
            "SELECT COUNT(*) FROM leaderboard WHERE pitches >= ?", FLOOR
        )[0][0]

    def test_bad_parameters_are_rejected(self, client):
        assert client.get("/api/leaderboard", params={"sort": "drop table"}).status_code == 422
        assert client.get("/api/leaderboard", params={"limit": 0}).status_code == 422
        assert client.get("/api/leaderboard", params={"min_pitches": -1}).status_code == 422
        assert client.get("/api/leaderboard", params={"season": "1999"}).status_code == 404


class TestPitcherApi:
    def test_every_pitcher_detail_agrees_with_the_pitches_table(self, client):
        """All 825 pitchers: the splits add up to the leaderboard line."""
        board = sql("SELECT pitcher, pitches, expected_whiff, actual_whiff, avg_stuff FROM leaderboard")
        for pitcher, pitches, expected, actual, stuff in board:
            line = client.get(f"/api/pitchers/{pitcher}").json()["seasons"][0]
            assert line["pitches"] == pitches
            for split in ("pitch_types", "months", "game_types"):
                assert sum(r["pitches"] for r in line[split]) == pitches, (pitcher, split)
            weighted = lambda key: sum(r[key] * r["pitches"] for r in line["pitch_types"]) / pitches
            assert weighted("expected_whiff") == pytest.approx(expected, abs=1e-9)
            assert weighted("actual_whiff") == pytest.approx(actual, abs=1e-9)
            assert weighted("avg_stuff") == pytest.approx(stuff, abs=1e-6)
            assert weighted("stuff_whiff") == pytest.approx(line["stuff_whiff"], abs=1e-9)

    def test_one_pitcher_against_plain_sql(self, client):
        line = client.get("/api/pitchers/695243").json()["seasons"][0]
        types = sql(
            "SELECT pitch_type, COUNT(*), AVG(stuff_whiff), AVG(expected_whiff), AVG(actual_whiff) "
            "FROM pitches WHERE pitcher = 695243 AND season = 2026 GROUP BY 1 ORDER BY 2 DESC, 1"
        )
        got = [
            (t["pitch_type"], t["pitches"], t["stuff_whiff"], t["expected_whiff"], t["actual_whiff"])
            for t in line["pitch_types"]
        ]
        assert got == [tuple(t) for t in types]
        months = sql(
            "SELECT substr(game_date, 1, 7), COUNT(*) FROM pitches "
            "WHERE pitcher = 695243 AND season = 2026 GROUP BY 1 ORDER BY 1"
        )
        assert [(m["month"], m["pitches"]) for m in line["months"]] == [tuple(m) for m in months]
        stages = dict(sql(
            "SELECT game_type, COUNT(*) FROM pitches WHERE pitcher = 695243 AND season = 2026 GROUP BY 1"
        ))
        assert {g["game_type"]: g["pitches"] for g in line["game_types"]} == stages
        assert line["stuff_whiff"] == pytest.approx(
            sql("SELECT AVG(stuff_whiff) FROM pitches WHERE pitcher = 695243 AND season = 2026")[0][0]
        )
        assert line["rank"] == 1 and line["qualified"] is True
        assert line["of"] == sql("SELECT COUNT(*) FROM leaderboard WHERE pitches >= ?", FLOOR)[0][0]

    def test_an_unqualified_pitcher_has_a_line_but_no_rank(self, client):
        pitcher, pitches = sql("SELECT pitcher, pitches FROM leaderboard ORDER BY pitches LIMIT 1")[0]
        line = client.get(f"/api/pitchers/{pitcher}").json()["seasons"][0]
        assert line["pitches"] == pitches < FLOOR
        assert line["qualified"] is False
        assert line["rank"] is None and line["whiff_over_expected_rank"] is None

    def test_unknown_pitcher_is_404(self, client):
        response = client.get("/api/pitchers/1")
        assert response.status_code == 404
        assert response.headers["content-type"].startswith("application/json")


class TestTinyDatabase:
    """Exact arithmetic on a few hand-written rows."""

    def test_breakdown_math(self, tiny_db):
        c = TestClient(create_app(tiny_db, REPORTS))
        body = c.get("/api/pitchers/11").json()
        assert body["name"] == "Al Able"
        assert [s["season"] for s in body["seasons"]] == [2026, 2025]
        line = body["seasons"][0]
        assert line["pitches"] == 5
        assert line["stuff_whiff"] == pytest.approx((0.45 + 0.25 + 0.35 + 0.15 + 0.25) / 5)
        assert line["expected_whiff"] == pytest.approx(0.30)
        assert line["actual_whiff"] == pytest.approx(0.40)
        assert line["whiff_over_expected"] == pytest.approx(0.10)
        ff = line["pitch_types"][0]
        assert (ff["pitch_type"], ff["pitches"]) == ("FF", 3)
        assert ff["share"] == pytest.approx(0.6)
        assert ff["stuff_whiff"] == pytest.approx(0.35)
        assert ff["avg_stuff"] == pytest.approx(105.0)
        assert ff["expected_whiff"] == pytest.approx(0.30)
        assert ff["actual_whiff"] == pytest.approx(2 / 3)
        assert ff["whiff_over_expected"] == pytest.approx(2 / 3 - 0.30)
        assert [m["month"] for m in line["months"]] == ["2026-04", "2026-05", "2026-10"]
        assert [m["pitches"] for m in line["months"]] == [3, 1, 1]
        assert [(g["game_type"], g["pitches"]) for g in line["game_types"]] == [("R", 4), ("F", 1)]
        # The 2025 line only sees 2025 pitches.
        assert [p["pitch_type"] for p in body["seasons"][1]["pitch_types"]] == ["CU"]

    def test_nobody_qualifies_so_nobody_is_ranked(self, tiny_db):
        c = TestClient(create_app(tiny_db, REPORTS))
        assert c.get("/api/leaderboard").json()["total"] == 0
        everyone = c.get("/api/leaderboard", params={"min_pitches": 0}).json()
        assert [r["pitcher"] for r in everyone["rows"]] == [11, 22]
        assert all(r["rank"] is None for r in everyone["rows"])
        assert everyone["seasons"] == [2026, 2025]

    def test_pages_render_with_no_qualified_pitchers(self, tiny_db):
        c = TestClient(create_app(tiny_db, REPORTS))
        board = c.get("/")
        assert board.status_code == 200 and "No pitchers match" in board.text
        everyone = c.get("/", params={"min": 0}).text
        assert "Al Able" in everyone and "Bo Baker" in everyone
        pitcher = c.get("/pitchers/11").text
        assert "Not ranked, under 100 swings" in pitcher
        assert "Four-seam fastball" in pitcher
        assert "XX" in pitcher  # an unknown pitch code is shown as is
        assert "Wild Card Series" in pitcher
        assert "Other seasons" in pitcher
        assert "Oct 2026" in pitcher
        assert c.get("/model").status_code == 200

    def test_season_switch(self, tiny_db):
        c = TestClient(create_app(tiny_db, REPORTS))
        body = c.get("/api/leaderboard", params={"season": "2025", "min_pitches": 0}).json()
        assert [r["pitcher"] for r in body["rows"]] == [11]

    def test_a_rewritten_database_is_picked_up(self, tiny_db):
        c = TestClient(create_app(tiny_db, REPORTS))
        assert c.get("/api/health").json()["leaderboard_rows"] == 3
        con = sqlite3.connect(tiny_db)
        con.execute("INSERT INTO leaderboard VALUES (33, 2026, 'Cole, Cy', 1, 100.0, 0.1, 0.0, -0.1)")
        con.commit()
        con.close()
        assert c.get("/api/health").json()["leaderboard_rows"] == 4


class TestModelCard:
    def test_key_numbers_are_the_repo_numbers(self, client):
        body = client.get("/api/model-card").json()
        assert body["key_numbers"] == {
            "full_auc": "0.788",
            "stuff_auc": "0.767",
            "baseline_auc": "0.613",
            "stuff_auc_before_vaa": "0.652",
        }
        assert [p["auc"] for p in body["performance"]] == [0.6133, 0.7674, 0.7875]
        assert len(body["calibration"]) == 10
        assert len(body["validity_by_threshold"]) == 7
        assert body["predictive_validity"][-1]["predictor"] == "50/50 blend"

    def test_saved_model_numbers_come_from_meta(self, client, meta):
        live = client.get("/api/model-card").json()["live_model"]
        assert live["model_version"] == meta["model_version"]
        assert live["full"]["train_auc"] == float(meta["train_auc"])
        assert live["full"]["test_auc"] == float(meta["test_auc"])
        assert live["stuff"]["train_auc"] == float(meta["stuff_train_auc"])
        assert live["stuff"]["test_auc"] == float(meta["stuff_test_auc"])
        assert live["stuff_reference_mean"] == float(meta["stuff_reference_mean"])
        # The saved fit agrees with the repo's headline numbers to within 0.001.
        assert abs(live["full"]["test_auc"] - 0.7875) < 0.001
        assert abs(live["stuff"]["test_auc"] - 0.7674) < 0.001

    def test_coverage_matches_the_pitches_table(self, client, meta):
        body = client.get("/api/model-card").json()
        first, last, count = sql("SELECT MIN(game_date), MAX(game_date), COUNT(*) FROM pitches")[0]
        assert body["coverage"]["season_start"] == first
        assert body["coverage"]["last_pitch_date"] == last
        assert body["coverage"]["live_scoring"] is False
        assert body["database"]["rows_scored"] == count == int(meta["rows_scored"])
        assert body["database"]["pitchers"] == sql("SELECT COUNT(*) FROM leaderboard")[0][0]
        backfill = sql("SELECT COUNT(*) FROM pitches WHERE game_date >= ?", meta["backfill_start"])[0][0]
        assert body["database"]["backfill_rows_scored"] == backfill
        held_out = sql("SELECT COUNT(*) FROM pitches WHERE game_date < ?", meta["backfill_start"])[0][0]
        assert body["live_model"]["held_out_swings"] == held_out

    def test_page_says_the_season_is_over_and_what_is_covered(self, client, meta):
        text = client.get("/model").text
        assert "The 2026 MLB season is over" in text
        assert "Nothing on this site scores live games" in text
        assert fmt_date(meta["last_pitch_date"]) in text
        assert f"{int(meta['rows_scored']):,} scored swings" in text
        assert "0.788" in text and "0.767" in text and "0.652" in text
        assert fmt_auc(meta["stuff_test_auc"]) in text and fmt_auc(meta["train_auc"]) in text
        assert meta["model_version"] in text
        assert "committed in this repo's reports" not in text  # the old seed banner
        for key in meta:
            assert f">{key}</th>" in text

    def test_every_figure_is_served(self, client):
        body = client.get("/api/model-card").json()
        assert len(body["figures"]) == len(reports.FIGURES) == 5
        for fig in body["figures"]:
            response = client.get(fig["url"])
            assert response.status_code == 200
            assert response.headers["content-type"] == "image/png"

    def test_only_listed_figures_are_served_from_the_repo(self, client):
        for path in (
            "/reports/pitcher_stuff_grades.csv",
            "/reports/live_pipeline.json",
            "/reports/..%2Fpyproject.toml",
            "/reports/%2e%2e/README.md",
            "/src/pitchquality/web/app.py",
            "/data/pitchquality.db",
            "/models/whiff-2025.joblib",
        ):
            assert client.get(path).status_code == 404, path

    def test_round_half_up(self):
        assert reports.round_half_up("0.7875") == "0.788"
        assert reports.round_half_up("0.7674") == "0.767"
        assert reports.round_half_up("0.6133") == "0.613"


class TestPages:
    def test_leaderboard_page(self, client, meta):
        text = client.get("/").text
        qualified = sql("SELECT COUNT(*) FROM leaderboard WHERE pitches >= ?", FLOOR)[0][0]
        top = sql(
            "SELECT pitcher, player_name FROM leaderboard WHERE pitches >= ? ORDER BY avg_stuff DESC LIMIT 1",
            FLOOR,
        )[0]
        assert "Stuff leaderboard" in text
        # Jake's copy (relay #156), word for word, between the title and the board.
        explainer = (
            "Stuff score = expected swing-and-miss rate from the pitch itself (velocity, spin, "
            "movement, approach angle), scaled so 100 = league average. No location, no count."
        )
        assert text.index("<h1>") < text.index(explainer) < text.index('<table class="board">')
        assert display_name(top[1]) in text
        assert f'href="/pitchers/{top[0]}"' in text
        assert f"Page 1 of {-(-qualified // 50)}" in text
        assert f"{int(meta['rows_scored']):,}" in text
        assert fmt_date(meta["last_pitch_date"]) in text
        assert "Stuff+ leaderboard" not in text
        assert "committed in this repo's reports" not in text

    def test_all_pitchers_view_shows_tiny_samples_unranked(self, client):
        name = display_name(sql("SELECT player_name FROM leaderboard ORDER BY avg_stuff DESC LIMIT 1")[0][0])
        assert name not in client.get("/").text
        everyone = client.get("/", params={"min": 0}).text
        assert name in everyone
        assert "Under 100 swings, not ranked" in everyone

    def test_leaderboard_search_sort_and_paging(self, client):
        assert "Andrés Muñoz" in client.get("/", params={"q": "munoz"}).text
        assert "No pitchers match" in client.get("/", params={"q": "zzzzzz"}).text
        last = client.get("/", params={"page": 999})
        assert last.status_code == 200 and "Next</span>" in last.text
        assert 'aria-sort="ascending"' in client.get("/", params={"sort": "name"}).text
        # An unknown sort key falls back to the default instead of erroring.
        assert client.get("/", params={"sort": "nonsense"}).status_code == 200

    def test_search_text_is_escaped(self, client):
        text = client.get("/", params={"q": "<script>alert(1)</script>"}).text
        assert "<script>alert(1)</script>" not in text
        assert "&lt;script&gt;" in text

    def test_pitcher_page(self, client):
        text = client.get("/pitchers/695243").text
        line = sql("SELECT pitches, avg_stuff FROM leaderboard WHERE pitcher = 695243")[0]
        qualified = sql("SELECT COUNT(*) FROM leaderboard WHERE pitches >= ?", FLOOR)[0][0]
        assert "Mason Miller" in text
        assert f"Rank 1 of {qualified}" in text
        assert fmt_stuff(line[1]) in text
        assert "Four-seam fastball" in text and "By month" in text
        assert "Regular season and postseason" in text
        assert "baseballsavant.mlb.com/savant-player/695243" in text
        assert "no pitch-level rows" not in text

    def test_a_sample_of_pitcher_pages_render(self, client):
        ids = [r[0] for r in sql("SELECT pitcher FROM leaderboard ORDER BY pitcher")][::25]
        for pitcher in ids:
            assert client.get(f"/pitchers/{pitcher}").status_code == 200, pitcher

    def test_unknown_pitcher_page_is_html_404(self, client):
        response = client.get("/pitchers/1")
        assert response.status_code == 404
        assert "Not found" in response.text

    def test_static_assets(self, client):
        assert client.get("/static/style.css").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/favicon.svg").status_code == 200


class TestReadOnly:
    def test_connection_refuses_writes(self, tiny_db):
        con = Database(tiny_db)._connect()
        with pytest.raises(sqlite3.OperationalError):
            con.execute("DELETE FROM leaderboard")
        con.close()

    def test_requests_leave_the_file_untouched(self, tiny_db):
        c = TestClient(create_app(tiny_db, REPORTS))
        before = tiny_db.read_bytes()
        for path in ("/?min=0", "/pitchers/11", "/model", "/api/leaderboard", "/api/model-card"):
            assert c.get(path).status_code == 200
        assert tiny_db.read_bytes() == before

    def test_missing_file_raises_instead_of_creating_one(self, tmp_path):
        path = tmp_path / "absent.db"
        with pytest.raises(DatabaseUnavailable):
            Database(path).snapshot()
        assert not path.exists()


class TestFormatting:
    def test_display_name(self):
        assert display_name("Miller, Mason") == "Mason Miller"
        assert display_name("Natera Jr., Samy") == "Samy Natera Jr."
        assert display_name("Ohtani") == "Ohtani"

    def test_fold(self):
        assert fold("Muñoz, Andrés") == "munoz, andres"

    def test_rates(self):
        assert fmt_rate(0.4433) == ".443"
        assert fmt_rate(None) == "n/a"
        assert fmt_signed(0.0601) == "+.060"
        assert fmt_signed(-0.0231) == "−.023"
        assert fmt_signed(-0.0002) == ".000"

    def test_stuff_and_auc(self):
        assert fmt_stuff(107.45357) == "107.5"
        assert fmt_stuff(None) == "n/a"
        assert fmt_auc("0.7668424416913047") == "0.7668"
        assert fmt_auc(None) == "not recorded"

    def test_dates(self):
        assert fmt_date("2026-10-28") == "Oct 28, 2026"
        assert fmt_date(None) == "unknown"
        assert fmt_date("late October") == "late October"


def test_the_service_does_not_import_the_modeling_stack():
    """Render installs requirements-web.txt only: no pandas, no scikit-learn."""
    code = (
        "import sys; import pitchquality.web.app; "
        "heavy = {'pandas', 'numpy', 'sklearn', 'matplotlib', 'pyarrow', 'pybaseball', 'joblib'}; "
        "print(sorted(heavy & set(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=True, cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert out.stdout.strip() == "[]"
