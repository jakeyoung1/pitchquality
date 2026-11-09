"""Tests for the live API and dashboard (``pitchquality.web``).

Two databases are used. ``seeded`` is built by the seed from the real reports
committed in this repo, so those tests check the site against the published
numbers. ``contract_db`` is a small synthetic file holding only the minimum
columns from the storage contract, with pitch rows, to prove the API runs on
what the pipeline is required to write. Synthetic rows never leave this file.
"""
import csv
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pitchquality.web import facts, reports, seed
from pitchquality.web.app import create_app, fmt_date, fmt_rate, fmt_signed, fmt_stuff
from pitchquality.web.db import Database, DatabaseUnavailable, display_name, fold

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def grades_csv() -> list[dict]:
    with (REPORTS / "pitcher_stuff_grades.csv").open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def seeded(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("seeded") / "pitchquality.db"
    seed.build(path, REPORTS)
    return path


@pytest.fixture(scope="module")
def client(seeded) -> TestClient:
    return TestClient(create_app(seeded, REPORTS))


@pytest.fixture()
def contract_db(tmp_path) -> Path:
    """Minimum contract columns only: no player_name, no extension meta keys."""
    path = tmp_path / "pitchquality.db"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE pitches (game_date TEXT, pitcher INTEGER, pitch_type TEXT,
                              stuff_score REAL, expected_whiff REAL, actual_whiff INTEGER);
        CREATE TABLE leaderboard (pitcher INTEGER, season INTEGER, pitches INTEGER,
                                  avg_stuff REAL, expected_whiff REAL, actual_whiff REAL,
                                  whiff_over_expected REAL);
        CREATE TABLE meta (key TEXT, value TEXT);
        """
    )
    con.executemany(
        "INSERT INTO leaderboard VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (11, 2026, 4, 0.30, 0.30, 0.50, 0.20),
            (22, 2026, 2, 0.20, 0.20, 0.00, -0.20),
            (11, 2025, 9, 0.25, 0.25, 0.25, 0.00),
        ],
    )
    con.executemany(
        "INSERT INTO pitches VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("2026-04-02", 11, "FF", 0.40, 0.40, 1),
            ("2026-04-02", 11, "FF", 0.20, 0.20, 0),
            ("2026-04-09", 11, "FF", 0.30, 0.30, 1),
            ("2026-05-01", 11, "SL", 0.30, 0.30, 0),
            ("2026-05-01", 22, "CH", 0.20, 0.20, 0),
            ("2026-05-03", 22, "XX", 0.20, 0.20, 0),
            ("2025-06-01", 11, "CU", 0.25, 0.25, 0),
        ],
    )
    con.executemany(
        "INSERT INTO meta VALUES (?, ?)",
        [
            ("model_version", "9.9.9"),
            ("last_fetch_date", "2026-09-27"),
            ("train_auc", "0.81"),
            ("test_auc", "0.7875"),
        ],
    )
    con.commit()
    con.close()
    return path


class TestSeed:
    def test_leaderboard_matches_the_committed_report_row_for_row(self, seeded):
        con = sqlite3.connect(seeded)
        rows = con.execute(
            "SELECT pitcher, player_name, pitches, actual_whiff, expected_whiff, "
            "avg_stuff, whiff_over_expected FROM leaderboard ORDER BY rowid"
        ).fetchall()
        con.close()
        expected = [
            (
                int(g["pitcher"]),
                g["player_name"],
                int(g["swings"]),
                float(g["actual_whiff_rate"]),
                float(g["expected_whiff_rate"]),
                float(g["stuff_plus"]),
                float(g["whiff_over_expected"]),
            )
            for g in grades_csv()
        ]
        assert rows == expected

    def test_schema_has_every_contract_column(self, seeded):
        con = sqlite3.connect(seeded)
        cols = lambda t: {r[1] for r in con.execute(f"PRAGMA table_info({t})")}
        assert cols("pitches") >= {
            "game_date", "pitcher", "pitch_type", "stuff_score", "expected_whiff", "actual_whiff",
        }
        assert cols("leaderboard") >= {
            "pitcher", "season", "pitches", "avg_stuff", "expected_whiff", "actual_whiff",
            "whiff_over_expected",
        }
        assert cols("meta") == {"key", "value"}
        con.close()

    def test_no_pitch_rows_are_made_up(self, seeded):
        con = sqlite3.connect(seeded)
        assert con.execute("SELECT COUNT(*) FROM pitches").fetchone()[0] == 0
        con.close()

    def test_meta_comes_from_the_reports_and_leaves_train_auc_empty(self, seeded):
        con = sqlite3.connect(seeded)
        meta = dict(con.execute("SELECT key, value FROM meta"))
        con.close()
        with (REPORTS / "model_performance.csv").open(newline="", encoding="utf-8") as fh:
            perf = {r["model"].split(" ")[0]: r["auc"] for r in csv.DictReader(fh)}
        assert meta["test_auc"] == perf["full"]
        assert meta["test_auc_stuff"] == perf["stuff"]
        assert meta["train_auc"] is None
        assert meta["last_fetch_date"] == facts.REPORT_LAST_GAME_DATE
        assert {"model_version", "last_fetch_date", "train_auc", "test_auc"} <= set(meta)

    def test_if_missing_never_overwrites_an_existing_database(self, tmp_path):
        path = tmp_path / "pitchquality.db"
        path.write_bytes(b"pipeline wrote this")
        assert seed.main(["--db", str(path), "--reports", str(REPORTS), "--if-missing"]) == 0
        assert path.read_bytes() == b"pipeline wrote this"

    def test_main_builds_when_the_file_is_missing(self, tmp_path):
        path = tmp_path / "data" / "pitchquality.db"
        assert seed.main(["--db", str(path), "--reports", str(REPORTS), "--if-missing"]) == 0
        assert Database(path).snapshot().rows


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

    def test_report_window_matches_the_fetch_script(self):
        fetch = (ROOT / "scripts_fetch.py").read_text(encoding="utf-8")
        assert f'"2026": ("{facts.REPORT_FIRST_GAME_DATE}", "{facts.REPORT_LAST_GAME_DATE}")' in fetch

    def test_sample_size_guidance_is_in_the_readme(self):
        assert f"below roughly {facts.MODEL_LEADS_BELOW_SWINGS} swings" in self.readme
        assert f"with {facts.OUTCOME_LEADS_FROM_SWINGS} swings in hand" in self.readme

    def test_pitch_type_names_cover_the_modeled_types(self):
        source = (ROOT / "src" / "pitchquality" / "features.py").read_text(encoding="utf-8")
        block = source.split("MODELED_PITCH_TYPES = {")[1].split("}")[0]
        modeled = {code.strip().strip('"') for code in block.split(",") if code.strip()}
        assert modeled == set(facts.PITCH_TYPE_NAMES)


class TestHealth:
    def test_ok_when_the_database_is_readable(self, client):
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["leaderboard_rows"] == len(grades_csv())

    def test_503_when_the_database_is_missing(self, tmp_path):
        c = TestClient(create_app(tmp_path / "nope.db", REPORTS))
        assert c.get("/api/health").status_code == 503
        assert c.get("/api/leaderboard").status_code == 503
        page = c.get("/")
        assert page.status_code == 503
        assert "Database unavailable" in page.text

    def test_503_when_a_contract_column_is_missing(self, tmp_path):
        path = tmp_path / "bad.db"
        con = sqlite3.connect(path)
        con.executescript(
            "CREATE TABLE leaderboard (pitcher INTEGER, season INTEGER);"
            "CREATE TABLE meta (key TEXT, value TEXT);"
            "CREATE TABLE pitches (pitcher INTEGER);"
        )
        con.close()
        response = TestClient(create_app(path, REPORTS)).get("/api/health")
        assert response.status_code == 503
        assert "avg_stuff" in response.json()["detail"]


class TestLeaderboardApi:
    def test_default_order_is_the_reports_stuff_ranking(self, client):
        body = client.get("/api/leaderboard", params={"limit": 15}).json()
        report = grades_csv()[:15]
        assert [r["pitcher"] for r in body["rows"]] == [int(g["pitcher"]) for g in report]
        assert body["rows"][0]["name"] == display_name(report[0]["player_name"])
        assert body["rows"][0]["avg_stuff"] == float(report[0]["stuff_plus"])
        assert body["total"] == len(grades_csv())
        assert body["season"] == 2026
        assert body["sample_unit"] == "swings"

    def test_search_ignores_accents_and_case(self, client):
        body = client.get("/api/leaderboard", params={"q": "MUNOZ, andres"}).json()
        assert [r["name"] for r in body["rows"]] == ["Andrés Muñoz"]

    def test_search_by_mlbam_id(self, client):
        body = client.get("/api/leaderboard", params={"q": "695243"}).json()
        assert [r["pitcher"] for r in body["rows"]] == [695243]

    def test_min_pitches_filters_on_the_sample(self, client):
        body = client.get("/api/leaderboard", params={"min_pitches": 1000, "limit": 1000}).json()
        expected = sum(int(g["swings"]) >= 1000 for g in grades_csv())
        assert body["total"] == expected > 0
        assert all(r["pitches"] >= 1000 for r in body["rows"])

    def test_sort_and_order(self, client):
        top = client.get("/api/leaderboard", params={"sort": "whiff_over_expected", "limit": 1}).json()
        best = max(grades_csv(), key=lambda g: float(g["whiff_over_expected"]))
        assert top["rows"][0]["pitcher"] == int(best["pitcher"])
        low = client.get(
            "/api/leaderboard", params={"sort": "pitches", "order": "asc", "limit": 1}
        ).json()
        assert low["rows"][0]["pitches"] == min(int(g["swings"]) for g in grades_csv())

    def test_paging(self, client):
        first = client.get("/api/leaderboard", params={"limit": 10}).json()["rows"]
        second = client.get("/api/leaderboard", params={"limit": 10, "offset": 10}).json()["rows"]
        assert [r["rank"] for r in first] == list(range(1, 11))
        assert [r["rank"] for r in second] == list(range(11, 21))

    def test_bad_parameters_are_rejected(self, client):
        assert client.get("/api/leaderboard", params={"sort": "drop table"}).status_code == 422
        assert client.get("/api/leaderboard", params={"limit": 0}).status_code == 422
        assert client.get("/api/leaderboard", params={"season": "1999"}).status_code == 404


class TestPitcherApi:
    def test_season_line_matches_the_report(self, client):
        miller = next(g for g in grades_csv() if g["pitcher"] == "695243")
        body = client.get("/api/pitchers/695243").json()
        line = body["seasons"][0]
        assert body["name"] == "Mason Miller"
        assert line["pitches"] == int(miller["swings"])
        assert line["expected_whiff"] == float(miller["expected_whiff_rate"])
        assert line["actual_whiff"] == float(miller["actual_whiff_rate"])
        assert line["rank"] == 1 and line["of"] == len(grades_csv())

    def test_no_pitch_breakdown_is_invented_for_the_seed(self, client):
        line = client.get("/api/pitchers/695243").json()["seasons"][0]
        assert line["pitch_types"] == [] and line["months"] == []

    def test_unknown_pitcher_is_404(self, client):
        response = client.get("/api/pitchers/1")
        assert response.status_code == 404
        assert response.headers["content-type"].startswith("application/json")


class TestContractMinimumDatabase:
    """The API must run on the minimum columns the pipeline is required to write."""

    def test_names_fall_back_to_the_pitcher_id(self, contract_db):
        c = TestClient(create_app(contract_db, REPORTS))
        body = c.get("/api/leaderboard").json()
        assert body["seasons"] == [2026, 2025]
        assert body["season"] == 2026
        assert [r["name"] for r in body["rows"]] == ["11", "22"]
        assert body["sample_unit"] == "pitches"

    def test_pitch_breakdown_aggregates_the_pitches_table(self, contract_db):
        c = TestClient(create_app(contract_db, REPORTS))
        body = c.get("/api/pitchers/11").json()
        assert [s["season"] for s in body["seasons"]] == [2026, 2025]
        current = body["seasons"][0]
        ff, sl = current["pitch_types"]
        assert (ff["pitch_type"], ff["pitches"]) == ("FF", 3)
        assert ff["share"] == pytest.approx(0.75)
        assert ff["expected_whiff"] == pytest.approx(0.30)
        assert ff["actual_whiff"] == pytest.approx(2 / 3)
        assert ff["whiff_over_expected"] == pytest.approx(2 / 3 - 0.30)
        assert (sl["pitch_type"], sl["pitches"]) == ("SL", 1)
        assert [m["month"] for m in current["months"]] == ["2026-04", "2026-05"]
        assert [m["pitches"] for m in current["months"]] == [3, 1]
        # The 2025 line only sees 2025 pitches.
        assert [p["pitch_type"] for p in body["seasons"][1]["pitch_types"]] == ["CU"]

    def test_pages_render_without_extension_columns(self, contract_db):
        c = TestClient(create_app(contract_db, REPORTS))
        board = c.get("/")
        assert board.status_code == 200
        assert "Avg stuff leaderboard" in board.text
        assert "committed in this repo" not in board.text  # not a seeded build
        pitcher = c.get("/pitchers/11")
        assert pitcher.status_code == 200
        assert "Four-seam fastball" in pitcher.text
        assert "Apr 2026" in pitcher.text
        assert "Other seasons" in pitcher.text
        unknown_type = c.get("/pitchers/22")
        assert "XX" in unknown_type.text
        model = c.get("/model")
        assert "Sep 27, 2026" in model.text
        assert c.get("/api/model-card").json()["train_auc"] == pytest.approx(0.81)

    def test_season_switch(self, contract_db):
        c = TestClient(create_app(contract_db, REPORTS))
        body = c.get("/api/leaderboard", params={"season": "2025"}).json()
        assert [r["pitcher"] for r in body["rows"]] == [11]


class TestModelCard:
    def test_key_numbers_come_from_the_performance_report(self, client):
        body = client.get("/api/model-card").json()
        assert body["key_numbers"] == {
            "full_auc": "0.788",
            "stuff_auc": "0.767",
            "baseline_auc": "0.613",
            "stuff_auc_before_vaa": "0.652",
        }
        assert body["test_auc"] == pytest.approx(0.7875)
        assert body["train_auc"] is None
        assert body["coverage"]["live_scoring"] is False
        assert body["coverage"]["seeded_from_reports"] is True
        assert [p["auc"] for p in body["performance"]] == [0.6133, 0.7674, 0.7875]
        assert len(body["calibration"]) == 10
        assert len(body["validity_by_threshold"]) == 7
        assert body["predictive_validity"][-1]["predictor"] == "50/50 blend"

    def test_page_says_the_season_is_over_and_what_is_covered(self, client):
        text = client.get("/model").text
        assert "The 2026 MLB season is over" in text
        assert "Nothing on this site scores live games" in text
        assert "Aug 19, 2026" in text
        assert "0.788" in text and "0.767" in text and "0.652" in text
        assert "not recorded" in text  # train_auc

    def test_every_figure_is_served(self, client):
        body = client.get("/api/model-card").json()
        assert len(body["figures"]) == len(reports.FIGURES) == 5
        for fig in body["figures"]:
            response = client.get(fig["url"])
            assert response.status_code == 200
            assert response.headers["content-type"] == "image/png"

    def test_only_listed_figures_are_served_from_reports(self, client):
        assert client.get("/reports/pitcher_stuff_grades.csv").status_code == 404
        assert client.get("/reports/run_output.txt").status_code == 404
        assert client.get("/reports/..%2Fpyproject.toml").status_code == 404
        assert client.get("/reports/%2e%2e/README.md").status_code == 404
        assert client.get("/src/pitchquality/web/app.py").status_code == 404
        assert client.get("/data/pitchquality.db").status_code == 404

    def test_round_half_up(self):
        assert reports.round_half_up("0.7875") == "0.788"
        assert reports.round_half_up("0.7674") == "0.767"
        assert reports.round_half_up("0.6133") == "0.613"


class TestPages:
    def test_leaderboard_page(self, client):
        text = client.get("/").text
        assert "Stuff+ leaderboard" in text
        assert "Mason Miller" in text
        assert 'href="/pitchers/695243"' in text
        assert "Page 1 of 12" in text  # 554 pitchers, 50 per page
        assert "committed in this repo" in text  # seeded-build notice

    def test_leaderboard_search_sort_and_paging(self, client):
        assert "Andrés Muñoz" in client.get("/", params={"q": "munoz"}).text
        assert "No pitchers match" in client.get("/", params={"q": "zzzzzz"}).text
        last = client.get("/", params={"page": 999})
        assert last.status_code == 200 and "Page 12 of 12" in last.text
        by_name = client.get("/", params={"sort": "name"}).text
        assert 'aria-sort="ascending"' in by_name
        # An unknown sort key falls back to the default instead of erroring.
        assert client.get("/", params={"sort": "nonsense"}).status_code == 200

    def test_search_text_is_escaped(self, client):
        text = client.get("/", params={"q": "<script>alert(1)</script>"}).text
        assert "<script>alert(1)</script>" not in text
        assert "&lt;script&gt;" in text

    def test_pitcher_page(self, client):
        text = client.get("/pitchers/695243").text
        assert "Mason Miller" in text
        assert "Rank 1 of 554" in text
        assert "141.3" in text
        assert "no pitch-level rows yet" in text
        assert "baseballsavant.mlb.com/savant-player/695243" in text
        assert "Josh Hader" in text  # next by rank

    def test_unknown_pitcher_page_is_html_404(self, client):
        response = client.get("/pitchers/1")
        assert response.status_code == 404
        assert "Not found" in response.text

    def test_static_assets(self, client):
        assert client.get("/static/style.css").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/favicon.svg").status_code == 200


class TestReadOnly:
    def test_connection_refuses_writes(self, seeded):
        con = Database(seeded)._connect()
        with pytest.raises(sqlite3.OperationalError):
            con.execute("DELETE FROM leaderboard")
        con.close()

    def test_requests_leave_the_file_untouched(self, seeded, client):
        before = seeded.read_bytes()
        for path in ("/", "/pitchers/695243", "/model", "/api/leaderboard", "/api/model-card"):
            assert client.get(path).status_code == 200
        assert seeded.read_bytes() == before

    def test_missing_file_raises_instead_of_creating_one(self, tmp_path):
        path = tmp_path / "absent.db"
        with pytest.raises(DatabaseUnavailable):
            Database(path).snapshot()
        assert not path.exists()

    def test_a_rewritten_database_is_picked_up(self, contract_db):
        c = TestClient(create_app(contract_db, REPORTS))
        assert c.get("/api/health").json()["leaderboard_rows"] == 3
        con = sqlite3.connect(contract_db)
        con.execute("INSERT INTO leaderboard VALUES (33, 2026, 1, 0.1, 0.1, 0.0, -0.1)")
        con.commit()
        con.close()
        assert c.get("/api/health").json()["leaderboard_rows"] == 4


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

    def test_stuff_handles_both_scales(self):
        assert fmt_stuff(141.3) == "141.3"
        assert fmt_stuff(0.3833) == ".383"

    def test_dates(self):
        assert fmt_date("2026-08-19") == "Aug 19, 2026"
        assert fmt_date(None) == "unknown"
        assert fmt_date("late August") == "late August"


def test_the_service_does_not_import_the_modeling_stack():
    """Render installs requirements-web.txt only: no pandas, no scikit-learn."""
    code = (
        "import sys; import pitchquality.web.app, pitchquality.web.seed; "
        "heavy = {'pandas', 'numpy', 'sklearn', 'matplotlib', 'pyarrow', 'pybaseball'}; "
        "print(sorted(heavy & set(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=True, cwd=ROOT,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert out.stdout.strip() == "[]"
