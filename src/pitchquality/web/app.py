"""FastAPI service: a JSON API and a server-rendered dashboard.

Run locally with ``uvicorn --app-dir src pitchquality.web.app:app``.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from urllib.parse import quote, urlencode

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import __version__
from . import facts, reports
from .db import REPO_ROOT, Database, DatabaseUnavailable, Snapshot, fold

HERE = Path(__file__).resolve().parent
REPO_URL = "https://github.com/jakeyoung1/pitchquality"
PAGE_SIZE = 50
# The database stores every pitcher with a scored swing. The board starts at
# the research threshold; 0 shows everyone.
DEFAULT_MIN = facts.QUALIFIED_MIN_SWINGS
MIN_SAMPLE_CHOICES = (DEFAULT_MIN, 250, 500, 1000, 0)

SORTS = {
    "name": ("name", False),
    "pitches": ("pitches", True),
    "avg_stuff": ("avg_stuff", True),
    "expected_whiff": ("expected_whiff", True),
    "actual_whiff": ("actual_whiff", True),
    "whiff_over_expected": ("whiff_over_expected", True),
}
PUBLIC_FIELDS = (
    "pitcher",
    "name",
    "season",
    "pitches",
    "avg_stuff",
    "expected_whiff",
    "actual_whiff",
    "whiff_over_expected",
    "qualified",
    "rank",
)


# ---------------------------------------------------------------- formatting


def fmt_rate(value) -> str:
    """Baseball style: 0.4433 reads as ``.443``."""
    if value is None:
        return "n/a"
    text = f"{abs(value):.3f}"
    text = text[1:] if text.startswith("0") else text
    return f"-{text}" if value < 0 else text


def fmt_signed(value) -> str:
    if value is None:
        return "n/a"
    text = f"{abs(value):.3f}"
    text = text[1:] if text.startswith("0") else text
    if round(value, 3) == 0:
        return text
    return ("+" if value > 0 else "−") + text


def fmt_stuff(value) -> str:
    """Stuff scores sit on a 100/10 scale and read to one decimal."""
    return "n/a" if value is None else f"{value:.1f}"


def fmt_auc(value, places: int = 4) -> str:
    """AUCs as stored in ``meta`` (TEXT) or a report CSV."""
    number = to_float(value)
    return "not recorded" if number is None else f"{number:.{places}f}"


def fmt_int(value) -> str:
    return "n/a" if value is None else f"{value:,}"


def fmt_date(value) -> str:
    """``2026-08-19`` reads as ``Aug 19, 2026``; anything else passes through."""
    if not value:
        return "unknown"
    try:
        d = date.fromisoformat(str(value)[:10])
    except ValueError:
        return str(value)
    return f"{d:%b} {d.day}, {d.year}"


def fmt_month(value) -> str:
    try:
        d = date.fromisoformat(f"{value}-01")
    except (TypeError, ValueError):
        return str(value)
    return f"{d:%b %Y}"


# ------------------------------------------------------------------- helpers


def stuff_tier(row: dict) -> str:
    """CSS class for a stuff score, by standard deviations from the mean of
    the season's ranked pitchers. Unranked pitchers get no tint."""
    z = row.get("stuff_z")
    if z is None:
        return ""
    if z >= 2:
        return "tier-hot2"
    if z >= 1:
        return "tier-hot1"
    if z <= -2:
        return "tier-cold2"
    if z <= -1:
        return "tier-cold1"
    return ""


def coverage(meta: dict) -> dict:
    return {
        "season_start": meta.get("season_start"),
        "season_end": meta.get("season_end"),
        "last_pitch_date": meta.get("last_pitch_date") or meta.get("last_fetch_date"),
        "last_fetch_date": meta.get("last_fetch_date"),
        "backfill_start": meta.get("backfill_start"),
        "live_scoring": False,
    }


def public(row: dict) -> dict:
    return {key: row[key] for key in PUBLIC_FIELDS}


def to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def filter_sort(rows: list[dict], q: str, min_pitches: int, sort: str, order: str) -> list[dict]:
    needle = fold(q.strip())
    if needle:
        rows = [r for r in rows if all(part in r["search"] for part in needle.split())]
    if min_pitches:
        rows = [r for r in rows if r["pitches"] >= min_pitches]
    field, desc_default = SORTS.get(sort, SORTS["avg_stuff"])
    desc = desc_default if order not in ("asc", "desc") else order == "desc"
    if field == "name":
        return sorted(rows, key=lambda r: (fold(r["name"]), r["pitcher"]), reverse=desc)
    # Ties fall back to the stuff score, then the id, so paging is stable.
    return sorted(rows, key=lambda r: (r[field], r["avg_stuff"], -r["pitcher"]), reverse=desc)


def strip_points(peers: list[dict], field: str, selected: dict) -> dict | None:
    """Positions for the distribution strip on a pitcher page."""
    values = [r[field] for r in peers]
    if len(values) < 2 or not selected["qualified"]:
        return None
    lo, hi = min(values), max(values)
    if hi == lo:
        return None
    scale = lambda v: round(20 + 960 * (v - lo) / (hi - lo), 1)  # noqa: E731
    ordered = sorted(values)
    return {
        "ticks": [scale(v) for v in values],
        "selected": scale(selected[field]),
        "median": scale(ordered[len(ordered) // 2]),
        "lo": lo,
        "hi": hi,
        "median_value": ordered[len(ordered) // 2],
    }


def reading(row: dict, stuff_whiff: float | None) -> list[str]:
    """Plain-language notes on one season line, built only from its numbers."""
    notes: list[str] = []
    n = row["pitches"]
    if stuff_whiff is not None:
        gap = (row["expected_whiff"] - stuff_whiff) * 100
        if round(gap, 1) == 0:
            notes.append(
                f"Stuff alone predicts a {fmt_rate(stuff_whiff)} whiff rate, and adding "
                "location and count leaves that where it is."
            )
        else:
            notes.append(
                f"Stuff alone predicts a {fmt_rate(stuff_whiff)} whiff rate. With location "
                f"and count the full model expects {fmt_rate(row['expected_whiff'])}, so where "
                f"and when the pitches were thrown {'adds' if gap > 0 else 'costs'} "
                f"{abs(gap):.1f} whiffs per 100 swings."
            )
    woe = row["whiff_over_expected"]
    if round(woe, 3) == 0:
        notes.append("The actual whiff rate matches what the full model expects.")
    else:
        notes.append(
            f"Hitters missed {abs(woe) * 100:.1f} {'more' if woe > 0 else 'fewer'} times per "
            "100 swings than the full model expects from stuff, location, and count."
        )
    if n < facts.QUALIFIED_MIN_SWINGS:
        notes.append(
            f"With {n:,} swings this pitcher is under the {facts.QUALIFIED_MIN_SWINGS}-swing "
            "minimum and is not ranked."
        )
    elif n < facts.MODEL_LEADS_BELOW_SWINGS:
        notes.append(
            f"At {n:,} swings the sample is small. Below roughly "
            f"{facts.MODEL_LEADS_BELOW_SWINGS} swings, the stuff model's expected whiff rate "
            "forecast the rest of the season better than the pitcher's own whiff rate did "
            "in the repo's validity test."
        )
    elif n >= facts.OUTCOME_LEADS_FROM_SWINGS:
        notes.append(
            f"At {n:,} swings the pitcher's own whiff rate is the better forecast. With "
            f"{facts.OUTCOME_LEADS_FROM_SWINGS} or more swings in hand, results out-predicted "
            "the stuff model in the repo's validity test."
        )
    return notes


# ----------------------------------------------------------------------- app


def create_app(db_path: Path | str | None = None, reports_dir: Path | str | None = None) -> FastAPI:
    database = Database(db_path)
    reports_path = Path(
        reports_dir or os.environ.get("PITCHQUALITY_REPORTS") or REPO_ROOT / "reports"
    ).resolve()

    app = FastAPI(
        title="pitchquality",
        version=__version__,
        description=(
            "Expected-whiff pitch grades on Statcast data. Read-only API over "
            "data/pitchquality.db."
        ),
    )
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.filters.update(
        rate=fmt_rate,
        signed=fmt_signed,
        stuff=fmt_stuff,
        intc=fmt_int,
        auc=fmt_auc,
        nicedate=fmt_date,
        nicemonth=fmt_month,
        half_up=reports.round_half_up,
    )
    templates.env.globals.update(
        repo_url=REPO_URL,
        pitch_names=facts.PITCH_TYPE_NAMES,
        game_type_names=facts.GAME_TYPE_NAMES,
        qualified_min=facts.QUALIFIED_MIN_SWINGS,
    )

    def wants_json(request: Request) -> bool:
        return request.url.path.startswith("/api/")

    def page(request: Request, name: str, context: dict, status_code: int = 200) -> HTMLResponse:
        meta = context.setdefault("meta", {})
        context.setdefault("coverage", coverage(meta))
        return templates.TemplateResponse(request, name, context, status_code=status_code)

    @app.exception_handler(DatabaseUnavailable)
    async def database_unavailable(request: Request, exc: DatabaseUnavailable):
        if wants_json(request):
            return JSONResponse({"detail": f"database unavailable: {exc}"}, status_code=503)
        return page(request, "error.html", {"title": "Database unavailable", "detail": str(exc)}, 503)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        if wants_json(request):
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        try:
            meta = database.snapshot().meta
        except DatabaseUnavailable:
            meta = {}
        title = "Not found" if exc.status_code == 404 else "Something went wrong"
        context = {"title": title, "detail": str(exc.detail), "meta": meta}
        return page(request, "error.html", context, exc.status_code)

    def pick_season(snap: Snapshot, season: str | None) -> int:
        if season is None:
            if not snap.seasons:
                raise HTTPException(404, "the leaderboard table is empty")
            return snap.seasons[0]
        for known in snap.seasons:
            if str(known) == season:
                return known
        raise HTTPException(404, f"no leaderboard rows for season {season}")

    def season_payload(line: dict) -> dict:
        detail = database.pitch_breakdown(line["pitcher"], line["season"])
        return {
            **public(line),
            "of": line["season_qualified"],
            "whiff_over_expected_rank": line["woe_rank"],
            "stuff_whiff": detail["total"]["stuff_whiff"],
            "pitch_types": detail["by_type"],
            "months": detail["by_month"],
            "game_types": detail["by_stage"],
        }

    def pitcher_payload(snap: Snapshot, pitcher_id: str) -> dict:
        lines = snap.find(pitcher_id)
        if not lines:
            raise HTTPException(404, f"no pitcher {pitcher_id} on the leaderboard")
        return {
            "pitcher": lines[0]["pitcher"],
            "name": lines[0]["name"],
            "qualified_min": facts.QUALIFIED_MIN_SWINGS,
            "seasons": [season_payload(line) for line in lines],
        }

    def model_card_payload(snap: Snapshot) -> dict:
        card = reports.load(reports_path)
        meta = snap.meta
        return {
            "meta": meta,
            "coverage": coverage(meta),
            "database": {
                "seasons": list(snap.seasons),
                "pitchers": len(snap.rows),
                "rows_scored": to_int(meta.get("rows_scored")),
                "backfill_rows_scored": to_int(meta.get("backfill_rows_scored")),
            },
            "live_model": {
                "model_version": meta.get("model_version"),
                "full": {
                    "train_auc": to_float(meta.get("train_auc")),
                    "test_auc": to_float(meta.get("test_auc")),
                },
                "stuff": {
                    "train_auc": to_float(meta.get("stuff_train_auc")),
                    "test_auc": to_float(meta.get("stuff_test_auc")),
                },
                "held_out_swings": card["live_held_out_swings"],
                "stuff_reference_mean": to_float(meta.get("stuff_reference_mean")),
                "stuff_reference_std": to_float(meta.get("stuff_reference_std")),
            },
            "key_numbers": card["key_numbers"],
            "data": facts.DATA_ROWS,
            "performance": card["performance"],
            "calibration": card["calibration"],
            "predictive_validity": card["predictive_validity"],
            "validity_by_threshold": card["validity_by_threshold"],
            "limitations": [{"title": t, "detail": d} for t, d in facts.LIMITATIONS],
            "figures": [
                {"url": f"/reports/{f['file']}", "alt": f["alt"]} for f in card["figures"]
            ],
        }

    # ------------------------------------------------------------------ API

    @app.get("/api/health", tags=["api"])
    def health():
        """200 when the database is readable and matches the schema, else 503."""
        snap = database.snapshot()
        return {
            "status": "ok",
            "leaderboard_rows": len(snap.rows),
            "last_fetch_date": snap.meta.get("last_fetch_date"),
            "last_pitch_date": snap.meta.get("last_pitch_date"),
            "model_version": snap.meta.get("model_version"),
            "schema_version": snap.meta.get("schema_version"),
        }

    @app.get("/api/leaderboard", tags=["api"])
    def api_leaderboard(
        season: str | None = None,
        q: str = Query("", max_length=80, description="Filter by pitcher name or id"),
        min_pitches: int = Query(
            DEFAULT_MIN,
            ge=0,
            description="Minimum scored swings. The database keeps every pitcher; 0 returns all.",
        ),
        sort: str = Query("avg_stuff", pattern="^(" + "|".join(SORTS) + ")$"),
        order: str = Query("", pattern="^(asc|desc|)$"),
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ):
        """Pitchers for one season. `pitches` counts scored swings, `expected_whiff`
        is the full model's mean, and `rank` is by stuff score among pitchers with
        at least 100 swings (null below that)."""
        snap = database.snapshot()
        chosen = pick_season(snap, season)
        rows = filter_sort(snap.season_rows(chosen), q, min_pitches, sort, order)
        return {
            "season": chosen,
            "seasons": list(snap.seasons),
            "min_pitches": min_pitches,
            "qualified_min": facts.QUALIFIED_MIN_SWINGS,
            "last_pitch_date": snap.meta.get("last_pitch_date"),
            "total": len(rows),
            "limit": limit,
            "offset": offset,
            "rows": [public(r) for r in rows[offset : offset + limit]],
        }

    @app.get("/api/pitchers/{pitcher_id}", tags=["api"])
    def api_pitcher(pitcher_id: str):
        """One pitcher's season lines with pitch-type, monthly, and game-type splits
        aggregated from the pitches table."""
        return pitcher_payload(database.snapshot(), pitcher_id)

    @app.get("/api/model-card", tags=["api"])
    def api_model_card():
        return model_card_payload(database.snapshot())

    # ---------------------------------------------------------------- pages

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def leaderboard_page(
        request: Request,
        season: str | None = None,
        q: str = Query("", max_length=80),
        min_swings: int = Query(DEFAULT_MIN, alias="min", ge=0),
        sort: str = "avg_stuff",
        order: str = "",
        page_number: int = Query(1, alias="page", ge=1),
    ):
        snap = database.snapshot()
        chosen = pick_season(snap, season)
        season_rows = snap.season_rows(chosen)
        qualified = snap.qualified(chosen)
        sort = sort if sort in SORTS else "avg_stuff"
        order = order if order in ("asc", "desc") else ""
        rows = filter_sort(season_rows, q, min_swings, sort, order)
        pages = max(1, -(-len(rows) // PAGE_SIZE))
        page_number = min(page_number, pages)
        start = (page_number - 1) * PAGE_SIZE
        desc = SORTS[sort][1] if not order else order == "desc"
        spread = max((abs(r["whiff_over_expected"]) for r in qualified), default=0) or 1

        def link(**changes) -> str:
            params = {
                "season": str(chosen) if len(snap.seasons) > 1 else "",
                "q": q.strip(),
                "min": "" if min_swings == DEFAULT_MIN else min_swings,
                "sort": "" if sort == "avg_stuff" else sort,
                "order": order,
                "page": "",
            }
            params.update(changes)
            query = urlencode({k: v for k, v in params.items() if v not in ("", None)})
            return "/" + (f"?{query}" if query else "")

        def sort_link(key: str) -> str:
            if key == sort:
                return link(sort=key if key != "avg_stuff" else "", order="asc" if desc else "desc")
            return link(sort=key if key != "avg_stuff" else "", order="")

        context = {
            "meta": snap.meta,
            "title": f"{chosen} leaderboard",
            "season": chosen,
            "seasons": snap.seasons,
            "rows": rows[start : start + PAGE_SIZE],
            "total": len(rows),
            "season_total": len(season_rows),
            "season_qualified": len(qualified),
            "season_sample": sum(r["pitches"] for r in season_rows),
            "q": q.strip(),
            "min": min_swings,
            "min_choices": MIN_SAMPLE_CHOICES,
            "sort": sort,
            "desc": desc,
            "page_number": page_number,
            "pages": pages,
            "first_shown": start + 1 if rows else 0,
            "last_shown": min(start + PAGE_SIZE, len(rows)),
            "spread": spread,
            "link": link,
            "sort_link": sort_link,
            "tier": stuff_tier,
            "filtered": bool(q.strip()) or min_swings != DEFAULT_MIN,
        }
        return page(request, "leaderboard.html", context)

    @app.get("/pitchers/{pitcher_id}", response_class=HTMLResponse, include_in_schema=False)
    def pitcher_page(request: Request, pitcher_id: str):
        snap = database.snapshot()
        payload = pitcher_payload(snap, pitcher_id)
        current = snap.find(pitcher_id)[0]
        detail = payload["seasons"][0]
        peers = snap.qualified(current["season"])
        index = current["rank"] - 1 if current["qualified"] else None
        context = {
            "meta": snap.meta,
            "title": payload["name"],
            "pitcher": payload,
            "line": current,
            "detail": detail,
            "other_seasons": payload["seasons"][1:],
            "notes": reading(current, detail["stuff_whiff"]),
            "strip": strip_points(peers, "avg_stuff", current),
            "woe_strip": strip_points(peers, "whiff_over_expected", current),
            "previous": peers[index - 1] if index else None,
            "next": peers[index + 1] if index is not None and index + 1 < len(peers) else None,
            "tier": stuff_tier(current),
            "savant_url": f"https://baseballsavant.mlb.com/savant-player/{quote(current['id'])}",
        }
        return page(request, "pitcher.html", context)

    @app.get("/model", response_class=HTMLResponse, include_in_schema=False)
    def model_card_page(request: Request):
        snap = database.snapshot()
        context = {
            "meta": snap.meta,
            "title": "Model card",
            "card": model_card_payload(snap),
            "research": reports.load(reports_path),
            "seasons": snap.seasons,
        }
        return page(request, "model_card.html", context)

    @app.get("/reports/{name}", include_in_schema=False)
    def report_figure(name: str):
        """Serve the report figures by exact name. Nothing else under reports/."""
        if name not in reports.FIGURES or not (reports_path / name).is_file():
            raise HTTPException(404, "no such figure")
        return FileResponse(
            reports_path / name,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    return app


app = create_app()
