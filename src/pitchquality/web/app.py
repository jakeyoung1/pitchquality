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
MIN_SAMPLE_CHOICES = (0, 250, 500, 1000)

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
    """Stuff+ style indexes get one decimal; raw probabilities read as rates."""
    if value is None:
        return "n/a"
    return f"{value:.1f}" if abs(value) > 1.5 else fmt_rate(value)


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


def sample_label(meta: dict) -> str:
    return "Swings" if meta.get("sample_unit") == "swings" else "Pitches"


def stuff_label(meta: dict) -> str:
    return "Stuff+" if meta.get("stuff_scale") == "stuff_plus" else "Avg stuff"


def stuff_tier(value, meta: dict) -> str:
    """CSS class for a Stuff+ cell: each 10 points is one standard deviation."""
    if value is None or meta.get("stuff_scale") != "stuff_plus":
        return ""
    if value >= 120:
        return "tier-hot2"
    if value >= 110:
        return "tier-hot1"
    if value <= 80:
        return "tier-cold2"
    if value <= 90:
        return "tier-cold1"
    return ""


def coverage(meta: dict) -> dict:
    return {
        "first_game_date": meta.get("first_game_date"),
        "last_fetch_date": meta.get("last_fetch_date"),
        "seeded_from_reports": (meta.get("source") or "").startswith("seed"),
        "live_scoring": False,
    }


def public(row: dict) -> dict:
    return {key: row[key] for key in PUBLIC_FIELDS}


def to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def filter_sort(rows: list[dict], q: str, min_pitches: int, sort: str, order: str) -> list[dict]:
    needle = fold(q.strip())
    if needle:
        rows = [r for r in rows if all(part in r["search"] for part in needle.split())]
    if min_pitches:
        rows = [r for r in rows if (r["pitches"] or 0) >= min_pitches]
    field, desc_default = SORTS.get(sort, SORTS["avg_stuff"])
    desc = desc_default if order not in ("asc", "desc") else order == "desc"
    if field == "name":
        return sorted(rows, key=lambda r: fold(r["name"]), reverse=desc)
    present = [r for r in rows if r[field] is not None]
    absent = [r for r in rows if r[field] is None]
    present.sort(key=lambda r: (r[field], -r["rank"]), reverse=desc)
    return present + absent


def strip_points(rows: list[dict], field: str, selected: dict) -> dict | None:
    """Positions for the distribution strip on a pitcher page."""
    values = [r[field] for r in rows if r[field] is not None]
    if len(values) < 2 or selected[field] is None:
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


def reading(row: dict, meta: dict) -> list[str]:
    """Plain-language notes on one season line, built only from its numbers."""
    notes: list[str] = []
    woe = row["whiff_over_expected"]
    unit = "swings" if meta.get("sample_unit") == "swings" else "pitches"
    if woe is not None:
        points = abs(woe) * 100
        if round(woe, 3) == 0:
            notes.append("Whiff rate matches what the model expects from the stuff.")
        elif woe > 0:
            notes.append(
                f"Misses {points:.1f} more bats per 100 {unit} than the stuff alone predicts. "
                "A gap in this direction points at deception, sequencing, or a release "
                "trait the model does not capture."
            )
        else:
            notes.append(
                f"Misses {points:.1f} fewer bats per 100 {unit} than the stuff alone predicts."
            )
    if meta.get("sample_unit") == "swings" and row["pitches"] is not None:
        n = row["pitches"]
        if n < facts.MODEL_LEADS_BELOW_SWINGS:
            notes.append(
                f"At {n:,} swings the sample is small. Below roughly "
                f"{facts.MODEL_LEADS_BELOW_SWINGS} swings the model's expected whiff rate "
                "forecast the second half better than the pitcher's own whiff rate did."
            )
        elif n >= facts.OUTCOME_LEADS_FROM_SWINGS:
            notes.append(
                f"At {n:,} swings the pitcher's own whiff rate is the better forecast. "
                f"With {facts.OUTCOME_LEADS_FROM_SWINGS} or more swings in hand, results "
                "out-predicted the model in the repo's validity test."
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
        nicedate=fmt_date,
        nicemonth=fmt_month,
        half_up=reports.round_half_up,
    )
    templates.env.globals.update(repo_url=REPO_URL, pitch_names=facts.PITCH_TYPE_NAMES)

    def wants_json(request: Request) -> bool:
        return request.url.path.startswith("/api/")

    def page(request: Request, name: str, context: dict, status_code: int = 200) -> HTMLResponse:
        meta = context.get("meta") or {}
        context.setdefault("meta", meta)
        context.setdefault("coverage", coverage(meta))
        context.setdefault("sample_label", sample_label(meta))
        context.setdefault("stuff_label", stuff_label(meta))
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

    def pick_season(snap: Snapshot, season: str | None):
        if season is None:
            if not snap.seasons:
                raise HTTPException(404, "the leaderboard table is empty")
            return snap.seasons[0]
        for known in snap.seasons:
            if str(known) == season:
                return known
        raise HTTPException(404, f"no leaderboard rows for season {season}")

    def pitcher_payload(snap: Snapshot, pitcher_id: str) -> dict:
        lines = snap.find(pitcher_id)
        if not lines:
            raise HTTPException(404, f"no pitcher {pitcher_id} on the leaderboard")
        seasons = []
        for line in lines:
            if snap.has_pitch_rows:
                detail = database.pitch_breakdown(line["pitcher"], line["season"])
            else:
                detail = {"by_type": [], "by_month": []}
            seasons.append(
                {
                    **public(line),
                    "of": line["season_size"],
                    "whiff_over_expected_rank": line["woe_rank"],
                    "pitch_types": detail["by_type"],
                    "months": detail["by_month"],
                }
            )
        return {"pitcher": lines[0]["pitcher"], "name": lines[0]["name"], "seasons": seasons}

    def model_card_payload(snap: Snapshot) -> dict:
        card = reports.load(reports_path)
        meta = snap.meta
        return {
            "meta": meta,
            "coverage": coverage(meta),
            "train_auc": to_float(meta.get("train_auc")),
            "test_auc": to_float(meta.get("test_auc")),
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
        """200 when the database is readable and matches the contract, else 503."""
        snap = database.snapshot()
        return {
            "status": "ok",
            "leaderboard_rows": len(snap.rows),
            "last_fetch_date": snap.meta.get("last_fetch_date"),
            "model_version": snap.meta.get("model_version"),
        }

    @app.get("/api/leaderboard", tags=["api"])
    def api_leaderboard(
        season: str | None = None,
        q: str = Query("", max_length=80, description="Filter by pitcher name or id"),
        min_pitches: int = Query(0, ge=0, description="Minimum scored sample"),
        sort: str = Query("avg_stuff", pattern="^(" + "|".join(SORTS) + ")$"),
        order: str = Query("", pattern="^(asc|desc|)$"),
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ):
        snap = database.snapshot()
        chosen = pick_season(snap, season)
        rows = filter_sort(snap.season_rows(chosen), q, min_pitches, sort, order)
        return {
            "season": chosen,
            "seasons": list(snap.seasons),
            "sample_unit": snap.meta.get("sample_unit") or "pitches",
            "last_fetch_date": snap.meta.get("last_fetch_date"),
            "total": len(rows),
            "limit": limit,
            "offset": offset,
            "rows": [public(r) for r in rows[offset : offset + limit]],
        }

    @app.get("/api/pitchers/{pitcher_id}", tags=["api"])
    def api_pitcher(pitcher_id: str):
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
        min: int = Query(0, ge=0),  # noqa: A002 - short on purpose, it shows in the URL
        sort: str = "avg_stuff",
        order: str = "",
        page_number: int = Query(1, alias="page", ge=1),
    ):
        snap = database.snapshot()
        chosen = pick_season(snap, season)
        season_rows = snap.season_rows(chosen)
        sort = sort if sort in SORTS else "avg_stuff"
        order = order if order in ("asc", "desc") else ""
        rows = filter_sort(season_rows, q, min, sort, order)
        pages = max(1, -(-len(rows) // PAGE_SIZE))
        page_number = min_int(page_number, pages)
        start = (page_number - 1) * PAGE_SIZE
        desc = SORTS[sort][1] if not order else order == "desc"
        spread = max((abs(r["whiff_over_expected"] or 0) for r in season_rows), default=0) or 1

        def link(**changes) -> str:
            params = {
                "season": str(chosen) if len(snap.seasons) > 1 else "",
                "q": q.strip(),
                "min": min or "",
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
            "season_sample": sum(r["pitches"] or 0 for r in season_rows),
            "q": q.strip(),
            "min": min,
            "min_choices": MIN_SAMPLE_CHOICES,
            "sort": sort,
            "desc": desc,
            "page_number": page_number,
            "pages": pages,
            "first_shown": start + 1 if rows else 0,
            "last_shown": min_int(start + PAGE_SIZE, len(rows)),
            "spread": spread,
            "link": link,
            "sort_link": sort_link,
            "tier": lambda value: stuff_tier(value, snap.meta),
            "filtered": bool(q.strip() or min),
        }
        return page(request, "leaderboard.html", context)

    @app.get("/pitchers/{pitcher_id}", response_class=HTMLResponse, include_in_schema=False)
    def pitcher_page(request: Request, pitcher_id: str):
        snap = database.snapshot()
        payload = pitcher_payload(snap, pitcher_id)
        lines = snap.find(pitcher_id)
        current = lines[0]
        peers = sorted(snap.season_rows(current["season"]), key=lambda r: r["rank"])
        index = current["rank"] - 1
        context = {
            "meta": snap.meta,
            "title": payload["name"],
            "pitcher": payload,
            "line": current,
            "detail": payload["seasons"][0],
            "other_seasons": payload["seasons"][1:],
            "notes": reading(current, snap.meta),
            "strip": strip_points(peers, "avg_stuff", current),
            "woe_strip": strip_points(peers, "whiff_over_expected", current),
            "previous": peers[index - 1] if index > 0 else None,
            "next": peers[index + 1] if index + 1 < len(peers) else None,
            "tier": stuff_tier(current["avg_stuff"], snap.meta),
            "savant_url": (
                f"https://baseballsavant.mlb.com/savant-player/{quote(current['id'])}"
                if current["id"].isdigit()
                else None
            ),
            "has_pitch_rows": snap.has_pitch_rows,
        }
        return page(request, "pitcher.html", context)

    @app.get("/model", response_class=HTMLResponse, include_in_schema=False)
    def model_card_page(request: Request):
        snap = database.snapshot()
        card = model_card_payload(snap)
        context = {
            "meta": snap.meta,
            "title": "Model card",
            "card": card,
            "seasons": snap.seasons,
            "pitchers": len(snap.season_rows(snap.seasons[0])) if snap.seasons else 0,
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


def min_int(a: int, b: int) -> int:
    return a if a < b else b


app = create_app()
