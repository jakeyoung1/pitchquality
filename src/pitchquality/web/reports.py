"""Load the model-card numbers from the CSVs the pipeline writes to ``reports/``.

Plain ``csv`` on purpose: the deployed service does not install pandas.
"""
from __future__ import annotations

import csv
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from pathlib import Path

from . import facts

# The only files under reports/ the site serves.
FIGURES = {
    "calibration.png": "Predicted versus observed whiff rate by decile. The points "
    "sit close to the diagonal until the top decile.",
    "crossover.png": "Correlation with second-half whiff rate by first-half sample "
    "size. The stuff model leads below roughly 125 swings and the pitcher's own "
    "whiff rate leads above it.",
    "whiff_by_velocity.png": "Observed and predicted whiff rate by pitch velocity.",
    "movement_map.png": "Pitch movement map from the held-out 2026 season.",
    "stuff_vs_outcome.png": "Expected whiff rate from the stuff model against "
    "actual whiff rate, one point per pitcher.",
}


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def round_half_up(value: str, places: int = 3) -> str:
    """Round a decimal string the way a reader would: ``0.7875`` gives ``0.788``."""
    quantum = Decimal(1).scaleb(-places)
    return str(Decimal(value).quantize(quantum, rounding=ROUND_HALF_UP))


def _numeric(rows: list[dict[str, str]], text_columns: tuple[str, ...] = ()) -> list[dict]:
    out = []
    for row in rows:
        shaped: dict = {}
        for key, value in row.items():
            if key in text_columns:
                shaped[key] = value
            else:
                number = float(value)
                shaped[key] = int(number) if number.is_integer() and "." not in value else number
        out.append(shaped)
    return out


@lru_cache(maxsize=4)
def load(reports_dir: Path) -> dict:
    performance = _numeric(_rows(reports_dir / "model_performance.csv"), ("model",))
    raw_performance = _rows(reports_dir / "model_performance.csv")

    def auc(prefix: str) -> str | None:
        for row in raw_performance:
            if row["model"].startswith(prefix):
                return row["auc"]
        return None

    full, stuff, baseline = auc("full"), auc("stuff"), auc("baseline")
    key_numbers = {
        "full_auc": round_half_up(full) if full else None,
        "stuff_auc": round_half_up(stuff) if stuff else None,
        "baseline_auc": round_half_up(baseline) if baseline else None,
        "stuff_auc_before_vaa": facts.STUFF_AUC_BEFORE_VAA,
    }
    return {
        "key_numbers": key_numbers,
        "performance": performance,
        "calibration": _numeric(_rows(reports_dir / "calibration.csv")),
        "predictive_validity": _numeric(_rows(reports_dir / "blend_validity.csv"), ("predictor",)),
        "validity_by_threshold": _numeric(_rows(reports_dir / "validity_by_threshold.csv")),
        "figures": [
            {"file": name, "alt": alt}
            for name, alt in FIGURES.items()
            if (reports_dir / name).is_file()
        ],
    }
