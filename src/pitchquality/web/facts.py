"""Numbers the dashboard quotes that live in prose rather than in a CSV.

Every constant here is copied from a file committed to this repo, named next to
it. ``tests/test_web.py`` asserts each one still appears in its source, so the
site cannot drift from the README without a test failing.
"""
from __future__ import annotations

# README.md, "The result that mattered": the stuff model before approach
# angles were added.
STUFF_AUC_BEFORE_VAA = "0.652"

# README.md, "Data" table. Swings and whiff rates also appear in
# reports/run_output.txt.
DATA_ROWS = [
    {
        "season": "2025",
        "pitches": "711,897",
        "swings": "335,769",
        "whiff_rate": ".2316",
        "role": "train",
    },
    {
        "season": "2026 (through Aug 19)",
        "pitches": "562,799",
        "swings": "264,629",
        "whiff_rate": ".2296",
        "role": "held-out test",
    },
]

# scripts_fetch.py, WINDOWS["2026"]: the date range behind the committed
# reports. The seed uses these for the database's coverage metadata.
REPORT_FIRST_GAME_DATE = "2026-03-26"
REPORT_LAST_GAME_DATE = "2026-08-19"

# README.md, crossover figure and "Where the model earns its keep": the model
# out-predicts a pitcher's own whiff rate below roughly 125 swings, and the
# pitcher's own results win with 300 swings in hand.
MODEL_LEADS_BELOW_SWINGS = 125
OUTCOME_LEADS_FROM_SWINGS = 300

# README.md, "Known limitations". Wording follows the README.
LIMITATIONS = [
    (
        "Overconfident at the top",
        "Calibration is tight through the eighth decile, then drifts: the top "
        "decile predicts .767 and observes .720. High-stuff pitches are graded "
        "slightly too generously.",
    ),
    (
        "Compressed at the velocity extremes",
        "The model under-predicts whiffs above 99 mph and over-predicts below "
        "90, which is regularization pulling toward the mean.",
    ),
    (
        "Whiff is not run prevention",
        "A swing-and-miss model says nothing about contact quality. A complete "
        "pitch-quality metric needs an expected-damage component alongside "
        "this one.",
    ),
    (
        "No hitter adjustment",
        "Facing a lineup that chases is not distinguished from beating a "
        "disciplined one.",
    ),
    (
        "Single-season training",
        "No multi-year stability testing, and no accounting for within-season "
        "Statcast calibration drift.",
    ),
    (
        "Grades describe a window",
        "They are not projections. There is no aging curve and no regression "
        "to a population prior.",
    ),
]

# Statcast pitch-type codes the model covers (features.MODELED_PITCH_TYPES).
PITCH_TYPE_NAMES = {
    "FF": "Four-seam fastball",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "ST": "Sweeper",
    "CU": "Curveball",
    "KC": "Knuckle curve",
    "CH": "Changeup",
    "FS": "Splitter",
    "SV": "Slurve",
}
