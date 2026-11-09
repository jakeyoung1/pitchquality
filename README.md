# pitchquality — an expected-whiff model for pitch evaluation

[![CI](https://github.com/jakeyoung1/pitchquality/actions/workflows/ci.yml/badge.svg)](https://github.com/jakeyoung1/pitchquality/actions/workflows/ci.yml)

Estimates the probability that a swing misses, from the physical properties of
the pitch. Trained on 2025 Statcast, evaluated on a **held-out 2026 season** the
model never saw.

The point is not the AUC. The point is separating **stuff** — what a pitcher
does to the ball, which travels with him — from **results**, which are
contaminated by the hitters he faced, where he located, and luck.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -c requirements-model.txt -e ".[dev]"
.venv/bin/python scripts_fetch.py      # download + cache Statcast (~2 min)
.venv/bin/python -m pitchquality.cli   # fit, evaluate, write reports/
.venv/bin/python -m pytest tests -q    # 58 tests
```

## Data

| Season | Pitches | Swings | Whiff rate | Role |
|---|---|---|---|---|
| 2025 | 711,897 | 335,769 | .2316 | train |
| 2026 (through Aug 19) | 562,799 | 264,629 | .2296 | held-out test |

Whiff is conditioned on a swing. Modeling `P(whiff | pitch)` would confound
pitch quality with plate discipline — a pitch nobody offers at scores well for
entirely the wrong reason.

## Held-out performance (train 2025 → test 2026)

| Model | AUC | Log loss | Brier |
|---|---|---|---|
| Baseline (pitch-type mean whiff rate) | .6133 | .5233 | .17158 |
| **Stuff** (release characteristics only) | **.7674** | .4465 | .14187 |
| Full (stuff + location + count) | .7875 | .4303 | .13598 |

## The result that mattered

The first version of the stuff model scored **.652 AUC** and *lost* the
predictive-validity test — a pitcher's own past whiff rate forecast his future
whiff rate better than the model did. That is the correct reason to reject a
model, so the question became what it could not see.

The answer was **approach angle**. Statcast publishes trajectory as
position/velocity/acceleration at y = 50 ft; the angle the hitter actually sees
has to be integrated forward to the plate. A flat vertical approach angle at the
top of the zone misses bats that the same velocity and spin would not miss on a
steeper plane, and VAA is not recoverable from velocity and movement alone.

Adding VAA and HAA moved the stuff model from **.652 → .767 AUC**.

## Does the grade actually predict anything?

Splitting 2026 at its median date and forecasting second-half whiff rate
(330 pitchers, ≥100 swings per half):

| Predictor | r | R² |
|---|---|---|
| First-half **expected** whiff (model) | .6443 | .4151 |
| First-half **actual** whiff (outcome) | .6369 | .4056 |
| **50/50 blend of the two** | **.6827** | **.4661** |

The honest reading: the model beats raw outcomes, but *narrowly*. The defensible
claim is not that stuff replaces results — it is that stuff carries information
results do not, which is why the blend beats either alone.

### Where the model earns its keep

Sweeping the first-half sample threshold, with the second-half target held fixed
at ≥150 swings so only the predictor's sample varies:

| Min swings (H1) | n | r model | r outcome | model advantage |
|---|---|---|---|---|
| 25 | 335 | .6192 | .5878 | **+.0314** |
| 50 | 324 | .6214 | .6212 | +.0002 |
| 75 | 319 | .6319 | .6242 | +.0076 |
| 100 | 302 | .6501 | .6424 | +.0077 |
| 150 | 268 | .6748 | .6863 | −.0115 |
| 200 | 211 | .6308 | .6631 | −.0323 |
| 300 | 133 | .6151 | .6724 | **−.0573** |

Whiff rate stabilizes quickly, so with 300 swings in hand a pitcher's own
results win and the model should not be used in preference to them.

![Correlation with second-half whiff rate by first-half sample size: the stuff model leads below roughly 125 swings, the pitcher's own whiff rate leads above it](reports/crossover.png)

The model's
advantage is concentrated exactly where scouting decisions are hardest: a
prospect with 25 swings of data, a reliever just called up, a deadline target
three weeks into a new pitch.

## Face validity

Top of the 2026 Stuff+ leaderboard: Mason Miller, Josh Hader, Andrés Muñoz,
Ryan Helsley, Brendon Little, Jeremiah Estrada, Fernando Cruz, Jeff Hoffman,
Blake Snell, Jacob Misiorowski. A stuff model that did *not* surface these arms
would be broken regardless of its AUC.

`whiff_over_expected` — actual minus expected — is the residual worth reading.
Pitchers persistently above it are getting misses their raw stuff does not
explain, which points at deception, sequencing, or a release the model has not
captured.

## Known limitations

- **Overconfident at the top.** Calibration is tight through the eighth decile,
  then drifts: the top decile predicts .767 and observes .720. High-stuff pitches
  are graded slightly too generously.

  <img src="reports/calibration.png" alt="Calibration plot of predicted versus observed whiff rate, close to the diagonal until the top decile" width="480">

- **Compressed at the velocity extremes.** The model under-predicts whiffs above
  99 mph and over-predicts below 90 — regularization pulling toward the mean.
- **Whiff is not run prevention.** A swing-and-miss model says nothing about
  contact quality. A complete pitch-quality metric needs an expected-damage
  component alongside this one.
- **No hitter adjustment.** Facing a lineup that chases is not distinguished
  from beating a disciplined one.
- **Single-season training.** No multi-year stability testing, and no accounting
  for within-season Statcast calibration drift.
- **Grades are descriptive of a window**, not projections. There is no aging
  curve and no regression to a population prior.

## Layout

```
src/pitchquality/
  features.py   targets, handedness mirroring, approach angles, FB differentials
  model.py      three models, held-out evaluation, calibration table
  grades.py     Stuff+ index, predictive validity, sample-size sweep, blending
  plots.py      calibration, velocity curve, movement maps, scatter
  cli.py        end-to-end pipeline
tests/          58 tests
reports/        CSVs + figures written by the pipeline
```

## Live data pipeline (2026 season complete)

`data/pitchquality.db` is a committed, ready-to-read SQLite snapshot. The pipeline
backfilled **August 20 through October 28, 2026**, including postseason games.
October 28 is the final game date in the
[MLB season schedule](https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate=2026-08-20&endDate=2026-11-09).
The backfill downloaded 165,286 source pitches and scored **78,317 eligible
swings**. With the original March 26-August 19 window, the DB contains **342,865
scored swings** from 728,085 source pitches. Excluded rows are takes, bunts,
unsupported pitch types, and swings missing the inputs already required by
`features.build()`. `leaderboard.pitches` counts eligible swings, not all pitches
thrown. No sample threshold is applied in storage.
`reports/backfill_coverage.json` verifies all 559 completed games against the MLB
schedule, with no missing or extra games. The rain-cancelled September 27 game
823490 had no pitches and is documented separately in that report.

The original models were fit once on the unchanged 2025 window (711,897 source
pitches, 335,769 eligible swings), under the
[Jake-approved clarification on relay #132](https://github.com/jakeyoung1/agent-relay/issues/132#issuecomment-6751376907).
The original 2026 holdout remains March 26-August 19. It has 264,548 eligible
swings in this run. Actual AUCs from this fit:

| Model | Train AUC (2025) | Held-out AUC (2026 through Aug 19) |
|---|---|---|
| Stuff | 0.78798527 | 0.76684244 |
| Full | 0.80852965 | 0.78747845 |

These values are in `meta`, `models/whiff-2025.json`, and
`reports/live_pipeline.json`; the baseline run log and performance CSV are in
`reports/live_baseline_*`. The earlier research tables and figures above retain
their original run's values. Live consumers should use DB metadata. No features,
targets, hyperparameters, training windows, or model prediction code changed.

### Run and replay

From the repo root, with Python 3.12:

```bash
pip install -c requirements-model.txt -e ".[dev]"
python -m pitchquality.live
python -m pytest -q
```

On Windows, use `.venv\Scripts\python.exe` after activating/installing in the venv.
The normal live command loads `models/whiff-2025.joblib` and never fits a model.
It clamps the fetch date to yesterday or October 28, whichever is earlier.
The shipped DB is complete, so the 2026 offseason returns a successful `no-op`
without fetching data, loading estimators, or changing the database. The pipeline
is scoped to 2026; enabling another season is a separate configuration change.

To explicitly replay source corrections after season completion:

```bash
python -m pitchquality.live --backfill
```

This refetches the original 2026 window and August 20-October 28, reuses the frozen
estimators, and replaces the season in one transaction. Normal refresh reuses
the original window's parquet cache and fetches the tail through the new date.
The whole season is processed together so each pitch's fastball differentials
use the same season-wide reference window. A larger window can revise earlier
scores as the pitcher throws more fastballs; the feature algorithm is unchanged.
SQLite uses the Statcast identity `(game_pk, at_bat_number, pitch_number)` to avoid
duplicates. Replays also remove previously scored rows that become ineligible.
Failures preserve existing pitches, leaderboard rows, and the fetch watermark.
Source parquet files are local caches, not committed data. Caches from older
versions of `scripts_fetch.py` need refetching to acquire pitch identities.

The one-time artifact was created with:

```bash
python scripts_fetch.py
python -m pitchquality.cli --save-model models/whiff-2025.joblib
```

That research command refuses to overwrite an existing bundle. Do not run it for
refreshes. The bundle contains both estimators and their feature ordering; its
JSON manifest records the dependency versions, training/evaluation windows,
metrics, and SHA-256. Loading checks the manifest, hash, and feature order. Use
the pinned environment because
[scikit-learn does not support loading models across different versions](https://scikit-learn.org/stable/model_persistence.html).
Load only this trusted repository's joblib file.

### Storage contract and score definitions

The exact schema is [`src/pitchquality/schema.sql`](src/pitchquality/schema.sql)
and was [posted on relay #132](https://github.com/jakeyoung1/agent-relay/issues/132#issuecomment-6751472326)
for the API builder. All rates are fractions, not percentages.

- `pitches.expected_whiff`: full-model P(whiff | swing).
- `pitches.stuff_whiff`: stuff-model P(whiff | swing).
- `pitches.actual_whiff`: 0 or 1, using the original whiff definition; foul tips
  are contact.
- `pitches.stuff_score`: `100 + 10 * (stuff_whiff - mean) / std`, using the frozen
  2025 training-swing prediction distribution (population standard deviation).
  The reference mean/std are stored in the model manifest and `meta`. This
  pitch-level scale differs from the historical report's scaling across pitcher
  means. It stays comparable between refresh batches.
- `leaderboard`: one row per pitcher/season, arithmetic means of the pitch
  scores and probabilities. `whiff_over_expected = actual_whiff - expected_whiff`.
- `game_type`: source Statcast code (`R`, `F`, `D`, `L`, `W`); postseason is
  included in the 2026 season. `player_name` is retained for API display.
- `meta`: TEXT key/value pairs. `train_auc` and `test_auc` refer to the full
  model; `stuff_train_auc`/`stuff_test_auc` to the stuff model. `last_fetch_date`
  is the inclusive successfully processed date, not the run time.
  `last_pitch_date` is the latest scored game. Model version, schema version,
  scope, season bounds, score reference, source counts, and scored counts are
  also recorded. See `reports/live_pipeline.json` for every populated key.

The API can open this DB read-only. Consumers should retain the committed
snapshot as a fallback and check `model_version`/`schema_version` when replacing it.

### Free scheduled refresh and API handoff

`.github/workflows/refresh.yml` runs daily at **13:17 UTC** on GitHub's standard
Ubuntu runner, and supports manual dispatch. It uses read-only repository
permissions and the built-in `GITHUB_TOKEN`; no paid service or external secret
is required. It becomes scheduled only after Jake merges the PR. Public-repo
scheduled workflows may be disabled by GitHub after 60 days of repository
inactivity, so the completed-season snapshot remains the durable fallback.

Each run restores the latest non-expired artifact from a successful run of the
same workflow/branch, or starts with the committed DB. It uploads the resulting
`pitchquality-live` artifact containing `pitchquality.db`, retained for 90 days.
An empty offseason refresh is still successful and keeps that snapshot available.
Nothing pushes to main, publishes a release, or deploys an application.

For an API consumer, download `pitchquality-live` from the latest successful
**Refresh 2026 pitch data** run in the Actions UI and put `pitchquality.db` at
`data/pitchquality.db`. An already authenticated GitHub CLI can also download it:

```bash
gh run list --repo jakeyoung1/pitchquality --workflow refresh.yml --branch main --status success
gh run download RUN_ID --repo jakeyoung1/pitchquality --name pitchquality-live --dir data
```

This PR supplies the database and refresh artifact. API hosting and any automatic
artifact ingestion by a hosted service belong to relay #133; this pipeline does
not deploy or automatically update a running API process.
