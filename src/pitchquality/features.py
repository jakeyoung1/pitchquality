"""Target definitions and feature engineering for the expected-whiff model.

Two conventions matter here and both are deliberate:

1. **Whiff is conditioned on a swing.** Modeling P(whiff | pitch) confounds
   pitch quality with plate discipline — a pitch nobody offers at scores well
   for the wrong reason. Every model in this package is fit on swings only.

2. **Everything is mirrored into a right-handed frame.** Raw ``pfx_x`` has the
   opposite sign for a lefty throwing the same shape, so an unmirrored model
   has to spend capacity relearning each pitch type twice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Statcast `description` values that represent a swing. Bunts are excluded:
# a missed bunt is not a swing-and-miss in any sense the model should learn.
SWING_EVENTS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play",
}

# A whiff is bat-through-the-zone-and-missed. `foul_tip` is contact, so it is a
# swing but not a whiff — this matches Statcast's own whiff-rate definition.
WHIFF_EVENTS = {"swinging_strike", "swinging_strike_blocked"}

# Physical characteristics of the pitch, known at release, independent of where
# it ended up. This is the "stuff" feature set.
STUFF_FEATURES = [
    "release_speed",
    "release_spin_rate",
    "release_extension",
    "pfx_x_mir",
    "pfx_z",
    "release_pos_x_mir",
    "release_pos_z",
    "spin_axis_mir",
    "arm_angle",
    "velo_diff_fb",
    "pfx_x_diff_fb",
    "pfx_z_diff_fb",
    "vaa",
    "haa",
    "vaa_diff_fb",
]

# Where the pitch crossed the plate, plus the count. Adding these turns a
# "stuff" model into a "pitching" model.
LOCATION_FEATURES = ["plate_x_mir", "plate_z", "balls", "strikes"]

CATEGORICAL = ["pitch_type", "same_hand"]

# Pitch types with enough volume to model. Knuckleballs, eephuses and position
# players are dropped rather than modeled on a handful of observations.
MODELED_PITCH_TYPES = {
    "FF", "SI", "FC", "SL", "ST", "CU", "KC", "CH", "FS", "SV",
}

# Pitches a pitcher's other offerings are measured against.
FASTBALL_TYPES = {"FF", "SI", "FC"}


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``is_swing`` and ``is_whiff`` columns."""
    out = df.copy()
    desc = out["description"].astype("string")
    out["is_swing"] = desc.isin(SWING_EVENTS).fillna(False)
    out["is_whiff"] = desc.isin(WHIFF_EVENTS).fillna(False)
    return out


def mirror_handedness(df: pd.DataFrame) -> pd.DataFrame:
    """Express every pitch in a right-handed pitcher's frame of reference.

    Horizontal quantities flip sign for left-handers; spin axis reflects about
    the vertical (0/360) axis. Vertical quantities are untouched.
    """
    out = df.copy()
    is_lhp = (out["p_throws"] == "L").to_numpy()
    flip = np.where(is_lhp, -1.0, 1.0)

    for src, dst in [
        ("pfx_x", "pfx_x_mir"),
        ("plate_x", "plate_x_mir"),
        ("release_pos_x", "release_pos_x_mir"),
    ]:
        out[dst] = out[src].astype(float).to_numpy() * flip

    axis = out["spin_axis"].astype(float).to_numpy()
    out["spin_axis_mir"] = np.where(is_lhp, (360.0 - axis) % 360.0, axis)

    # Platoon state, not raw handedness: what matters is whether the matchup is
    # same-side, and that is symmetric across LHP/RHP once mirrored.
    out["same_hand"] = (out["p_throws"] == out["stand"]).astype(int)
    return out


def add_fastball_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """Add each pitch's velocity and movement gap from its pitcher's fastball.

    A changeup is not good in absolute terms; it is good relative to the
    fastball it is meant to look like. Pitchers with no qualifying fastball in
    the window get 0.0 differentials, which the model reads as "no separation
    information" rather than as a missing value.
    """
    missing = {"pfx_x_mir", "vaa"} - set(df.columns)
    if missing:
        raise ValueError(
            f"add_fastball_differentials requires {sorted(missing)}; run "
            "mirror_handedness() and add_approach_angles() first."
        )
    out = df.copy()
    fb = out[out["pitch_type"].isin(FASTBALL_TYPES)]
    ref = (
        fb.groupby("pitcher")
        .agg(
            fb_velo=("release_speed", "mean"),
            fb_pfx_x=("pfx_x_mir", "mean"),
            fb_pfx_z=("pfx_z", "mean"),
            fb_vaa=("vaa", "mean"),
        )
        .reset_index()
    )
    out = out.merge(ref, on="pitcher", how="left")
    out["velo_diff_fb"] = (out["release_speed"] - out["fb_velo"]).fillna(0.0)
    out["pfx_x_diff_fb"] = (out["pfx_x_mir"] - out["fb_pfx_x"]).fillna(0.0)
    out["pfx_z_diff_fb"] = (out["pfx_z"] - out["fb_pfx_z"]).fillna(0.0)
    out["vaa_diff_fb"] = (out["vaa"] - out["fb_vaa"]).fillna(0.0)
    return out.drop(columns=["fb_velo", "fb_pfx_x", "fb_pfx_z", "fb_vaa"])


def add_approach_angles(df: pd.DataFrame) -> pd.DataFrame:
    """Compute vertical and horizontal approach angle at the front of the plate.

    Statcast publishes the trajectory as position/velocity/acceleration at
    y = 50 ft, so the angle the hitter actually sees has to be integrated
    forward to the plate. Vertical approach angle is one of the better-known
    drivers of four-seam whiff — a flat VAA at the top of the zone misses bats
    that the same velocity and spin would not miss on a steeper plane — and it
    is not recoverable from velocity and movement alone, which is why the first
    version of this model could not see it.
    """
    out = df.copy()
    y_plate = 17.0 / 12.0  # front edge of the plate, in feet

    vy0 = out["vy0"].astype(float).to_numpy()
    vx0 = out["vx0"].astype(float).to_numpy()
    vz0 = out["vz0"].astype(float).to_numpy()
    ax_ = out["ax"].astype(float).to_numpy()
    ay_ = out["ay"].astype(float).to_numpy()
    az_ = out["az"].astype(float).to_numpy()

    with np.errstate(invalid="ignore"):
        # Ball travels in -y; solve for velocity at the plate, then the time to get there.
        vy_f = -np.sqrt(np.maximum(vy0**2 - 2 * ay_ * (50.0 - y_plate), 0.0))
        t = np.where(ay_ != 0, (vy_f - vy0) / ay_, np.nan)
        vz_f = vz0 + az_ * t
        vx_f = vx0 + ax_ * t
        out["vaa"] = np.degrees(np.arctan2(vz_f, np.abs(vy_f)))
        out["haa"] = np.degrees(np.arctan2(vx_f, np.abs(vy_f)))

    # Mirror horizontal approach angle into the right-handed frame, same as
    # every other horizontal quantity.
    is_lhp = (out["p_throws"] == "L").to_numpy()
    out["haa"] = np.where(is_lhp, -out["haa"], out["haa"])
    return out


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: targets, mirroring, differentials, and row filtering."""
    out = add_targets(df)
    out = out[out["pitch_type"].isin(MODELED_PITCH_TYPES)]
    out = mirror_handedness(out)
    out = add_approach_angles(out)
    out = add_fastball_differentials(out)
    out = out[out["is_swing"]]

    required = ["release_speed", "pfx_x_mir", "pfx_z", "plate_x_mir", "plate_z"]
    out = out.dropna(subset=required)
    return out.reset_index(drop=True)
