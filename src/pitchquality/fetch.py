"""The existing Statcast download/cache path, shared by research and live runs."""
from pathlib import Path

import pandas as pd
from pybaseball import statcast

KEEP = [
    "game_date", "game_pk", "at_bat_number", "pitch_number", "game_type",
    "pitcher", "batter", "player_name", "pitch_type", "description",
    "release_speed", "release_spin_rate", "release_extension", "release_pos_x", "release_pos_z",
    "pfx_x", "pfx_z", "plate_x", "plate_z", "spin_axis", "arm_angle",
    "p_throws", "stand", "balls", "strikes", "zone", "vx0", "vy0", "vz0", "ax", "ay", "az",
]

WINDOWS = {
    "2025": ("2025-03-27", "2025-09-28"),
    "2026": ("2026-03-26", "2026-08-19"),
}


def fetch_cached(start: str, end: str, path: Path, *, force: bool = False) -> pd.DataFrame:
    """Cache only successful responses; errors never become empty cache files."""
    if path.exists() and not force:
        return pd.read_parquet(path)
    print(f"fetching Statcast {start} -> {end}", flush=True)
    df = statcast(start_dt=start, end_dt=end, verbose=False)
    df = df[[c for c in KEEP if c in df.columns]]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    try:
        df.to_parquet(tmp, index=False)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"{len(df):,} rows -> {path}", flush=True)
    return df
