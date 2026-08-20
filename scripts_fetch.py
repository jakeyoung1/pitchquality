"""Download Statcast pitch-level data and cache it as parquet."""
import sys, warnings, time
warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
from pybaseball import statcast

KEEP = [
    "game_date","game_pk","pitcher","batter","player_name","pitch_type","description",
    "release_speed","release_spin_rate","release_extension","release_pos_x","release_pos_z",
    "pfx_x","pfx_z","plate_x","plate_z","spin_axis","arm_angle",
    "p_throws","stand","balls","strikes","zone","vx0","vy0","vz0","ax","ay","az",
]

WINDOWS = {
    "2025": ("2025-03-27", "2025-09-28"),
    "2026": ("2026-03-26", "2026-08-19"),
}

out = Path("data"); out.mkdir(exist_ok=True)
for name, (a, b) in WINDOWS.items():
    f = out / f"statcast_{name}.parquet"
    if f.exists():
        print(f"{name}: cached, {len(pd.read_parquet(f)):,} rows", flush=True); continue
    t0 = time.time()
    print(f"{name}: fetching {a} -> {b}", flush=True)
    df = statcast(start_dt=a, end_dt=b, verbose=False)
    df = df[[c for c in KEEP if c in df.columns]]
    df.to_parquet(f, index=False)
    print(f"{name}: {len(df):,} rows in {time.time()-t0:.0f}s -> {f}", flush=True)
print("DONE", flush=True)
