"""Download the original training and held-out windows; importing does no work."""
from pathlib import Path

from pitchquality.fetch import KEEP, WINDOWS, fetch_cached


def main() -> None:
    for name, (start, end) in WINDOWS.items():
        path = Path("data") / f"statcast_{name}.parquet"
        df = fetch_cached(start, end, path)
        print(f"{name}: {len(df):,} rows -> {path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
