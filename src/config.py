"""Shared paths and experiment constants. Every script imports from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"              # temporary monthly OD csv (deleted after aggregation)
INTERIM = DATA / "interim"      # per-month aggregates (gitignored)
PROCESSED = DATA / "processed"  # model-ready tables (committed, small parquet)
EXTERNAL = DATA / "external"    # weather / calendar / events
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

for _p in (RAW, INTERIM, PROCESSED, EXTERNAL, RESULTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# Main study window (proposal): 2025-10 .. 2026-09
MAIN_START, MAIN_END = "2025-10", "2026-09"
# Extra history used only for the extended multi-month backtest
HIST_START = "2024-10"

# Time split (proposal 3.2)
TRAIN_END = "2026-07-31 23:00"
VALID_START, VALID_END = "2026-08-01 00:00", "2026-08-31 23:00"
TEST_START, TEST_END = "2026-09-01 00:00", "2026-09-30 23:00"

PEAK_HOURS = [7, 8, 17, 18]   # weekday peak hours used for "peak MAE"
CROWD_QUANTILE = 0.95         # per-station P95 of (entries+exits) on training data
RAIN_MM = 1.0                 # "rainy hour" threshold

# Taipei City reference point for gridded weather
TPE_LAT, TPE_LON = 25.0478, 121.5170

PANEL_PATH = PROCESSED / "panel_station_hour.parquet"
PANEL_HIST_PATH = PROCESSED / "panel_station_hour_hist.parquet"
