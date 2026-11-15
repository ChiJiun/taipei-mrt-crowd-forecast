"""Daily office calendar (holiday / make-up workday) from TaiwanCalendar.

Output external/calendar.csv: date, is_holiday, is_makeup_workday, holiday_name,
  is_weekend, day_type ('workday'|'holiday'), days_to_next_holiday, days_since_last_holiday,
  is_long_weekend (part of a >=3-day consecutive holiday block), holiday_block_len
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import EXTERNAL  # noqa: E402

URL = "https://cdn.jsdelivr.net/gh/ruyut/TaiwanCalendar/data/{y}.json"


def main(years=(2024, 2025, 2026, 2027)):
    rows = []
    for y in years:
        r = subprocess.run(["curl", "-sSfL", "-m", "60", URL.format(y=y)], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"skip {y}: {r.stderr.strip()}")
            continue
        rows += json.loads(r.stdout)
    cal = pd.DataFrame(rows)
    cal["date"] = pd.to_datetime(cal["date"], format="%Y%m%d")
    cal = cal.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    cal["is_holiday"] = cal["isHoliday"].astype(bool)
    cal["is_weekend"] = cal["date"].dt.dayofweek >= 5
    cal["is_makeup_workday"] = cal["is_weekend"] & ~cal["is_holiday"]
    cal["holiday_name"] = cal["description"].fillna("")
    cal["day_type"] = cal["is_holiday"].map({True: "holiday", False: "workday"})

    # consecutive holiday blocks (long weekends / Lunar New Year ...)
    block = (cal["is_holiday"] != cal["is_holiday"].shift()).cumsum()
    cal["holiday_block_len"] = cal.groupby(block)["is_holiday"].transform("size").where(cal["is_holiday"], 0)
    cal["is_long_weekend"] = cal["holiday_block_len"] >= 3
    hol_dates = cal.loc[cal.is_holiday, "date"]
    nxt = pd.merge_asof(cal[["date"]], hol_dates.rename("nh").to_frame(), left_on="date", right_on="nh",
                        direction="forward", allow_exact_matches=False)["nh"]
    prv = pd.merge_asof(cal[["date"]], hol_dates.rename("ph").to_frame(), left_on="date", right_on="ph",
                        direction="backward", allow_exact_matches=False)["ph"]
    cal["days_to_next_holiday"] = (nxt - cal["date"]).dt.days.clip(upper=30).fillna(30).astype(int)
    cal["days_since_last_holiday"] = (cal["date"] - prv).dt.days.clip(upper=30).fillna(30).astype(int)
    cols = ["date", "is_holiday", "is_makeup_workday", "is_weekend", "holiday_name", "day_type",
            "holiday_block_len", "is_long_weekend", "days_to_next_holiday", "days_since_last_holiday"]
    cal[cols].to_csv(EXTERNAL / "calendar.csv", index=False)
    print(f"calendar: {len(cal)} days, {cal.is_holiday.sum()} holidays, "
          f"{cal.is_makeup_workday.sum()} make-up workdays")


if __name__ == "__main__":
    main()
