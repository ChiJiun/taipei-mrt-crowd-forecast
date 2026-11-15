"""Merge the hand-collected event lists (data/external/events_raw/*.csv, gathered by web search, every row
with source_url) into data/external/events.csv: applies Dome corrections (wikipedia CPBL schedules) and adds
explicit station mappings for location-varying events (燈節, 路跑/遊行).
Usage: python src/10_events_merge.py   (then python src/10_events.py)
"""
import sys, pandas as pd
sys.path.insert(0, __import__("os").path.dirname(__file__))
from config import EXTERNAL  # noqa: E402
S = str(EXTERNAL / "events_raw"); out = EXTERNAL / "events.csv"
parts = [pd.read_csv(f"{S}/{f}", dtype=str) for f in ("events_other.csv", "events_dome_arena.csv", "events_dome_additions.csv")
         if __import__("os").path.exists(f"{S}/{f}")]
ev = pd.concat(parts, ignore_index=True)
cor = pd.read_csv(f"{S}/events_dome_corrections.csv", dtype=str)
nfix = 0
for c in cor.itertuples(index=False):
    m = (ev.venue == "台北大巨蛋") & (ev.date == c.date) & (ev.name == c.existing_name)
    if m.sum() != 1:
        print("correction not matched uniquely:", c.date, c.existing_name, m.sum()); continue
    ev.loc[m, ["name", "start_time", "end_time", "source_url", "start_time_confidence"]] = [
        c.corrected_name, c.corrected_start, c.corrected_end, c.source_url, "high"]
    nfix += 1
print("corrections applied:", nfix)
def stations(r):
    n, v = r["name"], r["venue"]
    if v == "台北燈節":
        return "西門:1.0;北門:0.6;小南門:0.3" if n.startswith("2025") else "圓山:1.0;西門:0.6;北門:0.3"
    if v == "路跑/遊行":
        if "國慶" in n: return "台大醫院:1.0;中正紀念堂:0.5"
        if "同志" in n: return "市政府:1.0;國父紀念館:0.5;台北101/世貿:0.3"
        if "渣打" in n: return "台大醫院:1.0;中正紀念堂:0.5;大直:0.3"
        if "臺北馬拉松" in n or "台北馬拉松" in n: return "市政府:1.0;國父紀念館:0.5"
    return ""
ev["stations"] = ev.apply(stations, axis=1)
assert not ((ev.venue.isin(["台北燈節", "路跑/遊行"])) & (ev.stations == "")).any()
ev = ev.sort_values(["date", "start_time", "venue"]).drop_duplicates(["name", "venue", "date"])
ev.to_csv(out, index=False)
print(len(ev), ev.venue.value_counts().to_dict())
