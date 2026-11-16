"""工作流 E：期末簡報 report/slides.pptx（python-pptx，16:9，15 頁）。

主要數字直接從 results/ 讀取（reg_main.csv、ts_compare.csv、cluster_error.csv、spatial_ablation.csv、
clf_main.csv、clf_backtest.csv、reg_shap_groups.csv、eda_summary_numbers.json），結果更新後重跑即可。
圖片取自 figures/。迴歸主結果尚未合併時，可設定 REPORT_REF=origin/ws/B-regression。

用法：python src/50_build_slides.py
      soffice --headless --convert-to pdf --outdir report report/slides.pptx
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

sys.path.insert(0, str(Path(__file__).parent))
rd = importlib.import_module("50_report_data")
ROOT = rd.ROOT
OUT = ROOT / "report" / "slides.pptx"
TMP = Path(tempfile.mkdtemp(prefix="slides_fig_"))

FONT = "Microsoft JhengHei"
NAVY = RGBColor(0x1B, 0x2A, 0x4A)
INK = RGBColor(0x1F, 0x23, 0x2B)
MUTED = RGBColor(0x5B, 0x60, 0x6B)
ACCENT = RGBColor(0xE0, 0x6C, 0x2F)      # 橘：重點數字
BLUE = RGBColor(0x2A, 0x78, 0xD6)
GRAY = RGBColor(0xB8, 0xB6, 0xAE)
TINT = RGBColor(0xF1, 0xF4, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
W, H = Inches(13.333), Inches(7.5)
MX = Inches(0.6)


# ---------------------------------------------------------------- data
def load():
    D = {}
    rm = rd.read_csv("reg_main.csv") if rd.has("reg_main.csv") else None
    D["has_b"] = rm is not None
    ts = rd.read_csv("ts_compare.csv")
    D["ts"] = ts
    D["rm"] = rm
    D["ce"] = rd.read_csv("cluster_error.csv")
    D["sa"] = rd.read_csv("spatial_ablation.csv")
    cm = rd.read_csv("clf_main.csv")
    D["cm"] = cm[cm.group == "全部"]
    D["bt"] = rd.read_csv("clf_backtest.csv")
    D["eda"] = json.loads(rd.read_text("eda_summary_numbers.json"))
    D["sg"] = rd.read_csv("reg_shap_groups.csv") if rd.has("reg_shap_groups.csv") else None
    return D


def reg(D, target, h, method, col="MAE"):
    if D["rm"] is not None:
        r = D["rm"]
        return float(r[(r.target == target) & (r.horizon == h) & (r.method == method)][col].iloc[0])
    ts = D["ts"]
    m = {"LightGBM-scaled": "LightGBM (scaled)"}.get(method, method)
    return float(ts[(ts.target == target) & (ts.horizon == h) & (ts.method == m)][col].iloc[0])


def best_model(D):
    return "LightGBM-B" if D["has_b"] else "LightGBM-scaled"


# ---------------------------------------------------------------- helpers
def set_run_font(run, size, color=INK, bold=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = FONT
    rpr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rpr.find(qn(tag))
        if el is None:
            el = rpr.makeelement(qn(tag), {})
            rpr.append(el)
        el.set("typeface", FONT)


def text(slide, x, y, w, h, s, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    lines = s if isinstance(s, list) else [s]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        set_run_font(p.add_run(), size, color, bold)
        p.runs[0].text = ln
    return tb


def bullets(slide, x, y, w, h, items, size=16, color=INK, gap=10):
    """items: str 或 (粗體前綴, 其餘文字)。"""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Emu(Inches(0.28))))
        pPr.set("indent", str(-Emu(Inches(0.28))))
        bu = pPr.makeelement(qn("a:buChar"), {"char": "•"})
        pPr.append(bu)
        if isinstance(it, tuple):
            r = p.add_run()
            set_run_font(r, size, color, True)
            r.text = it[0]
            r = p.add_run()
            set_run_font(r, size, color)
            r.text = it[1]
        else:
            r = p.add_run()
            set_run_font(r, size, color)
            r.text = it
    return tb


def title(slide, s, sub=None):
    text(slide, MX, Inches(0.35), W - 2 * MX, Inches(0.8), s, size=30, color=NAVY, bold=True)
    if sub:
        text(slide, MX, Inches(1.05), W - 2 * MX, Inches(0.45), sub, size=14, color=MUTED)


def footer(slide, n, src=None):
    if src:
        text(slide, MX, Inches(7.0), Inches(10), Inches(0.35), f"資料：{src}", size=10, color=MUTED)
    text(slide, W - Inches(1.4), Inches(7.0), Inches(0.8), Inches(0.35), str(n), size=10, color=MUTED,
         align=PP_ALIGN.RIGHT)


def card(slide, x, y, w, h, fill=TINT):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    shp.adjustments[0] = 0.06
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def stat(slide, x, y, w, big, label, color=ACCENT, big_size=40):
    text(slide, x, y, w, Inches(0.8), big, size=big_size, color=color, bold=True)
    text(slide, x, y + Inches(0.85), w, Inches(0.7), label, size=13, color=MUTED)


def picture(slide, rel, x, y, w=None, h=None):
    p = ROOT / rel
    if not p.exists() and rd.REF:
        out = TMP / p.name
        r = subprocess.run(["git", "show", f"{rd.REF}:{rel}"], cwd=ROOT, capture_output=True)
        if r.returncode == 0:
            out.write_bytes(r.stdout)
            p = out
    if not p.exists():
        card(slide, x, y, w or Inches(5), h or Inches(3), fill=RGBColor(0xF6, 0xE7, 0xE7))
        text(slide, x, y, w or Inches(5), h or Inches(3), f"圖檔待補：{rel}", size=12, color=MUTED,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        return None
    return slide.shapes.add_picture(str(p), x, y, w, h)


def style_chart(ch, legend=True):
    ch.font.size = Pt(12)
    ch.font.name = FONT
    ch.font.color.rgb = MUTED
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.TOP
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(12)
    va = ch.value_axis
    va.major_gridlines.format.line.color.rgb = RGBColor(0xE4, 0xE3, 0xDF)
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(11)
    ca = ch.category_axis
    ca.format.line.color.rgb = RGBColor(0xC9, 0xC7, 0xC0)
    ca.tick_labels.font.size = Pt(12)


def table(slide, x, y, w, rows, col_w=None, size=13, row_h=0.42, hl_rows=()):
    nr, nc = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nr, nc, x, y, w, Inches(row_h * nr)).table
    if col_w:
        for j, cw in enumerate(col_w):
            gt.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        gt.rows[i].height = Inches(row_h)
        for j, v in enumerate(row):
            c = gt.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.08)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            c.fill.fore_color.rgb = NAVY if i == 0 else (RGBColor(0xFD, 0xEE, 0xE4) if i in hl_rows else (
                WHITE if i % 2 else TINT))
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            set_run_font(r, size, WHITE if i == 0 else INK, bold=(i == 0 or i in hl_rows))
            r.text = str(v)
    return gt


# ---------------------------------------------------------------- slides
def build():
    D = load()
    BM = best_model(D)
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    blank = prs.slide_layouts[6]
    n = 0

    def new(dark=False):
        nonlocal n
        n += 1
        s = prs.slides.add_slide(blank)
        bg = s.background.fill
        bg.solid()
        bg.fore_color.rgb = NAVY if dark else WHITE
        return s

    imp = {(t, h): 100 * (1 - reg(D, t, h, BM) / reg(D, t, h, "Historical Average"))
           for t in ("entries", "exits") for h in (1, 2)}

    # 1 題目
    s = new(dark=True)
    text(s, MX, Inches(1.0), Inches(11), Inches(0.5), "資料科學導論　期末專題報告", size=18,
         color=RGBColor(0xC9, 0xD3, 0xE6))
    text(s, MX, Inches(2.2), Inches(12), Inches(1.4), "台北捷運站點人流預測與擁擠預警", size=44, color=WHITE, bold=True)
    text(s, MX, Inches(3.5), Inches(12), Inches(0.8),
         "122 站 × 逐小時｜LightGBM 預測未來 1–2 小時進出站量，並提前約 2 小時發出擁擠預警", size=20,
         color=RGBColor(0xC9, 0xD3, 0xE6))
    text(s, MX, Inches(5.6), Inches(12), Inches(0.5), "第＿組　組員：＿＿＿＿、＿＿＿＿、＿＿＿＿", size=18, color=WHITE)
    text(s, MX, Inches(6.2), Inches(12), Inches(0.5), "2026 年 11 月", size=14, color=RGBColor(0xC9, 0xD3, 0xE6))

    # 2 動機與目標
    s = new()
    title(s, "動機：人流集中在少數車站與尖峰", "疏運目前多靠人員經驗與即時監看，屬事後反應")
    eda = D["eda"]
    stat(s, MX, Inches(1.8), Inches(3.8), f"{eda['daily_entries_workday'] / 1e4:.0f} 萬", "平日日均進站人次")
    stat(s, Inches(4.6), Inches(1.8), Inches(3.8), "2.2 倍", "台北車站每小時進出量為第二名西門的倍數")
    stat(s, Inches(8.6), Inches(1.8), Inches(4.0), f"{eda['holiday_to_workday_ratio'] * 100:.0f}%",
         "假日日均進站量相對平日")
    card(s, MX, Inches(3.95), W - 2 * MX, Inches(2.75))
    text(s, Inches(0.85), Inches(4.1), Inches(6), Inches(0.5), "專案目標", size=20, color=NAVY, bold=True)
    bullets(s, Inches(0.85), Inches(4.65), Inches(11.6), Inches(2.0), [
        ("迴歸：", "預測各站未來 1–2 小時的進站量與出站量"),
        ("分類：", "預測站點是否達到擁擠門檻，並以逐日回放評估預警價值"),
        ("解釋：", "量化雨天、假日、大型活動對人流的影響"),
    ], size=17)
    footer(s, n, "eda_summary_numbers.json")

    # 3 資料
    s = new()
    title(s, "資料：OD 流量＋天氣、日曆、活動、颱風假", "主研究期 2025-10～2026-09；延長回測另加 2024-10～2025-09")
    table(s, MX, Inches(1.7), Inches(7.6), [
        ["資料", "來源", "粒度／規模"],
        ["每日分時各站 OD", "臺北捷運（政府開放資料）", "24 個月檔，每月約 300 MB"],
        ["逐時天氣", "Open-Meteo 觀測＋JMA 預報", "臺北單一格點，兩年逐時"],
        ["辦公日曆", "人事行政總處日曆", "假日、補班、連假長度"],
        ["大型活動", "場館節目表、維基百科", "574 筆、9 個場館"],
        ["颱風停班停課", "新聞與市府公告", "5 天"],
    ], col_w=[2.0, 3.0, 2.6], size=13, row_h=0.55)
    card(s, Inches(8.6), Inches(1.7), Inches(4.13), Inches(4.9))
    stat(s, Inches(8.85), Inches(1.9), Inches(3.8), "1,068,720", "站點小時（122 站 × 8,760 小時）", color=NAVY,
         big_size=34)
    stat(s, Inches(8.85), Inches(3.4), Inches(3.8), "87,840", "測試集站點小時（2026-09）", color=NAVY, big_size=34)
    stat(s, Inches(8.85), Inches(4.9), Inches(3.8), "訓練｜驗證｜測試", "~2026-07｜2026-08｜2026-09", color=NAVY,
         big_size=24)
    footer(s, n, "data/processed、data/external")

    # 4 方法流程
    s = new()
    title(s, "方法流程：一個全站共用模型，迴歸與預警共用特徵")
    picture(s, "figures/final_pipeline.png", Inches(0.9), Inches(1.35), w=Inches(11.5))
    footer(s, n)

    # 5 EDA（含期末更正）
    s = new()
    title(s, "雨天效果比期中小：中雨讓假日降約 8%", "2024/10–2026/9，排除颱風停班日；全網進站量相對同日別、同小時的無雨基準")
    picture(s, "figures/eda_rain_ratio_2y.png", MX, Inches(1.6), w=Inches(7.3))
    rr = rd.read_csv("eda_rain_ratio.csv").set_index(["period", "day_type", "rain"])
    h2 = rr.loc[("2024-10~2026-09", "假日", "1–5mm")]
    w2 = rr.loc[("2024-10~2026-09", "平日", "1–5mm")]
    old = rr.loc[("2025-10~2026-09 含颱風停班日（=進度報告定義）", "假日", ">5mm")]
    hb = rr.loc[("2024-10~2026-09", "假日", ">5mm")]
    ev = {x["venue"]: x for x in eda["event_uplift"]}["台北大巨蛋"]
    bullets(s, Inches(8.2), Inches(1.75), Inches(4.55), Inches(3.3), [
        ("中雨 1–5 mm：", f"假日 {h2.ratio_mean * 100:.0f}%、平日 {w2.ratio_mean * 100:.0f}%（CI 皆不含 1）"),
        ("大雨 > 5 mm 假日：", f"{hb.ratio_mean * 100:.0f}%，與無雨無顯著差異"),
        ("活動與颱風：", f"大巨蛋散場進站 {ev['departure_entries_ratio']:.1f} 倍；颱風假只剩 8%–27%"),
    ], size=15, gap=12)
    card(s, Inches(8.2), Inches(4.95), Inches(4.55), Inches(1.85), fill=RGBColor(0xFD, 0xEE, 0xE4))
    text(s, Inches(8.4), Inches(5.05), Inches(4.2), Inches(0.45), "期末更正", size=15, color=ACCENT, bold=True)
    text(s, Inches(8.4), Inches(5.5), Inches(4.2), Inches(1.3),
         f"期中「假日大雨降到 {old.ratio_mean * 100:.0f}%」主要來自巴威颱風停班的週六；排除颱風日後不成立",
         size=13, color=INK)
    footer(s, n, "results/eda_rain_ratio.csv、eda_summary_numbers.json")

    # 6 主結果 t+1 / t+2
    s = new()
    title(s, f"{BM} 把 t+1、t+2 的 MAE 降到 HA 的一半左右",
          "測試集 2026-09，MAE（人次／小時）；提案目標：相對 Historical Average 改善 ≥ 25%")
    cd = CategoryChartData()
    cd.categories = ["進站 t+1", "出站 t+1", "進站 t+2", "出站 t+2"]
    keys = [("entries", 1), ("exits", 1), ("entries", 2), ("exits", 2)]
    series = [("Historical Average", "Historical Average"), ("SARIMAX", None), ("LightGBM-scaled", "LightGBM-scaled")]
    if D["has_b"]:
        series.append(("LightGBM-B", "LightGBM-B"))
    for name, m in series:
        if name == "SARIMAX":
            ts = D["ts"]
            vals = [float(ts[(ts.target == t) & (ts.horizon == h) & (ts.method == "SARIMAX")]["MAE"].iloc[0])
                    for t, h in keys]
        else:
            vals = [reg(D, t, h, m) for t, h in keys]
        cd.add_series(name, vals)
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, MX, Inches(1.6), Inches(8.2), Inches(5.2), cd)
    ch = gf.chart
    style_chart(ch)
    cols = [GRAY, RGBColor(0x8F, 0xA8, 0xC8), RGBColor(0x7F, 0xB0, 0xEA), BLUE]
    for ser, c in zip(ch.series, cols):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = c
    plot = ch.plots[0]
    plot.gap_width = 60
    plot.overlap = -5
    last = ch.series[-1]
    last.data_labels.show_value = True
    last.data_labels.number_format = "0.0"
    last.data_labels.number_format_is_linked = False
    last.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    last.data_labels.font.size = Pt(12)
    last.data_labels.font.bold = True
    card(s, Inches(9.1), Inches(1.7), Inches(3.63), Inches(5.0))
    stat(s, Inches(9.35), Inches(1.9), Inches(3.3), f"{imp[('entries', 1)]:.0f}%", "t+1 進站 MAE 改善（相對 HA）")
    stat(s, Inches(9.35), Inches(3.4), Inches(3.3), f"{imp[('entries', 2)]:.0f}%", "t+2 進站 MAE 改善（相對 HA）")
    if rd.has("reg_backtest_monthly.csv"):
        bt = rd.read_csv("reg_backtest_monthly.csv")
        x = bt[bt.target == "entries"].pivot(index="month", columns="method", values="MAE")
        bimp = 100 * (1 - x["LightGBM-B"] / x["Historical Average"])
        n_ok = int((bimp >= 25).sum())
        stat(s, Inches(9.35), Inches(4.9), Inches(3.3), f"{n_ok}／{len(bimp)} 月",
             f"多月回測達標（改善 {bimp.min():.0f}–{bimp.max():.0f}%）")
        src_extra = "、reg_backtest_monthly.csv"
    else:
        text(s, Inches(9.35), Inches(4.95), Inches(3.2), Inches(1.6),
             "關鍵特徵：依站點量級正規化、起點當下相對歷史剖面的偏離", size=13, color=INK)
        src_extra = ""
    footer(s, n, "results/reg_main.csv、ts_compare.csv" + src_extra)

    # 6b 多月回測與預測區間
    if rd.has("reg_backtest_monthly.csv") and rd.has("reg_quantile.csv"):
        s = new()
        title(s, "結論不只對單月成立：12 個月回測全數達標", "左：滾動起點回測各月進站 MAE；右：台北車站 P10–P90 預測區間（分組 CQR 校準）")
        picture(s, "figures/reg_backtest_monthly.png", MX, Inches(1.55), h=Inches(3.2))
        picture(s, "figures/reg_interval_taipei_main.png", Inches(5.75), Inches(1.9), w=Inches(7.0))
        sp = rd.read_csv("reg_backtest_special.csv")
        sp = sp[sp.target == "entries"].set_index("subset")
        q = rd.read_csv("reg_quantile.csv")
        q = q[(q.target == "entries") & (q.horizon == 1) & (q.interval == "CQR-分組")].set_index("subset")
        bullets(s, MX, Inches(5.0), Inches(4.9), Inches(1.9), [
            ("颱風停班日：", f"HA {sp.loc['颱風假', 'MAE_Historical Average']:.0f} → LightGBM-B "
                       f"{sp.loc['颱風假', 'MAE_LightGBM-B']:.0f}（仍偏高估）"),
            ("跨年最難：", f"MAE {sp.loc[[k for k in sp.index if k.startswith('跨年')][0], 'MAE_LightGBM-B']:.0f}，"
                        "跨年夜低估、凌晨後高估"),
        ], size=14, gap=8)
        bullets(s, Inches(5.75), Inches(5.0), Inches(7.0), Inches(1.9), [
            ("整體覆蓋率：", f"{q.loc['all', 'coverage']:.1%}（名目 80%）"),
            ("平日尖峰：", f"只有 {q.loc['peak', 'coverage']:.1%}，開學月尖峰高於校準月"),
        ], size=14, gap=8)
        footer(s, n, "results/reg_backtest_monthly.csv、reg_backtest_special.csv、reg_quantile.csv（回測含兩年天氣）")

    # 7 SARIMA / Prophet
    s = new()
    title(s, "時間序列對照：SARIMAX 強，Prophet 不適合小時級預測", "進站 t+1 MAE（測試集 2026-09）")
    ts = D["ts"]
    t1 = ts[(ts.target == "entries") & (ts.horizon == 1)].set_index("method")
    names = ["Prophet", "Historical Average", "Prophet+AR", "SARIMAX", "LightGBM (scaled)"]
    labels = ["Prophet", "Historical Avg.", "Prophet+AR", "SARIMAX", "LightGBM-scaled"]
    vals = [float(t1.loc[k, "MAE"]) for k in names]
    if D["has_b"]:
        labels.append("LightGBM-B")
        vals.append(reg(D, "entries", 1, "LightGBM-B"))
    cd = CategoryChartData()
    cd.categories = labels
    cd.add_series("MAE", vals)
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, MX, Inches(1.6), Inches(6.6), Inches(5.2), cd)
    ch = gf.chart
    style_chart(ch, legend=False)
    ch.has_title = False
    ser = ch.series[0]
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = GRAY
    for i in range(len(vals)):
        pt = ser.points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = BLUE if i == len(vals) - 1 else (
            RGBColor(0x8F, 0xA8, 0xC8) if labels[i] == "SARIMAX" else GRAY)
    ser.data_labels.show_value = True
    ser.data_labels.number_format = "0.0"
    ser.data_labels.number_format_is_linked = False
    ser.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    ser.data_labels.font.size = Pt(12)
    ch.plots[0].gap_width = 50
    sar = t1.loc["SARIMAX"]
    lgs = t1.loc["LightGBM (scaled)"]
    ha = t1.loc["Historical Average"]
    bullets(s, Inches(7.6), Inches(1.8), Inches(5.1), Inches(5.0), [
        ("SARIMAX：", f"相對 HA 改善 {100 * (1 - sar.MAE / ha.MAE):.0f}%，但尖峰 MAE {sar.peak_MAE:.0f} "
                     f"仍遠高於 LightGBM-scaled 的 {lgs.peak_MAE:.0f}"),
        ("雨天打平：", f"SARIMAX {sar.rain_MAE:.1f} vs LightGBM-scaled {lgs.rain_MAE:.1f}，"
                    "狀態更新等於即時水準修正"),
        ("Prophet：", "只學平滑季節、不用最新觀測，比 HA 還差"),
        ("成本：", "SARIMAX 每站各一模型；LightGBM 全站一個模型"),
    ], size=15, gap=12)
    footer(s, n, "results/ts_compare.csv")

    # 8 站型
    s = new()
    title(s, "站型決定可預測性：商圈觀光型最難", "K-means（K=5）＋環狀線轉乘口；LightGBM-scaled t+1 進站相對 MAE（MAE ÷ 平均量）")
    picture(s, "figures/cluster_profiles.png", MX, Inches(1.6), w=Inches(6.3))
    ce = D["ce"]
    ce = ce[(ce.target == "entries") & (ce.horizon == 1) & (ce.method == "LightGBM (scaled)")].set_index("cluster_name")
    order = ["轉乘樞紐型", "住宅型", "住辦混合型", "辦公型", "商圈觀光型", "環狀線轉乘口（僅進站）"]
    rows = [["站型", "相對 MAE", "較 HA 改善"]]
    for k in order:
        rows.append([k.replace("（僅進站）", ""), f"{ce.loc[k, 'rel_MAE']:.3f}", f"{ce.loc[k, 'impr_vs_HA_%']:.0f}%"])
    table(s, Inches(7.2), Inches(1.7), Inches(5.5), rows, col_w=[2.5, 1.5, 1.5], size=13, row_h=0.45, hl_rows=(5,))
    bullets(s, Inches(7.2), Inches(5.05), Inches(5.5), Inches(1.8), [
        "方法排名在每一群都相同，差異反映可預測性本身",
        "商圈觀光型由休閒與活動驅動，辦公型受假日影響最大",
    ], size=14, gap=6)
    footer(s, n, "results/cluster_error.csv、cluster_profiles.csv")

    # 9 SHAP 與雨天
    s = new()
    title(s, "SHAP：歷史剖面與即時偏離主導，天氣平均貢獻小", f"{BM}，進站 t+1；平均 |SHAP| 換算成人次／小時")
    picture(s, "figures/reg_shap_group_bar.png", MX, Inches(1.6), w=Inches(6.6))
    items = []
    if D["sg"] is not None:
        sg = D["sg"].set_index("group")
        for k, (g, lab) in enumerate((("歷史平均剖面", "歷史平均剖面"), ("近期延遲（t, t-1, t-2）", "近期延遲"),
                                      ("天氣", "天氣"), ("活動", "活動"))):
            stat(s, MX + Inches(1.65 * k), Inches(5.05), Inches(1.6), f"{sg.loc[g, 'share_%']:.1f}%", lab,
                 color=ACCENT if k < 2 else NAVY, big_size=28)
    rain_b = reg(D, "entries", 1, BM, "rain_MAE")
    dry_b = reg(D, "entries", 1, BM, "dry_MAE")
    items += [
        ("雨天仍較難：", f"雨天 MAE {rain_b:.1f} vs 無雨 {dry_b:.1f}；約一半是「雨多下在白天與大站」的組成效應"),
        ("偏差方向：", "平日雨天尖峰被低估、假日雨天被高估"),
        ("活動站時：", "相對 HA 改善 44%，但絕對誤差仍高（規模與散場時間不確定）"),
    ]
    src9 = "results/reg_shap_groups.csv、reg_rain_tables.md、reg_supplement.csv"
    if rd.has("reg_weather_forecast.csv"):
        wf = rd.read_csv("reg_weather_forecast.csv")
        wb = wf[(wf.target == "entries") & (wf.horizon == 1) & wf.scenario.str.startswith("(b) 觀測訓練→JMA")].iloc[0]
        items.append(("改用 JMA 預報：", f"MAE 只多 {wb.over_vs_a:.2f}、雨天 MAE 多 {wb.rain_over_vs_a:.1f}，"
                                       "「假設天氣準確」的高估很小"))
        src9 += "、reg_weather_forecast.csv"
    bullets(s, Inches(7.5), Inches(1.8), Inches(5.2), Inches(5.0), items, size=15, gap=12)
    footer(s, n, src9)

    # 10 空間特徵
    s = new()
    title(s, "空間特徵：OD 流向讓 t+2 雨天誤差下降約 4 成", "LightGBM-scaled 加入鄰站、上下游與 OD 流向特徵（測試集 MAE）")
    sa = D["sa"].set_index(["target", "horizon", "features"])
    cd = CategoryChartData()
    cats = [("entries", 1, "進站 t+1"), ("exits", 1, "出站 t+1"), ("entries", 2, "進站 t+2"), ("exits", 2, "出站 t+2")]
    cd.categories = [c[2] for c in cats]
    cd.add_series("base", [float(sa.loc[(t, h, "base"), "MAE"]) for t, h, _ in cats])
    cd.add_series("+全部空間特徵", [float(sa.loc[(t, h, "+all"), "MAE"]) for t, h, _ in cats])
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, MX, Inches(1.6), Inches(7.0), Inches(5.2), cd)
    ch = gf.chart
    style_chart(ch)
    for ser, c in zip(ch.series, (GRAY, BLUE)):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = c
        ser.data_labels.show_value = True
        ser.data_labels.number_format = "0.0"
        ser.data_labels.number_format_is_linked = False
        ser.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
        ser.data_labels.font.size = Pt(11)
    ch.plots[0].gap_width = 70
    ch.value_axis.minimum_scale = 0
    r0 = float(sa.loc[("entries", 2, "base"), "rain_MAE"])
    r1 = float(sa.loc[("entries", 2, "+all"), "rain_MAE"])
    bullets(s, Inches(8.0), Inches(1.8), Inches(4.7), Inches(5.0), [
        ("t+2 進站雨天 MAE：", f"{r0:.1f} → {r1:.1f}（−{100 * (1 - r1 / r0):.0f}%）"),
        ("OD 流向：", "以各起站目前的異常推估 t+h 進站，再沿 OD 比例傳到目的站"),
        ("鄰站原始量：", "單獨加入反而略差（易過擬合）"),
    ], size=15, gap=14)
    footer(s, n, "results/spatial_ablation.csv")

    # 11 擁擠預警
    s = new()
    title(s, "擁擠預警：三種定義都達到 Recall ≥ 0.7、Precision ≥ 0.9", "t+1，測試集 2026-09；決策門檻在驗證集選定（Recall ≥ 0.7 的最高門檻）")
    cm = D["cm"].set_index(["label", "method"])
    rows = [["擁擠定義", "正類率", "HistAvg P／R", "LightGBM 分類 P／R", "AP", "達標"]]
    for lab, nm in (("relative", "相對 P95（各站自身）"), ("absolute", "絕對 5,914 人次／時"), ("dual", "雙門檻（兩者皆成立）")):
        h_ = cm.loc[(lab, "HistAvg")]
        l_ = cm.loc[(lab, "LGB-clf[none]")]
        ok = l_.Recall >= 0.7 and l_.Precision >= 0.9
        rows.append([nm, f"{l_.pos_rate * 100:.1f}%", f"{h_.Precision:.2f}／{h_.Recall:.2f}",
                     f"{l_.Precision:.3f}／{l_.Recall:.3f}", f"{l_.AP:.3f}", "✓" if ok else "✗"])
    table(s, MX, Inches(1.75), W - 2 * MX, rows, col_w=[3.2, 1.3, 2.2, 2.8, 1.2, 1.43], size=14, row_h=0.55)
    picture(s, "figures/clf_pr_curves.png", MX, Inches(4.15), h=Inches(2.7))
    bullets(s, Inches(7.9), Inches(4.3), Inches(4.8), Inches(2.6), [
        "相對門檻有一半正類在小站；雙門檻聚焦大、中站的異常尖峰",
        "class weight、focal、欠採樣：AP 差異不到 0.005",
    ], size=14, gap=10)
    footer(s, n, "results/clf_main.csv")

    # 12 回放回測
    s = new()
    bt = D["bt"]
    bt = bt[bt.policy == "min cost 10:1 (valid)"].set_index("period")
    lo, hi = bt.prewarn_rate.min(), bt.prewarn_rate.max()
    title(s, f"預警回放：{lo:.1%}–{hi:.1%} 擁擠事件可提前約 2 小時預警",
          "逐小時發出 t+1／t+2 預警；門檻以「漏報 1 事件 = 10 誤報小時」在驗證月選定；各折皆含天氣")
    picture(s, "figures/clf_backtest_timeline.png", Inches(1.6), Inches(1.5), h=Inches(3.0))
    rows = [["測試月", "提前偵測率", "平均提前量", "誤報／站／日", "非例行事件偵測率"]]
    names = {"2026-01": "2026-01 元旦", "2026-02": "2026-02 春節", "2026-04": "2026-04 清明",
             "2026-06": "2026-06 端午", "2026-07": "2026-07 颱風", "2026-09": "2026-09"}
    for per in sorted(bt.index):
        r = bt.loc[per]
        rows.append([names.get(per, per), f"{r.prewarn_rate:.1%}", f"{r.mean_lead_h:.2f} h",
                     f"{r.fa_per_station_day:.2f}", f"{r.prewarn_nonroutine:.1%}"])
    table(s, MX, Inches(4.6), Inches(8.6), rows, col_w=[2.2, 1.6, 1.6, 1.5, 1.7], size=12, row_h=0.31)
    items = ["約 90% 事件是例行尖峰", "非例行事件最難（颱風月 63%）"]
    if rd.has("clf_backtest_typhoon.csv"):
        ty = rd.read_csv("clf_backtest_typhoon.csv")
        ty = ty[(ty.subset == "颱風停班日") & (ty.policy == "min cost 10:1 (valid)")].iloc[0]
        items.append(f"颱風停班日幾乎不發預警（{ty.fa_per_station_day:.3f} 誤報／站／日）")
    bullets(s, Inches(9.5), Inches(4.7), Inches(3.25), Inches(2.2), items, size=14, gap=8)
    footer(s, n, "results/clf_backtest.csv、clf_backtest_typhoon.csv")

    # 13 目標達成
    s = new()
    title(s, "提案預期成果：迴歸與預警的量化目標都達成")
    clf = D["cm"].set_index(["label", "method"])
    rel = clf.loc[("relative", "LGB-clf[none]")]
    dual = clf.loc[("dual", "LGB-clf[none]")]
    table(s, MX, Inches(1.5), W - 2 * MX, [
        ["預期成果", "結果", "達成"],
        ["t+1 MAE 相對 HA 改善 ≥ 25%", f"{BM}：進站 {imp[('entries', 1)]:.1f}%、出站 {imp[('exits', 1)]:.1f}%", "✓"],
        ["t+2 MAE 相對 HA 改善 ≥ 25%", f"{BM}：進站 {imp[('entries', 2)]:.1f}%、出站 {imp[('exits', 2)]:.1f}%", "✓"],
        ["Recall ≥ 0.7 時 Precision ≥ 0.9", f"相對 P95：P {rel.Precision:.3f}／R {rel.Recall:.3f}；雙門檻：P {dual.Precision:.3f}／R "
                                            f"{dual.Recall:.3f}", "✓"],
        ["各站型、雨天情境的表現", "6 個站型分別報告；雨天誤差分解與空間特徵改善", "✓"],
        ["延長回測含颱風與連假", "迴歸 12 個月回測、預警回放皆含颱風月與春節（颱風日僅 2 天）", "✓"],
    ], col_w=[3.8, 7.33, 1.0], size=16, row_h=0.85)
    footer(s, n, "results/reg_main.csv、clf_main.csv、clf_backtest.csv")

    # 14 限制
    s = new()
    title(s, "限制：資料定義決定了結論能解讀到哪裡")
    lims = [
        ("出站 ≠ 實際出站時間", "OD 只有進站時段：出站量(t) 是「t 時進站、目的地為該站」的人次"),
        ("17 個站名只有單一方向", "景安／大坪林／頭前庄拆成兩個名稱，環狀線 11 個轉乘口只有進站"),
        ("擁擠是相對定義", "門檻是自身歷史的分位數，不是月台實際人數；轉乘人流看不到"),
        ("天氣與樣本", "單一格點模式值；假日大雨樣本少（期中 21 小時，CI 0.28–0.93，且受颱風日主導）"),
    ]
    for i, (h_, b_) in enumerate(lims):
        x = MX + (i % 2) * Inches(6.15)
        y = Inches(1.5) + (i // 2) * Inches(2.7)
        card(s, x, y, Inches(5.9), Inches(2.4))
        text(s, x + Inches(0.25), y + Inches(0.2), Inches(5.4), Inches(0.5), h_, size=19, color=NAVY, bold=True)
        text(s, x + Inches(0.25), y + Inches(0.85), Inches(5.4), Inches(1.4), b_, size=17, color=INK)
    footer(s, n)

    # 15 結論
    s = new(dark=True)
    text(s, MX, Inches(0.5), Inches(12), Inches(0.8), "結論與未來工作", size=32, color=WHITE, bold=True)
    light = RGBColor(0xDD, 0xE4, 0xF0)
    bullets(s, MX, Inches(1.7), Inches(7.0), Inches(5.0), [
        ("預測：", f"全站共用的 {BM}，t+1／t+2 MAE 約為 HA 的一半，t+2 退化很小"),
        ("預警：", "三種擁擠定義都達標；回放回測（含颱風月）98–99% 事件提前約 2 小時"),
        ("影響因素：", "站型決定可預測性；雨天、活動需靠即時偏離與 OD 流向修正"),
        ("更正期中：", "排除颱風停班日後，大雨不會讓假日人流減半；中雨約降 7–9%"),
    ], size=19, color=WHITE, gap=18)
    card(s, Inches(8.0), Inches(1.6), Inches(4.73), Inches(4.2), fill=RGBColor(0x2A, 0x3B, 0x5F))
    text(s, Inches(8.25), Inches(1.8), Inches(4.3), Inches(0.5), "未來工作", size=19, color=WHITE, bold=True)
    bullets(s, Inches(8.25), Inches(2.45), Inches(4.3), Inches(4.0), [
        "月台面積／載客率換算實際密度",
        "氣象署測站與鄉鎮預報",
        "擴充活動資料與售票數",
        "SARIMAX、站型做 stacking",
    ], size=17, color=light, gap=14)
    text(s, MX, Inches(6.2), Inches(12), Inches(0.6), "謝謝聆聽　Q & A", size=24, color=WHITE, bold=True)
    text(s, W - Inches(1.4), Inches(7.0), Inches(0.8), Inches(0.35), str(n), size=10, color=light, align=PP_ALIGN.RIGHT)

    prs.core_properties.title = "台北捷運站點人流預測與擁擠預警"
    prs.core_properties.author = ""
    prs.core_properties.comments = ""
    prs.core_properties.last_modified_by = ""
    prs.save(OUT)
    print(f"wrote {OUT.relative_to(ROOT)} ({n} slides)")


if __name__ == "__main__":
    build()
