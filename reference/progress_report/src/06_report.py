"""產生期末專題進度報告 docx（主題三）"""
import json, os
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(f"{ROOT}/results/model_results.json")); C = R["crowd"]
d = Document()
for s in d.sections:
    s.top_margin = s.bottom_margin = Cm(1.4); s.left_margin = s.right_margin = Cm(1.7)
st = d.styles["Normal"]; st.font.name = "Microsoft JhengHei"; st.font.size = Pt(9.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft JhengHei")
st.paragraph_format.space_after = Pt(2)

def H(t, lvl=1):
    p = d.add_paragraph(); r = p.add_run(t); r.bold = True; r.font.size = Pt(13 if lvl == 1 else 11)
    p.paragraph_format.space_before = Pt(6); return p
def P(t): return d.add_paragraph(t)
def B(t): return d.add_paragraph(t, style="List Bullet")
def T(rows, widths=None, size=8.5):
    t = d.add_table(rows=len(rows), cols=len(rows[0])); t.style = "Light Grid Accent 1"
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i, j); c.text = ""; r = c.paragraphs[0].add_run(str(v)); r.font.size = Pt(size); r.bold = (i == 0)
            c.paragraphs[0].paragraph_format.space_after = Pt(0)
    return t
def cap(t):
    p = d.add_paragraph(t); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in p.runs: r.font.size = Pt(8.5)

p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("資料科學導論　期末專題進度報告"); r.font.size = Pt(11)
p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("台北捷運站點人流預測與擁擠預警"); r.bold = True; r.font.size = Pt(16)
p = d.add_paragraph("第＿組　組員：＿＿＿＿、＿＿＿＿、＿＿＿＿　　2026 年 10 月 23 日"); p.alignment = WD_ALIGN_PARAGRAPH.CENTER

H("一、簡介")
P("背景與動機：台北捷運平日日均進站約 230 萬人次，但人流高度集中於少數轉乘站與通勤尖峰（本資料中台北車站平均每小時進出站合計約 1.26 萬人次，為第二名西門的 2.2 倍）。遇到豪雨、連假或大型活動，月台壅塞會降低通勤品質並增加安全隱患；目前疏運多依人員經驗與即時監看，屬事後反應。若能提前一至兩小時預測各站進出站量並對「是否擁擠」預警，營運單位即可提前調整班距與人流管制。")
P("專案目標：(1) 預測各站未來 1–2 小時的進站量與出站量（迴歸）；(2) 預測站點下一小時是否達到擁擠門檻（二元分類）；(3) 分析雨天、假日等因素對人流的影響，並評估預警結果的實用價值。方法上以時間序列基準與梯度提升模型（LightGBM）為主，評估指標為 MAE、RMSE 與 Precision／Recall／AUC。")

H("二、資料集")
T([["資料", "來源", "內容與粒度", "用途"],
   ["每日分時各站 OD 流量", "臺北大眾捷運公司（政府資料開放平臺 dataset 128506）", "日期、小時、進站、出站、人次；月檔約 300MB", "聚合成各站每小時進／出站量（目標變數）"],
   ["逐時天氣", "Open-Meteo 歷史再分析（臺北市單一格點）", "氣溫、降雨量、相對濕度", "天氣特徵"],
   ["辦公日曆", "TaiwanCalendar（依人事行政總處日曆整理）", "每日是否放假、補班", "假日特徵"],
   ["大型活動時程", "尚未取得", "跨年、演唱會、大巨蛋等", "後續加入"]])
P("")
P("處理方式：從 2025 年 10 月至 2026 年 9 月共 12 個月的 OD 檔，逐月下載後即時聚合為「站點 × 小時」的進站量與出站量並刪除原始檔（程式見 src/01）。排除觀察期間內新開通或停用的站點後保留 122 站，補齊缺漏時段為 0，共 1,068,720 筆站點小時資料（src/02）。目標為預測 t+1 小時的進／出站量。")

H("三、目前進度")
H("3.1 探索性資料分析", 2)
d.add_picture(f"{ROOT}/figures/fig1_hourly_profile.png", width=Cm(7.5)); d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
cap("圖一　全網平均每小時進站量（2025/10–2026/9）")
B("日別與時段：平日呈現 8 時與 17–18 時的雙尖峰；假日沒有早高峰，且日均進站 176 萬人次，約為平日（230 萬）的 76%。")
B("降雨：同日別、同小時下，假日降雨 1–5 mm 時全網進站量為無雨的 86%、超過 5 mm 時降至 55%（僅 21 小時樣本）；平日因通勤剛性，降雨 > 5 mm 仍有 94%。顯示雨天對非通勤人流的抑制遠大於通勤人流，是非線性且與日別交互的影響。")
H("3.2 基準方法與初步模型", 2)
P("以「在 t 時刻預測 t+1 小時」為第一個任務，依時間切分：2025/10–2026/7 為訓練集、2026/8 為驗證集（early stopping）、2026/9 為測試集（87,840 筆站點小時，平均每站每小時進站 729 人）。比較方法：Last Value（以目前小時作預測）、Seasonal Naive（上週同時段）、Historical Average（站點 × 平／假日 × 小時平均）、LightGBM（Poisson 損失；特徵含時段、星期、假日、站點、1/2/3/24/168 小時延遲、24/168 小時滾動平均、氣溫與降雨）。")
P("第一版 LightGBM 的整體 MAE 雖優於基準，但在「平日尖峰（7、8、17、18 時）」反而輸給 Seasonal Naive（進站 131.2 vs 106.9）；原因推測是各站量級差異極大（台北車站與小站相差百倍以上），Poisson 損失被大站與非尖峰時段主導。因此我們改為將目標除以該站近一週平均量級（並以量級加權），再乘回原尺度，同時加入歷史平均作為特徵（表中 LightGBM-scaled）。")
rows = [["方法", "進站 MAE", "進站 RMSE", "進站 尖峰MAE", "出站 MAE", "出站 RMSE", "出站 尖峰MAE"]]
for k in ["Last Value", "Seasonal Naive", "Historical Average", "LightGBM", "LightGBM (scaled)"]:
    e, x = R["entry"][k], R["exit"][k]
    rows.append([k, f"{e['MAE']:.1f}", f"{e['RMSE']:.1f}", f"{e['MAE_peak']:.1f}", f"{x['MAE']:.1f}", f"{x['RMSE']:.1f}", f"{x['MAE_peak']:.1f}"])
cap("表一　t+1 小時進／出站量預測結果（測試集：2026 年 9 月，單位：人次）"); T(rows)
e, h = R["entry"], R["entry"]["Historical Average"]
P("")
P(f"Historical Average 是最強的非機器學習基準，說明人流具穩定的日週週期；LightGBM-scaled 相對它將進站 MAE 降低 {100*(1-e['LightGBM (scaled)']['MAE']/h['MAE']):.1f}%、RMSE 降低 {100*(1-e['LightGBM (scaled)']['RMSE']/h['RMSE']):.1f}%，平日尖峰 MAE 降低 {100*(1-e['LightGBM (scaled)']['MAE_peak']/h['MAE_peak']):.1f}%，出站結果相近。特徵重要度（第一版 gain）以一週前同時段占 50.7%最高，其次為一天前同時段（20.9%）與目標小時（9.3%），降雨僅是次要特徵。雨天（降雨 ≥ 1 mm）時段的 MAE 為 107，約為無雨時段（47）的 2.3 倍，顯示雨天是模型最難預測的情境，也是後續改進的重點。")
d.add_picture(f"{ROOT}/figures/fig2_taipei_main_week.png", width=Cm(9)); d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
cap("圖二　台北車站進站量預測 vs 實際（測試集 9/14–9/20）")
H("3.3 擁擠預警（分類）初步結果", 2)
P(f"擁擠定義：以各站訓練期「進站＋出站」總量的 P95 作為該站擁擠門檻，下一小時總量達門檻即為正類（測試集正類率 {100*C['positive_rate_test']:.1f}%）。在驗證集上選擇使 Recall ≥ 0.7 的最高決策門檻，結果如下：")
T([["方法", "AUC", "AP", "Precision", "Recall", "F1"]] +
  [[k, f"{C[k]['AUC']:.3f}", f"{C[k]['AP']:.3f}", f"{C[k]['P']:.3f}", f"{C[k]['R']:.3f}", f"{C[k]['F1']:.3f}"] for k in ["HistAvg", "LightGBM"]])
cap("表二　擁擠預警（測試集：2026 年 9 月）")
P("LightGBM 在 Recall 0.74 時 Precision 達 0.96，高於歷史平均基準（Recall 0.58、Precision 0.95）。需要注意：這裡的擁擠門檻是「相對自身歷史」的分位數，並非月台實際人數或容量，因此尚不能直接等同現場壅塞。")
H("3.4 目前限制", 2)
B("天氣為臺北市單一格點再分析值，且直接使用目標時段實際天氣（等同預報完全準確）；之後改用中央氣象署各測站觀測並區分觀測與預報。")
B("OD 資料只有進出站量，沒有月台候車人數或列車載客率，擁擠門檻為相對定義；大型活動資料尚未蒐集，測試期（單月）也未涵蓋颱風或跨年。")
B("目前只做 t+1 小時，尚未做 t+2、未使用站間（空間）特徵與時序模型。")

H("四、未來計畫")
T([["時程", "工作內容"],
   ["第 1–2 週", "蒐集大型活動時程（大巨蛋、南港展覽館、跨年等）與氣象署測站資料，加入特徵；預測範圍延伸至 t+2；以 SHAP 量化雨天、假日、活動的影響，並分析雨天誤差偏高的原因"],
   ["第 3 週", "加入 SARIMA／Prophet 作時間序列對照；依站點類型（轉乘站、辦公型、住宅型、觀光型，以 K-means 分群）分別比較誤差"],
   ["第 4 週", "加入空間特徵（鄰近站與同線上下游站前一小時流量、OD 流向）；改以相對／絕對雙門檻定義擁擠，並處理類別不平衡"],
   ["第 5 週", "回測評估：逐日回放測試期，統計預警提前量、漏報（Recall）與誤報成本；以分位數預測給出尖峰人流區間；延長測試期至多個月以含颱風與連假"],
   ["第 6 週", "彙整結果、撰寫期末報告與簡報"]], size=9)
P("")
P("預期成果：t+1、t+2 兩個範圍上，LightGBM 相對 Historical Average 的 MAE 改善維持在 25% 以上；擁擠預警在 Recall ≥ 0.7 下 Precision 不低於 0.9，並報告各站型與雨天情境的表現。")
P("程式與資料處理流程皆置於 topic3_mrt/（src/01–06 依序執行，結果存於 results/、圖存於 figures/）。")
d.save(f"{ROOT}/期末專題進度報告_主題三_捷運人流預測.docx")
