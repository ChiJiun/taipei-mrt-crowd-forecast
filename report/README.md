# 期末報告與簡報（工作流 E）

| 檔案 | 內容 |
|---|---|
| `final_report.md` | 期末報告原稿（唯一需要手動編輯的文字來源） |
| `final_report.docx` | 由 md 轉出的 Word 版（`src/50_build_docx.py`） |
| `slides.pptx`／`slides.pdf` | 期末簡報 16 頁（`src/50_build_slides.py`，數字直接讀 `results/`） |

組員欄保留「第＿組　組員：＿＿＿＿」的空白，請在 Word／PowerPoint 中自行填寫（或改 md 與 `src/50_build_slides.py` 第 1 頁後重新產生）。

## 重新產生

```bash
pip install -r requirements.txt python-docx python-pptx
python src/50_figures.py                 # figures/final_method_mae.png、final_goals.png、final_pipeline.png
python src/50_check_report_numbers.py    # 抽查報告數字與 results/*.csv 是否一致（不一致會回傳非 0）
python src/50_build_docx.py              # report/final_report.docx
python src/50_build_slides.py            # report/slides.pptx
soffice --headless --convert-to pdf --outdir report report/slides.pptx   # report/slides.pdf
```

檢查簡報排版（每頁轉 png）：

```bash
pdftoppm -png -r 80 report/slides.pdf /tmp/slide
```

## 修改報告時

- 文字只改 `final_report.md`，再跑 `50_build_docx.py`。支援的 Markdown：`#`／`##`／`###` 標題、段落、`- ` 項目、`1. ` 編號、管線表格、`![圖說](../figures/x.png){width=12cm}`、程式碼區塊與 `**粗體**`／`` `程式碼` ``。
- 圖說寫成「圖X　…」放在圖片語法的方括號內；表格說明寫成獨立一行「表X　…」放在表格正上方。
- 改了數字就同步更新 `src/50_check_report_numbers.py` 的對應項目，確保每個關鍵數字都能追溯到 `results/`。
- 字型：Word 與簡報都使用「Microsoft JhengHei」（微軟正黑體）；在沒有此字型的環境（如 LibreOffice／Linux）會自動替代。

## 從尚未合併的分支讀取結果（選用）

若之後某個工作流的結果還在分支上、尚未合併到 main，可設定 `REPORT_REF` 從該分支讀取同名檔案（只用於草稿；定稿一律以 main 為準）：

```bash
REPORT_REF=origin/<branch> python src/50_check_report_numbers.py
REPORT_REF=origin/<branch> python src/50_build_docx.py
REPORT_REF=origin/<branch> python src/50_build_slides.py
```

`50_check_report_numbers.py --strict` 會把「結果檔不存在而略過」也當成失敗，定稿前請用這個模式確認。
