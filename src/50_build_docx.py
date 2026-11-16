"""工作流 E：把 report/final_report.md 轉成 report/final_report.docx（python-docx）。

格式比照進度報告（reference/progress_report/src/06_report.py）：微軟正黑體 9.5pt、窄邊界、
標題置中、「第＿組　組員：＿＿＿＿」空白、表格用 Light Grid Accent 1、圖下／表上置中說明。

支援的 Markdown 子集：# / ## / ### 標題、段落、- 項目、1. 編號、| 表格 |、
![圖說](路徑){width=12cm}、``` 程式碼區塊、> 引言、**粗體** 與 `程式碼`。
草稿說明區塊（<!-- DRAFT-NOTE --> … <!-- /DRAFT-NOTE -->）會以灰色斜體保留。

用法：python src/50_build_docx.py   （圖檔不存在時，若設定 REPORT_REF 會從該分支取出）
"""
from __future__ import annotations

import importlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

sys.path.insert(0, str(Path(__file__).parent))
rd = importlib.import_module("50_report_data")
ROOT = rd.ROOT
SRC = ROOT / "report" / "final_report.md"
OUT = ROOT / "report" / "final_report.docx"
FONT = "Microsoft JhengHei"
TMP = Path(tempfile.mkdtemp(prefix="report_fig_"))


def set_font(run, size=None, bold=None, italic=None, color=None, mono=False):
    name = "Consolas" if mono else FONT
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), FONT)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def add_inline(p, text, size=None, bold=False, italic=False, color=None):
    """**粗體** 與 `code` 拆成多個 run。"""
    for tok in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
        if not tok:
            continue
        if tok.startswith("**"):
            set_font(p.add_run(tok[2:-2]), size, True, italic, color)
        elif tok.startswith("`"):
            set_font(p.add_run(tok[1:-1]), size, bold, italic, color or "3A3A3A", mono=True)
        else:
            set_font(p.add_run(tok), size, bold, italic, color)


def resolve_image(path: str) -> Path | None:
    p = (SRC.parent / path).resolve()
    if p.exists():
        return p
    rel = p.relative_to(ROOT)
    if rd.REF:
        out = TMP / p.name
        r = subprocess.run(["git", "show", f"{rd.REF}:{rel.as_posix()}"], cwd=ROOT, capture_output=True)
        if r.returncode == 0:
            out.write_bytes(r.stdout)
            return out
    return None


def build():
    d = Document()
    for s in d.sections:
        s.top_margin = s.bottom_margin = Cm(1.6)
        s.left_margin = s.right_margin = Cm(1.8)
    st = d.styles["Normal"]
    st.font.name = FONT
    st.font.size = Pt(9.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    st.paragraph_format.space_after = Pt(3)
    st.paragraph_format.line_spacing = 1.15

    def caption(text):
        p = d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(6)
        add_inline(p, text, size=8.5, bold=True)

    def table(rows):
        hdr = [c.strip() for c in rows[0].strip().strip("|").split("|")]
        aligns = [c.strip() for c in rows[1].strip().strip("|").split("|")]
        body = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows[2:]]
        t = d.add_table(rows=1 + len(body), cols=len(hdr))
        t.style = "Light Grid Accent 1"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        # 欄寬依內容顯示長度分配（中文字算 2），避免短欄被平均分配擠到換行
        def disp(v):
            v = re.sub(r"\*\*|`", "", v)
            return sum(2 if ord(ch) > 0x2E7F else 1 for ch in v)
        need = [min(max(max(disp(r[j]) for r in body) if body else 0, disp(hdr[j]) * 0.7, 7) + 2, 46)
                for j in range(len(hdr))]
        total = min(17.4, max(sum(need) * 0.22, 9))
        widths = [Cm(total * n / sum(need)) for n in need]
        for j, col in enumerate(t.columns):
            col.width = widths[j]
        for row in t.rows:
            for j, c in enumerate(row.cells):
                c.width = widths[j]
        for i, row in enumerate([hdr] + body):
            for j, v in enumerate(row):
                c = t.cell(i, j)
                c.text = ""
                para = c.paragraphs[0]
                para.paragraph_format.space_after = Pt(0)
                if i > 0 and aligns[j].endswith(":") and not aligns[j].startswith(":"):
                    para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                elif aligns[j].startswith(":") and aligns[j].endswith(":"):
                    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                add_inline(para, v, size=8.5, bold=(i == 0))
                # 小表格不跨頁：除最後一列外都與下一段同頁
                if len(body) <= 16 and i < len(body):
                    para.paragraph_format.keep_with_next = True
        d.add_paragraph().paragraph_format.space_after = Pt(2)

    lines = SRC.read_text(encoding="utf-8").splitlines()
    i = 0
    first = True
    in_note = False
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s == "<!-- DRAFT-NOTE -->":
            in_note = True
            i += 1
            continue
        if s == "<!-- /DRAFT-NOTE -->":
            in_note = False
            i += 1
            continue
        if not s or s.startswith("<!--"):
            i += 1
            continue
        if first:  # 課名行
            p = d.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_inline(p, s, size=11)
            first = False
            i += 1
            continue
        if s.startswith("# "):
            p = d.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_inline(p, s[2:], size=16, bold=True)
        elif s.startswith("第＿組"):
            p = d.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(8)
            add_inline(p, s)
        elif s.startswith("## "):
            p = d.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.keep_with_next = True
            add_inline(p, s[3:], size=13, bold=True)
        elif s.startswith("### "):
            p = d.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.keep_with_next = True
            add_inline(p, s[4:], size=11, bold=True)
        elif s.startswith(">"):
            p = d.add_paragraph()
            add_inline(p, s.lstrip("> "), size=8.5, italic=True, color="6D6B64" if in_note else None)
        elif s.startswith("```"):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                p = d.add_paragraph()
                p.paragraph_format.space_after = Pt(0)
                set_font(p.add_run(lines[i]), 8, mono=True)
                i += 1
        elif s.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)(\{width=([\d.]+)cm\})?", s)
            cap_text, path, width = m.group(1), m.group(2), float(m.group(4) or 14)
            img = resolve_image(path)
            if img:
                d.add_picture(str(img), width=Cm(min(width, 17)))
                d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                d.paragraphs[-1].paragraph_format.keep_with_next = True
            else:
                p = d.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                add_inline(p, f"［圖檔待補：{path}］", italic=True, color="B00000")
            caption(cap_text)
        elif re.match(r"^表[一二三四五六七八九十]+　", s):
            p = d.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            add_inline(p, s, size=8.5, bold=True)
        elif s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            table(rows)
            continue
        elif s.startswith("- "):
            p = d.add_paragraph(style="List Bullet")
            add_inline(p, s[2:])
        elif re.match(r"^\d+\. ", s):
            # 不用 List Number：python-docx 的編號會跨清單延續
            p = d.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.6)
            p.paragraph_format.first_line_indent = Cm(-0.6)
            num, rest = s.split(" ", 1)
            add_inline(p, f"{num} {rest}")
        else:
            p = d.add_paragraph()
            add_inline(p, s)
        i += 1
    d.core_properties.title = "台北捷運站點人流預測與擁擠預警"
    d.core_properties.author = ""
    d.core_properties.comments = ""
    d.core_properties.last_modified_by = ""
    d.save(OUT)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    build()
