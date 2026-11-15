"""工作流 E 共用：讀取 results/ 底下的結果檔。

若檔案尚未合併到目前分支，可用環境變數 REPORT_REF（例如 origin/ws/B-regression）
指定一個 git ref，從該 ref 讀取同名檔案（只用於草稿階段；最終版一律以 main 上的檔案為準）。
"""
from __future__ import annotations

import io
import os
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
REF = os.environ.get("REPORT_REF", "").strip()


def source_of(name: str) -> str:
    """回傳結果檔的實際來源（'local' 或 git ref），找不到時回傳 ''。"""
    if (RESULTS / name).exists():
        return "local"
    if REF:
        r = subprocess.run(["git", "cat-file", "-e", f"{REF}:results/{name}"], cwd=ROOT,
                           capture_output=True)
        if r.returncode == 0:
            return REF
    return ""


def read_text(name: str) -> str:
    src = source_of(name)
    if src == "local":
        return (RESULTS / name).read_text(encoding="utf-8")
    if src:
        return subprocess.run(["git", "show", f"{src}:results/{name}"], cwd=ROOT, check=True,
                              capture_output=True).stdout.decode("utf-8")
    raise FileNotFoundError(f"results/{name}（可設定 REPORT_REF 從其他分支讀取）")


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(read_text(name)))


def has(name: str) -> bool:
    return bool(source_of(name))
