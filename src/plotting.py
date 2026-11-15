"""Matplotlib setup with a CJK font so Chinese labels render."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

_CANDIDATES = ["Noto Sans CJK TC", "Noto Sans CJK JP", "Noto Sans TC", "Microsoft JhengHei",
               "PingFang TC", "Heiti TC", "WenQuanYi Zen Hei", "Arial Unicode MS"]


def setup():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for c in _CANDIDATES:
        if c in names:
            plt.rcParams["font.sans-serif"] = [c, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"
    return plt


plt = setup()
