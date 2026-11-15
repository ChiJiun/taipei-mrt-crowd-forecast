"""Workstream C-2: K-means station typology (transfer hub / office / residential / leisure ...).

Features (training period only, <= TRAIN_END), computed per *physical* station:
  - weekday 24h entry profile and exit profile (each normalised to proportions)
  - holiday / weekday daily-volume ratio
  - AM-peak (7-9h) and PM-peak (17-19h) entries / exits ratio (log)
  - volume level log(mean hourly entries+exits)
  - transfer station flag (data/external/station_lines.csv)

Data quirks handled here (see results/ts_cluster_summary.md):
  - The OD files split three transfer stations into an entry-only and an exit-only name
    (大坪林/G大坪林, 景安/O景安, 頭前庄/O頭前庄). They are merged into one physical station
    for clustering and both names get the same cluster.
  - Circular-line (Y) gates (中原, 中和, 十四張, 幸福, 新北產業園區, 景平, 板新, 橋和, 秀朗橋,
    新埔民生, Y板橋) only have entries recorded (riders entering the TRTC network from the Y line),
    so exit-based features are undefined. They are excluded from K-means and assigned the
    rule-based group 「環狀線轉乘口（僅進站）」.

Usage: python src/31_clustering.py [--k K]   (default K=5, see comment in main)
Outputs: results/cluster_assignments.csv, results/cluster_features.csv, results/cluster_k_selection.csv,
         results/cluster_profiles.csv, figures/cluster_profiles.png, figures/cluster_pca.png,
         figures/cluster_k_selection.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from config import EXTERNAL, FIGURES, RESULTS, TRAIN_END  # noqa: E402
from features import load_calendar, load_panel  # noqa: E402
from plotting import plt  # noqa: E402

Y_GROUP = "環狀線轉乘口（僅進站）"
SEED = 42


def station_lines() -> pd.DataFrame:
    return pd.read_csv(EXTERNAL / "station_lines.csv")


def physical_panel(panel: pd.DataFrame, sl: pd.DataFrame):
    """Return (physical-station panel for K-means, list of entry-only Y gates)."""
    m = panel.groupby("station", observed=True)[["entries", "exits"]].mean()
    entry_only = m.index[(m.exits < 1) & (m.entries >= 1)].tolist()
    exit_only = m.index[(m.entries < 1) & (m.exits >= 1)].tolist()
    phys = sl.set_index("station")["physical_station"]
    # an entry-only gate is "split" if its physical station also has an exit-only gate
    split_phys = {phys[s] for s in exit_only}
    y_gates = [s for s in entry_only if phys[s] not in split_phys]
    keep = panel[~panel.station.isin(y_gates)].copy()
    keep["pstation"] = keep.station.astype(str).map(lambda s: phys[s] if phys[s] in split_phys else s)
    pp = keep.groupby(["pstation", "ts"], observed=True)[["entries", "exits"]].sum().reset_index()
    return pp, y_gates


def build_features(pp: pd.DataFrame, sl: pd.DataFrame) -> pd.DataFrame:
    cal = load_calendar().set_index("date")
    pp = pp[pp.ts <= TRAIN_END].copy()
    pp["hol"] = pp.ts.dt.normalize().map(cal["is_holiday"]).fillna(False).astype(bool)
    pp["hour"] = pp.ts.dt.hour
    wd = pp[~pp.hol]
    feats = {}
    for col, tag in (("entries", "in"), ("exits", "out")):
        prof = wd.groupby(["pstation", "hour"])[col].mean().unstack("hour")
        prof = prof.div(prof.sum(axis=1), axis=0)
        prof.columns = [f"{tag}_h{h:02d}" for h in prof.columns]
        feats[tag] = prof
    daily = pp.assign(date=pp.ts.dt.normalize(), tot=pp.entries + pp.exits) \
        .groupby(["pstation", "date", "hol"])["tot"].sum().reset_index()
    dm = daily.groupby(["pstation", "hol"])["tot"].mean().unstack("hol")
    sc = pd.DataFrame(index=dm.index)
    sc["holiday_ratio"] = dm[True] / dm[False]

    def ratio(hours):
        g = wd[wd.hour.isin(hours)].groupby("pstation")[["entries", "exits"]].sum()
        return np.log((g.entries + 1) / (g.exits + 1))
    sc["am_log_in_out"] = ratio([7, 8, 9])
    sc["pm_log_in_out"] = ratio([17, 18, 19])
    sc["log_volume"] = np.log(pp.groupby("pstation")[["entries", "exits"]].mean().sum(axis=1))
    tr = sl.groupby("physical_station")["is_transfer"].max()
    tr = tr.reindex(sc.index).fillna(sl.set_index("station")["is_transfer"].reindex(sc.index)).fillna(0)
    sc["is_transfer"] = tr.astype(int)
    return pd.concat([feats["in"], feats["out"], sc], axis=1)


def design_matrix(F: pd.DataFrame) -> np.ndarray:
    """Standardise; weight blocks so each profile counts like 2 scalar features in total."""
    prof_cols = [c for c in F.columns if c.startswith(("in_h", "out_h"))]
    sc_cols = [c for c in F.columns if c not in prof_cols]
    Z = StandardScaler().fit_transform(F.values)
    w = np.ones(F.shape[1])
    for tag in ("in_h", "out_h"):
        idx = [i for i, c in enumerate(F.columns) if c.startswith(tag)]
        w[idx] = np.sqrt(2.0 / len(idx))
    assert len(sc_cols) == 5
    return Z * w


def name_clusters(F: pd.DataFrame, lab: np.ndarray) -> dict[int, str]:
    """Data-driven naming from centroid ranks (rules documented in results/ts_cluster_summary.md):
    hub = highest transfer share (+ volume); among the rest: residential = highest AM entries/exits,
    office = lowest AM entries/exits, leisure = highest holiday/weekday ratio, others = mixed."""
    c = F.assign(cl=lab).groupby("cl").mean(numeric_only=True)
    names = {}
    hub = (c.is_transfer + (c.log_volume - c.log_volume.mean()) / c.log_volume.std() * 0.25).idxmax()
    names[hub] = "轉乘樞紐型"
    rest = c.drop(index=hub)
    if len(rest):
        names[rest.am_log_in_out.idxmax()] = "住宅型"
        rest = rest.drop(index=rest.am_log_in_out.idxmax())
    if len(rest):
        names[rest.am_log_in_out.idxmin()] = "辦公型"
        rest = rest.drop(index=rest.am_log_in_out.idxmin())
    if len(rest):
        names[rest.holiday_ratio.idxmax()] = "商圈觀光型"
        rest = rest.drop(index=rest.holiday_ratio.idxmax())
    for i, k in enumerate(rest.sort_values("log_volume", ascending=False).index):
        names[k] = "住辦混合型" if len(rest) == 1 else f"住辦混合型{i + 1}"
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None)
    a = ap.parse_args()
    panel = load_panel()
    sl = station_lines()
    pp, y_gates = physical_panel(panel, sl)
    F = build_features(pp, sl)
    F.round(5).to_csv(RESULTS / "cluster_features.csv")
    X = design_matrix(F)

    rows = []
    for k in range(2, 11):
        km = KMeans(k, n_init=50, random_state=SEED).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_)})
    ks = pd.DataFrame(rows).round(4)
    ks.to_csv(RESULTS / "cluster_k_selection.csv", index=False)
    print(ks.to_string(index=False))
    # Silhouette is highest for K=2..4 (station types form a continuum, not well-separated blobs);
    # K=5 is the smallest K that separates the leisure/shopping type from the mixed type, i.e. the
    # four proposal types + one mixed group, and sits at the elbow (see summary md).
    k = a.k or 5
    km = KMeans(k, n_init=50, random_state=SEED).fit(X)
    lab = km.labels_
    names = name_clusters(F, lab)
    # stable cluster ids: order by mean log volume (descending)
    order = F.assign(cl=lab).groupby("cl").log_volume.mean().sort_values(ascending=False).index
    remap = {old: i for i, old in enumerate(order)}
    pst = pd.DataFrame({"pstation": F.index, "cluster": [remap[l] for l in lab],
                        "cluster_name": [names[l] for l in lab]})

    phys = sl.set_index("station")["physical_station"]
    m = panel.groupby("station", observed=True)[["entries", "exits"]].mean()
    out = []
    for s in panel.station.cat.categories:
        if s in y_gates:
            out.append({"station": s, "cluster": k, "cluster_name": Y_GROUP})
            continue
        key = s if s in pst.pstation.values else phys[s]
        r = pst.set_index("pstation").loc[key]
        out.append({"station": s, "cluster": int(r.cluster), "cluster_name": r.cluster_name})
    asg = pd.DataFrame(out)
    asg = asg.merge(sl[["station", "physical_station", "lines", "is_transfer"]], on="station")
    asg["mean_entries"] = asg.station.map(m.entries).round(1)
    asg["mean_exits"] = asg.station.map(m.exits).round(1)
    asg.sort_values(["cluster", "mean_entries"], ascending=[True, False]).to_csv(
        RESULTS / "cluster_assignments.csv", index=False)

    # cluster summary table
    Fc = F.assign(cluster=pst.set_index("pstation").cluster.reindex(F.index).values)
    summ = Fc.groupby("cluster")[["holiday_ratio", "am_log_in_out", "pm_log_in_out", "log_volume", "is_transfer"]].mean()
    summ["n_physical"] = Fc.groupby("cluster").size()
    summ["cluster_name"] = pst.drop_duplicates("cluster").set_index("cluster").cluster_name
    summ["am_in_out"] = np.exp(summ.am_log_in_out)
    summ["pm_in_out"] = np.exp(summ.pm_log_in_out)
    summ["mean_volume"] = np.exp(summ.log_volume)
    summ["examples"] = Fc.assign(v=Fc.log_volume).sort_values("v", ascending=False).groupby("cluster") \
        .apply(lambda d: "、".join(d.index[:5]))
    summ = summ[["cluster_name", "n_physical", "mean_volume", "holiday_ratio", "am_in_out", "pm_in_out",
                 "is_transfer", "examples"]].round(3)
    summ.to_csv(RESULTS / "cluster_profiles.csv")
    print(summ.to_string())
    print(asg.groupby("cluster_name").station.apply(lambda s: "、".join(s)).to_string())

    # ---- figures
    ncl = k
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    hrs = np.arange(24)
    for ax, tag, ttl in ((axes[0], "in_h", "平日進站曲線（占全日比例）"), (axes[1], "out_h", "平日出站曲線（占全日比例）")):
        cols = [c for c in F.columns if c.startswith(tag)]
        for c in range(ncl):
            sub = Fc[Fc.cluster == c][cols]
            mu = sub.mean().values
            ax.plot(hrs, mu, color=cmap(c), lw=2, label=f"{c}: {summ.loc[c, 'cluster_name']} (n={len(sub)})")
            ax.fill_between(hrs, sub.quantile(0.25).values, sub.quantile(0.75).values, color=cmap(c), alpha=0.12)
        ax.set_title(ttl)
        ax.set_xlabel("小時")
        ax.set_xticks(range(0, 24, 3))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("比例")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"各站型平均平日進／出站曲線（K={k}，陰影為群內 25–75%）")
    fig.savefig(FIGURES / "cluster_profiles.png", dpi=150)
    plt.close(fig)

    pca = PCA(2, random_state=SEED).fit(X)
    P = pca.transform(X)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    cl = Fc.cluster.values
    for c in range(ncl):
        mk = cl == c
        ax.scatter(P[mk, 0], P[mk, 1], s=18 + 6 * np.exp(F.log_volume.values[mk]) / 500, color=cmap(c), alpha=0.75,
                   label=f"{c}: {summ.loc[c, 'cluster_name']}", edgecolor="white", lw=0.5)
    top = F.log_volume.sort_values(ascending=False).index[:12].tolist()
    extreme = list(F.index[np.argsort(np.abs(P[:, 0]))[-6:]]) + list(F.index[np.argsort(np.abs(P[:, 1]))[-6:]])
    for s in dict.fromkeys(top + extreme):
        i = F.index.get_loc(s)
        ax.annotate(s, (P[i, 0], P[i, 1]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1（{ev[0]:.0%}）")
    ax.set_ylabel(f"PC2（{ev[1]:.0%}）")
    ax.set_title("站點分群 PCA 散佈圖（點大小∝量級）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(FIGURES / "cluster_pca.png", dpi=150)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(ks.k, ks.inertia, "o-", color="tab:blue")
    ax1.set_xlabel("K")
    ax1.set_ylabel("Inertia（elbow）", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(ks.k, ks.silhouette, "s--", color="tab:orange")
    ax2.set_ylabel("Silhouette", color="tab:orange")
    ax1.axvline(k, color="gray", ls=":")
    ax1.set_title(f"K 的選擇（選定 K={k}）")
    fig.savefig(FIGURES / "cluster_k_selection.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
