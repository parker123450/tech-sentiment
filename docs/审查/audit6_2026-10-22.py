#!/usr/bin/env python3
"""第六轮讨论用：核对每个描述指标在历次顶底的敏感度、领先滞后、价格回声、反应对象、刻度和冗余。只读。"""
import sys, os, warnings
import numpy as np, pandas as pd
PROJ = "/Users/liangjiafu/Desktop/科技情绪全景"
sys.path.insert(0, PROJ)
os.chdir(PROJ)
warnings.filterwarnings("ignore")
import engine as E

D = E.load()
M = E.market(D)
c = M.c
eps = E.episodes(c.to_numpy())
done = [(pk, tr) for pk, tr, en in eps if en is not None]
peaks = [pk for pk, tr, en in eps]
troughs = [tr for pk, tr in done]
n = len(M)
idx = M.index
print(f"数据 {idx[0].date()} → {idx[-1].date()}，{n} 天；完成的底 {len(troughs)}，顶 {len(peaks)}")

r_sox = np.log(c).diff()
r_spx = np.log(D["^GSPC"].c).reindex(idx).diff() if "^GSPC" in D else None
beta = (r_sox.rolling(250).cov(r_spx) / r_spx.rolling(250).var())
r_sox_only = r_sox - beta * r_spx           # 半导体自己的那部分涨跌


def auc(x, label):
    """秩和 AUC：x 越大越像 label=1。"""
    s = pd.DataFrame({"x": x, "y": label}).dropna()
    if s.y.sum() < 5 or (1 - s.y).sum() < 5:
        return np.nan
    r = s.x.rank()
    n1, n0 = s.y.sum(), (1 - s.y).sum()
    return float((r[s.y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def window_label(points, back, fwd):
    lab = np.zeros(n)
    for p in points:
        lab[max(0, p - back): min(n, p + fwd + 1)] = 1
    return pd.Series(lab, index=idx)


def analyze(x, sign, points, back, fwd, lab_back, lab_fwd):
    """x: 原始序列；sign: +1 表示越高越像这种极端。返回一组指标。"""
    v = (x * sign).astype(float)
    pct = E.exp_pct(v).reindex(idx)              # 在自己全部历史里的分位（只用当天以前）
    lab = window_label(points, lab_back, lab_fwd)
    out = {"auc": auc(v, lab), "auc_pct": auc(pct, lab)}
    # 命中：顶/底窗口里分位到过 ≥ 90 的比例；底噪：任意一天的同长窗口里到过 ≥ 90 的比例
    hits, leads, wmax = [], [], []
    valid_pts = [p for p in points if np.isfinite(pct.iloc[max(0, p - back): p + fwd + 1]).any()]
    for p in valid_pts:
        w = pct.iloc[max(0, p - back): min(n, p + fwd + 1)]
        hits.append(bool(w.max() >= 90))
        wmax.append(w.max())
        ww = v.iloc[max(0, p - 60): min(n, p + 15 + 1)]
        if ww.notna().any():
            leads.append(int(idx.get_loc(ww.idxmax()) - p))
    out["n_pts"] = len(valid_pts)
    out["hit"] = float(np.mean(hits)) if hits else np.nan
    out["wmax_med"] = float(np.median(wmax)) if wmax else np.nan
    hi = (pct >= 90).astype(float)
    base = hi.rolling(back + fwd + 1, min_periods=1).max().shift(-fwd)
    out["base"] = float(base.mean())
    out["lead_med"] = float(np.median(leads)) if leads else np.nan
    out["lead_q1"], out["lead_q3"] = (float(np.percentile(leads, 25)), float(np.percentile(leads, 75))) if leads else (np.nan, np.nan)
    # 独立触发次数（分位 ≥ 90，相邻 10 天只算一次），其中落在任何顶/底 ±60 天以外的比例 = 虚警
    ev = E.dedup((pct >= 90).fillna(False).to_numpy())
    near = window_label(points, 60, 60).to_numpy()
    out["n_ev"] = len(ev)
    out["false"] = float(np.mean([near[i] == 0 for i in ev])) if ev else np.nan
    # 价格回声：和 20 日涨跌、回撤、离 200 日线的相关；三者一起能解释的比例
    P = pd.DataFrame({"v": v, "r20": M.r20, "ddv": M.ddv, "extv": M.extv}).dropna()
    if len(P) > 500:
        out["corr_r20"] = float(P.v.corr(P.r20)); out["corr_dd"] = float(P.v.corr(P.ddv))
        X = np.c_[np.ones(len(P)), P[["r20", "ddv", "extv"]].to_numpy()]
        b, *_ = np.linalg.lstsq(X, P.v.to_numpy(), rcond=None)
        res = P.v.to_numpy() - X @ b
        out["r2_px"] = float(1 - res.var() / P.v.var())
    else:
        out["corr_r20"] = out["corr_dd"] = out["r2_px"] = np.nan
    # 反应对象：日变化和半导体 / 标普 / 半导体独有涨跌的相关
    dv = v.diff()
    out["re_sox"] = float(dv.corr(r_sox)); out["re_spx"] = float(dv.corr(r_spx)); out["re_soxonly"] = float(dv.corr(r_sox_only))
    return out


# ---------------------------------------------------------------- 指标清单（原始值，方向）
nhnl = (M.br_nh - M.br_nl)
RAW = {
    # 期权定价
    "VIX": (M.vix, "期权"), "VIX÷50日均": (M.vix / M.vix50, "期权"), "VIX÷VIX3M": (M.t3, "期权"), "VIX9D÷VIX": (M.t9, "期权"),
    "VXN": (M.vxn, "期权"), "VXN÷50日均": (M.vxn / M.vxn50, "期权"), "VVIX": (M.vvix, "期权"), "SKEW": (M["skew"], "期权"),
    # 成交与仓位
    "股票认沽/认购10日": (M.pc10, "仓位"), "认沽/认购÷近一年中位": (M.pcjump, "仓位"), "指数认沽/认购10日": (M.pcidx10, "仓位"),
    "SOXL÷SOXS成交额分位": (M.bb_pct, "仓位"), "SMH相对成交量": (M.smh_relv, "量能"), "SMH涨跌日量比": (M.smh_udv, "量能"),
    # 信用
    "Baa 20日走阔": (M.baa20, "信用"), "HYG÷IEF 20日": (M.hyief20, "信用"),
    # 参与度与跨市场
    "成分股50日线上方%": (M.br_pct50, "参与度"), "成分股200日线上方%": (M.br_pct200, "参与度"), "新高减新低": (nhnl, "参与度"),
    "等权÷市值60日": (M.rsp60, "参与度"), "KOSPI波动倍数": (M.kvol, "跨市场"),
    # 价格与速度（对照用）
    "回撤÷波动": (M.ddv, "价格"), "离200日线÷波动": (M.extv, "价格"), "20日Z": (M.z20, "价格"), "10日加速度": (M.acc, "价格"), "RSI": (M.rsi, "价格"),
    # 合成
    "定价恐慌": (M.g_fear, "合成"), "抛售恐慌": (M.g_sell, "合成"), "过热温度": (M.g_hot, "合成"), "情绪分": (M.score, "合成"), "超出价格的恐慌": (M.fear_x, "合成"),
}
# 底部：越高越像底（sign=+1）或越低越像底（−1）；顶部同理
BOT_SIGN = {"VIX": 1, "VIX÷50日均": 1, "VIX÷VIX3M": 1, "VIX9D÷VIX": 1, "VXN": 1, "VXN÷50日均": 1, "VVIX": 1, "SKEW": 1,
            "股票认沽/认购10日": 1, "认沽/认购÷近一年中位": 1, "指数认沽/认购10日": 1, "SOXL÷SOXS成交额分位": -1, "SMH相对成交量": 1, "SMH涨跌日量比": -1,
            "Baa 20日走阔": 1, "HYG÷IEF 20日": -1, "成分股50日线上方%": -1, "成分股200日线上方%": -1, "新高减新低": -1, "等权÷市值60日": -1, "KOSPI波动倍数": 1,
            "回撤÷波动": -1, "离200日线÷波动": -1, "20日Z": -1, "10日加速度": -1, "RSI": -1,
            "定价恐慌": 1, "抛售恐慌": 1, "过热温度": -1, "情绪分": -1, "超出价格的恐慌": 1}
TOP_SIGN = {"VIX": -1, "VIX÷50日均": -1, "VIX÷VIX3M": -1, "VIX9D÷VIX": -1, "VXN": -1, "VXN÷50日均": -1, "VVIX": 1, "SKEW": 1,
            "股票认沽/认购10日": -1, "认沽/认购÷近一年中位": -1, "指数认沽/认购10日": -1, "SOXL÷SOXS成交额分位": 1, "SMH相对成交量": 1, "SMH涨跌日量比": 1,
            "Baa 20日走阔": -1, "HYG÷IEF 20日": 1, "成分股50日线上方%": 1, "成分股200日线上方%": 1, "新高减新低": 1, "等权÷市值60日": -1, "KOSPI波动倍数": -1,
            "回撤÷波动": 1, "离200日线÷波动": 1, "20日Z": 1, "10日加速度": 1, "RSI": 1,
            "定价恐慌": -1, "抛售恐慌": -1, "过热温度": 1, "情绪分": 1, "超出价格的恐慌": -1}

rows_b, rows_t = [], []
for k, (x, grp) in RAW.items():
    if x is None or x.notna().sum() < 500:
        continue
    b = analyze(x, BOT_SIGN[k], troughs, 20, 5, 5, 5)
    t = analyze(x, TOP_SIGN[k], peaks, 40, 5, 10, 5)
    rows_b.append({"指标": k, "组": grp, **b}); rows_t.append({"指标": k, "组": grp, **t})
B, T = pd.DataFrame(rows_b).set_index("指标"), pd.DataFrame(rows_t).set_index("指标")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.float_format", lambda v: f"{v:.2f}")


def show(df, title):
    print(f"\n### {title}")
    cols = ["组", "n_pts", "auc", "auc_pct", "hit", "base", "wmax_med", "lead_med", "lead_q1", "lead_q3", "n_ev", "false", "corr_r20", "corr_dd", "r2_px", "re_sox", "re_spx", "re_soxonly"]
    print(df[cols].to_string())


show(B, "底部（低点前 20 天到后 5 天；AUC 的标签是低点 ±5 天）")
show(T, "顶部（高点前 40 天到后 5 天；AUC 的标签是高点前 10 天到后 5 天）")

# ---------------------------------------------------------------- 刻度：登记的 0→100 刻度在历史分布里的位置
print("\n### 刻度核对（各项 0–1）：全历史里到 1（饱和）的比例、≥ 0.5 的比例；底部/顶部窗口里的最大值中位数；平时的中位数")
items = E.gauge_items(M)
sc = []
for key, v in items.items():
    v = pd.Series(v, index=idx).astype(float)
    kind = {"gf": "底", "gs": "底", "gh": "顶"}[key[:2]]
    pts, back, fwd = (troughs, 20, 5) if kind == "底" else (peaks, 40, 5)
    wm = [v.iloc[max(0, p - back): p + fwd + 1].max() for p in pts if v.iloc[max(0, p - back): p + fwd + 1].notna().any()]
    sc.append({"项": key, "类": kind, "有数据天": int(v.notna().sum()), "饱和%": v.ge(1).mean() * 100, "≥0.5%": v.ge(0.5).mean() * 100,
               "平时中位": v.median(), "顶底窗口最大中位": np.median(wm) if wm else np.nan, "窗口到1的比例": np.mean([w >= 1 for w in wm]) if wm else np.nan, "n": len(wm)})
print(pd.DataFrame(sc).set_index("项").to_string())

# ---------------------------------------------------------------- 冗余：恐惧侧指标（分位）的秩相关和主成分
print("\n### 冗余：底部侧指标的秩相关（2007 年以来，认沽/认购有数据后）")
fear_keys = ["VIX", "VIX÷50日均", "VIX÷VIX3M", "VXN", "VXN÷50日均", "VVIX", "SKEW", "股票认沽/认购10日", "认沽/认购÷近一年中位", "SOXL÷SOXS成交额分位",
             "Baa 20日走阔", "HYG÷IEF 20日", "成分股50日线上方%", "新高减新低", "KOSPI波动倍数", "SMH相对成交量", "RSI", "20日Z", "回撤÷波动"]
F = pd.DataFrame({k: RAW[k][0] * BOT_SIGN[k] for k in fear_keys}).loc["2007-11-01":].dropna()
C = F.rank().corr()
print(f"样本 {len(F)} 天")
print(C.round(2).to_string())
Z = (F.rank() - F.rank().mean()) / F.rank().std()
w, V = np.linalg.eigh(np.cov(Z.T))
w, V = w[::-1], V[:, ::-1]
print("\n主成分解释比例：", np.round(w / w.sum(), 2)[:5])
pc1 = pd.Series(V[:, 0] * np.sign(V[:, 0].sum()), index=fear_keys).round(2)
pc2 = pd.Series(V[:, 1], index=fear_keys).round(2)
print("第一主成分载荷：\n", pc1.sort_values(ascending=False).to_string())
print("第二主成分载荷：\n", pc2.sort_values(ascending=False).to_string())

# 简单聚类：相关 ≥ 0.6 的归一堆
print("\n相关 ≥ 0.6 的配对：")
for i, a in enumerate(fear_keys):
    for b_ in fear_keys[i + 1:]:
        if C.loc[a, b_] >= 0.6:
            print(f"  {a} ~ {b_}: {C.loc[a, b_]:.2f}")

# ---------------------------------------------------------------- 跨指数共通性：纳指、标普自己的大跌，哪些市场层面的指标同样在底部到极端
print("\n### 跨指数共通性：纳指（跌 25%）、标普（跌 15%）自己的底部，各指标在窗口里分位 ≥ 90 的比例")
common = {}
for sym, th in (("^NDX", -0.25), ("^GSPC", -0.15), ("^SOX", -0.25)):
    px = D[sym].c.reindex(idx).ffill()
    F2 = E.px_features(D[sym])
    e2 = E.episodes(px.to_numpy(), th)
    tr2 = [tr for pk, tr, en in e2 if en is not None]
    pk2 = [pk for pk, tr, en in e2]
    rowb, rowt = {}, {}
    cand = {"VIX": M.vix, "VIX÷50日均": M.vix / M.vix50, "VIX÷VIX3M": M.t3, "VXN÷50日均": M.vxn / M.vxn50, "VVIX": M.vvix, "SKEW": M["skew"],
            "股票认沽/认购10日": M.pc10, "认沽/认购÷近一年中位": M.pcjump, "Baa 20日走阔": M.baa20, "HYG÷IEF 20日": M.hyief20,
            "等权÷市值60日": M.rsp60, "成分股50日线上方%": M.br_pct50, "新高减新低": nhnl, "SOXL÷SOXS成交额分位": M.bb_pct}
    own = {"自己RSI": F2.rsi.reindex(idx), "自己20日Z": F2.z20.reindex(idx), "自己离200日线÷波动": F2.extv.reindex(idx), "自己相对成交量": F2.relv.reindex(idx)}
    for k, x in {**cand, **own}.items():
        sb = BOT_SIGN.get(k, {"自己RSI": -1, "自己20日Z": -1, "自己离200日线÷波动": -1, "自己相对成交量": 1}.get(k))
        st = TOP_SIGN.get(k, {"自己RSI": 1, "自己20日Z": 1, "自己离200日线÷波动": 1, "自己相对成交量": 1}.get(k))
        pb = E.exp_pct(x * sb).reindex(idx); pt = E.exp_pct(x * st).reindex(idx)
        hb = [pb.iloc[max(0, p - 20): p + 6].max() >= 90 for p in tr2 if pb.iloc[max(0, p - 20): p + 6].notna().any()]
        ht = [pt.iloc[max(0, p - 40): p + 6].max() >= 90 for p in pk2 if pt.iloc[max(0, p - 40): p + 6].notna().any()]
        rowb[k] = f"{np.mean(hb):.0%}({len(hb)})" if hb else "—"; rowt[k] = f"{np.mean(ht):.0%}({len(ht)})" if ht else "—"
    common[f"{sym} 底"] = rowb; common[f"{sym} 顶"] = rowt
    print(f"  {sym}: 完成的底 {len(tr2)}，顶 {len(pk2)}")
print(pd.DataFrame(common).to_string())

# ---------------------------------------------------------------- 反应速度：各恐惧项从平时（≤ 0.2）到亮（≥ 0.8）用几天；亮了以后持续几天
print("\n### 反应速度（0–1 的项）：从 ≤ 0.2 升到 ≥ 0.8 的中位天数；≥ 0.8 持续的中位天数；次数")
sp = []
for key, v in items.items():
    v = pd.Series(v, index=idx).astype(float).dropna()
    if v.empty:
        continue
    on = (v >= 0.8).to_numpy(); low = (v <= 0.2).to_numpy()
    rises, runs = [], []
    i = 0
    while i < len(v):
        if on[i] and (i == 0 or not on[i - 1]):
            j = i
            while j > 0 and not low[j]:
                j -= 1
            rises.append(i - j)
            k2 = i
            while k2 < len(v) and on[k2]:
                k2 += 1
            runs.append(k2 - i)
            i = k2
        else:
            i += 1
    sp.append({"项": key, "次数": len(rises), "升起用天(中位)": np.median(rises) if rises else np.nan, "持续天(中位)": np.median(runs) if runs else np.nan})
print(pd.DataFrame(sp).set_index("项").to_string())

B.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit6_bottom.csv"))
T.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit6_top.csv"))
