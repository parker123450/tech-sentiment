#!/usr/bin/env python3
"""第六轮讨论用：流动性（美联储资产负债表、国库账户、逆回购、金融条件指数）、利率、美元、债券波动，有没有前瞻性。只读。
下载缓存在本目录 liq_cache.csv。"""
import sys, os, warnings, json
import numpy as np, pandas as pd
PROJ = "/Users/liangjiafu/Desktop/科技情绪全景"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ); os.chdir(PROJ); warnings.filterwarnings("ignore")
import engine as E, build_sentiment as bs

CACHE = os.path.join(HERE, "liq_cache.csv")
FREDS = {"WALCL": "美联储总资产(百万)", "WTREGEN": "国库账户TGA(十亿)", "RRPONTSYD": "隔夜逆回购(十亿)", "WRESBAL": "银行准备金(十亿)",
         "DGS10": "10年国债", "DGS2": "2年国债", "DGS30": "30年国债", "DFII10": "10年实际利率", "DGS3MO": "3个月国债", "NFCI": "芝加哥金融条件指数"}
YAHOOS = {"^MOVE": "债券波动MOVE", "DX-Y.NYB": "美元指数"}
if os.path.exists(CACHE):
    raw = pd.read_csv(CACHE, parse_dates=["date"])
else:
    rows = []
    for k in FREDS:
        try:
            r = bs.fred(k); rows += [[k, d, v] for d, v, *_ in r]; print(f"  {k} {len(r)} 行 {r[0][0]}→{r[-1][0]}")
        except Exception as e:
            print("  !", k, e)
    for k in YAHOOS:
        res = bs.yahoo(k)
        if res:
            r = bs.rows_from(res); rows += [[k, d, c] for d, c, *_ in r]; print(f"  {k} {len(r)} 行 {r[0][0]}→{r[-1][0]}")
    raw = pd.DataFrame(rows, columns=["sym", "date", "v"]); raw["date"] = pd.to_datetime(raw.date); raw.to_csv(CACHE, index=False)
S = {k: g.set_index("date").v.sort_index() for k, g in raw.groupby("sym")}

D = E.load(); M = E.market(D); idx = M.index; c = M.c
f20, f60, mdd = E.forward(c)
frv = E.fwd_rv(c)
big = E.big_drop(c, M.lv)
eps = E.episodes(c.to_numpy())


def daily(s, lag):
    """对齐到半导体交易日：先 ffill 再往后挪 lag 个交易日（发布滞后，防止用到未来）。"""
    return s.reindex(idx.union(s.index)).ffill().reindex(idx).shift(lag)


# 净流动性（十亿美元）= 总资产 − 国库账户 − 逆回购；周三数据周四公布，按 2 个交易日滞后
walcl = S["WALCL"] / 1000
tga = S["WTREGEN"]; rrp = S["RRPONTSYD"]
rrp_w = rrp.reindex(walcl.index, method="ffill").fillna(0)
nl = (walcl - tga.reindex(walcl.index, method="ffill") - rrp_w).dropna()
NL = daily(nl, 2); WAL = daily(walcl, 2); TGA = daily(tga, 2); RRP = daily(rrp, 1); RES = daily(S["WRESBAL"], 2)
X = {
    "净流动性 4 周变化(占总资产%)": (NL - NL.shift(20)) / WAL * 100,
    "净流动性 13 周变化(占总资产%)": (NL - NL.shift(65)) / WAL * 100,
    "总资产 13 周变化%": WAL.pct_change(65) * 100,
    "国库账户 4 周变化(十亿)": TGA - TGA.shift(20),
    "逆回购 4 周变化(十亿)": RRP - RRP.shift(20),
    "准备金 13 周变化%": RES.pct_change(65) * 100,
    "金融条件 NFCI 水平": daily(S["NFCI"], 5),
    "NFCI 4 周变化": daily(S["NFCI"], 5) - daily(S["NFCI"], 5).shift(20),
    "10 年国债 20 日变化(bp)": (daily(S["DGS10"], 1) - daily(S["DGS10"], 1).shift(20)) * 100,
    "10 年国债 60 日变化(bp)": (daily(S["DGS10"], 1) - daily(S["DGS10"], 1).shift(60)) * 100,
    "30 年国债 20 日变化(bp)": (daily(S["DGS30"], 1) - daily(S["DGS30"], 1).shift(20)) * 100,
    "实际利率 20 日变化(bp)": (daily(S["DFII10"], 1) - daily(S["DFII10"], 1).shift(20)) * 100,
    "期限利差 10 年−2 年(bp)": (daily(S["DGS10"], 1) - daily(S["DGS2"], 1)) * 100,
    "期限利差 10 年−3 月(bp)": (daily(S["DGS10"], 1) - daily(S["DGS3MO"], 1)) * 100,
    "债券波动 MOVE 水平": daily(S["^MOVE"], 1),
    "MOVE 20 日变化%": daily(S["^MOVE"], 1).pct_change(20) * 100,
    "美元指数 20 日变化%": daily(S["DX-Y.NYB"], 1).pct_change(20) * 100,
    # 对照：现有指标
    "对照 VIX": M.vix, "对照 VIX÷50 日均": M.vix / M.vix50, "对照 认沽/认购 10 日": M.pc10, "对照 Baa 20 日走阔": M.baa20,
    "对照 成分股 50 日线上方%": M.br_pct50, "对照 回撤÷波动": M.ddv, "对照 20 日 Z": M.z20,
}
era = np.where(idx < pd.Timestamp(E.SPLIT), "前", "后")
rows = []
for name, x in X.items():
    x = pd.Series(np.asarray(x, float), index=idx)
    r = {"指标": name, "起": x.first_valid_index().strftime("%Y-%m") if x.notna().any() else "—"}
    for e in ("前", "后"):
        m = (era == e) & x.notna()
        df = pd.DataFrame({"x": x[m], "f20": f20[m], "f60": f60[m], "rv": frv[m], "big": big[m].astype(float)})
        r[f"n{e}"] = int(m.sum())
        for k in ("f20", "f60", "rv", "big"):
            d = df[["x", k]].dropna()
            r[f"{k}{e}"] = d.x.corr(d[k], method="spearman") if len(d) > 300 else np.nan
    # 极端十分位（只用当天以前的分位）之后 60 天比同样回撤深度的日子（项目的标准口径）
    pct = E.exp_pct(x).reindex(idx)
    for tag, mask in (("低", pct <= 10), ("高", pct >= 90)):
        st = E.event_stats(mask.fillna(False), c, M.ddv)
        r[f"{tag}10%:前"] = f"{st['pre']['n']}次 {st['pre'].get('ex60') if st['pre']['n'] else '—'}"
        r[f"{tag}10%:后"] = f"{st['post']['n']}次 {st['post'].get('ex60') if st['post']['n'] else '—'}"
    rows.append(r)
R = pd.DataFrame(rows).set_index("指标")
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)


def fmt(v):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:+.2f}" if isinstance(v, float) else str(v))


print("\n### 和之后的关系（秩相关；f20/f60 = 之后 20/60 天涨跌，rv = 之后 21 天实际波动，big = 60 天内再跌半个年化波动；前 = 2015 前，后 = 2015 后）")
cols = ["起", "n前", "f20前", "f60前", "rv前", "big前", "n后", "f20后", "f60后", "rv后", "big后"]
print(R[cols].applymap(lambda v: fmt(v) if not isinstance(v, (int, np.integer, str)) else v).to_string())
print("\n### 极端十分位之后 60 天和同样回撤深度的日子比（次数 超额）")
print(R[["低10%:前", "低10%:后", "高10%:前", "高10%:后"]].to_string())

# ---------------------------------------------------------------- 顶底附近：流动性和利率在顶/底前 120 天到后 20 天里何时到极端
print("\n### 顶底附近的时间（相对顶/底的天数；· = 窗口里没到过 90% 分位）")
n = len(idx)
peaks = [pk for pk, tr, en in eps]; troughs = [tr for pk, tr, en in eps if en is not None]
def seq(points, items, back, fwd, title):
    out = []
    for p in points:
        r = {"日期": idx[p].strftime("%Y-%m")}
        for name, (x, s) in items.items():
            v = pd.Series(np.asarray(x, float) * s, index=idx); pct = E.exp_pct(v).reindex(idx)
            w = pct.iloc[max(0, p - back): min(n, p + fwd + 1)]
            r[name] = (f"{idx.get_loc(w.idxmax()) - p:+d}" + ("" if w.max() >= 90 else "·")) if w.notna().any() else "—"
        out.append(r)
    print(f"\n{title}"); print(pd.DataFrame(out).set_index("日期").to_string())
top_items = {"净流动性13周↓": (X["净流动性 13 周变化(占总资产%)"], -1), "总资产13周↓": (X["总资产 13 周变化%"], -1), "TGA4周↑": (X["国库账户 4 周变化(十亿)"], 1),
             "NFCI↑": (X["NFCI 4 周变化"], 1), "10年60日↑": (X["10 年国债 60 日变化(bp)"], 1), "实际利率20日↑": (X["实际利率 20 日变化(bp)"], 1),
             "MOVE↑": (X["MOVE 20 日变化%"], 1), "美元20日↑": (X["美元指数 20 日变化%"], 1), "对照 认沽/认购低": (M.pc10, -1), "对照 广度50日": (M.br_pct50, 1)}
seq(peaks, top_items, 120, 20, "顶部（高点前 120 天到后 20 天），看流动性收紧、利率上行是否先于顶")
bot_items = {"净流动性13周↑": (X["净流动性 13 周变化(占总资产%)"], 1), "总资产13周↑": (X["总资产 13 周变化%"], 1), "TGA4周↓": (X["国库账户 4 周变化(十亿)"], -1),
             "NFCI↓": (X["NFCI 4 周变化"], -1), "10年60日↓": (X["10 年国债 60 日变化(bp)"], -1), "实际利率20日↓": (X["实际利率 20 日变化(bp)"], -1),
             "MOVE峰": (X["债券波动 MOVE 水平"], 1), "美元20日↓": (X["美元指数 20 日变化%"], -1), "对照 VIX": (M.vix, 1), "对照 广度50日": (M.br_pct50, -1)}
seq(troughs, bot_items, 120, 20, "底部（低点前 120 天到后 20 天），看流动性转松、利率下行、债券波动见顶是否先于底")

# 水平 vs 变化的提醒：净流动性水平和半导体价格水平的相关（都是趋势，会虚高）
both = pd.DataFrame({"nl": NL, "px": c}).dropna()
print(f"\n提醒：净流动性水平和半导体指数水平的相关 {both.nl.corr(both.px):.2f}（两条都在涨，这种相关没有意义）；"
      f"13 周变化对 13 周变化 {both.nl.pct_change(65).corr(both.px.pct_change(65)):.2f}（同期），"
      f"13 周变化对之后 60 天 {pd.Series(X['净流动性 13 周变化(占总资产%)']).corr(pd.Series(f60, index=idx)):.2f}")
R.to_csv(os.path.join(HERE, "liq_test_out.csv"))
