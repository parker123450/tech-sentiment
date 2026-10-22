"""2026-10-22 指标客观性审查（结果见同目录的 .md）。在仓库根目录运行：python3 docs/审查/audit_2026-10-22.py
一次性审查：指标里有多少主观成分、衡量得是否有效。不改项目代码，只读 engine 的计算结果。"""
import json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))   # 仓库根目录
sys.path.insert(0, ROOT)
import engine as E
import research as R

D = E.load()
M = E.market(D)
eps_rows, eps = E.episode_table(M)
c01 = lambda x: x.clip(0, 1)
out = {}
since = M.index >= "1995-06-01"
X = M[since]


def spear(a, b):
    m = a.notna() & b.notna()
    return float(a[m].rank().corr(b[m].rank()))


# ---------------- 1 价格回声：分数有多少只是"最近涨跌"的翻版
price = pd.DataFrame({"z20": X.z20, "ddv": X.ddv, "extv": X.extv})
f20 = np.log(X.c.shift(-20) / X.c)
f60 = np.log(X.c.shift(-60) / X.c)
echo = {}
for col in ("score", "g_fear", "g_sell", "g_hot"):
    y = X[col]
    m = y.notna() & price.notna().all(axis=1)
    A = np.c_[np.ones(m.sum()), price[m].to_numpy()]
    beta, *_ = np.linalg.lstsq(A, y[m].to_numpy(), rcond=None)
    r2 = 1 - np.var(y[m].to_numpy() - A @ beta) / np.var(y[m].to_numpy())
    echo[col] = {"corr_past20": float(y.corr(X.z20)), "corr_dd": float(y.corr(X.ddv)), "r2_price": float(r2),
                 "corr_f20": float(y.corr(f20)), "corr_f60": float(y.corr(f60))}
out["echo"] = echo


# ---------------- 2 主观刻度和权重：换一套"同样说得通"的选择，结果变多少
def items(M, k):
    """k: 每一项刻度的 (起点偏移, 跨度倍数)；返回三个温度和情绪分"""
    g = lambda name, a, span, x: c01((x - (a + k[name][0] * span)) / (span * k[name][1]))   # 起点挪动 k0 个跨度，跨度乘 k1
    near = c01((M.dd + 0.10) / 0.07)
    f = pd.DataFrame({
        "vix": g("vix", 15, 15, M.vix), "vixspike": g("vixspike", 1, 0.4, M.vix / M.vix50),
        "inv": g("inv", 0.85, 0.15, M.t3), "vxn": g("vxn", 18, 18, M.vxn), "vxnspike": g("vxnspike", 1, 0.4, M.vxn / M.vxn50),
        "pcjump": g("pcjump", 1, 0.3, M.pcjump), "credit": g("credit", 0, 20, M.baa20), "hyief": g("hyief", 0, 2, -M.hyief20)})
    grp = {"大盘": ["vix", "vixspike", "inv"], "科技": ["vxn", "vxnspike"], "成交": ["pcjump"], "信用": ["credit", "hyief"]}
    s = pd.DataFrame({"washout": g("washout", 0, 40, 50 - M.br_pct50), "delever": g("delever", 0, 40, 50 - M.bb_pct), "korea": g("korea", 1, 1, M.kvol)})
    h = pd.DataFrame({"ext": g("ext", 0, 1, M.extv), "divtop": M.r_divtop.astype(float).where(M.dd.notna()),
                      "shrink": M.r_shrink.astype(float).where(M.smh_relv.notna() | M.dd.notna()),
                      "narrow": near * np.fmax(g("narrow", 0, 20, 80 - M.br_pct50), g("narrow2", 0, 2, -M.rsp60)),
                      "frenzy": np.fmax(g("frenzy", 0, 35, 50 - M.pc_pct), g("frenzy2", 0, 40, M.bb_pct - 50)),
                      "spotvol": near * g("spotvol", 0, 0.2, M.vix / M.vixmin20 - 1)})
    return f, grp, s, h


NAMES = ["vix", "vixspike", "inv", "vxn", "vxnspike", "pcjump", "credit", "hyief", "washout", "delever", "korea", "ext", "narrow", "narrow2", "frenzy", "frenzy2", "spotvol"]
base_k = {n: (0.0, 1.0) for n in NAMES}


def score_from(M, k, weights="group", combine="mean"):
    f, grp, s, h = items(M, k)
    gf = (pd.concat([f[c].mean(axis=1) for c in grp.values()], axis=1).mean(axis=1) if weights == "group" else f.mean(axis=1)) * 100
    gs, gh = s.mean(axis=1) * 100, h.mean(axis=1) * 100
    panic = pd.concat([gf, gs], axis=1)
    p = panic.mean(axis=1) if combine == "mean" else panic.max(axis=1) if combine == "max" else gf if combine == "fear" else gs
    return gh - p


base = score_from(M, base_k)
assert np.allclose(base.dropna(), M.score.dropna().reindex(base.dropna().index), atol=1e-9), "复算和 engine 不一致"
band = lambda s: pd.cut(s, [-1e9, -60, -20, 20 - 1e-9, 60 - 1e-9, 1e9], labels=False)
rng = np.random.default_rng(11)
draws = []
for _ in range(300):
    k = {n: (rng.uniform(-0.15, 0.15), rng.uniform(0.7, 1.3)) for n in NAMES}
    sc = score_from(M, k)
    m = since & sc.notna() & base.notna()
    draws.append({"rho": spear(sc[m], base[m]), "band_diff": float((band(sc[m]) != band(base[m])).mean()), "today": float(sc.iloc[-1]),
                  "mad": float((sc[m] - base[m]).abs().median())})
dr = pd.DataFrame(draws)
out["scales"] = {"rho_min": dr.rho.min(), "rho_med": dr.rho.median(), "band_diff_med": dr.band_diff.median(), "band_diff_p90": dr.band_diff.quantile(0.9),
                 "today_lo": dr.today.quantile(0.05), "today_hi": dr.today.quantile(0.95), "mad_med": dr.mad.median()}
variants = {}
for name, kw in (("组内平均改成全部等权", dict(weights="all")), ("恐慌取两者较大", dict(combine="max")), ("只用定价恐慌", dict(combine="fear")), ("只用抛售恐慌", dict(combine="sell"))):
    sc = score_from(M, base_k, **kw)
    m = since & sc.notna() & base.notna()
    variants[name] = {"rho": spear(sc[m], base[m]), "band_diff": float((band(sc[m]) != band(base[m])).mean()), "today": float(sc.iloc[-1])}
out["variants"] = variants
bands_shift = {}
for d in (-10, 10):
    bb = lambda s: pd.cut(s, [-1e9, -60 + d, -20 + d, 20 + d - 1e-9, 60 + d - 1e-9, 1e9], labels=False)
    m = since & base.notna()
    bands_shift[d] = float((bb(base[m]) != band(base[m])).mean())
out["band_cut_shift"] = bands_shift

# ---------------- 3 不带主观权重的对照：所有原始指标的第一主成分（数据自己找出的"共同成分"）
raw = pd.DataFrame({"vix": M.vix, "vixspike": M.vix / M.vix50, "t3": M.t3, "vxn": M.vxn, "vxnspike": M.vxn / M.vxn50, "pcjump": M.pcjump,
                    "baa20": M.baa20, "hyief": -M.hyief20, "washout": -M.br_pct50, "korea": M.kvol, "ext": -M.extv, "near_hi": -M.dd, "rsp60": M.rsp60})
R0 = raw[since].dropna()
Z = (R0 - R0.mean()) / R0.std()
w, V = np.linalg.eigh(np.cov(Z.T))
pc1 = pd.Series(Z.to_numpy() @ V[:, -1], index=Z.index)
if pc1.corr(Z.vix) < 0:
    pc1 = -pc1
out["pca"] = {"start": str(Z.index[0].date()), "share_var": float(w[-1] / w.sum()), "corr_score": float(-pc1.corr(base.reindex(Z.index))),
              "rho_score": spear(-pc1, base.reindex(Z.index)),
              "loadings": {k: round(float(v), 2) for k, v in zip(Z.columns, V[:, -1] * np.sign(pc1.corr(Z.vix)))}}

# ---------------- 4 能不能分辨历次大跌的低点和之前的高点（AUC：随机挑一个低点附近的日子和一个其他日子，分数更低的概率）
def auc(score, pos):
    s, p = score[since].dropna(), pos[since].reindex(score[since].dropna().index).fillna(False)
    a, b = s[p].to_numpy(), s[~p].to_numpy()
    r = pd.Series(np.r_[a, b]).rank().to_numpy()
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def near(dates, w=5):
    m = pd.Series(False, index=M.index)
    pos = {t: i for i, t in enumerate(M.index.strftime("%Y-%m-%d"))}
    for d in dates:
        if d in pos:
            i = pos[d]
            m.iloc[max(0, i - w): i + w + 1] = True
    return m


lows = near([e["tr"] for e in eps_rows if not e.get("ongoing")])
highs = near([e["pk"] for e in eps_rows])
out["auc"] = {"low_score": 1 - auc(base, lows), "low_fear": auc(M.g_fear, lows), "low_sell": auc(M.g_sell, lows), "low_price_dd": 1 - auc(M.ddv, lows),
              "high_score": auc(base, highs), "high_hot": auc(M.g_hot, highs), "high_price_ext": auc(M.extv, highs), "high_price_dd": auc(M.dd, highs),
              "n_low": len([e for e in eps_rows if not e.get("ongoing")]), "n_high": len(eps_rows)}
# 低于 −60 的日子里，有多少真的在大跌低点前后 60 天内
near60 = near([e["tr"] for e in eps_rows if not e.get("ongoing")], 60)
ext = since & (base <= -60)
out["precision"] = {"days_le_-60": int(ext.sum()), "share_near_low60": float(near60[ext].mean()),
                    "share_all_days_near_low60": float(near60[since].mean())}

# ---------------- 5 情绪分各档之后的表现（同深度对照、两段、相邻 10 天只算一次）
pred = {}
for name, mask, expect in (("情绪分 ≤ −60", base <= -60, 1), ("情绪分 ≤ −20", base <= -20, 1), ("情绪分 ≥ 20", base >= 20, -1), ("情绪分 ≥ 60", base >= 60, -1)):
    st = E.event_stats(mask.fillna(False).to_numpy(), M.c, M.ddv, valid_from="1995-06-01")
    pred[name] = {"label": R.label(st, expect)[0], **{e: {k: st[e].get(k) for k in ("n", "f60", "ex60")} for e in ("pre", "post")}}
out["predict"] = pred

# ---------------- 6 "有历史支持"的规则：随便挑同样多、同样扎堆的日子，有多大概率也能过关（循环平移 400 次）
labels = json.load(open(os.path.join(ROOT, "research_results.json"), encoding="utf-8"))["labels"]
rules = {}
for k, s in E.SPEC.items():
    col = {"fear": "f_", "frag": "r_", "turn": "t_"}.get(s["group"])
    if col and col + k in M:
        rules[k] = (M[col + k].fillna(False).astype(bool).to_numpy(), s.get("expect", 1 if s["group"] in ("fear", "turn") else -1), s["name"])
for k, cb in E.COMBOS.items():
    rules[k] = (cb["f"](M).fillna(False).astype(bool).to_numpy(), cb["expect"], cb["name"])
perm = {}
n = len(M)
tested = [k for k in rules if k in labels]
for k in tested:
    mask, expect, nm = rules[k]
    if expect == 0:
        continue
    st = E.event_stats(mask, M.c, M.ddv, valid_from="1995-06-01")
    obs = R.label(st, expect)[0]
    if obs != "有历史支持":
        continue
    a = min(st["pre"]["ex60"] * expect, st["post"]["ex60"] * expect)
    hits = 0
    shifts = rng.integers(250, n - 250, 400)
    for sh in shifts:
        m2 = np.roll(mask, sh)
        s2 = E.event_stats(m2, M.c, M.ddv, valid_from="1995-06-01")
        if R.label(s2, expect)[0] == "有历史支持" and min(s2["pre"]["ex60"] * expect, s2["post"]["ex60"] * expect) >= a:
            hits += 1
    pass_any = 0
    for sh in shifts[:150]:
        s2 = E.event_stats(np.roll(mask, sh), M.c, M.ddv, valid_from="1995-06-01")
        pass_any += R.label(s2, expect)[0] == "有历史支持"
    perm[k] = {"name": nm, "n": [st["pre"]["n"], st["post"]["n"]], "ex60": [st["pre"]["ex60"], st["post"]["ex60"]],
               "p_as_strong": hits / 400, "p_pass_label": pass_any / 150}
out["perm"] = perm
out["n_tested"] = len([k for k in tested if rules[k][1] != 0])
out["n_supported"] = sum(1 for k in tested if labels[k] == "有历史支持")
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
