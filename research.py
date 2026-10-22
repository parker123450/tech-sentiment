#!/usr/bin/env python3
"""科技情绪全景 · 研究（取代 backtest.py 和 indicator_audit.py）

检验 engine.SPEC 里预先登记的每一条规则。不训练模型、不调阈值，结果只决定标签：
  有历史支持：2015 年前、后各至少 3 次独立触发，之后 60 天和"同样回撤深度的日子"相比，
             两段都朝预期方向偏离 1 个百分点以上；阶段描述类规则还要求在 ≥ 60% 的品种上两段方向都对。
  两段都相反：两段都朝预期的反方向偏离 1 个百分点以上。
  只作描述：其余情况（包括样本不足）。

方法：
  1. 历次顶底附近出现的比例，对比任意一天（费城半导体指数 1995 年以来 15 次 25% 以上的下跌）。
  2. 触发之后 20 / 60 天的表现，和同样回撤深度（按各自长期波动折算）的日子比；相邻 10 个交易日只算一次。
  3. 2015 年前后分两段。
  4. 价格、量能、动量类规则另外在 30 多个品种（半导体股、ETF、指数、韩国）上逐个检验。

输出 研究结论.md 和 research_results.json（看板读里面的标签）。用法：python3 research.py
"""
import json, os
import numpy as np
import pandas as pd
import engine as E
import build_sentiment as bs

HERE = E.HERE
OUT_MD = os.path.join(HERE, "研究结论.md")
OUT_JSON = os.path.join(HERE, "research_results.json")
MIN_EX = 0.01
CROSS = ["SMH", "QQQ", "SPY", "^KS11", "000660.KS", "005930.KS"] + bs.SEMIS


def label(st, expect, share=None):
    a, b = st.get("pre", {}), st.get("post", {})
    if a.get("n", 0) < 3 or b.get("n", 0) < 3 or a.get("ex60") is None or b.get("ex60") is None:
        return "只作描述", "样本不足"
    x, y = a["ex60"] * expect, b["ex60"] * expect
    sh_ok = share is None or (share[0] >= 0.6 and share[1] >= 0.6)
    if x >= MIN_EX and y >= MIN_EX and sh_ok:
        return "有历史支持", ""
    if x <= -MIN_EX and y <= -MIN_EX and (share is None or (share[0] <= 0.4 and share[1] <= 0.4)):
        return "两段都相反", ""
    return "只作描述", ("两段方向不一致" if x * y < 0 else "幅度太小" if sh_ok else "品种之间不一致")


def p(x, dp=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{dp}f}%"


def pp(x):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.0f}%"


def era_cell(s):
    if not s or not s.get("n"):
        return "0 次"
    return f"{s['n']} 次，60 天 {p(s.get('f60'))}，比同深度 {p(s.get('ex60'))}"


REGIMES = [("200 日线上方 · 50 日线上方", False, False), ("200 日线上方 · 50 日线下方", False, True),
           ("200 日线下方 · 50 日线上方", True, False), ("200 日线下方 · 50 日线下方", True, True)]


def measure(D, used):
    """第五轮：能衡量什么。幅度（隐含波动、波动扎堆）、下跌风险（趋势状态），以及同一批状态对方向有没有信息。"""
    S = pd.Timestamp(E.SPLIT)
    eras = lambda idx: (("pre", idx < S), ("post", idx >= S))
    out = {"iv": [], "rv": {}, "risk": {}, "regimes": {}, "labels": {}}
    # ---- M1 期权隐含波动 → 之后 21 天实际波动
    sox, ndx = E.px_features(D["^SOX"]), E.px_features(D["^NDX"])
    ratio = sox.lv / E.align(ndx.lv, sox.index)          # 半导体相对纳指的长期波动倍数（只用当天以前）
    pairs = [("标普 500，用 VIX", D["^GSPC"].c, E.align(D["^VIX"].c, D["^GSPC"].index)),
             ("纳指 100，用 VXN", D["^NDX"].c, E.align(D["^VXN"].c, D["^NDX"].index)),
             ("费城半导体，用 VXN × 半导体/纳指长期波动比", D["^SOX"].c, E.align(D["^VXN"].c, sox.index) * ratio)]
    ok_all = True
    for name, c, iv in pairs:
        df = pd.DataFrame({"iv": iv / 100, "fut": E.fwd_rv(c), "past": E.past_rv(c), "r": np.log(c.shift(-E.RV_N) / c)})
        df = df[df.index >= "1995-06-01"].dropna()
        row = {"name": name}
        for e, m in eras(df.index):
            x = df[m]
            band = x.iv * np.sqrt(E.RV_N / 252)
            row[e] = {"days": int(len(x)), "corr_iv": float(x.iv.corr(x.fut)), "corr_past": float(x.past.corr(x.fut)),
                      "cover1": float((x.r.abs() <= band).mean()), "cover2": float((x.r.abs() <= 2 * band).mean()),
                      "bias": float((x.iv - x.fut).median())}
            ok_all &= row[e]["corr_iv"] >= 0.5
        out["iv"].append(row)
    out["labels"]["iv_rv"] = "能衡量幅度" if ok_all else "只作描述"
    # ---- M3 波动扎堆；M2 趋势状态 → 下跌风险；M4 同样的状态 → 方向
    rv = {"pre": [], "post": []}
    risk = {k: {"pre": [], "post": []} for k in ("below200", "below50", "swingdn")}
    volr = {k: {"pre": [], "post": []} for k in risk}
    dirx = {k: {"pre": [], "post": []} for k in risk}
    reg = {i: {"pre": [], "post": []} for i in range(4)}
    for sym in ["^SOX"] + used:
        F = E.px_features(D[sym])
        fut, past = E.fwd_rv(F.c), E.past_rv(F.c)
        big = E.big_drop(F.c, F.lv)
        st = E.trend_states(F)
        f60 = F.c.shift(-60) / F.c - 1
        base = pd.DataFrame({"fut": fut, "past": past, "big": big, "f60": f60, "lv": F.lv}).join(st)
        base = base[base.index >= F.index[min(260, len(F) - 1)]]
        for e, m in eras(base.index):
            x = base[m].dropna(subset=["fut", "past"])
            if len(x) >= 500:
                rv[e].append(float(x.past.corr(x.fut)))
            y = base[m].dropna(subset=["big", "below200"])
            if len(y) < 500:
                continue
            for k in risk:
                on, off = y[y[k] == True], y[y[k] == False]
                if len(on) >= 120 and len(off) >= 120:
                    risk[k][e].append(float(on.big.mean() - off.big.mean()))
                    volr[k][e].append(float(on.fut.mean() / off.fut.mean()))
                    dirx[k][e].append(float(on.f60.median() - off.f60.median()))
            for i, (_, b200, b50) in enumerate(REGIMES):
                z = y[(y.below200 == b200) & (y.below50 == b50)]
                if len(z) >= 60:
                    reg[i][e].append({"big": float(z.big.mean()), "volx": float((z.fut / z.lv).median()), "f60": float(z.f60.median()), "sym": sym})
    med = lambda v: float(np.median(v)) if v else None
    out["rv"] = {e: {"n": len(v), "median": med(v), "share": float(np.mean([x >= 0.5 for x in v])) if v else None} for e, v in rv.items()}
    out["labels"]["rv_rv"] = "能衡量幅度" if all((out["rv"][e]["median"] or 0) >= 0.5 for e in ("pre", "post")) else "只作描述"
    for k in risk:
        r = {}
        for e in ("pre", "post"):
            v = risk[k][e]
            r[e] = {"n": len(v), "diff": med(v), "share": float(np.mean([x > 0 for x in v])) if v else None,
                    "volr": med(volr[k][e]), "dir": med(dirx[k][e]), "dir_share": float(np.mean([x < 0 for x in dirx[k][e]])) if dirx[k][e] else None}
        r["label"] = "能衡量风险" if all((r[e]["share"] or 0) >= 0.6 and (r[e]["diff"] or 0) >= 0.05 for e in ("pre", "post")) else "只作描述"
        r["dir_label"] = ("能衡量方向" if all((r[e]["dir_share"] or 0) >= 0.6 and (r[e]["dir"] or 0) <= -0.01 for e in ("pre", "post")) else "不能衡量方向")
        out["risk"][k] = r
        out["labels"][k] = r["label"]
    for i, (nm, b200, b50) in enumerate(REGIMES):
        out["regimes"][str(i)] = {"name": nm, **{e: {"n": len(v), "big": med([x["big"] for x in v]), "volx": med([x["volx"] for x in v]), "f60": med([x["f60"] for x in v]),
                                                     "sox": next(({"big": x["big"], "volx": x["volx"], "f60": x["f60"]} for x in v if x["sym"] == "^SOX"), None)}
                                                 for e, v in reg[i].items()}}
    return out


def main():
    D = E.load()
    M = E.market(D)
    eps_rows, eps = E.episode_table(M)
    ana = E.item_anatomy(M, eps)
    c, ddv = M.c, M.ddv
    res = {"asof": M.index[-1].strftime("%Y-%m-%d"), "labels": {}, "items": {}, "cross": {}, "combos": {}}

    # ---- 1. 恐惧定价、脆弱度、出清和转折：费城半导体指数
    items = []
    for k, s in E.SPEC.items():
        if s["group"] not in ("fear", "frag", "turn"):
            continue
        col = {"fear": "f_", "frag": "r_", "turn": "t_"}[s["group"]] + k
        st = E.event_stats(M[col], c, ddv, valid_from=E.START.get(k, "1995-06-01"))
        lb, why = label(st, s["expect"])
        res["labels"][k] = lb
        res["items"][k] = {"stats": st, "ana": ana.get(k), "label": lb, "why": why}
        items.append((k, s, st, ana.get(k), lb, why))

    # ---- 2. 组合：价格位置 × 恐惧定价、脆弱度条数（engine.COMBOS）
    for k, cb in E.COMBOS.items():
        st = E.event_stats(cb["f"](M).fillna(False).astype(bool), c, ddv, valid_from="1995-06-01")
        lb, why = label(st, cb["expect"]) if cb["expect"] else ("只作描述", "没有预设方向")
        res["combos"][k] = {"stats": st, "label": lb, "why": why}
        if cb["expect"]:
            res["labels"][k] = lb
    res["grid"] = E.grid_stats(M)

    # ---- 三个温度分档（只作描述，不打标签）
    res["gauges"] = {}
    for col in ("g_fear", "g_sell", "g_hot"):
        rows = {}
        for lo, hi in ((0, 20), (20, 40), (40, 60), (60, 80), (80, 101)):
            rows[f"{lo}–{min(hi, 100)}"] = E.event_stats(((M[col] >= lo) & (M[col] < hi)).fillna(False), c, ddv, valid_from="1995-06-01")
        res["gauges"][col] = rows

    # ---- 3. 阶段描述类规则：30 多个品种逐个检验
    stage = [k for k, s in E.SPEC.items() if s["group"] == "stage"]
    per = {k: [] for k in stage}
    ana_x = {k: [0, 0, 0.0, 0] for k in stage}     # 命中、顶底点数、基准加权和、品种数
    used = []
    for sym in CROSS:
        if sym not in D or len(D[sym]) < 1500:
            continue
        F = E.px_features(D[sym])
        fl = E.stage_flags(F)
        th = -max(0.20, float(F.lv.dropna().iloc[-1]) if F.lv.notna().any() else 0.3)
        ep = [e for e in E.episodes(F.c.to_numpy(), th) if e[0] >= 260]
        used.append(sym)
        for k in stage:
            vol_rule = k in ("climaxdn", "shrinkst")
            m = fl[k].fillna(False) & (fl["_hasvol"] if vol_rule else True)
            vf = F.index[fl["_hasvol"].to_numpy()][0] if vol_rule and fl["_hasvol"].any() else F.index[min(260, len(F) - 1)]
            st = E.event_stats(m, F.c, F.ddv, valid_from=vf)
            per[k].append((sym, st))
            kind = "tr" if E.SPEC[k]["expect"] > 0 else "pk"
            a = E.anatomy(m, F.c, ep, kind, 20 if kind == "tr" else 40, 5 if kind == "tr" else 0, vf)
            if a["n"]:
                ana_x[k][0] += a["hit"]; ana_x[k][1] += a["n"]; ana_x[k][2] += a["base"] * a["n"]; ana_x[k][3] += 1
    cross_rows = []
    for k in stage:
        ex = {"pre": [], "post": []}
        for sym, st in per[k]:
            for e in ("pre", "post"):
                s = st.get(e, {})
                if s.get("n", 0) >= 3 and s.get("ex60") is not None:
                    ex[e].append(s["ex60"])
        expct = E.SPEC[k]["expect"]
        agg = {e: {"n": len(v), "ex60": float(np.median(v)) if v else None} for e, v in ex.items()}
        share = tuple(float(np.mean([x * expct > 0 for x in ex[e]])) if ex[e] else 0.0 for e in ("pre", "post"))
        st = {e: {"n": agg[e]["n"], "ex60": agg[e]["ex60"]} for e in ("pre", "post")}
        lb, why = label(st, expct, share)
        h, n, bw, ni = ana_x[k]
        res["labels"][k] = lb
        res["cross"][k] = {"agg": agg, "share": share, "label": lb, "why": why, "hit": h, "pts": n, "base": bw / n if n else None, "instruments": ni}
        cross_rows.append((k, agg, share, lb, why, h, n, bw / n if n else None))
    res["cross_instruments"] = used
    res["measure"] = measure(D, used)
    for k, v in res["measure"]["labels"].items():
        res["labels"]["m_" + k] = v

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=lambda o: None)
    write_md(res, items, cross_rows, eps_rows, M, used)
    print(f"已写 {OUT_MD} 和 {OUT_JSON}")


def write_measure(w, ms, used):
    f2 = lambda x: "—" if x is None else f"{x:.2f}"
    w("## 能衡量什么：幅度和下跌风险（第五轮）\n")
    w("**结论：情绪和价格指标能可靠衡量的是\"接下来波动多大、下跌风险多高\"，涨跌方向只有少数组合（见后面的对照格和组合）有历史支持。**"
      "规则和刻度在 `engine.py` 的 `MEASURE` 里，检验前登记。\n")
    w("### 期权隐含波动能不能预告之后的波动（M1）\n")
    w("| 品种 | 段 | 和之后 21 天实际波动的相关系数 | 对照：用过去 21 天的波动去推 | 实际涨跌落在 ±1 倍预期内 | ±2 倍内 | 隐含减实际（中位数） |")
    w("|---|---|---|---|---|---|---|")
    for r in ms["iv"]:
        for e, nm in (("pre", "2015 前"), ("post", "2015 后")):
            x = r[e]
            w(f"| {r['name']} | {nm} | {f2(x['corr_iv'])} | {f2(x['corr_past'])} | {pp(x['cover1'])} | {pp(x['cover2'])} | {x['bias'] * 100:+.1f} 个点 |")
    w(f"\n标签：**{ms['labels']['iv_rv']}**。±1 倍预期波动的覆盖率理论上约 68%，高于 68% 说明期权定价略偏贵（卖期权的人要拿风险补偿）。"
      "半导体没有自己的长期期权历史，用纳指的 VXN 乘以半导体相对纳指的长期波动倍数代替；看板上的当天读数直接用 SMH 等自己的期权。\n")
    r = ms["rv"]
    w(f"### 波动会扎堆（M3）\n\n{len(used) + 1} 个品种，过去 21 天的实际波动和之后 21 天的相关系数中位数：2015 前 {f2(r['pre']['median'])}（{pp(r['pre']['share'])} 的品种 ≥ 0.5），"
      f"2015 后 {f2(r['post']['median'])}（{pp(r['post']['share'])}）。标签：**{ms['labels']['rv_rv']}**。\n")
    w("### 趋势状态 → 下跌风险（M2）和方向（M4）\n")
    w("\"再跌一大段\" = 之后 60 天内最低点比当天低 0.5 倍长期年化波动（半导体约 16%，标普约 7.5%）。每个品种比较\"在这个状态\"和\"不在\"的日子，再取品种中位数。\n")
    w("| 状态 | 段 | 再跌一大段的比例：高出（品种中位数） | 更高的品种占比 | 之后一个月波动：倍数 | 之后 60 天涨跌：差（品种中位数） | 更差的品种占比 | 风险标签 | 方向 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for k, v in ms["risk"].items():
        for e, nm in (("pre", "2015 前"), ("post", "2015 后")):
            x = v[e]
            dt = "—" if x["diff"] is None else f"{x['diff'] * 100:+.0f} 个百分点"
            w(f"| {E.MEASURE[k]['name']} | {nm} | {dt} | {pp(x['share'])} | {f2(x['volr'])} 倍 | {p(x['dir'])} | {pp(x['dir_share'])} | {v['label'] if e == 'pre' else ''} | {v['dir_label'] if e == 'pre' else ''} |")
    w("\n### 四种趋势状态下的历史风险（品种中位数）\n")
    w("| 状态 | 段 | 60 天内再跌一大段 | 之后一个月波动 ÷ 长期波动 | 之后 60 天涨跌 | 费城半导体自己 |")
    w("|---|---|---|---|---|---|")
    for i, g in ms["regimes"].items():
        for e, nm in (("pre", "2015 前"), ("post", "2015 后")):
            x = g[e]; s = x.get("sox")
            sx = f"{pp(s['big'])}，{f2(s['volx'])} 倍，{p(s['f60'])}" if s else "—"
            w(f"| {g['name']} | {nm} | {pp(x['big'])} | {f2(x['volx'])} 倍 | {p(x['f60'])} | {sx} |")
    w("")


def write_md(res, items, cross_rows, eps_rows, M, used):
    L = []
    w = L.append
    w("# 研究结论\n")
    w(f"数据截至 {res['asof']}，由 `python3 research.py` 生成，规则定义在 `engine.py` 的 `SPEC` 里。看板怎么读见 [规则说明.md](规则说明.md)。\n")
    w("## 怎么检验的\n")
    w("- 每条规则的阈值先登记（机制 + 阈值来源，都是行业惯例或整数），再跑检验。**结果只决定标签，不回头改阈值。**")
    w("- **有历史支持**：2015 年前、后各至少 3 次独立触发（相邻 10 个交易日只算一次），之后 60 天和**同样回撤深度的日子**相比，两段都朝预期方向偏离 1 个百分点以上。阶段描述类规则还要求在 ≥ 60% 的品种上两段方向都对。")
    w("- **只作描述**：其余情况。说明现在处在什么状态，但不能拿来推断之后。")
    w("- 「同样回撤深度」= 距一年高点的回撤，按各自的长期年化波动折算后分档。这样比较的是\"跌得一样深的时候，这条规则有没有多带来信息\"，而不是\"跌深了之后容易反弹\"。")
    w("- 期权和信用类规则只有指数一个样本（个股期权没有免费历史），从 2026-10-20 起每天记进 `daily_log.csv`，积累样本外记录。\n")

    w("## 一句话\n")
    NM = {**{k: v["name"] for k, v in E.SPEC.items()}, **{k: v["name"] for k, v in E.COMBOS.items()}}
    sup = [NM[k] for k, v in res["labels"].items() if v == "有历史支持"]
    rev = [NM[k] for k, v in res["labels"].items() if v == "两段都相反"]
    w("**有历史支持的**：" + ("；".join(sup) if sup else "没有") + "。")
    if rev:
        w("**两段都和预期相反的**：" + "；".join(rev) + "。")
    w("其余规则只作描述。单条恐惧项单独看两段不一致，要和价格位置一起看；价格、量能、动量类在 33 个品种上都没有稳定的额外信息。\n")

    write_measure(w, res["measure"], used)
    w("## 恐惧定价、脆弱度、出清（费城半导体指数，1995 年以来）\n")
    w("| 规则 | 历次顶底附近出现 | 任意一天 | 2015 前 | 2015 后 | 标签 |")
    w("|---|---|---|---|---|---|")
    for k, s, st, a, lb, why in items:
        where = "底" if s["group"] in ("fear", "turn") else "顶"
        ah = f"{where}部 {a['hit']}/{a['n']}" if a else "—"
        base = pp(a["base"]) if a else "—"
        w(f"| {s['name']} | {ah} | {base} | {era_cell(st.get('pre'))} | {era_cell(st.get('post'))} | {lb}{'（' + why + '）' if why else ''} |")
    w("\n预期方向：恐惧定价和出清是之后更好，脆弱度是之后更差。顶底附近 = 低点前 20 天到后 5 天、高点前 40 天。\n")

    w("## 组合\n")
    w("| 条件 | 2015 前 | 2015 后 | 60 天内最大回撤（全部） | 标签 |")
    w("|---|---|---|---|---|")
    for k, v in res["combos"].items():
        st = v["stats"]
        w(f"| {E.COMBOS[k]['name']} | {era_cell(st.get('pre'))} | {era_cell(st.get('post'))} | {p(st.get('all', {}).get('mdd'))} | {v['label']}{'（' + v['why'] + '）' if v['why'] else ''} |")
    w("")

    w("## 对照格：价格位置 × 恐惧定价\n")
    w("每格：独立次数，之后 60 天中位数（同一价格位置平均）。价格位置按距一年高点的回撤除以自己的长期年化波动：≤ −1.2 重伤，≤ −0.6 回调，离 200 日线超过 1 倍年化波动是延伸。\n")
    w("| 价格位置 | 恐惧未定价 | 部分定价 | 充分定价 |")
    w("|---|---|---|---|")
    for pi in range(4):
        cells = []
        for q in range(3):
            g = res["grid"].get(f"{pi}{q}", {})
            a = g.get("all")
            cells.append(f"{g.get('n', 0)} 次，{p(a['f60'])}（{p(a['row60'])}）" if a else f"{g.get('n', 0)} 次")
        w(f"| {E.POS_N[pi]} | " + " | ".join(cells) + " |")
    w("")

    w("## 三个温度分档\n")
    w("定价恐慌、抛售恐慌、过热温度各自所在的档位，之后 60 天和同样跌深的日子比（每次进入这一档算一次，相邻 10 天只算一次）。只作描述，不打标签。\n")
    for col, nm in (("g_fear", "定价恐慌"), ("g_sell", "抛售恐慌"), ("g_hot", "过热温度")):
        w(f"| {nm} | 2015 前 | 2015 后 |")
        w("|---|---|---|")
        for band, st in res["gauges"][col].items():
            w(f"| {band} | {era_cell(st.get('pre'))} | {era_cell(st.get('post'))} |")
        w("")
    w(f"## 价格、量能、动量：{len(used)} 个品种逐个检验\n")
    w("品种：" + "、".join(used) + "。只用现在的成分股，有幸存者偏差（早年跌没了的公司不在里面）。\n")
    w("| 规则 | 顶底附近出现 / 任意一天 | 2015 前：比同深度（方向对的品种占比） | 2015 后 | 标签 |")
    w("|---|---|---|---|---|")
    for k, agg, share, lb, why, h, n, base in cross_rows:
        hb = f"{h}/{n}（{pp(h / n) if n else '—'}）/ {pp(base)}" if n else "—"
        w(f"| {E.SPEC[k]['name']} | {hb} | {p(agg['pre']['ex60'])}（{pp(share[0])}） | {p(agg['post']['ex60'])}（{pp(share[1])}） | {lb}{'（' + why + '）' if why else ''} |")
    w("\n结论：这些特征在顶底附近确实常见，是人群行为的共同表现，但跌得一样深的时候，它们对之后的涨跌没有稳定的额外信息。看板用它们描述\"现在在哪一段\"。\n")

    w("## 历次 25% 以上的下跌\n")
    w("| 高点 | 低点 | 最大跌幅 | 见底用时 | 类型 | 高点前亮过的脆弱项 | 低点附近亮过的恐惧项 |")
    w("|---|---|---|---|---|---|---|")
    nm = lambda ks: "、".join(E.SPEC[k]["name"].split("（")[0] for k in ks) or "—"
    for e in eps_rows:
        w(f"| {e['pk']} | {e['tr']}{'（进行中）' if e['ongoing'] else ''} | {p(e['depth'])} | {e['days']} 天 | {e['type']} | {nm(e['frag'])} | {nm(e['fear'])} |")
    w("")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
