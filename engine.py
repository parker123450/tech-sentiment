#!/usr/bin/env python3
"""科技情绪全景 · 计算引擎（唯一的计算源，看板、通知、研究都用它）

主线：情绪 = 价格走到了哪一步 × 市场有没有为恐惧（或贪婪）付钱。
  - 趋势、动量、量能、参与度只描述"现在在哪一段"。口径无量纲（涨跌按各自的波动折算、成交量除以自己的均量），
    同一条规则对半导体指数、英伟达、KOSPI 都通用。
  - 期权和信用说明"情绪有没有被定价"：和全部历史比（只用当天以前的数据），再加固定刻度。
  - 不合成总分。输出三样东西：对照格（价格位置 × 恐惧定价）、底部清单和出清确认、脆弱度清单。

所有阈值登记在 SPEC 里（机制 + 阈值来源），写好后才跑研究；研究结果只决定标签，不回头改阈值。

用法：
  python3 engine.py             # 读 vol_history.csv 和 semis_history.csv，打印今天的读数
  python3 engine.py --selftest  # 把数据截到过去某天重算，检查结果和全量计算完全一致（没有用到未来数据）
"""
import json, os, sys, warnings
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
VOL_CSV = os.path.join(HERE, "vol_history.csv")
SEMIS_CSV = os.path.join(HERE, "semis_history.csv")
RESEARCH_JSON = os.path.join(HERE, "research_results.json")
ADJ = {"HYG", "IEF", "RSP", "SPY"}   # 算比值时用含分红价格
SPLIT = "2015-01-01"                 # 研究分两段验证
DEDUP = 10                           # 相邻 10 个交易日内只算一次
SQ = np.sqrt(252)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------- 规则登记（改阈值前先改这里，并写明理由）
# group: fear 恐惧定价（底部清单）/ frag 脆弱度 / turn 出清和转折 / stage 阶段描述（跨品种）
# expect: 预期之后 60 天相对"同样回撤深度的日子"更好（+1）还是更差（−1）
SPEC = {
    "vix30":    dict(group="fear", expect=1, name="VIX ≥ 30",
                     mech="标普期权的保险价格进入恐慌区。", src="业内通用刻度 20 / 30 / 40"),
    "vixspike": dict(group="fear", expect=1, name="VIX ≥ 自己 50 日均值的 1.4 倍",
                     mech="保险价格在短时间里被抢高。看的是相对冲击，不受各年代 VIX 基准高低的影响。", src="整数倍数，未调"),
    "inv":      dict(group="fear", expect=1, name="VIX 期限结构倒挂（VIX ÷ VIX3M ≥ 1）",
                     mech="近月保险比 3 个月的还贵：恐慌集中在眼前。", src="结构分界线 1.0（2006 年起有数据）"),
    "pchi":     dict(group="fear", expect=1, name="股票认沽/认购比（10 日均）到历史最高 10%",
                     mech="大量买认沽做保护或看空。", src="全部历史的十分位（2006 年起有数据）"),
    "credit":   dict(group="fear", expect=1, name="Baa 信用利差 20 天走阔 ≥ 20bp",
                     mech="信用市场要求更高的风险补偿，压力从股票扩散到信用。", src="整数刻度"),

    "ext":      dict(group="frag", expect=-1, name="离 200 日线超过 1 倍年化波动",
                     mech="涨幅远超自己的正常波动，追涨资金集中。", src="1 倍长期年化波动（半导体约 +32%，标普约 +15%）"),
    "divtop":   dict(group="frag", expect=-1, name="新高时 RSI 比上一个高点低 5 点以上",
                     mech="价格还在创新高，推动力已经在减弱（顶背离）。", src="RSI(14) 惯例；60 日新高、相隔 10–60 天"),
    "shrink":   dict(group="frag", expect=-1, name="新高时成交量萎缩（SMH 20 日均量 < 50 日均量 × 0.9）",
                     mech="价格新高，跟进的买盘却在减少。", src="整数比例"),
    "narrow":   dict(group="frag", expect=-1, name="指数离高点 3% 以内，但成分股在 50 日线上方的比例 < 60%，或等权标普 60 日跑输 ≥ 2%",
                     mech="上涨只靠少数大票，多数股票没有跟上。", src="整数刻度"),
    "frenzy":   dict(group="frag", expect=-1, name="认沽/认购比在历史最低 15%，或 3 倍做多÷做空成交额在历史最高 10%",
                     mech="大量买认购、加杠杆做多。", src="全部历史的分位"),
    "spotvol":  dict(group="frag", expect=-1, name="价格离高点 5% 以内，VIX 已经从 20 日低点抬升 20%",
                     mech="价格还在高位，已经有人开始花钱买保护。", src="整数比例"),

    "clear":    dict(group="turn", expect=1, name="恐慌退潮确认",
                     mech="恐惧充分定价之后，保险价格回落、价格不再创新低：被迫的对冲和抛售告一段落。",
                     src="过去 20 天恐惧定价 ≥ 3 条；VIX÷VIX3M < 0.95、VIX 比 20 日峰值低 20% 以上、价格 5 天没创新低"),
    "thrust":   dict(group="turn", expect=1, name="普涨回归：成分股从普跌到普涨不到两周（Zweig 广度推力）",
                     mech="从普跌到普涨只用了不到两周：资金在集体回来。", src="Zweig 原始阈值：上涨家数占比 10 日均值 10 天内从 < 0.40 升到 > 0.615"),
    "climax":   dict(group="turn", expect=1, name="恐慌抛售：SMH 放量 2 倍大跌且 RSI ≤ 30",
                     mech="恐慌性抛售集中出现。", src="2 倍均量、RSI 30 惯例"),

    "rsi30":    dict(group="stage", expect=1, name="RSI ≤ 30", mech="超卖。", src="RSI 惯例"),
    "rsi70":    dict(group="stage", expect=-1, name="RSI ≥ 70", mech="超买。", src="RSI 惯例"),
    "z20dn":    dict(group="stage", expect=1, name="20 日跌幅 ≥ 2 个标准差", mech="急跌。", src="2 个标准差"),
    "z20up":    dict(group="stage", expect=-1, name="20 日涨幅 ≥ 2 个标准差", mech="急涨。", src="2 个标准差"),
    "climaxdn": dict(group="stage", expect=1, name="放量 2 倍下跌", mech="集中抛售。", src="2 倍 50 日均量"),
    "divbot":   dict(group="stage", expect=1, name="新低时 RSI 比上一个低点高 5 点以上（底背离）", mech="还在创新低，下跌动能在减弱。", src="同顶背离"),
    "extstage": dict(group="stage", expect=-1, name="离 200 日线超过 1 倍年化波动", mech="同脆弱度第 1 条，按品种检验。", src="同上"),
    "divtopst": dict(group="stage", expect=-1, name="新高时 RSI 比上一个高点低 5 点以上（顶背离）", mech="同脆弱度第 2 条，按品种检验。", src="同上"),
    "shrinkst": dict(group="stage", expect=-1, name="新高时成交量萎缩", mech="同脆弱度第 3 条，按品种检验。", src="同上"),
}
# 组合（对照格的核心）：同样按上面的标签规则检验。跌深 = 价格位置是回调或重伤
COMBOS = {
    "deep_full": dict(expect=1, name="跌深 + 恐惧充分定价（≥ 3 条）", f=lambda M: (M.pos >= 2) & (M.fear_n >= 3)),
    "deep_part": dict(expect=0, name="跌深 + 恐惧部分定价（1–2 条）", f=lambda M: (M.pos >= 2) & M.fear_n.between(1, 2)),
    "deep_none": dict(expect=-1, name="跌深 + 恐惧没有定价（0 条）", f=lambda M: (M.pos >= 2) & (M.fear_n == 0)),
    "fear3": dict(expect=1, name="恐惧定价 ≥ 3 条（不论价格）", f=lambda M: M.fear_n >= 3),
    "frag3": dict(expect=-1, name="脆弱度 ≥ 3 条", f=lambda M: M.frag_n >= 3),
    "frag4": dict(expect=-1, name="脆弱度 ≥ 4 条", f=lambda M: M.frag_n >= 4),
}
# 能衡量什么（第五轮，2026-10-22 登记，检验前写好）：情绪和价格能可靠衡量的是"接下来波动多大、下跌风险多高"，
# 方向只有少数组合有历史支持。research.py 按下面的刻度检验，结果只决定标签。
#   kind = 幅度：标签"能衡量幅度"要求两段（2015 前后）相关系数都 ≥ 0.5（通用的"强相关"刻度）。
#   kind = 风险：标签"能衡量风险"要求两段都有 ≥ 60% 的品种在这个状态下"60 天内再跌一大段"的比例更高，
#                且品种中位数高出 ≥ 5 个百分点。"再跌一大段" = 60 天内最低点比当天低 0.5 倍长期年化波动（半导体约 16%，标普约 7.5%）。
#   同一批状态也检验方向（之后 60 天涨跌），沿用原来的"有历史支持"规则，预计不成立，如实写出。
MEASURE = {
    "iv_rv":    dict(kind="幅度", name="期权隐含波动 → 之后 21 天的实际波动",
                     mech="期权价格里含着市场对未来波动的预期。卖期权的人要为看错付钱，所以它通常略偏高，但高低方向是准的。",
                     src="相关系数 ≥ 0.5；±1 倍预期波动的覆盖率理论上约 68%"),
    "rv_rv":    dict(kind="幅度", name="过去 21 天的实际波动 → 之后 21 天的实际波动",
                     mech="波动会扎堆：大波动之后常跟着大波动。", src="相关系数 ≥ 0.5"),
    "below200": dict(kind="风险", name="价格在 200 日线下方", mech="长期趋势向下时，下跌更容易延续，波动也更大。", src="200 日线惯例"),
    "below50":  dict(kind="风险", name="价格在 50 日线下方", mech="中期趋势向下。", src="50 日线惯例"),
    "swingdn":  dict(kind="风险", name="摆动结构向下（高点、低点都在降低）", mech="道氏理论对下降趋势的定义。", src="摆动按 1 个月度标准差确认"),
}
BIG_K, BIG_N, RV_N = 0.5, 60, 21


def fwd_rv(c, n=RV_N):
    """之后 n 天的实际年化波动（研究和历史统计用，用到之后的价格）。"""
    return np.log(c).diff().rolling(n, min_periods=n).std().shift(-n) * SQ


def past_rv(c, n=RV_N):
    return np.log(c).diff().rolling(n, min_periods=n).std() * SQ


def big_drop(c, lv, k=BIG_K, n=BIG_N):
    """之后 n 天内的最低收盘价比今天低 k 倍长期年化波动（研究和历史统计用）。"""
    lo = c[::-1].rolling(n, min_periods=1).min()[::-1].shift(-1)
    out = (lo / c - 1) <= -k * lv
    return out.where(c.shift(-n).notna() & lv.notna())


def trend_states(F):
    """每个品种通用的趋势状态（只用当天以前的数据）。"""
    return pd.DataFrame({"below200": F.c < F.ma200, "below50": F.c < F.ma50, "swingdn": F.swing <= -2}, index=F.index).where(F.ma200.notna())


def expected_move(iv_pct, days):
    """期权隐含波动（年化 %）折算成 days 个交易日的 ±1 倍预期涨跌幅。"""
    return None if iv_pct is None or not np.isfinite(iv_pct) else iv_pct / 100 * np.sqrt(days / 252)


FEAR = [k for k, v in SPEC.items() if v["group"] == "fear"]
FRAG = [k for k, v in SPEC.items() if v["group"] == "frag"]
# 三个温度（2026-10-20 用户同意）：0–100，每一项按"离公认的恐慌 / 过热水平还有多远"折成 0–1，刻度都是整数或惯例。
#   定价恐慌：有没有人花钱为下跌买保险（期权和信用）。按机制分 4 组，组内平均、组间再平均，避免 VIX 家族占太多权重。
#   抛售恐慌：股票被卖得多狠（半导体普跌、去杠杆、韩国剧烈波动）。描述阶段用，不是信号。
#   过热温度：脆弱度 6 条连续化。
# 三个分开显示，不合成一个数。研究里它们都没有稳定的预测力（见 研究结论.md），只作描述；有历史支持的仍是清单条数。
GAUGES = {
    "fear": {"name": "定价恐慌", "items": [
        ("vix", "VIX", "15 → 30", "大盘期权"), ("vixspike", "VIX 相对 50 日均值", "1.0 → 1.4 倍", "大盘期权"),
        ("inv", "期限结构 VIX÷VIX3M", "0.85 → 1.0", "大盘期权"),
        ("vxn", "VXN 纳指科技期权", "18 → 36", "科技期权"), ("vxnspike", "VXN 相对 50 日均值", "1.0 → 1.4 倍", "科技期权"),
        ("pcjump", "认沽/认购比 ÷ 自己近一年中位数", "1.0 → 1.3 倍", "期权成交"),
        ("credit", "Baa 利差 20 日走阔", "0 → 20bp", "信用"), ("hyief", "HYG÷IEF 20 日跌幅", "0 → 2%", "信用")]},
    "sell": {"name": "抛售恐慌", "items": [
        ("washout", "成分股在 50 日线上方的比例", "50% → 10%", "普跌"), ("delever", "3 倍做多÷做空成交额分位", "50% → 10%", "去杠杆"),
        ("korea", "KOSPI 20 日波动 ÷ 长期波动", "1 → 2 倍", "韩国")]},
    "hot": {"name": "过热温度", "items": [
        ("ext", "离 200 日线", "0 → 1 倍年化波动", "延伸"), ("divtop", "顶背离", "没有 / 有", "动量"), ("shrink", "缩量新高", "没有 / 有", "量能"),
        ("narrow", "上涨变窄", "离高点 10% → 3% 以内，且成分股 50 日线上方 80% → 60%（或等权跑输 0 → 2%）", "参与度"),
        ("frenzy", "认购狂热或杠杆", "认沽/认购分位 50% → 15%，或杠杆多空比分位 50% → 90%", "仓位"),
        ("spotvol", "价涨波动也涨", "离高点 10% → 3% 以内，且 VIX 比 20 日低点高 0 → 20%", "期权")]},
}
GPFX = {"fear": "gf_", "sell": "gs_", "hot": "gh_"}
GCOL = {"fear": "g_fear", "sell": "g_sell", "hot": "g_hot"}
POS_N = ["延伸", "正常", "回调", "重伤"]          # 价格位置
FEAR_N = ["未定价", "部分定价", "充分定价"]       # 恐惧定价：0 条 / 1–2 条 / ≥ 3 条
CELL_TEXT = {
    (0, 0): "价格大幅延伸，恐惧没有定价：典型的追涨环境，看脆弱度清单。",
    (0, 1): "价格还在高位，已经有人在为下跌付钱。", (0, 2): "价格还在高位，已经有人在为下跌付钱。",
    (1, 0): "价格正常，期权和信用都平静。",
    (1, 1): "价格还没怎么跌，期权和信用已经紧张：恐惧走在价格前面。", (1, 2): "价格还没怎么跌，期权和信用已经紧张：恐惧走在价格前面。",
    (2, 0): "价格已经受伤，但还没人为恐惧付钱：按历次底部的共同特征，真正的恐慌还没来过，也就谈不上退潮。",
    (2, 1): "价格受伤，恐惧开始被定价，但还不充分。",
    (2, 2): "价格受伤，恐惧也已经充分定价：历史上恐慌见底的典型组合，但之后来回波动大。",
    (3, 0): "价格重伤，但还没人为恐惧付钱：按历次底部的共同特征，真正的恐慌还没来过，也就谈不上退潮。",
    (3, 1): "价格重伤，恐惧开始被定价，但还不充分。",
    (3, 2): "价格重伤，恐惧也已经充分定价：历史上恐慌见底的典型组合，但之后来回波动大。",
}


# ---------------------------------------------------------------- 数据
def load_csv(path):
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, dtype={"symbol": str, "date": str}, low_memory=False)
    out = {}
    for sym, g in df.groupby("symbol", sort=False):
        g = g.drop_duplicates("date", keep="last").sort_values("date")
        num = lambda k: pd.to_numeric(g[k], errors="coerce").to_numpy() if k in g else np.full(len(g), np.nan)
        c, a = num("close"), num("adjclose")
        f = pd.DataFrame({"c": c, "a": np.where(np.isfinite(a), a, c), "v": np.nan_to_num(num("volume")),
                          "o": num("open"), "h": num("high"), "l": num("low")}, index=pd.DatetimeIndex(pd.to_datetime(g["date"].to_numpy())))
        f = f[np.isfinite(f.c) & (f.c > 0)]
        if len(f):
            out[sym] = f
    return out


def load():
    D = load_csv(VOL_CSV)
    for k, v in load_csv(SEMIS_CSV).items():
        D.setdefault(k, v)
    return D


def until(D, t):
    """截到某天为止的数据（自检用）。"""
    t = pd.Timestamp(t)
    return {k: v[v.index <= t] for k, v in D.items()}


# ---------------------------------------------------------------- 通用工具（只用当天以前的数据）
def align(s, idx, limit=4):
    """每个日期取不晚于它、且不超过 limit 天前的最新值。"""
    if s is None or not len(s):
        return pd.Series(np.nan, index=idx)
    s = s[~s.index.duplicated(keep="last")].sort_index().astype(float)
    pos = s.index.searchsorted(idx, side="right") - 1
    ok = pos >= 0
    vals = np.full(len(idx), np.nan)
    vals[ok] = s.to_numpy()[pos[ok]]
    age = np.full(len(idx), np.inf)
    age[ok] = (idx[ok] - s.index[pos[ok]]).days
    vals[age > limit] = np.nan
    return pd.Series(vals, index=idx)


def exp_pct(s, min_n=250):
    """每天的值在它自己全部历史（到当天为止）里的分位，0–100。"""
    s = s.dropna()
    if len(s) < min_n:
        return pd.Series(np.nan, index=s.index)
    return s.expanding(min_periods=min_n).rank(pct=True) * 100


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn)


def recent(b, n):
    """过去 n 天（含今天）里出现过。"""
    return b.fillna(False).astype(float).rolling(n, min_periods=1).max() > 0


def divergence(new, R, sign):
    """创 60 日新高（新低）时，RSI 比上一次新高（新低，10–60 天以前）低（高）5 点以上。"""
    out = np.zeros(len(new), bool)
    nv, Rv = new.to_numpy(), R.to_numpy()
    li, lr = None, None
    for i in range(len(nv)):
        if not nv[i]:
            continue
        if li is not None and 10 <= i - li <= 60 and np.isfinite(Rv[i]) and np.isfinite(lr) and sign * (lr - Rv[i]) >= 5:
            out[i] = True
        if li is None or i - li >= 10:
            li, lr = i, Rv[i]
    return pd.Series(out, index=new.index)


def swing(c, sd):
    """摆动结构（实时确认的之字形）：价格从极值反向走出 1 个月度标准差（按长期波动），才确认那个极值是高点或低点。
    返回 structure（2 上升结构：高点低点都抬高；−2 下降结构；0 震荡）、brk（1 突破前高、−1 跌破前低）、pivots。"""
    x = np.log(c.to_numpy())
    th = (sd * np.sqrt(20)).to_numpy()
    n = len(x)
    st, brk = np.zeros(n), np.zeros(n)
    piv = []          # (i, 'H'/'L', price)
    H, L = [], []
    if n < 2:
        return pd.Series(st, index=c.index), pd.Series(brk, index=c.index), piv
    dr, ext, ei = 1, x[0], 0
    for i in range(1, n):
        t = th[i] if np.isfinite(th[i]) else 0.08
        if dr == 1:
            if x[i] >= ext:
                ext, ei = x[i], i
            elif ext - x[i] >= t:
                H.append(ext); piv.append((ei, "H", float(np.exp(ext)))); dr, ext, ei = -1, x[i], i
        else:
            if x[i] <= ext:
                ext, ei = x[i], i
            elif x[i] - ext >= t:
                L.append(ext); piv.append((ei, "L", float(np.exp(ext)))); dr, ext, ei = 1, x[i], i
        if len(H) >= 2 and len(L) >= 2:
            hh, hl = H[-1] > H[-2], L[-1] > L[-2]
            st[i] = 2 if hh and hl else -2 if not hh and not hl else 0
        if H and x[i] > H[-1]:
            brk[i] = 1
        elif L and x[i] < L[-1]:
            brk[i] = -1
    return pd.Series(st, index=c.index), pd.Series(brk, index=c.index), piv


def episodes(px, th=-0.25):
    """每一次从过去一年最高收盘价下跌 25% 以上；创新高或从低点反弹 50% 算结束（和旧看板一致）。"""
    n, eps = len(px), []
    inE, pk, tr, last_end = False, 0, 0, 0
    def win_max(i):
        m = max(last_end, i - 251)
        for k in range(m, i + 1):
            if px[k] > px[m]:
                m = k
        return m
    for i in range(1, n):
        if not inE:
            m = win_max(i)
            if px[i] / px[m] - 1 <= th:
                inE, pk, tr = True, m, i
        else:
            if px[i] < px[tr]:
                tr = i
            if px[i] >= px[pk] or px[i] >= px[tr] * 1.5:
                eps.append((pk, tr, i)); inE = False; last_end = i
    if inE:
        eps.append((pk, tr, None))
    return eps


# ---------------------------------------------------------------- 每个品种都适用的价格、动量、量能特征
def px_features(d):
    c = d.c
    v = d.v.where(d.v > 0)
    lr = np.log(c).diff()
    sd = lr.rolling(252, min_periods=120).std()
    lv = (sd * SQ).expanding(min_periods=250).median()              # 长期年化波动（只用当天以前）
    ma20, ma50, ma200 = (c.rolling(n, min_periods=n).mean() for n in (20, 50, 200))
    hi = c.rolling(252, min_periods=60).max()
    F = pd.DataFrame(index=c.index)
    F["c"], F["lv"], F["sd"] = c, lv, sd
    F["dd"] = c / hi - 1
    F["ddv"] = F.dd / lv
    F["ext"] = c / ma200 - 1
    F["extv"] = F.ext / lv
    F["ma20"], F["ma50"], F["ma200"] = ma20, ma50, ma200
    F["slope200"] = ma200 / ma200.shift(20) - 1
    F["r20"] = c / c.shift(20) - 1
    F["r10"] = c / c.shift(10) - 1
    F["r10p"] = c.shift(10) / c.shift(20) - 1
    dlv = lv / SQ                                                    # 长期日波动：速度一律按它折算，不随最近一年的波动漂移
    F["z20"] = np.log(c / c.shift(20)) / (dlv * np.sqrt(20))
    F["acc"] = (np.log(c / c.shift(10)) - np.log(c.shift(10) / c.shift(20))) / (dlv * np.sqrt(20))
    F["rsi"] = R = rsi(c)
    v50 = v.rolling(50, min_periods=50).mean()
    F["relv"] = v / v50
    up = v.where(c > c.shift(1), 0).rolling(20, min_periods=20).sum()
    dn = v.where(c < c.shift(1), 0).rolling(20, min_periods=20).sum()
    F["udv"] = up / dn.where(dn > 0)
    rng = (d.h - d.l).where(d.h > d.l)
    F["pos"] = (c - d.l) / rng
    F["newhi"] = c >= c.rolling(60, min_periods=60).max()
    F["newlo"] = c <= c.rolling(60, min_periods=60).min()
    F["div_top"] = divergence(F.newhi, R, 1)
    F["div_bot"] = divergence(F.newlo, R, -1)
    F["shrink_hi"] = F.newhi & (v.rolling(20, min_periods=20).mean() < v50 * 0.9)
    F["climax_dn"] = (lr < 0) & (F.relv >= 2)
    F["swing"], F["brk"], piv = swing(c, dlv)
    F.attrs["pivots"] = piv
    return F


def stage_flags(F):
    """阶段描述类规则（研究里拿 33 个品种逐个检验）。"""
    return pd.DataFrame({
        "rsi30": F.rsi <= 30, "rsi70": F.rsi >= 70, "z20dn": F.z20 <= -2, "z20up": F.z20 >= 2,
        "climaxdn": F.climax_dn, "divbot": F.div_bot, "extstage": F.extv >= 1,
        "divtopst": F.div_top, "shrinkst": F.shrink_hi, "_hasvol": F.relv.notna(),
    }, index=F.index)


# ---------------------------------------------------------------- 成分股参与度
def breadth(D, idx, members=None):
    import build_sentiment as bs   # 成分股名单只登记在一处
    members = members or bs.SEMIS
    cl = pd.DataFrame({s: D[s].c for s in members if s in D}).reindex(idx)
    out = pd.DataFrame(index=idx)
    if cl.empty:
        for k in ("n", "pct50", "pct200", "nh", "nl", "adv", "zema", "thrust"):
            out[k] = np.nan
        return out
    m50 = cl.rolling(50, min_periods=50).mean()
    m200 = cl.rolling(200, min_periods=200).mean()
    hi60 = cl.rolling(60, min_periods=60).max()
    lo60 = cl.rolling(60, min_periods=60).min()
    n50, n200, n60 = m50.notna().sum(1), m200.notna().sum(1), hi60.notna().sum(1)
    ok = lambda n: n >= 15
    out["n"] = n50
    out["pct50"] = ((cl > m50).sum(1) / n50 * 100).where(ok(n50))
    out["pct200"] = ((cl > m200).sum(1) / n200 * 100).where(ok(n200))
    out["nh"] = (cl >= hi60).sum(1).where(ok(n60))
    out["nl"] = (cl <= lo60).sum(1).where(ok(n60))
    ch = cl.diff()
    a, b = (ch > 0).sum(1), (ch < 0).sum(1)
    out["adv"] = (a / (a + b)).where((a + b) >= 15)
    out["zema"] = out.adv.ewm(span=10, adjust=False).mean()
    low10 = out.zema.rolling(10, min_periods=1).min()
    out["thrust"] = (out.zema > 0.615) & (low10 < 0.40)
    return out


# ---------------------------------------------------------------- 市场层面（以费城半导体指数的交易日为准）
def market(D):
    sox = D["^SOX"]
    idx = sox.index
    X = px_features(sox)
    M = X.drop(columns=["newhi", "newlo"]).copy()
    M.attrs["pivots"] = X.attrs["pivots"]
    get = lambda k: D[k].c if k in D else None
    adj = lambda k: D[k].a if k in D else None

    smh = px_features(D["SMH"]) if "SMH" in D else None
    for k in ("relv", "udv", "climax_dn", "shrink_hi", "rsi", "newhi"):
        M["smh_" + k] = align(smh[k].astype(float), idx) if smh is not None else np.nan
    M["smh_climax_dn"] = M.smh_climax_dn > 0.5
    M["smh_shrink_hi"] = M.smh_shrink_hi > 0.5

    spx = D.get("^GSPC")
    M["spx_lv"] = align(px_features(spx).lv, idx) if spx is not None else np.nan

    vix, v3m, v9d = get("^VIX"), get("^VIX3M"), get("^VIX9D")
    M["vix"] = align(vix, idx)
    M["vix50"] = align(vix.rolling(50, min_periods=50).mean(), idx) if vix is not None else np.nan
    M["vixmin20"] = align(vix.rolling(20, min_periods=20).min(), idx) if vix is not None else np.nan
    M["vixmax20"] = align(vix.rolling(20, min_periods=20).max(), idx) if vix is not None else np.nan
    M["vix_pct"] = align(exp_pct(vix), idx) if vix is not None else np.nan
    t3 = (vix / v3m).dropna() if vix is not None and v3m is not None else None
    t9 = (v9d / vix).dropna() if vix is not None and v9d is not None else None
    M["t3"] = align(t3, idx)
    M["t3_pct"] = align(exp_pct(t3), idx) if t3 is not None else np.nan
    M["t9"] = align(t9, idx)
    for k, col in (("^VXN", "vxn"), ("^VVIX", "vvix"), ("^SKEW", "skew")):
        s = get(k)
        M[col] = align(s, idx)
        M[col + "_pct"] = align(exp_pct(s), idx) if s is not None else np.nan

    pc = get("PC_EQ")
    pc10 = pc.rolling(10, min_periods=10).mean() if pc is not None else None
    M["pc10"] = align(pc10, idx)
    M["pc_pct"] = align(exp_pct(pc10), idx) if pc10 is not None else np.nan
    pci = get("PC_IDX")
    M["pcidx10"] = align(pci.rolling(10, min_periods=10).mean(), idx) if pci is not None else np.nan

    baa = get("BAA10Y")
    M["baa"] = align(baa, idx)
    M["baa_pct"] = align(exp_pct(baa), idx) if baa is not None else np.nan
    M["baa20"] = align((baa - baa.shift(20)) * 100, idx) if baa is not None else np.nan
    hy = get("HYOAS")
    M["hyoas"] = align(hy, idx)
    M["hyoas20"] = align((hy - hy.shift(20)) * 100, idx) if hy is not None else np.nan
    hyg, ief = adj("HYG"), adj("IEF")
    if hyg is not None and ief is not None:
        r = (hyg / ief).dropna()
        M["hyief20"] = align((r / r.shift(20) - 1) * 100, idx)
    else:
        M["hyief20"] = np.nan
    rsp, spy = adj("RSP"), adj("SPY")
    if rsp is not None and spy is not None:
        r = (rsp / spy).dropna()
        M["rsp60"] = align((r / r.shift(60) - 1) * 100, idx)
    else:
        M["rsp60"] = np.nan
    if "SOXL" in D and "SOXS" in D:
        L, S_ = D["SOXL"], D["SOXS"]
        j = L.index.intersection(S_.index)
        dl, ds = (L.c * L.v).reindex(j), (S_.c * S_.v).reindex(j)
        ok = (dl > 0) & (ds > 0)
        bb = (dl[ok].rolling(10, min_periods=10).mean() / ds[ok].rolling(10, min_periods=10).mean()).dropna()
        M["bb"] = align(bb, idx)
        M["bb_pct"] = align(exp_pct(bb), idx)
    else:
        M["bb"] = M["bb_pct"] = np.nan

    B = breadth(D, idx)
    for k in B.columns:
        M["br_" + k] = B[k]
    M["br_thrust"] = M.br_thrust.fillna(False).astype(bool)

    # ---- 恐惧定价（底部清单）
    yes = lambda s: s.fillna(False).astype(bool)
    M["f_vix30"] = yes(M.vix >= 30)
    M["f_vixspike"] = yes(M.vix >= M.vix50 * 1.4)
    M["f_inv"] = yes(M.t3 >= 1)
    M["f_pchi"] = yes(M.pc_pct >= 90)
    M["f_credit"] = yes(M.baa20 >= 20)
    M["fear_n"] = sum(M["f_" + k].astype(int) for k in FEAR)
    M["fear_avail"] = (M.vix.notna().astype(int) * 2 + M.t3.notna().astype(int) + M.pc_pct.notna().astype(int) + M.baa20.notna().astype(int))
    M["fear_lv"] = np.where(M.fear_n >= 3, 2, np.where(M.fear_n >= 1, 1, 0))

    # ---- 脆弱度
    near = lambda p: M.dd >= -p
    M["r_ext"] = yes(M.extv >= 1)
    M["r_divtop"] = recent(M.div_top, 20) & near(0.05)
    M["r_shrink"] = recent(M.smh_shrink_hi, 20) & near(0.05)
    M["r_narrow"] = near(0.03) & (yes(M.br_pct50 < 60) | yes(M.rsp60 <= -2))
    M["r_frenzy"] = yes(M.pc_pct <= 15) | yes(M.bb_pct >= 90)
    M["r_spotvol"] = near(0.05) & yes(M.vix >= M.vixmin20 * 1.2)
    M["frag_n"] = sum(M["r_" + k].astype(int) for k in FRAG)

    # ---- 出清和转折
    low20, low5 = M.c.rolling(20, min_periods=20).min(), M.c.rolling(5, min_periods=5).min()
    M["c_recent"] = M.fear_n.rolling(20, min_periods=1).max() >= 3
    M["c_term"] = yes(M.t3 < 0.95) | M.t3.isna()          # 2006 年以前没有 VIX3M，这一条不要求
    M["c_vixoff"] = yes(M.vix <= M.vixmax20 * 0.8)
    M["c_nolow"] = yes(low5 > low20)
    M["t_clear"] = M.c_recent & M.c_term & M.c_vixoff & M.c_nolow
    M["t_thrust"] = M.br_thrust
    M["t_climax"] = M.smh_climax_dn & yes(M.smh_rsi <= 30)

    # ---- 价格位置
    pos = np.where(M.ddv <= -1.2, 3, np.where(M.ddv <= -0.6, 2, np.where(M.extv >= 1, 0, 1)))
    M["pos"] = np.where(M.ddv.isna(), np.nan, pos)
    M["eq_spx"] = M.dd / M.lv * M.spx_lv          # 按波动折算成"相当于标普跌了多少"

    # ---- 三个温度（GAUGES）
    c01 = lambda x: x.clip(0, 1)
    vxn = get("^VXN")
    M["vxn50"] = align(vxn.rolling(50, min_periods=50).mean(), idx) if vxn is not None else np.nan
    M["pcjump"] = align(pc10 / pc10.rolling(250, min_periods=120).median(), idx) if pc10 is not None else np.nan
    ks = D.get("^KS11")
    if ks is not None:
        kf = px_features(ks)
        M["kvol"] = align(np.log(ks.c).diff().rolling(20, min_periods=20).std() * SQ / kf.lv, idx)
    else:
        M["kvol"] = np.nan
    M["gf_vix"] = c01((M.vix - 15) / 15)
    M["gf_vixspike"] = c01((M.vix / M.vix50 - 1) / 0.4)
    M["gf_inv"] = c01((M.t3 - 0.85) / 0.15)
    M["gf_vxn"] = c01((M.vxn - 18) / 18)
    M["gf_vxnspike"] = c01((M.vxn / M.vxn50 - 1) / 0.4)
    M["gf_pcjump"] = c01((M.pcjump - 1) / 0.3)
    M["gf_credit"] = c01(M.baa20 / 20)
    M["gf_hyief"] = c01(-M.hyief20 / 2)
    M["gs_washout"] = c01((50 - M.br_pct50) / 40)
    M["gs_delever"] = c01((50 - M.bb_pct) / 40)
    M["gs_korea"] = c01(M.kvol - 1)
    nearv = c01((M.dd + 0.10) / 0.07)                 # 离高点 3% 以内 = 1，10% 以外 = 0
    M["gh_ext"] = c01(M.extv)
    M["gh_divtop"] = M.r_divtop.astype(float).where(M.dd.notna())
    M["gh_shrink"] = M.r_shrink.astype(float).where(M.smh_relv.notna() | M.dd.notna())
    M["gh_narrow"] = nearv * np.fmax(c01((80 - M.br_pct50) / 20), c01(-M.rsp60 / 2))
    M["gh_frenzy"] = np.fmax(c01((50 - M.pc_pct) / 35), c01((M.bb_pct - 50) / 40))
    M["gh_spotvol"] = nearv * c01((M.vix / M.vixmin20 - 1) / 0.2)
    for kind, spec in GAUGES.items():
        groups = {}
        for k, _, _, grp in spec["items"]:
            groups.setdefault(grp, []).append(GPFX[kind] + k)
        gm = pd.concat([M[cols].mean(axis=1).rename(g) for g, cols in groups.items()], axis=1)
        M[GCOL[kind]] = (gm.mean(axis=1) if kind == "fear" else M[[GPFX[kind] + k for k, *_ in spec["items"]]].mean(axis=1)) * 100
        M[GCOL[kind] + "_pct"] = exp_pct(M[GCOL[kind]]).reindex(M.index)
    return M


# ---------------------------------------------------------------- 历史统计（用到之后的价格，只用于统计过去）
def forward(c):
    n = len(c)
    a = c.to_numpy()
    f20 = np.r_[a[20:] / a[:-20] - 1, np.full(20, np.nan)]
    f60 = np.r_[a[60:] / a[:-60] - 1, np.full(60, np.nan)]
    lo = pd.Series(a[::-1]).rolling(60, min_periods=60).min().to_numpy()[::-1]   # 从当天起 60 天内最低（含当天）
    lo_next = np.r_[lo[1:], np.nan]                                                 # 从次日起 60 天
    mdd = lo_next / a - 1
    mdd[n - 61:] = np.nan
    return f20, f60, mdd


def dedup(mask):
    out, last = [], -99
    for i, x in enumerate(np.asarray(mask, bool)):
        if x and i - last > DEDUP:
            out.append(i)
        if x:
            last = i
    return out


DD_EDGES = [-1.2, -0.9, -0.6, -0.3, -0.1]   # 回撤深度（按长期年化波动折算）分档，用来做"同样回撤深度"的对照


def event_stats(mask, c, ddv, era_split=SPLIT, valid_from=None):
    """触发（相邻 10 天只算一次）之后的表现，并和"同样回撤深度的日子"比；分 2015 前后。"""
    f20, f60, mdd = forward(c)
    idx = c.index
    bucket = np.digitize(np.nan_to_num(ddv.to_numpy(), nan=0.0), DD_EDGES)
    valid = np.isfinite(f60) & np.isfinite(ddv.to_numpy())
    if valid_from is not None:
        valid &= idx >= pd.Timestamp(valid_from)
    era = np.where(idx < pd.Timestamp(era_split), 0, 1)
    ev_all = dedup(np.asarray(mask, bool) & valid)
    res = {}
    for e, name in ((None, "all"), (0, "pre"), (1, "post")):
        ev = [i for i in ev_all if e is None or era[i] == e]
        base = valid & ((era == e) if e is not None else True)
        if not ev:
            res[name] = {"n": 0}
            continue
        ex, exd = [], []
        for i in ev:
            m = base & (bucket == bucket[i])
            if m.sum() >= 50:
                ex.append(f60[i] - np.median(f60[m])); exd.append(mdd[i] - np.median(mdd[m]))
        res[name] = {
            "n": len(ev), "f20": float(np.median(f20[ev])), "f60": float(np.median(f60[ev])), "up60": float(np.mean(f60[ev] > 0)),
            "mdd": float(np.median(mdd[ev])), "base_f60": float(np.median(f60[base])),
            "ex60": float(np.median(ex)) if ex else None, "exmdd": float(np.median(exd)) if exd else None,
        }
    return res


def anatomy(mask, c, eps, kind, back, fwd_, valid_from=None):
    """历次顶（kind='pk'）或底（'tr'）附近 [−back, +fwd] 天里出现过的比例，和任意一天比。"""
    b = pd.Series(np.asarray(mask, bool), index=c.index).astype(int)
    w = (b.rolling(back + fwd_ + 1, min_periods=1).max().shift(-fwd_).fillna(0) > 0).to_numpy()
    vf = pd.Timestamp(valid_from) if valid_from else c.index[0]
    pts = [(e[0] if kind == "pk" else e[1]) for e in eps]
    pts = [i for i in pts if c.index[i] >= vf]
    hits = [bool(w[i]) for i in pts]
    base = w[(c.index >= vf)].mean() if len(c) else np.nan
    return {"n": len(pts), "hit": int(sum(hits)), "base": float(base), "which": [c.index[i].strftime("%Y-%m") for i, h in zip(pts, hits) if h]}


START = {"inv": "2006-09-01", "pchi": "2007-11-01", "frenzy": "2007-11-01", "narrow": "2004-01-01", "shrink": "2001-01-01", "climax": "2001-01-01", "thrust": "2001-01-01"}


def grid_stats(M):
    c, ddv = M.c, M.ddv
    pos, lv = M.pos.to_numpy(), M.fear_lv.to_numpy()
    out = {}
    f20, f60, mdd = forward(c)
    era = np.where(M.index < pd.Timestamp(SPLIT), 0, 1)
    for p in range(4):
        rowm = (pos == p) & np.isfinite(f60)
        for q in range(3):
            m = (pos == p) & (lv == q)
            ev = [i for i in dedup(m) if np.isfinite(f60[i])]
            cell = {"days": int(m.sum()), "n": len(ev)}
            for e, nm in ((0, "pre"), (1, "post"), (None, "all")):
                eve = [i for i in ev if e is None or era[i] == e]
                rm = rowm & ((era == e) if e is not None else True)
                if eve and rm.sum() >= 30:
                    cell[nm] = {"n": len(eve), "f20": float(np.median(f20[eve])), "f60": float(np.median(f60[eve])),
                                "row60": float(np.median(f60[rm])), "up60": float(np.mean(f60[eve] > 0)), "mdd": float(np.median(mdd[eve]))}
            out[f"{p}{q}"] = cell
    return out


def episode_table(M):
    c = M.c
    px = c.to_numpy()
    eps = episodes(px)
    rows = []
    for pk, tr, en in eps:
        sl = slice(pk, tr + 1)
        vx = M.vix.iloc[sl].max()
        bq = M.baa.iloc[sl].max()
        lo, hi = max(0, tr - 20), min(len(c) - 1, tr + 5)
        fear_lit = [k for k in FEAR if M["f_" + k].iloc[lo:hi + 1].any()]
        frag_lit = [k for k in FRAG if M["r_" + k].iloc[max(0, pk - 40):pk + 1].any()]
        rec = None
        for i in range(tr, len(px)):
            if px[i] >= px[pk]:
                rec = i - tr
                break
        rows.append({"pk": c.index[pk].strftime("%Y-%m-%d"), "tr": c.index[tr].strftime("%Y-%m-%d"), "end": None if en is None else c.index[en].strftime("%Y-%m-%d"),
                     "ipk": int(pk), "itr": int(tr), "depth": float(px[tr] / px[pk] - 1), "days": int(tr - pk), "rec": rec,
                     "vix": None if not np.isfinite(vx) else float(vx), "baa": None if not np.isfinite(bq) else float(bq),
                     "type": "系统性" if (np.isfinite(vx) and vx >= 40) or (np.isfinite(bq) and bq >= 3.5) else "板块型",
                     "fear": fear_lit, "frag": frag_lit, "ongoing": en is None})
    return rows, eps


def item_anatomy(M, eps):
    c = M.c
    out = {}
    for k in FEAR:
        out[k] = anatomy(M["f_" + k], c, eps, "tr", 20, 5, START.get(k, "1995-06-01"))
    for k in FRAG:
        out[k] = anatomy(M["r_" + k], c, eps, "pk", 40, 0, START.get(k, "1995-06-01"))
    return out


# ---------------------------------------------------------------- 文字
def pct(x, dp=1, sign=True):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 100:+.{dp}f}%" if sign else f"{x * 100:.{dp}f}%"


def f2(x, dp=2):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{dp}f}"


def vix_zone(v):
    return "—" if not np.isfinite(v) else "平静" if v < 15 else "正常" if v < 20 else "紧张" if v < 30 else "恐慌" if v < 40 else "极度恐慌"


def term_zone(t):
    return "—" if not np.isfinite(t) else "很陡（市场很放心）" if t < 0.85 else "正常" if t < 0.95 else "趋平（眼前的风险变贵）" if t < 1 else "倒挂（恐慌集中在眼前）"


def ma_text(r):
    c, a, b, s = r.c, r.ma50, r.ma200, r.slope200
    if not (np.isfinite(a) and np.isfinite(b)):
        return "均线数据不足"
    if c > a > b and s > 0:
        return "多头排列（价格 > 50 日线 > 200 日线）"
    if c < a < b and s < 0:
        return "空头排列（价格 < 50 日线 < 200 日线）"
    return ("200 日线上方" if c > b else "200 日线下方") + "，均线交织"


def swing_text(r):
    s = {2: "上升结构（高点、低点都在抬高）", -2: "下降结构（高点、低点都在降低）"}.get(int(r.swing) if np.isfinite(r.swing) else 0, "震荡结构")
    b = {1: "，刚突破前一个摆动高点", -1: "，已跌破前一个摆动低点"}.get(int(r.brk) if np.isfinite(r.brk) else 0, "")
    return s + b


def speed_text(z):
    return "—" if not np.isfinite(z) else "急跌" if z <= -2 else "下跌" if z <= -1 else "平稳" if z < 1 else "上涨" if z < 2 else "急涨"


def acc_text(r):
    if not (np.isfinite(r.acc) and np.isfinite(r.r10)):
        return ""
    if r.r10 < 0:
        return "下跌在加速" if r.acc <= -1 else "下跌在减速" if r.acc >= 1 else "节奏没变"
    return "上涨在加速" if r.acc >= 1 else "上涨在减速" if r.acc <= -1 else "节奏没变"


def pos_text(r):
    eq = r.eq_spx
    if not np.isfinite(r.pos):
        return "—"
    p = int(r.pos)
    if p == 0:
        return f"延伸：离 200 日线 {pct(r.ext)}，超过自己 1 倍年化波动"
    if p == 1:
        return f"正常：距一年高点 {pct(r.dd)}（相当于标普 {pct(eq)}）"
    return f"{POS_N[p]}：距一年高点 {pct(r.dd)}（按波动折算，相当于标普 {pct(eq)}）"


def layers(M, D):
    r = M.iloc[-1]
    L = []
    piv = M.attrs.get("pivots", [])
    lh = next((p for p in reversed(piv) if p[1] == "H"), None)
    ll = next((p for p in reversed(piv) if p[1] == "L"), None)
    near = ""
    if lh and ll:
        near = (f"现价离上一个摆动低点 {ll[2]:,.0f}（{M.index[ll[0]].strftime('%m-%d')}）{pct(r.c / ll[2] - 1)}，"
                f"离上一个摆动高点 {lh[2]:,.0f}（{M.index[lh[0]].strftime('%m-%d')}）{pct(r.c / lh[2] - 1)}。")
    L.append({"id": "trend", "name": "趋势", "role": "描述阶段",
              "text": f"{ma_text(r)}；{swing_text(r)}。{near}",
              "nums": [["距一年高点", pct(r.dd)], ["相当于标普", pct(r.eq_spx)], ["离 200 日线", pct(r.ext)]]})
    L.append({"id": "mom", "name": "动量与速度", "role": "描述阶段",
              "text": f"10 天 {pct(r.r10)}、20 天 {pct(r.r20)}，按自己的长期波动算是{speed_text(r.z20)}（{f2(r.z20, 1)} 个标准差），{acc_text(r)}。"
                      + ("最近 20 天出现过顶背离。" if recent(M.div_top, 20).iloc[-1] else "")
                      + ("最近 20 天出现过底背离。" if recent(M.div_bot, 20).iloc[-1] else ""),
              "nums": [["RSI(14)", f2(r.rsi, 0)], ["10 日", pct(r.r10)], ["20 日", pct(r.r20)], ["Z 分数", f2(r.z20, 1)]]})
    cl = M.index[M.smh_climax_dn.to_numpy()]
    last_cl = cl[-1].strftime("%Y-%m-%d") if len(cl) else "—"
    L.append({"id": "vol", "name": "量能", "role": "描述阶段",
              "text": f"SMH 今天成交量是 50 日均量的 {f2(r.smh_relv)} 倍；20 天上涨日÷下跌日成交量 {f2(r.smh_udv)}"
                      f"（{'买盘主导' if r.smh_udv > 1.2 else '卖盘主导' if r.smh_udv < 0.8 else '买卖均衡'}）；最近一次放量 2 倍下跌：{last_cl}。"
                      f"3 倍做多÷做空成交额 {f2(r.bb)} 倍（历史 {f2(r.bb_pct, 0)}% 分位）。",
              "nums": [["相对成交量", f2(r.smh_relv)], ["上涨÷下跌量", f2(r.smh_udv)], ["杠杆多空比", f2(r.bb)]]})
    L.append({"id": "breadth", "name": "参与度", "role": "描述，也进清单",
              "text": (f"{int(r.br_n)} 只成分股里 {f2(r.br_pct50, 0)}% 在 50 日线上方、{f2(r.br_pct200, 0)}% 在 200 日线上方；"
                       f"60 日新高 {f2(r.br_nh, 0)} 家、新低 {f2(r.br_nl, 0)} 家。" if np.isfinite(r.br_pct50) else "成分股数据不足。")
                      + (f"等权标普 60 日比市值加权 {r.rsp60:+.1f}%。" if np.isfinite(r.rsp60) else ""),
              "nums": [["50 日线上方", f2(r.br_pct50, 0) + "%"], ["200 日线上方", f2(r.br_pct200, 0) + "%"], ["新高−新低", f2(r.br_nh - r.br_nl, 0)]]})
    L.append({"id": "price", "name": "恐惧/贪婪定价", "role": "核心信号",
              "text": f"VIX {f2(r.vix, 1)}（{vix_zone(r.vix)}，历史 {f2(r.vix_pct, 0)}% 分位）；期限结构 {f2(r.t3, 3)}（{term_zone(r.t3)}）；"
                      f"股票认沽/认购比 10 日均 {f2(r.pc10)}（历史 {f2(r.pc_pct, 0)}% 分位）；Baa 利差 {f2(r.baa)}%（历史 {f2(r.baa_pct, 0)}% 分位），20 天 {f2(r.baa20, 0)}bp。",
              "nums": [["VIX", f2(r.vix, 1)], ["VIX÷VIX3M", f2(r.t3, 3)], ["认沽/认购", f2(r.pc10)], ["Baa", f2(r.baa) + "%"]]})
    kr = []
    for k, n in (("^KS11", "KOSPI"), ("000660.KS", "海力士"), ("005930.KS", "三星")):
        if k not in D:
            continue
        F = px_features(D[k])
        q = F.iloc[-1]
        spx = M.spx_lv.iloc[-1]
        kr.append(f"{n} 距高点 {pct(q.dd)}（相当于标普 {pct(q.dd / q.lv * spx)}），{speed_text(q.z20)}，{ma_text(q).split('（')[0]}")
    L.append({"id": "cross", "name": "跨市场", "role": "确认", "text": "；".join(kr) + "。" if kr else "韩国数据不足。", "nums": []})
    return L


GBANDS = {
    "fear": ["很低：几乎没人在为下跌付钱", "偏低：保护比平时贵一点", "中等：恐惧在被定价", "高：大家在花大钱买保险", "极高：恐慌性定价"],
    "sell": ["很低：没有恐慌性抛售", "偏低：有些卖压", "中等：抛售在扩散", "高：恐慌性抛售", "极高：几乎全面杀跌"],
    "hot": ["很低", "偏低", "中等：上涨开始变脆", "高：上涨很脆弱", "极高：多项脆弱特征同时出现"],
}


def gauge_text(kind, v, pc):
    if v is None:
        return "数据不足"
    word = {"fear": "定价更恐慌", "sell": "卖得更狠", "hot": "更热"}[kind]
    return GBANDS[kind][min(4, int(v // 20))] + (f"。比 1995 年以来 {pc:.0f}% 的日子{word}" if pc is not None else "")


def gauges_now(M):
    r = M.iloc[-1]
    out = {}
    for kind, spec in GAUGES.items():
        col, pfx = GCOL[kind], GPFX[kind]
        v, pc = num(r[col]), num(r[col + "_pct"])
        w = M[col].iloc[-252:]
        out[kind] = {"name": spec["name"], "v": None if v is None else round(v), "pct": None if pc is None else round(pc),
                     "text": gauge_text(kind, v, pc), "d5": num(r[col] - M[col].iloc[-6]) if len(M) > 6 else None,
                     "items": [[nm, None if num(r[pfx + k]) is None else round(num(r[pfx + k]) * 100), rng, grp] for k, nm, rng, grp in spec["items"]],
                     "hi1y": num(w.max()), "hi1y_d": w.idxmax().strftime("%Y-%m-%d") if w.notna().any() else None}
    return out


def headline(M, grid):
    r = M.iloc[-1]
    p, q = int(r.pos), int(r.fear_lv)
    g = grid.get(f"{p}{q}", {})
    st = g.get("all")
    hist = (f"1994 年以来处在这一格 {g['n']} 次（相邻 10 天只算一次），之后 60 天中位数 {pct(st['f60'])}，同一价格位置的平均 {pct(st['row60'])}。"
            if st else "这一格历史样本太少。")
    stage = f"{ma_text(r).split('（')[0]} · {speed_text(r.z20)} · {POS_N[p]}（相当于标普 {pct(r.eq_spx)}）"
    return {"pos": p, "fear": q, "fear_n": int(r.fear_n), "frag_n": int(r.frag_n), "stage": stage, "gauges": gauges_now(M),
            "title": f"{POS_N[p]} × 恐惧{FEAR_N[q]}", "text": CELL_TEXT[(p, q)], "hist": hist}


# ---------------------------------------------------------------- 看板数据
def num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def r6(x, dp=4):
    x = num(x)
    return None if x is None else round(x, dp)


CHART = [("^SOX", "费城半导体"), ("SMH", "SMH"), ("^NDX", "纳指100"), ("^GSPC", "标普500"), ("NVDA", "英伟达"),
         ("^KS11", "KOSPI"), ("000660.KS", "SK 海力士"), ("005930.KS", "三星电子")]


def chart_data(D, M):
    out = []
    for k, n in CHART:
        if k not in D:
            continue
        d = D[k]
        F = M if k == "^SOX" else px_features(d)
        vol = d.v
        vnote = None
        if k == "^SOX" and "SMH" in D:                     # 指数没有成交量，用 SMH 的
            vol = align(D["SMH"].v, d.index, limit=0).fillna(0)
            vnote = "SMH 成交量"
        rows = [[t.strftime("%Y-%m-%d"), r6(o, 2), r6(h, 2), r6(l, 2), r6(c, 2), int(v) if np.isfinite(v) else 0]
                for t, o, h, l, c, v in zip(d.index, d.o, d.h, d.l, d.c, vol)]
        climax = F.index[(F.smh_climax_dn if k == "^SOX" else F.climax_dn).fillna(False).astype(bool).to_numpy()]
        piv = F.attrs.get("pivots", [])
        out.append({"k": k, "n": n, "vnote": vnote, "rows": rows,
                    "rsi": [r6(x, 1) for x in F.rsi], "climax": [t.strftime("%Y-%m-%d") for t in climax],
                    "pivots": [[F.index[i].strftime("%Y-%m-%d"), t, round(p, 2)] for i, t, p in piv[-40:]]})
    return out


def research_labels():
    try:
        with open(RESEARCH_JSON, encoding="utf-8") as f:
            return json.load(f).get("labels", {})
    except Exception:
        return {}


# ---------------------------------------------------------------- 第五轮：能衡量的东西（幅度、下跌风险），只作展示，不进任何判断
REGIME_N = ["200 日线上方 · 50 日线上方", "200 日线上方 · 50 日线下方", "200 日线下方 · 50 日线上方", "200 日线下方 · 50 日线下方"]
IV_PAIR = {"^GSPC": 0, "^NDX": 1, "^SOX": 2}          # research.py M1 检验过的三组，对应 measure["iv"] 的顺序


def research_measure():
    try:
        with open(RESEARCH_JSON, encoding="utf-8") as f:
            return json.load(f).get("measure", {})
    except Exception:
        return {}


def sox_iv(D, M):
    """半导体没有自己的波动指数：VXN × 半导体/纳指的长期波动比（和 research.py M1 同一口径，只用当天以前）。"""
    if "^NDX" not in D:
        return pd.Series(np.nan, index=M.index)
    return M.vxn * M.lv / align(px_features(D["^NDX"]).lv, M.index)


def regime(M):
    """0–3，顺序同 REGIME_N。"""
    return (2 * (M.c < M.ma200) + (M.c < M.ma50)).where(M.ma200.notna() & M.ma50.notna())


def iv_rv_text(x):
    if x is None:
        return "没有实际波动可比"
    return ("保险便宜：期权给的波动比最近实际的还低" if x < 1 else "正常：期权平时就比实际波动贵一点" if x < 1.5
            else "保险偏贵：期权给的波动比最近实际的高一半以上")


def cone(base, iv_pct, n=RV_N):
    """从最新收盘往后 n 个交易日的 ±1、±2 倍预期范围：[t, −2σ, −1σ, +1σ, +2σ]。"""
    s = iv_pct / 100
    return [[t] + [round(base * float(np.exp(k * s * np.sqrt(t / 252))), 2) for k in (-2, -1, 1, 2)] for t in range(1, n + 1)]


def opt_name(code):
    """TSM270115C00380000 → 2027-01-15 到期、行权价 380 的认购"""
    try:
        body = code[-15:]
        d, cp, k = body[:6], body[6], int(body[7:]) / 1000
        return f"20{d[:2]}-{d[2:4]}-{d[4:]} 到期、行权价 {k:g} 的{'认购' if cp == 'C' else '认沽'}"
    except Exception:
        return code


def money_text(v):
    a = abs(v)
    return f"{a / 1e8:.2f} 亿美元" if a >= 1e8 else f"{a / 1e4:,.0f} 万美元"


def measure_now(D, M, options=None):
    rm, r, days = research_measure(), M.iloc[-1], [t.strftime("%Y-%m-%d") for t in M.index]
    siv, rv = sox_iv(D, M), past_rv(M.c) * 100
    ratio = (siv / rv).replace([np.inf, -np.inf], np.nan)
    ratio_pct = exp_pct(ratio).reindex(M.index)
    chains = {m["symbol"]: m for m in ((options or {}).get("chains") or []) if num(m.get("iv30"))}
    cover = {k: rm["iv"][i] for k, i in IV_PAIR.items() if len(rm.get("iv", [])) > i}

    moves = []
    def add(k, name, iv, rv_now, src, date):
        iv, rv_now = num(iv), num(rv_now)
        if iv is None:
            return
        x = iv / rv_now if rv_now else None
        moves.append({"k": k, "name": name, "iv": r6(iv, 1), "week": r6(expected_move(iv, 5)), "month": r6(expected_move(iv, RV_N)),
                      "rv": r6(rv_now, 1), "ratio": r6(x, 2), "read": iv_rv_text(x), "src": src, "date": date})
    add("^SOX", "费城半导体", siv.iloc[-1], rv.iloc[-1], "VXN × 半导体/纳指长期波动比（近似）", days[-1])
    for s, m in chains.items():
        add(s, m.get("name") or s, m["iv30"], past_rv(D[s].c).iloc[-1] * 100 if s in D else None, "期权链 30 天隐含波动", m.get("date"))

    # K 线上的预期范围：指数用波动指数，ETF、个股用自己的期权链；KOSPI 用韩国 ETF EWY（含汇率）
    src = {"^SOX": (lambda ix: align(siv, ix).iloc[-1], "VXN × 半导体/纳指长期波动比"),
           "^NDX": (lambda ix: align(D["^VXN"].c, ix).iloc[-1] if "^VXN" in D else None, "VXN"),
           "^GSPC": (lambda ix: align(D["^VIX"].c, ix).iloc[-1] if "^VIX" in D else None, "VIX")}
    for k, s, nm in (("SMH", "SMH", "SMH 期权链"), ("NVDA", "NVDA", "英伟达期权链"), ("^KS11", "EWY", "韩国 ETF EWY 期权链（含汇率）")):
        if s in chains:
            src[k] = (lambda ix, v=chains[s]["iv30"]: v, nm)
    cones = {}
    for k, (f, nm) in src.items():
        if k not in D:
            continue
        c = D[k].c.dropna()
        iv = num(f(c.index))
        if iv is None or c.empty:
            continue
        cv = cover.get(k)
        cones[k] = {"from": c.index[-1].strftime("%Y-%m-%d"), "base": r6(c.iloc[-1], 2), "iv": r6(iv, 1), "src": nm,
                    "cover": {e: [r6(cv[e]["cover1"], 3), r6(cv[e]["cover2"], 3)] for e in ("pre", "post")} if cv else None,
                    "pts": cone(float(c.iloc[-1]), iv)}

    # 下跌风险：趋势状态 + 历史上这种状态下 60 天内再跌半个年化波动的比例（research.py M2）
    reg = regime(M)
    i = None if not np.isfinite(reg.iloc[-1]) else int(reg.iloc[-1])
    run = 0
    if i is not None:
        for v in reg.iloc[::-1]:
            if v != i:
                break
            run += 1
    RG = rm.get("regimes", {})
    rows = []
    for j in range(4):
        g = RG.get(str(j), {})
        e = {x: g.get(x) or {} for x in ("pre", "post")}
        rows.append({"i": j, "name": REGIME_N[j], **{f"{w}_{x}": r6(((e[x].get("sox") or {}) if w == "sox" else e[x]).get("big"), 3) for w in ("sox", "all") for x in ("pre", "post")},
                     "volx": [r6(((e[x].get("sox") or {}).get("volx")), 2) for x in ("pre", "post")]})
    b2 = (rm.get("risk") or {}).get("below200") or {}
    thr = num(BIG_K * r.lv)
    risk = {"i": i, "name": REGIME_N[i] if i is not None else None, "days": run, "d200": r6(r.c / r.ma200 - 1), "d50": r6(r.c / r.ma50 - 1),
            "thr": r6(thr), "label": (rm.get("labels") or {}).get("below200"), "rows": rows,
            "volr": [r6((b2.get(x) or {}).get("volr"), 2) for x in ("pre", "post")],
            "share": [r6((b2.get(x) or {}).get("share"), 2) for x in ("pre", "post")]}

    # 顶部三件事：白话
    sx = next((m for m in moves if m["k"] == "^SOX"), None)
    smh = next((m for m in moves if m["k"] == "SMH"), None)
    ref = smh or sx
    three = []
    if ref:
        cv = cover.get("^SOX")
        hit = (f"历史上半导体实际一个月的涨跌，约 {cv['pre']['cover1']:.0%}（2015 年前）和 {cv['post']['cover1']:.0%}（2015 年后）的时间落在 ±1 倍范围内，"
               f"几乎都落在 ±2 倍内。" if cv else "")
        extra = f"按 VXN 折算的半导体指数是 ±{sx['month']:.1%}。" if smh and sx else ""
        three.append({"k": "move", "t": "接下来可能波动多大", "big": f"±{ref['month']:.1%}", "unit": "未来一个月",
                      "text": f"{'SMH 期权' if smh else '按 VXN 折算'}给出的预期：一个月 ±{ref['month']:.1%}，一周 ±{ref['week']:.1%}（这是 1 倍范围）。{extra}{hit}",
                      "sub": f"保险价格：隐含 {ref['iv']:.0f}% ÷ 最近 21 天实际 {ref['rv']:.0f}% = {ref['ratio']:.2f}，{ref['read']}。" if ref.get("ratio") else ref["read"],
                      "label": (rm.get("labels") or {}).get("iv_rv"), "jump": "kline-sec"})
    if i is not None:
        me = rows[i]
        best = rows[0]
        lo = min(x for x in (rows[2]["sox_pre"], rows[2]["sox_post"], rows[3]["sox_pre"], rows[3]["sox_post"]) if x is not None) if all(rows[j]["sox_pre"] is not None for j in (2, 3)) else None
        hi = max(x for x in (rows[2]["sox_pre"], rows[2]["sox_post"], rows[3]["sox_pre"], rows[3]["sox_post"]) if x is not None) if lo is not None else None
        txt = f"半导体在 {REGIME_N[i]}（连续 {run} 天；离 200 日线 {pct(risk['d200'])}，离 50 日线 {pct(risk['d50'])}）。"
        if me["sox_pre"] is not None:
            txt += f"历史上这种状态下，60 天内再跌 {thr:.0%} 以上（半个年化波动）的比例：2015 年前 {me['sox_pre']:.0%}，2015 年后 {me['sox_post']:.0%}。"
            if i < 2 and lo is not None:
                txt += f"跌破 200 日线后是 {lo:.0%}–{hi:.0%}。"
            elif i >= 2 and best["sox_pre"] is not None:
                txt += f"两条线都在上方时只有 {best['sox_pre']:.0%}–{best['sox_post']:.0%}。"
        three.append({"k": "risk", "t": "下跌风险高不高", "big": f"{me['sox_post']:.0%}" if me["sox_post"] is not None else "—",
                      "unit": "60 天内再跌 " + (f"{thr:.0%}" if thr else "—") + " 的历史比例（2015 年后）",
                      "text": txt, "sub": "200 日线上下是 33 个品种、两段历史都成立的风险信号；它只说明风险大小，不说明方向（涨跌中位数差不多）。",
                      "label": risk["label"], "jump": "kline-sec", "tone": ["calm", "calm", "alert", "alert"][i]})
    return {"moves": moves, "cones": cones, "risk": risk, "three": three,
            "ratio": {"v": r6(ratio.iloc[-1], 2), "pct": r6(ratio_pct.iloc[-1], 0)},
            "series": {"siv": siv, "rv": rv, "reg": reg}}


def options_read(M, options, moves):
    """期权五问：每问一句结论、一个色调、一句为什么重要。只读已有的数，不做判断。"""
    r, O = M.iloc[-1], options or {}
    CH = {m["symbol"]: m for m in (O.get("chains") or [])}
    nm = lambda s: s if s in ("SMH", "SOXX", "QQQ", "EWY") else (CH.get(s) or {}).get("name") or s     # ETF 用代码，个股用中文名
    out = []
    # 1 预期波动和保险贵贱
    mv = {m["k"]: m for m in moves}
    ref = mv.get("SMH") or mv.get("^SOX")
    if ref:
        vz = f"VIX {r.vix:.1f}（历史 {r.vix_pct:.0f}% 分位）" if np.isfinite(r.vix) and np.isfinite(r.vix_pct) else ""
        tone = "fear" if np.isfinite(r.vix) and r.vix >= 30 else "watch" if (ref.get("ratio") or 0) >= 1.5 or (np.isfinite(r.vix) and r.vix >= 20) else "calm"
        others = "、".join(f"{nm(k)} ±{m['month']:.0%}" for k, m in mv.items() if k in ("NVDA", "QQQ", "EWY"))
        out.append({"q": "市场预期接下来波动多大？保险贵还是便宜？", "tone": tone,
                    "a": f"{nm(ref['k']) if ref['k'] != '^SOX' else '半导体指数'} 一个月 ±{ref['month']:.1%}、一周 ±{ref['week']:.1%}。{ref['read']}（隐含 {ref['iv']:.0f}% vs 最近实际 {ref['rv'] or 0:.0f}%）。" + (f"其他：{others}。" if others else "") + (f"大盘 {vz}。" if vz else ""),
                    "why": "隐含波动就是期权的价格，是市场花真钱给出的波动预期。历史检验里它比用过去的波动去推更准（见说明）。保险贵说明有人愿意多付钱防下跌；保险比实际波动还便宜，说明大家很放心。"})
    # 2 有没有人买下跌保护
    sk = [(s, CH[s]["skew25"]) for s in ("SMH", "NVDA", "QQQ") if s in CH and num(CH[s].get("skew25")) is not None]
    if sk or np.isfinite(r.pc10):
        pp = r.pc_pct if np.isfinite(r.pc_pct) else None
        tone = "fear" if pp is not None and pp >= 90 else "greed" if pp is not None and pp <= 15 else "calm"
        a = ("下跌保护比同样虚值的上涨期权贵：" + "，".join(f"{nm(s)} {v:.1f} 个波动点" for s, v in sk)) if sk else ""
        a = (a + "。" if a else "") + (f"股票期权认沽/认购比 10 日均 {r.pc10:.2f}，历史 {pp:.0f}% 分位" + ("，大量买保护。" if pp >= 90 else "，认购明显更多，偏追涨。" if pp <= 15 else "，不算极端。") if pp is not None else "")
        out.append({"q": "有没有人在买下跌保护？", "tone": tone, "a": a,
                    "why": "偏度 = 25 delta 认沽的隐含波动 − 25 delta 认购的隐含波动，白话就是\"下跌保险比上涨期权贵多少\"。指数通常 3–8 个点，个股更低；突然拉大说明有人抢着买保护。认沽/认购比看的是成交，越高越多人买保护。"})
    # 3 担心眼前还是以后
    if np.isfinite(r.t3):
        inv = [nm(s) for s, m in CH.items() if num(m.get("term")) is not None and m["term"] >= 1.02]
        tone = "fear" if r.t3 >= 1 else "watch" if r.t3 >= 0.95 else "calm"
        a = f"VIX ÷ VIX3M = {r.t3:.2f}" + ("，已经倒挂：恐慌集中在眼前。" if r.t3 >= 1 else "，趋平：眼前的担心在升温。" if r.t3 >= 0.95 else "，没有倒挂：远月比近月贵，是平常状态。")
        if inv and r.t3 < 0.95:
            a += f"但 {'、'.join(inv)} 的近月比 3 个月贵：担心集中在这些标的自己的近期事件（常是财报季）。"
        elif inv:
            a += f"{'、'.join(inv)} 的近月也比 3 个月贵。"
        out.append({"q": "担心的是眼前还是以后？", "tone": tone, "a": a,
                    "why": "平时越远的期权越贵（时间越长不确定越多）。近月反而更贵叫\"倒挂\"，说明大家担心的是这几天就要发生的事。历次大跌的恐慌顶点几乎都伴随倒挂；倒挂解除是恐慌退潮的条件之一。"})
    # 4 大资金押在哪（持仓变化）
    OI = O.get("oi") or {}
    if OI.get("top"):
        new = [x for x in OI["top"] if x["doi"] > 0][:3]
        by = OI.get("by_symbol") or {}
        cs, ps = sum(v.get("c", 0) for v in by.values()), sum(v.get("p", 0) for v in by.values())
        a = "；".join(f"{x['symbol']} {opt_name(x['option'])}，新增 {x['doi']:,.0f} 张（约 {money_text(x['prem'])}）" for x in new)
        a = (f"新增持仓金额最大的：{a}。" if a else "") + f"全部合计：认购持仓 {'+' if cs >= 0 else '−'}{abs(cs):,.0f} 张，认沽 {'+' if ps >= 0 else '−'}{abs(ps):,.0f} 张。"
        out.append({"q": "大资金押在哪？", "tone": "calm", "a": a, "date": f"{OI.get('from')} → {OI.get('to')}",
                    "why": "持仓（未平仓合约）每天收盘后结算一次。持仓增加是新开的仓，减少是平仓。只看成交分不清开仓平仓，看持仓变化才知道新钱押在哪个到期日和价位。深度实值认购常是替代股票的长线做多。"})
    # 5 今天盘中谁在主动买卖
    FL = O.get("flow") or {}
    if FL.get("symbols"):
        S = FL["symbols"].values()
        rough = np.mean([v.get("rough") or 0 for v in S]) >= 0.5
        net = sum(v.get("net") or 0 for v in S)
        cb, pb = sum(v.get("cb") or 0 for v in S), sum(v.get("pb") or 0 for v in S)
        a = (f"今天到目前为止，主动买认购 {money_text(cb)}，主动买认沽 {money_text(pb)}，净看多权利金 {'+' if net >= 0 else '−'}{money_text(net)}。"
             + ("但今天第一次抓取前已经有大量成交，只能粗略归类，不作判断。" if rough else f"整体{'偏多' if net > 0 else '偏空' if net < 0 else '持平'}。"))
        out.append({"q": "今天盘中谁在主动买卖？", "tone": "quiet" if rough else "calm", "a": a, "date": FL.get("day"), "rough": bool(rough),
                    "why": "成交价靠近卖价算主动买，靠近买价算主动卖，用每 30 分钟两次快照之间新增的成交量近似。这部分还没有历史可以检验，只作参考；数据间隔超过 75 分钟会标为粗略。"})
    return out


def build(D=None, options=None):
    D = D or load()
    M = market(D)
    grid = grid_stats(M)
    eps_rows, eps = episode_table(M)
    ana = item_anatomy(M, eps)
    labels = research_labels()
    r = M.iloc[-1]
    items = []
    for k, s in SPEC.items():
        if s["group"] not in ("fear", "frag", "turn"):
            continue
        col = {"fear": "f_", "frag": "r_", "turn": "t_"}[s["group"]] + k
        on = bool(M[col].iloc[-1])
        last = M.index[M[col].to_numpy()]
        items.append({"id": k, "group": s["group"], "name": s["name"], "mech": s["mech"], "src": s["src"], "on": on,
                      "last": last[-1].strftime("%Y-%m-%d") if len(last) else None,
                      "ana": ana.get(k), "label": labels.get(k)})
    combos = []
    for k, cb in COMBOS.items():
        m = cb["f"](M).fillna(False).astype(bool)
        last = M.index[m.to_numpy()]
        combos.append({"id": k, "name": cb["name"], "on": bool(m.iloc[-1]), "label": labels.get(k),
                       "last": last[-1].strftime("%Y-%m-%d") if len(last) else None})
    clear_conds = [["过去 20 天恐惧定价 ≥ 3 条", bool(r.c_recent)], ["VIX÷VIX3M < 0.95", bool(r.c_term and np.isfinite(r.t3))],
                   ["VIX 比 20 日峰值低 20% 以上", bool(r.c_vixoff)], ["价格 5 天没创新低", bool(r.c_nolow)]]
    days = [t.strftime("%Y-%m-%d") for t in M.index]
    daily = {"d": days, "pos": [None if not np.isfinite(x) else int(x) for x in M.pos], "fear": M.fear_n.astype(int).tolist(),
             "frag": M.frag_n.astype(int).tolist(), "clear": M.t_clear.astype(int).tolist(), "thrust": M.t_thrust.astype(int).tolist(),
             "climax": M.t_climax.astype(int).tolist(), "vix": [r6(x, 2) for x in M.vix], "t3": [r6(x, 3) for x in M.t3],
             "pct50": [r6(x, 1) for x in M.br_pct50], "pct200": [r6(x, 1) for x in M.br_pct200],
             "nhnl": [r6(a - b, 0) for a, b in zip(M.br_nh, M.br_nl)], "pc10": [r6(x, 3) for x in M.pc10],
             "sox": [r6(x, 2) for x in M.c], "dd": [r6(x, 4) for x in M.dd]}
    for kind in GAUGES:
        daily["g" + kind] = [r6(x, 0) for x in M[GCOL[kind]]]
        daily["c" + kind] = {k: [None if not np.isfinite(x) else int(round(x * 100)) for x in M[GPFX[kind] + k]] for k, *_ in GAUGES[kind]["items"]}
    daily["gitems"] = {kind: [[k, nm, rng, grp] for k, nm, rng, grp in GAUGES[kind]["items"]] for kind in GAUGES}
    ms = measure_now(D, M, options)
    S = ms.pop("series")
    hd = headline(M, grid)
    gs = hd["gauges"]
    ms["three"].append({"k": "mood", "t": "情绪走到哪一步", "big": hd["title"], "unit": "价格位置 × 恐惧定价",
                        "text": f"恐惧定价 {hd['fear_n']}/5 条，脆弱度 {hd['frag_n']}/6 条。{CELL_TEXT[(hd['pos'], hd['fear'])]}",
                        "sub": "三个温度（只作描述）：" + "，".join(f"{g['name']} {g['v'] if g['v'] is not None else '—'}" for g in gs.values()) + "。",
                        "label": None, "jump": "overview", "tone": "fear" if hd["fear"] == 2 else "greed" if hd["frag_n"] >= 4 else "calm"})
    daily["siv"], daily["rv"], daily["reg"] = [r6(x, 1) for x in S["siv"]], [r6(x, 1) for x in S["rv"]], [None if not np.isfinite(x) else int(x) for x in S["reg"]]
    smh_iv = {row[0]: row[1] for row in ((options or {}).get("hist") or {}).get("SMH", [])}
    chg = M.c.pct_change() * 100
    z1 = np.log(M.c).diff() / (M.lv / SQ)
    table = []
    for k in range(max(1, len(M) - 60), len(M)):
        q, d = M.iloc[k], days[k]
        table.append([d, r6(q.c, 2), r6(chg.iloc[k], 2), r6(z1.iloc[k], 1), r6(S["rv"].iloc[k], 1), r6(q.vix, 1), r6(q.vxn, 1), r6(S["siv"].iloc[k], 1),
                      r6(smh_iv.get(d), 1), r6(q.g_fear, 0), r6(q.g_sell, 0), r6(q.g_hot, 0), int(q.fear_n), int(q.frag_n),
                      daily["reg"][k], daily["pos"][k]])
    vixtab = []
    for nm, v, p, note in (("VIX", r.vix, r.vix_pct, vix_zone(r.vix)), ("VIX ÷ VIX3M", r.t3, r.t3_pct, term_zone(r.t3)),
                           ("VIX9D ÷ VIX", r.t9, np.nan, "远低于 1 说明眼前很放心" if np.isfinite(r.t9) and r.t9 < 0.9 else "接近或高于 1：眼前的风险被定价得更贵" if np.isfinite(r.t9) else "—"),
                           ("VXN（纳指）", r.vxn, r.vxn_pct, vix_zone(r.vxn)), ("VVIX", r.vvix, r.vvix_pct, "VIX 自己的波动，急升常领先 VIX"),
                           ("SKEW", r.skew, r.skew_pct, "深度虚值认沽的贵贱，牛市里也常偏高"),
                           ("股票认沽/认购比 10 日均", r.pc10, r.pc_pct, "越高越多人买保护"), ("指数认沽/认购比 10 日均", r.pcidx10, np.nan, "机构对冲为主，只作参考"),
                           ("Baa 信用利差 %", r.baa, r.baa_pct, f"20 天 {f2(r.baa20, 0)}bp"), ("高收益利差 %", r.hyoas, np.nan, f"20 天 {f2(r.hyoas20, 0)}bp（只有近 3 年）"),
                           ("HYG÷IEF 20 日 %", r.hyief20, np.nan, "含分红；跌超 2% 是明显收紧")):
        vixtab.append([nm, r6(v, 3), r6(p, 0), note])
    return {
        "asof": days[-1], "head": hd, "layers": layers(M, D), "items": items, "combos": combos, "clear": clear_conds,
        "grid": grid, "episodes": eps_rows, "chart": chart_data(D, M), "daily": daily, "vixtab": vixtab,
        "now": {"vix": r6(r.vix, 2), "t3": r6(r.t3, 3), "pc10": r6(r.pc10, 3), "baa": r6(r.baa, 2), "dd": r6(r.dd), "eq": r6(r.eq_spx),
                "lv": r6(r.lv), "spx_lv": r6(r.spx_lv), "pct50": r6(r.br_pct50, 1), "fear_avail": int(r.fear_avail)},
        "options": options, "measure": ms, "table": table, "oread": options_read(M, options, ms["moves"]),
    }, M


# ---------------------------------------------------------------- 样本外记录
LOG_CSV = os.path.join(HERE, "daily_log.csv")
LOG_COLS = ["date", "built", "g_fear", "g_sell", "g_hot", "pos", "fear_n", "frag_n", "clear", "fear_on", "frag_on", "sox", "dd", "eq_spx", "z20", "rsi",
            "vix", "t3", "pc10", "baa", "baa20", "pct50", "pct200"]


def _log_row(r, d, built):
    row = {"date": d, "built": built, "pos": POS_N[int(r.pos)] if np.isfinite(r.pos) else "", "fear_n": int(r.fear_n), "frag_n": int(r.frag_n),
           "clear": int(bool(r.t_clear)), "fear_on": " ".join(k for k in FEAR if r["f_" + k]), "frag_on": " ".join(k for k in FRAG if r["r_" + k])}
    for k, col, dp in (("sox", "c", 2), ("dd", "dd", 4), ("eq_spx", "eq_spx", 4), ("z20", "z20", 2), ("rsi", "rsi", 1), ("vix", "vix", 2), ("t3", "t3", 3),
                       ("pc10", "pc10", 3), ("baa", "baa", 2), ("baa20", "baa20", 0), ("pct50", "br_pct50", 1), ("pct200", "br_pct200", 1), ("g_fear", "g_fear", 0), ("g_sell", "g_sell", 0), ("g_hot", "g_hot", 0)):
        v = r6(r[col], dp)
        row[k] = "" if v is None else v
    return row


def _closing_row(row, d):
    """这一行是不是在那天收盘以后记的（或者是补记的）。盘中记的行之后会被收盘数据覆盖。"""
    if not row:
        return False
    b = row.get("built") or ""
    if b.startswith("补记"):
        return True
    try:
        from zoneinfo import ZoneInfo
        ny = pd.Timestamp(b).tz_convert(ZoneInfo("America/New_York"))
    except Exception:
        return False
    return ny.strftime("%Y-%m-%d") > d or (ny.strftime("%Y-%m-%d") == d and (ny.hour, ny.minute) >= (16, 5))


def append_log(M, built):
    """每个交易日一行（盘中多次运行时覆盖当天那一行），记录当时看到的读数，以后用来检验样本外表现。
    日志开始以后漏掉的交易日（比如那天运行失败、Mac 一直睡着）会补上，built 标"补记"。
    全量计算和截到当天的计算完全一致（--selftest 检查），所以补记的值就是那天收盘时会看到的读数。"""
    import csv
    d = M.index[-1].strftime("%Y-%m-%d")
    rows = {}
    if os.path.exists(LOG_CSV):
        with open(LOG_CSV, encoding="utf-8") as f:
            for x in csv.DictReader(f):
                rows[x["date"]] = x
    if rows:
        first = min(rows)
        for t in M.index[(M.index > pd.Timestamp(first)) & (M.index < M.index[-1])]:
            ds = t.strftime("%Y-%m-%d")
            if ds not in rows:
                rows[ds] = _log_row(M.loc[t], ds, "补记 " + built)
    if not _closing_row(rows.get(d), d):        # 收盘后已经记过的那一行不再覆盖：保留"当时看到的读数"
        rows[d] = _log_row(M.iloc[-1], d, built)
    tmp = LOG_CSV + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS, extrasaction="ignore")
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])
    os.replace(tmp, LOG_CSV)


# ---------------------------------------------------------------- 自检
CHECK_COLS = ["g_fear", "g_sell", "g_hot", "g_fear_pct", "g_sell_pct", "dd", "ddv", "extv", "z20", "acc", "rsi", "swing", "brk", "vix_pct", "t3_pct", "pc_pct", "baa_pct", "bb_pct", "br_pct50", "br_pct200",
              "br_zema", "fear_n", "frag_n", "pos", "eq_spx", "t_clear", "t_thrust", "t_climax", "smh_relv", "lv"]


def selftest(n=10, seed=7):
    D = load()
    full = market(D)
    extra = lambda D_, M_: pd.DataFrame({"siv": sox_iv(D_, M_), "rv21": past_rv(M_.c), "reg": regime(M_)})   # 第五轮的展示列也要过检
    full = full.join(extra(D, full))
    rng = np.random.default_rng(seed)
    cand = full.index[(full.index >= "2008-01-01")][:-5]
    picks = sorted(rng.choice(len(cand), n, replace=False))
    bad = 0
    for k in picks:
        t = cand[k]
        Dt = until(D, t)
        part = market(Dt)
        part = part.join(extra(Dt, part))
        cols = CHECK_COLS + ["siv", "rv21", "reg"]
        a, b = full.loc[t, cols].astype(float), part.loc[t, cols].astype(float)
        diff = [c for c in cols if not ((np.isnan(a[c]) and np.isnan(b[c])) or np.isclose(a[c], b[c], rtol=1e-9, atol=1e-9))]
        print(t.date(), "一致" if not diff else "不一致：" + ", ".join(f"{c} 全量 {a[c]} 截断 {b[c]}" for c in diff))
        bad += bool(diff)
    print("自检通过：截断重算和全量计算完全一致。" if not bad else f"自检失败：{bad} 个日期不一致。")
    return bad == 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    data, M = build()
    h = data["head"]
    print(f"数据截至 {data['asof']}")
    print(f"{h['title']}｜{h['stage']}")
    print(h["text"])
    print(h["hist"])
    print(f"恐惧定价 {h['fear_n']}/5，脆弱度 {h['frag_n']}/6")
    for g in h["gauges"].values():
        print(f"{g['name']} {g['v']}：{g['text']}｜" + "、".join(f"{a} {b}" for a, b, *_ in g["items"]))
    for L in data["layers"]:
        print(f"- {L['name']}：{L['text']}")
    for it in data["items"]:
        a = it["ana"]
        hit = f"  历次{'底' if it['group'] == 'fear' else '顶'}部 {a['hit']}/{a['n']}（任意一天 {a['base'] * 100:.0f}%）" if a else ""
        print(f"  [{'亮' if it['on'] else '  '}] {it['name']}{hit}")
