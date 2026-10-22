#!/usr/bin/env python3
"""科技情绪全景 · 强信号通知（Mac 桌面通知；设了环境变量 NTFY_TOPIC 时同时推到手机 ntfy App）

build_sentiment.py --notify 在每次刷新后调用。口径全部来自 engine.py（2026-10-20 第四轮）。只推这几条，每条附一句历史参考：
  1. 恐惧充分定价：恐惧定价第一次达到 3 条（前 10 个交易日都没有）
  2. 恐慌退潮确认：恐惧充分定价之后，期限结构回落、VIX 从峰值回落 20%、价格 5 天没创新低
  3. 脆弱度 ≥ 4 条（第一天）
  4. VIX 期限结构倒挂：VIX ÷ VIX3M 今天 > 1，前一天还没有
  5. VIX 单日异常跳升：今天的变化是过去一年日变化的 2 倍标准差以上
  6. 费城半导体单日跌超 4%
  7. 恐慌抛售：SMH 放量 2 倍下跌且 RSI ≤ 30
同一条信号同一个交易日只推一次（记录在 .notify_state.json）。盘中运行时用的是实时价，收盘前可能还会变。

第六轮（2026-10-23）加两样：
  - 每日摘要：每个交易日收盘后推一条：情绪分、三个因子、周期位置、今天动了什么、期权给的下个月幅度、下跌风险（_digest）。
  - 看门狗：美东 18:30 以后，看板的数据日期还不是最近一个交易日，推"数据没更新"（_stale，每天最多一次）。

单独测试：python3 notify.py --test   （发一条测试通知）
         python3 notify.py --dry    （只打印现在会推什么，包括每日摘要，不弹通知）
"""
import json, os, subprocess, sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, ".notify_state.json")
sys.path.insert(0, HERE)
import engine as E  # noqa: E402


NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "").strip()      # 云端（GitHub Actions）从仓库的 Secrets 传进来
SITE_URL = os.environ.get("SITE_URL", "").strip()          # 点通知打开看板


def ntfy(title, msg):
    """ntfy.sh 免费推送：手机装 ntfy App 订阅同一个频道名就能收到。频道名就是密码，别公开。"""
    import urllib.request
    body = {"topic": NTFY_TOPIC, "title": f"科技情绪全景 · {title}", "message": msg, "tags": ["bar_chart"]}
    if SITE_URL:
        body["click"] = SITE_URL
    req = urllib.request.Request("https://ntfy.sh/", data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=20).read()
    except Exception as e:
        print("ntfy 推送失败：", e)


def send(title, msg):
    if sys.platform == "darwin":
        q = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')
        subprocess.run(["osascript", "-e", f'display notification "{q(msg)}" with title "科技情绪全景" subtitle "{q(title)}" sound name "Glass"'], check=False)
    if NTFY_TOPIC:
        ntfy(title, msg)


def ref(mask, M, label=None):
    """1995 年以来这条信号之后的表现，和同样跌深的日子比（相邻 10 天只算一次，不含今天）。"""
    m = np.asarray(mask, bool).copy()
    m[-1] = False
    st = E.event_stats(m, M.c, M.ddv, valid_from="1995-06-01").get("all", {})
    if st.get("n", 0) < 3:
        return ""
    ex = st.get("ex60")
    s = f"1995 年以来 {st['n']} 次，之后 60 天中位数 {st['f60'] * 100:+.1f}%" + (f"，比同样跌深的日子 {ex * 100:+.1f}%" if ex is not None else "") + "。"
    return s + (f"（{label}）" if label else "")


def signals():
    D = E.load()
    M = E.market(D)
    labels = E.research_labels()
    i = len(M) - 1
    day = M.index[i].strftime("%Y-%m-%d")
    r = M.iloc[i]
    out = []

    def fresh(mask):   # 今天是这一段的第一天（前 10 个交易日都没有）
        m = np.asarray(mask, bool)
        return bool(m[i]) and not m[max(0, i - E.DEDUP):i].any()

    f3 = (M.fear_n >= 3).to_numpy()
    if fresh(f3):
        on = "、".join(E.SPEC[k]["name"] for k in E.FEAR if r["f_" + k])
        out.append(("fear3", "恐惧充分定价", f"{int(r.fear_n)}/5 条：{on}。价格位置：{E.POS_N[int(r.pos)]}（相当于标普 {E.pct(r.eq_spx)}）。" + ref(f3, M)))
    cl = M.t_clear.to_numpy()
    if fresh(cl):
        out.append(("clear", "恐慌退潮确认", f"恐惧充分定价之后，VIX 已从 20 日峰值回落 20% 以上，期限结构 {E.f2(r.t3, 3)}，价格 5 天没创新低。" + ref(cl, M, labels.get("clear"))))
    r4 = (M.frag_n >= 4).to_numpy()
    if fresh(r4):
        on = "、".join(E.SPEC[k]["name"].split("（")[0] for k in E.FRAG if r["r_" + k])
        out.append(("frag4", "脆弱度 ≥ 4 条", f"{int(r.frag_n)}/6：{on}。只说明上涨在变脆，不说明哪天见顶。" + ref(r4, M, labels.get("frag4"))))

    t3 = M.t3.to_numpy()
    inv = np.isfinite(t3) & (t3 > 1)
    inv_new = np.r_[False, inv[1:] & ~inv[:-1]]
    if inv_new[i]:
        out.append(("t3", "VIX 期限结构倒挂", f"VIX ÷ VIX3M = {t3[i]:.3f}，30 天波动率高过 3 个月，恐慌集中在眼前。" + ref(inv_new, M)))

    vix = D.get("^VIX")
    if vix is not None and len(vix) > 260:
        d = vix.c.diff()
        z = (d - d.shift(1).rolling(252).mean()) / d.shift(1).rolling(252).std()
        za = E.align(z, M.index).to_numpy()
        da = E.align(d, M.index).to_numpy()
        if np.isfinite(za[i]) and za[i] >= 2:
            out.append(("vixz", "VIX 单日异常跳升", f"VIX 这天 {da[i]:+.2f} 点，是过去一年日波动的 {za[i]:.1f} 倍标准差。" + ref(np.isfinite(za) & (za >= 2), M)))

    r1 = (M.c / M.c.shift(1) - 1).to_numpy()
    if np.isfinite(r1[i]) and r1[i] <= -0.04:
        out.append(("sox4", "费城半导体单日跌超 4%", f"今天 {r1[i] * 100:+.1f}%。" + ref(np.isfinite(r1) & (r1 <= -0.04), M)))

    cx = M.t_climax.to_numpy()
    if cx[i]:
        out.append(("climax", "恐慌抛售（SMH 放量大跌）", f"成交量是 50 日均量的 {E.f2(r.smh_relv)} 倍、RSI {E.f2(r.smh_rsi, 0)}。" + ref(cx, M, labels.get("climax"))))
    return day, out


def ny_now():
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York"))


def last_closed(now, hm=(16, 15)):
    """最近一个已经收盘（16:15 以后）的工作日。美股休市日不认识，会当成交易日。"""
    d = now.date()
    if d.weekday() >= 5 or (now.hour, now.minute) < hm:
        d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.isoformat()


def digest(page):
    """收盘后的每日摘要：一条消息说清今天。数字都来自 engine.build()，这里只拼文字。"""
    S, F, st = page.get("score") or {}, page.get("factors") or {}, page.get("state") or {}
    lines = []
    if S.get("v") is not None:
        lines.append(f"情绪分 {S['v']:+d}（{S.get('lean', '')}）" + (f"，5 天 {S['d5']:+.0f}" if S.get("d5") is not None else ""))
    fs = [f"{f['name']} {f['v']:+d}" if f.get("two") else f"{f['name']} {f['v']}" for f in F.values() if f.get("v") is not None]
    if fs:
        lines.append("；".join(f"{x}（{f['word']}）" for x, f in zip(fs, [f for f in F.values() if f.get("v") is not None])))
    if st:
        lines.append(f"周期位置：{st['name']}（第 {st['days']} 天）" + (f"，{'、'.join(st['sub'])}" if st.get("sub") else ""))
    ch = page.get("changes") or []
    lines.append("今天：" + ("；".join(c["t"] for c in ch[:4]) if ch else "没有指标的灯亮起或熄灭"))
    ms = page.get("measure") or {}
    for c in ms.get("three") or []:
        if c.get("k") == "move":
            lines.append(f"期权给的下个月预期波动 {c['big']}（只说幅度，不说方向）")
    rk = ms.get("risk") or {}
    if rk.get("name") and rk.get("i") is not None:
        row = (rk.get("rows") or [{}] * 4)[rk["i"]]
        p = row.get("sox_post")
        lines.append(f"趋势：{rk['name']}" + (f"，历史上 60 天内再跌 {rk['thr'] * 100:.0f}% 的比例 {p * 100:.0f}%" if p is not None and rk.get("thr") else ""))
    return f"{page.get('asof')} 收盘", "\n".join(lines)


def run(dry=False, page=None):
    day, out = signals()
    try:
        state = json.load(open(STATE, encoding="utf-8"))
    except Exception:
        state = {}
    sent = 0
    # 每日摘要和看门狗（第六轮）
    now = ny_now()
    exp = last_closed(now)
    if dry and day != exp:                                  # --dry 时不管是不是收盘后，都打印一份摘要预览
        page = page or E.build()[0]
        title, msg = digest(page)
        print(f"[每日摘要预览（收盘后才会真的推送；现在最近收盘的交易日是 {exp}）] {title}\n{msg}")
    if day == exp and (dry or state.get("_digest") != day):
        if page is None:
            page = E.build()[0]
        title, msg = digest(page)
        print(f"[每日摘要] {title}\n{msg}")
        if not dry:
            send(title, msg)
            state["_digest"] = day
            sent += 1
    elif day < last_closed(now, (18, 30)) and state.get("_stale") != exp:
        msg = f"看板的数据停在 {day}，最近一个交易日 {exp} 的收盘数据还没取到。可能是数据源没更新或自动更新没跑；美股休市日会误报。"
        print(f"[看门狗] {msg}")
        if not dry:
            send("数据没更新", msg)
            state["_stale"] = exp
            sent += 1
    for key, title, msg in out:
        if state.get(key) == day:
            continue
        print(f"[通知] {title}：{msg}")
        if not dry:
            send(title, msg)
            state[key] = day
            sent += 1
    if not out:
        print(f"[通知] {day} 没有强信号。")
    if not dry:
        with open(STATE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
    return sent


if __name__ == "__main__":
    if "--test" in sys.argv:
        send("测试通知", "通知可以正常弹出。之后只有出现强信号时才会提醒你。")
    else:
        run(dry="--dry" in sys.argv)
