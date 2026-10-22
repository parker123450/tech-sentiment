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

单独测试：python3 notify.py --test   （发一条测试通知）
         python3 notify.py --dry    （只打印现在会推什么，不弹通知）
"""
import json, os, subprocess, sys
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


def run(dry=False):
    day, out = signals()
    try:
        state = json.load(open(STATE, encoding="utf-8"))
    except Exception:
        state = {}
    sent = 0
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
