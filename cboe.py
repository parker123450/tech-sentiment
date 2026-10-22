#!/usr/bin/env python3
"""科技情绪全景 · CBOE 期权数据（只用 Python 标准库）

两类数据：
  1. 认沽/认购比（put/call ratio）日度历史：股票期权、指数期权、全部。
     2006-11 到 2019-10 用 CBOE 的历史 CSV，之后按天读 CBOE 的每日统计 JSON。
     抓过的存进 pc_history.csv，之后每次只补新的日子。
  2. 期权链快照（CBOE 延迟 15 分钟报价）：半导体 ETF、龙头股、纳指、韩国 ETF。
     算出 30 天隐含波动、期限结构、25 delta 偏度、认沽认购成交比和持仓比。
     没有历史可下载，所以每天存一行到 options_history.csv，慢慢攒出自己的历史。
"""
import csv, gzip, json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

UA = {"User-Agent": "Mozilla/5.0"}
PC_CSV = {"equity": "equitypc", "index": "indexpc", "total": "totalpc"}
PC_JSON_NAME = {"equity": "EQUITY PUT/CALL RATIO", "index": "INDEX PUT/CALL RATIO", "total": "TOTAL PUT/CALL RATIO"}
CHAIN_SYMBOLS = [("SMH", "半导体 ETF"), ("SOXX", "半导体 ETF（iShares）"), ("NVDA", "英伟达"), ("AVGO", "博通"), ("AMD", "AMD"),
                 ("TSM", "台积电"), ("MU", "美光"), ("QQQ", "纳指 100 ETF"), ("EWY", "韩国 ETF")]


def _get(url, timeout=40):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


# ---------------------------------------------------------------- put/call history
def _pc_from_csv(kind):
    txt = _get(f"https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/{PC_CSV[kind]}.csv")
    out = {}
    for line in txt.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 5 or not p[0][:1].isdigit():
            continue
        try:
            d = datetime.strptime(p[0], "%m/%d/%Y").strftime("%Y-%m-%d")
            out[d] = (float(p[1]), float(p[2]), float(p[4]))  # call, put, ratio
        except ValueError:
            pass
    return out


def _pc_from_json(d):
    try:
        j = json.loads(_get(f"https://cdn.cboe.com/data/us/options/market_statistics/daily/{d}_daily_options", timeout=20))
    except Exception:
        return d, None  # 休市日或还没发布
    row = {}
    ratios = {r["name"]: r["value"] for r in j.get("ratios", [])}
    for kind, name in PC_JSON_NAME.items():
        sec = {"equity": "EQUITY OPTIONS", "index": "INDEX OPTIONS", "total": "SUM OF ALL PRODUCTS"}[kind]
        vol = next((x for x in j.get(sec, []) if x.get("name") == "VOLUME"), None)
        try:
            row[kind] = (float(vol["call"]), float(vol["put"]), float(ratios[name])) if vol else None
        except (KeyError, TypeError, ValueError):
            row[kind] = None
    return d, row


def put_call_history(cache_path, log=print):
    """returns {kind: [(date, ratio, call, put), ...]} and keeps cache_path up to date"""
    rows, tried = {}, set()
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                tried.add(r["date"])
                if r["equity"]:
                    rows[r["date"]] = {k: (float(r[k + "_call"]), float(r[k + "_put"]), float(r[k])) if r[k] else None for k in PC_CSV}
    else:
        log("  第一次运行：下载 2006–2019 年的认沽/认购比历史 CSV")
        hist = {k: _pc_from_csv(k) for k in PC_CSV}
        for d in hist["equity"]:
            rows[d] = {k: hist[k].get(d) for k in PC_CSV}
            tried.add(d)
    start = date(2019, 10, 5)
    today = date.today()
    todo = []
    d = start
    while d <= today:
        s = d.isoformat()
        if d.weekday() < 5 and s not in tried:
            todo.append(s)
        d += timedelta(days=1)
    # 最近 5 个工作日即使试过也再试一次（当天的数据收盘后才发布）
    recent = [x for x in sorted(tried) if x >= (today - timedelta(days=7)).isoformat() and x not in rows]
    todo = sorted(set(todo) | set(recent))
    if todo:
        log(f"  补抓认沽/认购比 {len(todo)} 天" + ("（第一次会多花一两分钟）" if len(todo) > 50 else ""))
        with ThreadPoolExecutor(8) as ex:
            for d, row in ex.map(_pc_from_json, todo):
                tried.add(d)
                if row and row.get("equity"):
                    rows[d] = row
    with open(cache_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date"] + [c for k in PC_CSV for c in (k, k + "_call", k + "_put")])
        for d in sorted(tried):
            r = rows.get(d)
            if not r:
                w.writerow([d] + [""] * 9)  # 记下休市日，下次不再抓
                continue
            line = [d]
            for k in PC_CSV:
                v = r.get(k)
                line += [v[2], v[0], v[1]] if v else ["", "", ""]
            w.writerow(line)
    out = {k: [] for k in PC_CSV}
    for d in sorted(rows):
        for k in PC_CSV:
            v = rows[d].get(k)
            if v and v[2] > 0:
                out[k].append((d, v[2], v[0], v[1]))
    return out


# ---------------------------------------------------------------- option chain snapshot
def _parse(sym):
    # ROOT + YYMMDD + C/P + strike×1000 (8 位)
    tail = sym[-15:]
    return datetime.strptime(tail[:6], "%y%m%d").date(), tail[6], int(tail[7:]) / 1000


def _interp(points, x):
    """linear interpolation on sorted (x, y) points; clamps at the ends"""
    pts = sorted(points)
    if not pts:
        return None
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y0


def fetch_chain(sym):
    return json.loads(_get(f"https://cdn.cboe.com/api/global/delayed_quotes/options/{sym}.json", timeout=60))


def chain_metrics(sym, j=None):
    j = j or fetch_chain(sym)
    D = j["data"]
    ts = datetime.strptime(j["timestamp"][:10], "%Y-%m-%d").date()
    spot = D.get("current_price") or D.get("close")
    by_exp = {}
    cv = pv = coi = poi = 0.0
    for o in D["options"]:
        try:
            exp, cp, k = _parse(o["option"])
        except Exception:
            continue
        dte = (exp - ts).days
        vol, oi = o.get("volume") or 0, o.get("open_interest") or 0
        if cp == "C": cv += vol; coi += oi
        else: pv += vol; poi += oi
        iv = o.get("iv") or 0
        if dte < 2 or iv <= 0.01 or iv > 5:
            continue
        by_exp.setdefault(dte, []).append((cp, k, iv, o.get("delta") or 0))
    atm, skew = [], []
    for dte, opts in sorted(by_exp.items()):
        near = sorted(opts, key=lambda x: abs(x[1] - spot))[:4]          # 最接近现价的认购和认沽
        if len(near) >= 2 and abs(near[0][1] / spot - 1) < 0.05:
            atm.append((dte, sum(x[2] for x in near[:2]) / 2 * 100))
        puts = [x for x in opts if x[0] == "P" and -0.40 < x[3] < -0.10]
        calls = [x for x in opts if x[0] == "C" and 0.10 < x[3] < 0.40]
        if puts and calls:
            p = min(puts, key=lambda x: abs(x[3] + 0.25)); c = min(calls, key=lambda x: abs(x[3] - 0.25))
            if abs(p[3] + 0.25) < 0.08 and abs(c[3] - 0.25) < 0.08:
                skew.append((dte, (p[2] - c[2]) * 100))
    iv30 = D.get("iv30") or _interp(atm, 30)
    iv_near, iv_3m = _interp(atm, 30), _interp(atm, 91)
    traded = (D.get("last_trade_time") or "")[:10] or ts.isoformat()   # 盘前抓到的是上一个交易日收盘的数据
    return {"symbol": sym, "date": traded, "spot": spot, "chg_pct": D.get("price_change_percent"),
            "iv30": iv30, "iv30_chg": D.get("iv30_change"),
            "term": (iv_near / iv_3m) if iv_near and iv_3m else None,      # >1 = 近月比 3 个月贵，恐慌集中在眼前
            "skew25": _interp(skew, 30),                                     # 25 delta 认沽 IV − 认购 IV，波动率点
            "pc_vol": pv / cv if cv else None, "pc_oi": poi / coi if coi else None,
            "vol": cv + pv, "atm_curve": [(d, round(v, 2)) for d, v in atm if d <= 400]}


# ---------------------------------------------------------------- who is buying and who is selling (approximation)
# CBOE 的免费期权链没有逐笔成交方向。近似办法：两次快照之间每个合约新增的成交量，
# 按快照那一刻的最后成交价靠近卖价（主动买）还是买价（主动卖）来分；夹在中间的记为"中间价"。
# 局限：一段时间的成交只能按那一刻的最后成交价归类；数据延迟约 15 分钟；组合单拆不开。

def _side(last, bid, ask):
    if not last or last <= 0 or bid is None or ask is None or ask <= 0 or ask <= bid:
        return "m"
    sp = ask - bid
    return "b" if last >= ask - 0.25 * sp else "s" if last <= bid + 0.25 * sp else "m"


def _dte(name, day):
    try:
        return (_parse(name)[0] - datetime.strptime(day, "%Y-%m-%d").date()).days
    except Exception:
        return 999


def _trade_day(j):
    return (j["data"].get("last_trade_time") or j["timestamp"])[:10]


ROUGH_GAP = 75     # 分钟。定时任务每 30 分钟一次，GitHub 的定时常晚 5–30 分钟，所以 75 分钟以内都算连续


def update_flows(chains, flow_dir, log=print):
    """chains: [(symbol, raw json)]. Adds the volume traded since the previous snapshot, contract by contract, to that
    contract's trading day. A stretch is marked rough when the previous snapshot is more than ROUGH_GAP minutes old (or
    there is none), because its whole volume then gets classified by a single quote."""
    os.makedirs(flow_dir, exist_ok=True)
    now_dt = datetime.now().astimezone()
    now = now_dt.isoformat(timespec="minutes")
    seen_path = os.path.join(flow_dir, "seen.json")
    seen = json.load(open(seen_path, encoding="utf-8")) if os.path.exists(seen_path) else {}
    states = {}
    def state(day):
        if day not in states:
            sp = os.path.join(flow_dir, f"state_{day}.json")
            states[day] = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else {"day": day, "polls": [], "sym": {}}
        return states[day]
    def sym_state(day, sym):
        return state(day)["sym"].setdefault(sym, {"acc": {k: 0.0 for k in ("cb", "cs", "cm", "pb", "ps", "pm")}, "rough_prem": 0.0, "con": {}, "curve": []})
    for sym, j in chains:
        sv = seen.setdefault(sym, {"t": None, "c": {}})
        gap = None
        if sv["t"]:
            try:
                gap = (now_dt - datetime.fromisoformat(sv["t"])).total_seconds() / 60
            except ValueError:
                gap = None
        rough = gap is None or gap > ROUGH_GAP
        for o in j["data"]["options"]:
            name, vol = o.get("option"), o.get("volume") or 0
            d = (o.get("last_trade_time") or "")[:10]
            if not name or not d:
                continue
            prev = sv["c"].get(name)
            dv = vol - prev[1] if prev and prev[0] == d else vol      # 换了交易日：从零开始
            sv["c"][name] = [d, vol]
            if dv <= 0 or _dte(name, d) <= 1:                          # 当天和次日到期的（末日期权）以日内投机为主，不计入
                continue
            last = o.get("last_trade_price") or 0
            side = _side(last, o.get("bid"), o.get("ask"))
            prem = dv * last * 100                                   # 权利金（美元），一张合约 100 股
            S = sym_state(d, sym)
            S["acc"][("c" if name[-9] == "C" else "p") + side] += prem
            if rough:
                S["rough_prem"] += prem
            c = S["con"].setdefault(name, [0, 0, 0])
            c["bsm".index(side)] += dv
        sv["t"] = now
        day = _trade_day(j)
        S = sym_state(day, sym)
        a = S["acc"]
        S["curve"].append([now, round((a["cb"] - a["cs"]) - (a["pb"] - a["ps"]))])
        if now not in state(day)["polls"]:
            state(day)["polls"].append(now)
    for day, st in states.items():
        with open(os.path.join(flow_dir, f"state_{day}.json"), "w", encoding="utf-8") as f:
            json.dump(st, f, separators=(",", ":"))
    with open(seen_path, "w", encoding="utf-8") as f:
        json.dump(seen, f, separators=(",", ":"))
    # 只留最近 5 天的状态文件（每天的汇总另存在 options_flow_daily.csv）
    old = sorted(x for x in os.listdir(flow_dir) if x.startswith("state_"))
    for x in old[:-5]:
        os.remove(os.path.join(flow_dir, x))
    _save_daily(flow_dir)
    return summarize_flows(chains, flow_dir)


def _save_daily(flow_dir):
    path = os.path.join(os.path.dirname(flow_dir.rstrip("/")), "options_flow_daily.csv")
    rows = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows[(r["date"], r["symbol"])] = r
    for x in sorted(os.listdir(flow_dir)):
        if not x.startswith("state_"):
            continue
        st = json.load(open(os.path.join(flow_dir, x), encoding="utf-8"))
        for sym, S in st["sym"].items():
            a = S["acc"]
            tot = sum(a.values())
            rows[(st["day"], sym)] = {"date": st["day"], "symbol": sym, **{k: round(a[k]) for k in a},
                                      "net": round((a["cb"] - a["cs"]) - (a["pb"] - a["ps"])), "polls": len(S["curve"]),
                                      "rough_share": round(S.get("rough_prem", 0) / tot, 3) if tot else ""}
    cols = ["date", "symbol", "cb", "cs", "cm", "pb", "ps", "pm", "net", "polls", "rough_share"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for k in sorted(rows):
            w.writerow({c: rows[k].get(c, "") for c in cols})
    for x in sorted(x for x in os.listdir(flow_dir) if x.startswith("state_"))[:-5]:   # 每天的汇总已经写进上面的 CSV，逐合约的状态只留最近 5 天
        os.remove(os.path.join(flow_dir, x))


def save_contracts(chains, folder):
    """near-the-money contracts once per snapshot date, so the next day can see open interest change"""
    os.makedirs(folder, exist_ok=True)
    by_date = {}
    for sym, j in chains:
        D = j["data"]; spot = D.get("current_price") or D.get("close") or 0
        snap = datetime.strptime(j["timestamp"][:10], "%Y-%m-%d").date()
        for o in D["options"]:
            try:
                exp, cp, k = _parse(o["option"])
            except Exception:
                continue
            if spot <= 0 or (exp - snap).days > 180 or abs(k / spot - 1) > 0.3:
                continue
            by_date.setdefault(snap.isoformat(), []).append([sym, o["option"], o.get("open_interest") or 0, o.get("volume") or 0, o.get("last_trade_price") or 0, round(o.get("delta") or 0, 3), round(o.get("iv") or 0, 4)])
    for d, rows in by_date.items():
        with gzip.open(os.path.join(folder, f"{d}.csv.gz"), "wt", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "option", "oi", "volume", "last", "delta", "iv"])
            w.writerows(rows)
    files = sorted(x for x in os.listdir(folder) if x.endswith(".csv.gz"))
    for x in files[:-120]:      # 留最近 120 天
        os.remove(os.path.join(folder, x))


def oi_change(folder):
    """open interest change between the last two snapshot dates = positions opened (+) or closed (−) on the trading day in between"""
    files = sorted(x for x in os.listdir(folder) if x.endswith(".csv.gz")) if os.path.isdir(folder) else []
    if len(files) < 2:
        return None
    def load(x):
        with gzip.open(os.path.join(folder, x), "rt", encoding="utf-8") as f:
            return {r["option"]: r for r in csv.DictReader(f)}
    old, new = load(files[-2]), load(files[-1])
    agg, top = {}, []
    for name, r in new.items():
        if name not in old:
            continue
        d = float(r["oi"]) - float(old[name]["oi"])
        if not d:
            continue
        a = agg.setdefault(r["symbol"], {"c": 0, "p": 0})
        a["c" if name[-9] == "C" else "p"] += d
        prem = abs(d) * float(old[name]["last"] or r["last"] or 0) * 100
        top.append({"symbol": r["symbol"], "option": name, "doi": d, "prem": prem, "delta": float(r["delta"] or 0)})
    top.sort(key=lambda x: -x["prem"])
    return {"from": files[-2][:10], "to": files[-1][:10], "by_symbol": agg, "top": top[:12]}


def summarize_flows(chains, flow_dir):
    if not chains:
        return None
    day = max(_trade_day(j) for _, j in chains)
    sp = os.path.join(flow_dir, f"state_{day}.json")
    if not os.path.exists(sp):
        return None
    st = json.load(open(sp, encoding="utf-8"))
    syms = {}
    for sym, S in st["sym"].items():
        a = S["acc"]
        tot = sum(a.values())
        syms[sym] = {**{k: round(v) for k, v in a.items()}, "net": round((a["cb"] - a["cs"]) - (a["pb"] - a["ps"])), "curve": S["curve"][-40:],
                     "rough": round(S.get("rough_prem", 0) / tot, 3) if tot else 0}
    # 大单异动：当天成交量超过持仓、权利金超过 100 万美元
    big = []
    for sym, j in chains:
        if _trade_day(j) != day:
            continue
        D = j["data"]; spot = D.get("current_price") or D.get("close") or 0
        con = st["sym"].get(sym, {}).get("con", {})
        for o in D["options"]:
            vol, oi, last = o.get("volume") or 0, o.get("open_interest") or 0, o.get("last_trade_price") or 0
            prem = vol * last * 100
            if vol < 500 or vol < oi or prem < 1e6 or _dte(o["option"], day) <= 1:
                continue
            try:
                exp, cp, k = _parse(o["option"])
            except Exception:
                continue
            b, s, m = con.get(o["option"], [0, 0, 0])
            tot = b + s + m
            big.append({"symbol": sym, "type": cp, "exp": exp.isoformat(), "strike": k, "spot": spot, "vol": vol, "oi": oi, "prem": prem,
                        "buy": b / tot if tot else None, "sell": s / tot if tot else None, "iv": o.get("iv"), "delta": o.get("delta")})
    big.sort(key=lambda x: -x["prem"])
    hist = {}
    path = os.path.join(os.path.dirname(flow_dir.rstrip("/")), "options_flow_daily.csv")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                hist.setdefault(r["symbol"], []).append([r["date"], float(r["net"] or 0)])
    return {"day": day, "polls": st["polls"], "symbols": syms, "unusual": big[:15], "hist": hist}


def option_snapshots(hist_path, log=print, flow_dir=None, contracts_dir=None):
    out, raw = [], []
    def one(s):
        try:
            j = fetch_chain(s[0])
            return j, chain_metrics(s[0], j)
        except Exception as e:
            log(f"  ! {s[0]} 期权链下载失败：{e}")
            return None, None
    with ThreadPoolExecutor(4) as ex:
        for (sym, name), (j, m) in zip(CHAIN_SYMBOLS, ex.map(one, CHAIN_SYMBOLS)):
            if m:
                m["name"] = name
                out.append(m)
                raw.append((sym, j))
    # 每个标的每天一行，攒自己的历史（同一天重复运行时覆盖）
    keep = {}
    cols = ["date", "symbol", "spot", "iv30", "term", "skew25", "pc_vol", "pc_oi", "vol"]
    if os.path.exists(hist_path):
        with open(hist_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                keep[(r["date"], r["symbol"])] = r
    for m in out:
        keep[(m["date"], m["symbol"])] = {c: ("" if m.get(c) is None else round(m[c], 4) if isinstance(m[c], float) else m[c]) for c in cols}
    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for k in sorted(keep):
            w.writerow({c: keep[k].get(c, "") for c in cols})
    hist = {}
    for (d, s), r in sorted(keep.items()):
        hist.setdefault(s, []).append([d] + [float(r[c]) if r.get(c) not in ("", None) else None for c in ("iv30", "term", "skew25", "pc_vol")])
    flow = oi = None
    if flow_dir and raw:
        try:
            flow = update_flows(raw, flow_dir, log)
        except Exception as e:
            log(f"  ! 主动买卖统计出错：{e}")
    if contracts_dir and raw:
        try:
            save_contracts(raw, contracts_dir)
            oi = oi_change(contracts_dir)
        except Exception as e:
            log(f"  ! 持仓变化统计出错：{e}")
    return out, hist, flow, oi


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    t = time.time()
    pc = put_call_history(os.path.join(here, "pc_history.csv"))
    print({k: (len(v), v[0][0], v[-1][0]) for k, v in pc.items()}, f"{time.time() - t:.0f}s")
    snaps, _, flow, oi = option_snapshots(os.path.join(here, "options_history.csv"), flow_dir=os.path.join(here, "options_flow"), contracts_dir=os.path.join(here, "options_contracts"))
    if flow:
        for s, v in flow["symbols"].items():
            print(s, {k: v[k] for k in ("cb", "cs", "pb", "ps", "net", "rough")})
        print("大单：", [(u["symbol"], u["type"], u["exp"], u["strike"], u["vol"], round(u["prem"] / 1e6, 1)) for u in flow["unusual"][:5]])
    for m in snaps:
        print(m["symbol"], m["date"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if k in ("spot", "iv30", "term", "skew25", "pc_vol", "pc_oi")})
