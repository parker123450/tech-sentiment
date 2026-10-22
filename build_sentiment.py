#!/usr/bin/env python3
"""科技情绪全景 · 数据脚本

做三件事：
  1. 从 Yahoo Finance 拉 1990 年以来的日线（开高低收和成交量，含 29 只半导体成分股），从 FRED 拉信用利差，
     从 CBOE 拉认沽/认购比和期权链，存成 vol_history.csv 和 semis_history.csv
  2. 调用 engine.py 计算（需要 numpy、pandas，Anaconda 自带），生成 sentiment.html（双击即可打开的完整看板）
  3. 把当天的读数追加进 daily_log.csv（样本外记录）

用法（在终端里）：
  cd ~/Desktop/科技情绪全景
  python3 build_sentiment.py              # 跑一次并打开看板
  python3 build_sentiment.py --every 15   # 每 15 分钟刷新一次（盘中用），看板会自动重新载入
  python3 build_sentiment.py --no-open    # 只更新，不打开浏览器

定时任务（launchd，每 30 分钟）用的是：
  python3 build_sentiment.py --no-open --market-hours --reload 30 --notify
  --market-hours  不在美股交易时段（美东工作日 9:25–16:35）就直接退出
  --reload 30     看板开着时每 30 分钟自己重新载入
  --notify        出现强信号时弹 macOS 通知（见 notify.py）
"""
import argparse, csv, json, os, re, ssl, sys, time, webbrowser
import urllib.error, urllib.parse, urllib.request
from datetime import datetime, timezone, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "sentiment_template.html")
OUT_HTML = os.path.join(HERE, "sentiment.html")
OUT_CSV = os.path.join(HERE, "vol_history.csv")
PC_CACHE = os.path.join(HERE, "pc_history.csv")          # CBOE 认沽/认购比历史（第一次运行时补齐，之后只补新的）
OPT_HIST = os.path.join(HERE, "options_history.csv")     # 每天存一行期权链快照，攒自己的 IV 历史
SEMIS_CSV = os.path.join(HERE, "semis_history.csv")      # 半导体成分股日线（参与度和跨品种验证用）
CLOUD_DIR = os.path.join(HERE, "cloud")                  # 手机上看的私有网页：index.html + data.json
FETCH_STATE = os.path.join(HERE, ".fetch_state.json")    # 哪天做过全量下载、哪天的收盘数据已经算过（补跑用）
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ADJ = {"HYG", "IEF", "RSP", "SPY"}                        # 这几只算比值时用含分红的价格，否则分红会让比值一路下滑

SYMBOLS = [
    "^VIX", "^VIX9D", "^VIX3M", "^VXN",                    # 期权：隐含波动（第六轮去掉 VVIX、SKEW：核对里没有信息）
    "^GSPC", "^NDX", "SPY", "RSP", "QQQ",                  # 大盘、广度、实际波动
    "^SOX", "SMH", "SOXX", "SOXL", "SOXS",                 # 半导体指数、ETF、3 倍多空
    "HYG", "IEF",                                          # 信用：高收益债和国债
    "^KS11", "000660.KS", "005930.KS",                     # KOSPI、SK 海力士、三星
    "^MOVE", "DX-Y.NYB",                                   # 宏观背景（第六轮）：债券波动、美元指数，只作背景
]
# 半导体成分股（费城半导体指数和 SMH 的主要成员，2026-10 的名单）。只用现在的成员，早年的广度有幸存者偏差。
SEMIS = ["NVDA", "AVGO", "AMD", "TSM", "ASML", "QCOM", "TXN", "MU", "AMAT", "LRCX", "KLAC", "ADI", "INTC", "MRVL", "NXPI",
         "MCHP", "ON", "MPWR", "TER", "ENTG", "SWKS", "ARM", "GFS", "COHR", "LSCC", "STM", "UMC", "ASX", "MTSI"]
# 高收益和投资级利差（%）。ICE 授权限制，FRED 现在只给最近 3 年；
# BAA10Y（穆迪 Baa 公司债 − 10 年期国债，1986 年起）补长期历史，看板用它区分系统性和板块型下跌
FRED = {"HYOAS": "BAMLH0A0HYM2", "IGOAS": "BAMLC0A0CM", "BAA10Y": "BAA10Y",
        # 宏观背景（第六轮，只作背景，检验过没有前瞻性）：美联储总资产（百万美元）、国库账户（百万美元）、隔夜逆回购（十亿美元）、10 年和 2 年国债
        "WALCL": "WALCL", "WTREGEN": "WTREGEN", "RRPONTSYD": "RRPONTSYD", "DGS10": "DGS10", "DGS2": "DGS2"}
# 注意：完整的浏览器 UA（带 Chrome/...）没有 cookie 时会被 Yahoo 直接回 429，简短的 UA 反而正常
UA = {"User-Agent": "Mozilla/5.0"}


def ssl_context():
    try:
        import certifi  # Anaconda 和多数环境自带
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


CTX = ssl_context()


def get(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return r.read().decode("utf-8")


def yahoo(sym, since=None):
    p2 = int(time.time()) + 86400
    p1 = int(since) if since else 631152000   # 默认 1990-01-01
    q = urllib.parse.quote(sym)
    last = None
    for host in ("query1", "query2"):
        url = f"https://{host}.finance.yahoo.com/v8/finance/chart/{q}?period1={p1}&period2={p2}&interval=1d&includePrePost=false"
        for attempt in range(3):
            try:
                j = json.loads(get(url))
                res = (j.get("chart") or {}).get("result")
                if res:
                    return res[0]
                last = "empty result"
                break
            except ssl.SSLError as e:
                raise
            except urllib.error.HTTPError as e:
                last = e
                time.sleep((5 if e.code == 429 else 1.5) * (attempt + 1))  # 429 = 限流，等久一点
            except Exception as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
    print(f"  ! {sym} 下载失败：{last}")
    return None


def rows_from(res):
    ts = res.get("timestamp") or []
    q = (res.get("indicators") or {}).get("quote", [{}])[0]
    off = timedelta(seconds=(res.get("meta") or {}).get("gmtoffset", 0) or 0)
    out = {}
    meta = res.get("meta") or {}
    mt = meta.get("regularMarketTime")
    mday = (datetime.fromtimestamp(mt, tz=timezone.utc) + off).strftime("%Y-%m-%d") if mt else None
    for i, t in enumerate(ts):
        c = (q.get("close") or [None])[i] if i < len(q.get("close") or []) else None
        d = (datetime.fromtimestamp(t, tz=timezone.utc) + off).strftime("%Y-%m-%d")
        pick = lambda k: (q.get(k) or [None] * len(ts))[i] if i < len(q.get(k) or []) else None
        if c is None and i == len(ts) - 1 and d == mday and meta.get("regularMarketPrice"):
            # Yahoo 偶尔把最新一根日线的收盘价留空（收盘后要过一阵才补），这时用同一天的最新成交价补上
            c = meta["regularMarketPrice"]
            q = dict(q)
            for k, mk in (("high", "regularMarketDayHigh"), ("low", "regularMarketDayLow"), ("volume", "regularMarketVolume")):
                if pick(k) is None and meta.get(mk) is not None:
                    q[k] = list(q.get(k) or [None] * len(ts)); q[k][i] = meta[mk]
        if c is None:
            continue
        v = pick("volume") or 0
        adj = ((res.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose") or []
        a = adj[i] if i < len(adj) and adj[i] is not None else c
        out[d] = [d, round(float(c), 6), float(v), pick("open"), pick("high"), pick("low"), round(float(a), 6)]  # 同一天重复时保留最后一条
    return [out[k] for k in sorted(out)]


def fred(series_id):
    """FRED 偶尔返回网页（限流或维护页），所以先校验表头和日期，不对就换个 User-Agent 重试。"""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    last = None
    for attempt, ua in enumerate(("Mozilla/5.0", "Python-urllib/3", "tech-sentiment/1.0 (personal research)")):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua})
            with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
                lines = r.read().decode("utf-8", "replace").splitlines()
            if not lines or not lines[0].lower().startswith(("observation_date", "date")):
                raise ValueError("返回的不是数据表（可能是网页或限流）")
            rows = []
            for line in lines[1:]:
                p = line.strip().split(",")
                if len(p) >= 2 and ISO.match(p[0]) and p[1] not in ("", "."):
                    try:
                        rows.append([p[0], float(p[1]), 0, None, None, None])
                    except ValueError:
                        pass
            if len(rows) < 100:
                raise ValueError(f"只有 {len(rows)} 行，不像完整数据")
            return rows
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(last)


def load_cache():
    """上次存下的数据（vol_history.csv）。某个品种这次下载失败时沿用它。只收日期和收盘价都合法的行。"""
    out = {}
    if not os.path.exists(OUT_CSV):
        return out
    fl = lambda x: float(x) if x not in (None, "") else None
    with open(OUT_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = r.get("date") or ""
            if not ISO.match(d):
                continue
            try:
                c = float(r["close"])
            except (TypeError, ValueError):
                continue
            if not c == c or c <= 0:
                continue
            try:
                out.setdefault(r["symbol"], {})[d] = [d, c, fl(r.get("volume")) or 0.0, fl(r.get("open")), fl(r.get("high")), fl(r.get("low")), fl(r.get("adjclose")) or c]
            except ValueError:
                continue
    return {k: [v[d] for d in sorted(v)] for k, v in out.items()}


def read_state():
    try:
        return json.load(open(FETCH_STATE, encoding="utf-8"))
    except Exception:
        return {}


def write_state(st):
    with open(FETCH_STATE, "w", encoding="utf-8") as f:
        json.dump(st, f)


def write_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def alert_failure(err):
    """看板没能更新时弹一次通知（同一天只弹一次）。"""
    try:
        import notify
        st = json.load(open(notify.STATE, encoding="utf-8")) if os.path.exists(notify.STATE) else {}
        today = datetime.now().strftime("%Y-%m-%d")
        if st.get("_build_fail") == today:
            return
        notify.send("看板更新失败", f"{str(err)[:100]}。看板还是上一次的版本，详情见 ~/Library/Logs/tech-sentiment.log")
        st["_build_fail"] = today
        with open(notify.STATE, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False)
    except Exception:
        pass


def keep_recent_gaps(new_rows, old_rows, days=45):
    """Yahoo 的全量历史偶尔漏掉最近的某一天（2026-10-22 全量下载时 30 多个品种都漏了 10-21）。
    全量替换历史时，缓存里有、这次没返回、又在最近 days 天以内的日子补回来（也包括比这次更新的那几天）。"""
    if not new_rows or not old_rows:
        return new_rows, 0
    got = {x[0] for x in new_rows}
    cut = (datetime.strptime(new_rows[-1][0], "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")
    back = [x for x in old_rows if x[0] not in got and x[0] >= cut]
    return sorted(new_rows + back, key=lambda x: x[0]), len(back)


def update_semis(log=print):
    """半导体成分股日线。每天第一次运行拉 1990 年以来全量（拆股后 Yahoo 会改写历史），之后只补最近 10 天。"""
    rows = {}
    fresh_full = True
    if os.path.exists(SEMIS_CSV):
        fresh_full = datetime.fromtimestamp(os.path.getmtime(SEMIS_CSV)).date() != datetime.now().date()
        with open(SEMIS_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.setdefault(r["symbol"], {})[r["date"]] = [r["date"], r["close"], r["volume"], r["open"], r["high"], r["low"], r["adjclose"]]
    ok = 0
    for s in SEMIS:
        cached = rows.get(s, {})
        old = {} if fresh_full else cached
        since = None
        if old:
            since = datetime.strptime(max(old), "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() - 10 * 86400
        res = yahoo(s, since)
        if not res:
            continue
        new = rows_from(res)
        if not new:
            continue
        if fresh_full:
            new, nb = keep_recent_gaps(new, [cached[d] for d in sorted(cached)])
            if nb:
                log(f"  {s} 这次全量少了 {nb} 天，用存下的数据补回")
        for r in new:
            old[r[0]] = r
        rows[s] = old
        ok += 1
        time.sleep(0.3)
    with open(SEMIS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "date", "close", "volume", "open", "high", "low", "adjclose"])
        for s in SEMIS:
            for d in sorted(rows.get(s, {})):
                r = rows[s][d]
                w.writerow([s] + ["" if x is None else x for x in r])
    log(f"  半导体成分股 {ok}/{len(SEMIS)} 只（{'全量' if fresh_full else '只补最近 10 天'}）")


def us_market_open():
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("America/New_York"))
    return now.weekday() < 5 and (9, 25) <= (now.hour, now.minute) <= (16, 35)


def ny_now():
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York"))


def last_session():
    """最近一个已经收盘的美股交易日（不认节假日：节假日那天补跑也只是重算一遍）。"""
    now = ny_now()
    d = now.date()
    if not (now.weekday() < 5 and (now.hour, now.minute) >= (16, 5)):
        d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.isoformat()


def needs_catchup():
    """收盘数据还没算过（比如 Mac 在美股时段睡着了），而且 3 小时内没补跑过。"""
    st = read_state()
    if st.get("closing_done", "") >= last_session():
        return False
    try:
        if (datetime.now().astimezone() - datetime.fromisoformat(st.get("catchup_try", ""))).total_seconds() < 3 * 3600:
            return False
    except ValueError:
        pass
    st["catchup_try"] = datetime.now().astimezone().isoformat(timespec="seconds")
    write_state(st)
    return True


def render_only(open_browser=False, site=None):
    """不下载：用存下的行情和上一次的期权数据重新计算、生成看板（改界面、改计算时用）。"""
    import engine
    options = None
    try:
        options = json.load(open(os.path.join(CLOUD_DIR, "data.json"), encoding="utf-8")).get("options")
    except Exception:
        pass
    page, M = engine.build(options=options)
    page["built"] = datetime.now().astimezone().isoformat(timespec="seconds")
    write_page(page, 0, site)
    print(f"已重新生成 {OUT_HTML}（没有下载新数据），数据截至 {page['asof']}")
    if open_browser:
        webbrowser.open("file://" + urllib.parse.quote(OUT_HTML))


HEAD = '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'


def to_json(x):
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else None)


def write_site(page, tpl, folder, keep=800):
    """网站版（GitHub Pages）：页面不带数据；data.json 里 K 线只放最近 keep 天，完整历史放 history.json，页面打开后在后台再取。"""
    os.makedirs(folder, exist_ok=True)
    core = dict(page, lazy="history.json")
    core["chart"] = []
    for c in page["chart"]:
        rows = c["rows"][-keep:]
        core["chart"].append(dict(c, rows=rows, rsi=c["rsi"][-keep:], climax=[d for d in c["climax"] if d >= rows[0][0]]))
    write_atomic(os.path.join(folder, "index.html"), HEAD + tpl.replace("<!--__REFRESH__-->", ""))
    write_atomic(os.path.join(folder, "data.json"), to_json(core))
    write_atomic(os.path.join(folder, "history.json"), to_json({"chart": page["chart"]}))


def write_page(page, every, site=None):
    data = to_json(page)
    with open(TEMPLATE, encoding="utf-8") as f:
        tpl = f.read()
    if "/*__DATA__*/null" not in tpl:
        raise RuntimeError("sentiment_template.html 不是最新版，请用最新的模板替换。")
    refresh = f'<meta http-equiv="refresh" content="{every * 60}">' if every else ""
    html = HEAD + tpl.replace("<!--__REFRESH__-->", refresh).replace("/*__DATA__*/null", data)
    write_atomic(OUT_HTML, html)
    # 云端版本：页面不带数据，数据单独放 data.json
    os.makedirs(CLOUD_DIR, exist_ok=True)
    write_atomic(os.path.join(CLOUD_DIR, "index.html"), tpl.replace("<!--__REFRESH__-->", ""))
    write_atomic(os.path.join(CLOUD_DIR, "data.json"), data)
    if site:
        write_site(page, tpl, site)
    return html


def build_once(open_browser, every, site=None):
    series, health = {}, {}
    cache = load_cache()
    state = read_state()
    today = datetime.now().strftime("%Y-%m-%d")
    full = state.get("full_day") != today        # 每天第一次全量（拆股、分红后 Yahoo 会改写历史），之后只补最近 10 天
    def keep_old(s, why):
        old = cache.get(s)
        if old:
            series[s] = {"rows": old}
            health[s] = {"status": "cached", "last": old[-1][0]}
            print(f"  ! {s} {why}，沿用上次的数据（截至 {old[-1][0]}）")
        else:
            health[s] = {"status": "missing"}
            print(f"  ! {s} {why}，也没有存过的数据")
    print(f"下载 Yahoo 日线（{'全量' if full else '只补最近 10 天'}）：")
    ok_full = 0
    for s in SYMBOLS:
        old = cache.get(s) or []
        since = None if full or len(old) < 250 else datetime.strptime(old[-1][0], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() - 10 * 86400
        try:
            r = yahoo(s, since)
        except ssl.SSLError as e:
            print("\nSSL 证书错误：", e)
            print("如果你用的是 python.org 安装的 Python，请先双击运行 /Applications/Python 3.x/Install Certificates.command，")
            print("或者改用 Anaconda 自带的 python3。")
            sys.exit(1)
        rows = rows_from(r) if r else []
        if not rows:
            keep_old(s, "下载失败")
            continue
        if since is not None or len(rows) < 0.9 * len(old):     # 增量，或者全量反而变短了：和存下的合并
            m = {x[0]: x for x in old}
            m.update((x[0], x) for x in rows)
            rows = [m[k] for k in sorted(m)]
        else:
            ok_full += 1
            rows, nb = keep_recent_gaps(rows, old)              # 全量替换历史，但最近 45 天里缓存有、这次没返回的日子补回来
            if nb:
                print(f"  {s} 这次全量少了 {nb} 天，用存下的数据补回")
        series[s] = {"rows": rows}
        health[s] = {"status": "ok", "last": rows[-1][0]}
        print(f"  {s:<10} {len(rows):>6} 天  {rows[0][0]} → {rows[-1][0]}")
        time.sleep(0.4)
    if full and ok_full >= len(SYMBOLS) - 2:
        state["full_day"] = today
        write_state(state)
    print("下载半导体成分股：")
    try:
        update_semis()
    except ssl.SSLError:
        raise
    except Exception as e:
        print("  ! 成分股下载出错（看板其他部分照常生成）：", e)
    print("下载 FRED 信用利差：")
    for key, sid in FRED.items():
        try:
            rows = fred(sid)
            series[key] = {"rows": rows}
            health[key] = {"status": "ok", "last": rows[-1][0]}
            print(f"  {key:<10} {len(rows):>6} 天  {rows[0][0]} → {rows[-1][0]}")
        except Exception as e:
            keep_old(key, f"下载失败（{e}）")
    if not any(h["status"] == "ok" for h in health.values()):
        raise RuntimeError("一条数据都没下载到，检查一下网络，或者稍后再试。")
    options = None
    try:
        import cboe
        print("下载 CBOE 期权数据：")
        pc = cboe.put_call_history(PC_CACHE, log=print)
        for kind, key in (("equity", "PC_EQ"), ("index", "PC_IDX"), ("total", "PC_TOT")):
            if pc.get(kind):
                series[key] = {"rows": [[d, r, 0, None, None, None, r] for d, r, _, _ in pc[kind]]}
                print(f"  {key:<10} {len(pc[kind]):>6} 天  {pc[kind][0][0]} → {pc[kind][-1][0]}")
        snaps, ohist, oflow, ooi = cboe.option_snapshots(OPT_HIST, log=print, flow_dir=os.path.join(HERE, "options_flow"), contracts_dir=os.path.join(HERE, "options_contracts"))
        print(f"  期权链快照 {len(snaps)} 个标的：" + " ".join(m["symbol"] for m in snaps))
        if oflow:
            print(f"  主动买卖（近似）{oflow['day']}：第 {len(oflow['polls'])} 次快照，大单 {len(oflow['unusual'])} 张")
        options = {"chains": snaps, "hist": ohist, "flow": oflow, "oi": ooi}
    except Exception as e:
        print("  ! CBOE 期权数据出错（看板其他部分照常生成）：", e)

    # 认沽/认购比这次没取到时，也沿用存下的
    for key in ("PC_EQ", "PC_IDX", "PC_TOT"):
        if key not in series and cache.get(key):
            series[key] = {"rows": cache[key]}
            health[key] = {"status": "cached", "last": cache[key][-1][0]}
    # 回测用的完整数据：先写临时文件再替换，中途出错也不会把上次的好数据冲掉
    tmp = OUT_CSV + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "date", "close", "volume", "open", "high", "low", "adjclose"])
        for s, v in series.items():
            for r in v["rows"]:
                w.writerow([s, r[0], r[1], r[2]] + ["" if x is None else x for x in (r[3], r[4], r[5], r[6] if len(r) > 6 else r[1])])
    os.replace(tmp, OUT_CSV)

    # 看板：所有计算交给 engine.py（唯一的计算源）
    import engine
    built = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        page, M = engine.build(options=options)
    except Exception as e:
        alert_failure(e)
        raise
    page["built"] = built
    page["health"] = {"stale": [[k, h["last"]] for k, h in health.items() if h["status"] == "cached"],
                      "missing": [k for k, h in health.items() if h["status"] == "missing"]}
    engine.append_log(M, built)
    nyt = ny_now()
    if nyt.date().isoformat() > page["asof"] or nyt.hour >= 16:      # 收盘以后算的：这一天的收盘数据已经有了
        st = read_state()
        st["closing_done"] = max(st.get("closing_done", ""), page["asof"])
        write_state(st)
    html = write_page(page, every, site)
    print(f"\n已生成 {OUT_HTML}（{len(html) / 1e6:.1f} MB）和 {OUT_CSV}，时间 {built}")
    if open_browser:
        webbrowser.open("file://" + urllib.parse.quote(OUT_HTML))
    return page


def main():
    ap = argparse.ArgumentParser(description="科技情绪全景 数据脚本")
    ap.add_argument("--every", type=int, default=0, help="每隔多少分钟重新拉一次（0 = 只跑一次）")
    ap.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    ap.add_argument("--market-hours", action="store_true", help="不在美股交易时段就直接退出（给定时任务用）")
    ap.add_argument("--reload", type=int, default=0, help="看板每隔多少分钟自己重新载入（不循环拉数据）")
    ap.add_argument("--notify", action="store_true", help="出现强信号时弹 macOS 通知")
    ap.add_argument("--render-only", action="store_true", help="不下载，用存下的数据重新生成看板")
    ap.add_argument("--site", help="另外输出一份网站（index.html、data.json、history.json）到这个文件夹，给 GitHub Pages 用")
    a = ap.parse_args()
    if a.render_only:
        render_only(open_browser=not a.no_open, site=a.site)
        return
    if a.market_hours and not us_market_open():
        if not needs_catchup():
            print(datetime.now().astimezone().isoformat(timespec="seconds"), "不在美股交易时段，跳过。")
            if a.notify:
                # 不下载也检查一次：收盘后的每日摘要有没有发、数据是不是停在了前一个交易日（看门狗）。
                # 否则补跑有 3 小时节流，收盘后两次定时都会被跳过，当晚的"数据没更新"提醒永远发不出去。
                try:
                    import notify
                    notify.run()
                except Exception as e:
                    print("通知检查出错：", e)
            return
        print(datetime.now().astimezone().isoformat(timespec="seconds"), f"不在美股交易时段，但 {last_session()} 的收盘数据还没算过，补跑一次。")
    first = True
    while True:
        try:
            page = build_once(open_browser=first and not a.no_open, every=a.every or a.reload, site=a.site)
            first = False
            if a.notify:
                try:
                    import notify  # 需要 numpy 和 pandas（Anaconda 自带）
                    notify.run(page=page)       # 传进去，每日摘要不用再算一遍
                except Exception as e:
                    print("通知检查出错：", e)
        except Exception as e:
            print("出错了：", e)
            alert_failure(e)
        if not a.every:
            break
        print(f"{a.every} 分钟后再次刷新，按 Ctrl+C 停止。")
        time.sleep(a.every * 60)


if __name__ == "__main__":
    main()
