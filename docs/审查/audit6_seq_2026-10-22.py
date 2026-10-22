import sys, os, warnings, numpy as np, pandas as pd
sys.path.insert(0, "/Users/liangjiafu/Desktop/科技情绪全景"); os.chdir("/Users/liangjiafu/Desktop/科技情绪全景"); warnings.filterwarnings("ignore")
import engine as E
D = E.load(); M = E.market(D); idx = M.index; n = len(M)
eps = E.episodes(M.c.to_numpy())
def seq(points, items, back, fwd, title):
    print(f"\n### {title}：各指标在窗口里到达极端的那天（相对顶/底的天数；括号里是窗口里的最高分位，<90 用 · 标）")
    rows = []
    for p in points:
        d = idx[p].strftime("%Y-%m")
        r = {"日期": d}
        for name, (x, s) in items.items():
            v = (x * s).astype(float); pct = E.exp_pct(v).reindex(idx)
            w = pct.iloc[max(0, p - back): min(n, p + fwd + 1)]
            if w.notna().any():
                j = idx.get_loc(w.idxmax()) - p
                r[name] = f"{j:+d}" + ("" if w.max() >= 90 else "·")
            else:
                r[name] = "—"
        rows.append(r)
    print(pd.DataFrame(rows).set_index("日期").to_string())
bot = {"广度50日": (M.br_pct50, -1), "RSI": (M.rsi, -1), "20日Z": (M.z20, -1), "VIX÷50": (M.vix / M.vix50, 1), "倒挂": (M.t3, 1), "VIX": (M.vix, 1),
       "认沽/认购": (M.pc10, 1), "Baa20": (M.baa20, 1), "HYG÷IEF": (M.hyief20, -1), "多空额": (M.bb_pct, -1)}
seq([tr for pk, tr, en in eps if en is not None], bot, 60, 15, "底部（低点前 60 天到后 15 天）")
top = {"等权÷市值": (M.rsp60, -1), "广度50日": (M.br_pct50, 1), "新高新低": (M.br_nh - M.br_nl, 1), "认沽/认购低": (M.pc10, -1), "多空额高": (M.bb_pct, 1),
       "离200日": (M.extv, 1), "RSI": (M.rsi, 1), "20日Z": (M.z20, 1), "VIX比20日低点": (M.vix / M.vixmin20, 1)}
seq([pk for pk, tr, en in eps], top, 60, 10, "顶部（高点前 60 天到后 10 天）")
