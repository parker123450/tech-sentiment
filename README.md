# 科技情绪全景

半导体和科技股的情绪看板：价格走到了哪一步 × 市场有没有为恐惧（或贪婪）付钱，外加期权给出的预期波动和趋势风险。

- 看板：`https://<用户名>.github.io/<仓库名>/`（手机、电脑同一个网址）
- 美股盘中每 30 分钟自动更新一次，收盘后再补一次（`.github/workflows/update.yml`）
- 出现强信号时推送到手机 ntfy App（仓库 Secrets 里的 `NTFY_TOPIC`）
- 规则和口径见 `规则说明.md`，历史检验见 `研究结论.md`

数据来自 Yahoo Finance、FRED、CBOE 的公开行情，仅作情绪参考，不构成投资建议。

## 文件

| 文件 | 作用 |
|---|---|
| `build_sentiment.py` | 抓数据，调用 `engine.py` 生成看板 |
| `engine.py` | 全部计算（唯一的计算源） |
| `research.py` | 检验每条规则，生成 `研究结论.md`、`research_results.json` |
| `cboe.py` | CBOE 期权数据 |
| `notify.py` | 强信号通知（Mac 桌面通知 / ntfy） |
| `sentiment_template.html` | 看板模板，只负责画 |
| `seed/` | 云端第一次运行时用的初始小文件；之后这些文件存在 `data` 分支 |

## 本机运行

```bash
pip install -r requirements.txt
python3 build_sentiment.py          # 拉数据、生成 sentiment.html 并打开
python3 engine.py --selftest        # 检查没有用到未来数据
```
