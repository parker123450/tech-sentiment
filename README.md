# 科技情绪全景

半导体和科技股的情绪看板：价格走到了哪一步 × 市场有没有为恐惧（或贪婪）付钱，外加期权给出的预期波动和趋势风险。用客观指标代替感觉，等恐慌出清再买，也能看出过热。

## 现在做到哪（2026-10-22）

**已经完成**
- 第一到第三轮：数据管线（Yahoo、FRED、CBOE）、看板、预警通知、期权数据。
- 第四轮：去掉"贴着历史调出来的综合分"，改成对照格 + 清单 + 三个温度；每条规则先登记再检验，结果只决定标签。
- 第五轮：
  - 找出真正能衡量的：期权隐含波动能衡量之后一个月的**波动幅度**，跌破 200 日线能衡量**下跌风险**，方向测不出来。
  - 看板改成白色。顶部"今天该知道的三件事"，加了每日一览、K 线上的期权预期范围、期权五个白话问题。
  - 云端自动更新（GitHub Actions + Pages）和手机推送（ntfy）的代码。

- 2026-10-22 部署完成：仓库、GitHub Pages、ntfy 推送都已设好，第一次云端运行成功（所有数据源都能从 GitHub 服务器下载）。之后美股盘中每 30 分钟自动更新。

- 原来的 Claude 定时任务和 Mac 本机定时任务都已停用，现在只有云端在跑。
- 恢复一个总分：**情绪分** = 过热 − (定价恐慌 + 抛售恐慌) ÷ 2，−100 极度恐慌到 +100 过热，放在看板最上面，每日一览和 K 线副图里也有。
- 指标客观性审查（[docs/审查](docs/审查/2026-10-22_指标客观性审查.md)）：刻度的主观性影响小；情绪分描述恐慌底部准（AUC 0.93），描述顶部一般，约三分之二是价格回声；方向性规则的证据比标签弱。改进方案待你决定。

**等你做**：手机装 ntfy App，订阅推送频道。

**以后**
- 样本外记录攒够一两年后，检查对照格和清单在没见过的数据上是否还成立。
- 待定：跨资产（美元兑日元、美债、美元兑韩元、比特币）、财报和 FOMC 日历、接入盈透（暂缓）。

## 去哪看

| 想看什么 | 在哪 |
|---|---|
| 今天的看板 | [parker123450.github.io/tech-sentiment](https://parker123450.github.io/tech-sentiment/)（手机、电脑都行） |
| 看板长什么样 | [总览](docs/screenshots/overview.png) · [K 线](docs/screenshots/kline.png) · [期权](docs/screenshots/options.png) |
| 每条规则怎么定、怎么读 | [规则说明.md](规则说明.md) |
| 历史检验结果 | [研究结论.md](研究结论.md) |
| 目标、原则、框架、文件、待办（完整交接说明） | [CLAUDE.md](CLAUDE.md) |
| 每一轮的计划 | [docs/计划](docs/计划) |
| 指标是否有效、有多少主观成分 | [docs/审查](docs/审查) |
| 每天的读数记录（样本外） | `data` 分支的 `daily_log.csv` |

## 在别的电脑或手机上继续

- 看进度：打开这个仓库的 GitHub 页面，就是这份说明。
- 接着改：浏览器打开 claude.ai/code，选这个仓库（第一次要按提示授权 Claude 访问它），Claude 会先读 `CLAUDE.md`。
- 换一台电脑本机跑：clone 仓库，然后按下面"本机运行"做。

更新方式：美股盘中每 30 分钟自动更新一次，收盘后再补一次（`.github/workflows/update.yml`）；出现强信号时推送到手机 ntfy App（仓库 Secrets 里的 `NTFY_TOPIC`）。数据来自 Yahoo Finance、FRED、CBOE 的公开行情，仅作情绪参考，不构成投资建议。

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
| `docs/` | 每轮的计划、看板截图 |
| `CLAUDE.md` | 交接说明：目标、原则、框架、文件、待办 |

## 本机运行

```bash
pip install -r requirements.txt
python3 build_sentiment.py          # 拉数据、生成 sentiment.html 并打开
python3 engine.py --selftest        # 检查没有用到未来数据
```
