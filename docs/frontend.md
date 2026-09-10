# A-share Portfolio Advisor Streamlit Frontend

这是 FastAPI 后端的 Streamlit 操作台，支持：

- 运行“智能选股后再平衡”、固定股票池、自定义股票池或仅持仓模式；
- 输入现金和当前持仓；
- 从后端 SQLite 自动恢复上次保存的现金和持仓；
- 提交异步投资建议任务并自动轮询；
- 任务完成后按风控目标股数和参考价模拟成交，自动结转为下一次的当前组合；
- 展示最终目标股数和每条确定性风控调整；
- 查看三路 Analyst、Shortlist、Bull/Bear 辩论、Research Manager；
- 查看 Trader、三路 Risk Reviewer 和 Portfolio Manager；
- 查看调用轨迹、阶段健康度、市场快照及完整 JSON；
- 从后端 SQLite 读取历史决策档案。

每次提交时，后端会先保存一份不可变的组合输入快照。任务完成后，最终风控结果
仍作为历史记录保留，同时在同一 SQLite 事务中更新“当前组合”：买入按参考价重算
加权平均成本，卖出保留剩余股份的原成本，现金按模拟买卖金额增减。在线建议没有
完整的成交费用和滑点配置，因此这一步不计费用或滑点，也不会连接券商或提交订单。
新买股份按 A 股 T+1 规则暂记为不可卖。若组合在任务执行期间已被其他页面更新，
旧任务只归档而不会覆盖较新的组合版本。

平均成本在前端支持 8 位小数，后端使用 `Decimal` 和文本字段持久化，不会再强制
截断为 2 位小数。

## 启动

先在仓库根目录安装并启动后端：

```bash
.venv/bin/pip install -e '.[frontend]'
.venv/bin/ashare-agent serve --host 127.0.0.1 --port 8000
```

另开终端启动 Streamlit：

```bash
.venv/bin/streamlit run src/ashare_agent/frontend/app.py
```

前端不会从服务端环境自动读取 `BACKEND_API_KEY`；API Key 只由当前页面会话输入。
后端地址默认锁定为 `ADVISOR_API_URL`，只有受信任的本地调试环境才应设置
`ADVISOR_ALLOW_CUSTOM_API_URL=true`。

默认连接 `http://127.0.0.1:8000`。也可以设置：

```dotenv
ADVISOR_API_URL=http://127.0.0.1:8000
ADVISOR_ALLOW_CUSTOM_API_URL=false
```

若后端启用了 `BACKEND_API_KEY`，在页面的 API Key 输入框中输入相同值。它只保
存在当前 Streamlit 页面会话中。组合、结转结果和决策历史则保存在后端配置的
`DATABASE_PATH`（默认 `data/ashare_advisor.db`），前后端关闭并重启后仍会恢复。
前端不会连接券商，也不会提交订单。
