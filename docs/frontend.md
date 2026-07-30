# A-share Portfolio Advisor Streamlit Frontend

这是 FastAPI 后端的 Streamlit 操作台，支持：

- 运行“智能选股后再平衡”、固定股票池、自定义股票池或仅持仓模式；
- 输入现金和当前持仓；
- 提交异步投资建议任务并自动轮询；
- 展示最终目标股数和每条确定性风控调整；
- 查看三路 Analyst、Shortlist、Bull/Bear 辩论、Research Manager；
- 查看 Trader、三路 Risk Reviewer 和 Portfolio Manager；
- 查看调用轨迹、阶段健康度、市场快照及完整 JSON；
- 从后端 SQLite 读取历史决策档案。

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
存在当前 Streamlit 页面会话中。前端不会连接券商，也不会提交订单。
