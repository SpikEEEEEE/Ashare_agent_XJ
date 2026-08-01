# Ashare Agent XJ

把“全市场量化选股”和“LLM 组合操作建议”合并成一条可审计流水线：

```text
确定研究截止时点
  → 拉取并标准化横截面数据
  → LightGBM 选股
  → 保存不可变 CandidatePool
  → 候选股 ∪ 当前持仓
  → 单 LLM / 多 Agent 组合研究
  → 确定性市场风控
  → 保存建议与完整审计信息
```

本版本实际支持 **A 股 + Tushare**。项目从第一版就把市场身份、数据源、选股器、
决策引擎和风控策略拆成独立接口；港股身份规则已经建模，但港股行情适配、动态
每股整手、港币资金与港股风控尚未实现，不能把当前 A 股策略直接用于港股。

服务只生成分析建议，不连接券商，也不会创建或提交订单。

## 主要能力

- Tushare 全 A 股增量下载、缓存和单位标准化；
- 动量、反转、波动率、量价、Amihud 等横截面特征；
- 严格按当时可见数据训练的 LightGBM Top-K 选择器；
- DeepSeek 受限 DSL 特征生成、本地安全筛选和筛选时点血缘校验；
- 带模型版本、配置哈希、数据来源、排名和分数的 `CandidatePool`；
- 动态股票池自动与当前持仓取并集，确保落选持仓仍可被减仓或清仓；
- 单次 LLM 或股票池多 Agent 决策；
- A 股整手、T+1 可卖数量、现金、单股上限和最大持仓数风控；
- FastAPI、SQLite WAL、幂等键、有界任务队列和子进程硬超时；
- 在线建议回测、选股滚动回测和 Streamlit 工作台。

## 项目结构

```text
src/ashare_agent/
├── domain/                 # InstrumentId、CandidatePool、组合和行情领域对象
├── ports/                  # 数据、选股、决策和风控 Protocol
├── adapters/               # Tushare、LightGBM 选股桥接、LLM、注册工厂
├── selection/              # 原量化选股、特征 DSL 和滚动回测
├── agents/                 # 股票池多 Agent 图和确定性权重分配
├── services/               # 动态选股 + 投顾主编排、任务执行
├── repositories/           # SQLite 与不可变 CandidatePool JSON 仓库
├── api/                    # FastAPI 路由
├── backtest/               # 历史组合回测
└── frontend/               # Streamlit 工作台
```

更详细的边界和扩展说明见
[docs/architecture.md](docs/architecture.md)。

## 安装

需要 Python 3.11–3.14：

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test,frontend]'
cp .env.example .env
```

在 `.env` 中至少设置：

```dotenv
TUSHARE_TOKEN=你的_Tushare_Token
LLM_API_KEY=你的模型_API_Key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=你有权限使用的模型名
```

真实密钥不要写入 JSON、Python 文件或 Git。

## 启动

```bash
.venv/bin/ashare-agent serve --host 127.0.0.1 --port 8000
```

打开：

- Swagger UI：<http://127.0.0.1:8000/docs>
- Liveness：<http://127.0.0.1:8000/health/live>
- Readiness：<http://127.0.0.1:8000/health/ready>

启动前端：

```bash
.venv/bin/streamlit run src/ashare_agent/frontend/app.py
```

## 一次请求完成“选股 + 操作建议”

先创建组合：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/portfolios \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "当前组合",
    "cash": "100000",
    "positions": [{
      "symbol": "600519.SH",
      "shares": 100,
      "available_shares": 100,
      "average_cost": "1500"
    }]
  }'
```

再创建动态决策任务：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/decision-runs \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: my-run-001' \
  -d '{
    "portfolio_id": "上一步返回的 portfolio id",
    "mode": "rebalance",
    "market": "CN",
    "universe_source": "fresh_selection"
  }'
```

`fresh_selection` 会在后台经历 `selecting_universe` 状态，重新计算选股结果，
同时让数据源执行正常的增量尾部更新，选股完成后自动把候选池交给投顾。它不会
每次强制重下全部历史。其他股票池模式：

- `static`：使用 `config/universe.yaml`；
- `selected`：使用最新候选池；不存在或交易日已过期时自动重新选股；
- 请求体直接传 `universe`：使用自定义股票池；
- `holdings_only`：只分析现有持仓。

也可以查看当前候选池：

```text
GET  /api/v1/candidate-pools/latest
GET  /api/v1/universe?source=selected
```

为避免把长时间下载/训练暴露成无界同步 HTTP 请求，服务不提供直接刷新端点。
在线请求请使用 `fresh_selection`，它会进入有界任务队列并受任务硬超时保护；
运维或研究场景用命令行刷新：

```bash
.venv/bin/ashare-agent select-pool
```

只有在确认缓存损坏或需要重建全部历史分区时才使用 `--force-refresh`；该操作会
产生大量 Provider 请求。`DATA_MODE=offline_only` 时，选股和投顾都只读缓存，
缺少必要分区会安全失败且不会创建远程客户端。

动态流水线只接受行情 Provider 明确判定的“已完成交易日”，并校验规范数据帧的
最大日期、Provider cutoff 和模型评分日三者完全一致。Tushare 历史行业/ST 信息
按主数据快照时点使用；无法证明当时可见的状态会标记为未知并从资格筛选中排除，
不会用今天的名称或行业倒填历史。

## 独立使用选股研究工具

原选股能力保留在同一安装包内：

```bash
.venv/bin/ashare-select make-demo-data \
  --output data/demo_market.csv --stocks 80 --days 520

.venv/bin/ashare-select select \
  --input data/demo_market.csv \
  --config config/selection.demo.json \
  --output data/selection_demo
```

CSV 是规范化研究数据接口，不会猜测交易资格。除 OHLCV、成交额等行情字段外，
必须显式提供非空的 `is_st`、`st_status_known`、`is_suspended`、
`is_limit_up` 和 `is_limit_down` 布尔列；缺失或无法解析时安全失败。
`make-demo-data` 生成的文件已经满足这份契约。

Tushare 实际选股：

```bash
.venv/bin/ashare-select tushare-select \
  --start-date 20240101 \
  --config config/selection.json \
  --output data/manual_selection
```

## 历史组合回测

组合操作频率和候选池重选频率是两个独立参数。例如，每个交易日运行一次组合决策、
每周收盘后重选一次候选池：

```bash
.venv/bin/ashare-backtest \
  --start 2024-01-01 \
  --end 2024-06-30 \
  --decision-frequency daily \
  --selection-frequency weekly \
  --max-decisions 130 \
  --engine single_llm
```

- `--decision-frequency`：`daily`、`weekly` 或 `monthly`，控制组合研究和下一交易日
  开盘调仓的频率；
- `--selection-frequency`：`once`、`daily`、`weekly` 或 `monthly`；`once` 保持配置文件
  中的固定股票池，其他值会在相应周期末用当时可见数据重新生成候选池；
- 动态候选池会一直复用到下一次重选；当前持仓始终并入每日分析范围，所以落选持仓
  仍可被减仓或清仓；
- `--max-decisions` 是 LLM 成本保护。日频半年通常超过默认的 24 次，需要显式提高；
- 旧的 `--rebalance` 参数仍可运行，但已弃用，等价于 `--decision-frequency`。

回测结果的 `summary.json` 会保存两个频率和 `selection_count`；`decisions.json` 会为
每次决策记录实际使用的 `selection_session`、`candidate_pool_id` 和候选股票列表。

## 测试

```bash
PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python -m pytest -p no:cacheprovider
```

测试覆盖原有投顾、选股、回测和前端，并新增：

- A 股/港股证券身份与整手规则边界；
- 不可变候选池持久化和完整性校验；
- 历史时点只引用更早且策略配置一致的候选池，避免 buffer 引入未来成分股；
- 负数特征窗口、历史主数据穿越、缺失中间交易日和 AI 特征筛选期穿越；
- LightGBM 结果到 `CandidatePool` 的正式交接；
- 动态选股后与当前持仓取并集，再进入投顾的端到端编排。

## 当前限制

- 只实现 Tushare A 股生产适配器和 A 股风控；
- 不支持跨市场组合、汇率、港股实时数据或港股交易费用；
- 第一次实际选股需要下载较长历史，耗时和 Tushare 权限取决于账号；
- 量化结果和 LLM 输出都不代表收益保证，实盘前必须做独立样本外验证。
