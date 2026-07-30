# 架构与扩展边界

## 设计目标

统一项目不让选股模块直接依赖投顾实现，也不让投顾读取某个临时
`candidates.csv`。两个能力只通过不可变的 `CandidatePool` 领域对象连接。

```text
SelectionDataProvider
        │ canonical selection frame
        ▼
CandidatePoolSelector ──► CandidatePoolRepository
        │                         │
        └──────── CandidatePool ◄─┘
                         │
                         ▼
DecisionService: candidates ∪ current holdings
                         │
          MarketDataProvider + DecisionEngine
                         │
                         ▼
                    RiskPolicy
```

## 稳定领域契约

### InstrumentId

`InstrumentId(mic, local_code)` 是领域主键：

- `XSHG:600519`
- `XSHE:000001`
- `XHKG:00700`

`local_code` 始终是字符串，因此港股前导零不会丢失。`600519.SH` 和
`00700.HK` 只是 Provider/API 别名。

`MarketProfile` 负责时区、币种、MIC 与 Provider 后缀映射、代码格式和默认整手。
港股整手不是常量，必须从证券主数据取得。

### CandidatePool

候选池包含：

- schema、pool id、生成时间和内容 digest；
- 市场、研究时间、数据交易日和下一有效交易日；
- 策略、模型版本、配置哈希；
- 数据源与标准化来源；
- 父候选池；
- 每只证券的标准身份、Provider symbol、排名、分数、理由和信号；
- 训练诊断。

候选池以不可变 JSON 保存，`latest-<market>.json` 只是一条原子更新的指针。
决策任务同时固化 pool id、digest 和股票代码列表。
组合快照也固化 `market_id`，不能在服务切换市场后把原市场持仓误用于另一个市场。

### 数据端口

当前有两类端口，因为它们的访问模式不同：

- `SelectionDataProvider`：全市场、跨日期的横截面批量数据；
- `MarketDataProvider`：对候选股票读取投顾需要的行情、财务、新闻和日历。

同一个外部 Provider 可以实现两个端口，但应用服务不依赖具体 Provider。
缓存路径必须包含 `用途/provider/market/schema_version`，避免切换数据源或市场后
复用错误数据。

### 策略端口

- `CandidatePoolSelector`：选择并持久化候选池；
- `DecisionEngine`：输出结构化目标仓位意图；
- `RiskPolicy`：应用市场和组合的确定性约束。

LLM 永远不承担最后一道交易规则校验，也没有券商工具权限。

## 当前注册项

```text
MarketProfile:
  CN  identity/timezone/currency/fixed board lot
  HK  identity/timezone/currency; board lot must come from master data

MarketDataProvider:
  tushare/CN

SelectionDataProvider:
  tushare/CN
  csv/CN  (研究和契约测试)

CandidatePoolSelector:
  lightgbm/CN

RiskPolicy:
  CN
```

虽然 HK 身份已建模，但没有注册 HK 行情、选择器和风控，因此服务会明确拒绝运行，
不会悄悄套用 A 股规则。

## 新增另一个数据源

1. 实现 `SelectionDataProvider` 和/或 `MarketDataProvider`。
2. 在适配器内部完成一次且仅一次单位转换。
3. 输出规范字段，不把 Provider 原始字段泄露到应用层。
4. 将缓存放在新的 Provider/市场/schema 命名空间。
5. 通过 `register_market_data_provider()` 注册构建器。
6. 为相同规范输入增加“不同数据源产生相同候选池”的契约测试。

选股规范中，成交量是股数、成交额是实际货币金额、价格同时区分原始价格和复权
研究价格。禁止把 Tushare 的“手/千元/万元”直接传到领域层。

Provider 必须让 `data_cutoff` 等于规范帧中的最大交易日。Selector 的 cutoff 由
`MarketDataProvider.latest_completed_session()` 显式传入；Provider 声称的
cutoff、帧最大日期和最终评分日任一不一致都会失败，不能由适配器自行猜测盘中
数据是否已经完成。

## 新增港股市场

至少需要：

1. HK 交易日历和行情/财务/新闻 Provider；
2. 证券主数据及每只股票的 board lot；
3. HK 专用可卖、费用、结算和停牌规则；
4. HK 专用选择器配置与流动性阈值；
5. HKD 资金模型；跨市场组合还需要 `Money` 和汇率 Provider；
6. HK 历史回测成交规则；
7. Fake HK Provider 契约测试通过后再接真实 API。

不要把 A 股的 ST、涨跌停、六位代码、100 股整手、T+1 或人民币阈值放进通用
选择器。

## 运行时一致性

动态决策先由投顾行情 Provider 确定已完成交易日，再解析候选池。缓存候选池如果
不是同一交易日，会自动重新选择；新候选池仍不匹配时任务安全失败。这样选股和
建议共享一个数据截止日，不会把未完成交易日的数据混入操作建议。

候选池成员与当前持仓取并集。候选排名和选股信号会进入每只股票的 LLM 上下文；
非候选的现有持仓会明确标记为“候选池外持仓”，仍可被保持、减仓或清仓。

选股 buffer 只允许引用 `data_session` 严格早于当前研究日、且 `strategy_id` 与
稳定配置哈希均一致的候选池。历史回放不会读取 `latest` 指针，因此后来生成的
股票池不能污染过去结果。`latest` 指针只会单调前进；池文件采用进程锁、原子
写入和完整内容 digest。同一业务时点发生数据修订时，以独立 `created_at` 推进
最新版，而不是比较内容哈希的字典序。

配置哈希只覆盖策略真正使用的参数、显式策略/应用版本和生成特征文件内容，不含
缓存目录或 API 地址等部署路径。数据快照另有规范化 DataFrame 指纹。这样可以
区分“策略变化”和“数据变化”，也不会因换机器目录不同而制造伪版本。

Tushare 的 `stock_basic` 和 `namechange` 缓存带有快照日期。当前行业/名称只在
能够证明对应交易日可见时使用；历史 ST 状态无法证明时设置
`st_status_known=false`，资格筛选 fail closed。中间交易日分区缺失直接失败，
只有请求尾端尚未入库的交易日可以跳过。`daily_basic` 与 `adj_factor` 必须覆盖
每一条日行情键，换手率、市值、涨跌停状态和复权因子也必须满足数值契约；部分
返回、空值或非法值不会被填成“正常”。

CSV 适配器定位为规范化研究数据和 Provider 契约入口，因此同样失败关闭：
`is_st`、`st_status_known`、`is_suspended`、`is_limit_up` 和
`is_limit_down` 必须显式存在、非空且可解析为布尔值。新数据源应先把自身状态
语义映射到这份规范契约，而不是依赖默认值。

AI 生成特征的公式即使本身因果，“哪些公式被筛中”也可能包含未来信息。正式
artifact 必须声明 `screening.label_data_end`（手工 artifact 可声明
`available_from`），且评分日必须严格晚于该日期。策略语义变更通过显式
`strategy_version` 阻断旧候选池 buffer 的跨版本复用。
