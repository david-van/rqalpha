# JoinQuant 与 RQAlpha API 对比分析

> 生成依据：
>
> - JoinQuant 本地文档：`plantform_api/joinquant_api.html`
> - RQAlpha 本地文档：`docs/source/api/base_api.rst`、`docs/source/api/extend_api.rst`
> - RQAlpha 运行语义源码：`rqalpha/core/strategy.py`、`rqalpha/__init__.py`、相关 `rqalpha/apis/*`

## 1. 结论先行

如果把聚宽策略迁移到 RQAlpha，**最容易踩坑的不是单个行情 API，而是策略生命周期与调度模型不一致**。

最关键的差异有三类：

1. **生命周期函数名不同**
   - JoinQuant：`initialize` / `handle_data` / `before_trading_start` / `after_trading_end`
   - RQAlpha：`init` / `handle_bar` / `before_trading` / `after_trading` / `open_auction`
2. **定时调度接口不同**
   - JoinQuant：`run_daily/run_weekly/run_monthly`
   - RQAlpha：`scheduler.run_daily/run_weekly/run_monthly`
3. **数据与交易 API 表面相似，但默认行为、返回结构、字段语义不完全一致**
   - 例如 `get_price`、`get_bars`、`order_target_value` 等不能简单做字符串替换。

从能力覆盖上看：

- **JoinQuant 更偏“平台化一体服务”**：研究、回测、模拟盘、因子库、基本面、文件读写、消息推送、创建回测任务等都集成在平台 API 中。
- **RQAlpha 更偏“策略引擎 + 本地数据/扩展数据接口”**：基础 API 更聚焦撮合、账户、订阅、调度、历史行情；需要更丰富数据时走 `RQDatac` 扩展 API。

因此，**JoinQuant → RQAlpha 的迁移不能按“同名函数一一替换”来做，必须按“语义映射”来做**。

---

## 2. 对比方法与判定标准

本文采用四档判定：

- **等价**：接口名称可能不同，但主要用途、调用位置、核心语义基本一致
- **部分等价**：能覆盖一部分能力，但参数、时机、返回值、默认行为不同
- **不同**：名字相似，但运行语义差异较大，不建议直接替换
- **无直接对应**：另一平台没有官方直接对等 API

另外需要特别注意两个误区：

1. **同名不等价**：例如 `get_price` 在两个平台都存在，但返回结构、复权语义、上下文依赖都不完全一致。
2. **文档出现不代表正式基础 API 完全等价**：JoinQuant 有些能力属于平台服务；RQAlpha 有些能力属于扩展 API（依赖 `RQDatac`）。

---

## 3. 生命周期 API 对比

这是迁移时最重要的一组。

| JoinQuant | RQAlpha | 判定 | 说明 |
|---|---|---|---|
| `initialize(context)` | `init(context)` | 部分等价 | 都是初始化入口，但函数名不同；RQAlpha 只识别 `init`，不识别 `initialize` |
| `handle_data(context, data)` | `handle_bar(context, bar_dict)` | 不同 | 都是主策略逻辑入口，但参数结构和触发模型不同；RQAlpha 不识别 `handle_data` |
| `handle_tick(context, tick)` | `handle_tick(context, tick)` | 等价/部分等价 | 名字接近，但订阅与触发细节仍需按平台确认 |
| `before_trading_start(context)` | `before_trading(context)` | 部分等价 | 都是盘前回调，但名字不同 |
| `after_trading_end(context)` | `after_trading(context)` | 部分等价 | 都是盘后回调，但名字不同 |
| `on_event(context, event)` | `subscribe_event(...)` + 事件系统 | 部分等价 | JoinQuant 有统一 `on_event` 回调；RQAlpha 更偏事件订阅模型 |
| `on_strategy_end(context)` | 无直接对应的约定函数 | 无直接对应 | RQAlpha 文档中的约定函数没有同名策略结束回调 |
| `process_initialize(context)` | 无直接对应 | 无直接对应 | JoinQuant 用于进程重启时初始化不可持久化对象；RQAlpha 没有同名约定函数 |
| `after_code_changed(context)` | 无直接对应 | 无直接对应 | JoinQuant 模拟盘代码更新相关；RQAlpha 本地运行模型无同名接口 |

### 关键迁移提醒

- JoinQuant 策略里如果写的是 `initialize`，迁移到 RQAlpha 必须改成 `init`。
- JoinQuant 策略里如果主逻辑写在 `handle_data`，迁移到 RQAlpha 通常要改成 `handle_bar`。
- 这也是你前面那个 ETF 动量策略“回测成功但零交易”的直接根因之一。

---

## 4. 定时调度 API 对比

| JoinQuant | RQAlpha | 判定 | 说明 |
|---|---|---|---|
| `run_daily(func, ...)` | `scheduler.run_daily(function, time_rule=None)` | 部分等价 | 能力相似，但调用入口不同，回调签名也不同 |
| `run_weekly(func, ...)` | `scheduler.run_weekly(function, weekday/tradingday, time_rule=None)` | 部分等价 | 参数体系不同 |
| `run_monthly(func, ...)` | `scheduler.run_monthly(function, tradingday, time_rule=None)` | 部分等价 | 参数体系不同 |
| `unschedule_all()` | 无直接对应文档 API | 无直接对应 | JoinQuant 支持取消全部定时任务；RQAlpha 文档未见对等公开接口 |

### 语义差异

#### JoinQuant

- 调度函数是全局函数：`run_daily/run_weekly/run_monthly`
- 允许 `time='09:30'`、`time='every_bar'`、`reference_security=...`
- 文档明确建议：**尽量不要同时混用 `run_daily` 和 `handle_data`**

#### RQAlpha

- 调度器挂在 `scheduler` 对象上：`scheduler.run_daily/run_weekly/run_monthly`
- 文档中的回调签名要求是：`function(context, bar_dict)`
- 只能在 `init` 内调用
- 时间规则使用 `time_rule=market_open(...) / market_close(...) / physical_time(...)`

### 迁移建议

JoinQuant：

```python
run_daily(func, time='14:50')
```

迁到 RQAlpha 不能直接照抄，应改成类似：

```python
scheduler.run_daily(func, time_rule=physical_time(hour=14, minute=50))
```

并且 `func` 的签名要符合 RQAlpha 的要求。

---

## 5. 策略设置 / 运行环境 API 对比

### 5.1 基准、税费、滑点

| JoinQuant | RQAlpha | 判定 | 说明 |
|---|---|---|---|
| `set_benchmark` | 文档中未单列，但策略里常用，概念存在 | 部分等价 | 两边都支持设置基准，但文档组织方式不同 |
| `set_order_cost` | RQAlpha 通过账户/交易成本模块配置，策略里常见 `set_commission` 风格更常见 | 部分等价 | JoinQuant 强调 `OrderCost`，RQAlpha 更偏引擎成本模型 |
| `set_slippage` | 概念存在，策略里也常用 | 部分等价 | 都支持滑点，但对象模型和默认值说明不同 |
| `set_commission`（JoinQuant 已废弃） | RQAlpha 里策略代码常见 `set_commission(...)` 风格 | 不同 | JoinQuant 已废弃并建议改 `set_order_cost`；RQAlpha 生态中仍常见佣金设置接口 |

### 5.2 JoinQuant 平台设置项，RQAlpha 无直接对等公开 API

| JoinQuant | RQAlpha | 判定 | 说明 |
|---|---|---|---|
| `set_option('use_real_price', True)` | 无同名公开 API | 无直接对应 | RQAlpha 复权逻辑是另一套模型，不是通过这个平台选项暴露 |
| `order_volume_ratio` | 无直接对应 | 无直接对应 | 聚宽的撮合/成交量比例控制 |
| `match_with_order_book` | 无直接对应 | 无直接对应 | 聚宽盘口撮合模式 |
| `set_universe` | `update_universe` / `subscribe` / `unsubscribe` | 部分等价 | 都能影响标的池，但模型不同 |
| `disable_cache` | 无直接对应公开策略 API | 无直接对应 | 聚宽平台级缓存控制 |
| `set_option('avoid_future_data', True)` | 无直接同名公开 API | 无直接对应 | RQAlpha 没有同名策略级开关 |

---

## 6. 行情与历史数据 API 对比

这是两边最容易“看起来很像，实际上差很多”的部分。

### 6.1 基础行情类

| JoinQuant | RQAlpha | 判定 | 说明 |
|---|---|---|---|
| `get_price` | `get_price`（扩展 API，依赖 RQDatac） | 部分等价 | 名字相同，但 RQAlpha 的 `get_price` 属于扩展 API，不是基础引擎 API |
| `history` | 无同名公开基础 API | 无直接对应 | RQAlpha 常用的是 `history_bars` |
| `attribute_history` | `history_bars` | 部分等价 | 都是历史 K 线/字段查询，但调用形式和返回值不同 |
| `get_bars` | `history_bars` | 部分等价 | 都能拿 bar，但参数和返回结构不同 |
| `get_current_tick` | `current_snapshot` / `handle_tick` / `history_ticks` | 部分等价 | RQAlpha 没有同名公开基础 API |
| `get_ticks` | `history_ticks` | 部分等价 | 都是 tick 历史查询，但数据覆盖范围和字段未必完全一致 |
| `get_current_data` | `current_snapshot(order_book_id)` | 部分等价 | 都提供当前时点数据，但返回模型不同 |
| `get_extras` | 无单一直接对应 | 无直接对应 | 聚宽聚合了基金净值、ST、期货结算价、持仓量等额外字段；RQAlpha 倾向拆分为多个接口或扩展数据 |

### 6.2 合约/证券基础信息

| JoinQuant | RQAlpha | 判定 | 说明 |
|---|---|---|---|
| `get_all_securities` | `all_instruments` | 部分等价 | 都返回全量标的信息，但字段体系不同 |
| `get_security_info` | `instruments` | 部分等价 | 都是单标的详细信息 |
| `get_all_trade_days` | `get_trading_dates` | 部分等价 | 都能取交易日历 |
| `get_trade_days` | `get_trading_dates` / `get_previous_trading_date` / `get_next_trading_date` | 部分等价 | RQAlpha 更拆分 |
| `get_trade_day` | 无同名直接对应 | 无直接对应 | JoinQuant 支持“按标的+时刻反推交易日” |

### 6.3 指数 / 行业 / 概念 / 板块

| JoinQuant | RQAlpha | 判定 | 说明 |
|---|---|---|---|
| `get_index_stocks` | `index_components` | 部分等价 |
| `get_index_weights` | `index_weights` | 等价/部分等价 |
| `get_industry_stocks` | `industry` / `get_industry`（扩展） | 部分等价 |
| `get_concept_stocks` | `concept`（扩展） | 部分等价 |
| `get_industries` | `get_industry` / `get_instrument_industry` | 部分等价 |
| `get_concepts` | `concept` | 部分等价 |
| `get_industry` | `get_instrument_industry` / `get_industry` | 部分等价 |
| `get_concept` | 无单一直接对应 | 无直接对应 |
| `sector` | `sector` | 部分等价 |

### 6.4 公司基本面 / 因子 / 研究数据

| JoinQuant | RQAlpha | 判定 | 说明 |
|---|---|---|---|
| `get_fundamentals` | 无同名基础 API | 无直接对应 |
| `get_fundamentals_continuously` | 无同名基础 API | 无直接对应 |
| `finance.run_query` | 无直接对应基础 API | 无直接对应 |
| `macro.run_query` | `econ.get_reserve_ratio` / `econ.get_money_supply`（扩展） | 部分等价 | RQAlpha 只有少量宏观扩展接口，不是通用 query 模型 |
| `get_history_fundamentals` | `get_pit_financials_ex` / `current_performance`（扩展） | 部分等价 |
| `get_valuation` | 无直接对应公开 API | 无直接对应 |
| `get_all_factors` | `get_factor`（扩展） | 部分等价 |
| `get_factor_values` | `get_factor`（扩展） | 部分等价 |
| `get_factor_kanban_values` | 无直接对应 | 无直接对应 |
| `alpha101` / `alpha191` | 无直接对应 | 无直接对应 |
| `technical_analysis` | 无直接对应 | 无直接对应 |
| `neutralize` / `winsorize` / `winsorize_med` / `standardlize` | 无直接对应策略 API | 无直接对应 |

### 6.5 RQAlpha 行情/数据侧独有或更突出接口

| RQAlpha | JoinQuant | 判定 | 说明 |
|---|---|---|---|
| `history_bars` | `attribute_history` / `get_bars` | 部分等价 |
| `current_snapshot` | `get_current_data` / `get_current_tick` | 部分等价 |
| `get_yield_curve` | 无直接对应公开 API | 无直接对应 |
| `get_dividend` | JoinQuant 主要通过别的查询路径组合 | 部分等价/无直接对应 |
| `is_suspended` | `get_current_data()[security].paused` 一类语义 | 部分等价 |
| `is_st_stock` | `get_extras('is_st', ...)` | 部分等价 |
| `get_price_change_rate` | 无直接对应 | 无直接对应 |
| `get_split` | JoinQuant 可通过历史字段/公司行为间接获得，但无直接同名策略 API | 无直接对应 |
| `get_turnover_rate` | JoinQuant 没有同名策略 API | 无直接对应 |
| `get_stock_connect` | JoinQuant 有 `finance.run_query` 可查相关表 | 部分等价 |
| `current_performance` | JoinQuant 无同名公开策略 API | 无直接对应 |
| `get_securities_margin` / `get_margin_stocks` | JoinQuant 有融资融券专区 API，但组织方式不同 | 部分等价 |

---

## 7. 交易 API 对比

### 7.1 通用下单

| JoinQuant | RQAlpha | 判定 | 说明 |
|---|---|---|---|
| `order` | `order` | 部分等价 | 名字相同，但参数语义不同；JoinQuant 文档里是“按股数下单”，RQAlpha 文档里是“智能下单” |
| `order_target` | `order_to` / `order_target_*` 家族 | 部分等价 | JoinQuant 有明确目标股数；RQAlpha 的接口拆得更细 |
| `order_value` | `order_value` | 部分等价 | 名字相同，但证券/期货支持范围、风控细节不同 |
| `order_target_value` | `order_target_value` | 部分等价 | 最接近的一组，但仍不能假定完全兼容 |
| `cancel_order` | `cancel_order` | 等价/部分等价 |
| `get_open_orders` | `get_open_orders` | 等价/部分等价 |

### 7.2 JoinQuant 有，RQAlpha 无直接同名接口

| JoinQuant | RQAlpha | 判定 | 说明 |
|---|---|---|---|
| `get_orders` | 无直接对应文档 API | 无直接对应 |
| `get_trades` | 无直接对应文档 API | 无直接对应 |
| `inout_cash` | `deposit` / `withdraw` | 部分等价 |
| `batch_submit_orders` | 无直接对应 | 无直接对应 |
| `batch_cancel_orders` | 无直接对应 | 无直接对应 |

### 7.3 RQAlpha 有，JoinQuant 无直接同名接口

| RQAlpha | JoinQuant | 判定 | 说明 |
|---|---|---|---|
| `submit_order` | 无直接同名公开接口 | 无直接对应 |
| `order_shares` | `order` | 部分等价 |
| `order_lots` | 无直接对应 | 无直接对应 |
| `order_percent` | 无直接同名公开接口 | 无直接对应 |
| `order_target_percent` | 无直接同名公开接口 | 无直接对应 |
| `order_target_portfolio` | JoinQuant 组合优化/篮子下单能部分覆盖 | 部分等价 |
| `buy_open` / `sell_close` / `sell_open` / `buy_close` | JoinQuant 期货用 `order_target/order_value` + `side` 组织 | 部分等价 |
| `exercise` | 无直接对应 | 无直接对应 |

### 7.4 订单类型

| JoinQuant | RQAlpha | 判定 | 说明 |
|---|---|---|---|
| `OrderStyle` | `MarketOrder` / `LimitOrder` / `TWAPOrder` / `VWAPOrder` | 部分等价 |
| 停止单（文档中作为 OrderStyle 说明） | 无明显同名基础 API | 无直接对应 |
| `OrderStatus` | `ORDER_STATUS` / `Order` 状态 | 部分等价 |

---

## 8. 账户 / 组合 / 持仓对象对比

| JoinQuant | RQAlpha | 判定 | 说明 |
|---|---|---|---|
| `g` | `context` 自定义字段 / 全局变量体系 | 部分等价 | JoinQuant 强调 `g` 全局变量对象；RQAlpha 主要依赖 `context`，也有全局变量支持 |
| `Context` | `StrategyContext` | 部分等价 |
| `Portfolio` | `Portfolio` | 等价/部分等价 |
| `SubPortfolio` | `Account` / 多账户体系 | 部分等价 |
| `Position` | `StockPosition` / `FuturePosition` | 部分等价 |
| `data[security]` (`SecurityUnitData`) | `bar_dict[order_book_id]` (`BarObject`) | 不同 | 都是盘中可访问行情对象，但对象模型不同 |
| `tick` 对象 | `TickObject` | 部分等价 |
| `Trade` 对象 | `Order` / `Trade` 相关模型 | 部分等价 |
| `Event` 对象 | `EVENT` + 事件总线 | 部分等价 |

RQAlpha 在对象模型上更明确地区分：

- `StrategyContext`
- `RunInfo`
- `BarObject`
- `TickObject`
- `Order`
- `Portfolio`
- `Account`
- `StockPosition`
- `FuturePosition`

JoinQuant 则更强调平台对象：

- `g`
- `Context`
- `SubPortfolio`
- `Portfolio`
- `Position`
- `SecurityUnitData`

---

## 9. 其他工具 / 平台能力对比

### 9.1 JoinQuant 独有或明显更强的平台能力

| JoinQuant | RQAlpha | 判定 | 说明 |
|---|---|---|---|
| `record` | RQAlpha 文档里有 `plot` 概念，但生态与展示方式不同 | 部分等价 |
| `send_message` | 无直接对应 | 无直接对应 |
| `log` | `logger` / `user_log` 系列 | 部分等价 |
| `write_file` | 无直接对应官方策略 API | 无直接对应 |
| `read_file` | 无直接对应官方策略 API | 无直接对应 |
| `create_backtest` | 无直接对应 | 无直接对应 |
| `get_backtest` | 无直接对应 | 无直接对应 |
| `normalize_code` | 无直接对应官方策略 API | 无直接对应 |

### 9.2 RQAlpha 独有或更显式暴露的能力

| RQAlpha | JoinQuant | 判定 | 说明 |
|---|---|---|---|
| `open_auction` | 无直接同名生命周期函数 | 无直接对应 |
| `update_universe` | `set_universe` | 部分等价 |
| `subscribe` / `unsubscribe` | JoinQuant 的 tick 订阅体系命名不同 | 部分等价 |
| `subscribe_event` | `on_event` | 部分等价 |
| `deposit` / `withdraw` / `finance` / `repay` | `inout_cash` / 融资融券 API | 部分等价 |
| `scheduler.*` | `run_daily/run_weekly/run_monthly` | 部分等价 |

---

## 10. JoinQuant 有而 RQAlpha 文档中没有直接对应的 API 清单

以下 API 在 JoinQuant 文档中明确出现，但在当前 RQAlpha 本地文档中**没有直接同名或明确对等接口**。这类最容易在迁移时需要“重写而不是替换”。

### 生命周期 / 平台控制

- `initialize`
- `handle_data`
- `before_trading_start`
- `after_trading_end`
- `on_event`
- `on_strategy_end`
- `process_initialize`
- `after_code_changed`
- `unschedule_all`

### 平台设置项

- `use_real_price`
- `order_volume_ratio`
- `match_with_order_book`
- `set_universe`
- `disable_cache`
- `avoid_future_data`

### 数据 / 研究 / 因子 / 平台服务

- `history`
- `attribute_history`
- `get_current_tick`
- `get_ticks`
- `get_current_data`
- `get_extras`
- `get_all_factors`
- `get_factor_values`
- `get_factor_kanban_values`
- `get_fundamentals`
- `get_fundamentals_continuously`
- `finance.run_query`
- `macro.run_query`
- `get_billboard_list`
- `get_index_stocks`
- `get_industry_stocks`
- `get_concept_stocks`
- `get_industries`
- `get_concepts`
- `get_all_securities`
- `get_security_info`
- `get_all_trade_days`
- `get_trade_days`
- `get_money_flow`
- `get_concept`
- `get_call_auction`
- `get_trade_day`
- `get_history_fundamentals`
- `get_valuation`
- `alpha101`
- `alpha191`
- `technical_analysis`
- `neutralize`
- `winsorize`
- `winsorize_med`
- `standardlize`

### 交易 / 订单 / 平台输出

- `order_target`
- `get_orders`
- `get_trades`
- `inout_cash`
- `batch_submit_orders`
- `batch_cancel_orders`
- `record`
- `send_message`
- `write_file`
- `read_file`
- `create_backtest`
- `get_backtest`
- `normalize_code`

### 融资融券 / 期货平台 API

- `unsubscribe_all`
- `margincash_interest_rate`
- `margincash_margin_rate`
- `marginsec_interest_rate`
- `marginsec_margin_rate`
- `margincash_open`
- `margincash_close`
- `margincash_direct_refund`
- `marginsec_open`
- `marginsec_close`
- `marginsec_direct_refund`
- `get_margincash_stocks`
- `get_marginsec_stocks`
- `get_mtss`
- `get_dominant_future`
- `futures_margin_rate`
- `is_dangerous`

> 说明：这里的“没有直接对应”是指**当前本地 RQAlpha 文档没有直接给出同名/同级公开 API**，不代表完全做不到；很多能力可能需要改写成别的接口组合、订阅机制或扩展数据调用。

---

## 11. RQAlpha 有而 JoinQuant 文档中没有直接对应的 API 清单

以下 API 在 RQAlpha 本地文档中明确存在，但在 JoinQuant 本地 HTML 文档中没有直接出现为同级公开 API。

### 生命周期 / 调度

- `init`
- `handle_bar`
- `open_auction`
- `before_trading`
- `after_trading`
- `scheduler.run_daily`
- `scheduler.run_weekly`
- `scheduler.run_monthly`

### 交易接口

- `submit_order`
- `order_to`
- `order_shares`
- `order_lots`
- `order_percent`
- `order_target_percent`
- `order_target_portfolio`
- `buy_open`
- `sell_close`
- `sell_open`
- `buy_close`
- `exercise`

### 账户 / 查询 / 运行控制

- `get_position`
- `get_positions`
- `all_instruments`
- `instruments`
- `active_instrument`
- `instrument_history`
- `active_instruments`
- `instruments_history`
- `history_bars`
- `current_snapshot`
- `get_previous_trading_date`
- `get_next_trading_date`
- `history_ticks`
- `get_yield_curve`
- `sector`
- `get_dividend`
- `is_suspended`
- `is_st_stock`
- `update_universe`
- `subscribe_event`
- `deposit`
- `withdraw`
- `finance`
- `repay`

### 扩展 API（RQDatac）

- `get_price_change_rate`
- `get_split`
- `get_securities_margin`
- `concept`
- `get_margin_stocks`
- `get_shares`
- `get_turnover_rate`
- `get_factor`
- `get_instrument_industry`
- `get_stock_connect`
- `current_performance`
- `get_pit_financials_ex`
- `index_components`
- `index_weights`
- `get_dominant`
- `get_member_rank`
- `get_warehouse_stocks`
- `get_dominant_price`
- `econ.get_reserve_ratio`
- `econ.get_money_supply`

---

## 12. 迁移时最需要人工改写的 10 个点

如果你要把 JoinQuant 策略迁到 RQAlpha，下面这 10 个点最值得优先处理：

1. **`initialize` 改成 `init`**
2. **`handle_data` 改成 `handle_bar`**
3. **`before_trading_start` 改成 `before_trading`**
4. **`after_trading_end` 改成 `after_trading`**
5. **`run_daily/run_weekly/run_monthly` 改成 `scheduler.run_*`**
6. **`data[security]` 风格改成 `bar_dict[order_book_id]` 或 `current_snapshot(...)`**
7. **`attribute_history/history/get_bars` 迁移到 `history_bars` 或扩展 `get_price`**
8. **`set_universe` 改成 `update_universe` / `subscribe` / `unsubscribe` 的组合**
9. **`get_fundamentals/query/valuation/...` 这类研究型 API 不能简单替换，通常要改成 RQDatac 扩展接口或外部数据源**
10. **不要假设同名下单 API 完全兼容，特别是 `order` / `order_target_value` / 期货 side 语义**

---

## 13. 针对本仓库当前问题的直接解释

你前面的 `strategies/ETF动量/backtest.py` 正是一个典型例子：

- 它使用了 JoinQuant / 类 JoinQuant 风格的：
  - `initialize`
  - `handle_data`
  - `run_daily(...)`
- 但当前执行引擎是 RQAlpha

因此最先出问题的不是数据源，而是**生命周期函数根本没有被 RQAlpha 按预期识别**。

这也是为什么会出现：

- 回测能跑完
- 但没有任何交易
- `trades` 为空
- `positions` 为空

---

## 14. 本文档的边界与未决项

为了避免误导，以下内容需要特别说明：

1. **本文比较的是“本地可见文档 + 本地源码证据”**，不是联网抓取最新线上文档。
2. **JoinQuant 文档包含大量平台能力**，其中有些能力不只是“策略 API”，还包括平台服务与研究环境接口。
3. **RQAlpha 扩展 API 依赖 `RQDatac`**，因此在“能力是否存在”上，要区分：
   - RQAlpha 基础引擎 API
   - RQAlpha + RQDatac 扩展 API
4. 某些接口虽然可以“功能上替代”，但**并不意味着可以做无脑字符串替换**。

---

## 15. 建议的迁移顺序

如果后面你要把聚宽策略批量迁移到 RQAlpha，推荐按这个顺序做：

1. **先改生命周期与调度**
2. **再改行情 API**
3. **再改交易 API**
4. **最后处理基本面 / 因子 / 平台服务 API**

因为前两步决定策略“能不能正常跑”，后两步决定“结果准不准、功能全不全”。

---

## 16. 附：本地文档中提取到的 RQAlpha 正式 API 主清单

### 基础 API

- 生命周期：`init`、`handle_bar`、`handle_tick`、`open_auction`、`before_trading`、`after_trading`
- 订单类型：`MarketOrder`、`LimitOrder`、`TWAPOrder`、`VWAPOrder`
- 交易：`submit_order`、`order`、`order_to`、`order_shares`、`order_lots`、`order_value`、`order_percent`、`order_target_value`、`order_target_percent`、`order_target_portfolio`、`buy_open`、`sell_close`、`sell_open`、`buy_close`、`cancel_order`、`get_open_orders`、`exercise`
- 持仓：`get_position`、`get_positions`
- 数据：`all_instruments`、`instruments`、`active_instrument`、`instrument_history`、`active_instruments`、`instruments_history`、`history_bars`、`current_snapshot`、`get_trading_dates`、`get_previous_trading_date`、`get_next_trading_date`、`history_ticks`、`get_yield_curve`、`industry`、`sector`、`get_dividend`、`is_suspended`、`is_st_stock`、`get_future_contracts`
- 其他：`update_universe`、`subscribe`、`unsubscribe`、`subscribe_event`、`deposit`、`withdraw`、`finance`、`repay`、`scheduler.run_daily`、`scheduler.run_weekly`、`scheduler.run_monthly`

### 扩展 API（RQDatac）

- 行情：`get_price`、`get_price_change_rate`
- 股票：`get_split`、`get_securities_margin`、`concept`、`get_margin_stocks`、`get_shares`、`get_turnover_rate`、`get_factor`、`get_industry`、`get_instrument_industry`、`get_stock_connect`、`current_performance`、`get_pit_financials_ex`
- 指数：`index_components`、`index_weights`
- 期货：`get_dominant`、`get_member_rank`、`get_warehouse_stocks`、`get_dominant_price`
- 宏观：`econ.get_reserve_ratio`、`econ.get_money_supply`
