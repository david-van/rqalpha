#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
七星高照ETF轮动策略 V1.7 — RQAlpha 版本

原策略来源：聚宽 https://www.joinquant.com/post/70855
原作者：屌丝逆袭量化
优化时间：2026-3-29

策略逻辑：
- 基于加权动量得分(R² × 年化收益率)对ETF池进行排名
- 支持盈利保护、溢价率过滤、成交量过滤、短期动量过滤
- 无目标时自动切换到防御ETF（货币基金）
- 支持多时间点盈利保护独立检查

迁移说明：
- 溢价率过滤依赖基金净值数据（RQAlpha原生不支持），默认关闭
- 成交量放量过滤使用日线级别近似替代JoinQuant的分钟线累积
"""

import math
import numpy as np

from rqalpha.api import (
    history_bars,
    order_target_value,
    logger,
    scheduler,
    current_snapshot,
    is_suspended,
    instruments,
    get_previous_trading_date,
)
from rqalpha.mod.rqalpha_mod_sys_scheduler.scheduler import physical_time


# ==================== 辅助函数 ====================

def _name(code):
    """获取ETF名称"""
    try:
        return instruments(code).symbol
    except Exception:
        return code


# ==================== 盈利保护模块 ====================

def check_profit_protection(code, context, lookback=None, threshold=None):
    """
    检查是否触发盈利保护：从最近N日最高点回撤超过阈值

    Returns:
        bool: True 表示应触发盈利保护
    """
    if not context.enable_profit_protection:
        return False

    lb = lookback or context.profit_protection_lookback
    th = threshold or context.profit_protection_threshold

    high = history_bars(code, lb, '1d', 'high', adjust_type='pre')
    if high is None or len(high) < lb:
        logger.debug(f"{code} {_name(code)} 历史数据不足{lb}天，无法检查盈利保护")
        return False

    max_high = float(high.max())
    current_price = current_snapshot(code).last

    if current_price <= max_high * (1 - th):
        pullback = (1 - current_price / max_high) * 100
        logger.info(
            f"🔻 {code} {_name(code)} 触发盈利保护："
            f"当前价{current_price:.3f}，最近{lb}日最高{max_high:.3f}，"
            f"回撤{pullback:.2f}% > {th*100:.0f}%"
        )
        return True
    return False


# ==================== 溢价率模块 ====================

def get_premium_rate(code, date):
    """
    获取指定日期的溢价率（使用前一日净值，适合盘中判断）

    RQAlpha 原生不支持基金净值查询。如需启用溢价率过滤：
    1. 接入 RQData: from rqdatac import fund; fund.get_nav(code, date)
    2. 自行维护净值文件

    当前实现：无法获取净值时返回 (None, None, None)，溢价率过滤自动跳过。

    Returns:
        (premium_rate, price, net_value) 或 (None, None, None)
    """
    price = history_bars(code, 1, '1d', 'close', adjust_type='pre')[-1]

    net_value = _try_get_nav(code, date)
    if net_value is None or net_value <= 0:
        return None, None, None

    premium_rate = (price - net_value) / net_value
    return premium_rate, price, net_value


def _try_get_nav(code, date):
    """
    尝试获取基金净值，默认返回 None。
    用户可替换此函数接入自己的净值数据源。
    """
    # ---- RQData ----
    # try:
    #     from rqdatac import fund
    #     nav = fund.get_nav(code, date, date)
    #     if nav is not None and len(nav) > 0:
    #         return float(nav.iloc[0]['unit_nav'])
    # except Exception:
    #     pass

    # ---- 本地文件 ----
    # try:
    #     import pickle
    #     with open(r'D:\datas\fund_nav.pkl', 'rb') as f:
    #         nav_data = pickle.load(f)
    #     return nav_data.get(code, {}).get(str(date))
    # except Exception:
    #     pass

    return None


# ==================== 成交量检查 ====================

def get_volume_ratio(context, code, lookback=None, threshold=None):
    """
    检查当日成交量与过去N日均量的比值，若超过阈值则返回比值
    """
    lb = lookback or context.volume_lookback
    th = threshold or context.volume_threshold

    try:
        vols = history_bars(code, lb + 1, '1d', 'volume', include_now=True)
        if vols is None or len(vols) < lb + 1:
            return None

        today_vol = float(vols[-1])
        avg_vol = float(vols[:-1].mean())
        if avg_vol <= 0:
            return None

        ratio = today_vol / avg_vol
        if ratio > th:
            logger.debug(f"{code} {_name(code)} 成交量比{ratio:.2f} > {th}")
            return ratio
        return None
    except Exception as e:
        logger.warn(f"成交量计算失败 {code}: {e}")
        return None


# ==================== 动量计算模块 ====================

def _annualized_returns(price_series, lookback_days):
    """加权年化收益率（线性衰减加权回归）"""
    recent = price_series[-(lookback_days + 1):]
    y = np.log(recent)
    x = np.arange(len(y))
    weights = np.linspace(1, 2, len(y))
    slope, _ = np.polyfit(x, y, 1, w=weights)
    return math.exp(slope * 250) - 1


def calculate_momentum_metrics(context, code):
    """
    计算单只ETF的动量指标，应用所有过滤条件。

    过滤顺序:
      1. 盈利保护 — 当前价从近期高点回撤超阈值则排除
      2. 成交量放量 — 放量且年化收益过高则排除
      3. 短期动量 — 短期动量不足则排除
      4. 长期动量 — 加权回归计算年化收益率 × R² 作为得分
      5. 近3日单日跌幅 — 任一日跌幅超阈值则排除

    Returns:
        dict or None
    """
    try:
        etf_name = _name(code)
        lookback = max(context.lookback_days, context.short_lookback_days) + 20

        close = history_bars(code, lookback, '1d', 'close', adjust_type='pre')
        if len(close) < context.lookback_days:
            logger.debug(f"{code} {etf_name} 历史数据不足{len(close)}天，跳过")
            return None

        current_price = current_snapshot(code).last
        price_series = np.append(close, current_price)

        # ---- 1. 盈利保护 ----
        if check_profit_protection(code, context):
            logger.info(f"🚫 {code} {etf_name} 触发盈利保护，从排名中排除")
            return None

        # ---- 2. 成交量过滤 ----
        if context.enable_volume_check:
            vol_ratio = get_volume_ratio(context, code)
            if vol_ratio is not None:
                ann = _annualized_returns(price_series, context.lookback_days)
                if ann > context.volume_return_limit:
                    logger.info(
                        f"📉 {code} {etf_name} 成交量放量{vol_ratio:.1f}倍，"
                        f"且年化{ann*100:.1f}% > 阈值{context.volume_return_limit*100:.1f}%，过滤"
                    )
                    return None

        # ---- 3. 短期动量过滤 ----
        n = context.short_lookback_days
        if len(price_series) >= n + 1:
            short_ret = price_series[-1] / price_series[-(n + 1)] - 1
            short_annualized = (1 + short_ret) ** (250 / n) - 1
        else:
            short_annualized = 0.0

        if context.use_short_momentum_filter and short_annualized < context.short_momentum_threshold:
            logger.debug(
                f"{code} {etf_name} 短期动量{short_annualized*100:.1f}% "
                f"< 阈值{context.short_momentum_threshold*100:.1f}%，过滤"
            )
            return None

        # ---- 4. 长期动量（加权回归 + R²）----
        recent = price_series[-(context.lookback_days + 1):]
        y = np.log(recent)
        x = np.arange(len(y))
        weights = np.linspace(1, 2, len(y))
        slope, intercept = np.polyfit(x, y, 1, w=weights)
        annualized_returns = math.exp(slope * 250) - 1

        y_pred = slope * x + intercept
        ss_res = np.sum(weights * (y - y_pred) ** 2)
        y_mean = np.average(y, weights=weights)
        ss_tot = np.sum(weights * (y - y_mean) ** 2)
        r_squared = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

        score = annualized_returns * r_squared

        # ---- 5. 近3日单日跌幅过滤 ----
        if len(price_series) >= 4:
            day1 = price_series[-1] / price_series[-2]
            day2 = price_series[-2] / price_series[-3]
            day3 = price_series[-3] / price_series[-4]
            if min(day1, day2, day3) < context.loss_threshold:
                logger.info(
                    f"⚠️ {code} {etf_name} "
                    f"近3日有单日跌幅超{(1 - context.loss_threshold)*100:.1f}%，直接排除"
                )
                return None

        return {
            'etf': code,
            'etf_name': etf_name,
            'annualized_returns': annualized_returns,
            'r_squared': r_squared,
            'score': score,
            'current_price': current_price,
            'short_annualized': short_annualized,
        }

    except Exception as e:
        logger.warn(f"计算{code} {_name(code)}时出错: {e}")
        return None


# ==================== 排名缓存 ====================

def get_cached_rankings(context):
    """获取缓存的ETF排名，同一交易日内多次调用结果一致"""
    today = context.now.date()
    if getattr(context, '_rankings_cache_date', None) != today:
        logger.info("重新计算ETF排名...")
        ranked = _get_ranked_etfs(context)
        context._rankings_cache_date = today
        context._rankings_cache = ranked
    else:
        logger.debug("使用缓存的ETF排名")
    return context._rankings_cache


def _get_ranked_etfs(context):
    """计算所有ETF的动量得分，过滤后按得分降序排列"""
    etf_metrics = []
    for etf in context.etf_pool:
        # 未上市过滤
        if context.now.date() < instruments(etf).listed_date.date():
            continue
        # 停牌过滤
        if is_suspended(etf):
            logger.debug(f"{etf} {_name(etf)} 停牌，跳过")
            continue

        metrics = calculate_momentum_metrics(context, etf)
        if metrics is None:
            continue

        # 得分范围过滤
        if context.min_score_threshold < metrics['score'] < context.max_score_threshold:
            etf_metrics.append(metrics)
        else:
            logger.debug(
                f"{etf} {metrics['etf_name']} "
                f"得分{metrics['score']:.2f}超出阈值，过滤"
            )

    etf_metrics.sort(key=lambda x: x['score'], reverse=True)
    return etf_metrics


# ==================== 防御ETF ====================

def _defensive_available(context):
    """检查防御ETF是否可交易（未停牌、未涨跌停）"""
    code = context.defensive_etf
    snap = current_snapshot(code)
    if is_suspended(code):
        logger.debug(f"防御ETF {code} {_name(code)} 停牌")
        return False
    if snap.last >= snap.limit_up:
        logger.debug(f"防御ETF {code} {_name(code)} 涨停")
        return False
    if snap.last <= snap.limit_down:
        logger.debug(f"防御ETF {code} {_name(code)} 跌停")
        return False
    return True


# ==================== 下单 ====================

def _order_to(code, target_value, context):
    """
    智能下单：根据目标市值调整持仓，处理停牌、涨跌停、最小交易金额

    Returns:
        bool: 是否成功下单
    """
    name = _name(code)
    snap = current_snapshot(code)
    price = snap.last

    if is_suspended(code):
        logger.info(f"{code} {name} 停牌，跳过")
        return False

    if price == 0:
        logger.info(f"{code} {name} 当前价格0，跳过")
        return False

    pos = context.portfolio.positions.get(code)
    current_val = pos.market_value if pos else 0.0

    # 5% 容差
    if target_value > 0 and abs(current_val - target_value) <= target_value * 0.05:
        return False

    # 涨跌停
    if target_value > current_val and price >= snap.limit_up:
        logger.info(f"{code} {name} 涨停，跳过买入")
        return False
    if target_value < current_val and price <= snap.limit_down:
        logger.info(f"{code} {name} 跌停，跳过卖出")
        return False

    # 最小交易金额
    trade_val = abs(target_value - current_val)
    if 0 < trade_val < context.min_money:
        logger.info(f"{code} {name} 交易金额{trade_val:.2f} < {context.min_money}，跳过")
        return False

    # T+1 处理
    if target_value < current_val:
        sellable = pos.sellable if pos else 0
        if sellable == 0:
            logger.info(f"{code} {name} 当天买入不可卖出")
            return False

    # 计算目标股数（按100股取整），若与当前持仓一致则无需交易
    target_amount = int(target_value / price)
    target_amount = (target_amount // 100) * 100
    if target_amount <= 0 and target_value > 0:
        target_amount = 100
    cur_amount = pos.quantity if pos else 0
    diff = target_amount - cur_amount

    if diff == 0:
        return False

    order_result = order_target_value(code, target_value)
    if order_result:
        action = "买入" if diff > 0 else "卖出"
        logger.info(f"📦 {action}: {code} {name} 数量{abs(diff)} 目标市值{target_value:.2f}")
        return True
    else:
        logger.warn(f"下单失败: {code} {name}")
        return False


# ==================== 交易逻辑 ====================

def check_positions(context, bar_dict=None):
    """每日开盘检查持仓状态"""
    for code in list(context.portfolio.positions.keys()):
        pos = context.portfolio.positions[code]
        if pos.quantity > 0:
            logger.info(
                f"📊 持仓：{code} {_name(code)} "
                f"数量{pos.quantity} 成本{pos.avg_price:.3f} "
                f"现价{pos.last_price:.3f}"
            )


def profit_protection_check(context, bar_dict=None):
    """
    独立执行的盈利保护检查函数
    遍历所有持仓，若触发盈利保护则卖出
    """
    if not context.enable_profit_protection:
        logger.debug("盈利保护模块已关闭，跳过检查")
        return

    logger.info("========== 盈利保护独立检查开始 ==========")
    for code in list(context.portfolio.positions.keys()):
        if code not in context.etf_pool and code != context.defensive_etf:
            continue
        pos = context.portfolio.positions[code]
        if pos.quantity > 0:
            if check_profit_protection(code, context):
                if _order_to(code, 0, context):
                    logger.info(f"🛡️ 盈利保护卖出（独立检查）：{code} {_name(code)}")
    logger.info("========== 盈利保护独立检查完成 ==========")


def etf_sell_trade(context, bar_dict=None):
    """卖出不符合条件的持仓（排名变化、溢价率过高）"""
    logger.info("========== 卖出操作开始 ==========")

    ranked = get_cached_rankings(context)
    # 目标ETF列表（得分前N名）
    target_etfs = []
    for m in ranked[:context.holdings_num]:
        if m['score'] >= context.min_score_threshold:
            target_etfs.append(m['etf'])

    defensive_ok = _defensive_available(context)
    if not target_etfs and defensive_ok:
        target_etfs = [context.defensive_etf]

    target_set = set(target_etfs)

    # 卖出不在目标的持仓
    for code in list(context.portfolio.positions.keys()):
        if code not in context.etf_pool and code != context.defensive_etf:
            continue
        if code not in target_set:
            pos = context.portfolio.positions[code]
            if pos.quantity > 0:
                if _order_to(code, 0, context):
                    logger.info(f"📤 卖出不在目标的持仓：{code} {_name(code)}")

    # 溢价率检查
    if context.enable_premium_filter:
        prev_date = get_previous_trading_date(context.now, n=1)
        for code in list(context.portfolio.positions.keys()):
            if code not in context.etf_pool and code != context.defensive_etf:
                continue
            pos = context.portfolio.positions[code]
            if pos.quantity > 0:
                premium, _, _ = get_premium_rate(code, prev_date)
                if premium is not None and premium > context.premium_threshold:
                    if _order_to(code, 0, context):
                        logger.info(
                            f"🚨 溢价率过高 {code} {_name(code)} "
                            f"溢价率{premium*100:.2f}% > {context.premium_threshold*100:.0f}%，卖出"
                        )

    logger.info("========== 卖出操作完成 ==========")


def etf_buy_trade(context, bar_dict=None):
    """买入符合条件的ETF，等权分配，按排名顺序逐个尝试直到凑够持仓数量"""
    logger.info("========== 买入操作开始 ==========")

    ranked = get_cached_rankings(context)

    # 打印排名前5
    top_n = min(5, len(ranked))
    if top_n > 0:
        logger.info(f"=== ETF排名前{top_n} ===")
        for i, m in enumerate(ranked[:top_n]):
            logger.info(
                f"排名{i+1}: {m['etf']} {m['etf_name']} "
                f"得分{m['score']:.4f} 年化{m['annualized_returns']*100:.2f}% "
                f"R²={m['r_squared']:.4f}"
            )

    # 确定目标ETF列表：依次尝试排名靠前的ETF
    target_etfs = []
    prev_date = None
    if context.enable_premium_filter:
        prev_date = get_previous_trading_date(context.now, n=1)

    for m in ranked:
        if len(target_etfs) >= context.holdings_num:
            break

        if m['score'] < context.min_score_threshold:
            continue

        code = m['etf']

        # 盈利保护检查（买入前再次检查，防止卖了又买）
        if context.enable_profit_protection and check_profit_protection(code, context):
            logger.info(f"🚫 {code} {_name(code)} 触发盈利保护，从买入候选列表中排除")
            continue

        # 溢价率过滤
        if context.enable_premium_filter:
            premium, price, net = get_premium_rate(code, prev_date)
            if premium is None:
                logger.info(f"⚠️ {code} {_name(code)} 无法获取溢价率，视为不合格，跳过")
                continue
            if premium > context.premium_threshold:
                logger.info(
                    f"🚫 {code} {_name(code)} "
                    f"溢价率{premium*100:.2f}% > {context.premium_threshold*100:.0f}%，跳过"
                )
                continue
            logger.info(
                f"✅ {code} {_name(code)} "
                f"溢价率{premium*100:.2f}% ≤ {context.premium_threshold*100:.0f}%，通过"
            )

        target_etfs.append(code)
        logger.info(f"🎯 目标ETF {len(target_etfs)}: {code} {m['etf_name']} 得分{m['score']:.4f}")

    # 防御模式
    if not target_etfs:
        if _defensive_available(context):
            target_etfs = [context.defensive_etf]
            logger.info(f"🛡️ 进入防御模式，选择防御ETF：{context.defensive_etf} {_name(context.defensive_etf)}")
        else:
            logger.info("💤 无目标ETF且防御不可用，保持空仓")
            return

    # 检查是否有持仓需要先卖出（不在目标列表的持仓）
    current_positions = [
        c for c in context.portfolio.positions
        if c in context.etf_pool or c == context.defensive_etf
    ]
    to_sell = [c for c in current_positions if c not in target_etfs]
    if to_sell:
        names = [_name(c) for c in to_sell]
        logger.info(f"尚有持仓需要卖出：{list(zip(to_sell, names))}，等待卖出完成再买入")
        return

    # 等权分配
    total_val = context.portfolio.total_value
    target_per_etf = total_val / len(target_etfs)

    for code in target_etfs:
        pos = context.portfolio.positions.get(code)
        current_val = pos.market_value if pos else 0.0
        if abs(current_val - target_per_etf) > target_per_etf * 0.05 or current_val == 0:
            _order_to(code, target_per_etf, context)

    logger.info("========== 买入操作完成 ==========")


# ==================== RQAlpha 入口 ====================

def init(context):
    """初始化函数：设置ETF池、核心参数、调度任务"""

    logger.info("========== 策略初始化开始 ==========")

    # ---- ETF池 ----
    context.etf_pool = [
        "518880.XSHG",   # 黄金ETF
        "159985.XSHE",   # 豆粕ETF
        "501018.XSHG",   # 南方原油
        "161226.XSHE",   # 白银LOF
        "513100.XSHG",   # 纳指ETF
        "159915.XSHE",   # 创业板ETF
        "511220.XSHG",   # 城投债ETF
    ]

    # ---- 核心参数 ----
    context.lookback_days = 25
    context.holdings_num = 1
    context.defensive_etf = "511880.XSHG"   # 银华日利（货币ETF）
    context.min_money = 5000

    # ---- 盈利保护参数 ----
    context.enable_profit_protection = True
    context.profit_protection_lookback = 1
    context.profit_protection_threshold = 0.05
    context.profit_protection_check_times = ['11:00']

    # ---- 动量过滤参数 ----
    context.loss_threshold = 0.97
    context.min_score_threshold = 0
    context.max_score_threshold = 100.0
    context.use_short_momentum_filter = True
    context.short_lookback_days = 10
    context.short_momentum_threshold = 0.0

    # ---- 成交量过滤 ----
    context.enable_volume_check = True
    context.volume_lookback = 5
    context.volume_threshold = 2
    context.volume_return_limit = 1.0

    # ---- 溢价率过滤（默认关闭，需外部净值数据源）----
    context.enable_premium_filter = False
    context.premium_threshold = 0.20

    # ---- 参数覆盖（run.py 通过 strategy_params 注入）----
    params = getattr(context, 'strategy_params', None)
    if params:
        for k, v in params.items():
            setattr(context, k, v)

    # ---- 运行时变量 ----
    context._rankings_cache_date = None
    context._rankings_cache = None

    # ---- 注册调度 ----
    scheduler.run_daily(check_positions, time_rule=physical_time(hour=9, minute=10))
    scheduler.run_daily(etf_sell_trade, time_rule=physical_time(hour=14, minute=0))
    scheduler.run_daily(etf_buy_trade, time_rule=physical_time(hour=14, minute=1))

    for check_time in context.profit_protection_check_times:
        h, m = check_time.split(':')
        scheduler.run_daily(
            profit_protection_check,
            time_rule=physical_time(hour=int(h), minute=int(m))
        )
        logger.info(f"已注册盈利保护检查时间：{check_time}")

    logger.info(
        f"策略初始化完成：ETF池{len(context.etf_pool)}只，"
        f"动量周期{context.lookback_days}天，持仓{context.holdings_num}只"
    )
    logger.info(
        f"盈利保护开关：{'开启' if context.enable_profit_protection else '关闭'}，"
        f"回看周期{context.profit_protection_lookback}天，"
        f"回撤阈值{context.profit_protection_threshold*100:.0f}%"
    )
    if context.enable_premium_filter:
        logger.info(f"溢价率过滤已启用，阈值：{context.premium_threshold*100:.0f}%")
    else:
        logger.info("溢价率过滤未启用")
    logger.info("========== 策略初始化完成 ==========")


def handle_bar(context, bar_dict):
    """所有交易逻辑已通过 scheduler.run_daily 注册"""
    pass
