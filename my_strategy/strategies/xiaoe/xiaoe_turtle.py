#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
海龟仓位管理策略 — 基于 xiaoe_articles 博主股票池

核心理念:
  博主选股提供 alpha（入场/出场时机），海龟规则提供仓位管理框架：
  - ATR 波动率加权：波动大的股票少配，波动小的多配
  - 金字塔加仓：涨得好的股票不断加码
  - 止损保护：宽止损防黑天鹅，不防正常回调

与 buy_and_hold 的区别:
  - 共用同一套池子进出（池变化 = 调仓信号）
  - 但每只股票的仓位由 ATR 决定，不是等权
  - 趋势走出来的股票会金字塔加仓，走不出来的维持轻仓
"""

from pathlib import Path

import numpy as np

from my_strategy.common_file import project_root
from my_strategy.strategies.xiaoe.xiaoe_strategy import PoolLoader
from rqalpha.api import (
    history_bars, order_target_value, logger, scheduler,
)
from rqalpha.mod.rqalpha_mod_sys_scheduler.scheduler import physical_time


# ============================================================
# 策略参数
# ============================================================
TURTLE_PARAMS = {
    'atr_days': 20,          # ATR 计算周期
    'risk_pct': 0.03,        # 1 unit 的账户风险比例
    'max_units': 3,          # 单只股票最大加仓数
    'pyramid_atr': 0.5,      # 加仓间距 (× ATR)
    'max_total_units': 15,   # 全账户 unit 上限
    'stop_atr': 5.0,         # 硬止损距离 (很宽, 只防黑天鹅)
    'reentry_days': 20,      # 止损后重新入场需突破的通道
}


# ============================================================
# 技术指标
# ============================================================

def _bars(code, n, field):
    result = history_bars(code, n, '1d', field, adjust_type='pre')
    if result is None or len(result) < n:
        return None
    return np.asarray(result, dtype=float)


def _latest_close(code):
    bars = _bars(code, 1, 'close')
    return float(bars[-1]) if bars is not None else None


def _atr(code, n):
    """Wilder's ATR"""
    highs = _bars(code, n + 1, 'high')
    lows = _bars(code, n + 1, 'low')
    closes = _bars(code, n + 1, 'close')
    if highs is None or lows is None or closes is None:
        return None

    trs = []
    for i in range(1, len(highs)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)

    atr_val = trs[0]
    for tr in trs[1:]:
        atr_val = (atr_val * (n - 1) + tr) / n
    return atr_val


def _entry_breakout(code, entry_days):
    close = _latest_close(code)
    highs = _bars(code, entry_days + 1, 'high')
    if close is None or highs is None:
        return None
    return close > float(np.max(highs[:-1]))


# ============================================================
# 仓位计算
# ============================================================

def _calc_unit_value(total_value, price, atr, risk_pct):
    """1 unit = (risk_pct × total_value × price) / atr, 封顶 40%"""
    if atr <= 0 or price <= 0:
        return 0.0
    raw = (risk_pct * total_value * price) / atr
    return min(raw, total_value * 0.40)


def _total_units(context):
    return sum(s.get('units', 0) for s in context.turtle.values())


def _make_state():
    return {
        'units': 0,
        'unit_value': 0.0,
        'avg_entry': 0.0,
        'last_entry': 0.0,
        'highest_close': 0.0,
        'ever_entered': False,
    }


# ============================================================
# 策略生命周期
# ============================================================

def init(context):
    csv_dir = str(Path(project_root) / 'my_strategy' / 'strategies' / 'xiaoe_articles')
    params = TURTLE_PARAMS.copy()

    injected = getattr(context, 'strategy_params', None)
    if injected:
        plain = {}
        for k, v in injected.items():
            if hasattr(v, 'items'):
                plain[k] = dict(v)
            else:
                plain[k] = v
        params.update(plain)

    context.params = params
    context.loader = PoolLoader(csv_dir if 'pool_csv_dir' not in params else params['pool_csv_dir'])
    context.pool = context.loader.get_pool(context.now)
    context.turtle = {}
    context.old_pool = []

    logger.info(
        f"turtle 就绪 risk={params['risk_pct']:.1%} "
        f"atr={params['atr_days']}d max_units={params['max_units']} "
        f"stop={params['stop_atr']}×ATR pool={context.pool}"
    )

    scheduler.run_daily(sell_trade, time_rule=physical_time(hour=9, minute=35))
    scheduler.run_daily(buy_trade, time_rule=physical_time(hour=9, minute=37))


def before_trading(context):
    new_pool = context.loader.get_pool(context.now)
    context.old_pool = context.pool
    if set(new_pool) != set(context.pool):
        removed = set(context.pool) - set(new_pool)
        added = set(new_pool) - set(context.pool)
        if removed or added:
            logger.info(f"[池变化] +{list(added)} -{list(removed)}")
        context.pool = new_pool


def sell_trade(context, bar_dict):
    """出场: 出池清仓 + 宽止损防黑天鹅"""
    pool_set = set(context.pool)
    stop_atr = context.params['stop_atr']
    atr_days = context.params['atr_days']

    holdings = [c for c in context.portfolio.positions
                if context.portfolio.positions[c].quantity > 0]

    for code in holdings:
        state = context.turtle.get(code)

        # 出池 → 无条件清仓
        if code not in pool_set:
            if state:
                state['units'] = 0
            order_target_value(code, 0)
            logger.info(f"[turtle] 出池 {code}")
            continue

        if state is None or state['units'] == 0:
            continue

        price = _latest_close(code)
        if price is None:
            continue

        if price > state['highest_close']:
            state['highest_close'] = price

        # 宽止损: 最高点回撤 > stop_atr × ATR 才卖
        atr = _atr(code, atr_days)
        if atr is None or atr <= 0:
            continue

        hard_stop = state['avg_entry'] - stop_atr * atr
        trail_stop = state['highest_close'] - stop_atr * atr
        stop = max(hard_stop, trail_stop)

        if price <= stop:
            state['units'] = 0
            order_target_value(code, 0)
            logger.info(f"[turtle] 黑天鹅止损 {code} {price:.2f}≤{stop:.2f}")


def buy_trade(context, bar_dict):
    """入场: 池内股票首次入池→买, 已持仓→金字塔加仓, 止损后→突破再入"""
    params = context.params
    pool = context.pool
    if not pool:
        return

    total_value = context.portfolio.total_value

    for code in pool:
        state = context.turtle.get(code)
        if state is None:
            state = _make_state()
            context.turtle[code] = state

        # 同步
        if state['units'] > 0:
            pos = context.portfolio.positions.get(code)
            if not pos or pos.quantity <= 0:
                state['units'] = 0

        if state['units'] >= params['max_units']:
            continue

        if _total_units(context) >= params['max_total_units']:
            continue

        price = _latest_close(code)
        atr = _atr(code, params['atr_days'])
        if price is None or atr is None or atr <= 0:
            continue

        if state['units'] == 0:
            # 曾经入场过 → 需要突破通道才重新入场
            if state['ever_entered']:
                entry = _entry_breakout(code, params['reentry_days'])
                if entry is None or not entry:
                    continue
                logger.info(f"[turtle] 突破重新入场 {code}")

            # 首次入池 → 直接入场
            unit_value = _calc_unit_value(total_value, price, atr, params['risk_pct'])
            if unit_value <= 0:
                continue

            state['units'] = 1
            state['unit_value'] = unit_value
            state['avg_entry'] = price
            state['last_entry'] = price
            state['highest_close'] = price
            state['ever_entered'] = True

            order_target_value(code, unit_value)
            logger.info(
                f"[turtle] 入场 {code} unit=1/{params['max_units']} "
                f"v={unit_value:.0f}"
            )
        else:
            # 金字塔加仓
            threshold = state['last_entry'] + params['pyramid_atr'] * atr
            if price < threshold:
                continue

            new_units = state['units'] + 1
            current_value = context.portfolio.positions[code].market_value
            target_value = current_value + state['unit_value']

            state['units'] = new_units
            state['last_entry'] = price
            state['avg_entry'] = (
                state['avg_entry'] * (new_units - 1) + price
            ) / new_units
            if price > state['highest_close']:
                state['highest_close'] = price

            order_target_value(code, target_value)
            logger.info(
                f"[turtle] 加仓 {code} unit={new_units}/{params['max_units']} "
                f"target={target_value:.0f}"
            )


def handle_bar(context, bar_dict):
    pass
