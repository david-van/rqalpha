# -*- coding: utf-8 -*-
"""
S2 红利指数再平衡策略
----------------------
持4只红利ETF，按月等权再平衡，带止盈止损。

资金: 20000元 (pindex=1)
ETF: 中证红利(25%) / 上证红利(20%) / 红利低波(30%) / 红利低波50(25%)
止损: -8% | 止盈: +20%
"""
from jqdata import *
import numpy as np
import pandas as pd
import datetime


# ==================== 简易日志 ====================
def _slog(sid, module, msg, level='info'):
    prefix = f"[{sid}] {module}"
    text = f"{prefix} | {msg}"
    getattr(log, level)(text)


# ==================== 策略配置 ====================
def s2_init_config(context):
    g.hongli_strategy = {
        'name': '4支红利指数', 'pindex': 1,
        'etfs': {
            '中证红利ETF': '515080.XSHG',
            '上证红利ETF': '510880.XSHG',
            '红利低波ETF': '512890.XSHG',
            '红利低波50ETF': '515450.XSHG',
        },
        'target_ratio': {
            '中证红利ETF': 0.25,
            '上证红利ETF': 0.20,
            '红利低波ETF': 0.30,
            '红利低波50ETF': 0.25,
        },
        'lookback_days': 180,
        'min_lookback': 60,
        'std_multiplier': 1.5,
        'vol_adjust': True,
        'vol_lookback': 60,
        'min_trade_value': 1000,
        'stop_loss': -0.08,         # 单只亏损-8%止损
        'stop_gain': 0.20,          # 单只盈利+20%止盈
        'rebalance_freq': 'monthly',
        'hk_max_ratio': 0.15,
        'debug': True,
        'rebalance_cap': None,
    }


# ==================== 定时任务注册 ====================
def s2_setup_schedule():
    run_daily(hongli_before_trading, 'before_open')
    run_monthly(hongli_trade_rebalance, 1, time='10:00')


# ==================== 策略函数 ====================
def hongli_before_trading(context):
    """盘前检查止损止盈"""
    strategy = g.hongli_strategy
    pindex = strategy['pindex']
    positions = context.subportfolios[pindex].positions
    cdata = get_current_data()

    for etf_name, etf_code in strategy['etfs'].items():
        if etf_code not in positions:
            continue
        pos = positions[etf_code]
        if pos.total_amount == 0:
            continue
        if cdata[etf_code].paused:
            continue
        current_price = cdata[etf_code].last_price
        if current_price >= cdata[etf_code].high_limit:
            continue
        if current_price <= cdata[etf_code].low_limit:
            continue

        pnl_ratio = (current_price - pos.avg_cost) / pos.avg_cost

        if pnl_ratio <= strategy['stop_loss']:
            _slog('S2', '止损', f'{etf_name}({etf_code}) 亏损{pnl_ratio*100:.1f}%, 触发止损', 'warning')
            order_target(etf_code, 0, pindex=pindex)

        elif pnl_ratio >= strategy['stop_gain']:
            _slog('S2', '止盈', f'{etf_name}({etf_code}) 盈利{pnl_ratio*100:.1f}%, 触发止盈', 'warning')
            order_target(etf_code, 0, pindex=pindex)


def hongli_get_etf_price(etf_code, context):
    try:
        return get_current_data()[etf_code].last_price
    except:
        return None


def hongli_trade_rebalance(context):
    """每月1号按目标权重再平衡"""
    strategy = g.hongli_strategy
    pindex = strategy['pindex']
    total_value = context.subportfolios[pindex].total_value

    _slog('S2', '再平衡', f'总资产: {total_value:.2f}')

    valid_etfs = {k: v for k, v in strategy['etfs'].items()
                  if not get_current_data()[v].paused}
    if not valid_etfs:
        log.warn(f"[{strategy['name']}] 没有有效的ETF数据，跳过")
        return

    for etf_name, etf_code in valid_etfs.items():
        target_value = total_value * strategy['target_ratio'].get(etf_name, 0.25)
        position = context.subportfolios[pindex].positions.get(etf_code, None)
        current_value = position.value if position else 0

        if abs(current_value - target_value) < 1000:
            continue

        current_price = hongli_get_etf_price(etf_code, context)
        if not current_price:
            continue

        diff_amount = int((target_value - current_value) / current_price / 100) * 100
        if abs(diff_amount) > 0:
            order(etf_code, diff_amount, pindex=pindex)
            log.info(f"[{strategy['name']}] 调仓 {etf_name}: {diff_amount}股")
