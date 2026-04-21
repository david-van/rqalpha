#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc    : PT策略 → RQAlpha 转换版
#             ETF核心资产动量轮动策略

import math
import numpy as np
import pandas as pd
from rqalpha.api import *
from rqalpha.apis import *
# from rqalpha.mod.rqalpha_mod_sys_scheduler import scheduler
from rqalpha.mod.rqalpha_mod_sys_scheduler.scheduler import physical_time


# ==================== 初始化 ====================
def init(context):
    """
    RQAlpha入口函数: init (PT中为 initialize)
    benchmark / commission / slippage 等均在运行时 config 中设置
    """

    # ---------- 参数 ----------
    context.etf_pool = [
        '518880.XSHG',  # 黄金ETF（大宗商品）        原 518880.SS
        '513100.XSHG',  # 纳指100（海外资产）         原 513100.SS
        '159915.XSHE',  # 创业板100（成长/科技/中小盘） 原 159915.SZ
        '510180.XSHG',  # 沪深300（价值/蓝筹/中大盘）  原 510300.SS
    ]
    context.m_days = 25           # 动量参考天数
    context.initial_capital = 10000  # 策略初始资金（元）
    context.strategy_value = None    # 策略总价值，首次运行或重启时自动初始化
    context.target_list = []         # 当日目标ETF列表

    # ---------- 定时任务 ----------
    # 9:35 → market_open(minutes=5)   先卖
    # 9:37 → market_open(minutes=7)   后买
    scheduler.run_daily(sell_trade, time_rule=physical_time(hour=9, minute=35))
    scheduler.run_daily(buy_trade,  time_rule=physical_time(hour=9, minute=37))

    logger.info('策略初始化完成')


# ==================== handle_bar（空实现） ====================
def handle_bar(context, bar_dict):
    pass


# ==================== 动量打分 ====================
def get_rank(context):
    """
    基于年化收益 × 判定系数(R²) 的动量打分排名

    PT  : get_history(count, frequency, field, security_list, fq) → DataFrame
    RQA : history_bars(order_book_id, bar_count, frequency, fields) → numpy array
    """
    score_list = []

    for etf in context.etf_pool:
        # ---- 获取历史收盘价 ----
        # history_bars 返回 numpy 1-D array（单字段时）
        # adjust_type='pre' 等价于 PT 的 fq='pre'（前复权）
        try:
            close_arr = history_bars(
                etf,
                context.m_days,
                '1d',
                'close',
                adjust_type='pre'
            )
        except Exception as e:
            logger.warn(f'获取 {etf} 历史数据异常: {e}')
            score_list.append(float('-inf'))
            continue

        if close_arr is None or len(close_arr) == 0:
            score_list.append(float('-inf'))
            continue

        # ---- 线性回归打分 ----
        y = np.log(close_arr)
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        annualized_returns = math.pow(math.exp(slope), 250) - 1
        r_squared = 1 - (
                np.sum((y - (slope * x + intercept)) ** 2)
                / ((len(y) - 1) * np.var(y, ddof=1))
        )
        score = annualized_returns * r_squared
        score_list.append(score)

    # ---- 排序 ----
    df = pd.DataFrame(index=context.etf_pool, data={'score': score_list})
    df = df.sort_values(by='score', ascending=False)
    rank_list = list(df.index)
    logger.info('\n' + str(df))
    return rank_list


# ==================== 卖出逻辑 (9:35) ====================
def sell_trade(context, bar_dict):
    """
    计算动量排名 → 卖出非目标ETF
    注意：RQAlpha scheduler 回调签名为 (context, bar_dict)
    """
    target_num = 1
    context.target_list = get_rank(context)[:target_num]

    # ---- 更新策略总价值 ----
    managed_value = 0.0
    for etf in context.etf_pool:
        pos = context.portfolio.positions.get(etf, None)
        if pos is not None and pos.quantity > 0:
            managed_value += pos.market_value

    if context.strategy_value is None:
        context.strategy_value = managed_value if managed_value > 0 else context.initial_capital
        logger.info(f'策略价值初始化: {context.strategy_value:.2f}')
    elif managed_value > 0:
        context.strategy_value = managed_value

    # ---- 卖出不在目标中的持仓 ----
    hold_list = [
        etf for etf in context.portfolio.positions
        if context.portfolio.positions[etf].quantity > 0
    ]
    for etf in hold_list:
        if etf not in context.target_list:
            order_obj = order_target_value(etf, 0)
            if order_obj is not None:
                logger.info(f'卖出 {etf}')
            else:
                logger.warn(f'卖出失败 {etf}')
        else:
            logger.info(f'继续持有 {etf}')


# ==================== 买入逻辑 (9:37) ====================
def buy_trade(context, bar_dict):
    """
    卖出成交后 → 买入目标ETF
    注意：RQAlpha scheduler 回调签名为 (context, bar_dict)
    """
    target_num = 1

    hold_list = [
        etf for etf in context.portfolio.positions
        if context.portfolio.positions[etf].quantity > 0
    ]

    if len(hold_list) < target_num:
        available = min(context.portfolio.cash, context.strategy_value)
        value = available / (target_num - len(hold_list))
        for etf in context.target_list:
            if etf not in hold_list:
                order_obj = order_target_value(etf, value)
                if order_obj is not None:
                    logger.info(f'买入 {etf}，目标金额: {value:.2f}')
                else:
                    logger.warn(f'买入失败 {etf}，可用资金: {context.portfolio.cash:.2f}')
