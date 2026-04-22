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
    context.etf_pool = [
        '518880.XSHG',  # 黄金ETF（大宗商品）        原 518880.SS
        '513100.XSHG',  # 纳指100（海外资产）         原 513100.SS
        '159915.XSHE',  # 创业板100（成长/科技/中小盘） 原 159915.SZ
        '510180.XSHG',  # 沪深300（价值/蓝筹/中大盘）  原 510300.SS
    ]
    context.m_days = 25
    context.target_list = []
    scheduler.run_daily(sell_trade, time_rule=physical_time(hour=9, minute=35))
    scheduler.run_daily(buy_trade, time_rule=physical_time(hour=9, minute=37))
    logger.info('策略初始化完成')


# ==================== handle_bar（空实现） ====================
def handle_bar(context, bar_dict):
    pass


# ==================== 动量打分 ====================
def get_rank(context):
    """年化收益 × R² 动量打分"""
    score_list = []

    for etf in context.etf_pool:
        # ---- 获取历史收盘价 ----
        # history_bars 返回 numpy 1-D array（单字段时）
        # adjust_type='pre' 等价于 PT 的 fq='pre'（前复权）
        try:
            close_arr = history_bars(etf, context.m_days, '1d', 'close', adjust_type='pre')
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
    """9:35 计算排名，卖出非目标"""
    target_num = 1
    context.target_list = get_rank(context)[:target_num]
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


def buy_trade(context, bar_dict):
    """9:37 买入目标ETF —— 直接用账户总价值，天然复利"""
    target_num = 1
    hold_list = [
        etf for etf in context.portfolio.positions
        if context.portfolio.positions[etf].quantity > 0
    ]

    if len(hold_list) < target_num:
        # ============================================
        # 核心改动：直接用全部可用现金，利润自动滚入
        # ============================================
        value = context.portfolio.cash / (target_num - len(hold_list))
        for etf in context.target_list:
            if etf not in hold_list:
                order_obj = order_target_value(etf, value)
                if order_obj is not None:
                    logger.info(f'买入 {etf}，目标金额: {value:.2f}')
                else:
                    logger.warn(f'买入失败 {etf}，可用资金: {context.portfolio.cash:.2f}')
