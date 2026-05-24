#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2026/5/10
# @Author  : david_van
# @Desc    : RQAlpha版ETF池统计策略（从聚宽策略转换）
#            每半年（1月和7月）统计一次市场ETF数量，并输出CSV文件

import os
import pandas as pd
from rqalpha.api import *
from rqalpha.apis import *


def init(context):
    """
    初始化函数，整个回测期间只运行一次
    """
    scheduler.run_monthly(semi_annual_etf_check, tradingday=1)
    context.last_half_year_flag = None
    context.output_dir = "etf_output"
    os.makedirs(context.output_dir, exist_ok=True)
    logger.info("========== 策略初始化完成 ==========")
    logger.info("策略说明：每半年（1月和7月）统计一次市场ETF数量，并输出CSV文件")


def semi_annual_etf_check(context, bar_dict):
    """
    每月第一个交易日运行，但只在1月和7月真正执行ETF统计
    """
    today = context.now
    current_month = today.month
    current_year = today.year

    # 只在每年的1月和7月执行
    if current_month not in [1, 7]:
        return

    # 防止同月重复执行
    half_year_flag = "{}-H{}".format(current_year, '1' if current_month == 1 else '2')
    if context.last_half_year_flag == half_year_flag:
        return
    context.last_half_year_flag = half_year_flag

    # ==================== 获取ETF列表 ====================
    date_str = today.strftime('%Y-%m-%d')
    etf_df = all_instruments(type='ETF', date=date_str)
    etf_count = len(etf_df)

    # ==================== 整理数据 ====================
    output_df = etf_df.copy()
    # 把 order_book_id 作为 code 列放到最前面
    code_col = output_df.pop('order_book_id')
    output_df.insert(0, 'code', code_col)

    # 添加统计日期列和交易所列
    output_df.insert(0, 'stat_date', date_str)
    output_df['exchange'] = output_df['code'].apply(
        lambda x: 'XSHG' if '.XSHG' in str(x) else 'XSHE'
    )

    # ==================== 按交易所统计 ====================
    sh_count = len(output_df[output_df['exchange'] == 'XSHG'])
    sz_count = len(output_df[output_df['exchange'] == 'XSHE'])

    # ==================== 写入CSV文件 ====================
    out_dir = context.output_dir

    # 1. 写入完整ETF列表（每次覆盖，保存最新一期）
    latest_path = os.path.join(out_dir, "etf_list_latest.csv")
    output_df.to_csv(latest_path, index=False, encoding='utf-8')

    # 2. 按期归档，文件名带日期标记
    archive_path = os.path.join(out_dir, "etf_list_{}.csv".format(date_str))
    output_df.to_csv(archive_path, index=False, encoding='utf-8')

    # 3. 追加写入汇总统计记录
    summary_path = os.path.join(out_dir, "etf_summary.csv")
    summary_exists = os.path.exists(summary_path)
    with open(summary_path, 'a', encoding='utf-8') as f:
        if not summary_exists:
            f.write("stat_date,half_year_flag,total_count,sh_count,sz_count\n")
        f.write("{},{},{},{},{}\n".format(
            date_str, half_year_flag, etf_count, sh_count, sz_count
        ))

    # ==================== 日志输出 ====================
    logger.info("=" * 60)
    logger.info("【半年度ETF统计报告】")
    logger.info("  统计日期：{}".format(date_str))
    logger.info("  半年度标记：{}".format(half_year_flag))
    logger.info("  当前市场ETF总数量：{} 只".format(etf_count))
    logger.info("  上交所ETF数量：{} 只".format(sh_count))
    logger.info("  深交所ETF数量：{} 只".format(sz_count))
    logger.info("  文件已写入：")
    logger.info("    - {} （最新一期完整列表）".format(latest_path))
    logger.info("    - {} （归档列表）".format(archive_path))
    logger.info("    - {} （历次汇总记录）".format(summary_path))
    logger.info("=" * 60)

    # 打印前10只ETF示例
    sample_count = min(10, etf_count)
    logger.info("  前{}只ETF示例：".format(sample_count))
    for i in range(sample_count):
        row = output_df.iloc[i]
        symbol_name = row.get('symbol', 'N/A')
        logger.info("    {}. {} - {}".format(i + 1, row['code'], symbol_name))


def handle_bar(context, bar_dict):
    pass
