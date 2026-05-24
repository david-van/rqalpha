#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
策略: pool_oldest
一句话: 永远持有池中待最久的那只股票。

规则 (就两条，无参数):
  1. 持仓被博主清出池 → 卖, 换到当前池中待最久的那只
  2. 其他情况 → 不动
"""

import bisect
from pathlib import Path
import pandas as pd

from my_strategy.common_file import project_root
from rqalpha.api import (
    order_target_value, logger, scheduler,
)
from rqalpha.mod.rqalpha_mod_sys_scheduler.scheduler import physical_time


class PoolTracker:
    def __init__(self, csv_dir):
        self._first_seen = {}
        self._snapshots = []
        for f in sorted(Path(csv_dir).glob("*holdings*.csv")):
            df = pd.read_csv(f, encoding="utf-8-sig")
            for _, row in df.iterrows():
                codes_str = str(row.get("持有股票代码", ""))
                if not codes_str or codes_str == "nan":
                    continue
                codes = [c.strip() for c in codes_str.split("|") if c.strip()]
                if codes:
                    d = pd.Timestamp(str(row["日期"]).strip())
                    self._snapshots.append((d, codes))
                    for c in codes:
                        if c not in self._first_seen:
                            self._first_seen[c] = d
        self._snapshots.sort(key=lambda x: x[0])
        self._dates = [s[0] for s in self._snapshots]

    def get_pool(self, dt):
        idx = bisect.bisect_right(self._dates, dt) - 1
        return self._snapshots[idx][1] if idx >= 0 else []

    def oldest(self, pool):
        if not pool:
            return None
        return min(pool, key=lambda c: self._first_seen.get(c, pd.Timestamp("2099-01-01")))


def init(context):
    csv_dir = str(Path(project_root) / 'my_strategy' / 'strategies' / 'xiaoe_articles')
    injected = getattr(context, 'strategy_params', None)
    if injected and 'pool_csv_dir' in injected:
        csv_dir = injected['pool_csv_dir']

    context.tracker = PoolTracker(csv_dir)
    context.target = None
    scheduler.run_daily(sell_trade, time_rule=physical_time(hour=9, minute=35))
    scheduler.run_daily(buy_trade, time_rule=physical_time(hour=9, minute=37))
    logger.info(f"pool_oldest 就绪")


def sell_trade(context, bar_dict):
    pool = context.tracker.get_pool(context.now)
    context.target = context.tracker.oldest(pool)

    holdings = [c for c in context.portfolio.positions
                if context.portfolio.positions[c].quantity > 0]
    for code in holdings:
        if code != context.target:
            order_target_value(code, 0)
            logger.info(f"卖出 {code}")


def buy_trade(context, bar_dict):
    if context.target is None:
        return
    holdings = [c for c in context.portfolio.positions
                if context.portfolio.positions[c].quantity > 0]
    if context.target in holdings:
        return
    order_target_value(context.target, context.portfolio.cash)
    logger.info(f"买入 {context.target}")


def handle_bar(context, bar_dict):
    pass
