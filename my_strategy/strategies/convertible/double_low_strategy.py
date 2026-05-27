"""
可转债双低策略

双低值 = 转债价格 + 转股溢价率
每月调仓，选双低值最小的 N 只，等权持有。

参数（通过 config.extra.context_vars.strategy_params 注入）:
    top_n:       持仓数量（默认 20）
    max_price:   价格上限（默认 130）
    min_remain:  最低剩余规模，亿元（默认 1.0）
"""
import os
import pickle

import numpy as np
import pandas as pd
from rqalpha.api import *


def _load_remain_sizes():
    pkl_path = os.path.join(r'D:\datas\bundle', 'cb_instruments.pk')
    if not os.path.exists(pkl_path):
        return {}
    with open(pkl_path, 'rb') as f:
        insts = pickle.load(f)
    return {i['order_book_id']: float(i.get('remain_size', 0) or 0) for i in insts}


def init(context):
    params = getattr(context, 'strategy_params', {})
    context.top_n = params.get('top_n', 20)
    context.max_price = params.get('max_price', 130)
    context.min_remain = params.get('min_remain', 1.0)
    context.last_rebalance_month = None
    context.remain_sizes = _load_remain_sizes()

    instruments = all_instruments('Convertible')
    logger.info(f'可转债数量: {len(instruments)}, top_n={context.top_n}, '
                f'max_price={context.max_price}, min_remain={context.min_remain}')


def handle_bar(context, bar_dict):
    current_month = context.now.month
    if current_month == context.last_rebalance_month:
        return
    context.last_rebalance_month = current_month

    instruments = all_instruments('Convertible')
    scores = []

    for _, ins in instruments.iterrows():
        ob = ins['order_book_id']

        # 跳过已退市的
        delist = ins.get('de_listed_date')
        if delist is not None and str(delist) != '0000-00-00':
            try:
                if pd.Timestamp(delist) < context.now:
                    continue
            except Exception:
                pass

        remain = context.remain_sizes.get(ob, 0)
        if remain < context.min_remain:
            continue

        try:
            bar = history_bars(ob, 1, '1d', ['close', 'cb_over_rate'])
        except Exception:
            continue

        if bar is None or len(bar) == 0:
            continue

        price = bar['close'][0]
        premium = bar['cb_over_rate'][0]

        if np.isnan(price) or np.isnan(premium) or price <= 0:
            continue
        if price > context.max_price:
            continue

        dual_low = price + premium
        scores.append((ob, dual_low, price, premium))

    if len(scores) == 0:
        logger.warning('无双低标的可选')
        return

    scores.sort(key=lambda x: x[1])
    selected = scores[:context.top_n]

    logger.info(f'=== {context.now.strftime("%Y-%m-%d")} 调仓, 选出 {len(selected)} 只 ===')
    for ob, dl, p, prem in selected[:5]:
        logger.info(f'  {ob}: price={p:.1f} premium={prem:.1f}% dual_low={dl:.1f}')

    top_set = {ob for ob, *_ in selected}
    for pos in list(context.portfolio.positions.values()):
        if pos.order_book_id not in top_set and pos.market_value > 0:
            order_target_percent(pos.order_book_id, 0)

    weight = 1.0 / len(selected)
    for ob, *_ in selected:
        order_target_percent(ob, weight)


def after_trading(context):
    pass
