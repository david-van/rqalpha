"""
可转债双低策略

双低值 = 转债价格 + 转股溢价率
每月调仓，选双低值最小的 N 只，等权持有。

参数（通过 config.extra.context_vars.strategy_params 注入）:
    top_n:     持仓数量（默认 20）
    max_price: 价格上限（默认 130）
"""
import pickle
import pandas as pd
from rqalpha.api import *


def init(context):
    params = getattr(context, 'strategy_params', {})
    context.top_n = params.get('top_n', 20)
    context.max_price = params.get('max_price', 130)
    context.last_rebalance_month = None

    # 加载债券池，保留 listed_date 用于回测时按上市日期过滤
    with open(r'D:\datas\bundle\cb_instruments.pk', 'rb') as f:
        cb_list = pickle.load(f)
    context.cb_pool = []
    for i in cb_list:
        context.cb_pool.append((
            i['order_book_id'],
            i.get('listed_date', '0000-00-00'),
            i.get('de_listed_date', '0000-00-00'),
        ))

    logger.info(f'可转债池: {len(context.cb_pool)} 只')


def handle_bar(context, bar_dict):
    current_month = context.now.month
    if current_month == context.last_rebalance_month:
        return
    context.last_rebalance_month = current_month

    scores = []
    too_high = 0

    for ob, listed_date, delisted_date in context.cb_pool:
        if listed_date != '0000-00-00' and pd.Timestamp(listed_date) > context.now:
            continue
        if delisted_date != '0000-00-00' and pd.Timestamp(delisted_date) < context.now:
            continue

        bar = history_bars(ob, 1, '1d', ['close', 'cb_over_rate'])
        price = bar['close'][0]
        premium = bar['cb_over_rate'][0]

        if price > context.max_price:
            too_high += 1
            continue

        dual_low = price + premium
        scores.append((ob, dual_low, price, premium))

    if len(scores) == 0:
        logger.warning(f'调仓 {context.now.strftime("%Y-%m")}: 无敌低标的')
        return

    scores.sort(key=lambda x: x[1])
    selected = scores[:context.top_n]

    logger.info(
        f'调仓 {context.now.strftime("%Y-%m-%d")}: '
        f'候选 {len(scores)} | 超限价 {too_high} | 选中 {len(selected)}'
    )
    for ob, dl, p, prem in selected[:5]:
        logger.info(f'  {ob}: price={p:.1f} premium={prem:.1f}% dual_low={dl:.1f}')

    # 卖出不在池中的
    top_set = {ob for ob, *_ in selected}
    for pos in list(context.portfolio.positions.values()):
        if pos.order_book_id not in top_set and pos.market_value > 0:
            order_target_percent(pos.order_book_id, 0)

    # 等权买入
    weight = 1.0 / len(selected)
    for ob, *_ in selected:
        order_target_percent(ob, weight)


def after_trading(context):
    pass
