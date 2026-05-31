"""
可转债双低策略

双低值 = 转债价格 + 转股溢价率
每月调仓，选双低值最小的 N 只，等权持有。

参数（通过 config.extra.context_vars.strategy_params 注入）:
    top_n:     持仓数量（默认 20）
    max_price: 价格上限（默认 130）
"""
import os
import pickle
import pandas as pd
from rqalpha.api import *
from rqalpha.utils import is_valid_price


def init(context):
    params = getattr(context, 'strategy_params', {})
    context.top_n = params.get('top_n', 20)
    context.max_price = params.get('max_price', 130)
    context.last_rebalance_month = None

    # 已知数据不全的债券，回测时跳过
    context.skip_bonds = {
        '124017.SZ',  # 数据起始晚于上市63天（2022-08-25上市, 数据从2022-10-27开始）
        '124024.SZ',  # 数据起始晚于上市21天 + 中间缺465天
        '123029.SZ',  # 数据提前448天终止（2024-05-27断档, 2025-08-18才退市）
        '123099.SZ',  # 数据中间缺42天（2024-05-27 ~ 2024-07-08）
        '128044.SZ',  # maturity_date=2024-08-14 但 de_listed_date 为空, 数据含大量僵尸记录
        '113575.SH',  # maturity_date=2026-04-09 但 de_listed_date 为空
    }

    with open(r'D:\datas\bundle\cb_instruments.pk', 'rb') as f:
        cb_list = pickle.load(f)

    # 加载赎回公告数据（文件不存在时回退为空 dict，兼容旧数据）
    _redemption_path = r'D:\datas\bundle\cb_redemption.pk'
    _redemption = {}
    if os.path.exists(_redemption_path):
        with open(_redemption_path, 'rb') as f:
            _redemption = pickle.load(f)
        logger.info(f'赎回数据: {len(_redemption)} 条')

    context.cb_pool = []
    context.redemption_info = {}  # order_book_id -> (ann_date, call_reg_date)
    context.bond_dates = {}       # order_book_id -> (de_listed_date, maturity_date)
    for i in cb_list:
        code = i['order_book_id']
        r = _redemption.get(code, {})
        ann = r.get('ann_date', '0000-00-00')
        reg = r.get('call_reg_date', '0000-00-00')
        dl = i.get('de_listed_date', '0000-00-00')
        mat = i.get('maturity_date', '0000-00-00')
        context.cb_pool.append((
            code,
            i.get('listed_date', '0000-00-00'),
            dl,
            mat,
            ann,
            reg,
        ))
        context.redemption_info[code] = (ann, reg)
        context.bond_dates[code] = (dl, mat)

    logger.info(f'可转债池: {len(context.cb_pool)} 只, 跳过数据不全: {len(context.skip_bonds)} 只')


def handle_bar(context, bar_dict):
    # 每日检查：持仓债券如已发赎回公告，立即卖出
    for pos in list(context.portfolio.positions.values()):
        if pos.market_value <= 0:
            continue
        ann_date, _ = context.redemption_info.get(pos.order_book_id, ('0000-00-00', '0000-00-00'))
        if ann_date != '0000-00-00' and pd.Timestamp(ann_date) <= context.now:
            logger.info(f'赎回公告卖出: {pos.order_book_id} 公告日 {ann_date}')
            order_target_percent(pos.order_book_id, 0)
            continue
        # 自然到期检查：de_listed_date 为空时，用 maturity_date 判断
        dl_date, mat_date = context.bond_dates.get(pos.order_book_id, ('0000-00-00', '0000-00-00'))
        if dl_date == '0000-00-00' and mat_date not in ('0000-00-00', '2999-12-31'):
            if pd.Timestamp(mat_date) < context.now:
                logger.info(f'到期卖出: {pos.order_book_id} 到期日 {mat_date}')
                order_target_percent(pos.order_book_id, 0)

    current_month = context.now.month
    if current_month == context.last_rebalance_month:
        return
    context.last_rebalance_month = current_month

    scores = []
    too_high = 0

    for ob, listed_date, de_listed_date, maturity_date, ann_date, call_reg_date in context.cb_pool:
        if listed_date != '0000-00-00' and pd.Timestamp(listed_date) > context.now:
            continue
        if de_listed_date != '0000-00-00' and pd.Timestamp(de_listed_date) < context.now:
            continue
        # de_listed_date 为空时，用 maturity_date 兜底判断是否已到期
        if de_listed_date == '0000-00-00' and maturity_date not in ('0000-00-00', '2999-12-31'):
            if pd.Timestamp(maturity_date) < context.now:
                continue
        # 赎回公告已发 → 已不可交易
        if ann_date != '0000-00-00' and pd.Timestamp(ann_date) <= context.now:
            continue

        bar = history_bars(ob, 1, '1d', ['close', 'cb_over_rate'])
        if len(bar) == 0:
            if ob in context.skip_bonds:
                continue

        price = bar['close'][0]
        premium = bar['cb_over_rate'][0]

        if not is_valid_price(price):
            continue
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
