"""
可转债最小测试策略 - 验证 CBDataSource + Mod 端到端工作
"""
from rqalpha.api import *


def init(context):
    context.traded = False
    instruments = all_instruments('Convertible')
    logger.info(f'可转债数量: {len(instruments)}')
    context.s1 = '128119.SZ'


def handle_bar(context, bar_dict):
    if context.traded:
        return

    ob = context.s1
    price = bar_dict[ob].close
    logger.info(f'{ob} price={price}')

    bars = history_bars(ob, 5, '1d', ['close', 'volume', 'total_turnover'])
    logger.info(f'{ob} history: {bars["close"]}')

    # 直接用 order_shares 买10张（1手=10张）
    try:
        order_shares(ob, 10)
        context.traded = True
        logger.info(f'{ob} order_shares(10) 已下单')
    except Exception as e:
        logger.error(f'{ob} order_shares 失败: {e}')

    # 不交易的那个，验证数据
    ob2 = '110092.SH'
    try:
        p2 = bar_dict[ob2].close
        logger.info(f'{ob2} price={p2}')
    except Exception:
        pass


__config__ = {
    "base": {
        "start_date": "2025-01-01",
        "end_date": "2025-04-01",
        "frequency": "1d",
        "data_bundle_path": r"D:\datas\bundle",
        "accounts": {"stock": 100000},
    },
    "mod": {
        "cb": {
            "enabled": True,
            "lib": "my_mod.cb_bundle.rqalpha_mod_cb"
        },
        "sys_transaction_cost": {
            "enabled": True,
            "stock_commission_multiplier": 0.03,
            "stock_min_commission": 0,
            "tax_multiplier": 0,
        },
        "sys_simulation": {
            "enabled": True,
            "matching_type": "next_bar",
        },
        "sys_analyser": {
            "enabled": True,
            "plot": False,
        },
    },
    "extra": {
        "log_level": "info",
    },
}
