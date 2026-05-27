"""调试可转债下单流程"""
import traceback
from rqalpha.api import *
from rqalpha.environment import Environment

def init(context):
    context.s1 = '128119.SZ'

def handle_bar(context, bar_dict):
    ob = context.s1
    # 验证数据
    bar = bar_dict[ob]
    logger.info(f'{ob} close={bar.close} open={bar.open} volume={bar.volume}')

    # 逐步检查下单链路
    env = Environment.get_instance()

    # 1. instrument
    ins = env.data_proxy.instrument(ob)
    logger.info(f'instrument: {ins.order_book_id} type={ins.type} market={ins.market}')

    # 2. get_bar via env
    try:
        b = env.get_bar(ob)
        logger.info(f'env.get_bar: close={b.close}')
    except Exception as e:
        logger.error(f'env.get_bar FAILED: {traceback.format_exc()}')

    # 3. 尝试下单并打印完整异常
    try:
        from rqalpha.const import SIDE, POSITION_EFFECT
        account = env.portfolio.accounts['STOCK']
        position = account.get_position(ob, 'LONG')
        logger.info(f'account={account}, position={position}, market_value={position.market_value}')

        o = order_shares(ob, 10)
        logger.info(f'order result: {o}')
    except Exception as e:
        logger.error(f'FULL TRACEBACK:\n{traceback.format_exc()}')
        raise


__config__ = {
    "base": {
        "start_date": "2025-01-01",
        "end_date": "2025-01-05",
        "frequency": "1d",
        "data_bundle_path": r"D:\datas\bundle",
        "accounts": {"stock": 100000},
    },
    "mod": {
        "cb": {"enabled": True, "lib": "my_mod.cb_bundle.rqalpha_mod_cb"},
        "sys_transaction_cost": {"enabled": True, "stock_commission_multiplier": 0.03, "stock_min_commission": 0, "tax_multiplier": 0},
        "sys_simulation": {"enabled": True, "matching_type": "current_bar"},
        "sys_analyser": {"enabled": True, "plot": False},
    },
    "extra": {"log_level": "info"},
}
