# -*- coding: utf-8 -*-
"""
S3 QDII ETF/LOF 折溢价套利策略
-------------------------------
筛选含海外关键词的QDII基金，计算场内价格相对净值的折溢价率。
折价买入，溢价卖出，结合持仓回撤止损和超时退出。

资金: 20000元 (pindex=2)
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


# ==================== 证券名称缓存 ====================
_security_name_cache = {}


def get_security_name(security):
    """获取证券显示名称"""
    if security in _security_name_cache:
        return _security_name_cache[security]
    try:
        info = get_security_info(security)
        name = info.display_name if hasattr(info, 'display_name') else info.name
        _security_name_cache[security] = name
        return name
    except Exception:
        return str(security)


# ==================== 策略配置 ====================
def s3_init_config(context):
    g.qdii_strategy = {
        'name': 'QDII套利', 'pindex': 2, 'ratio': pd.Series(),
        'daily_loss_limit': 0.02,           # 单日亏损阈值
        'hold_time_limit': 10,              # 最长持仓天数
        'partial_stop_threshold': 0.10,     # 持仓峰值回撤10%减半仓
        'full_stop_threshold': 0.20,        # 持仓峰值回撤20%清仓
        'min_daily_volume': 10000,          # 最低日均成交量(手)
        'min_daily_money': 2e6,             # 最低日均成交额(元)
        'cool_down_days': 3,                # 卖出后冷却天数
        'previous_day_value': None,          # 上一日组合净值
        'last_update_date': None,            # 上次更新日期
        'position_peak_value': {},           # 持仓峰值市值
        'position_entry_date': {},           # 持仓建仓日期
        'last_sell_date': {},               # 上次卖出日期
        'rebalance_cap': None,               # 再平衡截流上限
    }


# ==================== 定时任务注册 ====================
def s3_setup_schedule():
    run_daily(qdii_iUpdate, '9:35')
    run_daily(qdii_iTrader, '9:36')
    run_daily(qdii_iUpdate, '10:31')
    run_daily(qdii_iTrader, '10:32')
    run_daily(qdii_iReport, 'after_close')


# ==================== 风险检查 ====================
def qdii_check_daily_loss(context, strategy):
    """检查当日是否触发单日亏损限制"""
    current_date = context.current_dt.date()
    current_value = context.subportfolios[strategy['pindex']].total_value

    if strategy['previous_day_value'] is None or strategy['last_update_date'] != current_date:
        strategy['previous_day_value'] = current_value
        strategy['last_update_date'] = current_date
        return

    daily_return = (current_value - strategy['previous_day_value']) / strategy['previous_day_value']
    if daily_return < -strategy['daily_loss_limit']:
        log.warn(f"[{strategy['name']}] 触发单日亏损限制: {daily_return:.2%}")
        strategy['previous_day_value'] = current_value
        strategy['last_update_date'] = current_date


def qdii_check_pre_trade_risk(context, strategy):
    """交易前检查：昨是否触发单日亏损"""
    if strategy['previous_day_value'] is None:
        return True
    current_value = context.subportfolios[strategy['pindex']].total_value
    daily_return = (current_value - strategy['previous_day_value']) / strategy['previous_day_value']
    if daily_return < -strategy['daily_loss_limit']:
        log.info(f"[{strategy['name']}] 昨日亏损 {daily_return:.2%}，暂停新买入")
        return False
    return True


def qdii_update_position_peaks(context, strategy):
    """更新各持仓的峰值市值"""
    pindex = strategy['pindex']
    for stock in context.subportfolios[pindex].positions:
        current_value = context.subportfolios[pindex].positions[stock].value
        if stock not in strategy['position_peak_value']:
            strategy['position_peak_value'][stock] = current_value
        elif current_value > strategy['position_peak_value'][stock]:
            strategy['position_peak_value'][stock] = current_value


def qdii_check_partial_stop_loss(context, strategy):
    """从持仓峰值回撤止损：>20%全清，>10%减半"""
    pindex = strategy['pindex']
    cdata = get_current_data()

    for stock in list(context.subportfolios[pindex].positions.keys()):
        if stock not in strategy['position_peak_value']:
            continue
        current_value = context.subportfolios[pindex].positions[stock].value
        peak_value = strategy['position_peak_value'][stock]

        if peak_value > 0:
            drawdown = (peak_value - current_value) / peak_value

            if drawdown > strategy['full_stop_threshold']:
                log.warn(f"[{strategy['name']}] 触发全部止损: {stock}, 回撤: {drawdown:.2%}")
                order_target(stock, 0, LimitOrderStyle(0.99 * cdata[stock].last_price), pindex=pindex)
                if stock in strategy['position_peak_value']:
                    del strategy['position_peak_value'][stock]

            elif drawdown > strategy['partial_stop_threshold']:
                current_amount = context.subportfolios[pindex].positions[stock].total_amount
                target_amount = int(current_amount * 0.5)
                if target_amount > 0:
                    log.info(f"[{strategy['name']}] 触发部分止损: {stock}, 回撤: {drawdown:.2%}")
                    order_target(stock, target_amount, LimitOrderStyle(0.99 * cdata[stock].last_price), pindex=pindex)


# ==================== 流动性过滤 ====================
def qdii_check_fund_liquidity(funds, context, strategy):
    """过滤日均成交量和成交额不足的基金"""
    if not funds:
        return []
    try:
        volume_data = history(20, '1d', 'volume', funds, df=False)
        money_data = history(20, '1d', 'money', funds, df=False)
        liquid_funds = []
        for fund in funds:
            vol_arr = volume_data.get(fund)
            mon_arr = money_data.get(fund)
            if vol_arr is None or mon_arr is None:
                continue
            avg_vol = np.mean(vol_arr)
            avg_mon = np.mean(mon_arr)
            if avg_vol >= strategy['min_daily_volume'] and avg_mon >= strategy['min_daily_money']:
                liquid_funds.append(fund)
        return liquid_funds
    except:
        return funds


def qdii_record_position_entry(stock, current_date, strategy):
    strategy['position_entry_date'][stock] = current_date


def qdii_should_exit_position(context, stock, cdata, strategy):
    """判断是否应该退出持仓：不在折溢价列表 或 溢价超5%"""
    if stock not in strategy['ratio']:
        return True
    if strategy['ratio'][stock] >= 0.05:
        return True
    return False


# ==================== 折溢价率更新 ====================
def qdii_iUpdate(context):
    """更新QDII基金的折溢价率"""
    strategy = g.qdii_strategy
    keys = ['标普', '美国', '纳指', '纳斯达克', '法国', '德国', '日经', '亚太', '东南亚', '印度', '沙特']
    black_list = ['562060.XSHG']
    dt_last = context.previous_date

    # 筛选海外QDII基金
    all_fund = get_all_securities(['fund'], dt_last)
    try:
        pattern = '|'.join(keys)
        mask = all_fund.display_name.str.contains(pattern, na=False)
        funds = all_fund[mask].index.tolist()
        funds = [s for s in funds if s not in black_list]
    except:
        funds = []

    # 流动性过滤
    funds = qdii_check_fund_liquidity(funds, context, strategy)

    if funds:
        # 计算折溢价率 = 场内价 / 净值 - 1
        value = get_extras('unit_net_value', funds, end_date=dt_last, count=1).iloc[0]
        price = history(1, '1d', 'close', funds, df=False)
        price_dict = price
        r = pd.Series({k: price_dict[k][0] for k in price_dict if k in value.index}) / value - 1.0
        r = r[r < 0.15].sort_values()
        strategy['ratio'] = r

        log.info(f"[{strategy['name']}] 筛选出 {len(strategy['ratio'])} 只符合条件的基金")
        if not r.empty:
            log.info(f"[{strategy['name']}] === 折溢价率明细 ===")
            for code, prem_rate in r.items():
                fund_name = get_security_name(code)
                log.info(f"[{strategy['name']}]   {code} {fund_name}: 折溢价率 {prem_rate:.2%}")
    else:
        strategy['ratio'] = pd.Series()
        log.info(f"[{strategy['name']}] 无满足条件的基金")


def _get_bid1_sell_price(security, fallback_price):
    """获取买一价作为卖出限价"""
    sell_price = None
    try:
        tick = get_current_tick(security)
        if tick is not None and tick.get('b1_p', 0) > 0:
            sell_price = round(tick['b1_p'], 3)
    except Exception:
        pass
    if sell_price is None:
        sell_price = round(fallback_price * 0.99, 3)
    return sell_price


# ==================== 交易执行 ====================
def qdii_iTrader(context):
    """QDII折溢价套利交易：折价买/溢价卖/回撤止损"""
    strategy = g.qdii_strategy
    pindex = strategy['pindex']

    # 更新峰值 + 止损检查
    qdii_update_position_peaks(context, strategy)
    qdii_check_partial_stop_loss(context, strategy)

    if strategy['ratio'].empty:
        log.error(f"[{strategy['name']}] 无折溢价数据，跳过交易")
        return

    r = strategy['ratio']
    base_position_size = context.subportfolios[pindex].total_value / max(3, len(r))

    # 再平衡截流
    if strategy.get('rebalance_cap') is not None:
        cap_position_size = strategy['rebalance_cap'] / max(3, len(r))
        if base_position_size > cap_position_size:
            _slog('S3', '再平衡', f'截流: {base_position_size:.0f}→{cap_position_size:.0f}元')
            base_position_size = cap_position_size

    cdata = get_current_data()
    max_last_price = max((cdata[s].last_price for s in r.index), default=0)
    min_position = 100 * max_last_price if max_last_price > 0 else 600
    position_size = max(base_position_size, min_position)
    lm_value = 0.8 * position_size   # 仓位下限(<80%补仓)
    hm_value = 1.3 * position_size   # 仓位上限(>130%减仓)
    cash_size = 0.1 * context.subportfolios[pindex].total_value
    current_date = context.current_dt.date()

    # 风控：昨日亏损暂停新买
    can_buy_new = qdii_check_pre_trade_risk(context, strategy)
    if not can_buy_new:
        log.info(f"[{strategy['name']}] 风控限制：暂停新买入")

    # ===== 卖出 =====
    log.info(f"[{strategy['name']}] === 卖出检查 | 当前持仓: {len(context.subportfolios[pindex].positions)} 只 ===")

    for stock in list(context.subportfolios[pindex].positions.keys()):
        if cdata[stock].paused:
            continue

        # 持仓超时检查
        if stock in strategy['position_entry_date']:
            hold_days = (current_date - strategy['position_entry_date'][stock]).days
            if hold_days > strategy['hold_time_limit']:
                if qdii_should_exit_position(context, stock, cdata, strategy):
                    log.info(f"[{strategy['name']}] 持仓超时卖出: {stock}, 持有{hold_days}天")
                    _sell_price = _get_bid1_sell_price(stock, cdata[stock].last_price)
                    _down_limit = getattr(cdata[stock], 'low_limit', None)
                    _high_limit = getattr(cdata[stock], 'high_limit', None)
                    if _down_limit and _sell_price < _down_limit:
                        _sell_price = _down_limit + 0.001
                    if _high_limit and _sell_price > _high_limit:
                        _sell_price = _high_limit - 0.001
                    order_target(stock, 0, LimitOrderStyle(_sell_price), pindex=pindex)
                    for key in ['position_entry_date', 'position_peak_value']:
                        if stock in strategy[key]:
                            del strategy[key][stock]
                continue

        # 不在目标列表
        if stock not in r.index:
            log.info(f"[{strategy['name']}] 不在目标列表，卖出: {stock}")
            _sell_price = _get_bid1_sell_price(stock, cdata[stock].last_price)
            _down_limit = getattr(cdata[stock], 'low_limit', None)
            _high_limit = getattr(cdata[stock], 'high_limit', None)
            if _down_limit and _sell_price < _down_limit:
                _sell_price = _down_limit + 0.001
            if _high_limit and _sell_price > _high_limit:
                _sell_price = _high_limit - 0.001
            order_target(stock, 0, LimitOrderStyle(_sell_price), pindex=pindex)
            for key in ['position_entry_date', 'position_peak_value']:
                if stock in strategy[key]:
                    del strategy[key][stock]

    if not can_buy_new:
        return

    # ===== 买入 =====
    log.info(f"[{strategy['name']}] === 买入检查 | 可用现金: {context.subportfolios[pindex].available_cash:.2f} ===")

    for stock in r.index:
        if context.subportfolios[pindex].available_cash < cash_size:
            break
        if cdata[stock].paused:
            continue

        if stock not in context.subportfolios[pindex].positions:
            # 新买入：折价才买
            if r[stock] < 0:
                log.info(f"[{strategy['name']}] 新买入: {stock}, 折价{r[stock]:.2%}")
                _buy_price = _get_safe_price(cdata, stock, 1.01)
                order_result = order_target_value(stock, position_size, LimitOrderStyle(_buy_price), pindex=pindex)
                if order_result is not None and order_result.filled > 0:
                    qdii_record_position_entry(stock, current_date, strategy)

        elif context.subportfolios[pindex].positions[stock].value < lm_value:
            # 仓位不足80%：折价补仓
            if r[stock] < 0:
                log.info(f"[{strategy['name']}] 补仓: {stock}")
                _buy_price = _get_safe_price(cdata, stock, 1.01)
                order_target_value(stock, position_size, LimitOrderStyle(_buy_price), pindex=pindex)

        elif context.subportfolios[pindex].positions[stock].value > hm_value:
            # 仓位超130%：溢价减仓
            if r[stock] > 0:
                log.info(f"[{strategy['name']}] 减仓: {stock}, 溢价{r[stock]:.2%}")
                _sell_price = _get_safe_price(cdata, stock, 0.99)
                order_target_value(stock, position_size, LimitOrderStyle(_sell_price), pindex=pindex)


def _get_safe_price(cdata, stock, multiplier):
    """计算安全限价（避开涨跌停边界）"""
    price = round(multiplier * cdata[stock].last_price, 3)
    _down_limit = getattr(cdata[stock], 'low_limit', None)
    _high_limit = getattr(cdata[stock], 'high_limit', None)
    if _down_limit and price < _down_limit:
        price = round(_down_limit + 0.001, 3)
    if _high_limit and price > _high_limit:
        price = round(_high_limit - 0.001, 3)
    return price


# ==================== 收盘报告 ====================
def qdii_iReport(context):
    """收盘后统计持仓和日亏损"""
    strategy = g.qdii_strategy
    pindex = strategy['pindex']
    qdii_check_daily_loss(context, strategy)

    cdata = get_current_data()
    tvalue = context.subportfolios[pindex].total_value
    ptable = pd.DataFrame(columns=['amount', 'value', 'weight', 'name', 'hold_days'])
    current_date = context.current_dt.date()

    for stock in context.subportfolios[pindex].positions:
        ps = context.subportfolios[pindex].positions[stock]
        hold_days = '-'
        if stock in strategy['position_entry_date']:
            hold_days = str((current_date - strategy['position_entry_date'][stock]).days)
        ptable.loc[stock] = [ps.total_amount, int(ps.value),
                             100 * ps.value / tvalue, cdata[stock].name, hold_days]

    ptable = ptable.sort_values(by='weight', ascending=False)
    log.info(f"[{strategy['name']}] 持仓数量: {len(ptable)}")
    log.info(f"[{strategy['name']}] 总资产: {context.subportfolios[pindex].total_value / 10000:.2f}万")
