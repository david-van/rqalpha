# -*- coding: utf-8 -*-
"""
S5 小市值策略（独立版V3）
--------------------------
从399101.XSHE（中证等权指数）成分股中选取小市值股票，每周二调仓。
含九项排雷审计、ATR止损、成本保护、MACD顶背离检测、涨停卖出。

资金: 50000元 (pindex=4)
"""
from jqdata import *
import numpy as np
import pandas as pd
import datetime
import math


# ==================== 简易日志 ====================
def _slog(sid, module, msg, level='info'):
    prefix = f"[{sid}] {module}"
    text = f"{prefix} | {msg}"
    getattr(log, level)(text)


# ==================== 工具函数 ====================
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


# ==================== 策略配置 ====================
def s5_init_config(context):
    g.small_strategy = {
        'name': 'S5_小市值(独立版V3)', 'pindex': 4,
        'filter_market_cap_min': 10,           # 市值范围(亿): 10-100
        'filter_market_cap_max': 100,
        'max_stock_price': 50,                 # 最高股价(元)
        'min_listed_days': 375,                # 最少上市天数
        'max_position_count': 5,               # 最大持仓数
        'min_buy_amount': 200,                 # 最小买入股数
        'enable_dynamic_stock_num': True,      # 按指数偏离MA10动态调仓数
        'defensive_etf': '511880.XSHG',
        'stoploss_limit': 0.09,               # 固定止损线(-9%)
        'stoploss_market': 0.05,              # 市场暴跌阈值(5%)
        'stoploss_strategy': 3,               # 3=ATR+成本保护+市场
        'yesterday_HL_list': [],              # 昨日涨停列表
        'reason_to_sell': '',                  # 卖出原因
        'not_buy_again': [],                   # 当日不再买入
        'limitup_sold_today': [],               # 今日涨停卖出列表
        'initial_total_value': 0,              # 初始市值
        'position_size': 0,                    # 仓位大小
        # ATR止损
        'enable_atr_stop_loss': True,
        'atr_period': 14,
        'atr_multiplier': 2.0,
        'atr_stop_prices': {},
        # 成本保护止损
        'enable_cost_protection': True,
        'cost_protection_profit_threshold_1': 0.15,   # 盈利15%
        'cost_protection_profit_threshold_2': 0.30,   # 盈利30%
        'cost_protection_stop_line_1': 0.00,           # 第一档止损线(保本)
        'cost_protection_stop_line_2': 0.10,           # 第二档止损线(盈利10%)
        # MACD背离
        'DBL_control': True,
        'dbl': [],
        'check_macd_divergence_days': 10,
        # 审计 & 分红 & 冷却
        'enable_audit_filter': True,
        'enable_bonus_filter': True,
        'audit_tolerance': 2,                  # 审计容差项数
        'enable_cooldown': True,
        'cooldown_days': 2,
        'no_buy_stocks': {},
        # 运行时状态
        'days': 0,
        'first_run': True,
        'stocks': [],
        'rebalance_cap': None,
    }


# ==================== 定时任务 ====================
def s5_setup_schedule():
    run_daily(small_before_trading, 'before_open')
    run_weekly(small_iSell, 2, '09:50')
    run_weekly(small_iBuy, 2, '10:00')
    run_daily(small_daily_stoploss, '10:05')
    run_daily(small_check_macd_divergence_daily, '09:31')
    run_daily(small_check_limit_up, '14:30')
    run_daily(small_iReport, 'after_close')


# ==================== 盘前 ====================
def small_before_trading(context):
    strategy = g.small_strategy
    pindex = strategy['pindex']
    strategy['days'] = strategy.get('days', 0) + 1
    small_update_yesterday_limit_up(context, strategy)


def small_update_yesterday_limit_up(context, strategy):
    pindex = strategy['pindex']
    strategy['yesterday_HL_list'] = []
    positions = context.subportfolios[pindex].positions
    if not positions:
        return
    df = get_price(list(positions.keys()), end_date=context.previous_date,
                   frequency='daily', fields=['close', 'high_limit'],
                   count=1, panel=False, fill_paused=False)
    if not df.empty:
        df = df[df['close'] == df['high_limit']]
        strategy['yesterday_HL_list'] = list(df.code)


# ==================== MACD 背离检测 ====================
def small_check_macd_divergence_daily(context):
    if g.log_trade_detail:
        _slog('S5', 'MACD', '背离检测开始')
    strategy = g.small_strategy
    if strategy['DBL_control']:
        small_check_macd_divergence(context, strategy)


def small_check_macd_divergence(context, strategy, market_index="399101.XSHE", end_days=0):
    if not strategy['dbl'] and "9:31" in str(context.current_dt.time()):
        return

    def detect_divergence():
        fast, slow, sign = 12, 26, 9
        rows = (fast + slow + sign) * 5
        grid = attribute_history(market_index, rows + 10, fields=["close"]).dropna()
        if end_days < 0:
            grid = grid.iloc[:end_days]
        if len(grid) < rows:
            return False
        try:
            grid["dif"], grid["dea"], grid["macd"] = small_macd(grid.close, fast, slow, sign)
            mask = (grid["macd"] < 0) & (grid["macd"].shift(1) >= 0)
            if mask.sum() < 2:
                return False
            key2, key1 = mask[mask].index[-2], mask[mask].index[-1]
            price_cond = grid.close[key2] < grid.close[key1]
            dif_cond = grid.dif[key2] > grid.dif[key1] > 0
            macd_cond = grid.macd.iloc[-2] > 0 > grid.macd.iloc[-1]
            if len(grid["dif"]) > 20:
                trend_cond = grid["dif"].iloc[-10:].mean() < grid["dif"].iloc[-20:-10].mean()
            else:
                trend_cond = False
            return price_cond and dif_cond and macd_cond and trend_cond
        except Exception as e:
            _slog('S5', 'MACD', f'{market_index} 检测错误: {e}', 'error')
            return False

    if detect_divergence():
        strategy['dbl'].append(1)
        _slog('S5', 'MACD', f'检测到{market_index}顶背离信号!', 'warning')
    else:
        strategy['dbl'].append(0)
    if len(strategy['dbl']) > 60:
        strategy['dbl'] = strategy['dbl'][-60:]


def small_macd(close, short=12, long=26, m=9):
    close_s = pd.Series(close)
    dif = close_s.ewm(span=short, min_periods=short - 1, adjust=False).mean() - \
          close_s.ewm(span=long, min_periods=long - 1, adjust=False).mean()
    dea = dif.ewm(span=m, min_periods=m - 1, adjust=False).mean()
    return dif, dea, (dif - dea) * 2


# ==================== 选股流水线 ====================
def small_choice_small(context, strategy, nchoice):
    target_list = small_get_stocks_v3(context, strategy)
    strategy['stocks'] = target_list
    return target_list[:nchoice]


def small_get_stocks_v3(context, strategy):
    """选股：基础过滤→财务筛选→排雷→分红→冷却"""
    initial_list = small_filter_stocks(context, strategy,
                                        get_index_stocks("399101.XSHE"))

    q = (
        query(
            valuation.code, valuation.market_cap,
            income.net_profit, income.operating_revenue,
        )
        .filter(
            valuation.code.in_(initial_list),
            valuation.market_cap.between(
                strategy['filter_market_cap_min'],
                strategy['filter_market_cap_max']),
            income.operating_revenue > 1e8,
            indicator.roe > 0,
            indicator.roa > 0,
            income.net_profit > 2000000,
        )
        .order_by(valuation.market_cap.asc())
        .limit(strategy['max_position_count'] * 5)
    )
    candidate_list = list(get_fundamentals(q, date=context.previous_date).code)

    current_date = context.current_dt.date()
    start_audit_date = datetime.date(2025, 1, 1)

    if current_date > start_audit_date:
        audited_list = small_apply_nine_point_audit(context, strategy, candidate_list)
    else:
        audited_list = small_filter_audit(context, strategy, candidate_list)

    final_list = small_bonus_filter(context, strategy, audited_list)
    final_list = small_filter_cooldown_stocks(context, strategy, final_list)

    if not final_list:
        return [strategy['defensive_etf']]

    last_prices = history(1, unit="1d", field="close", security_list=final_list)
    positions = context.subportfolios[strategy['pindex']].positions
    return [s for s in final_list
            if s in positions or last_prices[s][-1] <= strategy['max_stock_price']][:strategy['max_position_count']]


def small_filter_stocks(context, strategy, stock_list):
    """基础过滤：停牌/ST/退市/板块/涨跌停/次新股/异常高换手"""
    current_data = get_current_data()
    last_prices = history(1, unit="1m", field="close", security_list=stock_list)
    positions = context.subportfolios[strategy['pindex']].positions
    filtered_stocks = []

    for stock in stock_list:
        if current_data[stock].paused:
            continue
        if current_data[stock].is_st:
            continue
        if "退" in current_data[stock].name:
            continue
        # 排除科创板(688/689)、北交所(8xx)、老三板(4xx)
        code = stock.split('.')[0]
        if code.startswith(('688', '689')) or code.startswith('8') or code.startswith('4'):
            continue
        # 涨停不买
        if not (stock in positions or last_prices[stock][-1] < current_data[stock].high_limit):
            continue
        # 跌停不买
        if not (stock in positions or last_prices[stock][-1] > current_data[stock].low_limit):
            continue
        # 上市天数不够
        start_date = get_security_info(stock).start_date
        if hasattr(start_date, 'date'):
            start_date = start_date.date()
        if context.previous_date - start_date < datetime.timedelta(days=strategy['min_listed_days']):
            continue
        filtered_stocks.append(stock)
    return filtered_stocks


# ==================== 九项排雷审计 ====================
def small_apply_nine_point_audit(context, strategy, stock_list):
    if not stock_list:
        return []

    yesterday = context.previous_date
    curr_year = yesterday.year
    curr_month = yesterday.month

    if curr_month <= 4:
        report_year = curr_year - 2
    else:
        report_year = curr_year - 1
    report_date_str = f"{report_year}-12-31"

    q = query(
        valuation.code, indicator.adjusted_profit, income.net_profit,
        cash_flow.subtotal_operate_cash_inflow, cash_flow.subtotal_operate_cash_outflow,
        balance.good_will, balance.equities_parent_company_owners,
        balance.total_liability, balance.total_assets,
        balance.shortterm_loan, balance.cash_equivalents
    ).filter(valuation.code.in_(stock_list))

    fund_df = get_fundamentals(q, date=yesterday)
    if not fund_df.empty:
        fund_df = fund_df.set_index('code')
        fund_df.fillna(0, inplace=True)
    else:
        return stock_list

    final_list = []
    max_tolerance = strategy.get('audit_tolerance', 2)

    for stock in stock_list:
        score = 0
        hit_reasons = []
        try:
            stock_name = get_security_info(stock).display_name if get_security_info(stock) else ""

            # 1. 年报披露时间检查
            if hasattr(finance, 'STK_INCOME_STATEMENT'):
                q_time = query(finance.STK_INCOME_STATEMENT.pub_date).filter(
                    finance.STK_INCOME_STATEMENT.code == stock,
                    finance.STK_INCOME_STATEMENT.end_date == report_date_str,
                    finance.STK_INCOME_STATEMENT.pub_date <= yesterday
                ).limit(1)
                time_df = finance.run_query(q_time)
                if not time_df.empty:
                    actual_date = time_df['pub_date'].iloc[0]
                    if isinstance(actual_date, str):
                        actual_date = datetime.datetime.strptime(actual_date, '%Y-%m-%d').date()
                    elif hasattr(actual_date, 'date'):
                        actual_date = actual_date.date()
                    if actual_date and actual_date > datetime.date(report_year + 1, 4, 20):
                        score += 1
                        hit_reasons.append("年报迟发")

            # 2. 业绩预告
            if hasattr(finance, 'STK_FIN_FORCAST'):
                q_forcast = query(finance.STK_FIN_FORCAST).filter(
                    finance.STK_FIN_FORCAST.code == stock,
                    finance.STK_FIN_FORCAST.end_date == report_date_str,
                    finance.STK_FIN_FORCAST.pub_date <= yesterday
                ).limit(1)
                forcast_df = finance.run_query(q_forcast)
                if not forcast_df.empty:
                    type_id = forcast_df['type_id'].iloc[0]
                    if type_id in [3, 4, 5, 9, 10]:
                        score += 1
                        hit_reasons.append("业绩预告不良")

            # 3. 审计意见（一票否决）
            if hasattr(finance, 'STK_AUDIT_OPINION'):
                q_audit = query(finance.STK_AUDIT_OPINION).filter(
                    finance.STK_AUDIT_OPINION.code == stock,
                    finance.STK_AUDIT_OPINION.end_date == report_date_str,
                    finance.STK_AUDIT_OPINION.pub_date <= yesterday
                ).limit(1)
                audit_df = finance.run_query(q_audit)
                if not audit_df.empty:
                    opinion_id = audit_df['opinion_type_id'].iloc[0]
                    if opinion_id in [3, 4, 5]:
                        _slog('S5', '排雷', f'{stock}({stock_name}) 审计异常, 一票否决', 'warning')
                        continue

            # 4-7. 财务硬指标
            if stock in fund_df.index:
                row = fund_df.loc[stock]
                adj_p = row['adjusted_profit']
                net_p = row['net_profit']
                cash_net = row['subtotal_operate_cash_inflow'] - row['subtotal_operate_cash_outflow']

                if adj_p < 0 or (net_p != 0 and adj_p / net_p < 0.5):
                    score += 1
                    hit_reasons.append("主业存疑")
                if net_p > 0 and cash_net < 0:
                    score += 1
                    hit_reasons.append("现金流异常")

                equity = row['equities_parent_company_owners']
                gw = row['good_will']
                if equity > 0 and (gw / equity) > 0.3:
                    score += 1
                    hit_reasons.append("商誉占比高")

                t_liab = row['total_liability']
                t_assets = row['total_assets']
                st_loan = row['shortterm_loan']
                cash_val = row['cash_equivalents']
                debt_ratio = (t_liab / t_assets) if t_assets > 0 else 0
                if debt_ratio > 0.70 or st_loan > cash_val:
                    score += 1
                    hit_reasons.append(f"资金链紧绷(负债率{debt_ratio*100:.0f}%)")

            # 8. 大股东质押率
            if hasattr(finance, 'STK_SHARES_PLEDGE'):
                q_pledge = query(finance.STK_SHARES_PLEDGE).filter(
                    finance.STK_SHARES_PLEDGE.code == stock,
                    finance.STK_SHARES_PLEDGE.pub_date <= yesterday
                ).order_by(finance.STK_SHARES_PLEDGE.pub_date.desc()).limit(1)
                pledge_df = finance.run_query(q_pledge)
                if not pledge_df.empty:
                    ratio_col = 'pledge_proportion' if 'pledge_proportion' in pledge_df.columns else (
                        'pledge_ratio' if 'pledge_ratio' in pledge_df.columns else None)
                    if ratio_col is not None:
                        val = pledge_df[ratio_col].iloc[0]
                        if pd.notna(val) and val > 80:
                            score += 1
                            hit_reasons.append("大股东高质押")

            # 9. 监管立案调查
            if hasattr(finance, 'STK_INVESTIGATION'):
                q_inv = query(finance.STK_INVESTIGATION).filter(
                    finance.STK_INVESTIGATION.code == stock,
                    finance.STK_INVESTIGATION.pub_date >= f"{curr_year - 1}-01-01",
                    finance.STK_INVESTIGATION.pub_date <= yesterday
                ).limit(1)
                inv_df = finance.run_query(q_inv)
                if not inv_df.empty:
                    score += 1
                    hit_reasons.append("曾遭监管立案调查")

            if score > 0:
                log.info(f"[排雷透视] {stock}({stock_name}) 踩中 {score} 项: {' | '.join(hit_reasons)}")

            if score < max_tolerance:
                final_list.append(stock)
            else:
                _slog('S5', '排雷', f'{stock}({stock_name}) 踩雷{score}项>容忍度{max_tolerance}, 拦截', 'warning')

        except Exception as e:
            _slog('S5', '排雷', f'{stock} 异常: {e}', 'error')
            final_list.append(stock)

    return final_list


# ==================== 审计过滤（简化版，2025年前） ====================
def small_filter_audit(context, strategy, code_list):
    if not code_list:
        return []
    previous_date = context.previous_date
    last_year = (previous_date.replace(year=previous_date.year - 3, month=1, day=1)).strftime("%Y-%m-%d")
    q = query(
        finance.STK_AUDIT_OPINION.code,
        finance.STK_AUDIT_OPINION.pub_date,
        finance.STK_AUDIT_OPINION.opinion_type_id,
    ).filter(
        finance.STK_AUDIT_OPINION.code.in_(code_list),
        finance.STK_AUDIT_OPINION.pub_date >= last_year,
        finance.STK_AUDIT_OPINION.pub_date <= context.previous_date
    )
    df = finance.run_query(q)
    values_to_check = [3, 4, 5, 7]
    if not df.empty and "opinion_type_id" in df.columns:
        bad_stocks = set(df[df["opinion_type_id"].isin(values_to_check)]['code'])
        return [s for s in code_list if s not in bad_stocks]
    return code_list


# ==================== 分红过滤 ====================
def small_bonus_filter(context, strategy, stock_list):
    if not strategy.get('enable_bonus_filter'):
        return stock_list
    year = context.previous_date.year
    end_date = context.previous_date

    if end_date.month == 5:
        start_date = datetime.datetime(year=year, month=1, day=1)
        q = query(
            finance.STK_XR_XD.code,
            finance.STK_XR_XD.bonus_ratio_rmb,
        ).filter(
            finance.STK_XR_XD.board_plan_pub_date > start_date,
            finance.STK_XR_XD.board_plan_pub_date <= end_date,
            finance.STK_XR_XD.implementation_pub_date <= end_date,
            finance.STK_XR_XD.implementation_pub_date != None,
            finance.STK_XR_XD.bonus_ratio_rmb > 0,
            finance.STK_XR_XD.code.in_(stock_list),
        )
        expected_bonus_df = finance.run_query(q)
        bonus_list = expected_bonus_df["code"].unique().tolist() if len(expected_bonus_df) > 0 else []
    else:
        reprot_date = datetime.datetime(year=year - 1, month=12, day=31)
        q = query(
            finance.STK_XR_XD.code,
        ).filter(
            finance.STK_XR_XD.report_date == reprot_date,
            finance.STK_XR_XD.bonus_type == "年度分红",
            finance.STK_XR_XD.implementation_pub_date <= end_date,
            finance.STK_XR_XD.board_plan_bonusnote == "不分配不转增",
            finance.STK_XR_XD.code.in_(stock_list),
        )
        no_year_bonus = finance.run_query(q)
        no_year_bonus_list = no_year_bonus["code"].unique().tolist()
        bonus_list = [code for code in stock_list if code not in no_year_bonus_list]
        bonus_list = small_short_by_market_cap(context, bonus_list)

    if len(bonus_list) < strategy['max_position_count']:
        bonus_list.extend(
            [x for x in small_short_by_market_cap(context, stock_list) if x not in bonus_list]
            [:strategy['max_position_count'] - len(bonus_list)]
        )
    return bonus_list


# ==================== 冷却期过滤 ====================
def small_filter_cooldown_stocks(context, strategy, stock_list):
    if not strategy.get('enable_cooldown'):
        return stock_list
    current_date = context.current_dt.date()
    cooldown_days = strategy.get('cooldown_days', 2)
    valid_stocks = []
    for stock in stock_list:
        if stock in strategy.get('no_buy_stocks', {}):
            stop_date = strategy['no_buy_stocks'][stock]
            trade_days = get_trade_days(start_date=stop_date, end_date=current_date)
            passed_days = len(trade_days) - 1
            if passed_days < cooldown_days:
                _slog('S5', '冷却', f'{stock} 止损于{stop_date}, {passed_days}日, 仍在冷却期')
                continue
            else:
                del strategy['no_buy_stocks'][stock]
        valid_stocks.append(stock)
    return valid_stocks


def small_short_by_market_cap(context, stock_list):
    short_q = (
        query(valuation.code, valuation.market_cap)
        .filter(valuation.code.in_(stock_list), valuation.day == context.previous_date)
        .order_by(valuation.market_cap.asc())
    )
    short_df = get_fundamentals(short_q, date=context.previous_date)
    return short_df["code"].unique().tolist() if not short_df.empty else []


# ==================== ATR 止损 ====================
def small_calculate_atr(security, context, period=14):
    df = get_price(security, end_date=context.previous_date, count=period + 1,
                   frequency="daily", fields=["high", "low", "close"])
    if len(df) < period + 1:
        return None
    df["pre_close"] = df["close"].shift(1)
    df["tr1"] = df["high"] - df["low"]
    df["tr2"] = abs(df["high"] - df["pre_close"])
    df["tr3"] = abs(df["low"] - df["pre_close"])
    df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
    return df["tr"].iloc[-period:].mean()


def small_update_atr_stop_prices(context, strategy):
    if not strategy['enable_atr_stop_loss']:
        return
    pindex = strategy['pindex']
    positions = context.subportfolios[pindex].positions
    for stock in positions.keys():
        atr = small_calculate_atr(stock, context, strategy['atr_period'])
        if atr:
            if stock not in strategy['atr_stop_prices']:
                strategy['atr_stop_prices'][stock] = positions[stock].avg_cost - (
                    strategy['atr_multiplier'] * atr)
            else:
                trailing_stop = positions[stock].price - (strategy['atr_multiplier'] * atr)
                if trailing_stop > strategy['atr_stop_prices'][stock]:
                    strategy['atr_stop_prices'][stock] = trailing_stop


def small_check_atr_stop_loss(context, strategy):
    if not strategy['enable_atr_stop_loss']:
        return
    pindex = strategy['pindex']
    positions = context.subportfolios[pindex].positions
    for stock in list(positions.keys()):
        if stock in strategy['atr_stop_prices'] and positions[stock].price <= strategy['atr_stop_prices'][stock]:
            if positions[stock].closeable_amount > 0:
                _sp = _get_bid1_sell_price(stock, positions[stock].price)
                _sell_qty = positions[stock].closeable_amount
                order_target(stock, 0, LimitOrderStyle(_sp), pindex=pindex)
                _slog('S5', 'ATR止损', f'触发卖出 {stock} {_sell_qty}股@{_sp:.2f}')
            else:
                del strategy['atr_stop_prices'][stock]
                continue
            if strategy.get('enable_cooldown'):
                strategy['no_buy_stocks'][stock] = context.current_dt.date()
            del strategy['atr_stop_prices'][stock]


# ==================== 成本保护止损 ====================
def small_check_cost_protection_stop_loss(context, strategy):
    if not strategy['enable_cost_protection']:
        return
    pindex = strategy['pindex']
    positions = context.subportfolios[pindex].positions
    for stock in list(positions.keys()):
        price = positions[stock].price
        avg_cost = positions[stock].avg_cost
        profit_ratio = (price - avg_cost) / avg_cost

        # 翻倍止盈
        if price >= avg_cost * 2:
            if positions[stock].closeable_amount > 0:
                _sp = _get_bid1_sell_price(stock, price)
                _sell_qty = positions[stock].closeable_amount
                order_target(stock, 0, LimitOrderStyle(_sp), pindex=pindex)
                _slog('S5', '翻倍止盈', f'卖出 {stock} {_sell_qty}股@{_sp:.2f} 盈利{profit_ratio*100:.1f}%')
            if stock in strategy['atr_stop_prices']:
                del strategy['atr_stop_prices'][stock]
            continue

        # 成本保护分档
        stop_loss_line = -strategy['stoploss_limit']
        if profit_ratio >= strategy['cost_protection_profit_threshold_2']:
            stop_loss_line = strategy['cost_protection_stop_line_2']
        elif profit_ratio >= strategy['cost_protection_profit_threshold_1']:
            stop_loss_line = strategy['cost_protection_stop_line_1']

        if profit_ratio < stop_loss_line:
            if positions[stock].closeable_amount > 0:
                _sp = _get_bid1_sell_price(stock, price)
                _sell_qty = positions[stock].closeable_amount
                order_target(stock, 0, LimitOrderStyle(_sp), pindex=pindex)
                _slog('S5', '成本保护', f'止损卖出 {stock} {_sell_qty}股@{_sp:.2f} 盈亏{profit_ratio*100:+.1f}%')
            if strategy.get('enable_cooldown'):
                strategy['no_buy_stocks'][stock] = context.current_dt.date()
            if stock in strategy['atr_stop_prices']:
                del strategy['atr_stop_prices'][stock]


# ==================== 市场暴跌止损 ====================
def small_check_market_stop_loss(context, strategy):
    stock_df = get_price(
        security=get_index_stocks("399101.XSHE"),
        end_date=context.previous_date, frequency="daily",
        fields=["close", "open"], count=1, panel=False,
    )
    down_ratio = (stock_df["close"] / stock_df["open"] - 1).mean()
    if down_ratio <= -strategy['stoploss_market']:
        pindex = strategy['pindex']
        log.warn(f"[{strategy['name']}] 大盘惨跌，平均降幅 {down_ratio:.2%}")
        for stock in list(context.subportfolios[pindex].positions.keys()):
            pos = context.subportfolios[pindex].positions[stock]
            if pos.closeable_amount > 0:
                _sp = _get_bid1_sell_price(stock, pos.price)
                _sell_qty = pos.closeable_amount
                order_target(stock, 0, LimitOrderStyle(_sp), pindex=pindex)
                _slog('S5', '市场止损', f'清仓卖出 {stock} {_sell_qty}股@{_sp:.2f}')
            if stock in strategy['atr_stop_prices']:
                del strategy['atr_stop_prices'][stock]


# ==================== 卖出 ====================
def small_iSell(context):
    """每周二 09:50 执行卖出"""
    strategy = g.small_strategy
    pindex = strategy['pindex']
    sub_portfolio = context.subportfolios[pindex]
    if g.log_trade_detail:
        _slog('S5', '卖出', f'决策开始, {len(sub_portfolio.positions)}只持仓')

    strategy['limitup_sold_today'] = []
    strategy['not_buy_again'] = []

    # 顶背离检测
    divergence_risk_flag = False
    if strategy['DBL_control']:
        if len(strategy['dbl']) < 2:
            small_check_macd_divergence(context, strategy)
        if len(strategy['dbl']) > 0 and 1 in strategy['dbl'][-strategy['check_macd_divergence_days']:]:
            divergence_risk_flag = True

    # 动态调整持仓数量
    if strategy['enable_dynamic_stock_num']:
        ma_para = 10
        today = context.previous_date
        start_date = today - datetime.timedelta(days=ma_para * 2)
        index_df = get_price("399101.XSHE", start_date=start_date, end_date=today, frequency="daily")
        index_df["ma"] = index_df["close"].rolling(window=ma_para).mean()
        last_row = index_df.iloc[-1]
        diff = last_row["close"] - last_row["ma"]
        strategy['max_position_count'] = (
            3 if diff >= 500 else
            3 if 200 <= diff < 500 else
            4 if -200 <= diff < 200 else
            5 if -500 <= diff < -200 else
            6
        )

    # 选股
    strategy['stocks'] = small_choice_small(context, strategy, strategy['max_position_count'])

    cdata = get_current_data()
    positions = context.subportfolios[pindex].positions

    if divergence_risk_flag:
        log.warning(f"[{strategy['name']}] 顶背离触发，今日只卖不买")
        for stock in list(positions.keys()):
            if stock in strategy['yesterday_HL_list']:
                continue
            if cdata[stock].paused:
                continue
            _sell_qty = positions[stock].closeable_amount
            order_target(stock, 0, LimitOrderStyle(
                _get_bid1_sell_price(stock, cdata[stock].last_price)), pindex=pindex)
            _slog('S5', '卖出', f'非持仓列表 {stock} {_sell_qty}股@{cdata[stock].last_price:.2f}')
    else:
        for s in list(positions.keys()):
            if cdata[s].paused:
                continue
            if s not in strategy['stocks'] and s not in strategy['yesterday_HL_list']:
                _sell_qty = positions[s].closeable_amount
                order_target(s, 0, LimitOrderStyle(
                    _get_bid1_sell_price(s, cdata[s].last_price)), pindex=pindex)
                _slog('S5', '卖出', f'非选股池 {s} {_sell_qty}股@{cdata[s].last_price:.2f}')


def small_daily_stoploss(context):
    """每日 10:05 执行止损检查"""
    strategy = g.small_strategy
    pindex = strategy['pindex']
    positions = context.subportfolios[pindex].positions
    if not positions:
        return

    if strategy['enable_atr_stop_loss']:
        small_update_atr_stop_prices(context, strategy)
        small_check_atr_stop_loss(context, strategy)

    if strategy.get('stoploss_strategy', 3) in [1, 3]:
        small_check_cost_protection_stop_loss(context, strategy)

    if strategy.get('stoploss_strategy', 3) in [2, 3]:
        small_check_market_stop_loss(context, strategy)


# ==================== 买入 ====================
def small_iBuy(context):
    """每周二 10:00 执行买入"""
    strategy = g.small_strategy
    pindex = strategy['pindex']
    if g.log_trade_detail:
        _slog('S5', '买入', f'决策开始, 可用资金{context.subportfolios[pindex].available_cash:,.0f}')

    # 顶背离不买
    if strategy['DBL_control'] and 1 in strategy['dbl'][-strategy['check_macd_divergence_days']:]:
        _slog('S5', '买入', '顶背离模式, 暂停买入')
        return

    current_positions = context.subportfolios[pindex].positions
    available_cash = context.subportfolios[pindex].available_cash
    cdata = get_current_data()

    buy_list = [s for s in strategy.get('stocks', [])
                if s not in current_positions or current_positions[s].total_amount == 0]
    position_size = available_cash / len(buy_list) if buy_list else 0

    # 再平衡截流
    if strategy.get('rebalance_cap') is not None and buy_list:
        cap = strategy['rebalance_cap']
        existing_value = sum(pos.value for pos in current_positions.values())
        max_buy_value = max(0, cap - existing_value)
        position_size_capped = max_buy_value / len(buy_list)
        if position_size_capped < position_size:
            _slog('S5', '再平衡', f'截流: {position_size:.0f}→{position_size_capped:.0f}元/只')
            position_size = position_size_capped

    for s in buy_list:
        if available_cash < 100 * cdata[s].last_price:
            break
        if cdata[s].paused:
            continue

        if s in current_positions:
            buy_value = max(0, position_size - current_positions[s].value)
        else:
            buy_value = position_size

        if buy_value < 5000:
            continue

        buy_value = min(buy_value, available_cash)
        buy_amount = small_calculate_dynamic_buy_amount(
            buy_value, cdata[s].last_price, available_cash, strategy)

        min_cost = 100 * cdata[s].last_price
        if buy_amount * cdata[s].last_price < min_cost:
            continue
        if buy_amount > 0:
            order(s, buy_amount,
                  style=LimitOrderStyle(round(1.01 * cdata[s].last_price, 3)),
                  pindex=pindex)
            _slog('S5', '买入',
                  f'{s} {buy_amount}股@{cdata[s].last_price:.2f} 金额{buy_amount*cdata[s].last_price:,.0f}')
            available_cash -= buy_amount * cdata[s].last_price


def small_calculate_dynamic_buy_amount(target_value, current_price, available_cash, strategy):
    max_stock_price = strategy.get('max_stock_price', 50)
    min_buy_amount = strategy.get('min_buy_amount', 200)

    if current_price > max_stock_price:
        return 0

    theoretical_amount = int(target_value / current_price)
    buy_amount = int(theoretical_amount / min_buy_amount) * min_buy_amount
    if buy_amount < min_buy_amount:
        buy_amount = min_buy_amount if (min_buy_amount * current_price <= target_value) else 0

    if buy_amount * current_price > available_cash * 0.95:
        max_affordable = int(available_cash * 0.95 / current_price / min_buy_amount) * min_buy_amount
        buy_amount = max(0, max_affordable)
    return buy_amount


# ==================== 涨停卖出 ====================
def small_check_limit_up(context):
    """14:30 检查昨日涨停股今日开板→卖出"""
    strategy = g.small_strategy
    pindex = strategy['pindex']
    if not strategy['yesterday_HL_list']:
        return

    cdata = get_current_data()
    positions = context.subportfolios[pindex].positions
    sold_count = 0

    for stock in strategy['yesterday_HL_list'][:]:
        if cdata[stock].paused:
            continue
        try:
            current_price = get_price(stock, end_date=context.current_dt,
                                      frequency='1m', fields='close',
                                      skip_paused=False, count=1).iloc[0, 0]
        except:
            current_price = cdata[stock].last_price

        if current_price < cdata[stock].high_limit - 0.001:
            last_price = cdata[stock].last_price
            log_price = last_price if last_price > 0 else current_price
            _sell_qty = positions[stock].closeable_amount
            order_target(stock, 0, LimitOrderStyle(_get_bid1_sell_price(stock, log_price)), pindex=pindex)
            _slog('S5', '涨停回落', f'卖出 {stock} {_sell_qty}股 昨涨停今日开板@{log_price:.2f}')
            sold_count += 1
            strategy['yesterday_HL_list'].remove(stock)
            strategy['limitup_sold_today'].append(stock)

    if sold_count > 0:
        strategy['reason_to_sell'] = 'limitup'
        small_buy_with_remaining_cash(context, strategy, sold_count)
        strategy['reason_to_sell'] = ''


def small_buy_with_remaining_cash(context, strategy, buy_num=None):
    """涨停卖出后用剩余资金补仓"""
    pindex = strategy['pindex']
    positions = context.subportfolios[pindex].positions
    if buy_num is None:
        max_new_positions = min(strategy['max_position_count'] - len(positions), 3)
    else:
        max_new_positions = buy_num
    if max_new_positions <= 0:
        return

    cdata = get_current_data()
    cash = context.subportfolios[pindex].available_cash
    buy_candidates = [s for s in strategy.get('stocks', [])
                      if s not in positions
                      and s not in strategy.get('not_buy_again', [])
                      and not cdata[s].paused
                      and s not in strategy.get('limitup_sold_today', [])]
    if not buy_candidates:
        return

    position_size = cash / len(buy_candidates) if buy_candidates else 0
    max_buy = min(len(buy_candidates), max_new_positions)
    bought_count = 0

    for stock in buy_candidates[:max_buy]:
        buy_value = min(position_size, cash)
        min_cost = 100 * cdata[stock].last_price
        if buy_value < min_cost:
            continue
        try:
            buy_amount = small_calculate_dynamic_buy_amount(buy_value, cdata[stock].last_price, cash, strategy)
            if buy_amount <= 0:
                continue
            actual_value = buy_amount * cdata[stock].last_price
            order(stock, buy_amount, style=LimitOrderStyle(round(1.01 * cdata[stock].last_price, 3)), pindex=pindex)
            _slog('S5', '补仓买入', f'{stock} {buy_amount}股@{cdata[stock].last_price:.2f} 金额{actual_value:,.0f}')
            strategy.setdefault('not_buy_again', []).append(stock)
            cash -= actual_value
            bought_count += 1
            min_price = min((cdata[s].last_price for s in buy_candidates), default=0)
            min_unit = 100 * min_price
            if cash < min_unit:
                break
        except Exception as e:
            log.error(f"[{strategy['name']}] 补仓买入出错: {e}")
            continue
    return bought_count > 0


# ==================== 收盘报告 ====================
def small_iReport(context):
    strategy = g.small_strategy
    pindex = strategy['pindex']
    if 'initial_total_value' not in strategy:
        strategy['initial_total_value'] = context.subportfolios[pindex].total_value

    cdata = get_current_data()
    tvalue = context.subportfolios[pindex].total_value
    total_profit = int(tvalue - context.subportfolios[pindex].inout_cash)
    base_diff = int(tvalue - strategy['initial_total_value'])
    base_ratio = (100 * (tvalue - strategy['initial_total_value']) / strategy['initial_total_value']
                  if strategy['initial_total_value'] > 0 else 0)

    ptable = pd.DataFrame(columns=['数量', '价值', '名称'])
    positions = context.subportfolios[pindex].positions
    for s in positions:
        ps = positions[s]
        ptable.loc[s] = [ps.total_amount, int(ps.value), cdata[s].name]
    ptable = ptable.sort_values(by='价值', ascending=False)

    log.info(f"[{strategy['name']}] 持仓明细:\n{ptable}")
    log.info(f"[{strategy['name']}] 总盈亏:{total_profit}, 基准盈亏:{base_diff}({base_ratio:.2f}%)")
