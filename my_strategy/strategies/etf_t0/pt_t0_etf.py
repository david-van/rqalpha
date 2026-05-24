#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _typing import *


def initialize(context):
    set_benchmark("513030.SS")
    set_universe("513030.SS")

    set_commission(commission_ratio=0.00005, min_commission=0.1, type="ETF")
    set_slippage(0)

    g.security1 = "513030.SS"
    g.yclose = 1
    g.open = 1
    g.thresholdMin = 0.003
    g.thresholdMax = 0.5
    g.starting_cash = _total_value(context)

    g.enable_compare_csv = False
    g.compare_dir = "t0_etf_compare"
    g.compare_csv = (
        get_research_path() + g.compare_dir + "/t0_etf_open_vs_actual_buy.csv"
    )
    g.compare_count = 0
    g.compare_more_expensive_count = 0
    g.compare_cheaper_count = 0
    g.compare_equal_count = 0
    g.compare_total_amount = 0
    g.compare_total_open_value = 0
    g.compare_total_cost_diff = 0
    if g.enable_compare_csv:
        _init_compare_csv()

    run_daily(context, before_market_open, "09:00")
    run_daily(context, market_open, "09:31")
    run_daily(context, after_market_close, "14:45")
    run_daily(context, after_market_close2, "15:05")


def handle_data(context, data):
    pass


def before_market_open(context):
    pass


def market_open(context):
    g.open = _get_day_open(g.security1)
    g.yclose = _get_yesterday_close(g.security1)

    if (g.open - g.yclose >= g.thresholdMin) and (g.open - g.yclose <= g.thresholdMax):
        order_value(g.security1, _available_cash(context))


def after_market_close(context):
    positions = context.portfolio.positions
    if len(positions) > 0:
        order_target(g.security1, 0)


def after_market_close2(context):
    trades = get_trades()
    if len(trades) > 0:
        _append_buy_compare_record(context, trades)
        log.info("昨日收盘价: " + str(g.yclose) + ", 今日开盘价: " + str(g.open))
        for trade in _iter_trades(trades):
            log.info("成交记录: " + str(trade))
        log.info(
            "一天结束, 余额: "
            + str(round(_available_cash(context), 0))
            + ", 总价值: "
            + str(round(_total_value(context), 0))
            + ", 收益: "
            + str(round(_total_value(context) - g.starting_cash, 0))
        )
        log.info("##############################################################")
    else:
        log.info("今日无交易")


def _get_day_open(security):
    history = get_history(
        1,
        frequency="1d",
        field="open",
        security_list=security,
        fq="pre",
        include=True,
    )
    return history["open"][-1]


def _get_yesterday_close(security):
    history = get_history(
        1,
        frequency="1d",
        field=["open", "close", "high", "low"],
        security_list=security,
        fq="pre",
        include=False,
    )
    return history["close"][-1]


def _available_cash(context):
    if hasattr(context.portfolio, "available_cash"):
        return context.portfolio.available_cash
    return context.portfolio.cash


def _total_value(context):
    if hasattr(context.portfolio, "total_value"):
        return context.portfolio.total_value
    return context.portfolio.portfolio_value


def _iter_trades(trades):
    if isinstance(trades, dict):
        return trades.values()
    return trades


def _init_compare_csv():
    create_dir(g.compare_dir)
    with open(g.compare_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "date",
                "security",
                "yclose",
                "open_price",
                "actual_buy_price",
                "actual_buy_amount",
                "price_diff",
                "price_diff_pct",
                "cost_diff",
                "cum_trade_count",
                "cum_more_expensive_count",
                "cum_cheaper_count",
                "cum_equal_count",
                "cum_actual_buy_amount",
                "cum_cost_diff",
                "cum_weighted_price_diff",
                "cum_weighted_diff_pct",
            ]
        )
        writer.writerow(
            [
                "日期",
                "标的",
                "昨日收盘价",
                "今日开盘价",
                "实际买入价",
                "实际买入数量",
                "价差",
                "价差比例",
                "成本差额",
                "累计买入天数",
                "累计高于开盘次数",
                "累计低于开盘次数",
                "累计等于开盘次数",
                "累计买入数量",
                "累计成本差额",
                "累计加权价差",
                "累计加权差异比例",
            ]
        )


def _append_buy_compare_record(context, trades):
    if not g.enable_compare_csv:
        return

    buy_amount = 0
    buy_value = 0
    for trade in _iter_trades(trades):
        trade_info = _parse_trade(trade)
        if trade_info is None:
            continue
        security, direction, amount, price = trade_info
        if (
            not _same_security(security, g.security1)
            or direction != "buy"
            or amount <= 0
            or price <= 0
        ):
            continue
        buy_amount += amount
        buy_value += amount * price

    if buy_amount <= 0:
        return

    actual_buy_price = buy_value / buy_amount
    price_diff = actual_buy_price - g.open
    price_diff_pct = actual_buy_price / g.open - 1 if g.open else 0
    cost_diff = price_diff * buy_amount

    g.compare_count += 1
    if price_diff > 0:
        g.compare_more_expensive_count += 1
    elif price_diff < 0:
        g.compare_cheaper_count += 1
    else:
        g.compare_equal_count += 1
    g.compare_total_amount += buy_amount
    g.compare_total_open_value += g.open * buy_amount
    g.compare_total_cost_diff += cost_diff

    cum_weighted_price_diff = (
        g.compare_total_cost_diff / g.compare_total_amount
        if g.compare_total_amount
        else 0
    )
    cum_weighted_diff_pct = (
        g.compare_total_cost_diff / g.compare_total_open_value
        if g.compare_total_open_value
        else 0
    )

    with open(g.compare_csv, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                context.current_dt.strftime("%Y-%m-%d"),
                g.security1,
                round(g.yclose, 6),
                round(g.open, 6),
                round(actual_buy_price, 6),
                int(buy_amount),
                round(price_diff, 6),
                round(price_diff_pct, 8),
                round(cost_diff, 2),
                g.compare_count,
                g.compare_more_expensive_count,
                g.compare_cheaper_count,
                g.compare_equal_count,
                int(g.compare_total_amount),
                round(g.compare_total_cost_diff, 2),
                round(cum_weighted_price_diff, 6),
                round(cum_weighted_diff_pct, 8),
            ]
        )


def _parse_trade(trade):
    trade = _normalize_trade_row(trade)

    if isinstance(trade, dict):
        security = trade.get("stock_code", trade.get("symbol", ""))
        amount = _to_float(trade.get("business_amount", trade.get("filled_amount", 0)))
        price = _to_float(trade.get("business_price", trade.get("price", 0)))
        direction = _parse_direction(
            trade.get("entrust_bs", trade.get("direction", ""))
        )
        if direction is None:
            return None
        return security, direction, amount, price

    if isinstance(trade, (list, tuple)):
        if len(trade) < 6:
            return None
        security = trade[2]
        direction = _parse_direction(trade[3])
        amount = _to_float(trade[4])
        price = _to_float(trade[5])
        if direction is None:
            return None
        return security, direction, amount, price

    security = getattr(trade, "symbol", getattr(trade, "security", ""))
    amount = _to_float(getattr(trade, "amount", 0))
    price = _to_float(getattr(trade, "price", 0))
    direction = _parse_direction(
        getattr(trade, "entrust_direction", getattr(trade, "direction", ""))
    )
    if direction is None:
        return None
    return security, direction, amount, price


def _normalize_trade_row(trade):
    while (
        isinstance(trade, (list, tuple))
        and len(trade) == 1
        and isinstance(trade[0], (list, tuple, dict))
    ):
        trade = trade[0]
    return trade


def _parse_direction(value):
    direction_text = str(value).strip().lower()
    if direction_text in ("1", "buy", "b", "买", "买入", "证券买入"):
        return "buy"
    if direction_text in ("2", "sell", "s", "卖", "卖出", "证券卖出"):
        return "sell"
    if "buy" in direction_text or "买" in direction_text:
        return "buy"
    if "sell" in direction_text or "卖" in direction_text:
        return "sell"
    return None


def _same_security(left, right):
    return _normalize_security(left) == _normalize_security(right)


def _normalize_security(security):
    text = str(security).strip().upper()
    return text.replace(".XSHG", ".SS").replace(".XSHE", ".SZ").replace(".SH", ".SS")


def _to_float(value):
    try:
        return float(value)
    except Exception:
        return 0
