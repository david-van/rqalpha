#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2026/6/14 21:00
# @Author  : david_van
# @Desc    :
# 克隆自聚宽文章：https://www.joinquant.com/post/69062
# 标题：ETF轮动策略研究(四之一)：动态ETF池分析
# 作者：whn

# 策略名称：高收益ETF轮动策略 (动态池版)
# 核心算法：WLS加权线性回归

import numpy as np
import math
import pandas as pd
import re
from datetime import datetime, timedelta
from jqdata import *
from sklearn.cluster import AffinityPropagation

# ==================== 配置区 (支持参数注入) ====================
# 策略执行时间窗 (开始时间, 结束时间)
EXECUTION_BEG_TIME_PLACEHOLDER = "10:31"
EXECUTION_END_TIME_PLACEHOLDER = "10:32"

# 轮动策略核心与过滤参数
# 评分区间控制 (最低分, 最高分)
EXECUTION_SCORE_RANGE = (0.0, 6.0)
# 止损与跌幅过滤 (是否开启止损, 止损跌幅阈值)
EXECUTION_LOSE_PARAM = (True, 0.95)
# 均线过滤 (是否开启, 均线周期)
EXECUTION_MA_PARAM = (False, 20)
# 成交量校验 (是否开启, 考察天数, 成交量倍数阈值)
EXECUTION_VOLUME_PARAM = (False, 5, 1.6)
# R2 相关性过滤 (是否开启, R2 阈值)
EXECUTION_R2_PARAM = (False, 0.4)
# 短期动能过滤 (是否开启, 考察天数, 动能得分阈值)
EXECUTION_SHORT_MOMENTUM_PARAM = (False, 10, 0.0)
# 年化收益过滤 (是否开启, 最小年化收益)
EXECUTION_ANNUAL_RETURN_PARAM = (False, 1.0)
# 日内跌幅止损 (是否开启, 止损比例)
EXECUTION_DAY_LIMIT_PARAM = (True, 0.95)
# RSI 过滤 (是否开启, RSI 周期, 考察天数, 下限阈值)
EXECUTION_RSI_PARAM = (False, 6, 1, 98)
# 市场过滤参数: (过滤 A 股, 过滤跨境/QDII, 过滤债基/货币)
EXECUTION_FLITER_MARKET = (False, False, True)
# AP 聚类参数: (是否使用 AP 聚类, 阻尼系数, 偏好参数, 相关性计算窗口)
EXECUTION_AP_JOIN_MARKET = (True, 0.8, None, 60)


class Config:
    AVOID_FUTURE_DATA = True
    USE_REAL_PRICE = True
    BENCHMARK = "513100.XSHG"

    # 滑点与费率
    SLIPPAGE_FUND = 0.001
    SLIPPAGE_STOCK = 0.003
    COMMISSION_STOCK_OPEN = 0.0002
    COMMISSION_STOCK_CLOSE = 0.0002
    COMMISSION_MIN = 0

    # 策略参数
    HOLD_COUNT = 1
    M_DAYS = 25
    MIN_MONEY = 500

    # 动态池搜索参数
    POOL_MIN_LISTED_DAYS = 60  # 上市至少60天
    POOL_VOL_LOOKBACK = 20  # 考察成交额的天数
    POOL_MIN_AVG_MONEY = 5000 * 10000  # 日均成交额不低于5000万
    POOL_REFRESH_INTERVAL = 10  # 每10个交易日重新扫描一次全市场
    ETF_WHITELIST = ["518880.XSHG", "511380.XSHG", "511260.XSHG", "511090.XSHG"]
    A_SHARE_INDEX_REGEX = r"^(H\d{5}\.CSI|\d{6}\.(CSI|XSHG|XSHE))$"

    # 市场过滤参数
    POOL_FILTER_A_SHARE, POOL_FILTER_QDII, POOL_FILTER_BOND_MONEY = (
        EXECUTION_FLITER_MARKET
    )

    # AP 聚类参数
    POOL_USE_AP_CLUSTERING, AP_DAMPING, AP_PREFERENCE, AP_CORR_WINDOW = (
        EXECUTION_AP_JOIN_MARKET
    )

    # 评分与过滤
    MIN_SCORE, MAX_SCORE = EXECUTION_SCORE_RANGE
    ENABLE_LOSS_FILTER, DROP_3DAY_LIMIT = EXECUTION_LOSE_PARAM
    ENABLE_MA_FILTER, MA_FILTER_DAYS = EXECUTION_MA_PARAM
    ENABLE_VOLUME_CHECK, VOLUME_LOOKBACK, VOLUME_THRESHOLD = EXECUTION_VOLUME_PARAM
    ENABLE_R2_FILTER, R2_THRESHOLD = EXECUTION_R2_PARAM
    ENABLE_SHORT_MOMENTUM, SHORT_LOOKBACK_DAYS, SHORT_MOMENTUM_THRESHOLD = (
        EXECUTION_SHORT_MOMENTUM_PARAM
    )
    ENABLE_ANNUAL_RETURN_FILTER, MIN_ANNUALIZED_RETURN = EXECUTION_ANNUAL_RETURN_PARAM
    ENABLE_RSI_FILTER, RSI_PERIOD, RSI_LOOKBACK_DAYS, RSI_THRESHOLD = (
        EXECUTION_RSI_PARAM
    )
    ENABLE_INTRADAY_STOP_LOSS, STOP_LOSS_PCT = EXECUTION_DAY_LIMIT_PARAM

    DEFENSIVE_ETF = "511880.XSHG"  # 银华日利


# ==================== 初始化 ====================
def initialize(context):
    set_benchmark(Config.BENCHMARK)
    set_option("avoid_future_data", Config.AVOID_FUTURE_DATA)
    set_option("use_real_price", Config.USE_REAL_PRICE)

    log.set_level("order", "error")
    log.set_level("system", "error")

    set_slippage(FixedSlippage(Config.SLIPPAGE_FUND), type="fund")
    set_slippage(FixedSlippage(Config.SLIPPAGE_STOCK), type="stock")

    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0,
            open_commission=Config.COMMISSION_STOCK_OPEN,
            close_commission=Config.COMMISSION_STOCK_CLOSE,
            close_today_commission=0,
            min_commission=Config.COMMISSION_MIN,
        ),
        type="stock",
    )
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0,
            open_commission=Config.COMMISSION_STOCK_OPEN,
            close_commission=Config.COMMISSION_STOCK_CLOSE,
            close_today_commission=0,
            min_commission=Config.COMMISSION_MIN,
        ),
        type="fund",
    )
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0,
            open_commission=0,
            close_commission=0,
            close_today_commission=0,
            min_commission=0,
        ),
        type="mmf",
    )

    # 全局变量
    g.etf_pool = []
    g.pool_refresh_count = 0
    g.target_list = []

    # 定时刷新动态池
    run_daily(check_pool_refresh, "09:00")

    # 核心交易任务
    run_daily(etf_trade_sell, EXECUTION_BEG_TIME_PLACEHOLDER)
    run_daily(etf_trade_buy, EXECUTION_END_TIME_PLACEHOLDER)

    # 简单的定时止损监控（避免每分钟高频耗时）
    run_daily(simple_stop_loss, "09:31")
    run_daily(simple_stop_loss, "10:00")
    run_daily(simple_stop_loss, "11:00")
    run_daily(simple_stop_loss, "13:30")
    run_daily(simple_stop_loss, "14:30")

    # 状态输出
    run_daily(print_positions, "09:30")
    run_daily(print_positions, "15:00")


# ==================== 逻辑检查模块 ====================
def check_volume_anomaly_optimized(
    filtered_metrics, context, lookback=5, threshold=1.0
):
    """批量检测成交量是否异常放量 (避免循环调用 API)"""
    if not filtered_metrics:
        return []
    codes = [m["etf"] for m in filtered_metrics]

    try:
        # 获取上一个交易日
        last_trading_day = get_trade_days(end_date=context.current_dt, count=2)[0]
        # 批量获取历史成交量 (不含今日)
        hist = get_price(
            codes,
            count=lookback,
            end_date=last_trading_day,
            frequency="1d",
            fields=["volume"],
            panel=False,
        )
        if hist is None or hist.empty:
            return filtered_metrics
        avg_vols = hist.groupby("code")["volume"].mean()

        # 获取当日最新累计成交量 (处理回测环境下 get_current_data 或 daily 获取 volume 的限制)
        try:
            # 优先尝试快捷方式
            cur_vols_df = get_price(
                codes,
                count=1,
                end_date=context.current_dt,
                frequency="daily",
                fields=["volume"],
                panel=False,
            )
            cur_vols = (
                cur_vols_df.set_index("code")["volume"]
                if not cur_vols_df.empty
                else pd.Series()
            )
        except:
            # 触发 avoid_future_data=True 回测安全保护时的降级方案
            # 通过叠加当日分钟线 (1m) 来安全获得截止目前的累计成交量
            today_str = context.current_dt.strftime("%Y-%m-%d")
            cur_vols_df = get_price(
                codes,
                start_date=today_str,
                end_date=context.current_dt,
                frequency="1m",
                fields=["volume"],
                panel=False,
            )
            cur_vols = (
                cur_vols_df.groupby("code")["volume"].sum()
                if not cur_vols_df.empty
                else pd.Series()
            )

        passed = []
        for m in filtered_metrics:
            etf = m["etf"]
            avg_vom = avg_vols.get(etf, 0)
            cur_vom = cur_vols.get(etf, 0)

            ratio = cur_vom / avg_vom if avg_vom > 0 else 0
            m["volume_ratio"] = ratio
            if ratio <= threshold:
                passed.append(m)
        return passed
    except Exception as e:
        log.error(f"成交量检查异常: {e}")
        return filtered_metrics


def check_defensive_etf_available(context):
    """检查防御ETF是否可交易"""
    current_data = get_current_data()
    etf = Config.DEFENSIVE_ETF
    try:
        info = current_data[etf]
        if info.paused:
            log.info(f"防御性ETF {etf} 今日停牌")
            return False
        if not math.isnan(info.high_limit) and info.last_price >= info.high_limit:
            return False
        if not math.isnan(info.low_limit) and info.last_price <= info.low_limit:
            return False
        return True
    except:
        return False


# ==================== 动态池核心逻辑 ====================
def check_pool_refresh(context):
    """每日检查是否需要刷新池子（20个交易日刷新一次或初始为空时）"""
    if not g.etf_pool or g.pool_refresh_count <= 0:
        log.info("--- 💢 开始刷新全市场动态ETF池 ---")
        g.etf_pool = get_dynamic_etf_pool(context)
        log.info(f"--- 动态池刷新完成，当前成分数量: {len(g.etf_pool)} ---")
        pool_details = [
            f"[{c}] {get_security_info(c).display_name}" for c in g.etf_pool
        ]
        for i in range(0, len(pool_details), 3):
            log.info("  |  ".join(pool_details[i : i + 3]))
        g.pool_refresh_count = Config.POOL_REFRESH_INTERVAL
    else:
        g.pool_refresh_count -= 1


def get_dynamic_etf_pool(context):
    """
    动态 ETF 池生成，共四步：
      Step 1  全市场扫描 + 关键词预过滤
      Step 2  上市年龄过滤
      Step 3  流动性过滤（成交额 + 价格）
      Step 4  去重
      Step 5  AP 聚类优化
    返回去重后的 ETF 代码列表。
    """
    current_date = context.current_dt.date()

    # ── Step 1: 全市场扫描 + 关键词预过滤 ──────────────────────────────
    all_etfs = get_all_securities(types=["etf"])

    exclude_keywords = []
    if Config.POOL_FILTER_BOND_MONEY:
        exclude_keywords += ["债", "货币", "理财"]
    if Config.POOL_FILTER_QDII:
        exclude_keywords += [
            "QDII",
            "标普",
            "纳指",
            "道琼斯",
            "恒生",
            "H股",
            "日经",
            "德国",
            "法国",
            "英国",
            "美国",
            "海外",
        ]
    if Config.POOL_FILTER_A_SHARE:
        exclude_keywords += [
            "300",
            "500",
            "50",
            "800",
            "1000",
            "2000",
            "中证",
            "创业板",
            "科创",
            "沪深",
            "上证",
            "深证",
            "A股",
        ]
    if exclude_keywords:
        pattern = "|".join(exclude_keywords)
        all_etfs = all_etfs[~all_etfs["display_name"].str.contains(pattern, regex=True)]

    # ── Step 2: 上市年龄过滤 ──────────────────────────────────────────
    min_date = current_date - timedelta(days=Config.POOL_MIN_LISTED_DAYS)
    candidates = all_etfs[all_etfs["start_date"] < min_date].index.tolist()
    if not candidates:
        return []

    # ── Step 3: 流动性过滤 ────────────────────────────────────────────
    # 取上一个交易日，避免 FutureDataError
    last_trading_day = get_trade_days(end_date=current_date, count=2)[0]

    # 3a. 成交额过滤
    prices = get_price(
        candidates,
        count=Config.POOL_VOL_LOOKBACK,
        end_date=last_trading_day,
        frequency="daily",
        fields=["money"],
        panel=False,
    )
    if prices is None or prices.empty:
        return []

    avg_money = prices.groupby("code")["money"].mean()
    liquid_etfs = avg_money[avg_money >= Config.POOL_MIN_AVG_MONEY].index.tolist()

    # 3b. 价格过滤：均价 > 90 元 → 债券 / 货币类，剔除
    if Config.POOL_FILTER_BOND_MONEY and liquid_etfs:
        hist_avg = get_price(
            liquid_etfs,
            count=1,
            end_date=last_trading_day,
            frequency="daily",
            fields=["avg"],
            panel=False,
        )
        if hist_avg is not None and not hist_avg.empty:
            last_avg = hist_avg.groupby("code")["avg"].last()
            bond_etfs = last_avg[last_avg > 90].index.tolist()
            liquid_etfs = [e for e in liquid_etfs if e not in bond_etfs]

    if not liquid_etfs:
        return []

    # ── Step 4: 指数级去重与精准过滤 (始终执行) ────────────────────────
    index_deduped = _dedup_by_index(liquid_etfs, avg_money, current_date)

    # 这里的白名单保底纳入 Step 4 后半部分，确保进入聚类前包含核心资产
    for wl in Config.ETF_WHITELIST:
        if wl in liquid_etfs and wl not in index_deduped:
            index_deduped.append(wl)

    # ── Step 5: AP 聚类优化 (基于指数去重后的结果进行二次压缩) ──────────
    if Config.POOL_USE_AP_CLUSTERING:
        final_pool = _dedup_by_ap_clustering(index_deduped, avg_money, last_trading_day)
        if final_pool is None:
            log.warn("AP 聚类失败，保留指数去重结果")
            final_pool = index_deduped
    else:
        final_pool = index_deduped

    return final_pool


def _dedup_by_ap_clustering(candidate_etfs, avg_money, last_trading_day):
    """
    AP 聚类优化。
    对输入的候选池计算日收益率相关性，使用 AffinityPropagation 自动分簇，
    每簇选成交额最大的 ETF 作为代表。
    成功返回列表；异常返回 None。
    """
    from collections import defaultdict

    try:
        # 获取收盘价，构造收益率矩阵
        prices_ap = get_price(
            candidate_etfs,
            count=Config.AP_CORR_WINDOW + 1,
            end_date=last_trading_day,
            frequency="daily",
            fields=["close"],
            panel=False,
        )
        if prices_ap is None or prices_ap.empty:
            return None

        close_pivot = prices_ap.pivot(index="time", columns="code", values="close")
        close_pivot = close_pivot.fillna(method="ffill").dropna(axis=1, how="any")
        returns = close_pivot.pct_change().dropna()
        valid_etfs = returns.columns.tolist()
        if len(valid_etfs) < 2:
            return None

        # AP 聚类
        corr_matrix = returns.corr()
        ap = AffinityPropagation(
            damping=Config.AP_DAMPING,
            preference=Config.AP_PREFERENCE,
            affinity="precomputed",
            # random_state=42
        )
        ap.fit(corr_matrix)

        cluster_dict = defaultdict(list)
        for etf, label in zip(valid_etfs, ap.labels_):
            cluster_dict[label].append(etf)

        # 每簇选成交额最大的作为代表
        final_pool = [
            max(etfs, key=lambda x: avg_money.get(x, 0))
            for etfs in cluster_dict.values()
        ]

        # 补回因数据缺失未参与聚类的原候选标的
        final_pool += [e for e in candidate_etfs if e not in valid_etfs]

        return final_pool

    except Exception as e:
        log.warn(f"AP 聚类异常: {e}")
        return None


def _dedup_by_index(liquid_etfs, avg_money, current_date):
    """
    基于指数的精准过滤与去重。
    1. 查询 ETF 追踪指数
    2. 执行 A 股正则硬过滤
    3. 同一指数只保留成交额最大的一只
    """
    # 查询指数关联信息
    try:
        q = query(finance.FUND_INVEST_TARGET).filter(
            finance.FUND_INVEST_TARGET.code.in_(liquid_etfs)
        )
        df_target = finance.run_query(q)
    except Exception as e:
        log.warn(f"无法查询指数信息: {e}，将返回全部流动性池")
        return liquid_etfs

    # 1. A 股指数代码正则过滤
    if Config.POOL_FILTER_A_SHARE and not df_target.empty:
        mask = df_target["traced_index_code"].str.match(
            Config.A_SHARE_INDEX_REGEX, na=False
        )
        a_codes = df_target.loc[mask, "code"].unique().tolist()
        if a_codes:
            liquid_etfs = [c for c in liquid_etfs if c not in a_codes]
            df_target = df_target[~df_target["code"].isin(a_codes)]

    if df_target.empty:
        return liquid_etfs

    # 2. 取当前日期前最新一条指数记录
    df_target["start_date"] = pd.to_datetime(df_target["start_date"])
    df_target = df_target[df_target["start_date"].dt.date <= current_date]
    current_index_map = df_target.sort_values(
        "start_date", ascending=False
    ).drop_duplicates("code", keep="first")

    # group_key：优先用 traced_index_code，为空时退回 traced_index_name
    current_index_map["group_key"] = (
        current_index_map["traced_index_code"]
        .replace("", None)
        .fillna(current_index_map["traced_index_name"])
    )
    current_index_map["avg_money"] = current_index_map["code"].map(avg_money)

    # 3. 同一 group_key 只保留成交额最大的一只；无 group_key 的全部保留
    valid_grouped = current_index_map[current_index_map["group_key"].notna()]
    no_index_etfs = [c for c in liquid_etfs if c not in valid_grouped["code"].tolist()]

    best_etfs = valid_grouped.sort_values("avg_money", ascending=False).drop_duplicates(
        "group_key", keep="first"
    )
    final_pool = best_etfs["code"].tolist() + no_index_etfs

    return final_pool


# ==================== 策略交易逻辑 ====================
def calculate_rsi(prices, period=6):
    if len(prices) < period + 1:
        return np.array([])
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    alpha = 2.0 / (period + 1)
    avg_gains = np.zeros(len(deltas))
    avg_losses = np.zeros(len(deltas))
    avg_gains[period - 1], avg_losses[period - 1] = (
        np.mean(gains[:period]),
        np.mean(losses[:period]),
    )
    for i in range(period, len(deltas)):
        avg_gains[i] = (gains[i] * alpha) + (avg_gains[i - 1] * (1 - alpha))
        avg_losses[i] = (losses[i] * alpha) + (avg_losses[i - 1] * (1 - alpha))
    rs = np.divide(
        avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses != 0
    )
    rsi = 100 - (100 / (1 + rs))
    res = np.full(len(prices), np.nan)
    res[1:] = rsi
    return res[period:]


def wls_score(prices):
    y, x = np.log(prices), np.arange(len(prices))
    w = np.linspace(1, 2, len(y))
    slope, intercept = np.polyfit(x, y, 1, w=w)
    ann_ret = math.exp(slope * 250) - 1
    y_pred = slope * x + intercept
    ss_res = np.sum(w * (y - y_pred) ** 2)
    ss_tot = np.sum(w * (y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    return ann_ret * r2, {"r2": r2, "ret": ann_ret}


def apply_filters(metrics_list, context):
    """应用所有配置的风险过滤器并输出详细日志"""
    steps = [
        ("动量范围", lambda m: m["passed_momentum"], True),
        ("短期回归", lambda m: m["passed_short_mom"], Config.ENABLE_SHORT_MOMENTUM),
        ("R²", lambda m: m["passed_r2"], Config.ENABLE_R2_FILTER),
        (
            "年化收益",
            lambda m: m["passed_annual_ret"],
            Config.ENABLE_ANNUAL_RETURN_FILTER,
        ),
        ("均线过滤", lambda m: m["passed_ma"], Config.ENABLE_MA_FILTER),
        ("风控(跌幅)", lambda m: m["passed_loss"], Config.ENABLE_LOSS_FILTER),
        ("RSI", lambda m: m["passed_rsi"], Config.ENABLE_RSI_FILTER),
    ]

    filtered = metrics_list[:]
    log.info(f"== 过滤前初始标的数量: {len(filtered)} ==")

    for name, condition, is_enabled in steps:
        if is_enabled:
            before_len = len(filtered)
            filtered = [m for m in filtered if condition(m)]
            log.info(
                f"[{name}] 过滤器已开启, 过滤前: {before_len}, 过滤后: {len(filtered)}"
            )

    # 成交量异常检测 (已优化为批量模式)
    if Config.ENABLE_VOLUME_CHECK and filtered:
        before_len = len(filtered)
        filtered = check_volume_anomaly_optimized(
            filtered, context, Config.VOLUME_LOOKBACK, Config.VOLUME_THRESHOLD
        )
        log.info(
            f"[成交量] 过滤器已开启, 过滤前: {before_len}, 过滤后: {len(filtered)}"
        )

    return filtered


def get_target_list(context):
    """筛选目标ETF列表 (已高性能向量化重构)"""
    if not g.etf_pool:
        return []

    # 1. 批量获取所有标的历史收盘价 (单次 API 调用)
    # 获取上一个交易日
    last_trading_day = get_trade_days(end_date=context.current_dt, count=2)[0]

    # 确定需要的最大历史天数 (对齐 Opt 版本)
    lookback = (
        max(
            Config.M_DAYS,
            Config.SHORT_LOOKBACK_DAYS if Config.ENABLE_SHORT_MOMENTUM else 0,
            Config.RSI_PERIOD + Config.RSI_LOOKBACK_DAYS
            if Config.ENABLE_RSI_FILTER
            else 0,
            Config.MA_FILTER_DAYS if Config.ENABLE_MA_FILTER else 0,
        )
        + 5
    )

    df = get_price(
        g.etf_pool,
        count=lookback,
        end_date=last_trading_day,
        frequency="1d",
        fields=["close"],
        panel=False,
    )

    if df is None or df.empty:
        return []

    curr_data = get_current_data()
    all_metrics = []

    # 2. 向量化计算评分
    for etf, group in df.groupby("code"):
        # 获取实时价
        cur_price = curr_data[etf].last_price
        if math.isnan(cur_price) or cur_price <= 0:
            continue

        hist_close = group["close"].values
        if len(hist_close) < Config.M_DAYS:
            continue
        prices = np.append(hist_close, cur_price)

        # 计算 WLS 评分
        score, details = wls_score(prices[-(Config.M_DAYS + 1) :])

        # 预计算各项指标
        res = {
            "etf": etf,
            "score": score,
            "r2": details["r2"],
            "passed_momentum": Config.MIN_SCORE < score < Config.MAX_SCORE,
            "passed_annual_ret": details["ret"] >= Config.MIN_ANNUALIZED_RETURN,
            "passed_r2": details["r2"] > Config.R2_THRESHOLD,
            "passed_ma": cur_price >= np.mean(prices[-Config.MA_FILTER_DAYS :])
            if len(prices) >= Config.MA_FILTER_DAYS
            else True,
            "passed_loss": True,
            "passed_rsi": True,
        }

        # 3. 补充各子过滤器指标
        short_look = Config.SHORT_LOOKBACK_DAYS
        if len(prices) > short_look:
            short_ret = prices[-1] / prices[-(short_look + 1)] - 1
            res["passed_short_mom"] = (
                (1 + short_ret) ** (250 / short_look) - 1
            ) >= Config.SHORT_MOMENTUM_THRESHOLD
        else:
            res["passed_short_mom"] = False

        if len(prices) >= 4:
            if (
                min(
                    [
                        prices[-1] / prices[-2],
                        prices[-2] / prices[-3],
                        prices[-3] / prices[-4],
                    ]
                )
                < Config.DROP_3DAY_LIMIT
            ):
                res["passed_loss"] = False

        if Config.ENABLE_RSI_FILTER:
            rsis = calculate_rsi(prices, Config.RSI_PERIOD)
            if len(rsis) >= Config.RSI_LOOKBACK_DAYS and np.any(
                rsis[-Config.RSI_LOOKBACK_DAYS :] > Config.RSI_THRESHOLD
            ):
                if cur_price < np.mean(prices[-5:]):
                    res["passed_rsi"] = False

        all_metrics.append(res)

    # 评分排序
    all_metrics.sort(key=lambda x: x["score"], reverse=True)

    # 4. 风险过滤管道
    filtered_metrics = apply_filters(all_metrics, context)

    if filtered_metrics:
        top3_info = [f"{x['etf']}:{x['score']:.3f}" for x in filtered_metrics[:3]]
        log.info(f"排名最前的ETF: {top3_info}")
    else:
        log.info(f"所有目标被风险过滤器拦截，当前无符合条件的 ETF。")

    return [x["etf"] for x in filtered_metrics]


def smart_order_target(security, value, context):
    """带手数控制与风控的智能下单"""
    current_data = get_current_data()
    pos = context.portfolio.positions.get(security)
    current_shares = pos.total_amount if pos else 0
    closeable = pos.closeable_amount if pos else 0

    # 清仓逻辑
    if value == 0:
        if closeable > 0:
            order(security, -closeable)
            log.info(f"💰 [真实交易] 卖出清仓 {security}, 委托数量: {closeable} 股")
        return

    price = current_data[security].last_price
    if math.isnan(price) or price <= 0 or current_data[security].paused:
        if current_data[security].paused:
            log.info(f"{security}: 今日停牌或无数据，放弃买入")
        return
    if (
        price >= current_data[security].high_limit
        or price <= current_data[security].low_limit
    ):
        return

    # 计算目标股数 (100股整数倍)
    target_shares = (int(value / price) // 100) * 100
    diff = target_shares - current_shares

    if abs(diff) * price < Config.MIN_MONEY:
        return

    if diff > 0:
        o = order(security, diff)
        if o:
            log.info(
                f"💰 [真实交易] 买入建仓/加仓 {security}, 委托买入: {diff} 股, 约合 {diff * price:.2f} 元"
            )
    elif diff < 0:
        actual_sell = min(abs(diff), closeable)
        if actual_sell > 0:
            o = order(security, -actual_sell)
            if o:
                log.info(
                    f"💰 [真实交易] 卖出减仓 {security}, 委托卖出: {actual_sell} 股, 约合 {actual_sell * price:.2f} 元"
                )


def etf_trade_sell(context):
    """执行卖出逻辑 (10:31)"""
    log.info(f"=== 开始执行卖出逻辑 ({EXECUTION_BEG_TIME_PLACEHOLDER}) ===")
    raw_targets = get_target_list(context)
    ideal = raw_targets[: Config.HOLD_COUNT]

    if not ideal:
        if check_defensive_etf_available(context):
            ideal = [Config.DEFENSIVE_ETF]
            log.info(f"🛡️ 开启防御模式，今日理想目标为避险ETF: {ideal}")
        else:
            log.info("💤 无有效目标且防御ETF不可交易，空仓状态")
            ideal = []

    g.target_list = ideal
    current_positions = list(context.portfolio.positions.keys())

    for etf in current_positions:
        pos = context.portfolio.positions[etf]
        if pos.total_amount == 0:
            continue

        if etf not in ideal:
            if etf == Config.DEFENSIVE_ETF:
                reason = "由守转攻，防御阶段结束"
            elif not ideal or (len(ideal) == 1 and ideal[0] == Config.DEFENSIVE_ETF):
                reason = "触发防御避险机制"
            else:
                reason = "动量或排名掉队"
            smart_order_target(etf, 0, context)
            positions = list(context.portfolio.positions.keys())
            pos_str = ", ".join(positions) if positions else "空仓"
            log.info(f"卖出 {etf}: {reason}; 当前持仓: {pos_str}")


def etf_trade_buy(context):
    """执行买入逻辑 (10:32)"""
    log.info(f"=== 开始执行买入逻辑 ({EXECUTION_END_TIME_PLACEHOLDER}) ===")
    targets = g.target_list
    if not targets:
        log.info("无目标标的")
        return

    log.info(f"最终买入目标: {targets}")
    per_value = context.portfolio.total_value / len(targets)
    available_cash = context.portfolio.available_cash
    current_data = get_current_data()

    for etf in targets:
        if etf not in current_data or current_data[etf].paused:
            log.info(f"⏭️ 目标 {etf} 今日停牌或无数据，跳过买入")
            continue

        last_price = current_data[etf].last_price
        if math.isnan(last_price) or last_price <= 0:
            continue

        pos = context.portfolio.positions.get(etf)
        current_shares = pos.total_amount if pos else 0
        current_value = current_shares * last_price

        # 资金门槛拦截：仓位不足目标90%时才考虑买入
        if current_value < per_value * 0.9:
            need_buy_value = per_value - current_value
            actual_buy_value = min(need_buy_value, available_cash)
            # 严格防范资金连一手都不够的情况
            if actual_buy_value <= max(Config.MIN_MONEY, last_price * 100):
                log.info(
                    f"⚠️ 买入金额不足（需{need_buy_value:.2f}元，可用{available_cash:.2f}元），不买入碎片金额。"
                )
                continue
            safe_target_value = current_value + actual_buy_value
            smart_order_target(etf, safe_target_value, context)
        else:
            smart_order_target(etf, per_value, context)

        available_cash = context.portfolio.available_cash


def simple_stop_loss(context):
    """日内止损监控 (遵循 T+1)"""
    if not Config.ENABLE_INTRADAY_STOP_LOSS:
        return
    for etf, pos in context.portfolio.positions.items():
        if (
            etf == Config.DEFENSIVE_ETF
            or pos.total_amount == 0
            or pos.closeable_amount == 0
        ):
            continue

        current_data = get_current_data()
        current_price = current_data[etf].last_price
        cost_price = pos.avg_cost

        if current_price <= cost_price * Config.STOP_LOSS_PCT:
            loss_percent = (current_price / cost_price - 1) * 100
            log.info(
                f"🚨 [日内监控] {context.current_dt.strftime('%H:%M')} 触发止损: {etf}，当前价: {current_price:.3f}, 成本: {cost_price:.3f}, 亏损: {loss_percent:.2f}%"
            )

            o = order(etf, -pos.closeable_amount)
            if o:
                log.info(
                    f"✅ [日内监控] 止损卖出委托下达: {etf}, 数量: {pos.closeable_amount} 股"
                )


def print_positions(context):
    """打印当前持仓状态"""
    positions = list(context.portfolio.positions.keys())
    if not positions:
        log.info("=== 当前持仓: 空仓 ===")
    else:
        log.info(f"=== 当前持仓: {positions} ===")

    if context.current_dt.hour >= 14:
        log.info("")
