#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2026/7/5 22:46
# @Author  : david_van
# @Desc    :
# 克隆自聚宽文章：https://www.joinquant.com/post/75205
# 标题：ETF轮动策略研究(四之三)：动态池策略的参数回测与结果分析
# 作者：whn

# 克隆自聚宽文章：https://www.joinquant.com/post/67438
# 修正优化版：使用动态ETF池

import numpy as np                      # 数值计算
import math                             # 数学函数
import pandas as pd                     # 数据处理
from jqdata import *                    # 聚宽数据接口
from datetime import datetime, date, timedelta  # 日期时间处理

# ==================== 自动化优化参数占位符与 Config 类 ====================

EXECUTION_BEG_TIME_PLACEHOLDER = '13:10'
EXECUTION_END_TIME_PLACEHOLDER = '13:11'

EXECUTION_SCORE_THRESHOLD = (0, 5)
EXECUTION_SHORT_MOMENTUM_PARAM = (False, 10, 0.0)      # (启用, 天数, 阈值)
EXECUTION_R2_PARAM = (True, 0.4)                      # (启用, 阈值)
EXECUTION_ANNUAL_RETURN_PARAM = (True, 1.0)           # (启用, 阈值)
EXECUTION_MA_PARAM = (False, 20)                      # (启用, 天数)
EXECUTION_VOLUME_PARAM = (True, 5, 1.2)               # (启用, 回看, 阈值)
EXECUTION_LOSS_PARAM = (True, 0.97)                   # (启用, 阈值)
EXECUTION_RSI_PARAM = (False, 6, 1, 98)                # (启用, 周期, 回看, 阈值)

EXECUTION_FIXED_STOPLOSS = (True, 0.95)               # (启用, 阈值)
EXECUTION_PCT_STOPLOSS = (False, 0.95)                # (启用, 阈值)
EXECUTION_ATR_STOPLOSS = (False, 14, 2, True, True)   # (启用, 周期, 倍数, 跟踪, 排除防御)
EXECUTION_SELL_COOLDOWN = (True, 3)                  # (启用, 天数)

# 动态ETF池配置
EXECUTION_FLITER_MARKET = (False, False, True)        # (过滤A股, 过滤跨境/QDII, 过滤债基/货币)


class Config:
    # 策略全局配置类
    AVOID_FUTURE_DATA = True
    USE_REAL_PRICE = True
    BENCHMARK = "513100.XSHG"
    HOLDINGS_NUM = 1
    LOOKBACK_DAYS = 25
    DEFENSIVE_ETF = "511880.XSHG"     # 防御型ETF
    SAFE_HAVEN_ETF = '511660.XSHG'    # 冷却期避险ETF
    MIN_MONEY = 5000                  # 最小交易金额

    # 动态池搜索参数
    POOL_MIN_LISTED_DAYS = 60                           # 上市至少60天
    POOL_VOL_LOOKBACK = 20                              # 考察成交额的天数
    POOL_MIN_AVG_MONEY = 5000 * 10000                   # 日均成交额不低于5000万
    POOL_REFRESH_INTERVAL = 20                          # 每20个交易日重新扫描一次全市场
    ETF_WHITELIST = ['518880.XSHG', '511380.XSHG', '511260.XSHG', '511090.XSHG']
    A_SHARE_INDEX_REGEX = r'^(H\d{5}\.CSI|\d{6}\.(CSI|XSHG|XSHE))$'
    POOL_FILTER_A_SHARE, POOL_FILTER_QDII, POOL_FILTER_BOND_MONEY = EXECUTION_FLITER_MARKET

    # 动量计算
    MIN_SCORE_THRESHOLD, MAX_SCORE_THRESHOLD = EXECUTION_SCORE_THRESHOLD

    # 过滤器配置
    USE_SHORT_MOMENTUM_FILTER, SHORT_LOOKBACK_DAYS, SHORT_MOMENTUM_THRESHOLD = EXECUTION_SHORT_MOMENTUM_PARAM
    ENABLE_R2_FILTER, R2_THRESHOLD = EXECUTION_R2_PARAM
    ENABLE_ANNUAL_RETURN_FILTER, MIN_ANNUAL_RETURN = EXECUTION_ANNUAL_RETURN_PARAM
    ENABLE_MA_FILTER, MA_FILTER_DAYS = EXECUTION_MA_PARAM
    ENABLE_VOLUME_CHECK, VOLUME_LOOKBACK, VOLUME_THRESHOLD = EXECUTION_VOLUME_PARAM
    ENABLE_LOSS_FILTER, LOSS = EXECUTION_LOSS_PARAM
    USE_RSI_FILTER, RSI_PERIOD, RSI_LOOKBACK_DAYS, RSI_THRESHOLD = EXECUTION_RSI_PARAM

    # 止损配置
    DAILY_TIME_STOP_LOSS = ['09:35', '10:00', '11:00', '13:30', '14:30']
    USE_FIXED_STOP_LOSS, FIXED_STOP_LOSS_THRESHOLD = EXECUTION_FIXED_STOPLOSS
    USE_PCT_STOP_LOSS, PCT_STOP_LOSS_THRESHOLD = EXECUTION_PCT_STOPLOSS
    USE_ATR_STOP_LOSS, ATR_PERIOD, ATR_MULTIPLIER, ATR_TRAILING_STOP, ATR_EXCLUDE_DEFENSIVE = EXECUTION_ATR_STOPLOSS

    # 冷却期
    SELL_COOLDOWN_ENABLED, SELL_COOLDOWN_DAYS = EXECUTION_SELL_COOLDOWN


# --- 策略主控与初始化 ---
def show_params_log():
    infos = f"""策略参数初始化完成:
=== 基础参数 ===
- ETF池模式: 动态池
- 动量计算周期: {Config.LOOKBACK_DAYS} 天
- 持仓数量: {Config.HOLDINGS_NUM}
- 防御ETF: {Config.DEFENSIVE_ETF}
- 冷却期避险ETF: {Config.SAFE_HAVEN_ETF}
- 动态池刷新间隔: {Config.POOL_REFRESH_INTERVAL} 个交易日
- 动态池最少上市天数: {Config.POOL_MIN_LISTED_DAYS} 天
- 动态池最小日均成交额: {Config.POOL_MIN_AVG_MONEY / 10000:.0f} 万
=== 过滤条件 ===
- 动量得分过滤: {'启用' if (Config.MIN_SCORE_THRESHOLD > -1e9 or Config.MAX_SCORE_THRESHOLD < 1e9) else '禁用'} (范围: [{Config.MIN_SCORE_THRESHOLD}, {Config.MAX_SCORE_THRESHOLD}])
- 短期动量过滤: {'启用' if Config.USE_SHORT_MOMENTUM_FILTER else '禁用'} (周期: {Config.SHORT_LOOKBACK_DAYS}天)
- R²过滤: {'启用' if Config.ENABLE_R2_FILTER else '禁用'} (阈值 > {Config.R2_THRESHOLD:.3f})
- 年化收益率过滤: {'启用' if Config.ENABLE_ANNUAL_RETURN_FILTER else '禁用'} (阈值 > {Config.MIN_ANNUAL_RETURN:.1%})
- 均线过滤: {'启用' if Config.ENABLE_MA_FILTER else '禁用'} ({Config.MA_FILTER_DAYS}日均线)
- 成交量过滤: {'启用' if Config.ENABLE_VOLUME_CHECK else '禁用'} (均量比 < {Config.VOLUME_THRESHOLD:.2f})
- 短期风控过滤: {'启用' if Config.ENABLE_LOSS_FILTER else '禁用'} (最大单日跌幅限制 < {1 - Config.LOSS:.1%})
- RSI过滤: {'启用' if Config.USE_RSI_FILTER else '禁用'} (周期: {Config.RSI_PERIOD}, 超买阈值 > {Config.RSI_THRESHOLD})
- 止损机制: 固定比例止损({'启用' if Config.USE_FIXED_STOP_LOSS else '禁用'}), 当日跌幅止损({'启用' if Config.USE_PCT_STOP_LOSS else '禁用'}), ATR动下止损({'启用' if Config.USE_ATR_STOP_LOSS else '禁用'})
"""
    log.info(infos)


def initialize(context):                # 初始化策略
    set_option("avoid_future_data", Config.AVOID_FUTURE_DATA)       # 避免未来函数
    set_option("use_real_price", Config.USE_REAL_PRICE)          # 使用真实价格

    set_slippage(PriceRelatedSlippage(0.0001), type="fund")  # 设置滑点

    set_order_cost(OrderCost(open_tax=0, close_tax=0, open_commission=0.0001, close_commission=0.0001, close_today_commission=0.0001, min_commission=5,), type="fund")  # 设置交易费用

    log.set_level('order', 'error')     # 降低日志级别
    log.set_level('system', 'error')
    log.set_level('strategy', 'info')

    set_benchmark(Config.BENCHMARK)        # 设置基准

    # 从 Config 类初始全局状态 g
    # 全局变量初始化
    g.positions = {}                    # 记录目标持仓
    g.position_highs = {}               # 记录持仓最高价（用于ATR跟踪）
    g.position_stop_prices = {}         # 记录ATR止损价
    g.target_etfs_list = []             # 今日目标ETF列表
    g.cooldown_end_date = None          # 冷却期结束日期
    g.etf_name_memory = {}              # ETF名称缓存
    g.etf_info_cache = {}               # ETF基本信息缓存 (上市日期等)
    g.atr_daily_cache = {}              # ATR每日缓存 (避免分时重复计算)
    g.etf_pool = []                     # 动态ETF池
    g.pool_refresh_count = 0            # 动态池刷新计数

    run_daily(check_pool_refresh, time='09:00')       # 刷新动态ETF池
    run_daily(check_positions, time='09:10')          # 盘前检查持仓
    run_daily(etf_sell_trade, time=EXECUTION_BEG_TIME_PLACEHOLDER)         # 卖出交易
    run_daily(etf_buy_trade, time=EXECUTION_END_TIME_PLACEHOLDER)          # 买入交易

    for tm in Config.DAILY_TIME_STOP_LOSS:
        run_daily(minute_level_stop_loss, time=tm)          # 固定比例止损
        run_daily(minute_level_pct_stop_loss, time=tm)      # 当日跌幅止损
        run_daily(minute_level_atr_stop_loss, time=tm)      # ATR动态止损

    show_params_log()


# --- 动态池逻辑 ---
def check_pool_refresh(context):
    if not g.etf_pool or g.pool_refresh_count <= 0:
        log.info("--- 开始刷新动态ETF池 ---")
        g.etf_pool = get_dynamic_etf_pool(context)
        log.info(f"--- 动态池刷新完成，当前成分数量: {len(g.etf_pool)} ---")
        if g.etf_pool:
            pool_details = [f"[{c}] {get_security_info(c).display_name}" for c in g.etf_pool]
            for i in range(0, len(pool_details), 3):
                log.info("  |  ".join(pool_details[i:i + 3]))
        g.pool_refresh_count = Config.POOL_REFRESH_INTERVAL
    else:
        g.pool_refresh_count -= 1


def get_dynamic_etf_pool(context):
    current_date = context.current_dt.date()

    # Step 1: 全市场扫描 + 关键词预过滤
    all_etfs = get_all_securities(types=['etf'])

    exclude_keywords = []
    if Config.POOL_FILTER_BOND_MONEY:
        exclude_keywords += ["债", "货币", "理财"]
    if Config.POOL_FILTER_QDII:
        exclude_keywords += ["QDII", "标普", "纳指", "道琼斯", "恒生", "H股", "日经",
                             "德国", "法国", "英国", "美国", "海外"]
    if Config.POOL_FILTER_A_SHARE:
        exclude_keywords += ["300", "500", "50", "800", "1000", "2000",
                             "中证", "创业板", "科创", "沪深", "上证", "深证", "A股"]
    if exclude_keywords:
        pattern = "|".join(exclude_keywords)
        all_etfs = all_etfs[~all_etfs['display_name'].str.contains(pattern, regex=True)]

    # Step 2: 上市年龄过滤
    min_date = current_date - timedelta(days=Config.POOL_MIN_LISTED_DAYS)
    candidates = all_etfs[all_etfs['start_date'] < min_date].index.tolist()
    if not candidates:
        return []

    # Step 3: 流动性过滤
    last_trading_day = get_trade_days(end_date=current_date, count=2)[0]
    prices = get_price(
        candidates,
        count=Config.POOL_VOL_LOOKBACK,
        end_date=last_trading_day,
        frequency='daily',
        fields=['money'],
        panel=False
    )
    if prices is None or prices.empty:
        return []

    avg_money = prices.groupby('code')['money'].mean()
    liquid_etfs = avg_money[avg_money >= Config.POOL_MIN_AVG_MONEY].index.tolist()

    if Config.POOL_FILTER_BOND_MONEY and liquid_etfs:
        hist_avg = get_price(
            liquid_etfs,
            count=1,
            end_date=last_trading_day,
            frequency='daily',
            fields=['avg'],
            panel=False
        )
        if hist_avg is not None and not hist_avg.empty:
            last_avg = hist_avg.groupby('code')['avg'].last()
            bond_etfs = last_avg[last_avg > 90].index.tolist()
            liquid_etfs = [e for e in liquid_etfs if e not in bond_etfs]

    if not liquid_etfs:
        return []

    # Step 4: 指数级去重与精准过滤
    final_pool = _dedup_by_index(liquid_etfs, avg_money, current_date)

    for wl in Config.ETF_WHITELIST:
        if wl in liquid_etfs and wl not in final_pool:
            final_pool.append(wl)

    return sorted(list(set(final_pool)))


def _dedup_by_index(liquid_etfs, avg_money, current_date):
    try:
        q = query(finance.FUND_INVEST_TARGET).filter(
            finance.FUND_INVEST_TARGET.code.in_(liquid_etfs)
        )
        df_target = finance.run_query(q)
    except Exception as e:
        log.warning(f"无法查询指数信息: {e}，将返回全部流动性池")
        return liquid_etfs

    if Config.POOL_FILTER_A_SHARE and not df_target.empty:
        mask = df_target['traced_index_code'].str.match(Config.A_SHARE_INDEX_REGEX, na=False)
        a_codes = df_target.loc[mask, 'code'].unique().tolist()
        if a_codes:
            liquid_etfs = [c for c in liquid_etfs if c not in a_codes]
            df_target = df_target[~df_target['code'].isin(a_codes)]

    if df_target.empty:
        return liquid_etfs

    df_target['start_date'] = pd.to_datetime(df_target['start_date'])
    df_target = df_target[df_target['start_date'].dt.date <= current_date]
    current_index_map = (
        df_target.sort_values('start_date', ascending=False)
        .drop_duplicates('code', keep='first')
    )

    current_index_map['group_key'] = (
        current_index_map['traced_index_code']
        .replace('', None)
        .fillna(current_index_map['traced_index_name'])
    )
    current_index_map['avg_money'] = current_index_map['code'].map(avg_money)

    valid_grouped = current_index_map[current_index_map['group_key'].notna()]
    no_index_etfs = [c for c in liquid_etfs if c not in valid_grouped['code'].tolist()]

    best_etfs = (
        valid_grouped.sort_values('avg_money', ascending=False)
        .drop_duplicates('group_key', keep='first')
    )
    return best_etfs['code'].tolist() + no_index_etfs


# --- 指标计算与筛选 ---
def get_security_name(security):
    try:
        if security in g.etf_name_memory:
            return g.etf_name_memory[security]
        name = get_current_data()[security].name
        g.etf_name_memory[security] = name
        return name
    except:
        return security


def calculate_all_metrics_for_etf(context, etf, pre_prices=None, pre_current_data=None, pre_today_vol=None):  # 计算单个ETF的所有指标
    try:
        etf_name = get_security_name(etf)         # 获取ETF名称

        current_data = pre_current_data if pre_current_data is not None else get_current_data()

        if pre_prices is not None:
            prices_df = pre_prices
        else:
            lookback = max(                           # 确定所需历史数据长度
                Config.LOOKBACK_DAYS,
                Config.SHORT_LOOKBACK_DAYS,
                Config.RSI_PERIOD + Config.RSI_LOOKBACK_DAYS,
                Config.MA_FILTER_DAYS,
                Config.VOLUME_LOOKBACK
            ) + 20
            prices_df = attribute_history(etf, lookback, '1d', ['close', 'high', 'low', 'volume'])  # 获取历史价格

        if len(prices_df) < max(Config.LOOKBACK_DAYS, Config.MA_FILTER_DAYS):
            return None

        current_price = current_data[etf].last_price
        price_series = np.append(prices_df["close"].values, current_price)  # 拼接当前价格

        # --- 动量指标计算 ---
        recent_price_series = price_series[-(Config.LOOKBACK_DAYS + 1):]
        y = np.log(recent_price_series)
        x = np.arange(len(y))
        weights = np.linspace(1, 2, len(y))
        slope, intercept = np.polyfit(x, y, 1, w=weights)
        annualized_returns = math.exp(slope * 250) - 1
        ss_res = np.sum(weights * (y - (slope * x + intercept)) ** 2)
        ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot else 0
        momentum_score = annualized_returns * r_squared

        # --- 短期动量和均线 ---
        if len(price_series) >= Config.SHORT_LOOKBACK_DAYS + 1:
            short_return = price_series[-1] / price_series[-(Config.SHORT_LOOKBACK_DAYS + 1)] - 1
            short_annualized = (1 + short_return) ** (250 / Config.SHORT_LOOKBACK_DAYS) - 1
        else:
            short_annualized = -np.inf

        ma_price = np.mean(price_series[-Config.MA_FILTER_DAYS:])
        current_above_ma = current_price >= ma_price

        # --- 成交量比指标 ---
        if pre_prices is not None and pre_today_vol is not None:
            # 批量模式优化：直接计算，不调用 API
            hist_v = prices_df["volume"].iloc[-Config.VOLUME_LOOKBACK:]
            avg_v = hist_v.mean()
            volume_ratio = pre_today_vol / avg_v if avg_v > 0 else 0
        else:
            volume_ratio = get_volume_ratio(context, etf, show_detail_log=False)

        # --- 其他过滤指标 ---
        day_ratios = []
        passed_loss_filter = True
        if len(price_series) >= 4:
            day1, day2, day3 = price_series[-1] / price_series[-2], price_series[-2] / price_series[-3], price_series[-3] / price_series[-4]
            day_ratios = [day1, day2, day3]
            if min(day_ratios) < Config.LOSS:
                passed_loss_filter = False

        current_rsi = 0
        passed_rsi_filter = True
        if Config.USE_RSI_FILTER and len(price_series) >= Config.RSI_PERIOD + Config.RSI_LOOKBACK_DAYS:
            rsi_values = calculate_rsi(price_series, Config.RSI_PERIOD)
            if len(rsi_values) >= Config.RSI_LOOKBACK_DAYS:
                recent_rsi = rsi_values[-Config.RSI_LOOKBACK_DAYS:]
                current_rsi = recent_rsi[-1]
                if np.any(recent_rsi > Config.RSI_THRESHOLD):
                    ma5 = np.mean(price_series[-5:]) if len(price_series) >= 5 else current_price
                    if current_price < ma5:
                        passed_rsi_filter = False

        return {
            'etf': etf, 'etf_name': etf_name, 'momentum_score': momentum_score,
            'annualized_returns': annualized_returns, 'r_squared': r_squared,
            'short_annualized': short_annualized, 'current_price': current_price,
            'ma_price': ma_price, 'volume_ratio': volume_ratio, 'day_ratios': day_ratios,
            'current_rsi': current_rsi, 'passed_momentum': Config.MIN_SCORE_THRESHOLD <= momentum_score <= Config.MAX_SCORE_THRESHOLD,
            'passed_short_mom': short_annualized >= Config.SHORT_MOMENTUM_THRESHOLD,
            'passed_r2': r_squared > Config.R2_THRESHOLD, 'passed_annual_ret': annualized_returns >= Config.MIN_ANNUAL_RETURN,
            'passed_ma': current_above_ma, 'passed_volume': volume_ratio is not None and volume_ratio < Config.VOLUME_THRESHOLD,
            'passed_loss': passed_loss_filter, 'passed_rsi': passed_rsi_filter,
        }
    except Exception as e:
        log.warning(f"计算 {etf} 指标出错: {e}")
        return None


def apply_filters(metrics_list):        # 应用所有过滤条件
    steps = [
        ('动量得分', lambda m: m['passed_momentum'], True),
        ('短期动量', lambda m: m['passed_short_mom'], Config.USE_SHORT_MOMENTUM_FILTER),
        ('R²', lambda m: m['passed_r2'], Config.ENABLE_R2_FILTER),
        ('年化收益率', lambda m: m['passed_annual_ret'], Config.ENABLE_ANNUAL_RETURN_FILTER),
        ('均线', lambda m: m['passed_ma'], Config.ENABLE_MA_FILTER),
        ('成交量', lambda m: m['passed_volume'], Config.ENABLE_VOLUME_CHECK),
        ('短期风控', lambda m: m['passed_loss'], Config.ENABLE_LOSS_FILTER),
        ('RSI', lambda m: m['passed_rsi'], Config.USE_RSI_FILTER),
    ]

    filtered = metrics_list[:]
    for name, condition, is_enabled in steps:  # 逐个应用启用的过滤器
        if is_enabled:
            filtered = [m for m in filtered if condition(m)]
    return filtered


def get_final_ranked_etfs(context):     # 主筛选函数：列表预热、批量取数、计算、排序
    all_metrics = []

    if not g.etf_pool:
        g.etf_pool = get_dynamic_etf_pool(context)

    etf_set = sorted(list(set(g.etf_pool)))
    end_date = context.previous_date    # 昨日日期

    if not etf_set:
        log.info("动态ETF池为空")
        return []

    # 1. 预读全局数据，减少循环内调用
    curr_data_obj = get_current_data()
    try:
        h = get_price(etf_set, count=1, end_date=end_date, frequency='daily', fields=['money'])
        yesterday_money = h['money'].iloc[0]
    except Exception:
        yesterday_money = pd.Series(dtype=float)

    # 2. 第一轮过滤：初步剔除停牌、无量及未上市标的
    valid_etfs = []
    for etf in etf_set:
        # 2.1 剔除停牌
        if curr_data_obj[etf].paused:
            continue

        # 2.2 剔除昨日无交易数据的标的
        if pd.isna(yesterday_money.get(etf, np.nan)):
            continue

        # 2.3 检查上市日期（利用缓存减少调用）
        try:
            if etf in g.etf_info_cache:
                start_date_raw = g.etf_info_cache[etf]
            else:
                info = get_security_info(etf)
                start_date_raw = info.start_date if info else None
                g.etf_info_cache[etf] = start_date_raw

            # 统一日期格式
            start_date = start_date_raw.date() if isinstance(start_date_raw, datetime) else start_date_raw
            if not isinstance(start_date, date) or end_date < start_date:
                continue
        except Exception:
            continue

        valid_etfs.append(etf)

    if not valid_etfs:
        log.info("今日无符合基础条件（上市/交易/停牌）的候选ETF")
        return []

    # 3. 批量获取有效标的历史行情（不再拉取全量池数据）
    lookback = max(
        Config.LOOKBACK_DAYS,
        Config.SHORT_LOOKBACK_DAYS,
        Config.RSI_PERIOD + Config.RSI_LOOKBACK_DAYS,
        Config.MA_FILTER_DAYS,
        Config.VOLUME_LOOKBACK
    ) + 20

    # 3.1 批量日线数据
    bulk_prices = get_price(valid_etfs, count=lookback, end_date=end_date, frequency='daily', fields=['close', 'high', 'low', 'volume'])

    # 3.2 批量1分钟成交量数据（用于今日至今累计量）
    today_v_stats = get_price(valid_etfs, start_date=context.current_dt.date(), end_date=context.current_dt, frequency='1m', fields=['volume'])

    # 处理 1m 累计成交量之和
    if hasattr(today_v_stats, 'major_axis'):  # Panel
        today_v_sums = today_v_stats['volume'].sum()
    elif isinstance(getattr(today_v_stats, 'index', None), pd.MultiIndex):  # MultiIndex DF
        today_v_sums = today_v_stats['volume'].groupby(level=1).sum()
    else:  # Simple DF or Empty
        today_v_sums = today_v_stats['volume'].sum() if 'volume' in today_v_stats else pd.Series()

    # 4. 第二轮循环：计算具体指标
    for etf in valid_etfs:
        # 从批量数据中提取对应标的的片段
        if hasattr(bulk_prices, 'major_axis'):
            ticker_prices = bulk_prices.loc[:, :, etf]
        elif isinstance(getattr(bulk_prices, 'index', None), pd.MultiIndex):
            ticker_prices = bulk_prices.xs(etf, axis=0, level=1)
        else:
            ticker_prices = bulk_prices[etf] if etf in bulk_prices else None

        metrics = calculate_all_metrics_for_etf(
            context,
            etf,
            pre_prices=ticker_prices,
            pre_current_data=curr_data_obj,
            pre_today_vol=today_v_sums.get(etf, 0)
        )
        if metrics:
            all_metrics.append(metrics)

    # 5. 指标处理、排序与过滤
    for item in all_metrics:
        score = item.get('momentum_score')
        if pd.isna(score) or (isinstance(score, float) and np.isnan(score)):
            item['momentum_score'] = float('-inf')

    # 初步排序
    all_metrics.sort(key=lambda x: x.get('momentum_score', float('-inf')), reverse=True)
    # 应用详细过滤器
    final_list = apply_filters(all_metrics)
    # 再次排序并返回
    final_list.sort(key=lambda x: x.get('momentum_score', float('-inf')), reverse=True)

    if final_list:
        top_names = [f"{m['etf_name']}({m['etf']}):{m['momentum_score']:.4f}" for m in final_list[:5]]
        log.info(f"符合条件的候选ETF(前5): {top_names}")
    else:
        log.info("今日无符合过滤条件的动量ETF")

    return final_list


# --- 止损风控 ---
def minute_level_stop_loss(context):    # 分钟级固定比例止损
    if not Config.USE_FIXED_STOP_LOSS:
        return
    if is_in_cooldown(context):
        return

    pos_keys = list(context.portfolio.positions.keys())
    current_data = get_current_data()
    for security in pos_keys:
        position = context.portfolio.positions[security]
        if position.total_amount <= 0:
            continue
        if security not in current_data:
            try:
                _ = current_data[security]
            except:
                continue
        current_price = current_data[security].last_price
        cost_price = position.avg_cost
        stop_line = cost_price * Config.FIXED_STOP_LOSS_THRESHOLD
        if current_price <= 0 or cost_price <= 0:
            continue
        if current_price <= stop_line:
            security_name = get_security_name(security)
            log.info(f"🚨 [分钟级] 固定比例止损卖出, 时刻:{context.current_dt}, ETF:{security}-{security_name}, 损失:{100 * (current_price - cost_price) / cost_price:.2f}%")
            success = smart_order_target_value(security, 0, context)
            if success:
                g.position_highs.pop(security, None)
                g.position_stop_prices.pop(security, None)
                enter_safe_haven_and_set_cooldown(context, trigger_reason="分钟级固定止损")


def minute_level_pct_stop_loss(context):  # 分钟级当日跌幅止损
    if not Config.USE_PCT_STOP_LOSS:
        return
    if is_in_cooldown(context):
        return

    current_data = get_current_data()
    for security in list(context.portfolio.positions.keys()):
        position = context.portfolio.positions[security]
        if position.total_amount <= 0:
            continue
        if security not in current_data:
            try:
                _ = current_data[security]
            except:
                continue
        today_open = current_data[security].day_open
        if not today_open or today_open <= 0:
            continue
        current_price = current_data[security].last_price
        if current_price <= 0:
            continue
        if current_price <= today_open * Config.PCT_STOP_LOSS_THRESHOLD:
            security_name = get_security_name(security)
            log.info(f"🚨 [分钟级] 当日跌幅止损卖出: {security} {security_name}")
            success = smart_order_target_value(security, 0, context)
            if success:
                g.position_highs.pop(security, None)
                g.position_stop_prices.pop(security, None)
                enter_safe_haven_and_set_cooldown(context, trigger_reason="分钟级当日跌幅止损")


def minute_level_atr_stop_loss(context):  # 分钟级ATR动态止损
    if not Config.USE_ATR_STOP_LOSS:
        return
    if is_in_cooldown(context):
        return

    current_data = get_current_data()
    for security in list(context.portfolio.positions.keys()):
        position = context.portfolio.positions[security]
        if position.total_amount <= 0:
            continue
        if Config.ATR_EXCLUDE_DEFENSIVE and security == Config.DEFENSIVE_ETF:
            continue
        try:
            if security not in current_data:
                try:
                    _ = current_data[security]
                except:
                    continue
            current_price = current_data[security].last_price
            if current_price <= 0:
                continue
            cost_price = position.avg_cost
            if cost_price <= 0:
                continue
            current_atr, _, success, _ = calculate_atr(security, Config.ATR_PERIOD, context)
            if not success or current_atr <= 0:
                continue
            if security not in g.position_highs:
                g.position_highs[security] = current_price
            else:
                g.position_highs[security] = max(g.position_highs[security], current_price)
            if Config.ATR_TRAILING_STOP:
                atr_stop_price = g.position_highs[security] - Config.ATR_MULTIPLIER * current_atr
            else:
                atr_stop_price = cost_price - Config.ATR_MULTIPLIER * current_atr
            g.position_stop_prices[security] = atr_stop_price
            if current_price <= atr_stop_price:
                success = smart_order_target_value(security, 0, context)
                if success:
                    g.position_highs.pop(security, None)
                    g.position_stop_prices.pop(security, None)
                    enter_safe_haven_and_set_cooldown(context, trigger_reason="分钟级ATR动态止损")
        except Exception:
            pass


# --- 交易执行 ---
def smart_order_target_value(security, target_value, context):  # 智能下单函数
    current_data = get_current_data()
    security_name = get_security_name(security)
    if current_data[security].paused:
        return False
    if current_data[security].last_price >= current_data[security].high_limit:
        return False
    if current_data[security].last_price <= current_data[security].low_limit:
        return False
    current_price = current_data[security].last_price
    if current_price <= 0:
        return False

    target_amount = int(target_value / current_price)
    target_amount = (target_amount // 100) * 100
    if target_amount <= 0 and target_value > 0:
        target_amount = 100

    current_position = context.portfolio.positions.get(security, None)
    current_amount = current_position.total_amount if current_position else 0
    amount_diff = target_amount - current_amount
    trade_value = abs(amount_diff) * current_price

    if 0 < trade_value < Config.MIN_MONEY:
        return False

    if amount_diff < 0:
        closeable = current_position.closeable_amount if current_position else 0
        if closeable == 0:
            return False
        amount_diff = -min(abs(amount_diff), closeable)

    if amount_diff != 0:
        order_result = order(security, amount_diff)
        if order_result:
            if amount_diff > 0:
                log.info(f"📦 买入 {security_name}({security})，金额: {amount_diff * current_price:.2f}")
                if security in g.etf_pool:
                    g.position_highs[security] = current_price
            else:
                log.info(f"📤 卖出 {security_name}({security})，金额: {abs(amount_diff) * current_price:.2f}")
            return True
    return False


def etf_sell_trade(context):            # 卖出交易主逻辑
    if is_in_cooldown(context):
        return
    ranked_etfs = get_final_ranked_etfs(context)
    target_etfs = [m['etf'] for m in ranked_etfs[:Config.HOLDINGS_NUM]] if ranked_etfs else []

    if not target_etfs:
        if check_defensive_etf_available(context):
            target_etfs = [Config.DEFENSIVE_ETF]
            log.info(f"🛡️ 防御模式开启: {Config.DEFENSIVE_ETF}")

    g.target_etfs_list = target_etfs
    target_set = set(target_etfs)

    for security in list(context.portfolio.positions.keys()):
        if security not in target_set:
            smart_order_target_value(security, 0, context)


def etf_buy_trade(context):             # 买入交易主逻辑
    exit_safe_haven_if_cooldown_ends(context)
    if is_in_cooldown(context):
        return

    target_etfs = g.target_etfs_list
    if not target_etfs:
        return

    etfs_to_buy = [e for e in target_etfs if e not in context.portfolio.positions or context.portfolio.positions[e].total_amount == 0]
    if not etfs_to_buy:
        return

    available_cash = context.portfolio.available_cash
    per_value = available_cash // len(etfs_to_buy)

    for etf in etfs_to_buy:
        smart_order_target_value(etf, per_value, context)


# --- 工具函数与底层支撑 ---
def check_positions(context):
    g.atr_daily_cache = {}  # 每日清空 ATR 缓存
    current_data = get_current_data()
    for security in context.portfolio.positions:
        pos = context.portfolio.positions[security]
        if pos.total_amount > 0:
            _ = current_data[security]
            log.info('=' * 85)
            log.info(f'📊 持仓检查: {get_security_name(security)}({security}), 数量: {pos.total_amount}')


def check_defensive_etf_available(context):
    d = Config.DEFENSIVE_ETF
    info = get_current_data()[d]
    return not info.paused and info.last_price < info.high_limit and info.last_price > info.low_limit


def get_volume_ratio(context, security, lookback_days=None, threshold=None, show_detail_log=True):
    lookback_days = lookback_days or Config.VOLUME_LOOKBACK
    try:
        hist = attribute_history(security, lookback_days, '1d', ['volume'])
        if len(hist) < lookback_days:
            return None
        avg_v = hist['volume'].mean()
        if avg_v == 0:
            return None
        cur_v = get_price(security, start_date=context.current_dt.date(), end_date=context.current_dt, frequency='1m', fields=['volume'])['volume'].sum()
        return cur_v / avg_v
    except:
        return None


def calculate_rsi(prices, period=6):
    if len(prices) < period + 1:
        return np.array([])
    deltas = np.diff(prices)
    alpha = 2.0 / (period + 1)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_g = np.zeros(len(deltas))
    avg_l = np.zeros(len(deltas))
    avg_g[period - 1] = np.mean(gains[:period])
    avg_l[period - 1] = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_g[i] = gains[i] * alpha + avg_g[i - 1] * (1 - alpha)
        avg_l[i] = losses[i] * alpha + avg_l[i - 1] * (1 - alpha)
    rsi = 100 - (100 / (1 + avg_g / (avg_l + 1e-9)))
    return np.append(np.full(period, np.nan), rsi)


def calculate_atr(security, period=14, context=None):
    # 尝试从今日缓存获取
    if security in g.atr_daily_cache:
        cached_date, cached_atr = g.atr_daily_cache[security]
        if context and context.current_dt.date() == cached_date:
            return cached_atr, [], True, "缓存成功"

    try:
        hist = attribute_history(security, period + 5, '1d', ['high', 'low', 'close'])
        if len(hist) < period + 1:
            return 0, [], False, "数据不足"
        h, l, c = hist['high'].values, hist['low'].values, hist['close'].values
        tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
        atr = np.mean(tr[-period:])

        # 存入缓存
        if context:
            g.atr_daily_cache[security] = (context.current_dt.date(), atr)

        return atr, tr, True, "成功"
    except:
        return 0, [], False, "失败"


def is_in_cooldown(context):
    return Config.SELL_COOLDOWN_ENABLED and g.cooldown_end_date and context.current_dt.date() <= g.cooldown_end_date


def set_cooldown(context):
    if Config.SELL_COOLDOWN_ENABLED:
        g.cooldown_end_date = context.current_dt.date() + pd.Timedelta(days=Config.SELL_COOLDOWN_DAYS)
        log.info(f"🔒 冷却期开启至: {g.cooldown_end_date}")


def enter_safe_haven_and_set_cooldown(context, trigger_reason=""):
    if not Config.SELL_COOLDOWN_ENABLED:
        return
    for s in list(context.portfolio.positions.keys()):
        smart_order_target_value(s, 0, context)
        g.position_highs.pop(s, None)
    v = context.portfolio.total_value
    if v > Config.MIN_MONEY:
        smart_order_target_value(Config.SAFE_HAVEN_ETF, v * 0.99, context)
    set_cooldown(context)


def exit_safe_haven_if_cooldown_ends(context):
    if Config.SELL_COOLDOWN_ENABLED and g.cooldown_end_date and context.current_dt.date() > g.cooldown_end_date:
        if Config.SAFE_HAVEN_ETF in context.portfolio.positions:
            smart_order_target_value(Config.SAFE_HAVEN_ETF, 0, context)
        g.cooldown_end_date = None