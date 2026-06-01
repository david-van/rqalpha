# -*- coding: utf-8 -*-
"""
S4 博收益ETF轮动策略 v5.3
===========================
三态市场判定（正常/震荡/走弱）× 自适应持仓数 × 多道动量过滤。
走弱期切换全球/海外ETF池，正常/震荡期使用合并池（固定精选+动态行业Top100）。

核心特点:
- 三态市场: 六指数MA10/MA20判定，带确认机制防"一日游"
- 动量打分: 加权对数回归 × R²
- 多道过滤: R²/MA/成交量/连跌/溢价/拉普拉斯/高斯
- 防频换: 持仓连续N天不在Top-K才换
- 组合回撤分级: 3%预警 / 10%减半 / 12%切防御 / 20%清仓

资金: 40000元 (pindex=3)
"""
from jqdata import *
from functools import wraps
import numpy as np
import pandas as pd
import datetime
import math
import re


# ==================== 日志 ====================
def _slog(sid, module, msg, level='info'):
    prefix = f"[{sid}] {module}"
    text = f"{prefix} | {msg}"
    getattr(log, level)(text)


def _slog_global(module, msg, level='info'):
    text = f"[全局] {module} | {msg}"
    getattr(log, level)(text)


# ==================== 证券名称 ====================
def get_security_name(security):
    try:
        info = get_security_info(security)
        return info.display_name if hasattr(info, 'display_name') else info.name
    except Exception:
        return str(security)


# ==================== ETF池定义 ====================
GLOBAL_ETF_POOL = [
    # 大宗商品ETF
    '518880.XSHG',  # 黄金ETF
    '501018.XSHG',  # 南方原油
    '161226.XSHE',  # 国投白银LOF
    '159985.XSHE',  # 豆粕ETF
    '159980.XSHE',  # 有色ETF
    # 海外ETF
    '513310.XSHG',  # 中韩芯片
    '159518.XSHE',  # 标普油气ETF
    '159509.XSHE',  # 纳指科技ETF
    '513100.XSHG',  # 纳指ETF
    '513520.XSHG',  # 日经ETF
    '513500.XSHG',  # 标普500
    '159502.XSHE',  # 标普生物科技ETF
    '513400.XSHG',  # 道琼斯
    '513030.XSHG',  # 德国ETF
    '513290.XSHG',  # 纳指生物
    '520830.XSHG',  # 沙特ETF
    '159529.XSHE',  # 标普消费ETF
    '164824.XSHE',  # 印度基金LOF
    "513080.XSHG",  # 法国ETF
    "513730.XSHG",  # 东南亚ETF
    "511010.XSHG",  # 国债ETF
    "511220.XSHG",  # 城投债ETF
]

CHINA_ETF_POOL = [
    # 港股ETF
    '513090.XSHG', '513120.XSHG', '513180.XSHG', '513330.XSHG', '513750.XSHG',
    '159892.XSHE', '513190.XSHG', '159605.XSHE', '513630.XSHG', '159323.XSHE',
    '510900.XSHG', '513920.XSHG', '513970.XSHG',
    # 指数ETF
    '511380.XSHG', '512050.XSHG', '510500.XSHG', '159915.XSHE', '510300.XSHG',
    '512100.XSHG', '159949.XSHE', '588080.XSHG', '159967.XSHE', '588220.XSHG',
    '563300.XSHG', '510760.XSHG',
    # 行业ETF
    '588200.XSHG', '515880.XSHG', '159981.XSHE', '512880.XSHG', '513350.XSHG',
    '159326.XSHE', '159516.XSHE', '159206.XSHE', '512480.XSHG', '159363.XSHE',
    '159870.XSHE', '512400.XSHG', '159755.XSHE', '588170.XSHG', '159992.XSHE',
    '159995.XSHE', '512890.XSHG', '515220.XSHG', '159566.XSHE', '159819.XSHE',
    '512800.XSHG', '512690.XSHG', '515050.XSHG', '562500.XSHG', '512170.XSHG',
    '517520.XSHG', '159869.XSHE', '512070.XSHG', '159611.XSHE', '562800.XSHG',
    '515120.XSHG', '512010.XSHG', '510880.XSHG', '515790.XSHG', '515980.XSHG',
    '512660.XSHG', '159928.XSHE', '512710.XSHG', '560860.XSHG', '515030.XSHG',
    '159766.XSHE', '159218.XSHE', '159852.XSHE', '516160.XSHG', '516150.XSHG',
    '159227.XSHE', '159583.XSHE', '588790.XSHG', '159865.XSHE', '512980.XSHG',
    '159851.XSHE', '561360.XSHG', '561980.XSHG', '562590.XSHG', '512200.XSHG',
    '159732.XSHE', '159667.XSHE', '516510.XSHG', '159840.XSHE', '159998.XSHE',
    '159825.XSHE', '512670.XSHG', '159883.XSHE', '515210.XSHG', '515400.XSHG',
    '159256.XSHE', '561330.XSHG', '515170.XSHG', '159638.XSHE', '516520.XSHG',
    '513360.XSHG', '516190.XSHG',
]

FIXED_ETF_POOL = list(dict.fromkeys(GLOBAL_ETF_POOL + CHINA_ETF_POOL))


# ==================== 工具函数 ====================
def _get_bid1_sell_price(security, fallback_price):
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


def _coerce_scalar_price(x):
    if x is None:
        return None
    try:
        arr = np.asarray(x, dtype=float).ravel()
        if arr.size == 0:
            return None
        return float(arr[0])
    except Exception:
        return None


def _get_intraday_price_with_fallback(context, security):
    try:
        cd = get_current_data()
        if security in cd:
            cur = _coerce_scalar_price(cd[security].last_price)
            if cur is not None and cur > 0:
                return cur
    except Exception:
        pass
    try:
        dt = getattr(context, 'current_dt', None)
        dfm = get_price(security, end_date=dt, count=1, frequency='1m',
                        fields=['close'], panel=False, skip_paused=False)
        if dfm is not None and not dfm.empty:
            cur = _coerce_scalar_price(dfm['close'].iloc[-1] if 'close' in dfm.columns else dfm.iloc[-1, -1])
            if cur is not None and cur > 0:
                return cur
    except Exception:
        pass
    return None


def _boyishouyi_sp(context):
    return context.subportfolios[3]


# ==================== 数学工具 ====================
def boyishouyi_calculate_momentum_score(price_series, lookback_days):
    if len(price_series) < lookback_days + 1:
        return None, None, None
    recent_price_series = price_series[-(lookback_days + 1):]
    y = np.log(recent_price_series)
    x = np.arange(len(y))
    weights = np.linspace(1, 2, len(y))
    W = weights ** 2
    W_sum = np.sum(W)
    x_bar = np.sum(W * x) / W_sum
    y_bar = np.sum(W * y) / W_sum
    dx = x - x_bar
    dy = y - y_bar
    variance_x = np.sum(W * dx**2)
    if variance_x == 0:
        return 0, 0, 0
    slope = np.sum(W * dx * dy) / variance_x
    intercept = y_bar - slope * x_bar
    annualized_returns = math.exp(slope * 250) - 1
    y_pred = slope * x + intercept
    ss_res = np.sum(weights * (y - y_pred) ** 2)
    ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot else 0
    momentum_score = annualized_returns * r_squared
    return momentum_score, annualized_returns, r_squared


def laplace_filter(price, s=0.05):
    alpha = 1 - np.exp(-s)
    L = np.zeros(len(price))
    L[0] = price[0]
    for t in range(1, len(price)):
        L[t] = alpha * price[t] + (1 - alpha) * L[t - 1]
    return L


def gaussian_filter_last_two(price, sigma=1.2):
    n = len(price)
    if n < 2:
        return 0.0, 0.0
    idx_1 = np.arange(n)
    weights_1 = np.exp(-((idx_1 + 1) ** 2) / (2 * sigma ** 2))[::-1]
    weights_1 /= np.sum(weights_1)
    g1 = np.sum(price * weights_1)
    price_2 = price[:-1]
    idx_2 = np.arange(n - 1)
    weights_2 = np.exp(-((idx_2 + 1) ** 2) / (2 * sigma ** 2))[::-1]
    weights_2 /= np.sum(weights_2)
    g2 = np.sum(price_2 * weights_2)
    return g1, g2


# ==================== S4 配置初始化 ====================
def s4_init_config(context):
    pindex = 3
    g.boyishouyi_pindex = pindex

    # ETF池
    g.global_etf_pool = GLOBAL_ETF_POOL[:]
    g.china_etf_pool = CHINA_ETF_POOL[:]
    g.fixed_etf_pool = FIXED_ETF_POOL[:]

    # 池状态
    g.avg_etf_money_threshold = None
    g.global_liquidity_threshold_divisor = 3000
    g.filtered_fixed_pool = []
    g.dynamic_etf_pool = []
    g.merged_etf_pool = []
    g.ranked_etfs_result = []
    g.filtered_global_pool = []

    # 三态市场
    g.market_regime = '震荡期'
    g.is_a_share_weak = False
    g.regime_prev_day = None
    g.regime_prev_prev_day = None
    g.regime_flip_flop_count = 0
    g.regime_switch_confirm_days = 2
    g.regime_switch_pending_raw = None
    g.regime_switch_pending_streak = 0
    g.regime_last_change_date = None
    g.normal_ma_lookback = 10
    g.regime_ma20_lookback = 20
    g.weak_period_ma_lookback = 10
    g.regime_weak_below_ma20_min = 4
    g.regime_normal_above_ma10_min = 4

    # 震荡期滤波器参数
    g.gaussian_sigma = 1.2
    g.gaussian_min_slope = 0.002
    g.gaussian_use_relative_slope = True
    g.gaussian_min_slope_relative = 0.001

    # 持仓数
    g.holdings_num = 1
    g.holdings_num_normal = 1
    g.holdings_num_oscillation = 1
    g.holdings_num_weak = 1

    # 核心参数
    g.defensive_etf = "511880.XSHG"
    g.min_money = 10
    g.lookback_days = 25
    g.min_score_threshold = 0
    g.max_score_threshold = 5
    g.score_threshold_ratio = 0.9
    g.short_momentum_lookback = 21
    g.short_momentum_min_score = 0
    g.short_momentum_max_score = 6

    # 过滤开关
    g.enable_r2_filter = True
    g.r2_threshold = 0.4
    g.enable_ma_filter = True
    g.ma_lookback = 10
    g.ma_threshold = 1.0001
    g.enable_volume_check = True
    g.volume_lookback = 5
    g.volume_threshold = 1.8
    g.enable_volume_threshold_buffer = True
    g.volume_threshold_buffer = 0.1
    g.enable_loss_filter = True
    g.loss = 0.97
    g.enable_premium_filter = False
    g.max_premium_rate = 30
    g.enable_laplace_filter = True
    g.laplace_s_param = 0.05
    g.enable_laplace_s_regime_differentiation = True
    g.laplace_s_param_normal = 0.05
    g.laplace_s_param_weak = 0.04
    g.laplace_min_slope = 0.0005

    # 震荡期特殊参数
    g.enable_smoothed_momentum_input = False
    g.smoothed_ma_window = 5
    g.smoothed_momentum_only_in_range = True
    g.enable_range_r2_veto = False
    g.r2_threshold_range_bound = 0.9
    g.enable_range_momentum_floor = False
    g.range_momentum_min = 0.0
    g.range_momentum_max = 2.0
    g.enable_range_short_momentum_limits = True
    g.range_short_momentum_min = 0.0
    g.range_short_momentum_max = 6.0
    g.enable_switch_hysteresis = False
    g.switch_buffer_normal = 0.10
    g.switch_buffer_range = 0.40
    g.enable_dual_positive_momentum = True
    g.dual_positive_only_in_range = True
    g.whipsaw_options_only_in_range = True
    g.log_whipsaw_filter_detail = True

    # P0-2 参数补丁
    g.enable_p0_state_tuning_patch = True
    g.normal_r2_threshold_override = 0.39
    g.normal_laplace_min_slope_override = 0.0024

    # 走弱期过滤开关
    g.weak_apply_r2_filter = False
    g.weak_apply_ma_filter = True
    g.weak_apply_volume_filter = False
    g.weak_apply_loss_filter = True
    g.weak_apply_premium_filter = True
    g.weak_apply_laplace_filter = True

    # 防频换
    g.normal_max_days_not_rank1 = 5
    g.normal_max_days_not_topk = 5
    g.oscillation_max_days_not_topk = 5
    g.normal_not_rank1_streak = 0
    g.normal_streak_hold_code = None
    g.normal_not_topk_streaks = {}
    g.oscillation_anti_churn_enabled = True
    g.weak_anti_churn_enabled = True

    # 日志开关
    g.log_trade_detail = True
    g.log_ranking_detail = True
    g.log_filter_detail = True
    g.log_pool_update_details = False
    g.log_first_step_ranking = False

    # 回撤监控
    g.max_portfolio_value = 0
    g.drawdown_threshold = 0.03
    g.drawdown_records = []
    g.enable_drawdown_risk_actions = True
    g.dd_half_position_threshold = 0.10
    g.dd_switch_defensive_threshold = 0.12
    g.dd_flat_threshold = 0.20
    g.dd_partial_close_keep_fraction = 0.50
    g.dd_action_cooldown_date = None
    g.dd_reset_peak_after_action = True
    g.dd_monitor_time = '10:31'
    g.dd_valuation_use_mtm_last_price = True

    # 止损
    g.use_fixed_stop_loss = True
    g.fixedStopLossThreshold = 0.92
    g.use_pct_stop_loss = False
    g.pct_stop_loss_threshold = 0.95
    g.enable_stop_loss_rebuy_cooldown = True
    g.stop_loss_rebuy_cooldown_trade_days = 2
    g.stop_loss_rebuy_cutoff_time = '13:10'
    g.stop_loss_rebuy_first_allowed_date = {}

    # 动量软上限
    g.enable_momentum_soft_cap = False
    g.momentum_soft_cap_penalty = 0.05
    g.momentum_soft_cap_normal_only = True

    # MA跌破统计
    g.enable_holdings_ma_break_nextday_stats = True
    g.ma_break_nextday_pending = []
    g.ma_break_nextday_stats = {}

    # 市场状态绩效
    g.regime_day_counts = {'正常期': 0, '震荡期': 0, '走弱期': 0}
    g.regime_return_factors = {'正常期': 1.0, '震荡期': 1.0, '走弱期': 1.0}
    g.regime_win_counts = {'正常期': 0, '震荡期': 0, '走弱期': 0}
    g.regime_loss_counts = {'正常期': 0, '震荡期': 0, '走弱期': 0}
    g.regime_flat_counts = {'正常期': 0, '震荡期': 0, '走弱期': 0}
    g.regime_sum_pos_daily_ret = {'正常期': 0.0, '震荡期': 0.0, '走弱期': 0.0}
    g.regime_sum_neg_daily_ret = {'正常期': 0.0, '震荡期': 0.0, '走弱期': 0.0}
    g.prev_eod_portfolio_value = None

    # 运行时状态
    g.target_etfs_list = []
    g.trade_entry_open = {}
    g.trade_roundtrip_history = []
    g.exit_reason_stats = {}
    g.pending_sm3up_sell_followups = []
    g.last_metrics_by_etf_code = {}
    g.etf_names_dict = {}
    g.cache_date = None
    g.yesterday_close_cache = {}
    g.boyishouyi_rebalance_cap = None
    g.log_market_status_details = False
    g.defensive_switch_confirm_days = 2
    g.defensive_switch_pending_streak = 0
    g.defensive_switch_last_signal_date = None

    # 预加载证券名称
    _compile_boyi_patterns()


def _compile_boyi_patterns():
    pass  # S4 独立运行时不需要动态池构建关键词


# ==================== 定时任务 ====================
def s4_setup_schedule():
    run_daily(boyishouyi_morning_routine, '09:00')
    run_daily(boyishouyi_drawdown_monitor_routine, '10:31')
    run_daily(boyishouyi_check_weak_period_daily, '09:40')
    run_daily(boyishouyi_midday_routine, '10:00')
    run_daily(boyishouyi_afternoon_sell, '13:10')
    run_daily(boyishouyi_afternoon_buy, '13:15')
    run_daily(boyishouyi_holdings_ma_break_1450_check, '14:50')
    run_daily(boyishouyi_reset_daily_flags, '15:10')
    run_daily(boyishouyi_after_close_regime_statistics, 'after_close')
    run_daily(boyishouyi_minute_stop_loss, 'every_bar')
    run_daily(boyishouyi_minute_pct_stop_loss, 'every_bar')
    run_daily(boyishouyi_record_daily_positions, '15:30')
    run_daily(boyishouyi_output_positions_summary, 'after_close')


# ==================== 市场状态判定 ====================
def boyishouyi_resolve_market_regime(context):
    indexes = {
        '沪深300': '000300.XSHG',
        '深证综指(399101)': '399101.XSHE',
        '创业板': '399006.XSHE',
        '中证A500': '000510.XSHG',
        '中证1000': '000852.XSHG',
        '国证2000(代中证2000)': '399303.XSHE',
    }
    n_index = len(indexes)
    bars_need = max(g.regime_ma20_lookback, g.normal_ma_lookback) + 1
    below_ma20, above_ma10, n_ok = 0, 0, 0

    for name, code in indexes.items():
        df = attribute_history(code, bars_need, '1d', ['close'], skip_paused=False)
        if df is None or len(df) < g.regime_ma20_lookback:
            continue
        cur = df['close'][-1]
        ma10 = df['close'][-g.normal_ma_lookback:].mean()
        ma20 = df['close'][-g.regime_ma20_lookback:].mean()
        if cur < ma20:
            below_ma20 += 1
        if cur > ma10:
            above_ma10 += 1
        n_ok += 1

    weak_min = int(getattr(g, 'regime_weak_below_ma20_min', 6))
    normal_min = int(getattr(g, 'regime_normal_above_ma10_min', 3))

    if n_ok < n_index:
        raw_regime = '震荡期'
    elif below_ma20 >= weak_min:
        raw_regime = '走弱期'
    elif above_ma10 >= normal_min:
        raw_regime = '正常期'
    else:
        raw_regime = '震荡期'

    today = context.current_dt.date()
    effective_before = getattr(g, 'market_regime', '震荡期')
    last_change = getattr(g, 'regime_last_change_date', None)
    n_need = int(getattr(g, 'regime_switch_confirm_days', 2))
    if n_need < 1:
        n_need = 1

    log.info(f"[S4] 市场指标: below_ma20={below_ma20}/{n_ok} above_ma10={above_ma10}/{n_ok} → {raw_regime}")

    if last_change is None or raw_regime == effective_before or n_need <= 1:
        g.market_regime = raw_regime
        g.regime_last_change_date = today
        g.is_a_share_weak = raw_regime == '走弱期'
        g.regime_switch_pending_raw = None
        g.regime_switch_pending_streak = 0
        if last_change is not None and raw_regime != effective_before:
            _slog_global('市场', f'切换: {effective_before}→{raw_regime}')
    else:
        pending = getattr(g, 'regime_switch_pending_raw', None)
        streak = int(getattr(g, 'regime_switch_pending_streak', 0))
        if pending == raw_regime:
            streak += 1
        else:
            streak = 1
        g.regime_switch_pending_streak = streak
        g.regime_switch_pending_raw = raw_regime

        if streak >= n_need:
            g.market_regime = raw_regime
            g.regime_last_change_date = today
            g.is_a_share_weak = raw_regime == '走弱期'
            g.regime_switch_pending_raw = None
            g.regime_switch_pending_streak = 0
            _slog_global('市场', f'确认切换: {effective_before}→{raw_regime} (连续{streak}日)')
        else:
            g.market_regime = effective_before
            g.is_a_share_weak = effective_before == '走弱期'

    # 隔日跳回检测
    prev = getattr(g, 'regime_prev_day', None)
    prev2 = getattr(g, 'regime_prev_prev_day', None)
    new_reg = g.market_regime
    if prev is not None and prev2 is not None and new_reg == prev2 and new_reg != prev:
        g.regime_flip_flop_count = int(getattr(g, 'regime_flip_flop_count', 0)) + 1
        log.info(f"[S4] 隔日跳回 #{g.regime_flip_flop_count}: {prev2}→{prev}→{new_reg}")
    g.regime_prev_prev_day = prev
    g.regime_prev_day = new_reg

    _slog_global('市场', f'生效: {g.market_regime}')
    return g.market_regime


# ==================== 晨间流水线 ====================
def boyishouyi_morning_routine(context):
    _slog_global('晨间', '流水线启动')
    boyishouyi_check_positions(context)
    boyishouyi_calculate_global_etf_threshold(context)
    _slog_global('晨间', '流水线完毕')


def boyishouyi_check_positions(context):
    current_data = get_current_data()
    for security in _boyishouyi_sp(context).positions:
        position = _boyishouyi_sp(context).positions[security]
        if position.total_amount > 0:
            security_name = get_security_name(security)
            _slog('S4', '持仓', f'{security} {security_name}, {position.total_amount}股 '
                               f'成本{position.avg_cost:.3f} 现价{position.price:.3f}')


# ==================== 流动性阈值计算 ====================
def boyishouyi_calculate_global_etf_threshold(context):
    try:
        df_etf = get_all_securities(['etf'], date=context.current_dt)
        etf_list = df_etf.index.tolist()
        if not etf_list:
            g.avg_etf_money_threshold = 10000000
            return
        trade_days = get_trade_days(end_date=context.previous_date, count=3)
        df = get_price(etf_list, start_date=trade_days[0], end_date=context.previous_date,
                       frequency='daily', fields=['money'], panel=False, skip_paused=True)
        if df is None or df.empty:
            g.avg_etf_money_threshold = 10000000
            return
        daily_totals = df.groupby('time')['money'].sum()
        if len(daily_totals) < 3:
            g.avg_etf_money_threshold = 10000000
            return
        avg_total_money = daily_totals.mean()
        div = float(getattr(g, 'global_liquidity_threshold_divisor', 3000))
        g.avg_etf_money_threshold = avg_total_money / max(div, 1)
        _slog('S4', '流动性', f'全市场ETF日均成交{avg_total_money/1e8:.2f}亿, 门槛{g.avg_etf_money_threshold/1e4:.0f}万')
    except Exception as e:
        _slog('S4', '流动性', f'计算异常: {e}', 'warning')
        g.avg_etf_money_threshold = 10000000


# ==================== 走弱期/早盘 ====================
def boyishouyi_check_weak_period_daily(context):
    boyishouyi_resolve_market_regime(context)
    boyishouyi_midday_routine(context)


def boyishouyi_drawdown_monitor_routine(context):
    boyishouyi_monitor_drawdown(context)


def boyishouyi_refresh_holdings_num_by_regime(context):
    regime = getattr(g, 'market_regime', '震荡期')
    prev = int(getattr(g, 'holdings_num', 1))
    target_map = {
        '正常期': int(getattr(g, 'holdings_num_normal', prev)),
        '震荡期': int(getattr(g, 'holdings_num_oscillation', prev)),
        '走弱期': int(getattr(g, 'holdings_num_weak', prev)),
    }
    g.holdings_num = max(1, target_map.get(regime, prev))
    _slog('S4', '持仓数', f'{regime}, holdings_num={g.holdings_num}')


def boyishouyi_midday_routine(context):
    _slog('S4', '早盘', '流水线启动')

    if g.market_regime == '走弱期':
        boyishouyi_filter_global_pool_by_volume(context)
        _slog('S4', '池更新', f'走弱期过滤后全球池: {len(g.filtered_global_pool)}只')
    else:
        boyishouyi_filter_fixed_pool_by_volume(context)
        boyishouyi_daily_merge_etf_pools(context)
        _slog('S4', '池更新', f'{g.market_regime}合并池: {len(g.merged_etf_pool)}只')

    _slog('S4', '早盘', '流水线完毕')


def boyishouyi_filter_global_pool_by_volume(context):
    if getattr(g, 'avg_etf_money_threshold', None) is None:
        boyishouyi_calculate_global_etf_threshold(context)
    if not g.global_etf_pool:
        g.filtered_global_pool = []
        return
    dynamic_threshold = g.avg_etf_money_threshold
    end_date = context.previous_date
    TRADE_DAYS_COUNT = 3
    try:
        price_data = get_price(g.global_etf_pool, end_date=end_date, count=TRADE_DAYS_COUNT,
                               frequency='daily', fields=['money'], panel=False)
        if price_data is None or price_data.empty:
            g.filtered_global_pool = g.global_etf_pool[:]
            return
        total_money = price_data.groupby('code')['money'].sum()
        avg_daily_money = total_money / TRADE_DAYS_COUNT
        qualified = avg_daily_money[avg_daily_money > dynamic_threshold]
        g.filtered_global_pool = qualified.index.tolist()
        _slog('S4', '全球池', f'过滤后: {len(g.filtered_global_pool)}只')
    except Exception as e:
        _slog('S4', '全球池', f'异常: {e}', 'warning')
        g.filtered_global_pool = g.global_etf_pool[:]


def boyishouyi_filter_fixed_pool_by_volume(context):
    if getattr(g, 'avg_etf_money_threshold', None) is None:
        boyishouyi_calculate_global_etf_threshold(context)
    if not g.fixed_etf_pool:
        g.filtered_fixed_pool = []
        return
    dynamic_threshold = g.avg_etf_money_threshold
    end_date = context.previous_date
    TRADE_DAYS_COUNT = 3
    try:
        price_data = get_price(g.fixed_etf_pool, end_date=end_date, count=TRADE_DAYS_COUNT,
                               frequency='daily', fields=['money'], panel=False)
        if price_data is None or price_data.empty:
            g.filtered_fixed_pool = g.fixed_etf_pool[:]
            return
        total_money = price_data.groupby('code')['money'].sum()
        avg_daily_money = total_money / TRADE_DAYS_COUNT
        qualified = avg_daily_money[avg_daily_money > dynamic_threshold]
        g.filtered_fixed_pool = qualified.index.tolist()
    except Exception as e:
        _slog('S4', '固定池', f'异常: {e}', 'warning')
        g.filtered_fixed_pool = g.fixed_etf_pool[:]


def boyishouyi_daily_merge_etf_pools(context):
    if not hasattr(g, 'filtered_fixed_pool'):
        g.filtered_fixed_pool = g.fixed_etf_pool[:]
    merged = list(set(g.filtered_fixed_pool + g.dynamic_etf_pool))
    merged.sort()
    g.merged_etf_pool = merged


# ==================== 动量计算和过滤 ====================
def boyishouyi_get_volume_ratio(hist_volumes, today_vol, context, lookback_days=None):
    if lookback_days is None:
        lookback_days = g.volume_lookback
    try:
        if hist_volumes is None or len(hist_volumes) < lookback_days:
            return None
        past_n_days_vol = hist_volumes[-lookback_days:]
        avg_volume = np.mean(past_n_days_vol)
        if avg_volume == 0:
            return None
        now = context.current_dt
        elapsed_minutes = (now.hour - 9) * 60 + now.minute - 30
        if now.hour >= 13:
            elapsed_minutes -= 90
        elapsed_minutes = max(1, min(elapsed_minutes, 240))
        projected_today_vol = today_vol * (240.0 / elapsed_minutes)
        return projected_today_vol / avg_volume if avg_volume > 0 else 0
    except Exception:
        return None


def boyishouyi_calculate_premium_rate(etf, context):
    try:
        etf_price = getattr(g, 'etf_yesterday_close_batch', {}).get(etf)
        if etf_price is None or pd.isna(etf_price):
            df = get_price(etf, start_date=context.previous_date, end_date=context.previous_date,
                          fields=['close'])
            if df is None or len(df) == 0:
                return None, False
            etf_price = df['close'].iloc[-1]
        nav = getattr(g, 'etf_yesterday_nav_batch', {}).get(etf)
        if nav is None or pd.isna(nav):
            nav_df = get_extras('unit_net_value', etf, start_date=context.previous_date,
                               end_date=context.previous_date)
            if nav_df is None or len(nav_df) == 0:
                return None, False
            nav = nav_df.iloc[-1].values[0]
        if nav <= 0 or pd.isna(nav):
            return None, False
        premium_rate = (etf_price - nav) / nav * 100
        return premium_rate, premium_rate <= g.max_premium_rate
    except Exception:
        return None, True


def _whipsaw_global_period_ok():
    if not getattr(g, 'whipsaw_options_only_in_range', True):
        return True
    return getattr(g, 'market_regime', '震荡期') == '震荡期'


def _effective_r2_threshold_whipsaw():
    base = float(getattr(g, 'r2_threshold', 0.4))
    if (getattr(g, 'enable_p0_state_tuning_patch', False)
            and getattr(g, 'market_regime', '震荡期') == '正常期'):
        return float(getattr(g, 'normal_r2_threshold_override', base))
    if not getattr(g, 'enable_range_r2_veto', False):
        return base
    if not _whipsaw_global_period_ok():
        return base
    if getattr(g, 'market_regime', '震荡期') != '震荡期':
        return base
    return float(getattr(g, 'r2_threshold_range_bound', 0.9))


def _effective_laplace_s():
    base = float(getattr(g, 'laplace_s_param', 0.05))
    if not getattr(g, 'enable_laplace_s_regime_differentiation', False):
        return base
    regime = getattr(g, 'market_regime', '震荡期')
    if regime == '正常期':
        return float(getattr(g, 'laplace_s_param_normal', 0.07))
    if regime == '走弱期':
        return float(getattr(g, 'laplace_s_param_weak', 0.10))
    return base


def _effective_laplace_min_slope():
    base = float(getattr(g, 'laplace_min_slope', 0.002))
    if (getattr(g, 'enable_p0_state_tuning_patch', False)
            and getattr(g, 'market_regime', '震荡期') == '正常期'):
        return float(getattr(g, 'normal_laplace_min_slope_override', base))
    return base


def boyishouyi_calculate_all_metrics_for_etf(etf, etf_name, hist_closes, hist_volumes,
                                               current_price, today_vol, context):
    try:
        current_price = _coerce_scalar_price(current_price)
        if current_price is None or current_price <= 0:
            return None
        price_series = np.append(hist_closes, current_price)
        short_lb = int(getattr(g, 'short_momentum_lookback', 21))
        need_len = max(g.lookback_days, short_lb)
        if len(price_series) < need_len * 0.8:
            return None

        momentum_score, annualized_returns, r_squared = boyishouyi_calculate_momentum_score(
            price_series, g.lookback_days)
        if momentum_score is None:
            return None
        short_momentum_score, _, _ = boyishouyi_calculate_momentum_score(price_series, short_lb)

        effective_r2_threshold = _effective_r2_threshold_whipsaw()
        passed_r2 = r_squared > effective_r2_threshold

        # 动量分数范围检查
        effective_min_score = float(getattr(g, 'min_score_threshold', 0))
        effective_max_score = float(getattr(g, 'max_score_threshold', 5))
        momentum_rank_score = momentum_score
        momentum_soft_capped = False

        soft_cap_enabled = bool(getattr(g, 'enable_momentum_soft_cap', False))
        soft_cap_normal_only = bool(getattr(g, 'momentum_soft_cap_normal_only', True))
        regime = getattr(g, 'market_regime', '震荡期')
        apply_soft_cap = soft_cap_enabled and ((not soft_cap_normal_only) or regime == '正常期')
        if apply_soft_cap and momentum_score > effective_max_score:
            penalty = float(getattr(g, 'momentum_soft_cap_penalty', 0.2))
            momentum_rank_score = effective_max_score + (momentum_score - effective_max_score) * penalty
            momentum_soft_capped = True
            passed_momentum = momentum_score >= effective_min_score
        else:
            passed_momentum = (effective_min_score <= momentum_score <= effective_max_score)

        # 成交量比
        volume_ratio = boyishouyi_get_volume_ratio(hist_volumes, today_vol, context, g.volume_lookback)
        effective_volume_threshold = float(getattr(g, 'volume_threshold', 0))
        if getattr(g, 'enable_volume_threshold_buffer', False):
            effective_volume_threshold += max(0.0, float(getattr(g, 'volume_threshold_buffer', 0.0)))
        passed_volume = (volume_ratio is not None and volume_ratio < effective_volume_threshold)

        # 连跌过滤
        passed_loss_filter = True
        day_ratios = []
        if len(price_series) >= 4:
            day1 = price_series[-1] / price_series[-2]
            day2 = price_series[-2] / price_series[-3]
            day3 = price_series[-3] / price_series[-4]
            day_ratios = [day1, day2, day3]
            if min(day_ratios) < g.loss:
                passed_loss_filter = False

        # MA过滤
        passed_ma, ma_value = True, None
        if len(price_series) >= g.ma_lookback:
            ma_value = np.mean(price_series[-g.ma_lookback:])
            passed_ma = current_price > ma_value * g.ma_threshold
        else:
            passed_ma = False

        # 溢价率
        premium_rate, passed_premium = boyishouyi_calculate_premium_rate(etf, context)

        # 拉普拉斯滤波
        laplace_value, laplace_slope, passed_laplace = 0, 0, False
        gaussian_value, gaussian_slope, passed_gaussian = 0.0, 0.0, False

        if len(price_series) >= 10:
            try:
                effective_laplace_s = _effective_laplace_s()
                laplace_values = laplace_filter(price_series, s=effective_laplace_s)
                if len(laplace_values) >= 2:
                    laplace_value = laplace_values[-1]
                    laplace_slope = laplace_values[-1] - laplace_values[-2]
                    passed_laplace = (current_price > laplace_values[-1]
                                      and laplace_slope > _effective_laplace_min_slope())
                g1, g2 = gaussian_filter_last_two(price_series, sigma=g.gaussian_sigma)
                gaussian_value = g1
                if getattr(g, 'gaussian_use_relative_slope', False):
                    gaussian_slope = ((g1 - g2) / g2) if abs(g2) > 1e-12 else 0.0
                    g_min = float(getattr(g, 'gaussian_min_slope_relative', 0.001))
                else:
                    gaussian_slope = g1 - g2
                    g_min = g.gaussian_min_slope
                passed_gaussian = (current_price > g1 and gaussian_slope > g_min)
            except Exception:
                pass

        # 双正动量过滤
        dual_pos_active = getattr(g, 'enable_dual_positive_momentum', False)
        if dual_pos_active and getattr(g, 'dual_positive_only_in_range', True):
            dual_pos_active = regime == '震荡期'
        passed_dual_positive = (momentum_score > 0 and short_momentum_score > 0
                                ) if short_momentum_score is not None else False

        return {
            'etf': etf, 'etf_name': etf_name,
            'momentum_score': momentum_score, 'momentum_rank_score': momentum_rank_score,
            'momentum_soft_capped': momentum_soft_capped,
            'short_momentum_score': short_momentum_score,
            'annualized_returns': annualized_returns, 'r_squared': r_squared,
            'effective_r2_threshold': effective_r2_threshold,
            'effective_min_score_threshold': effective_min_score,
            'effective_max_score_threshold': effective_max_score,
            'passed_momentum': passed_momentum, 'passed_r2': passed_r2,
            'passed_ma': passed_ma, 'passed_volume': passed_volume,
            'passed_loss': passed_loss_filter, 'passed_premium': passed_premium,
            'passed_laplace': passed_laplace, 'passed_gaussian': passed_gaussian,
            'passed_dual_positive': passed_dual_positive,
            'dual_positive_filter_active': dual_pos_active,
            'current_price': current_price, 'volume_ratio': volume_ratio,
            'effective_volume_threshold': effective_volume_threshold,
            'day_ratios': day_ratios, 'premium_rate': premium_rate,
            'ma_value': ma_value,
            'laplace_value': laplace_value, 'laplace_slope': laplace_slope,
            'effective_laplace_s': effective_laplace_s,
            'gaussian_value': gaussian_value, 'gaussian_slope': gaussian_slope,
        }
    except Exception as e:
        log.debug(f"[S4] 指标计算 {etf} 失败: {e}")
        return None


def boyishouyi_apply_filters(metrics_list):
    regime = getattr(g, 'market_regime', '震荡期')
    is_weak = regime == '走弱期'
    steps = [
        ('动量得分', lambda m: m['passed_momentum'], True),
        ('R²', lambda m: m['passed_r2'],
         g.enable_r2_filter and (not is_weak or getattr(g, 'weak_apply_r2_filter', False))),
        ('均线', lambda m: m['passed_ma'],
         g.enable_ma_filter and is_weak and getattr(g, 'weak_apply_ma_filter', False)),
        ('成交量', lambda m: m['passed_volume'],
         g.enable_volume_check and (not is_weak or getattr(g, 'weak_apply_volume_filter', False))),
        ('短期风控', lambda m: m['passed_loss'],
         g.enable_loss_filter and (not is_weak or getattr(g, 'weak_apply_loss_filter', False))),
        ('溢价率', lambda m: m['passed_premium'],
         g.enable_premium_filter and (not is_weak or getattr(g, 'weak_apply_premium_filter', False))),
        ('拉普拉斯滤波', lambda m: m['passed_laplace'],
         g.enable_laplace_filter and (
             regime == '正常期' or (is_weak and getattr(g, 'weak_apply_laplace_filter', False)))),
        ('高斯滤波', lambda m: m['passed_gaussian'], regime == '震荡期'),
    ]

    filtered = metrics_list[:]
    for _, check_fn, active in steps:
        if active:
            filtered = [m for m in filtered if check_fn(m)]
    return filtered


def boyishouyi_get_final_ranked_etfs(context):
    """主排名函数：计算指标→过滤→候选池→防频换"""
    all_metrics = []
    etf_set = list(g.merged_etf_pool)
    end_date = context.previous_date
    regime = getattr(g, 'market_regime', '震荡期')
    _slog('S4', '动量', f'合并池{len(etf_set)}只, 状态:{regime}')

    short_lb = int(getattr(g, 'short_momentum_lookback', 21))
    lookback = max(g.lookback_days, short_lb, g.volume_lookback, g.ma_lookback) + 20
    today = context.current_dt.date()
    current_data = get_current_data()
    safe_lookback = lookback + 20

    hist_df = get_price(etf_set, count=safe_lookback, end_date=end_date, frequency='1d',
                        fields=['close', 'volume'], panel=False)
    today_vol_df = get_price(etf_set, start_date=today, end_date=context.current_dt,
                             frequency='1m', fields=['volume'], panel=False, fill_paused=False)
    if hist_df is None or hist_df.empty:
        _slog('S4', '动量', '无法获取历史价格', 'warning')
        return []

    # 昨收和净值缓存
    g.etf_yesterday_close_batch = {}
    g.etf_yesterday_nav_batch = {}
    try:
        y_price_df = get_price(etf_set, start_date=end_date, end_date=end_date,
                               fields=['close'], panel=False)
        if y_price_df is not None and not y_price_df.empty:
            g.etf_yesterday_close_batch = y_price_df.groupby('code')['close'].last().to_dict()
        nav_df = get_extras('unit_net_value', etf_set, start_date=end_date, end_date=end_date)
        if nav_df is not None and not nav_df.empty:
            g.etf_yesterday_nav_batch = nav_df.iloc[-1].to_dict()
    except Exception:
        pass

    today_vols = today_vol_df.groupby('code')['volume'].sum() if (
        today_vol_df is not None and not today_vol_df.empty) else pd.Series(dtype=float)
    close_pivot = hist_df.pivot(index='time', columns='code', values='close')
    volume_pivot = hist_df.pivot(index='time', columns='code', values='volume')

    for etf in etf_set:
        if current_data[etf].paused:
            continue
        if etf not in close_pivot.columns:
            continue
        raw_closes = close_pivot[etf].values
        raw_volumes = volume_pivot[etf].values
        valid_mask = (~np.isnan(raw_volumes)) & (raw_volumes > 0)
        hist_closes = raw_closes[valid_mask][-lookback:]
        hist_volumes = raw_volumes[valid_mask][-lookback:]
        if len(hist_closes) < max(g.lookback_days, short_lb):
            continue
        etf_name = get_security_name(etf)
        current_price = current_data[etf].last_price
        today_vol = today_vols.get(etf, 0)
        metrics = boyishouyi_calculate_all_metrics_for_etf(
            etf, etf_name, hist_closes, hist_volumes, current_price, today_vol, context)
        if metrics:
            all_metrics.append(metrics)

    g.last_metrics_by_etf_code = {m['etf']: m for m in all_metrics}
    all_metrics.sort(key=lambda x: x.get('momentum_rank_score', float('-inf')), reverse=True)

    # 过滤
    filtered_list = boyishouyi_apply_filters(all_metrics)
    filtered_list.sort(key=lambda x: x.get('momentum_score', float('-inf')), reverse=True)

    top_10 = filtered_list[:10]
    if not top_10:
        _slog('S4', '动量', '无通过过滤的ETF')
        return []

    # 候选池
    score_key = 'momentum_rank_score'
    if len(top_10) >= g.holdings_num:
        reference_score = top_10[g.holdings_num - 1].get(score_key, float('-inf'))
        ratio = g.score_threshold_ratio if regime != '走弱期' else 1.0
        score_threshold = reference_score * ratio
        candidate_pool = [item for item in top_10 if item.get(score_key, float('-inf')) >= score_threshold]
    else:
        candidate_pool = top_10[:]

    # 防频换 (single-holding mode)
    if g.holdings_num == 1 and candidate_pool:
        current_holding = None
        for sec in _boyishouyi_sp(context).positions:
            if _boyishouyi_sp(context).positions[sec].total_amount > 0:
                current_holding = sec
                break
        top1_etf = candidate_pool[0]['etf']
        if current_holding and current_holding == top1_etf:
            g.normal_not_rank1_streak = 0
            g.normal_streak_hold_code = current_holding
        elif current_holding and current_holding != top1_etf:
            g.normal_not_rank1_streak = g.normal_not_rank1_streak + 1
            g.normal_streak_hold_code = current_holding
            max_streak = g.normal_max_days_not_rank1
            if g.normal_not_rank1_streak < max_streak:
                _slog('S4', '防频换', f'{current_holding}不在第1名, 连续{g.normal_not_rank1_streak}日<{max_streak}日, 保留')
                return [{'etf': current_holding, 'etf_name': get_security_name(current_holding),
                         'momentum_score': 0}]  # 保留持仓

    _slog('S4', '排名', f'候选池{len(candidate_pool)}只, 持仓数{g.holdings_num}')
    return candidate_pool


# ==================== 卖出执行 ====================
def boyishouyi_afternoon_sell(context):
    _slog('S4', '卖出', '流水线启动')

    if g.market_regime == '走弱期':
        if hasattr(g, 'filtered_global_pool') and g.filtered_global_pool:
            g.merged_etf_pool = list(set(g.filtered_global_pool))
        else:
            g.merged_etf_pool = list(set(g.global_etf_pool))
    else:
        _slog('S4', '卖出', f'{g.market_regime}: 合并池{len(g.merged_etf_pool)}只')

    boyishouyi_refresh_holdings_num_by_regime(context)
    g.ranked_etfs_result = boyishouyi_get_final_ranked_etfs(context)
    boyishouyi_execute_sell_trades(context)
    _slog('S4', '卖出', '流水线完毕')


def boyishouyi_execute_sell_trades(context):
    """根据排名结果卖出不在目标列表的持仓"""
    final_list = getattr(g, 'ranked_etfs_result', [])
    target_etfs = [m['etf'] for m in final_list[:g.holdings_num]]

    # 防御模式
    if not target_etfs:
        if boyishouyi_check_defensive_etf_available(context):
            target_etfs = [g.defensive_etf]
            _slog('S4', '卖出', '切防御ETF')

    target_set = set(target_etfs)
    for sec in list(_boyishouyi_sp(context).positions.keys()):
        pos = _boyishouyi_sp(context).positions[sec]
        if pos.total_amount <= 0:
            continue
        if sec not in target_set:
            if boyishouyi_smart_order(sec, 0, context, exit_reason='排名淘汰', pindex=3):
                _slog('S4', '卖出', f'{sec} {get_security_name(sec)}')


# ==================== 买入执行 ====================
def boyishouyi_afternoon_buy(context):
    _slog('S4', '买入', '流水线启动')
    boyishouyi_execute_buy_trades(context)
    _slog('S4', '买入', '流水线完毕')


def boyishouyi_execute_buy_trades(context):
    final_list = getattr(g, 'ranked_etfs_result', [])
    target_etfs = [m['etf'] for m in final_list[:g.holdings_num]]

    if not target_etfs:
        if boyishouyi_check_defensive_etf_available(context):
            target_etfs = [g.defensive_etf]
        else:
            return

    # 过滤已持有
    buy_candidates = []
    for etf in target_etfs:
        pos = _boyishouyi_sp(context).positions.get(etf)
        if not pos or pos.total_amount == 0:
            buy_candidates.append(etf)

    if not buy_candidates:
        return

    # 等权分配
    total_val = _boyishouyi_sp(context).total_value
    target_per = total_val / len(target_etfs)
    if getattr(g, 'boyishouyi_rebalance_cap', None) is not None:
        cap_per = g.boyishouyi_rebalance_cap / len(target_etfs)
        target_per = min(target_per, cap_per)

    available = _boyishouyi_sp(context).available_cash
    for etf in buy_candidates:
        if available < g.min_money:
            break
        buy_value = min(target_per, available)
        current_price = get_current_data()[etf].last_price
        if current_price <= 0:
            continue
        buy_amount = int(buy_value / current_price / 100) * 100
        if buy_amount >= 100:
            order(etf, buy_amount, pindex=3)
            _slog('S4', '买入', f'{etf} {get_security_name(etf)} {buy_amount}股@{current_price:.3f}')
            available -= buy_amount * current_price


# ==================== 智能下单 ====================
def boyishouyi_smart_order(security, target_value, context, exit_reason="", pindex=3):
    current_data = get_current_data()
    if current_data[security].paused:
        return False
    price = current_data[security].last_price
    if price <= 0:
        return False

    cur_pos = context.subportfolios[pindex].positions.get(security)
    cur_amount = cur_pos.total_amount if cur_pos else 0
    target_amount = int(target_value / price) if price > 0 else 0
    target_amount = (target_amount // 100) * 100
    diff = target_amount - cur_amount

    if diff > 0:
        if price >= current_data[security].high_limit:
            return False
    elif diff < 0:
        if price <= current_data[security].low_limit:
            return False
        closeable = cur_pos.closeable_amount if cur_pos else 0
        if closeable == 0:
            return False
        diff = -min(abs(diff), closeable)

    if diff != 0:
        if diff < 0:
            sell_price = _get_bid1_sell_price(security, price)
            order(security, diff, style=LimitOrderStyle(sell_price), pindex=pindex)
        else:
            order(security, diff, pindex=pindex)
        return True
    return False


def boyishouyi_check_defensive_etf_available(context):
    data = get_current_data()
    etf = g.defensive_etf
    if data[etf].paused:
        return False
    if data[etf].last_price >= data[etf].high_limit:
        return False
    if data[etf].last_price <= data[etf].low_limit:
        return False
    return True


# ==================== 组合回撤监控 ====================
def boyishouyi_monitor_drawdown(context):
    try:
        current_value = _boyishouyi_sp(context).total_value
        if current_value > g.max_portfolio_value:
            g.max_portfolio_value = current_value
        if g.max_portfolio_value <= 0:
            return
        current_drawdown = (g.max_portfolio_value - current_value) / g.max_portfolio_value
        if current_drawdown < g.drawdown_threshold:
            return

        _slog('S4', '回撤', f'预警: {current_drawdown:.2%} (净值{current_value:,.0f})', 'warning')

        if not getattr(g, 'enable_drawdown_risk_actions', False):
            return

        today = context.current_dt.date()
        if getattr(g, 'dd_action_cooldown_date', None) == today:
            return

        th_flat = float(getattr(g, 'dd_flat_threshold', 0.20))
        th_def = float(getattr(g, 'dd_switch_defensive_threshold', 0.12))
        th_half = float(getattr(g, 'dd_half_position_threshold', 0.10))

        if current_drawdown >= th_flat:
            _slog('S4', '回撤', f'全部清仓! {current_drawdown:.2%}', 'error')
            for sec in list(_boyishouyi_sp(context).positions.keys()):
                boyishouyi_smart_order(sec, 0, context, exit_reason='回撤清仓', pindex=3)
            g.dd_action_cooldown_date = today
        elif current_drawdown >= th_def:
            if boyishouyi_check_defensive_etf_available(context):
                for sec in list(_boyishouyi_sp(context).positions.keys()):
                    if sec != g.defensive_etf:
                        boyishouyi_smart_order(sec, 0, context, exit_reason='切防御', pindex=3)
                g.dd_action_cooldown_date = today
        elif current_drawdown >= th_half:
            for sec in list(_boyishouyi_sp(context).positions.keys()):
                pos = _boyishouyi_sp(context).positions[sec]
                ca = min(pos.closeable_amount, pos.total_amount // 2)
                ca = (ca // 100) * 100
                if ca >= 100:
                    order(sec, -ca, pindex=3)
            g.dd_action_cooldown_date = today

        if getattr(g, 'dd_reset_peak_after_action', True) and g.dd_action_cooldown_date == today:
            g.max_portfolio_value = current_value
    except Exception as e:
        _slog('S4', '回撤', f'异常: {e}', 'error')


# ==================== 分钟止损 ====================
def boyishouyi_minute_stop_loss(context):
    """固定止损: 现价 <= 成本价 * fixedStopLossThreshold"""
    if not getattr(g, 'use_fixed_stop_loss', False):
        return
    now = context.current_dt
    if not (("09:25" <= now.strftime("%H:%M") <= "11:30")
            or ("13:00" <= now.strftime("%H:%M") <= "14:57")):
        return
    threshold = getattr(g, 'fixedStopLossThreshold', 0.92)
    for sec in list(_boyishouyi_sp(context).positions.keys()):
        pos = _boyishouyi_sp(context).positions[sec]
        if pos.total_amount <= 0 or pos.closeable_amount <= 0:
            continue
        current_data = get_current_data()
        if current_data[sec].paused:
            continue
        if current_data[sec].last_price <= pos.avg_cost * threshold:
            sell_price = _get_bid1_sell_price(sec, current_data[sec].last_price)
            order_target(sec, 0, LimitOrderStyle(sell_price), pindex=3)
            _slog('S4', '固定止损', f'{sec} {get_security_name(sec)} 现价{current_data[sec].last_price:.3f} <= 成本{pos.avg_cost:.3f}*{threshold}')


def boyishouyi_minute_pct_stop_loss(context):
    """百分比止损: 现价 <= 昨收 * pct_stop_loss_threshold"""
    if not getattr(g, 'use_pct_stop_loss', False):
        return
    now = context.current_dt
    if not (("09:25" <= now.strftime("%H:%M") <= "11:30")
            or ("13:00" <= now.strftime("%H:%M") <= "14:57")):
        return
    threshold = getattr(g, 'pct_stop_loss_threshold', 0.95)
    for sec in list(_boyishouyi_sp(context).positions.keys()):
        pos = _boyishouyi_sp(context).positions[sec]
        if pos.total_amount <= 0 or pos.closeable_amount <= 0:
            continue
        current_data = get_current_data()
        if current_data[sec].paused:
            continue
        yesterday_close = g.yesterday_close_cache.get(sec)
        if yesterday_close is None:
            try:
                df = get_price(sec, end_date=context.previous_date, count=1, frequency='1d', fields=['close'])
                yesterday_close = df['close'].iloc[0] if not df.empty else None
            except:
                continue
            if yesterday_close:
                g.yesterday_close_cache[sec] = yesterday_close
        if yesterday_close and current_data[sec].last_price <= yesterday_close * threshold:
            sell_price = _get_bid1_sell_price(sec, current_data[sec].last_price)
            order_target(sec, 0, LimitOrderStyle(sell_price), pindex=3)
            _slog('S4', '百分比止损', f'{sec} 现价{current_data[sec].last_price:.3f} <= 昨收{yesterday_close:.3f}*{threshold}')


# ==================== MA跌破观测 ====================
def boyishouyi_holdings_ma_break_1450_check(context):
    if not getattr(g, 'enable_holdings_ma_break_nextday_stats', False):
        return
    signal_date = context.current_dt.date()
    prev_eod = context.previous_date
    if hasattr(prev_eod, 'date'):
        prev_eod = prev_eod.date()
    holdings = [s for s, p in _boyishouyi_sp(context).positions.items() if p.total_amount > 0]
    if not holdings:
        return
    for sec in holdings:
        try:
            hd = get_price(sec, end_date=prev_eod, count=35, frequency='1d',
                           fields=['close'], panel=False, skip_paused=True)
            if hd is None or len(hd) < 30:
                continue
            closes = hd['close'].astype(float).values
            ma5, ma10, ma20, ma30 = (float(np.mean(closes[-n:])) for n in (5, 10, 20, 30))
        except Exception:
            continue
        px = _get_intraday_price_with_fallback(context, sec)
        if px is None or px <= 0:
            continue
        below_any = px < ma5 or px < ma10 or px < ma20 or px < ma30
        if below_any:
            if not hasattr(g, 'ma_break_nextday_pending') or g.ma_break_nextday_pending is None:
                g.ma_break_nextday_pending = []
            g.ma_break_nextday_pending.append({
                'signal_date': signal_date, 'code': sec, 'px_1450': float(px),
                'ma5': ma5, 'ma10': ma10, 'ma20': ma20, 'ma30': ma30,
                'below_ma5': px < ma5, 'below_ma10': px < ma10,
                'below_ma20': px < ma20, 'below_ma30': px < ma30,
            })


# ==================== 收盘 ====================
def boyishouyi_reset_daily_flags(context):
    g.cache_date = None
    g.yesterday_close_cache = {}


def boyishouyi_after_close_regime_statistics(context):
    v = _boyishouyi_sp(context).total_value
    prev = getattr(g, 'prev_eod_portfolio_value', None)
    reg = getattr(g, 'market_regime', '震荡期')

    if not isinstance(getattr(g, 'regime_day_counts', None), dict):
        g.regime_day_counts = {'正常期': 0, '震荡期': 0, '走弱期': 0}
    g.regime_day_counts[reg] = int(g.regime_day_counts.get(reg, 0)) + 1
    if prev is not None and prev > 0:
        daily_ret = (v - prev) / prev
        g.regime_return_factors[reg] = g.regime_return_factors.get(reg, 1.0) * (1.0 + daily_ret)
        if daily_ret > 0:
            g.regime_win_counts[reg] = int(g.regime_win_counts.get(reg, 0)) + 1
        elif daily_ret < 0:
            g.regime_loss_counts[reg] = int(g.regime_loss_counts.get(reg, 0)) + 1
    g.prev_eod_portfolio_value = v


def boyishouyi_record_daily_positions(context):
    _slog_global('收盘', '持仓清单')
    for sec in _boyishouyi_sp(context).positions:
        pos = _boyishouyi_sp(context).positions[sec]
        if pos.total_amount > 0:
            log.info(f"  持仓: {sec} {get_security_name(sec)} {pos.total_amount}股 @{pos.price:.3f}")


def boyishouyi_output_positions_summary(context):
    regime = getattr(g, 'market_regime', '震荡期')
    _slog_global('收盘', f'状态:{regime} 持仓:{len(_boyishouyi_sp(context).positions)}只')
