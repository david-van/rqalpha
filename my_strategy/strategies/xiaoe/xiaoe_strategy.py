#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xiaoe_articles 股票池策略 — 多模式分配，最大化收益

模式:
  equal_weight      — 等权持有池中所有股票，池变化时调仓
  momentum_top      — 动量打分，集中持有前 N 只
  momentum_weighted — 动量加权持有全部池内股票（推荐）
  newest            — 只持有最新加入池的股票（博主最强信号）
  oldest            — 只持有待在池中最久的股票
  buy_and_hold      — 买入持有，不在池内股票间做再平衡，只在池变化时操作
  buy_hold_low_vol_hybrid
                   — 60% 买入持有底仓 + 低波动增强仓
  low_vol_rebalance_hybrid
                   — 全池等权底仓 + 低波动增强仓，定期整体再平衡
  low_position_hybrid
                   — 全池等权底仓 + 低位/不过热增强仓
  momentum_skip_recent_hybrid
                   — 全池等权底仓 + 中期动量增强仓，跳过最近涨幅
  pullback_trend_hybrid
                   — 全池等权底仓 + 中期趋势中短期回调增强仓
"""

import bisect
import math
from pathlib import Path

import numpy as np
import pandas as pd

from my_strategy.common_file import project_root
from rqalpha.api import (
    history_bars, order_target_value, logger, scheduler,
)
from rqalpha.mod.rqalpha_mod_sys_scheduler.scheduler import physical_time


# ============================================================
# 动态股票池加载
# ============================================================
class PoolLoader:
    def __init__(self, csv_dir):
        self._first_seen = {}
        self._snapshots = []
        for f in sorted(Path(csv_dir).glob("*holdings*.csv")):
            df = pd.read_csv(f, encoding="utf-8-sig")
            for _, row in df.iterrows():
                codes_str = str(row.get("持有股票代码", ""))
                if not codes_str or codes_str == "nan":
                    continue
                codes = [c.strip() for c in codes_str.split("|") if c.strip()]
                if codes:
                    d = pd.Timestamp(str(row["日期"]).strip())
                    self._snapshots.append((d, codes))
                    for c in codes:
                        if c not in self._first_seen:
                            self._first_seen[c] = d
        self._snapshots.sort(key=lambda x: x[0])

    def get_pool(self, dt):
        """返回 dt 时刻的股票池列表"""
        dates = [s[0] for s in self._snapshots]
        idx = bisect.bisect_right(dates, dt) - 1
        return self._snapshots[idx][1] if idx >= 0 else []

    def first_seen(self, code):
        return self._first_seen.get(code, pd.Timestamp("2099-01-01"))


# ============================================================
# 动量打分
# ============================================================
def momentum_score(code, m_days, decay_ratio):
    """年化收益 × R² 动量得分，支持衰减加权回归"""
    close = history_bars(code, m_days, '1d', 'close', adjust_type='pre')
    if close is None or len(close) < m_days:
        return float('-inf')
    y = np.log(np.asarray(close))
    x = np.arange(len(y))
    if decay_ratio > 1.0:
        w = np.linspace(1.0, decay_ratio, len(y))
        slope, intercept = np.polyfit(x, y, 1, w=w)
        y_pred = slope * x + intercept
        ss_res = np.sum(w * (y - y_pred) ** 2)
        y_mean = np.average(y, weights=w)
        ss_tot = np.sum(w * (y - y_mean) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    else:
        slope, intercept = np.polyfit(x, y, 1)
        ss_res = np.sum((y - (slope * x + intercept)) ** 2)
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    ann = math.pow(math.exp(slope), 250) - 1
    return ann * r2


def score_all(pool, m_days, decay_ratio):
    """对池中所有股票打分，返回 {code: score}"""
    scores = {}
    for code in pool:
        scores[code] = momentum_score(code, m_days, decay_ratio)
    return scores


# ============================================================
# 默认参数
# ============================================================
DEFAULT_PARAMS = {
    'mode': 'momentum_weighted',
    'top_n': 2,
    'm_days': 25,
    'ema_days': 60,
    'decay_ratio': 1.0,
    'switch_threshold': 1.0,
    'rebalance_days': 5,   # 无池变化时，至少间隔 N 个交易日才调仓 (momentum 模式)
    'base_ratio': 0.6,     # buy_hold_low_vol_hybrid 的买入持有底仓比例
    'enhance_ratio': 0.4,  # buy_hold_low_vol_hybrid 的增强仓上限
    'low_vol_days': 60,    # 低波动计算窗口
    'enhance_top_n': 2,    # 增强仓加给波动率最低的 N 只
    'low_position_days': 250,
    'short_return_days': 20,
    'max_recent_return': 0.30,
    'vol_filter_days': 60,
    'exclude_top_vol_pct': 0.30,
    'max_position_ratio': 0.35,
    'momentum_lookback_days': 120,
    'momentum_skip_days': 20,
    'min_mid_momentum': 0.0,
    'tactical_reduce': False,
    'entry_gain_reduce': 0.60,
    'entry_gain_cut': 0.80,
    'recent_gain_reduce': 0.25,
    'market_filter': False,
    'market_index': '000300.XSHG',
    'market_ma_days': 200,
    'weak_market_exposure': 0.60,
}


# ============================================================
# 策略入口
# ============================================================
def init(context):
    csv_dir = str(Path(project_root) / 'my_strategy' / 'strategies' / 'xiaoe_articles')
    params = DEFAULT_PARAMS.copy()

    injected = getattr(context, 'strategy_params', None)
    if injected:
        # 递归转换 RqAttrDict → plain dict
        plain = {}
        for k, v in injected.items():
            if hasattr(v, 'items'):
                plain[k] = dict(v)
            else:
                plain[k] = v
        params.update(plain)

    context.params = params
    context.loader = PoolLoader(csv_dir if 'pool_csv_dir' not in params else params['pool_csv_dir'])
    context.pool = context.loader.get_pool(context.now)
    context.old_pool = []           # buy_and_hold 模式追踪上一期池子
    context.targets = {}
    context.pending_targets = None  # targets to execute today only
    context.ema_pending_targets = {}
    context.stock_slots = {}        # per-stock cash slot for buy_and_hold_ema
    context.hybrid_base_units = {}   # virtual shares for buy-and-hold base sleeve
    context.hybrid_enhance_units = {}
    context.hybrid_pending_values = None
    context.hybrid_last_rebalance = None
    context.hybrid_days_since_rebalance = 0
    context.pool_entry_prices = {}
    context.last_rebalance = None   # 上次调仓日期
    context.pool_changed = True     # 首日强制调仓

    logger.info(
        f"xiaoe_strategy 就绪 mode={params['mode']} "
        f"ema_days={params.get('ema_days')} pool={context.pool}"
    )

    scheduler.run_daily(sell_trade, time_rule=physical_time(hour=9, minute=35))
    scheduler.run_daily(buy_trade, time_rule=physical_time(hour=9, minute=37))


def before_trading(context):
    """检查池是否变化"""
    if getattr(context, 'hybrid_last_rebalance', None) is not None:
        context.hybrid_days_since_rebalance = getattr(context, 'hybrid_days_since_rebalance', 0) + 1

    new_pool = context.loader.get_pool(context.now)
    if set(new_pool) != set(context.pool):
        removed = set(context.pool) - set(new_pool)
        for code in removed:
            getattr(context, 'pool_entry_prices', {}).pop(code, None)
        logger.info(f"[池变化] {context.pool} → {new_pool}")
        context.pool = new_pool
        context.pool_changed = True


def _current_weight(context, code):
    """某只股票的当前市值权重"""
    pos = context.portfolio.positions.get(code)
    tv = context.portfolio.total_value
    return pos.market_value / tv if pos and tv > 0 else 0.0


def _compute_buy_and_hold(context):
    """buy_and_hold 模式：只在池变化时操作，已持有股票不做再平衡

    规则:
      1. 股票被博主清出池 → 卖出
      2. 新股票入池 → 等权买入（资金优先用现金/卖股所得，不够再从留存股均分卖出）
      3. 留存股票（新旧池都在）→ 仓位不动，除非需要腾资金给新股
      4. 多余现金 → 均分给留存股票加仓
    """
    new_pool = context.pool
    old_pool = getattr(context, 'old_pool', [])

    if not new_pool:
        return {}

    # 首次调用（无 old_pool）：等权建仓
    if not old_pool:
        context.old_pool = list(new_pool)
        w = 1.0 / len(new_pool)
        return {c: w for c in new_pool}

    context.old_pool = list(new_pool)

    removed = [s for s in old_pool if s not in new_pool]
    kept    = [s for s in old_pool if s in new_pool]
    added   = [s for s in new_pool if s not in old_pool]

    # 池没变 → 不操作
    if not removed and not added:
        return {}

    pool_sz = len(new_pool)
    target_w = 1.0 / pool_sz

    targets = {}

    # --- Step 1: 移除的股票 → 权重归零 ---
    for s in removed:
        targets[s] = 0.0

    # --- Step 2: 计算可动用资金权重 ---
    removed_w = sum(_current_weight(context, s) for s in removed)
    cash_w = context.portfolio.cash / context.portfolio.total_value
    available_w = removed_w + cash_w

    if not added:
        # --- 场景 A: 只有移除，没有新增 ---
        # 卖出释放的资金均分给留存的旧股票
        if available_w > 0.0001 and kept:
            per_extra = available_w / len(kept)
            for s in kept:
                targets[s] = _current_weight(context, s) + per_extra
        return targets

    # --- 场景 B: 有新增股票 ---
    needed_w = target_w * len(added)

    if needed_w <= available_w:
        # --- 资金充足：新股买入 + 余钱均分给旧股 ---
        for s in added:
            targets[s] = target_w
        surplus_w = available_w - needed_w
        if surplus_w > 0.0001 and kept:
            per_surplus = surplus_w / len(kept)
            for s in kept:
                targets[s] = _current_weight(context, s) + per_surplus
        else:
            for s in kept:
                targets[s] = _current_weight(context, s)
    else:
        # --- 资金不足：从旧股均分卖出，凑钱买新股 ---
        shortfall_w = needed_w - available_w

        if not kept:
            # 完全换仓：直接等权
            for s in new_pool:
                targets[s] = target_w
            return targets

        per_deduct = shortfall_w / len(kept)
        for s in kept:
            targets[s] = max(0.0, _current_weight(context, s) - per_deduct)
        for s in added:
            targets[s] = target_w

    return targets


def _is_buy_hold_low_vol_hybrid_mode(context):
    return context.params['mode'] in (
        'buy_hold_low_vol_hybrid',
        'buy_and_hold_low_vol_hybrid',
        'bh_low_vol_hybrid',
    )


def _is_low_vol_rebalance_hybrid_mode(context):
    return context.params['mode'] in (
        'low_vol_rebalance_hybrid',
        'low_vol_hybrid',
        'low_vol_hybrid50',
    )


def _is_constrained_rebalance_hybrid_mode(context):
    return context.params['mode'] in (
        'low_position_hybrid',
        'low_position_tactical_hybrid',
        'momentum_skip_recent_hybrid',
        'pullback_trend_hybrid',
    )


def _hybrid_log_tag(context):
    return str(context.params.get('mode', 'hybrid'))


def _latest_price(code):
    close = history_bars(code, 1, '1d', 'close', adjust_type='pre')
    if close is None or len(close) == 0:
        return None
    price = float(np.asarray(close, dtype=float)[-1])
    if not np.isfinite(price) or price <= 0:
        return None
    return price


def _prices_for(codes):
    prices = {}
    for code in codes:
        price = _latest_price(code)
        if price is not None:
            prices[code] = price
    return prices


def _units_to_values(units, prices, pool=None):
    pool_set = set(pool) if pool is not None else None
    values = {}
    for code, unit in (units or {}).items():
        if pool_set is not None and code not in pool_set:
            continue
        price = prices.get(code)
        if price is None:
            continue
        value = float(unit) * price
        if value > 1:
            values[code] = value
    return values


def _values_to_units(values, prices):
    units = {}
    for code, value in values.items():
        price = prices.get(code)
        if price is None or price <= 0 or value <= 1:
            continue
        units[code] = float(value) / price
    return units


def _volatility_score(code, days):
    close = history_bars(code, days + 1, '1d', 'close', adjust_type='pre')
    if close is None or len(close) < days + 1:
        return float('inf')

    close = np.asarray(close, dtype=float)
    if np.any(close[:-1] <= 0):
        return float('inf')

    returns = close[1:] / close[:-1] - 1
    returns = returns[np.isfinite(returns)]
    if len(returns) < days:
        return float('inf')
    vol = float(np.std(returns, ddof=1))
    return vol if np.isfinite(vol) else float('inf')


def _low_vol_candidates(pool, days, top_n):
    scored = []
    for code in pool:
        vol = _volatility_score(code, days)
        if vol != float('inf'):
            scored.append((code, vol))
    scored.sort(key=lambda item: (item[1], item[0]))
    return [code for code, _ in scored[:top_n]]


def _close_series(code, days):
    close = history_bars(code, days, '1d', 'close', adjust_type='pre')
    if close is None or len(close) < days:
        return None
    close = np.asarray(close, dtype=float)
    close = close[np.isfinite(close)]
    if len(close) < days or np.any(close <= 0):
        return None
    return close


def _recent_return(code, days):
    close = _close_series(code, days + 1)
    if close is None:
        return None
    return float(close[-1] / close[-1 - days] - 1)


def _low_position_features(context, code):
    lookback = int(context.params.get('low_position_days', 250))
    recent_days = int(context.params.get('short_return_days', 20))
    vol_days = int(context.params.get('vol_filter_days', 60))
    days = max(lookback, recent_days, vol_days) + 1
    close = _close_series(code, days)
    if close is None:
        return None

    current = float(close[-1])
    high = float(np.max(close[-lookback:]))
    if high <= 0:
        return None

    recent_ret = float(close[-1] / close[-1 - recent_days] - 1)
    vol_slice = close[-vol_days - 1:]
    vol_ret = vol_slice[1:] / vol_slice[:-1] - 1
    vol = float(np.std(vol_ret, ddof=1)) if len(vol_ret) >= vol_days else float('inf')
    if not np.isfinite(vol):
        return None

    drawdown = current / high - 1
    return {
        'drawdown': float(drawdown),
        'recent_ret': recent_ret,
        'vol': vol,
    }


def _low_position_candidates(context, pool, top_n):
    rows = []
    for code in pool:
        features = _low_position_features(context, code)
        if features is None:
            continue
        if features['recent_ret'] > float(context.params.get('max_recent_return', 0.30)):
            continue
        rows.append((code, features))

    if not rows:
        return []

    exclude_pct = max(0.0, min(1.0, float(context.params.get('exclude_top_vol_pct', 0.30))))
    if len(rows) > 1 and exclude_pct > 0:
        vol_sorted = sorted(rows, key=lambda item: item[1]['vol'], reverse=True)
        exclude_n = min(len(rows) - 1, int(math.ceil(len(rows) * exclude_pct)))
        excluded = {code for code, _ in vol_sorted[:exclude_n]}
        rows = [(code, features) for code, features in rows if code not in excluded]

    rows.sort(
        key=lambda item: (
            item[1]['drawdown'],
            item[1]['recent_ret'],
            item[1]['vol'],
            item[0],
        )
    )
    return [code for code, _ in rows[:top_n]]


def _momentum_skip_recent_score(context, code):
    lookback = int(context.params.get('momentum_lookback_days', 120))
    skip = int(context.params.get('momentum_skip_days', 20))
    close = _close_series(code, lookback + 1)
    if close is None or lookback <= skip:
        return None
    return float(close[-1 - skip] / close[-1 - lookback] - 1)


def _pullback_trend_score(context, code):
    mid = _momentum_skip_recent_score(context, code)
    if mid is None:
        return None
    min_mid = float(context.params.get('min_mid_momentum', 0.0))
    if mid < min_mid:
        return None
    recent_days = int(context.params.get('momentum_skip_days', 20))
    recent = _recent_return(code, recent_days)
    if recent is None:
        return None
    if recent > float(context.params.get('max_recent_return', 0.30)):
        return None
    return float(mid - recent)


def _score_candidates(context, pool, top_n, score_func):
    scored = []
    for code in pool:
        score = score_func(context, code)
        if score is None or not np.isfinite(score):
            continue
        scored.append((code, float(score)))
    scored.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return [code for code, _ in scored[:top_n]]


def _sync_pool_entry_prices(context, prices):
    entry_prices = getattr(context, 'pool_entry_prices', {}) or {}
    for code in context.pool:
        price = prices.get(code)
        if code not in entry_prices and price is not None and price > 0:
            entry_prices[code] = price
    context.pool_entry_prices = entry_prices


def _apply_target_value_cap(targets, total_value, max_ratio):
    if total_value <= 0 or max_ratio is None or max_ratio <= 0:
        return targets

    cap_value = total_value * max_ratio
    capped = dict(targets)
    for _ in range(10):
        excess = 0.0
        for code, value in list(capped.items()):
            if value > cap_value:
                excess += value - cap_value
                capped[code] = cap_value
        if excess <= 1:
            break

        receivers = [code for code, value in capped.items() if value < cap_value - 1]
        if not receivers:
            break
        add_value = excess / len(receivers)
        for code in receivers:
            capped[code] = min(cap_value, capped[code] + add_value)

    return capped


def _market_exposure(context):
    if not context.params.get('market_filter', False):
        return 1.0

    index = context.params.get('market_index', '000300.XSHG')
    ma_days = int(context.params.get('market_ma_days', 200))
    close = history_bars(index, ma_days + 1, '1d', 'close', adjust_type='pre')
    if close is None or len(close) < ma_days:
        return 1.0

    close = np.asarray(close, dtype=float)
    latest = float(close[-1])
    ma = float(np.mean(close[-ma_days:]))
    if not np.isfinite(latest) or not np.isfinite(ma) or ma <= 0:
        return 1.0
    if latest >= ma:
        return 1.0
    return max(0.0, min(1.0, float(context.params.get('weak_market_exposure', 0.60))))


def _apply_market_filter(context, targets):
    exposure = _market_exposure(context)
    if exposure >= 0.999:
        return targets
    logger.info(f"[{_hybrid_log_tag(context)}] market exposure={exposure:.0%}")
    return {code: value * exposure for code, value in targets.items()}


def _apply_tactical_reduce(context, targets, base_values, prices):
    if not context.params.get('tactical_reduce', False) and context.params['mode'] != 'low_position_tactical_hybrid':
        return targets

    reduced = dict(targets)
    entry_prices = getattr(context, 'pool_entry_prices', {}) or {}
    recent_days = int(context.params.get('short_return_days', 20))
    reduce_gain = float(context.params.get('entry_gain_reduce', 0.60))
    cut_gain = float(context.params.get('entry_gain_cut', 0.80))
    recent_threshold = float(context.params.get('recent_gain_reduce', 0.25))

    for code in list(reduced):
        entry = entry_prices.get(code)
        price = prices.get(code)
        if entry is None or price is None or entry <= 0:
            continue
        since_entry = float(price / entry - 1)
        recent = _recent_return(code, recent_days)
        if recent is None:
            continue

        base_value = base_values.get(code, 0.0)
        if since_entry >= cut_gain:
            reduced[code] = min(reduced[code], base_value)
            logger.info(f"[{_hybrid_log_tag(context)}] cut enhance {code} gain={since_entry:.2%}")
        elif since_entry >= reduce_gain and recent >= recent_threshold:
            reduced[code] = min(reduced[code], base_value)
            logger.info(
                f"[{_hybrid_log_tag(context)}] reduce enhance {code} "
                f"gain={since_entry:.2%} recent={recent:.2%}"
            )
    return reduced


def _enhance_candidates(context, tradable_pool):
    mode = context.params['mode']
    top_n = int(context.params.get('enhance_top_n', 2))
    if mode in ('low_position_hybrid', 'low_position_tactical_hybrid'):
        return _low_position_candidates(context, tradable_pool, top_n)
    if mode == 'momentum_skip_recent_hybrid':
        return _score_candidates(context, tradable_pool, top_n, _momentum_skip_recent_score)
    if mode == 'pullback_trend_hybrid':
        return _score_candidates(context, tradable_pool, top_n, _pullback_trend_score)
    return []


def _compute_constrained_rebalance_hybrid_values(context):
    pool = list(context.pool)
    prices = _prices_for(pool)
    tradable_pool = [code for code in pool if code in prices]

    if not tradable_pool:
        context.pool_changed = False
        return {}

    _sync_pool_entry_prices(context, prices)

    total_value = context.portfolio.total_value
    base_ratio = float(context.params.get('base_ratio', 0.7))
    enhance_ratio = float(context.params.get('enhance_ratio', 0.3))

    base_budget = max(0.0, total_value * base_ratio)
    enhance_budget = max(0.0, total_value * enhance_ratio)
    base_per_stock = base_budget / len(tradable_pool)
    base_values = {code: base_per_stock for code in tradable_pool}
    targets = dict(base_values)

    candidates = _enhance_candidates(context, tradable_pool)
    if candidates and enhance_budget > 0:
        enhance_per_stock = enhance_budget / len(candidates)
        for code in candidates:
            targets[code] = targets.get(code, 0.0) + enhance_per_stock
    elif enhance_budget > 0:
        fallback_extra = enhance_budget / len(tradable_pool)
        for code in tradable_pool:
            targets[code] = targets.get(code, 0.0) + fallback_extra

    targets = _apply_target_value_cap(
        targets,
        total_value,
        float(context.params.get('max_position_ratio', 0.35)),
    )
    targets = _apply_tactical_reduce(context, targets, base_values, prices)
    targets = _apply_market_filter(context, targets)

    context.hybrid_last_rebalance = context.now
    context.hybrid_days_since_rebalance = 0
    context.pool_changed = False
    logger.info(
        f"[{_hybrid_log_tag(context)}] candidates={candidates} "
        f"base={base_ratio:.0%} enhance={enhance_ratio:.0%}"
    )
    return {code: value for code, value in targets.items() if value > 1}


def _sync_hybrid_base_sleeve(context, prices):
    """Sync the buy-and-hold base sleeve only when the pool changes."""
    pool = list(context.pool)
    base_ratio = float(context.params.get('base_ratio', 0.6))
    base_budget = max(0.0, context.portfolio.total_value * base_ratio)

    if not pool or base_budget <= 0:
        context.hybrid_base_units = {}
        return {}

    old_units = getattr(context, 'hybrid_base_units', {}) or {}
    base_values = _units_to_values(old_units, prices, pool)

    if not base_values:
        per_stock = base_budget / len(pool)
        base_values = {code: per_stock for code in pool if code in prices}
    else:
        per_new_stock = base_budget / len(pool)
        for code in pool:
            if code not in base_values and code in prices:
                base_values[code] = per_new_stock

        current_total = sum(base_values.values())
        if current_total > 0:
            if current_total > base_budget:
                scale = base_budget / current_total
                base_values = {code: value * scale for code, value in base_values.items()}
            elif current_total < base_budget and base_values:
                extra = (base_budget - current_total) / len(base_values)
                base_values = {code: value + extra for code, value in base_values.items()}

    context.hybrid_base_units = _values_to_units(base_values, prices)
    return base_values


def _current_hybrid_base_values(context, prices):
    return _units_to_values(getattr(context, 'hybrid_base_units', {}) or {}, prices, context.pool)


def _compute_hybrid_enhance_values(context, prices, base_values):
    pool = list(context.pool)
    enhance_ratio = float(context.params.get('enhance_ratio', 0.4))
    top_n = int(context.params.get('enhance_top_n', 2))
    low_vol_days = int(context.params.get('low_vol_days', 60))

    base_total = sum(base_values.values())
    capacity = max(0.0, context.portfolio.total_value - base_total)
    enhance_budget = min(max(0.0, context.portfolio.total_value * enhance_ratio), capacity)
    candidates = _low_vol_candidates(pool, low_vol_days, top_n)
    if not candidates or enhance_budget <= 0:
        context.hybrid_enhance_units = {}
        return {}

    per_stock = enhance_budget / len(candidates)
    enhance_values = {code: per_stock for code in candidates if code in prices}
    context.hybrid_enhance_units = _values_to_units(enhance_values, prices)
    logger.info(
        f"[bh_low_vol] enhance candidates={candidates} "
        f"budget={enhance_budget:.2f}"
    )
    return enhance_values


def _should_rebalance_hybrid(context):
    if context.pool_changed:
        return True
    if getattr(context, 'hybrid_last_rebalance', None) is None:
        return True
    days = getattr(context, 'hybrid_days_since_rebalance', 0)
    return days >= int(context.params.get('rebalance_days', 20))


def _compute_buy_hold_low_vol_hybrid_values(context):
    pool = list(context.pool)
    codes = set(pool)
    codes.update(getattr(context, 'hybrid_base_units', {}) or {})
    codes.update(getattr(context, 'hybrid_enhance_units', {}) or {})
    prices = _prices_for(codes)

    if not pool:
        context.hybrid_base_units = {}
        context.hybrid_enhance_units = {}
        return {}

    if context.pool_changed or not getattr(context, 'hybrid_base_units', None):
        base_values = _sync_hybrid_base_sleeve(context, prices)
    else:
        base_values = _current_hybrid_base_values(context, prices)

    enhance_values = _compute_hybrid_enhance_values(context, prices, base_values)

    targets = {}
    for code, value in base_values.items():
        targets[code] = targets.get(code, 0.0) + value
    for code, value in enhance_values.items():
        targets[code] = targets.get(code, 0.0) + value

    context.hybrid_last_rebalance = context.now
    context.hybrid_days_since_rebalance = 0
    context.pool_changed = False
    return {code: value for code, value in targets.items() if value > 1}


def _compute_low_vol_rebalance_hybrid_values(context):
    """Full-rebalance low-vol hybrid: base equal weight + low-vol tilt."""
    pool = list(context.pool)
    prices = _prices_for(pool)
    tradable_pool = [code for code in pool if code in prices]

    if not tradable_pool:
        context.pool_changed = False
        return {}

    total_value = context.portfolio.total_value
    base_ratio = float(context.params.get('base_ratio', 0.5))
    enhance_ratio = float(context.params.get('enhance_ratio', 0.5))
    low_vol_days = int(context.params.get('low_vol_days', 60))
    top_n = int(context.params.get('enhance_top_n', 2))

    targets = {}
    base_budget = max(0.0, total_value * base_ratio)
    base_per_stock = base_budget / len(tradable_pool)
    for code in tradable_pool:
        targets[code] = base_per_stock

    candidates = _low_vol_candidates(tradable_pool, low_vol_days, top_n)
    enhance_budget = max(0.0, total_value * enhance_ratio)
    if candidates and enhance_budget > 0:
        enhance_per_stock = enhance_budget / len(candidates)
        for code in candidates:
            targets[code] = targets.get(code, 0.0) + enhance_per_stock
    elif enhance_budget > 0:
        # If volatility data is unavailable, stay fully invested through equal weight.
        fallback_extra = enhance_budget / len(tradable_pool)
        for code in tradable_pool:
            targets[code] = targets.get(code, 0.0) + fallback_extra

    context.hybrid_last_rebalance = context.now
    context.hybrid_days_since_rebalance = 0
    context.pool_changed = False
    logger.info(
        f"[low_vol_reb] candidates={candidates} "
        f"base={base_ratio:.0%} enhance={enhance_ratio:.0%}"
    )
    return {code: value for code, value in targets.items() if value > 1}


def _position_quantity(pos):
    return getattr(pos, 'quantity', 0) if pos else 0


def _position_market_value(pos):
    return getattr(pos, 'market_value', 0.0) if pos else 0.0


def _is_holding(context, code):
    pos = context.portfolio.positions.get(code)
    return _position_quantity(pos) > 0


def _slot_cash(context, code):
    slot = context.stock_slots.setdefault(code, {'cash': 0.0})
    return float(slot.get('cash', 0.0))


def _set_slot_cash(context, code, value):
    context.stock_slots.setdefault(code, {'cash': 0.0})['cash'] = max(0.0, float(value))


def _is_buy_and_hold_ema_mode(context):
    return context.params['mode'] in ('buy_and_hold_ema', 'buy_and_hold_ema60')


def _sell_buy_hold_low_vol_hybrid(context):
    if not _should_rebalance_hybrid(context):
        context.hybrid_pending_values = None
        return

    targets = _compute_buy_hold_low_vol_hybrid_values(context)
    context.hybrid_pending_values = targets

    holdings = [c for c in context.portfolio.positions
                if context.portfolio.positions[c].quantity > 0]
    for code in holdings:
        pos = context.portfolio.positions.get(code)
        current_value = _position_market_value(pos)
        target_value = targets.get(code, 0.0)
        tolerance = max(100.0, target_value * 0.001)
        if current_value > target_value + tolerance:
            order_target_value(code, target_value)
            logger.info(
                f"[bh_low_vol] sell/reduce {code} "
                f"{current_value:.2f}->{target_value:.2f}"
            )


def _buy_buy_hold_low_vol_hybrid(context):
    targets = getattr(context, 'hybrid_pending_values', None)
    if targets is None:
        return

    for code, target_value in targets.items():
        pos = context.portfolio.positions.get(code)
        current_value = _position_market_value(pos)
        tolerance = max(100.0, target_value * 0.001)
        if current_value + tolerance >= target_value:
            continue
        order_target_value(code, target_value)
        logger.info(
            f"[bh_low_vol] buy/add {code} "
            f"{current_value:.2f}->{target_value:.2f}"
        )

    context.hybrid_pending_values = None


def _sell_low_vol_rebalance_hybrid(context):
    if not _should_rebalance_hybrid(context):
        context.hybrid_pending_values = None
        return

    targets = _compute_low_vol_rebalance_hybrid_values(context)
    context.hybrid_pending_values = targets

    holdings = [c for c in context.portfolio.positions
                if context.portfolio.positions[c].quantity > 0]
    for code in holdings:
        pos = context.portfolio.positions.get(code)
        current_value = _position_market_value(pos)
        target_value = targets.get(code, 0.0)
        tolerance = max(100.0, target_value * 0.001)
        if current_value > target_value + tolerance:
            order_target_value(code, target_value)
            logger.info(
                f"[low_vol_reb] sell/reduce {code} "
                f"{current_value:.2f}->{target_value:.2f}"
            )


def _buy_low_vol_rebalance_hybrid(context):
    targets = getattr(context, 'hybrid_pending_values', None)
    if targets is None:
        return

    for code, target_value in targets.items():
        pos = context.portfolio.positions.get(code)
        current_value = _position_market_value(pos)
        tolerance = max(100.0, target_value * 0.001)
        if current_value + tolerance >= target_value:
            continue
        order_target_value(code, target_value)
        logger.info(
            f"[low_vol_reb] buy/add {code} "
            f"{current_value:.2f}->{target_value:.2f}"
        )

    context.hybrid_pending_values = None


def _sell_constrained_rebalance_hybrid(context):
    if not _should_rebalance_hybrid(context):
        context.hybrid_pending_values = None
        return

    targets = _compute_constrained_rebalance_hybrid_values(context)
    context.hybrid_pending_values = targets

    holdings = [c for c in context.portfolio.positions
                if context.portfolio.positions[c].quantity > 0]
    for code in holdings:
        pos = context.portfolio.positions.get(code)
        current_value = _position_market_value(pos)
        target_value = targets.get(code, 0.0)
        tolerance = max(100.0, target_value * 0.001)
        if current_value > target_value + tolerance:
            order_target_value(code, target_value)
            logger.info(
                f"[{_hybrid_log_tag(context)}] sell/reduce {code} "
                f"{current_value:.2f}->{target_value:.2f}"
            )


def _buy_constrained_rebalance_hybrid(context):
    targets = getattr(context, 'hybrid_pending_values', None)
    if targets is None:
        return

    for code, target_value in targets.items():
        pos = context.portfolio.positions.get(code)
        current_value = _position_market_value(pos)
        tolerance = max(100.0, target_value * 0.001)
        if current_value + tolerance >= target_value:
            continue
        order_target_value(code, target_value)
        logger.info(
            f"[{_hybrid_log_tag(context)}] buy/add {code} "
            f"{current_value:.2f}->{target_value:.2f}"
        )

    context.hybrid_pending_values = None


def _ema_log_tag(context):
    ema_days = int(context.params.get('ema_days', 60))
    return f"ema{ema_days}-slot"


def _ema_signal(context, code):
    """Return True if close is above EMA, False if below, None if data is not enough."""
    ema_days = int(context.params.get('ema_days', 60))
    close = history_bars(code, ema_days + 1, '1d', 'close', adjust_type='pre')
    if close is None or len(close) < ema_days:
        return None

    close = pd.Series(np.asarray(close, dtype=float))
    ema = close.ewm(span=ema_days, adjust=False).mean().iloc[-1]
    return float(close.iloc[-1]) >= float(ema)


def _ema_account_value(context, code):
    pos = context.portfolio.positions.get(code)
    if _position_quantity(pos) > 0:
        return float(_position_market_value(pos))
    return _slot_cash(context, code)


def _reduce_ema_account(context, code, value):
    if value <= 0:
        return

    if _is_holding(context, code):
        pos = context.portfolio.positions.get(code)
        target_value = max(0.0, _position_market_value(pos) - value)
        order_target_value(code, target_value)
        logger.info(f"[{_ema_log_tag(context)}] reduce {code} to {target_value:.2f}")
    else:
        _set_slot_cash(context, code, _slot_cash(context, code) - value)


def _add_ema_account(context, code, value):
    if value <= 0:
        return

    if _is_holding(context, code):
        pos = context.portfolio.positions.get(code)
        context.ema_pending_targets[code] = _position_market_value(pos) + value
    else:
        _set_slot_cash(context, code, _slot_cash(context, code) + value)


def _sync_buy_and_hold_ema_pool(context):
    """Handle pool changes while keeping each stock's EMA cash slot isolated."""
    context.ema_pending_targets = {}
    context.stock_slots = getattr(context, 'stock_slots', {}) or {}

    new_pool = list(context.pool)
    old_pool = list(getattr(context, 'old_pool', []) or [])

    if not new_pool:
        holdings = [c for c in context.portfolio.positions
                    if context.portfolio.positions[c].quantity > 0]
        for code in holdings:
            order_target_value(code, 0)
            logger.info(f"[{_ema_log_tag(context)}] sell removed {code}")
        context.stock_slots.clear()
        context.old_pool = []
        context.pool_changed = False
        context.last_rebalance = context.now
        return

    if not old_pool:
        cash_per_stock = context.portfolio.total_value / len(new_pool)
        for code in new_pool:
            _set_slot_cash(context, code, cash_per_stock)
        context.old_pool = list(new_pool)
        context.pool_changed = False
        context.last_rebalance = context.now
        return

    removed = [s for s in old_pool if s not in new_pool]
    kept = [s for s in old_pool if s in new_pool]
    added = [s for s in new_pool if s not in old_pool]

    if not removed and not added:
        context.old_pool = list(new_pool)
        context.pool_changed = False
        return

    old_cash_slots = sum(_slot_cash(context, s) for s in old_pool if not _is_holding(context, s))
    unassigned_cash = max(0.0, context.portfolio.cash - old_cash_slots)
    removed_value = sum(_ema_account_value(context, s) for s in removed)
    available_value = removed_value + unassigned_cash

    for code in removed:
        if _is_holding(context, code):
            order_target_value(code, 0)
            logger.info(f"[{_ema_log_tag(context)}] sell removed {code}")
        context.stock_slots.pop(code, None)

    target_new_value = context.portfolio.total_value / len(new_pool)
    needed_value = target_new_value * len(added)

    for code in added:
        _set_slot_cash(context, code, target_new_value)

    if added and needed_value > available_value:
        shortfall = needed_value - available_value
        if kept:
            per_deduct = shortfall / len(kept)
            for code in kept:
                _reduce_ema_account(context, code, per_deduct)
        else:
            per_added = available_value / len(added) if added else 0.0
            for code in added:
                _set_slot_cash(context, code, per_added)
    else:
        surplus = available_value - needed_value
        if surplus > 0.0001 and kept:
            per_extra = surplus / len(kept)
            for code in kept:
                _add_ema_account(context, code, per_extra)

    context.old_pool = list(new_pool)
    context.pool_changed = False
    context.last_rebalance = context.now


def _sell_buy_and_hold_ema(context):
    _sync_buy_and_hold_ema_pool(context)

    pending = getattr(context, 'ema_pending_targets', {}) or {}
    pool_set = set(context.pool)

    holdings = [c for c in context.portfolio.positions
                if context.portfolio.positions[c].quantity > 0]
    for code in holdings:
        if code not in pool_set:
            order_target_value(code, 0)
            logger.info(f"[{_ema_log_tag(context)}] sell non-pool {code}")

    for code in context.pool:
        context.stock_slots.setdefault(code, {'cash': 0.0})
        above = _ema_signal(context, code)
        if above is None:
            if not _is_holding(context, code):
                pending.pop(code, None)
            continue

        if _is_holding(context, code):
            if above:
                continue
            pos = context.portfolio.positions.get(code)
            _set_slot_cash(context, code, _position_market_value(pos))
            pending.pop(code, None)
            order_target_value(code, 0)
            logger.info(f"[{_ema_log_tag(context)}] sell below ema {code}")
        elif above and _slot_cash(context, code) > 0:
            pending[code] = _slot_cash(context, code)

    context.ema_pending_targets = pending


def _buy_buy_and_hold_ema(context):
    targets = getattr(context, 'ema_pending_targets', None)
    if not targets:
        return

    pool_set = set(context.pool)
    for code, target_value in list(targets.items()):
        if code not in pool_set or target_value <= 0:
            continue
        if _ema_signal(context, code) is not True:
            continue
        order_target_value(code, target_value)
        _set_slot_cash(context, code, 0.0)
        logger.info(f"[{_ema_log_tag(context)}] buy above ema {code} value {target_value:.2f}")

    context.ema_pending_targets = {}


def _compute_targets(context):
    """根据模式计算目标权重 {code: weight}"""
    mode = context.params['mode']
    if mode == 'buy_and_hold':
        return _compute_buy_and_hold(context)

    pool = context.pool
    if not pool:
        return {}

    if mode == 'equal_weight':
        w = 1.0 / len(pool)
        return {c: w for c in pool}

    elif mode == 'momentum_top':
        top_n = context.params['top_n']
        m_days = context.params['m_days']
        decay = context.params['decay_ratio']
        scores = score_all(pool, m_days, decay)
        ranked = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
        candidates = ranked[:top_n]
        w = 1.0 / len(candidates)
        return {c: w for c in candidates}

    elif mode == 'momentum_weighted':
        m_days = context.params['m_days']
        decay = context.params['decay_ratio']
        scores = score_all(pool, m_days, decay)
        valid = {c: s for c, s in scores.items() if s != float('-inf') and s > 0}
        if not valid:
            # 所有都是负动量 → 等权
            w = 1.0 / len(pool)
            return {c: w for c in pool}
        total = sum(valid.values())
        return {c: s / total for c, s in valid.items()}

    elif mode == 'newest':
        top_n = context.params.get('top_n', 1)
        # 按 first_seen 降序 = 最新加入的排前面
        newest = sorted(pool, key=lambda c: context.loader.first_seen(c), reverse=True)
        candidates = newest[:top_n]
        w = 1.0 / len(candidates)
        return {c: w for c in candidates}

    elif mode == 'oldest':
        oldest = min(pool, key=lambda c: context.loader.first_seen(c))
        return {oldest: 1.0}

    return {}


def _apply_threshold(context, old_targets, new_targets):
    """阈值切换：新目标必须显著优于旧目标才切换"""
    threshold = context.params.get('switch_threshold', 1.0)
    if threshold <= 1.0:
        return new_targets
    # 仅对 momentum_top 模式下 1 只持仓有意义
    if len(old_targets) != 1 or len(new_targets) != 1:
        return new_targets
    old_code = list(old_targets.keys())[0]
    new_code = list(new_targets.keys())[0]
    if old_code == new_code:
        return new_targets
    m_days = context.params['m_days']
    decay = context.params['decay_ratio']
    old_score = momentum_score(old_code, m_days, decay)
    new_score = momentum_score(new_code, m_days, decay)
    if old_score == float('-inf') or new_score == float('-inf'):
        return new_targets
    required = old_score + abs(old_score) * (threshold - 1)
    if new_score <= required:
        logger.info(f"[阈值] 维持 {old_code}({old_score:.4f}) "
                    f"新目标 {new_code}({new_score:.4f}) 未超 {required:.4f}")
        return old_targets
    logger.info(f"[阈值] 切换 {old_code} → {new_code}")
    return new_targets


def _should_rebalance(context):
    """判断今天是否需要调仓"""
    if context.pool_changed:
        return True
    mode = context.params['mode']
    # equal_weight / newest / oldest / buy_and_hold 只在池变化时调仓
    if mode in ('equal_weight', 'newest', 'oldest', 'buy_and_hold'):
        return False
    # momentum 模式按 rebalance_days 周期调仓
    if context.last_rebalance is None:
        return True
    days_passed = (context.now - context.last_rebalance).days
    return days_passed >= context.params.get('rebalance_days', 5)


def sell_trade(context, bar_dict):
    """计算目标并卖出非持有标的"""
    if _is_low_vol_rebalance_hybrid_mode(context):
        _sell_low_vol_rebalance_hybrid(context)
        return

    if _is_constrained_rebalance_hybrid_mode(context):
        _sell_constrained_rebalance_hybrid(context)
        return

    if _is_buy_hold_low_vol_hybrid_mode(context):
        _sell_buy_hold_low_vol_hybrid(context)
        return

    if _is_buy_and_hold_ema_mode(context):
        _sell_buy_and_hold_ema(context)
        return

    if not _should_rebalance(context):
        context.pending_targets = None
        return

    new_targets = _compute_targets(context)

    # 阈值保护
    old_targets = getattr(context, 'targets', {}) or {}
    new_targets = _apply_threshold(context, old_targets, new_targets)

    context.targets = new_targets
    context.pending_targets = new_targets
    context.last_rebalance = context.now
    context.pool_changed = False

    holdings = [c for c in context.portfolio.positions
                if context.portfolio.positions[c].quantity > 0]
    for code in holdings:
        if code not in new_targets:
            order_target_value(code, 0)
            logger.info(f"卖出 {code}")


def buy_trade(context, bar_dict):
    """按目标权重买入"""
    if _is_low_vol_rebalance_hybrid_mode(context):
        _buy_low_vol_rebalance_hybrid(context)
        return

    if _is_constrained_rebalance_hybrid_mode(context):
        _buy_constrained_rebalance_hybrid(context)
        return

    if _is_buy_hold_low_vol_hybrid_mode(context):
        _buy_buy_hold_low_vol_hybrid(context)
        return

    if _is_buy_and_hold_ema_mode(context):
        _buy_buy_and_hold_ema(context)
        return

    targets = getattr(context, 'pending_targets', None)
    if targets is None:
        return

    if not targets:
        context.pending_targets = None
        return

    total_value = context.portfolio.total_value
    holdings = {c: context.portfolio.positions[c].quantity > 0
                for c in context.portfolio.positions}

    for code, weight in targets.items():
        target_val = total_value * weight
        order_target_value(code, target_val)
        if not holdings.get(code):
            logger.info(f"买入 {code} 权重 {weight:.2%}")
    context.pending_targets = None


def handle_bar(context, bar_dict):
    pass
