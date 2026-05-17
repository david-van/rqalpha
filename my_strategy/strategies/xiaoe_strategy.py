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
"""

import bisect
import math
from pathlib import Path

import numpy as np
import pandas as pd

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
}


# ============================================================
# 策略入口
# ============================================================
def init(context):
    csv_dir = str(Path(__file__).parent / "xiaoe_articles")
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
    context.stock_slots = {}        # per-stock cash slot for buy_and_hold_ema60
    context.last_rebalance = None   # 上次调仓日期
    context.pool_changed = True     # 首日强制调仓

    logger.info(f"xiaoe_strategy 就绪 mode={params['mode']} pool={context.pool}")

    scheduler.run_daily(sell_trade, time_rule=physical_time(hour=9, minute=35))
    scheduler.run_daily(buy_trade, time_rule=physical_time(hour=9, minute=37))


def before_trading(context):
    """检查池是否变化"""
    new_pool = context.loader.get_pool(context.now)
    if set(new_pool) != set(context.pool):
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
        logger.info(f"[ema60-slot] reduce {code} to {target_value:.2f}")
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


def _sync_buy_and_hold_ema60_pool(context):
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
            logger.info(f"[ema60-slot] sell removed {code}")
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
            logger.info(f"[ema60-slot] sell removed {code}")
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


def _sell_buy_and_hold_ema60(context):
    _sync_buy_and_hold_ema60_pool(context)

    pending = getattr(context, 'ema_pending_targets', {}) or {}
    pool_set = set(context.pool)

    holdings = [c for c in context.portfolio.positions
                if context.portfolio.positions[c].quantity > 0]
    for code in holdings:
        if code not in pool_set:
            order_target_value(code, 0)
            logger.info(f"[ema60-slot] sell non-pool {code}")

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
            logger.info(f"[ema60-slot] sell below ema {code}")
        elif above and _slot_cash(context, code) > 0:
            pending[code] = _slot_cash(context, code)

    context.ema_pending_targets = pending


def _buy_buy_and_hold_ema60(context):
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
        logger.info(f"[ema60-slot] buy above ema {code} value {target_value:.2f}")

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
    if context.params['mode'] == 'buy_and_hold_ema60':
        _sell_buy_and_hold_ema60(context)
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
    if context.params['mode'] == 'buy_and_hold_ema60':
        _buy_buy_and_hold_ema60(context)
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
