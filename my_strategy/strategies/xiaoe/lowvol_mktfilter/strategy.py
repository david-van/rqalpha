#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
小鹅通股票池策略 — 低波动增强 + 大盘趋势风控（周频版）

策略来源：
  博主「疯狂的里海」在小鹅通平台披露的持仓快照（xiaoe_articles/*.csv），
  我们把它视为一个「有 alpha 的股票池」，在其上做量化仓位管理。

策略目标：
  在池子上实现「高绝对收益 + 低回撤」。
  预期（10万资金、2020-2026、真实成本、周频）：总收益 ~346%、最大回撤 ~29%、夏普 ~1.06。

三层逻辑：
  1) 底仓 50%：等权持有当前池内所有可交易股票（吃池子 alpha，熬得住）
  2) 增强仓 50%：额外加仓给池内「60 日波动率最低」的 top2（买得低、不过热）
  3) 风控：沪深300 跌破 60 日均线 → 股票总暴露降到 60%（留 40% 现金）

交易规则：
  - 周频再平衡（每 5 个交易日），或池子变化时触发
  - 先卖后买（9:35 卖出超出的 → 9:37 买入不足的）
  - 下单前做停牌 / 涨跌停 / T+1 检查

关键设计：
  - market_filter_lag=True：用「昨日收盘价」判断均线，消除当日收盘价前视
  - 信号用前复权价（history_bars adjust_type='pre'），成交用真实价（RQAlpha 内部处理）
"""

import bisect
from pathlib import Path

import numpy as np
import pandas as pd

from my_strategy.common_file import project_root
from rqalpha.api import (
    history_bars, order_target_value, logger, scheduler,
    current_snapshot, is_suspended,
)
from rqalpha.mod.rqalpha_mod_sys_scheduler.scheduler import physical_time


# ============================================================
# 【平台适配层】
# ============================================================
class PlatformAdapter:
    """统一平台 API 访问入口"""

    @staticmethod
    def get_bars(code, n, field, adjust_type='pre'):
        arr = history_bars(code, n, '1d', field, adjust_type=adjust_type)
        return np.asarray(arr, dtype=float) if arr is not None else None

    @staticmethod
    def get_snapshot(code):
        return current_snapshot(code)

    @staticmethod
    def get_current_price(code):
        snap = current_snapshot(code)
        return float(snap.last) if snap is not None else None

    @staticmethod
    def check_suspended(code):
        if is_suspended(code):
            return True
        snap = current_snapshot(code)
        return snap is not None and getattr(snap, 'volume', 0) == 0

    @staticmethod
    def get_position(context, code):
        return context.portfolio.positions.get(code)


# ============================================================
# 【动态股票池】
# ============================================================
class PoolLoader:
    """从持仓快照 CSV 加载股票池，支持 as-of 查询（≤dt 的最新快照）"""

    def __init__(self, csv_dir):
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
        self._snapshots.sort(key=lambda x: x[0])
        self._dates = [s[0] for s in self._snapshots]

    def get_pool(self, dt):
        idx = bisect.bisect_right(self._dates, dt) - 1
        return self._snapshots[idx][1] if idx >= 0 else []


# ============================================================
# 【信号】
# ============================================================
def volatility_score(code, days):
    """60 日收益率标准差（前复权），数据不足返回 inf"""
    close = PlatformAdapter.get_bars(code, days + 1, 'close', adjust_type='pre')
    if close is None or len(close) < days + 1:
        return float('inf')
    if np.any(close[:-1] <= 0):
        return float('inf')
    rets = close[1:] / close[:-1] - 1
    rets = rets[np.isfinite(rets)]
    if len(rets) < days:
        return float('inf')
    vol = float(np.std(rets, ddof=1))
    return vol if np.isfinite(vol) else float('inf')


def low_vol_candidates(codes, days, top_n):
    """返回池内波动率最低的 top_n 只"""
    scored = []
    for code in codes:
        vol = volatility_score(code, days)
        if vol != float('inf'):
            scored.append((code, vol))
    scored.sort(key=lambda x: (x[1], x[0]))
    return [code for code, _ in scored[:top_n]]


def market_exposure(context):
    """大盘趋势风控暴露系数：沪深300 跌破 MA → 降仓"""
    p = context.params
    if not p['market_filter']:
        return 1.0
    index = p['market_index']
    ma_days = int(p['market_ma_days'])
    lag = bool(p['market_filter_lag'])
    need = ma_days + (2 if lag else 1)
    close = PlatformAdapter.get_bars(index, need, 'close', adjust_type='pre')
    if close is None or len(close) < need:
        return 1.0
    if lag:
        latest = float(close[-2])                    # 昨日收盘（无前视）
        ma = float(np.mean(close[-ma_days - 1:-1]))  # 昨日为止的均线
    else:
        latest = float(close[-1])
        ma = float(np.mean(close[-ma_days:]))
    if not np.isfinite(latest) or not np.isfinite(ma) or ma <= 0:
        return 1.0
    if latest >= ma:
        return 1.0
    return max(0.0, min(1.0, float(p['weak_market_exposure'])))


# ============================================================
# 【仓位计算】
# ============================================================
def compute_targets(context):
    """计算每只股票的目标市值（底仓等权 + 低波动增强 × 风控暴露）"""
    p = context.params

    # 可交易池：非停牌、有有效价格
    tradable = {}
    for code in context.pool:
        if PlatformAdapter.check_suspended(code):
            continue
        price = PlatformAdapter.get_current_price(code)
        if price is not None and price > 0:
            tradable[code] = price
    if not tradable:
        return {}

    total_value = context.portfolio.total_value
    n = len(tradable)
    base_budget = max(0.0, total_value * p['base_ratio'])
    enhance_budget = max(0.0, total_value * p['enhance_ratio'])

    targets = {code: base_budget / n for code in tradable}

    candidates = low_vol_candidates(list(tradable.keys()), p['low_vol_days'], p['enhance_top_n'])
    if candidates and enhance_budget > 0:
        enhance_per = enhance_budget / len(candidates)
        for code in candidates:
            targets[code] = targets.get(code, 0.0) + enhance_per
    elif enhance_budget > 0:
        for code in tradable:
            targets[code] = targets.get(code, 0.0) + enhance_budget / n

    # 风控暴露
    exposure = market_exposure(context)
    targets = {code: v * exposure for code, v in targets.items()}
    return targets


# ============================================================
# 【下单执行器】
# ============================================================
def smart_order(context, code, target_value):
    """下单前做停牌 / 涨跌停 / T+1 检查，调用 order_target_value"""
    if PlatformAdapter.check_suspended(code):
        logger.info(f"[xiaoe] {code} 停牌，跳过")
        return False

    snap = PlatformAdapter.get_snapshot(code)
    price = float(snap.last) if snap is not None else 0.0
    if price <= 0:
        logger.info(f"[xiaoe] {code} 价格异常，跳过")
        return False

    pos = PlatformAdapter.get_position(context, code)
    current = pos.market_value if pos else 0.0

    # 涨停不买 / 跌停不卖
    limit_up = getattr(snap, 'limit_up', None)
    limit_down = getattr(snap, 'limit_down', None)
    if target_value > current and limit_up is not None and price >= limit_up:
        logger.info(f"[xiaoe] {code} 涨停，跳过买入")
        return False
    if target_value < current and limit_down is not None and price <= limit_down:
        logger.info(f"[xiaoe] {code} 跌停，跳过卖出")
        return False

    # T+1：当天买入不可卖出
    if target_value < current:
        sellable = getattr(pos, 'sellable', None) if pos else None
        if sellable == 0:
            logger.info(f"[xiaoe] {code} 当天买入不可卖(T+1)，跳过")
            return False

    order_target_value(code, target_value)
    return True


# ============================================================
# 【交易函数】
# ============================================================
def should_rebalance(context):
    if context.pool_changed:
        return True
    if getattr(context, 'last_rebalance', None) is None:
        return True
    return context.days_since_rebalance >= int(context.params['rebalance_days'])


def tolerance_of(context, target_value):
    return max(
        float(context.params['rebalance_tolerance_floor']),
        float(target_value) * context.params['rebalance_tolerance_pct'],
    )


def sell_trade(context, bar_dict=None):
    """先卖：卖出超出目标的持仓"""
    if not should_rebalance(context):
        context.pending_targets = None
        return

    targets = compute_targets(context)
    context.pending_targets = targets
    context.last_rebalance = context.now
    context.days_since_rebalance = 0
    context.pool_changed = False

    for code in list(context.portfolio.positions.keys()):
        pos = context.portfolio.positions[code]
        if pos.quantity <= 0:
            continue
        current = pos.market_value
        target = targets.get(code, 0.0)  # 离池股票 target=0 → 清仓
        tol = tolerance_of(context, target)
        if current > target + tol:
            smart_order(context, code, target)


def buy_trade(context, bar_dict=None):
    """后买：买入不足目标的持仓"""
    targets = getattr(context, 'pending_targets', None)
    if not targets:
        return
    for code, target in targets.items():
        pos = PlatformAdapter.get_position(context, code)
        current = pos.market_value if pos else 0.0
        tol = tolerance_of(context, target)
        if current + tol < target:
            smart_order(context, code, target)
    context.pending_targets = None


# ============================================================
# 【策略生命周期】
# ============================================================
def before_trading(context):
    new_pool = context.loader.get_pool(context.now)
    if set(new_pool) != set(context.pool):
        logger.info(f"[xiaoe] 池变化: {len(context.pool)} → {len(new_pool)} 只")
        context.pool = new_pool
        context.pool_changed = True
    context.days_since_rebalance = getattr(context, 'days_since_rebalance', 0) + 1


def handle_bar(context, bar_dict):
    pass


# ============================================================
# 【配置层】
# ============================================================
DEFAULT_PARAMS = {
    # 股票池目录（None = 默认 xiaoe_articles）
    'pool_csv_dir': None,

    # 仓位结构
    'base_ratio': 0.5,        # 底仓比例（全池等权）
    'enhance_ratio': 0.5,     # 增强仓比例（低波动 top N）
    'enhance_top_n': 2,       # 增强仓压给最低波动的 N 只
    'low_vol_days': 60,       # 波动率计算窗口

    # 再平衡
    'rebalance_days': 5,              # 周频（5 个交易日）
    'rebalance_tolerance_pct': 0.001, # 再平衡容差比例
    'rebalance_tolerance_floor': 100.0,  # 容差下限（元）

    # 风控：市场趋势过滤
    'market_filter': True,
    'market_index': '000300.XSHG',
    'market_ma_days': 60,
    'weak_market_exposure': 0.60,
    'market_filter_lag': True,   # 用昨日收盘判断，消除前视
}


def _to_plain_dict(obj):
    if hasattr(obj, 'items'):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    return obj


# ============================================================
# 【平台入口】
# ============================================================
def init(context):
    injected = getattr(context, 'strategy_params', None)
    injected = _to_plain_dict(injected) if injected else {}

    params = dict(DEFAULT_PARAMS)
    params.update(injected)
    context.params = params

    csv_dir = params['pool_csv_dir'] or str(
        Path(project_root) / 'my_strategy' / 'strategies' / 'xiaoe_articles'
    )
    context.loader = PoolLoader(csv_dir)
    context.pool = context.loader.get_pool(context.now)
    context.pool_changed = True
    context.days_since_rebalance = 0
    context.last_rebalance = None
    context.pending_targets = None

    # 日频下所有 run_daily 在同一个 bar 上按注册顺序串行执行，先卖后买。
    scheduler.run_daily(sell_trade, time_rule=physical_time(hour=9, minute=35))
    scheduler.run_daily(buy_trade, time_rule=physical_time(hour=9, minute=37))

    logger.info(
        f"[xiaoe] 就绪 base={params['base_ratio']:.0%} enhance={params['enhance_ratio']:.0%} "
        f"top{params['enhance_top_n']} lowvol{params['low_vol_days']}d "
        f"rebal{params['rebalance_days']}d mkt_ma{params['market_ma_days']} "
        f"filter={params['market_filter']} pool={len(context.pool)}只"
    )
