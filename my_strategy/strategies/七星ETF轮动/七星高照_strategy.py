#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
七星高照ETF轮动策略 V1.8 — RQAlpha 版本

原策略来源：聚宽 https://www.joinquant.com/post/70855
原作者：屌丝逆袭量化
优化时间：2026-3-29

策略逻辑：
- 基于加权动量得分(R² × 年化收益率)对ETF池进行排名
- 支持盈利保护、溢价率过滤、成交量过滤、短期动量过滤
- 无目标时保持空仓

迁移说明：
- 溢价率过滤依赖基金净值数据（RQAlpha原生不支持），默认关闭
- 成交量放量过滤使用日线级别近似替代JoinQuant的分钟线累积
"""

import math
import numpy as np
from abc import ABC, abstractmethod

from rqalpha.api import (
    history_bars,
    order_target_value,
    logger,
    scheduler,
    current_snapshot,
    is_suspended,
    instruments,
    get_previous_trading_date,
)
from rqalpha.mod.rqalpha_mod_sys_scheduler.scheduler import physical_time


# ============================================================
# 【平台适配层】
# ============================================================

class PlatformAdapter:
    """统一平台 API 访问入口"""

    @staticmethod
    def get_bars(code, n, field, include_now=False):
        """获取最近 n 日指定字段，返回 numpy 1-D array"""
        arr = history_bars(code, n, '1d', field, include_now=include_now)
        return np.asarray(arr) if arr is not None else None

    @staticmethod
    def get_current_price(code):
        return current_snapshot(code).last

    @staticmethod
    def get_snapshot(code):
        return current_snapshot(code)

    @staticmethod
    def check_suspended(code):
        return is_suspended(code)

    @staticmethod
    def get_instrument(code):
        return instruments(code)

    @staticmethod
    def get_name(code):
        try:
            return instruments(code).symbol
        except Exception:
            logger.warn(f'获取{code} 信息失败')
            return code

    @staticmethod
    def get_position(context, code):
        return context.portfolio.positions.get(code)

    @staticmethod
    def get_holdings(context, pool_codes):
        """获取在池内的有持仓代码列表"""
        return [
            c for c in context.portfolio.positions
            if c in pool_codes
        ]

    @staticmethod
    def order_target(code, value):
        return order_target_value(code, value)

    @staticmethod
    def log_info(msg):
        logger.info(msg)

    @staticmethod
    def log_warn(msg):
        logger.warn(msg)

    @staticmethod
    def log_debug(msg):
        logger.debug(msg)

    @staticmethod
    def register_schedule(fn, hour, minute):
        scheduler.run_daily(fn, time_rule=physical_time(hour=hour, minute=minute))

    @staticmethod
    def prev_trading_date(dt):
        return get_previous_trading_date(dt, n=1)


# ============================================================
# 【打分器】
# ============================================================

class BaseScorer(ABC):
    """打分器抽象基类"""
    name = 'base'

    def __init__(self, combine_weight=1.0, **kw):
        # combine_weight: 多打分器合成时的权重（七星高照只有一个 scorer，恒为 1.0）
        self.combine_weight = combine_weight
        self.params = kw

    @abstractmethod
    def score(self, context, etf_pool) -> dict:
        """返回 {code: score}"""


class MomentumR2Scorer(BaseScorer):
    """加权线性回归 R² × 年化收益率"""
    name = 'momentum_r2'

    def __init__(self, combine_weight=1.0, m_days=25, decay_weight=2.0, extra_fetch=20):
        super().__init__(combine_weight=combine_weight)
        self.m_days = m_days
        # combine_weight: 多打分器合成时的权重，七星高照只有一个 scorer，恒为 1.0
        # decay_weight: 线性回归时序衰减权重 = np.linspace(1, decay_weight, n)
        #   默认 2.0 = 最新数据权重是最旧数据的 2 倍（原版 np.linspace(1, 2, n)）
        #   设为 1.0 即等权回归
        self.decay_weight = decay_weight
        # extra_fetch: 额外获取天数，确保 price_series 长度覆盖过滤器所需
        self.extra_fetch = extra_fetch

    def score(self, context, etf_pool):
        out = {}
        cache = {}
        for etf in etf_pool:
            try:
                # 多取 extra_fetch 天数据，供 ShortMomentumFilter 等过滤器使用
                need_days = self.m_days + self.extra_fetch + 1
                close = PlatformAdapter.get_bars(etf, need_days, 'close')
                if close is None or len(close) < self.m_days:
                    out[etf] = float('-inf')
                    continue

                current_price = PlatformAdapter.get_current_price(etf)
                price_series = np.append(close, current_price)
                recent = price_series[-(self.m_days + 1):]
                y = np.log(recent)
                x = np.arange(len(y))

                if self.decay_weight > 1.0:
                    w = np.linspace(1.0, self.decay_weight, len(y))
                    slope, intercept = np.polyfit(x, y, 1, w=w)
                    y_pred = slope * x + intercept
                    ss_res = np.sum(w * (y - y_pred) ** 2)
                    y_mean = np.average(y, weights=w)
                    ss_tot = np.sum(w * (y - y_mean) ** 2)
                    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
                else:
                    slope, intercept = np.polyfit(x, y, 1)
                    y_pred = slope * x + intercept
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

                ann = math.exp(slope * 250) - 1
                score = ann * r2

                # 缓存中间结果供过滤器复用
                cache[etf] = {
                    'annualized_returns': ann,
                    'r_squared': r2,
                    'price_series': price_series,
                    'current_price': current_price,
                }
                out[etf] = score
            except Exception as e:
                PlatformAdapter.log_warn(f"计算{etf} {PlatformAdapter.get_name(etf)}时出错: {e}")
                out[etf] = float('-inf')
        context._scorer_cache = cache
        return out


# ============================================================
# 【过滤器】
# ============================================================

class BaseFilter(ABC):
    """过滤器抽象基类"""
    name = 'base'

    def __init__(self, enabled=True, **kw):
        self.enabled = enabled
        self.params = kw

    @abstractmethod
    def filter(self, context, ranked_list, scores) -> list:
        """返回过滤后的 ETF 列表"""


class ProfitProtectionFilter(BaseFilter):
    """盈利保护过滤：当前价从近期最高点回撤超阈值则排除"""
    name = 'profit_protection'

    def __init__(self, enabled=True, lookback=1, threshold=0.05):
        super().__init__(enabled=enabled)
        self.lookback = lookback
        self.threshold = threshold

    def check(self, code, context):
        """检查单只ETF是否触发盈利保护（供独立检查复用）"""
        if not self.enabled:
            return False
        high = PlatformAdapter.get_bars(code, self.lookback, 'high')
        if high is None or len(high) < self.lookback:
            PlatformAdapter.log_debug(
                f"{code} {PlatformAdapter.get_name(code)} 历史数据不足{self.lookback}天，无法检查盈利保护"
            )
            return False
        max_high = float(high.max())
        current_price = PlatformAdapter.get_current_price(code)
        if current_price <= max_high * (1 - self.threshold):
            pullback = (1 - current_price / max_high) * 100
            PlatformAdapter.log_info(
                f"🔻 {code} {PlatformAdapter.get_name(code)} 触发盈利保护："
                f"当前价{current_price:.3f}，最近{self.lookback}日最高{max_high:.3f}，"
                f"回撤{pullback:.2f}% > {self.threshold*100:.0f}%"
            )
            return True
        return False

    def filter(self, context, ranked_list, scores):
        if not self.enabled:
            return ranked_list
        result = []
        for etf in ranked_list:
            if self.check(etf, context):
                PlatformAdapter.log_info(
                    f"🚫 {etf} {PlatformAdapter.get_name(etf)} 触发盈利保护，从排名中排除"
                )
            else:
                result.append(etf)
        return result


class VolumeFilter(BaseFilter):
    """成交量放量过滤：放量且年化收益过高则排除"""
    name = 'volume'

    def __init__(self, enabled=True, lookback=5, threshold=2.0, return_limit=1.0):
        super().__init__(enabled=enabled)
        self.lookback = lookback
        self.threshold = threshold
        self.return_limit = return_limit

    def filter(self, context, ranked_list, scores):
        if not self.enabled:
            return ranked_list
        result = []
        for etf in ranked_list:
            try:
                vols = PlatformAdapter.get_bars(etf, self.lookback + 1, 'volume', include_now=True)
                if vols is None or len(vols) < self.lookback + 1:
                    result.append(etf)
                    continue
                today_vol = float(vols[-1])
                avg_vol = float(vols[:-1].mean())
                if avg_vol <= 0:
                    result.append(etf)
                    continue
                ratio = today_vol / avg_vol
                if ratio > self.threshold:
                    PlatformAdapter.log_debug(
                        f"{etf} {PlatformAdapter.get_name(etf)} 成交量比{ratio:.2f} > {self.threshold}"
                    )
                    # 需要年化收益判断是否排除
                    ann = self._get_annualized(context, etf)
                    if ann is not None and ann > self.return_limit:
                        PlatformAdapter.log_info(
                            f"📉 {etf} {PlatformAdapter.get_name(etf)} 成交量放量{ratio:.1f}倍，"
                            f"且年化{ann*100:.1f}% > 阈值{self.return_limit*100:.1f}%，过滤"
                        )
                        continue
                result.append(etf)
            except Exception as e:
                PlatformAdapter.log_warn(f"成交量计算失败 {etf}: {e}")
                result.append(etf)
        return result

    def _get_annualized(self, context, etf):
        """从 scorer 缓存或独立计算年化收益率"""
        cache = getattr(context, '_scorer_cache', {})
        if etf in cache:
            return cache[etf].get('annualized_returns')
        return None


class ShortMomentumFilter(BaseFilter):
    """短期动量过滤：短期动量不足则排除"""
    name = 'short_momentum'

    def __init__(self, enabled=True, lookback_days=10, threshold=0.0):
        super().__init__(enabled=enabled)
        self.lookback_days = lookback_days
        self.threshold = threshold

    def filter(self, context, ranked_list, scores):
        if not self.enabled:
            return ranked_list
        result = []
        for etf in ranked_list:
            name = PlatformAdapter.get_name(etf)
            cache = getattr(context, '_scorer_cache', {})
            entry = cache.get(etf, {})
            price_series = entry.get('price_series')
            if price_series is not None and len(price_series) >= self.lookback_days + 1:
                n = self.lookback_days
                short_ret = price_series[-1] / price_series[-(n + 1)] - 1
                short_annualized = (1 + short_ret) ** (250 / n) - 1
            else:
                short_annualized = 0.0

            if short_annualized < self.threshold:
                PlatformAdapter.log_debug(
                    f"{etf} {name} 短期动量{short_annualized*100:.1f}% "
                    f"< 阈值{self.threshold*100:.1f}%，过滤"
                )
            else:
                result.append(etf)
        return result


class SingleDayLossFilter(BaseFilter):
    """近3日单日跌幅过滤：任一日跌幅超阈值则排除"""
    name = 'single_day_loss'

    def __init__(self, enabled=True, threshold=0.97):
        super().__init__(enabled=enabled)
        self.threshold = threshold

    def filter(self, context, ranked_list, scores):
        if not self.enabled:
            return ranked_list
        result = []
        for etf in ranked_list:
            cache = getattr(context, '_scorer_cache', {})
            entry = cache.get(etf, {})
            price_series = entry.get('price_series')
            if price_series is not None and len(price_series) >= 4:
                day1 = price_series[-1] / price_series[-2]
                day2 = price_series[-2] / price_series[-3]
                day3 = price_series[-3] / price_series[-4]
                if min(day1, day2, day3) < self.threshold:
                    PlatformAdapter.log_info(
                        f"⚠️ {etf} {PlatformAdapter.get_name(etf)} "
                        f"近3日有单日跌幅超{(1 - self.threshold)*100:.1f}%，直接排除"
                    )
                    continue
            result.append(etf)
        return result


class ScoreRangeFilter(BaseFilter):
    """得分范围过滤：得分必须在 (min, max) 区间内"""
    name = 'score_range'

    def __init__(self, enabled=True, min_score=0.0, max_score=100.0):
        super().__init__(enabled=enabled)
        self.min_score = min_score
        self.max_score = max_score

    def filter(self, context, ranked_list, scores):
        if not self.enabled:
            return ranked_list
        result = []
        for etf in ranked_list:
            s = scores.get(etf, float('-inf'))
            if self.min_score < s < self.max_score:
                result.append(etf)
            else:
                PlatformAdapter.log_debug(
                    f"{etf} {PlatformAdapter.get_name(etf)} "
                    f"得分{s:.2f}超出阈值，过滤"
                )
        return result


class PremiumFilter(BaseFilter):
    """溢价率过滤：溢价率过高则排除（需要外部净值数据源，默认关闭）"""
    name = 'premium'

    def __init__(self, enabled=False, threshold=0.20):
        super().__init__(enabled=enabled)
        self.threshold = threshold

    def get_premium_rate(self, code, date):
        """获取溢价率，无法获取时返回 (None, None, None)"""
        price_arr = PlatformAdapter.get_bars(code, 1, 'close')
        if price_arr is None or len(price_arr) == 0:
            return None, None, None
        price = float(price_arr[-1])
        net_value = self._try_get_nav(code, date)
        if net_value is None or net_value <= 0:
            return None, None, None
        premium_rate = (price - net_value) / net_value
        return premium_rate, price, net_value

    def _try_get_nav(self, code, date):
        return None

    def filter(self, context, ranked_list, scores):
        if not self.enabled:
            return ranked_list
        result = []
        prev_date = PlatformAdapter.prev_trading_date(context.now)
        for etf in ranked_list:
            premium, _, _ = self.get_premium_rate(etf, prev_date)
            if premium is not None and premium > self.threshold:
                PlatformAdapter.log_info(
                    f"🚫 {etf} {PlatformAdapter.get_name(etf)} "
                    f"溢价率{premium*100:.2f}% > {self.threshold*100:.0f}%，跳过"
                )
            else:
                result.append(etf)
        return result


# ============================================================
# 【下单执行器】
# ============================================================

def smart_order(context, code, target_value):
    """下单：停牌/涨跌停/最小金额/T+1 检查，调用 order_target_value"""
    name = PlatformAdapter.get_name(code)
    snap = PlatformAdapter.get_snapshot(code)
    price = snap.last

    if PlatformAdapter.check_suspended(code):
        PlatformAdapter.log_info(f"{code} {name} 停牌，跳过")
        return False
    if price == 0:
        PlatformAdapter.log_info(f"{code} {name} 当前价格0，跳过")
        return False

    pos = PlatformAdapter.get_position(context, code)
    current_val = pos.market_value if pos else 0.0

    if target_value > 0 and abs(current_val - target_value) <= target_value * 0.05:
        return False

    if target_value > current_val and price >= snap.limit_up:
        PlatformAdapter.log_info(f"{code} {name} 涨停，跳过买入")
        return False
    if target_value < current_val and price <= snap.limit_down:
        PlatformAdapter.log_info(f"{code} {name} 跌停，跳过卖出")
        return False


    if target_value < current_val:
        sellable = pos.sellable if pos else 0
        if sellable == 0:
            PlatformAdapter.log_info(f"{code} {name} 当天买入不可卖出")
            return False

    if PlatformAdapter.order_target(code, target_value):
        PlatformAdapter.log_info(f"📦 下单: {code} {name} 目标市值{target_value:.2f}")
        return True
    PlatformAdapter.log_warn(f"下单失败: {code} {name}")
    return False


# ============================================================
# 【策略引擎】
# ============================================================

class StrategyEngine:
    """组合打分 + 过滤 + 选股 + 排名缓存"""

    def __init__(self, scorer, filters, top_n, etf_pool):
        self.scorer = scorer
        self.filters = filters
        self.top_n = top_n
        self.etf_pool = etf_pool
        self._cache_date = None
        self._cached_rankings = None
        self._cached_scores = None

    def select(self, context):
        """计算排名、过滤、返回候选ETF列表"""
        today = context.now.date()
        if self._cache_date != today:
            self._refresh_cache(context)
            self._cache_date = today
        else:
            PlatformAdapter.log_debug("使用缓存的ETF排名")

        ranked = list(self._cached_rankings)
        scores = dict(self._cached_scores)

        for f in self.filters:
            if not f.enabled:
                continue
            before = list(ranked)
            ranked = f.filter(context, ranked, scores)
            if before != ranked:
                removed = [e for e in before if e not in ranked]
                if removed:
                    PlatformAdapter.log_debug(
                        f"过滤器[{f.name}] 移除了: {removed}"
                    )

        candidates = ranked[:self.top_n]
        if not candidates:
            PlatformAdapter.log_info("💤 无目标ETF，保持空仓")
        return candidates

    def _refresh_cache(self, context):
        PlatformAdapter.log_info("重新计算ETF排名...")
        etf_metrics = []
        scores = {}

        # 先打分
        raw_scores = self.scorer.score(context, self.etf_pool)

        for etf in self.etf_pool:
            # 未上市过滤
            try:
                if context.now.date() < PlatformAdapter.get_instrument(etf).listed_date.date():
                    continue
            except Exception:
                PlatformAdapter.log_warn(f"未上市过滤 {etf}")
                pass
            # 停牌过滤
            if PlatformAdapter.check_suspended(etf):
                PlatformAdapter.log_debug(f"{etf} {PlatformAdapter.get_name(etf)} 停牌，跳过")
                continue

            score = raw_scores.get(etf, float('-inf'))
            if score == float('-inf'):
                continue

            scores[etf] = score

            cache = getattr(context, '_scorer_cache', {})
            entry = cache.get(etf, {})
            ann = entry.get('annualized_returns', 0)
            r2 = entry.get('r_squared', 0)
            current_price = entry.get('current_price', 0)

            # 短期动量 (从缓存读取，用于日志展示)
            short_lb = context.params['filter_short_momentum']['lookback_days']
            price_series = entry.get('price_series')
            if price_series is not None and len(price_series) >= short_lb + 1:
                n = short_lb
                short_ret = price_series[-1] / price_series[-(n + 1)] - 1
                short_annualized = (1 + short_ret) ** (250 / n) - 1
            else:
                short_annualized = 0.0

            etf_metrics.append({
                'etf': etf,
                'etf_name': PlatformAdapter.get_name(etf),
                'annualized_returns': ann,
                'r_squared': r2,
                'score': score,
                'current_price': current_price,
                'short_annualized': short_annualized,
            })

        etf_metrics.sort(key=lambda x: x['score'], reverse=True)
        self._cached_rankings = [m['etf'] for m in etf_metrics]
        self._cached_metrics = etf_metrics
        self._cached_scores = scores

    def get_rankings(self, context):
        """返回缓存的排名详情列表（用于日志打印）"""
        if self._cache_date != context.now.date():
            self._refresh_cache(context)
            self._cache_date = context.now.date()
        return self._cached_metrics


# ============================================================
# 【交易函数】
# ============================================================

def check_positions(context, bar_dict=None):
    """每日开盘检查持仓状态"""
    for code in list(context.portfolio.positions.keys()):
        pos = context.portfolio.positions[code]
        if pos.quantity > 0:
            PlatformAdapter.log_info(
                f"📊 持仓：{code} {PlatformAdapter.get_name(code)} "
                f"数量{pos.quantity} 成本{pos.avg_price:.3f} "
                f"现价{pos.last_price:.3f}"
            )


def sell_trade(context, bar_dict=None):
    """卖出不符合条件的持仓"""
    PlatformAdapter.log_info("========== 卖出操作开始 ==========")

    candidates = context.engine.select(context)
    target_set = set(candidates)

    for code in list(context.portfolio.positions.keys()):
        if code not in context.params['etf_pool']:
            continue
        if code not in target_set:
            pos = context.portfolio.positions[code]
            if pos.quantity > 0:
                if smart_order(context, code, 0):
                    PlatformAdapter.log_info(f"📤 卖出不在目标的持仓：{code} {PlatformAdapter.get_name(code)}")

    PlatformAdapter.log_info("========== 卖出操作完成 ==========")


def buy_trade(context, bar_dict=None):
    """买入符合条件的ETF，等权分配"""
    PlatformAdapter.log_info("========== 买入操作开始 ==========")

    engine = context.engine
    params = context.params

    # 打印排名前5
    rankings = engine.get_rankings(context)
    top_n = min(5, len(rankings))
    if top_n > 0:
        PlatformAdapter.log_info(f"=== ETF排名前{top_n} ===")
        for i, m in enumerate(rankings[:top_n]):
            PlatformAdapter.log_info(
                f"排名{i+1}: {m['etf']} {m['etf_name']} "
                f"得分{m['score']:.4f} 年化{m['annualized_returns']*100:.2f}% "
                f"R²={m['r_squared']:.4f}"
            )

    candidates = engine.select(context)
    if not candidates:
        PlatformAdapter.log_info("💤 无目标ETF，保持空仓")
        return

    # 安全检查：sell_trade 是否已清完不在目标的持仓
    current_positions = PlatformAdapter.get_holdings(context, params['etf_pool'])
    to_sell = [c for c in current_positions if c not in candidates]
    if to_sell:
        names = [PlatformAdapter.get_name(c) for c in to_sell]
        PlatformAdapter.log_info(
            f"尚有持仓需要卖出：{list(zip(to_sell, names))}，等待卖出完成再买入"
        )
        return

    # 等权分配
    total_val = context.portfolio.total_value
    target_per_etf = total_val / len(candidates)

    for code in candidates:
        pos = PlatformAdapter.get_position(context, code)
        current_val = pos.market_value if pos else 0.0
        if abs(current_val - target_per_etf) > target_per_etf * 0.05 or current_val == 0:
            smart_order(context, code, target_per_etf)

    PlatformAdapter.log_info("========== 买入操作完成 ==========")


# ============================================================
# 【配置层】
# ============================================================

DEFAULT_PARAMS = {
    # === ETF池 ===
    'etf_pool': [
        "518880.XSHG",   # 黄金ETF
        "159985.XSHE",   # 豆粕ETF
        "501018.XSHG",   # 南方原油
        "161226.XSHE",   # 白银LOF
        "513100.XSHG",   # 纳指ETF
        "159915.XSHE",   # 创业板ETF
        "511220.XSHG",   # 城投债ETF
    ],

    # === 核心参数 ===
    'holdings_num': 1,

    # === 打分器 ===
    'scorer': {
        'm_days': 25,
        'decay_weight': 2.0,   # np.linspace(1, decay_weight, n)，最新/最旧权重比
    },

    # === 过滤器 ===
    'filter_profit_protection': {
        'enabled': True, 'lookback': 1, 'threshold': 0.05,
    },
    'filter_volume': {
        'enabled': True, 'lookback': 5, 'threshold': 2, 'return_limit': 1.0,
    },
    'filter_short_momentum': {
        'enabled': True, 'lookback_days': 10, 'threshold': 0.0,
    },
    'filter_single_day_loss': {
        'enabled': True, 'threshold': 0.97,
    },
    'filter_score_range': {
        'min': 0.0, 'max': 100.0,
    },
    'filter_premium': {
        'enabled': False, 'threshold': 0.20,
    },
}

def _to_plain_dict(obj):
    """递归将 RqAttrDict 转为普通 dict"""
    if hasattr(obj, 'items'):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    return obj


def _deep_merge_params(defaults, injected):
    """将注入参数合并到嵌套 DEFAULT_PARAMS 中"""
    params = {k: (v.copy() if isinstance(v, dict) else v) for k, v in defaults.items()}
    for k, v in injected.items():
        if k in params and isinstance(params[k], dict) and isinstance(v, dict):
            params[k].update(v)
        else:
            params[k] = v
    return params


def build_components(params):
    """根据参数字典构建 scorer + filters"""
    sp = params['scorer']
    # 计算额外获取天数：确保 price_series 足够所有过滤器使用
    short_lb = params['filter_short_momentum'].get('lookback_days', 10)
    extra_fetch = max(20, short_lb - sp['m_days'] + 5)
    scorer = MomentumR2Scorer(
        combine_weight=1.0,           # 多打分器合成权重，七星高照只有一个 scorer，无实际作用
        m_days=sp['m_days'],
        decay_weight=sp['decay_weight'],  # np.linspace(1, decay_weight, n)，控制时序衰减
        extra_fetch=extra_fetch,
    )

    filters = []
    filter_defs = [
        ('filter_profit_protection', ProfitProtectionFilter),
        ('filter_volume', VolumeFilter),
        ('filter_short_momentum', ShortMomentumFilter),
        ('filter_single_day_loss', SingleDayLossFilter),
        ('filter_score_range', ScoreRangeFilter),
        ('filter_premium', PremiumFilter),
    ]
    for key, cls in filter_defs:
        cfg = params.get(key, {})
        if not isinstance(cfg, dict):
            continue
        f_params = cfg.copy()
        # 对于有 enabled 字段的，其余是构造参数
        enabled = f_params.pop('enabled', True)
        filters.append(cls(enabled=enabled, **f_params))

    return scorer, filters


# ============================================================
# 【平台入口】
# ============================================================

def _common_init(context):
    # 读取外部注入的参数
    injected = getattr(context, 'strategy_params', None)
    if injected:
        injected = _to_plain_dict(injected)
    else:
        injected = {}

    params = _deep_merge_params(DEFAULT_PARAMS, injected)

    scorer, filters = build_components(params)

    engine = StrategyEngine(
        scorer=scorer,
        filters=filters,
        top_n=params['holdings_num'],
        etf_pool=params['etf_pool'],
    )

    context.params = params
    context.engine = engine

    PlatformAdapter.register_schedule(check_positions, hour=9, minute=10)
    PlatformAdapter.register_schedule(sell_trade, hour=14, minute=0)
    PlatformAdapter.register_schedule(buy_trade, hour=14, minute=1)

    PlatformAdapter.log_info(
        f"策略初始化完成：ETF池{len(params['etf_pool'])}只，"
        f"动量周期{params['scorer']['m_days']}天，持仓{params['holdings_num']}只"
    )
    pp = params['filter_profit_protection']
    PlatformAdapter.log_info(
        f"盈利保护开关：{'开启' if pp['enabled'] else '关闭'}，"
        f"回看周期{pp['lookback']}天，"
        f"回撤阈值{pp['threshold']*100:.0f}%"
    )
    if params['filter_premium']['enabled']:
        PlatformAdapter.log_info(f"溢价率过滤已启用，阈值：{params['filter_premium']['threshold']*100:.0f}%")
    else:
        PlatformAdapter.log_info("溢价率过滤未启用")
    PlatformAdapter.log_info("========== 策略初始化完成 ==========")


def init(context):
    PlatformAdapter.log_info("========== 策略初始化开始 ==========")
    _common_init(context)


def handle_bar(context, bar_dict):
    """所有交易逻辑已通过 scheduler.run_daily 注册"""
    pass
