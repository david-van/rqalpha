#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2026/4/22 21:34
# @Author  : david_van
# @Desc : ETF 动量轮动策略 (RQAlpha / 聚宽 / PTrade 通用框架)
# @Usage: 迁移到其他平台时，只需：
#         1) 替换顶部 PLATFORM 常量
#         2) 修改 PlatformAdapter 中对应平台分支
#         3) 调整 ETF_POOL 代码后缀

import math
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

# ============================================================
# 【平台标识】迁移时改这里
# ============================================================
PLATFORM = 'rqalpha'   # 可选: 'rqalpha' | 'joinquant' | 'ptrade'


# ============================================================
# 【平台适配层】—— 所有平台相关 API 只出现在这个类里
# ============================================================
class PlatformAdapter:
    """
    屏蔽各平台 API 差异。策略代码只调用本类方法，不直接调用平台函数。
    """

    # ---------- 数据获取 ----------
    @staticmethod
    def get_close(code, n):
        """获取最近 n 日前复权收盘价，返回 numpy 1-D array"""
        if PLATFORM == 'rqalpha':
            from rqalpha.api import history_bars
            arr = history_bars(code, n, '1d', 'close', adjust_type='pre')
            return np.asarray(arr) if arr is not None else None

        elif PLATFORM == 'joinquant':
            # 聚宽：attribute_history
            arr = attribute_history(code, n, '1d', ['close'], df=False, fq='pre')['close']  # noqa
            return np.asarray(arr)

        elif PLATFORM == 'ptrade':
            # PTrade：get_history
            df = get_history(n, '1d', 'close', code, fq='pre', include=False)  # noqa
            return np.asarray(df[code].values)

    # ---------- 账户查询 ----------
    @staticmethod
    def get_position_qty(context, code):
        if PLATFORM == 'rqalpha':
            pos = context.portfolio.positions.get(code)
            return pos.quantity if pos else 0
        elif PLATFORM == 'joinquant':
            pos = context.portfolio.positions.get(code)
            return pos.total_amount if pos else 0
        elif PLATFORM == 'ptrade':
            pos = context.portfolio.positions.get(code)
            return pos.amount if pos else 0

    @staticmethod
    def get_cash(context):
        if PLATFORM == 'rqalpha':
            return context.portfolio.cash
        elif PLATFORM == 'joinquant':
            return context.portfolio.available_cash
        elif PLATFORM == 'ptrade':
            return context.portfolio.cash

    @staticmethod
    def get_holdings(context):
        """返回当前持仓（数量>0）的代码列表"""
        return [c for c in context.portfolio.positions
                if PlatformAdapter.get_position_qty(context, c) > 0]

    # ---------- 下单 ----------
    @staticmethod
    def order_value(code, value):
        """按目标金额调仓，返回是否成功"""
        if PLATFORM == 'rqalpha':
            from rqalpha.api import order_target_value
            return order_target_value(code, value) is not None
        elif PLATFORM == 'joinquant':
            return order_target_value(code, value) is not None  # noqa
        elif PLATFORM == 'ptrade':
            return order_target_value(code, value) is not None  # noqa

    # ---------- 日志 ----------
    @staticmethod
    def log_info(msg):
        if PLATFORM == 'rqalpha':
            from rqalpha.api import logger
            logger.info(msg)
        else:
            log.info(msg)  # noqa  (聚宽/PTrade)

    @staticmethod
    def log_warn(msg):
        if PLATFORM == 'rqalpha':
            from rqalpha.api import logger
            logger.warn(msg)
        else:
            log.warn(msg)  # noqa

    # ---------- 调度注册 ----------
    @staticmethod
    def register_schedule(sell_fn, buy_fn):
        if PLATFORM == 'rqalpha':
            from rqalpha.api import scheduler
            from rqalpha.mod.rqalpha_mod_sys_scheduler.scheduler import physical_time
            scheduler.run_daily(sell_fn, time_rule=physical_time(hour=9, minute=35))
            scheduler.run_daily(buy_fn,  time_rule=physical_time(hour=9, minute=37))
        elif PLATFORM == 'joinquant':
            run_daily(sell_fn, time='09:35')  # noqa
            run_daily(buy_fn,  time='09:37')  # noqa
        elif PLATFORM == 'ptrade':
            # PTrade 的 run_daily 需要在 initialize 中传 context
            run_daily(_ctx_holder['ctx'], sell_fn, time='9:35')  # noqa
            run_daily(_ctx_holder['ctx'], buy_fn,  time='9:37')  # noqa


# ============================================================
# 【打分器】—— 纯业务逻辑，不接触平台 API（只通过 adapter）
# ============================================================
class BaseScorer(ABC):
    name = 'base'
    def __init__(self, weight=1.0, **kw):
        self.weight = weight
        self.params = kw
    @abstractmethod
    def score(self, context, etf_pool) -> dict: ...


class MomentumR2Scorer(BaseScorer):
    """年化收益 × R² 动量打分"""
    name = 'momentum_r2'
    def __init__(self, weight=1.0, m_days=25):
        super().__init__(weight=weight)
        self.m_days = m_days

    def score(self, context, etf_pool):
        out = {}
        for etf in etf_pool:
            try:
                close = PlatformAdapter.get_close(etf, self.m_days)
                if close is None or len(close) < self.m_days:
                    out[etf] = float('-inf'); continue
                y = np.log(close); x = np.arange(len(y))
                slope, intercept = np.polyfit(x, y, 1)
                ann = math.pow(math.exp(slope), 250) - 1
                r2 = 1 - np.sum((y - (slope*x+intercept))**2) / ((len(y)-1)*np.var(y, ddof=1))
                out[etf] = ann * r2
            except Exception as e:
                PlatformAdapter.log_warn(f'{self.name} {etf} 异常: {e}')
                out[etf] = float('-inf')
        return out


class SimpleMomentumScorer(BaseScorer):
    """简单 N 日涨幅"""
    name = 'momentum_simple'
    def __init__(self, weight=1.0, m_days=20):
        super().__init__(weight=weight)
        self.m_days = m_days

    def score(self, context, etf_pool):
        out = {}
        for etf in etf_pool:
            close = PlatformAdapter.get_close(etf, self.m_days)
            if close is None or len(close) < self.m_days:
                out[etf] = float('-inf')
            else:
                out[etf] = (close[-1] / close[0]) - 1
        return out


# ============================================================
# 【过滤器】
# ============================================================
class BaseFilter(ABC):
    name = 'base'
    def __init__(self, enabled=True, **kw):
        self.enabled = enabled
        self.params = kw
    @abstractmethod
    def filter(self, context, ranked_list, scores) -> list: ...


class MinScoreFilter(BaseFilter):
    """分数阈值过滤：低于阈值不买（空仓保护）"""
    name = 'min_score'
    def __init__(self, enabled=True, threshold=0.0):
        super().__init__(enabled=enabled)
        self.threshold = threshold
    def filter(self, context, ranked_list, scores):
        if not self.enabled: return ranked_list
        return [e for e in ranked_list if scores.get(e, float('-inf')) > self.threshold]


class MATrendFilter(BaseFilter):
    """均线趋势过滤：价格必须 > MA(N) 才允许买入"""
    name = 'ma_trend'
    def __init__(self, enabled=True, ma_period=20):
        super().__init__(enabled=enabled)
        self.ma_period = ma_period

    def filter(self, context, ranked_list, scores):
        if not self.enabled:
            return ranked_list
        result = []
        for etf in ranked_list:
            close = PlatformAdapter.get_close(etf, self.ma_period)
            if close is not None and len(close) >= self.ma_period:
                if close[-1] > np.mean(close):
                    result.append(etf)
                else:
                    PlatformAdapter.log_info(f'[MA过滤] {etf} 跌破MA{self.ma_period}，剔除')
        return result


class VolatilityFilter(BaseFilter):
    """波动率过滤：年化波动率超过上限不买"""
    name = 'volatility'
    def __init__(self, enabled=True, n_days=20, max_vol=0.5):
        super().__init__(enabled=enabled)
        self.n_days = n_days
        self.max_vol = max_vol

    def filter(self, context, ranked_list, scores):
        if not self.enabled:
            return ranked_list
        result = []
        for etf in ranked_list:
            close = PlatformAdapter.get_close(etf, self.n_days)
            if close is None or len(close) < self.n_days:
                continue
            ret = np.diff(np.log(close))
            vol = np.std(ret, ddof=1) * np.sqrt(250)
            if vol <= self.max_vol:
                result.append(etf)
            else:
                PlatformAdapter.log_info(f'[波动过滤] {etf} 年化波动 {vol:.2%} > {self.max_vol:.2%}，剔除')
        return result


# ============================================================
# 【策略引擎】—— 合成打分 + 过滤 + 选股
# ============================================================
class StrategyEngine:
    def __init__(self, scorers, filters, top_n=1):
        self.scorers = scorers
        self.filters = filters
        self.top_n = top_n

    def _combine(self, context, etf_pool):
        """多打分器加权合成"""
        all_scores = {etf: 0.0 for etf in etf_pool}
        total_w = sum(s.weight for s in self.scorers) or 1.0
        detail = {}
        for scorer in self.scorers:
            s = scorer.score(context, etf_pool)
            detail[scorer.name] = s
            for etf, v in s.items():
                if v == float('-inf'):
                    all_scores[etf] = float('-inf')
                elif all_scores[etf] != float('-inf'):
                    all_scores[etf] += v * scorer.weight / total_w
        return all_scores, detail

    def select(self, context, etf_pool):
        scores, detail = self._combine(context, etf_pool)
        ranked = sorted(scores.keys(), key=lambda e: scores[e], reverse=True)

        # 打印打分明细
        df = pd.DataFrame(detail)
        df['FINAL'] = pd.Series(scores)
        df = df.sort_values('FINAL', ascending=False)
        PlatformAdapter.log_info('\n' + str(df))

        # 依次过滤
        for f in self.filters:
            if not f.enabled:
                continue
            before = list(ranked)
            ranked = f.filter(context, ranked, scores)
            if before != ranked:
                PlatformAdapter.log_info(f'过滤器[{f.name}]: {before} -> {ranked}')

        return ranked[:self.top_n]


# ============================================================
# 【策略配置】—— 调参/增删组件只改这里
# ============================================================

# ETF 池（迁移平台时按后缀规范调整）
# RQAlpha / 聚宽 : .XSHG / .XSHE
# PTrade         : .SS   / .SZ
ETF_POOL_MAP = {
    'rqalpha':   ['518880.XSHG', '513100.XSHG', '159915.XSHE', '510180.XSHG'],
    'joinquant': ['518880.XSHG', '513100.XSHG', '159915.XSHE', '510180.XSHG'],
    'ptrade':    ['518880.SS',   '513100.SS',   '159915.SZ',   '510180.SS'],
}

# ============================================================
# 【策略配置】—— 改为工厂函数，支持外部参数覆盖
# ============================================================

DEFAULT_PARAMS = {
    'top_n': 1,
    'scorer_momentum_r2': {'enabled': True, 'weight': 1.0, 'm_days': 25},
    'scorer_momentum_simple': {'enabled': False, 'weight': 0.3, 'm_days': 20},
    'filter_min_score': {'enabled': False, 'threshold': 0.0},
    'filter_ma_trend':  {'enabled': False, 'ma_period': 20},
    'filter_volatility':{'enabled': False, 'n_days': 20, 'max_vol': 0.5},
}

def build_components(params):
    """根据参数字典构建 scorers / filters 列表"""
    scorers = []
    if params['scorer_momentum_r2']['enabled']:
        p = params['scorer_momentum_r2']
        scorers.append(MomentumR2Scorer(weight=p['weight'], m_days=p['m_days']))
    if params['scorer_momentum_simple']['enabled']:
        p = params['scorer_momentum_simple']
        scorers.append(SimpleMomentumScorer(weight=p['weight'], m_days=p['m_days']))

    filters = [
        MinScoreFilter(**params['filter_min_score']),
        MATrendFilter(**params['filter_ma_trend']),
        VolatilityFilter(**params['filter_volatility']),
    ]
    return scorers, filters, params['top_n']

# ============================================================
# 【全局状态】—— PTrade 需要 context 做调度，用个小 holder 过渡
# ============================================================
_ctx_holder = {'ctx': None}


# ============================================================
# 【交易逻辑】—— 纯业务，不接触平台 API（只走 adapter）
# ============================================================
def sell_trade(context, bar_dict=None):
    """9:35 计算排名，卖出非目标"""
    context.target_list = context.engine.select(context, context.etf_pool)
    holdings = PlatformAdapter.get_holdings(context)
    for etf in holdings:
        if etf not in context.target_list:
            if PlatformAdapter.order_value(etf, 0):
                PlatformAdapter.log_info(f'卖出 {etf}')
            else:
                PlatformAdapter.log_warn(f'卖出失败 {etf}')
        else:
            PlatformAdapter.log_info(f'继续持有 {etf}')


def buy_trade(context, bar_dict=None):
    """9:37 买入目标 ETF，可用现金平均分配"""
    holdings = PlatformAdapter.get_holdings(context)
    need = [e for e in context.target_list if e not in holdings]
    if not need:
        return
    cash = PlatformAdapter.get_cash(context)
    value = cash / len(need)
    for etf in need:
        if PlatformAdapter.order_value(etf, value):
            PlatformAdapter.log_info(f'买入 {etf}，目标金额: {value:.2f}')
        else:
            PlatformAdapter.log_warn(f'买入失败 {etf}，可用资金: {cash:.2f}')


# ============================================================
# 【平台入口】—— 三个平台通用的 init / handle_bar
# ============================================================
def _common_init(context):
    # 读取外部注入的参数（批量回测用），没有就用默认值
    params = DEFAULT_PARAMS.copy()
    injected = getattr(context, 'strategy_params', None)
    if injected:
        # 深合并：逐键更新
        for k, v in injected.items():
            if isinstance(v, dict) and k in params:
                params[k].update(v)
            else:
                params[k] = v

    context.etf_pool = ETF_POOL_MAP[PLATFORM]
    context.target_list = []
    context.params = params

    scorers, filters, top_n = build_components(params)
    context.engine = StrategyEngine(scorers, filters, top_n=top_n)

    _ctx_holder['ctx'] = context
    PlatformAdapter.register_schedule(sell_trade, buy_trade)
    PlatformAdapter.log_info(f'策略初始化完成 [平台={PLATFORM}] 参数={params}')


# ---------- RQAlpha 入口 ----------
def init(context):
    _common_init(context)

def handle_bar(context, bar_dict):
    pass


# ---------- 聚宽入口（同名函数，平台切换时各自生效）----------
# 聚宽要求的入口是 `initialize(context)` —— 取消下方注释即可
# def initialize(context):
#     _common_init(context)


# ---------- PTrade 入口 ----------
# def initialize(context):
#     _common_init(context)
# def handle_data(context, data):
#     pass