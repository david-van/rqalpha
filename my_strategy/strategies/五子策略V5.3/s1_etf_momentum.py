# -*- coding: utf-8 -*-
"""
S1 ETF 动量轮动策略
--------------------
对40+只ETF（商品/国际/香港/A股/债券）做加权对数回归动量排名，
选top-1持有，多层过滤（短期动量/连跌/溢价/成交量），无信号切货基。

资金: 50000元 (pindex=0)
"""
from jqdata import *
from functools import wraps
import numpy as np
import pandas as pd
import datetime
import math


# ==================== 简易日志 ====================
def _slog(sid, module, msg, level='info'):
    prefix = f"[{sid}] {module}"
    text = f"{prefix} | {msg}"
    getattr(log, level)(text)


# ==================== 证券名称获取 ====================
def get_security_name(security):
    try:
        info = get_security_info(security)
        return info.display_name if hasattr(info, 'display_name') else info.name
    except Exception:
        return str(security)


# ==================== 策略配置 ====================
def s1_init_config(context):
    g.etf4_strategy = {
        'name': 'ETF轮动', 'pindex': 0,
        'etf_pool_bak': [
            "518880.XSHG",   # 黄金ETF
            "159985.XSHE",   # 豆粕ETF
            "501018.XSHG",   # 南方原油
            "161226.XSHE",   # 白银LOF
            "513100.XSHG",   # 纳指ETF
            "159915.XSHE",   # 创业板ETF
            "511220.XSHG",   # 城投债ETF
        ],
        'etf_pool': [
            # 大宗商品ETF
            "518880.XSHG",  # 黄金ETF
            "159980.XSHE",  # 有色ETF
            "159985.XSHE",  # 豆粕ETF
            "501018.XSHG",  # 南方原油
            "161226.XSHE",  # 白银LOF
            "159981.XSHE",  # 能源化工ETF
            # 国际ETF
            "513100.XSHG",  # 纳指ETF
            "159509.XSHE",  # 纳指科技ETF
            "513290.XSHG",  # 纳指生物ETF
            "513500.XSHG",  # 标普500ETF
            "159529.XSHE",  # 标普消费
            "513400.XSHG",  # 道琼斯ETF
            "513520.XSHG",  # 日经225ETF
            "513030.XSHG",  # 德国30ETF
            "513080.XSHG",  # 法国ETF
            "513310.XSHG",  # 中韩半导体ETF
            "513730.XSHG",  # 东南亚ETF
            # 香港ETF
            "159792.XSHE",  # 港股互联ETF
            "513130.XSHG",  # 恒生科技
            "513050.XSHG",  # 中概互联网ETF
            "159920.XSHE",  # 恒生ETF
            "513690.XSHG",  # 港股红利
            # 指数ETF
            "510300.XSHG",  # 沪深300ETF
            "510500.XSHG",  # 中证500ETF
            "510050.XSHG",  # 上证50ETF
            "510210.XSHG",  # 上证ETF
            "159915.XSHE",  # 创业板ETF
            "588080.XSHG",  # 科创50
            "512100.XSHG",  # 中证1000ETF
            "563360.XSHG",  # A500-ETF
            "563300.XSHG",  # 中证2000ETF
            # 风格ETF
            "512890.XSHG",  # 红利低波ETF
            "159967.XSHE",  # 创业板成长ETF
            "512040.XSHG",  # 价值ETF
            "159201.XSHE",  # 自由现金流ETF
            # 债券ETF
            "511380.XSHG",  # 可转债ETF
            "511010.XSHG",  # 国债ETF
            "511220.XSHG",  # 城投债ETF
        ],
        'lookback_days': 25,
        'holdings_num': 1,
        'min_money': 5000,
        'defensive_etf': "511880.XSHG",     # 银华日利（货基）
        'loss': 0.97,                        # 近3日单日跌幅阈值
        'min_score_threshold': 0,
        'max_score_threshold': 100.0,
        'enable_profit_protection': True,
        'profit_protection_lookback': 1,
        'profit_protection_threshold': 0.05,
        'profit_protection_check_times': ['11:00'],
        'enable_volume_check': True,
        'volume_lookback': 5,
        'volume_threshold': 2,
        'volume_return_limit': 1,
        'use_short_momentum_filter': True,
        'short_lookback_days': 10,
        'short_momentum_threshold': 0.0,
        'enable_premium_filter': True,
        'premium_threshold': 0.20,
        'positions': {},
        'rankings_cache': {'date': None, 'data': None},
        'rebalance_cap': None,
    }


# ==================== 定时任务 ====================
def s1_setup_schedule():
    strategy = g.etf4_strategy
    run_daily(etf4_check_positions, '09:10')
    run_daily(etf4_sell_trade, '13:16')
    run_daily(etf4_buy_trade, '13:21')
    for check_time in strategy.get('profit_protection_check_times', []):
        run_daily(etf4_profit_protection_check, check_time)


# ==================== 盈利保护 ====================
def etf4_profit_protection_check(context):
    """遍历持仓，触发盈利保护则卖出"""
    strategy = g.etf4_strategy
    pindex = strategy['pindex']
    if not strategy.get('enable_profit_protection', False):
        return
    _slog('S1', '盈利保护', '检查开始')
    etf_pool = strategy['etf_pool']
    defensive_etf = strategy.get('defensive_etf', '511880.XSHG')

    for sec in list(context.subportfolios[pindex].positions.keys()):
        if sec not in etf_pool and sec != defensive_etf:
            continue
        pos = context.subportfolios[pindex].positions[sec]
        if pos.total_amount > 0:
            if etf4_check_profit_protection(sec, context, strategy):
                if etf4_smart_order_target_value(sec, 0, context, strategy):
                    _slog('S1', '盈利保护', f'卖出: {sec} {get_security_name(sec)}')
    _slog('S1', '盈利保护', '检查完毕')


def etf4_check_profit_protection(security, context, strategy, lookback=None, threshold=None):
    """从近N日最高点回撤超阈值则触发"""
    if not strategy.get('enable_profit_protection', False):
        return False
    lookback = lookback or strategy.get('profit_protection_lookback', 1)
    threshold = threshold or strategy.get('profit_protection_threshold', 0.05)

    hist = attribute_history(security, lookback, '1d', ['high'])
    if hist.empty or len(hist) < lookback:
        return False

    max_high = hist['high'].max()
    current_price = get_current_data()[security].last_price

    if current_price <= max_high * (1 - threshold):
        _slog('S1', '盈利保护',
              f'{security} {get_security_name(security)} 现价{current_price:.3f}, '
              f'{lookback}日最高{max_high:.3f}, 回撤{(1-current_price/max_high)*100:.2f}%>{threshold*100:.0f}%')
        return True
    return False


# ==================== 溢价率 ====================
def etf4_get_premium_rate(code, date, max_back_days=5):
    """获取最近可用的溢价率 = (场内价/净值 - 1)"""
    price_data = get_price(code, start_date=date, end_date=date, frequency='daily', fields=['close'])
    if price_data.empty:
        return None, None, None

    price = price_data['close'].iloc[0]
    net_value = None
    start_date = date - datetime.timedelta(days=max_back_days * 2)
    trade_days = get_trade_days(start_date=start_date, end_date=date)
    trade_days = [pd.to_datetime(d).date() for d in trade_days]

    for dt in reversed(trade_days):
        if dt > date:
            continue
        net_data = get_extras('unit_net_value', code, start_date=dt, end_date=dt, df=True)
        if not net_data.empty and not pd.isna(net_data[code].iloc[0]):
            net_value = net_data[code].iloc[0]
            break
        try:
            q = query(finance.FUND_NET_VALUE).filter(
                finance.FUND_NET_VALUE.code == code,
                finance.FUND_NET_VALUE.day == dt
            )
            net_df = finance.run_query(q)
            if not net_df.empty:
                net_value = net_df['net_value'].iloc[0]
                break
        except:
            continue

    if net_value is None:
        return None, None, None
    premium_rate = (price - net_value) / net_value
    return premium_rate, price, net_value


# ==================== 成交量比率 ====================
def etf4_get_volume_ratio(context, security, strategy):
    """当日实时成交量 / 过去N日均量，放量超阈值返回比值"""
    lookback = strategy.get('volume_lookback', 5)
    threshold = strategy.get('volume_threshold', 2)
    try:
        hist = attribute_history(security, lookback, '1d', ['volume'])
        if hist.empty or len(hist) < lookback:
            return None
        avg_vol = hist['volume'].mean()
        today = context.current_dt.date()
        df_vol = get_price(security, start_date=today, end_date=context.current_dt,
                           frequency='1m', fields=['volume'], skip_paused=False, fq='pre')
        if df_vol is None or df_vol.empty:
            return None
        current_vol = df_vol['volume'].sum()
        ratio = current_vol / avg_vol if avg_vol > 0 else 0
        if ratio > threshold:
            return ratio
        return None
    except Exception:
        return None


# ==================== 持仓检查 ====================
def etf4_check_positions(context):
    """每日开盘检查持仓状态（仅日志）"""
    strategy = g.etf4_strategy
    pindex = strategy['pindex']
    for sec in context.subportfolios[pindex].positions:
        pos = context.subportfolios[pindex].positions[sec]
        if pos.total_amount > 0:
            _slog('S1', '持仓',
                  f'{sec} {get_security_name(sec)} 数量{pos.total_amount} 成本{pos.avg_cost:.3f} 现价{pos.price:.3f}')


# ==================== 卖出 ====================
def etf4_sell_trade(context):
    """卖出不在目标列表的持仓"""
    strategy = g.etf4_strategy
    pindex = strategy['pindex']
    _slog('S1', '卖出', '开始')

    etf_pool = strategy['etf_pool']
    defensive_etf = strategy.get('defensive_etf', '511880.XSHG')

    ranked = etf4_get_cached_rankings(context, strategy)
    target_etfs = []
    for m in ranked[:strategy['holdings_num']]:
        if m['score'] >= strategy['min_score_threshold']:
            target_etfs.append(m['etf'])

    # 防御模式：无目标且防御可用 → 切货基
    defensive_available = etf4_check_defensive_etf_available(context, strategy)
    if not target_etfs and defensive_available:
        target_etfs = [defensive_etf]

    target_set = set(target_etfs)

    for sec in list(context.subportfolios[pindex].positions.keys()):
        if sec not in etf_pool and sec != defensive_etf:
            continue
        if sec not in target_set:
            pos = context.subportfolios[pindex].positions[sec]
            if pos.total_amount > 0:
                if etf4_smart_order_target_value(sec, 0, context, strategy):
                    _slog('S1', '卖出', f'不在目标: {sec} {get_security_name(sec)}')

    _slog('S1', '卖出', '完毕')


# ==================== 买入 ====================
def etf4_buy_trade(context):
    """买入排名靠前的ETF，等权分配"""
    strategy = g.etf4_strategy
    pindex = strategy['pindex']
    _slog('S1', '买入', '开始')

    etf_pool = strategy['etf_pool']
    defensive_etf = strategy.get('defensive_etf', '511880.XSHG')

    ranked = etf4_get_cached_rankings(context, strategy)

    # 打印排名前5
    _slog('S1', '轮动', 'ETF排名前5')
    for i, m in enumerate(ranked[:5]):
        log.info(f"[S1] 排名{i+1}: {m['etf']} {m['etf_name']} "
                 f"得分{m['score']:.4f} 年化{m['annualized_returns']*100:.2f}% R²={m['r_squared']:.4f}")

    # 确定目标ETF
    target_etfs = []
    for m in ranked:
        if len(target_etfs) >= strategy['holdings_num']:
            break
        target_etfs.append(m['etf'])
        _slog('S1', '轮动', f'目标ETF {len(target_etfs)}: {m["etf"]} {m["etf_name"]} 得分{m["score"]:.4f}')

    if not target_etfs:
        if etf4_check_defensive_etf_available(context, strategy):
            target_etfs = [defensive_etf]
            _slog('S1', '防御', f'进入防御模式: {defensive_etf} {get_security_name(defensive_etf)}')
        else:
            _slog('S1', '买入', '无目标ETF且防御不可用，保持空仓')
            return

    # 检查是否有需要先卖出的持仓
    current_etf_pos = [s for s in context.subportfolios[pindex].positions
                       if s in etf_pool or s == defensive_etf]
    to_sell = [s for s in current_etf_pos if s not in target_etfs]
    if to_sell:
        _slog('S1', '买入', f'尚有{len(to_sell)}只待卖出，等待卖出完成')
        return

    # 等权分配
    total_val = context.subportfolios[pindex].total_value
    target_per_etf = total_val / len(target_etfs)

    if strategy.get('rebalance_cap') is not None:
        cap_per_etf = strategy['rebalance_cap'] / len(target_etfs)
        if target_per_etf > cap_per_etf:
            _slog('S1', '再平衡', f'截流: {target_per_etf:.0f}→{cap_per_etf:.0f}元/只')
            target_per_etf = cap_per_etf

    for etf in target_etfs:
        current_val = 0
        if etf in context.subportfolios[pindex].positions:
            pos = context.subportfolios[pindex].positions[etf]
            if pos.total_amount > 0:
                current_val = pos.total_amount * pos.price

        # 5%容差调仓
        if abs(current_val - target_per_etf) > target_per_etf * 0.05 or current_val == 0:
            if etf4_smart_order_target_value(etf, target_per_etf, context, strategy):
                action = '买入' if current_val < target_per_etf else '调仓'
                _slog('S1', action, f'{etf} {get_security_name(etf)} 目标金额{target_per_etf:.2f}')

    _slog('S1', '买入', '完毕')


# ==================== 排名缓存 ====================
def etf4_get_cached_rankings(context, strategy):
    """日内缓存排名结果"""
    today = context.current_dt.date()
    if strategy['rankings_cache'].get('date') != today:
        _slog('S1', '轮动', '重新计算排名')
        ranked = etf4_get_ranked_etfs(context, strategy)
        strategy['rankings_cache'] = {'date': today, 'data': ranked}
    return strategy['rankings_cache'].get('data', [])


def etf4_get_ranked_etfs(context, strategy):
    """计算所有ETF动量得分，过滤后按得分降序"""
    etf_pool = strategy['etf_pool']
    lookback = max(strategy['lookback_days'], strategy.get('short_lookback_days', 10)) + 20

    hist_df = get_price(etf_pool, end_date=context.previous_date, count=lookback,
                        frequency='1d', fields=['close'], panel=False)
    if hist_df is None or hist_df.empty:
        return []

    close_df = hist_df.pivot(index='time', columns='code', values='close')
    current_data = get_current_data()
    etf_metrics = []

    for etf in etf_pool:
        if current_data[etf].paused:
            continue
        if etf not in close_df.columns:
            continue
        closes = close_df[etf].dropna()
        if len(closes) < strategy['lookback_days']:
            continue

        current_price = current_data[etf].last_price
        metrics = etf4_calculate_momentum(etf, closes, current_price, context, strategy)
        if metrics and strategy['min_score_threshold'] < metrics['score'] < strategy['max_score_threshold']:
            etf_metrics.append(metrics)

    etf_metrics.sort(key=lambda x: x['score'], reverse=True)
    return etf_metrics


# ==================== 动量计算（含全部过滤） ====================
def etf4_calculate_momentum(etf, closes, current_price, context, strategy):
    """计算单只ETF的加权对数回归动量 + 全部过滤"""
    try:
        etf_name = get_security_name(etf)
        lookback = max(strategy['lookback_days'], strategy.get('short_lookback_days', 10)) + 20
        prices = attribute_history(etf, lookback, '1d', ['close', 'high'])
        if len(prices) < strategy['lookback_days']:
            return None
        price_series = np.append(closes.values, current_price)

        # 1. 盈利保护
        if etf4_check_profit_protection(etf, context, strategy):
            return None

        # 2. 溢价率过滤
        if strategy.get('enable_premium_filter', False):
            prev_date = get_trade_days(end_date=context.current_dt.date(), count=2)[0]
            premium, _, _ = etf4_get_premium_rate(etf, prev_date)
            if premium is not None and premium > strategy.get('premium_threshold', 0.20):
                _slog('S1', '溢价过滤',
                      f'{etf} {etf_name} 溢价率{premium*100:.2f}%>{strategy.get("premium_threshold",0.20)*100:.0f}%')
                return None

        # 3. 成交量过滤
        if strategy.get('enable_volume_check', False):
            vol_ratio = etf4_get_volume_ratio(context, etf, strategy)
            if vol_ratio is not None:
                annualized = etf4_get_annualized_returns(price_series, strategy['lookback_days'])
                if annualized > strategy.get('volume_return_limit', 1):
                    _slog('S1', '量能过滤', f'{etf} {etf_name} 放量{vol_ratio:.1f}倍，年化{annualized*100:.1f}%')
                    return None

        # 4. 短期动量过滤
        if strategy.get('use_short_momentum_filter', False):
            short_lookback = strategy.get('short_lookback_days', 10)
            if len(price_series) >= short_lookback + 1:
                short_return = price_series[-1] / price_series[-(short_lookback + 1)] - 1
                short_annualized = (1 + short_return) ** (250 / short_lookback) - 1
            else:
                short_annualized = 0
            if short_annualized < strategy.get('short_momentum_threshold', 0.0):
                return None
        else:
            short_annualized = 0

        # 5. 加权对数回归计算动量得分
        recent_prices = price_series[-(strategy['lookback_days'] + 1):]
        if len(recent_prices) < strategy['lookback_days'] + 1:
            return None

        y = np.log(recent_prices)
        x = np.arange(len(y))
        weights = np.linspace(1, 2, len(y))
        slope, intercept = np.polyfit(x, y, 1, w=weights)
        annualized_returns = math.exp(slope * 250) - 1

        # R² (趋势稳定性)
        ss_res = np.sum(weights * (y - (slope * x + intercept)) ** 2)
        ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0

        score = annualized_returns * r_squared

        # 6. 近3日单日跌幅过滤
        if len(price_series) >= 4:
            day1 = price_series[-1] / price_series[-2]
            day2 = price_series[-2] / price_series[-3]
            day3 = price_series[-3] / price_series[-4]
            if min(day1, day2, day3) < strategy['loss']:
                _slog('S1', '跌幅过滤', f'{etf} {etf_name} 近3日有单日跌幅>{(1-strategy["loss"])*100:.1f}%')
                return None

        return {
            'etf': etf,
            'etf_name': etf_name,
            'annualized_returns': annualized_returns,
            'r_squared': r_squared,
            'score': score,
            'short_annualized': short_annualized,
        }
    except Exception as e:
        log.warning(f'[S1] 计算{etf}动量出错: {e}')
        return None


def etf4_get_annualized_returns(price_series, lookback_days):
    """计算加权年化收益率"""
    recent = price_series[-(lookback_days + 1):]
    y = np.log(recent)
    x = np.arange(len(y))
    weights = np.linspace(1, 2, len(y))
    slope, _ = np.polyfit(x, y, 1, w=weights)
    return math.exp(slope * 250) - 1


# ==================== 防御ETF检查 ====================
def etf4_check_defensive_etf_available(context, strategy):
    """检查货基是否可交易"""
    data = get_current_data()
    etf = strategy.get('defensive_etf', '511880.XSHG')
    if data[etf].paused:
        return False
    if data[etf].last_price >= data[etf].high_limit:
        return False
    if data[etf].last_price <= data[etf].low_limit:
        return False
    return True


# ==================== 智能下单 ====================
def etf4_smart_order_target_value(security, target_value, context, strategy):
    """智能下单调仓：T+1/涨跌停/停牌/最小金额保护"""
    pindex = strategy['pindex']
    current_data = get_current_data()
    name = get_security_name(security)

    if current_data[security].paused:
        return False

    price = current_data[security].last_price
    if price == 0:
        return False

    target_amount = int(target_value / price)
    target_amount = (target_amount // 100) * 100
    if target_amount <= 0 and target_value > 0:
        target_amount = 100

    cur_pos = context.subportfolios[pindex].positions.get(security, None)
    cur_amount = cur_pos.total_amount if cur_pos else 0
    diff = target_amount - cur_amount

    # 涨跌停方向检查
    if diff > 0:
        if current_data[security].last_price >= current_data[security].high_limit:
            return False
    elif diff < 0:
        if current_data[security].last_price <= current_data[security].low_limit:
            return False

    # 最小交易金额
    trade_val = abs(diff) * price
    if 0 < trade_val < strategy['min_money']:
        return False

    # T+1: 卖出时检查可卖数量
    if diff < 0:
        closeable = cur_pos.closeable_amount if cur_pos else 0
        if closeable == 0:
            return False
        diff = -min(abs(diff), closeable)

    if diff != 0:
        if diff < 0:
            # 卖出：用买一价确保成交
            sell_price = None
            try:
                tick = get_current_tick(security)
                if tick is not None and tick.get('b1_p', 0) > 0:
                    sell_price = round(tick['b1_p'], 3)
            except Exception:
                pass
            if sell_price is None:
                sell_price = round(price * 0.99, 3)
            low_limit = getattr(current_data[security], 'low_limit', None)
            if low_limit and sell_price < low_limit:
                sell_price = round(low_limit + 0.001, 3)
            order(security, diff, style=LimitOrderStyle(sell_price), pindex=pindex)
        else:
            order(security, diff, pindex=pindex)
        return True
    return False
