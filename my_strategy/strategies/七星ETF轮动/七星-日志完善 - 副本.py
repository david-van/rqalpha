import numpy as np
import math
import datetime
import pandas as pd
from jqdata import *

# ==================== 全局参数（和策略完全一致） ====================
class g:
    # ETF池
    etf_pool = [
        # 大宗商品ETF
        "518880.XSHG",  # 黄金ETF
        "159980.XSHE",  # 有色ETF
        "159985.XSHE",  # 豆粕ETF
        "501018.XSHG",  # 南方原油
        '161226.XSHE',  # 白银LOF
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
    ]
    
    # 核心参数
    lookback_days = 25               # 动量计算周期
    holdings_num = 1                 # 候选数量
    defensive_etf = "511880.XSHG"    # 防御ETF
    min_money = 5000                 # 最小交易金额

    # 盈利保护
    enable_profit_protection = True
    profit_protection_lookback = 1
    profit_protection_threshold = 0.05

    loss = 0.97                      # 近3日单日跌幅阈值
    min_score_threshold = 0
    max_score_threshold = 100.0

    # 成交量过滤
    enable_volume_check = True
    volume_lookback = 5
    volume_threshold = 2
    volume_return_limit = 1

    # 短期动量过滤
    use_short_momentum_filter = True
    short_lookback_days = 10
    short_momentum_threshold = 0.0

    # 溢价率过滤
    enable_premium_filter = True
    premium_threshold = 0.20

# ==================== 工具函数 ====================
def get_name(security):
    try:
        return get_security_info(security).display_name
    except:
        return "未知"

def get_premium_rate(code, date, max_back_days=5):
    price_data = get_price(code, start_date=date, end_date=date, frequency='daily', fields=['close'])
    if price_data.empty:
        return None, None, None
    price = price_data['close'].iloc[0]

    net_value = None
    used_date = date
    start_date = date - datetime.timedelta(days=max_back_days*2)
    trade_days = get_trade_days(start_date=start_date, end_date=date)
    trade_days = [pd.to_datetime(d).date() for d in trade_days]
    
    for dt in reversed(trade_days):
        if dt > date:
            continue
        net_data = get_extras('unit_net_value', code, start_date=dt, end_date=dt, df=True)
        if not net_data.empty and not pd.isna(net_data[code].iloc[0]):
            net_value = net_data[code].iloc[0]
            used_date = dt
            break
        try:
            q = query(finance.FUND_NET_VALUE).filter(
                finance.FUND_NET_VALUE.code == code,
                finance.FUND_NET_VALUE.day == dt
            )
            net_df = finance.run_query(q)
            if not net_df.empty:
                net_value = net_df['net_value'].iloc[0]
                used_date = dt
                break
        except:
            continue

    if net_value is None:
        return None, None, None

    premium_rate = (price - net_value) / net_value
    return premium_rate, price, net_value

def check_profit_protection(security, lookback=None, threshold=None):
    lookback = lookback or g.profit_protection_lookback
    threshold = threshold or g.profit_protection_threshold
    hist = attribute_history(security, lookback, '1d', ['high'])
    if hist.empty or len(hist) < lookback:
        return False
    max_high = hist['high'].max()
    current_price = get_price(security, end_date=datetime.date.today(), count=1, fields='close', frequency='daily')['close'][0]
    return current_price <= max_high * (1 - threshold)

# def get_volume_ratio(security, current_dt, lookback=None, threshold=None):
#     lookback = lookback or g.volume_lookback
#     threshold = threshold or g.volume_threshold
#     try:
#         hist = attribute_history(security, lookback, '1d', ['volume'])
#         if hist.empty or len(hist) < lookback:
#             return None
#         avg_vol = hist['volume'].mean()
#         df_vol = get_price(security, start_date=current_dt, end_date=current_dt, frequency='1m', fields=['volume'])
#         if df_vol is None or df_vol.empty:
#             return None
#         current_vol = df_vol['volume'].sum()
#         ratio = current_vol / avg_vol if avg_vol > 0 else 0
#         return ratio if ratio > threshold else None
#     except:
#         return None
# 研究代码 - 修复正确的成交量过滤 ✅
def get_volume_ratio(security, current_dt, lookback=None, threshold=None):
    
    lookback = lookback or g.volume_lookback
    threshold = threshold or g.volume_threshold
    try:
        # 用今天 14:00:00（和模拟交易完全一致）
#         currtime = datetime.datetime.combine(
#             datetime.date.today(),
#             datetime.time(14, 0, 0)
#         )
        # 改成当前真实时间（和系统时间完全一致，时分秒都实时）
        currtime = datetime.datetime.now()
        hist = attribute_history(security, lookback, '1d', ['volume'])
        if hist.empty or len(hist) < lookback:
            return None
        avg_vol = hist['volume'].mean()

        # ✅ 恢复 1m 逻辑 + 传入完整时间（关键修复）
        df_vol = get_price(security, start_date=current_dt, end_date=currtime, frequency='1m', fields=['volume'])
        if df_vol is None or df_vol.empty:
            return None
        current_vol = df_vol['volume'].sum()

        ratio = current_vol / avg_vol if avg_vol > 0 else 0

        if ratio > threshold:
            print(f"📉 {security} {get_name(security)} 成交量放量{ratio:.1f}倍，超过阈值{threshold}倍")
            return ratio
        else:
            return None
    except:
        return None
    
def get_annualized_returns(price_series, lookback_days):
    recent = price_series[-(lookback_days + 1):]
    y = np.log(recent)
    x = np.arange(len(y))
    weights = np.linspace(1, 2, len(y))
    slope, _ = np.polyfit(x, y, 1, w=weights)
    return math.exp(slope * 250) - 1

# ==================== 核心动量计算（完整保留策略逻辑） ====================
def calculate_momentum_metrics(etf, current_dt):
    try:
        name = get_name(etf)
        lookback = max(g.lookback_days, g.short_lookback_days) + 20
        prices = attribute_history(etf, lookback, '1d', ['close', 'high'])
        if len(prices) < g.lookback_days:
            return None

        current_price = get_price(etf, end_date=current_dt, count=1, fields='close', frequency='daily')['close'][0]
        price_series = np.append(prices["close"].values, current_price)

        # 盈利保护过滤
        if check_profit_protection(etf):
            print(f"🛡️ 盈利保护过滤: {etf} {get_name(etf)} | 当前价格低于近期高点{int(g.profit_protection_threshold*100)}%")
            return None

        # 溢价率过滤
        if g.enable_premium_filter:
            prev_date = get_trade_days(end_date=current_dt, count=2)[0]
            premium, _, _ = get_premium_rate(etf, prev_date)
            if premium is not None:
                if premium > g.premium_threshold:
                    print(f"❌ 溢价过滤: {etf} {get_name(etf)} | 溢价率: {premium*100:.2f}% > {g.premium_threshold*100:.0f}%")
                    return None
            else:
                # 无法获取溢价率，跳过该过滤条件（不过滤）
                print(f"⚠️ 溢价获取失败: {etf} {get_name(etf)} | 无法获取溢价率，跳过溢价率过滤")
            

#         # 成交量过滤
#         if g.enable_volume_check:
#             vol_ratio = get_volume_ratio(etf, current_dt)
#             if vol_ratio is not None:
#                 annualized = get_annualized_returns(price_series, g.lookback_days)
#                 if annualized > g.volume_return_limit:
#                     return None
        # 研究代码：和模拟交易完全一致的过滤逻辑 + 输出原因
        if g.enable_volume_check:
            vol_ratio = get_volume_ratio(etf, current_dt)
            if vol_ratio is not None:
                annualized = get_annualized_returns(price_series, g.lookback_days)
                if annualized > g.volume_return_limit:
                    print(f"📉 过滤：{etf} {get_name(etf)} 成交量放量{vol_ratio:.1f}倍，且年化{annualized*100:.1f}% > 阈值{g.volume_return_limit*100:.1f}%，过滤")
                    return None

        # 短期动量过滤
        if len(price_series) >= g.short_lookback_days + 1:
            short_return = price_series[-1] / price_series[-(g.short_lookback_days + 1)] - 1
            short_annualized = (1 + short_return) ** (250 / g.short_lookback_days) - 1
        else:
            short_annualized = 0

        if g.use_short_momentum_filter and short_annualized < g.short_momentum_threshold:
            print(f"⏳ 短期动量过滤: {etf} {get_name(etf)} | 短期年化{short_annualized*100:.2f}% < 阈值{g.short_momentum_threshold*100:.0f}%")
            return None

        # 长期动量得分
        recent = price_series[-(g.lookback_days + 1):]
        y = np.log(recent)
        x = np.arange(len(y))
        weights = np.linspace(1, 2, len(y))
        slope, intercept = np.polyfit(x, y, 1, w=weights)
        annualized_returns = math.exp(slope * 250) - 1

        ss_res = np.sum(weights * (y - (slope * x + intercept)) ** 2)
        ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        score = annualized_returns * r_squared
           # ==================== ✅ 这里改成纯手动计算（无 np.polyfit）====================
#         recent = price_series[-(g.lookback_days + 1):]
#         y = np.log(recent)
#         x = np.arange(len(y))
#         weights = np.linspace(1, 2, len(y))

#         # 手动加权线性回归（100% 等价 np.polyfit(x, y, 1, w=weights)）
#         W = weights ** 2  # polyfit 内部会平方权重
#         W_sum = np.sum(W)
#         xw = np.sum(W * x) / W_sum
#         yw = np.sum(W * y) / W_sum
#         cov_xy = np.sum(W * (x - xw) * (y - yw))
#         var_x = np.sum(W * (x - xw) ** 2)

#         if var_x == 0:
#             slope = 0
#         else:
#             slope = cov_xy / var_x
#         intercept = yw - slope * xw

#         # 年化收益
#         annualized_returns = math.exp(slope * 250) - 1

#         # R² 计算（完全和原来一样）
#         y_pred = slope * x + intercept
#         ss_res = np.sum(weights * (y - y_pred) ** 2)
#         ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
#         r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0
#         score = annualized_returns * r_squared
        # ==================== 手动计算结束 ====================

        # 近3日跌幅过滤
        if len(price_series) >= 4:
            day1 = price_series[-1] / price_series[-2]
            day2 = price_series[-2] / price_series[-3]
            day3 = price_series[-3] / price_series[-4]
            min_return = min(day1, day2, day3)
            if min_return < g.loss:
                print(f"📉 跌幅过滤: {etf} {get_name(etf)} | 近3日最小单日收益{(min_return-1)*100:.2f}% < 阈值{(g.loss-1)*100:.0f}%")
                return None

        return {
            'etf': etf,
            'name': name,
            'score': score,
            'annual': annualized_returns,
            'r2': r_squared,
            'short_annual': short_annualized,
            'premium': premium  
        }
    except:
        return None

# ==================== 主运行函数 ====================
def run_etf_rank():
    today = datetime.date.today()
#     today = datetime.date(2026, 5, 7)  # 你想哪天就改哪天
    print(f"========== 七星高照ETF轮动 - 今日选股 {today} ==========\n")
    
    # 计算所有ETF得分
    results = []
    for etf in g.etf_pool:
        res = calculate_momentum_metrics(etf, today)
        if res is not None:
            if g.min_score_threshold < res['score'] < g.max_score_threshold:
                results.append(res)
    
    # 排序
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    
#     # 输出排名
#     print("【ETF排名前10】")
#     for i, item in enumerate(results[:10]):
#         print(f"第{i+1}名 | {item['etf']} {item['name']} | 得分:{item['score']:.4f} | 年化:{item['annual']*100:.2f}% | R²:{item['r2']:.4f}")
    print("【ETF排名前10（含溢价率）】")
#     for i, item in enumerate(results[:10]):
    for i, item in enumerate(results):
        print(f"第{i+1}名 | {item['etf']} {item['name']} | 得分:{item['score']:.4f} | 年化:{item['annual']*100:.2f}% | R²:{item['r2']:.4f} | 溢价率:{item.get('premium', 0)*100:.2f}%")
    
    # 最终选中标的
    target_etfs = []
    for item in results[:g.holdings_num]:
        if item['score'] >= g.min_score_threshold:
            target_etfs.append(item)
    
    if not target_etfs:
        target_etfs = [{'etf': g.defensive_etf, 'name': get_name(g.defensive_etf)}]
    
    print("\n========== 今日最终选中ETF ==========")
    for t in target_etfs:
        print(f"✅ {t['etf']} {t['name']}")
    
    return results, target_etfs

# ==================== 一键运行 ====================
if __name__ == '__main__':
    rank_list, target_list = run_etf_rank()