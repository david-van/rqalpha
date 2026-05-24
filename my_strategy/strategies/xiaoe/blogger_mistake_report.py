#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : david_van
# @Desc   : 博主"错误"量化报告
#
# 目的: 验证博主是否有可被算法套利的系统性错误。
#   1. 出池后股票表现 -> 博主"卖飞"还是"卖对"?
#   2. 进池前股票表现 -> 博主"买在低位"还是"追高"?
#   3. 进池后股票表现 -> 博主"买入及时"还是"买完就套"?
#   4. 案例: TOP "卖飞" / TOP "扛太久"

import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


from my_strategy.common_file import project_root

BUNDLE = r"D:\datas\bundle"
POOL_DIR = Path(project_root) / "my_strategy" / "strategies" / "xiaoe_articles"
RESULT_DIR = Path(project_root) / 'my_strategy' / 'strategies' / 'batch_results' / 'lihai_pool'
RESULT_DIR.mkdir(parents=True, exist_ok=True)

FORWARD_DAYS = [30, 60, 90, 180, 365]
BACKWARD_DAYS = [30, 60, 90]


# ============================================================
def load_pool_timeline(pool_dir):
    rows = []
    for f in Path(pool_dir).glob("*holdings*.csv"):
        df = pd.read_csv(f, encoding="utf-8-sig")
        for _, row in df.iterrows():
            codes_str = str(row.get("持有股票代码", ""))
            if not codes_str or codes_str == "nan":
                continue
            codes = tuple(sorted(c.strip() for c in codes_str.split("|") if c.strip()))
            if codes:
                rows.append({'date': str(row["日期"]).strip(), 'codes': codes})
    df = pd.DataFrame(rows).drop_duplicates('date').sort_values('date').reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    return df


def detect_events(pool_df):
    events = []
    prev_codes = set()
    for _, row in pool_df.iterrows():
        cur_codes = set(row['codes'])
        date = row['date']
        for code in cur_codes - prev_codes:
            events.append({'date': date, 'code': code, 'type': '进池'})
        for code in prev_codes - cur_codes:
            events.append({'date': date, 'code': code, 'type': '出池'})
        prev_codes = cur_codes
    return pd.DataFrame(events)


def load_adj_close(code):
    path = os.path.join(BUNDLE, 'stocks.h5')
    with h5py.File(path, 'r') as f:
        if code not in f:
            return None
        arr = f[code][:]
    df = pd.DataFrame(arr)
    df['date'] = pd.to_datetime(df['datetime'] // 1000000, format='%Y%m%d')
    df = df[['date', 'close']].copy()
    with h5py.File(os.path.join(BUNDLE, 'ex_cum_factor.h5'), 'r') as f:
        if code in f:
            fac = pd.DataFrame(f[code][:])
            ymd = (fac['start_date'] // 1000000).astype('int64').replace(0, 19000101)
            fac['date'] = pd.to_datetime(ymd, format='%Y%m%d', errors='coerce')
            fac = fac.dropna(subset=['date'])[['date', 'ex_cum_factor']].sort_values('date')
            df = pd.merge_asof(df.sort_values('date'), fac, on='date')
            df['ex_cum_factor'] = df['ex_cum_factor'].fillna(1.0)
            df['close'] = df['close'] * df['ex_cum_factor']
    return df.set_index('date')['close'].sort_index()


def shift_trading_days(price_series, base_date, n_days):
    """从 base_date 起向前/后偏移 n 个交易日"""
    idx = price_series.index
    if base_date <= idx[0] or base_date > idx[-1]:
        return None
    pos = idx.searchsorted(base_date, side='right') - 1
    target = pos + n_days
    if target < 0 or target >= len(idx):
        return None
    return price_series.iloc[target]


def compute_event_returns(events_df, price_data):
    rows = []
    for _, ev in events_df.iterrows():
        code = ev['code']
        if code not in price_data:
            continue
        s = price_data[code]
        try:
            p0 = s.asof(ev['date'])
        except Exception:
            continue
        if pd.isna(p0) or p0 == 0:
            continue
        rec = {'date': ev['date'], 'code': code, 'type': ev['type'], 'p0': float(p0)}
        for d in BACKWARD_DAYS:
            pb = shift_trading_days(s, ev['date'], -d)
            rec[f'前{d}d'] = float(p0 / pb - 1) if pb and pb > 0 else np.nan
        for d in FORWARD_DAYS:
            pf = shift_trading_days(s, ev['date'], d)
            rec[f'后{d}d'] = float(pf / p0 - 1) if pf and pf > 0 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    if df.empty or not cols:
        print('  无数据'); return
    rows = []
    for c in cols:
        s = df[c].dropna()
        if s.empty:
            continue
        rows.append({
            '窗口': c,
            '样本': len(s),
            '均值(%)': f"{s.mean() * 100:+.1f}",
            '中位数(%)': f"{s.median() * 100:+.1f}",
            '上涨占比': f"{(s > 0).mean() * 100:.0f}%",
            'P25(%)': f"{s.quantile(0.25) * 100:+.1f}",
            'P75(%)': f"{s.quantile(0.75) * 100:+.1f}",
            '最大(%)': f"{s.max() * 100:+.1f}",
            '最小(%)': f"{s.min() * 100:+.1f}",
        })
    print(pd.DataFrame(rows).to_string(index=False))


def plot_distributions(exits, enters):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, d in enumerate([30, 60, 90]):
        col = f'后{d}d'
        ax = axes[0, i]
        ex_s = exits[col].dropna() * 100
        en_s = enters[col].dropna() * 100
        bins = np.linspace(-50, 100, 31)
        ax.hist(ex_s.clip(-50, 100), bins=bins, alpha=0.7, color='#d62728', label=f'出池后(n={len(ex_s)})')
        ax.hist(en_s.clip(-50, 100), bins=bins, alpha=0.5, color='#2ca02c', label=f'进池后(n={len(en_s)})')
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.axvline(ex_s.median(), color='#d62728', linestyle=':', linewidth=2, label=f'出池后中位 {ex_s.median():.1f}%')
        ax.axvline(en_s.median(), color='#2ca02c', linestyle=':', linewidth=2, label=f'进池后中位 {en_s.median():.1f}%')
        ax.set_title(f'事件后 {d} 个交易日收益分布')
        ax.set_xlabel('收益率 %'); ax.legend(fontsize=8)
    for i, d in enumerate([30, 60, 90]):
        col = f'前{d}d'
        ax = axes[1, i]
        ex_s = exits[col].dropna() * 100
        en_s = enters[col].dropna() * 100
        bins = np.linspace(-50, 100, 31)
        ax.hist(ex_s.clip(-50, 100), bins=bins, alpha=0.7, color='#d62728', label=f'出池前(n={len(ex_s)})')
        ax.hist(en_s.clip(-50, 100), bins=bins, alpha=0.5, color='#2ca02c', label=f'进池前(n={len(en_s)})')
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.axvline(ex_s.median(), color='#d62728', linestyle=':', linewidth=2, label=f'出池前中位 {ex_s.median():.1f}%')
        ax.axvline(en_s.median(), color='#2ca02c', linestyle=':', linewidth=2, label=f'进池前中位 {en_s.median():.1f}%')
        ax.set_title(f'事件前 {d} 个交易日收益分布')
        ax.set_xlabel('收益率 %'); ax.legend(fontsize=8)
    plt.suptitle('博主进出池前后股票表现分布', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = RESULT_DIR / 'blogger_mistake_dist.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'分布图: {out}')


# ============================================================
def main():
    print('=' * 78)
    print('博主错误量化报告')
    print('=' * 78)
    pool_df = load_pool_timeline(POOL_DIR)
    print(f'池子时间范围: {pool_df["date"].min().date()} ~ {pool_df["date"].max().date()}')
    print(f'池子快照数  : {len(pool_df)}')

    events = detect_events(pool_df)
    n_in = (events["type"] == "进池").sum()
    n_out = (events["type"] == "出池").sum()
    print(f'事件总数    : 进池 {n_in}, 出池 {n_out}')

    all_codes = sorted(set(events['code'].unique()))
    print(f'\n加载 {len(all_codes)} 只股票复权价格...')
    price_data = {}
    missing = []
    for code in all_codes:
        s = load_adj_close(code)
        if s is None or len(s) == 0:
            missing.append(code)
        else:
            price_data[code] = s
    if missing:
        print(f'  缺失 {len(missing)} 只: {missing[:5]}...')

    ev_ret = compute_event_returns(events, price_data)
    exits = ev_ret[ev_ret['type'] == '出池'].copy()
    enters = ev_ret[ev_ret['type'] == '进池'].copy()

    # ============ 出池后表现 ============
    print('\n' + '=' * 78)
    print('【1】出池后股票表现 —— 博主"卖飞"还是"卖对"?')
    print('=' * 78)
    print('解读: 中位数为负 = 博主卖对了 (卖完继续跌)')
    print('      中位数为正 = 博主卖飞了 (卖完反而涨)')
    summarize(exits, '后')

    # ============ 出池前表现 ============
    print('\n' + '=' * 78)
    print('【2】出池前股票表现 —— 博主"扛得久不久"?')
    print('=' * 78)
    print('解读: 中位数大幅为负 = 博主扛了大跌才割肉 (价值陷阱拖延症)')
    print('      中位数接近0或为正 = 博主见势不妙就走')
    summarize(exits, '前')

    # ============ 进池前表现 ============
    print('\n' + '=' * 78)
    print('【3】进池前股票表现 —— 博主"抄底"还是"追高"?')
    print('=' * 78)
    print('解读: 中位数为负 = 博主在跌完后买 (抄底型)')
    print('      中位数为正 = 博主在涨上去后买 (追高型)')
    summarize(enters, '前')

    # ============ 进池后表现 ============
    print('\n' + '=' * 78)
    print('【4】进池后股票表现 —— 博主"买完就涨"还是"买完就套"?')
    print('=' * 78)
    print('解读: 中位数为正 = 博主选股+择时双优 (买完真的涨)')
    print('      中位数为负 = 博主选股可以但择时差 (买在了短期顶)')
    summarize(enters, '后')

    # ============ 案例 ============
    print('\n' + '=' * 78)
    print('【案例 A】"卖飞" TOP 10 —— 出池后 90 日涨幅最大的')
    print('=' * 78)
    flew = exits.dropna(subset=['后90d']).sort_values('后90d', ascending=False).head(10)
    show_cols = ['date', 'code', '前60d', '前30d', '后30d', '后60d', '后90d', '后180d']
    print(flew[show_cols].to_string(index=False))

    print('\n' + '=' * 78)
    print('【案例 B】"扛太久" TOP 10 —— 出池前 60 日跌幅最大的')
    print('=' * 78)
    bag = exits.dropna(subset=['前60d']).sort_values('前60d').head(10)
    print(bag[show_cols].to_string(index=False))

    print('\n' + '=' * 78)
    print('【案例 C】"买完就套" TOP 10 —— 进池后 90 日跌幅最大的')
    print('=' * 78)
    stuck = enters.dropna(subset=['后90d']).sort_values('后90d').head(10)
    print(stuck[show_cols].to_string(index=False))

    print('\n' + '=' * 78)
    print('【案例 D】"买完就涨" TOP 10 —— 进池后 90 日涨幅最大的')
    print('=' * 78)
    fly = enters.dropna(subset=['后90d']).sort_values('后90d', ascending=False).head(10)
    print(fly[show_cols].to_string(index=False))

    # 保存
    ev_ret.to_csv(RESULT_DIR / 'blogger_mistake_events.csv', index=False, encoding='utf-8-sig')
    print(f'\n所有事件: {RESULT_DIR / "blogger_mistake_events.csv"}')

    plot_distributions(exits, enters)

    # ============ 综合判读 ============
    print('\n' + '=' * 78)
    print('综合判读 (用于判断算法能否套利)')
    print('=' * 78)

    out_60d_med = exits['后60d'].median()
    out_90d_med = exits['后90d'].median()
    bag_60d_med = exits['前60d'].median()
    enter_60d_med = enters['后60d'].median()
    enter_pre_60d_med = enters['前60d'].median()

    print(f'出池后 60日中位数 : {out_60d_med * 100:+.1f}%   ← 卖对/卖飞')
    print(f'出池后 90日中位数 : {out_90d_med * 100:+.1f}%')
    print(f'出池前 60日中位数 : {bag_60d_med * 100:+.1f}%   ← 是否扛过深跌')
    print(f'进池前 60日中位数 : {enter_pre_60d_med * 100:+.1f}%   ← 抄底/追高')
    print(f'进池后 60日中位数 : {enter_60d_med * 100:+.1f}%   ← 买完表现')


if __name__ == '__main__':
    main()
