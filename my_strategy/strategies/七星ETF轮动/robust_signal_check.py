#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检验 A：动量分数预测能力 — Spearman 秩相关分析

验证 R² × 年化收益率 这个分数对未来收益是否有预测能力。
不走回测交易，只做统计检验。

运行:
    python my_strategy/strategies/七星ETF轮动/robust_signal_check.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from my_strategy.common_file import project_root
from rqalpha import run_code

ETFS = [
    "518880.XSHG",   # 黄金ETF
    "159985.XSHE",   # 豆粕ETF
    "501018.XSHG",   # 南方原油
    "161226.XSHE",   # 白银LOF
    "513100.XSHG",   # 纳指ETF
    "159915.XSHE",   # 创业板ETF
    "511220.XSHG",   # 城投债ETF
]

RESULT_DIR = Path(project_root) / 'my_strategy' / 'strategies' / 'batch_results' / '七星高照' / 'robustness'
RESULT_DIR.mkdir(exist_ok=True, parents=True)

M_DAYS = 25
DECAY_WEIGHT = 2.0
EXTRA_FETCH = 30

DATA_COLLECTION_CODE = r'''
import numpy as np
import math
import pickle
from rqalpha.api import history_bars, scheduler, logger


def compute_score(prices, m_days, decay_weight):
    if prices is None or len(prices) < m_days:
        return float('-inf'), 0.0, 0.0
    recent = prices[-(m_days + 1):]
    y = np.log(recent)
    x = np.arange(len(y), dtype=float)
    if decay_weight > 1.0:
        w = np.linspace(1.0, decay_weight, len(y))
        slope = np.polyfit(x, y, 1, w=w)[0]
        y_pred = slope * x + np.polyfit(x, y, 1, w=w)[1]
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
    return score, ann, r2


def init(context):
    context.records = []
    context.etf_list = list(context.etf_pool)
    context.m = context.m_days
    context.dw = context.decay_weight
    context.extra = context.extra_fetch
    context.out = context.output_path
    scheduler.run_daily(collect)


def collect(context, bar_dict):
    today = context.now.date().isoformat()
    need = context.m + context.extra
    for etf in context.etf_list:
        try:
            closes = history_bars(etf, need, '1d', 'close')
            if closes is None or len(closes) < context.m:
                continue
            prices = np.asarray(closes)
            score, ann, r2 = compute_score(prices, context.m, context.dw)
            context.records.append({
                'date': today,
                'etf': etf,
                'score': float(score),
                'close': float(prices[-1]),
            })
        except Exception:
            pass


def after_trading(context):
    with open(context.out, 'wb') as f:
        pickle.dump(context.records, f)
'''


def run_data_collection(start_date, end_date, etf_pool, output_path):
    config = {
        "base": {
            "data_bundle_path": r"D:\datas\bundle",
            "start_date": start_date,
            "end_date": end_date,
            "frequency": "1d",
            "accounts": {"stock": 20000},
        },
        "mod": {
            "sys_transaction_cost": {"enabled": True},
            "sys_simulation": {"enabled": True, "matching_type": "current_bar", "slippage": 0},
            "sys_analyser": {"enabled": True, "plot": False},
        },
        "extra": {
            "log_level": "warning",
            "context_vars": {
                "etf_pool": list(etf_pool),
                "m_days": M_DAYS,
                "decay_weight": DECAY_WEIGHT,
                "extra_fetch": EXTRA_FETCH,
                "output_path": str(output_path),
            },
        },
    }
    print(f"  运行数据采集回测 ({start_date} ~ {end_date})...")
    run_code(DATA_COLLECTION_CODE, config=config)


def analyze_signal(records, forward_days_list=(5, 10, 20)):
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])

    close_pivot = df.pivot(index='date', columns='etf', values='close')
    score_pivot = df.pivot(index='date', columns='etf', values='score')

    results = {}
    for n in forward_days_list:
        fwd_ret = close_pivot.shift(-n) / close_pivot - 1
        daily_corrs = []
        for i, date in enumerate(score_pivot.index):
            if i + n >= len(score_pivot.index):
                break
            day_scores = score_pivot.iloc[i]
            day_rets = fwd_ret.iloc[i]
            mask = day_scores.notna() & day_rets.notna() & (~np.isinf(day_scores))
            valid_scores = day_scores[mask]
            valid_rets = day_rets[mask]
            if len(valid_scores) < 5:
                continue
            corr, pval = spearmanr(valid_scores, valid_rets)
            daily_corrs.append({'date': date, 'corr': corr, 'pval': pval, 'n': len(valid_scores)})

        daily_df = pd.DataFrame(daily_corrs)
        mean_corr = daily_df['corr'].mean()
        pct_positive = (daily_df['corr'] > 0).mean()
        sig_pct = (daily_df['pval'] < 0.05).mean()

        rolling = daily_df.set_index('date')['corr'].rolling(60, min_periods=30).mean()

        results[n] = {
            'daily': daily_df,
            'mean_corr': mean_corr,
            'pct_positive': pct_positive,
            'sig_pct': sig_pct,
            'rolling': rolling,
        }
    return results


def run_test_a(start_date="2019-01-01", end_date="2025-12-31"):
    print("\n" + "=" * 60)
    print("检验 A：动量分数预测能力 — Spearman 秩相关分析")
    print("=" * 60)

    pkl_path = RESULT_DIR / "signal_raw_data.pkl"

    print("\n[1/2] 采集数据...")
    run_data_collection(start_date, end_date, ETFS, pkl_path)

    with open(pkl_path, 'rb') as f:
        records = pickle.load(f)
    print(f"  采集完成: {len(records)} 条记录")

    print("\n[2/2] 计算秩相关系数...")
    results = analyze_signal(records)

    print("\n" + "-" * 40)
    print(f"{'前瞻天数':<10} {'均值相关系数':<14} {'正相关占比':<12} {'显著占比(p<0.05)':<18}")
    print("-" * 40)

    summary_rows = []
    for n in (5, 10, 20):
        r = results[n]
        print(f"{n}天{'':<7} {r['mean_corr']:>+.4f}        {r['pct_positive']:>7.1%}        {r['sig_pct']:>7.1%}")
        summary_rows.append({
            'forward_days': n,
            'mean_spearman': r['mean_corr'],
            'pct_positive': r['pct_positive'],
            'pct_significant': r['sig_pct'],
        })

    summary_df = pd.DataFrame(summary_rows)
    csv_path = RESULT_DIR / "signal_spearman_summary.csv"
    summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n汇总: {csv_path}")

    # 滚动相关系数图
    try:
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        for idx, n in enumerate((5, 10, 20)):
            ax = axes[idx]
            r = results[n]
            ax.plot(r['rolling'].index, r['rolling'].values, linewidth=0.8, color='steelblue')
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
            ax.axhline(y=r['mean_corr'], color='red', linestyle='--', linewidth=0.5,
                       label=f'均值={r["mean_corr"]:.3f}')
            ax.set_ylabel(f'{n}天前瞻')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel('日期')
        fig.suptitle('动量分数 vs 未来收益 — 滚动60日 Spearman 相关系数', fontsize=12)
        fig.tight_layout()

        png_path = RESULT_DIR / "signal_rolling_corr.png"
        fig.savefig(png_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"图表: {png_path}")
    except Exception as e:
        print(f"  绘图失败: {e}")

    # 诊断结论
    print("\n" + "=" * 60)
    print("诊断结论")
    print("=" * 60)

    avg_corr = results[10]['mean_corr']
    pct_pos = results[10]['pct_positive']

    if avg_corr > 0.10 and pct_pos > 0.55:
        print("[有效] 动量分数对未来收益有正向预测能力")
        print("  回测收益可能来自真实 alpha，而非过拟合")
    elif avg_corr > 0.05:
        print("[偏弱] 动量分数有一定正向预测力但不够强")
        print("  策略可能有部分真实 alpha，但也可能存在过拟合")
    else:
        print("[无效] 动量分数与未来收益无明显正相关")
        print("  回测收益很可能来自过拟合或其他因素，非信号本身的有效性")

    print(f"\n基准判断: 10天前瞻 Spearman 均值 = {avg_corr:+.4f}, 正相关占比 = {pct_pos:.1%}")
    return results


if __name__ == "__main__":
    start = sys.argv[2] if len(sys.argv) > 3 and sys.argv[1] == '--start' else "2019-01-01"
    end = sys.argv[4] if len(sys.argv) > 5 and sys.argv[3] == '--end' else "2025-12-31"
    run_test_a(start, end)
