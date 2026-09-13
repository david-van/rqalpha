#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc : xiaoe 低波动基线 + 组合级风控叠加 —— 收益-回撤前沿扫描
#
# 目标：绝对收益跟得上（≥ buy_and_hold 247%），同时把回撤明显压下来。
# 思路：以 low_vol_rebalance_hybrid（b50/e50/v60/n2/r1，日度）为强收益底座，
#       在其上叠加三类可独立开关的组合级风控：
#         - market_filter  市场趋势过滤（MA 之下降仓）
#         - dd_breaker     回撤熔断（组合自高点回撤超阈值 → 降仓）
#         - vol_target     波动率目标仓位（高波动期降暴露）
#
# 运行（仓库根目录）:
#     .venv/Scripts/python.exe -u my_strategy/strategies/xiaoe/run_xiaoe_risk.py
#
# 输出:
#     my_strategy/strategies/batch_results/xiaoe_pool/risk_overlay/summary_*.csv
#     my_strategy/strategies/batch_results/xiaoe_pool/risk_overlay/frontier_*.png
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from my_strategy.common_file import project_root
from my_strategy.strategies.xiaoe.run_xiaoe import (
    make_base_config, run_one, _override, POOL_DIR, STRATEGY_FILE,
)

RESULT_DIR = Path(project_root) / 'my_strategy' / 'strategies' / 'batch_results' / 'xiaoe_pool' / 'risk_overlay'
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def _lv(**kw):
    """low_vol_rebalance_hybrid 底座（日度 b50/e50/v60/n2）"""
    base = dict(
        base_ratio=0.5,
        enhance_ratio=0.5,
        low_vol_days=60,
        enhance_top_n=2,
        rebalance_days=1,
    )
    base.update(kw)
    return _override("low_vol_rebalance_hybrid", **base)


EXPERIMENTS = [
    # 参照
    ("buy_and_hold", _override("buy_and_hold")),
    # 底座（无风控）
    ("lv_r1_base", _lv()),

    # 市场趋势过滤（MA 之下降仓）
    ("lv_r1_mkt200_w60", _lv(market_filter=True, market_ma_days=200, weak_market_exposure=0.60)),
    ("lv_r1_mkt200_w40", _lv(market_filter=True, market_ma_days=200, weak_market_exposure=0.40)),
    ("lv_r1_mkt120_w60", _lv(market_filter=True, market_ma_days=120, weak_market_exposure=0.60)),
    ("lv_r1_mkt60_w60", _lv(market_filter=True, market_ma_days=60, weak_market_exposure=0.60)),

    # 回撤熔断
    ("lv_r1_dd15_60", _lv(dd_breaker=True, dd_threshold=-0.15, dd_exposure=0.60)),
    ("lv_r1_dd10_50", _lv(dd_breaker=True, dd_threshold=-0.10, dd_exposure=0.50)),

    # 波动率目标
    ("lv_r1_vol20", _lv(vol_target=0.20)),
    ("lv_r1_vol25", _lv(vol_target=0.25)),

    # 组合叠加
    ("lv_r1_mkt200w60_dd15", _lv(market_filter=True, market_ma_days=200, weak_market_exposure=0.60,
                                  dd_breaker=True, dd_threshold=-0.15, dd_exposure=0.60)),
    ("lv_r1_mkt200w60_vol20", _lv(market_filter=True, market_ma_days=200, weak_market_exposure=0.60,
                                   vol_target=0.20)),
]


def extract_cost(result):
    try:
        trades = result.get('sys_analyser', {}).get('trades')
        if trades is not None and len(trades) and 'transaction_cost' in trades.columns:
            return float(trades['transaction_cost'].fillna(0).sum())
    except Exception:
        pass
    return None


def extract_metrics(tag, result):
    summary = result.get('sys_analyser', {}).get('summary', {})
    return {
        'tag': tag,
        'total_returns': summary.get('total_returns'),
        'annualized': summary.get('annualized_returns'),
        'sharpe': summary.get('sharpe'),
        'max_drawdown': summary.get('max_drawdown'),
        'turnover': summary.get('turnover'),
        'cost': extract_cost(result),
    }


def main():
    rows = []
    t_start = time.time()
    for tag, override in EXPERIMENTS:
        t0 = time.time()
        try:
            result = run_one(tag, override)
            rows.append(extract_metrics(tag, result))
            print(f"[ok] {tag}  {time.time() - t0:.0f}s", flush=True)
        except Exception as e:
            print(f"[fail] {tag}: {e}", flush=True)
            rows.append({'tag': tag, 'error': str(e)})

    df = pd.DataFrame(rows)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = RESULT_DIR / f'summary_{stamp}.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # 格式化展示
    show = df.copy()
    for c in ('total_returns', 'annualized', 'max_drawdown'):
        if c in show.columns:
            show[c] = show[c].map(lambda x: '' if pd.isna(x) else f'{x:.2%}')
    if 'sharpe' in show.columns:
        show['sharpe'] = show['sharpe'].map(lambda x: '' if pd.isna(x) else f'{x:.3f}')
    if 'turnover' in show.columns:
        show['turnover'] = show['turnover'].map(lambda x: '' if pd.isna(x) else f'{x:.2f}')
    if 'cost' in show.columns:
        show['cost'] = show['cost'].map(lambda x: '' if pd.isna(x) else f'{x:.0f}')
    print('\n===== 收益-回撤扫描结果 =====')
    print(show.to_string(index=False))
    print(f'\n已保存: {csv_path}')
    print(f'总耗时: {time.time() - t_start:.0f}s')

    # 前沿图
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        valid = df.dropna(subset=['total_returns', 'max_drawdown']).copy()
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.scatter(valid['max_drawdown'] * 100, valid['total_returns'] * 100, s=60, color='#1f77b4')
        for _, r in valid.iterrows():
            ax.annotate(r['tag'], (r['max_drawdown'] * 100, r['total_returns'] * 100),
                        fontsize=8, textcoords='offset points', xytext=(4, 4))
        ax.set_xlabel('最大回撤 (%)')
        ax.set_ylabel('累计收益 (%)')
        ax.set_title('xiaoe 低波动基线 + 风控叠加：收益-回撤前沿')
        ax.grid(True, alpha=0.3)
        png_path = RESULT_DIR / f'frontier_{stamp}.png'
        plt.savefig(png_path, dpi=130, bbox_inches='tight')
        plt.close()
        print(f'前沿图: {png_path}')
    except Exception as e:
        print(f'绘图失败: {e}')

    return df


if __name__ == '__main__':
    main()
