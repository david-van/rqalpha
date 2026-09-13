#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
小鹅通低波动策略 — 回测脚本

运行（仓库根目录）:
    .venv/Scripts/python.exe -u my_strategy/strategies/xiaoe/lowvol_mktfilter/run_backtest.py

结果输出:
    my_strategy/strategies/batch_results/xiaoe_pool/lowvol_mktfilter/
"""
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from my_strategy.common_file import project_root
from rqalpha import run_file

STRATEGY_FILE = str(
    Path(project_root) / 'my_strategy' / 'strategies' / 'xiaoe' / 'lowvol_mktfilter' / 'strategy.py'
)
RESULT_DIR = Path(project_root) / 'my_strategy' / 'strategies' / 'batch_results' / 'xiaoe_pool' / 'lowvol_mktfilter'
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def make_base_config(tag, start="2020-01-01", end="2026-01-01"):
    return {
        "base": {
            "strategy_file": STRATEGY_FILE,
            "data_bundle_path": r"D:\datas\bundle",
            "start_date": start,
            "end_date": end,
            "frequency": "1d",
            "slippage": "0.0",
            "accounts": {"stock": 100000},
        },
        "mod": {
            "sys_transaction_cost": {
                "enabled": True,
                # 真实成本：佣金万8、最低5元、卖出印花税0.05%
                "stock_commission_multiplier": 1.0,
                "stock_min_commission": 5,
                "tax_multiplier": 1.0,
            },
            "sys_simulation": {
                "enabled": True,
                "matching_type": "current_bar",
                "slippage": 0,
            },
            "sys_analyser": {
                "enabled": True,
                "plot": False,
                "benchmark": "000300.XSHG",
                "output_file": str(RESULT_DIR / f"{tag}.pkl"),
            },
        },
        "extra": {
            "log_level": "error",
            "context_vars": {},
        },
    }


# 默认配置 = 最终推荐参数（10万 + 周频 + MA60滞后 + 真实成本）
FINAL = {}

EXPERIMENTS = [
    ("final", FINAL),
    ("final_nofilter", {"market_filter": False}),
    ("final_daily", {"rebalance_days": 1}),
]


def extract_metrics(tag, result):
    summary = result.get('sys_analyser', {}).get('summary', {})
    trades = result.get('sys_analyser', {}).get('trades')
    cost = None
    n_trades = None
    if trades is not None and len(trades):
        n_trades = len(trades)
        if 'transaction_cost' in trades.columns:
            cost = float(trades['transaction_cost'].fillna(0).sum())
    return {
        'tag': tag,
        'total_returns': summary.get('total_returns'),
        'annualized': summary.get('annualized_returns'),
        'sharpe': summary.get('sharpe'),
        'max_drawdown': summary.get('max_drawdown'),
        'turnover': summary.get('turnover'),
        'trades': n_trades,
        'cost': cost,
    }


def main():
    rows = []
    t0 = time.time()
    for tag, override in EXPERIMENTS:
        t = time.time()
        config = make_base_config(tag)
        config["extra"]["context_vars"] = {"strategy_params": override}
        try:
            result = run_file(STRATEGY_FILE, config=config)
            rows.append(extract_metrics(tag, result))
            print(f"[ok] {tag}  {time.time() - t:.0f}s", flush=True)
        except Exception as e:
            print(f"[fail] {tag}: {e}", flush=True)
            rows.append({'tag': tag, 'error': str(e)})

    df = pd.DataFrame(rows)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = RESULT_DIR / f'summary_{stamp}.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    show = df.copy()
    for c in ('total_returns', 'annualized', 'max_drawdown'):
        show[c] = show[c].map(lambda x: '' if pd.isna(x) else f'{x:.2%}')
    for c in ('sharpe', 'turnover'):
        show[c] = show[c].map(lambda x: '' if pd.isna(x) else f'{x:.2f}')
    if 'cost' in show.columns:
        show['cost'] = show['cost'].map(lambda x: '' if pd.isna(x) else f'{x:.0f}')
    print('\n===== 回测结果 =====')
    print(show.to_string(index=False))
    print(f'\n已保存: {csv_path}')
    print(f'总耗时: {time.time() - t0:.0f}s')
    return df


if __name__ == '__main__':
    main()
