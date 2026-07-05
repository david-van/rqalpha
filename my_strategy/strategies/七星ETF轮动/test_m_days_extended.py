#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""临时测试：行业ETF池拉长动量周期 — 只测 m_days=60/100/120，对比已有m25"""
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from my_strategy.common_file import project_root
from rqalpha import run_file

STRATEGY_FILE = os.path.join(project_root, 'my_strategy/strategies/七星ETF轮动/七星高照_strategy.py')
OUTPUT_DIR = Path(project_root) / 'my_strategy' / 'strategies' / 'batch_results' / '七星高照_指数行业ETF' / 'score_only_m_days'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

TEST_VALUES = [60, 100, 120]


def make_config(tag):
    return {
        "base": {
            "strategy_file": STRATEGY_FILE,
            "data_bundle_path": r"D:\datas\bundle",
            "start_date": "2024-01-01",
            "end_date": "2025-12-31",
            "frequency": "1d",
            "accounts": {"stock": 20000},
        },
        "mod": {
            "sys_transaction_cost": {
                "enabled": True,
                "stock_commission_multiplier": 0.25,
                "stock_min_commission": 0,
                "tax_multiplier": 0,
            },
            "sys_simulation": {
                "enabled": True,
                "matching_type": "current_bar",
                "slippage": 0,
            },
            "sys_analyser": {
                "enabled": True,
                "plot": False,
                "benchmark": "510300.XSHG",
                "output_file": str(OUTPUT_DIR / f"{tag}.pkl"),
            },
        },
        "extra": {
            "log_level": "info",
            "log_file": str(Path(__file__).with_name("backtest.log")),
            "context_vars": {
                "strategy_params": {
                    "filter_profit_protection": {"enabled": False},
                    "filter_volume": {"enabled": False},
                    "filter_short_momentum": {"enabled": False},
                    "filter_single_day_loss": {"enabled": False},
                    "filter_score_range": {"enabled": False},
                    "scorer": {"m_days": None},
                },
            },
        },
    }


def main():
    results = []
    for m_val in TEST_VALUES:
        tag = f"m{m_val:02d}"
        print(f"\n{'='*60}")
        print(f"Running: {tag}")
        print(f"{'='*60}")

        config = make_config(tag)
        config["extra"]["context_vars"]["strategy_params"]["scorer"]["m_days"] = m_val

        try:
            result = run_file(STRATEGY_FILE, config=config)
            s = result.get('sys_analyser', {}).get('summary', {})
            row = {
                'tag': tag,
                'm_days': m_val,
                'total_returns': s.get('total_returns'),
                'annualized': s.get('annualized_returns'),
                'sharpe': s.get('sharpe'),
                'max_drawdown': s.get('max_drawdown'),
                'win_rate': s.get('win_rate'),
                'benchmark_total': s.get('benchmark_total_returns'),
                'benchmark_annu': s.get('benchmark_annualized_returns'),
            }
            results.append(row)
            print(f"  OK: total={row['total_returns']:.4f}, sharpe={row['sharpe']:.4f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({'tag': tag, 'error': str(e)})

    df = pd.DataFrame(results)
    print("\n\n" + "="*60)
    print("结果汇总 (连同已有 m25 基准对比)")
    print("="*60)
    print("已有 m25 (base_off): total=-0.0903, annualized=-0.0480, sharpe=-0.011, max_dd=0.562")
    print(df.to_string(index=False))

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = OUTPUT_DIR / f'summary_{stamp}.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n已保存: {csv_path}")


if __name__ == "__main__":
    main()
