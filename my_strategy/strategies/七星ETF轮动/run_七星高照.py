#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
七星高照ETF轮动策略 — 批量回测

运行:
    python my_strategy/strategies/七星ETF轮动/run_七星高照.py

结果输出:
    my_strategy/strategies/batch_results/七星高照/
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime

from my_strategy.common_file import project_root
from rqalpha import run_file


# ==================== 基础配置 ====================
STRATEGY_FILE = os.path.join(project_root, 'my_strategy/strategies/七星ETF轮动/七星高照_strategy.py')
RESULT_DIR = Path(project_root) / 'my_strategy' / 'strategies' / 'batch_results' / '七星高照'
RESULT_DIR.mkdir(exist_ok=True, parents=True)


def make_base_config(tag: str):
    """生成一份基础 config，output_file 按 tag 区分"""
    return {
        "base": {
            "strategy_file": STRATEGY_FILE,
            "data_bundle_path": r"D:\datas\bundle",
            "start_date": "2020-01-01",
            "end_date":   "2026-01-01",
            "frequency":  "1d",
            "accounts":   {"stock": 20000},
        },
        "mod": {
            "sys_transaction_cost": {
                "enabled": True,
                "stock_commission_multiplier": 0.6667,   # 万2（base=万3 × 0.6667）
                "stock_min_commission": 5,
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
                "output_file": str(RESULT_DIR / f"{tag}.pkl"),
            },
        },
        "extra": {
            "log_level": "",
            "log_file": str(Path(__file__).with_name("backtest.log")),
            "context_vars": {},
        },
    }


# ==================== 实验组定义 ====================

# ---- 模式A: 基线单次回测（默认参数）----
EXPERIMENTS = [
    ("baseline", {}),
]

# ---- 模式B: ETF池扫描（小池 vs 大池）----
# EXPERIMENTS = [
#     ("small_pool", {}),
#     ("large_pool", {"etf_pool": [
#         "518880.XSHG", "159980.XSHE", "159985.XSHE", "501018.XSHG",
#         "161226.XSHE", "159981.XSHE", "513100.XSHG", "159509.XSHE",
#         "513290.XSHG", "513500.XSHG", "159529.XSHE", "513400.XSHG",
#         "513520.XSHG", "513030.XSHG", "513080.XSHG", "513310.XSHG",
#         "513730.XSHG", "159792.XSHE", "513130.XSHG", "513050.XSHG",
#         "159920.XSHE", "513690.XSHG", "510300.XSHG", "510500.XSHG",
#         "510050.XSHG", "510210.XSHG", "159915.XSHE", "588080.XSHG",
#         "512100.XSHG", "563360.XSHG", "563300.XSHG", "512890.XSHG",
#         "159967.XSHE", "512040.XSHG", "159201.XSHE", "511380.XSHG",
#         "511010.XSHG", "511220.XSHG",
#     ]}),
# ]

# ---- 模式C: 动量周期扫描 ----
# EXPERIMENTS = [
#     ("baseline", {}),
# ] + [
#     (f"lookback{d:02d}", {"lookback_days": d})
#     for d in [15, 20, 30, 35, 40]
# ]

# ---- 模式D: 持仓数扫描 ----
# EXPERIMENTS = [
#     (f"hold{n}", {"holdings_num": n})
#     for n in [1, 2, 3]
# ]

# ---- 模式E: 盈利保护阈值扫描 ----
# EXPERIMENTS = [
#     ("baseline", {}),
# ] + [
#     (f"prot{t*100:03.0f}", {"profit_protection_threshold": t})
#     for t in [0.03, 0.05, 0.07, 0.10]
# ]

# ---- 模式F: 短期动量阈值扫描 ----
# EXPERIMENTS = [
#     ("baseline", {}),
# ] + [
#     (f"short{t*100:03.0f}", {"short_momentum_threshold": t})
#     for t in [-0.1, -0.05, 0.0, 0.05, 0.1]
# ]

# ---- 模式G: 过滤开关对比 ----
# EXPERIMENTS = [
#     ("all_on", {}),
#     ("no_volume", {"enable_volume_check": False}),
#     ("no_short", {"use_short_momentum_filter": False}),
#     ("no_profit", {"enable_profit_protection": False}),
#     ("all_off", {
#         "enable_volume_check": False,
#         "use_short_momentum_filter": False,
#         "enable_profit_protection": False,
#     }),
# ]


# ==================== 单次回测 ====================
def run_one(tag: str, param_override: dict):
    config = make_base_config(tag)
    config["extra"]["context_vars"] = {
        "strategy_params": param_override,
    }
    print(f'\n========== 运行: {tag} ==========')
    print(f'参数覆盖: {param_override}')
    result = run_file(STRATEGY_FILE, config=config)
    return result


# ==================== 指标提取 ====================
def extract_metrics(tag: str, result: dict):
    """从 run_file 返回值里提取关键指标"""
    summary = result.get('sys_analyser', {}).get('summary', {})
    return {
        'tag':             tag,
        'total_returns':   summary.get('total_returns'),
        'annualized':      summary.get('annualized_returns'),
        'sharpe':          summary.get('sharpe'),
        'max_drawdown':    summary.get('max_drawdown'),
        'win_rate':        summary.get('win_rate'),
        'benchmark_total': summary.get('benchmark_total_returns'),
        'benchmark_annu':  summary.get('benchmark_annualized_returns'),
    }


# ==================== 主流程 ====================
def main():
    metrics_list = []
    portfolios = {}

    for tag, override in EXPERIMENTS:
        try:
            result = run_one(tag, override)
            metrics_list.append(extract_metrics(tag, result))
            pf = result.get('sys_analyser', {}).get('portfolio')
            if pf is not None:
                portfolios[tag] = pf['unit_net_value']
        except Exception as e:
            print(f'[{tag}] 回测失败: {e}')
            import traceback
            traceback.print_exc()
            metrics_list.append({'tag': tag, 'error': str(e)})

    # --- 汇总指标表 ---
    df = pd.DataFrame(metrics_list)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    print('\n\n========== 批量回测汇总 ==========')
    print(df.to_string(index=False))

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = RESULT_DIR / f'summary_{stamp}.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'\n汇总已保存: {csv_path}')

    # --- 净值曲线对比图 ---
    if len(portfolios) > 1:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            for tag, nv in portfolios.items():
                plt.plot(nv.index, nv.values, label=tag)
            plt.title('七星高照 ETF轮动 — 净值曲线对比')
            plt.xlabel('Date')
            plt.ylabel('Net Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            png_path = RESULT_DIR / f'compare_{stamp}.png'
            plt.savefig(png_path, dpi=120, bbox_inches='tight')
            plt.close()
            print(f'净值对比图: {png_path}')
        except Exception as e:
            print(f'绘图失败: {e}')

    return df


if __name__ == "__main__":
    main()
