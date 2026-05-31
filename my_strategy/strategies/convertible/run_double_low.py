#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可转债双低策略 — 批量回测 & 调参

运行:
    python my_strategy/strategies/convertible/run_double_low.py

结果输出:
    my_strategy/strategies/batch_results/convertible_double_low/
"""
import os
import pandas as pd
from pathlib import Path
from datetime import datetime

from my_strategy.common_file import project_root
from rqalpha import run_file


# ==================== 基础配置 ====================
STRATEGY_FILE = os.path.join(project_root, 'my_strategy/strategies/convertible/double_low_strategy.py')
RESULT_DIR = Path(project_root) / 'my_strategy' / 'strategies' / 'batch_results' / 'convertible_double_low'
RESULT_DIR.mkdir(exist_ok=True, parents=True)


def make_base_config(tag: str):
    return {
        "base": {
            "strategy_file": STRATEGY_FILE,
            "data_bundle_path": r"D:\datas\bundle",
            "start_date": "2021-01-01",
            "end_date": "2026-05-01",
            "frequency": "1d",
            "accounts": {"stock": 100000},
        },
        "mod": {
            "cb": {
                "enabled": True,
                "lib": "my_mod.cb_bundle.rqalpha_mod_cb",
            },
            "sys_transaction_cost": {
                "enabled": True,
                "stock_commission_multiplier": 0.03,
                "stock_min_commission": 0,
                "tax_multiplier": 0,
            },
            "sys_simulation": {
                "enabled": True,
                "matching_type": "next_bar",
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


# ==================== 实验组 ====================
# ---- 参数范围 ----
_TOP_N_LIST       = [10, 15, 20, 30]
_MAX_PRICE_LIST   = [120, 130, 140]


def _tag(**kw):
    parts = []
    for k, v in kw.items():
        if k == 'top_n' and v == 20: continue
        if k == 'max_price' and v == 130: continue
        parts.append(f"{k}{v}")
    return "_".join(parts) if parts else "baseline"


def _override(**kw):
    return kw


# ── 模式1: 基准单次回测 ──
EXPERIMENTS = [
    ("baseline", _override(top_n=10, max_price=130)),
]

# ── 模式2: top_n 参数扫描 ──
# EXPERIMENTS = [
#     (_tag(top_n=n), _override(top_n=n, max_price=130, ))
#     for n in _TOP_N_LIST
# ]

# ── 模式3: max_price 参数扫描 ──
# EXPERIMENTS = [
#     (_tag(max_price=p), _override(top_n=20, max_price=p, ))
#     for p in _MAX_PRICE_LIST
# ]

# ── 模式4: 三维全扫描 ──
# EXPERIMENTS = [
#     (_tag(top_n=n, max_price=p, min_remain=r), _override(top_n=n, max_price=p, min_remain=r))
#     for n in _TOP_N_LIST
#     for p in _MAX_PRICE_LIST
#     for r in _MIN_REMAIN_LIST
# ]


# ==================== 单次回测 ====================
def run_one(tag: str, param_override: dict):
    config = make_base_config(tag)
    config["extra"]["context_vars"] = {
        "strategy_params": param_override,
    }
    print(f'\n========== 运行: {tag} ==========')
    print(f'参数: {param_override}')
    result = run_file(STRATEGY_FILE, config=config)
    return result


# ==================== 指标提取 ====================
def extract_metrics(tag: str, result: dict):
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
            plt.title('Double Low Strategy — Unit Net Value Comparison')
            plt.xlabel('Date'); plt.ylabel('Net Value')
            plt.legend(); plt.grid(True, alpha=0.3)
            png_path = RESULT_DIR / f'compare_{stamp}.png'
            plt.savefig(png_path, dpi=120, bbox_inches='tight')
            plt.close()
            print(f'净值对比图: {png_path}')
        except Exception as e:
            print(f'绘图失败: {e}')

    return df


if __name__ == "__main__":
    main()
