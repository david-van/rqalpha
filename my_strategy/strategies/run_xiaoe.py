#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc : xiaoe_articles 股票池策略 — 批量回测 & 调参
"""
运行:
    python my_strategy/strategies/run_xiaoe.py

结果输出:
    my_strategy/strategies/batch_results/xiaoe_pool/
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime

from my_strategy.common_file import project_root
from rqalpha import run_file


# ==================== 基础配置 ====================
STRATEGY_FILE = os.path.join(project_root, 'my_strategy/strategies/xiaoe_strategy.py')
RESULT_DIR = Path(__file__).with_name('batch_results').joinpath("xiaoe_pool")
RESULT_DIR.mkdir(exist_ok=True, parents=True)

POOL_DIR = str(Path(project_root) / "my_strategy/strategies/xiaoe_articles")


def make_base_config(tag: str):
    return {
        "base": {
            "strategy_file": STRATEGY_FILE,
            "data_bundle_path": r"D:\datas\bundle",
            "start_date": "2020-01-01",
            "end_date":   "2026-01-01",
            "frequency":  "1d",
            "slippage":   '0.0',
            "accounts":   {"stock": 100000},
        },
        "mod": {
            "sys_transaction_cost": {
                "enabled": True,
                "stock_commission_multiplier": 0.0625,
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
# ---- 公共参数范围 ----
_M_DAYS_LIST      = [15, 20, 25, 30, 40, 50, 60]
_DECAY_RATIO_LIST  = [1.0, 1.5, 2.0, 3.0]
_TOP_N_LIST        = [1, 2, 3]
_THRESHOLD_LIST    = [1.0, 1.05, 1.10]


def _tag(mode, **kw):
    """生成 tag: mode_k1v1_k2v2，略去默认值"""
    parts = [mode]
    for k, v in kw.items():
        if k == 'top_n' and v == 1: continue
        if k == 'm_days' and v == 25: continue
        if k == 'decay_ratio' and v == 1.0: continue
        if k == 'switch_threshold' and v == 1.0: continue
        if isinstance(v, float):
            parts.append(f"{k}{v:.2f}")
        else:
            parts.append(f"{k}{v}")
    return "_".join(parts)


def _override(mode, **kw):
    """生成 xiaoe_strategy 的参数覆盖"""
    override = {"mode": mode, "pool_csv_dir": POOL_DIR}
    override.update(kw)
    return override


# ── 模式1: 对比（2 组）── buy_and_hold vs equal_weight
EXPERIMENTS = [
    ("buy_and_hold",     _override("buy_and_hold")),
    ("equal_weight",     _override("equal_weight")),
]

# ── 模式2: momentum_top 参数扫描（m_days × top_n，21 组）──
# EXPERIMENTS = [
#     (_tag("mtop", top_n=n, m_days=d),
#      _override("momentum_top", top_n=n, m_days=d))
#     for n in _TOP_N_LIST
#     for d in _M_DAYS_LIST
# ]

# ── 模式3: momentum_weighted 参数扫描（m_days × decay_ratio，28 组）──
# EXPERIMENTS = [
#     (_tag("mw", m_days=d, decay_ratio=r),
#      _override("momentum_weighted", m_days=d, decay_ratio=r))
#     for d in _M_DAYS_LIST
#     for r in _DECAY_RATIO_LIST
# ]

# ── 模式4: momentum_top 三维扫描（m_days × decay_ratio × threshold, 84 组）──
# EXPERIMENTS = [
#     (_tag("mtop", m_days=d, decay_ratio=r, switch_threshold=t),
#      _override("momentum_top", top_n=1, m_days=d, decay_ratio=r, switch_threshold=t))
#     for d in _M_DAYS_LIST
#     for r in _DECAY_RATIO_LIST
#     for t in _THRESHOLD_LIST
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
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        for tag, nv in portfolios.items():
            plt.plot(nv.index, nv.values, label=tag)
        plt.title('xiaoe_strategy — Unit Net Value Comparison')
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
