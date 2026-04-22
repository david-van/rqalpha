#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2026/4/22 21:31
# @Author  : david_van
# @Desc    :
import os
import copy
import pickle
import pandas as pd
from pathlib import Path
from datetime import datetime

from my_strategy.common_file import project_root
from rqalpha import run_file


# ==================== 基础配置 ====================
STRATEGY_FILE = os.path.join(project_root, 'my_strategy/strategies/momentum.py')
RESULT_DIR = Path(__file__).with_name('batch_results')
RESULT_DIR.mkdir(exist_ok=True)


def make_base_config(tag: str):
    """生成一份基础 config，output_file 按 tag 区分"""
    return {
        "base": {
            "strategy_file": STRATEGY_FILE,
            "data_bundle_path": r"D:\datas\bundle",
            "start_date": "2026-01-01",
            "end_date":   "2026-04-01",
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
                "plot": False,   # 批量跑时关掉单次弹图
                "benchmark": "000300.XSHG",
                "output_file": str(RESULT_DIR / f"{tag}.pkl"),
            },
        },
        "extra": {
            "log_level": "error",   # 批量跑时降噪
            "context_vars": {},     # ← 关键：注入参数的入口
        },
    }


# ==================== 定义实验组 ====================
# 每个实验 = (tag, 参数覆盖字典)
EXPERIMENTS = [
    ("baseline", {
        # 全默认
    }),
    # ("no_min_score", {
    #     "filter_min_score": {"enabled": False, "threshold": 0.0},
    # }),
    # ("with_ma_trend", {
    #     "filter_ma_trend": {"enabled": True, "ma_period": 20},
    # }),
    # ("with_vol_filter", {
    #     "filter_volatility": {"enabled": True, "n_days": 20, "max_vol": 0.4},
    # }),
    # ("all_filters_on", {
    #     "filter_min_score":  {"enabled": True,  "threshold": 0.0},
    #     "filter_ma_trend":   {"enabled": True,  "ma_period": 20},
    #     "filter_volatility": {"enabled": True,  "n_days": 20, "max_vol": 0.5},
    # }),
    # ("mdays_60", {
    #     "scorer_momentum_r2": {"enabled": True, "weight": 1.0, "m_days": 60},
    # }),
    # ("dual_scorer", {
    #     "scorer_momentum_r2":     {"enabled": True, "weight": 0.7, "m_days": 25},
    #     "scorer_momentum_simple": {"enabled": True, "weight": 0.3, "m_days": 20},
    # }),
]


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
    portfolios = {}   # 存每次的净值曲线，用于画图对比

    for tag, override in EXPERIMENTS:
        try:
            result = run_one(tag, override)
            metrics_list.append(extract_metrics(tag, result))
            # 收集净值曲线
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
        plt.title('Strategy Comparison (Unit Net Value)')
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