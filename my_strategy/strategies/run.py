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
RESULT_DIR = RESULT_DIR.joinpath("mdays_switch_threshold")
RESULT_DIR.mkdir(exist_ok=True)


def make_base_config(tag: str):
    """生成一份基础 config，output_file 按 tag 区分"""
    return {
        "base": {
            "strategy_file": STRATEGY_FILE,
            "data_bundle_path": r"D:\datas\bundle",
            "start_date": "2018-01-01",
            "end_date":   "2026-01-01",
            "frequency":  "1d",
            "slippage":   '0.0',
            "accounts":   {"stock": 10000},
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
#
# 【模式说明】切换下方三种模式之一，注释掉其余两种：
#
#   模式A — 单维扫描 m_days（参数敏感性）
#   模式B — 单维扫描 switch_threshold（阈值敏感性）
#   模式C — 二维网格搜索 m_days × switch_threshold（联合优化）
#
# tag 命名规则：
#   m{d}          → m_days=d,  switch_threshold=1.0（默认）
#   t{t}          → m_days=25（默认）, switch_threshold=t
#   m{d}_t{t}     → 两者均非默认
#   baseline      → 全默认（m_days=25, switch_threshold=1.0）

# ---- 可调范围 ----
_M_DAYS_LIST   = [20,25,   30, 35, 40]          # 根据上次扫描结果聚焦到甜区
_THRESHOLD_LIST = [1.0, 1.05, 1.10, 1.15, 1.20]

def _make_tag(d, t):
    t_int = round(t * 100)   # 用 round 避免 1.15*100=114.999... 的浮点误差
    if d == 25 and t == 1.0:
        return "baseline"
    if d == 25:
        return f"t{t_int:03d}"                  # t105 / t110 / t115 / t120
    if t == 1.0:
        return f"m{d:02d}"                      # m20 / m30 / m35 / m40
    return f"m{d:02d}_t{t_int:03d}"            # m30_t115

def _make_override(d, t):
    override = {}
    if d != 25:
        override["scorer_momentum_r2"] = {"m_days": d}
    if t != 1.0:
        override["switch_threshold"] = t
    return override

# 模式C：二维网格（m_days × switch_threshold），共 len(_M_DAYS_LIST) × len(_THRESHOLD_LIST) 组
EXPERIMENTS = [
    (_make_tag(d, t), _make_override(d, t))
    for d in _M_DAYS_LIST
    for t in _THRESHOLD_LIST
]

# 模式A：只扫 m_days（注释掉模式C，取消下方注释）
# _M_DAYS_LIST = [5, 8, 10, 15, 18, 20, 22, 25, 28, 30, 35, 40, 50, 60]
# EXPERIMENTS = [
#     ("baseline" if d == 25 else f"mdays_{d:02d}", {} if d == 25 else {"scorer_momentum_r2": {"m_days": d}})
#     for d in _M_DAYS_LIST
# ]

# 模式B：只扫 switch_threshold（注释掉模式C，取消下方注释）
# _THRESHOLD_LIST = [1.0, 1.05, 1.10, 1.15, 1.20]
# EXPERIMENTS = [
#     ("baseline" if t == 1.0 else f"t{int(t * 100):03d}", {} if t == 1.0 else {"switch_threshold": t})
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