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
RESULT_DIR = RESULT_DIR.joinpath("lihai_pool")
RESULT_DIR.mkdir(exist_ok=True)


def make_base_config(tag: str):
    """生成一份基础 config，output_file 按 tag 区分"""
    return {
        "base": {
            "strategy_file": STRATEGY_FILE,
            "data_bundle_path": r"D:\datas\bundle",
            "start_date": "2020-01-01",
            "end_date":   "2026-01-01",
            "frequency":  "1d",
            "slippage":   '0.0',
            "accounts":   {"stock": 50000},
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
# 【模式说明】切换下方模式之一，注释掉其余：
#
#   模式C — 二维网格 m_days × switch_threshold（默认）
#   模式D — 三维网格 m_days × switch_threshold × decay_ratio（5×5×5=125组）
#   模式E — 单扫 decay_ratio（固定 m_days=25, switch_threshold=1.0）
#
# tag 命名规则：
#   m{d}          → m_days=d
#   t{t}          → switch_threshold=t
#   r{decay}      → decay_ratio=decay
#   baseline      → 全默认 (m_days=25, switch_threshold=1.0, decay_ratio=1.0)

# ---- 可调范围 ----
_M_DAYS_LIST     = [20, 25, 30, 35, 40]
_THRESHOLD_LIST  = [1.0, 1.05, 1.10, 1.15, 1.20]
_DECAY_RATIO_LIST = [1.0, 1.5, 2.0, 2.5, 3.0]
# 动态池模式专用：从极短到极长，看 m_days 对收益曲线的影响
# _DYN_M_DAYS_LIST = [5, 10, 15, 20, 25, 30, 40, 50, 60,70,80, 90, 120, 200]
_DYN_M_DAYS_LIST = [55, 58, 60, 62, 65, 68, 70, 75, 80, 85, 90, 95, 100]


def _make_tag(d, t, r=1.0):
    r_int = round(r * 10)
    t_int = round(t * 100)
    parts = []
    if d != 25:
        parts.append(f"m{d:02d}")
    if t != 1.0:
        parts.append(f"t{t_int:03d}")
    if r != 1.0:
        parts.append(f"r{r_int:02d}")
    return "_".join(parts) if parts else "baseline"

def _make_override(d, t, r=1.0):
    override = {}
    scorer_override = {}
    if d != 25:
        scorer_override["m_days"] = d
    if r != 1.0:
        scorer_override["decay_ratio"] = r
    if scorer_override:
        override["scorer_momentum_r2"] = scorer_override
    if t != 1.0:
        override["switch_threshold"] = t
    return override

# 模式C：二维网格（m_days × switch_threshold），共 len(_M_DAYS_LIST) × len(_THRESHOLD_LIST) 组
# EXPERIMENTS = [
#     (_make_tag(d, t), _make_override(d, t))
#     for d in _M_DAYS_LIST
#     for t in _THRESHOLD_LIST
# ]

# 模式D：三维网格（m_days × switch_threshold × decay_ratio），样本多，按需开启
# EXPERIMENTS = [
#     (_make_tag(d, t, r), _make_override(d, t, r))
#     for d in _M_DAYS_LIST
#     for t in _THRESHOLD_LIST
#     for r in _DECAY_RATIO_LIST
# ]

# 模式E：单独扫 decay_ratio（固定 m_days=25, switch_threshold=1.0）
# EXPERIMENTS = [
#     (_make_tag(25, 1.0, r), _make_override(25, 1.0, r))
#     for r in _DECAY_RATIO_LIST
# ]

# 模式F：动态股票池（从 xiaoe_articles CSV 按日期加载持仓池）
# EXPERIMENTS = [
#     ("dynamic_pool", {
#         "pool_csv_dir": str(Path(project_root) / "xiaoe_articles"),
#         "scorer_momentum_r2": {"m_days": 50},
#     }),
# ]

# 模式G：动态股票池 × m_days 扫描（判断动量天数是否是问题）
# tag 命名：dyn_m{天数}
EXPERIMENTS = [
    (f"dyn_m{d:03d}", {
        "pool_csv_dir": str(Path(project_root) / "my_strategy/strategies/xiaoe_articles"),
        "scorer_momentum_r2": {"m_days": d},
    })
    for d in _DYN_M_DAYS_LIST
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