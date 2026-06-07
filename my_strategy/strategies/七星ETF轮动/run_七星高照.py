#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
七星高照ETF轮动策略 — 批量回测

运行:
    python my_strategy/strategies/七星ETF轮动/run_七星高照.py

结果输出:
    my_strategy/strategies/batch_results/七星高照/
"""

import json
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import combinations

from my_strategy.common_file import project_root
from rqalpha import run_file


# ==================== 基础配置 ====================
STRATEGY_FILE = os.path.join(project_root, 'my_strategy/strategies/七星ETF轮动/七星高照_strategy.py')
BASE_RESULT_DIR = Path(project_root) / 'my_strategy' / 'strategies' / 'batch_results' / '七星高照'
BASE_RESULT_DIR.mkdir(exist_ok=True, parents=True)

SWEEP_META = None  # 由下方实验组定义覆盖


def get_output_dir():
    """根据 SWEEP_META 决定输出子目录"""
    if SWEEP_META is None:
        return BASE_RESULT_DIR / "single"
    return BASE_RESULT_DIR / SWEEP_META["name"]


def make_base_config(tag: str):
    """生成一份基础 config，output_file 按 tag 区分"""
    output_dir = get_output_dir()
    output_dir.mkdir(exist_ok=True, parents=True)
    return {
        "base": {
            "strategy_file": STRATEGY_FILE,
            "data_bundle_path": r"D:\datas\bundle",
            "start_date": "2020-01-01",
            "end_date":   "2025-12-31",
            "frequency":  "1d",
            "accounts":   {"stock": 20000},
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
                "output_file": str(output_dir / f"{tag}.pkl"),
            },
        },
        "extra": {
            "log_level": "info",
            "log_file": str(Path(__file__).with_name("backtest.log")),
            "context_vars": {},
        },
    }


# ==================== 实验组定义 ====================
# 使用说明:
#   1. 取消注释一组 EXPERIMENTS
#   2. 取消注释对应的 SWEEP_META
#   3. 修改上方 start_date / end_date
#   4. 运行 python run_七星高照.py

# ---- 模式A: 基线单次回测(默认参数) ----
EXPERIMENTS = [
    ("baseline", {}),
]
SWEEP_META = None



# ---- 模式C: 动量周期 m_days 扫描 ----
# EXPERIMENTS = [
#     ("baseline", {}),
# ] + [
#     (f"m{d:02d}", {"scorer": {"m_days": d}})
#     for d in [15, 20, 30, 35, 40]
# ]
# SWEEP_META = {
#     "name": "m_days",
#     "dimensions": [{"name": "m_days", "display": "动量周期(天)"}],
#     "tag_values": {"baseline": [25], "m15": [15], "m20": [20],
#                    "m30": [30], "m35": [35], "m40": [40]}
# }

# ---- 模式D: 持仓数 holdings_num 扫描 ----
# EXPERIMENTS = [
#     (f"h{n}", {"holdings_num": n})
#     for n in [1, 2, 3]
# ]
# SWEEP_META = {
#     "name": "holdings_num",
#     "dimensions": [{"name": "holdings_num", "display": "持仓数"}],
#     "tag_values": {"h1": [1], "h2": [2], "h3": [3]}
# }

# ---- 模式E: 盈利保护阈值扫描 ----
# EXPERIMENTS = [
#     ("baseline", {}),
# ] + [
#     (f"pp{t*100:03.0f}", {"filter_profit_protection": {"threshold": t}})
#     for t in [0.03, 0.07, 0.10]
# ]
# SWEEP_META = {
#     "name": "profit_protection",
#     "dimensions": [{"name": "threshold", "display": "盈利保护阈值"}],
#     "tag_values": {"baseline": [0.05], "pp003": [0.03], "pp007": [0.07], "pp010": [0.10]}
# }

# ---- 模式F: 短期动量阈值扫描 ----
# EXPERIMENTS = [
#     ("baseline", {}),
# ] + [
#     (f"sm_neg{abs(t)*100:02.0f}" if t < 0 else f"sm_{t*100:02.0f}",
#      {"filter_short_momentum": {"threshold": t}})
#     for t in [-0.10, -0.05, 0.0, 0.05, 0.10]
# ]
# SWEEP_META = {
#     "name": "short_momentum",
#     "dimensions": [{"name": "threshold", "display": "短期动量阈值"}],
#     "tag_values": {"baseline": [0.0], "sm_neg10": [-0.10], "sm_neg05": [-0.05],
#                    "sm_00": [0.0], "sm_05": [0.05], "sm_10": [0.10]}
# }

# ---- 模式G1: 过滤器关闭消融（从全开出发，逐个关闭）----
CORE_FILTERS = {
    "pp": ("filter_profit_protection", "盈利保护"),
    "vol": ("filter_volume", "成交量"),
    "sm": ("filter_short_momentum", "短期动量"),
    "sdl": ("filter_single_day_loss", "单日跌幅"),
    "sr": ("filter_score_range", "得分范围"),
}


def core_filter_params(enabled_keys):
    """生成核心过滤器开关参数；上市/停牌过滤始终保留。"""
    enabled_keys = set(enabled_keys)
    return {
        param_name: {"enabled": key in enabled_keys}
        for key, (param_name, _) in CORE_FILTERS.items()
    }


def filter_label(keys):
    if not keys:
        return "核心过滤器全关"
    return "+".join(CORE_FILTERS[key][1] for key in keys)


def filter_tag(keys):
    if not keys:
        return "base_off"
    if len(keys) == 1:
        return f"only_{keys[0]}"
    if len(keys) == len(CORE_FILTERS):
        return "all_on"
    return "_".join(keys)


def build_filter_layer_experiments(layer):
    """生成指定层级的过滤器组合；每层都带 base_off 作为基准。"""
    filter_keys = list(CORE_FILTERS.keys())
    experiments = [("base_off", [])]

    if layer == "4_5":
        sizes = [4, len(filter_keys)]
    else:
        layer = int(layer)
        if layer < 1 or layer > 3:
            raise ValueError("FILTER_LAYER 只能是 1、2、3 或 '4_5'")
        sizes = [layer]

    for size in sizes:
        for keys in combinations(filter_keys, size):
            keys = list(keys)
            experiments.append((filter_tag(keys), keys))
    return experiments


# EXPERIMENTS = [
#     ("all_on", {}),
#     ("no_pp", {"filter_profit_protection": {"enabled": False}}),
#     ("no_vol", {"filter_volume": {"enabled": False}}),
#     ("no_sm", {"filter_short_momentum": {"enabled": False}}),
#     ("no_sdl", {"filter_single_day_loss": {"enabled": False}}),
#     ("no_sr", {"filter_score_range": {"enabled": False}}),
#     ("all_off", core_filter_params([])),
# ]
# SWEEP_META = {
#     "name": "filter_ablation",
#     "dimensions": [{"name": "variant", "display": "过滤器组合"}],
#     "tag_order": ["all_on", "no_pp", "no_vol", "no_sm", "no_sdl", "no_sr", "all_off"],
#     "tag_values": {"all_on": ["全部开启"], "no_pp": ["关盈利保护"],
#                    "no_vol": ["关成交量"], "no_sm": ["关短期动量"],
#                    "no_sdl": ["关单日跌幅"], "no_sr": ["关得分范围"],
#                    "all_off": ["核心过滤器全关"]},
# }

# ---- 模式G2: 过滤器分层分析（每次只跑一层，每层单独输出）----
FILTER_LAYER = '4_5'  # 可选: 1、2、3、"4_5"
LAYER_EXPERIMENTS = build_filter_layer_experiments(FILTER_LAYER)
LAYER_NAME = f"filter_layer_{FILTER_LAYER}"

EXPERIMENTS = [
    (tag, core_filter_params(keys))
    for tag, keys in LAYER_EXPERIMENTS
]
SWEEP_META = {
    "name": LAYER_NAME,
    "analysis_type": "filter_layer",
    "dimensions": [{"name": "variant", "display": "过滤器组合"}],
    "layer": FILTER_LAYER,
    "baseline": "base_off",
    "tag_order": [tag for tag, _ in LAYER_EXPERIMENTS],
    "filter_sets": {tag: keys for tag, keys in LAYER_EXPERIMENTS},
    "filter_labels": {key: label for key, (_, label) in CORE_FILTERS.items()},
    "tag_values": {
        tag: [filter_label(keys)]
        for tag, keys in LAYER_EXPERIMENTS
    },
}

# ---- 模式H: 衰减权重 decay_weight 扫描 ----
# EXPERIMENTS = [
#     ("baseline", {}),
# ] + [
#     (f"dw{d*10:02.0f}", {"scorer": {"decay_weight": d}})
#     for d in [1.0, 1.5, 2.5, 3.0]
# ]
# SWEEP_META = {
#     "name": "decay_weight",
#     "dimensions": [{"name": "decay_weight", "display": "衰减权重"}],
#     "tag_values": {"baseline": [2.0], "dw10": [1.0], "dw15": [1.5],
#                    "dw25": [2.5], "dw30": [3.0]}
# }

# ---- 模式I: 成交量阈值扫描 ----
# EXPERIMENTS = [
#     ("baseline", {}),
# ] + [
#     (f"vt{t*10:02.0f}", {"filter_volume": {"threshold": t}})
#     for t in [1.5, 2.5, 3.0]
# ]
# SWEEP_META = {
#     "name": "volume_threshold",
#     "dimensions": [{"name": "threshold", "display": "成交量阈值"}],
#     "tag_values": {"baseline": [2.0], "vt15": [1.5], "vt25": [2.5], "vt30": [3.0]}
# }

# ---- 模式J: 二维扫描 m_days x decay_weight ----
# EXPERIMENTS = [
#     (f"m{d:02d}_dw{w*10:02.0f}", {"scorer": {"m_days": d, "decay_weight": w}})
#     for d in [15, 20, 25, 30, 35]
#     for w in [1.0, 1.5, 2.0, 2.5, 3.0]
# ]
# SWEEP_META = {
#     "name": "m_days_x_decay",
#     "dimensions": [
#         {"name": "m_days", "display": "动量周期(天)"},
#         {"name": "decay_weight", "display": "衰减权重"}
#     ],
#     "tag_values": {
#         f"m{d:02d}_dw{w*10:02.0f}": [d, w]
#         for d in [15, 20, 25, 30, 35]
#         for w in [1.0, 1.5, 2.0, 2.5, 3.0]
#     }
# }


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
    output_dir = get_output_dir()
    print(f'输出目录: {output_dir}')

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
    csv_path = output_dir / f'summary_{stamp}.csv'
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
            png_path = output_dir / f'compare_{stamp}.png'
            plt.savefig(png_path, dpi=120, bbox_inches='tight')
            plt.close()
            print(f'净值对比图: {png_path}')
        except Exception as e:
            print(f'绘图失败: {e}')

    # --- 写 meta.json ---
    if SWEEP_META:
        meta_path = output_dir / "meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(SWEEP_META, f, ensure_ascii=False, indent=2)
        print(f'\n参数元信息: {meta_path}')

    return df


if __name__ == "__main__":
    main()
