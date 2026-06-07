#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
七星高照ETF轮动策略 — 批量回测

运行:
    python my_strategy/strategies/七星ETF轮动/run_七星高照.py

结果输出:
    my_strategy/strategies/batch_results/七星高照/
"""

import argparse
import json
import os
from datetime import datetime
from itertools import combinations
from pathlib import Path

import pandas as pd

from my_strategy.common_file import project_root
from rqalpha import run_file


# ==================== 基础配置 ====================
STRATEGY_FILE = os.path.join(project_root, 'my_strategy/strategies/七星ETF轮动/七星高照_strategy.py')
BASE_RESULT_DIR = Path(project_root) / 'my_strategy' / 'strategies' / 'batch_results' / '七星高照'
BASE_RESULT_DIR.mkdir(exist_ok=True, parents=True)

# 默认实验选择；直接运行本脚本时生效，命令行 --sweep / --layer 会覆盖这里。
#
# 常用实验组:
#   "filter_layer"              过滤器分层组合分析，需要 ACTIVE_SWEEP_ARGS = {"layer": "1"/"2"/"3"/"4_5"}
#   "score_only_m_days"         核心过滤器全关，只扫描动量周期 m_days
#   "score_only_decay_weight"   核心过滤器全关，只扫描回归衰减权重 decay_weight
#   "sdl_threshold"             核心过滤器全关，只开启单日跌幅过滤并扫描 threshold
#   "score_grid_m_days_decay"   核心过滤器全关，扫描 m_days × decay_weight
#   "score_sdl_grid_compact"    只开启单日跌幅过滤，扫描稳定区间三维组合
#   "score_sdl_grid_full"       只开启单日跌幅过滤，扫描完整三维组合
#
# 其他可用实验组:
#   "baseline", "m_days", "holdings_num", "profit_protection", "short_momentum",
#   "filter_ablation", "decay_weight", "volume_threshold", "m_days_x_decay",
#   "only_sdl_m_days", "only_sdl_decay_weight", "only_sdl_threshold"
#
# 示例:
#   ACTIVE_SWEEP = "score_only_m_days"
#   ACTIVE_SWEEP_ARGS = {}
#   ACTIVE_SWEEP = "filter_layer"
#   ACTIVE_SWEEP_ARGS = {"layer": "4_5"}
ACTIVE_SWEEP = "score_only_m_days"
ACTIVE_SWEEP_ARGS = {}

# 运行时状态，由 resolve_sweep() 根据 ACTIVE_SWEEP / 命令行参数生成。
# 不需要手动修改这两个变量；想切换实验只改 ACTIVE_SWEEP / ACTIVE_SWEEP_ARGS。
SWEEP_META = None       # 当前实验组的元信息，用于决定输出目录和写 meta.json
EXPERIMENTS = []        # 当前实验组要执行的组合列表，格式为 [(tag, 参数覆盖字典), ...]


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
# 使用方式:
#   1. 修改文件顶部 ACTIVE_SWEEP / ACTIVE_SWEEP_ARGS
#   2. 或运行时用 --sweep / --layer 覆盖
#   3. 先用 --dry-run 确认组合，再正式运行

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


def build_baseline_sweep():
    return [("baseline", {})], None


def build_m_days_sweep():
    values = [15, 20, 30, 35, 40]
    experiments = [("baseline", {})] + [
        (f"m{d:02d}", {"scorer": {"m_days": d}})
        for d in values
    ]
    meta = {
        "name": "m_days",
        "dimensions": [{"name": "m_days", "display": "动量周期(天)"}],
        "tag_order": [tag for tag, _ in experiments],
        "tag_values": {
            "baseline": [25],
            **{f"m{d:02d}": [d] for d in values},
        },
    }
    return experiments, meta


def build_holdings_num_sweep():
    values = [1, 2, 3]
    experiments = [
        (f"h{n}", {"holdings_num": n})
        for n in values
    ]
    meta = {
        "name": "holdings_num",
        "dimensions": [{"name": "holdings_num", "display": "持仓数"}],
        "tag_order": [tag for tag, _ in experiments],
        "tag_values": {f"h{n}": [n] for n in values},
    }
    return experiments, meta


def build_profit_protection_sweep():
    values = [0.03, 0.07, 0.10]
    experiments = [("baseline", {})] + [
        (f"pp{t * 100:03.0f}", {"filter_profit_protection": {"threshold": t}})
        for t in values
    ]
    meta = {
        "name": "profit_protection",
        "dimensions": [{"name": "threshold", "display": "盈利保护阈值"}],
        "tag_order": [tag for tag, _ in experiments],
        "tag_values": {
            "baseline": [0.05],
            **{f"pp{t * 100:03.0f}": [t] for t in values},
        },
    }
    return experiments, meta


def build_short_momentum_sweep():
    values = [-0.10, -0.05, 0.0, 0.05, 0.10]
    experiments = [("baseline", {})] + [
        (
            f"sm_neg{abs(t) * 100:02.0f}" if t < 0 else f"sm_{t * 100:02.0f}",
            {"filter_short_momentum": {"threshold": t}},
        )
        for t in values
    ]
    meta = {
        "name": "short_momentum",
        "dimensions": [{"name": "threshold", "display": "短期动量阈值"}],
        "tag_order": [tag for tag, _ in experiments],
        "tag_values": {
            "baseline": [0.0],
            **{
                (f"sm_neg{abs(t) * 100:02.0f}" if t < 0 else f"sm_{t * 100:02.0f}"): [t]
                for t in values
            },
        },
    }
    return experiments, meta


def build_filter_ablation_sweep():
    experiments = [
        ("all_on", {}),
        ("no_pp", {"filter_profit_protection": {"enabled": False}}),
        ("no_vol", {"filter_volume": {"enabled": False}}),
        ("no_sm", {"filter_short_momentum": {"enabled": False}}),
        ("no_sdl", {"filter_single_day_loss": {"enabled": False}}),
        ("no_sr", {"filter_score_range": {"enabled": False}}),
        ("all_off", core_filter_params([])),
    ]
    meta = {
        "name": "filter_ablation",
        "dimensions": [{"name": "variant", "display": "过滤器组合"}],
        "tag_order": [tag for tag, _ in experiments],
        "tag_values": {
            "all_on": ["全部开启"],
            "no_pp": ["关盈利保护"],
            "no_vol": ["关成交量"],
            "no_sm": ["关短期动量"],
            "no_sdl": ["关单日跌幅"],
            "no_sr": ["关得分范围"],
            "all_off": ["核心过滤器全关"],
        },
    }
    return experiments, meta


def build_filter_layer_sweep(layer="4_5"):
    layer = str(layer)
    layer_experiments = build_filter_layer_experiments(layer)
    experiments = [
        (tag, core_filter_params(keys))
        for tag, keys in layer_experiments
    ]
    meta = {
        "name": f"filter_layer_{layer}",
        "analysis_type": "filter_layer",
        "dimensions": [{"name": "variant", "display": "过滤器组合"}],
        "layer": layer,
        "baseline": "base_off",
        "tag_order": [tag for tag, _ in layer_experiments],
        "filter_sets": {tag: keys for tag, keys in layer_experiments},
        "filter_labels": {key: label for key, (_, label) in CORE_FILTERS.items()},
        "tag_values": {
            tag: [filter_label(keys)]
            for tag, keys in layer_experiments
        },
    }
    return experiments, meta


def merge_nested(base: dict, extra: dict | None = None) -> dict:
    """递归合并参数字典，返回新对象。"""
    merged = {}
    for key, value in base.items():
        merged[key] = value.copy() if isinstance(value, dict) else value
    if not extra:
        return merged
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = merged[key].copy()
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def only_sdl_params(extra: dict | None = None) -> dict:
    """只开启单日跌幅过滤，其余核心过滤器关闭。"""
    params = {
        "filter_profit_protection": {"enabled": False},
        "filter_volume": {"enabled": False},
        "filter_short_momentum": {"enabled": False},
        "filter_single_day_loss": {"enabled": True, "threshold": 0.97},
        "filter_score_range": {"enabled": False},
    }
    return merge_nested(params, extra)


def score_only_params(extra: dict | None = None) -> dict:
    """关闭核心过滤器，只测试动量打分参数；上市/停牌过滤保留默认行为。"""
    return merge_nested(core_filter_params([]), extra)


def sdl_only_params(extra: dict | None = None) -> dict:
    """核心过滤器全关后只开启单日跌幅过滤。"""
    return merge_nested(
        core_filter_params(["sdl"]),
        {"filter_single_day_loss": {"threshold": 0.97}, **(extra or {})},
    )


def build_score_only_m_days_experiments(values):
    return [
        (f"m{d:02d}", score_only_params({"scorer": {"m_days": d}}))
        for d in values
    ]


def build_score_only_decay_experiments(values):
    return [
        (f"dw{int(w * 10):02d}", score_only_params({"scorer": {"decay_weight": w}}))
        for w in values
    ]


def build_sdl_threshold_experiments(values):
    return [
        (f"sdl{int(t * 100):03d}", sdl_only_params({"filter_single_day_loss": {"threshold": t}}))
        for t in values
    ]


def build_score_grid_experiments(m_days_values, decay_weight_values):
    return [
        (
            f"m{m_days:02d}_dw{int(decay_weight * 10):02d}",
            score_only_params({"scorer": {"m_days": m_days, "decay_weight": decay_weight}}),
        )
        for m_days in m_days_values
        for decay_weight in decay_weight_values
    ]


def build_score_sdl_grid_experiments(m_days_values, decay_weight_values, threshold_values):
    return [
        (
            f"m{m_days:02d}_dw{int(decay_weight * 10):02d}_sdl{int(threshold * 100):03d}",
            sdl_only_params({
                "scorer": {"m_days": m_days, "decay_weight": decay_weight},
                "filter_single_day_loss": {"threshold": threshold},
            }),
        )
        for m_days in m_days_values
        for decay_weight in decay_weight_values
        for threshold in threshold_values
    ]


def build_only_sdl_m_days_experiments(values):
    return [
        (f"m{d:02d}", only_sdl_params({"scorer": {"m_days": d}}))
        for d in values
    ]


def build_only_sdl_decay_experiments(values):
    return [
        (f"dw{int(w * 10):02d}", only_sdl_params({"scorer": {"decay_weight": w}}))
        for w in values
    ]


def build_only_sdl_threshold_experiments(values):
    return [
        (f"sdl{int(t * 100):03d}", only_sdl_params({"filter_single_day_loss": {"threshold": t}}))
        for t in values
    ]


def build_only_sdl_m_days_sweep():
    values = [15, 20, 25, 30, 35, 40]
    experiments = build_only_sdl_m_days_experiments(values)
    meta = {
        "name": "only_sdl_m_days",
        "dimensions": [{"name": "m_days", "display": "动量周期(天)"}],
        "tag_order": [tag for tag, _ in experiments],
        "tag_values": {f"m{d:02d}": [d] for d in values},
    }
    return experiments, meta


def build_only_sdl_decay_weight_sweep():
    values = [1.0, 1.5, 2.0, 2.5, 3.0]
    experiments = build_only_sdl_decay_experiments(values)
    meta = {
        "name": "only_sdl_decay_weight",
        "dimensions": [{"name": "decay_weight", "display": "回归衰减权重"}],
        "tag_order": [tag for tag, _ in experiments],
        "tag_values": {f"dw{int(w * 10):02d}": [w] for w in values},
    }
    return experiments, meta


def build_only_sdl_threshold_sweep():
    values = [0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    experiments = build_only_sdl_threshold_experiments(values)
    meta = {
        "name": "only_sdl_threshold",
        "dimensions": [{"name": "threshold", "display": "单日跌幅阈值"}],
        "tag_order": [tag for tag, _ in experiments],
        "tag_values": {f"sdl{int(t * 100):03d}": [t] for t in values},
    }
    return experiments, meta


def build_score_only_m_days_sweep():
    values = [15, 20, 25, 30, 35, 40, 50, 60]
    experiments = build_score_only_m_days_experiments(values)
    meta = {
        "name": "score_only_m_days",
        "analysis_type": "score_only",
        "dimensions": [{"name": "m_days", "display": "动量周期(天)"}],
        "tag_order": [tag for tag, _ in experiments],
        "tag_values": {f"m{d:02d}": [d] for d in values},
    }
    return experiments, meta


def build_score_only_decay_weight_sweep():
    values = [1.0, 1.5, 2.0, 2.5, 3.0]
    experiments = build_score_only_decay_experiments(values)
    meta = {
        "name": "score_only_decay_weight",
        "analysis_type": "score_only",
        "dimensions": [{"name": "decay_weight", "display": "回归衰减权重"}],
        "tag_order": [tag for tag, _ in experiments],
        "tag_values": {f"dw{int(w * 10):02d}": [w] for w in values},
    }
    return experiments, meta


def build_sdl_threshold_sweep():
    values = [0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    experiments = build_sdl_threshold_experiments(values)
    meta = {
        "name": "sdl_threshold",
        "analysis_type": "single_filter_threshold",
        "dimensions": [{"name": "threshold", "display": "单日跌幅阈值"}],
        "tag_order": [tag for tag, _ in experiments],
        "tag_values": {f"sdl{int(t * 100):03d}": [t] for t in values},
    }
    return experiments, meta


def build_score_grid_m_days_decay_sweep():
    m_days_values = [15, 20, 25, 30, 35, 40, 50, 60]
    decay_weight_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    experiments = build_score_grid_experiments(m_days_values, decay_weight_values)
    meta = {
        "name": "score_grid_m_days_decay",
        "analysis_type": "score_grid",
        "dimensions": [
            {"name": "m_days", "display": "动量周期(天)"},
            {"name": "decay_weight", "display": "回归衰减权重"},
        ],
        "tag_order": [tag for tag, _ in experiments],
        "tag_values": {
            f"m{m_days:02d}_dw{int(decay_weight * 10):02d}": [m_days, decay_weight]
            for m_days in m_days_values
            for decay_weight in decay_weight_values
        },
    }
    return experiments, meta


def build_score_sdl_grid_compact_sweep():
    m_days_values = [15, 20, 25, 60]
    decay_weight_values = [2.0, 2.5, 3.0]
    threshold_values = [0.95, 0.96, 0.97, 0.98]
    experiments = build_score_sdl_grid_experiments(m_days_values, decay_weight_values, threshold_values)
    meta = {
        "name": "score_sdl_grid_compact",
        "analysis_type": "score_sdl_grid",
        "dimensions": [
            {"name": "m_days", "display": "动量周期(天)"},
            {"name": "decay_weight", "display": "回归衰减权重"},
            {"name": "sdl_threshold", "display": "单日跌幅阈值"},
        ],
        "tag_order": [tag for tag, _ in experiments],
        "tag_values": {
            f"m{m_days:02d}_dw{int(decay_weight * 10):02d}_sdl{int(threshold * 100):03d}": [
                m_days,
                decay_weight,
                threshold,
            ]
            for m_days in m_days_values
            for decay_weight in decay_weight_values
            for threshold in threshold_values
        },
    }
    return experiments, meta


def build_score_sdl_grid_full_sweep():
    m_days_values = [15, 20, 25, 30, 35, 40, 50, 60]
    decay_weight_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    threshold_values = [0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    experiments = build_score_sdl_grid_experiments(m_days_values, decay_weight_values, threshold_values)
    meta = {
        "name": "score_sdl_grid_full",
        "analysis_type": "score_sdl_grid",
        "dimensions": [
            {"name": "m_days", "display": "动量周期(天)"},
            {"name": "decay_weight", "display": "回归衰减权重"},
            {"name": "sdl_threshold", "display": "单日跌幅阈值"},
        ],
        "tag_order": [tag for tag, _ in experiments],
        "tag_values": {
            f"m{m_days:02d}_dw{int(decay_weight * 10):02d}_sdl{int(threshold * 100):03d}": [
                m_days,
                decay_weight,
                threshold,
            ]
            for m_days in m_days_values
            for decay_weight in decay_weight_values
            for threshold in threshold_values
        },
    }
    return experiments, meta


def build_decay_weight_sweep():
    values = [1.0, 1.5, 2.5, 3.0]
    experiments = [("baseline", {})] + [
        (f"dw{d * 10:02.0f}", {"scorer": {"decay_weight": d}})
        for d in values
    ]
    meta = {
        "name": "decay_weight",
        "dimensions": [{"name": "decay_weight", "display": "衰减权重"}],
        "tag_order": [tag for tag, _ in experiments],
        "tag_values": {
            "baseline": [2.0],
            **{f"dw{d * 10:02.0f}": [d] for d in values},
        },
    }
    return experiments, meta


def build_volume_threshold_sweep():
    values = [1.5, 2.5, 3.0]
    experiments = [("baseline", {})] + [
        (f"vt{t * 10:02.0f}", {"filter_volume": {"threshold": t}})
        for t in values
    ]
    meta = {
        "name": "volume_threshold",
        "dimensions": [{"name": "threshold", "display": "成交量阈值"}],
        "tag_order": [tag for tag, _ in experiments],
        "tag_values": {
            "baseline": [2.0],
            **{f"vt{t * 10:02.0f}": [t] for t in values},
        },
    }
    return experiments, meta


def build_m_days_x_decay_sweep():
    m_days_values = [15, 20, 25, 30, 35]
    decay_weight_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    experiments = [
        (f"m{d:02d}_dw{w * 10:02.0f}", {"scorer": {"m_days": d, "decay_weight": w}})
        for d in m_days_values
        for w in decay_weight_values
    ]
    meta = {
        "name": "m_days_x_decay",
        "dimensions": [
            {"name": "m_days", "display": "动量周期(天)"},
            {"name": "decay_weight", "display": "衰减权重"},
        ],
        "tag_order": [tag for tag, _ in experiments],
        "tag_values": {
            f"m{d:02d}_dw{w * 10:02.0f}": [d, w]
            for d in m_days_values
            for w in decay_weight_values
        },
    }
    return experiments, meta


SWEEP_BUILDERS = {
    "baseline": build_baseline_sweep,
    "m_days": build_m_days_sweep,
    "holdings_num": build_holdings_num_sweep,
    "profit_protection": build_profit_protection_sweep,
    "short_momentum": build_short_momentum_sweep,
    "filter_ablation": build_filter_ablation_sweep,
    "filter_layer": build_filter_layer_sweep,
    "score_only_m_days": build_score_only_m_days_sweep,
    "score_only_decay_weight": build_score_only_decay_weight_sweep,
    "sdl_threshold": build_sdl_threshold_sweep,
    "score_grid_m_days_decay": build_score_grid_m_days_decay_sweep,
    "score_sdl_grid_compact": build_score_sdl_grid_compact_sweep,
    "score_sdl_grid_full": build_score_sdl_grid_full_sweep,
    "only_sdl_m_days": build_only_sdl_m_days_sweep,
    "only_sdl_decay_weight": build_only_sdl_decay_weight_sweep,
    "only_sdl_threshold": build_only_sdl_threshold_sweep,
    "decay_weight": build_decay_weight_sweep,
    "volume_threshold": build_volume_threshold_sweep,
    "m_days_x_decay": build_m_days_x_decay_sweep,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="七星高照 ETF 轮动批量回测")
    parser.add_argument("--sweep", choices=sorted(SWEEP_BUILDERS), help="选择要运行的实验组")
    parser.add_argument("--layer", choices=["1", "2", "3", "4_5"], help="filter_layer 的层级")
    parser.add_argument("--list-sweeps", action="store_true", help="列出可用实验组后退出")
    parser.add_argument("--dry-run", action="store_true", help="只打印实验组合，不执行回测")
    return parser.parse_args(argv)


def resolve_sweep(args):
    sweep_name = args.sweep or ACTIVE_SWEEP
    sweep_args = dict(ACTIVE_SWEEP_ARGS) if sweep_name == ACTIVE_SWEEP else {}
    if args.layer is not None:
        if sweep_name != "filter_layer":
            raise ValueError("--layer 只能用于 filter_layer 实验组")
        sweep_args["layer"] = args.layer

    builder = SWEEP_BUILDERS[sweep_name]
    experiments, sweep_meta = builder(**sweep_args)
    return sweep_name, sweep_args, experiments, sweep_meta


def print_available_sweeps():
    print("可用实验组:")
    for name in sorted(SWEEP_BUILDERS):
        print(f"  - {name}")


def print_sweep_plan(sweep_name, sweep_args, experiments, sweep_meta):
    output_name = sweep_meta["name"] if sweep_meta else "single"
    output_dir = BASE_RESULT_DIR / output_name
    print(f"实验组: {sweep_name}")
    print(f"参数: {sweep_args}")
    print(f"输出目录: {output_dir}")
    print(f"组合数量: {len(experiments)}")
    for index, (tag, override) in enumerate(experiments, start=1):
        print(f"\n[{index:02d}] {tag}")
        print(json.dumps(override, ensure_ascii=False, indent=2))


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
def main(argv=None):
    global EXPERIMENTS, SWEEP_META

    args = parse_args(argv)
    if args.list_sweeps:
        print_available_sweeps()
        return None

    sweep_name, sweep_args, EXPERIMENTS, SWEEP_META = resolve_sweep(args)

    if args.dry_run:
        print_sweep_plan(sweep_name, sweep_args, EXPERIMENTS, SWEEP_META)
        return pd.DataFrame({"tag": [tag for tag, _ in EXPERIMENTS]})

    output_dir = get_output_dir()
    print(f'实验组: {sweep_name}')
    print(f'实验参数: {sweep_args}')
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
