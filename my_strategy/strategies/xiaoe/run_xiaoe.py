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
STRATEGY_FILE = os.path.join(project_root, 'my_strategy/strategies/xiaoe/xiaoe_strategy.py')
STRATEGY_FILE_TURTLE = os.path.join(project_root, 'my_strategy/strategies/xiaoe/xiaoe_turtle.py')
RESULT_DIR = Path(project_root) / 'my_strategy' / 'strategies' / 'batch_results' / 'xiaoe_pool'
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
_EMA_DAYS          = 60


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


# ── 当前研究主线：基线 + 低波动 + 低位/动量/回调增强 + 风控开关 ──
EXPERIMENTS = [
    ("buy_and_hold",
     _override("buy_and_hold")),
    ("equal_weight",
     _override("equal_weight")),
    ("dynamic_equal_reb_r1",
     _override(
         "low_vol_rebalance_hybrid",
         base_ratio=1.0,
         enhance_ratio=0.0,
         low_vol_days=60,
         enhance_top_n=2,
         rebalance_days=1,
     )),
    ("low_vol_reb_b50_e50_v60_n2_r20",
     _override(
         "low_vol_rebalance_hybrid",
         base_ratio=0.5,
         enhance_ratio=0.5,
         low_vol_days=60,
         enhance_top_n=2,
         rebalance_days=20,
     )),
    ("low_vol_reb_b50_e50_v60_n2_r1",
     _override(
         "low_vol_rebalance_hybrid",
         base_ratio=0.5,
         enhance_ratio=0.5,
         low_vol_days=60,
         enhance_top_n=2,
         rebalance_days=1,
     )),
    ("high_vol_reb_b50_e50_v60_n2_r1",
     _override(
         "high_vol_rebalance_hybrid",
         base_ratio=0.5,
         enhance_ratio=0.5,
         low_vol_days=60,
         vol_prefer="high",
         enhance_top_n=2,
         rebalance_days=1,
     )),
    ("high_vol_reb_b70_e30_v60_n2_r1",
     _override(
         "high_vol_rebalance_hybrid",
         base_ratio=0.7,
         enhance_ratio=0.3,
         low_vol_days=60,
         vol_prefer="high",
         enhance_top_n=2,
         rebalance_days=1,
     )),
    ("low_vol_reb_b60_e40_v60_n2_r20",
     _override(
         "low_vol_rebalance_hybrid",
         base_ratio=0.6,
         enhance_ratio=0.4,
         low_vol_days=60,
         enhance_top_n=2,
         rebalance_days=20,
     )),
    ("low_vol_reb_b70_e30_v60_n2_r20",
     _override(
         "low_vol_rebalance_hybrid",
         base_ratio=0.7,
         enhance_ratio=0.3,
         low_vol_days=60,
         enhance_top_n=2,
         rebalance_days=20,
     )),
    ("bh_low_vol_b60_e40_v60_n2_r20",
     _override(
         "buy_hold_low_vol_hybrid",
         base_ratio=0.6,
         enhance_ratio=0.4,
         low_vol_days=60,
         enhance_top_n=2,
         rebalance_days=20,
     )),
    ("low_pos_b70_e30_n2_r20",
     _override(
         "low_position_hybrid",
         base_ratio=0.7,
         enhance_ratio=0.3,
         enhance_top_n=2,
         rebalance_days=20,
         low_position_days=250,
         short_return_days=20,
         max_recent_return=0.30,
         vol_filter_days=60,
         exclude_top_vol_pct=0.30,
         max_position_ratio=0.35,
     )),
    ("low_pos_tac_b70_e30_n2_r20",
     _override(
         "low_position_tactical_hybrid",
         base_ratio=0.7,
         enhance_ratio=0.3,
         enhance_top_n=2,
         rebalance_days=20,
         low_position_days=250,
         short_return_days=20,
         max_recent_return=0.30,
         vol_filter_days=60,
         exclude_top_vol_pct=0.30,
         max_position_ratio=0.35,
         tactical_reduce=True,
         entry_gain_reduce=0.60,
         entry_gain_cut=0.80,
         recent_gain_reduce=0.25,
     )),
    ("mom_skip_b70_e30_n2_r20",
     _override(
         "momentum_skip_recent_hybrid",
         base_ratio=0.7,
         enhance_ratio=0.3,
         enhance_top_n=2,
         rebalance_days=20,
         momentum_lookback_days=120,
         momentum_skip_days=20,
         max_position_ratio=0.35,
     )),
    ("mom_skip_top2_n2_r1",
     _override(
         "momentum_skip_recent_hybrid",
         base_ratio=0.0,
         enhance_ratio=1.0,
         enhance_top_n=2,
         rebalance_days=1,
         momentum_lookback_days=120,
         momentum_skip_days=20,
         max_position_ratio=0.5,
     )),
    ("mom_skip_top2_n2_r20",
     _override(
         "momentum_skip_recent_hybrid",
         base_ratio=0.0,
         enhance_ratio=1.0,
         enhance_top_n=2,
         rebalance_days=20,
         momentum_lookback_days=120,
         momentum_skip_days=20,
         max_position_ratio=0.5,
     )),
    ("mom_skip_h50_n2_r1",
     _override(
         "momentum_skip_recent_hybrid",
         base_ratio=0.5,
         enhance_ratio=0.5,
         enhance_top_n=2,
         rebalance_days=1,
         momentum_lookback_days=120,
         momentum_skip_days=20,
         max_position_ratio=1.0,
     )),
    ("mom_skip_h50_n2_r20",
     _override(
         "momentum_skip_recent_hybrid",
         base_ratio=0.5,
         enhance_ratio=0.5,
         enhance_top_n=2,
         rebalance_days=20,
         momentum_lookback_days=120,
         momentum_skip_days=20,
         max_position_ratio=1.0,
     )),
    ("pullback_b70_e30_n2_r20",
     _override(
         "pullback_trend_hybrid",
         base_ratio=0.7,
         enhance_ratio=0.3,
         enhance_top_n=2,
         rebalance_days=20,
         momentum_lookback_days=120,
         momentum_skip_days=20,
         min_mid_momentum=0.0,
         max_recent_return=0.30,
         max_position_ratio=0.35,
     )),
    ("pullback_h50_n2_r1",
     _override(
         "pullback_trend_hybrid",
         base_ratio=0.5,
         enhance_ratio=0.5,
         enhance_top_n=2,
         rebalance_days=1,
         momentum_lookback_days=120,
         momentum_skip_days=20,
         min_mid_momentum=0.0,
         max_recent_return=10.0,
         max_position_ratio=1.0,
     )),
    ("pullback_h50_n2_r20",
     _override(
         "pullback_trend_hybrid",
         base_ratio=0.5,
         enhance_ratio=0.5,
         enhance_top_n=2,
         rebalance_days=20,
         momentum_lookback_days=120,
         momentum_skip_days=20,
         min_mid_momentum=0.0,
         max_recent_return=10.0,
         max_position_ratio=1.0,
     )),
    ("pullback_h70_n2_r1",
     _override(
         "pullback_trend_hybrid",
         base_ratio=0.7,
         enhance_ratio=0.3,
         enhance_top_n=2,
         rebalance_days=1,
         momentum_lookback_days=120,
         momentum_skip_days=20,
         min_mid_momentum=0.0,
         max_recent_return=10.0,
         max_position_ratio=1.0,
     )),
    ("pullback_h70_n2_r20",
     _override(
         "pullback_trend_hybrid",
         base_ratio=0.7,
         enhance_ratio=0.3,
         enhance_top_n=2,
         rebalance_days=20,
         momentum_lookback_days=120,
         momentum_skip_days=20,
         min_mid_momentum=0.0,
         max_recent_return=10.0,
         max_position_ratio=1.0,
     )),
    ("moderate_pullback_pos_h50_n2_r1",
     _override(
         "moderate_pullback_pos_trend_hybrid",
         base_ratio=0.5,
         enhance_ratio=0.5,
         enhance_top_n=2,
         rebalance_days=1,
         momentum_lookback_days=120,
         momentum_skip_days=20,
         min_mid_momentum=0.0,
         min_recent_return=-0.20,
         max_recent_return=0.20,
         max_position_ratio=1.0,
     )),
    ("moderate_pullback_pos_h50_n2_r20",
     _override(
         "moderate_pullback_pos_trend_hybrid",
         base_ratio=0.5,
         enhance_ratio=0.5,
         enhance_top_n=2,
         rebalance_days=20,
         momentum_lookback_days=120,
         momentum_skip_days=20,
         min_mid_momentum=0.0,
         min_recent_return=-0.20,
         max_recent_return=0.20,
         max_position_ratio=1.0,
     )),
    ("low_pos_tac_mkt200_b70_e30_n2_r20",
     _override(
         "low_position_tactical_hybrid",
         base_ratio=0.7,
         enhance_ratio=0.3,
         enhance_top_n=2,
         rebalance_days=20,
         low_position_days=250,
         short_return_days=20,
         max_recent_return=0.30,
         vol_filter_days=60,
         exclude_top_vol_pct=0.30,
         max_position_ratio=0.35,
         tactical_reduce=True,
         entry_gain_reduce=0.60,
         entry_gain_cut=0.80,
         recent_gain_reduce=0.25,
         market_filter=True,
         market_index="000300.XSHG",
         market_ma_days=200,
         weak_market_exposure=0.60,
     )),
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


# ── 模式5: 海龟仓位管理 ──
_TURTLE_BASE = {"pool_csv_dir": POOL_DIR}
EXPERIMENTS_TURTLE = [
    # ("turtle",          dict(risk_pct=0.03, max_units=3, stop_atr=5.0, reentry_days=20, **_TURTLE_BASE)),
    # ("turtle_r2",       dict(risk_pct=0.02, max_units=3, stop_atr=5.0, reentry_days=20, **_TURTLE_BASE)),
    # ("turtle_r4",       dict(risk_pct=0.04, max_units=3, stop_atr=5.0, reentry_days=20, **_TURTLE_BASE)),
    # ("turtle_u4",       dict(risk_pct=0.03, max_units=4, stop_atr=5.0, reentry_days=20, **_TURTLE_BASE)),
    # ("turtle_tight",    dict(risk_pct=0.03, max_units=3, stop_atr=3.0, reentry_days=20, **_TURTLE_BASE)),
]


# ==================== 单次回测 ====================
def run_one(tag: str, param_override: dict, strategy_file: str = None):
    config = make_base_config(tag)
    if strategy_file:
        config["base"]["strategy_file"] = strategy_file
    config["extra"]["context_vars"] = {
        "strategy_params": param_override,
    }
    print(f'\n========== 运行: {tag} ==========')
    print(f'参数覆盖: {param_override}')
    result = run_file(strategy_file or STRATEGY_FILE, config=config)
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

    for tag, override in EXPERIMENTS_TURTLE:
        try:
            result = run_one(tag, override, strategy_file=STRATEGY_FILE_TURTLE)
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
