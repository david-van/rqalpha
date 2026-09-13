#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
小鹅通低波动策略 — 过拟合验证脚本

验证三件事：
  1) 参数敏感性：逐维扫描关键参数，看结果是不是"尖锐峰值"还是"平稳平台"
  2) 时间段稳定性：把 2020-2026 切成两半，看策略在两段是否都有效
  3) 逐年收益：默认配置的逐年收益，看是否靠某一年撑起来的

运行（仓库根目录）:
    .venv/Scripts/python.exe -u my_strategy/strategies/xiaoe/lowvol_mktfilter/validate_overfit.py

结果输出:
    my_strategy/strategies/batch_results/xiaoe_pool/lowvol_mktfilter/
"""
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from my_strategy.common_file import project_root
from rqalpha import run_file

from my_strategy.strategies.xiaoe.lowvol_mktfilter.run_backtest import (
    STRATEGY_FILE, RESULT_DIR, make_base_config,
)

FINAL = {}
BASE = {"base_ratio": 0.5, "enhance_ratio": 0.5, "enhance_top_n": 2, "low_vol_days": 60,
        "rebalance_days": 5, "market_ma_days": 60, "weak_market_exposure": 0.60}


def _ov(**kw):
    base = dict(BASE)
    base.update(kw)
    return base


def _run(tag, override, start, end):
    config = make_base_config(tag, start=start, end=end)
    config["extra"]["context_vars"] = {"strategy_params": override}
    result = run_file(STRATEGY_FILE, config=config)
    summary = result.get('sys_analyser', {}).get('summary', {})
    trades = result.get('sys_analyser', {}).get('trades')
    n_trades = len(trades) if trades is not None else None
    return {
        'tag': tag,
        'total_returns': summary.get('total_returns'),
        'annualized': summary.get('annualized_returns'),
        'sharpe': summary.get('sharpe'),
        'max_drawdown': summary.get('max_drawdown'),
        'turnover': summary.get('turnover'),
        'trades': n_trades,
    }


def sensitivity():
    """参数敏感性：每次只改一个参数，其余保持默认"""
    experiments = [
        # (tag, override)
        ("lowvol30", _ov(low_vol_days=30)),
        ("lowvol60", _ov(low_vol_days=60)),
        ("lowvol120", _ov(low_vol_days=120)),
        ("top1", _ov(enhance_top_n=1)),
        ("top2", _ov(enhance_top_n=2)),
        ("top3", _ov(enhance_top_n=3)),
        ("mktma30", _ov(market_ma_days=30)),
        ("mktma60", _ov(market_ma_days=60)),
        ("mktma120", _ov(market_ma_days=120)),
        ("mktma200", _ov(market_ma_days=200)),
        ("base40", _ov(base_ratio=0.4, enhance_ratio=0.6)),
        ("base50", _ov(base_ratio=0.5, enhance_ratio=0.5)),
        ("base60", _ov(base_ratio=0.6, enhance_ratio=0.4)),
        ("base70", _ov(base_ratio=0.7, enhance_ratio=0.3)),
        ("rebal3", _ov(rebalance_days=3)),
        ("rebal5", _ov(rebalance_days=5)),
        ("rebal10", _ov(rebalance_days=10)),
        ("rebal20", _ov(rebalance_days=20)),
        ("weak40", _ov(weak_market_exposure=0.40)),
        ("weak60", _ov(weak_market_exposure=0.60)),
        ("weak80", _ov(weak_market_exposure=0.80)),
    ]
    rows = []
    for tag, override in experiments:
        t = time.time()
        try:
            rows.append(_run(tag, override, "2020-01-01", "2026-01-01"))
            print(f"[ok] {tag}  {time.time() - t:.0f}s", flush=True)
        except Exception as e:
            print(f"[fail] {tag}: {e}", flush=True)
            rows.append({'tag': tag, 'error': str(e)})
    return pd.DataFrame(rows)


def time_split():
    """时间段稳定性：前半段 2020-2022，后半段 2023-2025"""
    experiments = [
        ("full", FINAL, "2020-01-01", "2026-01-01"),
        ("first_half_2020_2022", FINAL, "2020-01-01", "2023-01-01"),
        ("second_half_2023_2025", FINAL, "2023-01-01", "2026-01-01"),
    ]
    rows = []
    for tag, override, start, end in experiments:
        t = time.time()
        try:
            rows.append(_run(tag, override, start, end))
            print(f"[ok] {tag}  {time.time() - t:.0f}s", flush=True)
        except Exception as e:
            print(f"[fail] {tag}: {e}", flush=True)
            rows.append({'tag': tag, 'error': str(e)})
    return pd.DataFrame(rows)


def yearly_breakdown():
    """默认配置的逐年收益"""
    config = make_base_config("final", start="2020-01-01", end="2026-01-01")
    config["extra"]["context_vars"] = {"strategy_params": FINAL}
    result = run_file(STRATEGY_FILE, config=config)
    nv = result['sys_analyser']['portfolio']['unit_net_value'].sort_index()
    yearly = {}
    for y, g in nv.groupby(nv.index.year):
        yearly[y] = float(g.iloc[-1] / g.iloc[0] - 1)
    return pd.Series(yearly)


def main():
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 70)
    print("[1/3] 参数敏感性")
    print("=" * 70)
    sens = sensitivity()
    sens_path = RESULT_DIR / f'sensitivity_{stamp}.csv'
    sens.to_csv(sens_path, index=False, encoding='utf-8-sig')
    show = sens.copy()
    for c in ('total_returns', 'annualized', 'max_drawdown'):
        show[c] = show[c].map(lambda x: '' if pd.isna(x) else f'{x:.2%}')
    show['sharpe'] = show['sharpe'].map(lambda x: '' if pd.isna(x) else f'{x:.2f}')
    print(show.to_string(index=False))
    print(f'已保存: {sens_path}\n')

    print("=" * 70)
    print("[2/3] 时间段稳定性")
    print("=" * 70)
    ts = time_split()
    ts_path = RESULT_DIR / f'timesplit_{stamp}.csv'
    ts.to_csv(ts_path, index=False, encoding='utf-8-sig')
    show = ts.copy()
    for c in ('total_returns', 'annualized', 'max_drawdown'):
        show[c] = show[c].map(lambda x: '' if pd.isna(x) else f'{x:.2%}')
    show['sharpe'] = show['sharpe'].map(lambda x: '' if pd.isna(x) else f'{x:.2f}')
    print(show.to_string(index=False))
    print(f'已保存: {ts_path}\n')

    print("=" * 70)
    print("[3/3] 逐年收益（默认配置）")
    print("=" * 70)
    yearly = yearly_breakdown()
    for y in sorted(yearly.index):
        print(f"  {y}: {yearly[y]:+.2%}")
    yearly_path = RESULT_DIR / f'yearly_{stamp}.csv'
    yearly.rename('return').to_csv(yearly_path, encoding='utf-8-sig')
    print(f'已保存: {yearly_path}\n')


if __name__ == '__main__':
    main()
