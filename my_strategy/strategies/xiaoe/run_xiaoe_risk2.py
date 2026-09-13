#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc : 风控叠加稳健性验证 —— 真实成本 + 趋势过滤滞后1天（消除前视）
#
# 真实成本 = rqalpha 默认：佣金万8、最低5元、卖出印花税0.05%
# 对比维度：
#   1) 成本敏感性：cheap(万0.5/无税) vs real(万8/5元/0.05%税)
#   2) 前视检验：   market_filter_lag=False(用当日收盘) vs True(只用昨日及以前)
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from my_strategy.common_file import project_root
from my_strategy.strategies.xiaoe.run_xiaoe import make_base_config, run_one, _override

RESULT_DIR = Path(project_root) / 'my_strategy' / 'strategies' / 'batch_results' / 'xiaoe_pool' / 'risk_overlay'
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def _lv(**kw):
    base = dict(
        base_ratio=0.5,
        enhance_ratio=0.5,
        low_vol_days=60,
        enhance_top_n=2,
        rebalance_days=1,
    )
    base.update(kw)
    return _override("low_vol_rebalance_hybrid", **base)


def _real_cost(config):
    config["mod"]["sys_transaction_cost"]["stock_commission_multiplier"] = 1.0
    config["mod"]["sys_transaction_cost"]["stock_min_commission"] = 5
    config["mod"]["sys_transaction_cost"]["tax_multiplier"] = 1.0
    return config


EXPERIMENTS = [
    # (tag, override, realistic_cost?)
    ("lv_r1_base_rc", _lv(), True),
    ("lv_r1_mkt60_w60_rc", _lv(market_filter=True, market_ma_days=60, weak_market_exposure=0.60), True),
    ("lv_r1_mkt60_w60_lag1_rc", _lv(market_filter=True, market_ma_days=60, weak_market_exposure=0.60,
                                    market_filter_lag=True), True),
    ("lv_r1_mkt200_w60_rc", _lv(market_filter=True, market_ma_days=200, weak_market_exposure=0.60), True),
    ("lv_r1_mkt200_w60_lag1_rc", _lv(market_filter=True, market_ma_days=200, weak_market_exposure=0.60,
                                      market_filter_lag=True), True),
    ("lv_r1_mkt200_w40_rc", _lv(market_filter=True, market_ma_days=200, weak_market_exposure=0.40), True),
    ("lv_r1_mkt200_w40_lag1_rc", _lv(market_filter=True, market_ma_days=200, weak_market_exposure=0.40,
                                      market_filter_lag=True), True),
]


def extract(result):
    summary = result.get('sys_analyser', {}).get('summary', {})
    cost = None
    try:
        trades = result.get('sys_analyser', {}).get('trades')
        if trades is not None and len(trades) and 'transaction_cost' in trades.columns:
            cost = float(trades['transaction_cost'].fillna(0).sum())
    except Exception:
        pass
    return {
        'total_returns': summary.get('total_returns'),
        'annualized': summary.get('annualized_returns'),
        'sharpe': summary.get('sharpe'),
        'max_drawdown': summary.get('max_drawdown'),
        'turnover': summary.get('turnover'),
        'cost': cost,
    }


def main():
    rows = []
    t0 = time.time()
    for tag, override, real in EXPERIMENTS:
        t = time.time()
        try:
            config = make_base_config(tag)
            if real:
                config = _real_cost(config)
            config["extra"]["context_vars"] = {"strategy_params": override}
            from rqalpha import run_file
            result = run_file(config["base"]["strategy_file"], config=config)
            row = {'tag': tag, **extract(result)}
            rows.append(row)
            print(f"[ok] {tag}  {time.time() - t:.0f}s", flush=True)
        except Exception as e:
            print(f"[fail] {tag}: {e}", flush=True)
            rows.append({'tag': tag, 'error': str(e)})

    df = pd.DataFrame(rows)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = RESULT_DIR / f'robustness_{stamp}.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    show = df.copy()
    for c in ('total_returns', 'annualized', 'max_drawdown'):
        if c in show.columns:
            show[c] = show[c].map(lambda x: '' if pd.isna(x) else f'{x:.2%}')
    for c in ('sharpe', 'turnover'):
        if c in show.columns:
            show[c] = show[c].map(lambda x: '' if pd.isna(x) else f'{x:.3f}')
    if 'cost' in show.columns:
        show['cost'] = show['cost'].map(lambda x: '' if pd.isna(x) else f'{x:.0f}')
    print('\n===== 真实成本 + 滞后1天稳健性 =====')
    print(show.to_string(index=False))
    print(f'\n已保存: {csv_path}')
    print(f'总耗时: {time.time() - t0:.0f}s')
    return df


if __name__ == '__main__':
    main()
