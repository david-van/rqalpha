# -*- coding: utf-8 -*-
# @Desc : 10万资金 + 周频再平衡 验证 + 容差敏感性
#
# 固定：10万初始资金、真实成本(万8/5元/0.05%税)、MA60滞后1天风控、50/50低波动增强
# 维度1：rebalance_days = 1/3/5/10/20 （用户要求至少5天，看换手-收益权衡）
# 维度2：在 rebalance_days=5 下，容差 pct = 0.1%/0.5%/2% （小资金/高价股下容差影响）
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from my_strategy.common_file import project_root
from my_strategy.strategies.xiaoe.run_xiaoe import make_base_config, _override

RESULT_DIR = Path(project_root) / 'my_strategy' / 'strategies' / 'batch_results' / 'xiaoe_pool' / 'risk_overlay'
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def _lv(**kw):
    base = dict(
        base_ratio=0.5,
        enhance_ratio=0.5,
        low_vol_days=60,
        enhance_top_n=2,
        market_filter=True,
        market_ma_days=60,
        weak_market_exposure=0.60,
        market_filter_lag=True,
    )
    base.update(kw)
    return _override("low_vol_rebalance_hybrid", **base)


def _real_cost(config):
    config["mod"]["sys_transaction_cost"]["stock_commission_multiplier"] = 1.0
    config["mod"]["sys_transaction_cost"]["stock_min_commission"] = 5
    config["mod"]["sys_transaction_cost"]["tax_multiplier"] = 1.0
    return config


EXPERIMENTS = [
    # 再平衡频率扫描（容差 0.1%）
    ("r1_mkt60_lag1", _lv(rebalance_days=1)),
    ("r3_mkt60_lag1", _lv(rebalance_days=3)),
    ("r5_mkt60_lag1", _lv(rebalance_days=5)),
    ("r10_mkt60_lag1", _lv(rebalance_days=10)),
    ("r20_mkt60_lag1", _lv(rebalance_days=20)),
    # 容差敏感性（固定周频 5 天）
    ("r5_mkt60_lag1_tol05", _lv(rebalance_days=5, rebalance_tolerance_pct=0.005)),
    ("r5_mkt60_lag1_tol2", _lv(rebalance_days=5, rebalance_tolerance_pct=0.02)),
]


def extract(result):
    summary = result.get('sys_analyser', {}).get('summary', {})
    trades = result.get('sys_analyser', {}).get('trades')
    cost = None
    n_trades = None
    if trades is not None and len(trades):
        n_trades = len(trades)
        if 'transaction_cost' in trades.columns:
            cost = float(trades['transaction_cost'].fillna(0).sum())
    return {
        'total_returns': summary.get('total_returns'),
        'annualized': summary.get('annualized_returns'),
        'sharpe': summary.get('sharpe'),
        'max_drawdown': summary.get('max_drawdown'),
        'turnover': summary.get('turnover'),
        'trades': n_trades,
        'cost': cost,
    }


def main():
    rows = []
    t0 = time.time()
    from rqalpha import run_file
    for tag, override in EXPERIMENTS:
        t = time.time()
        try:
            config = _real_cost(make_base_config(tag))
            config["extra"]["context_vars"] = {"strategy_params": override}
            result = run_file(config["base"]["strategy_file"], config=config)
            rows.append({'tag': tag, **extract(result)})
            print(f"[ok] {tag}  {time.time() - t:.0f}s", flush=True)
        except Exception as e:
            print(f"[fail] {tag}: {e}", flush=True)
            rows.append({'tag': tag, 'error': str(e)})

    df = pd.DataFrame(rows)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = RESULT_DIR / f'weekly_validate_{stamp}.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    show = df.copy()
    for c in ('total_returns', 'annualized', 'max_drawdown'):
        show[c] = show[c].map(lambda x: '' if pd.isna(x) else f'{x:.2%}')
    for c in ('sharpe', 'turnover'):
        show[c] = show[c].map(lambda x: '' if pd.isna(x) else f'{x:.2f}')
    if 'cost' in show.columns:
        show['cost'] = show['cost'].map(lambda x: '' if pd.isna(x) else f'{x:.0f}')
    print('\n===== 10万资金 + 周频再平衡验证 =====')
    print(show.to_string(index=False))
    print(f'\n已保存: {csv_path}')
    print(f'总耗时: {time.time() - t0:.0f}s')
    return df


if __name__ == '__main__':
    main()
