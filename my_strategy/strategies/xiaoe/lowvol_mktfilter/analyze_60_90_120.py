#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc : 跑 low_vol_days=90 回测，然后对比 60/90/120 三窗口：
#        1) 总指标 2) 逐年收益 3) 增强仓真实选股差异
import pickle
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from my_strategy.common_file import project_root
from rqalpha import run_file
from my_strategy.strategies.xiaoe.lowvol_mktfilter.run_backtest import (
    STRATEGY_FILE, RESULT_DIR, make_base_config,
)

NAMES = {
    "000830.XSHE": "鲁西化工", "000876.XSHE": "新希望", "002053.XSHE": "云南能投",
    "002381.XSHE": "双箭股份", "002478.XSHE": "常宝股份", "002507.XSHE": "涪陵榨菜",
    "002539.XSHE": "云图控股", "002841.XSHE": "视源股份", "300006.XSHE": "莱美药业",
    "300019.XSHE": "硅宝科技", "300092.XSHE": "科新机电", "300121.XSHE": "阳谷华泰",
    "300255.XSHE": "常山药业", "300401.XSHE": "花园生物", "300435.XSHE": "中泰股份",
    "300547.XSHE": "川环科技", "300596.XSHE": "利安隆", "300786.XSHE": "国林科技",
    "301373.XSHE": "凌玮科技", "600129.XSHG": "太极集团", "600452.XSHG": "涪陵电力",
    "603325.XSHG": "博隆技术", "603601.XSHG": "再升科技", "603612.XSHG": "索通发展",
    "603758.XSHG": "秦安股份", "605077.XSHG": "华康股份",
}

BASE = {"base_ratio": 0.5, "enhance_ratio": 0.5, "enhance_top_n": 2,
        "rebalance_days": 5, "market_filter": True, "market_ma_days": 60,
        "weak_market_exposure": 0.60, "market_filter_lag": True}


def run_lowvol90():
    pkl = RESULT_DIR / "lowvol90.pkl"
    if pkl.exists():
        print(f"lowvol90.pkl 已存在，跳过回测")
        return
    print("运行 lowvol90 回测...")
    config = make_base_config("lowvol90")
    params = dict(BASE)
    params["low_vol_days"] = 90
    config["extra"]["context_vars"] = {"strategy_params": params}
    run_file(STRATEGY_FILE, config=config)
    print("lowvol90 完成")


def load(tag):
    with open(RESULT_DIR / f"{tag}.pkl", "rb") as f:
        return pickle.load(f)


def yearly(tag):
    nv = load(tag)["portfolio"]["unit_net_value"].copy()
    if not isinstance(nv.index, pd.DatetimeIndex):
        nv.index = pd.to_datetime(nv.index)
    return {y: float(g.iloc[-1] / g.iloc[0] - 1) for y, g in nv.groupby(nv.index.year)}


def top2_days(tag):
    sp = load(tag)["stock_positions"].copy()
    if not isinstance(sp.index, pd.DatetimeIndex):
        sp.index = pd.to_datetime(sp.index)
    out = {}
    for dt, g in sp.groupby(sp.index):
        out[dt] = g.sort_values("market_value", ascending=False).head(2)["order_book_id"].tolist()
    return out


def main():
    run_lowvol90()
    TAGS = ["lowvol60", "lowvol90", "lowvol120"]

    # 1) 总指标
    print("\n=== 总指标对比 (60/90/120) ===")
    print(f"{'指标':<10}{'60天':>10}{'90天':>10}{'120天':>10}")
    for key, fmt in [("total_returns", ".2%"), ("annualized_returns", ".2%"),
                     ("max_drawdown", ".2%"), ("sharpe", ".3f"), ("turnover", ".1f")]:
        vals = [load(t)["summary"].get(key, 0) for t in TAGS]
        print(f"{key:<10}" + "".join(f"{v:>{10 if fmt!='.3f' else 10}}" for v in vals))

    # 2) 逐年
    print("\n=== 逐年收益 ===")
    y = {t: yearly(t) for t in TAGS}
    years = sorted({yy for d in y.values() for yy in d})
    print(f"{'年份':<6}{'60天':>10}{'90天':>10}{'120天':>10}")
    for yy in years:
        print(f"{yy:<6}{y['lowvol60'].get(yy,0):>10.1%}{y['lowvol90'].get(yy,0):>10.1%}{y['lowvol120'].get(yy,0):>10.1%}")

    # 3) 增强仓选股
    tops = {t: top2_days(t) for t in TAGS}
    dates = sorted(set().union(*[set(d.keys()) for d in tops.values()]))
    cnt = {t: Counter() for t in TAGS}
    for t in TAGS:
        for dt in dates:
            for c in tops[t].get(dt, []):
                cnt[t][c] += 1
    allc = set().union(*[set(c.keys()) for c in cnt.values()])
    print("\n=== 增强仓(top2)入选天数 (前12) ===")
    print(f"{'股票':<8}{'60天':>7}{'90天':>7}{'120天':>7}")
    for c in sorted(allc, key=lambda x: -cnt["lowvol60"][x])[:12]:
        print(f"{NAMES.get(c,c):<8}{cnt['lowvol60'][c]:>7}{cnt['lowvol90'][c]:>7}{cnt['lowvol120'][c]:>7}")

    # 4) 两两不一致
    print("\n=== 增强仓 top2 两两不一致天数 ===")
    for a, b in [("lowvol60", "lowvol90"), ("lowvol90", "lowvol120"), ("lowvol60", "lowvol120")]:
        diff = sum(1 for dt in dates if set(tops[a].get(dt, [])) != set(tops[b].get(dt, [])))
        print(f"  {a} vs {b}: {diff}/{len(dates)} 天不一致")


if __name__ == "__main__":
    main()
