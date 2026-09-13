#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc : 用回测真实持仓(stock_positions)重建各窗口"增强仓 top2"，
#        定位 30/60/120 之间到底在哪些时间段、拿了哪些不同的票。
import pickle
from collections import Counter
from pathlib import Path

import pandas as pd

from my_strategy.common_file import project_root

RESULT_DIR = Path(project_root) / 'my_strategy' / 'strategies' / 'batch_results' / 'xiaoe_pool' / 'lowvol_mktfilter'
TAGS = ["lowvol30", "lowvol60", "lowvol120"]

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


def load(tag):
    with open(RESULT_DIR / f"{tag}.pkl", "rb") as f:
        return pickle.load(f)


def daily_top2(tag):
    """按市值取每日 top2 = 增强仓，返回 {date: [code, code]}"""
    sp = load(tag)["stock_positions"].copy()
    if not isinstance(sp.index, pd.DatetimeIndex):
        sp.index = pd.to_datetime(sp.index)
    out = {}
    for dt, g in sp.groupby(sp.index):
        top2 = g.sort_values("market_value", ascending=False).head(2)["order_book_id"].tolist()
        out[dt] = top2
    return out


def main():
    tops = {t: daily_top2(t) for t in TAGS}
    dates = sorted(set().union(*[set(d.keys()) for d in tops.values()]))
    print(f"持仓快照日期数: {len(dates)}")

    # 1) 各窗口每只票被列为 top2 的天数
    print("\n=== 各股票被列入增强仓(top2)的天数 ===")
    cnt = {t: Counter() for t in TAGS}
    for t in TAGS:
        for dt in dates:
            for c in tops[t].get(dt, []):
                cnt[t][c] += 1
    allc = set().union(*[set(c.keys()) for c in cnt.values()])
    print(f"{'股票':<8}{'30天':>7}{'60天':>7}{'120天':>7}")
    for c in sorted(allc, key=lambda x: -cnt["lowvol30"][x]):
        print(f"{NAMES.get(c,c):<8}{cnt['lowvol30'][c]:>7}{cnt['lowvol60'][c]:>7}{cnt['lowvol120'][c]:>7}")

    # 2) 三窗口 top2 两两不一致天数
    print("\n=== 增强仓 top2 两两不一致天数 ===")
    for a, b in [("lowvol30", "lowvol60"), ("lowvol60", "lowvol120"), ("lowvol30", "lowvol120")]:
        diff = sum(1 for dt in dates if set(tops[a].get(dt, [])) != set(tops[b].get(dt, [])))
        print(f"  {a} vs {b}: {diff}/{len(dates)} 天不一致")


if __name__ == "__main__":
    main()
