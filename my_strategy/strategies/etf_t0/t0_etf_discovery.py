#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 RQAlpha 的 instruments.pk 中发现所有 T+0 ETF 并自动分类。

在 RQAlpha 中，`market_tplus=0` 表示 T+0 交易。
ETF 根据底层标的（underlying_order_book_id）可分为：
  - cross_border: 跨境 ETF，底层为海外指数（HSI/NDX/SPX/N225/GDAXI 等）
  - gold:        黄金 ETF，底层为 Au99.99 / SHAU
  - bond:        债券 ETF，底层为债券指数
  - money_market: 货币 ETF，底层为空
  - commodity:   商品期货 ETF
  - other:       其他
"""

from __future__ import annotations

import pickle
from pathlib import Path

# ---------------------------------------------------------------
# 跨境 ETF 的底层标的特征（海外交易所指数）
# ---------------------------------------------------------------
# INDX 后缀的来自 RQAlpha 自定义指数，如 HSI.INDX / NDX.INDX
# XSHG/XSHE 后缀的多为境内指数
CROSS_BORDER_PATTERNS = [
    # 港股
    "HSI", "HSCEI", "HSTECH", "HSHCI", "HSIII", "HSBIO", "HSSC",
    "HSSCHKY", "HSSCHI", "HSSCT", "HSSCNE", "HSSC50", "HSSCITI",
    "HSSCSOY", "HSSCAM", "HSIDI", "HSCGSI", "HSHYLV", "SPAHLVCP",
    "931239",  # 港股通消费
    "931454",  # 港股通新经济
    "931573",  # 港股通科技
    "931637",  # 港股通互联网
    "931722",  # 港股通创新药
    "931787",  # 港股通创新药 (variant)
    "931790",  # 港股芯片
    "930604",  # 中国互联网30
    "930709",  # 港股证券
    "930796",  # 东南亚科技
    "930839",  # 港股核心消费
    "930914",  # 港股高股息
    "930931",  # 港股通50
    "930957",  # 港股通中国100
    "930965",  # 港股医药
    "930967",  # 港股信息
    "931018",  # (bond, skip)
    "931024",  # 港股金融
    "931233",  # 港股汽车
    "931250",  # 港股创新药
    "931552",  # (bond)
    "931574",  # 港股科技
    "987008",  # 港股通科技30
    "987016",  # 港股通低波红利
    "987018",  # 港股通创新药
    "987022",  # 港股通消费
    "987024",  # 港股通互联网
    "932069",  # 港股医药
    "H30106",  # 港股金融
    "H30533",  # 中概互联50
    "H50069",  # 港股通100
    "H11098",  # (bond)
    "H11146",  # 港股通内地金融
    "H11153",  # 港股通央企
    "H11014",  # (bond)
    "CES100",  # CES港股通100
    # 美股
    "NDX", "SPX", "SPTR500N", "DJI", "NBI",
    "SPSIBI", "SPSIOP", "SP5CSSUP", "SPNCSCHN",
    # 欧洲
    "GDAXI", "CAC40",
    # 日本
    "N225", "TPX",
    # 东南亚 / 其他
    "EMASIAUP", "GPCSP006", "ASIATECP", "FISAULM", "BOVESPA",
    # MSCI 海外
    "750108",
]

# 黄金
GOLD_PATTERNS = ["Au99.99", "SHAU"]

# 债券
BOND_PATTERNS = [
    "CBA", "CBC", "H01077", "950045", "950175", "950245",
    "921128", "921130", "932160", "931161", "930865",
    "950041", "950109", "950243", "000140", "931078",
    "932200", "931018", "931552", "H11098", "H11014",
]

# 商品期货
COMMODITY_PATTERNS = ["IMCI", "000201", "000014"]

# 货币 ETF（underlying 为空）
MONEY_MARKET_NAMES = ["货币", "现金", "保證金", "保证金", "钱袋子", "添益", "快线"]


def classify_etf(item: dict) -> str:
    """根据 instruments.pk 中的一条记录判断 ETF 类别."""
    underlying = str(item.get("underlying_order_book_id") or "")
    name = str(item.get("symbol", ""))
    # 币 ETF 的 underlying 为空或 None 字符串就是货币 ETF
    if not underlying or underlying == "None":
        return "money_market"

    underlying_upper = underlying.upper()

    for pat in GOLD_PATTERNS:
        if pat.upper() in underlying_upper:
            return "gold"

    for pat in COMMODITY_PATTERNS:
        if pat.upper() in underlying_upper:
            return "commodity"

    for pat in BOND_PATTERNS:
        if pat.upper() in underlying_upper:
            return "bond"

    for pat in CROSS_BORDER_PATTERNS:
        if pat.upper() in underlying_upper:
            return "cross_border"

    # 名称兜底
    for kw in MONEY_MARKET_NAMES:
        if kw in name:
            return "money_market"

    return "other"


def discover_t0_etfs(bundle_path: str | None = None) -> list[dict]:
    """从 instruments.pk 发现所有 T+0 ETF，返回分类后的列表."""
    if bundle_path is None:
        bundle_path = r"D:\datas\bundle"

    pk_path = Path(bundle_path) / "instruments.pk"
    if not pk_path.exists():
        raise FileNotFoundError(f"instruments.pk 不存在: {pk_path}")

    with open(pk_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    results = []
    for item in data:
        if item.get("type") != "ETF":
            continue
        if item.get("market_tplus") != 0:
            continue
        if item.get("status") != "Active":
            continue

        category = classify_etf(item)
        results.append({
            "order_book_id": item["order_book_id"],
            "name": item.get("symbol", ""),
            "market": item.get("exchange", "")[-2:],  # SH or SZ
            "t0_type": category,
            "underlying": item.get("underlying_order_book_id") or "",
            "listed_date": item.get("listed_date", ""),
            "round_lot": item.get("round_lot", 100),
        })

    return sorted(results, key=lambda x: (x["t0_type"], x["order_book_id"]))


def print_summary(etfs: list[dict]) -> None:
    """打印各类别 ETF 汇总."""
    from collections import Counter
    cats = Counter(e["t0_type"] for e in etfs)
    print(f"\nT+0 ETF 总数: {len(etfs)}")
    print(f"  跨境 (cross_border): {cats.get('cross_border', 0)} 只")
    print(f"  黄金 (gold):          {cats.get('gold', 0)} 只")
    print(f"  债券 (bond):          {cats.get('bond', 0)} 只")
    print(f"  货币 (money_market):  {cats.get('money_market', 0)} 只")
    print(f"  商品 (commodity):     {cats.get('commodity', 0)} 只")
    print(f"  其他 (other):         {cats.get('other', 0)} 只")

    print(f"\n--- 跨境 ETF (适合高低开策略) ---")
    for e in etfs:
        if e["t0_type"] == "cross_border":
            print(f"  {e['order_book_id']:20s} {e['name'][:30]:30s} 底层: {e['underlying']}")

    if cats.get("other", 0) > 0:
        print(f"\n--- 其他/未分类 ---")
        for e in etfs:
            if e["t0_type"] == "other":
                print(f"  {e['order_book_id']:20s} {e['name'][:30]:30s} 底层: {e['underlying']}")


if __name__ == "__main__":
    etfs = discover_t0_etfs()
    print_summary(etfs)
