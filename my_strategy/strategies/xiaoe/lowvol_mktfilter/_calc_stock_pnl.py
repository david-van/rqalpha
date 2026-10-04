#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc : 计算策略在每只股票上的实际盈亏（已实现 + 期末浮盈浮亏），
#        口径：卖出回收 + 期末持仓市值 - 买入成本 - 手续费。
import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd

from my_strategy.common_file import project_root

RESULT_DIR = Path(project_root) / 'my_strategy' / 'strategies' / 'batch_results' / 'xiaoe_pool' / 'lowvol_mktfilter'

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


def trade_cost(row):
    if "transaction_cost" in row.index and pd.notna(row.get("transaction_cost")):
        return float(row["transaction_cost"])
    c = 0.0
    for k in ("commission", "tax"):
        if k in row.index and pd.notna(row.get(k)):
            c += float(row[k])
    return c


def main():
    with open(RESULT_DIR / "final.pkl", "rb") as f:
        d = pickle.load(f)

    trades = d["trades"].copy()
    if not isinstance(trades.index, pd.DatetimeIndex):
        trades.index = pd.to_datetime(trades.index)

    # 期末持仓：只取最后一天的快照（已卖出的票期末数量=0，不计入期末市值）
    sp = d["stock_positions"].copy()
    if not isinstance(sp.index, pd.DatetimeIndex):
        sp.index = pd.to_datetime(sp.index)
    final_snap = sp[sp.index == sp.index.max()]
    final_qty = {row["order_book_id"]: float(row["quantity"]) for _, row in final_snap.iterrows()}
    final_price = {row["order_book_id"]: float(row["last_price"]) for _, row in final_snap.iterrows()}

    agg = defaultdict(lambda: {"buy": 0.0, "sell": 0.0, "fee": 0.0, "n_buy": 0, "n_sell": 0})
    for _, row in trades.iterrows():
        code = row["order_book_id"]
        side = str(row.get("side", "")).upper()
        qty = abs(float(row["last_quantity"]))
        price = float(row["last_price"])
        fee = trade_cost(row)
        if side == "BUY":
            agg[code]["buy"] += price * qty
            agg[code]["n_buy"] += 1
        else:
            agg[code]["sell"] += price * qty
            agg[code]["n_sell"] += 1
        agg[code]["fee"] += fee

    rows = []
    for code, a in agg.items():
        fq = final_qty.get(code, 0.0)
        fp = final_price.get(code, 0.0)
        final_val = fq * fp
        pnl = a["sell"] + final_val - a["buy"] - a["fee"]
        rows.append((NAMES.get(code, code), code, a["n_buy"], a["n_sell"],
                     a["buy"], a["sell"], final_val, a["fee"], pnl))

    rows.sort(key=lambda x: -x[8])

    lines = ["股票名称\t代码\t买入次数\t卖出次数\t买入总额\t卖出总额\t期末市值\t手续费\t净盈亏(元)"]
    for name, code, nb, ns, buy, sell, fv, fee, pnl in rows:
        lines.append(f"{name}\t{code}\t{nb}\t{ns}\t{buy:.0f}\t{sell:.0f}\t{fv:.0f}\t{fee:.0f}\t{pnl:+.0f}")

    # 汇总
    total_pnl = sum(r[8] for r in rows)
    lines.append("")
    lines.append(f"合计净盈亏: {total_pnl:+.0f} 元")

    out = Path(project_root) / "my_strategy" / "strategies" / "xiaoe" / "lowvol_mktfilter" / "_stock_pnl.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("written", len(rows))


if __name__ == "__main__":
    main()
