#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2026/4/21 22:29
# @Author  : david_van
# @Desc    :
import pickle
import pandas as pd

# 读取 pkl 文件
with open("result.pkl", "rb") as f:
    result = pickle.load(f)

# ---- 查看回测摘要 ----
summary = result["summary"]
for key, value in summary.items():
    print(f"{key}: {value}")

# 常见字段：
# total_returns:        总收益率
# annualized_returns:   年化收益率
# max_drawdown:         最大回撤
# sharpe:               夏普比率
# volatility:           波动率

# ---- 查看每日净值曲线 ----
portfolio = result["portfolio"]
print(portfolio.head())
# 可以画图
portfolio["unit_net_value"].plot(title="策略净值曲线")

# ---- 查看交易记录 ----
trades = result["trades"]
print(trades)
result["trades"].to_csv("trades.csv", encoding="utf-8-sig")

# ---- 查看持仓记录 ----
positions = result["stock_positions"]
print(positions)
