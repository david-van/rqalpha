#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2026/4/21 22:29
# @Author  : david_van
# @Desc    :
import pickle
import pandas as pd

# 读取 pkl 文件
import pickle
from pathlib import Path

# ---- 配置：只需改这一处 ----
pkl_path = Path("batch_results/no_min_score.pkl")

with open(pkl_path, "rb") as f:
    result = pickle.load(f)

# 自动派生输出路径，例如：
#   batch_results/no_min_score.pkl
#   → batch_results/no_min_score_trades.csv
#   → batch_results/no_min_score_portfolio.csv
out_dir = pkl_path.parent          # batch_results
stem    = pkl_path.stem            # no_min_score

trades_csv    = out_dir / f"{stem}_trades.csv"
portfolio_csv = out_dir / f"{stem}_portfolio.csv"

# ---- 查看回测摘要 ----
summary = result["summary"]
for key, value in summary.items():
    print(f"{key}: {value}")

# ---- 查看每日净值曲线 ----
portfolio = result["portfolio"]
print(portfolio.head())
portfolio["unit_net_value"].plot(title=f"策略净值曲线 ({stem})")

# ---- 导出交易记录 ----
trades = result["trades"]
print(trades)
trades.to_csv(trades_csv, encoding="utf-8-sig")
print(f"交易记录已保存: {trades_csv}")

# ---- 导出组合记录 ----
portfolio = result["portfolio"]
print(portfolio)
portfolio.to_csv(portfolio_csv, encoding="utf-8-sig")
print(f"组合记录已保存: {portfolio_csv}")

# ---- 查看持仓记录 ----
positions = result["stock_positions"]
print(positions)

