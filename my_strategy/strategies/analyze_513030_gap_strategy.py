#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analyze 513030 open-gap intraday strategy from an RQAlpha bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


DEFAULT_BUNDLE = r"D:\datas\bundle"
DEFAULT_SECURITY = "513030.XSHG"
DEFAULT_START_DATE = "2023-01-03"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze buy-at-open/sell-at-close returns under different "
            "open-gap thresholds and buckets."
        )
    )
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE, help="RQAlpha bundle path")
    parser.add_argument("--security", default=DEFAULT_SECURITY, help="Security id")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--step", type=float, default=0.001, help="Gap bucket step")
    parser.add_argument("--min-gap", type=float, default=-0.02)
    parser.add_argument("--max-gap", type=float, default=0.04)
    parser.add_argument(
        "--commission-rate",
        type=float,
        default=0.0,
        help="One-way commission rate. Default 0 means gross return.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/gap_strategy_513030",
        help="Output directory",
    )
    return parser.parse_args()


def resolve_security(keys: list[str], security: str) -> str:
    security = security.upper()
    if security in keys:
        return security

    code = security.split(".")[0]
    matches = [key for key in keys if key.split(".")[0] == code]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"{security} not found in funds.h5")
    raise KeyError(f"{security} is ambiguous: {matches}")


def load_daily_bars(bundle: str, security: str) -> tuple[str, pd.DataFrame]:
    h5_path = Path(bundle) / "funds.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"funds.h5 not found: {h5_path}")

    with h5py.File(h5_path, "r") as store:
        order_book_id = resolve_security(list(store.keys()), security)
        data = store[order_book_id][:]

    df = pd.DataFrame.from_records(data)
    df["date"] = pd.to_datetime(
        df["datetime"].astype(str).str.slice(0, 8), format="%Y%m%d"
    )
    df = df.sort_values("date").reset_index(drop=True)
    return order_book_id, df


def prepare_trades(
    df: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
    commission_rate: float,
) -> pd.DataFrame:
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    df = df.copy()
    df = df[(df["open"] > 0) & (df["close"] > 0) & (df["prev_close"] > 0)]
    df["year"] = df["date"].dt.year
    df["gap"] = df["open"] - df["prev_close"]
    df["gap_pct"] = df["gap"] / df["prev_close"]
    df["gross_return"] = df["close"] / df["open"] - 1
    df["net_return"] = (
        df["close"] * (1 - commission_rate) / (df["open"] * (1 + commission_rate))
        - 1
    )
    df["pnl_price"] = df["close"] - df["open"]
    return df.reset_index(drop=True)


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    equity = np.r_[1.0, (1.0 + returns.to_numpy()).cumprod()]
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def summarize(group: pd.DataFrame, return_col: str) -> dict[str, float | int]:
    count = int(len(group))
    if count == 0:
        return {
            "交易次数": 0,
            "胜率": 0.0,
            "平均收益率": 0.0,
            "中位数收益率": 0.0,
            "累计收益率": 0.0,
            "最大回撤": 0.0,
            "平均价差收益": 0.0,
            "累计价差收益": 0.0,
            "盈亏比": np.nan,
        }

    returns = group[return_col]
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    equity_return = float((1.0 + returns).prod() - 1.0)
    profit_loss_ratio = float(gains / losses) if losses > 0 else np.nan
    return {
        "交易次数": count,
        "胜率": float((returns > 0).mean()),
        "平均收益率": float(returns.mean()),
        "中位数收益率": float(returns.median()),
        "累计收益率": equity_return,
        "最大回撤": max_drawdown(returns),
        "平均价差收益": float(group["pnl_price"].mean()),
        "累计价差收益": float(group["pnl_price"].sum()),
        "盈亏比": profit_loss_ratio,
    }


def build_threshold_tables(
    df: pd.DataFrame,
    step: float,
    min_gap: float,
    max_gap: float,
    return_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start_tick = int(round(min_gap / step))
    end_tick = int(round(max_gap / step))
    overall_rows = []
    yearly_rows = []

    for tick in range(start_tick, end_tick + 1):
        threshold = round(tick * step, 6)
        selected = df[df["gap"] >= threshold]
        overall_rows.append({"阈值": threshold, **summarize(selected, return_col)})

        for year, group in selected.groupby("year"):
            yearly_rows.append(
                {"年份": int(year), "阈值": threshold, **summarize(group, return_col)}
            )

    return pd.DataFrame(overall_rows), pd.DataFrame(yearly_rows)


def build_bucket_table(df: pd.DataFrame, step: float, return_col: str) -> pd.DataFrame:
    result = df.copy()
    result["高低开分档"] = (np.rint(result["gap"] / step) * step).round(6)

    rows = []
    for (year, bucket), group in result.groupby(["year", "高低开分档"]):
        rows.append(
            {
                "年份": int(year),
                "高低开分档": float(bucket),
                "平均高低开": float(group["gap"].mean()),
                "平均高低开比例": float(group["gap_pct"].mean()),
                **summarize(group, return_col),
            }
        )
    return pd.DataFrame(rows).sort_values(["年份", "高低开分档"])


def write_outputs(
    df: pd.DataFrame,
    overall: pd.DataFrame,
    yearly: pd.DataFrame,
    buckets: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    overall.to_csv(out_dir / "threshold_overall.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(out_dir / "threshold_by_year.csv", index=False, encoding="utf-8-sig")
    buckets.to_csv(out_dir / "gap_bucket_by_year.csv", index=False, encoding="utf-8-sig")

    trade_cols = [
        "date",
        "year",
        "open",
        "close",
        "prev_close",
        "gap",
        "gap_pct",
        "gross_return",
        "net_return",
        "pnl_price",
        "volume",
        "total_turnover",
    ]
    df[trade_cols].to_csv(out_dir / "daily_trades_source.csv", index=False, encoding="utf-8-sig")


def print_key_summary(overall: pd.DataFrame, yearly: pd.DataFrame, threshold: float) -> None:
    threshold = round(threshold, 6)
    row = overall[overall["阈值"] == threshold]
    if row.empty:
        return

    print(f"\n关键场景：高开 >= {threshold:.3f}，开盘买入，收盘卖出")
    print(row.to_string(index=False))

    yearly_part = yearly[yearly["阈值"] == threshold]
    if not yearly_part.empty:
        print("\n按年份：")
        print(yearly_part.to_string(index=False))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    order_book_id, raw = load_daily_bars(args.bundle, args.security)
    trades = prepare_trades(raw, args.start_date, args.end_date, args.commission_rate)
    if trades.empty:
        raise RuntimeError("No valid daily bars after filtering.")

    return_col = "net_return"
    overall, yearly = build_threshold_tables(
        trades, args.step, args.min_gap, args.max_gap, return_col
    )
    buckets = build_bucket_table(trades, args.step, return_col)
    write_outputs(trades, overall, yearly, buckets, out_dir)

    print(f"标的: {order_book_id}")
    print(f"数据区间: {trades['date'].min().date()} ~ {trades['date'].max().date()}")
    print(f"交易日数量: {len(trades)}")
    print(f"手续费: 单边 {args.commission_rate:.6f}")
    print(f"输出目录: {out_dir.resolve()}")
    print_key_summary(overall, yearly, 0.003)


if __name__ == "__main__":
    main()
