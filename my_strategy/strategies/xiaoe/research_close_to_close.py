#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Close-to-close research backtester for the xiaoe pool.

This script intentionally does not use RQAlpha matching. It reads adjusted
close prices from the local bundle and verifies whether old lightweight
research records such as dynamic_equal and low_vol_hybrid50 can be reproduced.
"""

from __future__ import annotations

import argparse
import bisect
import math
import os
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from my_strategy.common_file import project_root


BUNDLE = r"D:\datas\bundle"
START = "2020-01-02"
END = "2026-01-01"
BENCHMARK = "000300.XSHG"
POOL_DIR = Path(project_root) / "my_strategy" / "strategies" / "xiaoe_articles"
RESULT_DIR = Path(project_root) / "my_strategy" / "strategies" / "batch_results" / "xiaoe_pool" / "close_to_close"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


RULES = {
    "dynamic_equal": {"kind": "equal"},
    "low_vol_hybrid50": {"kind": "vol", "base_ratio": 0.50, "prefer": "low"},
    "high_vol_hybrid50": {"kind": "vol", "base_ratio": 0.50, "prefer": "high"},
    "low_vol_hybrid70": {"kind": "vol", "base_ratio": 0.70, "prefer": "low"},
    "high_vol_hybrid70": {"kind": "vol", "base_ratio": 0.70, "prefer": "high"},
}


def load_pool_snapshots(pool_dir: Path) -> pd.DataFrame:
    rows = []
    for file in sorted(pool_dir.glob("*holdings*.csv")):
        df = pd.read_csv(file, encoding="utf-8-sig")
        for _, row in df.iterrows():
            codes_str = str(row.get("持有股票代码", ""))
            if not codes_str or codes_str == "nan":
                continue
            codes = tuple(c.strip() for c in codes_str.split("|") if c.strip())
            if codes:
                rows.append({
                    "date": pd.Timestamp(str(row["日期"]).strip()),
                    "codes": codes,
                })
    snapshots = pd.DataFrame(rows)
    if snapshots.empty:
        raise RuntimeError(f"no pool snapshots found in {pool_dir}")
    return snapshots.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def pool_asof(snapshots: pd.DataFrame, snapshot_dates: list[pd.Timestamp], dt: pd.Timestamp) -> tuple[str, ...]:
    idx = bisect.bisect_right(snapshot_dates, dt) - 1
    if idx < 0:
        return tuple()
    return snapshots.iloc[idx]["codes"]


def load_adj_close(code: str, source: str = "stocks.h5") -> pd.Series | None:
    path = os.path.join(BUNDLE, source)
    with h5py.File(path, "r") as file:
        if code not in file:
            return None
        arr = file[code][:]

    df = pd.DataFrame(arr)
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["datetime"] // 1000000, format="%Y%m%d")
    df = df[["date", "close"]].copy().sort_values("date")

    if source == "stocks.h5":
        fac_path = os.path.join(BUNDLE, "ex_cum_factor.h5")
        with h5py.File(fac_path, "r") as file:
            if code in file:
                fac = pd.DataFrame(file[code][:])
                ymd = (fac["start_date"] // 1000000).astype("int64").replace(0, 19000101)
                fac["date"] = pd.to_datetime(ymd, format="%Y%m%d", errors="coerce")
                fac = fac.dropna(subset=["date"])[["date", "ex_cum_factor"]].sort_values("date")
                df = pd.merge_asof(df, fac, on="date")
                df["ex_cum_factor"] = df["ex_cum_factor"].fillna(1.0)
                df["close"] = df["close"] * df["ex_cum_factor"]

    return df.set_index("date")["close"].sort_index()


def load_trading_dates() -> pd.DatetimeIndex:
    benchmark = load_adj_close(BENCHMARK, source="indexes.h5")
    if benchmark is None or benchmark.empty:
        raise RuntimeError(f"cannot load benchmark calendar: {BENCHMARK}")
    return benchmark[(benchmark.index >= pd.Timestamp(START)) & (benchmark.index <= pd.Timestamp(END))].index


def align_prices(codes: set[str], trading_dates: pd.DatetimeIndex) -> dict[str, pd.Series]:
    prices = {}
    for code in sorted(codes):
        raw = load_adj_close(code)
        if raw is None or raw.empty:
            continue
        prices[code] = raw.reindex(trading_dates).ffill()
    return prices


def price_at(series: pd.Series, pos: int) -> float | None:
    if pos < 0 or pos >= len(series):
        return None
    value = series.iloc[pos]
    if pd.isna(value) or value <= 0:
        return None
    return float(value)


def realized_vol(series: pd.Series, pos: int, days: int) -> float | None:
    if pos < days:
        return None
    window = series.iloc[pos - days:pos + 1].dropna()
    if len(window) < days + 1 or (window <= 0).any():
        return None
    returns = window.pct_change().dropna()
    if len(returns) < days:
        return None
    value = float(returns.std(ddof=1))
    return value if math.isfinite(value) else None


def equal_weights(codes: list[str]) -> dict[str, float]:
    if not codes:
        return {}
    weight = 1.0 / len(codes)
    return {code: weight for code in codes}


def vol_hybrid_weights(
    codes: list[str],
    prices: dict[str, pd.Series],
    pos: int,
    base_ratio: float,
    prefer: str,
    vol_days: int,
    enhance_top_n: int,
) -> dict[str, float]:
    if not codes:
        return {}

    weights = {code: base_ratio / len(codes) for code in codes}
    scored = []
    for code in codes:
        vol = realized_vol(prices[code], pos, vol_days)
        if vol is not None:
            scored.append((code, vol))

    if not scored:
        return equal_weights(codes)

    reverse = prefer == "high"
    scored.sort(key=lambda item: (item[1], item[0]), reverse=reverse)
    picks = [code for code, _ in scored[:enhance_top_n]]
    enhance_ratio = max(0.0, 1.0 - base_ratio)
    if picks and enhance_ratio > 0:
        for code in picks:
            weights[code] = weights.get(code, 0.0) + enhance_ratio / len(picks)
    return weights


def target_weights(
    rule: str,
    codes: list[str],
    prices: dict[str, pd.Series],
    pos: int,
    vol_days: int,
    enhance_top_n: int,
) -> dict[str, float]:
    spec = RULES[rule]
    if spec["kind"] == "equal":
        return equal_weights(codes)
    if spec["kind"] == "vol":
        return vol_hybrid_weights(
            codes,
            prices,
            pos,
            float(spec["base_ratio"]),
            str(spec["prefer"]),
            vol_days,
            enhance_top_n,
        )
    raise ValueError(rule)


def run_rule(
    rule: str,
    snapshots: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    prices: dict[str, pd.Series],
    rebalance_days: int,
    vol_days: int,
    enhance_top_n: int,
) -> tuple[pd.Series, pd.DataFrame]:
    snapshot_dates = list(snapshots["date"])
    equity = 1.0
    cash = 1.0
    holdings: dict[str, float] = {}
    last_pool: tuple[str, ...] = tuple()
    logs = []
    curve = [(trading_dates[0], equity)]

    for pos in range(len(trading_dates) - 1):
        dt = trading_dates[pos]
        next_dt = trading_dates[pos + 1]
        raw_pool = pool_asof(snapshots, snapshot_dates, dt)
        codes = [
            code for code in raw_pool
            if code in prices and price_at(prices[code], pos) is not None
        ]
        pool_tuple = tuple(codes)
        should_rebalance = (
            pos == 0
            or rebalance_days <= 1
            or pos % rebalance_days == 0
            or pool_tuple != last_pool
        )

        if should_rebalance:
            weights = target_weights(rule, codes, prices, pos, vol_days, enhance_top_n)
            weight_sum = sum(weights.values())
            holdings = {code: equity * weight for code, weight in weights.items()}
            cash = max(0.0, equity * (1.0 - weight_sum))
            last_pool = pool_tuple
            logs.append({
                "date": dt,
                "pool_size": len(codes),
                "cash_weight": cash / equity if equity > 0 else 0,
                "weights": "|".join(f"{code}:{weight:.4f}" for code, weight in sorted(weights.items())),
            })

        next_holdings = {}
        for code, value in holdings.items():
            p0 = price_at(prices[code], pos)
            p1 = price_at(prices[code], pos + 1)
            if p0 is None or p1 is None:
                next_holdings[code] = value
            else:
                next_holdings[code] = value * p1 / p0
        holdings = next_holdings
        equity = cash + sum(holdings.values())
        curve.append((next_dt, equity))

    return pd.Series(dict(curve)).sort_index(), pd.DataFrame(logs)


def stats(equity: pd.Series) -> dict[str, float]:
    total = float(equity.iloc[-1] - 1)
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    annualized = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else np.nan
    drawdown = equity / equity.cummax() - 1
    returns = equity.pct_change().dropna()
    sharpe = (
        returns.mean() / returns.std(ddof=1) * np.sqrt(252)
        if len(returns) > 1 and returns.std(ddof=1) > 0
        else np.nan
    )
    return {
        "total_returns": total,
        "annualized": float(annualized),
        "max_drawdown": float(drawdown.min()),
        "sharpe": float(sharpe),
    }


def yearly_returns(equity: pd.Series) -> dict[int, float]:
    result = {}
    for year, group in equity.groupby(equity.index.year):
        start_idx = equity.index.searchsorted(pd.Timestamp(f"{year}-01-01"), side="left") - 1
        start_value = equity.iloc[start_idx] if start_idx >= 0 else group.iloc[0]
        result[int(year)] = float(group.iloc[-1] / start_value - 1)
    return result


def format_pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.2%}"


def main() -> pd.DataFrame:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebalance-days", type=int, default=1)
    parser.add_argument("--vol-days", type=int, default=60)
    parser.add_argument("--enhance-top-n", type=int, default=2)
    args = parser.parse_args()

    print("Loading pool snapshots...")
    snapshots = load_pool_snapshots(POOL_DIR)
    all_codes = {code for codes in snapshots["codes"] for code in codes}
    print(f"snapshots={len(snapshots)}, codes={len(all_codes)}")

    print("Loading adjusted close prices...")
    trading_dates = load_trading_dates()
    prices = align_prices(all_codes, trading_dates)
    missing = sorted(all_codes - set(prices))
    print(f"trading_days={len(trading_dates)}, price_series={len(prices)}, missing={len(missing)}")
    if missing:
        print("missing:", ", ".join(missing))

    curves = {}
    summary_rows = []
    yearly_rows = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for rule in RULES:
        print(f"Running {rule}...")
        equity, log = run_rule(
            rule,
            snapshots,
            trading_dates,
            prices,
            args.rebalance_days,
            args.vol_days,
            args.enhance_top_n,
        )
        curves[rule] = equity
        log.to_csv(RESULT_DIR / f"{rule}_trades_{stamp}.csv", index=False, encoding="utf-8-sig")

        row = {
            "rule": rule,
            "rebalance_days": args.rebalance_days,
            "vol_days": args.vol_days,
            "enhance_top_n": args.enhance_top_n,
        }
        row.update(stats(equity))
        summary_rows.append(row)

        year_row = {"rule": rule}
        year_row.update(yearly_returns(equity))
        yearly_rows.append(year_row)

    summary = pd.DataFrame(summary_rows)
    yearly = pd.DataFrame(yearly_rows)
    curves_df = pd.DataFrame(curves)

    summary_path = RESULT_DIR / f"summary_{stamp}.csv"
    yearly_path = RESULT_DIR / f"yearly_{stamp}.csv"
    curves_path = RESULT_DIR / f"curves_{stamp}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    yearly.to_csv(yearly_path, index=False, encoding="utf-8-sig")
    curves_df.to_csv(curves_path, encoding="utf-8-sig")

    display = summary.copy()
    for col in ("total_returns", "annualized", "max_drawdown", "sharpe"):
        if col == "sharpe":
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            display[col] = display[col].map(format_pct)
    print("\nClose-to-close summary:")
    print(display.to_string(index=False))
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved yearly : {yearly_path}")
    print(f"Saved curves : {curves_path}")
    return summary


if __name__ == "__main__":
    main()
