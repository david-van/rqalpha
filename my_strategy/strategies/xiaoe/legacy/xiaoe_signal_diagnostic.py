#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnose whether in-holding signals have predictive value for xiaoe pool stocks.

This is not a trading strategy. It samples pool constituents through time and
answers: after a state appears, what are the next 20/60/120 trading-day returns,
and do they beat the contemporaneous pool average?
"""

from __future__ import annotations

import bisect
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


BUNDLE = r"D:\datas\bundle"
START = "2020-01-01"
END = "2026-01-01"
BENCHMARK = "000300.XSHG"

SAMPLE_STEP = 20
FORWARD_DAYS = [20, 60, 120]
PAST_DAYS = [20, 60, 120]
EMA_DAYS = [20, 60, 120]
DRAWDOWN_LOOKBACK = 120


from my_strategy.common_file import project_root

POOL_DIR = Path(project_root) / "my_strategy" / "strategies" / "xiaoe_articles"
RESULT_DIR = Path(project_root) / 'my_strategy' / 'strategies' / 'batch_results' / 'xiaoe_pool' / 'diagnostics'
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def load_pool_snapshots(pool_dir: Path) -> pd.DataFrame:
    rows = []
    for file in sorted(pool_dir.glob("*holdings*.csv")):
        df = pd.read_csv(file, encoding="utf-8-sig")
        for _, row in df.iterrows():
            if len(row) < 4:
                continue
            date = pd.Timestamp(str(row.iloc[0]).strip())
            codes_str = str(row.iloc[3]).strip()
            if not codes_str or codes_str == "nan":
                continue
            codes = tuple(c.strip() for c in codes_str.split("|") if c.strip())
            if codes:
                rows.append({"date": date, "codes": codes})

    snapshots = pd.DataFrame(rows)
    if snapshots.empty:
        raise RuntimeError(f"no pool snapshots found in {pool_dir}")
    snapshots = snapshots.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return snapshots


def pool_asof(snapshots: pd.DataFrame, snapshot_dates: list[pd.Timestamp], dt: pd.Timestamp) -> tuple[str, ...]:
    idx = bisect.bisect_right(snapshot_dates, dt) - 1
    if idx < 0:
        return tuple()
    return snapshots.iloc[idx]["codes"]


def first_seen_dates(snapshots: pd.DataFrame) -> dict[str, pd.Timestamp]:
    first = {}
    for _, row in snapshots.iterrows():
        for code in row["codes"]:
            first.setdefault(code, row["date"])
    return first


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
    bench = load_adj_close(BENCHMARK, source="indexes.h5")
    if bench is None or bench.empty:
        raise RuntimeError(f"cannot load benchmark {BENCHMARK}")
    bench = bench[(bench.index >= pd.Timestamp(START)) & (bench.index <= pd.Timestamp(END))]
    return bench.index


def align_prices(codes: set[str], trading_dates: pd.DatetimeIndex) -> dict[str, pd.Series]:
    prices = {}
    for code in sorted(codes):
        raw = load_adj_close(code)
        if raw is None or raw.empty:
            continue
        aligned = raw.reindex(trading_dates).ffill()
        prices[code] = aligned
    return prices


def build_features(prices: dict[str, pd.Series]) -> dict[str, dict[str, pd.Series]]:
    features = {}
    for code, close in prices.items():
        per_code = {"close": close}
        for days in EMA_DAYS:
            ema = close.ewm(span=days, adjust=False, min_periods=days).mean()
            per_code[f"ema{days}"] = ema
            per_code[f"below_ema{days}"] = close < ema
            per_code[f"cross_below_ema{days}"] = (close < ema) & (close.shift(1) >= ema.shift(1))

        high = close.rolling(DRAWDOWN_LOOKBACK, min_periods=DRAWDOWN_LOOKBACK).max()
        per_code["dd120"] = close / high - 1
        for days in PAST_DAYS:
            per_code[f"ret_past{days}"] = close / close.shift(days) - 1
        for days in FORWARD_DAYS:
            per_code[f"ret_fwd{days}"] = close.shift(-days) / close - 1
        features[code] = per_code
    return features


def value_at(series: pd.Series, pos: int):
    if pos < 0 or pos >= len(series):
        return np.nan
    return series.iloc[pos]


def add_condition(records: list[dict], condition: str, base: dict, fwd: dict[int, float], excess: dict[int, float]):
    rec = dict(base)
    rec["condition"] = condition
    for days in FORWARD_DAYS:
        rec[f"fwd_{days}d"] = fwd.get(days, np.nan)
        rec[f"excess_{days}d"] = excess.get(days, np.nan)
    records.append(rec)


def collect_observations(
    snapshots: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    features: dict[str, dict[str, pd.Series]],
    first_seen: dict[str, pd.Timestamp],
) -> pd.DataFrame:
    records = []
    snapshot_dates = list(snapshots["date"])
    sample_positions = range(0, len(trading_dates) - max(FORWARD_DAYS), SAMPLE_STEP)

    for pos in sample_positions:
        dt = trading_dates[pos]
        pool = tuple(code for code in pool_asof(snapshots, snapshot_dates, dt) if code in features)
        if not pool:
            continue

        pool_fwd = {}
        for days in FORWARD_DAYS:
            vals = []
            for code in pool:
                val = value_at(features[code][f"ret_fwd{days}"], pos)
                if pd.notna(val):
                    vals.append(float(val))
            pool_fwd[days] = float(np.mean(vals)) if vals else np.nan

        past_rank = {}
        for days in (60, 120):
            vals = []
            for code in pool:
                val = value_at(features[code][f"ret_past{days}"], pos)
                if pd.notna(val):
                    vals.append((code, float(val)))
            if vals:
                ordered = sorted(vals, key=lambda x: x[1])
                past_rank[days] = {
                    "bottom": ordered[0][0],
                    "top": ordered[-1][0],
                }
            else:
                past_rank[days] = {"bottom": None, "top": None}

        for code in pool:
            close = value_at(features[code]["close"], pos)
            if pd.isna(close) or close <= 0:
                continue

            fwd = {}
            excess = {}
            for days in FORWARD_DAYS:
                val = value_at(features[code][f"ret_fwd{days}"], pos)
                fwd[days] = float(val) if pd.notna(val) else np.nan
                excess[days] = fwd[days] - pool_fwd[days] if pd.notna(fwd[days]) and pd.notna(pool_fwd[days]) else np.nan

            age_days = (dt - first_seen[code]).days
            entry_close = features[code]["close"].asof(first_seen[code])
            since_entry = float(close / entry_close - 1) if pd.notna(entry_close) and entry_close > 0 else np.nan
            dd120 = value_at(features[code]["dd120"], pos)
            ret60 = value_at(features[code]["ret_past60"], pos)
            ret120 = value_at(features[code]["ret_past120"], pos)

            base = {
                "date": dt,
                "code": code,
                "pool_size": len(pool),
                "age_days": age_days,
                "since_entry_return": since_entry,
                "dd120": float(dd120) if pd.notna(dd120) else np.nan,
                "past60": float(ret60) if pd.notna(ret60) else np.nan,
                "past120": float(ret120) if pd.notna(ret120) else np.nan,
            }

            add_condition(records, "pool_all", base, fwd, excess)

            if age_days < 60:
                add_condition(records, "age_0_60", base, fwd, excess)
            elif age_days < 180:
                add_condition(records, "age_60_180", base, fwd, excess)
            elif age_days < 360:
                add_condition(records, "age_180_360", base, fwd, excess)
            else:
                add_condition(records, "age_360_plus", base, fwd, excess)

            if pd.notna(since_entry):
                if since_entry >= 1.0:
                    add_condition(records, "since_entry_gain_ge_100pct", base, fwd, excess)
                elif since_entry >= 0.5:
                    add_condition(records, "since_entry_gain_50_100pct", base, fwd, excess)
                elif since_entry <= -0.2:
                    add_condition(records, "since_entry_loss_ge_20pct", base, fwd, excess)

            if pd.notna(dd120):
                if dd120 <= -0.4:
                    add_condition(records, "dd120_ge_40pct", base, fwd, excess)
                elif dd120 <= -0.3:
                    add_condition(records, "dd120_30_40pct", base, fwd, excess)
                elif dd120 <= -0.2:
                    add_condition(records, "dd120_20_30pct", base, fwd, excess)

            for days in EMA_DAYS:
                if bool(value_at(features[code][f"below_ema{days}"], pos)):
                    add_condition(records, f"below_ema{days}", base, fwd, excess)
                if bool(value_at(features[code][f"cross_below_ema{days}"], pos)):
                    add_condition(records, f"cross_below_ema{days}", base, fwd, excess)

            for days in (60, 120):
                if past_rank[days]["top"] == code:
                    add_condition(records, f"pool_top_past{days}", base, fwd, excess)
                if past_rank[days]["bottom"] == code:
                    add_condition(records, f"pool_bottom_past{days}", base, fwd, excess)

            if age_days >= 360 and past_rank[60]["bottom"] == code:
                add_condition(records, "old_360_plus_and_pool_bottom_past60", base, fwd, excess)
            if age_days >= 360 and bool(value_at(features[code]["below_ema120"], pos)):
                add_condition(records, "old_360_plus_and_below_ema120", base, fwd, excess)
            if pd.notna(dd120) and dd120 <= -0.3 and past_rank[60]["bottom"] == code:
                add_condition(records, "dd120_ge_30pct_and_pool_bottom_past60", base, fwd, excess)

    return pd.DataFrame(records)


def summarize(observations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, df in observations.groupby("condition"):
        row = {"condition": condition, "n": len(df)}
        for days in FORWARD_DAYS:
            fwd = df[f"fwd_{days}d"].dropna()
            exc = df[f"excess_{days}d"].dropna()
            row[f"fwd_{days}d_mean"] = fwd.mean()
            row[f"fwd_{days}d_median"] = fwd.median()
            row[f"fwd_{days}d_win_rate"] = (fwd > 0).mean()
            row[f"excess_{days}d_mean"] = exc.mean()
            row[f"excess_{days}d_median"] = exc.median()
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["excess_60d_mean", "fwd_60d_mean"], ascending=[True, True])
    return summary


def format_summary_for_console(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    pct_cols = [c for c in out.columns if c != "n" and c != "condition"]
    for col in pct_cols:
        out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    return out


def main():
    print("Loading xiaoe pool snapshots...")
    snapshots = load_pool_snapshots(POOL_DIR)
    first_seen = first_seen_dates(snapshots)
    all_codes = {code for codes in snapshots["codes"] for code in codes}
    print(f"snapshots={len(snapshots)}, codes={len(all_codes)}")
    print(f"pool range={snapshots['date'].min().date()} to {snapshots['date'].max().date()}")

    print("Loading bundle prices...")
    trading_dates = load_trading_dates()
    prices = align_prices(all_codes, trading_dates)
    missing = sorted(all_codes - set(prices))
    print(f"trading_days={len(trading_dates)}, price_series={len(prices)}, missing={len(missing)}")
    if missing:
        print("missing codes:", ", ".join(missing))

    print("Building signal observations...")
    features = build_features(prices)
    observations = collect_observations(snapshots, trading_dates, features, first_seen)
    summary = summarize(observations)

    obs_path = RESULT_DIR / "xiaoe_signal_observations.csv"
    summary_path = RESULT_DIR / "xiaoe_signal_summary.csv"
    observations.to_csv(obs_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\nWorst conditions by 60d excess return:")
    cols = [
        "condition", "n",
        "fwd_20d_mean", "excess_20d_mean",
        "fwd_60d_mean", "excess_60d_mean",
        "fwd_120d_mean", "excess_120d_mean",
    ]
    print(format_summary_for_console(summary[cols].head(20)).to_string(index=False))

    print("\nBest conditions by 60d excess return:")
    print(format_summary_for_console(summary[cols].tail(20).sort_values("excess_60d_mean", ascending=False)).to_string(index=False))

    print(f"\nSaved observations: {obs_path}")
    print(f"Saved summary     : {summary_path}")


if __name__ == "__main__":
    main()
