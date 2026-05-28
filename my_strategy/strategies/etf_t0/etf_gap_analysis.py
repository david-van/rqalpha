#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""跨品种 ETF 高低开日内策略分析（基于 MySQL 数据源）。

前置条件：已运行 etf_data_to_mysql.py 将 HDF5 数据导入 MySQL。

用法:
    # 默认分析：对所有活跃 ETF，按 0.001 步长扫描 [-0.02, 0.04] 区间
    python etf_gap_analysis.py

    # 自定义参数
    python etf_gap_analysis.py --start-date 2020-01-01 --end-date 2026-05-01 \
        --min-gap 0.0 --max-gap 0.02 --step 0.001

    # 指定标的
    python etf_gap_analysis.py --etfs "513030.XSHG,513100.XSHG,159920.XSHE"

    # 输出 CSV 到指定目录
    python etf_gap_analysis.py --out-dir ./my_analysis

输出文件:
    - threshold_overall.csv    各阈值下各 ETF 的全区间汇总
    - gap_bucket_by_etf.csv    按 gap 分档 + ETF + 年份的明细
    - top_thresholds.csv       每个 ETF 最优阈值的汇总排名
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pymysql
except ImportError:
    print("请先安装 pymysql: pip install pymysql")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from my_strategy.common_file import project_root

DEFAULT_OUT_DIR = (
    Path(project_root) / "my_strategy" / "strategies" / "batch_results" / "etf_gap_multi"
)

DEFAULT_DB = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "",
    "database": "rqalpha_etf",
    "charset": "utf8mb4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="跨品种 ETF 高低开日内策略 MySQL 分析"
    )
    parser.add_argument("--host", default=DEFAULT_DB["host"])
    parser.add_argument("--port", type=int, default=DEFAULT_DB["port"])
    parser.add_argument("--user", default=DEFAULT_DB["user"])
    parser.add_argument("--password", default=DEFAULT_DB["password"])
    parser.add_argument("--database", default=DEFAULT_DB["database"])
    parser.add_argument("--start-date", default="2017-01-01")
    parser.add_argument("--end-date", default="2026-05-01")
    parser.add_argument("--step", type=float, default=0.001, help="阈值扫描步长")
    parser.add_argument("--min-gap", type=float, default=-0.02)
    parser.add_argument("--max-gap", type=float, default=0.04)
    parser.add_argument("--etfs", default=None, help="逗号分隔的 ETF 列表，不指定则用 etf_universe 表")
    parser.add_argument(
        "--category",
        default="cross_border",
        help="ETF 类别: cross_border | gold | bond | money_market | commodity | all",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--commission",
        type=float,
        default=0.00005,
        help="单边手续费率（默认 0.00005 即万0.5，ETF 免印花税）",
    )
    return parser.parse_args()


def get_conn(args) -> pymysql.Connection:
    return pymysql.connect(
        host=args.host, port=args.port, user=args.user,
        password=args.password, database=args.database, charset="utf8mb4",
    )


def load_etf_list(conn, args) -> pd.DataFrame:
    """加载 ETF 标的列表."""
    if args.etfs:
        codes = [s.strip() for s in args.etfs.split(",") if s.strip()]
        return pd.DataFrame({"order_book_id": codes})

    try:
        if args.category == "all":
            df = pd.read_sql(
                "SELECT order_book_id, name FROM etf_universe WHERE active = 1 ORDER BY order_book_id",
                conn,
            )
        else:
            df = pd.read_sql(
                "SELECT order_book_id, name FROM etf_universe "
                "WHERE active = 1 AND t0_type = %s ORDER BY order_book_id",
                conn,
                params=(args.category,),
            )
        if not df.empty:
            return df
    except Exception:
        pass

    return pd.read_sql(
        "SELECT DISTINCT order_book_id FROM etf_daily_bars ORDER BY order_book_id",
        conn,
    )


def load_daily_bars(conn, args, etf_list: list[str]) -> pd.DataFrame:
    """从 MySQL 加载所有相关 ETF 的日线数据并计算派生字段."""
    placeholders = ", ".join(["%s"] * len(etf_list))
    sql = f"""
        SELECT
            order_book_id,
            trade_date,
            YEAR(trade_date)  AS trade_year,
            open,
            close,
            prev_close,
            open - prev_close                         AS gap,
            (open - prev_close) / NULLIF(prev_close, 0) AS gap_pct,
            close / NULLIF(open, 0) - 1               AS gross_return,
            (close * (1 - %s) / NULLIF(open * (1 + %s), 0) - 1) AS net_return,
            close - open                               AS pnl_price,
            volume,
            total_turnover
        FROM etf_daily_bars
        WHERE order_book_id IN ({placeholders})
          AND trade_date >= %s
          AND trade_date <= %s
          AND open > 0 AND close > 0 AND prev_close > 0
        ORDER BY order_book_id, trade_date
    """
    params = [args.commission, args.commission] + etf_list + [args.start_date, args.end_date]
    df = pd.read_sql(sql, conn, params=params)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["order_book_id"] = df["order_book_id"].astype(str)
    return df


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    equity = np.r_[1.0, (1.0 + returns.to_numpy()).cumprod()]
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def summarize(group: pd.DataFrame) -> dict:
    """对一组交易的收益率序列做统计汇总."""
    ret = group["net_return"]
    count = len(group)
    if count == 0:
        return {
            "trade_count": 0, "win_rate": 0.0, "avg_return": 0.0,
            "total_return": 0.0, "max_drawdown": 0.0,
            "avg_pnl_price": 0.0, "total_pnl_price": 0.0,
        }
    gains = ret[ret > 0].sum()
    losses = -ret[ret < 0].sum()
    return {
        "trade_count": count,
        "win_rate": float((ret > 0).mean()),
        "avg_return": float(ret.mean()),
        "total_return": float((1.0 + ret).prod() - 1.0),
        "max_drawdown": max_drawdown(ret),
        "avg_pnl_price": float(group["pnl_price"].mean()),
        "total_pnl_price": float(group["pnl_price"].sum()),
        "profit_loss_ratio": float(gains / losses) if losses > 0 else np.nan,
    }


def build_threshold_table(df: pd.DataFrame, args) -> pd.DataFrame:
    """遍历所有阈值 × ETF，生成汇总表."""
    thresholds = np.arange(args.min_gap, args.max_gap + args.step / 2, args.step).round(6)
    rows = []
    for th in thresholds:
        selected = df[df["gap"] >= th]
        for etf, group in selected.groupby("order_book_id"):
            rows.append({"threshold": float(th), "order_book_id": etf, **summarize(group)})
    return pd.DataFrame(rows)


def build_bucket_table(df: pd.DataFrame, args) -> pd.DataFrame:
    """按 gap 分档 + ETF + 年份 做汇总."""
    df = df.copy()
    df["gap_bucket"] = (np.rint(df["gap"] / args.step) * args.step).round(6)
    rows = []
    for (etf, year, bucket), group in df.groupby(["order_book_id", "trade_year", "gap_bucket"]):
        rows.append({
            "order_book_id": etf,
            "year": int(year),
            "gap_bucket": float(bucket),
            **summarize(group),
        })
    return pd.DataFrame(rows)


def build_top_summary(threshold_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """找出每个 ETF 累计收益最高的阈值，附加数据年数和年化收益率."""
    idx = threshold_df.groupby("order_book_id")["total_return"].idxmax()
    top = threshold_df.loc[idx].copy()

    # 计算每只 ETF 的数据年份跨度
    span = df.groupby("order_book_id")["trade_date"].agg(["min", "max"])
    span["years"] = (span["max"] - span["min"]).dt.days / 365.25
    top = top.merge(span[["years"]], on="order_book_id", how="left")
    top["annualized_return"] = (
        (1 + top["total_return"]) ** (1 / top["years"]) - 1
    )

    top = top.sort_values("total_return", ascending=False).reset_index(drop=True)
    return top[
        [
            "order_book_id", "threshold", "years", "trade_count", "win_rate",
            "avg_return", "total_return", "annualized_return", "max_drawdown",
            "profit_loss_ratio",
        ]
    ]


def write_outputs(
    threshold_df: pd.DataFrame,
    bucket_df: pd.DataFrame,
    top_df: pd.DataFrame,
    df: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold_df.to_csv(out_dir / "threshold_overall.csv", index=False, encoding="utf-8-sig")
    bucket_df.to_csv(out_dir / "gap_bucket_by_etf.csv", index=False, encoding="utf-8-sig")
    top_df.to_csv(out_dir / "top_thresholds.csv", index=False, encoding="utf-8-sig")
    cols = [
        "order_book_id", "trade_date", "trade_year",
        "open", "close", "prev_close", "gap", "gap_pct",
        "gross_return", "net_return", "pnl_price", "volume", "total_turnover",
    ]
    df[cols].to_csv(out_dir / "daily_trades_source.csv", index=False, encoding="utf-8-sig")
    print(f"\n输出文件 -> {out_dir.resolve()}/")
    for f in ["threshold_overall.csv", "gap_bucket_by_etf.csv",
              "top_thresholds.csv", "daily_trades_source.csv"]:
        print(f"  {f}")


def print_summary(top_df: pd.DataFrame, threshold_df: pd.DataFrame) -> None:
    """打印关键结果摘要."""
    print(f"\n{'='*80}")
    print("各 ETF 最优阈值（按累计收益率排序）")
    print(f"{'='*80}")
    cols = ["order_book_id", "threshold", "years", "trade_count", "win_rate",
            "total_return", "annualized_return", "max_drawdown"]
    display = top_df[cols].copy()
    display["win_rate"] = display["win_rate"].map(lambda x: f"{x:.1%}")
    display["total_return"] = display["total_return"].map(lambda x: f"{x:.2%}")
    display["annualized_return"] = display["annualized_return"].map(lambda x: f"{x:.2%}")
    display["max_drawdown"] = display["max_drawdown"].map(lambda x: f"{x:.2%}")
    display["threshold"] = display["threshold"].map(lambda x: f"{x:.4f}")
    display["years"] = display["years"].map(lambda x: f"{x:.2f}年" if pd.notna(x) else "-")
    print(display.to_string(index=False))

    print(f"\n{'='*80}")
    print("各 ETF 在阈值 0.003 下的表现（对比基准 513030 德国ETF）")
    print(f"{'='*80}")
    at_003 = threshold_df[threshold_df["threshold"] == 0.003].copy()
    if at_003.empty:
        # 取最近的阈值
        th_near = threshold_df.iloc[
            (threshold_df["threshold"] - 0.003).abs().argsort()
        ]["threshold"].iloc[0]
        at_003 = threshold_df[threshold_df["threshold"] == th_near].copy()
        print(f"  (实际使用阈值: {th_near:.4f})")

    at_003 = at_003.sort_values("total_return", ascending=False)
    cols2 = ["order_book_id", "threshold", "trade_count", "win_rate", "total_return", "max_drawdown"]
    display2 = at_003[cols2].copy()
    display2["win_rate"] = display2["win_rate"].map(lambda x: f"{x:.1%}")
    display2["total_return"] = display2["total_return"].map(lambda x: f"{x:.2%}")
    display2["max_drawdown"] = display2["max_drawdown"].map(lambda x: f"{x:.2%}")
    display2["threshold"] = display2["threshold"].map(lambda x: f"{x:.4f}")
    print(display2.to_string(index=False))


def main() -> None:
    args = parse_args()
    conn = get_conn(args)

    etf_df = load_etf_list(conn, args)
    etf_list = etf_df["order_book_id"].tolist()
    print(f"分析标的: {len(etf_list)} 只")
    for _, row in etf_df.iterrows():
        name_info = f"  {row.get('name', '')}" if "name" in row and pd.notna(row["name"]) else ""
        print(f"  {row['order_book_id']}{name_info}")

    print(f"\n从 MySQL 加载日线数据 {args.start_date} ~ {args.end_date} ...")
    df = load_daily_bars(conn, args, etf_list)
    conn.close()

    if df.empty:
        print("[ERROR] 无数据。请先运行 etf_data_to_mysql.py 导入数据")
        sys.exit(1)

    print(f"共 {len(df)} 行日线, {df['order_book_id'].nunique()} 只 ETF")
    print(f"日期范围: {df['trade_date'].min().date()} ~ {df['trade_date'].max().date()}")

    print("\n构建阈值扫描表 ...")
    threshold_df = build_threshold_table(df, args)

    print("构建分档明细表 ...")
    bucket_df = build_bucket_table(df, args)

    print("汇总最优阈值 ...")
    top_df = build_top_summary(threshold_df, df)

    out_dir = Path(args.out_dir)
    write_outputs(threshold_df, bucket_df, top_df, df, out_dir)
    print_summary(top_df, threshold_df)


if __name__ == "__main__":
    main()
