#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""将 RQAlpha HDF5 bundle 中的 ETF 日线数据导入 MySQL。

会自动从 instruments.pk 发现所有 T+0 ETF 并填充 etf_universe 表。

用法:
    # 首次使用：自动发现全部 T+0 ETF 并导入
    python etf_data_to_mysql.py --password YOUR_PASSWORD

    # 仅导入跨境 ETF（默认）
    python etf_data_to_mysql.py --category cross_border --password YOUR_PASSWORD

    # 导入全部 T+0 ETF
    python etf_data_to_mysql.py --category all --password YOUR_PASSWORD

    # 指定具体标的
    python etf_data_to_mysql.py --etfs "513030.XSHG,513100.XSHG" --password YOUR_PASSWORD

    # 增量更新（跳过已导入的）
    python etf_data_to_mysql.py --password YOUR_PASSWORD
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import pandas as pd

try:
    import pymysql
except ImportError:
    print("请先安装 pymysql: pip install pymysql")
    sys.exit(1)

from t0_etf_discovery import discover_t0_etfs

DEFAULT_BUNDLE = r"D:\datas\bundle"
BATCH_SIZE = 500

DEFAULT_DB = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "",
    "database": "rqalpha_etf",
    "charset": "utf8mb4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导入 ETF 日线到 MySQL")
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE, help="RQAlpha bundle 路径")
    parser.add_argument("--host", default=DEFAULT_DB["host"])
    parser.add_argument("--port", type=int, default=DEFAULT_DB["port"])
    parser.add_argument("--user", default=DEFAULT_DB["user"])
    parser.add_argument("--password", default=DEFAULT_DB["password"])
    parser.add_argument("--database", default=DEFAULT_DB["database"])
    parser.add_argument(
        "--category",
        default="cross_border",
        help="ETF 类别: cross_border | gold | bond | money_market | commodity | all",
    )
    parser.add_argument(
        "--etfs",
        default=None,
        help="逗号分隔的 ETF 代码。指定后覆盖 --category",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="使用 REPLACE INTO（覆盖已有数据）而不是 INSERT IGNORE",
    )
    parser.add_argument(
        "--skip-universe",
        action="store_true",
        help="跳过 etf_universe 表的更新",
    )
    return parser.parse_args()


def connect_mysql(cfg: dict, database: bool = True) -> pymysql.Connection:
    kwargs = {
        "host": cfg["host"], "port": cfg["port"],
        "user": cfg["user"], "password": cfg["password"],
        "charset": cfg["charset"],
    }
    if database:
        kwargs["database"] = cfg["database"]
    return pymysql.connect(**kwargs)


def ensure_database(cfg: dict):
    conn = connect_mysql(cfg, database=False)
    with conn.cursor() as cur:
        cur.execute(
            f"CREATE DATABASE IF NOT EXISTS {cfg['database']} "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
    conn.commit()
    conn.close()


def create_tables(conn):
    statements = [
        """CREATE TABLE IF NOT EXISTS etf_daily_bars (
            id              BIGINT          AUTO_INCREMENT PRIMARY KEY,
            order_book_id   VARCHAR(20)     NOT NULL COMMENT '标的代码',
            trade_date      DATE            NOT NULL COMMENT '交易日期',
            open            DECIMAL(12,4)   NOT NULL DEFAULT 0,
            high            DECIMAL(12,4)   NOT NULL DEFAULT 0,
            low             DECIMAL(12,4)   NOT NULL DEFAULT 0,
            close           DECIMAL(12,4)   NOT NULL DEFAULT 0,
            prev_close      DECIMAL(12,4)   NOT NULL DEFAULT 0,
            volume          DECIMAL(18,2)   NOT NULL DEFAULT 0,
            total_turnover  DECIMAL(18,2)   NOT NULL DEFAULT 0,
            UNIQUE INDEX uk_symbol_date (order_book_id, trade_date),
            INDEX idx_trade_date (trade_date),
            INDEX idx_order_book_id (order_book_id)
        ) ENGINE=InnoDB COMMENT 'ETF 日线行情'""",
        """CREATE TABLE IF NOT EXISTS etf_universe (
            order_book_id   VARCHAR(20)     PRIMARY KEY COMMENT '标的代码',
            name            VARCHAR(200)    DEFAULT '' COMMENT '中文简称',
            market          VARCHAR(10)     DEFAULT '' COMMENT '市场 SH/SZ',
            t0_type         VARCHAR(20)     NOT NULL DEFAULT 'cross_border' COMMENT '类别',
            underlying      VARCHAR(200)    DEFAULT '' COMMENT '底层标的',
            listed_date     DATE            DEFAULT NULL COMMENT '上市日期',
            round_lot       INT             DEFAULT 100 COMMENT '最小交易单位',
            active          TINYINT(1)      DEFAULT 1 COMMENT '是否活跃'
        ) ENGINE=InnoDB COMMENT 'T+0 ETF 标的池'""",
    ]
    with conn.cursor() as cur:
        for s in statements:
            cur.execute(s)
    conn.commit()


def sync_etf_universe(conn, bundle_path: str):
    """从 instruments.pk 发现 T+0 ETF 并同步到 etf_universe 表."""
    print("正在从 instruments.pk 发现 T+0 ETF ...")
    etfs = discover_t0_etfs(str(Path(bundle_path).parent))

    sql = """
        INSERT INTO etf_universe (order_book_id, name, market, t0_type, underlying, listed_date, round_lot, active)
        VALUES (%s, %s, %s, %s, %s, %s, %s, 1)
        ON DUPLICATE KEY UPDATE
            name = VALUES(name),
            market = VALUES(market),
            t0_type = VALUES(t0_type),
            underlying = VALUES(underlying),
            listed_date = VALUES(listed_date)
    """
    with conn.cursor() as cur:
        for e in etfs:
            listed = e["listed_date"]
            if not listed or listed == "0000-00-00":
                listed = None
            cur.execute(sql, (
                e["order_book_id"], e["name"], e["market"],
                e["t0_type"], e["underlying"], listed, e["round_lot"],
            ))
        conn.commit()

    print(f"  已同步 {len(etfs)} 只 T+0 ETF 到 etf_universe")
    return etfs


def resolve_orders(conn, args, bundle_path: str) -> list[dict]:
    """确定要导入的 ETF 列表."""
    # 先用 HDF5 中实际存在的 key 做白名单
    funds_h5 = Path(args.bundle) / "funds.h5"
    with h5py.File(funds_h5, "r") as f:
        h5_keys = set(f.keys())

    if args.etfs:
        codes = [s.strip() for s in args.etfs.split(",") if s.strip()]
        resolved = []
        for code in codes:
            upper = code.upper()
            if upper in h5_keys:
                resolved.append({"order_book_id": upper, "name": "", "t0_type": "manual"})
            else:
                matches = [k for k in h5_keys if k.split(".")[0] == upper]
                if len(matches) == 1:
                    resolved.append({"order_book_id": matches[0], "name": "", "t0_type": "manual"})
                else:
                    print(f"  [SKIP] {code} 不在 bundle 中")
        return resolved

    # 从数据库读取已配置标的
    try:
        with conn.cursor() as cur:
            if args.category == "all":
                cur.execute(
                    "SELECT order_book_id, name, t0_type FROM etf_universe WHERE active = 1"
                )
            else:
                cur.execute(
                    "SELECT order_book_id, name, t0_type FROM etf_universe "
                    "WHERE active = 1 AND t0_type = %s",
                    (args.category,),
                )
            rows = cur.fetchall()
        if rows:
            return [
                {"order_book_id": r[0], "name": r[1], "t0_type": r[2]} for r in rows
            ]
    except pymysql.err.ProgrammingError:
        pass

    # fallback: 直接发现
    etfs = discover_t0_etfs(str(Path(bundle_path).parent))
    if args.category != "all":
        etfs = [e for e in etfs if e["t0_type"] == args.category]
    return [
        {"order_book_id": e["order_book_id"], "name": e["name"], "t0_type": e["t0_type"]}
        for e in etfs
    ]


def read_etf_from_h5(bundle_path: str, order_book_id: str) -> pd.DataFrame | None:
    with h5py.File(bundle_path, "r") as store:
        if order_book_id not in store:
            return None
        raw = store[order_book_id][:]

    if len(raw) == 0:
        return None

    df = pd.DataFrame.from_records(raw)
    df["trade_date"] = pd.to_datetime(
        df["datetime"].astype(str).str.slice(0, 8), format="%Y%m%d"
    )
    df = df[(df["open"] > 0) & (df["close"] > 0) & (df["prev_close"] > 0)]
    return df[["trade_date", "open", "close", "high", "low", "prev_close", "volume", "total_turnover"]]


def batch_insert(conn, order_book_id: str, df: pd.DataFrame, use_replace: bool) -> int:
    if df.empty:
        return 0

    cmd = "REPLACE INTO" if use_replace else "INSERT IGNORE INTO"
    sql = f"""
        {cmd} etf_daily_bars
            (order_book_id, trade_date, open, close, high, low, prev_close, volume, total_turnover)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    params = [
        (
            order_book_id,
            str(r["trade_date"])[:10],
            float(r["open"]), float(r["close"]),
            float(r["high"]), float(r["low"]),
            float(r["prev_close"]),
            float(r["volume"]), float(r["total_turnover"]),
        )
        for _, r in df.iterrows()
    ]

    total = 0
    with conn.cursor() as cur:
        for i in range(0, len(params), BATCH_SIZE):
            cur.executemany(sql, params[i:i + BATCH_SIZE])
            total += cur.rowcount
        conn.commit()
    return total


def main() -> None:
    args = parse_args()
    funds_h5_path = str(Path(args.bundle) / "funds.h5")

    if not Path(funds_h5_path).exists():
        print(f"[ERROR] bundle 文件不存在: {funds_h5_path}")
        sys.exit(1)

    cfg = {
        "host": args.host, "port": args.port,
        "user": args.user, "password": args.password,
        "database": args.database, "charset": "utf8mb4",
    }

    ensure_database(cfg)
    conn = connect_mysql(cfg)
    print(f"[OK] 已连接 MySQL: {cfg['host']}:{cfg['port']}/{cfg['database']}")

    create_tables(conn)

    # 同步标的列表
    if not args.skip_universe:
        sync_etf_universe(conn, funds_h5_path)

    # 确定导入列表
    orders = resolve_orders(conn, args, funds_h5_path)
    print(f"\n待导入 ETF: {len(orders)} 只 (category={args.category})")
    for o in orders:
        name_str = f"  {o['name']}" if o["name"] else ""
        print(f"  {o['order_book_id']}{name_str}")

    # 逐只导入
    total_rows = 0
    success = 0
    for idx, ob_info in enumerate(orders, 1):
        ob = ob_info["order_book_id"]
        df = read_etf_from_h5(funds_h5_path, ob)
        if df is None or df.empty:
            print(f"  [{idx:3d}/{len(orders)}] {ob} — 无数据，跳过")
            continue

        n = batch_insert(conn, ob, df, args.replace)
        status = "REPLACE" if args.replace else "INSERT"
        print(f"  [{idx:3d}/{len(orders)}] {ob} — "
              f"{df['trade_date'].min().date()} ~ {df['trade_date'].max().date()}, "
              f"{len(df):,} 行, {status} {n} 行")
        total_rows += n
        success += 1

    conn.close()
    print(f"\n[DONE] {success}/{len(orders)} 只成功, 共 {total_rows:,} 行")


if __name__ == "__main__":
    main()
