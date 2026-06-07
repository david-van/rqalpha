#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""下载 QDII ETF 净值 + 日线数据。

从 sxsc_tushare 获取 QDII ETF 的 fund_nav(净值) 和 fund_daily(日线)，
写入 MySQL 或本地 CSV 文件，用于后续折溢价套利分析。

用法:
    # 默认：下载 QDII ETF 数据到 CSV 缓存
    python download_qdii_nav.py

    # 写入 MySQL（需要 MySQL 运行中）
    python download_qdii_nav.py --to-mysql

    # 强制全量重新下载
    python download_qdii_nav.py --force

    # 指定 ETF
    python download_qdii_nav.py --etfs "513100.SH,513500.SH"

依赖:
    pip install pymysql
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import sxsc_tushare as ts

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
START_DATE = '20170101'
END_DATE = '20260606'

# QDII ETF 筛选关键词（同 S3 策略）
QDII_KEYWORDS = [
    '标普', '美国', '纳指', '纳斯达克', '法国', '德国', '日经',
    '亚太', '东南亚', '印度', '沙特', '道琼斯', '中韩',
]

# 排除列表（非 QDII 但可能命中关键词）
BLACK_LIST = ['562060.XSHG']

# CSV 缓存目录
CACHE_DIR = Path(__file__).resolve().parent / 'cache'

# MySQL 配置（同 etf_t0）
MYSQL_CONFIG = {
    "host": "localhost", "port": 3306, "user": "root", "password": "",
    "database": "rqalpha_etf", "charset": "utf8mb4",
}

# 批量下载间隔（秒）
API_DELAY = 0.3
BATCH_SIZE = 10  # 每批下载多少只后报告一次


# ---------------------------------------------------------------------------
# 代码格式映射
# ---------------------------------------------------------------------------
def ts_to_rq(code: str) -> str:
    """tushare .SH/.SZ → RQAlpha .XSHG/.XSHE"""
    return code.replace('.SH', '.XSHG').replace('.SZ', '.XSHE')


def rq_to_ts(code: str) -> str:
    """RQAlpha .XSHG/.XSHE → tushare .SH/.SZ"""
    return code.replace('.XSHG', '.SH').replace('.XSHE', '.SZ')


# ---------------------------------------------------------------------------
# MySQL 操作
# ---------------------------------------------------------------------------
def _connect_mysql(pw: str = None) -> 'pymysql.Connection':
    import pymysql
    cfg = {**MYSQL_CONFIG}
    if pw:
        cfg['password'] = pw
    return pymysql.connect(**cfg)


def ensure_mysql_tables(conn):
    sql = """
        CREATE TABLE IF NOT EXISTS fund_nav (
            id            BIGINT          AUTO_INCREMENT PRIMARY KEY,
            ts_code       VARCHAR(20)     NOT NULL COMMENT 'tushare 代码，如 513100.SH',
            nav_date      DATE            NOT NULL COMMENT '净值日期',
            unit_nav      DECIMAL(14,8)   NOT NULL DEFAULT 0 COMMENT '单位净值',
            accum_nav     DECIMAL(14,8)   DEFAULT NULL COMMENT '累计净值',
            adj_nav       DECIMAL(14,8)   DEFAULT NULL COMMENT '复权净值',
            ann_date      DATE            DEFAULT NULL COMMENT '公告日期',
            UNIQUE INDEX uk_code_date (ts_code, nav_date),
            INDEX idx_nav_date (nav_date),
            INDEX idx_ts_code (ts_code)
        ) ENGINE=InnoDB COMMENT 'QDII ETF 净值数据（来自 sxsc_tushare fund_nav）';
    """
    with conn.cursor() as cur:
        for s in sql.split(';'):
            s = s.strip()
            if s:
                cur.execute(s)
    conn.commit()


def batch_insert_nav(conn, records: list[dict]) -> int:
    sql = """
        INSERT IGNORE INTO fund_nav (ts_code, nav_date, unit_nav, accum_nav, adj_nav, ann_date)
        VALUES (%(ts_code)s, %(nav_date)s, %(unit_nav)s, %(accum_nav)s, %(adj_nav)s, %(ann_date)s)
    """
    count = 0
    with conn.cursor() as cur:
        for i in range(0, len(records), 500):
            cur.executemany(sql, records[i:i + 500])
            count += cur.rowcount
        conn.commit()
    return count


# ---------------------------------------------------------------------------
# CSV 缓存操作
# ---------------------------------------------------------------------------
def _cache_path(ts_code: str, data_type: str) -> Path:
    """缓存文件路径: cache/513100.SH_nav.csv / cache/513100.SH_daily.csv"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{ts_code}_{data_type}.csv"


def load_from_cache(ts_code: str, data_type: str) -> pd.DataFrame | None:
    path = _cache_path(ts_code, data_type)
    if path.exists():
        df = pd.read_csv(path)
        if not df.empty:
            return df
    return None


def save_to_cache(ts_code: str, data_type: str, df: pd.DataFrame):
    path = _cache_path(ts_code, data_type)
    df.to_csv(path, index=False, encoding='utf-8-sig')


# ---------------------------------------------------------------------------
# QDII ETF 发现
# ---------------------------------------------------------------------------
def discover_qdii_etfs(api, etf_codes: list[str] | None = None) -> list[dict]:
    """发现 QDII ETF 列表。指定 codes 则直接使用，否则从 fund_basic 筛选。"""
    if etf_codes:
        return [{'ts_code': c, 'name': c} for c in etf_codes]

    print('正在获取全量 ETF 列表 ...')
    df = api.fund_basic(market='E', limit=2000,
                        fields='ts_code,name,fund_type,invest_type,benchmark,list_date,status')
    if df is None or df.empty:
        print('[ERROR] fund_basic 无数据')
        return []

    total = len(df)
    print(f'全量 ETF: {total} 只')

    # 筛选 QDII 关键词
    pattern = '|'.join(QDII_KEYWORDS)
    mask = df['name'].str.contains(pattern, na=False)
    qdii = df[mask]
    print(f'QDII 关键词筛选: {len(qdii)} 只')

    # 去黑名单
    qdii_ts_blacklist = [rq_to_ts(b) for b in BLACK_LIST]
    qdii = qdii[~qdii['ts_code'].isin(qdii_ts_blacklist)]

    # 只保留活跃的
    if 'status' in qdii.columns:
        qdii = qdii[qdii['status'].isin(['L', '正常', None, ''])]

    result = []
    for _, row in qdii.iterrows():
        result.append({
            'ts_code': row['ts_code'],
            'name': row['name'],
            'list_date': str(row.get('list_date', ''))[:10] if pd.notna(row.get('list_date')) else '',
        })

    print(f'最终 QDII ETF: {len(result)} 只')
    for r in result:
        print(f'  {r["ts_code"]:16s} {r["name"]}')

    return result


# ---------------------------------------------------------------------------
# 数据下载
# ---------------------------------------------------------------------------
def download_nav(api, ts_code: str, start_date: str, end_date: str,
                 force: bool = False) -> pd.DataFrame | None:
    """下载单只 ETF 净值数据。支持缓存增量。"""
    if not force:
        cached = load_from_cache(ts_code, 'nav')
        if cached is not None and not cached.empty:
            last_date = cached['nav_date'].max()
            if str(last_date) >= end_date:
                return cached
            # 增量：从最后日期开始
            start_date = str(last_date).replace('-', '')[:8]

    try:
        df = api.fund_nav(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return None

        df = df.drop_duplicates(subset=['nav_date'])
        # 保留需要的列
        cols = ['ts_code', 'nav_date', 'unit_nav', 'accum_nav', 'adj_nav', 'ann_date']
        df = df[[c for c in cols if c in df.columns]].copy()

        # 合并缓存
        if not force:
            cached = load_from_cache(ts_code, 'nav')
            if cached is not None and not cached.empty:
                df = pd.concat([cached, df], ignore_index=True)
                df = df.drop_duplicates(subset=['nav_date'])

        df = df.sort_values('nav_date').reset_index(drop=True)
        save_to_cache(ts_code, 'nav', df)
        return df
    except Exception as e:
        print(f'  [WARN] {ts_code} NAV 下载失败: {e}')
        return None


def download_daily(api, ts_code: str, start_date: str, end_date: str,
                   force: bool = False) -> pd.DataFrame | None:
    """下载单只 ETF 日线数据。支持缓存增量。"""
    if not force:
        cached = load_from_cache(ts_code, 'daily')
        if cached is not None and not cached.empty:
            last_date = cached['trade_date'].max()
            if str(last_date) >= end_date:
                return cached
            start_date = str(last_date).replace('-', '')[:8]

    try:
        df = api.fund_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return None

        cols = ['ts_code', 'trade_date', 'open', 'close', 'high', 'low',
                'pre_close', 'change', 'pct_chg', 'vol', 'amount']
        df = df[[c for c in cols if c in df.columns]].copy()

        if not force:
            cached = load_from_cache(ts_code, 'daily')
            if cached is not None and not cached.empty:
                df = pd.concat([cached, df], ignore_index=True)
                df = df.drop_duplicates(subset=['trade_date'])

        df = df.sort_values('trade_date').reset_index(drop=True)
        save_to_cache(ts_code, 'daily', df)
        return df
    except Exception as e:
        print(f'  [WARN] {ts_code} 日线下载失败: {e}')
        return None


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(etf_codes: list[str] | None = None, force: bool = False,
        to_mysql: bool = False, mysql_pw: str = None,
        start_date: str = START_DATE, end_date: str = END_DATE,
        env: str = 'prd'):
    api = ts.get_api(env=env, timeout=60)

    # 1. 发现 QDII ETF
    etfs = discover_qdii_etfs(api, etf_codes)
    if not etfs:
        print('未发现 QDII ETF，退出')
        return

    # 2. 下载数据
    total_nav, total_daily = 0, 0
    nav_records = []  # 用于 MySQL 批量写入

    for i, etf in enumerate(etfs):
        ts_code = etf['ts_code']
        name = etf['name']
        idx = i + 1
        print(f'\n[{idx}/{len(etfs)}] {ts_code} {name}')

        # 净值
        nav_df = download_nav(api, ts_code, start_date, end_date, force)
        if nav_df is not None and not nav_df.empty:
            print(f'  NAV: {len(nav_df)} 条 ({nav_df["nav_date"].min()} ~ {nav_df["nav_date"].max()})')
            total_nav += len(nav_df)
            if to_mysql:
                for _, row in nav_df.iterrows():
                    nav_records.append({
                        'ts_code': ts_code,
                        'nav_date': str(row['nav_date'])[:10],
                        'unit_nav': float(row['unit_nav']),
                        'accum_nav': float(row['accum_nav']) if pd.notna(row.get('accum_nav')) else None,
                        'adj_nav': float(row['adj_nav']) if pd.notna(row.get('adj_nav')) else None,
                        'ann_date': str(row['ann_date'])[:10] if pd.notna(row.get('ann_date')) else None,
                    })
        else:
            print(f'  NAV: 无数据')

        # 日线
        daily_df = download_daily(api, ts_code, start_date, end_date, force)
        if daily_df is not None and not daily_df.empty:
            print(f'  日线: {len(daily_df)} 条 ({daily_df["trade_date"].min()} ~ {daily_df["trade_date"].max()})')
            total_daily += len(daily_df)
        else:
            print(f'  日线: 无数据')

        time.sleep(API_DELAY)

    # 3. 写入 MySQL
    if to_mysql and nav_records:
        print(f'\n写入 MySQL: {len(nav_records)} 条净值记录 ...')
        conn = _connect_mysql(mysql_pw)
        ensure_mysql_tables(conn)
        inserted = batch_insert_nav(conn, nav_records)
        print(f'  实际写入 {inserted} 条')
        conn.close()

    print(f'\n=== 完成 ===')
    print(f'净值: {total_nav} 条, 日线: {total_daily} 条')
    print(f'缓存目录: {CACHE_DIR.resolve()}')


def main():
    parser = argparse.ArgumentParser(description='下载 QDII ETF 净值与日线数据')
    parser.add_argument('--etfs', default=None, help='指定 ETF 代码，逗号分隔，如 513100.SH,513500.SH')
    parser.add_argument('--force', action='store_true', help='强制全量重新下载')
    parser.add_argument('--to-mysql', action='store_true', help='写入 MySQL（需 MySQL 运行中）')
    parser.add_argument('--mysql-pw', default=None, help='MySQL 密码')
    parser.add_argument('--start-date', default=START_DATE, help='起始日期 YYYYMMDD')
    parser.add_argument('--end-date', default=END_DATE, help='截止日期 YYYYMMDD')
    parser.add_argument('--env', default='prd', choices=['prd', 'qa'], help='API 环境')
    args = parser.parse_args()

    etf_codes = None
    if args.etfs:
        etf_codes = [s.strip() for s in args.etfs.split(',') if s.strip()]

    run(etf_codes=etf_codes, force=args.force,
        to_mysql=args.to_mysql, mysql_pw=args.mysql_pw,
        start_date=args.start_date, end_date=args.end_date, env=args.env)


if __name__ == '__main__':
    main()
