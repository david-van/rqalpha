"""
可转债数据下载脚本

从 sxsc_tushare 下载可转债基础信息和日线数据，
输出为 RQAlpha bundle 兼容的 HDF5 和 instruments pickle 文件。

用法:
    python my_mod/cb_bundle/download_cb_data.py                # 测试模式: 只下载2只
    python my_mod/cb_bundle/download_cb_data.py --all          # 全量下载（增量更新）
    python my_mod/cb_bundle/download_cb_data.py --all --force  # 全量下载（强制覆盖）
"""
import os
import sys
import argparse
import pickle
from datetime import datetime

import numpy as np
import h5py

# 将 rqalpha 项目根目录加入 sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import sxsc_tushare as ts

# ── 配置 ────────────────────────────────────────────
BUNDLE_PATH = r'D:\datas\bundle'
CB_H5_FILE = 'convertibles.h5'
CB_PKL_FILE = 'cb_instruments.pk'
TEST_LIMIT = 2  # 测试模式下最多下载的债券数量
TEST_CODES = ['128119.SZ', '110092.SH']  # 测试模式下指定的债券代码（优先于 limit）

# 可转债日线 dtype（股票字段 + 可转债专属字段）
# 可转债日线 dtype — 包含 cb_daily API 返回的全部字段
CB_DTYPE = np.dtype([
    ('datetime',       '<i8'),
    ('open',           '<f8'),
    ('close',          '<f8'),
    ('high',           '<f8'),
    ('low',            '<f8'),
    ('prev_close',     '<f8'),
    ('change',         '<f8'),  # 涨跌额
    ('pct_chg',        '<f8'),  # 涨跌幅(%)
    ('limit_up',       '<f8'),
    ('limit_down',     '<f8'),
    ('volume',         '<f8'),
    ('total_turnover', '<f8'),
    ('cb_over_rate',   '<f8'),  # 转股溢价率(%)
    ('cb_value',       '<f8'),  # 转股价值
    ('bond_value',     '<f8'),  # 纯债价值
    ('bond_over_rate', '<f8'),  # 纯债溢价率(%)
])

# CB instrument 扩展字段映射
EXCHANGE_MAP = {'SH': 'XSHG', 'SZ': 'XSHE'}
BOARD_TYPE_MAP = {'SH': 'MainBoard', 'SZ': 'GEM'}


def safe_date(val):
    """将 pandas 的 NaN/NaT/None 转为 RQAlpha 默认日期字符串"""
    import pandas as pd
    if pd.isna(val) or val is None or val == '' or val == '0':
        return '0000-00-00'
    return str(val)


def date_str_to_int(d: str) -> int:
    """将 YYYYMMDD 字符串转为 int64 (YYYYMMDDHHMMSS)"""
    if not d or d in ('0000-00-00', '9999-12-31', '0'):
        return 0
    d = d.replace('-', '')
    return int(d) * 1000000  # YYYYMMDD → YYYYMMDDHHMMSS


def download_basic_info(api, test_codes: list = None, start_date: str = ''):
    """
    获取可转债基本信息，返回 Instrument 字典列表和 ts_code 列表。
    test_codes: 指定下载的债券代码列表，None 表示全量下载。
    start_date:  YYYYMMDD，过滤掉在此之前已退市的债券。
    """
    df = api.cb_basic()
    total = len(df)
    print(f'[cb_basic] 获取到 {total} 条记录')

    if test_codes:
        ts_codes = test_codes
        df = df[df['ts_code'].isin(ts_codes)]
    else:
        ts_codes = df['ts_code'].tolist()

    # 用 start_date 过滤已退市债券：delist_date < start_date 的跳过
    if start_date:
        start_int = int(start_date.replace('-', ''))
        before_filter = len(df)

        def still_alive(delist_val):
            d = safe_date(delist_val)
            if d == '0000-00-00':
                return True  # 还在市
            try:
                return int(d.replace('-', '')) >= start_int
            except ValueError:
                return True  # 无法解析的保留

        df = df[df['delist_date'].apply(still_alive)]
        filtered_out = before_filter - len(df)
        if filtered_out > 0:
            print(f'[cb_basic] 过滤掉 {filtered_out} 只已退市债券（退市日早于 {start_date}）')

    instruments = []
    for _, row in df.iterrows():
        ts_code = row['ts_code']
        exchange = EXCHANGE_MAP.get(ts_code.split('.')[-1], 'XSHG')

        inst = {
            'order_book_id': ts_code,
            'symbol': str(row.get('bond_short_name', ts_code)),
            'abbrev_symbol': ts_code.replace('.', ''),
            'type': 'Convertible',
            'exchange': exchange,
            'status': 'Active' if safe_date(row.get('delist_date')) == '0000-00-00'
                      else 'Delisted',
            'listed_date': safe_date(row.get('list_date')),
            'de_listed_date': safe_date(row.get('delist_date')),
            'maturity_date': safe_date(row.get('maturity_date')) or '2999-12-31',
            'round_lot': 10,   # 可转债一手 = 10张
            'market_tplus': 0,  # 可转债 T+0
            'board_type': 'MainBoard',
            'special_type': 'Normal',
            'industry_code': '',
            'industry_name': '',
            'sector_code': '',
            'sector_code_name': '',
            'issue_price': float(row.get('issue_price', 100.0) or 100.0),
            'par': float(row.get('par', 100.0) or 100.0),
            'remain_size': float(row.get('remain_size', 0) or 0),  # 剩余规模(亿元)
            'underlying_symbol': str(row.get('stk_code', '')),
            'conv_price': float(row.get('conv_price', 0) or 0),
            'trading_hours': '09:31-11:30,13:01-15:00',
            'trading_code': ts_code.replace('.', ''),
            'office_address': '',
            'province': '',
        }
        instruments.append(inst)

    print(f'[cb_basic] 待下载 {len(instruments)} 只债券的基本信息')
    return instruments, [i['order_book_id'] for i in instruments]


def _df_to_array(df) -> np.ndarray:
    """将 cb_daily 返回的 DataFrame 转为 CB_DTYPE 的 structured array"""
    records = []
    for _, row in df.iterrows():
        dt_int = int(str(row['trade_date']).replace('-', '')) * 1000000
        records.append((
            dt_int,
            float(row.get('open', 0) or 0),
            float(row.get('close', 0) or 0),
            float(row.get('high', 0) or 0),
            float(row.get('low', 0) or 0),
            float(row.get('pre_close', 0) or 0),
            float(row.get('change', 0) or 0),
            float(row.get('pct_chg', 0) or 0),
            0.0, 0.0,  # limit_up, limit_down
            float(row.get('vol', 0) or 0),
            float(row.get('amount', 0) or 0),
            float(row.get('cb_over_rate', 0) or 0),
            float(row.get('cb_value', 0) or 0),
            float(row.get('bond_value', 0) or 0),
            float(row.get('bond_over_rate', 0) or 0),
        ))
    arr = np.array(records, dtype=CB_DTYPE)
    arr.sort(order='datetime')
    return arr


def _flush_batch(h5, batch: dict, existing_h5: dict):
    """将一批债券数据写入 HDF5，返回 (新增数, 跳过数)"""
    added, skipped = 0, 0
    for ts_code, arr in batch.items():
        if len(arr) == 0:
            continue
        if ts_code in existing_h5:
            last_dt = existing_h5[ts_code]
            mask = arr['datetime'] > last_dt
            new_bars = arr[mask]
            if len(new_bars) == 0:
                skipped += 1
                continue
            old_data = h5[ts_code][:]
            merged = np.concatenate([old_data, new_bars])
            merged.sort(order='datetime')
            del h5[ts_code]
            h5.create_dataset(ts_code, data=merged)
            added += 1
            print(f'  [append] {ts_code}: +{len(new_bars)} bars (total {len(merged)})')
        else:
            h5.create_dataset(ts_code, data=arr)
            added += 1
            print(f'  [new]    {ts_code}: {len(arr)} bars')
    return added, skipped


def _load_existing_instruments(bundle_path):
    """加载已有的 cb_instruments.pk，不存在则返回 {}"""
    pkl_path = os.path.join(bundle_path, CB_PKL_FILE)
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            insts = pickle.load(f)
        return {i['order_book_id']: i for i in insts}
    return {}


def _load_existing_h5_keys(h5_path):
    """返回 HDF5 中已有的 ts_code 集合及其最后一条数据的 datetime"""
    existing = {}
    if os.path.exists(h5_path):
        with h5py.File(h5_path, 'r') as h5:
            for key in h5.keys():
                data = h5[key][:]
                if len(data) > 0:
                    existing[key] = int(data['datetime'][-1])
                else:
                    existing[key] = 0
    return existing


def run(codes=None, all=False, force=False, start_date='', end_date='',
        env='prd', bundle_path=BUNDLE_PATH, batch_size=20):
    """
    下载可转债数据，按批次边下载边保存，中断不丢已完成的批次。

    参数:
        codes:      指定债券代码列表。为 None 时配合 all=True 全量下载
        all:        True=全量下载所有可转债, False=仅下载 codes 指定的债券
        force:      True=强制覆盖已有数据, False=增量追加
        start_date: 起始日期 YYYYMMDD（空字符串=不限制）
        end_date:   截止日期 YYYYMMDD（空字符串=不限制）
        env:        API 环境 'prd'(公网) 或 'qa'(内网)
        bundle_path: bundle 目录路径
        batch_size: 每批下载多少只后保存一次（默认 20）
    """
    import time

    if codes is None and not all:
        codes = TEST_CODES
        mode = f'测试 ({codes})'
    elif codes:
        mode = f'指定债券 ({codes})'
    else:
        mode = '全量'

    start_date = start_date.replace('-', '') if start_date else ''
    end_date = end_date.replace('-', '') if end_date else ''
    print(f'=== 可转债数据下载 - {mode} ===')
    print(f'批次大小: {batch_size} 只/批')

    api = ts.get_api(env=env, timeout=60)

    # 0. 读取已有数据
    h5_path = os.path.join(bundle_path, CB_H5_FILE)
    existing_h5 = {} if force else _load_existing_h5_keys(h5_path)
    pkl_path = os.path.join(bundle_path, CB_PKL_FILE)
    existing_insts = {} if force else _load_existing_instruments(bundle_path)

    # 1. 基本信息
    instruments, ts_codes = download_basic_info(api, test_codes=codes, start_date=start_date)
    if not instruments:
        print('未获取到可转债基本信息，退出')
        return

    # 2. 合并 instruments（先更新内存，最后一次性写文件）
    for inst in instruments:
        oid = inst['order_book_id']
        if oid in existing_insts:
            existing_insts[oid].update(inst)
        else:
            existing_insts[oid] = inst

    # 3. 分批下载日线，每批立即写入 HDF5
    os.makedirs(bundle_path, exist_ok=True)
    h5_mode = 'w' if force else 'a'
    total_added, total_skipped, total_failed = 0, 0, 0

    with h5py.File(h5_path, h5_mode) as h5:
        for batch_start in range(0, len(ts_codes), batch_size):
            batch_codes = ts_codes[batch_start:batch_start + batch_size]
            batch_data = {}
            b = batch_start // batch_size + 1
            n_batches = (len(ts_codes) + batch_size - 1) // batch_size
            print(f'\n[批次 {b}/{n_batches}] 下载 {len(batch_codes)} 只债券 ...')

            for i, ts_code in enumerate(batch_codes):
                # 计算增量起始日期
                bond_start = start_date
                if ts_code in existing_h5:
                    last_dt_str = str(existing_h5[ts_code])[:8]
                    if last_dt_str > bond_start:
                        bond_start = last_dt_str

                extra = {}
                if bond_start:
                    extra['start_date'] = bond_start
                if end_date:
                    extra['end_date'] = end_date

                idx = batch_start + i + 1
                try:
                    df = api.cb_daily(ts_code=ts_code, **extra)
                    time.sleep(0.3)
                except Exception as e:
                    print(f'  [{idx}/{len(ts_codes)}] {ts_code} 失败: {e}')
                    total_failed += 1
                    continue

                if df is None or df.empty:
                    print(f'  [{idx}/{len(ts_codes)}] {ts_code} 无数据')
                    continue

                arr = _df_to_array(df)
                batch_data[ts_code] = arr
                print(f'  [{idx}/{len(ts_codes)}] {ts_code} → {len(arr)} 条')

            # 立即写入本批次
            if batch_data:
                added, skipped = _flush_batch(h5, batch_data, existing_h5)
                total_added += added
                total_skipped += skipped
                # 更新 existing_h5，使后续批次能看到本批次刚写入的数据
                for ts_code, arr in batch_data.items():
                    if len(arr) > 0:
                        existing_h5[ts_code] = int(arr['datetime'][-1])
                print(f'[批次 {b}/{n_batches}] 已保存: 新增 {added}, 跳过 {skipped}')
            else:
                print(f'[批次 {b}/{n_batches}] 无数据')

    # 4. 保存 instruments
    merged_instruments = list(existing_insts.values())
    with open(pkl_path, 'wb') as f:
        pickle.dump(merged_instruments, f, protocol=2)
    print(f'[save] instruments: {len(merged_instruments)} 只 → {pkl_path}')

    print(f'\n=== 完成 ===')
    print(f'写入: {total_added} 只, 跳过(无新数据): {total_skipped} 只, 失败: {total_failed} 只')
    print(f'Bundle 目录: {bundle_path}')


def main():
    """CLI 入口，解析命令行参数后调用 run()"""
    parser = argparse.ArgumentParser(description='下载可转债数据')
    parser.add_argument('--all', action='store_true', help='全量下载（默认测试模式仅下载2只）')
    parser.add_argument('--force', action='store_true', help='强制覆盖已有数据（默认增量追加）')
    parser.add_argument('--codes', nargs='*', default=None, help='指定债券代码，如 128119.SZ 110092.SH')
    parser.add_argument('--start', default='', help='起始日期 YYYYMMDD（如 20200101）')
    parser.add_argument('--end', default='', help='截止日期 YYYYMMDD（如 20241231）')
    parser.add_argument('--env', default='prd', choices=['prd', 'qa'], help='API 环境')
    parser.add_argument('--bundle-path', default=BUNDLE_PATH, help='bundle 目录')
    args = parser.parse_args()

    run(codes=args.codes, all=args.all, force=args.force,
        start_date=args.start, end_date=args.end,
        env=args.env, bundle_path=args.bundle_path)


if __name__ == '__main__':
    run(all= True,start_date='20180101')
