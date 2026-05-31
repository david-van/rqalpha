"""
可转债赎回公告数据下载脚本

从 sxsc_tushare 的 cb_call 接口批量获取赎回公告日期和登记日，
输出为独立的 cb_redemption.pk，不修改已有的任何文件。

用法:
    python my_mod/cb_bundle/download_cb_redemption.py
"""
import os
import sys
import pickle
import argparse

import pandas as pd

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import sxsc_tushare as ts

BUNDLE_PATH = r'D:\datas\bundle'
REDEMPTION_PKL_FILE = 'cb_redemption.pk'


PAGE_SIZE = 2000  # API 文档单次限量 2000


def _fetch_all(api):
    """分页获取全部 cb_call 记录，返回 DataFrame。"""
    frames = []
    offset = 0
    while True:
        chunk = api.cb_call(limit=PAGE_SIZE, offset=offset)
        if chunk is None or len(chunk) == 0:
            break
        frames.append(chunk)
        n = len(chunk)
        print(f'  [分页] offset={offset} → +{n} 条')
        offset += n
        if n < PAGE_SIZE:
            break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run(env='prd', bundle_path=BUNDLE_PATH):
    api = ts.get_api(env=env, timeout=60)

    df = _fetch_all(api)
    if len(df) == 0:
        print('[cb_call] 未获取到数据，退出')
        return
    print(f'[cb_call] 共计 {len(df)} 条赎回记录')

    # 筛选已执行的记录（有 call_reg_date 才算真正执行了赎回）
    executed = df[df['call_reg_date'].notna() & (df['call_reg_date'].astype(str).str.strip() != '')].copy()
    print(f'[cb_call] 已执行赎回: {len(executed)} 条, 涉及 {executed["ts_code"].nunique()} 只债券')

    # 同一只债券可能有多条执行记录（如先到赎后强赎），取登记日最早的那条
    executed = executed.sort_values('call_reg_date')
    executed = executed.drop_duplicates(subset='ts_code', keep='first')

    redemption = {}
    for _, row in executed.iterrows():
        code = row['ts_code']
        ann = str(row['ann_date']).strip() if pd.notna(row['ann_date']) else ''
        reg = str(row['call_reg_date']).strip()
        ctype = str(row.get('call_type', ''))
        redemption[code] = {
            'ann_date': ann,
            'call_reg_date': reg,
            'call_type': ctype,
        }

    pkl_path = os.path.join(bundle_path, REDEMPTION_PKL_FILE)
    with open(pkl_path, 'wb') as f:
        pickle.dump(redemption, f, protocol=2)
    print(f'[save] {len(redemption)} 条 → {pkl_path}')
    print('完成')


def main():
    parser = argparse.ArgumentParser(description='下载可转债赎回公告数据')
    parser.add_argument('--env', default='prd', choices=['prd', 'qa'], help='API 环境')
    parser.add_argument('--bundle-path', default=BUNDLE_PATH, help='bundle 目录')
    args = parser.parse_args()
    run(env=args.env, bundle_path=args.bundle_path)


if __name__ == '__main__':
    run()
