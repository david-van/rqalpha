#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从本地 bundle/instruments.pk 读取股票名称→代码映射，
在 holdings CSV 的"持有股票"列后面插入"持有股票代码"列。

用法: python build_name_code_map.py
"""

import pickle
import pandas as pd
from pathlib import Path

BUNDLE_PATH = Path(r"D:\datas\bundle\instruments.pk")
CSV_DIR = Path(__file__).parent

# 公司更名导致的名称不一致，手动补充
# NAME_OVERRIDES = {
#     "旭升股份": "603305.XSHG",   # 已更名为 旭升集团
#     "晨光文具": "603899.XSHG",   # 已更名为 晨光股份
# }


def load_name_to_code(bundle_path: Path) -> dict:
    """从 instruments.pk 加载 股票名称→代码 映射"""
    with open(bundle_path, "rb") as f:
        instruments = pickle.load(f)

    name_to_code = {}
    for inst in instruments:
        if inst["type"] == "CS":  # Common Stock
            name_to_code[inst["symbol"]] = inst["order_book_id"]

    # 合并手动覆盖（更名公司）
    # name_to_code.update(NAME_OVERRIDES)
    return name_to_code


def add_code_column(csv_path: Path, name_to_code: dict):
    """在 CSV 的 '持有股票' 列后插入 '持有股票代码' 列"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if "持有股票代码" in df.columns:
        print(f"  [跳过] {csv_path.name} — 已有 '持有股票代码' 列")
        return

    if "持有股票" not in df.columns:
        print(f"  [跳过] {csv_path.name} — 无 '持有股票' 列")
        return

    codes_col = []
    missing = set()

    for raw in df["持有股票"]:
        if pd.isna(raw) or str(raw).strip() == "":
            codes_col.append("")
            continue

        codes = []
        for name in str(raw).split("|"):
            name = name.strip()
            if not name:
                continue
            code = name_to_code.get(name)
            if code:
                codes.append(code)
            else:
                codes.append(f"??{name}??")
                missing.add(name)

        codes_col.append("|".join(codes))

    # 插入到"持有股票"列之后
    col_idx = df.columns.get_loc("持有股票")
    df.insert(col_idx + 1, "持有股票代码", codes_col)

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  [完成] {csv_path.name} ({len(df)} 行)")

    if missing:
        print(f"  [警告] 未找到代码: {sorted(missing)}")


def main():
    print("加载 instruments.pk ...")
    name_to_code = load_name_to_code(BUNDLE_PATH)
    print(f"共加载 {len(name_to_code)} 只股票的名称→代码映射\n")

    csv_files = sorted(CSV_DIR.glob("*holdings*.csv"))
    if not csv_files:
        print("未找到 holdings CSV 文件")
        return

    for csv_path in csv_files:
        add_code_column(csv_path, name_to_code)

    print("\n全部完成。")


if __name__ == "__main__":
    main()
