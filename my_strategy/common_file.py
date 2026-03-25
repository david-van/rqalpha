#!/usr/bin/env python
# @Date    : 2025/7/24 07:05
# @Author  : david_van
# @Desc    :
import glob
import os

import pandas as pd


def get_project_root() -> str:
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while not os.path.exists(os.path.join(current_dir, "pyproject.toml")):
        current_dir = os.path.dirname(current_dir)
    return current_dir


def safe_rename(df: pd.DataFrame, rename_dict: dict, warn_missing: bool = False) -> pd.DataFrame:
    """
    安全地重命名 DataFrame 的列名。

    参数:
        df (pd.DataFrame): 要操作的 DataFrame
        rename_dict (dict): 旧列名到新列名的映射
        warn_missing (bool): 如果为 True,当列不存在时抛出警告

    返回:
        pd.DataFrame: 新的 DataFrame(原 DataFrame 不会被修改)
    """
    existing_columns = set(df.columns)
    valid_rename = {}

    for old_name, new_name in rename_dict.items():
        if old_name in existing_columns:
            valid_rename[old_name] = new_name
        elif warn_missing:
            import warnings

            warnings.warn(f"列 '{old_name}' 不存在,跳过重命名", stacklevel=2)
    if len(valid_rename) == 0:
        return df
    return df.rename(columns=valid_rename)


def find_files_with_glob(directory, pattern):
    """
    使用通配符模式查找文件
    :param directory: 文件夹路径
    :param pattern: 匹配模式(如'*report*.docx')
    :return: 匹配的文件路径列表
    """
    directory = os.path.abspath(os.path.join(project_root, directory))
    return glob.glob(os.path.join(directory, pattern))


def find_one_file(directory, keyword, append_pattern: bool = False):
    """
    使用通配符模式查找文件
    :param directory: 文件夹路径
    :param keyword: 关键字
    :param append_pattern 是否追加*,类似sql中的like
    :return: 匹配的文件路径列表
    """
    pattern = "*" + keyword + "*" if append_pattern else keyword
    pb_file_list = find_files_with_glob(directory, pattern)
    if len(pb_file_list) != 1:
        raise ValueError(f"当前目录{directory}根据'{pattern}'找到的文件数为:{len(pb_file_list)},请检查")
    return pb_file_list[0]


def get_code_by_name(name: str) -> str:
    index_df = pd.read_csv(os.path.join(project_root, index_csv_file))
    for _index, row in index_df.iterrows():
        # 指数代码,指数名称,发布时间
        index_code = row["指数代码"]
        index_name = row["指数名称"]
        if index_name == name:
            return index_code


# 获取项目根目录
project_root = get_project_root()
print(f"项目根目录是: {project_root}")

index_path = r"data/index"
index_csv_file = index_path + r"/指数.csv"
index_etf_csv_file = index_path + r"/指数行业etf.csv"
pb_path = index_path + "/pb"
compare_path = index_path + "/compare"
fund_etf_hist_min = index_path + "/fund_etf_hist_min"
daily_path = index_path + "/daily"
points_path = index_path + "/points"
intermediate_path = index_path + "/intermediate"
index_show_path = index_path + "/show"
trade_record_path = index_path + "/trade_record/trend/use_atr"

trend_show_path = index_path + "/show/trend/use_atr"
trend_record_path = index_path + "/trade_record/trend"

stock_path = r"data/stock"
stock_show_path = index_path + "/show"


def change_trade_record_path(path: str):
    global trade_record_path
    trade_record_path = index_path + path


def get_trade_record_path() -> str:
    return trade_record_path


def change_trend_show_path(path: str):
    global trend_show_path
    trend_show_path = index_path + path


def get_trend_show_path() -> str:
    return trend_show_path
