#!/usr/bin/env python
# @Date    : 2025/7/24 07:05
# @Author  : david_van
# @Desc    :
import glob
import os

import pandas as pd

# RQAlpha bundle 目录的标志性文件：只有两者同时存在才认为是有效 bundle
_BUNDLE_MARKERS = ("instruments.pk", "stocks.h5")


def is_bundle_dir(path) -> bool:
    """判断 path 是否是一个有效的 RQAlpha bundle 目录"""
    if not path:
        return False
    p = os.path.expanduser(str(path).strip().strip('"').strip("'"))
    return all(os.path.isfile(os.path.join(p, m)) for m in _BUNDLE_MARKERS)


def _bundle_candidates():
    """按优先级生成候选 bundle 路径。

    1) 环境变量 RQALPHA_BUNDLE（显式指定，最高优先级）
    2) 常见位置：Linux 服务器 / 本机 Windows / 用户目录
    """
    env = os.environ.get("RQALPHA_BUNDLE") or os.environ.get("BUNDLE_PATH")
    if env:
        yield env
    yield "/home/data/bundle"
    yield "/home/data"
    yield os.path.join(os.path.expanduser("~"), "datas", "bundle")
    yield r"D:\datas\bundle"
    yield "D:/datas/bundle"
    yield os.path.join(os.path.expanduser("~"), ".rqalpha", "bundle")


def resolve_bundle_path(default=None):
    """解析 RQAlpha bundle 路径，跨平台可用。

    优先级：RQALPHA_BUNDLE 环境变量 > 常见位置自动探测 > default。
    找不到有效 bundle 时返回 default（不抛异常，交由 RQAlpha 自己报错）。

    Windows 本机放 D:\\datas\\bundle；Linux 服务器放 /home/data/bundle，
    两者都能自动命中，因此迁移机器时无需改代码。
    """
    for cand in _bundle_candidates():
        if is_bundle_dir(cand):
            return os.path.expanduser(str(cand).strip().strip('"').strip("'"))
    return default


def get_project_root() -> str:
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while not os.path.exists(os.path.join(current_dir, "pyproject.toml")):
        current_dir = os.path.dirname(current_dir)
    return current_dir

# 获取项目根目录
project_root = get_project_root()
print(f"项目根目录是: {project_root}")
