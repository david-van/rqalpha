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

# 获取项目根目录
project_root = get_project_root()
print(f"项目根目录是: {project_root}")
