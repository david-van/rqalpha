#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2026/3/25 23:22
# @Author  : david_van
# @Desc    : 聚宽 API 适配层，将 JQ 风格接口映射到 RQAlpha all_instruments 等基础 API

from typing import Union, Optional, List
from rqalpha.api import export_as_api
from rqalpha.core.execution_context import ExecutionContext
from rqalpha.const import EXECUTION_PHASE
from datetime import datetime
import pandas as pd
from rqalpha.utils.logger import user_system_log

from rqalpha.apis import *

# 聚宽 types 参数 -> RQAlpha INSTRUMENT_TYPE 映射
_JQ_TYPE_MAP = {
    'stock': ['CS'],
    'fund': ['ETF', 'LOF'],
    'index': ['INDX'],
    'futures': ['Future'],
    'etf': ['ETF'],
    'lof': ['LOF'],
}

# RQAlpha type 值 -> 聚宽 type 值的反向映射
_RQ_TO_JQ_TYPE = {
    'CS': 'stock',
    'ETF': 'etf',
    'LOF': 'lof',
    'INDX': 'index',
    'Future': 'futures',
}


@export_as_api
@ExecutionContext.enforce_phase(EXECUTION_PHASE.ON_INIT,
                                EXECUTION_PHASE.BEFORE_TRADING,
                                EXECUTION_PHASE.ON_BAR,
                                EXECUTION_PHASE.AFTER_TRADING,
                                EXECUTION_PHASE.SCHEDULED)
def get_all_securities(
        types: Union[str, List[str]] = 'stock',
        date: Optional[Union[str, datetime]] = None
) -> pd.DataFrame:
    """
    聚宽 get_all_securities 适配接口。

    部分适配说明：
    - 支持的 types: 'stock', 'fund', 'index', 'futures', 'etf', 'lof'
    - 不支持的 types: 'fja', 'fjb' (分级基金A/B，RQAlpha无对应类型，会被忽略并输出警告)
    - 返回 DataFrame 的 index 为证券代码 (order_book_id)
    - 返回列: display_name, name, start_date, end_date, type
      其中 display_name 和 name 均映射自 RQAlpha 的 symbol 字段
    """
    if isinstance(types, str):
        types = [types]

    # 收集需要查询的 RQAlpha instrument types
    rq_types = []
    unsupported = []
    for t in types:
        t_lower = t.lower()
        if t_lower in _JQ_TYPE_MAP:
            rq_types.extend(_JQ_TYPE_MAP[t_lower])
        else:
            unsupported.append(t)
    if unsupported:
        user_system_log.warning(
            'get_all_securities: 不支持的 types=%s，已忽略', unsupported
        )

    if not rq_types:
        return pd.DataFrame(columns=['display_name', 'name', 'start_date', 'end_date', 'type'])

    # 去重
    rq_types = list(set(rq_types))

    # 逐类型查询并合并（单类型查询返回完整属性列）
    frames = []
    for rq_type in rq_types:
        df = all_instruments(type=rq_type, date=date)
        if df.empty:
            continue
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=['display_name', 'name', 'start_date', 'end_date', 'type'])

    merged = pd.concat(frames, ignore_index=True)

    # 构造聚宽风格的返回 DataFrame
    result = pd.DataFrame(index=merged['order_book_id'])
    result.index.name = None
    result['display_name'] = merged['symbol'].values
    result['name'] = merged['symbol'].values
    result['start_date'] = pd.to_datetime(merged['listed_date'].values)
    result['end_date'] = pd.to_datetime(merged['de_listed_date'].values, errors="coerce")
    # # end_date 可能含 2999-12-31（表示未退市），超出 pd.Timestamp 纳秒精度范围，保留为 datetime.date
    # result['end_date'] = [d.date() if hasattr(d, 'date') else d for d in merged['de_listed_date']]
    # 将 RQAlpha type 映射回聚宽 type
    result['type'] = merged['type'].map(
        lambda x: _RQ_TO_JQ_TYPE.get(str(x), str(x))
    ).values

    return result