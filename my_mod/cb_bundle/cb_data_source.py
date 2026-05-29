"""
可转债自定义数据源

CBDataSource 继承 BaseDataSource，在已有股票/期货/指数数据的 bundle 基础上，
额外注册可转债的日线数据和合约信息。
"""
import os
import pickle
from typing import TYPE_CHECKING

import numpy as np

from rqalpha.const import INSTRUMENT_TYPE, MARKET
from rqalpha.data.base_data_source.data_source import BaseDataSource
from rqalpha.data.base_data_source.storages import DayBarStore
from rqalpha.model.instrument import Instrument
from rqalpha.environment import Environment

if TYPE_CHECKING:
    from rqalpha.interface import AbstractDataSource


# 可转债专属 DayBarStore：扩展 DEFAULT_DTYPE，HDF5 里没有的债券返回空时字段也完整
class CBDayBarStore(DayBarStore):
    DEFAULT_DTYPE = np.dtype([
        ('datetime',       np.uint64),
        ('open',           np.float64),
        ('close',          np.float64),
        ('high',           np.float64),
        ('low',            np.float64),
        ('prev_close',     np.float64),
        ('change',         np.float64),
        ('pct_chg',        np.float64),
        ('limit_up',       np.float64),
        ('limit_down',     np.float64),
        ('volume',         np.float64),
        ('total_turnover', np.float64),
        ('cb_over_rate',   np.float64),
        ('cb_value',       np.float64),
        ('bond_value',     np.float64),
        ('bond_over_rate', np.float64),
    ])


class CBDataSource(BaseDataSource):
    """
    在 BaseDataSource 基础上增加可转债 (Convertible) 数据支持。

    Bundle 目录中需要存在:
      - convertibles.h5    : DayBarStore 兼容的 HDF5 日线文件
      - cb_instruments.pk  : pickle 格式的可转债合约列表
    """

    def __init__(self, base_config):
        super().__init__(base_config)
        path = base_config.data_bundle_path

        def _p(name):
            return os.path.join(path, name)

        # 注册可转债合约
        cb_pkl = _p('cb_instruments.pk')
        if os.path.exists(cb_pkl):
            with open(cb_pkl, 'rb') as f:
                cb_instruments = pickle.load(f)
            insts = []
            for d in cb_instruments:
                if d.get('type') == 'Convertible':
                    insts.append(Instrument(d, self._future_info_store.get_tick_size, market=MARKET.CN))
            if insts:
                self.register_instruments(insts)
                print(f'[CBDataSource] 注册 {len(insts)} 只可转债合约')

        # 注册可转债日线存储（使用 CBDayBarStore 保证 dtype 一致）
        cb_h5 = _p('convertibles.h5')
        if os.path.exists(cb_h5):
            cb_store = CBDayBarStore(cb_h5)
            self.register_day_bar_store(INSTRUMENT_TYPE.CONVERTIBLE, cb_store, market=MARKET.CN)
            print(f'[CBDataSource] 注册可转债日线存储: {cb_h5}')


def inject_cb_data_source(env: Environment, config):
    """
    注入 CBDataSource 到 Environment。

    在 Mod 的 start_up() 中调用此函数:
        inject_cb_data_source(env, config.base)

    也可在策略文件中直接调用（需在 rqalpha.run() 之前）。
    """
    if not hasattr(env, 'data_source'):
        env.set_data_source(CBDataSource(config))
