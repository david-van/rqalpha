"""
RQAlpha 策略 IDE 类型声明模块（仅供 TYPE_CHECKING 使用）。

用法：

    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from strategy_lab._typing import *

该文件不会参与策略运行时的 API 注入，只用于给 IDE / Pyright / Pylance
提供静态符号和类型信息，避免在编写策略时出现整片爆红。

声明范围：
1. 基于 ``rqalpha/apis/api_base.py`` 导出的全局 API
2. 策略上下文 ``StrategyContext`` 的稳定属性
3. 策略运行时常见的 ``g`` 全局变量
"""

from __future__ import annotations

import logging as _logging
import types
from datetime import date, datetime
from typing import Any, Callable, Iterable, Optional, Union

import numpy as np
import pandas as pd

from rqalpha.const import (
    DEFAULT_ACCOUNT_TYPE,
    EVENT,
    MATCHING_TYPE,
    ORDER_STATUS,
    ORDER_TYPE,
    POSITION_DIRECTION,
    POSITION_EFFECT,
    RUN_TYPE,
    SIDE,
)
from rqalpha.core.events import Event
from rqalpha.model.instrument import Instrument
from rqalpha.model.order import LimitOrder, MarketOrder, Order, OrderStyle, TWAPOrder, VWAPOrder
from rqalpha.model.tick import TickObject
from rqalpha.portfolio import Account, Portfolio
from rqalpha.portfolio.position import Position


class RunInfo:
    start_date: date
    end_date: date
    frequency: str
    stock_starting_cash: float
    future_starting_cash: float
    slippage: float
    matching_type: str
    stock_commission_multiplier: float
    futures_commission_multiplier: float
    margin_multiplier: float
    run_type: str


class StrategyContext:
    """RQAlpha 策略上下文的稳定属性声明。"""

    universe: set[str]
    now: datetime
    run_info: RunInfo
    portfolio: Portfolio
    stock_account: Account
    future_account: Account
    config: Any

    def __setattr__(self, name: str, value: Any) -> None: ...
    def __getattr__(self, name: str) -> Any: ...


# ============================================================
# 策略运行时常见全局对象
# ============================================================

context: StrategyContext
g: types.SimpleNamespace
logger: _logging.Logger


# ============================================================
# api_base.py 中 export_as_api 注册出的类 / 常量
# ============================================================

print: Callable[..., Any]

LimitOrder = LimitOrder
MarketOrder = MarketOrder
VWAPOrder = VWAPOrder
TWAPOrder = TWAPOrder

ORDER_STATUS = ORDER_STATUS
SIDE = SIDE
POSITION_EFFECT = POSITION_EFFECT
POSITION_DIRECTION = POSITION_DIRECTION
ORDER_TYPE = ORDER_TYPE
RUN_TYPE = RUN_TYPE
MATCHING_TYPE = MATCHING_TYPE
EVENT = EVENT
DEFAULT_ACCOUNT_TYPE = DEFAULT_ACCOUNT_TYPE


# ============================================================
# api_base.py 中 export_as_api 注册出的函数
# ============================================================

def get_open_orders() -> list[Order]: ...


def submit_order(
    id_or_ins: str | Instrument,
    amount: float,
    side: SIDE,
    price_or_style: float | OrderStyle | LimitOrder | MarketOrder | VWAPOrder | TWAPOrder | None = None,
    price: float | None = None,
    style: OrderStyle | None = None,
    position_effect: POSITION_EFFECT | None = None,
) -> Order | None: ...


def cancel_order(order: Order) -> Order: ...


def update_universe(id_or_symbols: str | Instrument | Iterable[str] | Iterable[Instrument]) -> None: ...


def subscribe(id_or_symbols: str | Instrument | Iterable[str] | Iterable[Instrument]) -> None: ...


def unsubscribe(id_or_symbols: str | Instrument | Iterable[str] | Iterable[Instrument]) -> None: ...


def get_yield_curve(
    date: str | date | datetime | pd.Timestamp | None = None,
    tenor: str | None = None,
) -> pd.DataFrame: ...


def history_bars(
    order_book_id: str,
    bar_count: int,
    frequency: str,
    fields: str | list[str] | None = None,
    skip_suspended: bool = True,
    include_now: bool = False,
    adjust_type: str = "pre",
) -> np.ndarray: ...


def history_ticks(order_book_id: str, count: int) -> list[TickObject]: ...


def all_instruments(
    type: str | None = None,
    date: str | datetime | date | None = None,
) -> pd.DataFrame: ...


def instruments(id_or_symbols: str | Iterable[str]) -> None | Instrument | list[Instrument]: ...


def active_instrument(order_book_id: str | Instrument) -> Instrument: ...


def instrument_history(order_book_id: str) -> list[Instrument]: ...


def active_instruments(
    order_book_ids: str | Instrument | Iterable[str] | Iterable[Instrument],
) -> dict[str, Instrument]: ...


def instruments_history(
    order_book_ids: str | Instrument | Iterable[str] | Iterable[Instrument],
) -> list[Instrument]: ...


def get_trading_dates(
    start_date: str | date | datetime | pd.Timestamp,
    end_date: str | date | datetime | pd.Timestamp,
) -> pd.DatetimeIndex: ...


def get_previous_trading_date(
    date: str | date | datetime | pd.Timestamp,
    n: int = 1,
) -> date: ...


def get_next_trading_date(
    date: str | date | datetime | pd.Timestamp,
    n: int = 1,
) -> date: ...


def current_snapshot(id_or_symbol: str | Instrument) -> TickObject | None: ...


def get_positions() -> list[Position]: ...


def get_position(
    order_book_id: str,
    direction: POSITION_DIRECTION = POSITION_DIRECTION.LONG,
) -> Position: ...


def subscribe_event(
    event_type: EVENT,
    handler: Callable[[StrategyContext, Event], None],
) -> None: ...


def symbol(order_book_id: str | Iterable[str], sep: str = ", ") -> str: ...


def deposit(account_type: str, amount: float, receiving_days: int = 0) -> Any: ...


def withdraw(account_type: str, amount: float) -> None: ...


def finance(amount: float, account_type: str = DEFAULT_ACCOUNT_TYPE.STOCK) -> None: ...


def repay(amount: float, account_type: str = DEFAULT_ACCOUNT_TYPE.STOCK) -> None: ...
