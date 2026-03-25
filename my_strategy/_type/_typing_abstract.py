"""
RQAlpha 抽象交易 API 的 IDE 类型声明模块（仅供 TYPE_CHECKING 使用）。

用法：

    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from strategy_lab._typing_abstract import *
"""

from __future__ import annotations

from typing import Optional

from rqalpha.model.instrument import Instrument
from rqalpha.model.order import LimitOrder, MarketOrder, Order, OrderStyle, TWAPOrder, VWAPOrder

PRICE_OR_STYLE_TYPE = int | float | OrderStyle | None
TUPLE_PRICE_OR_STYLE_TYPE = (
    float
    | OrderStyle
    | None
    | tuple
    | tuple[PRICE_OR_STYLE_TYPE]
    | tuple[PRICE_OR_STYLE_TYPE, PRICE_OR_STYLE_TYPE]
)


def order_shares(
    id_or_ins: str | Instrument,
    amount: int,
    price_or_style: PRICE_OR_STYLE_TYPE = None,
    price: float | None = None,
    style: OrderStyle | None = None,
) -> Order | None: ...


def order_value(
    id_or_ins: str | Instrument,
    cash_amount: float,
    price_or_style: PRICE_OR_STYLE_TYPE = None,
    price: float | None = None,
    style: OrderStyle | None = None,
) -> Order | None: ...


def order_percent(
    id_or_ins: str | Instrument,
    percent: float,
    price_or_style: PRICE_OR_STYLE_TYPE = None,
    price: float | None = None,
    style: OrderStyle | None = None,
) -> Order | None: ...


def order_target_value(
    id_or_ins: str | Instrument,
    cash_amount: float,
    price_or_style: TUPLE_PRICE_OR_STYLE_TYPE = None,
    price: float | None = None,
    style: OrderStyle | None = None,
) -> Order | None: ...


def order_target_percent(
    id_or_ins: str | Instrument,
    percent: float,
    price_or_style: TUPLE_PRICE_OR_STYLE_TYPE = None,
    price: float | None = None,
    style: OrderStyle | None = None,
) -> Order | None: ...


def buy_open(
    id_or_ins: str | Instrument,
    amount: int,
    price_or_style: PRICE_OR_STYLE_TYPE = None,
    price: float | None = None,
    style: OrderStyle | None = None,
) -> Order | list[Order] | None: ...


def buy_close(
    id_or_ins: str | Instrument,
    amount: int,
    price_or_style: PRICE_OR_STYLE_TYPE = None,
    price: float | None = None,
    style: OrderStyle | None = None,
    close_today: bool = False,
) -> Order | list[Order] | None: ...


def sell_open(
    id_or_ins: str | Instrument,
    amount: int,
    price_or_style: PRICE_OR_STYLE_TYPE = None,
    price: float | None = None,
    style: OrderStyle | None = None,
) -> Order | list[Order] | None: ...


def sell_close(
    id_or_ins: str | Instrument,
    amount: float,
    price_or_style: PRICE_OR_STYLE_TYPE = None,
    price: float | None = None,
    style: OrderStyle | None = None,
    close_today: bool = False,
) -> Order | list[Order] | None: ...


def order(
    id_or_ins: str | Instrument,
    quantity: int,
    price_or_style: PRICE_OR_STYLE_TYPE = None,
    price: float | None = None,
    style: OrderStyle | None = None,
) -> list[Order]: ...


def order_to(
    id_or_ins: str | Instrument,
    quantity: int,
    price_or_style: PRICE_OR_STYLE_TYPE = None,
    price: float | None = None,
    style: OrderStyle | None = None,
) -> list[Order]: ...


def exercise(
    id_or_ins: str | Instrument,
    amount: int,
    convert: bool = False,
) -> Order | None: ...
