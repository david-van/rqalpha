"""
RQAlpha rqdatac 扩展 API 的 IDE 类型声明模块（仅供 TYPE_CHECKING 使用）。

用法：

    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from strategy_lab._typing_rqdatac import *
"""

from __future__ import annotations

import datetime
from typing import Any, Iterable

import pandas as pd


def get_split(
    order_book_ids: str | list[str],
    start_date: str | datetime.date | None = None,
) -> pd.DataFrame: ...


def index_components(
    order_book_id: str,
    date: str | datetime.date | None = None,
) -> list[str]: ...


def index_weights(
    order_book_id: str,
    date: str | datetime.date | None = None,
) -> pd.Series: ...


def concept(*concept_names: str) -> list[str]: ...


def get_margin_stocks(exchange: str | None = None, margin_type: str = "all") -> list[str]: ...


def get_price(
    order_book_ids: str | Iterable[str],
    start_date: datetime.date | str,
    end_date: datetime.date | datetime.datetime | str | None = None,
    frequency: str = "1d",
    fields: Iterable[str] | None = None,
    adjust_type: str = "pre",
    skip_suspended: bool = False,
    expect_df: bool = False,
) -> pd.DataFrame | pd.Series | Any: ...


def get_securities_margin(
    order_book_ids: str | Iterable[str],
    count: int = 1,
    fields: str | None = None,
    expect_df: bool = True,
) -> pd.Series | pd.DataFrame: ...


def get_shares(
    order_book_ids: str | list[str],
    count: int = 1,
    fields: str | None = None,
    expect_df: bool = False,
) -> pd.DataFrame | pd.Series: ...


def get_turnover_rate(
    order_book_ids: str | list[str],
    count: int = 1,
    fields: set[str] | None = None,
    expect_df: bool = False,
) -> pd.Series | pd.DataFrame | Any: ...


def get_price_change_rate(
    order_book_ids: str | list[str],
    count: int = 1,
    expect_df: bool = False,
) -> pd.DataFrame | pd.Series: ...


def get_factor(
    order_book_ids: str | list[str],
    factors: str | list[str],
    count: int = 1,
    universe: str | list[Any] | None = None,
    expect_df: bool = False,
) -> pd.DataFrame: ...


def get_industry(industry: str, source: str = "citics") -> list[str]: ...


def get_instrument_industry(
    order_book_ids: str | list[str],
    source: str = "citics",
    level: int = 1,
) -> pd.DataFrame: ...


def get_stock_connect(
    order_book_ids: str | list[str],
    count: int = 1,
    fields: str | None = None,
    expect_df: bool = False,
) -> pd.DataFrame: ...


def current_performance(
    order_book_id: str,
    info_date: str | None = None,
    quarter: str | None = None,
    interval: str = "1q",
    fields: str | list[str] | None = None,
) -> pd.DataFrame: ...


def get_dominant_future(underlying_symbol: str, rule: int = 0) -> str | None: ...


class econ:
    @staticmethod
    def get_reserve_ratio(reserve_type: str = "all", n: int = 1) -> pd.DataFrame | None: ...

    @staticmethod
    def get_money_supply(n: int = 1) -> pd.DataFrame | None: ...


class futures:
    @staticmethod
    def get_dominant(underlying_symbol: str, rule: int = 0) -> str | None: ...

    @staticmethod
    def get_member_rank(which: str, count: int = 1, rank_by: str = "short") -> pd.DataFrame: ...

    @staticmethod
    def get_warehouse_stocks(underlying_symbols: str | list[str], count: int = 1) -> pd.DataFrame: ...

    @staticmethod
    def get_dominant_price(
        underlying_symbols: str | list[str],
        start_date: Any = None,
        end_date: Any = None,
        frequency: str = "1d",
        fields: str | list[str] | None = None,
        adjust_type: str = "pre",
        adjust_method: str = "prev_close_spread",
    ) -> pd.DataFrame | None: ...


def get_fundamentals(
    query: Any,
    entry_date: Any = None,
    interval: str = "1d",
    report_quarter: bool = False,
    expect_df: bool = False,
    **kwargs: Any,
) -> pd.DataFrame | Any: ...


def get_financials(
    query: Any,
    quarter: str | None = None,
    interval: str = "4q",
    expect_df: bool = False,
) -> pd.DataFrame | pd.Series | Any: ...


def get_pit_financials(
    fields: Any,
    quarter: str | None = None,
    interval: Any = None,
    order_book_ids: Any = None,
    if_adjusted: str | int = "all",
) -> pd.DataFrame: ...


def get_pit_financials_ex(
    order_book_ids: str | list[str],
    fields: str | list[str],
    count: int,
    statements: str = "latest",
) -> pd.DataFrame | None: ...


def query(*entities: Any) -> Any: ...
