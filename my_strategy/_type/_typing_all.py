"""
RQAlpha 策略 IDE 类型声明推荐入口（仅供 TYPE_CHECKING 使用）。

这个文件不会参与运行时 API 注入，只是把 my_strategy._type 下拆分好的
三个 typing helper 聚合成一个更适合日常写策略的入口。

推荐用法：

    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from my_strategy._type._typing_all import *

如果你只想按需引入某一部分，也可以分别使用：

    from my_strategy._type._typing import *
    from my_strategy._type._typing_abstract import *
    from my_strategy._type._typing_rqdatac import *
"""

from __future__ import annotations

from my_strategy._type._typing import *
from my_strategy._type._typing_abstract import *
from my_strategy._type._typing_rqdatac import *
