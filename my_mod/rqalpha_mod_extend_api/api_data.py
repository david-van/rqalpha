#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2026/3/25 23:22
# @Author  : david_van
# @Desc    :

from typing import Union, Optional
from typing import Union, List, Optional, Dict, Any
from rqalpha.api import export_as_api
from rqalpha.core.execution_context import ExecutionContext
from rqalpha.const import EXECUTION_PHASE
from datetime import datetime
import pandas as pd
from rqalpha.interface import AbstractMod
from rqalpha.utils.logger import user_system_log

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
    user_system_log.info('inject api of get_all_securities')
    raise NotImplementedError