
import os
from typing import Union, Optional
from typing import Union, List, Optional, Dict, Any
from datetime import datetime
import pandas as pd
from rqalpha.interface import AbstractMod
from rqalpha.utils.logger import user_system_log


class ExtendAPIJukuanMod(AbstractMod):
    def __init__(self):
        # 注入API 一定要在初始化阶段，否则无法成功注入
        self._inject_api()

    def start_up(self, env, mod_config):
        pass

    def tear_down(self, code, exception=None):
        pass

    def _inject_api(self):
        from rqalpha.api import export_as_api
        from rqalpha.core.execution_context import ExecutionContext
        from rqalpha.const import EXECUTION_PHASE

        @export_as_api
        @ExecutionContext.enforce_phase(EXECUTION_PHASE.ON_INIT,
                                        EXECUTION_PHASE.BEFORE_TRADING,
                                        EXECUTION_PHASE.ON_BAR,
                                        EXECUTION_PHASE.AFTER_TRADING,
                                        EXECUTION_PHASE.SCHEDULED)
        def get_all_securities(
                self,
                types: Union[str, List[str]] = 'stock',
                date: Optional[Union[str, datetime]] = None
        ) -> pd.DataFrame:
            user_system_log.info('inject api of get_all_securities')
            raise NotImplementedError
