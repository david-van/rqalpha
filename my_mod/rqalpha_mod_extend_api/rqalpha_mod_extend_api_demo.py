from rqalpha.interface import AbstractMod
from rqalpha.utils.logger import user_system_log
from my_mod.rqalpha_mod_extend_api import api_data

class ExtendAPIJukuanMod(AbstractMod):
    def __init__(self):
        # 注入API 一定要在初始化阶段，否则无法成功注入
        self._inject_api()

    def start_up(self, env, mod_config):
        pass

    def tear_down(self, code, exception=None):
        pass

    def _inject_api(self):
        user_system_log.info('inject jukuan api')
