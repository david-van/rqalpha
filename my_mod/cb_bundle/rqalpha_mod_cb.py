"""
可转债数据源 Mod

在 RQAlpha 启动时自动注入 CBDataSource，替换默认的 BaseDataSource。
需要 bundle 目录中已存在 convertibles.h5 和 cb_instruments.pk。

使用方式 - 在策略配置中声明:
    "mod": {
        "cb": {
            "enabled": True,
            "lib": "my_mod.cb_bundle.rqalpha_mod_cb"
        }
    }
"""
from rqalpha.interface import AbstractMod


def load_mod():
    return CBMod()


class CBMod(AbstractMod):
    def __init__(self):
        pass

    def start_up(self, env, mod_config):
        from my_mod.cb_bundle.cb_data_source import inject_cb_data_source
        inject_cb_data_source(env, env.config.base)

    def tear_down(self, code, exception=None):
        pass
