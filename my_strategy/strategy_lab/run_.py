from pathlib import Path
import sys

from my_strategy.common_file import project_root

import os
from rqalpha import run_file


def main():
    strategy_file = os.path.join(project_root,   'my_strategy/strategy_lab/new_etf_clustering.py')
    # strategy_file = os.path.join(project_root,   'rqalpha/examples/macd.py')
    config = {
        "base": {
            "strategy_file": str(strategy_file),
            "data_bundle_path": r"D:\datas\bundle",
            "start_date": "2026-01-01",
            "end_date": "2026-02-01",
            "frequency": "1d",
            "accounts": {
                "stock": 10000,
            },
        },
        "extra": {
            "log_level": "info",
            "log_file": str(Path(__file__).with_name("backtest.log")),
        },
        "mod": {
            "rqalpha_mod_extend_api": {
                "enabled": True,
                "lib": "my_mod.rqalpha_mod_extend_api",
            }
        }
    }

    result = run_file(str(strategy_file), config=config)
    print(result)
    return result
    # return run_file(str(strategy_file), config=config)


if __name__ == "__main__":
    main()
