import os
from pathlib import Path
import sys

from my_strategy.common_file import project_root


from rqalpha import run_file


def main():
    strategy_file = os.path.join(project_root,   'my_strategy/strategies/ETF动量/run_backtest.py')
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
    }

    result = run_file(str(strategy_file), config=config)
    print(result)
    return result
    # return run_file(str(strategy_file), config=config)


if __name__ == "__main__":
    main()
