import os
from pathlib import Path
import sys

from my_strategy.common_file import project_root

from rqalpha import run_file


def main():
    strategy_file = os.path.join(project_root, 'my_strategy/strategies/pt_核心资产轮动.py')
    config = {
        "base": {
            "strategy_file": str(strategy_file),
            "data_bundle_path": r"D:\datas\bundle",
            "start_date": "2026-01-01",
            "end_date": "2026-03-01",
            "frequency": "1d",

            "slippage": '0.0',
            "accounts": {
                "stock": 10000,
            },
        },
        "mod": {
            "sys_transaction_cost": {
                "enabled": True,
                # 万0.5 = 默认万8 × 0.0625
                "stock_commission_multiplier": 0.0625,
                # 免五：最低佣金设为 0
                "stock_min_commission": 0,
                # ETF 买卖无印花税，设为 0
                "tax_multiplier": 0,
            },
            "sys_simulation": {
                "enabled": True,
                "matching_type": "current_bar",
                "slippage": 0,
            },
            "sys_analyser": {
                "enabled": True,
                "plot": True,
                "benchmark": "000300.XSHG",
                "output_file": "result.pkl",
            },
        },
        "extra": {
            "log_level": "info",
            "log_file": str(Path(__file__).with_name("backtest.log")),
        },
    }

    result = run_file(str(strategy_file), config=config)
    # print(result)
    # result["trades"].to_csv("trades.csv", encoding="utf-8-sig")
    return result
    # return run_file(str(strategy_file), config=config)


if __name__ == "__main__":
    main()
