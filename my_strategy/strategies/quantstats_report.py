#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
加载 RQAlpha 回测 pkl 文件，生成 QuantStats 专业分析报告。

用法: 直接修改底部 CONFIG 的参数，然后 python quantstats_report.py
依赖: pip install quantstats
"""

import pickle
from pathlib import Path

import pandas as pd
import quantstats as qs


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_returns(portfolio, col="unit_net_value"):
    nav = portfolio[col].dropna()
    return nav.pct_change().dropna()


def terminal_summary(returns, benchmark_returns=None):
    """打印终端摘要：核心指标、月度收益、最差分析、回撤期间"""
    print("\n" + "=" * 60)
    print("                   策 略 核 心 指 标")
    print("=" * 60)

    cagr = qs.stats.cagr(returns)
    sharpe = qs.stats.sharpe(returns)
    sortino = qs.stats.sortino(returns)
    max_dd = qs.stats.max_drawdown(returns)
    calmar = qs.stats.calmar(returns)
    volatility = qs.stats.volatility(returns)
    win_rate = qs.stats.win_rate(returns)
    avg_return = qs.stats.avg_return(returns)

    print(f"  年化收益率 (CAGR):    {cagr:>10.2%}")
    print(f"  夏普比率 (Sharpe):     {sharpe:>10.2f}")
    print(f"  索提诺比率 (Sortino):  {sortino:>10.2f}")
    print(f"  卡玛比率 (Calmar):     {calmar:>10.2f}")
    print(f"  最大回撤 (Max DD):     {max_dd:>10.2%}")
    print(f"  年化波动率 (Vol):       {volatility:>10.2%}")
    print(f"  胜率 (Win Rate):       {win_rate:>10.2%}")
    print(f"  日均收益率:            {avg_return:>10.4%}")

    # 月度收益表
    print("\n" + "-" * 60)
    print("              月 度 收 益 率 (%)")
    print("-" * 60)
    monthly = qs.stats.monthly_returns(returns)
    if isinstance(monthly, pd.DataFrame):
        print((monthly * 100).round(2).to_string())

    # 最差 5 天
    print("\n" + "-" * 60)
    print("              最 差 5 天")
    print("-" * 60)
    for d, r in returns.nsmallest(5).items():
        print(f"  {d.date()}  {r:+.4%}")

    # 回撤最深的 5 个时期
    print(f"\n  回撤最深的5个时期:")
    dd = qs.stats.to_drawdown_series(returns)
    dd_periods = _find_drawdown_periods(dd)
    dd_periods.sort(key=lambda x: x[2])
    for start, end, depth in dd_periods[:5]:
        days = (end - start).days
        print(f"  {start.date()} ~ {end.date()}  ({days}天) 最深回撤 {depth:.2%}")

    # 年度收益
    print("\n" + "-" * 60)
    print("              年 度 收 益")
    print("-" * 60)
    yearly = returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    for y, r in yearly.items():
        print(f"  {y.year}  {r:+.2%}")


def _find_drawdown_periods(dd_series):
    """从每日回撤序列中提取回撤期间"""
    periods = []
    in_dd = False
    start = None
    for d, v in dd_series.items():
        if v < 0 and not in_dd:
            start = d
            in_dd = True
        if v == 0 and in_dd:
            if start is not None:
                periods.append((start, d, dd_series.loc[start:d].min()))
            in_dd = False
            start = None
    if in_dd and start is not None:
        periods.append((start, dd_series.index[-1], dd_series.loc[start:].min()))
    return periods


def generate_report(pkl_path, output_html=None, title=None, show_terminal=True):
    """
    主函数：加载 pkl，打印终端摘要，生成 HTML 报告。

    参数:
        pkl_path:       pkl 文件路径
        output_html:    输出 HTML 路径，默认与 pkl 同名加 _qs_report.html
        title:          报告标题，默认自动从 pkl 提取策略名
        show_terminal:  是否打印终端摘要
    """
    pkl_path = Path(pkl_path)
    print(f"加载 {pkl_path} ...")
    result = load_pkl(pkl_path)

    # 提取收益率
    portfolio = result["portfolio"]
    returns = get_returns(portfolio)

    # 基准收益率
    bench_rets = None
    if "benchmark_portfolio" in result:
        bench_rets = get_returns(result["benchmark_portfolio"])
        common = returns.index.intersection(bench_rets.index)
        returns, bench_rets = returns.loc[common], bench_rets.loc[common]
        print(f"基准数据已加载，共 {len(common)} 个交易日")

    strategy_name = title or result.get("summary", {}).get("strategy_name", pkl_path.stem)
    print(f"策略: {strategy_name}")
    print(f"交易日: {returns.index[0].date()} ~ {returns.index[-1].date()}, 共 {len(returns)} 天")

    if show_terminal:
        terminal_summary(returns, bench_rets)

    # HTML 报告
    output = output_html or str(pkl_path.with_suffix(".qs_report.html"))
    print(f"\n生成 HTML 报告 → {output}")
    qs.reports.html(
        returns=returns,
        benchmark=bench_rets,
        output=output,
        title=strategy_name,
    )
    print("完成! 用浏览器打开即可查看。")


# ============================================================
#  配 置 区 - 修改这里的参数即可
# ============================================================
CONFIG = {
    "pkl_path":       "batch_results/lihai_pool/dynamic_pool.pkl",   # pkl 文件路径
    "output_html":    None,                            # 输出 HTML 路径，None=自动生成
    "title":          None,                            # 报告标题，None=自动提取
    "show_terminal":  True,                            # 是否打印终端摘要
}

if __name__ == "__main__":
    generate_report(**CONFIG)
