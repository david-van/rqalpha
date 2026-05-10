#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc : 多实验回测结果对比分析工具
# @Usage: python analysis.py
"""
读取 batch_results 目录下所有 .pkl 文件，生成：
  1. 终端打印：绩效总览表、逐年收益表、逐年夏普表、逐年回撤表
  2. analysis_summary.png    — 2×3 综合图（净值、回撤、换手、逐年收益、交易次数、Sharpe）
  3. analysis_curves.png     — 净值+回撤+滚动1年收益
  4. analysis_sensitivity.png — 参数敏感性曲线
  5. analysis_trades.png     — 交易行为分析
  6. analysis_yearly.png     — 逐年收益热力图
  7. analysis_risk_return.png — 风险收益散点图
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtic

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ============================================================
# 配置
# ============================================================
RESULT_DIR = Path(__file__).with_name("batch_results")
# todo 配置自己需要分析的详细目录
RESULT_DIR = RESULT_DIR.joinpath("mdays_switch_threshold")
TRADING_DAYS_PER_YEAR = 250


# ============================================================
# 数据加载
# ============================================================
def load_all_results(result_dir: Path) -> dict:
    """读取所有 .pkl 文件，返回 {tag: {summary, portfolio, trades}}"""
    results = {}
    for pkl_path in sorted(result_dir.glob("*.pkl")):
        tag = pkl_path.stem
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        results[tag] = {
            "summary": data.get("summary", {}),
            "portfolio": data.get("portfolio", pd.DataFrame()),
            "trades": data.get("trades", pd.DataFrame()),
        }
        print(f"已加载: {tag}  ({len(results[tag]['trades'])} 笔交易)")
    return results


# ============================================================
# 指标提取
# ============================================================
def extract_summary_table(results: dict) -> pd.DataFrame:
    """从 summary 提取核心指标"""
    rows = []
    for tag, data in results.items():
        s = data["summary"]
        rows.append(
            {
                "实验": tag,
                "累计收益": f"{s.get('total_returns', 0):.2%}",
                "年化收益": f"{s.get('annualized_returns', 0):.2%}",
                "夏普比率": f"{s.get('sharpe', 0):.3f}",
                "索提诺": f"{s.get('sortino', 0):.3f}",
                "最大回撤": f"{s.get('max_drawdown', 0):.2%}",
                "回撤天数": s.get("max_drawdown_duration_days", 0),
                "胜率": f"{s.get('win_rate', 0):.2%}",
                "盈亏比": f"{s.get('profit_loss_rate', 0):.2f}",
                "换手率": f"{s.get('turnover', 0):.1f}",
                "交易次数": len(data["trades"]),
                "年化收益(数值)": s.get("annualized_returns", 0),
                "最大回撤(数值)": s.get("max_drawdown", 0),
            }
        )
    df = pd.DataFrame(rows)
    if "年化收益(数值)" in df.columns:
        df = df.sort_values("年化收益(数值)", ascending=False)
    # 删掉辅助列
    df = df.drop(columns=["年化收益(数值)", "最大回撤(数值)"], errors="ignore")
    return df


def compute_yearly_metrics(portfolio: pd.DataFrame) -> pd.DataFrame:
    """从日频 portfolio 计算逐年指标"""
    if portfolio.empty or "unit_net_value" not in portfolio.columns:
        return pd.DataFrame()
    nv = portfolio["unit_net_value"]
    # 确保 index 是 datetime
    if not isinstance(nv.index, pd.DatetimeIndex):
        nv.index = pd.to_datetime(nv.index)
    yearly = nv.resample("YE").agg(["first", "last"])
    yearly["return"] = yearly["last"] / yearly["first"] - 1

    # 逐年最大回撤
    drawdowns = {}
    for year, group in nv.groupby(nv.index.year):
        peak = group.cummax()
        dd = group / peak - 1
        drawdowns[year] = dd.min()
    result = pd.DataFrame(
        {
            "year": yearly.index.year,
            "return": yearly["return"].values,
            "max_drawdown": drawdowns.values(),
        }
    )
    result = result.set_index("year")
    return result


def compute_yearly_sharpe(portfolio: pd.DataFrame) -> dict:
    """从日收益率计算逐年夏普"""
    if portfolio.empty or "unit_net_value" not in portfolio.columns:
        return {}
    nv = portfolio["unit_net_value"]
    if not isinstance(nv.index, pd.DatetimeIndex):
        nv.index = pd.to_datetime(nv.index)
    daily_ret = nv.pct_change().dropna()
    yearly_sharpe = {}
    for year, group in daily_ret.groupby(daily_ret.index.year):
        if len(group) > 1:
            mean = group.mean()
            std = group.std()
            yearly_sharpe[year] = (mean / std) * np.sqrt(TRADING_DAYS_PER_YEAR) if std > 0 else 0
        else:
            yearly_sharpe[year] = 0
    return yearly_sharpe


def compute_drawdown_series(nv: pd.Series) -> pd.Series:
    """计算回撤序列"""
    peak = nv.cummax()
    return nv / peak - 1


# ============================================================
# 图表 1: 综合面板 (2×3)
# ============================================================
def plot_summary_panel(results: dict, save_path: Path):
    """净值曲线 + 回撤 + 换手 + 逐年收益 + 交易次数 + Sharpe"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # --- 1.1 净值曲线 ---
    ax = axes[0, 0]
    for (tag, data), c in zip(results.items(), colors):
        pf = data["portfolio"]
        if pf.empty:
            continue
        nv = pf["unit_net_value"]
        if not isinstance(nv.index, pd.DatetimeIndex):
            nv.index = pd.to_datetime(nv.index)
        ax.plot(nv.index, nv.values, label=tag, color=c, linewidth=0.8)
    # 基准
    first_pf = list(results.values())[0]["portfolio"]
    if not first_pf.empty and "benchmark_unit_net_value" in first_pf.columns:
        bm = first_pf["benchmark_unit_net_value"]
        if not isinstance(bm.index, pd.DatetimeIndex):
            bm.index = pd.to_datetime(bm.index)
        ax.plot(bm.index, bm.values, label="沪深300", color="gray", linewidth=0.6, linestyle="--")
    ax.set_title("净值曲线")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    # --- 1.2 回撤曲线 ---
    ax = axes[0, 1]
    for (tag, data), c in zip(results.items(), colors):
        pf = data["portfolio"]
        if pf.empty:
            continue
        nv = pf["unit_net_value"]
        if not isinstance(nv.index, pd.DatetimeIndex):
            nv.index = pd.to_datetime(nv.index)
        dd = compute_drawdown_series(nv)
        ax.fill_between(dd.index, 0, dd.values, alpha=0.3, color=c, label=tag, linewidth=0.5)
    ax.set_title("回撤曲线")
    ax.yaxis.set_major_formatter(mtic.PercentFormatter(1.0))
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- 1.3 换手率柱状图 ---
    ax = axes[0, 2]
    tags = []
    turnovers = []
    trade_counts = []
    for tag, data in results.items():
        tags.append(tag)
        turnovers.append(data["summary"].get("turnover", 0))
        trade_counts.append(len(data["trades"]))
    x = np.arange(len(tags))
    bars = ax.bar(x, turnovers, color=colors[: len(tags)], edgecolor="white")
    ax.set_title("换手率")
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=8)
    for bar, v in zip(bars, turnovers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{v:.1f}",
                ha="center", fontsize=7)

    # --- 1.4 逐年收益分组柱状图 ---
    ax = axes[1, 0]
    plot_yearly_bar(ax, results, colors)

    # --- 1.5 交易次数 ---
    ax = axes[1, 1]
    bars = ax.bar(x, trade_counts, color=colors[: len(tags)], edgecolor="white")
    ax.set_title("交易总次数")
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=8)
    for bar, v in zip(bars, trade_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(v),
                ha="center", fontsize=8)

    # --- 1.6 夏普比率 ---
    ax = axes[1, 2]
    sharpes = [data["summary"].get("sharpe", 0) for data in results.values()]
    bars = ax.bar(x, sharpes, color=colors[: len(tags)], edgecolor="white")
    ax.set_title("夏普比率")
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=8)
    for bar, v in zip(bars, sharpes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{v:.3f}",
                ha="center", fontsize=7)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {save_path}")


def plot_yearly_bar(ax, results: dict, colors):
    """逐年收益分组柱状图"""
    all_yearly = {}
    years_set = set()
    for tag, data in results.items():
        ym = compute_yearly_metrics(data["portfolio"])
        if not ym.empty:
            all_yearly[tag] = ym
            years_set.update(ym.index.tolist())
    if not years_set:
        ax.set_title("逐年收益 (无数据)")
        return
    years = sorted(years_set)
    tags = list(all_yearly.keys())
    n_tags = len(tags)
    x = np.arange(len(years))
    width = 0.8 / max(n_tags, 1)

    for i, (tag, ym) in enumerate(all_yearly.items()):
        vals = [ym.loc[y, "return"] if y in ym.index else 0 for y in years]
        offset = (i - n_tags / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=tag, color=colors[i], edgecolor="white")

    ax.set_title("逐年收益")
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=0, fontsize=8)
    ax.yaxis.set_major_formatter(mtic.PercentFormatter(1.0))
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="black", linewidth=0.5)


# ============================================================
# 图表 2: 净值 + 回撤 + 滚动收益
# ============================================================
def plot_curves_panel(results: dict, save_path: Path):
    """三行子图：净值曲线、回撤曲线、滚动1年收益"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (tag, data), c in zip(results.items(), colors):
        pf = data["portfolio"]
        if pf.empty:
            continue
        nv = pf["unit_net_value"]
        if not isinstance(nv.index, pd.DatetimeIndex):
            nv.index = pd.to_datetime(nv.index)

        # 净值
        axes[0].plot(nv.index, nv.values, label=tag, color=c, linewidth=0.8)
        # 回撤
        dd = compute_drawdown_series(nv)
        axes[1].plot(dd.index, dd.values, color=c, linewidth=0.6, alpha=0.8)
        # 滚动1年收益
        rolling = nv.pct_change().rolling(TRADING_DAYS_PER_YEAR).apply(
            lambda x: (1 + x).prod() - 1, raw=False
        )
        axes[2].plot(rolling.index, rolling.values, label=tag, color=c, linewidth=0.8)

    # 基准净值
    first_pf = list(results.values())[0]["portfolio"]
    if not first_pf.empty and "benchmark_unit_net_value" in first_pf.columns:
        bm = first_pf["benchmark_unit_net_value"]
        if not isinstance(bm.index, pd.DatetimeIndex):
            bm.index = pd.to_datetime(bm.index)
        axes[0].plot(bm.index, bm.values, label="沪深300", color="gray",
                     linewidth=0.6, linestyle="--")

    axes[0].set_title("净值曲线")
    axes[0].legend(fontsize=7, loc="upper left")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("回撤曲线")
    axes[1].yaxis.set_major_formatter(mtic.PercentFormatter(1.0))
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color="black", linewidth=0.5)

    axes[2].set_title(f"滚动 {TRADING_DAYS_PER_YEAR} 日收益")
    axes[2].yaxis.set_major_formatter(mtic.PercentFormatter(1.0))
    axes[2].legend(fontsize=7, ncol=3)
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {save_path}")


# ============================================================
# 图表 3: 参数敏感性
# ============================================================
def plot_sensitivity(results: dict, save_path: Path):
    """X轴=阈值参数值，Y轴多条线分别为收益、夏普、回撤、换手率"""
    # 尝试从 tag 中提取 threshold 值
    pairs = []
    for tag, data in results.items():
        th = _extract_threshold(tag)
        if th is not None:
            pairs.append((th, tag, data))

    if len(pairs) < 2:
        print("参数敏感性: 至少需要 2 个实验（可从 tag 名中提取阈值参数值）")
        return

    pairs.sort(key=lambda x: x[0])
    thresholds = [p[0] for p in pairs]
    tags = [p[1] for p in pairs]
    metrics = {
        "年化收益": [p[2]["summary"].get("annualized_returns", 0) for p in pairs],
        "夏普比率": [p[2]["summary"].get("sharpe", 0) for p in pairs],
        "最大回撤": [p[2]["summary"].get("max_drawdown", 0) for p in pairs],
        "换手率": [p[2]["summary"].get("turnover", 0) for p in pairs],
        "索提诺": [p[2]["summary"].get("sortino", 0) for p in pairs],
    }

    # 自动判断扫描的是哪个参数，用于图表标题和 X 轴标签
    has_mdays = any("mdays" in p[1] for p in pairs)
    param_label = "m_days (回望天数)" if has_mdays else "阈值系数 (switch_threshold)"

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()

    for idx, (name, vals) in enumerate(metrics.items()):
        ax = axes_flat[idx]
        ax.plot(thresholds, vals, marker="o", linewidth=1.5, markersize=8)
        for th, v, t in zip(thresholds, vals, tags):
            ax.annotate(t, (th, v), textcoords="offset points", xytext=(0, 10),
                        fontsize=7, ha="center")
        ax.set_title(name)
        ax.set_xlabel(param_label)
        ax.grid(True, alpha=0.3)
        if name in ("最大回撤",):
            ax.yaxis.set_major_formatter(mtic.PercentFormatter(1.0))
        if name in ("年化收益",):
            ax.yaxis.set_major_formatter(mtic.PercentFormatter(1.0))

    # 第6个子图：参数值 vs 交易次数
    ax = axes_flat[5]
    trade_counts = [len(p[2]["trades"]) for p in pairs]
    ax.plot(thresholds, trade_counts, marker="o", color="darkred", linewidth=1.5, markersize=8)
    for th, v in zip(thresholds, trade_counts):
        ax.annotate(str(v), (th, v), textcoords="offset points", xytext=(0, 10),
                    fontsize=7, ha="center")
    ax.set_title("交易总次数")
    ax.set_xlabel(param_label)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"参数敏感性分析 — {param_label}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {save_path}")


def _parse_tag(tag: str) -> dict:
    """从 tag 名中解析参数值，返回 {'m_days': int, 'threshold': float}。

    支持的 tag 格式（来自 run.py 的 _make_tag）：
      baseline          → m_days=25, threshold=1.0
      m28               → m_days=28, threshold=1.0
      t110              → m_days=25, threshold=1.10
      m28_t110          → m_days=28, threshold=1.10
      mdays_28          → m_days=28, threshold=1.0   (旧格式兼容)
      switch_threshold_10 → m_days=25, threshold=1.10 (旧格式兼容)
    """
    import re
    m_days = 25       # 默认值
    threshold = 1.0   # 默认值

    # 新格式：m{DD} 或 m{DD}_t{TTT}
    # 用 (?<![a-z]) 和 (?!\d) 代替 \b，避免 _ 被当作单词字符导致边界失效
    m = re.search(r"(?<![a-z])m(\d{2})(?!\d)", tag)
    if m:
        m_days = int(m.group(1))

    # 新格式：t{TTT}（三位，如 t115 → 1.15）
    # 匹配 _t{TTT} 或 行首t{TTT}，避免 mdays 里的 d 被误匹配
    m = re.search(r"(?:^|_)t(\d{3})(?!\d)", tag)
    if m:
        threshold = int(m.group(1)) / 100.0

    # 旧格式兼容：mdays_XX
    m = re.search(r"mdays[_](\d+)", tag)
    if m:
        m_days = int(m.group(1))

    # 旧格式兼容：switch_threshold_XX
    m = re.search(r"switch_threshold[_]?(\d+)", tag)
    if m:
        digits = m.group(1)
        threshold = float(digits) / 100 + 1.0 if len(digits) <= 2 else float(digits) / 100

    return {"m_days": m_days, "threshold": threshold}


def _extract_threshold(tag: str):
    """单维敏感性图用：返回该 tag 的"主参数值"（用于 X 轴）。
    二维场景（tag 同时含 m 和 t）返回 None，由 plot_grid_heatmap 处理。
    """
    p = _parse_tag(tag)
    import re
    has_m = bool(re.search(r"\bm\d{2}\b", tag) or re.search(r"mdays[_]\d+", tag))
    has_t = bool(re.search(r"\bt\d{3}\b", tag) or re.search(r"switch_threshold", tag))

    if has_m and has_t:
        return None   # 二维点，不参与单维敏感性图
    if has_m or tag.lower() == "baseline":
        return p["m_days"]
    if has_t:
        return p["threshold"]
    if tag.lower() == "baseline":
        return p["m_days"]
    return None


# ============================================================
# 图表 4: 交易行为分析
# ============================================================
def plot_trade_analysis(results: dict, save_path: Path):
    """持仓时长分布 + 年度交易次数 + 交易品种切换"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # --- 4.1 年度交易次数 ---
    ax = axes[0, 0]
    all_yearly_trades = {}
    years_set = set()
    for tag, data in results.items():
        trades = data["trades"]
        if trades.empty:
            continue
        if not isinstance(trades.index, pd.DatetimeIndex):
            trades.index = pd.to_datetime(trades.index)
        if "datetime" in trades.columns and not isinstance(trades.index, pd.DatetimeIndex):
            pass  # index already used above
        yearly = trades.groupby(trades.index.year).size()
        all_yearly_trades[tag] = yearly
        years_set.update(yearly.index.tolist())

    if years_set:
        years = sorted(years_set)
        x = np.arange(len(years))
        width = 0.8 / max(len(all_yearly_trades), 1)
        for i, (tag, counts) in enumerate(all_yearly_trades.items()):
            vals = [counts.loc[y] if y in counts.index else 0 for y in years]
            offset = (i - len(all_yearly_trades) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=tag, color=colors[i], edgecolor="white")
        ax.set_title("年度交易次数")
        ax.set_xticks(x)
        ax.set_xticklabels(years, fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    # --- 4.2 换手率 vs 收益散点 ---
    ax = axes[0, 1]
    for (tag, data), c in zip(results.items(), colors):
        ann = data["summary"].get("annualized_returns", 0)
        turnover = data["summary"].get("turnover", 0)
        ax.scatter(turnover, ann, c=[c], s=100, label=tag, edgecolors="black", linewidth=0.5)
        ax.annotate(tag, (turnover, ann), textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xlabel("换手率")
    ax.set_ylabel("年化收益")
    ax.set_title("换手率 vs 年化收益")
    ax.yaxis.set_major_formatter(mtic.PercentFormatter(1.0))
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- 4.3 持仓时长分布 (箱线图) ---
    ax = axes[1, 0]
    holding_data = []
    labels = []
    for tag, data in results.items():
        trades = data["trades"]
        if trades.empty or len(trades) < 2:
            continue
        if not isinstance(trades.index, pd.DatetimeIndex):
            trades.index = pd.to_datetime(trades.index)
        # 按品种分组，计算连续持有天数
        durations = _compute_holding_durations(trades)
        if durations:
            holding_data.append(durations)
            labels.append(tag)
    if holding_data:
        bp = ax.boxplot(holding_data, labels=labels, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], colors[: len(labels)]):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_title("持仓天数分布")
        ax.set_ylabel("持仓天数")
        ax.grid(True, alpha=0.3, axis="y")

    # --- 4.4 累计交易成本 ---
    ax = axes[1, 1]
    for (tag, data), c in zip(results.items(), colors):
        trades = data["trades"]
        if trades.empty:
            continue
        cost_cols = [col for col in ["commission", "tax", "transaction_cost"] if col in trades.columns]
        if not cost_cols:
            continue
        # 使用第一个存在的成本列
        cost_col = cost_cols[0]
        if not isinstance(trades.index, pd.DatetimeIndex):
            trades.index = pd.to_datetime(trades.index)
        cumulative = trades[cost_col].cumsum()
        ax.plot(trades.index, cumulative.values, label=tag, color=c, linewidth=0.8)
    ax.set_title("累计交易成本")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {save_path}")


def _compute_holding_durations(trades: pd.DataFrame) -> list:
    """从交易记录中估算持仓天数列表"""
    if trades.empty:
        return []
    durations = []
    # 按 order_book_id 分组，找到买入→卖出配对
    for symbol, group in trades.groupby("order_book_id"):
        group = group.sort_index()
        buys = group[group["last_quantity"] > 0]
        sells = group[group["last_quantity"] < 0]
        # 简单配对：相邻买卖
        buy_times = list(buys.index)
        sell_times = list(sells.index)
        for bt in buy_times:
            # 找到最近的卖出
            later_sells = [st for st in sell_times if st > bt]
            if later_sells:
                days = (min(later_sells) - bt).days
                durations.append(days)
    return durations


# ============================================================
# 图表 5: 逐年收益热力图
# ============================================================
def plot_yearly_heatmap(results: dict, save_path: Path):
    """每年每个实验的收益率热力图"""
    all_yearly = {}
    years_set = set()
    for tag, data in results.items():
        ym = compute_yearly_metrics(data["portfolio"])
        if not ym.empty:
            all_yearly[tag] = ym["return"]
            years_set.update(ym.index.tolist())

    if not years_set:
        print("逐年热力图: 无数据")
        return

    years = sorted(years_set)
    tags = list(all_yearly.keys())
    matrix = np.zeros((len(tags), len(years)))
    for i, tag in enumerate(tags):
        for j, year in enumerate(years):
            matrix[i, j] = all_yearly[tag].get(year, np.nan)

    fig, ax = plt.subplots(figsize=(max(10, len(years) * 0.8), max(4, len(tags) * 0.5)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-0.4, vmax=0.8)

    # 标注
    for i in range(len(tags)):
        for j in range(len(years)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1%}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if abs(val) > 0.3 else "black")

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, fontsize=9)
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels(tags, fontsize=9)
    ax.set_title("逐年收益热力图", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, format=mtic.PercentFormatter(1.0), shrink=0.8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {save_path}")


# ============================================================
# 图表 6: 风险收益散点图
# ============================================================
def plot_risk_return(results: dict, save_path: Path):
    """X=最大回撤, Y=年化收益 的散点图"""
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (tag, data), c in zip(results.items(), colors):
        s = data["summary"]
        ann = s.get("annualized_returns", 0)
        mdd = s.get("max_drawdown", 0)
        sharpe = s.get("sharpe", 0)
        ax.scatter(mdd, ann, c=[c], s=200 + sharpe * 100, label=tag,
                   edgecolors="black", linewidth=0.8, alpha=0.85)
        ax.annotate(f"{tag}\n(Sharpe:{sharpe:.3f})",
                    (mdd, ann), textcoords="offset points", xytext=(8, 8), fontsize=8)

    ax.set_xlabel("最大回撤")
    ax.set_ylabel("年化收益")
    ax.set_title("风险收益散点图 (气泡大小=夏普比率)")
    ax.xaxis.set_major_formatter(mtic.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mtic.PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axvline(x=0, color="black", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {save_path}")


# ============================================================
# 图表 7: 二维参数网格热力图（m_days × switch_threshold）
# ============================================================
def plot_grid_heatmap(results: dict, save_path: Path):
    """当实验是二维网格时，画 m_days(行) × switch_threshold(列) 的热力图。
    每个格子显示夏普比率，颜色深浅表示高低。
    同时输出年化收益和最大回撤的对应热力图（共3张子图）。
    若检测到不是二维数据则跳过。
    """
    import re

    # 解析所有 tag
    parsed = {tag: _parse_tag(tag) for tag in results}
    m_days_vals   = sorted(set(p["m_days"]    for p in parsed.values()))
    threshold_vals = sorted(set(p["threshold"] for p in parsed.values()))

    # 至少要有 2×2 才算二维
    if len(m_days_vals) < 2 or len(threshold_vals) < 2:
        print("二维网格热力图: 数据不是二维网格，跳过")
        return

    def build_matrix(metric_key):
        mat = np.full((len(m_days_vals), len(threshold_vals)), np.nan)
        for tag, data in results.items():
            p = parsed[tag]
            i = m_days_vals.index(p["m_days"])
            j = threshold_vals.index(p["threshold"])
            mat[i, j] = data["summary"].get(metric_key, np.nan)
        return mat

    metrics = [
        ("sharpe",             "夏普比率",  "YlGn",   None,  None),
        ("annualized_returns", "年化收益",  "RdYlGn", -0.1,  0.5),
        ("max_drawdown",       "最大回撤",  "RdYlGn_r", -0.5, -0.1),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, max(4, len(m_days_vals) * 0.7 + 2)))

    for ax, (key, title, cmap, vmin, vmax) in zip(axes, metrics):
        mat = build_matrix(key)
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.8,
                     format=mtic.PercentFormatter(1.0) if key != "sharpe" else None)

        # 标注数值
        for i in range(len(m_days_vals)):
            for j in range(len(threshold_vals)):
                v = mat[i, j]
                if not np.isnan(v):
                    txt = f"{v:.3f}" if key == "sharpe" else f"{v:.1%}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8, fontweight="bold",
                            color="white" if (key == "sharpe" and v > mat[~np.isnan(mat)].mean()) else "black")

        ax.set_xticks(range(len(threshold_vals)))
        ax.set_xticklabels([f"{t:.2f}" for t in threshold_vals], fontsize=9)
        ax.set_yticks(range(len(m_days_vals)))
        ax.set_yticklabels([str(d) for d in m_days_vals], fontsize=9)
        ax.set_xlabel("switch_threshold", fontsize=10)
        ax.set_ylabel("m_days", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")

        # 标出最优格子（夏普最高 / 年化最高 / 回撤最小）
        if key == "sharpe":
            best = np.unravel_index(np.nanargmax(mat), mat.shape)
        elif key == "annualized_returns":
            best = np.unravel_index(np.nanargmax(mat), mat.shape)
        else:  # max_drawdown：值越大（越接近0）越好
            best = np.unravel_index(np.nanargmax(mat), mat.shape)
        ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1,
                                   fill=False, edgecolor="red", linewidth=2.5))

    fig.suptitle("二维参数网格热力图  (红框=最优)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {save_path}")


# ============================================================
# 终端：二维网格汇总表
# ============================================================
def print_grid_table(results: dict):
    """打印 m_days × switch_threshold 的夏普/年化收益透视表"""
    parsed = {tag: _parse_tag(tag) for tag in results}
    m_days_vals    = sorted(set(p["m_days"]    for p in parsed.values()))
    threshold_vals = sorted(set(p["threshold"] for p in parsed.values()))

    if len(m_days_vals) < 2 or len(threshold_vals) < 2:
        return  # 不是二维，跳过

    print("\n" + "=" * 100)
    print("【二维网格：夏普比率透视表】  行=m_days  列=switch_threshold")
    print("=" * 100)
    rows = []
    for d in m_days_vals:
        row = {"m_days": d}
        for t in threshold_vals:
            # 找到对应 tag
            match = [tag for tag, p in parsed.items() if p["m_days"] == d and p["threshold"] == t]
            if match:
                v = results[match[0]]["summary"].get("sharpe", np.nan)
                row[f"t={t:.2f}"] = f"{v:.3f}" if not np.isnan(v) else "-"
            else:
                row[f"t={t:.2f}"] = "-"
        rows.append(row)
    print(pd.DataFrame(rows).set_index("m_days").to_string())

    print("\n" + "=" * 100)
    print("【二维网格：年化收益透视表】  行=m_days  列=switch_threshold")
    print("=" * 100)
    rows = []
    for d in m_days_vals:
        row = {"m_days": d}
        for t in threshold_vals:
            match = [tag for tag, p in parsed.items() if p["m_days"] == d and p["threshold"] == t]
            if match:
                v = results[match[0]]["summary"].get("annualized_returns", np.nan)
                row[f"t={t:.2f}"] = f"{v:.2%}" if not np.isnan(v) else "-"
            else:
                row[f"t={t:.2f}"] = "-"
        rows.append(row)
    print(pd.DataFrame(rows).set_index("m_days").to_string())
    print("=" * 100 + "\n")
def print_terminal_tables(results: dict):
    """打印所有表格到终端"""
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    # --- 绩效总览 ---
    print("\n" + "=" * 100)
    print("【绩效总览表】")
    print("=" * 100)
    df_summary = extract_summary_table(results)
    print(df_summary.to_string(index=False))

    # --- 逐年收益 ---
    print("\n" + "=" * 100)
    print("【逐年收益表】")
    print("=" * 100)
    yearly_data = {}
    all_years = set()
    for tag, data in results.items():
        ym = compute_yearly_metrics(data["portfolio"])
        if not ym.empty:
            yearly_data[tag] = ym["return"]
            all_years.update(ym.index.tolist())
    if yearly_data:
        df_yearly = pd.DataFrame(yearly_data).T
        df_yearly.columns = [str(int(c)) for c in df_yearly.columns]
        df_yearly = df_yearly[sorted(df_yearly.columns)]
        # 格式化
        for col in df_yearly.columns:
            df_yearly[col] = df_yearly[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
        print(df_yearly.to_string())

    # --- 逐年夏普 ---
    print("\n" + "=" * 100)
    print("【逐年夏普比率表】")
    print("=" * 100)
    sharpe_data = {}
    all_years.clear()
    for tag, data in results.items():
        ys = compute_yearly_sharpe(data["portfolio"])
        if ys:
            sharpe_data[tag] = ys
            all_years.update(ys.keys())
    if sharpe_data:
        df_sharpe = pd.DataFrame(sharpe_data).T
        df_sharpe.columns = [str(int(c)) for c in df_sharpe.columns]
        df_sharpe = df_sharpe[sorted(df_sharpe.columns)]
        for col in df_sharpe.columns:
            df_sharpe[col] = df_sharpe[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
        print(df_sharpe.to_string())

    # --- 逐年最大回撤 ---
    print("\n" + "=" * 100)
    print("【逐年最大回撤表】")
    print("=" * 100)
    dd_data = {}
    all_years.clear()
    for tag, data in results.items():
        ym = compute_yearly_metrics(data["portfolio"])
        if not ym.empty and "max_drawdown" in ym.columns:
            dd_data[tag] = ym["max_drawdown"]
            all_years.update(ym.index.tolist())
    if dd_data:
        df_dd = pd.DataFrame(dd_data).T
        df_dd.columns = [str(int(c)) for c in df_dd.columns]
        df_dd = df_dd[sorted(df_dd.columns)]
        for col in df_dd.columns:
            df_dd[col] = df_dd[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
        print(df_dd.to_string())

    # --- 实验参数提取 ---
    print("\n" + "=" * 100)
    print("【参数敏感性速览】")
    print("=" * 100)
    for tag, data in results.items():
        th = _extract_threshold(tag)
        if th is not None:
            s = data["summary"]
            print(f"  threshold={th:.2f}  |  年化:{s.get('annualized_returns',0):.2%}  |  "
                  f"夏普:{s.get('sharpe',0):.3f}  |  回撤:{s.get('max_drawdown',0):.2%}  |  "
                  f"换手:{s.get('turnover',0):.1f}  |  交易:{len(data['trades'])}笔")
    print("=" * 100 + "\n")


# ============================================================
# 主入口
# ============================================================
def main():
    if not RESULT_DIR.exists():
        print(f"目录不存在: {RESULT_DIR}")
        return

    results = load_all_results(RESULT_DIR)
    if not results:
        print("未找到任何 .pkl 文件")
        return

    # 终端表格
    print_terminal_tables(results)
    print_grid_table(results)          # 二维网格时额外打印透视表

    # 图表
    print("\n生成图表...")
    plot_summary_panel(results, RESULT_DIR / "analysis_summary.png")
    plot_curves_panel(results, RESULT_DIR / "analysis_curves.png")
    plot_sensitivity(results, RESULT_DIR / "analysis_sensitivity.png")
    plot_trade_analysis(results, RESULT_DIR / "analysis_trades.png")
    plot_yearly_heatmap(results, RESULT_DIR / "analysis_yearly.png")
    plot_risk_return(results, RESULT_DIR / "analysis_risk_return.png")
    plot_grid_heatmap(results, RESULT_DIR / "analysis_grid_heatmap.png")  # 二维网格专属

    print("\n全部完成!")


if __name__ == "__main__":
    main()
