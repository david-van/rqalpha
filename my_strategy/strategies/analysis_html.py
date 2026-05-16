#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc : 多实验回测结果对比分析 — 交互式 HTML 仪表盘
# @Usage: python analysis_html.py
"""
读取 batch_results 目录下所有 .pkl 文件，生成单个交互式 HTML 仪表盘。
包含 6 个 Tab 页签：概览、曲线对比、参数敏感性、交易分析、年度分析、风险收益。
"""

import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# 数据加载 & 指标计算
# ============================================================
TRADING_DAYS_PER_YEAR = 250


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


def extract_summary_table(results: dict) -> pd.DataFrame:
    """从 summary 提取核心指标"""
    rows = []
    for tag, data in results.items():
        s = data["summary"]
        rows.append({
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
        })
    df = pd.DataFrame(rows)
    if "年化收益(数值)" in df.columns:
        df = df.sort_values("年化收益(数值)", ascending=False)
    df = df.drop(columns=["年化收益(数值)", "最大回撤(数值)"], errors="ignore")
    return df


def compute_yearly_metrics(portfolio: pd.DataFrame) -> pd.DataFrame:
    """从日频 portfolio 计算逐年指标"""
    if portfolio.empty or "unit_net_value" not in portfolio.columns:
        return pd.DataFrame()
    nv = portfolio["unit_net_value"]
    if not isinstance(nv.index, pd.DatetimeIndex):
        nv.index = pd.to_datetime(nv.index)
    yearly: pd.DataFrame = nv.resample("YE").agg(["first", "last"])
    yearly["return"] = yearly["last"] / yearly["first"] - 1

    drawdowns = {}
    for year, group in nv.groupby(nv.index.year):
        peak = group.cummax()
        dd = group / peak - 1
        drawdowns[year] = dd.min()
    result = pd.DataFrame({
        "year": yearly.index.year,
        "return": yearly["return"].values,
        "max_drawdown": drawdowns.values(),
    })
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


def _compute_holding_durations(trades: pd.DataFrame) -> list:
    """从交易记录中估算持仓天数列表"""
    if trades.empty:
        return []
    durations = []
    for symbol, group in trades.groupby("order_book_id"):
        group = group.sort_index()
        buys = group[group["last_quantity"] > 0]
        sells = group[group["last_quantity"] < 0]
        buy_times = list(buys.index)
        sell_times = list(sells.index)
        for bt in buy_times:
            later_sells = [st for st in sell_times if st > bt]
            if later_sells:
                days = (min(later_sells) - bt).days
                durations.append(days)
    return durations


def _parse_tag(tag: str) -> dict:
    """从 tag 名中解析参数值，返回 {'m_days': int, 'threshold': float, 'decay_ratio': float}"""
    import re
    m_days = 25
    threshold = 1.0
    decay_ratio = 1.0

    m = re.search(r"(?<![a-z])m(\d{2})(?!\d)", tag)
    if m:
        m_days = int(m.group(1))

    m = re.search(r"(?:^|_)t(\d{3})(?!\d)", tag)
    if m:
        threshold = int(m.group(1)) / 100.0

    m = re.search(r"(?:^|_)r(\d{2})(?!\d)", tag)
    if m:
        decay_ratio = int(m.group(1)) / 10.0

    m = re.search(r"mdays[_](\d+)", tag)
    if m:
        m_days = int(m.group(1))

    m = re.search(r"switch_threshold[_]?(\d+)", tag)
    if m:
        digits = m.group(1)
        threshold = float(digits) / 100 + 1.0 if len(digits) <= 2 else float(digits) / 100

    return {"m_days": m_days, "threshold": threshold, "decay_ratio": decay_ratio}


def _extract_threshold(tag: str, scan_param: str = None):
    """单维敏感性图用：返回该 tag 的主参数值（用于 X 轴）"""
    import re
    p = _parse_tag(tag)
    has_m = bool(re.search(r"(?<![a-z])m\d{2}(?!\d)", tag) or re.search(r"mdays[_]\d+", tag))
    has_t = bool(re.search(r"(?:^|_)t\d{3}(?!\d)", tag) or re.search(r"switch_threshold", tag))
    has_r = bool(re.search(r"(?:^|_)r\d{2}(?!\d)", tag))

    dims = sum([has_m, has_t, has_r])
    if dims >= 2:
        return None
    if has_r:
        return p["decay_ratio"]
    if has_m:
        return p["m_days"]
    if has_t:
        return p["threshold"]
    if tag.lower() == "baseline":
        if scan_param == "r":
            return p["decay_ratio"]
        elif scan_param == "t":
            return p["threshold"]
        else:
            return p["m_days"]
    return None


def _detect_scan_param(tags: list) -> str:
    """从非 baseline 的 tag 列表中检测扫描参数类型"""
    for tag in tags:
        if tag == "baseline":
            continue
        if tag.startswith("r"):
            return "r"
        if tag.startswith("m"):
            return "m"
        if tag.startswith("t"):
            return "t"
    return "m"

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs

# ============================================================
# 配置
# ============================================================
RESULT_DIR = Path(__file__).with_name("batch_results")
RESULT_DIR = RESULT_DIR.joinpath("lihai_pool")

# 颜色方案：尽量接近原 matplotlib tab10
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _get_color(i: int) -> str:
    return COLORS[i % len(COLORS)]


# ============================================================
# Tab 1: 概览 — 2×3 综合面板
# ============================================================
def build_overview_figure(results: dict) -> go.Figure:
    tags = list(results.keys())
    n = len(tags)

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "净值曲线", "回撤曲线", "换手率",
            "逐年收益", "交易总次数", "夏普比率",
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )

    # ---- (1,1) 净值曲线 ----
    for i, (tag, data) in enumerate(results.items()):
        pf = data["portfolio"]
        if pf.empty or "unit_net_value" not in pf.columns:
            continue
        nv = pf["unit_net_value"]
        if not isinstance(nv.index, pd.DatetimeIndex):
            nv.index = pd.to_datetime(nv.index)
        fig.add_trace(
            go.Scatter(
                x=nv.index, y=nv.values, mode="lines",
                name=tag, line=dict(color=_get_color(i), width=1.2),
                legendgroup=tag, showlegend=True,
                hovertemplate=f"{tag}<br>%{{x|%Y-%m-%d}}<br>净值: %{{y:.4f}}<extra></extra>",
            ),
            row=1, col=1,
        )

    # 基准线（沪深300）
    first_pf = list(results.values())[0]["portfolio"]
    if not first_pf.empty and "benchmark_unit_net_value" in first_pf.columns:
        bm = first_pf["benchmark_unit_net_value"]
        if not isinstance(bm.index, pd.DatetimeIndex):
            bm.index = pd.to_datetime(bm.index)
        fig.add_trace(
            go.Scatter(
                x=bm.index, y=bm.values, mode="lines",
                name="沪深300", line=dict(color="gray", width=1.0, dash="dash"),
                legendgroup="benchmark", showlegend=True,
                hovertemplate=f"沪深300<br>%{{x|%Y-%m-%d}}<br>净值: %{{y:.4f}}<extra></extra>",
            ),
            row=1, col=1,
        )

    # ---- (1,2) 回撤曲线 ----
    for i, (tag, data) in enumerate(results.items()):
        pf = data["portfolio"]
        if pf.empty or "unit_net_value" not in pf.columns:
            continue
        nv = pf["unit_net_value"]
        if not isinstance(nv.index, pd.DatetimeIndex):
            nv.index = pd.to_datetime(nv.index)
        dd = compute_drawdown_series(nv)
        fig.add_trace(
            go.Scatter(
                x=dd.index, y=dd.values,
                mode="lines", fill="tozeroy",
                name=tag, line=dict(color=_get_color(i), width=0.6),
                legendgroup=tag, showlegend=False,
                hovertemplate=f"{tag}<br>%{{x|%Y-%m-%d}}<br>回撤: %{{y:.2%}}<extra></extra>",
            ),
            row=1, col=2,
        )

    # ---- (1,3) 换手率柱状图 ----
    tag_list = []
    turnover_vals = []
    for tag, data in results.items():
        tag_list.append(tag)
        turnover_vals.append(data["summary"].get("turnover", 0))
    fig.add_trace(
        go.Bar(
            x=tag_list, y=turnover_vals,
            name="换手率", marker=dict(color=[_get_color(i) for i in range(len(tag_list))]),
            text=[f"{v:.1f}" for v in turnover_vals], textposition="outside",
            showlegend=False,
            hovertemplate="%{x}<br>换手率: %{y:.1f}<extra></extra>",
        ),
        row=1, col=3,
    )

    # ---- (2,1) 逐年收益分组柱状图 ----
    all_yearly = {}
    years_set = set()
    for tag, data in results.items():
        ym = compute_yearly_metrics(data["portfolio"])
        if not ym.empty:
            all_yearly[tag] = ym
            years_set.update(ym.index.tolist())

    if years_set:
        years = sorted(years_set)
        for i, (tag, ym) in enumerate(all_yearly.items()):
            vals = [ym.loc[y, "return"] if y in ym.index else 0 for y in years]
            fig.add_trace(
                go.Bar(
                    x=[str(y) for y in years], y=vals,
                    name=tag, marker=dict(color=_get_color(i)),
                    legendgroup=tag, showlegend=False,
                    hovertemplate=f"{tag}<br>%{{x}}<br>收益: %{{y:.2%}}<extra></extra>",
                ),
                row=2, col=1,
            )
        fig.update_xaxes(tickangle=0, row=2, col=1)

    # ---- (2,2) 交易次数 ----
    trade_counts = [len(data["trades"]) for data in results.values()]
    fig.add_trace(
        go.Bar(
            x=tag_list, y=trade_counts,
            name="交易次数", marker=dict(color=[_get_color(i) for i in range(len(tag_list))]),
            text=[str(v) for v in trade_counts], textposition="outside",
            showlegend=False,
            hovertemplate="%{x}<br>交易次数: %{y}<extra></extra>",
        ),
        row=2, col=2,
    )

    # ---- (2,3) 夏普比率 ----
    sharpe_vals = [data["summary"].get("sharpe", 0) for data in results.values()]
    fig.add_trace(
        go.Bar(
            x=tag_list, y=sharpe_vals,
            name="夏普比率", marker=dict(color=[_get_color(i) for i in range(len(tag_list))]),
            text=[f"{v:.3f}" for v in sharpe_vals], textposition="outside",
            showlegend=False,
            hovertemplate="%{x}<br>夏普比率: %{y:.3f}<extra></extra>",
        ),
        row=2, col=3,
    )

    # 全局布局
    fig.update_layout(
        title=dict(text="策略回测概览", font=dict(size=18)),
        barmode="group",
        hovermode="closest",
        autosize=True,
        legend=dict(font=dict(size=10), orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5),
    )
    fig.update_yaxes(title_text="单位净值", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=1, col=2)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_xaxes(tickangle=30, row=1, col=3)
    fig.update_xaxes(tickangle=30, row=2, col=2)
    fig.update_xaxes(tickangle=30, row=2, col=3)

    return fig


# ============================================================
# Tab 2: 曲线对比 — 净值 + 回撤 + 滚动收益
# ============================================================
def build_curves_figure(results: dict) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("净值曲线", "回撤曲线", f"滚动 {TRADING_DAYS_PER_YEAR} 日收益"),
    )

    for i, (tag, data) in enumerate(results.items()):
        pf = data["portfolio"]
        if pf.empty or "unit_net_value" not in pf.columns:
            continue
        nv = pf["unit_net_value"]
        if not isinstance(nv.index, pd.DatetimeIndex):
            nv.index = pd.to_datetime(nv.index)

        # 净值
        fig.add_trace(
            go.Scatter(
                x=nv.index, y=nv.values, mode="lines",
                name=tag, line=dict(color=_get_color(i), width=1.2),
                legendgroup=tag, showlegend=True,
                hovertemplate=f"{tag}<br>%{{x|%Y-%m-%d}}<br>净值: %{{y:.4f}}<extra></extra>",
            ),
            row=1, col=1,
        )

        # 回撤
        dd = compute_drawdown_series(nv)
        fig.add_trace(
            go.Scatter(
                x=dd.index, y=dd.values, mode="lines",
                name=tag, line=dict(color=_get_color(i), width=1.0),
                legendgroup=tag, showlegend=False,
                hovertemplate=f"{tag}<br>%{{x|%Y-%m-%d}}<br>回撤: %{{y:.2%}}<extra></extra>",
            ),
            row=2, col=1,
        )

        # 滚动收益
        rolling = nv.pct_change().rolling(TRADING_DAYS_PER_YEAR).apply(
            lambda x: (1 + x).prod() - 1, raw=False
        )
        fig.add_trace(
            go.Scatter(
                x=rolling.index, y=rolling.values, mode="lines",
                name=tag, line=dict(color=_get_color(i), width=1.0),
                legendgroup=tag, showlegend=False,
                hovertemplate=f"{tag}<br>%{{x|%Y-%m-%d}}<br>滚动收益: %{{y:.2%}}<extra></extra>",
            ),
            row=3, col=1,
        )

    # 基准净值
    first_pf = list(results.values())[0]["portfolio"]
    if not first_pf.empty and "benchmark_unit_net_value" in first_pf.columns:
        bm = first_pf["benchmark_unit_net_value"]
        if not isinstance(bm.index, pd.DatetimeIndex):
            bm.index = pd.to_datetime(bm.index)
        fig.add_trace(
            go.Scatter(
                x=bm.index, y=bm.values, mode="lines",
                name="沪深300", line=dict(color="gray", width=1.0, dash="dash"),
                legendgroup="benchmark", showlegend=True,
                hovertemplate=f"沪深300<br>%{{x|%Y-%m-%d}}<br>净值: %{{y:.4f}}<extra></extra>",
            ),
            row=1, col=1,
        )

    fig.update_layout(
        title=dict(text="曲线对比分析", font=dict(size=18)),
        hovermode="closest",
        autosize=True,
        legend=dict(font=dict(size=10), orientation="h", yanchor="top", y=-0.10, xanchor="center", x=0.5),
    )
    fig.update_yaxes(title_text="单位净值", row=1, col=1)
    fig.update_yaxes(title_text="回撤", tickformat=".0%", row=2, col=1)
    fig.update_yaxes(title_text="滚动收益", tickformat=".0%", row=3, col=1)
    # 零线
    fig.add_hline(y=0, line=dict(color="black", width=0.5), row=2, col=1)
    fig.add_hline(y=0, line=dict(color="black", width=0.5), row=3, col=1)

    return fig


# ============================================================
# Tab 3: 参数敏感性 — 2×3 面板 + 条件性 2D 网格热力图
# ============================================================
def build_sensitivity_figure(results: dict) -> go.Figure | None:
    scan_param = _detect_scan_param(list(results.keys()))

    pairs = []
    for tag, data in results.items():
        th = _extract_threshold(tag, scan_param=scan_param)
        if th is not None:
            pairs.append((th, tag, data))

    if len(pairs) < 2:
        return None

    pairs.sort(key=lambda x: x[0])
    thresholds = [p[0] for p in pairs]
    tags = [p[1] for p in pairs]

    metrics = [
        ("年化收益", [p[2]["summary"].get("annualized_returns", 0) for p in pairs], ".0%"),
        ("夏普比率", [p[2]["summary"].get("sharpe", 0) for p in pairs], ".3f"),
        ("最大回撤", [p[2]["summary"].get("max_drawdown", 0) for p in pairs], ".0%"),
        ("换手率", [p[2]["summary"].get("turnover", 0) for p in pairs], ".1f"),
        ("索提诺", [p[2]["summary"].get("sortino", 0) for p in pairs], ".3f"),
    ]
    trade_counts = [len(p[2]["trades"]) for p in pairs]

    if scan_param == "m":
        param_label = "m_days (回望天数)"
    elif scan_param == "r":
        param_label = "decay_ratio (衰减权重比)"
    else:
        param_label = "threshold (切换阈值)"

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[m[0] for m in metrics] + ["交易总次数"],
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    for idx, (name, vals, fmt) in enumerate(metrics):
        row = idx // 3 + 1
        col = idx % 3 + 1
        fig.add_trace(
            go.Scatter(
                x=thresholds, y=vals, mode="lines+markers+text",
                name=name, line=dict(width=1.8), marker=dict(size=8),
                text=tags, textposition="top center", textfont=dict(size=8),
                showlegend=False,
                hovertemplate=f"{name}=%{{y" + fmt + "}<br>参数=%{{x}}<extra></extra>",
            ),
            row=row, col=col,
        )

    # 第6个子图：交易次数
    fig.add_trace(
        go.Scatter(
            x=thresholds, y=trade_counts, mode="lines+markers+text",
            name="交易总次数", line=dict(color="darkred", width=1.8), marker=dict(size=8),
            text=[str(v) for v in trade_counts], textposition="top center", textfont=dict(size=8),
            showlegend=False,
            hovertemplate="交易次数=%{y}<br>参数=%{x}<extra></extra>",
        ),
        row=2, col=3,
    )

    fig.update_layout(
        title=dict(text=f"参数敏感性分析 — {param_label}", font=dict(size=18)),
        autosize=True,
        hovermode="closest",
    )
    for idx in range(6):
        row = idx // 3 + 1
        col = idx % 3 + 1
        fig.update_xaxes(title_text=param_label, row=row, col=col)

    return fig


def build_grid_heatmap_figure(results: dict) -> go.Figure | None:
    """2D 参数网格热力图（条件性生成）"""
    parsed = {tag: _parse_tag(tag) for tag in results}
    m_days_vals = sorted(set(p["m_days"] for p in parsed.values()))
    threshold_vals = sorted(set(p["threshold"] for p in parsed.values()))

    if len(m_days_vals) < 2 or len(threshold_vals) < 2:
        return None

    def build_matrix(metric_key):
        mat = np.full((len(m_days_vals), len(threshold_vals)), np.nan)
        for tag, data in results.items():
            p = parsed[tag]
            i = m_days_vals.index(p["m_days"])
            j = threshold_vals.index(p["threshold"])
            mat[i, j] = data["summary"].get(metric_key, np.nan)
        return mat

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("夏普比率", "年化收益", "最大回撤"),
        horizontal_spacing=0.10,
    )

    # 夏普比率
    mat_sharpe = build_matrix("sharpe")
    text_sharpe = np.array([[f"{v:.3f}" if not np.isnan(v) else "" for v in row] for row in mat_sharpe])
    fig.add_trace(
        go.Heatmap(
            z=mat_sharpe, x=[f"{t:.2f}" for t in threshold_vals],
            y=[str(d) for d in m_days_vals],
            text=text_sharpe, texttemplate="%{text}", textfont=dict(size=10),
            colorscale="YlGn", showscale=True, colorbar=dict(title="夏普"),
            hovertemplate="m_days=%{y}<br>threshold=%{x}<br>夏普=%{z:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # 年化收益
    mat_ret = build_matrix("annualized_returns")
    text_ret = np.array([[f"{v:.1%}" if not np.isnan(v) else "" for v in row] for row in mat_ret])
    fig.add_trace(
        go.Heatmap(
            z=mat_ret, x=[f"{t:.2f}" for t in threshold_vals],
            y=[str(d) for d in m_days_vals],
            text=text_ret, texttemplate="%{text}", textfont=dict(size=10),
            colorscale="RdYlGn", zmin=-0.1, zmax=0.5,
            showscale=True, colorbar=dict(title="年化收益"),
            hovertemplate="m_days=%{y}<br>threshold=%{x}<br>年化收益=%{z:.2%}<extra></extra>",
        ),
        row=1, col=2,
    )

    # 最大回撤
    mat_dd = build_matrix("max_drawdown")
    text_dd = np.array([[f"{v:.1%}" if not np.isnan(v) else "" for v in row] for row in mat_dd])
    fig.add_trace(
        go.Heatmap(
            z=mat_dd, x=[f"{t:.2f}" for t in threshold_vals],
            y=[str(d) for d in m_days_vals],
            text=text_dd, texttemplate="%{text}", textfont=dict(size=10),
            colorscale="RdYlGn_r", zmin=-0.5, zmax=-0.1,
            showscale=True, colorbar=dict(title="最大回撤"),
            hovertemplate="m_days=%{y}<br>threshold=%{x}<br>最大回撤=%{z:.2%}<extra></extra>",
        ),
        row=1, col=3,
    )

    # 标出最优格子（红色边框）
    for col, mat in enumerate([mat_sharpe, mat_ret, mat_dd], start=1):
        best = np.unravel_index(np.nanargmax(mat), mat.shape)
        fig.add_shape(
            type="rect",
            x0=best[1] - 0.5, y0=best[0] - 0.5,
            x1=best[1] + 0.5, y1=best[0] + 0.5,
            line=dict(color="red", width=3),
            fillcolor="rgba(0,0,0,0)",
            row=1, col=col,
        )

    fig.update_layout(
        title=dict(text="二维参数网格热力图 (红框=最优)", font=dict(size=18)),
        autosize=True,
    )
    fig.update_xaxes(title_text="switch_threshold", row=1, col=1)
    fig.update_xaxes(title_text="switch_threshold", row=1, col=2)
    fig.update_xaxes(title_text="switch_threshold", row=1, col=3)
    fig.update_yaxes(title_text="m_days", row=1, col=1)
    fig.update_yaxes(title_text="m_days", row=1, col=2)
    fig.update_yaxes(title_text="m_days", row=1, col=3)

    return fig


# ============================================================
# Tab 4: 交易分析 — 2×2 面板
# ============================================================
def build_trades_figure(results: dict) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "年度交易次数", "换手率 vs 年化收益",
            "持仓天数分布", "累计交易成本",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    tags = list(results.keys())
    n = len(tags)

    # ---- (1,1) 年度交易次数 ----
    all_yearly_trades = {}
    years_set = set()
    for tag, data in results.items():
        trades = data["trades"]
        if trades.empty:
            continue
        if not isinstance(trades.index, pd.DatetimeIndex):
            trades.index = pd.to_datetime(trades.index)
        yearly = trades.groupby(trades.index.year).size()
        all_yearly_trades[tag] = yearly
        years_set.update(yearly.index.tolist())

    if years_set:
        years = sorted(years_set)
        for i, (tag, counts) in enumerate(all_yearly_trades.items()):
            vals = [counts.loc[y] if y in counts.index else 0 for y in years]
            fig.add_trace(
                go.Bar(
                    x=[str(y) for y in years], y=vals,
                    name=tag, marker=dict(color=_get_color(i)),
                    legendgroup=tag, showlegend=(i == 0),
                    hovertemplate=f"{tag}<br>%{{x}}<br>交易次数: %{{y}}<extra></extra>",
                ),
                row=1, col=1,
            )

    # ---- (1,2) 换手率 vs 年化收益 ----
    for i, (tag, data) in enumerate(results.items()):
        ann = data["summary"].get("annualized_returns", 0)
        turnover = data["summary"].get("turnover", 0)
        sharpe = data["summary"].get("sharpe", 0)
        fig.add_trace(
            go.Scatter(
                x=[turnover], y=[ann], mode="markers+text",
                name=tag, marker=dict(color=_get_color(i), size=11, line=dict(color="black", width=0.5)),
                text=[tag], textposition="top right", textfont=dict(size=8),
                legendgroup=tag, showlegend=False,
                hovertemplate=f"{tag}<br>换手率: %{{x:.1f}}<br>年化收益: %{{y:.2%}}<br>夏普: {sharpe:.3f}<extra></extra>",
            ),
            row=1, col=2,
        )

    # ---- (2,1) 持仓天数分布 (箱线图) ----
    holding_data = []
    holding_labels = []
    for tag, data in results.items():
        trades = data["trades"]
        if trades.empty or len(trades) < 2:
            continue
        if not isinstance(trades.index, pd.DatetimeIndex):
            trades.index = pd.to_datetime(trades.index)
        durations = _compute_holding_durations(trades)
        if durations:
            holding_data.append(durations)
            holding_labels.append(tag)

    for i, (label, durations) in enumerate(zip(holding_labels, holding_data)):
        fig.add_trace(
            go.Box(
                y=durations, name=label,
                marker=dict(color=_get_color(i)),
                boxpoints=False,
                showlegend=False,
                hovertemplate=f"{label}<br>持仓天数: %{{y}}<extra></extra>",
            ),
            row=2, col=1,
        )

    # ---- (2,2) 累计交易成本 ----
    for i, (tag, data) in enumerate(results.items()):
        trades = data["trades"]
        if trades.empty:
            continue
        cost_cols = [col for col in ["commission", "tax", "transaction_cost"] if col in trades.columns]
        if not cost_cols:
            continue
        cost_col = cost_cols[0]
        if not isinstance(trades.index, pd.DatetimeIndex):
            trades.index = pd.to_datetime(trades.index)
        cumulative = trades[cost_col].cumsum()
        fig.add_trace(
            go.Scatter(
                x=trades.index, y=cumulative.values, mode="lines",
                name=tag, line=dict(color=_get_color(i), width=1.0),
                legendgroup=tag, showlegend=False,
                hovertemplate=f"{tag}<br>%{{x|%Y-%m-%d}}<br>累计成本: %{{y:.2f}}<extra></extra>",
            ),
            row=2, col=2,
        )

    fig.update_layout(
        title=dict(text="交易行为分析", font=dict(size=18)),
        barmode="group",
        hovermode="closest",
        autosize=True,
        legend=dict(font=dict(size=10)),
    )
    fig.update_yaxes(title_text="年化收益", tickformat=".0%", row=1, col=2)
    fig.update_xaxes(title_text="换手率", row=1, col=2)
    fig.update_yaxes(title_text="持仓天数", row=2, col=1)
    fig.update_yaxes(title_text="累计成本", row=2, col=2)

    return fig


# ============================================================
# Tab 5: 年度分析 — 热力图 + 分组柱状图
# ============================================================
def build_yearly_figure(results: dict) -> go.Figure | None:
    all_yearly = {}
    years_set = set()
    for tag, data in results.items():
        ym = compute_yearly_metrics(data["portfolio"])
        if not ym.empty:
            all_yearly[tag] = ym["return"]
            years_set.update(ym.index.tolist())

    if not years_set:
        return None

    years = sorted(years_set)
    tags = list(all_yearly.keys())
    matrix = np.zeros((len(tags), len(years)))
    for i, tag in enumerate(tags):
        for j, year in enumerate(years):
            matrix[i, j] = all_yearly[tag].get(year, np.nan)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("逐年收益热力图", "逐年收益分组柱状图"),
        vertical_spacing=0.12,
        row_heights=[0.45, 0.55],
    )

    # 热力图
    text_mat = np.array([[f"{v:.1%}" if not np.isnan(v) else "" for v in row] for row in matrix])
    fig.add_trace(
        go.Heatmap(
            z=matrix, x=[str(y) for y in years], y=tags,
            text=text_mat, texttemplate="%{text}", textfont=dict(size=10),
            colorscale="RdYlGn", zmin=-0.4, zmax=0.8,
            showscale=True, colorbar=dict(title="收益", tickformat=".0%"),
            hovertemplate="%{y}<br>%{x}<br>收益: %{z:.2%}<extra></extra>",
        ),
        row=1, col=1,
    )

    # 分组柱状图
    for i, (tag, ym_returns) in enumerate(all_yearly.items()):
        vals = [ym_returns.loc[y] if y in ym_returns.index else 0 for y in years]
        fig.add_trace(
            go.Bar(
                x=[str(y) for y in years], y=vals,
                name=tag, marker=dict(color=_get_color(i)),
                legendgroup=tag, showlegend=True,
                hovertemplate=f"{tag}<br>%{{x}}<br>收益: %{{y:.2%}}<extra></extra>",
            ),
            row=2, col=1,
        )

    fig.update_layout(
        title=dict(text="年度收益分析", font=dict(size=18)),
        barmode="group",
        hovermode="closest",
        autosize=True,
        legend=dict(font=dict(size=10), orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    )
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.add_hline(y=0, line=dict(color="black", width=0.5), row=2, col=1)

    return fig


# ============================================================
# Tab 6: 风险收益散点图
# ============================================================
def build_risk_return_figure(results: dict) -> go.Figure:
    fig = go.Figure()

    for i, (tag, data) in enumerate(results.items()):
        s = data["summary"]
        ann = s.get("annualized_returns", 0)
        mdd = s.get("max_drawdown", 0)
        sharpe = s.get("sharpe", 0)
        fig.add_trace(
            go.Scatter(
                x=[mdd], y=[ann], mode="markers+text",
                name=tag,
                marker=dict(
                    color=_get_color(i), size=15 + sharpe * 6,
                    line=dict(color="black", width=0.8),
                    sizemode="area",
                ),
                text=[tag], textposition="top right", textfont=dict(size=9),
                hovertemplate=f"<b>{tag}</b><br>最大回撤: %{{x:.2%}}<br>年化收益: %{{y:.2%}}<br>夏普比率: {sharpe:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text="风险收益散点图 (气泡大小=夏普比率)", font=dict(size=18)),
        xaxis=dict(title="最大回撤", tickformat=".0%"),
        yaxis=dict(title="年化收益", tickformat=".0%"),
        hovermode="closest",
        autosize=True,
        showlegend=False,
    )
    fig.add_hline(y=0, line=dict(color="black", width=0.5))
    fig.add_vline(x=0, line=dict(color="black", width=0.5))

    return fig


# ============================================================
# 绩效汇总表 HTML
# ============================================================
def build_summary_table_html(results: dict) -> str:
    """生成绩效汇总 HTML 表格"""
    df = extract_summary_table(results)
    if df.empty:
        return "<p>无数据</p>"

    # build header
    cols = df.columns.tolist()
    thead = "<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"

    # build body
    rows_html = ""
    for _, row in df.iterrows():
        cells = "".join(f"<td>{row[c]}</td>" for c in cols)
        rows_html += f"<tr>{cells}</tr>"

    return f"""<table class="summary-table">
<thead>{thead}</thead>
<tbody>{rows_html}</tbody>
</table>"""


# ============================================================
# HTML 仪表盘组装
# ============================================================
def build_dashboard_html(results: dict, output_path: Path) -> None:
    """生成完整的 HTML 仪表盘文件，图表延迟初始化"""
    import json as _json
    from collections import OrderedDict

    tags = list(results.keys())
    n = len(tags)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("生成图表...")

    # 构建各 Tab 的图表
    overview_fig = build_overview_figure(results)
    curves_fig = build_curves_figure(results)
    sensitivity_fig = build_sensitivity_figure(results)
    grid_heatmap_fig = build_grid_heatmap_figure(results)
    trades_fig = build_trades_figure(results)
    yearly_fig = build_yearly_figure(results)
    risk_return_fig = build_risk_return_figure(results)

    # Summary 表格
    summary_table_html = build_summary_table_html(results)

    # ---- 构建延迟初始化数据结构 ----
    # {tab_id: [(div_id, fig_spec_json), ...]}
    tab_figures = OrderedDict()
    tab_figures["overview"] = [("ov_main", overview_fig)]
    tab_figures["curves"] = [("cv_main", curves_fig)]

    sens_specs = []
    if sensitivity_fig is not None:
        sens_specs.append(("sens_main", sensitivity_fig))
    if grid_heatmap_fig is not None:
        sens_specs.append(("grid_main", grid_heatmap_fig))
    tab_figures["sensitivity"] = sens_specs

    tab_figures["trades"] = [("tr_main", trades_fig)]
    tab_figures["yearly"] = [("yr_main", yearly_fig)] if yearly_fig is not None else []
    tab_figures["risk"] = [("rr_main", risk_return_fig)]

    # 序列化为 JSON
    figure_specs_json = OrderedDict()
    for tab_id, specs in tab_figures.items():
        for div_id, fig in specs:
            figure_specs_json[div_id] = _json.loads(fig.to_json())

    # ---- 构建各 Tab 的 HTML 占位 div ----
    def _make_div(div_id: str) -> str:
        return f'<div id="{div_id}" class="plotly-graph-div"></div>'

    def _make_plot_container(div_id: str) -> str:
        return f'<div class="plot-container">{_make_div(div_id)}</div>'

    overview_content = _make_plot_container("ov_main")
    curves_content = _make_plot_container("cv_main")

    sensitivity_content = ""
    if sensitivity_fig is not None:
        sensitivity_content += _make_plot_container("sens_main")
    else:
        sensitivity_content += "<p class='no-data'>至少需要 2 个单维度实验才能展示参数敏感性</p>"
    if grid_heatmap_fig is not None:
        sensitivity_content += _make_plot_container("grid_main")

    trades_content = _make_plot_container("tr_main")

    yearly_content = ""
    if yearly_fig is not None:
        yearly_content = _make_plot_container("yr_main")
    else:
        yearly_content = "<p class='no-data'>无逐年数据</p>"

    risk_content = _make_plot_container("rr_main")

    # 读取 plotly.js（离线内嵌）
    plotly_js = get_plotlyjs()

    # HTML 模板
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>策略回测分析仪表盘</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Microsoft YaHei', 'SimHei', 'PingFang SC', 'DejaVu Sans', sans-serif;
      background: #f0f2f5; color: #333;
    }}
    .dashboard-header {{
      background: linear-gradient(135deg, #1a2a3a 0%, #2c3e50 100%);
      color: white; padding: 20px 28px;
    }}
    .dashboard-header h1 {{ font-size: 24px; font-weight: 600; }}
    .dashboard-header .meta {{
      font-size: 13px; opacity: 0.7; margin-top: 6px;
    }}
    .tab-bar {{
      display: flex; background: #fff; padding: 0 20px;
      border-bottom: 1px solid #e0e0e0; position: sticky; top: 0; z-index: 100;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}
    .tab-btn {{
      padding: 14px 22px; border: none; background: transparent;
      color: #666; cursor: pointer; font-size: 14px; font-family: inherit;
      border-bottom: 3px solid transparent; transition: all 0.2s;
      outline: none;
    }}
    .tab-btn:hover {{ color: #1a73e8; background: #f8f9ff; }}
    .tab-btn.active {{ color: #1a73e8; border-bottom-color: #1a73e8; font-weight: 600; }}
    .tab-panel {{ display: none; padding: 20px; }}
    .tab-panel.active {{ display: block; }}
    .plot-container {{
      background: white; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
      padding: 16px; margin-bottom: 20px;
    }}
    .plot-container h3 {{
      font-size: 16px; margin-bottom: 12px; color: #2c3e50;
      border-left: 4px solid #1a73e8; padding-left: 10px;
    }}
    .plotly-graph-div {{
      min-height: 450px;
    }}
    .summary-table {{
      width: 100%; border-collapse: collapse; font-size: 13px;
    }}
    .summary-table th {{
      background: #2c3e50; color: white; padding: 10px 12px;
      text-align: left; font-weight: 500; position: sticky; top: 0;
    }}
    .summary-table td {{
      padding: 8px 12px; border-bottom: 1px solid #ecf0f1;
    }}
    .summary-table tbody tr:hover {{ background: #f0f4ff; }}
    .summary-table tbody tr:nth-child(even) {{ background: #fafbfc; }}
    .summary-table tbody tr:nth-child(even):hover {{ background: #f0f4ff; }}
    .footer {{
      text-align: center; padding: 20px; color: #999; font-size: 12px;
    }}
    .no-data {{
      text-align: center; color: #999; padding: 40px; font-size: 14px;
    }}
  </style>
</head>
<body>
  <div class="dashboard-header">
    <h1>策略回测分析仪表盘</h1>
    <div class="meta">
      数据目录: {RESULT_DIR} | 实验数量: {n} | 实验标签: {", ".join(tags)} | 生成时间: {generated_at}
    </div>
  </div>

  <div class="tab-bar">
    <button class="tab-btn active" onclick="switchTab(event, 'overview')">概览</button>
    <button class="tab-btn" onclick="switchTab(event, 'curves')">曲线对比</button>
    <button class="tab-btn" onclick="switchTab(event, 'sensitivity')">参数敏感性</button>
    <button class="tab-btn" onclick="switchTab(event, 'trades')">交易分析</button>
    <button class="tab-btn" onclick="switchTab(event, 'yearly')">年度分析</button>
    <button class="tab-btn" onclick="switchTab(event, 'risk')">风险收益</button>
  </div>

  <div class="tab-panel active" id="overview">
    {overview_content}
    <div class="plot-container">
      <h3>绩效指标汇总</h3>
      {summary_table_html}
    </div>
  </div>

  <div class="tab-panel" id="curves">
    {curves_content}
  </div>

  <div class="tab-panel" id="sensitivity">
    {sensitivity_content}
  </div>

  <div class="tab-panel" id="trades">
    {trades_content}
  </div>

  <div class="tab-panel" id="yearly">
    {yearly_content}
  </div>

  <div class="tab-panel" id="risk">
    {risk_content}
  </div>

  <div class="footer">RQAlpha 策略回测分析 | Powered by Plotly</div>

  <script>{plotly_js}</script>
  <script>
    var FIGURES = {_json.dumps(figure_specs_json)};
    var _initialized = {{}};

    function ensurePlots(tabName) {{
      if (_initialized[tabName]) {{
        // 已初始化，只需 resize
        var panel = document.getElementById(tabName);
        if (panel) {{
          panel.querySelectorAll('.plotly-graph-div').forEach(function(el) {{
            Plotly.Plots.resize(el);
          }});
        }}
        return;
      }}
      _initialized[tabName] = true;
      var panel = document.getElementById(tabName);
      if (!panel) return;
      var divs = panel.querySelectorAll('.plotly-graph-div');
      divs.forEach(function(el) {{
        var spec = FIGURES[el.id];
        if (spec) {{
          Plotly.newPlot(el.id, spec.data, spec.layout);
        }}
      }});
    }}

    function switchTab(evt, tabName) {{
      document.querySelectorAll('.tab-panel').forEach(function(p) {{ p.classList.remove('active'); }});
      document.querySelectorAll('.tab-btn').forEach(function(b) {{ b.classList.remove('active'); }});
      document.getElementById(tabName).classList.add('active');
      evt.currentTarget.classList.add('active');
      // 延迟初始化或 resize（等待 display:block 生效）
      setTimeout(function() {{ ensurePlots(tabName); }}, 50);
    }}

    // 页面加载后立即初始化 overview（已显示）
    window.addEventListener('load', function() {{
      ensurePlots('overview');
    }});
  </script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"已保存仪表盘: {output_path}  ({file_size_mb:.1f} MB)")


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

    output_path = RESULT_DIR / "analysis_dashboard.html"
    build_dashboard_html(results, output_path)
    print("\n完成! 用浏览器打开 analysis_dashboard.html 即可查看交互式仪表盘。")


if __name__ == "__main__":
    main()
