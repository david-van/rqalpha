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

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs


warnings.filterwarnings("ignore")

# ============================================================
# 数据加载 & 指标计算
# ============================================================
TRADING_DAYS_PER_YEAR = 250

# ============================================================
# 配置
# ============================================================
RESULT_DIR = Path(__file__).with_name("batch_results")
RESULT_DIR = RESULT_DIR.joinpath("xiaoe_pool")

# Bundle 日线数据路径
BUNDLE_PATH = Path("D:/datas/bundle/stocks.h5")
_STOCK_DATA_CACHE = {}  # (code, start, end) → DataFrame

# 颜色方案：尽量接近原 matplotlib tab10
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

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

    m = re.search(r"(?<![a-z])m(\d{2,3})(?!\d)", tag)
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
    has_m = bool(re.search(r"(?<![a-z])m\d{2,3}(?!\d)", tag) or re.search(r"mdays[_]\d+", tag))
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
    import re
    for tag in tags:
        if tag == "baseline":
            continue
        if re.search(r"(?<![a-z])r\d{2}(?!\d)", tag):
            return "r"
        if re.search(r"(?<![a-z])m\d{2,3}(?!\d)", tag) or re.search(r"mdays[_]\d+", tag):
            return "m"
        if re.search(r"(?:^|_)t\d{3}(?!\d)", tag) or re.search(r"switch_threshold", tag):
            return "t"
    return "m"



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
        height=750,
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
        height=1100,
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
        height=700,
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
        height=750,
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
        height=800,
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
        height=650,
        showlegend=False,
    )
    fig.add_hline(y=0, line=dict(color="black", width=0.5))
    fig.add_vline(x=0, line=dict(color="black", width=0.5))

    return fig


# ============================================================
# Tab 7: 深度分析 — 单策略 4 张核心图表
# ============================================================

def _get_daily_returns(portfolio: pd.DataFrame) -> pd.Series | None:
    """从 portfolio 提取日收益率序列，统一处理 DatetimeIndex"""
    if portfolio.empty or "unit_net_value" not in portfolio.columns:
        return None
    nv = portfolio["unit_net_value"].dropna()
    if len(nv) < 5:
        return None
    if not isinstance(nv.index, pd.DatetimeIndex):
        nv.index = pd.to_datetime(nv.index)
    return nv.pct_change().dropna()


def build_monthly_heatmap(portfolio: pd.DataFrame) -> go.Figure | None:
    """月度收益热力图：行=年份，列=月份，色阶 RdYlGn"""
    daily_ret = _get_daily_returns(portfolio)
    if daily_ret is None:
        return None

    monthly = daily_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    if monthly.empty:
        return None

    records = []
    for dt, ret in monthly.items():
        records.append({"year": dt.year, "month": dt.month, "return": ret})
    df = pd.DataFrame(records)
    pivot = df.pivot_table(index="year", columns="month", values="return", sort=False)

    all_months = list(range(1, 13))
    pivot = pivot.reindex(columns=all_months)
    years = pivot.index.tolist()

    z = pivot.values
    text = [[f"{v:.1%}" if not np.isnan(v) else "" for v in row] for row in z]
    month_labels = [f"{m}月" for m in all_months]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z, x=month_labels, y=[str(y) for y in years],
        text=text, texttemplate="%{text}", textfont=dict(size=11),
        colorscale="RdYlGn", zmid=0,
        showscale=True, colorbar=dict(title="收益", tickformat=".0%"),
        hovertemplate="%{y}年 %{x}<br>收益: %{z:.2%}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="月度收益热力图", font=dict(size=14)),
        xaxis=dict(title="月份", side="bottom"),
        yaxis=dict(title="年份", autorange="reversed"),
        autosize=True, height=400, margin=dict(t=40, b=40, l=40, r=20),
    )
    return fig


def build_returns_distribution(portfolio: pd.DataFrame) -> go.Figure | None:
    """日收益分布直方图 + 正态拟合曲线"""
    daily_ret = _get_daily_returns(portfolio)
    if daily_ret is None:
        return None

    values = daily_ret.values
    mu, std = np.mean(values), np.std(values)
    from scipy.stats import skew, kurtosis  # noqa
    from scipy.stats import norm as scipy_norm

    x_norm = np.linspace(mu - 4 * std, mu + 4 * std, 200)
    pdf_norm = scipy_norm.pdf(x_norm, mu, std)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=values, histnorm="percent", nbinsx=80,
        name="日收益分布", marker=dict(color="#4a90d9", line=dict(color="white", width=0.5)),
        hovertemplate="收益: %{x:.3%}<br>比例: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x_norm, y=pdf_norm * 100 * (x_norm[1] - x_norm[0]) * len(values) / (len(values) / 100),
        mode="lines", name="正态拟合", line=dict(color="#d62728", width=2, dash="dash"),
        hovertemplate="正态拟合<extra></extra>",
    ))

    # 将实际直方图 bin 分布和正态拟合的差异重新绘制
    # 使用 method 2: 用密度归一化
    fig.data = ()
    fig.add_trace(go.Histogram(
        x=values, histnorm="probability density", nbinsx=80,
        name="日收益分布", marker=dict(color="#4a90d9", line=dict(color="white", width=0.5)),
        hovertemplate="收益: %{x:.3%}<br>密度: %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x_norm, y=pdf_norm, mode="lines",
        name="正态拟合", line=dict(color="#d62728", width=2, dash="dash"),
        hovertemplate="正态拟合<extra></extra>",
    ))

    s = skew(values, bias=False)
    k = kurtosis(values, bias=False)

    fig.update_layout(
        title=dict(text=f"日收益分布 (μ={mu:.4%}, σ={std:.4%}, 偏度={s:.2f}, 峰度={k:.2f})", font=dict(size=14)),
        xaxis=dict(title="日收益率", tickformat=".1%"),
        yaxis=dict(title="概率密度"),
        autosize=True, height=420, margin=dict(t=55, b=40, l=40, r=20),
        legend=dict(x=0.75, y=0.95),
    )
    return fig


def build_underwater_plot(portfolio: pd.DataFrame) -> go.Figure | None:
    """回撤水深图：面积填充 + 标记最大回撤"""
    if portfolio.empty or "unit_net_value" not in portfolio.columns:
        return None
    nv = portfolio["unit_net_value"].dropna()
    if len(nv) < 5:
        return None
    if not isinstance(nv.index, pd.DatetimeIndex):
        nv.index = pd.to_datetime(nv.index)

    dd = compute_drawdown_series(nv)

    min_idx = dd.idxmin()
    min_val = dd.min()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values, mode="lines",
        fill="tozeroy", fillcolor="rgba(214,39,40,0.25)",
        line=dict(color="#d62728", width=1.2),
        name="回撤深度",
        hovertemplate="%{x|%Y-%m-%d}<br>回撤: %{y:.2%}<extra></extra>",
    ))
    # 零线水平标记
    fig.add_hline(y=0, line=dict(color="gray", width=0.5, dash="dot"))

    # 标注最深回撤点
    fig.add_trace(go.Scatter(
        x=[min_idx], y=[min_val], mode="markers+text",
        name="最大回撤",
        marker=dict(color="red", size=12, symbol="x-thin", line=dict(width=2)),
        text=[f" {min_idx.strftime('%Y-%m-%d')}<br> {min_val:.2%}"],
        textposition="bottom right", textfont=dict(size=10, color="red"),
        showlegend=False,
        hovertemplate="最大回撤<br>%{x|%Y-%m-%d}<br>%{y:.2%}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=f"回撤水深图 (最大回撤 {min_val:.2%})", font=dict(size=14)),
        yaxis=dict(title="回撤", tickformat=".0%"),
        autosize=True, height=420, margin=dict(t=55, b=40, l=40, r=20),
        showlegend=False,
    )
    return fig


def build_rolling_and_worst(portfolio: pd.DataFrame) -> go.Figure | None:
    """双轴：滚动年化夏普 + 滚动年化波动率，附带 worst-N 天文本"""
    daily_ret = _get_daily_returns(portfolio)
    if daily_ret is None:
        return None

    window = min(TRADING_DAYS_PER_YEAR, len(daily_ret) // 2)
    if window < 20:
        return None

    rolling_mean = daily_ret.rolling(window).mean() * TRADING_DAYS_PER_YEAR
    rolling_std = daily_ret.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    rolling_sharpe = rolling_mean / rolling_std.replace(0, np.nan)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=rolling_sharpe.index, y=rolling_sharpe.values, mode="lines",
            name="滚动夏普", line=dict(color="#1f77b4", width=1.2),
            hovertemplate="%{x|%Y-%m-%d}<br>滚动夏普: %{y:.3f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=rolling_std.index, y=rolling_std.values, mode="lines",
            name="滚动波动率", line=dict(color="#ff7f0e", width=1.2, dash="dash"),
            hovertemplate="%{x|%Y-%m-%d}<br>滚动波动率: %{y:.3%}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="滚动夏普", secondary_y=False)
    fig.update_yaxes(title_text="滚动波动率", tickformat=".0%", secondary_y=True)

    fig.update_layout(
        title=dict(text=f"滚动 {window} 日夏普 & 波动率", font=dict(size=14)),
        hovermode="closest",
        autosize=True, height=420, margin=dict(t=55, b=40, l=40, r=40),
        legend=dict(x=0.75, y=0.95),
    )
    fig.add_hline(y=0, line=dict(color="gray", width=0.5), secondary_y=False)
    return fig


def build_worst_days_table(portfolio: pd.DataFrame, n: int = 10) -> str:
    """生成最差 N 天 HTML 表格"""
    daily_ret = _get_daily_returns(portfolio)
    if daily_ret is None:
        return "<p>无数据</p>"

    worst = daily_ret.nsmallest(n)
    rows = ""
    for dt, ret in worst.items():
        rows += f"<tr><td>{dt.strftime('%Y-%m-%d')}</td><td>{ret:.4%}</td></tr>"

    return f"""<table class="summary-table">
<thead><tr><th>日期</th><th>收益率</th></tr></thead>
<tbody>{rows}</tbody>
</table>"""


# ============================================================
# Tab 8: 持仓与交易 — 3 张图 + 1 张调仓明细表
# ============================================================

def _prepare_trades(trades: pd.DataFrame) -> pd.DataFrame | None:
    """统一预处理 trades DataFrame"""
    if trades.empty:
        return None
    df = trades.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    # 单笔成交金额
    if "transaction_cost" in df.columns:
        df["amount"] = df["last_price"] * df["last_quantity"].abs()
    else:
        df["amount"] = df["last_price"] * df["last_quantity"].abs()
    return df


def build_position_count_chart(portfolio: pd.DataFrame, trades: pd.DataFrame) -> go.Figure | None:
    """持仓数量 + 总市值双轴时序图"""
    df = _prepare_trades(trades)
    if df is None:
        return None

    # 逐日累计持仓数：按时间顺序累加，在每个交易日计算当前持仓数
    df_sorted = df.sort_index()
    positions = {}  # {stock: cumulative_qty}
    counts = []
    for ts, day_trades in df_sorted.groupby(df_sorted.index):
        for _, row in day_trades.iterrows():
            sid = row["order_book_id"]
            if sid not in positions:
                positions[sid] = 0
            if row["side"] == "BUY":
                positions[sid] += row["last_quantity"]
            else:
                positions[sid] -= abs(row["last_quantity"])
        n = sum(1 for q in positions.values() if q > 0)
        counts.append((ts, n))

    if not counts:
        return None

    count_series = pd.Series(
        [c[1] for c in counts],
        index=pd.DatetimeIndex([c[0] for c in counts]),
    )

    # 总市值
    tv = None
    if not portfolio.empty and "total_value" in portfolio.columns:
        tv = portfolio["total_value"]
        if not isinstance(tv.index, pd.DatetimeIndex):
            tv.index = pd.to_datetime(tv.index)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=count_series.index, y=count_series.values, mode="lines",
            name="持仓股票数", line=dict(color="#1f77b4", width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>持仓数: %{y}只<extra></extra>",
        ),
        secondary_y=False,
    )
    if tv is not None:
        fig.add_trace(
            go.Scatter(
                x=tv.index, y=tv.values, mode="lines",
                name="总市值", line=dict(color="#ff7f0e", width=1.2, dash="dash"),
                hovertemplate="%{x|%Y-%m-%d}<br>总市值: %{y:,.0f}<extra></extra>",
            ),
            secondary_y=True,
        )

    fig.update_yaxes(title_text="持仓股票数", secondary_y=False)
    fig.update_yaxes(title_text="总市值", tickformat=",.0f", secondary_y=True)
    fig.update_layout(
        title=dict(text="持仓数量 & 总市值", font=dict(size=14)),
        hovermode="closest",
        autosize=True, height=420, margin=dict(t=55, b=40, l=40, r=40),
        legend=dict(x=0.75, y=0.95),
    )
    return fig


def _read_stock_daily(code: str, start_dt, end_dt) -> pd.DataFrame | None:
    """从本地 bundle 读取个股日线数据，模块级缓存"""
    start_str = pd.Timestamp(start_dt).strftime("%Y%m%d")
    end_str = pd.Timestamp(end_dt).strftime("%Y%m%d")
    cache_key = (code, start_str, end_str)
    if cache_key in _STOCK_DATA_CACHE:
        return _STOCK_DATA_CACHE[cache_key].copy()

    if not BUNDLE_PATH.exists():
        return None

    try:
        with h5py.File(BUNDLE_PATH, "r") as f:
            if code not in f:
                return None
            ds = f[code]
            arr = ds[:]
            df = pd.DataFrame({
                name: arr[name] for name in ds.dtype.names
            })
        df["datetime"] = pd.to_datetime(df["datetime"].astype(str), format="%Y%m%d%H%M%S")
        df = df.set_index("datetime").sort_index()
        df = df.loc[start_dt:end_dt]
        if df.empty:
            return None
        _STOCK_DATA_CACHE[cache_key] = df
        return df.copy()
    except Exception:
        return None


def build_stock_kline_chart(trades: pd.DataFrame, code: str) -> go.Figure | None:
    """个股K线图：蜡烛图 + EMA20/50/60/120 + 买卖点标记 + 成交量副图"""
    stock_trades = trades[trades["order_book_id"] == code].copy()
    if stock_trades.empty:
        return None
    if not isinstance(stock_trades.index, pd.DatetimeIndex):
        stock_trades.index = pd.to_datetime(stock_trades.index)

    stock_trades = stock_trades.sort_index()
    t0, t1 = stock_trades.index[0], stock_trades.index[-1]
    # 起始留 padding 给 EMA 计算，结束延后 120 个交易日
    margin = pd.Timedelta(days=300)
    t1_ext = t1 + pd.Timedelta(days=200)  # 约120个交易日
    df = _read_stock_daily(code, t0 - margin, t1_ext)
    if df is None or df.empty:
        return None

    # EMA 预设：全部计算，默认只显示 20/50/60
    EMA_PRESETS = [5, 10, 20, 30, 50, 60, 120, 250]
    EMA_DEFAULT_ON = {20, 50, 60}
    EMA_COLORS = {
        5: "#17becf", 10: "#bcbd22", 20: "#ff7f0e", 30: "#e377c2",
        50: "#2ca02c", 60: "#1f77b4", 120: "#9467bd", 250: "#d62728",
    }
    for span in EMA_PRESETS:
        df[f"ema_{span}"] = df["close"].ewm(span=span, min_periods=span).mean()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.68, 0.32],
    )

    # 蜡烛图
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="K线", increasing=dict(line=dict(color="#d62728"), fillcolor="#d62728"),
        decreasing=dict(line=dict(color="#2ca02c"), fillcolor="#2ca02c"),
        hovertemplate="%{x|%Y-%m-%d}<br>开: %{open:.3f}<br>高: %{high:.3f}<br>低: %{low:.3f}<br>收: %{close:.3f}<extra></extra>",
    ), row=1, col=1)

    # EMA 线
    for span in EMA_PRESETS:
        label = f"EMA{span}"
        visible = span in EMA_DEFAULT_ON
        color = EMA_COLORS.get(span, "#888888")
        fig.add_trace(go.Scatter(
            x=df.index, y=df[f"ema_{span}"], mode="lines",
            name=label, line=dict(color=color, width=1.2),
            visible=visible,
            legendgroup=label,
            hovertemplate=f"{label}: %{{y:.3f}}<extra></extra>",
        ), row=1, col=1)

    # 买点 ▲
    buys = stock_trades[stock_trades["side"] == "BUY"]
    if not buys.empty:
        # 获取买入日期在日线中的收盘价作为标记Y坐标
        buy_y = []
        buy_x = []
        buy_text = []
        for ts, row in buys.iterrows():
            dt = ts if isinstance(ts, pd.Timestamp) else pd.Timestamp(ts)
            kline_row = df.loc[dt.strftime("%Y-%m-%d"):].head(1) if dt.strftime("%Y-%m-%d") in df.index.strftime("%Y-%m-%d") else None
            close_v = float(row["last_price"])
            buy_y.append(close_v)
            buy_x.append(dt)
            buy_text.append(f"买 {row.get('symbol','')}<br>{dt.strftime('%Y-%m-%d')}<br>价:{row['last_price']:.3f}<br>量:{int(row['last_quantity'])}股")
        fig.add_trace(go.Scatter(
            x=buy_x, y=buy_y, mode="markers",
            name="买入", marker=dict(symbol="triangle-up", color="#1f77b4", size=12, line=dict(color="#1f77b4", width=1)),
            legendgroup="buy", showlegend=True,
            text=buy_text, hoverinfo="text",
        ), row=1, col=1)

    # 卖点 ▼
    sells = stock_trades[stock_trades["side"] == "SELL"]
    if not sells.empty:
        sell_y = []
        sell_x = []
        sell_text = []
        for ts, row in sells.iterrows():
            dt = ts if isinstance(ts, pd.Timestamp) else pd.Timestamp(ts)
            close_v = float(row["last_price"])
            sell_y.append(close_v)
            sell_x.append(dt)
            sell_text.append(f"卖 {row.get('symbol','')}<br>{dt.strftime('%Y-%m-%d')}<br>价:{row['last_price']:.3f}<br>量:{int(row['last_quantity'])}股")
        fig.add_trace(go.Scatter(
            x=sell_x, y=sell_y, mode="markers",
            name="卖出", marker=dict(symbol="triangle-down", color="#9467bd", size=12, line=dict(color="#9467bd",
                                                                                                 width=1)),
            legendgroup="sell", showlegend=True,
            text=sell_text, hoverinfo="text",
        ), row=1, col=1)

    # 成交量
    colors_vol = ["#d62728" if df.loc[t, "close"] >= df.loc[t, "open"] else "#2ca02c" for t in df.index]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"], name="成交量",
        marker=dict(color=colors_vol, line=dict(width=0)),
        hovertemplate="%{x|%Y-%m-%d}<br>量: %{y:,.0f}<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        title=dict(text=f"{code} {stock_trades['symbol'].iloc[0] if 'symbol' in stock_trades.columns else ''}", font=dict(size=14)),
        hovermode="closest",
        autosize=True, height=700, margin=dict(t=50, b=40, l=40, r=40),
        legend=dict(x=1.02, y=0.98),
    )
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_xaxes(rangeslider=dict(visible=False), row=1, col=1)

    close_data = {
        "close": [round(float(v), 3) for v in df["close"].tolist()],
        "dates": df.index.strftime("%Y-%m-%d").tolist(),
    }
    return fig, close_data


def build_stock_trade_freq(trades: pd.DataFrame) -> go.Figure | None:
    """个股交易频次横向堆叠柱状图：买/卖分别着色"""
    df = _prepare_trades(trades)
    if df is None:
        return None

    freq = df.groupby(["symbol", "side"]).size().unstack(fill_value=0)
    for col in ["BUY", "SELL"]:
        if col not in freq.columns:
            freq[col] = 0
    freq["total"] = freq["BUY"] + freq["SELL"]
    freq = freq.sort_values("total", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=freq.index, x=freq["BUY"], name="买入",
        orientation="h", marker=dict(color="#d62728"),
        hovertemplate="%{y}<br>买入: %{x}次<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=freq.index, x=freq["SELL"], name="卖出",
        orientation="h", marker=dict(color="#2ca02c"),
        hovertemplate="%{y}<br>卖出: %{x}次<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"个股交易频次 ({len(freq)}只)", font=dict(size=14)),
        barmode="stack",
        autosize=True, height=max(300, len(freq) * 25),
        margin=dict(t=40, b=40, l=80, r=40),
        legend=dict(x=0.85, y=0.98),
        xaxis=dict(title="交易次数"),
    )
    return fig


def build_monthly_buysell(trades: pd.DataFrame) -> go.Figure | None:
    """月度买入金额 vs 卖出金额分组柱状图"""
    df = _prepare_trades(trades)
    if df is None:
        return None

    if "amount" not in df.columns:
        return None

    df["month"] = df.index.to_period("M").to_timestamp()
    monthly = df.groupby(["month", "side"])["amount"].sum().unstack(fill_value=0)
    for col in ["BUY", "SELL"]:
        if col not in monthly.columns:
            monthly[col] = 0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly.index, y=monthly["BUY"], name="买入金额",
        marker=dict(color="#d62728"),
        hovertemplate="%{x|%Y-%m}<br>买入: %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=monthly.index, y=monthly["SELL"], name="卖出金额",
        marker=dict(color="#2ca02c"),
        hovertemplate="%{x|%Y-%m}<br>卖出: %{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="月度买卖金额对比", font=dict(size=14)),
        barmode="group",
        autosize=True, height=420, margin=dict(t=40, b=40, l=40, r=20),
        legend=dict(x=0.75, y=0.98),
        yaxis=dict(title="金额", tickformat=",.0f"),
    )
    return fig


def build_trade_log_data(trades: pd.DataFrame) -> list | None:
    """从 trades 构建调仓记录列表，含每笔交易后的累计持仓量"""
    df = _prepare_trades(trades)
    if df is None:
        return None

    df = df.sort_index()
    positions = {}  # {code: cumulative_qty}
    records = []
    for _, row in df.iterrows():
        dt = row.name
        code = str(row["order_book_id"])
        side = str(row["side"])
        qty = int(row["last_quantity"])
        if code not in positions:
            positions[code] = 0
        if side == "BUY":
            positions[code] += qty
        else:
            positions[code] -= abs(qty)
        amt = row.get("amount", row.get("last_price", 0) * abs(row.get("last_quantity", 0)))
        cost = row.get("transaction_cost", row.get("commission", 0) + row.get("tax", 0))
        records.append({
            "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
            "code": code,
            "name": str(row.get("symbol", "")),
            "side": side,
            "qty": qty,
            "price": round(float(row["last_price"]), 3),
            "amount": round(float(amt), 2),
            "cost": round(float(cost), 4),
            "pos": positions[code],  # 本次交易后该股票的累计持仓
        })
    return records


def build_holdings_snapshot_data(trades: pd.DataFrame) -> dict | None:
    """构建每日持仓快照: {date_str: {code: {name, qty, cost, last_price}}}"""
    df = _prepare_trades(trades)
    if df is None:
        return None

    df = df.sort_index()
    positions = {}     # {code: {"qty": int, "cost": float, "name": str, "last_price": float}}

    snapshots = {}
    for ts, day_trades in df.groupby(df.index):
        for _, row in day_trades.iterrows():
            code = str(row["order_book_id"])
            if code not in positions:
                positions[code] = {"qty": 0, "cost": 0.0, "name": str(row.get("symbol", "")), "last_price": 0.0, "value": 0.0}
            side = str(row["side"])
            qty = int(row["last_quantity"])
            price = float(row["last_price"])
            # 买入金额用正值，卖出金额用负值（不影响市值计算）
            amt = row.get("amount", float(row["last_price"]) * abs(qty))
            if side == "BUY":
                positions[code]["qty"] += qty
                positions[code]["cost"] += float(amt)
            else:
                positions[code]["qty"] -= abs(qty)
                positions[code]["cost"] -= float(amt)
            positions[code]["last_price"] = price

        # 当天收盘后快照: 只保留持有中 (qty>0) 的股票
        hold = {}
        for code, p in positions.items():
            if p["qty"] > 0:
                hold[code] = {
                    "name": p["name"],
                    "qty": p["qty"],
                    "cost": round(p["cost"], 2),
                    "price": round(p["last_price"], 3),
                    "value": round(p["qty"] * p["last_price"], 2),
                }
        if hold:
            snapshots[pd.Timestamp(ts).strftime("%Y-%m-%d")] = hold

    return snapshots


# 调仓明细表的 HTML 骨架（固定结构，JS 动态填充数据）
TRADE_LOG_SHELL = """<div class="trade-log-toolbar">
  <input type="text" id="tl-search" placeholder="搜索股票代码/名称..." oninput="tlRender()" class="tl-search-input">
  <select id="tl-side-filter" onchange="tlRender()" class="tl-select">
    <option value="">全部方向</option>
    <option value="BUY">买入</option>
    <option value="SELL">卖出</option>
  </select>
  <label style="font-size:13px;margin-left:4px;">从</label>
  <input type="date" id="tl-date-from" onchange="tlRender()" class="tl-date-input">
  <label style="font-size:13px;">至</label>
  <input type="date" id="tl-date-to" onchange="tlRender()" class="tl-date-input">
  <span class="tl-info" id="tl-info"></span>
</div>
<div style="overflow-x:auto;">
  <table class="summary-table trade-log-table" id="tl-table">
    <thead>
      <tr>
        <th onclick="tlSort('date')" class="sortable">日期 &#9650;&#9660;</th>
        <th onclick="tlSort('code')" class="sortable">代码</th>
        <th onclick="tlSort('name')" class="sortable">名称</th>
        <th onclick="tlSort('side')" class="sortable">方向</th>
        <th onclick="tlSort('qty')" class="sortable">数量</th>
        <th onclick="tlSort('price')" class="sortable">价格</th>
        <th onclick="tlSort('amount')" class="sortable">金额</th>
        <th onclick="tlSort('cost')" class="sortable">手续费</th>
        <th onclick="tlSort('pos')" class="sortable">持仓(累计)</th>
      </tr>
    </thead>
    <tbody id="tl-tbody"></tbody>
  </table>
</div>
<div class="trade-log-pager" id="tl-pager"></div>"""


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

    # ---- 深度分析：每个实验单独生成 4 张图 + worst days 表 ----
    print("生成深度分析图表...")
    deep_figures_data = OrderedDict()
    for tag, data in results.items():
        pf = data["portfolio"]
        figs = OrderedDict()
        figs["deep_monthly"] = build_monthly_heatmap(pf)
        figs["deep_dist"] = build_returns_distribution(pf)
        figs["deep_underwater"] = build_underwater_plot(pf)
        figs["deep_rolling"] = build_rolling_and_worst(pf)
        figs["deep_worst_html"] = build_worst_days_table(pf)
        deep_figures_data[tag] = figs

    # ---- 持仓与交易：每个实验 3 张图 + 调仓明细表 ----
    print("生成持仓与交易图表...")
    holdings_figures_data = OrderedDict()
    for tag, data in results.items():
        pf = data["portfolio"]
        tr = data["trades"]
        figs = OrderedDict()
        figs["hold_pos"] = build_position_count_chart(pf, tr)
        figs["hold_freq"] = build_stock_trade_freq(tr)
        figs["hold_monthly"] = build_monthly_buysell(tr)
        figs["hold_log_data"] = build_trade_log_data(tr)
        figs["hold_snapshot"] = build_holdings_snapshot_data(tr)
        holdings_figures_data[tag] = figs

    # ---- K线图表：每个实验/每只股票一张图 ----
    print("生成K线图表...")
    kline_figures_data = OrderedDict()
    for tag, data in results.items():
        trades = data["trades"]
        stock_codes = sorted(set(trades["order_book_id"].dropna()))
        kline_figures_data[tag] = OrderedDict()
        for code in stock_codes:
            result = build_stock_kline_chart(trades, code)
            kline_figures_data[tag][code] = result

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

    # 序列化为 JSON（截断浮点精度以缩减体积）
    def _truncate_floats(obj, precision=4):
        if isinstance(obj, dict):
            return {k: _truncate_floats(v, precision) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_truncate_floats(v, precision) for v in obj]
        if isinstance(obj, float):
            return round(obj, precision)
        return obj

    figure_specs_json = OrderedDict()
    for tab_id, specs in tab_figures.items():
        for div_id, fig in specs:
            figure_specs_json[div_id] = _truncate_floats(_json.loads(fig.to_json()))

    # 深度分析：按 tag 嵌套的图表数据
    deep_figures_json = OrderedDict()
    for tag, figs in deep_figures_data.items():
        deep_figures_json[tag] = OrderedDict()
        for chart_type, fig in figs.items():
            if chart_type == "deep_worst_html":
                deep_figures_json[tag][chart_type] = fig  # HTML 字符串
            elif fig is not None:
                deep_figures_json[tag][chart_type] = _truncate_floats(_json.loads(fig.to_json()))
            else:
                deep_figures_json[tag][chart_type] = None

    # K线图表数据
    kline_figures_json = OrderedDict()
    kline_close_json = OrderedDict()
    for tag, figs in kline_figures_data.items():
        kline_figures_json[tag] = OrderedDict()
        kline_close_json[tag] = OrderedDict()
        for code, result in figs.items():
            if result is not None:
                fig, close_data = result
                kline_figures_json[tag][code] = _truncate_floats(_json.loads(fig.to_json()))
                kline_close_json[tag][code] = close_data
            else:
                kline_figures_json[tag][code] = None
                kline_close_json[tag][code] = None

    # 持仓与交易：按 tag 嵌套的图表数据
    holdings_figures_json = OrderedDict()
    for tag, figs in holdings_figures_data.items():
        holdings_figures_json[tag] = OrderedDict()
        for chart_type, fig in figs.items():
            if chart_type == "hold_log_data" or chart_type == "hold_snapshot":
                # 数据在 JS 侧单独注入，这里跳过
                pass
            elif fig is not None:
                holdings_figures_json[tag][chart_type] = _truncate_floats(_json.loads(fig.to_json()))
            else:
                holdings_figures_json[tag][chart_type] = None

    # 调仓记录 + 持仓快照 raw data（直接注入 JS）
    trade_log_json = OrderedDict()
    for tag, figs in holdings_figures_data.items():
        trade_log_json[tag] = {
            "records": figs.get("hold_log_data") or [],
            "snapshots": figs.get("hold_snapshot") or {},
        }

    # ---- 构建各 Tab 的 HTML 占位 div ----
    def _make_div(div_id: str) -> str:
        return f'<div id="{div_id}" class="plotly-graph-div"></div>'

    def _make_plot_container(div_id: str) -> str:
        return f'<div class="plot-container"><button class="fullscreen-btn" onclick="toggleFullscreen(this.parentElement)" title="全屏">&#x26F6;</button>{_make_div(div_id)}</div>'

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

    # 深度分析：第一个实验的默认 chart 数据 + 下拉框
    first_tag = tags[0] if tags else ""
    deep_select_options = "\n".join(
        '<option value="{v}" {sel}>{t}</option>'.format(
            v=_json.dumps(t, ensure_ascii=False).strip('"'),
            sel="selected" if t == first_tag else "",
            t=t,
        )
        for t in tags
    )
    deep_content = f"""<div class="deep-selector-bar">
      <label for="deep-selector">选择策略：</label>
      <select id="deep-selector" onchange="onDeepChange()">
        {deep_select_options}
      </select>
    </div>
    <div class="plot-container">
      <button class="fullscreen-btn" onclick="toggleFullscreen(this.parentElement)" title="全屏">&#x26F6;</button>
      {_make_div("deep_monthly")}
    </div>
    <div class="plot-container">
      <button class="fullscreen-btn" onclick="toggleFullscreen(this.parentElement)" title="全屏">&#x26F6;</button>
      {_make_div("deep_dist")}
    </div>
    <div class="plot-container">
      <button class="fullscreen-btn" onclick="toggleFullscreen(this.parentElement)" title="全屏">&#x26F6;</button>
      {_make_div("deep_underwater")}
    </div>
    <div class="plot-container">
      <button class="fullscreen-btn" onclick="toggleFullscreen(this.parentElement)" title="全屏">&#x26F6;</button>
      {_make_div("deep_rolling")}
    </div>
    <div class="plot-container" id="deep_worst_container">
      <h3>最差 10 天收益</h3>
      <div id="deep_worst_html"></div>
    </div>"""

    # 持仓与交易
    holdings_content = f"""<div class="deep-selector-bar">
      <label for="holdings-selector">选择策略：</label>
      <select id="holdings-selector" onchange="onHoldingsChange()">
        {deep_select_options}
      </select>
    </div>
    <div class="plot-container">
      <button class="fullscreen-btn" onclick="toggleFullscreen(this.parentElement)" title="全屏">&#x26F6;</button>
      {_make_div("hold_pos")}
    </div>
    <div class="plot-container">
      <button class="fullscreen-btn" onclick="toggleFullscreen(this.parentElement)" title="全屏">&#x26F6;</button>
      {_make_div("hold_freq")}
    </div>
    <div class="plot-container">
      <button class="fullscreen-btn" onclick="toggleFullscreen(this.parentElement)" title="全屏">&#x26F6;</button>
      {_make_div("hold_monthly")}
    </div>
    <div class="plot-container" id="holdings-snapshot-panel">
      <h3>持仓快照</h3>
      <div class="snapshot-bar">
        <label>查看日期：</label>
        <select id="snap-date-select" onchange="renderSnapshot()"></select>
        <button onclick="snapNav(-1)" class="snap-nav-btn">前一天</button>
        <button onclick="snapNav(1)" class="snap-nav-btn">后一天</button>
        <span class="tl-info" id="snap-info"></span>
      </div>
      <div style="overflow-x:auto;">
        <table class="summary-table" id="snap-table">
          <thead>
            <tr>
              <th>代码</th><th>名称</th><th>持仓数量</th><th>累计成本</th><th>最新价</th><th>占比</th><th>估算市值</th>
            </tr>
          </thead>
          <tbody id="snap-tbody"></tbody>
        </table>
      </div>
    </div>
    <div class="plot-container">
      <button class="fullscreen-btn" onclick="toggleFullscreen(this.parentElement)" title="全屏">&#x26F6;</button>
      <h3>个股K线分析</h3>
      <div class="snapshot-bar">
        <label>选择股票：</label>
        <select id="kline-stock-select" onchange="renderKlineChart()"></select>
      </div>
      <div class="snapshot-bar" style="flex-wrap:wrap;">
        <label style="font-size:13px;">EMA：</label>
        <span id="kline-ema-presets"></span>
        <input type="number" id="kline-ema-custom" placeholder="自定义周期" min="2" max="500" style="width:100px;padding:3px 8px;font-size:13px;border:1px solid #ddd;border-radius:4px;">
        <button onclick="addCustomEma()" style="padding:3px 12px;font-size:13px;border:1px solid #1a73e8;border-radius:4px;background:#1a73e8;color:white;cursor:pointer;">添加</button>
      </div>
      <div id="kline_chart" class="plotly-graph-div" style="min-height:650px;"></div>
    </div>
    <div class="plot-container">
      <h3>调仓记录明细</h3>
      <div id="hold_log_html">{TRADE_LOG_SHELL}</div>
    </div>"""

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
      padding: 16px; margin-bottom: 20px; position: relative;
    }}
    .plot-container h3 {{
      font-size: 16px; margin-bottom: 12px; color: #2c3e50;
      border-left: 4px solid #1a73e8; padding-left: 10px;
    }}
    .plotly-graph-div {{
      min-height: 550px;
    }}
    .fullscreen-btn {{
      position: absolute; top: 10px; right: 10px; z-index: 50;
      background: rgba(255,255,255,0.9); border: 1px solid #ddd; border-radius: 4px;
      padding: 4px 8px; cursor: pointer; font-size: 16px; color: #555;
      transition: all 0.2s; line-height: 1;
    }}
    .fullscreen-btn:hover {{ background: #1a73e8; color: white; border-color: #1a73e8; }}
    .plot-container:fullscreen {{
      background: white; padding: 20px; overflow: auto;
    }}
    .plot-container:fullscreen .plotly-graph-div {{
      min-height: 90vh !important; height: 90vh !important;
    }}
    .plot-container:-webkit-full-screen {{
      background: white; padding: 20px; overflow: auto;
    }}
    .plot-container:-webkit-full-screen .plotly-graph-div {{
      min-height: 90vh !important; height: 90vh !important;
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
    .deep-selector-bar {{
      display: flex; align-items: center; gap: 12px;
      background: white; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
      padding: 14px 20px; margin-bottom: 20px;
    }}
    .deep-selector-bar label {{
      font-size: 14px; font-weight: 600; color: #2c3e50;
    }}
    .deep-selector-bar select {{
      padding: 8px 16px; font-size: 14px; font-family: inherit;
      border: 1px solid #ddd; border-radius: 4px;
      background: white; cursor: pointer; min-width: 280px;
    }}
    .deep-selector-bar select:focus {{
      outline: none; border-color: #1a73e8; box-shadow: 0 0 0 2px rgba(26,115,232,0.15);
    }}
    .trade-log-toolbar {{
      display: flex; align-items: center; gap: 12px; margin-bottom: 12px; flex-wrap: wrap;
    }}
    .tl-search-input {{
      padding: 8px 14px; font-size: 14px; font-family: inherit;
      border: 1px solid #ddd; border-radius: 4px; min-width: 220px;
    }}
    .tl-search-input:focus {{
      outline: none; border-color: #1a73e8; box-shadow: 0 0 0 2px rgba(26,115,232,0.15);
    }}
    .tl-select {{
      padding: 8px 12px; font-size: 14px; font-family: inherit;
      border: 1px solid #ddd; border-radius: 4px; background: white; cursor: pointer;
    }}
    .tl-info {{
      font-size: 13px; color: #888; margin-left: auto;
    }}
    .tl-date-input {{
      padding: 6px 10px; font-size: 13px; font-family: inherit;
      border: 1px solid #ddd; border-radius: 4px; background: white;
    }}
    .trade-log-table {{
      font-size: 12px;
    }}
    .trade-log-table th.sortable {{
      cursor: pointer; user-select: none;
    }}
    .trade-log-table th.sortable:hover {{
      background: #1a3a5a;
    }}
    .tl-row-buy td {{ background: #fff5f5; }}
    .tl-row-sell td {{ background: #f5fff5; }}
    .tl-row-buy:hover td {{ background: #ffe0e0 !important; }}
    .tl-row-sell:hover td {{ background: #e0ffe0 !important; }}
    .tl-side-buy {{
      color: #d62728; font-weight: 600;
    }}
    .tl-side-sell {{
      color: #2ca02c; font-weight: 600;
    }}
    .tl-num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .trade-log-pager {{
      display: flex; align-items: center; justify-content: center; gap: 16px;
      margin-top: 14px; font-size: 13px;
    }}
    .trade-log-pager button {{
      padding: 6px 18px; border: 1px solid #ddd; border-radius: 4px;
      background: white; cursor: pointer; font-size: 13px; font-family: inherit;
    }}
    .trade-log-pager button:hover:not(:disabled) {{
      background: #1a73e8; color: white; border-color: #1a73e8;
    }}
    .trade-log-pager button:disabled {{
      color: #ccc; cursor: default;
    }}
    #hold_log_html {{
      overflow-x: auto;
    }}
    .snapshot-bar {{
      display: flex; align-items: center; gap: 10px; margin-bottom: 12px; flex-wrap: wrap;
    }}
    .snapshot-bar label {{
      font-size: 14px; font-weight: 600; color: #2c3e50;
    }}
    .snapshot-bar select {{
      padding: 6px 14px; font-size: 14px; font-family: inherit;
      border: 1px solid #ddd; border-radius: 4px; background: white; cursor: pointer;
    }}
    .snap-nav-btn {{
      padding: 6px 14px; border: 1px solid #ddd; border-radius: 4px;
      background: white; cursor: pointer; font-size: 13px; font-family: inherit;
    }}
    .snap-nav-btn:hover {{
      background: #1a73e8; color: white; border-color: #1a73e8;
    }}
    #snap-table {{
      font-size: 13px;
    }}
    #snap-table tbody tr:hover {{
      background: #f0f4ff;
    }}
    .ema-check {{
      display: inline-flex; align-items: center; gap: 2px;
      font-size: 12px; cursor: pointer; margin-right: 6px;
      background: #f0f4ff; padding: 2px 6px; border-radius: 3px;
    }}
    .ema-check input {{ margin: 0; cursor: pointer; }}
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
    <button class="tab-btn" onclick="switchTab(event, 'deep')">深度分析</button>
    <button class="tab-btn" onclick="switchTab(event, 'holdings')">持仓与交易</button>
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

  <div class="tab-panel" id="deep">
    {deep_content}
  </div>

  <div class="tab-panel" id="holdings">
    {holdings_content}
  </div>

  <div class="footer">RQAlpha 策略回测分析 | Powered by Plotly</div>

  <script>
    var FIGURES = {_json.dumps(figure_specs_json)};
    var DEEP_FIGURES = {_json.dumps(deep_figures_json, ensure_ascii=False)};
    var HOLDINGS_FIGURES = {_json.dumps(holdings_figures_json, ensure_ascii=False)};
    var TRADE_LOG_DATA = {_json.dumps(trade_log_json, ensure_ascii=False)};
    var KLINE_FIGURES = {_json.dumps(kline_figures_json, ensure_ascii=False)};
    var KLINE_CLOSE = {_json.dumps(kline_close_json, ensure_ascii=False)};
    var _initialized = {{}};
    var _plotlyConfig = {{responsive: true, displaylogo: false}};
    var _plotlyReady = false;

    function toggleFullscreen(container) {{
      if (!document.fullscreenElement) {{
        container.requestFullscreen().then(function() {{
          setTimeout(function() {{
            container.querySelectorAll('.plotly-graph-div').forEach(function(el) {{
              Plotly.Plots.resize(el);
            }});
          }}, 100);
        }});
      }} else {{
        document.exitFullscreen();
      }}
    }}

    document.addEventListener('fullscreenchange', function() {{
      if (!document.fullscreenElement) {{
        setTimeout(function() {{
          document.querySelectorAll('.plotly-graph-div').forEach(function(el) {{
            if (el._fullData) Plotly.Plots.resize(el);
          }});
        }}, 100);
      }}
    }});

    function renderDeepCharts(tag) {{
      var chartTypes = ['deep_monthly', 'deep_dist', 'deep_underwater', 'deep_rolling'];
      var tagData = DEEP_FIGURES[tag] || {{}};
      chartTypes.forEach(function(ct) {{
        var el = document.getElementById(ct);
        if (!el) return;
        Plotly.purge(el);
        var spec = tagData[ct];
        if (spec) {{
          Plotly.newPlot(ct, spec.data, spec.layout, _plotlyConfig);
        }}
      }});
      // 更新 worst days 表格
      var worstEl = document.getElementById('deep_worst_html');
      if (worstEl) {{
        worstEl.innerHTML = (tagData['deep_worst_html'] || '<p class=no-data>无数据</p>');
      }}
    }}

    function onDeepChange() {{
      if (!_plotlyReady) return;
      var sel = document.getElementById('deep-selector');
      if (sel) renderDeepCharts(sel.value);
    }}

    function renderHoldingsCharts(tag) {{
      var chartTypes = ['hold_pos', 'hold_freq', 'hold_monthly'];
      var tagData = HOLDINGS_FIGURES[tag] || {{}};
      chartTypes.forEach(function(ct) {{
        var el = document.getElementById(ct);
        if (!el) return;
        Plotly.purge(el);
        var spec = tagData[ct];
        if (spec) {{
          Plotly.newPlot(ct, spec.data, spec.layout, _plotlyConfig);
        }}
      }});
      // 更新调仓明细表
      _tlCurrentTag = tag;
      // 设置日期筛选默认范围
      var recs = _tlData();
      if (recs.length > 0) {{
        document.getElementById('tl-date-from').value = recs[0].date;
        document.getElementById('tl-date-to').value = recs[recs.length - 1].date;
      }}
      tlRefreshData();
      // 填充持仓快照日期选择器
      var d = TRADE_LOG_DATA[tag];
      var snaps = (d && d.snapshots) ? d.snapshots : {{}};
      var dates = Object.keys(snaps).sort();
      var sel = document.getElementById('snap-date-select');
      if (sel) {{
        sel.innerHTML = dates.map(function(dt) {{ return '<option value="' + dt + '">' + dt + '</option>'; }}).join('');
        sel.value = dates.length ? dates[dates.length - 1] : '';
      }}
      // 填充K线股票选择器
      var klineData = KLINE_FIGURES[tag] || {{}};
      var klineCodes = Object.keys(klineData).sort();
      var klineSel = document.getElementById('kline-stock-select');
      if (klineSel) {{
        klineSel.innerHTML = klineCodes.map(function(c) {{ return '<option value="' + c + '">' + c + '</option>'; }}).join('');
      }}
      renderSnapshot();
      renderKlineChart();
    }}

    function onHoldingsChange() {{
      if (!_plotlyReady) return;
      var sel = document.getElementById('holdings-selector');
      if (sel) renderHoldingsCharts(sel.value);
    }}

    // ---- 调仓记录明细表 (排序/搜索/分页) ----
    var _tlCurrentTag = '';
    var _tlSortKey = 'date';
    var _tlSortAsc = false;
    var _tlPage = 0;
    var _tlPageSize = 100;

    function _tlData() {{
      var d = TRADE_LOG_DATA[_tlCurrentTag];
      return (d && d.records) ? d.records : [];
    }}

    window.tlSort = function(key) {{
      if (_tlSortKey === key) {{ _tlSortAsc = !_tlSortAsc; }}
      else {{ _tlSortKey = key; _tlSortAsc = true; }}
      _tlPage = 0;
      tlRender();
    }};

    window.tlRefreshData = function() {{
      _tlSortKey = 'date'; _tlSortAsc = false; _tlPage = 0;
      tlRender();
    }};

    window.tlRender = function() {{
      var q = (document.getElementById('tl-search').value || '').toLowerCase();
      var sideFilter = document.getElementById('tl-side-filter').value;
      var dateFrom = document.getElementById('tl-date-from').value;
      var dateTo = document.getElementById('tl-date-to').value;
      var data = _tlData();
      var filtered = data.filter(function(r) {{
        if (q && r.code.toLowerCase().indexOf(q) === -1 && r.name.toLowerCase().indexOf(q) === -1) return false;
        if (sideFilter && r.side !== sideFilter) return false;
        if (dateFrom && r.date < dateFrom) return false;
        if (dateTo && r.date > dateTo) return false;
        return true;
      }});
      var sk = _tlSortKey;
      filtered.sort(function(a, b) {{
        var va = a[sk], vb = b[sk];
        if (typeof va === 'string') return _tlSortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
        return _tlSortAsc ? va - vb : vb - va;
      }});
      var infoEl = document.getElementById('tl-info');
      if (infoEl) infoEl.textContent = '共 ' + filtered.length + ' 笔交易';

      var totalPages = Math.ceil(filtered.length / _tlPageSize);
      if (_tlPage >= totalPages) _tlPage = Math.max(0, totalPages - 1);
      var page = filtered.slice(_tlPage * _tlPageSize, (_tlPage + 1) * _tlPageSize);
      var tbody = document.getElementById('tl-tbody');
      if (!tbody) return;
      tbody.innerHTML = page.map(function(r) {{
        return '<tr class="tl-row-' + r.side.toLowerCase() + '">' +
          '<td>' + r.date + '</td>' +
          '<td>' + r.code + '</td>' +
          '<td>' + r.name + '</td>' +
          '<td class="tl-side-' + r.side.toLowerCase() + '">' + (r.side === 'BUY' ? '买' : '卖') + '</td>' +
          '<td class="tl-num">' + r.qty.toLocaleString() + '</td>' +
          '<td class="tl-num">' + r.price.toFixed(3) + '</td>' +
          '<td class="tl-num">' + r.amount.toLocaleString(undefined, {{minimumFractionDigits:2}}) + '</td>' +
          '<td class="tl-num">' + r.cost.toFixed(4) + '</td>' +
          '<td class="tl-num"><b>' + (r.pos || 0).toLocaleString() + '</b></td>' +
          '</tr>';
      }}).join('');

      var pager = document.getElementById('tl-pager');
      if (!pager) return;
      if (totalPages <= 1) {{ pager.innerHTML = ''; }}
      else {{
        var html = '<button onclick="tlGoPage(' + (_tlPage-1) + ')" ' + (_tlPage===0?'disabled':'') + '>上一页</button>';
        html += '<span> ' + (_tlPage+1) + ' / ' + totalPages + ' </span>';
        html += '<button onclick="tlGoPage(' + (_tlPage+1) + ')" ' + (_tlPage>=totalPages-1?'disabled':'') + '>下一页</button>';
        pager.innerHTML = html;
      }}
    }};

    window.tlGoPage = function(p) {{
      _tlPage = p;
      tlRender();
    }};

    // ---- 持仓快照 ----
    window.renderSnapshot = function() {{
      var sel = document.getElementById('snap-date-select');
      if (!sel || !sel.value) return;
      var date = sel.value;
      var d = TRADE_LOG_DATA[_tlCurrentTag];
      var snaps = (d && d.snapshots) ? d.snapshots : {{}};
      var hold = snaps[date] || {{}};
      var codes = Object.keys(hold).sort();
      var totalValue = 0;
      codes.forEach(function(c) {{ totalValue += hold[c].value || 0; }});
      var rows = [];
      codes.forEach(function(c) {{
        var h = hold[c];
        var pct = totalValue > 0 ? (h.value || 0) / totalValue * 100 : 0;
        rows.push(
          '<tr>' +
          '<td>' + c + '</td>' +
          '<td>' + (h.name || '') + '</td>' +
          '<td class="tl-num">' + (h.qty || 0).toLocaleString() + '</td>' +
          '<td class="tl-num">' + (h.cost || 0).toLocaleString(undefined, {{minimumFractionDigits:2}}) + '</td>' +
          '<td class="tl-num">' + (h.price || 0).toFixed(3) + '</td>' +
          '<td class="tl-num"><b>' + pct.toFixed(1) + '%</b></td>' +
          '<td class="tl-num">' + (h.value || 0).toLocaleString(undefined, {{minimumFractionDigits:2}}) + '</td>' +
          '</tr>'
        );
      }});
      document.getElementById('snap-tbody').innerHTML = rows.join('');
      document.getElementById('snap-info').textContent = '共持有 ' + codes.length + ' 只股票，估算市值 ' + totalValue.toLocaleString(undefined, {{maximumFractionDigits:0}});
    }};

    window.renderKlineChart = function() {{
      var sel = document.getElementById('kline-stock-select');
      if (!sel || !sel.value) return;
      var code = sel.value;
      var tagData = KLINE_FIGURES[_tlCurrentTag] || {{}};
      var spec = tagData[code];
      var el = document.getElementById('kline_chart');
      Plotly.purge(el);
      if (spec) {{
        Plotly.newPlot('kline_chart', spec.data, spec.layout, _plotlyConfig);
        _klineCode = code;
        _customEmas = [];
        buildEmaPresets();
      }}
    }};

    var _klineCode = '';
    var _customEmas = [];

    function _emaPresets() {{
      // 从 Plotly 数据中提取所有 EMA trace 的 name
      var el = document.getElementById('kline_chart');
      if (!el || !el._fullData) return [];
      return el._fullData.filter(function(t) {{
        return t.name && t.name.startsWith('EMA');
      }}).map(function(t) {{ return parseInt(t.name.replace('EMA','')); }});
    }}

    function buildEmaPresets() {{
      var el = document.getElementById('kline_chart');
      var visMap = {{}};
      if (el && el._fullData) {{
        el._fullData.forEach(function(t) {{
          if (t.name && t.name.startsWith('EMA')) {{
            var n = parseInt(t.name.replace('EMA',''));
            visMap[n] = t.visible === true;
          }}
        }});
      }}
      var presets = _emaPresets();
      var html = presets.map(function(n) {{
        var chk = visMap[n] ? ' checked' : '';
        return '<label class="ema-check"><input type="checkbox" value="' + n + '" onchange="toggleEma(' + n + ')"' + chk + '>' + n + '</label>';
      }}).join('');
      document.getElementById('kline-ema-presets').innerHTML = html;
      // 加上已添加的自定义 EMA
      _customEmas.forEach(function(n) {{
        addCustomEmaCheck(n);
      }});
    }}

    window.toggleEma = function(span) {{
      var el = document.getElementById('kline_chart');
      if (!el || !el._fullData) return;
      var match = 'EMA' + span;
      var vis = el._fullData.map(function(t) {{
        if (t.name === match) return !(t.visible === true);
        return t.visible === true;
      }});
      Plotly.restyle(el, 'visible', vis);
    }};

    window.addCustomEma = function() {{
      var input = document.getElementById('kline-ema-custom');
      var n = parseInt(input.value);
      if (!n || n < 2 || n > 500) {{ alert('请输入 2-500 之间的数字'); return; }}
      // 检查是否已存在
      var all = _emaPresets().concat(_customEmas);
      if (all.indexOf(n) !== -1) {{ alert('EMA' + n + ' 已存在'); return; }}

      var closeData = KLINE_CLOSE[_tlCurrentTag] || {{}};
      var cd = closeData[_klineCode];
      if (!cd || !cd.close) return;

      // 计算自定义 EMA
      var alpha = 2 / (n + 1);
      var ema = [];
      var prev = cd.close[0];
      for (var i = 0; i < cd.close.length; i++) {{
        if (i === 0) {{ ema.push(cd.close[0]); }}
        else {{
          var val = alpha * cd.close[i] + (1 - alpha) * prev;
          ema.push(val);
          prev = val;
        }}
      }}

      // 取前 8 种颜色之外的循环颜色
      var colorCycle = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'];
      var color = colorCycle[_customEmas.length % colorCycle.length];

      Plotly.addTraces('kline_chart', {{
        x: cd.dates, y: ema, mode: 'lines',
        name: 'EMA' + n, line: {{color: color, width: 1.2}},
        legendgroup: 'EMA' + n,
        hovertemplate: 'EMA' + n + ': %{{y:.3f}}<extra></extra>',
      }});

      _customEmas.push(n);
      addCustomEmaCheck(n);
      input.value = '';
    }};

    function addCustomEmaCheck(n) {{
      var container = document.getElementById('kline-ema-presets');
      var label = document.createElement('label');
      label.className = 'ema-check';
      label.id = 'ema-check-' + n;
      label.innerHTML = '<input type="checkbox" value="' + n + '" onchange="toggleEma(' + n + ')" checked>' + n +
        ' <span onclick="removeCustomEma(' + n + ')" style="cursor:pointer;color:#d62728;font-size:11px;">×</span>';
      container.appendChild(label);
    }}

    window.removeCustomEma = function(n) {{
      var el = document.getElementById('kline_chart');
      if (!el || !el._fullData) return;
      var match = 'EMA' + n;
      var idx = -1;
      for (var i = 0; i < el._fullData.length; i++) {{
        if (el._fullData[i].name === match) {{ idx = i; break; }}
      }}
      if (idx >= 0) {{
        Plotly.deleteTraces('kline_chart', idx);
      }}
      _customEmas = _customEmas.filter(function(v) {{ return v !== n; }});
      var chk = document.getElementById('ema-check-' + n);
      if (chk) chk.remove();
    }};

    window.snapNav = function(dir) {{
      var sel = document.getElementById('snap-date-select');
      if (!sel || sel.options.length === 0) return;
      var idx = sel.selectedIndex + dir;
      if (idx < 0) idx = 0;
      if (idx >= sel.options.length) idx = sel.options.length - 1;
      sel.selectedIndex = idx;
      renderSnapshot();
    }};

    function ensurePlots(tabName) {{
      if (!_plotlyReady) return;
      if (_initialized[tabName] && tabName !== 'deep' && tabName !== 'holdings') {{
        var panel = document.getElementById(tabName);
        if (panel) {{
          panel.querySelectorAll('.plotly-graph-div').forEach(function(el) {{
            if (el.id && el.id.startsWith('deep_')) return;
            Plotly.Plots.resize(el);
          }});
        }}
        return;
      }}
      if (tabName === 'deep' || tabName === 'holdings') {{
        // 深度/持仓 tab 不设 initialized 标记，允许切换下拉重新渲染
        if (tabName === 'deep') {{
          var sel = document.getElementById('deep-selector');
          var tag = sel ? sel.value : '';
          if (tag) renderDeepCharts(tag);
        }} else {{
          var sel = document.getElementById('holdings-selector');
          var tag = sel ? sel.value : '';
          if (tag) renderHoldingsCharts(tag);
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
          Plotly.newPlot(el.id, spec.data, spec.layout, _plotlyConfig);
        }}
      }});
    }}

    function switchTab(evt, tabName) {{
      document.querySelectorAll('.tab-panel').forEach(function(p) {{ p.classList.remove('active'); }});
      document.querySelectorAll('.tab-btn').forEach(function(b) {{ b.classList.remove('active'); }});
      document.getElementById(tabName).classList.add('active');
      evt.currentTarget.classList.add('active');
      setTimeout(function() {{ ensurePlots(tabName); }}, 50);
    }}

    function onPlotlyLoaded() {{
      _plotlyReady = true;
      ensurePlots('overview');
    }}
  </script>
  <script>{plotly_js}</script>
  <script>onPlotlyLoaded();</script>
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
