#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate a compact HTML report for the 513030 open-gap CSV analysis."""

from __future__ import annotations

import argparse
import html
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots


DEFAULT_RESULT_DIR = Path(__file__).with_name("batch_results") / "gap_strategy_513030"

COL_THRESHOLD = "阈值"
COL_YEAR = "年份"
COL_BUCKET = "高低开分档"
COL_TRADES = "交易次数"
COL_WIN_RATE = "胜率"
COL_AVG_RETURN = "平均收益率"
COL_MEDIAN_RETURN = "中位数收益率"
COL_TOTAL_RETURN = "累计收益率"
COL_MAX_DRAWDOWN = "最大回撤"
COL_PROFIT_LOSS_RATIO = "盈亏比"

COLORS = {
    "blue": "#2563eb",
    "green": "#059669",
    "red": "#dc2626",
    "orange": "#ea580c",
    "purple": "#7c3aed",
    "gray": "#64748b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a lightweight Plotly HTML report from gap strategy CSVs."
    )
    parser.add_argument(
        "--result-dir",
        default=str(DEFAULT_RESULT_DIR),
        help="Directory containing threshold_overall.csv, threshold_by_year.csv, "
        "gap_bucket_by_year.csv and daily_trades_source.csv.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output HTML path. Default: <result-dir>/gap_strategy_report.html",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Comma separated thresholds for equity curves, e.g. -0.005,0,0.003. "
        "Default: auto picks representative thresholds.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Rows to show in the top threshold table.",
    )
    return parser.parse_args()


def read_csv(result_dir: Path, file_name: str) -> pd.DataFrame:
    path = result_dir / file_name
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def load_data(result_dir: Path) -> dict[str, pd.DataFrame]:
    data = {
        "overall": read_csv(result_dir, "threshold_overall.csv"),
        "yearly": read_csv(result_dir, "threshold_by_year.csv"),
        "buckets": read_csv(result_dir, "gap_bucket_by_year.csv"),
        "trades": read_csv(result_dir, "daily_trades_source.csv"),
    }
    data["trades"]["date"] = pd.to_datetime(data["trades"]["date"])

    numeric_cols = {
        "overall": [
            COL_THRESHOLD,
            COL_TRADES,
            COL_WIN_RATE,
            COL_AVG_RETURN,
            COL_MEDIAN_RETURN,
            COL_TOTAL_RETURN,
            COL_MAX_DRAWDOWN,
            COL_PROFIT_LOSS_RATIO,
        ],
        "yearly": [
            COL_YEAR,
            COL_THRESHOLD,
            COL_TRADES,
            COL_WIN_RATE,
            COL_AVG_RETURN,
            COL_TOTAL_RETURN,
            COL_MAX_DRAWDOWN,
            COL_PROFIT_LOSS_RATIO,
        ],
        "buckets": [
            COL_YEAR,
            COL_BUCKET,
            "平均高低开",
            "平均高低开比例",
            COL_TRADES,
            COL_WIN_RATE,
            COL_AVG_RETURN,
            COL_TOTAL_RETURN,
            COL_MAX_DRAWDOWN,
        ],
        "trades": [
            "year",
            "open",
            "close",
            "prev_close",
            "gap",
            "gap_pct",
            "gross_return",
            "net_return",
            "pnl_price",
            "volume",
            "total_turnover",
        ],
    }
    for key, cols in numeric_cols.items():
        for col in cols:
            if col in data[key].columns:
                data[key][col] = pd.to_numeric(data[key][col], errors="coerce")

    data["overall"] = data["overall"].sort_values(COL_THRESHOLD).reset_index(drop=True)
    data["yearly"] = data["yearly"].sort_values([COL_YEAR, COL_THRESHOLD]).reset_index(
        drop=True
    )
    data["buckets"] = data["buckets"].sort_values([COL_YEAR, COL_BUCKET]).reset_index(
        drop=True
    )
    data["trades"] = data["trades"].sort_values("date").reset_index(drop=True)
    return data


def pct(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.{digits}%}"


def num(value: float | int | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.{digits}f}"


def threshold_label(value: float) -> str:
    return f"{value:.3f}"


def pick_thresholds(overall: pd.DataFrame, raw_thresholds: str | None) -> list[float]:
    available = sorted(float(x) for x in overall[COL_THRESHOLD].dropna().unique())
    if not available:
        return []

    if raw_thresholds:
        requested = [float(x.strip()) for x in raw_thresholds.split(",") if x.strip()]
        picked = []
        for value in requested:
            nearest = min(available, key=lambda x: abs(x - value))
            if nearest not in picked:
                picked.append(nearest)
        return picked[:6]

    best_return = float(
        overall.loc[overall[COL_TOTAL_RETURN].idxmax(), COL_THRESHOLD]
    )
    best_drawdown = float(
        overall.loc[overall[COL_MAX_DRAWDOWN].idxmax(), COL_THRESHOLD]
    )
    anchors = [best_return, 0.003, 0.0, best_drawdown]
    picked = []
    for value in anchors:
        nearest = min(available, key=lambda x: abs(x - value))
        if nearest not in picked:
            picked.append(nearest)
    return picked[:4]


def make_threshold_overview(overall: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "阈值 vs 累计收益",
            "阈值 vs 最大回撤",
            "阈值 vs 胜率 / 平均单笔收益",
            "阈值 vs 交易次数",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": True}, {"secondary_y": False}],
        ],
        vertical_spacing=0.13,
        horizontal_spacing=0.1,
    )

    x = overall[COL_THRESHOLD]
    fig.add_trace(
        go.Scatter(
            x=x,
            y=overall[COL_TOTAL_RETURN],
            name="累计收益率",
            mode="lines+markers",
            line=dict(color=COLORS["blue"], width=2),
            hovertemplate="阈值 %{x:.3f}<br>累计收益 %{y:.2%}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=overall[COL_MAX_DRAWDOWN],
            name="最大回撤",
            mode="lines+markers",
            line=dict(color=COLORS["red"], width=2),
            hovertemplate="阈值 %{x:.3f}<br>最大回撤 %{y:.2%}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=overall[COL_WIN_RATE],
            name="胜率",
            mode="lines",
            line=dict(color=COLORS["green"], width=2),
            hovertemplate="阈值 %{x:.3f}<br>胜率 %{y:.2%}<extra></extra>",
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=overall[COL_AVG_RETURN],
            name="平均单笔收益",
            mode="lines",
            line=dict(color=COLORS["orange"], width=2, dash="dot"),
            hovertemplate="阈值 %{x:.3f}<br>平均收益 %{y:.3%}<extra></extra>",
        ),
        row=2,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Bar(
            x=x,
            y=overall[COL_TRADES],
            name="交易次数",
            marker_color=COLORS["gray"],
            hovertemplate="阈值 %{x:.3f}<br>交易次数 %{y}<extra></extra>",
        ),
        row=2,
        col=2,
    )

    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=1, col=2)
    fig.update_yaxes(tickformat=".0%", row=2, col=1, secondary_y=False)
    fig.update_yaxes(tickformat=".2%", row=2, col=1, secondary_y=True)
    fig.update_xaxes(title_text="高低开阈值", tickformat=".3f")
    fig.update_layout(
        height=760,
        margin=dict(l=50, r=40, t=80, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
    )
    return fig


def make_equity_curves(trades: pd.DataFrame, thresholds: list[float]) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.68, 0.32],
        subplot_titles=("按阈值过滤后的累计净值", "对应回撤"),
    )
    palette = [
        COLORS["blue"],
        COLORS["green"],
        COLORS["orange"],
        COLORS["purple"],
        COLORS["red"],
        COLORS["gray"],
    ]

    for idx, threshold in enumerate(thresholds):
        selected_return = trades["net_return"].where(trades["gap"] >= threshold, 0.0)
        equity = (1.0 + selected_return.fillna(0.0)).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        label = f"阈值 {threshold_label(threshold)}"
        color = palette[idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=trades["date"],
                y=equity,
                name=label,
                mode="lines",
                line=dict(color=color, width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>净值 %{y:.3f}<extra>"
                + label
                + "</extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=trades["date"],
                y=drawdown,
                name=f"{label} 回撤",
                mode="lines",
                line=dict(color=color, width=1.5),
                showlegend=False,
                hovertemplate="%{x|%Y-%m-%d}<br>回撤 %{y:.2%}<extra>"
                + label
                + "</extra>",
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="净值", row=1, col=1)
    fig.update_yaxes(title_text="回撤", tickformat=".0%", row=2, col=1)
    fig.update_layout(
        height=680,
        margin=dict(l=55, r=35, t=70, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
    )
    return fig


def make_yearly_heatmap(yearly: pd.DataFrame) -> go.Figure:
    pivot_return = yearly.pivot(
        index=COL_YEAR, columns=COL_THRESHOLD, values=COL_TOTAL_RETURN
    ).sort_index()
    pivot_trades = yearly.pivot(
        index=COL_YEAR, columns=COL_THRESHOLD, values=COL_TRADES
    ).reindex_like(pivot_return)

    fig = go.Figure(
        data=go.Heatmap(
            x=[threshold_label(float(x)) for x in pivot_return.columns],
            y=[str(int(y)) for y in pivot_return.index],
            z=pivot_return.values,
            customdata=pivot_trades.values,
            colorscale=[
                [0.0, "#b91c1c"],
                [0.48, "#f8fafc"],
                [1.0, "#047857"],
            ],
            zmid=0,
            colorbar=dict(title="累计收益"),
            hovertemplate=(
                "年份 %{y}<br>阈值 %{x}<br>累计收益 %{z:.2%}"
                "<br>交易次数 %{customdata}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="年度阈值热力图",
        height=430,
        margin=dict(l=45, r=35, t=55, b=55),
        xaxis_title="高低开阈值",
        yaxis_title="年份",
    )
    return fig


def make_bucket_heatmaps(buckets: pd.DataFrame) -> go.Figure:
    pivot_return = buckets.pivot(
        index=COL_YEAR, columns=COL_BUCKET, values=COL_AVG_RETURN
    ).sort_index()
    pivot_count = buckets.pivot(
        index=COL_YEAR, columns=COL_BUCKET, values=COL_TRADES
    ).reindex_like(pivot_return)

    x_labels = [threshold_label(float(x)) for x in pivot_return.columns]
    y_labels = [str(int(y)) for y in pivot_return.index]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("分档平均单笔收益", "分档交易次数"),
        horizontal_spacing=0.12,
    )
    fig.add_trace(
        go.Heatmap(
            x=x_labels,
            y=y_labels,
            z=pivot_return.values,
            colorscale=[
                [0.0, "#b91c1c"],
                [0.5, "#f8fafc"],
                [1.0, "#047857"],
            ],
            zmid=0,
            colorbar=dict(title="平均收益", x=0.46),
            hovertemplate="年份 %{y}<br>分档 %{x}<br>平均收益 %{z:.3%}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=x_labels,
            y=y_labels,
            z=pivot_count.values,
            colorscale="Blues",
            colorbar=dict(title="次数"),
            hovertemplate="年份 %{y}<br>分档 %{x}<br>交易次数 %{z}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(height=480, margin=dict(l=45, r=35, t=65, b=70))
    fig.update_xaxes(title_text="高低开分档")
    fig.update_yaxes(title_text="年份", row=1, col=1)
    return fig


def make_daily_scatter(trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=trades["gap"],
            y=trades["net_return"],
            mode="markers",
            marker=dict(
                size=8,
                color=trades["year"],
                colorscale="Viridis",
                opacity=0.72,
                colorbar=dict(title="年份"),
            ),
            customdata=np.stack(
                [
                    trades["date"].dt.strftime("%Y-%m-%d"),
                    trades["open"],
                    trades["close"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "%{customdata[0]}<br>高低开 %{x:.3f}<br>当日收益 %{y:.2%}"
                "<br>开盘 %{customdata[1]:.3f}<br>收盘 %{customdata[2]:.3f}"
                "<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line_width=1, line_color="#94a3b8")
    fig.add_vline(x=0, line_width=1, line_color="#94a3b8")
    fig.update_layout(
        title="每日高低开 vs 开盘买入收盘卖出收益",
        height=470,
        margin=dict(l=55, r=35, t=60, b=50),
        xaxis_title="高低开价差",
        yaxis_title="当日收益率",
        yaxis_tickformat=".1%",
    )
    return fig


def figure_div(fig: go.Figure, div_id: str) -> str:
    return fig.to_html(
        include_plotlyjs=False,
        full_html=False,
        div_id=div_id,
        config={"responsive": True, "displaylogo": False},
    )


def make_top_table(overall: pd.DataFrame, top_n: int) -> str:
    display = (
        overall.sort_values(COL_TOTAL_RETURN, ascending=False)
        .head(top_n)
        .loc[
            :,
            [
                COL_THRESHOLD,
                COL_TRADES,
                COL_WIN_RATE,
                COL_AVG_RETURN,
                COL_TOTAL_RETURN,
                COL_MAX_DRAWDOWN,
                COL_PROFIT_LOSS_RATIO,
            ],
        ]
        .copy()
    )
    display[COL_THRESHOLD] = display[COL_THRESHOLD].map(lambda x: f"{x:.3f}")
    for col in [COL_WIN_RATE, COL_AVG_RETURN, COL_TOTAL_RETURN, COL_MAX_DRAWDOWN]:
        display[col] = display[col].map(lambda x: pct(x, 2 if col != COL_AVG_RETURN else 3))
    display[COL_PROFIT_LOSS_RATIO] = display[COL_PROFIT_LOSS_RATIO].map(
        lambda x: num(x, 2)
    )

    rows = []
    for _, row in display.iterrows():
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        rows.append(f"<tr>{cells}</tr>")
    headers = "".join(f"<th>{html.escape(str(col))}</th>" for col in display.columns)
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def make_cards(data: dict[str, pd.DataFrame], thresholds: list[float]) -> str:
    overall = data["overall"]
    trades = data["trades"]
    best_return_row = overall.loc[overall[COL_TOTAL_RETURN].idxmax()]
    best_dd_row = overall.loc[overall[COL_MAX_DRAWDOWN].idxmax()]
    best_win_row = overall.loc[overall[COL_WIN_RATE].idxmax()]
    date_range = f"{trades['date'].min():%Y-%m-%d} 至 {trades['date'].max():%Y-%m-%d}"
    selected_labels = ", ".join(threshold_label(x) for x in thresholds) or "-"

    cards = [
        ("样本区间", date_range, f"{len(trades)} 个交易日"),
        (
            "累计收益最佳阈值",
            threshold_label(float(best_return_row[COL_THRESHOLD])),
            pct(float(best_return_row[COL_TOTAL_RETURN])),
        ),
        (
            "回撤最小阈值",
            threshold_label(float(best_dd_row[COL_THRESHOLD])),
            pct(float(best_dd_row[COL_MAX_DRAWDOWN])),
        ),
        (
            "胜率最高阈值",
            threshold_label(float(best_win_row[COL_THRESHOLD])),
            pct(float(best_win_row[COL_WIN_RATE])),
        ),
        ("净值曲线阈值", selected_labels, "可用 --thresholds 调整"),
    ]
    return "".join(
        "<div class='kpi'>"
        f"<div class='kpi-title'>{html.escape(title)}</div>"
        f"<div class='kpi-value'>{html.escape(value)}</div>"
        f"<div class='kpi-sub'>{html.escape(sub)}</div>"
        "</div>"
        for title, value, sub in cards
    )


def build_html(
    data: dict[str, pd.DataFrame],
    thresholds: list[float],
    top_n: int,
    result_dir: Path,
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    figures = [
        ("threshold_overview", make_threshold_overview(data["overall"])),
        ("equity_curves", make_equity_curves(data["trades"], thresholds)),
        ("yearly_heatmap", make_yearly_heatmap(data["yearly"])),
        ("bucket_heatmaps", make_bucket_heatmaps(data["buckets"])),
        ("daily_scatter", make_daily_scatter(data["trades"])),
    ]
    figure_html = "\n".join(
        "<section class='panel'>"
        f"<button type='button' class='fullscreen-btn' data-target='{div_id}' "
        "onclick='toggleFullscreen(this)'>全屏</button>"
        f"{figure_div(fig, div_id)}"
        "</section>"
        for div_id, fig in figures
    )

    csv_files = [
        "threshold_overall.csv",
        "threshold_by_year.csv",
        "gap_bucket_by_year.csv",
        "daily_trades_source.csv",
    ]
    csv_links = "".join(
        f"<span class='chip'>{html.escape(name)}</span>" for name in csv_files
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>513030 高低开策略分析</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --text: #172033;
      --muted: #64748b;
      --border: #d8dee9;
      --accent: #2563eb;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 28px 34px 18px;
      background: #ffffff;
      border-bottom: 1px solid var(--border);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 26px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
    }}
    main {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 22px 24px 38px;
    }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .kpi, .panel, .table-panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    .kpi {{
      padding: 14px 16px;
      min-width: 0;
    }}
    .kpi-title {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .kpi-value {{
      font-size: 20px;
      font-weight: 700;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .kpi-sub {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 4px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .panel {{
      padding: 10px 12px;
      margin: 16px 0;
      position: relative;
    }}
    .fullscreen-btn {{
      position: absolute;
      top: 12px;
      right: 14px;
      z-index: 5;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.92);
      color: #334155;
      cursor: pointer;
      font-size: 12px;
      line-height: 1;
      padding: 7px 10px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
    }}
    .fullscreen-btn:hover {{
      border-color: var(--accent);
      color: var(--accent);
    }}
    .panel:fullscreen {{
      width: 100vw;
      height: 100vh;
      margin: 0;
      padding: 48px 18px 18px;
      border: 0;
      border-radius: 0;
      box-sizing: border-box;
      overflow: hidden;
      background: #ffffff;
    }}
    .panel:-webkit-full-screen {{
      width: 100vw;
      height: 100vh;
      margin: 0;
      padding: 48px 18px 18px;
      border: 0;
      border-radius: 0;
      box-sizing: border-box;
      overflow: hidden;
      background: #ffffff;
    }}
    .panel:fullscreen .plotly-graph-div,
    .panel:-webkit-full-screen .plotly-graph-div {{
      height: calc(100vh - 66px) !important;
      width: 100% !important;
    }}
    .panel:fullscreen .fullscreen-btn,
    .panel:-webkit-full-screen .fullscreen-btn {{
      top: 12px;
      right: 18px;
    }}
    .table-panel {{
      padding: 18px;
      margin: 16px 0;
      overflow-x: auto;
    }}
    .table-panel h2 {{
      margin: 0 0 12px;
      font-size: 18px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid #e7ebf2;
      padding: 8px 10px;
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    th {{
      color: #334155;
      background: #f8fafc;
      font-weight: 650;
    }}
    .chips {{
      margin-top: 10px;
    }}
    .chip {{
      display: inline-block;
      margin: 5px 6px 0 0;
      padding: 4px 8px;
      border: 1px solid var(--border);
      border-radius: 999px;
      color: var(--muted);
      background: #f8fafc;
      font-size: 12px;
    }}
    @media (max-width: 1100px) {{
      .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 640px) {{
      header {{ padding: 22px 18px 16px; }}
      main {{ padding: 16px 12px 28px; }}
      .kpis {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <script>{get_plotlyjs()}</script>
  <header>
    <h1>513030 高低开策略分析</h1>
    <div class="meta">
      生成时间：{html.escape(generated_at)} · 数据目录：{html.escape(str(result_dir))}
      <div class="chips">{csv_links}</div>
    </div>
  </header>
  <main>
    <section class="kpis">{make_cards(data, thresholds)}</section>
    <section class="table-panel">
      <h2>累计收益排名前 {top_n} 的阈值</h2>
      {make_top_table(data["overall"], top_n)}
    </section>
    {figure_html}
  </main>
  <script>
    function getFullscreenElement() {{
      return document.fullscreenElement || document.webkitFullscreenElement || null;
    }}

    function requestPanelFullscreen(panel) {{
      if (panel.requestFullscreen) {{
        return panel.requestFullscreen();
      }}
      if (panel.webkitRequestFullscreen) {{
        return panel.webkitRequestFullscreen();
      }}
      return Promise.resolve();
    }}

    function exitPanelFullscreen() {{
      if (document.exitFullscreen) {{
        return document.exitFullscreen();
      }}
      if (document.webkitExitFullscreen) {{
        return document.webkitExitFullscreen();
      }}
      return Promise.resolve();
    }}

    function resizePlotlyPanel(panel) {{
      if (!window.Plotly || !panel) return;
      panel.querySelectorAll('.plotly-graph-div').forEach(function(el) {{
        Plotly.Plots.resize(el);
      }});
    }}

    function toggleFullscreen(button) {{
      var panel = button.closest('.panel');
      if (!panel) return;
      if (getFullscreenElement() === panel) {{
        exitPanelFullscreen();
      }} else {{
        requestPanelFullscreen(panel).then(function() {{
          setTimeout(function() {{ resizePlotlyPanel(panel); }}, 80);
        }});
      }}
    }}

    function syncFullscreenButtons() {{
      var active = getFullscreenElement();
      document.querySelectorAll('.fullscreen-btn').forEach(function(button) {{
        var panel = button.closest('.panel');
        button.textContent = panel && panel === active ? '退出全屏' : '全屏';
        if (panel) resizePlotlyPanel(panel);
      }});
      if (active) {{
        setTimeout(function() {{ resizePlotlyPanel(active); }}, 120);
      }}
    }}

    document.addEventListener('fullscreenchange', syncFullscreenButtons);
    document.addEventListener('webkitfullscreenchange', syncFullscreenButtons);
    window.addEventListener('resize', function() {{
      var active = getFullscreenElement();
      if (active) resizePlotlyPanel(active);
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    out_path = Path(args.out) if args.out else result_dir / "gap_strategy_report.html"

    data = load_data(result_dir)
    thresholds = pick_thresholds(data["overall"], args.thresholds)
    html_text = build_html(data, thresholds, args.top_n, result_dir.resolve())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")
    print(f"HTML report written: {out_path.resolve()}")


if __name__ == "__main__":
    main()
