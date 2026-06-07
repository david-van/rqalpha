#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
参数网格实验轻量分析。

适用于 score_grid_m_days_decay / score_sdl_grid_compact / score_sdl_grid_full
这类组合数量较多的扫描。脚本只读取已有 pkl 和 meta.json，不运行回测。
"""

import argparse
import html
import json
import pickle
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from my_strategy.common_file import project_root


BASE_DIR = Path(project_root) / "my_strategy" / "strategies" / "batch_results" / "七星高照"
DEFAULT_SWEEP = "score_grid_m_days_decay"
TOP_N = 20
TRADING_DAYS_PER_YEAR = 250


def load_pickle(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def load_meta(sweep_dir: Path) -> dict:
    meta_path = sweep_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"缺少 meta.json: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def tag_values(meta: dict, tag: str) -> list:
    values = meta.get("tag_values", {}).get(tag, [])
    return values if isinstance(values, list) else [values]


def load_results(sweep_dir: Path, meta: dict) -> tuple[pd.DataFrame, dict]:
    rows = []
    raw_results = {}
    dims = meta.get("dimensions", [])

    for pkl_path in sorted(sweep_dir.glob("*.pkl")):
        tag = pkl_path.stem
        data = load_pickle(pkl_path)
        summary = data.get("summary", {})
        trades = data.get("trades", pd.DataFrame())
        values = tag_values(meta, tag)

        annualized = summary.get("annualized_returns", np.nan)
        max_drawdown = summary.get("max_drawdown", np.nan)
        calmar = annualized / max_drawdown if max_drawdown and not pd.isna(max_drawdown) else np.nan
        row = {
            "tag": tag,
            "total_returns": summary.get("total_returns", np.nan),
            "annualized": annualized,
            "sharpe": summary.get("sharpe", np.nan),
            "sortino": summary.get("sortino", np.nan),
            "max_drawdown": max_drawdown,
            "calmar": calmar,
            "turnover": summary.get("turnover", np.nan),
            "trades": len(trades) if isinstance(trades, pd.DataFrame) else 0,
        }

        for idx, dim in enumerate(dims):
            row[dim["name"]] = values[idx] if idx < len(values) else np.nan

        rows.append(row)
        raw_results[tag] = data

    if not rows:
        raise RuntimeError(f"未找到任何 pkl 文件: {sweep_dir}")

    df = pd.DataFrame(rows)
    df["score"] = (
        df["sharpe"].fillna(-999)
        + df["annualized"].fillna(-999)
        - df["max_drawdown"].fillna(999)
    )
    return df, raw_results


def yearly_returns(portfolio: pd.DataFrame) -> pd.Series:
    if not isinstance(portfolio, pd.DataFrame) or portfolio.empty:
        return pd.Series(dtype=float)
    if "unit_net_value" not in portfolio.columns:
        return pd.Series(dtype=float)

    nv = portfolio["unit_net_value"].dropna().copy()
    if not isinstance(nv.index, pd.DatetimeIndex):
        nv.index = pd.to_datetime(nv.index)
    nv = nv.sort_index()

    rows = {}
    for year, group in nv.groupby(nv.index.year):
        if len(group) >= 2:
            rows[int(year)] = float(group.iloc[-1] / group.iloc[0] - 1)
    return pd.Series(rows, dtype=float)


def drawdown_series(portfolio: pd.DataFrame) -> pd.Series:
    if not isinstance(portfolio, pd.DataFrame) or portfolio.empty:
        return pd.Series(dtype=float)
    if "unit_net_value" not in portfolio.columns:
        return pd.Series(dtype=float)

    nv = portfolio["unit_net_value"].dropna().copy()
    if not isinstance(nv.index, pd.DatetimeIndex):
        nv.index = pd.to_datetime(nv.index)
    nv = nv.sort_index()
    return nv / nv.cummax() - 1


def candidate_tags(df: pd.DataFrame, limit: int = TOP_N) -> list[str]:
    tags = []
    rankings = [
        df.sort_values("annualized", ascending=False).head(5),
        df.sort_values("sharpe", ascending=False).head(5),
        df.sort_values("calmar", ascending=False).head(5),
        df.sort_values("max_drawdown", ascending=True).head(5),
    ]
    for ranking in rankings:
        for tag in ranking["tag"]:
            if tag not in tags:
                tags.append(tag)
            if len(tags) >= limit:
                return tags
    return tags


def pct(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{value:.{digits}%}"


def num(value, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{value:.{digits}f}"


def table_html(df: pd.DataFrame, columns: list[str], title: str | None = None, limit: int | None = None) -> str:
    view = df.head(limit).copy() if limit else df.copy()
    header = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    rows = []
    for _, row in view.iterrows():
        cells = []
        for col in columns:
            value = row.get(col, "")
            if col in {"annualized", "total_returns", "max_drawdown", "worst_year", "avg_year"}:
                text = pct(value)
            elif col in {"sharpe", "sortino", "calmar", "score"}:
                text = num(value)
            else:
                text = "" if pd.isna(value) else str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")

    heading = f"<h3>{html.escape(title)}</h3>" if title else ""
    return f"""
{heading}
<table class="data-table">
  <thead><tr>{header}</tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
"""


def build_overview_fig(df: pd.DataFrame) -> go.Figure:
    top_ret = df.sort_values("annualized", ascending=False).head(TOP_N)
    top_sharpe = df.sort_values("sharpe", ascending=False).head(TOP_N)
    top_calmar = df.sort_values("calmar", ascending=False).head(TOP_N)
    low_dd = df.sort_values("max_drawdown", ascending=True).head(TOP_N)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("年化收益 Top", "夏普 Top", "Calmar Top", "低回撤 Top"),
        horizontal_spacing=0.12,
        vertical_spacing=0.14,
    )
    specs = [
        (top_ret, "annualized", "年化收益", ".1%", 1, 1),
        (top_sharpe, "sharpe", "夏普", ".3f", 1, 2),
        (top_calmar, "calmar", "Calmar", ".3f", 2, 1),
        (low_dd, "max_drawdown", "最大回撤", ".1%", 2, 2),
    ]
    for data, key, name, fmt, row, col in specs:
        fig.add_trace(
            go.Bar(
                x=data[key],
                y=data["tag"],
                orientation="h",
                text=[f"{v:{fmt}}" for v in data[key]],
                textposition="auto",
                hovertemplate=f"%{{y}}<br>{name}=%{{x:{fmt}}}<extra></extra>",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.update_yaxes(autorange="reversed", row=row, col=col)

    fig.update_layout(title="参数组合总览排行", height=900, margin=dict(l=90, r=30, t=80, b=40))
    return fig


def _matrix(df: pd.DataFrame, x_name: str, y_name: str, metric: str) -> tuple[list, list, np.ndarray]:
    x_vals = sorted(df[x_name].dropna().unique())
    y_vals = sorted(df[y_name].dropna().unique())
    mat = np.full((len(y_vals), len(x_vals)), np.nan)
    for _, row in df.iterrows():
        if pd.isna(row.get(x_name)) or pd.isna(row.get(y_name)):
            continue
        x_idx = x_vals.index(row[x_name])
        y_idx = y_vals.index(row[y_name])
        mat[y_idx, x_idx] = row.get(metric, np.nan)
    return x_vals, y_vals, mat


def _heatmap_trace(
    df: pd.DataFrame,
    x_name: str,
    y_name: str,
    metric: str,
    fmt: str,
    title: str,
    showscale: bool = True,
) -> go.Heatmap:
    x_vals, y_vals, mat = _matrix(df, x_name, y_name, metric)
    text = np.array([[f"{v:{fmt}}" if not np.isnan(v) else "" for v in row] for row in mat])
    colorscale = "RdYlGn_r" if metric == "max_drawdown" else "RdYlGn"
    return go.Heatmap(
        z=mat,
        x=[str(v) for v in x_vals],
        y=[str(v) for v in y_vals],
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorscale=colorscale,
        showscale=showscale,
        colorbar=dict(title=title),
        hovertemplate=f"{y_name}=%{{y}}<br>{x_name}=%{{x}}<br>{title}=%{{z:{fmt}}}<extra></extra>",
    )


def build_heatmap_fig(df: pd.DataFrame, meta: dict) -> go.Figure | None:
    dims = meta.get("dimensions", [])
    if len(dims) < 2:
        return None

    y_name = dims[0]["name"]
    x_name = dims[1]["name"]
    slice_name = dims[2]["name"] if len(dims) >= 3 else None
    metrics = [
        ("annualized", ".1%", "年化收益"),
        ("sharpe", ".3f", "夏普"),
        ("max_drawdown", ".1%", "最大回撤"),
    ]

    if slice_name is None:
        fig = make_subplots(rows=1, cols=3, subplot_titles=[m[2] for m in metrics], horizontal_spacing=0.12)
        for col, (metric, fmt, title) in enumerate(metrics, start=1):
            fig.add_trace(_heatmap_trace(df, x_name, y_name, metric, fmt, title), row=1, col=col)
        fig.update_layout(title=f"{dims[0]['display']} × {dims[1]['display']} 热力图", height=520)
        return fig

    slice_vals = sorted(df[slice_name].dropna().unique())
    subplot_titles = []
    for value in slice_vals:
        for _, _, title in metrics:
            subplot_titles.append(f"{dims[2]['display']}={value} / {title}")

    fig = make_subplots(
        rows=len(slice_vals),
        cols=3,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.10,
        vertical_spacing=0.08,
    )
    for row_idx, value in enumerate(slice_vals, start=1):
        part = df[df[slice_name] == value]
        for col_idx, (metric, fmt, title) in enumerate(metrics, start=1):
            fig.add_trace(
                _heatmap_trace(part, x_name, y_name, metric, fmt, title, showscale=row_idx == 1),
                row=row_idx,
                col=col_idx,
            )

    fig.update_layout(
        title=f"{dims[0]['display']} × {dims[1]['display']}，按 {dims[2]['display']} 分片",
        height=max(520, 320 * len(slice_vals)),
    )
    return fig


def build_yearly_fig(candidate_df: pd.DataFrame, raw_results: dict) -> tuple[go.Figure | None, pd.DataFrame]:
    yearly_rows = []
    stats = []
    for tag in candidate_df["tag"]:
        yearly = yearly_returns(raw_results[tag].get("portfolio", pd.DataFrame()))
        for year, value in yearly.items():
            yearly_rows.append({"tag": tag, "year": int(year), "return": float(value)})
        if not yearly.empty:
            stats.append({
                "tag": tag,
                "profitable_years": int((yearly > 0).sum()),
                "years": int(len(yearly)),
                "worst_year": float(yearly.min()),
                "avg_year": float(yearly.mean()),
            })

    stats_df = pd.DataFrame(stats)
    if not yearly_rows:
        return None, stats_df

    yearly_df = pd.DataFrame(yearly_rows)
    pivot = yearly_df.pivot_table(index="tag", columns="year", values="return", aggfunc="first")
    pivot = pivot.reindex(candidate_df["tag"])
    text = pivot.map(lambda v: "" if pd.isna(v) else f"{v:.1%}")

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=pivot.index,
        text=text.values,
        texttemplate="%{text}",
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="年度收益"),
        hovertemplate="tag=%{y}<br>year=%{x}<br>return=%{z:.2%}<extra></extra>",
    ))
    fig.update_layout(title="候选参数年度收益稳定性", height=max(520, len(pivot) * 28 + 160))
    return fig, stats_df


def build_candidate_fig(candidate_df: pd.DataFrame, raw_results: dict) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("候选参数净值曲线", "候选参数回撤曲线"),
        vertical_spacing=0.12,
    )
    for tag in candidate_df["tag"]:
        pf = raw_results[tag].get("portfolio", pd.DataFrame())
        if not isinstance(pf, pd.DataFrame) or pf.empty or "unit_net_value" not in pf.columns:
            continue
        nv = pf["unit_net_value"].dropna().copy()
        if not isinstance(nv.index, pd.DatetimeIndex):
            nv.index = pd.to_datetime(nv.index)
        nv = nv.sort_index()
        fig.add_trace(go.Scatter(x=nv.index, y=nv, name=tag, mode="lines"), row=1, col=1)
        dd = nv / nv.cummax() - 1
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=tag, mode="lines", showlegend=False), row=2, col=1)

    fig.update_yaxes(title_text="单位净值", row=1, col=1)
    fig.update_yaxes(title_text="回撤", tickformat=".0%", row=2, col=1)
    fig.update_layout(title="候选参数对比", height=760, hovermode="x unified")
    return fig


def fig_html(fig: go.Figure | None) -> str:
    if fig is None:
        return "<p class='empty'>无可展示图表</p>"
    return fig.to_html(include_plotlyjs=False, full_html=False, config={"responsive": True})


def build_dashboard(sweep_name: str, sweep_dir: Path, meta: dict, df: pd.DataFrame, raw_results: dict) -> Path:
    dims = meta.get("dimensions", [])
    param_cols = [dim["name"] for dim in dims]
    metric_cols = ["annualized", "sharpe", "max_drawdown", "calmar", "total_returns", "trades", "score"]
    table_cols = ["tag", *param_cols, *metric_cols]

    ranked_df = df.sort_values("score", ascending=False)
    candidates = candidate_tags(df)
    candidate_df = df[df["tag"].isin(candidates)].copy()
    candidate_df["candidate_order"] = candidate_df["tag"].map({tag: idx for idx, tag in enumerate(candidates)})
    candidate_df = candidate_df.sort_values("candidate_order")

    overview_fig = build_overview_fig(df)
    heatmap_fig = build_heatmap_fig(df, meta)
    yearly_fig, yearly_stats = build_yearly_fig(candidate_df, raw_results)
    candidate_fig = build_candidate_fig(candidate_df, raw_results)

    yearly_stats_html = ""
    if not yearly_stats.empty:
        yearly_stats = yearly_stats.sort_values(["profitable_years", "worst_year", "avg_year"], ascending=False)
        yearly_stats_html = table_html(
            yearly_stats,
            ["tag", "profitable_years", "years", "worst_year", "avg_year"],
            "候选年度稳定性统计",
        )

    summary_text = (
        f"扫描: {html.escape(sweep_name)} | "
        f"实验数: {len(df)} | "
        f"维度: {html.escape(', '.join(dim['display'] for dim in dims))} | "
        f"生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}"
    )

    output_path = sweep_dir / "param_grid_dashboard.html"
    plotly_js = get_plotlyjs()
    dashboard = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{html.escape(sweep_name)} 参数网格分析</title>
  <style>
    body {{ margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; background: #f6f7f9; color: #222; }}
    header {{ padding: 18px 24px; background: #17202a; color: white; }}
    header h1 {{ margin: 0 0 8px; font-size: 22px; }}
    header p {{ margin: 0; color: #d7dde5; font-size: 13px; }}
    .tabs {{ display: flex; gap: 4px; padding: 12px 16px 0; background: white; border-bottom: 1px solid #ddd; position: sticky; top: 0; z-index: 2; }}
    .tab-btn {{ border: 0; background: #edf0f4; padding: 10px 14px; cursor: pointer; font-size: 14px; border-radius: 6px 6px 0 0; }}
    .tab-btn.active {{ background: #17202a; color: white; }}
    .tab {{ display: none; padding: 18px; }}
    .tab.active {{ display: block; }}
    .panel {{ background: white; border: 1px solid #e0e3e8; border-radius: 6px; padding: 14px; margin-bottom: 16px; }}
    .data-table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    .data-table th, .data-table td {{ border: 1px solid #e1e4e8; padding: 6px 8px; text-align: right; white-space: nowrap; }}
    .data-table th:first-child, .data-table td:first-child {{ text-align: left; }}
    .data-table th {{ background: #f0f2f5; position: sticky; top: 45px; }}
    .table-wrap {{ overflow-x: auto; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    h3 {{ margin: 8px 0 10px; font-size: 15px; }}
    .empty {{ color: #666; }}
  </style>
</head>
<body>
  <header>
    <h1>参数网格分析</h1>
    <p>{summary_text}</p>
  </header>
  <nav class="tabs">
    <button class="tab-btn active" onclick="showTab('overview', this)">总览排行</button>
    <button class="tab-btn" onclick="showTab('heatmap', this)">参数热力图</button>
    <button class="tab-btn" onclick="showTab('yearly', this)">年度稳定性</button>
    <button class="tab-btn" onclick="showTab('candidates', this)">候选参数对比</button>
  </nav>

  <section id="overview" class="tab active">
    <div class="panel">{fig_html(overview_fig)}</div>
    <div class="panel table-wrap">{table_html(ranked_df, table_cols, "综合评分排行", TOP_N)}</div>
  </section>

  <section id="heatmap" class="tab">
    <div class="panel">{fig_html(heatmap_fig)}</div>
  </section>

  <section id="yearly" class="tab">
    <div class="panel">{fig_html(yearly_fig)}</div>
    <div class="panel table-wrap">{yearly_stats_html}</div>
  </section>

  <section id="candidates" class="tab">
    <div class="panel">{fig_html(candidate_fig)}</div>
    <div class="panel table-wrap">{table_html(candidate_df, table_cols, "候选参数核心指标")}</div>
  </section>

  <script>{plotly_js}</script>
  <script>
    function showTab(id, btn) {{
      document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById(id).classList.add('active');
      btn.classList.add('active');
      setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
    }}
  </script>
</body>
</html>"""
    output_path.write_text(dashboard, encoding="utf-8")
    return output_path


def list_sweeps() -> None:
    rows = []
    if BASE_DIR.exists():
        for path in sorted(BASE_DIR.iterdir()):
            meta_path = path / "meta.json"
            if path.is_dir() and meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                dims = ", ".join(dim["display"] for dim in meta.get("dimensions", []))
                rows.append((path.name, dims))
    if not rows:
        print("未找到带 meta.json 的扫描目录")
        return
    print("可用扫描:")
    for name, dims in rows:
        print(f"  - {name}: {dims}")


def parse_args():
    parser = argparse.ArgumentParser(description="参数网格实验轻量分析")
    parser.add_argument("--sweep", default=DEFAULT_SWEEP, help="扫描目录名称")
    parser.add_argument("--list-sweeps", action="store_true", help="列出可用扫描后退出")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_sweeps:
        list_sweeps()
        return

    sweep_dir = BASE_DIR / args.sweep
    if not sweep_dir.exists():
        raise FileNotFoundError(f"扫描目录不存在: {sweep_dir}")

    meta = load_meta(sweep_dir)
    df, raw_results = load_results(sweep_dir, meta)
    output_path = build_dashboard(args.sweep, sweep_dir, meta, df, raw_results)
    print(f"已生成: {output_path}")


if __name__ == "__main__":
    main()
