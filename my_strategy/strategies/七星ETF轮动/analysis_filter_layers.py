#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
过滤器分层实验总览。

读取 filter_layer_1 / 2 / 3 / 4_5 的回测结果，汇总为一个轻量 HTML。
不运行回测，只分析已有 pkl 和 meta.json。
"""

import html
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = (
    PROJECT_ROOT
    / "my_strategy"
    / "strategies"
    / "batch_results"
    / "七星高照"
)
LAYER_DIRS = ["filter_layer_1", "filter_layer_2", "filter_layer_3", "filter_layer_4_5"]
OUTPUT_FILE = RESULT_ROOT / "filter_layers_summary.html"


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def _trade_cost(trades: pd.DataFrame) -> float:
    if not isinstance(trades, pd.DataFrame) or trades.empty:
        return 0.0
    if "transaction_cost" in trades.columns:
        return float(trades["transaction_cost"].fillna(0).sum())

    total = 0.0
    for col in ["commission", "tax"]:
        if col in trades.columns:
            total += float(trades[col].fillna(0).sum())
    return total


def _yearly_returns(portfolio: pd.DataFrame, tag: str, n_filters: int, filters: str) -> list[dict]:
    if not isinstance(portfolio, pd.DataFrame) or portfolio.empty:
        return []
    if "unit_net_value" not in portfolio.columns:
        return []

    nv = portfolio["unit_net_value"].dropna().copy()
    if not isinstance(nv.index, pd.DatetimeIndex):
        nv.index = pd.to_datetime(nv.index)
    nv = nv.sort_index()

    rows = []
    for year, group in nv.groupby(nv.index.year):
        if len(group) < 2:
            continue
        rows.append({
            "tag": tag,
            "n_filters": n_filters,
            "filters": filters,
            "year": int(year),
            "return": float(group.iloc[-1] / group.iloc[0] - 1),
        })
    return rows


def load_layer_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    yearly_rows = []
    seen_tags = set()
    filter_labels = {}

    for layer_name in LAYER_DIRS:
        layer_dir = RESULT_ROOT / layer_name
        meta_path = layer_dir / "meta.json"
        if not layer_dir.exists() or not meta_path.exists():
            print(f"跳过缺失目录: {layer_dir}")
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        filter_sets = meta.get("filter_sets", {})
        filter_labels.update(meta.get("filter_labels", {}))

        for pkl_path in sorted(layer_dir.glob("*.pkl")):
            tag = pkl_path.stem
            if tag in seen_tags:
                continue
            seen_tags.add(tag)

            data = _load_pickle(pkl_path)
            summary = data.get("summary", {})
            trades = data.get("trades", pd.DataFrame())
            keys = list(filter_sets.get(tag, []))
            filters = "+".join(filter_labels.get(k, k) for k in keys) if keys else "核心过滤器全关"
            cost = _trade_cost(trades)
            max_drawdown = summary.get("max_drawdown", np.nan)
            annualized = summary.get("annualized_returns", np.nan)
            calmar = annualized / max_drawdown if max_drawdown and not pd.isna(max_drawdown) else np.nan
            n_filters = len(keys)

            rows.append({
                "layer": meta.get("layer", ""),
                "source_dir": layer_name,
                "tag": tag,
                "filter_keys": keys,
                "n_filters": n_filters,
                "filters": filters,
                "total_returns": summary.get("total_returns", np.nan),
                "annualized": annualized,
                "sharpe": summary.get("sharpe", np.nan),
                "sortino": summary.get("sortino", np.nan),
                "max_drawdown": max_drawdown,
                "calmar": calmar,
                "turnover": summary.get("turnover", np.nan),
                "trades": len(trades) if isinstance(trades, pd.DataFrame) else 0,
                "cost": cost,
            })
            yearly_rows.extend(_yearly_returns(data.get("portfolio", pd.DataFrame()), tag, n_filters, filters))

    if not rows:
        raise RuntimeError("未找到任何 filter_layer 结果，请先运行分层回测。")

    df = pd.DataFrame(rows)
    if "base_off" not in set(df["tag"]):
        raise RuntimeError("未找到 base_off，无法计算相对基准指标。")

    base = df[df["tag"] == "base_off"].iloc[0]
    df["delta_total"] = df["total_returns"] - base["total_returns"]
    df["delta_annualized"] = df["annualized"] - base["annualized"]
    df["delta_sharpe"] = df["sharpe"] - base["sharpe"]
    df["drawdown_improve"] = base["max_drawdown"] - df["max_drawdown"]
    df["delta_trades"] = df["trades"] - base["trades"]
    df["delta_cost"] = df["cost"] - base["cost"]
    yearly_df = pd.DataFrame(yearly_rows)
    return df.sort_values(["n_filters", "tag"]).reset_index(drop=True), yearly_df


def compute_marginal_contribution(df: pd.DataFrame) -> pd.DataFrame:
    lookup = {frozenset(row["filter_keys"]): row for _, row in df.iterrows()}
    filter_order = ["pp", "vol", "sm", "sdl", "sr"]
    filter_labels = {
        "pp": "盈利保护",
        "vol": "成交量",
        "sm": "短期动量",
        "sdl": "单日跌幅",
        "sr": "得分范围",
    }
    rows = []

    for key in filter_order:
        diffs = []
        for keyset, base_row in lookup.items():
            if key in keyset:
                continue
            target_set = frozenset(set(keyset) | {key})
            target_row = lookup.get(target_set)
            if target_row is None:
                continue
            diffs.append({
                "delta_annualized": target_row["annualized"] - base_row["annualized"],
                "delta_sharpe": target_row["sharpe"] - base_row["sharpe"],
                "drawdown_improve": base_row["max_drawdown"] - target_row["max_drawdown"],
                "delta_trades": target_row["trades"] - base_row["trades"],
                "delta_cost": target_row["cost"] - base_row["cost"],
            })

        d = pd.DataFrame(diffs)
        rows.append({
            "filter": filter_labels[key],
            "delta_annualized": d["delta_annualized"].mean(),
            "delta_sharpe": d["delta_sharpe"].mean(),
            "drawdown_improve": d["drawdown_improve"].mean(),
            "delta_trades": d["delta_trades"].mean(),
            "delta_cost": d["delta_cost"].mean(),
        })

    return pd.DataFrame(rows)


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for i, current in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if i == j:
                continue
            better_or_equal = (
                other["annualized"] >= current["annualized"]
                and other["max_drawdown"] <= current["max_drawdown"]
            )
            strictly_better = (
                other["annualized"] > current["annualized"]
                or other["max_drawdown"] < current["max_drawdown"]
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return df.loc[keep].sort_values("annualized", ascending=False)


def build_figures(df: pd.DataFrame, marginal: pd.DataFrame, yearly_df: pd.DataFrame) -> dict:
    figures = {}
    top_ann = df.sort_values("annualized", ascending=False).head(12)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_ann["tag"],
        y=top_ann["annualized"],
        text=[f"{v:.1%}" for v in top_ann["annualized"]],
        textposition="outside",
        marker_color="#1f77b4",
        hovertemplate="%{x}<br>年化收益: %{y:.2%}<extra></extra>",
    ))
    fig.update_layout(
        title="Top 12 年化收益组合",
        height=460,
        yaxis_tickformat=".0%",
        xaxis_tickangle=-30,
        margin=dict(t=70, l=60, r=30, b=110),
    )
    figures["top_ann"] = fig

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["max_drawdown"],
        y=df["annualized"],
        mode="markers+text",
        text=df["tag"],
        textposition="top center",
        marker=dict(
            size=np.clip(df["cost"] / max(df["cost"].max(), 1) * 34 + 8, 8, 42),
            color=df["n_filters"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="过滤器数"),
            opacity=0.82,
            line=dict(color="white", width=1),
        ),
        customdata=np.stack([df["filters"], df["sharpe"], df["calmar"], df["cost"]], axis=-1),
        hovertemplate=(
            "%{text}<br>%{customdata[0]}"
            "<br>年化: %{y:.2%}<br>回撤: %{x:.2%}"
            "<br>夏普: %{customdata[1]:.3f}<br>Calmar: %{customdata[2]:.3f}"
            "<br>成本: %{customdata[3]:.2f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="收益-回撤散点图",
        height=620,
        xaxis_title="最大回撤",
        yaxis_title="年化收益",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        margin=dict(t=70, l=70, r=60, b=60),
    )
    figures["risk_return"] = fig

    best_ann = df.loc[df.groupby("n_filters")["annualized"].idxmax()].sort_values("n_filters")
    best_sharpe = df.loc[df.groupby("n_filters")["sharpe"].idxmax()].sort_values("n_filters")
    best_calmar = df.loc[df.groupby("n_filters")["calmar"].idxmax()].sort_values("n_filters")

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("每层最佳年化", "每层最佳夏普", "每层最佳 Calmar"),
        horizontal_spacing=0.10,
    )
    fig.add_trace(go.Bar(
        x=best_ann["n_filters"].astype(str),
        y=best_ann["annualized"],
        text=best_ann["tag"],
        marker_color="#1f77b4",
        hovertemplate="过滤器数 %{x}<br>%{text}<br>年化 %{y:.2%}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=best_sharpe["n_filters"].astype(str),
        y=best_sharpe["sharpe"],
        text=best_sharpe["tag"],
        marker_color="#2ca02c",
        hovertemplate="过滤器数 %{x}<br>%{text}<br>夏普 %{y:.3f}<extra></extra>",
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=best_calmar["n_filters"].astype(str),
        y=best_calmar["calmar"],
        text=best_calmar["tag"],
        marker_color="#ff7f0e",
        hovertemplate="过滤器数 %{x}<br>%{text}<br>Calmar %{y:.3f}<extra></extra>",
    ), row=1, col=3)
    fig.update_layout(title="分层最优组合", height=470, showlegend=False)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    figures["layer_best"] = fig

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("平均年化贡献", "平均夏普贡献", "平均回撤改善"),
        horizontal_spacing=0.10,
    )
    fig.add_trace(go.Bar(
        x=marginal["filter"],
        y=marginal["delta_annualized"],
        marker_color="#1f77b4",
        hovertemplate="%{x}<br>年化贡献 %{y:+.2%}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=marginal["filter"],
        y=marginal["delta_sharpe"],
        marker_color="#2ca02c",
        hovertemplate="%{x}<br>夏普贡献 %{y:+.3f}<extra></extra>",
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=marginal["filter"],
        y=marginal["drawdown_improve"],
        marker_color="#ff7f0e",
        hovertemplate="%{x}<br>回撤改善 %{y:+.2%}<extra></extra>",
    ), row=1, col=3)
    fig.update_layout(title="过滤器平均边际贡献", height=470, showlegend=False)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=1, col=3)
    figures["marginal"] = fig

    if not yearly_df.empty:
        tag_order = df.sort_values("annualized", ascending=False)["tag"].tolist()
        yearly_pivot = yearly_df.pivot_table(index="tag", columns="year", values="return", aggfunc="first")
        yearly_pivot = yearly_pivot.reindex([tag for tag in tag_order if tag in yearly_pivot.index])
        heat_text = yearly_pivot.map(lambda v: "" if pd.isna(v) else f"{v:.1%}")

        fig = go.Figure(data=go.Heatmap(
            z=yearly_pivot.values,
            x=[str(y) for y in yearly_pivot.columns],
            y=yearly_pivot.index,
            text=heat_text.values,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmid=0,
            colorbar=dict(title="年度收益"),
            hovertemplate="%{y}<br>%{x}: %{z:.2%}<extra></extra>",
        ))
        fig.update_layout(
            title="年度收益热力图",
            height=max(680, len(yearly_pivot.index) * 24),
            margin=dict(t=70, l=130, r=40, b=55),
            xaxis_title="年份",
            yaxis_title="组合",
        )
        figures["yearly_heatmap"] = fig

        top_tags = df.sort_values("annualized", ascending=False).head(8)["tag"].tolist()
        fig = go.Figure()
        for tag in top_tags:
            yd = yearly_df[yearly_df["tag"] == tag].sort_values("year")
            fig.add_trace(go.Scatter(
                x=yd["year"].astype(str),
                y=yd["return"],
                mode="lines+markers",
                name=tag,
                hovertemplate=f"{tag}<br>%{{x}}: %{{y:.2%}}<extra></extra>",
            ))
        fig.update_layout(
            title="Top 8 组合年度收益",
            height=520,
            yaxis_tickformat=".0%",
            xaxis_title="年份",
            yaxis_title="年度收益",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="top", y=-0.18),
            margin=dict(t=70, l=70, r=30, b=120),
        )
        figures["yearly_top_lines"] = fig

    return figures


def _fmt_pct(value) -> str:
    return "" if pd.isna(value) else f"{value:.2%}"


def _fmt_num(value, digits=3) -> str:
    return "" if pd.isna(value) else f"{value:.{digits}f}"


def table_html(df: pd.DataFrame, title: str, limit: int | None = None) -> str:
    cols = [
        ("组合", "tag"),
        ("过滤器数", "n_filters"),
        ("开启过滤器", "filters"),
        ("累计收益", "total_returns"),
        ("年化收益", "annualized"),
        ("夏普", "sharpe"),
        ("最大回撤", "max_drawdown"),
        ("Calmar", "calmar"),
        ("交易次数", "trades"),
        ("成本", "cost"),
    ]
    data = df.head(limit) if limit else df
    header = "".join(f"<th>{html.escape(label)}</th>" for label, _ in cols)
    rows = []
    for _, row in data.iterrows():
        cells = []
        for label, col in cols:
            value = row[col]
            if col in {"total_returns", "annualized", "max_drawdown"}:
                text = _fmt_pct(value)
            elif col in {"sharpe", "calmar"}:
                text = _fmt_num(value)
            elif col == "cost":
                text = _fmt_num(value, 2)
            else:
                text = str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"""
    <section class="panel">
      <h2>{html.escape(title)}</h2>
      <div class="table-wrap">
        <table>
          <thead><tr>{header}</tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </section>
    """


def yearly_best_table_html(yearly_df: pd.DataFrame) -> str:
    if yearly_df.empty:
        return """
        <section class="panel">
          <h2>每年最佳组合</h2>
          <p>无年度收益数据。</p>
        </section>
        """

    idx = yearly_df.groupby("year")["return"].idxmax()
    best = yearly_df.loc[idx].sort_values("year")
    header = "".join(f"<th>{label}</th>" for label in ["年份", "最佳组合", "开启过滤器", "年度收益", "过滤器数"])
    rows = []
    for _, row in best.iterrows():
        rows.append(
            "<tr>"
            f"<td>{int(row['year'])}</td>"
            f"<td>{html.escape(str(row['tag']))}</td>"
            f"<td>{html.escape(str(row['filters']))}</td>"
            f"<td>{_fmt_pct(row['return'])}</td>"
            f"<td>{int(row['n_filters'])}</td>"
            "</tr>"
        )
    return f"""
    <section class="panel">
      <h2>每年最佳组合</h2>
      <div class="table-wrap">
        <table>
          <thead><tr>{header}</tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </section>
    """


def _chart_panel(div_id: str) -> str:
    return f'<section class="panel"><div id="{div_id}" class="chart"></div></section>'


def render_html(df: pd.DataFrame, marginal: pd.DataFrame, yearly_df: pd.DataFrame, figures: dict) -> str:
    figure_json = {}
    for key, fig in figures.items():
        figure_json[key] = json.loads(fig.to_json())

    top_ann = df.sort_values("annualized", ascending=False)
    top_calmar = df.sort_values("calmar", ascending=False)
    pareto = pareto_front(df)

    best_ann = top_ann.iloc[0]
    best_calmar = top_calmar.iloc[0]
    all_on = df[df["tag"] == "all_on"].iloc[0] if "all_on" in set(df["tag"]) else None
    all_on_text = ""
    if all_on is not None:
        all_on_text = (
            f"all_on 年化 {_fmt_pct(all_on['annualized'])}，"
            f"低于最佳收益组合 {best_ann['tag']} 的 {_fmt_pct(best_ann['annualized'])}。"
        )

    overview_content = f"""
    <section class="panel">
      <h2>核心结论</h2>
      <div class="summary-grid">
        <div class="metric"><b>收益最高</b><span>{html.escape(best_ann['tag'])}：年化 {_fmt_pct(best_ann['annualized'])}，回撤 {_fmt_pct(best_ann['max_drawdown'])}</span></div>
        <div class="metric"><b>Calmar 最好</b><span>{html.escape(best_calmar['tag'])}：Calmar {_fmt_num(best_calmar['calmar'])}，年化 {_fmt_pct(best_calmar['annualized'])}</span></div>
        <div class="metric"><b>全开对照</b><span>{html.escape(all_on_text)}</span></div>
        <div class="metric"><b>最强边际过滤器</b><span>单日跌幅的平均年化贡献最高，建议作为核心过滤器观察。</span></div>
      </div>
    </section>
    {_chart_panel("top_ann")}
    {_chart_panel("risk_return")}
    {_chart_panel("layer_best")}
    {_chart_panel("marginal")}
    """

    yearly_content = f"""
    {_chart_panel("yearly_heatmap") if "yearly_heatmap" in figures else ""}
    {_chart_panel("yearly_top_lines") if "yearly_top_lines" in figures else ""}
    {yearly_best_table_html(yearly_df)}
    """

    ranking_content = f"""
    {table_html(top_ann, "Top 年化收益组合", 12)}
    {table_html(top_calmar, "Top Calmar 组合", 12)}
    {table_html(pareto, "收益-回撤 Pareto 组合")}
    """

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>过滤器分层实验总览</title>
  <style>
    body {{ font-family: "Microsoft YaHei", Arial, sans-serif; margin: 0; background: #f4f6f8; color: #222; }}
    header {{ background: #263544; color: white; padding: 22px 30px; }}
    header h1 {{ margin: 0 0 8px; font-size: 24px; }}
    header p {{ margin: 3px 0; color: #d8e0e8; }}
    main {{ padding: 20px 26px 36px; }}
    .tab-bar {{ display: flex; gap: 0; background: #fff; border-bottom: 1px solid #dfe5ec; position: sticky; top: 0; z-index: 10; padding: 0 24px; }}
    .tab-btn {{ border: 0; background: transparent; padding: 14px 22px; cursor: pointer; font-size: 14px; color: #53616f; border-bottom: 3px solid transparent; }}
    .tab-btn.active {{ color: #1267b1; border-bottom-color: #1267b1; font-weight: 600; }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}
    .panel {{ background: white; border-radius: 8px; padding: 16px; margin-bottom: 18px; box-shadow: 0 1px 5px rgba(0,0,0,.08); }}
    .panel h2 {{ margin: 0 0 12px; font-size: 17px; color: #263544; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }}
    .metric {{ background: #f8fafc; border: 1px solid #e3e8ef; border-radius: 6px; padding: 12px; }}
    .metric b {{ display: block; font-size: 14px; margin-bottom: 6px; }}
    .metric span {{ font-size: 13px; color: #53616f; }}
    .chart {{ min-height: 460px; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th {{ background: #263544; color: white; text-align: left; padding: 9px 10px; white-space: nowrap; }}
    td {{ border-bottom: 1px solid #e7ecf1; padding: 8px 10px; white-space: nowrap; }}
    tr:nth-child(even) td {{ background: #fafbfd; }}
  </style>
</head>
<body>
  <header>
    <h1>过滤器分层实验总览</h1>
    <p>读取目录：{html.escape(str(RESULT_ROOT))}</p>
    <p>唯一组合：{len(df)} 组；base_off 去重后只保留一份。</p>
  </header>
  <nav class="tab-bar">
    <button class="tab-btn active" onclick="switchTab('overview', this)">总览</button>
    <button class="tab-btn" onclick="switchTab('yearly', this)">年度收益</button>
    <button class="tab-btn" onclick="switchTab('ranking', this)">排行榜</button>
  </nav>
  <main>
    <div id="overview" class="tab-panel active">{overview_content}</div>
    <div id="yearly" class="tab-panel">{yearly_content}</div>
    <div id="ranking" class="tab-panel">{ranking_content}</div>
  </main>
  <script>{get_plotlyjs()}</script>
  <script>
    const FIGURES = {json.dumps(figure_json, ensure_ascii=False)};
    const config = {{responsive: true, displaylogo: false}};
    Object.entries(FIGURES).forEach(([id, spec]) => {{
      Plotly.newPlot(id, spec.data, spec.layout, config);
    }});
    function switchTab(id, btn) {{
      document.querySelectorAll('.tab-panel').forEach(function(panel) {{
        panel.classList.remove('active');
      }});
      document.querySelectorAll('.tab-btn').forEach(function(button) {{
        button.classList.remove('active');
      }});
      document.getElementById(id).classList.add('active');
      btn.classList.add('active');
      setTimeout(function() {{
        document.getElementById(id).querySelectorAll('.chart').forEach(function(el) {{
          if (el._fullData) Plotly.Plots.resize(el);
        }});
      }}, 50);
    }}
  </script>
</body>
</html>"""


def main():
    df, yearly_df = load_layer_results()
    marginal = compute_marginal_contribution(df)
    figures = build_figures(df, marginal, yearly_df)
    html_text = render_html(df, marginal, yearly_df, figures)
    OUTPUT_FILE.write_text(html_text, encoding="utf-8")
    print(f"唯一组合: {len(df)}")
    print(f"年度收益记录: {len(yearly_df)}")
    print(f"已保存: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
