#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""QDII ETF 折溢价套利 — 可视化报告。
从分析结果 CSV 生成 Plotly HTML 报告。

用法:
    python qdii_premium_report.py
    python qdii_premium_report.py --etf 513100.SH
"""

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

CACHE_DIR = Path(__file__).resolve().parent / 'cache'
RESULTS_DIR = Path(__file__).resolve().parent / 'analysis_results'
REPORT_DIR = Path(__file__).resolve().parent / 'reports'

COLORS = {
    "blue": "#2563eb", "green": "#059669", "red": "#dc2626",
    "orange": "#ea580c", "purple": "#7c3aed", "gray": "#64748b",
}
PALETTE = [
    "#2563eb", "#059669", "#dc2626", "#ea580c", "#7c3aed", "#0891b2",
    "#be185d", "#4f46e5", "#b45309", "#0d9488", "#9333ea", "#c2410c",
]

# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_results() -> dict[str, pd.DataFrame]:
    dfs = {}
    for name in ['premium_overall', 'premium_forward', 'premium_cycles', 'top_premium_etfs']:
        path = RESULTS_DIR / f'{name}.csv'
        if path.exists():
            dfs[name] = pd.read_csv(path)
        else:
            dfs[name] = pd.DataFrame()
    return dfs


def load_etf_data(ts_code: str) -> pd.DataFrame | None:
    """加载单只 ETF 合并数据。"""
    from qdii_premium_analysis import load_etf_data as _load
    return _load(ts_code)


# ---------------------------------------------------------------------------
# 格式化
# ---------------------------------------------------------------------------
def pct(v, d=2): return f"{v:.{d}%}" if pd.notna(v) else "-"
def num(v, d=2): return f"{v:.{d}f}" if pd.notna(v) else "-"


# ---------------------------------------------------------------------------
# 图表
# ---------------------------------------------------------------------------
def make_kpi_cards(best_df: pd.DataFrame) -> str:
    """顶部 KPI 卡片。"""
    cards = []
    for _, row in best_df.iterrows():
        etf = row['order_book_id']
        name = etf.split('.')[0]
        cards.append((
            f"{etf}",
            f"买 {row['buy_threshold']:.4f} / 卖 {row['sell_threshold']:.4f}",
            f"收益 {pct(row['total_return'])} | 年化 {pct(row['annualized_return'])} | "
            f"{int(row['n_cycles'])}轮 | 胜率 {pct(row['win_rate'])}"
        ))
    return "".join(
        f"<div class='kpi'><div class='kpi-title'>{t}</div>"
        f"<div class='kpi-value'>{v}</div><div class='kpi-sub'>{s}</div></div>"
        for t, v, s in cards
    )


def make_premium_ts(dfs_etf: dict[str, pd.DataFrame]) -> go.Figure:
    """折溢价率时序图：每个 ETF 一条折溢价率曲线 + 水平线标注买卖阈值。"""
    fig = go.Figure()
    for i, (code, df) in enumerate(dfs_etf.items()):
        if df is None or df.empty:
            continue
        c = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter(
            x=df['trade_date'], y=df['premium'] * 100,
            name=code.split('.')[0],
            mode='lines', line=dict(color=c, width=1.2),
            hovertemplate=f"{code}<br>%{{x|%Y-%m-%d}}<br>溢价率 %{{y:.2f}}%<extra></extra>",
        ))
    fig.add_hline(y=0, line_width=1, line_color='#94a3b8', line_dash='dash')
    fig.update_layout(
        title="折溢价率时序 (premium = close/unit_nav - 1)",
        height=420, margin=dict(l=50, r=30, t=50, b=45),
        yaxis_title="溢价率", yaxis_tickformat=".1f",
        xaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def make_forward_return_curve(forward_df: pd.DataFrame) -> go.Figure:
    """Forward return 分档图：折溢价率分档 vs 未来N日平均收益。"""
    if forward_df.empty:
        return go.Figure()
    fig = go.Figure()
    etfs = forward_df['order_book_id'].unique()
    for i, etf in enumerate(etfs):
        subset = forward_df[forward_df['order_book_id'] == etf].sort_values('premium_bucket')
        if subset.empty:
            continue
        c = PALETTE[i % len(PALETTE)]
        for horizon, dash_style in [(5, 'solid'), (10, 'dash'), (20, 'dot')]:
            col = f'fwd_{horizon}d_avg'
            if col in subset.columns:
                valid = subset.dropna(subset=[col])
                if valid.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=valid['premium_bucket'] * 100, y=valid[col] * 100,
                    name=f"{etf.split('.')[0]} {horizon}d",
                    mode='lines+markers', line=dict(color=c, width=1.5, dash=dash_style),
                    marker=dict(size=4),
                    hovertemplate=f"{etf} {horizon}d<br>溢价分档 %{{x:.1f}}%<br>Forward 收益 %{{y:.2f}}%<extra></extra>",
                ))
    fig.add_hline(y=0, line_width=1, line_color='#94a3b8', line_dash='dash')
    fig.add_vline(x=0, line_width=1, line_color='#94a3b8', line_dash='dash')
    fig.update_layout(
        title="折溢价分档 vs Forward Return (折价买入 → 未来N日收益)",
        height=440, margin=dict(l=50, r=30, t=50, b=45),
        xaxis_title="折溢价率分档", xaxis_tickformat=".1f",
        yaxis_title="Forward Return", yaxis_tickformat=".1f",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def make_threshold_heatmap(threshold_df: pd.DataFrame, etf: str) -> go.Figure:
    """单 ETF 的买卖阈值热力图：buy × sell → total_return。"""
    subset = threshold_df[threshold_df['order_book_id'] == etf]
    if subset.empty:
        return go.Figure()

    pivot = subset.pivot(index='buy_threshold', columns='sell_threshold', values='total_return')
    pivot = pivot.sort_index(ascending=False)

    fig = go.Figure(data=go.Heatmap(
        x=[f"{x:.4f}" for x in pivot.columns],
        y=[f"{y:.4f}" for y in pivot.index],
        z=pivot.values,
        colorscale=[
            [0.0, "#b91c1c"], [0.4, "#f8fafc"],
            [0.6, "#10b981"], [1.0, "#047857"],
        ],
        zmid=0, colorbar=dict(title="累计收益"),
        hovertemplate="买 %{y} 卖 %{x}<br>累计收益 %{z:.2%}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{etf} 买卖阈值扫描 (buy × sell → total_return)",
        height=400, margin=dict(l=65, r=35, t=50, b=50),
        xaxis_title="卖出阈值 (溢价)", yaxis_title="买入阈值 (折价)",
    )
    return fig


def make_equity_curve(df: pd.DataFrame, etf: str,
                      buy_threshold: float, sell_threshold: float, max_hold: int = 60) -> go.Figure:
    """单 ETF 在指定阈值下的净值曲线。"""
    from qdii_premium_analysis import simulate_arbitrage

    cycles = simulate_arbitrage(df, buy_threshold, sell_threshold, max_hold)
    if not cycles:
        return go.Figure()

    # 统一日期为字符串
    for c in cycles:
        c['buy_date'] = str(c['buy_date'])[:10]
        c['sell_date'] = str(c['sell_date'])[:10]

    # 构建日净值序列
    df = df.copy()
    df['date_str'] = df['trade_date'].astype(str).str[:10]
    holding = False
    eq = 1.0
    cycle_idx = 0
    equities = []

    for i in range(len(df)):
        trade_date = df['date_str'].iloc[i]
        if cycle_idx < len(cycles) and trade_date >= cycles[cycle_idx]['buy_date'] and not holding:
            holding = True
        if holding and cycle_idx < len(cycles):
            buy_price = cycles[cycle_idx]['buy_price']
            current_ret = df['close'].iloc[i] / buy_price - 1.0
            eq = 1.0 + current_ret
            if trade_date >= cycles[cycle_idx]['sell_date']:
                eq = 1.0 + cycles[cycle_idx]['cycle_return']
                equities.append({'trade_date': trade_date, 'equity': eq})
                holding = False
                cycle_idx += 1
                eq = 1.0
            else:
                equities.append({'trade_date': trade_date, 'equity': eq})
        else:
            equities.append({'trade_date': trade_date, 'equity': eq})

    eq_df = pd.DataFrame(equities)
    # 累积
    eq_df['cum_equity'] = eq_df['equity'].cumprod() if 'equity' in eq_df.columns else 1.0

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.06,
                        subplot_titles=(f"{etf} 套利净值曲线", "单轮收益分布"))

    fig.add_trace(go.Scatter(
        x=eq_df['trade_date'], y=eq_df['cum_equity'],
        mode='lines', name='套利净值', line=dict(color=COLORS['blue'], width=2),
    ), row=1, col=1)

    returns = [c['cycle_return'] * 100 for c in cycles]
    colors = [COLORS['green'] if r > 0 else COLORS['red'] for r in returns]
    fig.add_trace(go.Bar(
        x=[f"#{i+1}" for i in range(len(returns))],
        y=returns, marker_color=colors,
        name='单轮收益',
    ), row=2, col=1)

    fig.update_yaxes(title_text="净值", row=1, col=1)
    fig.update_yaxes(title_text="收益%", tickformat=".1f", row=2, col=1)
    fig.update_layout(height=520, margin=dict(l=50, r=30, t=60, b=45),
                      showlegend=False)
    return fig


def make_risk_scatter(best_df: pd.DataFrame) -> go.Figure:
    """ETF 对比散点图：年化收益 vs 最大回撤。"""
    if best_df.empty:
        return go.Figure()
    fig = go.Figure()
    for i, (_, row) in enumerate(best_df.iterrows()):
        fig.add_trace(go.Scatter(
            x=[row['max_drawdown'] * 100 if pd.notna(row['max_drawdown']) else 0],
            y=[row['annualized_return'] * 100 if pd.notna(row['annualized_return']) else 0],
            mode='markers+text',
            text=row['order_book_id'].split('.')[0] if '.' in str(row['order_book_id']) else str(row['order_book_id']),
            textposition='middle right',
            marker=dict(
                size=max(np.sqrt(row['n_cycles']) * 4, 10) if pd.notna(row['n_cycles']) else 10,
                color=PALETTE[i % len(PALETTE)],
                line=dict(width=1, color='white'),
            ),
            name=str(row['order_book_id']),
            hovertemplate=f"{row['order_book_id']}<br>年化 %{{y:.1f}}%<br>回撤 %{{x:.1f}}%<br>{row['n_cycles']:.0f}轮<extra></extra>",
        ))
    fig.add_hline(y=0, line_width=1, line_color='#94a3b8', line_dash='dash')
    fig.update_layout(
        title="风险收益散点 (气泡=交易轮次)",
        height=420, margin=dict(l=50, r=30, t=50, b=45),
        xaxis_title="最大回撤", xaxis_tickformat=".1f",
        yaxis_title="年化收益率", yaxis_tickformat=".1f",
        showlegend=False,
    )
    return fig


def make_cycles_table(cycles_df: pd.DataFrame) -> str:
    """套利周期明细表。"""
    if cycles_df.empty:
        return "<p>无交易记录</p>"
    display = cycles_df.sort_values('buy_date').copy()
    cols = ['order_book_id', 'buy_date', 'sell_date', 'hold_days', 'buy_premium',
            'sell_premium', 'cycle_return', 'exit_reason']
    display = display[[c for c in cols if c in display.columns]]
    for c in ['buy_premium', 'sell_premium', 'cycle_return']:
        if c in display.columns:
            display[c] = display[c].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else '-')
    display['buy_date'] = display['buy_date'].astype(str).str[:10]
    display['sell_date'] = display['sell_date'].astype(str).str[:10]

    headers = "".join(f"<th>{c}</th>" for c in display.columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
        for _, row in display.iterrows()
    )
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>"


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def figure_div(fig: go.Figure, div_id: str) -> str:
    return fig.to_html(include_plotlyjs=False, full_html=False, div_id=div_id,
                       config={"responsive": True, "displaylogo": False})


def build_html(results: dict, dfs_etf: dict[str, pd.DataFrame]) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best_df = results.get('top_premium_etfs', pd.DataFrame())
    forward_df = results.get('premium_forward', pd.DataFrame())
    threshold_df = results.get('premium_overall', pd.DataFrame())
    cycles_df = results.get('premium_cycles', pd.DataFrame())

    figures: list[tuple[str, go.Figure]] = []

    # 1. 折溢价时序
    figures.append(("premium_ts", make_premium_ts(dfs_etf)))

    # 2. Forward return
    figures.append(("forward_return", make_forward_return_curve(forward_df)))

    # 3. 风险散点
    figures.append(("risk_scatter", make_risk_scatter(best_df)))

    # 4. 每个 ETF 的阈值热力图 + 净值曲线
    if not best_df.empty:
        for i, (_, row) in enumerate(best_df.iterrows()):
            etf = row['order_book_id']
            if etf in dfs_etf and dfs_etf[etf] is not None and not dfs_etf[etf].empty:
                buy_th = float(row['buy_threshold'])
                sell_th = float(row['sell_threshold'])
                figures.append((f"heatmap_{i}", make_threshold_heatmap(threshold_df, etf)))
                figures.append((f"equity_{i}",
                                make_equity_curve(dfs_etf[etf], etf, buy_th, sell_th)))

    figure_html = "\n".join(
        "<section class='panel'>"
        f"<button type='button' class='fullscreen-btn' data-target='{div_id}' "
        "onclick='toggleFullscreen(this)'>全屏</button>"
        f"{figure_div(fig, div_id)}</section>"
        for div_id, fig in figures
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>QDII ETF 折溢价套利分析</title>
  <style>
    :root {{ color-scheme: light; --bg: #f6f8fb; --panel: #ffffff; --text: #172033;
      --muted: #64748b; --border: #d8dee9; --accent: #2563eb; }}
    body {{ margin: 0; background: var(--bg); color: var(--text);
      font-family: -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;
      line-height: 1.45; }}
    header {{ padding: 28px 34px 18px; background: #ffffff; border-bottom: 1px solid var(--border); }}
    h1 {{ margin: 0 0 8px; font-size: 26px; font-weight: 700; }}
    .meta {{ color: var(--muted); font-size: 13px; }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 22px 24px 38px; }}
    .kpis {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 10px; margin-bottom: 16px; }}
    .kpi, .panel, .table-panel {{ background: var(--panel); border: 1px solid var(--border);
      border-radius: 8px; box-shadow: 0 1px 2px rgba(15,23,42,0.04); }}
    .kpi {{ padding: 12px 14px; }}
    .kpi-title {{ font-size: 14px; font-weight: 700; margin-bottom: 4px; }}
    .kpi-value {{ font-size: 13px; color: var(--accent); margin-bottom: 2px; }}
    .kpi-sub {{ color: var(--muted); font-size: 11px; }}
    .panel {{ padding: 10px 12px; margin: 14px 0; position: relative; }}
    .fullscreen-btn {{ position: absolute; top: 12px; right: 14px; z-index: 5;
      border: 1px solid var(--border); border-radius: 6px;
      background: rgba(255,255,255,0.92); color: #334155; cursor: pointer;
      font-size: 12px; line-height: 1; padding: 7px 10px; }}
    .fullscreen-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
    .table-panel {{ padding: 16px; margin: 14px 0; overflow-x: auto; }}
    .table-panel h2 {{ margin: 0 0 10px; font-size: 17px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid #e7ebf2; padding: 7px 9px; text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: #334155; background: #f8fafc; font-weight: 650; }}
    @media (max-width: 768px) {{ .kpis {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <script>{get_plotlyjs()}</script>
  <header>
    <h1>QDII ETF 折溢价套利 — 分析报告</h1>
    <div class="meta">生成: {html.escape(generated_at)} · {len(dfs_etf)} 只 ETF</div>
  </header>
  <main>
    <section class="kpis">{make_kpi_cards(best_df)}</section>
    <section class="table-panel">
      <h2>套利周期明细</h2>
      {make_cycles_table(cycles_df)}
    </section>
    {figure_html}
  </main>
  <script>
    function toggleFullscreen(btn) {{
      var p = btn.closest('.panel');
      if (!p) return;
      var active = document.fullscreenElement || document.webkitFullscreenElement;
      if (active === p) {{
        (document.exitFullscreen || document.webkitExitFullscreen).call(document);
      }} else {{
        (p.requestFullscreen || p.webkitRequestFullscreen).call(p).then(function() {{
          setTimeout(function() {{
            if (window.Plotly) p.querySelectorAll('.plotly-graph-div')
              .forEach(function(el) {{ Plotly.Plots.resize(el); }});
          }}, 80);
        }});
      }}
    }}
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def run(etf_filter: str | None = None):
    results = load_results()
    best_df = results.get('top_premium_etfs', pd.DataFrame())

    if best_df.empty:
        print('[ERROR] 无分析结果。请先运行 qdii_premium_analysis.py')
        return

    etfs = best_df['order_book_id'].tolist()
    if etf_filter:
        etfs = [e for e in etfs if etf_filter in e]

    print(f"加载 {len(etfs)} 只 ETF 数据 ...")
    dfs_etf = {}
    for code in etfs:
        df = load_etf_data(code)
        if df is not None and not df.empty:
            dfs_etf[code] = df
            print(f"  {code:16s} {len(df):5d}天")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / 'qdii_arbitrage_report.html'
    html_text = build_html(results, dfs_etf)
    out_path.write_text(html_text, encoding='utf-8')
    print(f'\n报告已生成: {out_path.resolve()}')


def main():
    parser = argparse.ArgumentParser(description='QDII 折溢价套利报告')
    parser.add_argument('--etf', default=None, help='筛选特定 ETF')
    args = parser.parse_args()
    run(etf_filter=args.etf)


if __name__ == '__main__':
    main()
