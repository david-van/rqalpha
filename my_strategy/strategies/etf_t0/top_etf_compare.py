#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""ETF 高低开策略 Top 排行榜 HTML 报告。

从 top_thresholds.csv 中筛选（历史>1年 + 交易>100次），
按年化收益取 Top 15，生成对比可视化报告。

用法:
    python top_etf_compare.py              # 默认参数
    python top_etf_compare.py --top-n 10   # 只看前10
"""

from __future__ import annotations

import argparse
import html
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from my_strategy.common_file import project_root

try:
    import pymysql
except ImportError:
    print("请先安装 pymysql: pip install pymysql")
    sys.exit(1)

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# 配置（修改这里即可）
# ---------------------------------------------------------------------------
TOP_N = 15                          # 取前几名
MIN_YEARS = 1.0                     # 最少数据年数
MIN_TRADES = 100                    # 最少交易次数
CSV_PATH = (
    Path(project_root) / "my_strategy" / "strategies"
    / "batch_results" / "etf_gap_multi" / "top_thresholds.csv"
)
THRESHOLD_CSV = (
    Path(project_root) / "my_strategy" / "strategies"
    / "batch_results" / "etf_gap_multi" / "threshold_overall.csv"
)
OUT_DIR = (
    Path(project_root) / "my_strategy" / "strategies"
    / "batch_results" / "etf_reports"
)

DEFAULT_DB = {
    "host": "localhost", "port": 3306, "user": "root", "password": "",
    "database": "rqalpha_etf", "charset": "utf8mb4",
}

COLORS = {
    "blue": "#2563eb", "green": "#059669", "red": "#dc2626",
    "orange": "#ea580c", "purple": "#7c3aed", "gray": "#64748b",
}
PALETTE = [
    "#2563eb", "#059669", "#dc2626", "#ea580c", "#7c3aed", "#0891b2",
    "#be185d", "#4f46e5", "#b45309", "#0d9488", "#9333ea", "#c2410c",
    "#1d4ed8", "#15803d", "#a21caf",
]


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETF 高低开策略 Top 排行榜")
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--min-years", type=float, default=MIN_YEARS)
    parser.add_argument("--min-trades", type=int, default=MIN_TRADES)
    parser.add_argument("--csv", default=str(CSV_PATH))
    parser.add_argument("--threshold-csv", default=str(THRESHOLD_CSV))
    parser.add_argument("--host", default=DEFAULT_DB["host"])
    parser.add_argument("--port", type=int, default=DEFAULT_DB["port"])
    parser.add_argument("--user", default=DEFAULT_DB["user"])
    parser.add_argument("--password", default=DEFAULT_DB["password"])
    parser.add_argument("--database", default=DEFAULT_DB["database"])
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def get_conn(args) -> pymysql.Connection:
    return pymysql.connect(
        host=args.host, port=args.port, user=args.user,
        password=args.password, database=args.database, charset="utf8mb4",
    )


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_data(args) -> tuple[pd.DataFrame, list[dict]]:
    """读取 top_thresholds.csv，筛选后返回 top N 和完整的阈值扫描数据."""
    top = pd.read_csv(args.csv)
    # 筛选
    mask = (top["years"] > args.min_years) & (top["trade_count"] > args.min_trades)
    filtered = top[mask].copy()
    filtered = filtered.nlargest(args.top_n, "annualized_return").reset_index(drop=True)

    # 加载 ETF 名称
    conn = get_conn(args)
    try:
        name_df = pd.read_sql(
            "SELECT order_book_id, name FROM etf_universe", conn,
        )
        filtered = filtered.merge(name_df, on="order_book_id", how="left")
        filtered["name"] = filtered["name"].fillna("")
    except Exception:
        filtered["name"] = ""
    conn.close()

    # 加载阈值扫描明细（用于画每个 ETF 的阈值 vs 收益曲线）
    threshold_df = pd.read_csv(args.threshold_csv) if Path(args.threshold_csv).exists() else pd.DataFrame()

    top_list = filtered.to_dict("records")
    return top_list, threshold_df


def load_equity_data(args, etf: str, threshold: float) -> pd.DataFrame | None:
    """从 MySQL 加载单只 ETF 在指定阈值下的日线数据和净值."""
    conn = get_conn(args)
    sql = """
        SELECT trade_date, open, close, prev_close,
               open - prev_close AS gap,
               (close*(1-%s) / NULLIF(open*(1+%s), 0) - 1) AS net_return
        FROM etf_daily_bars
        WHERE order_book_id = %s AND trade_date >= '2017-01-01'
          AND open > 0 AND close > 0 AND prev_close > 0
        ORDER BY trade_date
    """
    params = [0.00005, 0.00005, etf]
    df = pd.read_sql(sql, conn, params=params)
    conn.close()
    if df.empty:
        return None
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["net_return"] = pd.to_numeric(df["net_return"], errors="coerce")
    df["signal"] = df["gap"] >= threshold
    df["strategy_return"] = df["net_return"].where(df["signal"], 0.0)
    df["equity"] = (1.0 + df["strategy_return"].fillna(0.0)).cumprod()
    return df


# ---------------------------------------------------------------------------
# 图表
# ---------------------------------------------------------------------------
def pct(v, d=1):
    return f"{v:.{d}%}" if pd.notna(v) else "-"


def num(v, d=2):
    return f"{v:.{d}f}" if pd.notna(v) else "-"


def make_summary_table(top_list: list[dict]) -> str:
    """Top ETF 对比表格."""
    headers = ["排名", "ETF代码", "名称", "阈值", "年限", "交易次",
               "胜率", "年化收益", "累计收益", "最大回撤"]
    rows = []
    for i, r in enumerate(top_list, 1):
        rows.append([
            str(i),
            r["order_book_id"],
            str(r.get("name", ""))[:20],
            f"{r['threshold']:.4f}",
            f"{r['years']:.2f}年",
            str(int(r["trade_count"])),
            pct(r["win_rate"]),
            pct(r["annualized_return"]),
            pct(r["total_return"]),
            pct(r["max_drawdown"]),
        ])
    thead = "".join(f"<th>{h}</th>" for h in headers)
    tbody = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>"


def make_risk_scatter(top_list: list[dict]) -> go.Figure:
    """风险收益散点图：年化收益 vs 最大回撤，气泡=交易次数，颜色=年限."""
    fig = go.Figure()
    for i, r in enumerate(top_list):
        fig.add_trace(go.Scatter(
            x=[r["max_drawdown"] * 100],
            y=[r["annualized_return"] * 100],
            mode="markers+text",
            text=r["order_book_id"].split(".")[0],
            textposition="middle right",
            marker=dict(
                size=max(np.sqrt(r["trade_count"]) * 0.6, 8),
                color=PALETTE[i % len(PALETTE)],
                line=dict(width=1, color="white"),
            ),
            name=f"{r['order_book_id']} {r.get('name', '')[:12]}",
            hovertemplate=(
                f"{r['order_book_id']} {r.get('name', '')}<br>"
                "年化收益 %{y:.1f}%<br>最大回撤 %{x:.1f}%<br>"
                f"交易{r['trade_count']:.0f}次 · {r['years']:.2f}年"
                "<extra></extra>"
            ),
        ))
    fig.update_layout(
        title="风险收益散点图（气泡大小=交易次数）",
        height=500, margin=dict(l=55, r=30, t=50, b=50),
        xaxis_title="最大回撤", yaxis_title="年化收益率",
        xaxis_tickformat=".1f", yaxis_tickformat=".1f",
        showlegend=False,
    )
    return fig


def make_equity_curves(args, top_list: list[dict]) -> go.Figure:
    """所有 ETF 在各自最优阈值下的净值曲线叠加."""
    fig = go.Figure()
    for i, r in enumerate(top_list):
        df = load_equity_data(args, r["order_book_id"], r["threshold"])
        if df is None or df.empty:
            continue
        label = f"{r['order_book_id'].split('.')[0]}"
        fig.add_trace(go.Scatter(
            x=df["trade_date"], y=df["equity"],
            name=label, mode="lines",
            line=dict(color=PALETTE[i], width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>" + label + " 净值 %{y:.3f}<extra></extra>",
        ))
    fig.update_layout(
        title=f"Top {len(top_list)} 净值曲线（各自最优阈值）",
        height=520, margin=dict(l=55, r=30, t=50, b=50),
        yaxis_title="净值",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def make_yearly_returns(args, top_list: list[dict]) -> go.Figure:
    """Top 15 各年份收益热力图."""
    rows_data = []
    for i, r in enumerate(top_list):
        df = load_equity_data(args, r["order_book_id"], r["threshold"])
        if df is None or df.empty:
            continue
        df["year"] = df["trade_date"].dt.year
        for yr, grp in df.groupby("year"):
            yr_ret = float((1.0 + grp["strategy_return"].fillna(0.0)).prod() - 1.0)
            rows_data.append({
                "ETF": r["order_book_id"].split(".")[0],
                "年份": int(yr),
                "年度收益": yr_ret,
            })
    if not rows_data:
        return go.Figure()

    yr_df = pd.DataFrame(rows_data)
    pivot = yr_df.pivot(index="ETF", columns="年份", values="年度收益")

    # 按 ETF 在 top_list 中的顺序排列
    etf_order = [r["order_book_id"].split(".")[0] for r in top_list]
    pivot = pivot.reindex([e for e in etf_order if e in pivot.index])

    fig = go.Figure(data=go.Heatmap(
        x=[str(int(y)) for y in pivot.columns],
        y=pivot.index.tolist(),
        z=pivot.values,
        colorscale=[
            [0.0, "#b91c1c"],
            [0.35, "#f59e0b"],
            [0.48, "#f8fafc"],
            [0.55, "#10b981"],
            [1.0, "#047857"],
        ],
        zmid=0,
        colorbar=dict(title="年度收益"),
        hovertemplate="%{y} %{x}<br>年度收益 %{z:.1%}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Top {len(top_list)} 各年度收益率（按最优阈值）",
        height=440,
        margin=dict(l=75, r=35, t=50, b=50),
        xaxis_title="年份", yaxis_title="",
    )
    return fig


def make_threshold_sensitivity(threshold_df: pd.DataFrame, top_list: list[dict]) -> go.Figure:
    """前6只 ETF 的阈值-累计收益敏感度曲线."""
    fig = go.Figure()
    top6_etfs = {r["order_book_id"] for r in top_list[:6]}
    for i, etf in enumerate(top6_etfs):
        subset = threshold_df[threshold_df["order_book_id"] == etf].sort_values("threshold")
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset["threshold"], y=subset["total_return"],
            name=f"{etf.split('.')[0]}",
            mode="lines+markers",
            line=dict(color=PALETTE[i], width=2),
            hovertemplate=f"{etf}<br>阈值 %{{x:.3f}}<br>累计收益 %{{y:.1%}}<extra></extra>",
        ))
    fig.update_layout(
        title="阈值 vs 累计收益（Top 6 对比）",
        height=420, margin=dict(l=55, r=30, t=50, b=50),
        xaxis_title="高低开阈值", xaxis_tickformat=".3f",
        yaxis_title="累计收益率", yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def figure_div(fig: go.Figure, div_id: str) -> str:
    return fig.to_html(include_plotlyjs=False, full_html=False, div_id=div_id,
                       config={"responsive": True, "displaylogo": False})


# ---------------------------------------------------------------------------
# HTML 组装
# ---------------------------------------------------------------------------
def build_html(args, top_list: list[dict], threshold_df: pd.DataFrame) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    figures = [
        ("risk_scatter", make_risk_scatter(top_list)),
        ("equity_curves", make_equity_curves(args, top_list)),
        ("yearly_returns", make_yearly_returns(args, top_list)),
        ("threshold_sensitivity", make_threshold_sensitivity(threshold_df, top_list)),
    ]
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
  <title>ETF 高低开策略 Top 排行</title>
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
    .panel, .table-panel {{ background: var(--panel); border: 1px solid var(--border);
      border-radius: 8px; box-shadow: 0 1px 2px rgba(15,23,42,0.04); }}
    .panel {{ padding: 10px 12px; margin: 16px 0; position: relative; }}
    .fullscreen-btn {{ position: absolute; top: 12px; right: 14px; z-index: 5;
      border: 1px solid var(--border); border-radius: 6px;
      background: rgba(255,255,255,0.92); color: #334155; cursor: pointer;
      font-size: 12px; line-height: 1; padding: 7px 10px;
      box-shadow: 0 1px 2px rgba(15,23,42,0.08); }}
    .fullscreen-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
    .table-panel {{ padding: 18px; margin: 16px 0; overflow-x: auto; }}
    .table-panel h2 {{ margin: 0 0 12px; font-size: 18px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e7ebf2; padding: 8px 10px;
      text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: #334155; background: #f8fafc; font-weight: 650; }}
    tr:nth-child(1) td {{ font-weight: 700; background: #fef3c7; }}
    tr:nth-child(2) td {{ font-weight: 700; background: #f1f5f9; }}
    tr:nth-child(3) td {{ font-weight: 700; background: #fef2f2; }}
    @media (max-width: 1100px) {{ main {{ padding: 16px 12px 28px; }} }}
  </style>
</head>
<body>
  <script>{get_plotlyjs()}</script>
  <header>
    <h1>ETF 高低开策略 — 年化收益 Top {len(top_list)}</h1>
    <div class="meta">
      生成时间：{html.escape(generated_at)}
      · 筛选条件：历史 &gt; {args.min_years} 年 · 交易 &gt; {args.min_trades} 次
      · 数据来源: {html.escape(args.csv)}
    </div>
  </header>
  <main>
    <section class="table-panel">
      <h2>Top {len(top_list)} 排名</h2>
      {make_summary_table(top_list)}
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


def main() -> None:
    args = parse_args()
    if not Path(args.csv).exists():
        print(f"[ERROR] top_thresholds.csv 不存在: {args.csv}")
        print("请先运行 etf_gap_analysis.py 生成数据")
        sys.exit(1)

    print("加载数据 ...")
    top_list, threshold_df = load_data(args)

    if not top_list:
        print("[ERROR] 没有满足条件的 ETF")
        sys.exit(1)

    print(f"筛选后 Top {len(top_list)}:")
    for r in top_list:
        print(f"  {r['order_book_id']:16s}  {r.get('name', '')[:20]:20s}  "
              f"年化{pct(r['annualized_return']):>7s}  累计{pct(r['total_return'])}  "
              f"回撤{pct(r['max_drawdown'])}  {r['trade_count']:.0f}次/{r['years']:.2f}年")

    out_path = Path(args.out) if args.out else (
        OUT_DIR / f"etf_top{args.top_n}_compare.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n生成 HTML ...")
    html_text = build_html(args, top_list, threshold_df)
    out_path.write_text(html_text, encoding="utf-8")
    print(f"报告已生成: {out_path.resolve()}")


if __name__ == "__main__":
    main()
