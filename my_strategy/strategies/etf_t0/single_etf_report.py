#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""生成单个 ETF 的高低开策略可视化 HTML 报告（MySQL 数据源）。

用法:
    python single_etf_report.py --etf 513030.XSHG
    python single_etf_report.py --etf 513100.XSHG --start-date 2020-01-01 --out report.html
    python single_etf_report.py --etf 513030.XSHG --thresholds -0.005,0,0.003,0.01
"""

from __future__ import annotations

import argparse
import html
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pymysql
except ImportError:
    print("请先安装 pymysql: pip install pymysql")
    sys.exit(1)

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from my_strategy.common_file import project_root

# ---------------------------------------------------------------------------
# 配置（修改这里即可，不需要命令行参数）
# ---------------------------------------------------------------------------
ETF_CODE = "513360.XSHG"          # 要分析的 ETF 代码
START_DATE = "2019-01-01"         # 回测起始日期
END_DATE = "2026-05-01"           # 回测结束日期
THRESHOLDS = None                 # 净值曲线展示的阈值，如 "-0.005,0,0.003"；None=自动选取
STEP = 0.001                      # 阈值扫描步长
MIN_GAP = -0.02                   # 阈值扫描下限
MAX_GAP = 0.02                    # 阈值扫描上限
COMMISSION = 0.00005              # 单边手续费率

DEFAULT_DB = {
    "host": "localhost", "port": 3306, "user": "root", "password": "",
    "database": "rqalpha_etf", "charset": "utf8mb4",
}
DEFAULT_OUT_DIR = Path(project_root) / "my_strategy" / "strategies" / "batch_results" / "etf_reports"

COL_THRESHOLD = "阈值"
COL_YEAR = "年份"
COL_BUCKET = "高低开分档"
COL_TRADES = "交易次数"
COL_WIN_RATE = "胜率"
COL_AVG_RETURN = "平均收益率"
COL_TOTAL_RETURN = "累计收益率"
COL_MAX_DRAWDOWN = "最大回撤"
COL_PROFIT_LOSS_RATIO = "盈亏比"

COLORS = {
    "blue": "#2563eb", "green": "#059669", "red": "#dc2626",
    "orange": "#ea580c", "purple": "#7c3aed", "gray": "#64748b",
}


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成单 ETF 高低开策略 HTML 报告")
    parser.add_argument("--etf", default=ETF_CODE, help="ETF 代码")
    parser.add_argument("--host", default=DEFAULT_DB["host"])
    parser.add_argument("--port", type=int, default=DEFAULT_DB["port"])
    parser.add_argument("--user", default=DEFAULT_DB["user"])
    parser.add_argument("--password", default=DEFAULT_DB["password"])
    parser.add_argument("--database", default=DEFAULT_DB["database"])
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--step", type=float, default=STEP)
    parser.add_argument("--min-gap", type=float, default=MIN_GAP)
    parser.add_argument("--max-gap", type=float, default=MAX_GAP)
    parser.add_argument("--commission", type=float, default=COMMISSION)
    parser.add_argument("--out", default=None, help="输出 HTML 路径")
    parser.add_argument("--top-n", type=int, default=10, help="排名表格行数")
    parser.add_argument(
        "--thresholds", default=THRESHOLDS,
        help="净值曲线要展示的阈值，逗号分隔",
    )
    return parser.parse_args()


def get_conn(args) -> pymysql.Connection:
    return pymysql.connect(
        host=args.host, port=args.port, user=args.user,
        password=args.password, database=args.database, charset="utf8mb4",
    )


# ---------------------------------------------------------------------------
# 数据加载（MySQL）
# ---------------------------------------------------------------------------
def load_etf_name(conn, etf_code: str) -> str:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM etf_universe WHERE order_book_id = %s", (etf_code,))
            row = cur.fetchone()
        return row[0] if row else etf_code
    except Exception:
        return etf_code


def _resolve_etf_code(conn, etf_code: str) -> str:
    """自动匹配后缀：如果用户只给 513030，补全为 513030.XSHG 或 .XSHE."""
    etf_upper = etf_code.upper()
    if "." in etf_upper:
        return etf_upper
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT order_book_id FROM etf_daily_bars WHERE order_book_id LIKE %s LIMIT 1",
                (etf_upper + ".%",),
            )
            row = cur.fetchone()
        if row:
            return row[0]
    except Exception:
        pass
    # fallback
    for suffix in (".XSHG", ".XSHE"):
        candidate = etf_upper + suffix
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM etf_daily_bars WHERE order_book_id = %s LIMIT 1",
                    (candidate,),
                )
                if cur.fetchone():
                    return candidate
        except Exception:
            pass
    return etf_upper + ".XSHG"


def load_data(conn, args) -> dict[str, pd.DataFrame]:
    """从 MySQL 加载 ETF 日线，计算 gap/return，构建阈值扫描表."""
    etf_code = _resolve_etf_code(conn, args.etf)

    sql = """
        SELECT order_book_id, trade_date, YEAR(trade_date) AS trade_year,
               open, close, prev_close,
               open - prev_close                                     AS gap,
               (open - prev_close) / NULLIF(prev_close, 0)            AS gap_pct,
               close / NULLIF(open, 0) - 1                           AS gross_return,
               (close*(1-%s) / NULLIF(open*(1+%s), 0) - 1)           AS net_return,
               close - open                                          AS pnl_price,
               volume, total_turnover
        FROM etf_daily_bars
        WHERE order_book_id = %s AND trade_date >= %s AND trade_date <= %s
          AND open > 0 AND close > 0 AND prev_close > 0
        ORDER BY trade_date
    """
    params = [args.commission, args.commission, etf_code, args.start_date, args.end_date]
    trades = pd.read_sql(sql, conn, params=params)
    if trades.empty:
        raise RuntimeError(f"{etf_code} 在指定日期范围内无数据")

    trades["trade_date"] = pd.to_datetime(trades["trade_date"])
    for c in ["gap", "gap_pct", "gross_return", "net_return", "pnl_price"]:
        trades[c] = pd.to_numeric(trades[c], errors="coerce")
    trades = trades.sort_values("trade_date").reset_index(drop=True)
    trades["year"] = trades["trade_date"].dt.year

    # ---- 阈值扫描: overall ----
    thresholds = np.arange(args.min_gap, args.max_gap + args.step / 2, args.step).round(6)
    overall_rows = []
    for th in thresholds:
        selected = trades[trades["gap"] >= th]
        overall_rows.append({
            COL_THRESHOLD: float(th),
            ** _summarize(selected),
        })
    overall = pd.DataFrame(overall_rows).sort_values(COL_THRESHOLD).reset_index(drop=True)

    # ---- 阈值 × 年份 ----
    yearly_rows = []
    for th in thresholds:
        selected = trades[trades["gap"] >= th]
        for yr, grp in selected.groupby("trade_year"):
            yearly_rows.append({
                COL_YEAR: int(yr),
                COL_THRESHOLD: float(th),
                ** _summarize(grp),
            })
    yearly = pd.DataFrame(yearly_rows).sort_values([COL_YEAR, COL_THRESHOLD]).reset_index(drop=True)

    # ---- gap 分档 × 年份 ----
    trades_cp = trades.copy()
    trades_cp[COL_BUCKET] = (np.rint(trades_cp["gap"] / args.step) * args.step).round(6)
    bucket_rows = []
    for (yr, bk), grp in trades_cp.groupby(["trade_year", COL_BUCKET]):
        bucket_rows.append({
            COL_YEAR: int(yr),
            COL_BUCKET: float(bk),
            "平均高低开": float(grp["gap"].mean()),
            "平均高低开比例": float(grp["gap_pct"].mean()),
            ** _summarize(grp),
        })
    buckets = pd.DataFrame(bucket_rows).sort_values([COL_YEAR, COL_BUCKET]).reset_index(drop=True)

    return {"overall": overall, "yearly": yearly, "buckets": buckets, "trades": trades, "etf_code": etf_code}


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    eq = np.r_[1.0, (1.0 + returns.to_numpy()).cumprod()]
    return float((eq / np.maximum.accumulate(eq) - 1.0).min())


def _summarize(group: pd.DataFrame) -> dict:
    ret = group["net_return"]
    n = len(group)
    if n == 0:
        return {COL_TRADES: 0, COL_WIN_RATE: 0.0, COL_AVG_RETURN: 0.0,
                COL_TOTAL_RETURN: 0.0, COL_MAX_DRAWDOWN: 0.0,
                "平均价差收益": 0.0, "累计价差收益": 0.0,
                COL_PROFIT_LOSS_RATIO: np.nan}
    gains = ret[ret > 0].sum()
    losses = -ret[ret < 0].sum()
    return {
        COL_TRADES: n,
        COL_WIN_RATE: float((ret > 0).mean()),
        COL_AVG_RETURN: float(ret.mean()),
        COL_TOTAL_RETURN: float((1.0 + ret).prod() - 1.0),
        COL_MAX_DRAWDOWN: _max_drawdown(ret),
        "平均价差收益": float(group["pnl_price"].mean()),
        "累计价差收益": float(group["pnl_price"].sum()),
        COL_PROFIT_LOSS_RATIO: float(gains / losses) if losses > 0 else np.nan,
    }


# ---------------------------------------------------------------------------
# 图表函数（复用自 gap_strategy_html.py）
# ---------------------------------------------------------------------------

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


def pick_thresholds(overall: pd.DataFrame, raw: str | None) -> list[float]:
    available = sorted(float(x) for x in overall[COL_THRESHOLD].dropna().unique())
    if not available:
        return []
    if raw:
        req = [float(x.strip()) for x in raw.split(",") if x.strip()]
        picked = []
        for v in req:
            nearest = min(available, key=lambda x: abs(x - v))
            if nearest not in picked:
                picked.append(nearest)
        return picked[:6]

    best_return = float(overall.loc[overall[COL_TOTAL_RETURN].idxmax(), COL_THRESHOLD])
    best_dd = float(overall.loc[overall[COL_MAX_DRAWDOWN].idxmax(), COL_THRESHOLD])
    anchors = [best_return, 0.003, 0.0, best_dd]
    picked = []
    for v in anchors:
        nearest = min(available, key=lambda x: abs(x - v))
        if nearest not in picked:
            picked.append(nearest)
    return picked[:4]


def make_threshold_overview(overall: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("阈值 vs 累计收益", "阈值 vs 最大回撤",
                        "阈值 vs 胜率 / 平均单笔收益", "阈值 vs 交易次数"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": True}, {"secondary_y": False}]],
        vertical_spacing=0.13, horizontal_spacing=0.1,
    )
    x = overall[COL_THRESHOLD]
    fig.add_trace(go.Scatter(x=x, y=overall[COL_TOTAL_RETURN], name="累计收益率",
        mode="lines+markers", line=dict(color=COLORS["blue"], width=2),
        hovertemplate="阈值 %{x:.3f}<br>累计收益 %{y:.2%}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=overall[COL_MAX_DRAWDOWN], name="最大回撤",
        mode="lines+markers", line=dict(color=COLORS["red"], width=2),
        hovertemplate="阈值 %{x:.3f}<br>最大回撤 %{y:.2%}<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=overall[COL_WIN_RATE], name="胜率",
        mode="lines", line=dict(color=COLORS["green"], width=2),
        hovertemplate="阈值 %{x:.3f}<br>胜率 %{y:.2%}<extra></extra>"), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=overall[COL_AVG_RETURN], name="平均单笔收益",
        mode="lines", line=dict(color=COLORS["orange"], width=2, dash="dot"),
        hovertemplate="阈值 %{x:.3f}<br>平均收益 %{y:.3%}<extra></extra>"), row=2, col=1, secondary_y=True)
    fig.add_trace(go.Bar(x=x, y=overall[COL_TRADES], name="交易次数",
        marker_color=COLORS["gray"],
        hovertemplate="阈值 %{x:.3f}<br>交易次数 %{y}<extra></extra>"), row=2, col=2)

    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=1, col=2)
    fig.update_yaxes(tickformat=".0%", row=2, col=1, secondary_y=False)
    fig.update_yaxes(tickformat=".2%", row=2, col=1, secondary_y=True)
    fig.update_xaxes(title_text="高低开阈值", tickformat=".3f")
    fig.update_layout(height=760, margin=dict(l=50, r=40, t=80, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1))
    return fig


def make_equity_curves(trades: pd.DataFrame, thresholds: list[float]) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.68, 0.32],
        subplot_titles=("按阈值过滤后的累计净值", "对应回撤"))
    palette = [COLORS["blue"], COLORS["green"], COLORS["orange"],
               COLORS["purple"], COLORS["red"], COLORS["gray"]]

    for idx, th in enumerate(thresholds):
        s = trades["net_return"].where(trades["gap"] >= th, 0.0)
        eq = (1.0 + s.fillna(0.0)).cumprod()
        dd = eq / eq.cummax() - 1.0
        label = f"阈值 {threshold_label(th)}"
        c = palette[idx % len(palette)]
        fig.add_trace(go.Scatter(x=trades["trade_date"], y=eq, name=label,
            mode="lines", line=dict(color=c, width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>净值 %{y:.3f}<extra>" + label + "</extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=trades["trade_date"], y=dd, name=f"{label} 回撤",
            mode="lines", line=dict(color=c, width=1.5), showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}<br>回撤 %{y:.2%}<extra>" + label + "</extra>"), row=2, col=1)

    fig.update_yaxes(title_text="净值", row=1, col=1)
    fig.update_yaxes(title_text="回撤", tickformat=".0%", row=2, col=1)
    fig.update_layout(height=680, margin=dict(l=55, r=35, t=70, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1))
    return fig


def make_yearly_heatmap(yearly: pd.DataFrame) -> go.Figure:
    pivot_ret = yearly.pivot(index=COL_YEAR, columns=COL_THRESHOLD,
                             values=COL_TOTAL_RETURN).sort_index()
    pivot_cnt = yearly.pivot(index=COL_YEAR, columns=COL_THRESHOLD,
                             values=COL_TRADES).reindex_like(pivot_ret)
    fig = go.Figure(data=go.Heatmap(
        x=[threshold_label(float(x)) for x in pivot_ret.columns],
        y=[str(int(y)) for y in pivot_ret.index],
        z=pivot_ret.values, customdata=pivot_cnt.values,
        colorscale=[[0.0, "#b91c1c"], [0.48, "#f8fafc"], [1.0, "#047857"]],
        zmid=0, colorbar=dict(title="累计收益"),
        hovertemplate="年份 %{y}<br>阈值 %{x}<br>累计收益 %{z:.2%}<br>交易次数 %{customdata}<extra></extra>",
    ))
    fig.update_layout(title="年度阈值热力图", height=430,
        margin=dict(l=45, r=35, t=55, b=55),
        xaxis_title="高低开阈值", yaxis_title="年份")
    return fig


def make_bucket_heatmaps(buckets: pd.DataFrame) -> go.Figure:
    pivot_ret = buckets.pivot(index=COL_YEAR, columns=COL_BUCKET,
                              values=COL_AVG_RETURN).sort_index()
    pivot_cnt = buckets.pivot(index=COL_YEAR, columns=COL_BUCKET,
                              values=COL_TRADES).reindex_like(pivot_ret)
    x_labels = [threshold_label(float(x)) for x in pivot_ret.columns]
    y_labels = [str(int(y)) for y in pivot_ret.index]
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=("分档平均单笔收益", "分档交易次数"), horizontal_spacing=0.12)
    fig.add_trace(go.Heatmap(x=x_labels, y=y_labels, z=pivot_ret.values,
        colorscale=[[0.0, "#b91c1c"], [0.5, "#f8fafc"], [1.0, "#047857"]],
        zmid=0, colorbar=dict(title="平均收益", x=0.46),
        hovertemplate="年份 %{y}<br>分档 %{x}<br>平均收益 %{z:.3%}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Heatmap(x=x_labels, y=y_labels, z=pivot_cnt.values,
        colorscale="Blues", colorbar=dict(title="次数"),
        hovertemplate="年份 %{y}<br>分档 %{x}<br>交易次数 %{z}<extra></extra>"), row=1, col=2)
    fig.update_layout(height=480, margin=dict(l=45, r=35, t=65, b=70))
    fig.update_xaxes(title_text="高低开分档")
    fig.update_yaxes(title_text="年份", row=1, col=1)
    return fig


def make_daily_scatter(trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=trades["gap"], y=trades["net_return"], mode="markers",
        marker=dict(size=8, color=trades["year"], colorscale="Viridis",
                    opacity=0.72, colorbar=dict(title="年份")),
        customdata=np.stack([
            trades["trade_date"].dt.strftime("%Y-%m-%d"),
            trades["open"], trades["close"],
        ], axis=-1),
        hovertemplate="%{customdata[0]}<br>高低开 %{x:.3f}<br>当日收益 %{y:.2%}"
                      "<br>开盘 %{customdata[1]:.3f}<br>收盘 %{customdata[2]:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_width=1, line_color="#94a3b8")
    fig.add_vline(x=0, line_width=1, line_color="#94a3b8")
    fig.update_layout(title="每日高低开 vs 开盘买入收盘卖出收益", height=470,
        margin=dict(l=55, r=35, t=60, b=50),
        xaxis_title="高低开价差", yaxis_title="当日收益率", yaxis_tickformat=".1%")
    return fig


def figure_div(fig: go.Figure, div_id: str) -> str:
    return fig.to_html(include_plotlyjs=False, full_html=False, div_id=div_id,
                       config={"responsive": True, "displaylogo": False})


# ---------------------------------------------------------------------------
# HTML 组件
# ---------------------------------------------------------------------------

def make_top_table(overall: pd.DataFrame, top_n: int) -> str:
    display = (overall.sort_values(COL_TOTAL_RETURN, ascending=False).head(top_n)
        .loc[:, [COL_THRESHOLD, COL_TRADES, COL_WIN_RATE, COL_AVG_RETURN,
                 COL_TOTAL_RETURN, COL_MAX_DRAWDOWN, COL_PROFIT_LOSS_RATIO]].copy())
    display[COL_THRESHOLD] = display[COL_THRESHOLD].map(lambda x: f"{x:.3f}")
    for c in [COL_WIN_RATE, COL_TOTAL_RETURN, COL_MAX_DRAWDOWN]:
        display[c] = display[c].map(lambda x: pct(x, 2))
    display[COL_AVG_RETURN] = display[COL_AVG_RETURN].map(lambda x: pct(x, 3))
    display[COL_PROFIT_LOSS_RATIO] = display[COL_PROFIT_LOSS_RATIO].map(lambda x: num(x, 2))
    rows = []
    for _, row in display.iterrows():
        cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in row)
        rows.append(f"<tr>{cells}</tr>")
    headers = "".join(f"<th>{html.escape(str(c))}</th>" for c in display.columns)
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def make_cards(data: dict, thresholds: list[float], etf_code: str, name: str) -> str:
    overall = data["overall"]
    trades = data["trades"]
    best_ret = overall.loc[overall[COL_TOTAL_RETURN].idxmax()]
    best_dd = overall.loc[overall[COL_MAX_DRAWDOWN].idxmax()]
    best_win = overall.loc[overall[COL_WIN_RATE].idxmax()]
    date_range = f"{trades['trade_date'].min():%Y-%m-%d} 至 {trades['trade_date'].max():%Y-%m-%d}"
    sel = ", ".join(threshold_label(x) for x in thresholds) or "-"

    num_years = (trades['trade_date'].max() - trades['trade_date'].min()).days / 365.25
    best_ann = (1 + float(best_ret[COL_TOTAL_RETURN])) ** (1 / num_years) - 1

    cards = [
        ("标的", f"{etf_code}", name),
        ("样本区间", date_range, f"{len(trades)} 个交易日"),
        ("累计收益最佳", threshold_label(float(best_ret[COL_THRESHOLD])),
         f"{pct(float(best_ret[COL_TOTAL_RETURN]))}  (年化 {pct(best_ann)})"),
        ("胜率最高阈值", threshold_label(float(best_win[COL_THRESHOLD])),
         pct(float(best_win[COL_WIN_RATE]))),
        ("净值曲线阈值", sel, "可用 --thresholds 调整"),
    ]
    return "".join(
        "<div class='kpi'><div class='kpi-title'>{}</div>"
        "<div class='kpi-value'>{}</div><div class='kpi-sub'>{}</div></div>".format(
            html.escape(t), html.escape(v), html.escape(s))
        for t, v, s in cards
    )


def build_html(data: dict, thresholds: list[float], top_n: int, etf_code: str, name: str) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = f"{etf_code} {name} 高低开策略分析"

    figures = [
        ("threshold_overview", make_threshold_overview(data["overall"])),
        ("equity_curves", make_equity_curves(data["trades"], thresholds)),
        ("yearly_heatmap", make_yearly_heatmap(data["yearly"])),
        ("bucket_heatmaps", make_bucket_heatmaps(data["buckets"])),
        ("daily_scatter", make_daily_scatter(data["trades"])),
    ]
    figure_html = "\n".join(
        "<section class='panel'><button type='button' class='fullscreen-btn' "
        f"data-target='{div_id}' onclick='toggleFullscreen(this)'>全屏</button>"
        f"{figure_div(fig, div_id)}</section>"
        for div_id, fig in figures
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
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
    .kpis {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .kpi, .panel, .table-panel {{ background: var(--panel); border: 1px solid var(--border);
      border-radius: 8px; box-shadow: 0 1px 2px rgba(15,23,42,0.04); }}
    .kpi {{ padding: 14px 16px; min-width: 0; }}
    .kpi-title {{ color: var(--muted); font-size: 12px; margin-bottom: 6px;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .kpi-value {{ font-size: 20px; font-weight: 700;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .kpi-sub {{ color: var(--muted); font-size: 12px; margin-top: 4px;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .panel {{ padding: 10px 12px; margin: 16px 0; position: relative; }}
    .fullscreen-btn {{ position: absolute; top: 12px; right: 14px; z-index: 5;
      border: 1px solid var(--border); border-radius: 6px;
      background: rgba(255,255,255,0.92); color: #334155; cursor: pointer;
      font-size: 12px; line-height: 1; padding: 7px 10px;
      box-shadow: 0 1px 2px rgba(15,23,42,0.08); }}
    .fullscreen-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
    .panel:fullscreen, .panel:-webkit-full-screen {{ width: 100vw; height: 100vh;
      margin: 0; padding: 48px 18px 18px; border: 0; border-radius: 0;
      box-sizing: border-box; overflow: hidden; background: #ffffff; }}
    .panel:fullscreen .plotly-graph-div, .panel:-webkit-full-screen .plotly-graph-div {{
      height: calc(100vh - 66px) !important; width: 100% !important; }}
    .panel:fullscreen .fullscreen-btn, .panel:-webkit-full-screen .fullscreen-btn {{
      top: 12px; right: 18px; }}
    .table-panel {{ padding: 18px; margin: 16px 0; overflow-x: auto; }}
    .table-panel h2 {{ margin: 0 0 12px; font-size: 18px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e7ebf2; padding: 8px 10px;
      text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: #334155; background: #f8fafc; font-weight: 650; }}
    @media (max-width: 1100px) {{ .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    @media (max-width: 640px) {{ header {{ padding: 22px 18px 16px; }}
      main {{ padding: 16px 12px 28px; }} .kpis {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <script>{get_plotlyjs()}</script>
  <header>
    <h1>{html.escape(title)}</h1>
    <div class="meta">生成时间：{html.escape(generated_at)} · 数据来源: MySQL rqalpha_etf</div>
  </header>
  <main>
    <section class="kpis">{make_cards(data, thresholds, etf_code, name)}</section>
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
    function requestPanelFullscreen(p) {{
      return p.requestFullscreen ? p.requestFullscreen() : p.webkitRequestFullscreen();
    }}
    function exitPanelFullscreen() {{
      return document.exitFullscreen ? document.exitFullscreen() : document.webkitExitFullscreen();
    }}
    function resizePlotlyPanel(p) {{
      if (!window.Plotly || !p) return;
      p.querySelectorAll('.plotly-graph-div').forEach(function(el) {{ Plotly.Plots.resize(el); }});
    }}
    function toggleFullscreen(btn) {{
      var p = btn.closest('.panel');
      if (!p) return;
      if (getFullscreenElement() === p) {{ exitPanelFullscreen(); }}
      else {{ requestPanelFullscreen(p).then(function() {{ setTimeout(function() {{ resizePlotlyPanel(p); }}, 80); }}); }}
    }}
    document.addEventListener('fullscreenchange', function() {{
      var a = getFullscreenElement();
      document.querySelectorAll('.fullscreen-btn').forEach(function(b) {{
        var p = b.closest('.panel');
        b.textContent = p && p === a ? '退出全屏' : '全屏';
        if (p) resizePlotlyPanel(p);
      }});
      if (a) setTimeout(function() {{ resizePlotlyPanel(a); }}, 120);
    }});
    document.addEventListener('webkitfullscreenchange', function() {{
      var a = getFullscreenElement();
      document.querySelectorAll('.fullscreen-btn').forEach(function(b) {{
        var p = b.closest('.panel');
        b.textContent = p && p === a ? '退出全屏' : '全屏';
        if (p) resizePlotlyPanel(p);
      }});
      if (a) setTimeout(function() {{ resizePlotlyPanel(a); }}, 120);
    }});
    window.addEventListener('resize', function() {{
      var a = getFullscreenElement();
      if (a) resizePlotlyPanel(a);
    }});
  </script>
</body>
</html>"""


def main() -> None:
    args = parse_args()
    conn = get_conn(args)

    etf_code = _resolve_etf_code(conn, args.etf)
    name = load_etf_name(conn, etf_code)
    print(f"分析标的: {etf_code}  {name}")

    data = load_data(conn, args)
    conn.close()

    trades = data["trades"]
    print(f"数据范围: {trades['trade_date'].min().date()} ~ {trades['trade_date'].max().date()}, "
          f"{len(trades)} 个交易日")

    thresholds = pick_thresholds(data["overall"], args.thresholds)

    out_path = Path(args.out) if args.out else (
        DEFAULT_OUT_DIR / f"{etf_code.replace('.', '_')}_gap_report.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html_text = build_html(data, thresholds, args.top_n, etf_code, name)
    out_path.write_text(html_text, encoding="utf-8")
    print(f"HTML 报告已生成: {out_path.resolve()}")


if __name__ == "__main__":
    main()
