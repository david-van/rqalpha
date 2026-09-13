#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc : 小鹅通低波动策略 — 回测结果交互式 HTML 报告
# @Usage: .venv/Scripts/python.exe -u my_strategy/strategies/xiaoe/lowvol_mktfilter/analysis_html.py
"""
读取 batch_results/xiaoe_pool/lowvol_mktfilter/ 下的 .pkl 文件，
生成单个交互式 HTML（plotly 离线内嵌，双击即可在浏览器打开，无需联网）。

包含：
  1) 绩效汇总表
  2) 净值 + 回撤曲线（主配置 / 时间分段 + 沪深300 基准）
  3) 参数敏感性（按参数分组，收益 / 回撤 / 夏普）
  4) 风险收益散点（气泡=夏普）
  5) 逐年收益热力图
  6) 月度收益热力图（final 配置）
"""
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs

from my_strategy.common_file import project_root

warnings.filterwarnings("ignore")

RESULT_DIR = Path(project_root) / 'my_strategy' / 'strategies' / 'batch_results' / 'xiaoe_pool' / 'lowvol_mktfilter'
OUTPUT_HTML = RESULT_DIR / 'analysis_dashboard.html'

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# ============================================================
# 参数说明映射：每个 tag 的完整参数（默认 = 最终推荐配置）
# ============================================================
DEFAULT_PARAMS = {
    "base_ratio": 0.5, "enhance_ratio": 0.5, "enhance_top_n": 2, "low_vol_days": 60,
    "rebalance_days": 5, "market_ma_days": 60, "weak_market_exposure": 0.60,
    "market_filter": True, "market_filter_lag": True,
}

TAG_LABELS = {
    "final": "最终配置（周频 + MA60风控）",
    "final_nofilter": "关闭大盘风控",
    "final_daily": "日频再平衡（1日）",
    "full": "全程（同 final，用于逐年）",
    "first_half_2020_2022": "前半段 2020-2022",
    "second_half_2023_2025": "后半段 2023-2025",
    "lowvol30": "波动窗口=30日",
    "lowvol60": "波动窗口=60日",
    "lowvol90": "波动窗口=90日",
    "lowvol120": "波动窗口=120日",
    "top1": "增强仓=1只",
    "top2": "增强仓=2只",
    "top3": "增强仓=3只",
    "mktma30": "均线周期=30日",
    "mktma60": "均线周期=60日",
    "mktma120": "均线周期=120日",
    "mktma200": "均线周期=200日",
    "base40": "底仓40%+增强60%",
    "base50": "底仓50%+增强50%",
    "base60": "底仓60%+增强40%",
    "base70": "底仓70%+增强30%",
    "rebal3": "每3日再平衡",
    "rebal5": "每5日再平衡",
    "rebal10": "每10日再平衡",
    "rebal20": "每20日再平衡",
    "weak40": "弱市暴露40%",
    "weak60": "弱市暴露60%",
    "weak80": "弱市暴露80%",
}

TAG_OVERRIDES = {
    "final": {}, "full": {},
    "final_nofilter": {"market_filter": False},
    "final_daily": {"rebalance_days": 1},
    "lowvol30": {"low_vol_days": 30},
    "lowvol60": {"low_vol_days": 60},
    "lowvol90": {"low_vol_days": 90},
    "lowvol120": {"low_vol_days": 120},
    "top1": {"enhance_top_n": 1},
    "top2": {"enhance_top_n": 2},
    "top3": {"enhance_top_n": 3},
    "mktma30": {"market_ma_days": 30},
    "mktma60": {"market_ma_days": 60},
    "mktma120": {"market_ma_days": 120},
    "mktma200": {"market_ma_days": 200},
    "base40": {"base_ratio": 0.4, "enhance_ratio": 0.6},
    "base50": {"base_ratio": 0.5, "enhance_ratio": 0.5},
    "base60": {"base_ratio": 0.6, "enhance_ratio": 0.4},
    "base70": {"base_ratio": 0.7, "enhance_ratio": 0.3},
    "rebal3": {"rebalance_days": 3},
    "rebal5": {"rebalance_days": 5},
    "rebal10": {"rebalance_days": 10},
    "rebal20": {"rebalance_days": 20},
    "weak40": {"weak_market_exposure": 0.40},
    "weak60": {"weak_market_exposure": 0.60},
    "weak80": {"weak_market_exposure": 0.80},
}

TAG_TIME_RANGE = {
    "first_half_2020_2022": ("2020-01-01", "2023-01-01"),
    "second_half_2023_2025": ("2023-01-01", "2026-01-01"),
}

# ============================================================
# 分类参数对比：每个维度一张表，参数值从小到大排列，高亮最终配置
# ============================================================
CATEGORY_TABLES = [
    ("大盘风控 · 开关", [("开启（最终配置）", "final"), ("关闭", "final_nofilter")]),
    ("大盘风控 · 均线周期", [("30日", "mktma30"), ("60日", "mktma60"), ("120日", "mktma120"), ("200日", "mktma200")]),
    ("大盘风控 · 弱市暴露", [("40%", "weak40"), ("60%", "weak60"), ("80%", "weak80")]),
    ("低波动增强 · 波动窗口", [("30日", "lowvol30"), ("60日", "lowvol60"), ("90日", "lowvol90"), ("120日", "lowvol120")]),
    ("低波动增强 · 增强只数", [("1只", "top1"), ("2只", "top2"), ("3只", "top3")]),
    ("仓位结构 · 底仓/增强", [("40%/60%", "base40"), ("50%/50%", "base50"), ("60%/40%", "base60"), ("70%/30%", "base70")]),
    ("再平衡频率", [("每3日", "rebal3"), ("每5日", "rebal5"), ("每10日", "rebal10"), ("每20日", "rebal20")]),
    ("时间段稳定性", [("全程 2020-2026", "full"), ("前半 2020-2022", "first_half_2020_2022"), ("后半 2023-2025", "second_half_2023_2025")]),
]

# 每个维度里代表「最终配置」的那一行（高亮）
DEFAULT_TAGS = {"final", "mktma60", "lowvol60", "top2", "base50", "rebal5", "weak60", "full"}


def full_params(tag: str) -> dict:
    p = dict(DEFAULT_PARAMS)
    p.update(TAG_OVERRIDES.get(tag, {}))
    return p


def time_range(tag: str) -> str:
    s, e = TAG_TIME_RANGE.get(tag, ("2020-01-01", "2026-01-01"))
    return f"{s[:4]}-{e[:4]}"


# 主配置 + 时间分段（用于曲线对比）
CURVE_TAGS = ["final", "final_nofilter", "final_daily"]

# 参数敏感性分组: (参数名, [(tag, 显示值), ...])
SENSITIVITY_GROUPS = [
    ("low_vol_days（波动率窗口）", [("lowvol30", "30"), ("lowvol60", "60"), ("lowvol90", "90"), ("lowvol120", "120")]),
    ("enhance_top_n（增强仓只数）", [("top1", "1"), ("top2", "2"), ("top3", "3")]),
    ("market_ma_days（均线周期）", [("mktma30", "30"), ("mktma60", "60"), ("mktma120", "120"), ("mktma200", "200")]),
    ("base_ratio（底仓/增强）", [("base40", "40/60"), ("base50", "50/50"), ("base60", "60/40"), ("base70", "70/30")]),
    ("rebalance_days（再平衡周期）", [("rebal3", "3"), ("rebal5", "5"), ("rebal10", "10"), ("rebal20", "20")]),
    ("weak_exposure（弱市暴露）", [("weak40", "40%"), ("weak60", "60%"), ("weak80", "80%")]),
]


def load_all_results() -> dict:
    results = {}
    for p in sorted(RESULT_DIR.glob("*.pkl")):
        with open(p, "rb") as f:
            data = pickle.load(f)
        results[p.stem] = data
    return results


def _as_dt_index(s):
    s = s.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    return s


def _get_nv(data) -> pd.Series | None:
    pf = data.get("portfolio", pd.DataFrame())
    if pf is None or pf.empty or "unit_net_value" not in pf.columns:
        return None
    return _as_dt_index(pf["unit_net_value"].dropna())


def drawdown_series(nv: pd.Series) -> pd.Series:
    return nv / nv.cummax() - 1


def summary_table_html(results: dict) -> str:
    rows = []
    for tag, d in results.items():
        s = d.get("summary", {})
        trades = d.get("trades", pd.DataFrame())
        rows.append({
            "tag": tag,
            "label": TAG_LABELS.get(tag, ""),
            "total": s.get("total_returns", 0),
            "ann": s.get("annualized_returns", 0),
            "sharpe": s.get("sharpe", 0),
            "mdd": s.get("max_drawdown", 0),
            "turnover": s.get("turnover", 0),
            "trades": len(trades),
        })
    df = pd.DataFrame(rows).sort_values("total", ascending=False)
    thead = "<tr><th>配置</th><th>说明</th><th>累计收益</th><th>年化</th><th>夏普</th><th>最大回撤</th><th>换手率</th><th>交易笔数</th></tr>"
    body = ""
    for _, r in df.iterrows():
        body += (f"<tr><td>{r['tag']}</td><td>{r['label']}</td><td>{r['total']:.2%}</td><td>{r['ann']:.2%}</td>"
                 f"<td>{r['sharpe']:.3f}</td><td>{r['mdd']:.2%}</td><td>{r['turnover']:.1f}</td>"
                 f"<td>{int(r['trades'])}</td></tr>")
    return f"<table class='summary-table'><thead>{thead}</thead><tbody>{body}</tbody></table>"


def param_table_html(results: dict) -> str:
    """每个配置的完整参数表，按最终推荐排序"""
    order = ["final", "final_nofilter", "final_daily",
             "lowvol30", "lowvol60", "lowvol90", "lowvol120",
             "top1", "top2", "top3",
             "mktma30", "mktma60", "mktma120", "mktma200",
             "base40", "base50", "base60", "base70",
             "rebal3", "rebal5", "rebal10", "rebal20",
             "weak40", "weak60", "weak80",
             "full", "first_half_2020_2022", "second_half_2023_2025"]
    tags = [t for t in order if t in results]
    tags += [t for t in sorted(results) if t not in tags]

    thead = ("<tr><th>配置</th><th>说明</th><th>底仓/增强</th><th>增强只数</th>"
             "<th>波动窗口</th><th>再平衡</th><th>均线周期</th><th>弱市暴露</th>"
             "<th>风控</th><th>时间区间</th></tr>")
    body = ""
    for tag in tags:
        p = full_params(tag)
        mf = "开" if p["market_filter"] else "关"
        body += (
            f"<tr><td>{tag}</td><td>{TAG_LABELS.get(tag, '')}</td>"
            f"<td>{p['base_ratio']:.0%}/{p['enhance_ratio']:.0%}</td>"
            f"<td>{p['enhance_top_n']}</td>"
            f"<td>{p['low_vol_days']}日</td>"
            f"<td>{p['rebalance_days']}日</td>"
            f"<td>{p['market_ma_days']}日</td>"
            f"<td>{p['weak_market_exposure']:.0%}</td>"
            f"<td>{mf}</td>"
            f"<td>{time_range(tag)}</td></tr>"
        )
    return f"<table class='summary-table'><thead>{thead}</thead><tbody>{body}</tbody></table>"


def grouped_tables_html(results: dict) -> str:
    """按分类生成参数对比表：每个维度一张表，高亮最终配置"""
    blocks = []
    for cat_name, pairs in CATEGORY_TABLES:
        valid = [(label, tag) for label, tag in pairs if tag in results]
        if not valid:
            continue
        thead = ("<tr><th>参数</th><th>配置</th><th>累计收益</th><th>年化</th>"
                 "<th>夏普</th><th>最大回撤</th><th>换手率</th><th>交易笔数</th></tr>")
        body = ""
        for label, tag in valid:
            s = results[tag].get("summary", {})
            trades = results[tag].get("trades", pd.DataFrame())
            hl = ' class="highlight"' if tag in DEFAULT_TAGS else ""
            body += (f"<tr{hl}><td>{label}</td><td>{tag}</td>"
                     f"<td>{s.get('total_returns', 0):.2%}</td>"
                     f"<td>{s.get('annualized_returns', 0):.2%}</td>"
                     f"<td>{s.get('sharpe', 0):.3f}</td>"
                     f"<td>{s.get('max_drawdown', 0):.2%}</td>"
                     f"<td>{s.get('turnover', 0):.1f}</td>"
                     f"<td>{len(trades)}</td></tr>")
        blocks.append(
            f"<h3>{cat_name}</h3>"
            f"<table class='summary-table'><thead>{thead}</thead><tbody>{body}</tbody></table>"
        )
    return "".join(blocks)


def curves_figure(results: dict) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("净值曲线", "回撤曲线"))
    # 时间分段（若存在）
    tags = [t for t in CURVE_TAGS if t in results]
    if "first_half_2020_2022" in results:
        tags += ["first_half_2020_2022", "second_half_2023_2025"]
    for i, tag in enumerate(tags):
        nv = _get_nv(results[tag])
        if nv is None:
            continue
        fig.add_trace(go.Scatter(x=nv.index, y=nv.values, mode="lines", name=tag,
                                 line=dict(color=COLORS[i % len(COLORS)], width=1.2)), row=1, col=1)
        dd = drawdown_series(nv)
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name=tag,
                                 line=dict(color=COLORS[i % len(COLORS)], width=0.8),
                                 showlegend=False), row=2, col=1)
    # 基准
    first = results.get("final") or next(iter(results.values()))
    pf = first.get("portfolio", pd.DataFrame())
    if pf is not None and not pf.empty and "benchmark_unit_net_value" in pf.columns:
        bm = _as_dt_index(pf["benchmark_unit_net_value"].dropna())
        fig.add_trace(go.Scatter(x=bm.index, y=bm.values, mode="lines", name="沪深300",
                                 line=dict(color="gray", width=1.0, dash="dash")), row=1, col=1)
    fig.update_yaxes(title_text="单位净值", row=1, col=1)
    fig.update_yaxes(title_text="回撤", tickformat=".0%", row=2, col=1)
    fig.add_hline(y=0, line=dict(color="black", width=0.5), row=2, col=1)
    fig.update_layout(title="净值与回撤", height=800, hovermode="closest",
                      legend=dict(orientation="h", y=-0.05))
    return fig


def sensitivity_figures(results: dict) -> list:
    figs = []
    for param_name, pairs in SENSITIVITY_GROUPS:
        valid = [(label, tag) for tag, label in pairs if tag in results]
        if len(valid) < 2:
            continue
        labels = [v[0] for v in valid]
        tags = [v[1] for v in valid]
        ann = [results[t]["summary"].get("annualized_returns", 0) for t in tags]
        mdd = [results[t]["summary"].get("max_drawdown", 0) for t in tags]
        sharpe = [results[t]["summary"].get("sharpe", 0) for t in tags]
        fig = make_subplots(rows=1, cols=3, subplot_titles=("年化收益", "最大回撤", "夏普"))
        fig.add_trace(go.Scatter(x=labels, y=ann, mode="lines+markers",
                                 line=dict(color=COLORS[0], width=2), marker=dict(size=8)), row=1, col=1)
        fig.add_trace(go.Scatter(x=labels, y=mdd, mode="lines+markers",
                                 line=dict(color=COLORS[3], width=2), marker=dict(size=8)), row=1, col=2)
        fig.add_trace(go.Scatter(x=labels, y=sharpe, mode="lines+markers",
                                 line=dict(color=COLORS[1], width=2), marker=dict(size=8)), row=1, col=3)
        fig.update_yaxes(tickformat=".0%", row=1, col=1)
        fig.update_yaxes(tickformat=".0%", row=1, col=2)
        fig.update_layout(title=param_name, height=320, showlegend=False, margin=dict(t=40, b=40))
        figs.append(fig)
    return figs


def risk_return_figure(results: dict) -> go.Figure:
    fig = go.Figure()
    for i, (tag, d) in enumerate(results.items()):
        s = d.get("summary", {})
        fig.add_trace(go.Scatter(
            x=[s.get("max_drawdown", 0)], y=[s.get("annualized_returns", 0)],
            mode="markers+text", name=tag, text=[tag], textposition="top right",
            textfont=dict(size=9),
            marker=dict(color=COLORS[i % len(COLORS)], size=14 + s.get("sharpe", 0) * 6,
                        line=dict(color="black", width=0.5)),
            hovertemplate=f"<b>{tag}</b><br>回撤 %{{x:.2%}}<br>年化 %{{y:.2%}}<br>夏普 {s.get('sharpe', 0):.3f}<extra></extra>",
        ))
    fig.update_layout(title="风险收益散点（气泡=夏普）", height=550,
                      xaxis=dict(title="最大回撤", tickformat=".0%"),
                      yaxis=dict(title="年化收益", tickformat=".0%"), showlegend=False)
    fig.add_hline(y=0, line=dict(color="black", width=0.5))
    return fig


def yearly_figure(results: dict) -> go.Figure | None:
    tag = "final" if "final" in results else (next(iter(results.keys())))
    nv = _get_nv(results[tag])
    if nv is None:
        return None
    yearly = {}
    for y, g in nv.groupby(nv.index.year):
        if len(g) >= 2:
            yearly[y] = float(g.iloc[-1] / g.iloc[0] - 1)
    years = sorted(yearly)
    vals = [yearly[y] for y in years]
    fig = go.Figure(go.Bar(
        x=[str(y) for y in years], y=vals,
        marker=dict(color=["#2ca02c" if v >= 0 else "#d62728" for v in vals]),
        text=[f"{v:.1%}" for v in vals], textposition="outside",
    ))
    fig.add_hline(y=0, line=dict(color="black", width=0.5))
    fig.update_layout(title=f"逐年收益（{tag}）", height=420, yaxis=dict(tickformat=".0%"))
    return fig


def monthly_heatmap(results: dict) -> go.Figure | None:
    tag = "final" if "final" in results else next(iter(results.keys()))
    nv = _get_nv(results[tag])
    if nv is None:
        return None
    daily_ret = nv.pct_change().dropna()
    monthly = daily_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    rec = [{"year": dt.year, "month": dt.month, "ret": r} for dt, r in monthly.items()]
    pivot = pd.DataFrame(rec).pivot_table(index="year", columns="month", values="ret", sort=False)
    pivot = pivot.reindex(columns=list(range(1, 13)))
    z = pivot.values
    text = [[f"{v:.1%}" if not np.isnan(v) else "" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{m}月" for m in range(1, 13)], y=[str(y) for y in pivot.index],
        text=text, texttemplate="%{text}", textfont=dict(size=10),
        colorscale="RdYlGn", zmid=0, colorbar=dict(title="收益", tickformat=".0%"),
    ))
    fig.update_layout(title=f"月度收益热力图（{tag}）", height=400,
                      yaxis=dict(autorange="reversed"))
    return fig


def build_dashboard(results: dict, output: Path):
    plotly_js = get_plotlyjs()

    figures = {
        "curves": curves_figure(results),
        "risk": risk_return_figure(results),
        "yearly": yearly_figure(results),
        "monthly": monthly_heatmap(results),
    }
    sens_figs = sensitivity_figures(results)

    def fig_div(fig, div_id):
        spec = json.loads(fig.to_json())
        return (f"<div id='{div_id}' class='plotly-graph-div'></div>\n"
                f"<script>FIGURES['{div_id}']={json.dumps(spec)};</script>")

    fig_blocks = []
    fig_blocks.append(fig_div(figures["curves"], "curves"))
    fig_blocks.append(fig_div(figures["risk"], "risk"))
    if figures["yearly"] is not None:
        fig_blocks.append(fig_div(figures["yearly"], "yearly"))
    if figures["monthly"] is not None:
        fig_blocks.append(fig_div(figures["monthly"], "monthly"))
    for i, f in enumerate(sens_figs):
        fig_blocks.append(fig_div(f, f"sens{i}"))

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>小鹅通低波动策略 — 回测报告</title>
<style>
  body {{ font-family: 'Microsoft YaHei','SimHei','PingFang SC',sans-serif; background:#f0f2f5; color:#333; margin:0; }}
  .header {{ background:linear-gradient(135deg,#1a2a3a,#2c3e50); color:#fff; padding:24px 32px; }}
  .header h1 {{ margin:0; font-size:24px; }}
  .header .meta {{ font-size:13px; opacity:.7; margin-top:6px; }}
  .section {{ background:#fff; border-radius:8px; box-shadow:0 1px 4px rgba(0,0,0,.08); padding:20px; margin:20px; }}
  .section h2 {{ font-size:17px; color:#2c3e50; border-left:4px solid #1a73e8; padding-left:10px; margin-top:0; }}
  .plotly-graph-div {{ min-height:480px; }}
  .summary-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  .summary-table th {{ background:#2c3e50; color:#fff; padding:10px 12px; text-align:left; }}
  .summary-table td {{ padding:8px 12px; border-bottom:1px solid #ecf0f1; }}
  .summary-table tbody tr:hover {{ background:#f0f4ff; }}
  .summary-table tr.highlight td {{ background:#fff3cd; font-weight:600; }}
  .section h3 {{ font-size:15px; color:#1a73e8; margin:18px 0 8px; }}
  .footer {{ text-align:center; color:#999; font-size:12px; padding:20px; }}
</style>
</head>
<body>
<div class="header">
  <h1>小鹅通低波动策略 — 回测报告</h1>
  <div class="meta">低波动增强 + 大盘趋势风控（周频）| 数据目录: {output.parent} | 生成时间: {generated_at}</div>
</div>

<div class="section"><h2>分类参数对比</h2>{grouped_tables_html(results)}<p style="color:#888;font-size:12px;margin-top:8px;">高亮行为「最终配置」，其余行只改该维度一个参数。</p></div>
<div class="section"><h2>绩效汇总</h2>{summary_table_html(results)}</div>
<div class="section"><h2>配置参数说明</h2>{param_table_html(results)}</div>
<div class="section"><h2>净值与回撤曲线</h2><div id="curves" class="plotly-graph-div"></div></div>
<div class="section"><h2>风险收益散点</h2><div id="risk" class="plotly-graph-div"></div></div>
<div class="section"><h2>逐年收益</h2><div id="yearly" class="plotly-graph-div"></div></div>
<div class="section"><h2>月度收益热力图</h2><div id="monthly" class="plotly-graph-div"></div></div>
<div class="section"><h2>参数敏感性</h2>{''.join(f'<div id="sens{i}" class="plotly-graph-div"></div>' for i in range(len(sens_figs)))}</div>

<div class="footer">RQAlpha 策略回测分析 | Powered by Plotly</div>

<script>{plotly_js}</script>
<script>
var FIGURES = {{}};
</script>
{''.join(fig_blocks)}
<script>
Object.keys(FIGURES).forEach(function(k) {{
  var spec = FIGURES[k];
  Plotly.newPlot(k, spec.data, spec.layout, {{responsive:true, displaylogo:false}});
}});
</script>
</body>
</html>"""

    output.write_text(html, encoding="utf-8")
    print(f"HTML 报告已生成: {output}")


def main():
    print("加载 .pkl 结果...")
    results = load_all_results()
    if not results:
        print(f"未找到 .pkl 文件: {RESULT_DIR}")
        return
    print(f"共 {len(results)} 个结果: {', '.join(sorted(results.keys()))}")
    build_dashboard(results, OUTPUT_HTML)


if __name__ == "__main__":
    main()
