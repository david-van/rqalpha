#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 result.pkl 提取月度/年度收益，与聚宽平台（策略收益.txt）对比
"""

import pickle

import pandas as pd


# ============================================================
#  第一步：从 result.pkl 读取并计算月度收益
# ============================================================
def load_rqalpha_monthly(pkl_path):
    """从 result.pkl 中提取月度收益率"""
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)

    portfolio = result['portfolio']

    if not isinstance(portfolio.index, pd.DatetimeIndex):
        portfolio.index = pd.to_datetime(portfolio.index)

    # 取每月最后一个交易日的净值
    monthly_nav = portfolio['unit_net_value'].resample('ME').last()

    # 月度收益率
    monthly_return = monthly_nav.pct_change()
    monthly_return.iloc[0] = monthly_nav.iloc[0] / 1.0 - 1

    # 格式化 index 为 "YYYY-MM"
    monthly_return.index = monthly_return.index.strftime('%Y-%m')

    return monthly_return


# ============================================================
#  第二步：从 策略收益.txt 读取聚宽月度收益
# ============================================================
def load_jukuan_monthly(txt_path):
    """读取聚宽导出的策略收益文件"""
    with open(txt_path, 'r', encoding='utf-8') as f:
        raw = f.read()

    print("===== 文件前500字符 =====")
    print(repr(raw[:500]))
    print("=========================\n")

    for sep in ['\t', ',', r'\s+']:
        try:
            if sep == r'\s+':
                df = pd.read_csv(
                    txt_path, sep=sep, engine='python',
                    na_values=['NaN', 'nan', ''],
                    encoding='utf-8'
                )
            else:
                df = pd.read_csv(
                    txt_path, sep=sep,
                    na_values=['NaN', 'nan', ''],
                    encoding='utf-8'
                )

            if df.shape[1] >= 2 and df.shape[0] > 0:
                print(f"成功解析，分隔符: {repr(sep)}, shape: {df.shape}")
                print(f"列名: {list(df.columns)}")
                print(f"前3行:\n{df.head(3)}\n")
                break
        except Exception as e:
            print(f"分隔符 {repr(sep)} 失败: {e}")
            continue
    else:
        raise ValueError("所有分隔符都无法解析文件")

    date_col = df.columns[0]
    df[date_col] = df[date_col].astype(str).str.strip()
    df = df.set_index(date_col)
    df.index.name = 'date'
    df.columns = df.columns.str.strip()

    print(f"日期范围: {df.index[0]} ~ {df.index[-1]}")
    print()

    col_1m = None
    for col in df.columns:
        if '1个月' in col or '1m' in col.lower():
            col_1m = col
            break

    if col_1m is None:
        col_1m = df.columns[0]
        print(f"⚠️  未找到'1个月'列，使用第一列: '{col_1m}'")

    jk_monthly = df[col_1m].astype(float)
    jk_monthly.name = 'jukuan'

    return jk_monthly


# ============================================================
#  第三步：月度对比
# ============================================================
def compare_monthly(rq_monthly, jk_monthly):
    df = pd.DataFrame({
        'rqalpha': rq_monthly,
        'jukuan': jk_monthly,
    })

    df_both = df.dropna(subset=['rqalpha', 'jukuan']).copy()
    df_both['diff'] = df_both['rqalpha'] - df_both['jukuan']
    df_both['diff_abs'] = df_both['diff'].abs()

    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    pd.set_option('display.max_rows', 200)

    print("=" * 70)
    print("月度收益对比：RQAlpha vs 聚宽")
    print("=" * 70)
    print(df_both[['rqalpha', 'jukuan', 'diff']].to_string())

    print("\n" + "=" * 70)
    print("月度差异统计")
    print("=" * 70)
    print(f"对比月份数:   {len(df_both)}")
    print(f"平均差异:     {df_both['diff'].mean():.6f}")
    print(f"平均绝对差异: {df_both['diff_abs'].mean():.6f}")
    print(f"最大差异:     {df_both['diff'].max():.6f}  ({df_both['diff'].idxmax()})")
    print(f"最小差异:     {df_both['diff'].min():.6f}  ({df_both['diff'].idxmin()})")
    print(f"差异标准差:   {df_both['diff'].std():.6f}")

    big_diff = df_both[df_both['diff_abs'] > 0.005]
    if len(big_diff) > 0:
        print(f"\n⚠️  差异 > 0.5% 的月份 ({len(big_diff)}个):")
        print(big_diff[['rqalpha', 'jukuan', 'diff']].to_string())
    else:
        print("\n✅ 所有月份差异均 < 0.5%")

    # 累计净值
    print("\n" + "=" * 70)
    print("累计净值对比")
    print("=" * 70)
    df_both['rq_cum'] = (1 + df_both['rqalpha']).cumprod()
    df_both['jk_cum'] = (1 + df_both['jukuan']).cumprod()
    df_both['cum_diff'] = df_both['rq_cum'] - df_both['jk_cum']
    print(df_both[['rq_cum', 'jk_cum', 'cum_diff']].to_string())

    return df_both


# ============================================================
#  第四步：年度对比
# ============================================================
def compare_yearly(rq_monthly, jk_monthly):
    """从月度收益聚合为年度收益并对比"""

    # 提取年份
    rq_years = pd.Series(rq_monthly.values, index=rq_monthly.index).copy()
    jk_years = pd.Series(jk_monthly.values, index=jk_monthly.index).copy()

    # 按年聚合：年收益 = 各月(1+r)连乘 - 1
    def aggregate_yearly(monthly_series):
        yearly = {}
        for idx, val in monthly_series.items():
            year = idx[:4]  # "2018-01" → "2018"
            if pd.notna(val):
                if year not in yearly:
                    yearly[year] = 1.0
                yearly[year] *= (1 + val)

        # 转为收益率
        for year in yearly:
            yearly[year] -= 1.0

        return pd.Series(yearly, dtype=float)

    rq_yearly = aggregate_yearly(rq_monthly)
    jk_yearly = aggregate_yearly(jk_monthly)

    df_year = pd.DataFrame({
        'rqalpha': rq_yearly,
        'jukuan': jk_yearly,
    })

    df_year = df_year.dropna(subset=['rqalpha', 'jukuan']).copy()
    df_year['diff'] = df_year['rqalpha'] - df_year['jukuan']
    df_year['diff_abs'] = df_year['diff'].abs()

    # 年度累计净值
    df_year['rq_cum'] = (1 + df_year['rqalpha']).cumprod()
    df_year['jk_cum'] = (1 + df_year['jukuan']).cumprod()
    df_year['cum_diff'] = df_year['rq_cum'] - df_year['jk_cum']

    pd.set_option('display.float_format', lambda x: f'{x:.4f}')

    print("\n" + "=" * 70)
    print("年度收益对比：RQAlpha vs 聚宽")
    print("=" * 70)
    print(df_year[['rqalpha', 'jukuan', 'diff']].to_string())

    print("\n" + "=" * 70)
    print("年度累计净值对比")
    print("=" * 70)
    print(df_year[['rq_cum', 'jk_cum', 'cum_diff']].to_string())

    print("\n" + "=" * 70)
    print("年度差异统计")
    print("=" * 70)
    print(f"对比年份数:   {len(df_year)}")
    print(f"平均年度差异: {df_year['diff'].mean():.6f}")
    print(f"平均绝对差异: {df_year['diff_abs'].mean():.6f}")
    print(f"最大差异:     {df_year['diff'].max():.6f}  ({df_year['diff'].idxmax()})")
    print(f"最小差异:     {df_year['diff'].min():.6f}  ({df_year['diff'].idxmin()})")

    # 胜率对比：两边同涨同跌的年份
    same_direction = ((df_year['rqalpha'] > 0) == (df_year['jukuan'] > 0)).sum()
    print(f"\n方向一致年份: {same_direction}/{len(df_year)}")

    # 哪一年差异最大，标注
    worst_year = df_year['diff_abs'].idxmax()
    print(f"差异最大年份: {worst_year}  "
          f"(RQ: {df_year.loc[worst_year, 'rqalpha']:.4f}, "
          f"JK: {df_year.loc[worst_year, 'jukuan']:.4f}, "
          f"差: {df_year.loc[worst_year, 'diff']:.4f})")

    return df_year


# ============================================================
#  主程序
# ============================================================
if __name__ == '__main__':
    pkl_path = 'result.pkl'
    txt_path = 'jukuan/策略收益.txt'

    # 加载数据
    rq_monthly = load_rqalpha_monthly(pkl_path)
    jk_monthly = load_jukuan_monthly(txt_path)

    # 月度对比
    df_month = compare_monthly(rq_monthly, jk_monthly)

    # 年度对比
    df_year = compare_yearly(rq_monthly, jk_monthly)
