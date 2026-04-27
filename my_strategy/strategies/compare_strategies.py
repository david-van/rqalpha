"""
用法: python compare_strategies.py strategy_sz180.pkl strategy_dividend_etf.pkl
"""
import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def compare(path1, path2, name1="上证180", name2="红利ETF"):
    d1, d2 = load(path1), load(path2)

    # ========== 1. Summary ==========
    print("\n" + "=" * 70)
    print("                    汇 总 指 标 对 比")
    print("=" * 70)
    s1 = d1.get("summary", {})
    s2 = d2.get("summary", {})
    df_s = pd.DataFrame({name1: s1, name2: s2})
    if not df_s.empty:
        print(df_s.to_string())

    # ========== 2. 净值 ==========
    nav1 = d1["portfolio"]["unit_net_value"]
    nav2 = d2["portfolio"]["unit_net_value"]
    common = nav1.index.intersection(nav2.index)
    nav1, nav2 = nav1.loc[common], nav2.loc[common]
    ret1, ret2 = nav1.pct_change(), nav2.pct_change()

    # ========== 3. 月度拆解 ==========
    m1 = nav1.resample("M").last().pct_change()
    m2 = nav2.resample("M").last().pct_change()
    md = (m1 - m2).dropna()

    print("\n" + "=" * 70)
    print(f"  月度差异（{name1} - {name2}）, 负值 = {name2}落后")
    print("=" * 70)
    md_df = pd.DataFrame({name1: m1, name2: m2, "差异": md}).dropna()
    print("落后最多的5个月:")
    print(md_df.nsmallest(5, "差异").to_string())
    print("\n领先最多的5个月:")
    print(md_df.nlargest(5, "差异").to_string())

    # ========== 4. 绘图 ==========
    fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=False)

    axes[0].plot(nav1, label=name1)
    axes[0].plot(nav2, label=name2)
    axes[0].set_title("净值曲线")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(nav1 - nav2, color="purple")
    axes[1].axhline(0, color="gray", ls="--")
    axes[1].set_title(f"净值差 ({name1} - {name2})")
    axes[1].grid(True, alpha=0.3)

    colors = ["green" if x >= 0 else "red" for x in md]
    axes[2].bar(md.index, md.values, width=20, color=colors, alpha=0.7)
    axes[2].set_title("月度收益差异")
    axes[2].grid(True, alpha=0.3)

    # 回撤对比
    dd1 = nav1 / nav1.cummax() - 1
    dd2 = nav2 / nav2.cummax() - 1
    axes[3].fill_between(dd1.index, dd1, 0, alpha=0.3, label=name1)
    axes[3].fill_between(dd2.index, dd2, 0, alpha=0.3, label=name2)
    axes[3].set_title("回撤对比")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("strategy_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n图片已保存: strategy_comparison.png")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        compare(sys.argv[1], sys.argv[2])
    else:
        # 直接修改这里的路径
        compare("batch_results/baseline.pkl", "batch_results/replace_180.pkl")
