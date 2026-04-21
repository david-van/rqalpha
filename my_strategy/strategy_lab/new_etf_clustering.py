#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2026/3/22 22:06
# @Author  : david_van
# @Desc    :
# from jqdata import *
# from jqfactor import *
import numpy as np
import pandas as pd
import datetime
import unicodedata
import builtins
import math
import networkx as nx
from sklearn.cluster import AffinityPropagation, DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from collections import defaultdict

from rqalpha.apis import *

# -------------------- 配置区域 (Initial Configuration) --------------------
CONFIG = {
    # 基础过滤参数
    "min_liquidity": 50e6,       # 最小成交额 (5000万)
    "min_listing_days": 60,      # 最小上市天数
    "correlation_window": 120,      # 计算相关性的回看窗口 (天)

    # 类型过滤配置 (True=剔除, False=保留)
    "filter_bond_money": True,    # 剔除债券、货币
    "filter_qdii": False,         # 剔除跨境/QDII
    "filter_gold": False,         # 剔除黄金
    "black_list": ["510900.XSHG"],  # 黑名单

    # 聚类方法选择 ("hierarchical", "ap", "mst", "dbscan")
    "clustering_method": "ap",

    # Hierarchical聚类参数
    "cluster_corr_threshold": 0.85, # 聚类相关性阈值 (决定簇的数量/大小)

    # AP聚类参数
    "ap_damping": 0.8,           # 阻尼系数 (0.5-1.0)，防止震荡，越大越稳定
    "ap_preference": None,       # 偏好值 (None = 使用中位数)，值越大簇越多

    # MST聚类参数
    "mst_n_clusters": 10,        # MST聚类目标簇数量 (切断由强到弱的K-1条边)

    # DBSCAN聚类参数
    "dbscan_eps": 0.5,           # DBSCAN 邻域半径
    "dbscan_min_samples": 2,     # DBSCAN 最小样本数

    # 动量评分参数
    "score_window": 25,          # 动量评分的回看窗口 (天)
    "min_r2": 0.7,               # 最小 R 方
    "max_annualized_returns": 9.99, # 最大年化收益 (过滤异常值)

    # PCA 分析参数
    "pca_rolling_window": 60,    # 滚动 PCA 窗口 (天)
    "pca_overlay_etf": "510300.XSHG", # 叠加显示的基准 ETF (如沪深300)，为 None 则不显示
}

# -------------------- 工具函数 --------------------------
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)

today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)
def get_history_data(security_list, end_date, count, field="close"):
    if isinstance(security_list, str):
        security_list = [security_list]

    df = get_price(list(security_list), end_date=end_date, count=count,
                   frequency='daily', fields=[field], panel=False)

    if df.empty:
        return pd.DataFrame()

    if 'code' in df.columns:
        index_col = 'time' if 'time' in df.columns else 'date' if 'date' in df.columns else None

        if index_col:
            # Pivot to Index=Date, Columns=Code
            return df.pivot(index=index_col, columns='code', values=field)
        else:
            return df.pivot(columns='code', values=field)

    if isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)) or (len(df) > 0 and isinstance(df.index[0], (datetime.date, datetime.datetime, pd.Timestamp))):
        if len(security_list) == 1:
            security_code = security_list[0]
            if field in df.columns:
                return df[[field]].rename(columns={field: security_code})

        return df[field] # Fallback

    if getattr(df, "ndim", 2) == 3:
        return df[field]

    if isinstance(df.index, pd.MultiIndex):
        unstacked = df.unstack(level=1)
        if field in unstacked.columns:
            return unstacked[field]
        if field in unstacked.columns.get_level_values(0):
            return unstacked.xs(field, axis=1, level=0, drop_level=True)
        return unstacked
    return df[field]
def _width(s: str) -> int:
    """返回字符串在终端里真实占的列数"""
    return builtins.sum(2 if unicodedata.east_asian_width(ch) in 'FW' else 1 for ch in s)

def pretty_print(df: pd.DataFrame, sep: str = '  ') -> None:
    """强制左对齐、屏幕宽度对齐、不换行打印 DataFrame"""
    str_df = df.astype(str)
    cols = list(df.columns)
    widths = {}
    for c in cols:
        w = max(_width(str(v)) for v in str_df[c])
        widths[c] = max(w, _width(c))

    header = sep.join(c.ljust(widths[c] - (_width(c) - len(c))) for c in cols)
    print(header)
    for _, row in str_df.iterrows():
        line = sep.join(
            row[c].ljust(widths[c] - (_width(row[c]) - len(row[c]))) for c in cols
        )
        print(line)
# -------------------- 1. 初始池筛选 --------------------------
def initial_etf_filter(target_date=None, verbose=True, config=CONFIG):
    """
    基础过滤：
    1. 获取所有ETF
    2. 剔除黑名单、非核心深市、特定前缀(债券/货币等)
    3. 剔除上市时间过短的
    """
    if verbose: print(f"Step 1: Initial Filtering (Date: {target_date})...")
    if target_date is None:
        target_date = datetime.date.today()

    df = get_all_securities(["etf"], date=target_date)

    # 白名单/黑名单
    blacklist = config['black_list']
    df = df[~df.index.isin(blacklist)]

    # 关键字剔除 (主要针对名字中包含特定标识的ETF)
    exclude_keywords = []
    if config["filter_gold"]:
        exclude_keywords.extend(["黄金", "上海金"])
    if config["filter_qdii"]:
        exclude_keywords.extend(["QDII", "标普", "纳指", "道琼斯", "恒生", "H股", "日经", "德国", "法国", "英国", "美国", "海外"])

    if exclude_keywords:
        if verbose: print(f"  -> Excluding keys: {exclude_keywords}")
        # using regex to filter
        pattern = "|".join(exclude_keywords)
        df = df[~df['display_name'].str.contains(pattern, regex=True)]

    # 上市时间检查
    # Use target_date instead of yesterday
    check_date = pd.Timestamp(target_date - datetime.timedelta(days=1))
    # check_date = target_date - datetime.timedelta(days=1)
    df = df[((check_date - df["start_date"]).dt.days >= config["min_listing_days"]) & (df["end_date"] > check_date)]

    # ----------------------------------------------------
    # 新增：价格过滤 (针对债券、货币)
    # 策略：如果最近1日均价 > 90，则认为是债券或货币基金
    # ----------------------------------------------------
    if config["filter_bond_money"]:
        if verbose: print("  -> Checking for high-priced ETFs (Bond/Money > 90)...")

        try:
            # avg_prices_df = history(1, "1d", "avg", df.index.tolist(), df=True)
            avg_prices_df = get_history_data(df.index.tolist(), target_date, 1, "avg")
            # Usually get_history_data returns index=Date, columns=Securities.

            if not avg_prices_df.empty:
                # avg_prices_df: rows=date (1 row), cols=securities
                last_prices = avg_prices_df.iloc[-1]

                # 找出价格 > 90 的代码
                high_price_etfs = last_prices[last_prices > 90].index.tolist()
                if high_price_etfs:
                    if verbose: print(f"  -> Filtering out {len(high_price_etfs)} ETFs with price > 90 (likely Bond/Money):")
                    # print(f"     Examples: {high_price_etfs[:5]}")
                    df = df[~df.index.isin(high_price_etfs)]
        except Exception as e:
            print(f"  [Warning] Failed to filter by price: {e}")

    initial_list = df.index.tolist()
    if verbose: print(f"  -> Found {len(initial_list)} candidate ETFs.")
    return initial_list
# -------------------- 2.1 AP 聚类筛选 (Affinity Propagation) --------------------------
def ap_clustering_filter(etf_list, config, target_date=None, verbose=True):
    """
    使用 Affinity Propagation (AP) 算法进行聚类筛选
    """
    if verbose: print(f"Step 2: Affinity Propagation Clustering (Damping={config['ap_damping']})...")

    # 1. 预先过滤流动性 (减少计算量)
    if target_date is None: target_date = datetime.date.today()

    # money_median_20d = history(20, "1d", "money", etf_list, df=True).median()
    money_df = get_history_data(etf_list, target_date, 20, "money")
    money_median_20d = money_df.median()
    valid_etfs = [etf for etf in etf_list if money_median_20d.get(etf, 0) > config["min_liquidity"]]

    if not valid_etfs:
        if verbose: print("  -> No ETFs passed liquidity filter.")
        return []

    # 2. 获取价格并计算收益率
    # price_data = history(config["correlation_window"], "1d", "close", valid_etfs, df=True)
    price_data = get_history_data(valid_etfs, target_date, config["correlation_window"], "close")
    price_data = price_data.fillna(method="ffill").dropna(axis=1, how="any")

    if price_data.empty:
        if verbose: print("  -> Price data empty.")
        return []

    returns = price_data.pct_change().dropna()
    final_etf_list = returns.columns.tolist()

    # 3. 计算相关性矩阵 (Spearman)
    if verbose: print(f"  -> Computing correlation matrix for {len(final_etf_list)} ETFs...")
    corr_matrix = returns.corr(method="spearman")

    # 4. 执行 AP 聚类
    # 注意：AP 接受相似度矩阵 (Similarity Matrix)，相关系数本身就是一种相似度 (-1 ~ 1)
    # affinity='precomputed' 表示我们直接传入矩阵
    ap = AffinityPropagation(
        damping=config["ap_damping"],
        preference=config["ap_preference"],
        affinity='precomputed',
        random_state=42
    )
    ap.fit(corr_matrix)

    labels = ap.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    if verbose: print(f"  -> AP converged into {n_clusters_} clusters.")

    # 5. 从每个簇中选择流动性最好的 ETF

    # 准备成交额数据
    # money_data = history(30, "1d", "money", final_etf_list, df=True).mean()
    money_data = get_history_data(final_etf_list, target_date, 30, "money").mean()

    cluster_dict = defaultdict(list)
    for etf, label in zip(final_etf_list, labels):
        cluster_dict[label].append(etf)

    selected_etfs = []

    if verbose: print("  -> Selecting best liquidity ETF from each cluster:")
    for label, etfs in cluster_dict.items():
        # 选出该组中 money_data 最大的
        best_etf = max(etfs, key=lambda x: money_data.get(x, 0))
        selected_etfs.append(best_etf)

        # 可选：打印每组详情
        display_name = get_security_info(best_etf).display_name

        if verbose: print(f"    Cluster {label}: Selected {display_name} ({best_etf}) from {len(etfs)} candidates.")

    return selected_etfs

# -------------------- 2.2 层次聚类筛选 (Hierarchical Clustering) --------------------------
def hierarchical_clustering_filter(etf_list, config, target_date=None, verbose=True):
    """
    使用层次聚类筛选 (移植自 oix_pools.py slow_track_filter)
    """
    if verbose: print(f"Step 2: Hierarchical Clustering (Corr Threshold={config['cluster_corr_threshold']})...")

    # 1. 预先过滤流动性
    if target_date is None: target_date = datetime.date.today()
    # money_median_20d = history(20, "1d", "money", etf_list, df=True).median()
    money_median_20d = get_history_data(etf_list, target_date, 20, "money").median()
    valid_etfs = [etf for etf in etf_list if money_median_20d.get(etf, 0) > config["min_liquidity"]]

    if not valid_etfs:
        if verbose: print("  -> No ETFs passed liquidity filter.")
        return []

    # 2. 获取价格并计算收益率
    # 注意: oix 使用 120 天，这里 config["correlation_window"] 也是 120
    # price_data = history(config["correlation_window"], "1d", "close", valid_etfs, df=True)
    price_data = get_history_data(valid_etfs, target_date, config["correlation_window"], "close")
    price_data = price_data.fillna(method="ffill").dropna(axis=1, how="any")

    if price_data.empty:
        if verbose: print("  -> Price data empty.")
        return []

    returns = price_data.pct_change().dropna()
    final_etf_list = returns.columns.tolist()

    # 3. 计算相关性矩阵 (Spearman)
    if verbose: print(f"  -> Computing correlation matrix for {len(final_etf_list)} ETFs...")
    corr_matrix = returns.corr(method="spearman")

    # 4. 转换为距离矩阵 & 聚类
    # distance = sqrt(2 * (1 - correlation))
    distance_matrix = np.sqrt(2 * (1 - corr_matrix))
    # distance_matrix 对角线设为0 (虽然公式结果也是0，确保数值稳定)
    np.fill_diagonal(distance_matrix.values, 0)

    condensed_dist = squareform(distance_matrix, checks=False)
    Z = linkage(condensed_dist, method="average")

    # 5. 聚类划分
    # threshold = sqrt(2 * (1 - corr_threshold))
    dist_threshold = np.sqrt(2 * (1 - config["cluster_corr_threshold"]))
    clusters = fcluster(Z, dist_threshold, criterion="distance")

    n_clusters_ = len(set(clusters))
    if verbose: print(f"  -> Converged into {n_clusters_} clusters.")

    # 6. 从每个簇中选择流动性最好的 ETF
    money_data = history(30, "1d", "money", final_etf_list, df=True).median() # or mean, oix uses mean in fast/slow logic varies slightly, oix slow uses median for filter but mean for selection? oix line 88 uses fillna(0) then selection uses mean.
    # oix line 88: money_data = history(30, "1d", "money", filtered_etfs, df=True).fillna(0)
    # oix selection: max(etfs, key=lambda etf: money_data[etf].mean())
    # Wait, money_data[etf] is a series. .mean() is scalar. Correct.
    # I will stick to simple scalar mean for sorting.
    money_means = money_data.mean() if isinstance(money_data, pd.DataFrame) else money_data # JQ returns DF usually.
    # Actually, let's just get the mean directly
    # money_means = history(30, "1d", "money", final_etf_list, df=True).mean()
    money_means = get_history_data(final_etf_list, target_date, 30, "money").mean()

    cluster_dict = defaultdict(list)
    for etf, label in zip(final_etf_list, clusters):
        cluster_dict[label].append(etf)

    selected_etfs = []

    if verbose: print("  -> Selecting best liquidity ETF from each cluster:")
    for label, etfs in cluster_dict.items():
        # 选出该组中 money_means 最大的
        best_etf = max(etfs, key=lambda x: money_means.get(x, 0))
        selected_etfs.append(best_etf)

        display_name = get_security_info(best_etf).display_name
        if verbose: print(f"    Cluster {label}: Selected {display_name} ({best_etf}) from {len(etfs)} candidates.")

    return selected_etfs

# -------------------- 2.3 MST 聚类筛选 (Minimum Spanning Tree) --------------------------
def mst_clustering_filter(etf_list, config, target_date=None, verbose=True):
    """
    使用最小生成树 (MST) 算法进行聚类筛选
    原理: 构建完全图 (权重=距离)，生成MST，切断最长的K-1条边，形成K个连通分量
    """
    if verbose: print(f"Step 2: MST Clustering (Target Clusters={config['mst_n_clusters']})...")

    # 1. 预先过滤流动性
    if target_date is None: target_date = datetime.date.today()
    # money_median_20d = history(20, "1d", "money", etf_list, df=True).median()
    money_median_20d = get_history_data(etf_list, target_date, 20, "money").median()
    valid_etfs = [etf for etf in etf_list if money_median_20d.get(etf, 0) > config["min_liquidity"]]

    if not valid_etfs:
        if verbose: print("  -> No ETFs passed liquidity filter.")
        return []

    # 2. 获取价格并计算收益率
    # price_data = history(config["correlation_window"], "1d", "close", valid_etfs, df=True)
    price_data = get_history_data(valid_etfs, target_date, config["correlation_window"], "close")
    price_data = price_data.fillna(method="ffill").dropna(axis=1, how="any")

    if price_data.empty:
        if verbose: print("  -> Price data empty.")
        return []

    returns = price_data.pct_change().dropna()
    final_etf_list = returns.columns.tolist() # 节点列表 of size N

    # 3. 构建距离矩阵 (Distance Matrix)
    # distance = sqrt(2 * (1 - correlation))
    if verbose: print(f"  -> Computing correlation & distance matrix for {len(final_etf_list)} ETFs...")
    corr_matrix = returns.corr(method="spearman")
    dist_matrix = np.sqrt(2 * (1 - corr_matrix))
    np.fill_diagonal(dist_matrix.values, 0) # 自身距离为0

    # 4. 构建完全图并生成 MST
    G = nx.Graph()
    # 添加节点
    G.add_nodes_from(final_etf_list)

    # 添加边 (完全图)
    # 优化: networkx 处理完全图比较慢，我们可以直接通过距离矩阵构建
    # 这里直接遍历矩阵的上三角部分添加边
    edges = []
    n = len(final_etf_list)
    for i in range(n):
        for j in range(i + 1, n):
            u = final_etf_list[i]
            v = final_etf_list[j]
            weight = dist_matrix.iloc[i, j]
            edges.append((u, v, weight))

    G.add_weighted_edges_from(edges)

    if verbose: print("  -> Building Minimum Spanning Tree...")
    mst = nx.minimum_spanning_tree(G)

    # 1. 计算标准化树长 (NTL)
    total_weight = sum(d['weight'] for u, v, d in mst.edges(data=True))
    ntl = total_weight / (len(mst.nodes()) - 1)
    # 2. 计算平均路径长度 (需要算拓扑距离，或是加权距离)
    # 这里通常算的是跳数(hops)，或者加权距离
    avg_path_len = nx.average_shortest_path_length(mst, weight='weight')
    # 3. 打印观测
    if verbose:
        print(f"  [Market Structure] NTL: {ntl:.4f} (Lower = Higher Risk/Systemic)")
        print(f"  [Market Structure] Avg Path: {avg_path_len:.4f}")

    # 5. 切断最长的 K-1 条边
    # 获取MST中所有边及其权重
    mst_edges = sorted(mst.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)

    # 需要移除的边数 = 目标簇数 - 1
    # 如果当前连通分量已经是1 (MST本身)，移除1条边变2个分量...
    # 注意: MST是连通的，初始分量为1
    num_to_remove = config['mst_n_clusters'] - 1

    if num_to_remove > 0 and len(mst_edges) >= num_to_remove:
        # 移除权重最大的前 num_to_remove 条边 (即距离最远/相关性最低的连接)
        # 这样会断开大的群组
        edges_to_remove = mst_edges[:num_to_remove]
        mst.remove_edges_from([(u, v) for u, v, d in edges_to_remove])
        if verbose: print(f"  -> Removed {num_to_remove} longest edges to form clusters.")

    # 6. 获取连通分量 (即聚类结果)
    components = list(nx.connected_components(mst))
    if verbose: print(f"  -> Generated {len(components)} clusters.")

    # 7. 从每个簇中选择流动性最好的 ETF
    # money_means = history(30, "1d", "money", final_etf_list, df=True).mean()
    money_means = get_history_data(final_etf_list, target_date, 30, "money").mean()

    selected_etfs = []
    if verbose: print("  -> Selecting best liquidity ETF from each cluster:")

    for i, comp in enumerate(components):
        etfs_in_cluster = list(comp)
        # 选出该组中 money_means 最大的
        best_etf = max(etfs_in_cluster, key=lambda x: money_means.get(x, 0))
        selected_etfs.append(best_etf)

        display_name = get_security_info(best_etf).display_name
        if verbose: print(f"    Cluster {i+1}: Selected {display_name} ({best_etf}) from {len(etfs_in_cluster)} candidates.")

    return selected_etfs

# -------------------- 2.4 DBSCAN 聚类筛选 (Density-Based) --------------------------
def dbscan_clustering_filter(etf_list, config, target_date=None, verbose=True):
    """
    使用 DBSCAN 进行基于密度的聚类
    原理: 将高密度区域划分为簇，稀疏区域标记为噪声 (-1)
    """
    eps = config.get('dbscan_eps', 0.5) # 默认距离阈值
    min_samples = config.get('dbscan_min_samples', 2) # 最小样本数

    if verbose: print(f"Step 2: DBSCAN Clustering (eps={eps}, min_samples={min_samples})...")

    # 1. 预先过滤流动性
    if target_date is None: target_date = datetime.date.today()
    money_median_20d = get_history_data(etf_list, target_date, 20, "money").median()
    valid_etfs = [etf for etf in etf_list if money_median_20d.get(etf, 0) > config["min_liquidity"]]

    if not valid_etfs:
        if verbose: print("  -> No ETFs passed liquidity filter.")
        return []

    # 2. 获取价格并计算收益率
    price_data = get_history_data(valid_etfs, target_date, config["correlation_window"], "close")
    price_data = price_data.fillna(method="ffill").dropna(axis=1, how="any")

    if price_data.empty:
        if verbose: print("  -> Price data empty.")
        return []

    returns = price_data.pct_change().dropna()
    final_etf_list = returns.columns.tolist()

    # 3. 计算相关性矩阵 & 距离矩阵
    if verbose: print(f"  -> Computing correlation & distance matrix for {len(final_etf_list)} ETFs...")
    corr_matrix = returns.corr(method="spearman")
    # distance = sqrt(2 * (1 - correlation))
    # DBSCAN metric='precomputed' 需要距离矩阵
    dist_matrix = np.sqrt(2 * (1 - corr_matrix))
    np.fill_diagonal(dist_matrix.values, 0)

    # 4. 执行 DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    db.fit(dist_matrix)

    labels = db.labels_

    # 统计簇
    unique_labels = set(labels)
    n_clusters_ = len(unique_labels) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    if verbose: print(f"  -> DBSCAN found {n_clusters_} clusters and {n_noise_} noise points.")

    # 5. 选择逻辑
    # 对于每个簇 (Label >= 0)，选流动性最好的
    # 对于噪声 (-1)，通常意味着它们独特，全部保留 (或视为单独的簇)

    money_means = get_history_data(final_etf_list, target_date, 30, "money").mean()
    cluster_dict = defaultdict(list)

    for etf, label in zip(final_etf_list, labels):
        cluster_dict[label].append(etf)

    selected_etfs = []

    if verbose: print("  -> Selecting best liquidity ETF from each cluster:")

    # 处理普通簇
    for label in unique_labels:
        if label == -1:
            continue # 处理噪声单独逻辑

        etfs = cluster_dict[label]
        best_etf = max(etfs, key=lambda x: money_means.get(x, 0))
        selected_etfs.append(best_etf)

        display_name = get_security_info(best_etf).display_name
        if verbose: print(f"    Cluster {label}: Selected {display_name} ({best_etf}) from {len(etfs)} candidates.")

    # 处理噪声 (Outliers)
    # 策略: 噪声代表独特资产，全部保留
    if -1 in cluster_dict:
        noise_etfs = cluster_dict[-1]
        if verbose: print(f"  -> Retaining {len(noise_etfs)} unique outliers (Noise):")
        for etf in noise_etfs:
            selected_etfs.append(etf)
            # if verbose: print(f"     Outlier: {get_security_info(etf).display_name} ({etf})")

    return selected_etfs
# -------------------- 4. 多轮聚合 (Multi-Round Aggregation) --------------------------
def multi_round_aggregation(freq="M", periods=6, end_date=None, verbose=False, **conf):
    """
    Perform aggregation for multiple dates looking back from a time series.
    freq: 'M' (Month End), 'W' (Weekly), etc.
    periods: Number of periods to look back.
    end_date: Setup end date.

    Output: Table of selected ETFs for each date.
    """
    if end_date is None:
        end_date = datetime.date.today()

    config = CONFIG.copy()
    config.update(**conf)
    dates = pd.date_range(end=end_date, periods=periods, freq=freq).tolist()
    # Convert to date objects
    dates = [d.date() for d in dates]

    print("=" * 60)
    print(f"Multi-Round Aggregation ({len(dates)} rounds)")
    print(f"Dates: {dates}")
    print(f"Method: {config['clustering_method']}")
    print("=" * 60)

    results = []

    for d in dates:
        print(f"\rProcessing Date: {d}")
        try:
            # 1. Initial Filter
            candidates = initial_etf_filter(target_date=d, verbose=verbose)
            if not candidates:
                print("  -> No candidates found.")
                results.append({"date": d, "count": 0, "etfs": ""})
                continue

            # 2. Clustering
            method = config["clustering_method"]
            if method == "ap":
                final_pool = ap_clustering_filter(candidates, config, target_date=d, verbose=verbose)
            elif method == "mst":
                final_pool = mst_clustering_filter(candidates, config, target_date=d, verbose=verbose)
            elif method == "dbscan":
                final_pool = dbscan_clustering_filter(candidates, config, target_date=d, verbose=verbose)
            else:
                final_pool = hierarchical_clustering_filter(candidates, config, target_date=d, verbose=verbose)

            # Record result
            # Convert list to string
            etf_str = ",".join([get_security_info(c).display_name for c in final_pool])
            results.append({"date": d, "count": len(final_pool), "etfs": etf_str, "codes": final_pool})

        except Exception as e:
            print(f"  [Error] Processing {d} failed: {e}")
            results.append({"date": d, "count": 0, "etfs": "ERROR"})

    # Output Table
    print("=" * 60)
    print("Multi-Round Aggregation Results")
    print("=" * 60)

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        pretty_print(res_df[["date", "count", "etfs"]])

    return res_df

def init(context):
    config = CONFIG.copy()
    today = datetime.date.today()
    target_date = today - datetime.timedelta(days=30)
    # config.update(**conf)
    print('-'*80)
    print(f"Starting {config['clustering_method'].upper()}-Based ETF Pool Generation")
    print("Configuration:", config)
    print('-'*80)

    candidates = initial_etf_filter(target_date=target_date)

    # 2. 聚类选择
    if config["clustering_method"] == "ap":
        final_pool = ap_clustering_filter(candidates, config, target_date=date)
        method_name = "Affinity Propagation"
    elif config["clustering_method"] == "mst":
        final_pool = mst_clustering_filter(candidates, config, target_date=date)
        method_name = "Minimum Spanning Tree (MST)"
    elif config["clustering_method"] == "dbscan":
        final_pool = dbscan_clustering_filter(candidates, config, target_date=date)
        method_name = "DBSCAN Clustering"
    else:
        final_pool = hierarchical_clustering_filter(candidates, config, target_date=date)
        method_name = "Hierarchical Clustering"

    print(f"{method_name} Complete. Pool Size: {len(final_pool)}")


