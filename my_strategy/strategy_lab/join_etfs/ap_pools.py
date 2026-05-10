from jqdata import *
import logging
from sklearn.metrics import adjusted_rand_score
from collections import defaultdict
import numpy as np
import pandas as pd
import datetime
import unicodedata
import builtins
import math
from pca_analysis import analyze_pool_pca
from join_method import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------- 配置区域 (Initial Configuration) --------------------
CONFIG = {
    # 基础过滤参数
    "min_liquidity": 50e6,       # 最小成交额 (5000万)
    "min_listing_days": 60,      # 最小上市天数
    "correlation_window": 120,   # 计算相关性的回看窗口 (天)
    "smoothing_steps": 1,        # 恢复原状：1表示不平滑，只计算当前窗口
    "smoothing_lag": 20,         # (此参数在steps=1时无效)
    
    # 类型过滤配置 (True=剔除, False=保留)
    "filter_bond_money": True,    # 剔除债券、货币
    "filter_qdii": False,         # 剔除跨境/QDII
    "filter_gold": False,         # 剔除黄金
    "filter_to_index": False,     # 剔除每个指数多余的ETF
    # 黑名单
    "black_list": ["510900.XSHG"],

    # 选择代表的方法 'money'-最大成交额，'market_cap'-最大市值
    "represent_method": "money",
    
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
    logger.info(header)
    for _, row in str_df.iterrows():
        line = sep.join(
            row[c].ljust(widths[c] - (_width(row[c]) - len(row[c]))) for c in cols
        )
        logger.info(line)

# -------------------- 1.0 去除指数对应的多余ETF ---------------

def filter_unique_etf_per_index(etf_list, target_date=None):
    """
    过滤 ETF，使得每个指数只保留一只代表 ETF。
    默认规则：
    1. 找到每只 ETF 在 target_date 时对应的指数。
    2. 对同一指数的 ETF，保留最早发布、code最小的那个
    """
    if not etf_list:
        return []

    # 查询指数关联信息
    try:
        q = query(finance.FUND_INVEST_TARGET).filter(
            finance.FUND_INVEST_TARGET.code.in_(etf_list)
        )
        df_all = finance.run_query(q)
    except Exception as e:
        logger.warning(f"无法查询指数信息: {e}")
        return etf_list

    if df_all.empty:
        pass

    # 处理时间过滤 (避免未来函数)
    if target_date is not None and not df_all.empty:
        if not isinstance(target_date, (datetime.date, datetime.datetime)):
             try:
                 target_date = pd.to_datetime(target_date).date()
             except:
                 pass
        df_all['start_date'] = pd.to_datetime(df_all['start_date'])
        df_all = df_all[df_all['start_date'].dt.date <= target_date]

    if df_all.empty:
        selected_codes = set()
    else:
        current_index_map = df_all.sort_values('start_date', ascending=False).drop_duplicates('code', keep='first')
        current_index_map['group_key'] = current_index_map['traced_index_code'].fillna(current_index_map['traced_index_name'])
        grouped = current_index_map[current_index_map['group_key'].notna()].copy()
        best_etfs = grouped.sort_values(['start_date', 'code'], ascending=[True, False]).drop_duplicates('group_key', keep='first')
        selected_codes = set(best_etfs['code'].tolist())
    
    whitelist = ['518880.XSHG'] 
    for wl in whitelist:
        if wl in etf_list:
            selected_codes.add(wl)

    final_list = [code for code in etf_list if code in selected_codes]
    dropped_count = len(etf_list) - len(final_list)
    if dropped_count > 0:
        logger.debug(f"  -> 指数去重过滤掉 {dropped_count} 只 ETF")
        
    return final_list

# -------------------- 1. 初始池筛选 --------------------------
def initial_etf_filter(target_date=None, config=CONFIG):
    """
    基础过滤：
    1. 获取所有ETF
    2. 剔除黑名单、特定类型ETF(黄金/债券/货币/QDII等)
    3. 剔除上市时间过短的
    4. 流动性过滤
    """
    logger.debug(f"第一步：初始过滤 (日期: {target_date})...")
    if target_date is None:
        target_date = datetime.date.today()
        
    df = get_all_securities(["etf"], date=target_date)
    
    # 白名单/黑名单
    blacklist = config['black_list']
    df = df[~df.index.isin(blacklist)]
    
    # 关键字剔除
    exclude_keywords = []
    # if config["filter_bond_money"]: 
    #     exclude_keywords.extend(["债", "货币", "理财"])
    if config["filter_gold"]:       
        exclude_keywords.extend(["黄金", "上海金"])
    if config["filter_qdii"]:       
        exclude_keywords.extend(["QDII", "标普", "纳指", "道琼斯", "恒生", "H股", "日经", "德国", "法国", "英国", "美国", "海外"])
    
    if exclude_keywords:
        logger.debug(f"  -> 剔除关键字: {exclude_keywords}")
        pattern = "|".join(exclude_keywords)
        df = df[~df['display_name'].str.contains(pattern, regex=True)]
    
    check_date = target_date - datetime.timedelta(days=1)
    df = df[((check_date - df["start_date"]).dt.days >= config["min_listing_days"]) & (df["end_date"] > check_date)]

    # 价格过滤 (针对债券、货币)
    if config["filter_bond_money"]:
        logger.debug("  -> 检查高价 ETF (债券/货币 > 90)...")
        
        try:
            avg_prices_df = get_history_data(df.index.tolist(), target_date, 1, "avg")
            
            if not avg_prices_df.empty:
                last_prices = avg_prices_df.iloc[-1]          
                high_price_etfs = last_prices[last_prices > 90].index.tolist()
                if high_price_etfs:
                    logger.debug(f"  -> 剔除 {len(high_price_etfs)} 个价格 > 90 的 ETF (可能是债券/货币):")
                    # logger.debug(f"     Examples: {high_price_etfs[:5]}")
                    df = df[~df.index.isin(high_price_etfs)]
        except Exception as e:
            logger.warning(f"  [警告] 价格过滤失败: {e}")

    initial_list = df.index.tolist()
    
    # 流动性过滤
    logger.debug("  -> 执行流动性过滤...")
    try:
        money_median_20d = get_history_data(initial_list, target_date, 20, "money").median()
        valid_etfs = [etf for etf in initial_list if money_median_20d.get(etf, 0) > config["min_liquidity"]]
        if not valid_etfs:
            logger.debug("  -> 无 ETF 通过流动性过滤。")
            initial_list = []
        else:
            initial_list = valid_etfs
    except Exception as e:
        logger.warning(f"  [警告] 流动性过滤失败: {e}")

    # 指数去重 (每指数保留一只)
    if config.get('filter_to_index', False):
        initial_list = filter_unique_etf_per_index(initial_list, target_date=None)
    logger.debug(f"  -> 找到 {len(initial_list)} 个候选 ETF。")
    return initial_list

# -------------------- 3. 动量评分 --------------------------
def get_best_etf(etf_list, config=CONFIG):
    """
    计算动量得分并排序
    """
    logger.info("第三步：动量评分与排名...")
    data = pd.DataFrame(index=etf_list, columns=["code", "name", "annualized_returns", "r2", "score"])
    
    for etf in etf_list:
        df = get_price(etf, end_date=today, frequency="daily", fields=["close"], count=config["score_window"], panel=False)
        if df.empty:
            continue
            
        prices = df["close"].values
        y = np.log(prices)
        x = np.arange(len(y))
        weights = np.linspace(1, 2, len(y))
        slope, intercept = np.polyfit(x, y, 1, w=weights)
        ann_ret = round(math.exp(slope * 250) - 1, 2)
        y_pred = slope * x + intercept
        ss_res = np.sum(weights * (y - y_pred) ** 2)
        ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
        r2 = round(1 - ss_res / ss_tot if ss_tot else 0, 2)
        score = round(ann_ret * r2, 2)
        data.loc[etf, "code"] = etf
        try:
            data.loc[etf, "name"] = get_security_info(etf).display_name
        except:
            data.loc[etf, "name"] = "Unknown"
        data.loc[etf, "annualized_returns"] = ann_ret
        data.loc[etf, "r2"] = r2
        data.loc[etf, "score"] = score
        if len(prices) >= 4:
             if min(prices[-1]/prices[-2], prices[-2]/prices[-3], prices[-3]/prices[-4]) < 0.95:
                 data.loc[etf, "score"] = 0

    filtered = data[
        (data["score"] > 0) & 
        (data["annualized_returns"] < config["max_annualized_returns"]) & 
        (data["r2"] > config["min_r2"])
    ].sort_values(by="score", ascending=False)
    
    return filtered

# -------------------- 4.1 单轮聚合 (Single-Round Aggregation) --------------------------
def test_aggregation(date = None, return_details = False, config = CONFIG):
    if date is None:
        date = datetime.date.today()
    elif isinstance(date, str):
        date = pd.to_datetime(date).date()
    logger.info("-" * 50)
    logger.info(f"开始基于 {config['clustering_method'].upper()} 的 ETF 池生成")
    logger.info(f"配置: {config}")
    logger.info("-" * 50)
    
    # 初始筛选
    candidates = initial_etf_filter(target_date=date)
    
    # 聚类选择
    if config["clustering_method"] == "ap":
        final_pool = ap_clustering_filter(candidates, config, target_date=date, return_details=return_details)
        method_name = "Affinity Propagation"
    elif config["clustering_method"] == "mst":
        final_pool = mst_clustering_filter(candidates, config, target_date=date, return_details=return_details)
        method_name = "Minimum Spanning Tree (MST)"
    elif config["clustering_method"] == "dbscan":
        final_pool = dbscan_clustering_filter(candidates, config, target_date=date, return_details=return_details)
        method_name = "DBSCAN Clustering"
    else:
        final_pool = hierarchical_clustering_filter(candidates, config, target_date=date, return_details=return_details)
        method_name = "Hierarchical Clustering"
        
    logger.info(f"{method_name} 完成。池大小: {len(final_pool)}")
    
    # PCA 分析
    # analyze_pool_pca(final_pool, config, overlay_code=config.get("pca_overlay_etf"))
    
    # 动量评分展示
    result_df = get_best_etf(final_pool)
    
    logger.info("\n最终选择的 ETF 池 (动量排名前列):")
    pretty_print(result_df)

# -------------------- 4.2 多轮聚合 (Multi-Round Aggregation) --------------------------
def multi_round_aggregation(freq="M", periods=6, end_date=None, config=CONFIG):
    """
    多轮聚合：回溯多个时间点进行聚类筛选。
    freq: 'M' (月末), 'W' (周末) 等。
    periods: 回溯的周期数。
    end_date: 结束日期。
    
    输出: 每个日期的入选ETF列表。
    """
    if end_date is None:
        end_date = datetime.date.today()
    
    dates = pd.date_range(end=end_date, periods=periods, freq=freq).tolist()
    dates = [d.date() for d in dates]
    
    logger.info("=" * 60)
    logger.info(f"多轮聚合 ({len(dates)} 轮)")
    logger.info(f"日期: {dates}")
    logger.info(f"方法: {config['clustering_method']}")
    logger.info("=" * 60)
    
    results = []
    
    for d in dates:
        logger.info(f"\n处理日期: {d}")
        try:
            candidates = initial_etf_filter(target_date=d)
            if not candidates:
                logger.info("  -> 未找到候选标的。")
                results.append({"date": d, "count": 0, "etfs": ""})
                continue
                
            method = config["clustering_method"]
            if method == "ap":
                final_pool = ap_clustering_filter(candidates, config, target_date=d)
            elif method == "mst":
                final_pool = mst_clustering_filter(candidates, config, target_date=d)
            elif method == "dbscan":
                final_pool = dbscan_clustering_filter(candidates, config, target_date=d)
            else:
                final_pool = hierarchical_clustering_filter(candidates, config, target_date=d)
            
            etf_str = ",".join([get_security_info(c).display_name for c in final_pool])
            results.append({"date": d, "count": len(final_pool), "etfs": etf_str, "codes": final_pool})
            
        except Exception as e:
            logger.error(f"  [错误] 处理日期 {d} 失败: {e}")
            results.append({"date": d, "count": 0, "etfs": "ERROR"})

    logger.info("\n" + "=" * 60)
    logger.info("多轮聚合结果")
    logger.info("=" * 60)
    
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        pretty_print(res_df[["date", "count", "etfs"]])
        
    return res_df

# -------------------- 5. 稳定性分析工具 --------------------------
def analyze_cluster_stability(base_date=None, lag_days=1, config=CONFIG):
    """
    研究两个相似时间点（base_date vs base_date - lag_days）的聚类结果偏差。
    用于量化观察聚类算法的时序不稳定性。
    """
    if base_date is None:
        base_date = datetime.date.today()
    elif isinstance(base_date, str):
        base_date = pd.to_datetime(base_date).date()
        
    prev_date = base_date - datetime.timedelta(days=lag_days)
    
    logger.info("=" * 60)
    logger.info(f"稳定性分析: {base_date} vs {prev_date} (滞后: {lag_days} 天)")
    logger.info(f"算法: {config['clustering_method']}")
    logger.info("=" * 60)
    
    # 第1轮: 基准日期
    logger.info(f"\n[第1轮] 处理 {base_date}...")
    pool1 = []
    details1 = {}
    try:
        c1 = initial_etf_filter(target_date=base_date)
        if config["clustering_method"] == "ap":
            pool1, details1 = ap_clustering_filter(c1, config, target_date=base_date, return_details=True)
        elif config["clustering_method"] == "mst":
            pool1, details1 = mst_clustering_filter(c1, config, target_date=base_date, return_details=True)
        elif config["clustering_method"] == "dbscan":
            pool1, details1 = dbscan_clustering_filter(c1, config, target_date=base_date, return_details=True)
        else:
            pool1, details1 = hierarchical_clustering_filter(c1, config, target_date=base_date, return_details=True)
    except Exception as e:
        logger.error(f"  第1轮出错: {e}")

    # 第2轮: 对比日期
    logger.info(f"[第2轮] 处理 {prev_date}...")
    pool2 = []
    details2 = {}
    try:
        c2 = initial_etf_filter(target_date=prev_date)
        if config["clustering_method"] == "ap":
            pool2, details2 = ap_clustering_filter(c2, config, target_date=prev_date, return_details=True)
        elif config["clustering_method"] == "mst":
            pool2, details2 = mst_clustering_filter(c2, config, target_date=prev_date, return_details=True)
        elif config["clustering_method"] == "dbscan":
            pool2, details2 = dbscan_clustering_filter(c2, config, target_date=prev_date, return_details=True)
        else:
            pool2, details2 = hierarchical_clustering_filter(c2, config, target_date=prev_date, return_details=True)
    except Exception as e:
        logger.error(f"  第2轮出错: {e}")
        
    set1 = set(pool1)
    set2 = set(pool2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    # 选股稳定性 (Jaccard系数)
    stability_score = len(intersection) / len(union) if union else 0
    
    # 结构稳定性 (ARI指数)
    common_candidates = set(details1.keys()) & set(details2.keys())
    ari_score = 0.0
    if common_candidates:
        labels_true = [details1[etf] for etf in common_candidates]
        labels_pred = [details2[etf] for etf in common_candidates]
        ari_score = adjusted_rand_score(labels_true, labels_pred)
    
    logger.info("\n" + "-"*30 + " 结果 " + "-"*30)
    logger.info(f"池1 选中 ({base_date}): {len(pool1)}")
    logger.info(f"池2 选中 ({prev_date}): {len(pool2)}")
    logger.info(f"选择稳定性 (Jaccard): {stability_score:.4f} (1.0 = 完全相同)")
    
    logger.info(f"\n公共候选分析 (结构):")
    logger.info(f"公共候选 ETF 数量: {len(common_candidates)}")
    logger.info(f"结构稳定性 (ARI):    {ari_score:.4f} (1.0 = 完全相同分组)")
    logger.info(f"  (ARI 衡量同一组资产聚类划分的相似度)")
    
    newly_added = set1 - set2
    dropped = set2 - set1
    logger.info("\n" + "-"*30 + " 结果 " + "-"*30)
    logger.info(f"池1 大小 ({base_date}): {len(pool1)}")
    logger.info(f"池2 大小 ({prev_date}): {len(pool2)}")
    logger.info(f"交集: {len(intersection)}")
    logger.info(f"稳定性得分 (Jaccard): {stability_score:.4f} (1.0 = 完全相同)")
    logger.info(f"\n变动: {len(newly_added) + len(dropped)}")
    if newly_added:
        names = [f"{get_security_info(c).display_name}" for c in newly_added]
        logger.info(f"  + 新增 ({len(newly_added)}): {', '.join(names)}")
    if dropped:
        names = [f"{get_security_info(c).display_name}" for c in dropped]
        logger.info(f"  - 移除 ({len(dropped)}): {', '.join(names)}")
        
    return stability_score

# -------------------- 主程序 --------------------------
if __name__ == "__main__":
    
    # 模式选择: "SINGLE" (单次), "MULTI" (多轮), "STABILITY" (稳定性)
    MODE = "STABILITY" 
    
    if MODE == "STABILITY":
        analyze_cluster_stability(lag_days=1) # Comparative Analysis
        exit()
    
    elif MODE == "SINGLE":
        test_aggregation(date = None, config = CONFIG)
        exit()
    
    elif MODE == "MULTI":
        # 示例: 最近6个月末
        multi_round_aggregation(freq="ME", periods=6)
        logger.info("\n" + "-"*30 + " 单轮 (今日) " + "-"*30)


