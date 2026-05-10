import pandas as pd
import numpy as np
import datetime
from collections import defaultdict
import logging
import os
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from jqdata import *
    from ap_pools import initial_etf_filter, ap_clustering_filter, mst_clustering_filter, dbscan_clustering_filter, hierarchical_clustering_filter, CONFIG
except ModuleNotFoundError:
    logger.warning("未检测到 jqdata，将无法使用。")

try:
    from pyecharts import options as opts
    from pyecharts.options import EmphasisOpts
    from pyecharts.charts import Sankey
    from pyecharts.commons.utils import JsCode
    import textwrap
    HAS_PYECHARTS = True
except ImportError:
    logger.warning("未检测到 pyecharts，方案二 (桑基图) 将不可用。")
    HAS_PYECHARTS = False

# -------------------- 核心工具：获取多期聚类状态 --------------------------
def get_cluster_series(end_date=None, periods=6, freq="M", config=None, cache_file=None, force_refresh=False):
    """
    获取一系列日期的聚类快照。
    cache_file: 如果提供路径，将尝试加载/保存结果到该文件 (pickle格式)
    force_refresh: 强制重新计算并覆盖缓存
    
    返回: list of dict -> [
        {
            'date': date_obj,
            'details': {etf_code: label},
            'representatives': {label: rep_etf_code},
            'clusters': {label: [members]},
            'label_map': {label: 'Label_Name'} (可选)
        }, ...
    ]
    """
    if cache_file and not force_refresh and os.path.exists(cache_file):
        logger.info(f"正在从缓存加载聚类数据: {cache_file} ...")
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                logger.info("  -> 加载成功。")
                return data
        except Exception as e:
            logger.warning(f"  -> 加载缓存失败: {e}，将重新计算。")

    if end_date is None:
        end_date = datetime.date.today()
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date).date()
        
    dates = pd.date_range(end=end_date, periods=periods, freq=freq).tolist()
    dates = [d.date() for d in dates]
    
    series_data = []
    
    logger.info(f"正在生成 {len(dates)} 期聚类数据...")
    
    for d in dates:
        logger.info(f"  -> 处理 {d}...")
        try:
            # 初始筛选
            candidates = initial_etf_filter(target_date=d, config=config)
            if not candidates:
                continue
                
            # 执行聚类 (强制 return_details=True)
            method = config["clustering_method"]
            if method == "ap":
                func = ap_clustering_filter
            elif method == "mst":
                func = mst_clustering_filter
            elif method == "dbscan":
                func = dbscan_clustering_filter
            else:
                func = hierarchical_clustering_filter
                
            selected_etfs, details = func(candidates, config, target_date=d, return_details=True)
            
            # 整理结构
            clusters = defaultdict(list)
            label_map = {}
            for etf, label in details.items():
                clusters[label].append(etf)
                try:
                    name = get_security_info(etf).display_name
                except:
                    name = str(rep)
                label_map[etf] = name
            representatives = {}
            for rep in selected_etfs:
                if rep in details:
                    lbl = details[rep]
                    representatives[lbl] = rep
            
            series_data.append({
                'date': d,
                'details': details,
                'representatives': representatives,
                'clusters': clusters,
                'label_map': label_map
            })
            
        except Exception as e:
            logger.error(f"  [ERROR] 处理 {d} 失败: {e}")
            
    # 保存缓存
    if cache_file and series_data:
        try:
            logger.info(f"正在保存缓存到: {cache_file} ...")
            if os.path.exists(cache_file): os.remove(cache_file)
            with open(cache_file, 'wb') as f:
                pickle.dump(series_data, f)
        except Exception as e:
            logger.error(f"  -> 保存缓存失败: {e}")
            
    return series_data

# -------------------- 方案一：族谱追踪 (Text/DataFrame) --------------------------

def track_cluster_genealogy(series_data, threshold=0.3):
    """
    追踪聚类家族的演变。
    逻辑：
    - 给第一期的每个 Cluster 分配唯一 Family_ID。
    - 对于下一期，计算每个新 Cluster 与上一期所有 Cluster 的 Jaccard 相似度。
    - 如果 Max(Jaccard) > threshold，则继承 Family_ID。
    - 否则，分配新的 Family_ID。
    
    返回: DataFrame (Index=Family_ID, Columns=Date, Value=Representative_Name (Size))
    """
    if not series_data:
        logger.warning("无数据可追踪。")
        return pd.DataFrame()
        
    family_history = defaultdict(dict)
    prev_map = {} 
    prev_members = {}
    next_family_id = 1
    all_dates = [x['date'] for x in series_data]
    date_columns = [d.strftime("%Y-%m-%d") for d in all_dates]

    logger.info("开始追踪族谱...")
    
    for i, step in enumerate(series_data):
        curr_date = step['date']
        curr_date_str = curr_date.strftime("%Y-%m-%d")
        curr_clusters = step['clusters'] # {label: [members]}
        curr_reps = step['representatives'] # {label: rep_code}
        
        curr_map = {} # {label: family_id}
        
        for label, members in curr_clusters.items():
            members_set = set(members)
            best_match_id = None
            best_iou = -1.0
            
            if i > 0:
                for p_label, p_members in prev_members.items():
                    intersection = len(members_set & p_members)
                    union = len(members_set | p_members)
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_match_label_prev = p_label
                
                if best_iou > threshold:
                    best_match_id = prev_map[best_match_label_prev]
            
            if best_match_id is not None:
                family_id = best_match_id
            else:
                family_id = next_family_id
                next_family_id += 1
            
            curr_map[label] = family_id
            
            rep_code = curr_reps.get(label, "Unknown")
            if 'label_map' in step and rep_code in step['label_map']:
                rep_name = step['label_map'][rep_code]
            else:
                try:
                    rep_name = get_security_info(rep_code).display_name
                except:
                    rep_name = rep_code
            size = len(members)
            
            info_str = f"{rep_name}({size})"
            family_history[family_id][curr_date_str] = info_str
            
        prev_map = curr_map
        prev_members = {k: set(v) for k, v in curr_clusters.items()}
        
    df = pd.DataFrame.from_dict(family_history, orient='index')    
    df = df.reindex(columns=date_columns)
    df = df.sort_index()
    df['Survival_Rate'] = df.notna().sum(axis=1) / len(date_columns)
    
    return df

# -------------------- 方案二：桑基图 (Sankey Diagram) --------------------------
def generate_sankey_chart(series_data, save_path="cluster_sankey.html", min_size=3):
    """
    生成桑基图 HTML。
    min_size: 最小簇大小，低于此大小的簇将被过滤以减少噪音
    """
    if not HAS_PYECHARTS:
        logger.error("无法生成桑基图: 缺少 pyecharts。")
        return
        
    if len(series_data) < 2:
        logger.warning("数据少于2期，无法生成流向图。")
        return

    nodes = []
    links = []
    
    node_map = {}
    existing_node_names = set()
    logger.info(f"开始生成桑基图结构 (过滤小簇 < {min_size})...")
    max_clusters_per_period = 0
    
    # 生成所有节点
    for i, step in enumerate(series_data):
        date_str = step['date'].strftime("%Y-%m-%d") # 完整日期保持唯一性
        date_short = step['date'].strftime("%y%m")  # 显示用短日期
        
        current_period_clusters = 0
        
        for label, members in step['clusters'].items():
            size = len(members)
            if size < min_size:
                continue
            current_period_clusters += 1
            rep_code = step['representatives'].get(label, "")
            if 'label_map' in step and rep_code in step['label_map']:
                 full_name = step['label_map'][rep_code].upper()
                 rep_name = full_name.replace("ETF", "").replace("联接", "")[:5]
            else:
                try:
                    full_name = get_security_info(rep_code).display_name.upper()
                    rep_name = full_name.replace("ETF", "").replace("联接", "")[:5]
                except:
                    rep_name = str(rep_code).upper()
                
            base_name = f"{date_short}\n{rep_name}({size})"
            unique_name = base_name
            suffix = 1
            while unique_name in existing_node_names:
                unique_name = f"{base_name}_{suffix}"
                suffix += 1
            
            existing_node_names.add(unique_name)
            member_names = []
            for etf in members:
                if 'label_map' in step and etf in step['label_map']:
                    member_names.append(step['label_map'][etf])
                else:
                    try:
                        member_names.append(get_security_info(etf).display_name)
                    except:
                        member_names.append(str(etf))
            members_str = ", ".join(member_names)
            wrapped_members = "<br/>".join(textwrap.wrap(members_str, width=40))
            
            nodes.append({"name": unique_name, "desc": wrapped_members})
            node_map[(i, label)] = unique_name
            
        max_clusters_per_period = max(max_clusters_per_period, current_period_clusters)
            
    # 生成连线 (Links)
    for i in range(len(series_data) - 1):
        curr_step = series_data[i]
        next_step = series_data[i+1]
        etf_to_next_label = next_step['details'] 
        for curr_label, curr_members in curr_step['clusters'].items():
            if (i, curr_label) not in node_map:
                continue
            source_node_name = node_map[(i, curr_label)]
            dest_counts = defaultdict(int)      
            for etf in curr_members:
                if etf in etf_to_next_label:
                    next_tgt_label = etf_to_next_label[etf]
                    if (i+1, next_tgt_label) in node_map:
                        dest_counts[next_tgt_label] += 1
                    else:
                        pass
                else:
                    pass
            
            for next_tgt_label, count in dest_counts.items():
                if count == 0: continue
                target_node_name = node_map[(i+1, next_tgt_label)]
                links.append({
                    "source": source_node_name,
                    "target": target_node_name,
                    "value": count
                })
                
    # 绘图
    chart_height = max(800, max_clusters_per_period * 50)
    logger.info(f"  -> 动态设置图表高度: {chart_height}px (最大簇数: {max_clusters_per_period})")
    
    try:
        sankey = (
            Sankey(init_opts=opts.InitOpts(width="100%", height=f"{chart_height}px"))
            .add(
                series_name="ETF 流向",
                nodes=nodes,
                links=links,
                pos_right = "5%",
                layout_iterations=256,
                orient='horizontal',
                linestyle_opt=opts.LineStyleOpts(opacity=0.3, curve=0.5, color="source"),
                label_opts=opts.LabelOpts(position="right", font_size=12),
                node_gap=22, # 节点间距
                node_width=22, # 节点宽度
                emphasis_opts=opts.EmphasisOpts(
                    focus='series',
                    blur_scope="global",
                    is_show_label_line=True,
                ),
                tooltip_opts=opts.TooltipOpts(
                    formatter=JsCode(
                        "function (params) { "
                        "if (params.dataType === 'node') { "
                        "return params.data.name.replace('\\n', '<br/>') + '<br/><hr/><b>包含的ETF:</b><br/>' + params.data.desc; "
                        "} else { "
                        "return params.data.source.replace('\\n', ' ') + ' -> ' + params.data.target.replace('\\n', ' ') + ' : ' + params.data.value; "
                        "} }"
                    )
                )
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"ETF 族谱演变 ({len(series_data)} 周期)"),
                toolbox_opts=opts.ToolboxOpts(is_show=True, feature={"saveAsImage": {"pixel_ratio": 2}})
            )
        )
        
        sankey.render(save_path)
        logger.info(f"桑基图已保存至: {save_path}")
    except Exception as e:
        logger.error(f"桑基图渲染失败: {e}")


# -------------------- 主入口 --------------------------
if __name__ == "__main__":
    # 示例运行参数
    PERIODS = 6
    FREQ = "M" # Month End
    CLUSTER_DIR = os.path.join(os.path.dirname(__file__), 'cluster_results')
    CACHE_FILE = os.path.join(CLUSTER_DIR, "cluster_data_cache2.pkl")
    
    logger.info(">>> 开始聚类演变分析程序 <<<")
    
    # 获取数据 (使用缓存)
    data_series = get_cluster_series(periods=PERIODS, freq=FREQ, cache_file=CACHE_FILE)
    
    if not data_series:
        logger.error("未获取到数据，程序退出。")
        exit()
        
    # 执行方案一：族谱表
    logger.info("\n>>> 方案一：生成族谱追踪表 <<<")
    df_genealogy = track_cluster_genealogy(data_series, threshold=0.3)
    if not df_genealogy.empty:
        print_df = df_genealogy.fillna("")
        print(print_df.drop(columns=['Survival_Rate']).to_string())
        csv_path = os.path.join(CLUSTER_DIR, "cluster_genealogy.csv")
        df_genealogy.to_csv(csv_path)
        logger.info(f"族谱表已保存至: {csv_path}")
    else:
        logger.info("族谱为空。")
        
    # 执行方案二：桑基图
    logger.info("\n>>> 方案二：生成桑基图 <<<")
    generate_sankey_chart(data_series, save_path=os.path.join(CLUSTER_DIR, "cluster_sankey.html"), min_size=1)    
    logger.info("分析完成。")
