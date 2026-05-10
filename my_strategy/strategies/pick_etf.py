#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2026/5/10 21:54
# @Author  : david_van
# @Desc    :

import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation
from typing import List, Dict, Optional, Tuple


class ETFPoolBuilder:
    """
    ETF精简池构建器（修正版）

    核心原则：让AP自适应地决定市场有多少种不同类型的ETF
    不强制逼近某个目标数量
    """

    def __init__(
            self,
            damping: float = 0.8,
            preference_quantile: float = 0.3,
            min_corr_days: int = 120,
            min_daily_volume: float = 5_000_000,
            max_iter: int = 500,
    ):
        """
        参数:
            damping: AP阻尼系数，0.5-1.0，越高越稳定，推荐0.8
            preference_quantile: 相关系数分位数，决定合并松紧度
                0.5 = 分得细（代表多）
                0.3 = 适中（推荐）
                0.1 = 分得粗（代表少）
            min_corr_days: 计算相关性所需的最少交易日
            min_daily_volume: 预过滤的日均成交额门槛（元）
            max_iter: AP最大迭代次数
        """
        self.damping = damping
        self.preference_quantile = preference_quantile
        self.min_corr_days = min_corr_days
        self.min_daily_volume = min_daily_volume
        self.max_iter = max_iter

    def build_pool(
            self,
            price_df: pd.DataFrame,
            volume_df: pd.DataFrame,
            etf_names: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        构建精简ETF池

        参数:
            price_df: 日收盘价 DataFrame, columns=ETF代码
            volume_df: 日成交额 DataFrame, columns=ETF代码
            etf_names: {代码: 名称} 字典（可选）

        返回:
            pool: 精简池ETF代码列表
            details: 聚类详情DataFrame
        """

        # ── Step 1: 预过滤 ──
        candidates = self._pre_filter(price_df, volume_df)
        print(f"[1/4] 预过滤: {len(price_df.columns)} → "
              f"{len(candidates)} 只ETF")

        if len(candidates) < 3:
            print("  ⚠️ 候选ETF不足3只，跳过聚类")
            return candidates, pd.DataFrame()

        # ── Step 2: 相关性矩阵 ──
        returns = price_df[candidates].pct_change().dropna()
        corr = returns.corr().values
        corr = np.nan_to_num(corr, nan=0.0)
        n = len(candidates)
        print(f"[2/4] 相关性矩阵: {n} × {n}")

        # ── Step 3: 计算preference ──
        # 取上三角（不含对角线）的所有相关系数
        triu_idx = np.triu_indices(n, k=1)
        all_corrs = corr[triu_idx]
        preference = np.quantile(all_corrs, self.preference_quantile)

        print(f"[3/4] preference = {preference:.4f} "
              f"(相关系数的{self.preference_quantile:.0%}分位数)")
        print(f"      含义: 组内相关性 ≥ {preference:.2f} 才值得合并")

        # ── Step 4: AP聚类 ──
        ap = AffinityPropagation(
            damping=self.damping,
            preference=preference,
            max_iter=self.max_iter,
            affinity='precomputed',
            random_state=42,
        )
        labels = ap.fit_predict(corr)

        if ap.cluster_centers_indices_ is None:
            print("  ⚠️ AP未收敛，提高damping重试")
            ap = AffinityPropagation(
                damping=min(self.damping + 0.1, 0.99),
                preference=preference,
                max_iter=self.max_iter * 2,
                affinity='precomputed',
                random_state=42,
            )
            labels = ap.fit_predict(corr)

        exemplar_indices = ap.cluster_centers_indices_
        n_clusters = len(exemplar_indices)

        print(f"[4/4] AP自然分组: {n_clusters} 个类型")
        print(f"      压缩比: {n} → {n_clusters} "
              f"({n_clusters/n*100:.1f}%)")

        # ── 构建精简池 ──
        pool = []
        rows = []

        for cid in range(n_clusters):
            mask = labels == cid
            members = [candidates[i] for i in range(n) if mask[i]]
            exemplar = candidates[exemplar_indices[cid]]

            # 选流动性最好的成员作为代表
            representative = self._pick_most_liquid(
                members, volume_df
            )
            pool.append(representative)

            # 记录详情
            rows.append({
                'cluster': cid,
                'representative': representative,
                'name': (etf_names or {}).get(representative, ''),
                'ap_exemplar': exemplar,
                'size': len(members),
                'members': members,
                'avg_corr': (
                    np.mean([
                        corr[candidates.index(a)][candidates.index(b)]
                        for a in members for b in members if a != b
                    ]) if len(members) > 1 else 1.0
                ),
            })

        details = pd.DataFrame(rows)

        # ── 池内相关性检查 ──
        pool_indices = [candidates.index(c) for c in pool]
        pool_corr = corr[np.ix_(pool_indices, pool_indices)]
        np.fill_diagonal(pool_corr, 0)
        max_corr = np.max(pool_corr)
        avg_corr = np.mean(np.abs(pool_corr[np.triu_indices(
            len(pool), k=1
        )]))

        print(f"\n池内质量检查:")
        print(f"  池内最高相关性: {max_corr:.3f}")
        print(f"  池内平均|相关性|: {avg_corr:.3f}")

        if max_corr > 0.85:
            pairs = []
            for i in range(len(pool)):
                for j in range(i+1, len(pool)):
                    if pool_corr[i][j] > 0.85:
                        pairs.append(
                            f"    {pool[i]} ↔ {pool[j]}: "
                            f"{pool_corr[i][j]:.2f}"
                        )
            print(f"  ⚠️ 存在高相关对:")
            for p in pairs[:5]:
                print(p)
        else:
            print(f"  ✅ 无高相关对 (all < 0.85)")

        return pool, details

    def _pre_filter(self, price_df, volume_df):
        valid = []
        for col in price_df.columns:
            if price_df[col].dropna().shape[0] < self.min_corr_days:
                continue
            if col in volume_df.columns:
                avg_vol = volume_df[col].tail(60).mean()
                if pd.isna(avg_vol) or avg_vol < self.min_daily_volume:
                    continue
            valid.append(col)
        return valid

    def _pick_most_liquid(self, members, volume_df):
        best, best_vol = members[0], 0
        for code in members:
            if code in volume_df.columns:
                vol = volume_df[code].tail(60).mean()
                if not pd.isna(vol) and vol > best_vol:
                    best_vol = vol
                    best = code
        return best