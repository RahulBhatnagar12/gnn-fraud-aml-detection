"""
Node and Edge Feature Engineering (Semester 7 - Task 5 & Table 4).

Supports:
1. Elliptic Bitcoin Dataset:
   - Splits 166 features into Local (0..93) vs Aggregated (94..165) for ablation studies.
2. IBM AML Dataset:
   - Vectorized node-level features: in-degree, out-degree, amount statistics (sum/mean/std/max),
     net flow, and activity burstiness (inter-transaction burstiness index & inter-arrival stats).
   - Causal edge-level features: log1p(amount_usd), time since sender/receiver's previous tx,
     payment format one-hot, and currency one-hot.
   - Feature-group index maps to measure how much each feature group adds.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def get_elliptic_feature_views(data: Data, local_feature_count: int = 94) -> Dict[str, torch.Tensor]:
    """Return Elliptic feature subsets for feature-group ablation (Table 4)."""
    x = data.x
    return {
        "local_94": x[:, :local_feature_count],
        "aggregated_72": x[:, local_feature_count:],
        "all_166": x,
    }


def _compute_burstiness_stats(
    df_events: pd.DataFrame,
    account_col: str,
    ts_col: str = "timestamp_sec",
) -> pd.DataFrame:
    """
    Compute activity burstiness per account from chronologically sorted events:
    - burstiness = (std_dt - mean_dt) / (std_dt + mean_dt + eps) in [-1, 1]
    - log_mean_dt, log_min_dt, active_span_hours
    """
    events = df_events[[account_col, ts_col]].sort_values([account_col, ts_col], kind="mergesort")
    events["dt"] = events.groupby(account_col)[ts_col].diff()

    dt_valid = events.dropna(subset=["dt"])
    dt_stats = dt_valid.groupby(account_col)["dt"].agg(["mean", "std", "min"]).rename(
        columns={"mean": "mean_dt", "std": "std_dt", "min": "min_dt"}
    )
    dt_stats["std_dt"] = dt_stats["std_dt"].fillna(0.0)
    denom = dt_stats["std_dt"] + dt_stats["mean_dt"] + 1e-6
    dt_stats["burstiness"] = (dt_stats["std_dt"] - dt_stats["mean_dt"]) / denom

    span = events.groupby(account_col)[ts_col].agg(["min", "max"])
    span["active_span_hours"] = (span["max"] - span["min"]) / 3600.0

    out = span[["active_span_hours"]].join(dt_stats[["burstiness", "mean_dt", "min_dt"]], how="left")
    out["burstiness"] = out["burstiness"].fillna(0.0)
    out["log_mean_dt"] = np.log1p(out["mean_dt"].fillna(0.0).clip(lower=0.0))
    out["log_min_dt"] = np.log1p(out["min_dt"].fillna(0.0).clip(lower=0.0))
    return out[["burstiness", "log_mean_dt", "log_min_dt", "active_span_hours"]]


def compute_ibm_aml_features(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    top_k_currencies: int = 6,
    top_k_formats: int = 7,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Compute vectorized node-level and edge-level features for a cleaned, chronologically
    sorted IBM AML dataframe. Scalers are fit strictly on the training slice to prevent leakage.

    Returns:
        node_features: (num_accounts, D_node) float32 array aligned to unique accounts order
        edge_features: (num_edges, D_edge) float32 array aligned to df rows
        node_feature_names: list of node feature names
        edge_feature_names: list of edge feature names
    """
    if not df["timestamp_sec"].is_monotonic_increasing:
        df = df.sort_values("timestamp_sec", kind="mergesort").reset_index(drop=True)

    unique_accounts = pd.Index(pd.concat([df["src_account"], df["dst_account"]], ignore_index=True).unique())
    n_edges = len(df)
    train_end = max(1, int(n_edges * train_ratio))

    # --- 1. Node-level Structural & Amount Statistics ---
    out_grp = df.groupby("src_account")["amount_usd"].agg(["count", "sum", "mean", "std", "max"]).rename(
        columns={
            "count": "out_degree",
            "sum": "out_amt_sum",
            "mean": "out_amt_mean",
            "std": "out_amt_std",
            "max": "out_amt_max",
        }
    )
    in_grp = df.groupby("dst_account")["amount_usd"].agg(["count", "sum", "mean", "std", "max"]).rename(
        columns={
            "count": "in_degree",
            "sum": "in_amt_sum",
            "mean": "in_amt_mean",
            "std": "in_amt_std",
            "max": "in_amt_max",
        }
    )

    node_df = pd.DataFrame(index=unique_accounts)
    node_df = node_df.join(in_grp, how="left").join(out_grp, how="left").fillna(0.0)
    node_df["total_degree"] = node_df["in_degree"] + node_df["out_degree"]
    node_df["degree_ratio"] = (node_df["in_degree"] + 1.0) / (node_df["out_degree"] + 1.0)
    node_df["net_flow_usd"] = node_df["in_amt_sum"] - node_df["out_amt_sum"]

    # Log-scale heavy-tailed monetary & degree features
    for col in [
        "in_degree", "out_degree", "total_degree",
        "in_amt_sum", "in_amt_mean", "in_amt_std", "in_amt_max",
        "out_amt_sum", "out_amt_mean", "out_amt_std", "out_amt_max",
    ]:
        node_df[col] = np.log1p(node_df[col].clip(lower=0.0))
    node_df["net_flow_usd"] = np.sign(node_df["net_flow_usd"]) * np.log1p(np.abs(node_df["net_flow_usd"]))

    # --- 2. Node-level Activity Burstiness ---
    src_events = df[["src_account", "timestamp_sec"]].rename(columns={"src_account": "account"})
    dst_events = df[["dst_account", "timestamp_sec"]].rename(columns={"dst_account": "account"})
    all_events = pd.concat([src_events, dst_events], ignore_index=True)
    burst_df = _compute_burstiness_stats(all_events, account_col="account")
    node_df = node_df.join(burst_df, how="left").fillna(0.0)

    node_feature_names = list(node_df.columns)
    node_matrix = node_df.to_numpy(dtype=np.float32)

    # Standardize node features using statistics over accounts active in training slice
    train_accounts = pd.Index(
        pd.concat([df.iloc[:train_end]["src_account"], df.iloc[:train_end]["dst_account"]]).unique()
    )
    train_acc_mask = unique_accounts.isin(train_accounts)
    if train_acc_mask.sum() > 0:
        mu_n = node_matrix[train_acc_mask].mean(axis=0, keepdims=True)
        sd_n = node_matrix[train_acc_mask].std(axis=0, keepdims=True) + 1e-6
    else:
        mu_n = node_matrix.mean(axis=0, keepdims=True)
        sd_n = node_matrix.std(axis=0, keepdims=True) + 1e-6
    node_matrix = ((node_matrix - mu_n) / sd_n).astype(np.float32)

    # --- 3. Edge-level Causal Features ---
    edge_df = pd.DataFrame(index=df.index)
    edge_df["log_amount_usd"] = np.log1p(df["amount_usd"].to_numpy(dtype=np.float64)).astype(np.float32)

    # Causal inter-transaction time since sender/receiver's previous transaction
    src_dt = df.groupby("src_account")["timestamp_sec"].diff().fillna(-1.0).to_numpy(dtype=np.float64)
    dst_dt = df.groupby("dst_account")["timestamp_sec"].diff().fillna(-1.0).to_numpy(dtype=np.float64)
    edge_df["is_first_src_tx"] = (src_dt < 0).astype(np.float32)
    edge_df["is_first_dst_tx"] = (dst_dt < 0).astype(np.float32)
    edge_df["log_dt_prev_src"] = np.log1p(np.clip(src_dt, 0.0, None)).astype(np.float32)
    edge_df["log_dt_prev_dst"] = np.log1p(np.clip(dst_dt, 0.0, None)).astype(np.float32)

    # Payment format one-hot (fit vocabulary on training slice)
    top_formats = df.iloc[:train_end]["Payment Format"].astype(str).value_counts().head(top_k_formats).index.tolist()
    fmt_series = df["Payment Format"].astype(str).where(df["Payment Format"].astype(str).isin(top_formats), "Other")
    for fmt in top_formats + ["Other"]:
        edge_df[f"fmt_{fmt}"] = (fmt_series == fmt).astype(np.float32)

    # Currency one-hot (fit vocabulary on training slice)
    top_currs = df.iloc[:train_end]["Payment Currency"].astype(str).value_counts().head(top_k_currencies).index.tolist()
    curr_series = df["Payment Currency"].astype(str).where(df["Payment Currency"].astype(str).isin(top_currs), "Other")
    for curr in top_currs + ["Other"]:
        edge_df[f"curr_{curr}"] = (curr_series == curr).astype(np.float32)

    edge_feature_names = list(edge_df.columns)
    edge_matrix = edge_df.to_numpy(dtype=np.float32)

    # Standardize continuous edge columns (first 5 numeric features except binary indicators)
    cont_cols = [0, 3, 4]
    mu_e = edge_matrix[:train_end, cont_cols].mean(axis=0, keepdims=True)
    sd_e = edge_matrix[:train_end, cont_cols].std(axis=0, keepdims=True) + 1e-6
    edge_matrix[:, cont_cols] = (edge_matrix[:, cont_cols] - mu_e) / sd_e

    return node_matrix, edge_matrix, node_feature_names, edge_feature_names


def get_ibm_node_feature_group_indices(node_feature_names: List[str]) -> Dict[str, List[int]]:
    """Group IBM AML node features for the feature-group study (Table 4)."""
    structural = [i for i, n in enumerate(node_feature_names) if "degree" in n]
    amounts = [i for i, n in enumerate(node_feature_names) if "amt" in n or "flow" in n]
    burstiness = [i for i, n in enumerate(node_feature_names) if "burst" in n or "dt" in n or "span" in n]
    return {
        "structural": structural,
        "amount_stats": amounts,
        "burstiness": burstiness,
        "all": list(range(len(node_feature_names))),
    }


def get_ibm_edge_feature_group_indices(edge_feature_names: List[str]) -> Dict[str, List[int]]:
    """Group IBM AML edge features for the feature-group study (Table 4)."""
    amount = [i for i, n in enumerate(edge_feature_names) if "amount" in n]
    temporal = [i for i, n in enumerate(edge_feature_names) if "dt" in n or "first" in n]
    categorical = [i for i, n in enumerate(edge_feature_names) if n.startswith("fmt_") or n.startswith("curr_")]
    return {
        "amount": amount,
        "temporal_recency": temporal,
        "format_and_currency": categorical,
        "all": list(range(len(edge_feature_names))),
    }
