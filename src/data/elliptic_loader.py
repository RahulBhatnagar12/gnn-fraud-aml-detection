"""
Elliptic Bitcoin Dataset Loader & Preprocessor (Semester 7 - Tasks 2, 3, 4).

Handles:
- Aligning elliptic_txs_features.csv, elliptic_txs_classes.csv, elliptic_txs_edgelist.csv by txId
- Mapping transaction IDs to contiguous 0-based node indices
- Encoding labels: illicit = 1, licit = 0, unknown = -1 (excluded from loss/eval, kept for message passing)
- Preserving discrete timestep (1..49) for temporal train/test split (no random split leakage)
- Generating a schema-faithful synthetic demo Elliptic dataset when Kaggle CSVs are absent
"""

from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

ELLIPTIC_FILES = (
    "elliptic_txs_features.csv",
    "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv",
)


def has_elliptic_files(data_dir: Union[str, Path]) -> bool:
    """Return True if all three Elliptic CSV files exist in data_dir."""
    data_dir = Path(data_dir)
    return all((data_dir / fname).exists() for fname in ELLIPTIC_FILES)


def generate_demo_elliptic_csvs(
    output_dir: Union[str, Path],
    num_nodes: int = 3500,
    num_timesteps: int = 49,
    seed: int = 42,
) -> Path:
    """
    Generate synthetic Elliptic-formatted CSV files (166 features: 94 local + 72 aggregated,
    49 timesteps, ~2% illicit, ~21% licit, ~77% unknown) so the pipeline runs end-to-end
    before downloading the full 203,769-node Kaggle dataset.
    """
    rng = np.random.default_rng(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tx_ids = np.arange(100000, 100000 + num_nodes, dtype=np.int64)
    timesteps = np.sort(rng.integers(1, num_timesteps + 1, size=num_nodes))

    # Label distribution mirroring Elliptic: ~2.3% illicit, ~20.7% licit, ~77% unknown
    probs = rng.random(num_nodes)
    raw_classes = np.where(probs < 0.025, "1", np.where(probs < 0.23, "2", "unknown"))

    # 94 local features (subtle signal for illicit transactions)
    local_feats = rng.normal(0.0, 1.0, size=(num_nodes, 94)).astype(np.float32)
    illicit_mask = raw_classes == "1"
    licit_mask = raw_classes == "2"
    local_feats[illicit_mask, :15] += rng.normal(0.75, 0.25, size=(illicit_mask.sum(), 15))
    local_feats[licit_mask, :15] -= rng.normal(0.15, 0.20, size=(licit_mask.sum(), 15))

    # Directed DAG-like edges within/across adjacent timesteps (homophily + flow)
    edges = []
    for t in range(1, num_timesteps + 1):
        curr_idx = np.where(timesteps == t)[0]
        next_idx = np.where((timesteps >= t) & (timesteps <= min(t + 1, num_timesteps)))[0]
        if len(curr_idx) == 0 or len(next_idx) < 2:
            continue
        for u in curr_idx:
            deg = int(rng.integers(1, 3))
            # Prefer connecting same-class nodes with higher probability to simulate graph homophily
            if raw_classes[u] == "1":
                illicit_candidates = next_idx[raw_classes[next_idx] == "1"]
                if len(illicit_candidates) > 0 and rng.random() < 0.55:
                    vs = rng.choice(illicit_candidates, size=min(deg, len(illicit_candidates)), replace=False)
                else:
                    vs = rng.choice(next_idx, size=min(deg, len(next_idx)), replace=False)
            else:
                vs = rng.choice(next_idx, size=min(deg, len(next_idx)), replace=False)
            for v in vs:
                if u != v:
                    edges.append((int(tx_ids[u]), int(tx_ids[v])))

    # 72 aggregated neighbor features computed from local_feats[:72]
    adj_sum = np.zeros((num_nodes, 72), dtype=np.float32)
    deg_cnt = np.ones((num_nodes, 1), dtype=np.float32)
    id_to_idx = {int(tid): i for i, tid in enumerate(tx_ids)}
    for u_tid, v_tid in edges:
        ui, vi = id_to_idx[u_tid], id_to_idx[v_tid]
        adj_sum[vi] += local_feats[ui, :72]
        adj_sum[ui] += local_feats[vi, :72]
        deg_cnt[vi] += 1.0
        deg_cnt[ui] += 1.0
    agg_feats = adj_sum / deg_cnt

    all_feats = np.hstack([local_feats, agg_feats])
    df_feat = pd.DataFrame(all_feats)
    df_feat.insert(0, "timestep", timesteps)
    df_feat.insert(0, "txId", tx_ids)
    df_feat.to_csv(out_path / "elliptic_txs_features.csv", header=False, index=False)

    df_cls = pd.DataFrame({"txId": tx_ids, "class": raw_classes})
    df_cls.to_csv(out_path / "elliptic_txs_classes.csv", index=False)

    df_edges = pd.DataFrame(edges, columns=["txId1", "txId2"])
    df_edges.to_csv(out_path / "elliptic_txs_edgelist.csv", index=False)
    return out_path


def ensure_elliptic_dataset(raw_dir: Union[str, Path], demo_dir: Union[str, Path], seed: int = 42) -> tuple[Path, bool]:
    """Return (dataset_dir, is_demo). Uses raw_dir if Kaggle CSVs exist, else generates demo_dir."""
    raw_path = Path(raw_dir)
    if has_elliptic_files(raw_path):
        return raw_path, False
    demo_path = Path(demo_dir)
    if not has_elliptic_files(demo_path):
        generate_demo_elliptic_csvs(demo_path, seed=seed)
    return demo_path, True


def load_elliptic_graph(
    data_dir: Union[str, Path],
    train_max_timestep: int = 34,
) -> Data:
    """
    Load Elliptic CSVs into a PyTorch Geometric directed Data object.

    Node features: 166 dims (0..93 local, 94..165 aggregated).
    Labels: illicit=1, licit=0, unknown=-1.
    Temporal split: timesteps <= train_max_timestep for train, > train_max_timestep for test.
    """
    data_dir = Path(data_dir)
    if not has_elliptic_files(data_dir):
        raise FileNotFoundError(
            f"Missing Elliptic CSV files in {data_dir}. Expected: {', '.join(ELLIPTIC_FILES)}"
        )

    df_features = pd.read_csv(data_dir / "elliptic_txs_features.csv", header=None)
    df_classes = pd.read_csv(data_dir / "elliptic_txs_classes.csv")
    df_edges = pd.read_csv(data_dir / "elliptic_txs_edgelist.csv")

    df_features = df_features.rename(columns={0: "txId", 1: "timestep"})
    df_features["txId"] = df_features["txId"].astype(np.int64)
    df_features["timestep"] = df_features["timestep"].astype(np.int64)

    # Contiguous 0-based index mapping
    tx_ids = df_features["txId"].to_numpy()
    tx_to_idx: Dict[int, int] = {int(tid): idx for idx, tid in enumerate(tx_ids)}

    # Align class labels via txId; normalize string/int class representations
    df_classes["txId"] = df_classes["txId"].astype(np.int64)
    df_classes["class_norm"] = df_classes["class"].astype(str).str.strip().str.lower()
    label_map = {"1": 1, "illicit": 1, "2": 0, "licit": 0, "unknown": -1}

    df_merged = df_features[["txId", "timestep"]].merge(
        df_classes[["txId", "class_norm"]], on="txId", how="left"
    )
    labels = df_merged["class_norm"].map(label_map).fillna(-1).to_numpy(dtype=np.int64)

    # Feature matrix (cols 2..167 -> 166 features)
    x = df_features.iloc[:, 2:].to_numpy(dtype=np.float32)
    timesteps = df_features["timestep"].to_numpy(dtype=np.int64)

    # Map directed edges; drop any edge referencing an unknown txId
    src_mapped = df_edges["txId1"].astype(np.int64).map(tx_to_idx)
    dst_mapped = df_edges["txId2"].astype(np.int64).map(tx_to_idx)
    valid_edges = src_mapped.notna() & dst_mapped.notna()
    edge_index = torch.tensor(
        np.vstack([src_mapped[valid_edges].to_numpy(dtype=np.int64),
                   dst_mapped[valid_edges].to_numpy(dtype=np.int64)]),
        dtype=torch.long,
    )

    labeled_mask = labels != -1
    train_mask = (timesteps <= train_max_timestep) & labeled_mask
    test_mask = (timesteps > train_max_timestep) & labeled_mask

    return Data(
        x=torch.from_numpy(x),
        edge_index=edge_index,
        y=torch.from_numpy(labels),
        timestep=torch.from_numpy(timesteps),
        labeled_mask=torch.from_numpy(labeled_mask),
        train_mask=torch.from_numpy(train_mask),
        test_mask=torch.from_numpy(test_mask),
    )


def verify_elliptic_stats(data: Data) -> Dict[str, Union[int, float]]:
    """Compute and return summary statistics for verification against the Elliptic paper."""
    y = data.y.cpu().numpy()
    t = data.timestep.cpu().numpy()
    num_nodes = int(data.num_nodes)
    num_edges = int(data.num_edges)
    num_illicit = int((y == 1).sum())
    num_licit = int((y == 0).sum())
    num_unknown = int((y == -1).sum())

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_features": int(data.num_features),
        "num_timesteps": int(len(np.unique(t))),
        "illicit_nodes": num_illicit,
        "licit_nodes": num_licit,
        "unknown_nodes": num_unknown,
        "illicit_pct_total": round(100.0 * num_illicit / max(num_nodes, 1), 2),
        "illicit_pct_labeled": round(100.0 * num_illicit / max(num_illicit + num_licit, 1), 2),
        "train_labeled_nodes": int(data.train_mask.sum().item()),
        "test_labeled_nodes": int(data.test_mask.sum().item()),
    }
