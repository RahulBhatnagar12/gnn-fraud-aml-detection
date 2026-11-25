"""
Edge Classification GNNs for IBM AML Dataset (Semester 7 - Tasks 6 & 7, Section 6).

Supports the two IBM AML edge-classification strategies described in Section 6:
1. Concatenated endpoint embeddings + edge attributes (GCN / GraphSAGE encoders)
2. Attention-based GAT layers that natively incorporate edge attributes (edge_dim)

Includes class-weighted loss, minority-class oversampling on training edges, and
NeighborLoader-ready / mini-batch edge scoring for large transaction graphs.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from src.utils.metrics import (
    BinaryFocalLoss,
    compute_binary_classification_metrics,
    compute_pos_weight,
    oversample_minority_indices,
)


class _EdgeMLPHead(nn.Module):
    def __init__(self, node_emb_dim: int, edge_feat_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_emb_dim + edge_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        target_edge_idx: torch.Tensor,
    ) -> torch.Tensor:
        src = edge_index[0, target_edge_idx]
        dst = edge_index[1, target_edge_idx]
        pair_feat = torch.cat([node_emb[src], node_emb[dst], edge_attr[target_edge_idx]], dim=-1)
        return self.mlp(pair_feat).squeeze(-1)


class GCNEdgeClassifier(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.edge_head = _EdgeMLPHead(hidden_dim, edge_dim, hidden_dim, dropout)
        self.dropout = dropout

    def encode_nodes(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        return h

    def forward_edges(
        self,
        node_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        target_edge_idx: torch.Tensor,
    ) -> torch.Tensor:
        return self.edge_head(node_emb, edge_index, edge_attr, target_edge_idx)


class SAGEEdgeClassifier(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(node_dim, hidden_dim, aggr="mean")
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.edge_head = _EdgeMLPHead(hidden_dim, edge_dim, hidden_dim, dropout)
        self.dropout = dropout

    def encode_nodes(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        return h

    def forward_edges(
        self,
        node_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        target_edge_idx: torch.Tensor,
    ) -> torch.Tensor:
        return self.edge_head(node_emb, edge_index, edge_attr, target_edge_idx)


class GATEdgeClassifier(nn.Module):
    """GAT encoder accepting edge attributes directly (edge_dim) + edge MLP classifier."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        out_per_head = max(1, hidden_dim // heads)
        gat_dim = out_per_head * heads
        self.conv1 = GATConv(node_dim, out_per_head, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(gat_dim)
        self.conv2 = GATConv(gat_dim, out_per_head, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(gat_dim)
        self.edge_head = _EdgeMLPHead(gat_dim, edge_dim, hidden_dim, dropout)
        self.dropout = dropout

    def encode_nodes(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.bn1(self.conv1(x, edge_index, edge_attr=edge_attr)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.bn2(self.conv2(h, edge_index, edge_attr=edge_attr)))
        return h

    def forward_edges(
        self,
        node_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        target_edge_idx: torch.Tensor,
    ) -> torch.Tensor:
        return self.edge_head(node_emb, edge_index, edge_attr, target_edge_idx)


def build_edge_gnn(
    arch: str,
    node_dim: int,
    edge_dim: int,
    hidden_dim: int = 64,
    heads: int = 4,
    dropout: float = 0.3,
) -> nn.Module:
    key = arch.lower()
    if key == "gcn":
        return GCNEdgeClassifier(node_dim, edge_dim, hidden_dim, dropout)
    if key == "graphsage":
        return SAGEEdgeClassifier(node_dim, edge_dim, hidden_dim, dropout)
    if key == "gat":
        return GATEdgeClassifier(node_dim, edge_dim, hidden_dim, heads, dropout)
    raise ValueError(f"Unsupported edge GNN architecture: {arch}")


def _tune_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Tune decision threshold on validation edges to maximize illicit F1 under severe imbalance."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.20, 0.90, 15):
        m = compute_binary_classification_metrics(y_true, y_prob, threshold=float(t))
        if m["illicit_f1"] > best_f1:
            best_f1 = m["illicit_f1"]
            best_t = float(t)
    return best_t


def train_and_eval_edge_gnn(
    arch: str,
    data: Data,
    hidden_dim: int = 64,
    heads: int = 4,
    dropout: float = 0.3,
    epochs: int = 40,
    lr: float = 0.005,
    weight_decay: float = 5e-4,
    use_class_weights: bool = True,
    use_oversampling: bool = True,
    oversample_ratio: float = 0.15,
    use_focal_loss: bool = False,
    seed: int = 42,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Train an edge-classification GNN on IBM AML using chronological masks.
    Note: Message passing uses only non-test edges during training (train + val)
    to avoid leaking future test structure into training representations.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device)
    edge_label = data.edge_label.to(device)

    train_idx = torch.where(data.train_mask)[0].to(device)
    val_idx = torch.where(data.val_mask)[0].to(device)
    test_idx = torch.where(data.test_mask)[0].to(device)

    # Message-passing graph for training uses only training edges (strict temporal isolation)
    train_mp_edge_index = edge_index[:, train_idx]
    train_mp_edge_attr = edge_attr[train_idx]

    if use_oversampling:
        sampled_train_idx = oversample_minority_indices(
            train_idx, edge_label, target_pos_ratio=oversample_ratio, seed=seed
        )
    else:
        sampled_train_idx = train_idx

    model = build_edge_gnn(arch, x.size(1), edge_attr.size(1), hidden_dim, heads, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    pos_weight = compute_pos_weight(edge_label[sampled_train_idx]) if use_class_weights else None
    if use_focal_loss:
        criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0, pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_ap = -1.0
    best_state = None
    best_thresh = 0.5

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        node_emb = model.encode_nodes(x, train_mp_edge_index, train_mp_edge_attr)
        logits = model.forward_edges(node_emb, edge_index, edge_attr, sampled_train_idx)
        loss = criterion(logits, edge_label[sampled_train_idx].float())
        loss.backward()
        optimizer.step()

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_mp_mask = data.train_mask | data.val_mask
            val_emb = model.encode_nodes(x, edge_index[:, val_mp_mask.to(device)], edge_attr[val_mp_mask.to(device)])
            val_logits = model.forward_edges(val_emb, edge_index, edge_attr, val_idx)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_y = edge_label[val_idx].cpu().numpy()
            val_metrics = compute_binary_classification_metrics(val_y, val_probs, threshold=0.5)
            if val_metrics["auc_pr"] >= best_val_ap:
                best_val_ap = val_metrics["auc_pr"]
                best_thresh = _tune_f1_threshold(val_y, val_probs)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        full_emb = model.encode_nodes(x, edge_index, edge_attr)
        test_logits = model.forward_edges(full_emb, edge_index, edge_attr, test_idx)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        y_test = edge_label[test_idx].cpu().numpy()

    metrics = compute_binary_classification_metrics(y_test, test_probs, threshold=best_thresh)
    return metrics, test_probs
