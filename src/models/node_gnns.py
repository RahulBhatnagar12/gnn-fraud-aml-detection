"""
Node Classification GNNs for Elliptic Bitcoin Dataset (Semester 7 - Tasks 6 & 7).

Models:
- GCNNodeClassifier (Kipf & Welling, 2017)
- GraphSAGENodeClassifier (Hamilton et al., 2017)
- GATNodeClassifier (Velickovic et al., 2018)
- HybridGCNNodeClassifier (combines GCN embeddings with raw features, per Weber et al., 2019)

Unlabelled nodes (y == -1) are kept in edge_index for message passing and masked out of loss & evaluation.
"""

from typing import Dict, Tuple

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
)


class GCNNodeClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.head = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.encode(x, edge_index)
        logits = self.head(emb).squeeze(-1)
        return logits, emb


class GraphSAGENodeClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.head = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.encode(x, edge_index)
        logits = self.head(emb).squeeze(-1)
        return logits, emb


class GATNodeClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        out_per_head = max(1, hidden_channels // heads)
        gat_out = out_per_head * heads
        self.conv1 = GATConv(in_channels, out_per_head, heads=heads, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(gat_out)
        self.conv2 = GATConv(gat_out, out_per_head, heads=heads, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(gat_out)
        self.head = nn.Linear(gat_out, 1)
        self.dropout = dropout

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.bn1(self.conv1(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.bn2(self.conv2(h, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.encode(x, edge_index)
        logits = self.head(emb).squeeze(-1)
        return logits, emb


class HybridGCNNodeClassifier(nn.Module):
    """Concatenates raw node features with GCN embeddings (Weber et al., 2019 skip architecture)."""

    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.head = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )
        self.dropout = dropout

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.encode(x, edge_index)
        combined = torch.cat([x, emb], dim=-1)
        logits = self.head(combined).squeeze(-1)
        return logits, emb


def build_node_gnn(
    arch: str,
    in_channels: int,
    hidden_channels: int = 64,
    heads: int = 4,
    dropout: float = 0.3,
) -> nn.Module:
    key = arch.lower()
    if key == "gcn":
        return GCNNodeClassifier(in_channels, hidden_channels, dropout)
    if key == "graphsage":
        return GraphSAGENodeClassifier(in_channels, hidden_channels, dropout)
    if key == "gat":
        return GATNodeClassifier(in_channels, hidden_channels, heads, dropout)
    if key == "hybrid_gcn":
        return HybridGCNNodeClassifier(in_channels, hidden_channels, dropout)
    raise ValueError(f"Unsupported node GNN architecture: {arch}")


def train_and_eval_node_gnn(
    arch: str,
    data: Data,
    x_override: torch.Tensor = None,
    hidden_channels: int = 64,
    heads: int = 4,
    dropout: float = 0.3,
    epochs: int = 50,
    lr: float = 0.005,
    weight_decay: float = 5e-4,
    use_focal_loss: bool = False,
    use_class_weights: bool = True,
    seed: int = 42,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Train a node classification GNN on Elliptic temporal train_mask (ignoring -1 unlabelled nodes
    in the loss while retaining them in edge_index for message passing) and evaluate on test_mask.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = (x_override if x_override is not None else data.x).to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    test_mask = data.test_mask.to(device)

    # Standardize features strictly using training labeled + train-timestep nodes
    train_t_mask = (data.timestep <= int(data.timestep[data.train_mask].max())).to(device)
    mu = x[train_t_mask].mean(dim=0, keepdim=True)
    sd = x[train_t_mask].std(dim=0, keepdim=True) + 1e-6
    x = (x - mu) / sd

    model = build_node_gnn(arch, x.size(1), hidden_channels, heads, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    pos_weight = compute_pos_weight(y[train_mask]) if use_class_weights else None
    if use_focal_loss:
        criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0, pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_loss = float("inf")
    best_state = None
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(x, edge_index)
        loss = criterion(logits[train_mask], y[train_mask].float())
        loss.backward()
        optimizer.step()
        if float(loss.item()) < best_loss:
            best_loss = float(loss.item())
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        logits, embeddings = model(x, edge_index)
        probs = torch.sigmoid(logits[test_mask]).cpu().numpy()
        y_test = y[test_mask].cpu().numpy()
        emb_np = embeddings.cpu().numpy()

    metrics = compute_binary_classification_metrics(y_test, probs, threshold=0.5)
    return metrics, probs, emb_np
