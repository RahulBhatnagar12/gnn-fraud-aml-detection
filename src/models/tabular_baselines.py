"""
Tabular Baselines (Semester 7 - Section 5.1 & Section 6).

Implements Logistic Regression and Random Forest baselines for:
- Elliptic node classification (comparing local 94 vs all 166 features, and raw + GNN embeddings)
- IBM AML edge classification (concatenating src node stats, dst node stats, and edge attributes)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data

from src.utils.metrics import compute_binary_classification_metrics


def build_ibm_edge_tabular_matrix(
    data: Data,
    edge_indices: torch.Tensor,
    node_cols: Optional[List[int]] = None,
    edge_cols: Optional[List[int]] = None,
) -> np.ndarray:
    """Construct edge-level tabular matrix [x_src || x_dst || edge_attr] for given edge indices."""
    x_np = data.x.cpu().numpy()
    e_np = data.edge_attr.cpu().numpy()
    if node_cols is not None:
        x_np = x_np[:, node_cols]
    if edge_cols is not None:
        e_np = e_np[:, edge_cols]

    idx_np = edge_indices.cpu().numpy()
    src = data.edge_index[0, idx_np].cpu().numpy()
    dst = data.edge_index[1, idx_np].cpu().numpy()
    return np.hstack([x_np[src], x_np[dst], e_np[idx_np]])


def run_tabular_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_tag: str = "all",
    seed: int = 42,
) -> List[Dict[str, object]]:
    """
    Train balanced Logistic Regression and Random Forest on temporal train split
    and evaluate illicit-class metrics on temporal test split.
    """
    results: List[Dict[str, object]] = []

    models: List[Tuple[str, object]] = [
        (
            "LogisticRegression",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=seed,
                )),
            ]),
        ),
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=150,
                max_depth=14,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            ),
        ),
    ]

    for name, clf in models:
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        metrics = compute_binary_classification_metrics(y_test, probs, threshold=0.5)
        results.append({
            "model": name,
            "features": feature_tag,
            **metrics,
        })

    return results
