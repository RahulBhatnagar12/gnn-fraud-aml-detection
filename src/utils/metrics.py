"""
Evaluation Metrics & Class-Imbalance Utilities (Semester 7 - Section 6 & Table 4).

Follows Section 6 design decisions:
- Avoid plain accuracy due to severe class imbalance (~2% illicit in Elliptic, <=0.1% in IBM AML)
- Report illicit-class Precision, Recall, F1, and AUC-PR (plus ROC-AUC)
- Provide weighted BCE pos_weight, Binary Focal Loss, and minority-class oversampling
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """Focal Loss with optional positive-class weighting for severe imbalance."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_f = targets.float()
        bce = F.binary_cross_entropy_with_logits(
            logits, targets_f, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets_f + (1.0 - probs) * (1.0 - targets_f)
        alpha_t = self.alpha * targets_f + (1.0 - self.alpha) * (1.0 - targets_f)
        loss = alpha_t * ((1.0 - p_t) ** self.gamma) * bce
        return loss.mean()


def compute_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    """Compute num_neg / max(num_pos, 1) for BCEWithLogitsLoss."""
    pos = int((labels == 1).sum().item())
    neg = int((labels == 0).sum().item())
    ratio = neg / max(pos, 1)
    return torch.tensor([ratio], dtype=torch.float32, device=labels.device)


def oversample_minority_indices(
    indices: torch.Tensor,
    labels: torch.Tensor,
    target_pos_ratio: float = 0.15,
    seed: int = 42,
) -> torch.Tensor:
    """
    Oversample illicit (label == 1) indices within a training index tensor so that
    minority samples reach approximately `target_pos_ratio` of the batch pool.
    """
    y_sub = labels[indices]
    pos_idx = indices[y_sub == 1]
    neg_idx = indices[y_sub == 0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return indices

    current_ratio = len(pos_idx) / (len(pos_idx) + len(neg_idx))
    if current_ratio >= target_pos_ratio:
        return indices

    desired_pos = int((target_pos_ratio / (1.0 - target_pos_ratio)) * len(neg_idx))
    desired_pos = max(desired_pos, len(pos_idx))

    rng = np.random.default_rng(seed)
    sampled_pos = rng.choice(pos_idx.cpu().numpy(), size=desired_pos, replace=True)
    combined = torch.cat([neg_idx.cpu(), torch.from_numpy(sampled_pos).long()])
    perm = torch.from_numpy(rng.permutation(len(combined))).long()
    return combined[perm].to(indices.device)


def compute_binary_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute illicit-class Precision, Recall, F1, AUC-PR, and ROC-AUC."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary", zero_division=0
    )

    if len(np.unique(y_true)) > 1:
        auc_pr = float(average_precision_score(y_true, y_prob))
        roc_auc = float(roc_auc_score(y_true, y_prob))
    else:
        auc_pr = 0.0
        roc_auc = 0.0

    return {
        "illicit_precision": round(float(precision), 4),
        "illicit_recall": round(float(recall), 4),
        "illicit_f1": round(float(f1), 4),
        "auc_pr": round(auc_pr, 4),
        "roc_auc": round(roc_auc, 4),
    }


def format_results_table(rows: List[Dict[str, Union[str, float]]]) -> str:
    """Format model benchmark rows into a clean markdown/console table."""
    df = pd.DataFrame(rows)
    return df.to_string(index=False)


def save_results_artifacts(
    rows: List[Dict[str, Union[str, float]]],
    output_dir: Union[str, Path],
    prefix: str,
) -> Dict[str, Path]:
    """Save benchmark comparison metrics to CSV and JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{prefix}_metrics.csv"
    json_path = out / f"{prefix}_metrics.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return {"csv": csv_path, "json": json_path}
