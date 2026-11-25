"""
Semester 7 Pipeline 1: Elliptic Bitcoin Node Classification
Covers Tasks 2-7 from the Semester 7 plan:
- Data cleaning, contiguous indexing, unknown=-1 masking, temporal split (t<=34 vs t>34)
- Graph construction & verification in PyTorch Geometric
- Feature-group study (local 94 vs all 166 features)
- Tabular baselines (Logistic Regression, Random Forest)
- GNN baselines & tuning (GCN, GraphSAGE, GAT, Hybrid GCN+Raw, RF+GCN embeddings)
"""

import json
from pathlib import Path
import sys
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.elliptic_loader import (
    ensure_elliptic_dataset,
    load_elliptic_graph,
    verify_elliptic_stats,
)
from src.features.feature_engineering import get_elliptic_feature_views
from src.models.node_gnns import train_and_eval_node_gnn
from src.models.tabular_baselines import run_tabular_baselines
from src.utils.metrics import format_results_table, save_results_artifacts


def _plot_benchmark(rows: List[Dict], output_path: Path) -> None:
    labels = [f"{r['model']} ({r['features']})" for r in rows]
    f1s = [r["illicit_f1"] for r in rows]
    aucs = [r["auc_pr"] for r in rows]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(x - w / 2, f1s, width=w, label="Illicit F1", color="#4c78a8")
    ax.bar(x + w / 2, aucs, width=w, label="AUC-PR", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Elliptic Temporal Split: Tabular Baselines vs Node GNNs (Semester 7)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    seed = int(cfg["seed"])
    ell_cfg = cfg["elliptic"]
    tr_cfg = cfg["training"]

    out_dir = PROJECT_ROOT / "outputs" / "elliptic"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset check / demo fallback
    data_dir, is_demo = ensure_elliptic_dataset(
        PROJECT_ROOT / ell_cfg["raw_dir"],
        PROJECT_ROOT / ell_cfg["demo_dir"],
        seed=seed,
    )
    source_tag = "DEMO synthetic Elliptic CSVs" if is_demo else "REAL Elliptic Kaggle CSVs"
    print(f"\n[Elliptic] Loading dataset from {data_dir} ({source_tag})")

    # 2. Construct PyG graph & verify statistics
    data = load_elliptic_graph(data_dir, train_max_timestep=int(ell_cfg["train_max_timestep"]))
    stats = verify_elliptic_stats(data)
    print("[Elliptic] Graph statistics:", json.dumps(stats, indent=2))
    (out_dir / "graph_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    proc_path = PROJECT_ROOT / ell_cfg["processed_path"]
    proc_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, proc_path)

    # 3. Feature-group study & Tabular Baselines
    feature_views = get_elliptic_feature_views(data, local_feature_count=int(ell_cfg["local_feature_count"]))
    y_train = data.y[data.train_mask].numpy()
    y_test = data.y[data.test_mask].numpy()

    all_rows: List[Dict] = []
    for view_name in ("local_94", "all_166"):
        x_view = feature_views[view_name].numpy()
        tab_rows = run_tabular_baselines(
            x_view[data.train_mask.numpy()],
            y_train,
            x_view[data.test_mask.numpy()],
            y_test,
            feature_tag=view_name,
            seed=seed,
        )
        all_rows.extend(tab_rows)

    # 4. Node GNN Models (GCN, GraphSAGE, GAT, Hybrid GCN+Raw)
    gnn_embeddings = {}
    for arch in ("gcn", "graphsage", "gat", "hybrid_gcn"):
        metrics, _, emb = train_and_eval_node_gnn(
            arch=arch,
            data=data,
            hidden_channels=int(tr_cfg["hidden_dim"]),
            heads=int(tr_cfg["gat_heads"]),
            dropout=float(tr_cfg["dropout"]),
            epochs=int(tr_cfg["epochs"]),
            lr=float(tr_cfg["lr"]),
            weight_decay=float(tr_cfg["weight_decay"]),
            use_class_weights=bool(tr_cfg["use_class_weights"]),
            seed=seed,
        )
        gnn_embeddings[arch] = emb
        all_rows.append({"model": arch.upper(), "features": "all_166", **metrics})

    # 5. Weber et al. (2019) Hybrid Check: Random Forest on [Raw 166 || GCN Embeddings]
    x_raw_np = data.x.numpy()
    x_hybrid = np.hstack([x_raw_np, gnn_embeddings["gcn"]])
    rf_hybrid = run_tabular_baselines(
        x_hybrid[data.train_mask.numpy()],
        y_train,
        x_hybrid[data.test_mask.numpy()],
        y_test,
        feature_tag="raw_166+gcn_emb",
        seed=seed,
    )
    all_rows.extend([r for r in rf_hybrid if r["model"] == "RandomForest"])

    print("\n[Elliptic] Semester 7 Benchmark Results (Temporal Split):")
    print(format_results_table(all_rows))

    save_results_artifacts(all_rows, out_dir, "elliptic")
    _plot_benchmark(all_rows, out_dir / "elliptic_benchmark.png")
    print(f"[Elliptic] Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
