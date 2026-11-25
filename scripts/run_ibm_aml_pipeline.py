"""
Semester 7 Pipeline 2: IBM AML Edge Classification & Exploratory Pattern Visualization
Covers Tasks 2-7 from the Semester 7 plan:
- Cleaning: (Bank, Account) concatenation, chronological sort, duplicate removal, currency->USD conversion
- Exploratory time-windowed subgraph extraction & ground-truth laundering pattern visualization
- Vectorized node-level (degree, amount stats, burstiness) & causal edge-level feature engineering
- Feature-group ablation study
- Tabular baselines (Logistic Regression, Random Forest) + Edge GNNs (GCN, GraphSAGE, GAT w/ edge_dim)
  under class-weighted loss & minority oversampling
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

from src.data.ibm_aml_loader import (
    clean_ibm_aml_dataframe,
    ensure_ibm_aml_dataset,
    load_ibm_aml_graph,
    parse_ibm_patterns_file,
    verify_ibm_aml_stats,
)
from src.data.subgraph_visualizer import (
    extract_time_window_subgraph,
    visualize_laundering_patterns,
    visualize_time_window_subgraph,
)
from src.features.feature_engineering import (
    compute_ibm_aml_features,
    get_ibm_node_feature_group_indices,
)
from src.models.edge_gnns import train_and_eval_edge_gnn
from src.models.tabular_baselines import (
    build_ibm_edge_tabular_matrix,
    run_tabular_baselines,
)
from src.utils.metrics import format_results_table, save_results_artifacts


def _plot_benchmark(rows: List[Dict], output_path: Path) -> None:
    labels = [f"{r['model']} ({r['features']})" for r in rows]
    f1s = [r["illicit_f1"] for r in rows]
    aucs = [r["auc_pr"] for r in rows]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(x - w / 2, f1s, width=w, label="Illicit F1", color="#54a24b")
    ax.bar(x + w / 2, aucs, width=w, label="AUC-PR", color="#e45756")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("IBM AML Chronological Split: Baselines vs Edge GNNs (Semester 7)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    seed = int(cfg["seed"])
    aml_cfg = cfg["ibm_aml"]
    tr_cfg = cfg["training"]

    out_dir = PROJECT_ROOT / "outputs" / "ibm_aml"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ensure dataset (real HI-Small / LI-Small if placed in data/raw/ibm_aml, else demo)
    tx_path, pat_path, is_demo = ensure_ibm_aml_dataset(
        PROJECT_ROOT / aml_cfg["raw_dir"],
        PROJECT_ROOT / aml_cfg["demo_dir"],
        transactions_file=aml_cfg["transactions_file"],
        patterns_file=aml_cfg["patterns_file"],
        seed=seed,
    )
    source_tag = "DEMO synthetic IBM AML files" if is_demo else "REAL IBM AML dataset"
    print(f"\n[IBM AML] Loading transactions from {tx_path} ({source_tag})")

    # 2. Clean & preprocess transactions
    df, cleaning_report = clean_ibm_aml_dataframe(tx_path)
    print("[IBM AML] Cleaning report:", json.dumps(cleaning_report, indent=2))
    (out_dir / "cleaning_report.json").write_text(json.dumps(cleaning_report, indent=2), encoding="utf-8")

    # 3. Exploratory time-windowed subgraph & pattern visualization (Sem 7 Task 4)
    window_g = extract_time_window_subgraph(df, window_hours=int(aml_cfg["window_hours"]))
    visualize_time_window_subgraph(window_g, out_dir / "time_window_subgraph.png")
    patterns = parse_ibm_patterns_file(pat_path)
    if patterns:
        visualize_laundering_patterns(patterns, out_dir / "sample_laundering_patterns.png")
        print(f"[IBM AML] Parsed {len(patterns)} ground-truth laundering patterns & saved visualizations.")

    # 4. Compute node-level & edge-level features
    node_feats, edge_feats, node_names, edge_names = compute_ibm_aml_features(
        df,
        train_ratio=float(aml_cfg["train_ratio"]),
    )
    print(f"[IBM AML] Engineered features -> Node dim: {len(node_names)}, Edge dim: {len(edge_names)}")

    # 5. Build PyG directed attributed graph & verify stats
    data, _ = load_ibm_aml_graph(
        df,
        node_features=node_feats,
        edge_features=edge_feats,
        train_ratio=float(aml_cfg["train_ratio"]),
        val_ratio=float(aml_cfg["val_ratio"]),
    )
    stats = verify_ibm_aml_stats(data)
    print("[IBM AML] Graph statistics:", json.dumps(stats, indent=2))
    (out_dir / "graph_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    proc_path = PROJECT_ROOT / aml_cfg["processed_path"]
    proc_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, proc_path)

    # 6. Feature-group study (how much each node feature group adds) + Tabular Baselines
    train_idx = torch.where(data.train_mask)[0]
    test_idx = torch.where(data.test_mask)[0]
    y_train = data.edge_label[train_idx].numpy()
    y_test = data.edge_label[test_idx].numpy()

    node_groups = get_ibm_node_feature_group_indices(node_names)
    ablation_rows: List[Dict] = []
    for grp_name, cols in node_groups.items():
        X_tr = build_ibm_edge_tabular_matrix(data, train_idx, node_cols=cols)
        X_te = build_ibm_edge_tabular_matrix(data, test_idx, node_cols=cols)
        res = run_tabular_baselines(X_tr, y_train, X_te, y_test, feature_tag=f"node_{grp_name}+edge", seed=seed)
        if grp_name == "all":
            ablation_rows.extend(res)
        else:
            ablation_rows.extend([r for r in res if r["model"] == "RandomForest"])

    # 7. Edge GNN Models (GCN, GraphSAGE, GAT with edge_dim)
    gnn_rows: List[Dict] = []
    for arch in ("gcn", "graphsage", "gat"):
        metrics, _ = train_and_eval_edge_gnn(
            arch=arch,
            data=data,
            hidden_dim=int(tr_cfg["hidden_dim"]),
            heads=int(tr_cfg["gat_heads"]),
            dropout=float(tr_cfg["dropout"]),
            epochs=int(tr_cfg["epochs"]),
            lr=float(tr_cfg["lr"]),
            weight_decay=float(tr_cfg["weight_decay"]),
            use_class_weights=bool(tr_cfg["use_class_weights"]),
            use_oversampling=bool(tr_cfg["use_oversampling"]),
            oversample_ratio=float(tr_cfg["oversample_ratio"]),
            seed=seed,
        )
        gnn_rows.append({"model": f"{arch.upper()}_Edge", "features": "all_node+edge", **metrics})

    all_rows = ablation_rows + gnn_rows
    print("\n[IBM AML] Semester 7 Benchmark Results (Chronological Split):")
    print(format_results_table(all_rows))

    save_results_artifacts(all_rows, out_dir, "ibm_aml")
    _plot_benchmark(all_rows, out_dir / "ibm_aml_benchmark.png")
    print(f"[IBM AML] Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
