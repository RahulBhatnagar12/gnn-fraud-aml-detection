# Graph Neural Networks for Fraud and Money-Laundering Detection — Semester 7 (Sem 1)

**Harcourt Butler Technical University (Session 2026–2027)**  
**Team:** Rahul Bhatnagar (230104045), Saket Pandey (230104052), Pranjal Shukla (230104039)  
**Guide:** Dr. Prabhat Verma  

---

## Scope: Semester 7 Progress (Up to, Excluding Semester 8 Subgraph Mining)

This repository contains the **Semester 7 (Phase 1)** implementation matching the mid-semester / end-semester plan:

1. **Dataset Acquisition & Preprocessing (`src/data/`)**
   - **Elliptic Bitcoin Dataset:** Aligns `txId`, maps to contiguous indices, encodes labels (`illicit=1`, `licit=0`, `unknown=-1`), preserves `timestep` for temporal splits (`1–34` train, `35–49` test).
   - **IBM AML Synthetic Dataset (`HI-Small` / `LI-Small`):** Combines `(Bank, Account)` into unique account identifiers, parses & sorts timestamps chronologically, checks/removes duplicates, converts multi-currency amounts to a common USD scale, and parses `*_Patterns.txt`.
   - Includes deterministic **demo dataset generators** so the entire pipeline runs immediately even before placing large Kaggle CSVs in `data/raw/`.

2. **Transaction Graph Construction & Pattern Visualization (`src/data/`)**
   - **Elliptic:** Directed node-classification graph (`Data` with `x`, `edge_index`, `y`, `timestep`, `train_mask`, `test_mask`). Unlabelled (`-1`) nodes stay in `edge_index` for message passing and are masked out of loss/evaluation.
   - **IBM AML:** Directed edge-classification graph where accounts are nodes and transactions are attributed edges (`edge_index`, `edge_attr`, `edge_label`, `edge_timestamp`).
   - **Time-Windowed Subgraphs:** Extracts time-windowed subgraphs and visualizes sample laundering patterns (Fan-In, Fan-Out, Cycle, Scatter-Gather) using NetworkX & Matplotlib.

3. **Node & Edge Feature Engineering (`src/features/`)**
   - **Elliptic Feature Groups:** Separates local (`94`) vs aggregated (`72`) features for ablation comparison.
   - **IBM AML Node Features:** In-degree, out-degree, in/out amount statistics (sum, mean, std, max), net flow, and activity burstiness (inter-transaction CV burstiness + peak hourly tx count).
   - **IBM AML Edge Features:** Common-scale USD amount (`log1p` & scaled), time since sender/receiver's previous transaction, payment format, and original currency.

4. **Tabular Baselines & GNN Models (`src/models/`)**
   - **Tabular Baselines:** Logistic Regression & Random Forest (Weber et al., 2019 comparison).
   - **Elliptic Node Models:** `GCN`, `GraphSAGE`, `GAT`, and `GCN + Raw Features` hybrid.
   - **IBM AML Edge Models:** `GraphSAGE` / `GCN` with concatenated endpoint embeddings + edge features, and `GAT` with native `edge_dim` attention, trained with class-weighted loss & minority oversampling.
   - **Class-Imbalance Evaluation (`src/utils/metrics.py`):** Illicit-class Precision, Recall, F1, and AUC-PR (instead of plain accuracy).

> **Note on Semester 8 Scope (Not Included Here):**  
> Suspicious-subgraph mining algorithms, alert ranking, GNN explainability, and the analyst dashboard are scheduled for **Semester 8** and are intentionally excluded from this Semester 7 codebase.

---

## Quickstart

```bash
# Activate environment (or use your existing Python 3.9+ venv with PyTorch & PyG)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run full Semester 1 pipeline (uses real CSVs in data/raw/ if available, or auto-generates demo data)
python3 scripts/run_all_sem1.py

# Or run datasets individually:
python3 scripts/run_elliptic_pipeline.py
python3 scripts/run_ibm_aml_pipeline.py
```

### Using Real Datasets
- **Elliptic:** Place `elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv` into `data/raw/elliptic/`.
- **IBM AML:** Place `HI-Small_Trans.csv` (and `HI-Small_Patterns.txt`, or `LI-Small_*`) into `data/raw/ibm_aml/`.
