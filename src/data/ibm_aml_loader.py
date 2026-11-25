"""
IBM AML Synthetic Dataset Loader & Preprocessor (Semester 7 - Tasks 2, 3, 4).

Handles:
- Loading HI-Small_Trans.csv / LI-Small_Trans.csv and optional *_Patterns.txt
- Combining (Bank, Account) into unique account identifiers
- Parsing timestamps and sorting records chronologically
- Checking/removing duplicates and invalid records
- Converting multi-currency amounts to a common USD reference scale
- Building a PyG Data object where accounts = nodes and transactions = directed attributed edges
- Chronological train/test edge split (no random split leakage)
"""

from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# Static reference exchange rates to USD for common IBM AML currencies
CURRENCY_TO_USD: Dict[str, float] = {
    "US Dollar": 1.0,
    "USD": 1.0,
    "Euro": 1.08,
    "EUR": 1.08,
    "UK Pound": 1.27,
    "GBP": 1.27,
    "Swiss Franc": 1.13,
    "CHF": 1.13,
    "Canadian Dollar": 0.73,
    "CAD": 0.73,
    "Australian Dollar": 0.65,
    "AUD": 0.65,
    "Yen": 0.0067,
    "JPY": 0.0067,
    "Yuan": 0.14,
    "CNY": 0.14,
    "Rupee": 0.012,
    "INR": 0.012,
    "Bitcoin": 65000.0,
    "BTC": 65000.0,
    "Mexican Peso": 0.055,
    "Brazilian Real": 0.19,
    "Saudi Riyal": 0.27,
    "Shekel": 0.27,
    "Ruble": 0.011,
}

PAYMENT_FORMATS = ("Wire", "ACH", "Cheque", "Credit Card", "Reinvestment", "Cash", "Bitcoin")


def generate_demo_ibm_aml_files(
    output_dir: Union[str, Path],
    tx_filename: str = "HI-Small_Trans.csv",
    patterns_filename: str = "HI-Small_Patterns.txt",
    num_accounts: int = 800,
    num_normal_txs: int = 12000,
    seed: int = 42,
) -> Tuple[Path, Path]:
    """
    Generate a realistic IBM-AML-schema transaction CSV and Patterns.txt file
    containing multi-currency accounts plus embedded laundering patterns
    (FAN-OUT, FAN-IN, CYCLE, SCATTER-GATHER, BIPARTITE).
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    banks = ["0010", "0020", "0030", "0040", "0050"]
    currencies = ["US Dollar", "Euro", "UK Pound", "Rupee", "Swiss Franc", "Yen"]
    formats = ["ACH", "Wire", "Credit Card", "Cheque", "Cash"]

    # Create bank_account pool (notice account IDs can collide across banks, testing Bank+Account uniqueness)
    accounts = [
        (banks[i % len(banks)], f"800{1000 + (i // len(banks)):04d}")
        for i in range(num_accounts)
    ]

    base_ts = pd.Timestamp("2022-09-01 00:00:00")
    records = []

    # 1. Background legitimate transactions over 14 days
    for _ in range(num_normal_txs):
        src_idx, dst_idx = rng.choice(num_accounts, size=2, replace=False)
        from_bank, from_acc = accounts[src_idx]
        to_bank, to_acc = accounts[dst_idx]
        curr = currencies[int(rng.integers(0, len(currencies)))]
        rate = CURRENCY_TO_USD[curr]
        amt_usd = float(np.exp(rng.normal(4.8, 1.1)))
        amt_paid = round(amt_usd / rate, 2)
        ts = base_ts + pd.Timedelta(seconds=int(rng.integers(0, 14 * 86400)))
        fmt = formats[int(rng.integers(0, len(formats)))]
        records.append({
            "Timestamp": ts.strftime("%Y/%m/%d %H:%M"),
            "From Bank": from_bank,
            "From Account": from_acc,
            "To Bank": to_bank,
            "To Account": to_acc,
            "Amount Received": amt_paid,
            "Receiving Currency": curr,
            "Amount Paid": amt_paid,
            "Payment Currency": curr,
            "Payment Format": fmt,
            "Is Laundering": 0,
        })

    # Helper to append a laundering pattern and collect pattern lines
    pattern_blocks: List[str] = []

    def add_pattern_tx(src_i: int, dst_i: int, ts_offset_sec: int, amt_usd: float, curr: str, fmt: str) -> Dict:
        fb, fa = accounts[src_i]
        tb, ta = accounts[dst_i]
        rate = CURRENCY_TO_USD[curr]
        amt = round(amt_usd / rate, 2)
        ts = (base_ts + pd.Timedelta(seconds=ts_offset_sec)).strftime("%Y/%m/%d %H:%M")
        row = {
            "Timestamp": ts,
            "From Bank": fb,
            "From Account": fa,
            "To Bank": tb,
            "To Account": ta,
            "Amount Received": amt,
            "Receiving Currency": curr,
            "Amount Paid": amt,
            "Payment Currency": curr,
            "Payment Format": fmt,
            "Is Laundering": 1,
        }
        records.append(row)
        return row

    # Inject multiple instances of each typology across both train (days 1-9) and test (days 10-14)
    cursor = 0
    for day_offset_days in (2, 5, 8, 11, 13):
        t0 = day_offset_days * 86400

        # FAN-OUT: 1 source -> 6 mules
        src = cursor % num_accounts
        mules = [(cursor + k) % num_accounts for k in range(1, 7)]
        cursor += 7
        blk = ["BEGIN LAUNDERING ATTEMPT - FAN-OUT"]
        for k, m in enumerate(mules):
            r = add_pattern_tx(src, m, t0 + k * 600, float(rng.uniform(4200, 4950)), "US Dollar", "ACH")
            blk.append(",".join(str(r[c]) for c in r))
        blk.append("END LAUNDERING ATTEMPT - FAN-OUT")
        pattern_blocks.append("\n".join(blk))

        # FAN-IN: 6 smurfs -> 1 collector
        collector = cursor % num_accounts
        smurfs = [(cursor + k) % num_accounts for k in range(1, 7)]
        cursor += 7
        blk = ["BEGIN LAUNDERING ATTEMPT - FAN-IN"]
        for k, s in enumerate(smurfs):
            r = add_pattern_tx(s, collector, t0 + 3600 + k * 420, float(rng.uniform(8800, 9850)), "Euro", "Wire")
            blk.append(",".join(str(r[c]) for c in r))
        blk.append("END LAUNDERING ATTEMPT - FAN-IN")
        pattern_blocks.append("\n".join(blk))

        # CYCLE: 4-node directed ring
        ring = [(cursor + k) % num_accounts for k in range(4)]
        cursor += 4
        blk = ["BEGIN LAUNDERING ATTEMPT - CYCLE"]
        for k in range(4):
            r = add_pattern_tx(ring[k], ring[(k + 1) % 4], t0 + 7200 + k * 900, float(rng.uniform(18000, 22000)), "US Dollar", "Wire")
            blk.append(",".join(str(r[c]) for c in r))
        blk.append("END LAUNDERING ATTEMPT - CYCLE")
        pattern_blocks.append("\n".join(blk))

        # SCATTER-GATHER: src -> 4 intermediaries -> sink
        sg_src = cursor % num_accounts
        sg_mid = [(cursor + k) % num_accounts for k in range(1, 5)]
        sg_sink = (cursor + 5) % num_accounts
        cursor += 6
        blk = ["BEGIN LAUNDERING ATTEMPT - SCATTER-GATHER"]
        for k, mid in enumerate(sg_mid):
            r1 = add_pattern_tx(sg_src, mid, t0 + 14400 + k * 300, float(rng.uniform(6000, 7500)), "UK Pound", "ACH")
            r2 = add_pattern_tx(mid, sg_sink, t0 + 18000 + k * 300, float(rng.uniform(5800, 7300)), "UK Pound", "Wire")
            blk.append(",".join(str(r1[c]) for c in r1))
            blk.append(",".join(str(r2[c]) for c in r2))
        blk.append("END LAUNDERING ATTEMPT - SCATTER-GATHER")
        pattern_blocks.append("\n".join(blk))

    df = pd.DataFrame(records)
    # Write with original IBM AML duplicate 'Account' header style for full compatibility
    df_out = df.rename(columns={"From Account": "Account", "To Account": "Account.1"})
    # Deliberately inject a few duplicates to exercise duplicate-removal cleaning
    df_out = pd.concat([df_out, df_out.iloc[:5]], ignore_index=True)

    tx_path = out_dir / tx_filename
    # Restore exact header line "Account" twice if desired, or standard pandas CSV
    df_out.to_csv(tx_path, index=False)

    pat_path = out_dir / patterns_filename
    pat_path.write_text("\n\n".join(pattern_blocks) + "\n", encoding="utf-8")
    return tx_path, pat_path


def ensure_ibm_aml_dataset(
    raw_dir: Union[str, Path],
    demo_dir: Union[str, Path],
    transactions_file: str = "HI-Small_Trans.csv",
    patterns_file: str = "HI-Small_Patterns.txt",
    seed: int = 42,
) -> Tuple[Path, Optional[Path], bool]:
    """Return (tx_csv_path, patterns_path_or_None, is_demo)."""
    raw_tx = Path(raw_dir) / transactions_file
    raw_pat = Path(raw_dir) / patterns_file
    if raw_tx.exists():
        return raw_tx, (raw_pat if raw_pat.exists() else None), False

    demo_tx = Path(demo_dir) / transactions_file
    demo_pat = Path(demo_dir) / patterns_file
    if not demo_tx.exists():
        generate_demo_ibm_aml_files(
            demo_dir, tx_filename=transactions_file, patterns_filename=patterns_file, seed=seed
        )
    return demo_tx, (demo_pat if demo_pat.exists() else None), True


def _standardize_ibm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize IBM AML column names across raw duplicate-'Account' CSVs and renamed variants."""
    cols = list(df.columns)
    rename_map = {}
    if "Account" in cols and "Account.1" in cols:
        rename_map["Account"] = "From Account"
        rename_map["Account.1"] = "To Account"
    df = df.rename(columns=rename_map)
    required = {
        "Timestamp",
        "From Bank",
        "From Account",
        "To Bank",
        "To Account",
        "Amount Paid",
        "Payment Currency",
        "Payment Format",
        "Is Laundering",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"IBM AML CSV missing expected columns: {sorted(missing)}")
    return df


def clean_ibm_aml_dataframe(
    csv_path: Union[str, Path],
    max_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Clean & preprocess IBM AML transactions (Section 5.3):
    1. Combine (From Bank, From Account) and (To Bank, To Account) into unique account IDs
    2. Parse timestamps and sort chronologically
    3. Check and remove duplicate / invalid (negative amount, self-loop/null) records
    4. Convert multi-currency amounts to a common USD scale
    """
    df = pd.read_csv(csv_path, nrows=max_rows, dtype={"From Bank": str, "To Bank": str})
    df = _standardize_ibm_columns(df)
    initial_rows = len(df)

    # Combine bank and account numbers into unique identifiers
    df["src_account"] = df["From Bank"].astype(str).str.strip() + "_" + df["From Account"].astype(str).str.strip()
    df["dst_account"] = df["To Bank"].astype(str).str.strip() + "_" + df["To Account"].astype(str).str.strip()

    # Parse timestamps and drop invalid rows
    df["parsed_ts"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["parsed_ts", "src_account", "dst_account", "Amount Paid"])

    # Drop non-positive amounts
    df["Amount Paid"] = pd.to_numeric(df["Amount Paid"], errors="coerce")
    df = df[df["Amount Paid"] > 0].copy()

    # Check & remove exact duplicate transaction records
    before_dedup = len(df)
    df = df.drop_duplicates(
        subset=["parsed_ts", "src_account", "dst_account", "Amount Paid", "Payment Currency", "Payment Format"]
    ).copy()
    duplicates_removed = before_dedup - len(df)

    # Convert amounts to common USD scale
    df["fx_rate"] = df["Payment Currency"].astype(str).str.strip().map(CURRENCY_TO_USD).fillna(1.0)
    df["amount_usd"] = (df["Amount Paid"] * df["fx_rate"]).astype(np.float64)

    # Chronological sort (mandatory for leakage-free temporal split and time-since-prev-tx features)
    df = df.sort_values("parsed_ts", kind="mergesort").reset_index(drop=True)
    df["timestamp_sec"] = (df["parsed_ts"].astype("int64") // 10**9).astype(np.int64)
    df["label"] = df["Is Laundering"].astype(np.int64)

    cleaning_report = {
        "initial_rows": initial_rows,
        "duplicates_removed": duplicates_removed,
        "final_clean_rows": len(df),
        "unique_accounts": int(pd.Index(pd.concat([df["src_account"], df["dst_account"]])).nunique()),
        "laundering_txs": int(df["label"].sum()),
    }
    return df, cleaning_report


def parse_ibm_patterns_file(patterns_path: Optional[Union[str, Path]]) -> List[Dict]:
    """
    Parse IBM AML *_Patterns.txt into structured laundering scheme records
    for Sem 7 pattern visualization and Sem 8 ground-truth evaluation.
    """
    if patterns_path is None or not Path(patterns_path).exists():
        return []

    patterns: List[Dict] = []
    current_type: Optional[str] = None
    current_edges: List[Dict] = []

    begin_re = re.compile(r"BEGIN LAUNDERING ATTEMPT\s*-\s*(.+)", re.IGNORECASE)
    end_re = re.compile(r"END LAUNDERING ATTEMPT", re.IGNORECASE)

    with open(patterns_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m_begin = begin_re.search(line)
            if m_begin:
                current_type = m_begin.group(1).strip()
                current_edges = []
                continue
            if end_re.search(line):
                if current_type and current_edges:
                    patterns.append({"typology": current_type, "edges": current_edges})
                current_type = None
                current_edges = []
                continue
            if current_type is not None:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 10:
                    src = f"{parts[1]}_{parts[2]}"
                    dst = f"{parts[3]}_{parts[4]}"
                    try:
                        amt = float(parts[7])
                    except ValueError:
                        amt = 0.0
                    curr = parts[8]
                    amt_usd = amt * CURRENCY_TO_USD.get(curr, 1.0)
                    current_edges.append({
                        "timestamp": parts[0],
                        "src": src,
                        "dst": dst,
                        "amount_usd": amt_usd,
                        "currency": curr,
                        "format": parts[9],
                    })
    return patterns


def load_ibm_aml_graph(
    df: pd.DataFrame,
    node_features: np.ndarray,
    edge_features: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
) -> Tuple[Data, Dict[str, int]]:
    """
    Construct a directed, attributed PyG Data object for IBM AML edge classification:
    - Nodes = unique (Bank, Account) identifiers
    - Edges = chronological transactions with edge_attr and edge_label
    - Chronological split: earliest train_ratio for train, next val_ratio for val, rest for test.
    """
    unique_accounts = pd.Index(pd.concat([df["src_account"], df["dst_account"]], ignore_index=True).unique())
    account_to_idx: Dict[str, int] = {acc: idx for idx, acc in enumerate(unique_accounts)}

    src_idx = df["src_account"].map(account_to_idx).to_numpy(dtype=np.int64)
    dst_idx = df["dst_account"].map(account_to_idx).to_numpy(dtype=np.int64)

    edge_index = torch.tensor(np.vstack([src_idx, dst_idx]), dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    edge_label = torch.tensor(df["label"].to_numpy(dtype=np.int64), dtype=torch.long)
    edge_timestamp = torch.tensor(df["timestamp_sec"].to_numpy(dtype=np.int64), dtype=torch.long)
    x = torch.tensor(node_features, dtype=torch.float32)

    n_edges = len(df)
    train_end = int(n_edges * train_ratio)
    val_end = int(n_edges * (train_ratio + val_ratio))

    train_mask = torch.zeros(n_edges, dtype=torch.bool)
    val_mask = torch.zeros(n_edges, dtype=torch.bool)
    test_mask = torch.zeros(n_edges, dtype=torch.bool)
    train_mask[:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_label=edge_label,
        edge_timestamp=edge_timestamp,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=len(unique_accounts),
    )
    return data, account_to_idx


def verify_ibm_aml_stats(data: Data) -> Dict[str, Union[int, float]]:
    """Verify IBM AML graph statistics and chronological split class ratios."""
    y = data.edge_label.cpu().numpy()
    num_edges = int(data.num_edges)
    illicit = int((y == 1).sum())
    return {
        "num_accounts_nodes": int(data.num_nodes),
        "num_transaction_edges": num_edges,
        "node_feature_dim": int(data.x.size(1)),
        "edge_feature_dim": int(data.edge_attr.size(1)),
        "illicit_edges": illicit,
        "licit_edges": int((y == 0).sum()),
        "illicit_pct": round(100.0 * illicit / max(num_edges, 1), 4),
        "train_edges": int(data.train_mask.sum().item()),
        "val_edges": int(data.val_mask.sum().item()),
        "test_edges": int(data.test_mask.sum().item()),
    }
