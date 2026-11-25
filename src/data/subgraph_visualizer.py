"""
Time-Windowed Subgraph Extraction & Ground-Truth Laundering Pattern Visualization
(Semester 7 - Task 4 & Table 4: 'create time-windowed subgraphs and visualise sample laundering patterns').

Note: This is exploratory graph visualization of known dataset windows/patterns in Sem 7,
NOT the Semester 8 automated suspicious-subgraph mining module.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def extract_time_window_subgraph(
    df: pd.DataFrame,
    window_start_sec: Optional[int] = None,
    window_hours: int = 24,
    max_edges: int = 300,
) -> nx.MultiDiGraph:
    """
    Extract a directed NetworkX subgraph from transactions falling inside
    [window_start_sec, window_start_sec + window_hours * 3600].
    If window_start_sec is None, centers on the first laundering transaction window.
    """
    if window_start_sec is None:
        illicit_rows = df[df["label"] == 1]
        if len(illicit_rows) > 0:
            center_ts = int(illicit_rows["timestamp_sec"].iloc[0])
            window_start_sec = center_ts - (window_hours * 3600) // 4
        else:
            window_start_sec = int(df["timestamp_sec"].min())

    window_end_sec = window_start_sec + window_hours * 3600
    sub_df = df[(df["timestamp_sec"] >= window_start_sec) & (df["timestamp_sec"] <= window_end_sec)]
    if len(sub_df) > max_edges:
        # Keep all illicit edges in the window plus a sample of licit context edges
        illicit_sub = sub_df[sub_df["label"] == 1]
        licit_sub = sub_df[sub_df["label"] == 0].head(max(0, max_edges - len(illicit_sub)))
        sub_df = pd.concat([illicit_sub, licit_sub], ignore_index=True)

    G = nx.MultiDiGraph()
    for _, row in sub_df.iterrows():
        G.add_edge(
            row["src_account"],
            row["dst_account"],
            amount_usd=float(row["amount_usd"]),
            label=int(row["label"]),
            payment_format=str(row["Payment Format"]),
        )
    return G


def visualize_time_window_subgraph(
    G: nx.MultiDiGraph,
    output_path: Union[str, Path],
    title: str = "IBM AML Time-Windowed Transaction Subgraph",
) -> Path:
    """Render and save a time-windowed subgraph highlighting illicit vs licit transactions."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No transactions in selected time window", ha="center", va="center")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

    pos = nx.spring_layout(G, seed=42, k=0.45)

    licit_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("label", 0) == 0]
    illicit_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("label", 0) == 1]

    illicit_nodes = {u for u, v in illicit_edges} | {v for u, v in illicit_edges}
    node_colors = ["#d62728" if n in illicit_nodes else "#4c78a8" for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=75, node_color=node_colors, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edgelist=licit_edges, edge_color="#b0b0b0", alpha=0.45, arrows=True, arrowsize=8, ax=ax
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=illicit_edges, edge_color="#d62728", width=2.0, alpha=0.9, arrows=True, arrowsize=12, ax=ax
    )

    ax.set_title(f"{title}\n(Nodes: {G.number_of_nodes()}, Licit edges: {len(licit_edges)}, Illicit edges: {len(illicit_edges)})")
    ax.axis("off")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def visualize_laundering_patterns(
    patterns: List[Dict],
    output_path: Union[str, Path],
    max_panels: int = 4,
) -> Optional[Path]:
    """
    Visualize up to `max_panels` distinct laundering pattern types (e.g., FAN-OUT, FAN-IN,
    CYCLE, SCATTER-GATHER) parsed from the IBM AML Patterns file.
    """
    if not patterns:
        return None

    # Pick one representative pattern per distinct typology
    selected: List[Dict] = []
    seen_types = set()
    for pat in patterns:
        t = pat["typology"]
        if t not in seen_types:
            seen_types.add(t)
            selected.append(pat)
        if len(selected) >= max_panels:
            break

    n_panels = len(selected)
    cols = min(2, n_panels)
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.8 * rows))
    if n_panels == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes.flat)

    for idx, pat in enumerate(selected):
        ax = axes_list[idx]
        G = nx.DiGraph()
        for e in pat["edges"]:
            G.add_edge(e["src"], e["dst"], amount_usd=e["amount_usd"])
        pos = nx.kamada_kawai_layout(G) if G.number_of_nodes() > 2 else nx.spring_layout(G, seed=42)
        short_labels = {n: n.split("_")[-1][-4:] for n in G.nodes()}
        edge_labels = {(u, v): f"${d['amount_usd']:,.0f}" for u, v, d in G.edges(data=True)}

        nx.draw_networkx_nodes(G, pos, node_size=420, node_color="#e45756", alpha=0.92, ax=ax)
        nx.draw_networkx_labels(G, pos, labels=short_labels, font_size=8, font_color="white", ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color="#333333", width=1.6, arrows=True, arrowsize=14, ax=ax)
        if len(edge_labels) <= 12:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)
        ax.set_title(f"Pattern: {pat['typology']} ({G.number_of_edges()} txs)")
        ax.axis("off")

    for j in range(n_panels, len(axes_list)):
        axes_list[j].axis("off")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Sample IBM AML Ground-Truth Laundering Patterns", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
