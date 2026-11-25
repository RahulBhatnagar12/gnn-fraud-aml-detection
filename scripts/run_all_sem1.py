"""
Run complete Semester 7 (Semester 1) pipeline for both Elliptic and IBM AML datasets.
Excludes Semester 8 components (suspicious-subgraph mining, alert ranking, explainability, dashboard).
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_elliptic_pipeline import main as run_elliptic
from scripts.run_ibm_aml_pipeline import main as run_ibm_aml


def main() -> None:
    print("=" * 72)
    print("GNN Fraud & Money-Laundering Detection — Semester 7 (Phase 1) Pipeline")
    print("=" * 72)
    run_elliptic()
    print("\n" + "-" * 72)
    run_ibm_aml()
    print("\n" + "=" * 72)
    print("Semester 7 pipeline completed. Outputs saved under outputs/elliptic and outputs/ibm_aml.")
    print("Semester 8 modules (subgraph mining, alert scoring, explainability, dashboard) remain separate.")


if __name__ == "__main__":
    main()
