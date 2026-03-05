from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot P3O violation terms from p3o_terms.csv")
    p.add_argument("--run_dir", type=str, required=True, help="Run directory containing p3o_terms.csv")
    p.add_argument("--csv", type=str, default=None, help="Optional explicit CSV path")
    p.add_argument("--out", type=str, default=None, help="Output png path (default: <run_dir>/p3o_violation.png)")
    return p.parse_args()


def load_rows(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row: Dict[str, float] = {}
            for k, v in r.items():
                try:
                    row[k] = float(v)
                except Exception:
                    continue
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    csv_path = Path(args.csv) if args.csv else (run_dir / "p3o_terms.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = load_rows(csv_path)
    if not rows:
        raise RuntimeError(f"CSV has no rows: {csv_path}")

    iters = [int(r.get("iteration", i)) for i, r in enumerate(rows)]
    penalty = [r.get("penalty_before_kappa", r.get("p3o_penalty", 0.0)) for r in rows]

    violation_keys = sorted({k for r in rows for k in r.keys() if k.startswith("violation_")})

    plt.figure(figsize=(10, 6))
    plt.plot(iters, penalty, label="sum_i ReLU(Lc_total_i)", linewidth=2.0)
    for k in violation_keys:
        plt.plot(iters, [r.get(k, 0.0) for r in rows], label=k)

    plt.title("P3O Constraint Violation (Before kappa)")
    plt.xlabel("Iteration")
    plt.ylabel("Violation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = Path(args.out) if args.out else (run_dir / "p3o_violation.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
