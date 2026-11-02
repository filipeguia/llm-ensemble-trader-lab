"""Generate the small synthetic sample CSV that ships with the repository

The bars come from a geometric Brownian motion with regime switches (see
src/llm_ensemble_lab/synthetic.py). They are fake: there is no real market
or instrument behind them

Usage from the repository root:
    python data/generate_sample.py
    python data/generate_sample.py --bars 2000 --seed 11 --out data/other.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_ensemble_lab.bars import write_bars_csv  # noqa: E402
from llm_ensemble_lab.synthetic import generate_bars  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bars", type=int, default=720)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=str(ROOT / "data" / "sample_synthetic_bars.csv"))
    args = parser.parse_args()
    bars = generate_bars(args.bars, seed=args.seed)
    write_bars_csv(args.out, bars)
    print(f"wrote {len(bars)} synthetic bars to {args.out}")


if __name__ == "__main__":
    main()
