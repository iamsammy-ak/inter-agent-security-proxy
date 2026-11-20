"""Run the benchmarks and write benchmarks/results/{injection,pii,latency}.{json,md}.

usage: python benchmarks/run_benchmarks.py [--suite all|injection|pii|latency]

The injection suite needs the dataset (python benchmarks/download_data.py). The bge-small rows need fastembed and
the BAAI/bge-small-en-v1.5 model (FASTEMBED_CACHE_PATH is honoured); without them only the hashing rows appear.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_proxy.evaluation import run_suites  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=["all", "injection", "pii", "latency"], default="all")
    args = parser.parse_args()
    command = "python benchmarks/run_benchmarks.py" + (f" --suite {args.suite}" if args.suite != "all" else "")
    run_suites(args.suite, ROOT / "benchmarks" / "results", command)


if __name__ == "__main__":
    main()
