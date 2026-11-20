"""Download the deepset/prompt-injections train and test splits (Apache-2.0) into the local cache.

usage: python benchmarks/download_data.py [--data-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_proxy.datasets import DATASET_URL, download_deepset, load_split  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()
    for split, path in download_deepset(args.data_dir).items():
        rows = load_split(split, args.data_dir)
        positives = sum(r.label for r in rows)
        print(f"{split}: {len(rows)} rows ({positives} injections) -> {path.name}")
    print(f"source: {DATASET_URL} (Apache-2.0)")


if __name__ == "__main__":
    main()
