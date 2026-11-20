"""Download and load the deepset/prompt-injections dataset (Apache-2.0).

The dataset is fetched page by page from the Hugging Face datasets-server HTTP API and cached as JSON Lines, so
no parquet tooling is needed. Only benchmark and training scripts call :func:`download_deepset`; tests never touch
the network.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel

DATASET_ID = "deepset/prompt-injections"
DATASET_URL = "https://huggingface.co/datasets/deepset/prompt-injections"
ROWS_API = "https://datasets-server.huggingface.co/rows"
SPLITS = ("train", "test")
PAGE = 100


class LabelledText(BaseModel):
    """One labelled example: ``label`` is 1 for a prompt injection and 0 for benign text."""

    text: str
    label: int


def default_data_dir() -> Path:
    """Directory for downloaded datasets (``AGENT_PROXY_DATA_DIR`` or ``.cache/datasets``)."""
    return Path(os.environ.get("AGENT_PROXY_DATA_DIR", Path(".cache") / "datasets"))


def split_path(split: str, data_dir: Path | None = None) -> Path:
    """Path of the cached JSON Lines file for one split."""
    return (data_dir or default_data_dir()) / "deepset-prompt-injections" / f"{split}.jsonl"


def download_deepset(data_dir: Path | None = None, *, timeout: float = 30.0) -> dict[str, Path]:
    """Fetch the train and test splits and write them as JSON Lines. Returns the file per split."""
    import httpx

    written: dict[str, Path] = {}
    with httpx.Client(timeout=timeout, headers={"User-Agent": "agent-proxy-benchmarks"}) as client:
        for split in SPLITS:
            rows: list[dict[str, object]] = []
            offset, total = 0, None
            while total is None or offset < total:
                params: dict[str, str | int] = {
                    "dataset": DATASET_ID,
                    "config": "default",
                    "split": split,
                    "offset": offset,
                    "length": PAGE,
                }
                response = client.get(ROWS_API, params=params)
                response.raise_for_status()
                payload = response.json()
                total = int(payload["num_rows_total"])
                page = [item["row"] for item in payload["rows"]]
                if not page:
                    break
                rows.extend({"text": str(r["text"]), "label": int(r["label"])} for r in page)
                offset += len(page)
            if total is not None and len(rows) != total:
                raise RuntimeError(f"{split}: expected {total} rows, received {len(rows)}")
            path = split_path(split, data_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            written[split] = path
    return written


def load_split(split: str, data_dir: Path | None = None) -> list[LabelledText]:
    """Load a cached split; raises ``FileNotFoundError`` with the download command when missing."""
    path = split_path(split, data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the dataset first: python benchmarks/download_data.py"
        )
    lines = path.read_text(encoding="utf-8").splitlines()
    return [LabelledText.model_validate_json(line) for line in lines if line.strip()]
