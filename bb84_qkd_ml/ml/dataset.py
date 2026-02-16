"""Dataset writer for BB84 QKD ML pipeline."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FILE_PATH = DATA_DIR / "dataset.csv"
HEADER = ["QBER", "BasisMatchRate", "KeyLength", "Label"]


def init_csv() -> None:
    """Initialize dataset file with header if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not FILE_PATH.exists() or FILE_PATH.stat().st_size == 0:
        with FILE_PATH.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)


def write_row(features: Sequence[float], label: int) -> None:
    """Append a row of features and label to the dataset."""
    with FILE_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(features) + [label])
