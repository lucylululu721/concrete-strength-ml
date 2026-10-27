from __future__ import annotations

from pathlib import Path

import pandas as pd


FEATURES = [
    "cement",
    "blast_furnace_slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
]
TARGET = "compressive_strength"


def clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the UCI dataset with concise, stable column names."""
    if frame.shape[1] != 9:
        raise ValueError(f"Expected 9 columns, received {frame.shape[1]}")
    cleaned = frame.copy()
    cleaned.columns = FEATURES + [TARGET]
    cleaned = cleaned.apply(pd.to_numeric, errors="raise")
    if cleaned.isna().any().any():
        raise ValueError("Unexpected missing values in the UCI dataset")
    return cleaned


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run `python -m src.download_data` first."
        )
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        expected = set(FEATURES + [TARGET])
        if set(frame.columns) != expected:
            raise ValueError("Processed CSV does not contain the expected columns")
        return frame[FEATURES + [TARGET]]
    return clean_columns(pd.read_excel(path))

