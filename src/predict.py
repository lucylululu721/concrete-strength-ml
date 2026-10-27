from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .data_utils import FEATURES


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict concrete compressive strength")
    parser.add_argument("--input", required=True, help="JSON file containing the eight inputs")
    parser.add_argument(
        "--model", default="models/best_model.joblib", help="trained joblib model"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    input_path = Path(args.input)
    model_path = Path(args.model)
    if not input_path.is_absolute():
        input_path = root / input_path
    if not model_path.is_absolute():
        model_path = root / model_path

    values = json.loads(input_path.read_text(encoding="utf-8"))
    missing = [name for name in FEATURES if name not in values]
    extra = [name for name in values if name not in FEATURES]
    if missing or extra:
        raise ValueError(f"Input keys mismatch. Missing={missing}; unexpected={extra}")
    row = pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)
    model = joblib.load(model_path)
    prediction = float(model.predict(row)[0])
    print(json.dumps({"predicted_compressive_strength_mpa": prediction}, indent=2))


if __name__ == "__main__":
    main()

