from pathlib import Path

import joblib
import pandas as pd

from src.data_utils import FEATURES, TARGET, load_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_processed_dataset_schema():
    frame = load_dataset(ROOT / "data" / "processed" / "concrete_strength.csv")
    assert frame.shape == (1030, 9)
    assert frame.columns.tolist() == FEATURES + [TARGET]
    assert not frame.isna().any().any()


def test_saved_model_predicts_finite_value():
    model = joblib.load(ROOT / "models" / "best_model.joblib")
    row = pd.DataFrame(
        [[540.0, 0.0, 0.0, 162.0, 2.5, 1040.0, 676.0, 28.0]],
        columns=FEATURES,
    )
    prediction = float(model.predict(row)[0])
    assert 0.0 < prediction < 100.0

