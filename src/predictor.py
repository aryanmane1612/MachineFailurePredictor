
import json
from pathlib import Path

import joblib
import pandas as pd


# Find the main project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"

# Load the saved model and its required feature names
MODEL_PATH = MODEL_DIR / "machine_failure_svm.joblib"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"

model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH, "r", encoding="utf-8") as file:
    FEATURE_COLUMNS = json.load(file)


def predict_machine_failure(features):
    """
    Predict machine condition from the 48 extracted features.

    Parameters
    ----------
    features : dict
        Dictionary containing all 48 expected feature names and values.

    Returns
    -------
    dict
        Predicted label and condition.
    """

    missing_features = [
        name for name in FEATURE_COLUMNS if name not in features
    ]

    if missing_features:
        raise ValueError(
            f"Missing {len(missing_features)} features: "
            f"{missing_features}"
        )

    # Arrange features in exactly the order used during training
    input_data = pd.DataFrame(
        [[features[name] for name in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS
    )

    prediction = int(model.predict(input_data)[0])

    return {
        "label": prediction,
        "condition": "Damaged" if prediction == 1 else "Healthy"
    }
