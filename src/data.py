"""Load, validate, clean, document, and split Bank Marketing data."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    CLASS_NAMES,
    DROPPED_FEATURES,
    METRICS_DIR,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)


EXPECTED_COLUMNS = {
    "age", "job", "marital", "education", "default", "balance",
    "housing", "loan", "contact", "day", "month", "duration",
    "campaign", "pdays", "previous", "poutcome", "y",
}


def load_dataset() -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Load the full UCI Bank Marketing CSV and apply agreed cleaning rules."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}. Place bank-full.csv in data/raw/."
        )

    raw_frame = pd.read_csv(RAW_DATA_PATH, sep=";")
    missing_columns = EXPECTED_COLUMNS - set(raw_frame.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    invalid_targets = set(raw_frame[TARGET_COLUMN].unique()) - {"no", "yes"}
    if invalid_targets:
        raise ValueError(f"Unexpected target labels: {sorted(invalid_targets)}")

    original_rows = len(raw_frame)
    original_feature_count = len(raw_frame.columns) - 1
    original_duplicates = int(raw_frame.duplicated().sum())

    # duration is known only after a call. The project predicts potential
    # subscribers before contact, so keeping it would cause temporal leakage.
    cleaned = raw_frame.drop(columns=DROPPED_FEATURES).copy()
    duplicates_after_feature_removal = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    target = cleaned[TARGET_COLUMN].map({"no": 0, "yes": 1}).astype(int)
    target.name = "target"
    features = cleaned.drop(columns=[TARGET_COLUMN])

    processed = features.copy()
    processed["target"] = target
    processed.to_csv(PROCESSED_DATA_PATH, index=False)

    categorical_columns = list(features.select_dtypes(exclude="number").columns)
    numeric_columns = list(features.select_dtypes(include="number").columns)
    class_counts = target.value_counts().sort_index()
    unknown_counts = {
        column: int((features[column] == "unknown").sum())
        for column in categorical_columns
    }

    metadata: dict[str, Any] = {
        "dataset_name": "Bank Marketing",
        "source": "UCI Machine Learning Repository",
        "source_url": "https://archive.ics.uci.edu/dataset/222/bank",
        "doi": "10.24432/C5K306",
        "task": "Binary classification",
        "prediction_objective": (
            "Predict whether a client will subscribe to a term deposit before contact"
        ),
        "raw_samples": original_rows,
        "processed_samples": int(len(features)),
        "original_features": original_feature_count,
        "model_features": int(features.shape[1]),
        "numeric_features": numeric_columns,
        "categorical_features": categorical_columns,
        "target_column_original": TARGET_COLUMN,
        "target_column_processed": "target",
        "class_mapping": {"0": CLASS_NAMES[0], "1": CLASS_NAMES[1]},
        "positive_class": "yes",
        "class_counts": {
            CLASS_NAMES[0]: int(class_counts.get(0, 0)),
            CLASS_NAMES[1]: int(class_counts.get(1, 0)),
        },
        "class_percentages": {
            CLASS_NAMES[0]: float(class_counts.get(0, 0) / len(target) * 100),
            CLASS_NAMES[1]: float(class_counts.get(1, 0) / len(target) * 100),
        },
        "missing_values": int(features.isna().sum().sum()),
        "unknown_category_counts": unknown_counts,
        "duplicates_in_raw_data": original_duplicates,
        "duplicates_after_dropping_duration": duplicates_after_feature_removal,
        "removed_duplicate_rows": original_rows - len(cleaned),
        "dropped_features": DROPPED_FEATURES,
        "preprocessing": [
            "Removed duration to prevent temporal leakage",
            "Removed duplicate rows after dropping duration",
            "Encoded target as no=0 and yes=1",
            "Kept unknown as an explicit categorical level",
            "Applied One-Hot Encoding inside each model pipeline",
            "Used a stratified train/test split",
        ],
    }

    with (METRICS_DIR / "dataset_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    return features, target, metadata


def split_dataset(
    features: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create one stratified split shared by all experiments."""

    return train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=target,
    )

