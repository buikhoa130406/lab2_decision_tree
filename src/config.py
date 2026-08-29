"""Shared configuration for the Bank Marketing classification project."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "bank-full.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "bank_marketing_processed.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"

TARGET_COLUMN = "y"
DROPPED_FEATURES = ["duration"]
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# The business-relevant class is a successful term-deposit subscription.
POSITIVE_CLASS = 1
CLASS_NAMES = ["no", "yes"]


def ensure_directories() -> None:
    """Create every directory used by generated artifacts."""

    for directory in (
        RAW_DATA_PATH.parent,
        PROCESSED_DATA_PATH.parent,
        FIGURES_DIR,
        METRICS_DIR,
        MODELS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

