"""Consistent classification evaluation with yes as the positive class."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline

from config import CLASS_NAMES, POSITIVE_CLASS


def evaluate_model(
    name: str, model: Pipeline, x_train: pd.DataFrame, y_train: pd.Series,
    x_test: pd.DataFrame, y_test: pd.Series,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    train_prediction = model.predict(x_train)
    test_prediction = model.predict(x_test)
    yes_index = list(model.classes_).index(POSITIVE_CLASS)
    yes_probability = model.predict_proba(x_test)[:, yes_index]
    matrix = confusion_matrix(y_test, test_prediction, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    test_accuracy = accuracy_score(y_test, test_prediction)
    tree = model.named_steps["classifier"]

    summary: dict[str, Any] = {
        "model": name,
        "train_accuracy": accuracy_score(y_train, train_prediction),
        "test_accuracy": test_accuracy,
        "error_rate": 1.0 - test_accuracy,
        "precision_yes": precision_score(y_test, test_prediction, pos_label=POSITIVE_CLASS, zero_division=0),
        "recall_yes": recall_score(y_test, test_prediction, pos_label=POSITIVE_CLASS, zero_division=0),
        "f1_yes": f1_score(y_test, test_prediction, pos_label=POSITIVE_CLASS, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, test_prediction),
        "pr_auc_yes": average_precision_score(y_test, yes_probability),
        "roc_auc_yes": roc_auc_score(y_test, yes_probability),
        "tree_depth": tree.get_depth(),
        "leaf_count": tree.get_n_leaves(),
        "node_count": tree.tree_.node_count,
        "true_positive_yes": int(true_positive),
        "false_negative_yes": int(false_negative),
        "false_positive_yes": int(false_positive),
        "true_negative_no": int(true_negative),
    }
    report = classification_report(
        y_test, test_prediction, labels=[0, 1], target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    return summary, report, matrix


def evaluate_all_models(
    models: dict[str, Pipeline], x_train: pd.DataFrame, y_train: pd.Series,
    x_test: pd.DataFrame, y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    summaries, reports, matrices = [], {}, {}
    for name, model in models.items():
        summary, report, matrix = evaluate_model(name, model, x_train, y_train, x_test, y_test)
        summaries.append(summary)
        reports[name] = report
        matrices[name] = matrix
    return pd.DataFrame(summaries), reports, matrices

