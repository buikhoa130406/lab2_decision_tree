"""Run the complete Bank Marketing Decision Tree project."""

from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.tree import export_text

from config import METRICS_DIR, MODELS_DIR, ensure_directories
from data import load_dataset, split_dataset
from evaluation import evaluate_all_models
from modeling import train_all_models
from visualization import create_all_figures, get_tree_components


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def main() -> None:
    ensure_directories()
    print("[1/5] Loading Bank Marketing data and removing duration...")
    features, target, metadata = load_dataset()
    x_train, x_test, y_train, y_test = split_dataset(features, target)

    split_summary = {
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "test_size": len(x_test) / len(features),
        "train_class_counts": y_train.value_counts().sort_index().to_dict(),
        "test_class_counts": y_test.value_counts().sort_index().to_dict(),
    }
    with (METRICS_DIR / "split_summary.json").open("w", encoding="utf-8") as file:
        json.dump(split_summary, file, indent=2, default=_json_default)

    print("[2/5] Training baseline and three improvement strategies...")
    training = train_all_models(x_train, y_train)

    print("[3/5] Evaluating every model on the untouched test set...")
    metrics, reports, matrices = evaluate_all_models(
        training.models, x_train, y_train, x_test, y_test
    )

    cv_scores = {
        name: detail["cv_f1_yes_mean"]
        for name, detail in training.tuning_summary.items()
    }
    selected_model_name = max(cv_scores, key=cv_scores.get)
    metrics["cv_f1_yes"] = metrics["model"].map(cv_scores)
    metrics["selected_by_training_cv"] = metrics["model"].eq(selected_model_name)

    metrics.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    training.controlled_complexity_cv.to_csv(
        METRICS_DIR / "controlled_complexity_cv.csv", index=False
    )
    training.pruning_cv.to_csv(METRICS_DIR / "pruning_cv.csv", index=False)
    training.class_weight_cv.to_csv(METRICS_DIR / "class_weight_cv.csv", index=False)
    with (METRICS_DIR / "classification_reports.json").open("w", encoding="utf-8") as file:
        json.dump(reports, file, indent=2, default=_json_default)
    with (METRICS_DIR / "tuning_summary.json").open("w", encoding="utf-8") as file:
        json.dump(training.tuning_summary, file, indent=2, default=_json_default)
    with (METRICS_DIR / "selected_model.json").open("w", encoding="utf-8") as file:
        json.dump({
            "model": selected_model_name,
            "selection_rule": "Highest mean 5-fold CV F1 for yes on training data",
            "cv_f1_yes": cv_scores[selected_model_name],
            "dataset": metadata["dataset_name"],
        }, file, indent=2)

    print("[4/5] Saving models, rules, and figures...")
    for name, model in training.models.items():
        filename = name.lower().replace(" ", "_").replace("-", "_") + ".joblib"
        joblib.dump(model, MODELS_DIR / filename)
    joblib.dump(training.models[selected_model_name], MODELS_DIR / "best_model.joblib")

    with (METRICS_DIR / "tree_rules.txt").open("w", encoding="utf-8") as file:
        for name, model in training.models.items():
            classifier, feature_names = get_tree_components(model)
            file.write(f"===== {name} =====\n")
            file.write(export_text(classifier, feature_names=feature_names, max_depth=6))
            file.write("\n")

    create_all_figures(
        features, target, training.models, metrics, matrices,
        training.pruning_cv, selected_model_name,
    )

    print("[5/5] Complete. Final comparison:")
    display_columns = [
        "model", "train_accuracy", "test_accuracy", "error_rate",
        "precision_yes", "recall_yes", "f1_yes", "pr_auc_yes",
        "tree_depth", "leaf_count", "cv_f1_yes", "selected_by_training_cv",
    ]
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(metrics[display_columns].round(4).to_string(index=False))
    print(f"\nSelected model by training CV: {selected_model_name}")


if __name__ == "__main__":
    main()

