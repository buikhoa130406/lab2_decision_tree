"""Generate EDA, evaluation, complexity, and tree figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree

from config import CLASS_NAMES, FIGURES_DIR


def _save_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close()


def get_tree_components(model: Pipeline):
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = list(preprocessor.get_feature_names_out())
    return classifier, feature_names


def plot_class_distribution(target: pd.Series) -> None:
    counts = target.map({0: "No", 1: "Yes"}).value_counts().reindex(["No", "Yes"])
    plt.figure(figsize=(7, 4.5))
    axis = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False)
    axis.set_title("Term-deposit subscription class distribution")
    axis.set_xlabel("Subscribed term deposit")
    axis.set_ylabel("Number of clients")
    for container in axis.containers:
        axis.bar_label(container)
    _save_figure("class_distribution.png")


def plot_numeric_correlation(features: pd.DataFrame) -> None:
    numeric = features.select_dtypes(include="number")
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation among numeric features (duration excluded)")
    _save_figure("numeric_feature_correlation.png")


def plot_selected_feature_distributions(features: pd.DataFrame, target: pd.Series) -> None:
    frame = features[["age", "balance", "campaign"]].copy()
    frame["Subscribed"] = target.map({0: "No", 1: "Yes"})
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, column in zip(axes, ["age", "balance", "campaign"]):
        sns.boxplot(data=frame, x="Subscribed", y=column, hue="Subscribed", legend=False, ax=axis, showfliers=False)
        axis.set_title(column.capitalize())
        axis.set_xlabel("")
    figure.suptitle("Selected numeric features by target class", y=1.02)
    _save_figure("selected_feature_distributions.png")


def plot_confusion_matrices(matrices: dict[str, np.ndarray]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 9))
    for axis, (name, matrix) in zip(axes.flat, matrices.items()):
        sns.heatmap(
            matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=axis,
        )
        axis.set_title(name)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
    figure.suptitle("Confusion matrices on the same test set", y=1.02)
    _save_figure("confusion_matrices.png")


def plot_model_comparison(metrics: pd.DataFrame) -> None:
    long_frame = metrics.melt(
        id_vars="model",
        value_vars=["test_accuracy", "precision_yes", "recall_yes", "f1_yes"],
        var_name="metric", value_name="score",
    )
    long_frame["metric"] = long_frame["metric"].map({
        "test_accuracy": "Accuracy", "precision_yes": "Precision (yes)",
        "recall_yes": "Recall (yes)", "f1_yes": "F1 (yes)",
    })
    plt.figure(figsize=(13, 6))
    axis = sns.barplot(data=long_frame, x="model", y="score", hue="metric")
    axis.set_ylim(0, 1.02)
    axis.set_title("Final test-set performance")
    axis.set_xlabel("")
    axis.set_ylabel("Score")
    axis.tick_params(axis="x", rotation=12)
    axis.legend(title="Metric", loc="lower right")
    _save_figure("model_comparison.png")


def plot_tree_complexity(metrics: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    sns.barplot(data=metrics, x="model", y="tree_depth", ax=axes[0])
    sns.barplot(data=metrics, x="model", y="leaf_count", ax=axes[1])
    sns.barplot(data=metrics, x="model", y="node_count", ax=axes[2])
    for axis, title in zip(axes, ["Tree depth", "Leaves", "Nodes"]):
        axis.set_title(title)
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=18)
    figure.suptitle("Model complexity comparison", y=1.02)
    _save_figure("tree_complexity.png")


def plot_pruning_curve(pruning_cv: pd.DataFrame) -> None:
    frame = pruning_cv.sort_values("param_classifier__ccp_alpha")
    alpha = frame["param_classifier__ccp_alpha"].astype(float)
    plt.figure(figsize=(8, 5))
    plt.plot(alpha, frame["mean_train_score"], marker="o", label="Train CV F1")
    plt.plot(alpha, frame["mean_test_score"], marker="o", label="Validation CV F1")
    plt.xlabel("ccp_alpha")
    plt.ylabel("F1 for yes class")
    plt.title("Effect of cost-complexity pruning")
    plt.legend()
    plt.grid(alpha=0.25)
    _save_figure("pruning_curve.png")


def plot_feature_importance(model: Pipeline) -> None:
    classifier, feature_names = get_tree_components(model)
    importance = pd.Series(classifier.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=False).head(15).sort_values()
    plt.figure(figsize=(9, 7))
    importance.plot(kind="barh")
    plt.title("Top feature importances of the CV-selected model")
    plt.xlabel("Impurity-based importance")
    _save_figure("best_model_feature_importance.png")


def plot_decision_tree(name: str, model: Pipeline) -> None:
    classifier, feature_names = get_tree_components(model)
    safe_name = name.lower().replace(" ", "_").replace("-", "_")
    plt.figure(figsize=(28, 14))
    plot_tree(
        classifier, feature_names=feature_names, class_names=CLASS_NAMES,
        filled=True, rounded=True, proportion=False, precision=2,
        max_depth=3, fontsize=7,
    )
    plt.title(f"{name} - first four levels")
    _save_figure(f"{safe_name}_tree_readable.png")


def create_all_figures(
    features: pd.DataFrame, target: pd.Series, models: dict[str, Pipeline],
    metrics: pd.DataFrame, matrices: dict[str, np.ndarray],
    pruning_cv: pd.DataFrame, selected_model_name: str,
) -> None:
    sns.set_theme(style="whitegrid")
    plot_class_distribution(target)
    plot_numeric_correlation(features)
    plot_selected_feature_distributions(features, target)
    plot_confusion_matrices(matrices)
    plot_model_comparison(metrics)
    plot_tree_complexity(metrics)
    plot_pruning_curve(pruning_cv)
    plot_feature_importance(models[selected_model_name])
    for name, model in models.items():
        plot_decision_tree(name, model)

