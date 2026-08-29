"""Baseline and three improvement strategies for Bank Marketing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from config import CV_FOLDS, POSITIVE_CLASS, RANDOM_STATE


YES_F1_SCORER = make_scorer(f1_score, pos_label=POSITIVE_CLASS)


@dataclass
class TrainingResult:
    """Models and training-only evidence used to select hyperparameters."""

    models: dict[str, Pipeline]
    tuning_summary: dict[str, dict[str, Any]]
    controlled_complexity_cv: pd.DataFrame
    pruning_cv: pd.DataFrame
    class_weight_cv: pd.DataFrame


def build_pipeline(features: pd.DataFrame, class_weight=None) -> Pipeline:
    """Create a leakage-safe categorical preprocessing and tree pipeline."""

    numeric_columns = list(features.select_dtypes(include="number").columns)
    categorical_columns = list(features.select_dtypes(exclude="number").columns)
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
            ("numeric", "passthrough", numeric_columns),
        ],
        verbose_feature_names_out=False,
    )
    classifier = DecisionTreeClassifier(
        random_state=RANDOM_STATE, class_weight=class_weight
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def _compact_cv_results(search: GridSearchCV) -> pd.DataFrame:
    columns = [
        name for name in search.cv_results_
        if name.startswith("param_") or name in {
            "mean_test_score", "std_test_score", "mean_train_score", "rank_test_score"
        }
    ]
    return (
        pd.DataFrame(search.cv_results_)[columns]
        .sort_values(["rank_test_score", "mean_test_score"], ascending=[True, False])
        .reset_index(drop=True)
    )


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key.replace("classifier__", ""): value for key, value in params.items()}


def train_all_models(x_train: pd.DataFrame, y_train: pd.Series) -> TrainingResult:
    """Train all models using F1 of yes as the primary CV score."""

    baseline = build_pipeline(x_train)
    baseline.fit(x_train, y_train)
    baseline_scores = cross_val_score(
        build_pipeline(x_train), x_train, y_train, cv=CV_FOLDS,
        scoring=YES_F1_SCORER, n_jobs=-1,
    )

    controlled_search = GridSearchCV(
        estimator=build_pipeline(x_train),
        param_grid={
            "classifier__criterion": ["gini", "entropy"],
            "classifier__max_depth": [3, 5, 7, 10, None],
            "classifier__min_samples_split": [2, 20, 50],
            "classifier__min_samples_leaf": [1, 10, 25],
        },
        scoring=YES_F1_SCORER, cv=CV_FOLDS, n_jobs=-1,
        return_train_score=True, refit=True,
    )
    controlled_search.fit(x_train, y_train)

    transformed_train = baseline.named_steps["preprocessor"].transform(x_train)
    pruning_path = DecisionTreeClassifier(
        random_state=RANDOM_STATE
    ).cost_complexity_pruning_path(transformed_train, y_train)
    candidate_alphas = np.unique(np.round(pruning_path.ccp_alphas[:-1], 8))
    if len(candidate_alphas) > 25:
        indices = np.linspace(0, len(candidate_alphas) - 1, 25).astype(int)
        candidate_alphas = np.unique(candidate_alphas[indices])
    if 0.0 not in candidate_alphas:
        candidate_alphas = np.insert(candidate_alphas, 0, 0.0)

    pruning_search = GridSearchCV(
        estimator=build_pipeline(x_train),
        param_grid={"classifier__ccp_alpha": candidate_alphas.tolist()},
        scoring=YES_F1_SCORER, cv=CV_FOLDS, n_jobs=-1,
        return_train_score=True, refit=True,
    )
    pruning_search.fit(x_train, y_train)

    balanced_search = GridSearchCV(
        estimator=build_pipeline(x_train, class_weight="balanced"),
        param_grid={
            "classifier__max_depth": [3, 5, 7, 10, None],
            "classifier__min_samples_split": [2, 20, 50],
            "classifier__min_samples_leaf": [1, 10, 25],
        },
        scoring=YES_F1_SCORER, cv=CV_FOLDS, n_jobs=-1,
        return_train_score=True, refit=True,
    )
    balanced_search.fit(x_train, y_train)

    models = {
        "Baseline": baseline,
        "Controlled complexity": controlled_search.best_estimator_,
        "Cost-complexity pruning": pruning_search.best_estimator_,
        "Balanced class weights": balanced_search.best_estimator_,
    }
    tuning_summary = {
        "Baseline": {
            "best_params": _clean_params(baseline.named_steps["classifier"].get_params()),
            "cv_f1_yes_mean": float(baseline_scores.mean()),
            "cv_f1_yes_std": float(baseline_scores.std()),
        },
        "Controlled complexity": {
            "best_params": _clean_params(controlled_search.best_params_),
            "cv_f1_yes_mean": float(controlled_search.best_score_),
        },
        "Cost-complexity pruning": {
            "best_params": _clean_params(pruning_search.best_params_),
            "cv_f1_yes_mean": float(pruning_search.best_score_),
        },
        "Balanced class weights": {
            "best_params": {**_clean_params(balanced_search.best_params_), "class_weight": "balanced"},
            "cv_f1_yes_mean": float(balanced_search.best_score_),
        },
    }
    return TrainingResult(
        models=models,
        tuning_summary=tuning_summary,
        controlled_complexity_cv=_compact_cv_results(controlled_search),
        pruning_cv=_compact_cv_results(pruning_search),
        class_weight_cv=_compact_cv_results(balanced_search),
    )

