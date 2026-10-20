from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from safe_on_clean_text.data.splits import load_split
from safe_on_clean_text.degradations import iter_prediction_inputs, load_condition_texts
from safe_on_clean_text.models.predictions import (
    prediction_rows,
    classification_metrics,
    predicted_labels,
    save_predictions,
)
from safe_on_clean_text.provenance import commit_hash, library_versions, timestamp, save_json


def build_vectorizer(max_features: int, ngram_range: tuple[int, int]) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        lowercase=True,
        strip_accents="unicode",
    )


def build_logreg(max_features: int, ngram_range: tuple[int, int], c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", build_vectorizer(max_features, ngram_range)),
            ("clf", LogisticRegression(C=c_value, max_iter=2000, solver="liblinear", random_state=seed)),
        ]
    )


def build_linear_svm(max_features: int, ngram_range: tuple[int, int], seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", build_vectorizer(max_features, ngram_range)),
            ("clf", LinearSVC(max_iter=5000, random_state=seed)),
        ]
    )


def fit_logreg_with_c_selection(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    c_grid: list[float],
    max_features: int,
    ngram_range: tuple[int, int],
    seed: int,
) -> tuple[Pipeline, float, dict[float, float]]:
    """
    Fit LogReg on train for every C; pick the C with the best validation macro-F1 (ties -> smaller C).
    Returns the model fitted on train with that C, the C, and the validation macro-F1 per C.
    """

    models: dict[float, Pipeline] = {}
    scores: dict[float, float] = {}
    for c_value in sorted(c_grid):
        model = build_logreg(max_features, ngram_range, c_value, seed)
        model.fit(train_df["text"].tolist(), train_df["label"].to_numpy())
        y_pred = model.predict(validation_df["text"].tolist())
        models[c_value] = model
        scores[c_value] = float(f1_score(validation_df["label"], y_pred, average="macro"))

    best_score = max(scores.values())
    best_c = min(c_value for c_value, score in scores.items() if score == best_score)

    return models[best_c], best_c, scores


def logreg_logits(model: Pipeline, texts: pd.Series) -> np.ndarray:
    """
    2-class logits (0, decision_function): their softmax equals predict_proba.
    """

    decision = model.decision_function(texts.tolist())
    logits = np.column_stack([np.zeros_like(decision), decision])

    probabilities = model.predict_proba(texts.tolist())
    assert np.allclose(1.0 / (1.0 + np.exp(-decision)), probabilities[:, 1])
    return logits


def train(config: dict) -> None:
    """
    TF-IDF + LogReg (C by validation macro-F1) and LinearSVC -> one run folder each (predictions on every split and
    condition, run_meta.json).
    """

    started_at = timestamp()
    start_time = time.perf_counter()

    seed = int(config["seed"])
    dataset = config["data"]["name"]
    data_dir = Path(config["data"]["processed_dir"])
    conditions = config["conditions"]
    baseline_config = config["baselines"]
    max_features = int(baseline_config["max_features"])
    ngram_range = tuple(baseline_config["ngram_range"])

    split_to_df = {
        name: load_split(data_dir / f"{name}.csv") for name in ["train", "validation", "calibration", "test"]
    }
    condition_texts = load_condition_texts(data_dir / "conditions.csv")

    logreg, best_c, c_scores = fit_logreg_with_c_selection(
        train_df=split_to_df["train"],
        validation_df=split_to_df["validation"],
        c_grid=[float(c_value) for c_value in baseline_config["logreg_c_grid"]],
        max_features=max_features,
        ngram_range=ngram_range,
        seed=seed,
    )

    svm = build_linear_svm(max_features, ngram_range, seed)
    svm.fit(split_to_df["train"]["text"].tolist(), split_to_df["train"]["label"].to_numpy())

    model_details = {
        "tfidf_logreg": {
            "logreg_c": best_c,
            "validation_macro_f1_by_c": {str(c_value): score for c_value, score in c_scores.items()},
            "in_conformal": True,
        },
        "tfidf_linear_svm": {"svm_c": float(svm.named_steps["clf"].C), "in_conformal": False},
    }

    for model_key, model in {"tfidf_logreg": logreg, "tfidf_linear_svm": svm}.items():
        prediction_frames: list[pd.DataFrame] = []
        clean_metrics: dict[str, dict[str, float]] = {}

        for split_name, condition, df, texts in iter_prediction_inputs(split_to_df, condition_texts, conditions):
            if model_key == "tfidf_logreg":
                frame = prediction_rows(
                    df["id"], split_name, condition, df["label"], texts, logits=logreg_logits(model, texts)
                )
            else:
                frame = prediction_rows(
                    df["id"],
                    split_name,
                    condition,
                    df["label"],
                    texts,
                    decision_scores=model.decision_function(texts.tolist()),
                )
            prediction_frames.append(frame)

            if condition == "clean":
                clean_metrics[split_name] = classification_metrics(frame["label"], predicted_labels(frame))

        run_dir = Path(config["paths"]["runs_dir"]) / dataset / model_key / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        save_predictions(pd.concat(prediction_frames, ignore_index=True), run_dir / "predictions.csv.gz")

        save_json(
            {
                "dataset": dataset,
                "model": model_key,
                "seed": seed,
                "commit": commit_hash(),
                **model_details[model_key],
                "vocabulary_size": len(model.named_steps["tfidf"].vocabulary_),
                "conditions": conditions,
                "n_rows": {name: len(df) for name, df in split_to_df.items()},
                "clean_metrics": clean_metrics,
                "total_seconds": round(time.perf_counter() - start_time, 1),
                "started_at": started_at,
                "finished_at": timestamp(),
                "versions": library_versions(),
                "config": {
                    "seed": seed,
                    "data": config["data"],
                    "baselines": baseline_config,
                    "conditions": conditions,
                },
            },
            run_dir / "run_meta.json",
        )

        print(f"Saved run folder to: {run_dir.resolve()}")
        print(pd.DataFrame(clean_metrics).T.round(4).to_string())

    print(f"\nLogReg C by validation macro-F1: {c_scores} -> C = {best_c}")
