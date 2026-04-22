from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets_unificados_usados"
PREPARED_DATASET = DATASETS_DIR / "training_dataset_balanced_v1.csv"
BASELINE_DATASET = DATASETS_DIR / "ranking_dataset.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "ranker_v1"

CATEGORICAL_COLUMNS = [
    "surface",
    "user_favorite_category",
    "user_favorite_device",
    "user_favorite_time_slot",
    "user_recent_favorite_category",
    "video_category",
]

# Excluye ids brutos y señales derivadas directamente del outcome observado.
BASELINE_EXCLUDED_COLUMNS = {
    "user_id",
    "video_id",
    "channel_id",
    "creator_user_id",
    "first_timestamp",
    "last_timestamp",
    "user_first_timestamp",
    "user_last_timestamp",
    "interaction_count",
    "avg_watch_percent",
    "max_watch_percent",
    "total_watch_time_s",
    "any_like",
    "any_comment",
    "any_subscription",
    "any_recommended",
    "any_click",
    "positive_label",
    "negative_label",
    "implicit_score_sum",
    "implicit_score_mean",
    "cf_confidence",
    "ranking_label",
    "watch_vs_user_avg",
}

PREPARED_EXCLUDED_COLUMNS = {
    "user_id",
    "video_id",
    "channel_id",
    "creator_user_id",
    "anchor_timestamp",
    "split",
    "example_source",
    "label",
}


@dataclass
class SplitData:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def get_label_column(df: pd.DataFrame) -> str:
    if "label" in df.columns:
        return "label"
    return "ranking_label"


def detect_device() -> str:
    try:
        result = xgb.build_info()
        if result.get("USE_CUDA") is True:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    if "last_timestamp" in df.columns:
        df["last_timestamp"] = pd.to_datetime(df["last_timestamp"], errors="coerce")
    return df


def split_by_time(df: pd.DataFrame) -> SplitData:
    if "last_timestamp" not in df.columns:
        raise ValueError("No existe `last_timestamp`; no puedo hacer split temporal.")
    ordered = df.sort_values("last_timestamp").reset_index(drop=True)
    n_rows = len(ordered)
    train_end = int(n_rows * 0.80)
    valid_end = int(n_rows * 0.90)

    train = ordered.iloc[:train_end].copy()
    valid = ordered.iloc[train_end:valid_end].copy()
    test = ordered.iloc[valid_end:].copy()

    return SplitData(train=train, valid=valid, test=test)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    label_column = get_label_column(df)
    excluded_columns = PREPARED_EXCLUDED_COLUMNS if label_column == "label" else BASELINE_EXCLUDED_COLUMNS

    y = df[label_column].astype(np.int32)
    X = df.drop(columns=[col for col in excluded_columns if col in df.columns]).copy()

    for column in CATEGORICAL_COLUMNS:
        if column in X.columns:
            X[column] = X[column].fillna("unknown").astype("category")

    for column in X.columns:
        if column in CATEGORICAL_COLUMNS:
            continue
        if X[column].dtype == "object":
            X[column] = X[column].fillna("unknown").astype("category")
        elif str(X[column].dtype).startswith("float") or str(X[column].dtype).startswith("int"):
            X[column] = pd.to_numeric(X[column], errors="coerce").fillna(0)

    return X, y


def split_dataset(df: pd.DataFrame) -> SplitData:
    if "split" in df.columns:
        train = df.loc[df["split"].eq("train")].copy()
        valid = df.loc[df["split"].eq("valid")].copy()
        test = df.loc[df["split"].eq("test")].copy()
        if len(train) and len(valid) and len(test):
            return SplitData(train=train, valid=valid, test=test)
    return split_by_time(df)


def precision_at_k(df: pd.DataFrame, scores: np.ndarray, k: int = 10) -> float:
    label_column = get_label_column(df)
    scored = df[["user_id", label_column]].copy()
    scored["score"] = scores
    scored = scored.sort_values(["user_id", "score"], ascending=[True, False])
    topk = scored.groupby("user_id", sort=False).head(k)
    per_user = topk.groupby("user_id", sort=False)[label_column].mean()
    return float(per_user.mean()) if not per_user.empty else 0.0


def recall_at_k(df: pd.DataFrame, scores: np.ndarray, k: int = 10) -> float:
    label_column = get_label_column(df)
    scored = df[["user_id", label_column]].copy()
    scored["score"] = scores
    scored = scored.sort_values(["user_id", "score"], ascending=[True, False])
    topk = scored.groupby("user_id", sort=False).head(k)

    hits = topk.groupby("user_id", sort=False)[label_column].sum()
    positives = df.groupby("user_id", sort=False)[label_column].sum()
    valid_users = positives[positives > 0].index
    recalls = (hits.reindex(valid_users, fill_value=0) / positives.reindex(valid_users)).fillna(0)
    return float(recalls.mean()) if not recalls.empty else 0.0


def candidate_stats(df: pd.DataFrame) -> dict[str, float | int]:
    counts = df.groupby("user_id", sort=False).size()
    return {
        "users": int(counts.shape[0]),
        "mean_candidates_per_user": round(float(counts.mean()), 6) if len(counts) else 0.0,
        "median_candidates_per_user": round(float(counts.median()), 6) if len(counts) else 0.0,
        "max_candidates_per_user": int(counts.max()) if len(counts) else 0,
    }


def train_model(split_data: SplitData, output_dir: Path) -> dict[str, float | int | str]:
    X_train, y_train = prepare_features(split_data.train)
    X_valid, y_valid = prepare_features(split_data.valid)
    X_test, y_test = prepare_features(split_data.test)

    device = detect_device()
    scale_pos_weight = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        min_child_weight=4,
        objective="binary:logistic",
        tree_method="hist",
        device=device,
        enable_categorical=True,
        eval_metric=["logloss", "auc", "aucpr"],
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        early_stopping_rounds=50,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=25,
    )

    valid_scores = model.predict_proba(X_valid)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]
    valid_candidate_stats = candidate_stats(split_data.valid)
    test_candidate_stats = candidate_stats(split_data.test)

    metrics = {
        "device": device,
        "train_rows": int(len(X_train)),
        "valid_rows": int(len(X_valid)),
        "test_rows": int(len(X_test)),
        "feature_count": int(X_train.shape[1]),
        "best_iteration": int(getattr(model, "best_iteration", model.n_estimators)),
        "valid_roc_auc": round(float(roc_auc_score(y_valid, valid_scores)), 6),
        "valid_average_precision": round(float(average_precision_score(y_valid, valid_scores)), 6),
        "valid_log_loss": round(float(log_loss(y_valid, valid_scores)), 6),
        "valid_precision_at_1": round(precision_at_k(split_data.valid, valid_scores, k=1), 6),
        "valid_precision_at_5": round(precision_at_k(split_data.valid, valid_scores, k=5), 6),
        "valid_precision_at_10": round(precision_at_k(split_data.valid, valid_scores, k=10), 6),
        "valid_recall_at_10": round(recall_at_k(split_data.valid, valid_scores, k=10), 6),
        "test_roc_auc": round(float(roc_auc_score(y_test, test_scores)), 6),
        "test_average_precision": round(float(average_precision_score(y_test, test_scores)), 6),
        "test_log_loss": round(float(log_loss(y_test, test_scores)), 6),
        "test_precision_at_1": round(precision_at_k(split_data.test, test_scores, k=1), 6),
        "test_precision_at_5": round(precision_at_k(split_data.test, test_scores, k=5), 6),
        "test_precision_at_10": round(precision_at_k(split_data.test, test_scores, k=10), 6),
        "test_recall_at_10": round(recall_at_k(split_data.test, test_scores, k=10), 6),
        "positive_rate_train": round(float(y_train.mean()), 6),
        "positive_rate_valid": round(float(y_valid.mean()), 6),
        "positive_rate_test": round(float(y_test.mean()), 6),
        "valid_candidate_users": valid_candidate_stats["users"],
        "valid_mean_candidates_per_user": valid_candidate_stats["mean_candidates_per_user"],
        "valid_median_candidates_per_user": valid_candidate_stats["median_candidates_per_user"],
        "valid_max_candidates_per_user": valid_candidate_stats["max_candidates_per_user"],
        "test_candidate_users": test_candidate_stats["users"],
        "test_mean_candidates_per_user": test_candidate_stats["mean_candidates_per_user"],
        "test_median_candidates_per_user": test_candidate_stats["median_candidates_per_user"],
        "test_max_candidates_per_user": test_candidate_stats["max_candidates_per_user"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(output_dir / "ranker_model.json")

    feature_importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance_gain": [model.get_booster().get_score(importance_type="gain").get(col, 0.0) for col in X_train.columns],
        }
    ).sort_values("importance_gain", ascending=False)
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    with open(output_dir / "training_config.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "dataset": str(PREPARED_DATASET if PREPARED_DATASET.exists() else BASELINE_DATASET),
                "categorical_columns": CATEGORICAL_COLUMNS,
                "excluded_columns": sorted(PREPARED_EXCLUDED_COLUMNS if PREPARED_DATASET.exists() else BASELINE_EXCLUDED_COLUMNS),
                "notes": [
                    "Trainer GPU para el ranker compartido.",
                    "Si existe un dataset preparado balanceado, se usa por defecto.",
                    "Sigue siendo un pipeline offline y no garantiza point-in-time safety completa.",
                ],
            },
            fh,
            indent=2,
        )

    return metrics


def main() -> None:
    dataset_path = PREPARED_DATASET if PREPARED_DATASET.exists() else BASELINE_DATASET
    if not dataset_path.exists():
        raise FileNotFoundError(f"No encuentro el dataset de entrenamiento: {dataset_path}")

    df = load_dataset(dataset_path)
    split_data = split_dataset(df)
    metrics = train_model(split_data, DEFAULT_OUTPUT_DIR)

    print("Training completed")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
