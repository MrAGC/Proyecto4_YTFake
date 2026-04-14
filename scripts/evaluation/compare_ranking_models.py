from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "reco_output_v2" / "training_dataset_balanced_v1.csv"
OUTPUT_DIR = PROJECT_ROOT / "models" / "ranker_compare_v1"

CATEGORICAL_COLUMNS = [
    "surface",
    "user_favorite_category",
    "user_favorite_device",
    "user_favorite_time_slot",
    "user_recent_favorite_category",
    "video_category",
]

EXCLUDED_COLUMNS = {
    "user_id",
    "video_id",
    "channel_id",
    "creator_user_id",
    "anchor_timestamp",
    "split",
    "example_source",
    "label",
}


def detect_device() -> str:
    try:
        result = xgb.build_info()
        if result.get("USE_CUDA") is True:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No encuentro el dataset de ranking preparado: {path}")
    return pd.read_csv(path)


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df.loc[df["split"].eq("train")].copy()
    valid_df = df.loc[df["split"].eq("valid")].copy()
    test_df = df.loc[df["split"].eq("test")].copy()
    return train_df, valid_df, test_df


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in EXCLUDED_COLUMNS]


def precision_at_k(df: pd.DataFrame, scores: np.ndarray, k: int) -> float:
    scored = df[["user_id", "label"]].copy()
    scored["score"] = scores
    scored = scored.sort_values(["user_id", "score"], ascending=[True, False])
    topk = scored.groupby("user_id", sort=False).head(k)
    per_user = topk.groupby("user_id", sort=False)["label"].mean()
    return float(per_user.mean()) if not per_user.empty else 0.0


def ndcg_at_k(df: pd.DataFrame, scores: np.ndarray, k: int) -> float:
    scored = df[["user_id", "label"]].copy()
    scored["score"] = scores
    scored = scored.sort_values(["user_id", "score"], ascending=[True, False])

    total = 0.0
    users = 0
    for _, group in scored.groupby("user_id", sort=False):
        labels = group["label"].to_numpy(dtype=np.float32)
        ranked = labels[:k]
        discounts = 1.0 / np.log2(np.arange(2, len(ranked) + 2))
        dcg = float(np.sum(ranked * discounts))
        ideal = np.sort(labels)[::-1][:k]
        idcg = float(np.sum(ideal * discounts[: len(ideal)]))
        if idcg <= 0:
            continue
        total += dcg / idcg
        users += 1
    return total / users if users else 0.0


def candidate_stats(df: pd.DataFrame) -> dict[str, float | int]:
    counts = df.groupby("user_id", sort=False).size()
    return {
        "users": int(counts.shape[0]),
        "mean_candidates_per_user": round(float(counts.mean()), 6) if len(counts) else 0.0,
        "median_candidates_per_user": round(float(counts.median()), 6) if len(counts) else 0.0,
        "max_candidates_per_user": int(counts.max()) if len(counts) else 0,
    }


def prepare_pointwise_dense_inputs(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    feature_columns = select_feature_columns(train_df)
    categorical_columns = [column for column in CATEGORICAL_COLUMNS if column in feature_columns]
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]

    cat_imputer = SimpleImputer(strategy="constant", fill_value="unknown")
    cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    num_imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    scaler = StandardScaler()

    X_train_cat = cat_encoder.fit_transform(cat_imputer.fit_transform(train_df[categorical_columns]))
    X_valid_cat = cat_encoder.transform(cat_imputer.transform(valid_df[categorical_columns]))
    X_test_cat = cat_encoder.transform(cat_imputer.transform(test_df[categorical_columns]))

    X_train_num = scaler.fit_transform(num_imputer.fit_transform(train_df[numeric_columns]))
    X_valid_num = scaler.transform(num_imputer.transform(valid_df[numeric_columns]))
    X_test_num = scaler.transform(num_imputer.transform(test_df[numeric_columns]))

    X_train = np.hstack([X_train_num, X_train_cat]).astype(np.float32)
    X_valid = np.hstack([X_valid_num, X_valid_cat]).astype(np.float32)
    X_test = np.hstack([X_test_num, X_test_cat]).astype(np.float32)

    y_train = train_df["label"].astype(np.int32).to_numpy()
    y_valid = valid_df["label"].astype(np.int32).to_numpy()
    y_test = test_df["label"].astype(np.int32).to_numpy()

    artifacts = {
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "cat_imputer": cat_imputer,
        "cat_encoder": cat_encoder,
        "num_imputer": num_imputer,
        "scaler": scaler,
    }
    return X_train, y_train, X_valid, y_valid, X_test, y_test, artifacts


def prepare_pairwise_ranker_inputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    group_sizes = df.groupby("user_id", sort=False).size()
    valid_users = group_sizes[group_sizes >= 2].index
    filtered = df.loc[df["user_id"].isin(valid_users)].copy()
    filtered = filtered.sort_values(["user_id", "video_id"]).reset_index(drop=True)

    X = filtered.drop(columns=[column for column in EXCLUDED_COLUMNS if column in filtered.columns]).copy()
    for column in CATEGORICAL_COLUMNS:
        if column in X.columns:
            X[column] = X[column].fillna("unknown").astype("category")

    for column in X.columns:
        if column in CATEGORICAL_COLUMNS:
            continue
        if X[column].dtype == "object":
            X[column] = X[column].fillna("unknown").astype("category")
        else:
            X[column] = pd.to_numeric(X[column], errors="coerce").fillna(0.0)

    y = filtered["label"].astype(np.int32).to_numpy()
    groups = filtered.groupby("user_id", sort=False).size().to_numpy(dtype=np.int32)
    return filtered, X, y, groups


def build_metrics(df: pd.DataFrame, labels: np.ndarray, scores: np.ndarray, probabilistic_scores: bool) -> dict[str, float | int | None]:
    stats = candidate_stats(df)
    metrics: dict[str, float | int | None] = {
        "rows": int(len(df)),
        "positive_rate": round(float(labels.mean()), 6),
        "roc_auc": round(float(roc_auc_score(labels, scores)), 6),
        "average_precision": round(float(average_precision_score(labels, scores)), 6),
        "log_loss": round(float(log_loss(labels, scores)), 6) if probabilistic_scores else None,
        "precision_at_1": round(precision_at_k(df, scores, k=1), 6),
        "precision_at_5": round(precision_at_k(df, scores, k=5), 6),
        "precision_at_10": round(precision_at_k(df, scores, k=10), 6),
        "ndcg_at_10": round(ndcg_at_k(df, scores, k=10), 6),
        "candidate_users": stats["users"],
        "mean_candidates_per_user": stats["mean_candidates_per_user"],
        "median_candidates_per_user": stats["median_candidates_per_user"],
        "max_candidates_per_user": stats["max_candidates_per_user"],
    }
    return metrics


def train_pointwise_dense_nn(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, object]:
    X_train, y_train, X_valid, y_valid, X_test, y_test, preprocess_artifacts = prepare_pointwise_dense_inputs(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
    )

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        batch_size=4096,
        learning_rate_init=0.001,
        max_iter=20,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=42,
        verbose=True,
    )
    model.fit(X_train, y_train)

    valid_scores = model.predict_proba(X_valid)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]

    model_dir = OUTPUT_DIR / "pointwise_dense_nn"
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "model.pkl", "wb") as file_handle:
        pickle.dump(
            {
                "model": model,
                "preprocess": preprocess_artifacts,
            },
            file_handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    metrics = {
        "valid": build_metrics(valid_df, y_valid, valid_scores, probabilistic_scores=True),
        "test": build_metrics(test_df, y_test, test_scores, probabilistic_scores=True),
        "training": {
            "train_rows": int(len(train_df)),
            "feature_count": int(X_train.shape[1]),
            "iterations": int(model.n_iter_),
        },
    }
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)

    return metrics


def train_pairwise_xgboost(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, object]:
    train_grouped, X_train, y_train, train_groups = prepare_pairwise_ranker_inputs(train_df)
    valid_grouped, X_valid, y_valid, valid_groups = prepare_pairwise_ranker_inputs(valid_df)
    test_grouped, X_test, y_test, _ = prepare_pairwise_ranker_inputs(test_df)

    device = detect_device()
    model = xgb.XGBRanker(
        objective="rank:pairwise",
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        min_child_weight=4,
        tree_method="hist",
        device=device,
        enable_categorical=True,
        eval_metric=["ndcg@10", "map@10"],
        random_state=42,
        early_stopping_rounds=30,
    )

    model.fit(
        X_train,
        y_train,
        group=train_groups,
        eval_set=[(X_valid, y_valid)],
        eval_group=[valid_groups],
        verbose=25,
    )

    valid_scores = model.predict(X_valid)
    test_scores = model.predict(X_test)

    model_dir = OUTPUT_DIR / "pairwise_xgboost"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(model_dir / "ranker_model.json")

    feature_importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance_gain": [model.get_booster().get_score(importance_type="gain").get(column, 0.0) for column in X_train.columns],
        }
    ).sort_values("importance_gain", ascending=False)
    feature_importance.to_csv(model_dir / "feature_importance.csv", index=False)

    metrics = {
        "valid": build_metrics(valid_grouped, y_valid, valid_scores, probabilistic_scores=False),
        "test": build_metrics(test_grouped, y_test, test_scores, probabilistic_scores=False),
        "training": {
            "train_rows": int(len(train_grouped)),
            "feature_count": int(X_train.shape[1]),
            "device": device,
            "best_iteration": int(getattr(model, "best_iteration", model.n_estimators)),
            "train_groups": int(len(train_groups)),
            "valid_groups": int(len(valid_groups)),
        },
    }
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)

    return metrics


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset(DATASET_PATH)
    train_df, valid_df, test_df = split_dataset(df)

    pointwise_metrics = train_pointwise_dense_nn(train_df, valid_df, test_df)
    pairwise_metrics = train_pairwise_xgboost(train_df, valid_df, test_df)

    comparison = {
        "best_model_by_test_ndcg_at_10": "pairwise_xgboost"
        if pairwise_metrics["test"]["ndcg_at_10"] >= pointwise_metrics["test"]["ndcg_at_10"]
        else "pointwise_dense_nn",
        "pointwise_dense_nn": pointwise_metrics,
        "pairwise_xgboost": pairwise_metrics,
        "notes": [
            "PointWise Dense NN usa el dataset balanceado y puntua cada candidato por separado.",
            "PairWise XGBoost usa objective rank:pairwise y el contexto extra del dataset.",
            "La comparacion se debe leer sobre todo con ndcg@10 y precision@1, no solo con AUC.",
        ],
    }

    with open(OUTPUT_DIR / "comparison_summary.json", "w", encoding="utf-8") as file_handle:
        json.dump(comparison, file_handle, indent=2)

    print("Ranking model comparison completed")
    print(f"  output_dir: {OUTPUT_DIR}")
    print(f"  best_model_by_test_ndcg_at_10: {comparison['best_model_by_test_ndcg_at_10']}")
    for model_name in ["pointwise_dense_nn", "pairwise_xgboost"]:
        print()
        print(model_name)
        for split_name in ["valid", "test"]:
            print(f"  {split_name}:")
            for metric_name, metric_value in comparison[model_name][split_name].items():
                print(f"    {metric_name}: {metric_value}")


if __name__ == "__main__":
    main()
