from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets_unificados_usados"
DATASET_PATH = DATASETS_DIR / "training_dataset_balanced_v1.csv"
METADATA_PATH = DATASETS_DIR / "training_dataset_balanced_v1_metadata.json"
CF_DATASET_PATH = DATASETS_DIR / "cf_interactions.csv"
VIDEOS_PATH = DATASETS_DIR / "video_features.csv"
RANKER_MODEL_PATH = PROJECT_ROOT / "models" / "ranker_compare_v1" / "pairwise_xgboost" / "ranker_model.json"
RETRIEVAL_METRICS_PATH = PROJECT_ROOT / "models" / "retrieval_multisource_v1" / "metrics.json"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "recommender_quality_v1"
SUMMARY_PATH = OUTPUT_DIR / "quality_summary.json"
REPORT_PATH = OUTPUT_DIR / "quality_report.md"

CATEGORICAL_COLUMNS = [
    "surface",
    "user_favorite_category",
    "user_favorite_device",
    "user_favorite_time_slot",
    "user_recent_favorite_category",
    "video_category",
]


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def dcg(labels: np.ndarray) -> float:
    discounts = 1.0 / np.log2(np.arange(2, len(labels) + 2))
    return float(np.sum(labels * discounts))


def average_precision(labels: np.ndarray, k: int) -> float:
    labels = labels[:k]
    positives = 0
    total = 0.0
    for index, label in enumerate(labels, start=1):
        if label <= 0:
            continue
        positives += 1
        total += positives / index
    return total / max(int(labels.sum()), 1)


def reciprocal_rank(labels: np.ndarray, k: int) -> float:
    for index, label in enumerate(labels[:k], start=1):
        if label > 0:
            return 1.0 / index
    return 0.0


def score_ranker(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    model = xgb.XGBRanker()
    model.load_model(RANKER_MODEL_PATH)

    features = df[feature_columns].copy()
    for column in CATEGORICAL_COLUMNS:
        if column in features.columns:
            features[column] = features[column].fillna("unknown").astype("category")
    for column in features.columns:
        if column in CATEGORICAL_COLUMNS:
            continue
        if features[column].dtype == "object":
            features[column] = features[column].fillna("unknown").astype("category")
        else:
            features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0)

    scored = df.copy()
    scored["score"] = model.predict(features)
    return scored


def build_ranking_metrics(scored: pd.DataFrame, train_seen: dict[int, set[int]], total_videos: int, k: int = 10) -> dict[str, object]:
    precision_values: list[float] = []
    hit_values: list[float] = []
    recall_values: list[float] = []
    ndcg_values: list[float] = []
    map_values: list[float] = []
    mrr_values: list[float] = []
    category_diversity_values: list[float] = []
    channel_diversity_values: list[float] = []
    novelty_values: list[float] = []
    seen_repeat_values: list[float] = []
    recommended_items: set[int] = set()
    evaluated_users = 0

    scored = scored.sort_values(["user_id", "score"], ascending=[True, False])
    for user_id, group in scored.groupby("user_id", sort=False):
        labels_all = group["label"].to_numpy(dtype=np.float32)
        relevant_count = int(labels_all.sum())
        if relevant_count <= 0:
            continue
        top = group.head(k)
        labels = top["label"].to_numpy(dtype=np.float32)
        recommended_ids = [int(video_id) for video_id in top["video_id"].tolist()]
        recommended_items.update(recommended_ids)

        precision_values.append(float(labels.mean()) if len(labels) else 0.0)
        hit_values.append(float(labels.sum() > 0))
        recall_values.append(float(labels.sum() / relevant_count))
        ideal = np.sort(labels_all)[::-1][: len(labels)]
        ideal_dcg = dcg(ideal)
        ndcg_values.append(dcg(labels) / ideal_dcg if ideal_dcg > 0 else 0.0)
        map_values.append(average_precision(labels, k=k))
        mrr_values.append(reciprocal_rank(labels, k=k))
        category_diversity_values.append(top["video_category"].nunique() / max(len(top), 1))
        channel_diversity_values.append(top["channel_id"].nunique() / max(len(top), 1))
        novelty_values.append(float(pd.to_numeric(top["is_recent_upload"], errors="coerce").fillna(0).mean()))
        user_seen = train_seen.get(int(user_id), set())
        seen_repeat_values.append(sum(video_id in user_seen for video_id in recommended_ids) / max(len(recommended_ids), 1))
        evaluated_users += 1

    def mean(values: list[float]) -> float:
        return round(float(np.mean(values)), 6) if values else 0.0

    return {
        "evaluated_users": evaluated_users,
        "precision_at_10": mean(precision_values),
        "hit_rate_at_10": mean(hit_values),
        "recall_at_10": mean(recall_values),
        "ndcg_at_10": mean(ndcg_values),
        "map_at_10": mean(map_values),
        "mrr_at_10": mean(mrr_values),
        "coverage_at_10": round(len(recommended_items) / max(total_videos, 1), 6),
        "unique_recommended_items": len(recommended_items),
        "category_diversity_at_10": mean(category_diversity_values),
        "channel_diversity_at_10": mean(channel_diversity_values),
        "novelty_at_10": mean(novelty_values),
        "seen_repeat_rate_at_10": mean(seen_repeat_values),
    }


def build_negative_sampling_summary(df: pd.DataFrame) -> dict[str, object]:
    total = len(df)
    source_counts = df["example_source"].fillna("unknown").value_counts().to_dict()
    label_counts = df["label"].value_counts().to_dict()
    negatives = df.loc[df["label"].eq(0)].copy()
    synthetic = negatives.loc[negatives["example_source"].eq("synthetic_negative")]
    observed = negatives.loc[negatives["example_source"].eq("observed_negative")]

    hard_proxy_rate = 0.0
    if not synthetic.empty:
        hard_proxy_rate = float(
            (
                synthetic["category_matches_user_pref"].eq(1)
                | synthetic["recent_category_match"].eq(1)
                | synthetic["user_follows_channel"].eq(1)
            ).mean()
        )

    return {
        "total_rows": total,
        "positive_rows": int(label_counts.get(1, 0)),
        "negative_rows": int(label_counts.get(0, 0)),
        "positive_rate": round(float(df["label"].mean()), 6),
        "source_counts": {str(key): int(value) for key, value in source_counts.items()},
        "observed_negative_rows": int(len(observed)),
        "synthetic_negative_rows": int(len(synthetic)),
        "synthetic_hard_proxy_rate": round(hard_proxy_rate, 6),
        "interpretation": [
            "Los negativos observados vienen de baja retencion sin acciones positivas.",
            "Los negativos sinteticos son videos no vistos, preferiblemente de categorias afines o recientes.",
            "Falta separar explicitamente hard negatives por retrieval/categoria/canal para que el report sea mas defendible.",
        ],
    }


def build_candidate_summary(df: pd.DataFrame) -> dict[str, object]:
    group_sizes = df.groupby(["split", "user_id"], sort=False).size().reset_index(name="candidates")
    split_rows: list[dict[str, object]] = []
    for split_name, group in group_sizes.groupby("split", sort=False):
        split_rows.append(
            {
                "split": str(split_name),
                "users": int(len(group)),
                "mean_candidates": round(float(group["candidates"].mean()), 3),
                "median_candidates": round(float(group["candidates"].median()), 3),
                "max_candidates": int(group["candidates"].max()),
            }
        )
    return {
        "by_split": split_rows,
        "warning": "La evaluacion del ranker tiene pocos candidatos por usuario; para demo esta bien, pero para produccion conviene evaluar con 20-100 candidatos realistas por usuario.",
    }


def build_chart_rows(metrics: dict[str, object], negative_summary: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    ranking_keys = [
        ("Precision@10", "precision_at_10"),
        ("HitRate@10", "hit_rate_at_10"),
        ("Recall@10", "recall_at_10"),
        ("NDCG@10", "ndcg_at_10"),
        ("MAP@10", "map_at_10"),
        ("MRR@10", "mrr_at_10"),
    ]
    ux_keys = [
        ("Coverage@10", "coverage_at_10"),
        ("Diversity categorias", "category_diversity_at_10"),
        ("Diversity canales", "channel_diversity_at_10"),
        ("Novelty@10", "novelty_at_10"),
        ("Repeat vistos", "seen_repeat_rate_at_10"),
    ]
    source_total = sum(negative_summary["source_counts"].values())
    return {
        "ranking": [
            {"label": label, "value": float(metrics.get(key, 0)), "pct": round(float(metrics.get(key, 0)) * 100, 2)}
            for label, key in ranking_keys
        ],
        "ux": [
            {"label": label, "value": float(metrics.get(key, 0)), "pct": round(float(metrics.get(key, 0)) * 100, 2)}
            for label, key in ux_keys
        ],
        "negative_sources": [
            {
                "label": str(label),
                "count": int(count),
                "pct": round((int(count) / max(source_total, 1)) * 100, 2),
            }
            for label, count in negative_summary["source_counts"].items()
        ],
    }


def write_markdown(summary: dict[str, object]) -> None:
    ranking = summary["ranking_metrics"]
    negative = summary["negative_sampling"]
    candidate = summary["candidate_pool"]
    retrieval = summary["retrieval_metrics"]
    lines = [
        "# Evaluacion De Calidad Del Recomendador",
        "",
        "## 1. Como se ha evaluado",
        "",
        "Se usa el split `test` del dataset de ranking balanceado. Para cada usuario se ordenan candidatos con el ranker `pairwise_xgboost` y se calculan metricas @10.",
        "",
        "Esta evaluacion sirve para demo y comparacion offline. La mejora pendiente es evaluar con pools de candidatos mas grandes por usuario.",
        "",
        "## 2. Ranking",
        "",
        f"- Precision@10: `{ranking['precision_at_10']}`",
        f"- HitRate@10: `{ranking['hit_rate_at_10']}`",
        f"- Recall@10: `{ranking['recall_at_10']}`",
        f"- NDCG@10: `{ranking['ndcg_at_10']}`",
        f"- MAP@10: `{ranking['map_at_10']}`",
        f"- MRR@10: `{ranking['mrr_at_10']}`",
        "",
        "## 3. Experiencia De Usuario",
        "",
        f"- Coverage@10: `{ranking['coverage_at_10']}`",
        f"- Diversity categorias@10: `{ranking['category_diversity_at_10']}`",
        f"- Diversity canales@10: `{ranking['channel_diversity_at_10']}`",
        f"- Novelty@10: `{ranking['novelty_at_10']}`",
        f"- Repeat vistos@10: `{ranking['seen_repeat_rate_at_10']}`",
        "",
        "## 4. Negative Sampling",
        "",
        f"- Filas positivas: `{negative['positive_rows']}`",
        f"- Filas negativas: `{negative['negative_rows']}`",
        f"- Rate de positivos: `{negative['positive_rate']}`",
        f"- Negativos observados: `{negative['observed_negative_rows']}`",
        f"- Negativos sinteticos: `{negative['synthetic_negative_rows']}`",
        f"- Hard-negative proxy en sinteticos: `{negative['synthetic_hard_proxy_rate']}`",
        "",
        "Interpretacion:",
        "",
    ]
    lines.extend(f"- {item}" for item in negative["interpretation"])
    lines.extend(
        [
            "",
            "## 5. Pool De Candidatos",
            "",
            candidate["warning"],
            "",
        ]
    )
    for row in candidate["by_split"]:
        lines.append(
            f"- `{row['split']}`: {row['users']} usuarios, media {row['mean_candidates']} candidatos, mediana {row['median_candidates']}, max {row['max_candidates']}"
        )
    if retrieval:
        lines.extend(
            [
                "",
                "## 6. Retrieval",
                "",
                f"- Test category_hit@50: `{retrieval.get('test', {}).get('category_hit@50', 'n/a')}`",
                f"- Test hit@50 exacto: `{retrieval.get('test', {}).get('hit@50', 'n/a')}`",
                "",
                "Lectura: el retrieval exacto por item es demasiado estricto para este dataset, pero recupera muy bien el tema.",
            ]
        )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = load_json(METADATA_PATH)
    feature_columns = list(metadata["feature_columns"])
    dataset = pd.read_csv(DATASET_PATH)
    test_df = dataset.loc[dataset["split"].eq("test")].copy()
    video_features = pd.read_csv(VIDEOS_PATH, usecols=["video_id"])
    cf_train = pd.read_csv(CF_DATASET_PATH, usecols=["user_id", "video_id", "split"])
    train_seen = (
        cf_train.loc[cf_train["split"].eq("train")]
        .groupby("user_id", sort=False)["video_id"]
        .agg(lambda values: {int(value) for value in values})
        .to_dict()
    )

    scored_test = score_ranker(test_df, feature_columns)
    ranking_metrics = build_ranking_metrics(scored_test, train_seen, total_videos=int(len(video_features)), k=10)
    negative_sampling = build_negative_sampling_summary(dataset)
    candidate_pool = build_candidate_summary(dataset)
    retrieval_metrics = load_json(RETRIEVAL_METRICS_PATH)

    summary = {
        "report_name": "recommender_quality_v1",
        "generated_from": {
            "ranking_dataset": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
            "ranker_model": str(RANKER_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "retrieval_metrics": str(RETRIEVAL_METRICS_PATH.relative_to(PROJECT_ROOT)),
        },
        "ranking_metrics": ranking_metrics,
        "negative_sampling": negative_sampling,
        "candidate_pool": candidate_pool,
        "retrieval_metrics": retrieval_metrics,
    }
    summary["charts"] = build_chart_rows(ranking_metrics, negative_sampling)
    summary["demo_takeaway"] = [
        "El ranker ordena bien dentro del pool evaluado y debe explicarse con NDCG, MAP, MRR y Precision@K.",
        "El retrieval se debe defender por category_hit@50 y candidate quality, no por acierto exacto de item.",
        "Los negativos observados son buenos; el siguiente salto es etiquetar hard negatives explicitamente.",
        "Para una evaluacion mas realista falta aumentar candidatos por usuario en test.",
    ]

    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary)
    print("Recommender quality evaluation completed")
    print(f"  summary: {SUMMARY_PATH}")
    print(f"  report : {REPORT_PATH}")


if __name__ == "__main__":
    main()
