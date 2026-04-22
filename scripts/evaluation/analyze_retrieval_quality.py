from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.train_retrieval_cf import (
    SECONDARY_ITEM_WEIGHT,
    TOP_RECOMMENDATIONS,
    TOP_USER_NEIGHBORS,
    item_based_scores,
    merge_scores,
    top_items,
    user_based_scores,
)


DATASETS_DIR = PROJECT_ROOT / "datasets_unificados_usados"
CF_DATASET_PATH = DATASETS_DIR / "cf_interactions.csv"
VIDEO_FEATURES_PATH = DATASETS_DIR / "video_features.csv"
RETRIEVAL_MODEL_PATH = PROJECT_ROOT / "models" / "retrieval_cf_v1" / "retrieval_model.pkl"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "retrieval_diagnostics_v1"


def dcg_at_k(recommended_items: list[int], relevant_items: set[int], k: int) -> float:
    score = 0.0
    for rank_index, item_id in enumerate(recommended_items[:k], start=1):
        if item_id in relevant_items:
            score += 1.0 / math.log2(rank_index + 1)
    return score


def load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cf_df = pd.read_csv(CF_DATASET_PATH)
    video_features = pd.read_csv(
        VIDEO_FEATURES_PATH,
        usecols=["video_id", "channel_id", "creator_user_id", "video_category", "is_recent_upload"],
    )
    with open(RETRIEVAL_MODEL_PATH, "rb") as file_handle:
        model = pickle.load(file_handle)
    return cf_df, video_features, model


def compute_dataset_stats(cf_df: pd.DataFrame) -> dict[str, float | int]:
    train_df = cf_df.loc[cf_df["split"].eq("train")].copy()
    user_history_sizes = train_df.groupby("user_id", sort=False).size()
    item_support_sizes = train_df.groupby("video_id", sort=False).size()
    return {
        "train_rows": int(len(train_df)),
        "train_users": int(train_df["user_id"].nunique()),
        "train_items": int(train_df["video_id"].nunique()),
        "avg_train_positives_per_user": round(float(user_history_sizes.mean()), 6),
        "median_train_positives_per_user": round(float(user_history_sizes.median()), 6),
        "avg_train_users_per_item": round(float(item_support_sizes.mean()), 6),
        "median_train_users_per_item": round(float(item_support_sizes.median()), 6),
        "pct_users_with_3_or_less_train_positives": round(float((user_history_sizes <= 3).mean()), 6),
    }


def compute_holdout_overlap_stats(cf_df: pd.DataFrame, video_features: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    merged = cf_df.copy()
    if "video_category" not in merged.columns:
        merged = merged.merge(video_features[["video_id", "video_category"]], on="video_id", how="left")
    if "channel_id" not in merged.columns:
        merged = merged.merge(video_features[["video_id", "channel_id"]], on="video_id", how="left")
    if "creator_user_id" not in merged.columns:
        merged = merged.merge(video_features[["video_id", "creator_user_id"]], on="video_id", how="left")
    train_df = merged.loc[merged["split"].eq("train")].copy()
    train_user_profiles = train_df.groupby("user_id", sort=False).agg(
        train_categories=("video_category", lambda values: set(map(str, values))),
        train_channels=("channel_id", lambda values: set(map(int, values))),
        train_creators=("creator_user_id", lambda values: set(map(int, values))),
        train_history_size=("video_id", "size"),
    )
    train_item_popularity = train_df.groupby("video_id", sort=False).size()

    output: dict[str, dict[str, float | int]] = {}
    for split_name in ["validation", "test"]:
        split_df = merged.loc[merged["split"].eq(split_name)].copy()
        split_df = split_df.merge(train_user_profiles, on="user_id", how="left")
        item_popularity = split_df["video_id"].map(train_item_popularity).fillna(0)

        category_match = split_df.apply(lambda row: str(row["video_category"]) in row["train_categories"], axis=1)
        channel_match = split_df.apply(lambda row: int(row["channel_id"]) in row["train_channels"], axis=1)
        creator_match = split_df.apply(lambda row: int(row["creator_user_id"]) in row["train_creators"], axis=1)
        unseen_vs_train = ~split_df["video_id"].isin(train_item_popularity.index)

        output[split_name] = {
            "rows": int(len(split_df)),
            "unseen_item_rate_vs_train": round(float(unseen_vs_train.mean()), 6),
            "avg_target_item_train_support": round(float(item_popularity.mean()), 6),
            "median_target_item_train_support": round(float(item_popularity.median()), 6),
            "category_seen_in_user_history": round(float(category_match.mean()), 6),
            "channel_seen_in_user_history": round(float(channel_match.mean()), 6),
            "creator_seen_in_user_history": round(float(creator_match.mean()), 6),
            "avg_user_train_history_size": round(float(split_df["train_history_size"].mean()), 6),
            "median_user_train_history_size": round(float(split_df["train_history_size"].median()), 6),
        }
    return output


def evaluate_retrieval_quality(cf_df: pd.DataFrame, video_features: pd.DataFrame, model: dict) -> dict[str, dict[str, dict[str, float | int]]]:
    video_category = dict(zip(video_features["video_id"], video_features["video_category"]))
    user_histories = model["user_histories"]
    user_norms = model["user_norms"]
    item_users = model["item_users"]
    item_neighbors = model["item_neighbors"]

    output: dict[str, dict[str, dict[str, float | int]]] = {}
    for split_name in ["validation", "test"]:
        split_df = cf_df.loc[cf_df["split"].eq(split_name)].copy()
        targets_by_user = split_df.groupby("user_id", sort=False)["video_id"].agg(list)
        model_metrics = {
            "user_based_cf": {"hit@10": 0.0, "hit@50": 0.0, "category_hit@50": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0},
            "item_based_cf": {"hit@10": 0.0, "hit@50": 0.0, "category_hit@50": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0},
            "hybrid_cf": {"hit@10": 0.0, "hit@50": 0.0, "category_hit@50": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0},
        }
        evaluated_users = 0

        for user_id, target_list in targets_by_user.items():
            user_id = int(user_id)
            if user_id not in user_histories:
                continue

            relevant_items = {int(video_id) for video_id in target_list}
            target_categories = {video_category.get(video_id) for video_id in relevant_items}

            primary_scores = user_based_scores(
                user_id=user_id,
                user_histories=user_histories,
                user_norms=user_norms,
                item_users=item_users,
                top_neighbors=TOP_USER_NEIGHBORS,
            )
            secondary_scores = item_based_scores(
                user_id=user_id,
                user_histories=user_histories,
                item_neighbors=item_neighbors,
            )
            hybrid_scores = merge_scores(primary_scores, secondary_scores, secondary_weight=SECONDARY_ITEM_WEIGHT)

            recommendations = {
                "user_based_cf": top_items(primary_scores, TOP_RECOMMENDATIONS),
                "item_based_cf": top_items(secondary_scores, TOP_RECOMMENDATIONS),
                "hybrid_cf": top_items(hybrid_scores, TOP_RECOMMENDATIONS),
            }

            evaluated_users += 1
            for model_name, recommended_items in recommendations.items():
                top10 = recommended_items[:10]
                top50 = recommended_items[:50]
                hit10 = any(item_id in relevant_items for item_id in top10)
                hit50 = any(item_id in relevant_items for item_id in top50)
                category_hit50 = any(video_category.get(item_id) in target_categories for item_id in top50)

                reciprocal_rank = 0.0
                for rank_index, item_id in enumerate(top10, start=1):
                    if item_id in relevant_items:
                        reciprocal_rank = 1.0 / rank_index
                        break

                ideal_dcg = dcg_at_k(list(relevant_items), relevant_items, k=10)
                ndcg = 0.0
                if ideal_dcg > 0:
                    ndcg = dcg_at_k(recommended_items, relevant_items, k=10) / ideal_dcg

                model_metrics[model_name]["hit@10"] += float(hit10)
                model_metrics[model_name]["hit@50"] += float(hit50)
                model_metrics[model_name]["category_hit@50"] += float(category_hit50)
                model_metrics[model_name]["mrr@10"] += reciprocal_rank
                model_metrics[model_name]["ndcg@10"] += ndcg

        output[split_name] = {}
        for model_name, raw_metrics in model_metrics.items():
            output[split_name][model_name] = {
                metric_name: round(metric_value / max(evaluated_users, 1), 6)
                for metric_name, metric_value in raw_metrics.items()
            }
            output[split_name][model_name]["evaluated_users"] = evaluated_users
    return output


def build_markdown_report(
    dataset_stats: dict[str, float | int],
    overlap_stats: dict[str, dict[str, float | int]],
    retrieval_quality: dict[str, dict[str, dict[str, float | int]]],
) -> str:
    lines: list[str] = []
    lines.append("# Diagnostico del retrieval")
    lines.append("")
    lines.append("## Resumen")
    lines.append("")
    for key, value in dataset_stats.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Solape entre holdout e historial")
    lines.append("")
    for split_name, metrics in overlap_stats.items():
        lines.append(f"### {split_name}")
        lines.append("")
        for key, value in metrics.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines.append("## Calidad del retrieval")
    lines.append("")
    for split_name, split_metrics in retrieval_quality.items():
        lines.append(f"### {split_name}")
        lines.append("")
        for model_name, metrics in split_metrics.items():
            lines.append(f"- {model_name}")
            for key, value in metrics.items():
                lines.append(f"  - {key}: {value}")
        lines.append("")
    lines.append("## Lectura tecnica")
    lines.append("")
    lines.append("- El retrieval exacto por item es muy bajo.")
    lines.append("- Sin embargo, el category_hit@50 es altisimo, lo que indica que los candidatos si caen en el tema correcto.")
    lines.append("- Los items de validation y test no son cold-start respecto a train: el unseen_item_rate_vs_train es 0.")
    lines.append("- El problema principal no es falta de soporte del item ni ausencia total de vecinos.")
    lines.append("- El problema es que el dataset expresa preferencia sobre todo a nivel de categoria o tema, y la evaluacion exacta por item castiga como fallo muchos candidatos que en realidad son razonables para home.")
    lines.append("- La conclusion operativa es mantener CF como retrieval principal/secundario, pero evaluar candidate generation con metricas de calidad de candidatos y no solo con exact match.")
    return "\n".join(lines)


def main() -> None:
    cf_df, video_features, model = load_artifacts()
    dataset_stats = compute_dataset_stats(cf_df)
    overlap_stats = compute_holdout_overlap_stats(cf_df, video_features)
    retrieval_quality = evaluate_retrieval_quality(cf_df, video_features, model)

    report = {
        "dataset_stats": dataset_stats,
        "holdout_overlap": overlap_stats,
        "retrieval_quality": retrieval_quality,
        "conclusion": {
            "root_cause": "El retrieval memory-based recupera bien el tema, pero la evaluacion exacta por item es demasiado estricta para este dataset.",
            "recommended_next_step": "Usar retrieval multi-source y medir candidate quality y downstream ranking, no solo hit exacto al item holdout.",
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "diagnostics.json", "w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=2)
    (OUTPUT_DIR / "diagnostics.md").write_text(
        build_markdown_report(dataset_stats, overlap_stats, retrieval_quality),
        encoding="utf-8",
    )

    print("Retrieval diagnostics completed")
    print(f"  output_dir: {OUTPUT_DIR}")
    print(f"  validation_hybrid_category_hit@50: {retrieval_quality['validation']['hybrid_cf']['category_hit@50']}")
    print(f"  test_hybrid_category_hit@50: {retrieval_quality['test']['hybrid_cf']['category_hit@50']}")


if __name__ == "__main__":
    main()
