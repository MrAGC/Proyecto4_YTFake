from __future__ import annotations

import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd

from train_retrieval_cf import (
    SECONDARY_ITEM_WEIGHT,
    TOP_ITEM_NEIGHBORS,
    TOP_RECOMMENDATIONS,
    TOP_USER_NEIGHBORS,
    build_item_neighbors,
    build_user_histories,
    item_based_scores,
    top_items,
    user_based_scores,
)


CF_DATASET_PATH = Path("reco_output_v2/cf_interactions.csv")
VIDEO_FEATURES_PATH = Path("reco_output_v2/video_features.csv")
USER_FEATURES_PATH = Path("reco_output_v2/user_features.csv")
USER_CHANNEL_FEATURES_PATH = Path("reco_output_v2/user_channel_features.csv")
OUTPUT_DIR = Path("models/retrieval_multisource_v1")

USER_CF_WEIGHT = 0.35
ITEM_CF_WEIGHT = 0.20
CATEGORY_WEIGHT = 0.25
CHANNEL_WEIGHT = 0.15
FRESHNESS_WEIGHT = 0.05

CF_TOP_K = 100
CATEGORY_TOP_K = 80
CHANNEL_TOP_K = 20
FRESH_TOP_K = 60
FINAL_TOP_K = 50


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cf_df = pd.read_csv(CF_DATASET_PATH)
    video_features = pd.read_csv(VIDEO_FEATURES_PATH)
    user_features = pd.read_csv(USER_FEATURES_PATH)
    user_channel_features = pd.read_csv(USER_CHANNEL_FEATURES_PATH)
    return cf_df, video_features, user_features, user_channel_features


def build_video_indexes(video_features: pd.DataFrame) -> tuple[dict[str, list[int]], dict[str, list[int]], dict[int, list[int]], list[int]]:
    ranked_videos = video_features.copy()
    ranked_videos["retrieval_score"] = (
        0.70 * ranked_videos["video_discovery_score"].fillna(0)
        + 0.20 * ranked_videos["video_engagement_score"].fillna(0)
        + 0.10 * ranked_videos["video_freshness_score"].fillna(0)
    )
    ranked_videos = ranked_videos.sort_values(
        ["retrieval_score", "video_freshness_score", "video_popularity_log"],
        ascending=False,
    )

    category_top_videos: dict[str, list[int]] = {}
    recent_category_top_videos: dict[str, list[int]] = {}
    for category, group in ranked_videos.groupby("video_category", sort=False):
        ordered = group["video_id"].astype(int).tolist()
        category_top_videos[str(category)] = ordered
        recent_group = group.loc[group["is_recent_upload"].eq(1), "video_id"].astype(int).tolist()
        recent_category_top_videos[str(category)] = recent_group if recent_group else ordered

    channel_top_videos: dict[int, list[int]] = {}
    for channel_id, group in ranked_videos.groupby("channel_id", sort=False):
        channel_top_videos[int(channel_id)] = group["video_id"].astype(int).tolist()

    global_recent = ranked_videos.loc[ranked_videos["is_recent_upload"].eq(1), "video_id"].astype(int).tolist()
    if not global_recent:
        global_recent = ranked_videos["video_id"].astype(int).tolist()

    return category_top_videos, recent_category_top_videos, channel_top_videos, global_recent


def build_user_channel_index(user_channel_features: pd.DataFrame) -> dict[int, list[dict[str, float | int]]]:
    df = user_channel_features.copy()
    df["retrieval_channel_score"] = (
        0.60 * df["channel_current_interest_score"].fillna(0)
        + 0.20 * df["user_follows_channel"].fillna(0)
        + 0.10 * (df["channel_recent_interactions_30d"].fillna(0) > 0).astype(int)
        - 0.20 * df["stale_follow_flag"].fillna(0)
    )
    df = df.sort_values(["user_id", "retrieval_channel_score"], ascending=[True, False])

    index: dict[int, list[dict[str, float | int]]] = {}
    for user_id, group in df.groupby("user_id", sort=False):
        records: list[dict[str, float | int]] = []
        for row in group.itertuples(index=False):
            if float(row.retrieval_channel_score) <= 0:
                continue
            records.append(
                {
                    "channel_id": int(row.channel_id),
                    "score": float(row.retrieval_channel_score),
                }
            )
            if len(records) == 6:
                break
        index[int(user_id)] = records
    return index


def build_user_preference_index(user_features: pd.DataFrame) -> dict[int, dict[str, str]]:
    df = user_features[
        [
            "user_id",
            "user_favorite_category",
            "user_recent_favorite_category",
        ]
    ].copy()
    return {
        int(row.user_id): {
            "favorite_category": str(row.user_favorite_category),
            "recent_favorite_category": str(row.user_recent_favorite_category),
        }
        for row in df.itertuples(index=False)
    }


def add_normalized_scores(target: dict[int, float], source_scores: dict[int, float], weight: float) -> None:
    if not source_scores:
        return
    max_score = max(source_scores.values()) or 1.0
    for item_id, score in source_scores.items():
        target[item_id] += weight * (score / max_score)


def weighted_ranked_list(items: list[int], weight: float, top_k: int) -> dict[int, float]:
    scores: dict[int, float] = {}
    for rank_index, item_id in enumerate(items[:top_k], start=1):
        scores[int(item_id)] = weight / math.sqrt(rank_index)
    return scores


def build_multisource_scores(
    user_id: int,
    seen_videos: set[int],
    user_histories: dict[int, dict[int, float]],
    user_norms: dict[int, float],
    item_users: dict[int, list[tuple[int, float]]],
    item_neighbors: dict[int, list[tuple[float, int]]],
    user_preferences: dict[int, dict[str, str]],
    user_channel_index: dict[int, list[dict[str, float | int]]],
    category_top_videos: dict[str, list[int]],
    recent_category_top_videos: dict[str, list[int]],
    channel_top_videos: dict[int, list[int]],
    global_recent: list[int],
) -> dict[int, float]:
    final_scores: dict[int, float] = defaultdict(float)

    primary_cf = user_based_scores(
        user_id=user_id,
        user_histories=user_histories,
        user_norms=user_norms,
        item_users=item_users,
        top_neighbors=TOP_USER_NEIGHBORS,
    )
    primary_cf = {item_id: score for item_id, score in primary_cf.items() if item_id not in seen_videos}
    add_normalized_scores(final_scores, dict(sorted(primary_cf.items(), key=lambda pair: pair[1], reverse=True)[:CF_TOP_K]), USER_CF_WEIGHT)

    secondary_cf = item_based_scores(
        user_id=user_id,
        user_histories=user_histories,
        item_neighbors=item_neighbors,
    )
    secondary_cf = {item_id: score for item_id, score in secondary_cf.items() if item_id not in seen_videos}
    add_normalized_scores(final_scores, dict(sorted(secondary_cf.items(), key=lambda pair: pair[1], reverse=True)[:CF_TOP_K]), ITEM_CF_WEIGHT)

    preferences = user_preferences.get(user_id, {})
    recent_category = preferences.get("recent_favorite_category", "unknown")
    favorite_category = preferences.get("favorite_category", "unknown")

    category_scores: dict[int, float] = {}
    if recent_category and recent_category != "unknown":
        recent_items = [item_id for item_id in recent_category_top_videos.get(recent_category, []) if item_id not in seen_videos]
        category_scores.update(weighted_ranked_list(recent_items, weight=1.0, top_k=CATEGORY_TOP_K))
    if favorite_category and favorite_category != "unknown":
        favorite_items = [item_id for item_id in category_top_videos.get(favorite_category, []) if item_id not in seen_videos]
        for item_id, score in weighted_ranked_list(favorite_items, weight=0.7, top_k=CATEGORY_TOP_K).items():
            category_scores[item_id] = category_scores.get(item_id, 0.0) + score
    add_normalized_scores(final_scores, category_scores, CATEGORY_WEIGHT)

    channel_scores: dict[int, float] = {}
    for channel_rank, channel_info in enumerate(user_channel_index.get(user_id, []), start=1):
        channel_id = int(channel_info["channel_id"])
        channel_weight = float(channel_info["score"]) / math.sqrt(channel_rank)
        channel_items = [item_id for item_id in channel_top_videos.get(channel_id, []) if item_id not in seen_videos]
        for item_id, score in weighted_ranked_list(channel_items, weight=channel_weight, top_k=CHANNEL_TOP_K).items():
            channel_scores[item_id] = channel_scores.get(item_id, 0.0) + score
    add_normalized_scores(final_scores, channel_scores, CHANNEL_WEIGHT)

    fresh_scores = weighted_ranked_list(
        [item_id for item_id in global_recent if item_id not in seen_videos],
        weight=1.0,
        top_k=FRESH_TOP_K,
    )
    add_normalized_scores(final_scores, fresh_scores, FRESHNESS_WEIGHT)

    return dict(final_scores)


def evaluate_multisource(
    split_df: pd.DataFrame,
    video_features: pd.DataFrame,
    user_histories: dict[int, dict[int, float]],
    user_norms: dict[int, float],
    item_users: dict[int, list[tuple[int, float]]],
    item_neighbors: dict[int, list[tuple[float, int]]],
    user_preferences: dict[int, dict[str, str]],
    user_channel_index: dict[int, list[dict[str, float | int]]],
    category_top_videos: dict[str, list[int]],
    recent_category_top_videos: dict[str, list[int]],
    channel_top_videos: dict[int, list[int]],
    global_recent: list[int],
) -> tuple[dict[str, float | int], pd.DataFrame]:
    video_category = dict(zip(video_features["video_id"], video_features["video_category"]))
    targets_by_user = split_df.groupby("user_id", sort=False)["video_id"].agg(list)

    metrics = {
        "hit@10": 0.0,
        "hit@50": 0.0,
        "category_hit@50": 0.0,
        "mrr@10": 0.0,
        "ndcg@10": 0.0,
        "mean_candidates_returned": 0.0,
    }
    examples: list[dict[str, object]] = []
    evaluated_users = 0

    for user_id, target_list in targets_by_user.items():
        user_id = int(user_id)
        seen_videos = set(user_histories.get(user_id, {}))
        if not seen_videos:
            continue

        relevant_items = {int(video_id) for video_id in target_list}
        target_categories = {video_category.get(item_id) for item_id in relevant_items}
        candidate_scores = build_multisource_scores(
            user_id=user_id,
            seen_videos=seen_videos,
            user_histories=user_histories,
            user_norms=user_norms,
            item_users=item_users,
            item_neighbors=item_neighbors,
            user_preferences=user_preferences,
            user_channel_index=user_channel_index,
            category_top_videos=category_top_videos,
            recent_category_top_videos=recent_category_top_videos,
            channel_top_videos=channel_top_videos,
            global_recent=global_recent,
        )
        ranked_items = top_items(candidate_scores, FINAL_TOP_K)
        if not ranked_items:
            continue

        evaluated_users += 1
        metrics["mean_candidates_returned"] += len(ranked_items)

        top10 = ranked_items[:10]
        top50 = ranked_items[:50]
        hit10 = any(item_id in relevant_items for item_id in top10)
        hit50 = any(item_id in relevant_items for item_id in top50)
        category_hit50 = any(video_category.get(item_id) in target_categories for item_id in top50)

        reciprocal_rank = 0.0
        for rank_index, item_id in enumerate(top10, start=1):
            if item_id in relevant_items:
                reciprocal_rank = 1.0 / rank_index
                break

        dcg = 0.0
        for rank_index, item_id in enumerate(top10, start=1):
            if item_id in relevant_items:
                dcg += 1.0 / math.log2(rank_index + 1)
        ideal_dcg = sum(1.0 / math.log2(index + 1) for index in range(2, min(len(relevant_items), 10) + 2))

        metrics["hit@10"] += float(hit10)
        metrics["hit@50"] += float(hit50)
        metrics["category_hit@50"] += float(category_hit50)
        metrics["mrr@10"] += reciprocal_rank
        metrics["ndcg@10"] += (dcg / ideal_dcg) if ideal_dcg > 0 else 0.0

        examples.append(
            {
                "user_id": user_id,
                "targets": ",".join(map(str, sorted(relevant_items))),
                "top10_candidates": ",".join(map(str, top10)),
                "target_categories": ",".join(sorted({str(category) for category in target_categories if category is not None})),
            }
        )

    if evaluated_users == 0:
        raise ValueError("No hay usuarios evaluables para la retrieval multi-source.")

    final_metrics = {
        key: round(value / evaluated_users, 6)
        for key, value in metrics.items()
        if key != "mean_candidates_returned"
    }
    final_metrics["mean_candidates_returned"] = round(metrics["mean_candidates_returned"] / evaluated_users, 6)
    final_metrics["evaluated_users"] = evaluated_users
    return final_metrics, pd.DataFrame(examples)


def main() -> None:
    cf_df, video_features, user_features, user_channel_features = load_inputs()
    train_df = cf_df.loc[cf_df["split"].eq("train")].copy()
    valid_df = cf_df.loc[cf_df["split"].eq("validation")].copy()
    test_df = cf_df.loc[cf_df["split"].eq("test")].copy()

    user_histories, user_norms, item_users, item_norms = build_user_histories(train_df)
    item_neighbors = build_item_neighbors(
        user_histories=user_histories,
        item_norms=item_norms,
        top_k=TOP_ITEM_NEIGHBORS,
    )
    category_top_videos, recent_category_top_videos, channel_top_videos, global_recent = build_video_indexes(video_features)
    user_preferences = build_user_preference_index(user_features)
    user_channel_index = build_user_channel_index(user_channel_features)

    valid_metrics, valid_examples = evaluate_multisource(
        split_df=valid_df,
        video_features=video_features,
        user_histories=user_histories,
        user_norms=user_norms,
        item_users=item_users,
        item_neighbors=item_neighbors,
        user_preferences=user_preferences,
        user_channel_index=user_channel_index,
        category_top_videos=category_top_videos,
        recent_category_top_videos=recent_category_top_videos,
        channel_top_videos=channel_top_videos,
        global_recent=global_recent,
    )
    test_metrics, test_examples = evaluate_multisource(
        split_df=test_df,
        video_features=video_features,
        user_histories=user_histories,
        user_norms=user_norms,
        item_users=item_users,
        item_neighbors=item_neighbors,
        user_preferences=user_preferences,
        user_channel_index=user_channel_index,
        category_top_videos=category_top_videos,
        recent_category_top_videos=recent_category_top_videos,
        channel_top_videos=channel_top_videos,
        global_recent=global_recent,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "retrieval_model.pkl", "wb") as file_handle:
        pickle.dump(
            {
                "weights": {
                    "user_cf": USER_CF_WEIGHT,
                    "item_cf": ITEM_CF_WEIGHT,
                    "category": CATEGORY_WEIGHT,
                    "channel": CHANNEL_WEIGHT,
                    "freshness": FRESHNESS_WEIGHT,
                },
                "category_top_videos": category_top_videos,
                "recent_category_top_videos": recent_category_top_videos,
                "channel_top_videos": channel_top_videos,
                "global_recent": global_recent,
                "user_preferences": user_preferences,
                "user_channel_index": user_channel_index,
                "cf_model": {
                    "user_histories": user_histories,
                    "user_norms": user_norms,
                    "item_users": item_users,
                    "item_neighbors": item_neighbors,
                },
            },
            file_handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    metrics = {
        "valid": valid_metrics,
        "test": test_metrics,
        "notes": [
            "Retrieval multi-source para home: mezcla CF, categoria reciente, canales activos y frescura.",
            "La metrica exacta por item sigue siendo dura; category_hit@50 es la lectura mas util para home.",
        ],
    }
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)

    valid_examples.head(200).to_csv(OUTPUT_DIR / "valid_examples.csv", index=False)
    test_examples.head(200).to_csv(OUTPUT_DIR / "test_examples.csv", index=False)

    print("Multisource retrieval completed")
    print(f"  output: {OUTPUT_DIR}")
    print("  valid metrics:")
    for key, value in valid_metrics.items():
        print(f"    {key}: {value}")
    print("  test metrics:")
    for key, value in test_metrics.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
