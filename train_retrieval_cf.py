from __future__ import annotations

import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


CF_DATASET_PATH = Path("reco_output_v2/cf_interactions.csv")
OUTPUT_DIR = Path("models/retrieval_cf_v1")

TOP_USER_NEIGHBORS = 40
TOP_ITEM_NEIGHBORS = 80
TOP_RECOMMENDATIONS = 50
SECONDARY_ITEM_WEIGHT = 0.35


def load_cf_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No encuentro el dataset CF: {path}")

    df = pd.read_csv(path)
    for column in ["implicit_score_sum", "implicit_score_mean", "cf_confidence"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return df


def build_user_histories(train_df: pd.DataFrame) -> tuple[dict[int, dict[int, float]], dict[int, float], dict[int, list[tuple[int, float]]], dict[int, float]]:
    user_histories: dict[int, dict[int, float]] = {}
    item_users: dict[int, list[tuple[int, float]]] = defaultdict(list)
    user_norm_squares: dict[int, float] = defaultdict(float)
    item_norm_squares: dict[int, float] = defaultdict(float)

    for row in train_df.itertuples(index=False):
        user_id = int(row.user_id)
        video_id = int(row.video_id)
        weight = float(row.implicit_score_sum)

        user_histories.setdefault(user_id, {})[video_id] = weight
        item_users[video_id].append((user_id, weight))
        user_norm_squares[user_id] += weight * weight
        item_norm_squares[video_id] += weight * weight

    user_norms = {
        user_id: math.sqrt(norm_sq)
        for user_id, norm_sq in user_norm_squares.items()
        if norm_sq > 0
    }
    item_norms = {
        item_id: math.sqrt(norm_sq)
        for item_id, norm_sq in item_norm_squares.items()
        if norm_sq > 0
    }
    return user_histories, user_norms, dict(item_users), item_norms


def build_item_neighbors(
    user_histories: dict[int, dict[int, float]],
    item_norms: dict[int, float],
    top_k: int,
) -> dict[int, list[tuple[float, int]]]:
    pair_dot_products: dict[tuple[int, int], float] = defaultdict(float)

    for history in user_histories.values():
        items = list(history.items())
        n_items = len(items)
        for left_idx in range(n_items):
            item_a, weight_a = items[left_idx]
            for right_idx in range(left_idx + 1, n_items):
                item_b, weight_b = items[right_idx]
                if item_a < item_b:
                    pair_key = (item_a, item_b)
                else:
                    pair_key = (item_b, item_a)
                pair_dot_products[pair_key] += weight_a * weight_b

    item_neighbors: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for (item_a, item_b), dot_product in pair_dot_products.items():
        denom = item_norms.get(item_a, 0.0) * item_norms.get(item_b, 0.0)
        if denom <= 0:
            continue
        similarity = dot_product / denom
        if similarity <= 0:
            continue
        item_neighbors[item_a].append((similarity, item_b))
        item_neighbors[item_b].append((similarity, item_a))

    for item_id, neighbors in item_neighbors.items():
        neighbors.sort(key=lambda pair: pair[0], reverse=True)
        item_neighbors[item_id] = neighbors[:top_k]

    return dict(item_neighbors)


def user_based_scores(
    user_id: int,
    user_histories: dict[int, dict[int, float]],
    user_norms: dict[int, float],
    item_users: dict[int, list[tuple[int, float]]],
    top_neighbors: int,
) -> dict[int, float]:
    user_history = user_histories.get(user_id)
    if not user_history:
        return {}

    query_norm = user_norms.get(user_id, 0.0)
    if query_norm <= 0:
        return {}

    shared_dot_products: dict[int, float] = defaultdict(float)
    seen_items = set(user_history)

    for item_id, query_weight in user_history.items():
        for neighbor_user_id, neighbor_weight in item_users.get(item_id, []):
            if neighbor_user_id == user_id:
                continue
            shared_dot_products[neighbor_user_id] += query_weight * neighbor_weight

    if not shared_dot_products:
        return {}

    neighbor_similarities: list[tuple[float, int]] = []
    for neighbor_user_id, dot_product in shared_dot_products.items():
        denom = query_norm * user_norms.get(neighbor_user_id, 0.0)
        if denom <= 0:
            continue
        similarity = dot_product / denom
        if similarity > 0:
            neighbor_similarities.append((similarity, neighbor_user_id))

    if not neighbor_similarities:
        return {}

    neighbor_similarities.sort(key=lambda pair: pair[0], reverse=True)
    candidate_scores: dict[int, float] = defaultdict(float)

    for similarity, neighbor_user_id in neighbor_similarities[:top_neighbors]:
        for candidate_video_id, candidate_weight in user_histories[neighbor_user_id].items():
            if candidate_video_id in seen_items:
                continue
            candidate_scores[candidate_video_id] += similarity * candidate_weight

    return dict(candidate_scores)


def item_based_scores(
    user_id: int,
    user_histories: dict[int, dict[int, float]],
    item_neighbors: dict[int, list[tuple[float, int]]],
) -> dict[int, float]:
    user_history = user_histories.get(user_id)
    if not user_history:
        return {}

    seen_items = set(user_history)
    candidate_scores: dict[int, float] = defaultdict(float)

    for item_id, history_weight in user_history.items():
        for similarity, candidate_video_id in item_neighbors.get(item_id, []):
            if candidate_video_id in seen_items:
                continue
            candidate_scores[candidate_video_id] += similarity * history_weight

    return dict(candidate_scores)


def merge_scores(
    primary_scores: dict[int, float],
    secondary_scores: dict[int, float],
    secondary_weight: float,
) -> dict[int, float]:
    if not primary_scores and not secondary_scores:
        return {}

    combined_scores: dict[int, float] = defaultdict(float)
    if primary_scores:
        primary_max = max(primary_scores.values()) or 1.0
        for item_id, score in primary_scores.items():
            combined_scores[item_id] += score / primary_max

    if secondary_scores:
        secondary_max = max(secondary_scores.values()) or 1.0
        for item_id, score in secondary_scores.items():
            combined_scores[item_id] += secondary_weight * (score / secondary_max)

    return dict(combined_scores)


def top_items(scores: dict[int, float], top_k: int) -> list[int]:
    if not scores:
        return []
    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [item_id for item_id, _ in ordered[:top_k]]


def dcg_at_k(recommended_items: list[int], relevant_items: set[int], k: int) -> float:
    gain = 0.0
    for index, item_id in enumerate(recommended_items[:k], start=1):
        if item_id in relevant_items:
            gain += 1.0 / math.log2(index + 1)
    return gain


def evaluate_split(
    split_df: pd.DataFrame,
    user_histories: dict[int, dict[int, float]],
    user_norms: dict[int, float],
    item_users: dict[int, list[tuple[int, float]]],
    item_neighbors: dict[int, list[tuple[float, int]]],
    catalog_size: int,
) -> tuple[dict[str, dict[str, float | int]], pd.DataFrame]:
    targets_by_user = split_df.groupby("user_id", sort=False)["video_id"].agg(list)
    models = {
        "user_based_cf": {"hit@10": 0.0, "hit@50": 0.0, "recall@10": 0.0, "recall@50": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0},
        "item_based_cf": {"hit@10": 0.0, "hit@50": 0.0, "recall@10": 0.0, "recall@50": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0},
        "hybrid_cf": {"hit@10": 0.0, "hit@50": 0.0, "recall@10": 0.0, "recall@50": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0},
    }
    unique_recommended_items = {model_name: set() for model_name in models}
    debug_rows: list[dict[str, object]] = []

    evaluated_users = 0
    for user_id, target_list in targets_by_user.items():
        user_id = int(user_id)
        if user_id not in user_histories:
            continue

        relevant_items = {int(video_id) for video_id in target_list}
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
        for model_name, recommended in recommendations.items():
            top10 = recommended[:10]
            top50 = recommended[:50]
            hits10 = len(relevant_items.intersection(top10))
            hits50 = len(relevant_items.intersection(top50))

            models[model_name]["hit@10"] += float(hits10 > 0)
            models[model_name]["hit@50"] += float(hits50 > 0)
            models[model_name]["recall@10"] += hits10 / max(len(relevant_items), 1)
            models[model_name]["recall@50"] += hits50 / max(len(relevant_items), 1)
            models[model_name]["ndcg@10"] += dcg_at_k(recommended, relevant_items, k=10)

            reciprocal_rank = 0.0
            for rank_index, item_id in enumerate(top10, start=1):
                if item_id in relevant_items:
                    reciprocal_rank = 1.0 / rank_index
                    break
            models[model_name]["mrr@10"] += reciprocal_rank
            unique_recommended_items[model_name].update(top50)

        debug_rows.append(
            {
                "user_id": user_id,
                "targets": ",".join(map(str, sorted(relevant_items))),
                "user_based_cf_top10": ",".join(map(str, recommendations["user_based_cf"][:10])),
                "item_based_cf_top10": ",".join(map(str, recommendations["item_based_cf"][:10])),
                "hybrid_cf_top10": ",".join(map(str, recommendations["hybrid_cf"][:10])),
            }
        )

    if evaluated_users == 0:
        raise ValueError("No hay usuarios evaluables en este split para retrieval.")

    output_metrics: dict[str, dict[str, float | int]] = {}
    for model_name, metric_dict in models.items():
        output_metrics[model_name] = {
            metric_name: round(metric_value / evaluated_users, 6)
            for metric_name, metric_value in metric_dict.items()
        }
        output_metrics[model_name]["evaluated_users"] = evaluated_users
        output_metrics[model_name]["coverage@50"] = round(len(unique_recommended_items[model_name]) / max(catalog_size, 1), 6)

    debug_df = pd.DataFrame(debug_rows)
    return output_metrics, debug_df


def main() -> None:
    cf_df = load_cf_dataset(CF_DATASET_PATH)
    train_df = cf_df.loc[cf_df["split"].eq("train")].copy()
    valid_df = cf_df.loc[cf_df["split"].eq("validation")].copy()
    test_df = cf_df.loc[cf_df["split"].eq("test")].copy()

    user_histories, user_norms, item_users, item_norms = build_user_histories(train_df)
    item_neighbors = build_item_neighbors(
        user_histories=user_histories,
        item_norms=item_norms,
        top_k=TOP_ITEM_NEIGHBORS,
    )

    valid_metrics, valid_debug = evaluate_split(
        split_df=valid_df,
        user_histories=user_histories,
        user_norms=user_norms,
        item_users=item_users,
        item_neighbors=item_neighbors,
        catalog_size=int(cf_df["video_id"].nunique()),
    )
    test_metrics, test_debug = evaluate_split(
        split_df=test_df,
        user_histories=user_histories,
        user_norms=user_norms,
        item_users=item_users,
        item_neighbors=item_neighbors,
        catalog_size=int(cf_df["video_id"].nunique()),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "retrieval_model.pkl", "wb") as file_handle:
        pickle.dump(
            {
                "user_histories": user_histories,
                "user_norms": user_norms,
                "item_users": item_users,
                "item_neighbors": item_neighbors,
                "config": {
                    "top_user_neighbors": TOP_USER_NEIGHBORS,
                    "top_item_neighbors": TOP_ITEM_NEIGHBORS,
                    "top_recommendations": TOP_RECOMMENDATIONS,
                    "secondary_item_weight": SECONDARY_ITEM_WEIGHT,
                },
            },
            file_handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    metrics = {
        "valid": valid_metrics,
        "test": test_metrics,
        "dataset": {
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "test_rows": int(len(test_df)),
            "train_users": int(train_df["user_id"].nunique()),
            "train_videos": int(train_df["video_id"].nunique()),
        },
        "notes": [
            "User-based CF es la retrieval principal.",
            "Item-based CF es la retrieval secundaria para rellenar y dar continuidad.",
            "La variante hybrid_cf usa score principal mas refuerzo del item-based.",
        ],
    }

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)

    valid_debug.head(200).to_csv(OUTPUT_DIR / "valid_recommendation_examples.csv", index=False)
    test_debug.head(200).to_csv(OUTPUT_DIR / "test_recommendation_examples.csv", index=False)

    print("Retrieval training completed")
    print(f"  train_rows: {len(train_df)}")
    print(f"  valid_rows: {len(valid_df)}")
    print(f"  test_rows : {len(test_df)}")
    print(f"  output    : {OUTPUT_DIR}")
    for split_name, split_metrics in (("valid", valid_metrics), ("test", test_metrics)):
        print()
        print(split_name)
        for model_name, model_metrics in split_metrics.items():
            print(f"  {model_name}:")
            for metric_name, metric_value in model_metrics.items():
                print(f"    {metric_name}: {metric_value}")


if __name__ == "__main__":
    main()
