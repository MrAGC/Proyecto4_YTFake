from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


RANKING_DATASET_PATH = Path("reco_output_v2/ranking_dataset.csv")
USER_FEATURES_PATH = Path("reco_output_v2/user_features.csv")
VIDEO_FEATURES_PATH = Path("reco_output_v2/video_features.csv")
USER_CHANNEL_FEATURES_PATH = Path("reco_output_v2/user_channel_features.csv")
USERS_PATH = Path("data/usuarios.csv")
FOLLOWS_PATH = Path("data/seguimientos_canales.csv")

OUTPUT_DATASET_PATH = Path("reco_output_v2/training_dataset_balanced_v1.csv")
OUTPUT_METADATA_PATH = Path("reco_output_v2/training_dataset_balanced_v1_metadata.json")

TARGET_POSITIVE_ROWS = 300_000
RANDOM_SEED = 42

CATEGORICAL_COLUMNS = [
    "surface",
    "user_favorite_category",
    "user_favorite_device",
    "user_favorite_time_slot",
    "user_recent_favorite_category",
    "video_category",
]

USER_FEATURE_COLUMNS = [
    "user_total_interactions",
    "user_unique_videos",
    "user_unique_categories",
    "user_unique_creators",
    "user_avg_watch_percent",
    "user_completion_rate",
    "user_like_rate",
    "user_comment_rate",
    "user_subscription_rate",
    "user_recommended_share",
    "user_avg_implicit_score",
    "user_avg_video_freshness",
    "user_active_days",
    "user_recommendation_clicks",
    "user_recommendation_impressions",
    "user_click_from_reco_rate",
    "user_favorite_category",
    "user_favorite_device",
    "user_favorite_time_slot",
    "user_recent_favorite_category",
    "user_recent_interactions_30d",
    "user_following_channels",
    "user_active_window_days",
]

VIDEO_FEATURE_COLUMNS = [
    "video_category",
    "video_duration_s_clean",
    "total_views",
    "total_likes",
    "total_comments",
    "suscriptores_ganados",
    "veces_recomendado",
    "total_clics",
    "video_like_rate",
    "video_comment_rate",
    "video_subscription_rate",
    "video_avg_watch_percent",
    "video_ctr",
    "video_popularity_log",
    "video_engagement_score",
    "video_age_days",
    "video_freshness_score",
    "is_recent_upload",
    "video_discovery_score",
    "has_title_metadata",
    "has_keyword_metadata",
    "channel_followers_total",
    "channel_total_videos",
    "channel_views_per_video",
    "channel_engagement_score",
]

USER_CHANNEL_FEATURE_COLUMNS = [
    "channel_recent_interactions_30d",
    "channel_recent_watch_percent_30d",
    "channel_recent_implicit_score_30d",
    "days_since_last_channel_watch",
    "user_follows_channel",
    "stale_follow_flag",
    "channel_current_interest_score",
]

DERIVED_FEATURE_COLUMNS = [
    "category_matches_user_pref",
    "recent_category_match",
    "creator_matches_user_pref",
    "recent_creator_match",
    "channel_matches_user_pref",
    "recent_channel_match",
    "item_vs_user_score_gap",
    "freshness_vs_user_avg",
]

OUTPUT_COLUMNS = [
    "user_id",
    "video_id",
    "channel_id",
    "creator_user_id",
    "anchor_timestamp",
    "split",
    "surface",
    "example_source",
    "label",
    *USER_FEATURE_COLUMNS,
    *VIDEO_FEATURE_COLUMNS,
    *USER_CHANNEL_FEATURE_COLUMNS,
    *DERIVED_FEATURE_COLUMNS,
]


def split_by_time(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values("last_timestamp").reset_index(drop=True)
    n_rows = len(ordered)
    train_end = int(n_rows * 0.80)
    valid_end = int(n_rows * 0.90)

    ordered["split"] = "train"
    ordered.loc[train_end:valid_end - 1, "split"] = "valid"
    ordered.loc[valid_end:, "split"] = "test"
    return ordered


def stratified_sample_by_split(df: pd.DataFrame, target_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    sampled_parts: list[pd.DataFrame] = []
    total_rows = len(df)
    for split_name, split_df in df.groupby("split", sort=False):
        split_target = int(round(target_rows * len(split_df) / total_rows))
        split_target = min(len(split_df), max(split_target, 1))
        indices = rng.choice(split_df.index.to_numpy(), size=split_target, replace=False)
        sampled_parts.append(split_df.loc[indices])
    return pd.concat(sampled_parts, ignore_index=True)


def build_observed_dataset(ranking_df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ranking_df = split_by_time(ranking_df)
    positives = ranking_df.loc[ranking_df["ranking_label"].eq(1)].copy()
    negatives = ranking_df.loc[ranking_df["ranking_label"].eq(0)].copy()

    sampled_positives = stratified_sample_by_split(positives, TARGET_POSITIVE_ROWS, rng)

    observed_positive_rows = sampled_positives[OUTPUT_COLUMNS].copy() if set(OUTPUT_COLUMNS).issubset(sampled_positives.columns) else None
    if observed_positive_rows is None:
        observed_positive_rows = sampled_positives.copy()

    observed_negative_rows = negatives.copy()

    return sampled_positives, observed_negative_rows, ranking_df


def prepare_observed_rows(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    prepared = df.copy()
    prepared = prepared.rename(columns={"last_timestamp": "anchor_timestamp", "ranking_label": "label"})
    prepared["surface"] = "home"
    prepared["example_source"] = source_name
    missing_columns = [col for col in OUTPUT_COLUMNS if col not in prepared.columns]
    for column in missing_columns:
        prepared[column] = 0
    prepared = prepared[OUTPUT_COLUMNS].copy()
    return prepared


def build_seen_videos(users_path: Path) -> dict[int, set[int]]:
    users_df = pd.read_csv(users_path, usecols=["user_id", "video_id"])
    seen: dict[int, set[int]] = {}
    for row in users_df.itertuples(index=False):
        seen.setdefault(int(row.user_id), set()).add(int(row.video_id))
    return seen


def load_feature_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[tuple[int, int]]]:
    user_features = pd.read_csv(USER_FEATURES_PATH)
    video_features = pd.read_csv(VIDEO_FEATURES_PATH)
    user_channel_features = pd.read_csv(USER_CHANNEL_FEATURES_PATH)
    follows = pd.read_csv(FOLLOWS_PATH, usecols=["user_id", "channel_id"])
    follow_pairs = {(int(row.user_id), int(row.channel_id)) for row in follows.itertuples(index=False)}
    return user_features, video_features, user_channel_features, follow_pairs


def build_lookup_tables(
    user_features: pd.DataFrame,
    video_features: pd.DataFrame,
    user_channel_features: pd.DataFrame,
) -> tuple[dict[int, dict], dict[int, dict], dict[tuple[int, int], dict], dict[str, np.ndarray], np.ndarray]:
    user_lookup = user_features.set_index("user_id").to_dict(orient="index")
    video_lookup = video_features.set_index("video_id").to_dict(orient="index")
    user_channel_lookup = user_channel_features.set_index(["user_id", "channel_id"]).to_dict(orient="index")

    video_df = video_features[["video_id", "video_category", "is_recent_upload"]].copy()
    category_pools = {
        str(category): group["video_id"].to_numpy(dtype=np.int64)
        for category, group in video_df.groupby("video_category", sort=False)
    }
    recent_pool = video_df.loc[video_df["is_recent_upload"].eq(1), "video_id"].to_numpy(dtype=np.int64)
    all_videos = video_df["video_id"].to_numpy(dtype=np.int64)

    return user_lookup, video_lookup, user_channel_lookup, category_pools, recent_pool if len(recent_pool) else all_videos


def choose_candidate_video(
    user_seen: set[int],
    preferred_categories: list[str],
    category_pools: dict[str, np.ndarray],
    recent_pool: np.ndarray,
    all_videos: np.ndarray,
    rng: np.random.Generator,
) -> int:
    candidate_pools = []
    for category in preferred_categories:
        if category and category in category_pools:
            candidate_pools.append(category_pools[category])
    candidate_pools.append(recent_pool)
    candidate_pools.append(all_videos)

    for pool in candidate_pools:
        if len(pool) == 0:
            continue
        for _ in range(20):
            candidate = int(pool[rng.integers(0, len(pool))])
            if candidate not in user_seen:
                return candidate

    for candidate in all_videos:
        candidate = int(candidate)
        if candidate not in user_seen:
            return candidate

    raise RuntimeError("No he encontrado un video no visto para construir un negativo sintetico.")


def build_synthetic_negative_row(
    anchor_row: pd.Series,
    candidate_video_id: int,
    user_lookup: dict[int, dict],
    video_lookup: dict[int, dict],
    user_channel_lookup: dict[tuple[int, int], dict],
    follow_pairs: set[tuple[int, int]],
) -> dict:
    user_id = int(anchor_row["user_id"])
    user_row = user_lookup[user_id]
    video_row = video_lookup[candidate_video_id]
    channel_id = int(video_row["channel_id"])
    creator_user_id = int(video_row["creator_user_id"])

    user_channel_row = user_channel_lookup.get((user_id, channel_id), {})
    follows_channel = 1 if (user_id, channel_id) in follow_pairs else int(user_channel_row.get("user_follows_channel", 0))

    row = {
        "user_id": user_id,
        "video_id": int(candidate_video_id),
        "channel_id": channel_id,
        "creator_user_id": creator_user_id,
        "anchor_timestamp": anchor_row["last_timestamp"],
        "split": anchor_row["split"],
        "surface": "home",
        "example_source": "synthetic_negative",
        "label": 0,
    }

    for column in USER_FEATURE_COLUMNS:
        row[column] = user_row.get(column, "unknown" if column in CATEGORICAL_COLUMNS else 0)

    for column in VIDEO_FEATURE_COLUMNS:
        row[column] = video_row.get(column, 0)

    row["channel_recent_interactions_30d"] = int(user_channel_row.get("channel_recent_interactions_30d", 0))
    row["channel_recent_watch_percent_30d"] = float(user_channel_row.get("channel_recent_watch_percent_30d", 0))
    row["channel_recent_implicit_score_30d"] = float(user_channel_row.get("channel_recent_implicit_score_30d", 0))
    row["days_since_last_channel_watch"] = int(user_channel_row.get("days_since_last_channel_watch", 999))
    row["user_follows_channel"] = follows_channel
    row["stale_follow_flag"] = int(user_channel_row.get("stale_follow_flag", 1 if follows_channel else 0))
    row["channel_current_interest_score"] = float(user_channel_row.get("channel_current_interest_score", 0))

    row["category_matches_user_pref"] = int(video_row["video_category"] == user_row.get("user_favorite_category", "unknown"))
    row["recent_category_match"] = int(video_row["video_category"] == user_row.get("user_recent_favorite_category", "unknown"))
    row["creator_matches_user_pref"] = 0
    row["recent_creator_match"] = 0
    row["channel_matches_user_pref"] = 0
    row["recent_channel_match"] = 0
    row["item_vs_user_score_gap"] = round(
        float(video_row.get("video_engagement_score", 0)) - float(user_row.get("user_avg_implicit_score", 0)),
        4,
    )
    row["freshness_vs_user_avg"] = round(
        float(video_row.get("video_freshness_score", 0)) - float(user_row.get("user_avg_video_freshness", 0)),
        4,
    )

    return row


def build_synthetic_negatives(
    positive_anchors: pd.DataFrame,
    observed_negative_count: int,
    user_lookup: dict[int, dict],
    video_lookup: dict[int, dict],
    user_channel_lookup: dict[tuple[int, int], dict],
    category_pools: dict[str, np.ndarray],
    recent_pool: np.ndarray,
    all_videos: np.ndarray,
    seen_videos: dict[int, set[int]],
    follow_pairs: set[tuple[int, int]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    target_negative_rows = len(positive_anchors)
    synthetic_needed = max(target_negative_rows - observed_negative_count, 0)
    synthetic_anchors = positive_anchors.sample(
        n=synthetic_needed,
        replace=synthetic_needed > len(positive_anchors),
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    synthetic_rows: list[dict] = []
    for anchor in synthetic_anchors.itertuples(index=False):
        anchor_series = pd.Series(anchor._asdict())
        user_id = int(anchor_series["user_id"])
        user_seen = seen_videos.get(user_id, set())
        preferred_categories = [
            str(anchor_series.get("user_recent_favorite_category", "unknown")),
            str(anchor_series.get("user_favorite_category", "unknown")),
            str(anchor_series.get("video_category", "unknown")),
        ]
        candidate_video_id = choose_candidate_video(
            user_seen=user_seen,
            preferred_categories=preferred_categories,
            category_pools=category_pools,
            recent_pool=recent_pool,
            all_videos=all_videos,
            rng=rng,
        )
        synthetic_rows.append(
            build_synthetic_negative_row(
                anchor_row=anchor_series,
                candidate_video_id=candidate_video_id,
                user_lookup=user_lookup,
                video_lookup=video_lookup,
                user_channel_lookup=user_channel_lookup,
                follow_pairs=follow_pairs,
            )
        )

    return pd.DataFrame(synthetic_rows, columns=OUTPUT_COLUMNS)


def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = 0

    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].fillna("unknown").astype(str)

    for column in OUTPUT_COLUMNS:
        if column in CATEGORICAL_COLUMNS or column in {"anchor_timestamp", "split", "surface", "example_source"}:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    df = df[OUTPUT_COLUMNS].copy()
    return df


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    ranking_df = pd.read_csv(RANKING_DATASET_PATH)
    ranking_df["last_timestamp"] = pd.to_datetime(ranking_df["last_timestamp"], errors="coerce")
    if ranking_df["last_timestamp"].isna().all():
        raise ValueError("No se pudo parsear `last_timestamp` en ranking_dataset.csv.")

    sampled_positives, observed_negatives, split_ranking_df = build_observed_dataset(ranking_df, rng)
    prepared_positives = prepare_observed_rows(sampled_positives, "observed_positive")
    prepared_observed_negatives = prepare_observed_rows(observed_negatives, "observed_negative")

    user_features, video_features, user_channel_features, follow_pairs = load_feature_tables()
    user_lookup, video_lookup, user_channel_lookup, category_pools, recent_pool = build_lookup_tables(
        user_features=user_features,
        video_features=video_features,
        user_channel_features=user_channel_features,
    )
    all_videos = video_features["video_id"].to_numpy(dtype=np.int64)
    seen_videos = build_seen_videos(USERS_PATH)

    synthetic_negatives = build_synthetic_negatives(
        positive_anchors=sampled_positives,
        observed_negative_count=len(prepared_observed_negatives),
        user_lookup=user_lookup,
        video_lookup=video_lookup,
        user_channel_lookup=user_channel_lookup,
        category_pools=category_pools,
        recent_pool=recent_pool,
        all_videos=all_videos,
        seen_videos=seen_videos,
        follow_pairs=follow_pairs,
        rng=rng,
    )

    final_df = pd.concat(
        [
            prepared_positives,
            prepared_observed_negatives,
            synthetic_negatives,
        ],
        ignore_index=True,
    )
    final_df = finalize_dataset(final_df)
    final_df = final_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    OUTPUT_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_DATASET_PATH, index=False)

    metadata = {
        "rows": int(len(final_df)),
        "positive_rows": int(final_df["label"].sum()),
        "negative_rows": int(len(final_df) - final_df["label"].sum()),
        "positive_rate": round(float(final_df["label"].mean()), 6),
        "observed_positive_rows": int(len(prepared_positives)),
        "observed_negative_rows": int(len(prepared_observed_negatives)),
        "synthetic_negative_rows": int(len(synthetic_negatives)),
        "categorical_columns": CATEGORICAL_COLUMNS,
        "feature_columns": [col for col in OUTPUT_COLUMNS if col not in {"user_id", "video_id", "channel_id", "creator_user_id", "anchor_timestamp", "split", "example_source", "label"}],
        "notes": [
            "Este dataset balancea observados y negativos sinteticos plausibles.",
            "Los negativos sinteticos se construyen con videos no vistos del usuario, preferiblemente en categorias afines o recientes.",
            "Sigue siendo una aproximacion offline; no garantiza point-in-time safety completa.",
        ],
    }

    with open(OUTPUT_METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print("Balanced training dataset built")
    print(f"  rows: {metadata['rows']}")
    print(f"  positive_rows: {metadata['positive_rows']}")
    print(f"  negative_rows: {metadata['negative_rows']}")
    print(f"  positive_rate: {metadata['positive_rate']}")
    print(f"  output: {OUTPUT_DATASET_PATH}")


if __name__ == "__main__":
    main()
