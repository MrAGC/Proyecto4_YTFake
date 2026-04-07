from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
CURRENT_DATA_DIR = DATA_DIR / "actuales"
OUTPUT_DIR = Path("reco_output")
FALLBACK_OUTPUT_DIR = Path("reco_output_v2")

USERS_PATH = CURRENT_DATA_DIR / "usuarios.csv"
VIDEOS_PATH = CURRENT_DATA_DIR / "videos.csv"
VIDEOS_COMPLETE_PATH = CURRENT_DATA_DIR / "videos_completo.csv"
NEW_VIDEOS_PATH = CURRENT_DATA_DIR / "videos_nuevos.csv"
CHANNELS_PATH = CURRENT_DATA_DIR / "canales.csv"
FOLLOWS_PATH = CURRENT_DATA_DIR / "seguimientos_canales.csv"


def safe_mode(series: pd.Series, default: str = "unknown") -> str:
    mode = series.dropna().mode()
    if mode.empty:
        return default
    return str(mode.iloc[0])


def safe_mode_int(series: pd.Series, default: int = 0) -> int:
    mode = series.dropna().mode()
    if mode.empty:
        return default
    return int(mode.iloc[0])


def parse_date_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    users = pd.read_csv(USERS_PATH)
    users = parse_date_columns(users, ["timestamp", "published_at"])

    videos = pd.read_csv(VIDEOS_PATH)
    videos = parse_date_columns(videos, ["published_at", "first_interaction_at", "last_interaction_at"])

    if NEW_VIDEOS_PATH.exists():
        new_videos = pd.read_csv(NEW_VIDEOS_PATH)
        new_videos = parse_date_columns(new_videos, ["published_at", "first_interaction_at", "last_interaction_at"])
        videos = pd.concat([videos, new_videos], ignore_index=True, sort=False)

    if VIDEOS_COMPLETE_PATH.exists():
        complete_videos = pd.read_csv(VIDEOS_COMPLETE_PATH)
        complete_videos = complete_videos[["video_id", "titulo", "que_pasa"]].copy()
        videos = videos.drop(columns=[col for col in ["titulo", "que_pasa"] if col in videos.columns])
        videos = videos.merge(complete_videos, on="video_id", how="left")

    channels = pd.read_csv(CHANNELS_PATH) if CHANNELS_PATH.exists() else pd.DataFrame()
    channels = parse_date_columns(
        channels,
        ["channel_created_at", "first_publish_at", "last_publish_at"],
    )

    follows = pd.read_csv(FOLLOWS_PATH) if FOLLOWS_PATH.exists() else pd.DataFrame()
    follows = parse_date_columns(follows, ["followed_at"])

    return users, videos, channels, follows


def build_creator_features(channels: pd.DataFrame) -> pd.DataFrame:
    if channels.empty:
        return pd.DataFrame(
            columns=[
                "channel_id",
                "creator_user_id",
                "channel_primary_category",
                "channel_total_videos",
                "channel_total_views",
                "channel_total_likes",
                "channel_total_comments",
                "channel_total_subscriptions_generated",
                "channel_followers_total",
                "channel_views_per_video",
                "channel_like_rate_proxy",
                "channel_engagement_score",
            ]
        )

    df = channels.copy()
    numeric_columns = [
        "total_videos",
        "total_views",
        "total_likes",
        "total_comments",
        "total_subscriptions_generated",
        "followers_total",
        "avg_video_watch_percent",
        "avg_video_ctr",
        "avg_video_like_rate",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    df["channel_views_per_video"] = np.divide(
        df["total_views"],
        df["total_videos"],
        out=np.zeros(len(df), dtype=float),
        where=df["total_videos"].to_numpy() > 0,
    ).round(4)
    df["channel_like_rate_proxy"] = np.divide(
        df["total_likes"],
        df["total_views"],
        out=np.zeros(len(df), dtype=float),
        where=df["total_views"].to_numpy() > 0,
    ).round(4)
    df["channel_engagement_score"] = (
        0.40 * df["avg_video_watch_percent"].clip(0, 1)
        + 0.25 * df["avg_video_like_rate"].clip(0, 1)
        + 0.15 * df["avg_video_ctr"].clip(0, 1)
        + 0.20 * np.tanh(np.log1p(df["followers_total"]) / 5)
    ).round(4)

    return df.rename(
        columns={
            "primary_category": "channel_primary_category",
            "total_videos": "channel_total_videos",
            "total_views": "channel_total_views",
            "total_likes": "channel_total_likes",
            "total_comments": "channel_total_comments",
            "total_subscriptions_generated": "channel_total_subscriptions_generated",
            "followers_total": "channel_followers_total",
        }
    )[
        [
            "channel_id",
            "creator_user_id",
            "channel_primary_category",
            "channel_total_videos",
            "channel_total_views",
            "channel_total_likes",
            "channel_total_comments",
            "channel_total_subscriptions_generated",
            "channel_followers_total",
            "channel_views_per_video",
            "channel_like_rate_proxy",
            "channel_engagement_score",
        ]
    ].copy()


def build_video_features(
    videos: pd.DataFrame,
    snapshot_timestamp: pd.Timestamp,
    creator_features: pd.DataFrame,
) -> pd.DataFrame:
    df = videos.copy()

    numeric_columns = [
        "channel_id",
        "creator_user_id",
        "video_duration_s",
        "total_views",
        "total_likes",
        "like_rate",
        "total_comments",
        "comment_rate",
        "suscriptores_ganados",
        "subscription_rate",
        "avg_watch_percent",
        "avg_watch_time_s",
        "veces_recomendado",
        "total_clics",
        "click_through_rate",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    valid_duration = df["video_duration_s"].between(1, 14_400)
    duration_fallback = int(df.loc[valid_duration, "video_duration_s"].median())
    df["video_duration_s_clean"] = (
        df["video_duration_s"]
        .where(valid_duration, duration_fallback)
        .fillna(duration_fallback)
        .round()
        .astype(int)
    )

    df["video_like_rate"] = df["like_rate"].fillna(0).clip(0, 1)
    df["video_comment_rate"] = df["comment_rate"].fillna(0).clip(0, 1)
    df["video_subscription_rate"] = df["subscription_rate"].fillna(0).clip(0, 1)
    df["video_avg_watch_percent"] = df["avg_watch_percent"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)
    df["video_ctr"] = df["click_through_rate"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)
    df["video_popularity_log"] = np.log1p(df["total_views"].fillna(0))

    df["video_engagement_score"] = (
        0.35 * df["video_avg_watch_percent"]
        + 0.25 * df["video_like_rate"]
        + 0.15 * df["video_comment_rate"]
        + 0.15 * df["video_subscription_rate"]
        + 0.10 * df["video_ctr"]
    ).round(4)

    if "published_at" in df.columns:
        video_age_days = (snapshot_timestamp.floor("D") - df["published_at"]).dt.days
        fallback_age = int(video_age_days.dropna().clip(lower=0).median()) if video_age_days.notna().any() else 180
        df["video_age_days"] = video_age_days.fillna(fallback_age).clip(lower=0).astype(int)
    else:
        df["video_age_days"] = 180

    df["video_freshness_score"] = np.exp(-df["video_age_days"] / 45).round(4)
    df["is_recent_upload"] = df["video_age_days"].le(14).astype(int)
    df["video_discovery_score"] = (
        0.75 * df["video_engagement_score"] + 0.25 * df["video_freshness_score"]
    ).round(4)

    df["video_category"] = df["category"].fillna("Unknown")
    df["channel_id"] = df["channel_id"].fillna(0).astype(int)
    df["creator_user_id"] = df["creator_user_id"].fillna(0).astype(int)
    df["has_title_metadata"] = df["titulo"].fillna("").str.strip().ne("").astype(int)
    df["has_keyword_metadata"] = df["que_pasa"].fillna("").str.strip().ne("").astype(int)

    if not creator_features.empty:
        df = df.merge(creator_features, on=["channel_id", "creator_user_id"], how="left")
    else:
        df["channel_followers_total"] = 0
        df["channel_total_videos"] = 0
        df["channel_views_per_video"] = 0.0
        df["channel_engagement_score"] = 0.0
        df["channel_primary_category"] = "unknown"

    fill_defaults = {
        "channel_followers_total": 0,
        "channel_total_videos": 0,
        "channel_views_per_video": 0.0,
        "channel_engagement_score": 0.0,
        "channel_primary_category": "unknown",
    }
    for column, default in fill_defaults.items():
        if column in df.columns:
            df[column] = df[column].fillna(default)

    return df[
        [
            "video_id",
            "channel_id",
            "creator_user_id",
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
    ].copy()


def clean_user_events(users: pd.DataFrame, video_features: pd.DataFrame) -> pd.DataFrame:
    df = users.copy()

    numeric_columns = [
        "tiempo_visto_s",
        "porcentaje_visto",
        "like",
        "comentario",
        "suscripcion",
        "fue_recomendado",
        "clic_recomendacion",
        "es_video_propio",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in ["like", "comentario", "suscripcion", "fue_recomendado", "clic_recomendacion", "es_video_propio"]:
        if column in df.columns:
            df[column] = df[column].fillna(0).clip(0, 1).astype(int)

    df = df.drop(columns=[col for col in ["channel_id", "creator_user_id", "video_category"] if col in df.columns])
    df = df.merge(
        video_features[
            [
                "video_id",
                "channel_id",
                "creator_user_id",
                "video_category",
                "video_duration_s_clean",
                "video_age_days",
                "video_freshness_score",
                "is_recent_upload",
            ]
        ],
        on="video_id",
        how="left",
    )

    df["tiempo_visto_s_clean"] = df["tiempo_visto_s"].fillna(0).clip(lower=0)
    valid_duration = df["video_duration_s_clean"].gt(0)
    df.loc[valid_duration, "tiempo_visto_s_clean"] = np.minimum(
        df.loc[valid_duration, "tiempo_visto_s_clean"],
        df.loc[valid_duration, "video_duration_s_clean"],
    )

    raw_watch_percent = df["porcentaje_visto"].replace([np.inf, -np.inf], np.nan)
    derived_watch_percent = np.where(
        valid_duration,
        df["tiempo_visto_s_clean"] / df["video_duration_s_clean"],
        np.nan,
    )

    df["watch_percent_clean"] = raw_watch_percent.where(raw_watch_percent.between(0, 1))
    df["watch_percent_clean"] = df["watch_percent_clean"].fillna(pd.Series(derived_watch_percent, index=df.index))
    df["watch_percent_clean"] = df["watch_percent_clean"].fillna(0).clip(0, 1)

    df["completion_flag"] = (df["watch_percent_clean"] >= 0.9).astype(int)
    df["positive_label"] = (
        (df["watch_percent_clean"] >= 0.7)
        | df["like"].eq(1)
        | df["comentario"].eq(1)
        | df["suscripcion"].eq(1)
        | df["clic_recomendacion"].eq(1)
    ).astype(int)
    df["negative_label"] = (
        (df["watch_percent_clean"] <= 0.2)
        & df["like"].eq(0)
        & df["comentario"].eq(0)
        & df["suscripcion"].eq(0)
        & df["clic_recomendacion"].eq(0)
    ).astype(int)

    df["implicit_score"] = (
        0.55 * df["watch_percent_clean"]
        + 0.15 * df["like"]
        + 0.10 * df["comentario"]
        + 0.15 * df["suscripcion"]
        + 0.05 * df["clic_recomendacion"]
    ).round(4)

    df["watch_date"] = df["timestamp"].dt.floor("D")
    df["fresh_watch_signal"] = (df["watch_percent_clean"] * df["video_freshness_score"]).round(4)

    return df


def build_user_features(clean_events: pd.DataFrame, follows: pd.DataFrame) -> pd.DataFrame:
    snapshot_timestamp = clean_events["timestamp"].max()
    recent_threshold = snapshot_timestamp - pd.Timedelta(days=30)
    recent_events = clean_events.loc[clean_events["timestamp"].ge(recent_threshold)].copy()
    grouped = clean_events.groupby("user_id", sort=False)

    user_features = grouped.agg(
        user_total_interactions=("video_id", "size"),
        user_unique_videos=("video_id", "nunique"),
        user_unique_categories=("video_category", "nunique"),
        user_unique_creators=("creator_user_id", "nunique"),
        user_avg_watch_percent=("watch_percent_clean", "mean"),
        user_completion_rate=("completion_flag", "mean"),
        user_like_rate=("like", "mean"),
        user_comment_rate=("comentario", "mean"),
        user_subscription_rate=("suscripcion", "mean"),
        user_recommended_share=("fue_recomendado", "mean"),
        user_avg_implicit_score=("implicit_score", "mean"),
        user_avg_video_freshness=("video_freshness_score", "mean"),
        user_active_days=("watch_date", "nunique"),
        user_first_timestamp=("timestamp", "min"),
        user_last_timestamp=("timestamp", "max"),
    ).reset_index()

    recommendation_summary = grouped.agg(
        user_recommendation_clicks=("clic_recomendacion", "sum"),
        user_recommendation_impressions=("fue_recomendado", "sum"),
    ).reset_index()
    recommendation_summary["user_click_from_reco_rate"] = np.divide(
        recommendation_summary["user_recommendation_clicks"],
        recommendation_summary["user_recommendation_impressions"],
        out=np.zeros(len(recommendation_summary), dtype=float),
        where=recommendation_summary["user_recommendation_impressions"].to_numpy() > 0,
    )

    user_preferences = grouped.agg(
        user_favorite_category=("video_category", safe_mode),
        user_favorite_device=("dispositivo", safe_mode),
        user_favorite_time_slot=("franja_horaria", safe_mode),
        user_favorite_channel_id=("channel_id", safe_mode_int),
        user_favorite_creator_id=("creator_user_id", safe_mode_int),
    ).reset_index()

    if recent_events.empty:
        recent_preferences = pd.DataFrame(
            columns=[
                "user_id",
                "user_recent_favorite_category",
                "user_recent_favorite_channel_id",
                "user_recent_favorite_creator_id",
                "user_recent_interactions_30d",
            ]
        )
    else:
        recent_preferences = (
            recent_events.groupby("user_id", sort=False)
            .agg(
                user_recent_favorite_category=("video_category", safe_mode),
                user_recent_favorite_channel_id=("channel_id", safe_mode_int),
                user_recent_favorite_creator_id=("creator_user_id", safe_mode_int),
                user_recent_interactions_30d=("video_id", "size"),
            )
            .reset_index()
        )

    user_features = user_features.merge(recommendation_summary, on="user_id", how="left")
    user_features = user_features.merge(user_preferences, on="user_id", how="left")
    user_features = user_features.merge(recent_preferences, on="user_id", how="left")

    if not follows.empty:
        following_summary = (
            follows.groupby("user_id", sort=False)["channel_id"]
            .nunique()
            .rename("user_following_channels")
            .reset_index()
        )
        user_features = user_features.merge(following_summary, on="user_id", how="left")
    else:
        user_features["user_following_channels"] = 0

    user_features["user_following_channels"] = user_features["user_following_channels"].fillna(0).astype(int)
    user_features["user_recent_interactions_30d"] = user_features["user_recent_interactions_30d"].fillna(0).astype(int)
    user_features["user_recent_favorite_channel_id"] = user_features["user_recent_favorite_channel_id"].fillna(0).astype(int)
    user_features["user_recent_favorite_creator_id"] = user_features["user_recent_favorite_creator_id"].fillna(0).astype(int)
    user_features["user_recent_favorite_category"] = user_features["user_recent_favorite_category"].fillna("unknown")
    user_features["user_active_window_days"] = (
        (user_features["user_last_timestamp"] - user_features["user_first_timestamp"]).dt.days + 1
    )

    numeric_to_round = [
        "user_avg_watch_percent",
        "user_completion_rate",
        "user_like_rate",
        "user_comment_rate",
        "user_subscription_rate",
        "user_recommended_share",
        "user_avg_implicit_score",
        "user_avg_video_freshness",
        "user_click_from_reco_rate",
    ]
    user_features[numeric_to_round] = user_features[numeric_to_round].round(4)

    return user_features


def build_user_channel_features(
    clean_events: pd.DataFrame,
    follows: pd.DataFrame,
    snapshot_timestamp: pd.Timestamp,
) -> pd.DataFrame:
    recent_threshold = snapshot_timestamp - pd.Timedelta(days=30)
    grouped = clean_events.groupby(["user_id", "channel_id", "creator_user_id"], sort=False)

    user_channel_features = grouped.agg(
        channel_first_watch_timestamp=("timestamp", "min"),
        channel_last_watch_timestamp=("timestamp", "max"),
        channel_lifetime_interactions=("video_id", "size"),
        channel_lifetime_watch_percent=("watch_percent_clean", "mean"),
        channel_lifetime_implicit_score=("implicit_score", "mean"),
    ).reset_index()

    recent_events = clean_events.loc[clean_events["timestamp"].ge(recent_threshold)].copy()
    if not recent_events.empty:
        recent_channel_features = (
            recent_events.groupby(["user_id", "channel_id", "creator_user_id"], sort=False)
            .agg(
                channel_recent_interactions_30d=("video_id", "size"),
                channel_recent_watch_percent_30d=("watch_percent_clean", "mean"),
                channel_recent_implicit_score_30d=("implicit_score", "mean"),
            )
            .reset_index()
        )
        user_channel_features = user_channel_features.merge(
            recent_channel_features,
            on=["user_id", "channel_id", "creator_user_id"],
            how="left",
        )
    else:
        user_channel_features["channel_recent_interactions_30d"] = 0
        user_channel_features["channel_recent_watch_percent_30d"] = 0.0
        user_channel_features["channel_recent_implicit_score_30d"] = 0.0

    if not follows.empty:
        follow_features = follows[["user_id", "channel_id", "followed_at"]].drop_duplicates()
        follow_features["user_follows_channel"] = 1
        user_channel_features = user_channel_features.merge(
            follow_features,
            on=["user_id", "channel_id"],
            how="left",
        )
    else:
        user_channel_features["followed_at"] = pd.NaT
        user_channel_features["user_follows_channel"] = 0

    user_channel_features["channel_recent_interactions_30d"] = (
        user_channel_features["channel_recent_interactions_30d"].fillna(0).astype(int)
    )
    user_channel_features["channel_recent_watch_percent_30d"] = (
        user_channel_features["channel_recent_watch_percent_30d"].fillna(0).round(4)
    )
    user_channel_features["channel_recent_implicit_score_30d"] = (
        user_channel_features["channel_recent_implicit_score_30d"].fillna(0).round(4)
    )
    user_channel_features["user_follows_channel"] = (
        user_channel_features["user_follows_channel"].fillna(0).astype(int)
    )
    user_channel_features["days_since_last_channel_watch"] = (
        snapshot_timestamp - user_channel_features["channel_last_watch_timestamp"]
    ).dt.days.clip(lower=0)
    user_channel_features["stale_follow_flag"] = (
        user_channel_features["user_follows_channel"].eq(1)
        & user_channel_features["channel_recent_interactions_30d"].eq(0)
        & user_channel_features["days_since_last_channel_watch"].ge(60)
    ).astype(int)
    user_channel_features["channel_current_interest_score"] = (
        0.70 * user_channel_features["channel_recent_implicit_score_30d"]
        + 0.20 * np.exp(-user_channel_features["days_since_last_channel_watch"] / 30)
        + 0.10 * user_channel_features["user_follows_channel"]
        - 0.15 * user_channel_features["stale_follow_flag"]
    ).round(4)

    return user_channel_features


def build_pair_interactions(clean_events: pd.DataFrame) -> pd.DataFrame:
    pair_df = (
        clean_events.groupby(["user_id", "video_id"], sort=False)
        .agg(
            channel_id=("channel_id", "first"),
            creator_user_id=("creator_user_id", "first"),
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
            interaction_count=("video_id", "size"),
            avg_watch_percent=("watch_percent_clean", "mean"),
            max_watch_percent=("watch_percent_clean", "max"),
            total_watch_time_s=("tiempo_visto_s_clean", "sum"),
            any_like=("like", "max"),
            any_comment=("comentario", "max"),
            any_subscription=("suscripcion", "max"),
            any_recommended=("fue_recomendado", "max"),
            any_click=("clic_recomendacion", "max"),
            positive_label=("positive_label", "max"),
            implicit_score_sum=("implicit_score", "sum"),
            implicit_score_mean=("implicit_score", "mean"),
            avg_video_freshness=("video_freshness_score", "mean"),
            min_video_age_days=("video_age_days", "min"),
        )
        .reset_index()
    )

    pair_df["negative_label"] = (
        pair_df["positive_label"].eq(0)
        & pair_df["max_watch_percent"].le(0.2)
        & pair_df["any_like"].eq(0)
        & pair_df["any_comment"].eq(0)
        & pair_df["any_subscription"].eq(0)
        & pair_df["any_click"].eq(0)
    ).astype(int)
    pair_df["cf_confidence"] = (1 + 10 * pair_df["implicit_score_sum"]).round(4)

    numeric_to_round = [
        "avg_watch_percent",
        "max_watch_percent",
        "total_watch_time_s",
        "implicit_score_sum",
        "implicit_score_mean",
        "avg_video_freshness",
    ]
    pair_df[numeric_to_round] = pair_df[numeric_to_round].round(4)

    return pair_df


def build_cf_splits(pair_df: pd.DataFrame) -> pd.DataFrame:
    cf_df = pair_df.loc[pair_df["positive_label"].eq(1)].copy()
    cf_df = cf_df.sort_values(["user_id", "last_timestamp", "video_id"]).reset_index(drop=True)
    cf_df["split"] = "train"

    positive_counts = cf_df.groupby("user_id")["video_id"].transform("size")
    order_in_user = cf_df.groupby("user_id").cumcount()

    validation_mask = (positive_counts >= 3) & order_in_user.eq(positive_counts - 2)
    test_mask = (positive_counts >= 2) & order_in_user.eq(positive_counts - 1)

    cf_df.loc[validation_mask, "split"] = "validation"
    cf_df.loc[test_mask, "split"] = "test"

    return cf_df[
        [
            "user_id",
            "video_id",
            "channel_id",
            "creator_user_id",
            "interaction_count",
            "implicit_score_sum",
            "implicit_score_mean",
            "cf_confidence",
            "first_timestamp",
            "last_timestamp",
            "split",
        ]
    ].copy()


def build_ranking_dataset(
    pair_df: pd.DataFrame,
    user_features: pd.DataFrame,
    video_features: pd.DataFrame,
    user_channel_features: pd.DataFrame,
    follows: pd.DataFrame,
) -> pd.DataFrame:
    labelled_pairs = pair_df.loc[
        pair_df["positive_label"].eq(1) | pair_df["negative_label"].eq(1)
    ].copy()
    labelled_pairs["ranking_label"] = labelled_pairs["positive_label"].astype(int)

    ranking_df = labelled_pairs.merge(user_features, on="user_id", how="left")
    ranking_df = ranking_df.merge(
        video_features.drop(columns=["channel_id", "creator_user_id"]),
        on="video_id",
        how="left",
    )

    ranking_df = ranking_df.merge(
        user_channel_features[
            [
                "user_id",
                "channel_id",
                "creator_user_id",
                "channel_recent_interactions_30d",
                "channel_recent_watch_percent_30d",
                "channel_recent_implicit_score_30d",
                "days_since_last_channel_watch",
                "user_follows_channel",
                "stale_follow_flag",
                "channel_current_interest_score",
            ]
        ],
        on=["user_id", "channel_id", "creator_user_id"],
        how="left",
    )

    if not follows.empty:
        ranking_df["user_follows_channel"] = ranking_df["user_follows_channel"].fillna(0).astype(int)
    else:
        ranking_df["user_follows_channel"] = 0

    ranking_df["channel_recent_interactions_30d"] = ranking_df["channel_recent_interactions_30d"].fillna(0).astype(int)
    ranking_df["channel_recent_watch_percent_30d"] = ranking_df["channel_recent_watch_percent_30d"].fillna(0).round(4)
    ranking_df["channel_recent_implicit_score_30d"] = ranking_df["channel_recent_implicit_score_30d"].fillna(0).round(4)
    ranking_df["days_since_last_channel_watch"] = ranking_df["days_since_last_channel_watch"].fillna(999).astype(int)
    ranking_df["stale_follow_flag"] = ranking_df["stale_follow_flag"].fillna(0).astype(int)
    ranking_df["channel_current_interest_score"] = ranking_df["channel_current_interest_score"].fillna(0).round(4)
    ranking_df["category_matches_user_pref"] = (
        ranking_df["video_category"] == ranking_df["user_favorite_category"]
    ).astype(int)
    ranking_df["recent_category_match"] = (
        ranking_df["video_category"] == ranking_df["user_recent_favorite_category"]
    ).astype(int)
    ranking_df["creator_matches_user_pref"] = (
        ranking_df["creator_user_id"] == ranking_df["user_favorite_creator_id"]
    ).astype(int)
    ranking_df["recent_creator_match"] = (
        ranking_df["creator_user_id"] == ranking_df["user_recent_favorite_creator_id"]
    ).astype(int)
    ranking_df["channel_matches_user_pref"] = (
        ranking_df["channel_id"] == ranking_df["user_favorite_channel_id"]
    ).astype(int)
    ranking_df["recent_channel_match"] = (
        ranking_df["channel_id"] == ranking_df["user_recent_favorite_channel_id"]
    ).astype(int)

    ranking_df["watch_vs_user_avg"] = (
        ranking_df["avg_watch_percent"] - ranking_df["user_avg_watch_percent"]
    ).round(4)
    ranking_df["item_vs_user_score_gap"] = (
        ranking_df["video_engagement_score"] - ranking_df["user_avg_implicit_score"]
    ).round(4)
    ranking_df["freshness_vs_user_avg"] = (
        ranking_df["video_freshness_score"] - ranking_df["user_avg_video_freshness"]
    ).round(4)

    return ranking_df


def save_outputs(
    clean_events: pd.DataFrame,
    user_features: pd.DataFrame,
    user_channel_features: pd.DataFrame,
    video_features: pd.DataFrame,
    creator_features: pd.DataFrame,
    pair_df: pd.DataFrame,
    cf_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_dir = OUTPUT_DIR
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        clean_events.head(1).to_csv(target_dir / "__write_probe__.csv", index=False)
        (target_dir / "__write_probe__.csv").unlink(missing_ok=True)
    except PermissionError:
        target_dir = FALLBACK_OUTPUT_DIR
        target_dir.mkdir(parents=True, exist_ok=True)

    clean_events[
        [
            "user_id",
            "video_id",
            "channel_id",
            "creator_user_id",
            "timestamp",
            "dispositivo",
            "franja_horaria",
            "video_category",
            "tiempo_visto_s_clean",
            "watch_percent_clean",
            "like",
            "comentario",
            "suscripcion",
            "fue_recomendado",
            "clic_recomendacion",
            "completion_flag",
            "positive_label",
            "negative_label",
            "video_age_days",
            "video_freshness_score",
            "fresh_watch_signal",
            "implicit_score",
        ]
    ].to_csv(target_dir / "eventos_limpios.csv", index=False)

    user_features.to_csv(target_dir / "user_features.csv", index=False)
    user_channel_features.to_csv(target_dir / "user_channel_features.csv", index=False)
    video_features.to_csv(target_dir / "video_features.csv", index=False)
    creator_features.to_csv(target_dir / "creator_features.csv", index=False)
    pair_df.to_csv(target_dir / "pair_interactions.csv", index=False)
    cf_df.to_csv(target_dir / "cf_interactions.csv", index=False)
    ranking_df.to_csv(target_dir / "ranking_dataset.csv", index=False)


def print_summary(
    clean_events: pd.DataFrame,
    user_features: pd.DataFrame,
    user_channel_features: pd.DataFrame,
    video_features: pd.DataFrame,
    creator_features: pd.DataFrame,
    pair_df: pd.DataFrame,
    cf_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> None:
    print("Prepared recommendation datasets")
    print(f"  clean events      : {len(clean_events):,}")
    print(f"  user features     : {len(user_features):,}")
    print(f"  user-channel feat : {len(user_channel_features):,}")
    print(f"  video features    : {len(video_features):,}")
    print(f"  creator features  : {len(creator_features):,}")
    print(f"  pair interactions : {len(pair_df):,}")
    print(f"  cf positives      : {len(cf_df):,}")
    print(f"  ranking rows      : {len(ranking_df):,}")
    print()
    print("CF split")
    print(cf_df["split"].value_counts().to_string())
    print()
    print("Ranking label distribution")
    print(ranking_df["ranking_label"].value_counts().sort_index().to_string())


def main() -> None:
    users, videos, channels, follows = load_inputs()
    snapshot_timestamp = users["timestamp"].max()
    creator_features = build_creator_features(channels)
    video_features = build_video_features(videos, snapshot_timestamp, creator_features)
    clean_events = clean_user_events(users, video_features)
    user_features = build_user_features(clean_events, follows)
    user_channel_features = build_user_channel_features(clean_events, follows, snapshot_timestamp)
    pair_df = build_pair_interactions(clean_events)
    cf_df = build_cf_splits(pair_df)
    ranking_df = build_ranking_dataset(pair_df, user_features, video_features, user_channel_features, follows)
    save_outputs(
        clean_events,
        user_features,
        user_channel_features,
        video_features,
        creator_features,
        pair_df,
        cf_df,
        ranking_df,
    )
    print_summary(
        clean_events,
        user_features,
        user_channel_features,
        video_features,
        creator_features,
        pair_df,
        cf_df,
        ranking_df,
    )


if __name__ == "__main__":
    main()
