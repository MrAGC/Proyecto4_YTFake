from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
OUTPUT_DIR = Path("reco_output")

USERS_PATH = DATA_DIR / "usuarios.csv"
VIDEOS_PATH = DATA_DIR / "videos.csv"


def safe_mode(series: pd.Series, default: str = "unknown") -> str:
    mode = series.dropna().mode()
    if mode.empty:
        return default
    return str(mode.iloc[0])


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    users = pd.read_csv(USERS_PATH, parse_dates=["timestamp"])
    videos = pd.read_csv(VIDEOS_PATH)
    return users, videos


def build_video_features(videos: pd.DataFrame) -> pd.DataFrame:
    df = videos.copy()

    numeric_columns = [
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

    df["video_category"] = df["category"].fillna("Unknown")
    df["has_title_metadata"] = df["titulo"].fillna("").str.strip().ne("").astype(int)
    df["has_keyword_metadata"] = df["que_pasa"].fillna("").str.strip().ne("").astype(int)

    return df[
        [
            "video_id",
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
            "has_title_metadata",
            "has_keyword_metadata",
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
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in ["like", "comentario", "suscripcion", "fue_recomendado", "clic_recomendacion"]:
        df[column] = df[column].fillna(0).clip(0, 1).astype(int)

    df = df.merge(
        video_features[["video_id", "video_category", "video_duration_s_clean"]],
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

    return df


def build_user_features(clean_events: pd.DataFrame) -> pd.DataFrame:
    grouped = clean_events.groupby("user_id", sort=False)

    user_features = grouped.agg(
        user_total_interactions=("video_id", "size"),
        user_unique_videos=("video_id", "nunique"),
        user_unique_categories=("video_category", "nunique"),
        user_avg_watch_percent=("watch_percent_clean", "mean"),
        user_completion_rate=("completion_flag", "mean"),
        user_like_rate=("like", "mean"),
        user_comment_rate=("comentario", "mean"),
        user_subscription_rate=("suscripcion", "mean"),
        user_recommended_share=("fue_recomendado", "mean"),
        user_avg_implicit_score=("implicit_score", "mean"),
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
    ).reset_index()

    user_features = user_features.merge(recommendation_summary, on="user_id", how="left")
    user_features = user_features.merge(user_preferences, on="user_id", how="left")
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
        "user_click_from_reco_rate",
    ]
    user_features[numeric_to_round] = user_features[numeric_to_round].round(4)

    return user_features


def build_pair_interactions(clean_events: pd.DataFrame) -> pd.DataFrame:
    pair_df = (
        clean_events.groupby(["user_id", "video_id"], sort=False)
        .agg(
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
) -> pd.DataFrame:
    labelled_pairs = pair_df.loc[
        pair_df["positive_label"].eq(1) | pair_df["negative_label"].eq(1)
    ].copy()
    labelled_pairs["ranking_label"] = labelled_pairs["positive_label"].astype(int)

    ranking_df = labelled_pairs.merge(user_features, on="user_id", how="left")
    ranking_df = ranking_df.merge(video_features, on="video_id", how="left")
    ranking_df["category_matches_user_pref"] = (
        ranking_df["video_category"] == ranking_df["user_favorite_category"]
    ).astype(int)

    ranking_df["watch_vs_user_avg"] = (
        ranking_df["avg_watch_percent"] - ranking_df["user_avg_watch_percent"]
    ).round(4)
    ranking_df["item_vs_user_score_gap"] = (
        ranking_df["video_engagement_score"] - ranking_df["user_avg_implicit_score"]
    ).round(4)

    return ranking_df


def save_outputs(
    clean_events: pd.DataFrame,
    user_features: pd.DataFrame,
    video_features: pd.DataFrame,
    pair_df: pd.DataFrame,
    cf_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clean_events[
        [
            "user_id",
            "video_id",
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
            "implicit_score",
        ]
    ].to_csv(OUTPUT_DIR / "eventos_limpios.csv", index=False)

    user_features.to_csv(OUTPUT_DIR / "user_features.csv", index=False)
    video_features.to_csv(OUTPUT_DIR / "video_features.csv", index=False)
    pair_df.to_csv(OUTPUT_DIR / "pair_interactions.csv", index=False)
    cf_df.to_csv(OUTPUT_DIR / "cf_interactions.csv", index=False)
    ranking_df.to_csv(OUTPUT_DIR / "ranking_dataset.csv", index=False)


def print_summary(
    clean_events: pd.DataFrame,
    user_features: pd.DataFrame,
    video_features: pd.DataFrame,
    pair_df: pd.DataFrame,
    cf_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> None:
    print("Prepared recommendation datasets")
    print(f"  clean events      : {len(clean_events):,}")
    print(f"  user features     : {len(user_features):,}")
    print(f"  video features    : {len(video_features):,}")
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
    users, videos = load_inputs()
    video_features = build_video_features(videos)
    clean_events = clean_user_events(users, video_features)
    user_features = build_user_features(clean_events)
    pair_df = build_pair_interactions(clean_events)
    cf_df = build_cf_splits(pair_df)
    ranking_df = build_ranking_dataset(pair_df, user_features, video_features)
    save_outputs(clean_events, user_features, video_features, pair_df, cf_df, ranking_df)
    print_summary(clean_events, user_features, video_features, pair_df, cf_df, ranking_df)


if __name__ == "__main__":
    main()
