from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
SOURCE_PATH = DATA_DIR / "youtube recommendation dataset.csv"
RNG_SEED = 42


def safe_mode(series: pd.Series, default: str = "Unknown") -> str:
    mode = series.dropna().mode()
    if mode.empty:
        return default
    return str(mode.iloc[0])


def parse_timestamps(raw_timestamp: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(raw_timestamp, format="mixed", errors="coerce")
    epoch_mask = parsed.isna()
    if epoch_mask.any():
        numeric_vals = pd.to_numeric(raw_timestamp[epoch_mask], errors="coerce")
        parsed.loc[epoch_mask] = pd.to_datetime(numeric_vals, unit="s", errors="coerce")
    return parsed


def assign_creators_to_videos(
    videos_agg: pd.DataFrame,
    user_ids: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    video_count = len(videos_agg)
    creator_target = min(len(user_ids), max(300, video_count // 12))
    creator_user_ids = np.sort(rng.choice(user_ids, size=creator_target, replace=False))

    category_weights = (
        videos_agg["category"]
        .value_counts(normalize=True)
        .sort_index()
    )
    categories = category_weights.index.to_list()
    creator_primary_categories = rng.choice(
        categories,
        size=creator_target,
        p=category_weights.to_numpy(),
    )

    for idx, category in enumerate(categories):
        creator_primary_categories[idx] = category

    creators_df = pd.DataFrame(
        {
            "channel_id": np.arange(1, creator_target + 1, dtype=int),
            "creator_user_id": creator_user_ids,
            "primary_category": creator_primary_categories,
        }
    )

    category_to_channels = {
        category: creators_df.loc[
            creators_df["primary_category"].eq(category),
            ["channel_id", "creator_user_id"],
        ].reset_index(drop=True)
        for category in categories
    }

    videos_agg = videos_agg.copy()
    videos_agg["channel_id"] = 0
    videos_agg["creator_user_id"] = 0

    for category, index in videos_agg.groupby("category", sort=False).groups.items():
        pool = category_to_channels.get(category)
        if pool is None or pool.empty:
            pool = creators_df[["channel_id", "creator_user_id"]]

        selected_idx = rng.integers(0, len(pool), size=len(index))
        videos_agg.loc[index, "channel_id"] = pool.iloc[selected_idx]["channel_id"].to_numpy()
        videos_agg.loc[index, "creator_user_id"] = pool.iloc[selected_idx]["creator_user_id"].to_numpy()

    return videos_agg, creators_df


def generate_publication_dates(
    first_interaction_at: pd.Series,
    rng: np.random.Generator,
) -> pd.Series:
    n_rows = len(first_interaction_at)
    age_bucket = rng.choice([0, 1, 2], size=n_rows, p=[0.15, 0.35, 0.50])

    lag_days = np.empty(n_rows, dtype=int)
    mask_new = age_bucket == 0
    mask_recent = age_bucket == 1
    mask_old = age_bucket == 2

    lag_days[mask_new] = rng.integers(0, 8, size=mask_new.sum())
    lag_days[mask_recent] = rng.integers(8, 46, size=mask_recent.sum())
    lag_days[mask_old] = rng.integers(46, 366, size=mask_old.sum())

    published_at = first_interaction_at.dt.floor("D") - pd.to_timedelta(lag_days, unit="D")
    return published_at


def build_video_table(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    videos_agg = (
        df.groupby("video_id", sort=False)
        .agg(
            video_duration_s=("video_duration", "first"),
            total_views=("user_id", "count"),
            total_likes=("liked", "sum"),
            total_comments=("commented", "sum"),
            suscriptores_ganados=("subscribed_after", "sum"),
            avg_watch_percent=("watch_percent", "mean"),
            avg_watch_time_s=("watch_time", "mean"),
            veces_recomendado=("recommended", "sum"),
            total_clics=("clicked", "sum"),
            category=("category", lambda x: x.mode().iloc[0]),
            first_interaction_at=("timestamp", "min"),
            last_interaction_at=("timestamp", "max"),
        )
        .reset_index()
    )

    videos_agg["like_rate"] = (videos_agg["total_likes"] / videos_agg["total_views"]).round(4)
    videos_agg["comment_rate"] = (videos_agg["total_comments"] / videos_agg["total_views"]).round(4)
    videos_agg["subscription_rate"] = (
        videos_agg["suscriptores_ganados"] / videos_agg["total_views"]
    ).round(4)
    videos_agg["click_through_rate"] = (
        videos_agg["total_clics"] / videos_agg["veces_recomendado"].replace(0, np.nan)
    ).round(4).fillna(0)
    videos_agg["avg_watch_percent"] = videos_agg["avg_watch_percent"].round(4)
    videos_agg["avg_watch_time_s"] = videos_agg["avg_watch_time_s"].round(2)

    user_ids = np.sort(df["user_id"].dropna().astype(int).unique())
    videos_agg, creators_df = assign_creators_to_videos(videos_agg, user_ids, rng)
    videos_agg["published_at"] = generate_publication_dates(videos_agg["first_interaction_at"], rng)

    videos_agg.insert(3, "titulo", "")
    videos_agg.insert(7, "que_pasa", "")

    videos_df = videos_agg[
        [
            "video_id",
            "channel_id",
            "creator_user_id",
            "titulo",
            "video_duration_s",
            "published_at",
            "category",
            "que_pasa",
            "first_interaction_at",
            "last_interaction_at",
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
    ].copy()
    videos_df["is_cold_start_video"] = 0

    return videos_df, creators_df


def build_users_table(df: pd.DataFrame, video_lookup: pd.DataFrame) -> pd.DataFrame:
    users_df = df[
        [
            "user_id",
            "video_id",
            "device",
            "timestamp",
            "watch_time_of_day",
            "watch_time",
            "watch_percent",
            "liked",
            "commented",
            "subscribed_after",
            "recommended",
            "clicked",
        ]
    ].copy().rename(
        columns={
            "device": "dispositivo",
            "watch_time_of_day": "franja_horaria",
            "watch_time": "tiempo_visto_s",
            "watch_percent": "porcentaje_visto",
            "liked": "like",
            "commented": "comentario",
            "subscribed_after": "suscripcion",
            "recommended": "fue_recomendado",
            "clicked": "clic_recomendacion",
        }
    )

    users_df = users_df.merge(
        video_lookup[
            [
                "video_id",
                "channel_id",
                "creator_user_id",
                "published_at",
                "category",
            ]
        ],
        on="video_id",
        how="left",
    ).rename(columns={"category": "video_category"})

    users_df["es_video_propio"] = users_df["user_id"].eq(users_df["creator_user_id"]).astype(int)
    users_df = users_df.sort_values(["user_id", "timestamp", "video_id"]).reset_index(drop=True)
    return users_df


def build_channel_follows(users_df: pd.DataFrame) -> pd.DataFrame:
    follows_df = users_df.loc[
        users_df["suscripcion"].eq(1),
        ["user_id", "channel_id", "creator_user_id", "video_id", "timestamp"],
    ].copy()

    follows_df = follows_df.loc[follows_df["user_id"].ne(follows_df["creator_user_id"])].copy()
    follows_df = follows_df.sort_values(["user_id", "channel_id", "timestamp", "video_id"])
    follows_df = follows_df.drop_duplicates(["user_id", "channel_id"], keep="first")
    follows_df = follows_df.rename(
        columns={
            "video_id": "source_video_id",
            "timestamp": "followed_at",
        }
    )
    follows_df["follow_origin"] = "watch_subscription"

    return follows_df[
        [
            "user_id",
            "channel_id",
            "creator_user_id",
            "source_video_id",
            "followed_at",
            "follow_origin",
        ]
    ].copy()


def build_channels_table(
    creators_df: pd.DataFrame,
    videos_df: pd.DataFrame,
    follows_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    channel_video_stats = (
        videos_df.groupby(["channel_id", "creator_user_id"], sort=False)
        .agg(
            primary_category=("category", safe_mode),
            total_videos=("video_id", "count"),
            total_views=("total_views", "sum"),
            total_likes=("total_likes", "sum"),
            total_comments=("total_comments", "sum"),
            total_subscriptions_generated=("suscriptores_ganados", "sum"),
            avg_video_watch_percent=("avg_watch_percent", "mean"),
            avg_video_ctr=("click_through_rate", "mean"),
            avg_video_like_rate=("like_rate", "mean"),
            first_publish_at=("published_at", "min"),
            last_publish_at=("published_at", "max"),
        )
        .reset_index()
    )

    follow_stats = (
        follows_df.groupby("channel_id", sort=False)
        .agg(
            followers_total=("user_id", "nunique"),
            first_follow_at=("followed_at", "min"),
            last_follow_at=("followed_at", "max"),
        )
        .reset_index()
    )

    channels_df = creators_df.merge(
        channel_video_stats,
        on=["channel_id", "creator_user_id"],
        how="left",
    ).merge(
        follow_stats,
        on="channel_id",
        how="left",
    )

    if "primary_category_x" in channels_df.columns or "primary_category_y" in channels_df.columns:
        channels_df["primary_category"] = channels_df.get("primary_category_y")
        if "primary_category_x" in channels_df.columns:
            channels_df["primary_category"] = channels_df["primary_category"].fillna(channels_df["primary_category_x"])
        channels_df = channels_df.drop(
            columns=[col for col in ["primary_category_x", "primary_category_y"] if col in channels_df.columns]
        )

    channels_df["followers_total"] = channels_df["followers_total"].fillna(0).astype(int)
    channels_df["total_videos"] = channels_df["total_videos"].fillna(0).astype(int)
    channels_df["total_views"] = channels_df["total_views"].fillna(0).astype(int)
    channels_df["total_likes"] = channels_df["total_likes"].fillna(0).astype(int)
    channels_df["total_comments"] = channels_df["total_comments"].fillna(0).astype(int)
    channels_df["total_subscriptions_generated"] = (
        channels_df["total_subscriptions_generated"].fillna(0).astype(int)
    )
    channels_df["avg_video_watch_percent"] = channels_df["avg_video_watch_percent"].fillna(0).round(4)
    channels_df["avg_video_ctr"] = channels_df["avg_video_ctr"].fillna(0).round(4)
    channels_df["avg_video_like_rate"] = channels_df["avg_video_like_rate"].fillna(0).round(4)
    channels_df["avg_video_watch_percent"] = channels_df["avg_video_watch_percent"].replace([np.inf, -np.inf], 0)
    channels_df["avg_video_ctr"] = channels_df["avg_video_ctr"].replace([np.inf, -np.inf], 0)
    channels_df["avg_video_like_rate"] = channels_df["avg_video_like_rate"].replace([np.inf, -np.inf], 0)

    channel_name_suffix = channels_df["channel_id"].astype(str).str.zfill(4)
    channels_df["channel_name"] = (
        channels_df["primary_category"].fillna("general").str.lower()
        + "_creator_"
        + channel_name_suffix
    )

    channel_start_lag = rng.integers(30, 366, size=len(channels_df))
    channels_df["channel_created_at"] = (
        channels_df["first_publish_at"].fillna(channels_df["last_publish_at"])
        - pd.to_timedelta(channel_start_lag, unit="D")
    )
    channels_df["channel_created_at"] = channels_df["channel_created_at"].fillna(channels_df["first_publish_at"])

    return channels_df[
        [
            "channel_id",
            "creator_user_id",
            "channel_name",
            "primary_category",
            "channel_created_at",
            "first_publish_at",
            "last_publish_at",
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
    ].copy()


def build_user_profiles(
    users_df: pd.DataFrame,
    follows_df: pd.DataFrame,
    creators_df: pd.DataFrame,
) -> pd.DataFrame:
    creator_map = creators_df.set_index("creator_user_id")["channel_id"]

    user_profiles = (
        users_df.groupby("user_id", sort=False)
        .agg(
            first_seen_at=("timestamp", "min"),
            last_seen_at=("timestamp", "max"),
            total_interactions=("video_id", "size"),
            unique_videos_seen=("video_id", "nunique"),
            unique_creators_seen=("creator_user_id", "nunique"),
            likes_given=("like", "sum"),
            comments_given=("comentario", "sum"),
            subscriptions_made=("suscripcion", "sum"),
            recommendation_clicks=("clic_recomendacion", "sum"),
        )
        .reset_index()
    )

    following_counts = (
        follows_df.groupby("user_id", sort=False)["channel_id"]
        .nunique()
        .rename("following_channels")
        .reset_index()
    )

    user_profiles = user_profiles.merge(following_counts, on="user_id", how="left")
    user_profiles["following_channels"] = user_profiles["following_channels"].fillna(0).astype(int)
    user_profiles["is_creator"] = user_profiles["user_id"].isin(creator_map.index).astype(int)
    user_profiles["owned_channel_id"] = user_profiles["user_id"].map(creator_map)
    user_profiles["account_role"] = np.where(
        user_profiles["is_creator"].eq(1),
        "viewer_creator",
        "viewer",
    )

    return user_profiles[
        [
            "user_id",
            "account_role",
            "is_creator",
            "owned_channel_id",
            "first_seen_at",
            "last_seen_at",
            "total_interactions",
            "unique_videos_seen",
            "unique_creators_seen",
            "likes_given",
            "comments_given",
            "subscriptions_made",
            "recommendation_clicks",
            "following_channels",
        ]
    ].copy()


def build_new_uploads_table(
    channels_df: pd.DataFrame,
    videos_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if channels_df.empty or videos_df.empty:
        return pd.DataFrame(columns=videos_df.columns)

    max_video_id = int(videos_df["video_id"].max())
    snapshot_date = videos_df["last_interaction_at"].max().floor("D")
    uploads_target = min(1000, max(250, len(channels_df) // 5))

    sampled_channels = channels_df.sample(n=uploads_target, replace=False, random_state=RNG_SEED).reset_index(drop=True)
    duration_by_category = (
        videos_df.groupby("category", sort=False)["video_duration_s"]
        .median()
        .round()
        .astype(int)
        .to_dict()
    )

    recent_offsets = rng.integers(0, 8, size=uploads_target)
    default_duration = int(videos_df["video_duration_s"].median())

    new_videos = pd.DataFrame(
        {
            "video_id": np.arange(max_video_id + 1, max_video_id + uploads_target + 1, dtype=int),
            "channel_id": sampled_channels["channel_id"].to_numpy(),
            "creator_user_id": sampled_channels["creator_user_id"].to_numpy(),
            "titulo": "",
            "video_duration_s": sampled_channels["primary_category"].map(duration_by_category).fillna(default_duration).astype(int),
            "published_at": snapshot_date - pd.to_timedelta(recent_offsets, unit="D"),
            "category": sampled_channels["primary_category"].to_numpy(),
            "que_pasa": "",
            "first_interaction_at": pd.NaT,
            "last_interaction_at": pd.NaT,
            "total_views": 0,
            "total_likes": 0,
            "like_rate": 0.0,
            "total_comments": 0,
            "comment_rate": 0.0,
            "suscriptores_ganados": 0,
            "subscription_rate": 0.0,
            "avg_watch_percent": 0.0,
            "avg_watch_time_s": 0.0,
            "veces_recomendado": 0,
            "total_clics": 0,
            "click_through_rate": 0.0,
            "is_cold_start_video": 1,
        }
    )

    return new_videos[videos_df.columns].copy()


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    print("Loading source dataset...")
    df = pd.read_csv(SOURCE_PATH)
    print(f"  rows: {len(df):,} | columns: {list(df.columns)}")

    liked_map = {"0": 0, "1": 1, "2": 1, "no": 0, "yes": 1}
    df["liked"] = df["liked"].map(liked_map).fillna(0).astype(int)

    category_map = {
        "Music": "Music",
        "MUsic": "Music",
        "music": "Music",
        "Gaming": "Gaming",
        "gamingg": "Gaming",
        "Education": "Education",
        "Ed": "Education",
        "Comedy": "Comedy",
        "COMEDY": "Comedy",
        "Tech": "Tech",
        "Tech ": "Tech",
        "Sports": "Sports",
        "News": "News",
        "Lifestyle": "Lifestyle",
    }
    df["category"] = df["category"].map(category_map).fillna(df["category"])
    df["timestamp"] = parse_timestamps(df["timestamp"])
    df["video_duration"] = pd.to_numeric(df["video_duration"], errors="coerce")
    df["watch_time"] = pd.to_numeric(df["watch_time"], errors="coerce").fillna(0).clip(lower=0)
    df["watch_percent"] = pd.to_numeric(df["watch_percent"], errors="coerce")
    df["watch_percent"] = df["watch_percent"].replace([np.inf, -np.inf], np.nan)

    valid_duration = df["video_duration"].gt(0)
    derived_watch_percent = np.where(
        valid_duration,
        df["watch_time"] / df["video_duration"],
        np.nan,
    )
    df["watch_percent"] = df["watch_percent"].where(df["watch_percent"].between(0, 1))
    df["watch_percent"] = df["watch_percent"].fillna(pd.Series(derived_watch_percent, index=df.index))
    df["watch_percent"] = df["watch_percent"].fillna(0).clip(0, 1)

    print("Building video catalog with creators and publish dates...")
    videos_df, creators_df = build_video_table(df, rng)

    print("Building user interactions...")
    users_df = build_users_table(df, videos_df)

    print("Building channel follows and user profiles...")
    follows_df = build_channel_follows(users_df)
    channels_df = build_channels_table(creators_df, videos_df, follows_df, rng)
    user_profiles_df = build_user_profiles(users_df, follows_df, creators_df)
    new_uploads_df = build_new_uploads_table(channels_df, videos_df, rng)

    final_videos_df = pd.concat([videos_df, new_uploads_df], ignore_index=True, sort=False)
    final_videos_df = final_videos_df.sort_values("video_id").reset_index(drop=True)

    existing_videos_path = DATA_DIR / "videos.csv"
    if existing_videos_path.exists():
        existing_videos = pd.read_csv(existing_videos_path, usecols=["video_id", "titulo", "que_pasa"])
        final_videos_df = final_videos_df.drop(columns=["titulo", "que_pasa"], errors="ignore")
        final_videos_df = final_videos_df.merge(existing_videos, on="video_id", how="left")
        final_videos_df["titulo"] = final_videos_df["titulo"].fillna("")
        final_videos_df["que_pasa"] = final_videos_df["que_pasa"].fillna("")
        ordered_columns = [
            "video_id",
            "channel_id",
            "creator_user_id",
            "titulo",
            "video_duration_s",
            "published_at",
            "category",
            "que_pasa",
            "first_interaction_at",
            "last_interaction_at",
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
            "is_cold_start_video",
        ]
        final_videos_df = final_videos_df[ordered_columns].copy()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    users_df.to_csv(DATA_DIR / "usuarios.csv", index=False)
    user_profiles_df.to_csv(DATA_DIR / "usuarios_perfiles.csv", index=False)
    channels_df.to_csv(DATA_DIR / "canales.csv", index=False)
    follows_df.to_csv(DATA_DIR / "seguimientos_canales.csv", index=False)
    final_videos_df.to_csv(DATA_DIR / "videos.csv", index=False)

    print()
    print("Generated datasets")
    print(f"  usuarios.csv              -> {len(users_df):,} rows")
    print(f"  usuarios_perfiles.csv     -> {len(user_profiles_df):,} rows")
    print(f"  canales.csv               -> {len(channels_df):,} rows")
    print(f"  seguimientos_canales.csv  -> {len(follows_df):,} rows")
    print(f"  videos.csv                -> {len(final_videos_df):,} rows")


if __name__ == "__main__":
    main()
