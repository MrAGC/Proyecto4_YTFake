from __future__ import annotations

import csv
import difflib
import json
import math
import pickle
import re
import unicodedata
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import xgboost as xgb

from ml.train_retrieval_cf import top_items
from ml.train_retrieval_multisource import build_multisource_scores


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RECO_DIR = BASE_DIR / "reco_output_v2"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

METADATA_PATH = RECO_DIR / "training_dataset_balanced_v1_metadata.json"
USER_FEATURES_PATH = RECO_DIR / "user_features.csv"
VIDEO_FEATURES_PATH = RECO_DIR / "video_features.csv"
USER_CHANNEL_FEATURES_PATH = RECO_DIR / "user_channel_features.csv"
VIDEOS_PATH = DATA_DIR / "videos.csv"
USERS_PROFILES_PATH = DATA_DIR / "usuarios_perfiles.csv"
CHANNELS_PATH = DATA_DIR / "canales.csv"
FOLLOWS_PATH = DATA_DIR / "seguimientos_canales.csv"
INTERACTIONS_PATH = DATA_DIR / "usuarios.csv"
LOCAL_SESSION_EVENTS_PATH = DATA_DIR / "local_session_events.csv"
LOCAL_SEARCH_EVENTS_PATH = DATA_DIR / "local_search_events.csv"
LOCAL_ENGAGEMENT_EVENTS_PATH = DATA_DIR / "local_engagement_events.csv"
RANKER_MODEL_PATH = MODELS_DIR / "ranker_compare_v1" / "pairwise_xgboost" / "ranker_model.json"
RETRIEVAL_HOME_MODEL_PATH = MODELS_DIR / "retrieval_multisource_v1" / "retrieval_model.pkl"
RETRIEVAL_METRICS_PATH = MODELS_DIR / "retrieval_multisource_v1" / "metrics.json"
RANKER_COMPARISON_PATH = MODELS_DIR / "ranker_compare_v1" / "comparison_summary.json"
RETRIEVAL_DIAGNOSTICS_PATH = REPORTS_DIR / "retrieval_diagnostics_v1" / "diagnostics.json"
QUALITY_SUMMARY_PATH = REPORTS_DIR / "recommender_quality_v1" / "quality_summary.json"

CATEGORICAL_COLUMNS = [
    "surface",
    "user_favorite_category",
    "user_favorite_device",
    "user_favorite_time_slot",
    "user_recent_favorite_category",
    "video_category",
]
CATEGORICAL_SAFE_VALUES = {
    "surface": {"home", "watch_next", "search"},
    "user_favorite_category": {"Comedy", "Education", "Gaming", "Lifestyle", "Music", "News", "Sports", "Tech"},
    "user_favorite_device": {"Desktop", "Mobile", "TV", "Tablet"},
    "user_favorite_time_slot": {"Afternoon", "Evening", "Morning", "Night"},
    "user_recent_favorite_category": {"Comedy", "Education", "Gaming", "Lifestyle", "Music", "News", "Sports", "Tech", "unknown"},
    "video_category": {"Comedy", "Education", "Gaming", "Lifestyle", "Music", "News", "Sports", "Tech"},
}
CATEGORICAL_FALLBACKS = {
    "surface": "home",
    "user_favorite_category": "Education",
    "user_favorite_device": "Mobile",
    "user_favorite_time_slot": "Morning",
    "user_recent_favorite_category": "unknown",
    "video_category": "Education",
}

DEFAULT_VIDEO_LIMIT = 240
SESSION_HISTORY_LIMIT = 100
HISTORICAL_HISTORY_LIMIT = 10
HISTORY_PAGE_LIMIT = 240
SEARCH_STOPWORDS = {
    "a",
    "al",
    "algo",
    "asi",
    "bien",
    "cada",
    "como",
    "con",
    "contra",
    "cual",
    "cuando",
    "de",
    "del",
    "desde",
    "despues",
    "durante",
    "el",
    "en",
    "entre",
    "era",
    "es",
    "esta",
    "este",
    "esto",
    "fue",
    "hace",
    "hay",
    "la",
    "las",
    "lo",
    "los",
    "mas",
    "me",
    "mi",
    "mis",
    "muy",
    "no",
    "para",
    "pero",
    "por",
    "que",
    "se",
    "sin",
    "sobre",
    "su",
    "sus",
    "te",
    "todo",
    "un",
    "una",
    "unos",
    "y",
}
SEARCH_CATEGORY_ALIASES = {
    "comedia": "Comedy",
    "comedy": "Comedy",
    "educacion": "Education",
    "education": "Education",
    "estudio": "Education",
    "futbol": "Sports",
    "deporte": "Sports",
    "deportes": "Sports",
    "sports": "Sports",
    "gaming": "Gaming",
    "juego": "Gaming",
    "juegos": "Gaming",
    "videojuego": "Gaming",
    "videojuegos": "Gaming",
    "lifestyle": "Lifestyle",
    "vida": "Lifestyle",
    "musica": "Music",
    "music": "Music",
    "noticia": "News",
    "noticias": "News",
    "news": "News",
    "tecnologia": "Tech",
    "technology": "Tech",
    "tech": "Tech",
}


@dataclass
class UserSummary:
    user_id: int
    favorite_category: str
    recent_category: str
    favorite_device: str
    following_channels: int
    total_interactions: int


@dataclass
class SessionState:
    recent_video_ids: deque[int] = field(default_factory=lambda: deque(maxlen=SESSION_HISTORY_LIMIT))
    engaged_video_ids: deque[int] = field(default_factory=lambda: deque(maxlen=SESSION_HISTORY_LIMIT))
    liked_video_ids: set[int] = field(default_factory=set)
    subscribed_channel_ids: set[int] = field(default_factory=set)
    views_in_session: int = 0


@dataclass
class SearchDocument:
    video_id: int
    title: str
    normalized_title: str
    tokens: set[str]
    trigrams: set[str]
    views: int


@dataclass
class SearchIndex:
    documents: list[SearchDocument]
    token_index: dict[str, list[int]]
    trigram_index: dict[str, list[int]]


class RecommendationService:
    def __init__(self) -> None:
        with open(METADATA_PATH, "r", encoding="utf-8") as file_handle:
            self.feature_columns = json.load(file_handle)["feature_columns"]

        self.user_features = pd.read_csv(USER_FEATURES_PATH)
        self.video_features = pd.read_csv(VIDEO_FEATURES_PATH)
        self.user_channel_features = pd.read_csv(USER_CHANNEL_FEATURES_PATH)
        self.videos = pd.read_csv(VIDEOS_PATH, low_memory=False)
        self.user_profiles = pd.read_csv(USERS_PROFILES_PATH)
        self.channels = pd.read_csv(CHANNELS_PATH)
        self.follows = pd.read_csv(FOLLOWS_PATH, usecols=["user_id", "channel_id"])
        self.interactions = pd.read_csv(
            INTERACTIONS_PATH,
            usecols=["user_id", "video_id", "timestamp"],
            parse_dates=["timestamp"],
        )

        self.user_feature_map = self.user_features.set_index("user_id").to_dict(orient="index")
        self.video_feature_map = self.video_features.set_index("video_id").to_dict(orient="index")
        self.video_catalog_map = self.videos.set_index("video_id").to_dict(orient="index")
        self.user_channel_map = self.user_channel_features.set_index(["user_id", "channel_id"]).to_dict(orient="index")
        self.user_profile_map = self.user_profiles.set_index("user_id").to_dict(orient="index")
        self.channel_map = self.channels.set_index("channel_id").to_dict(orient="index")
        self.follow_pairs = {
            (int(row.user_id), int(row.channel_id))
            for row in self.follows.itertuples(index=False)
        }
        self.followed_channels_by_user = self._build_followed_channels_by_user()
        self.seen_by_user = (
            self.interactions.groupby("user_id", sort=False)["video_id"]
            .agg(lambda values: {int(value) for value in values})
            .to_dict()
        )
        self.historical_history_by_user = self._build_historical_history()
        self.creator_video_ids = self._build_creator_video_index()
        self.channels_by_creator = self._build_channels_by_creator()
        self.search_index = self._build_search_index()
        self.session_states: dict[int, SessionState] = {}
        self.guest_states: dict[str, SessionState] = {}
        self._ensure_local_session_store()
        self._ensure_local_search_store()
        self._ensure_local_engagement_store()
        self._load_local_session_events()
        self._load_local_engagement_events()

        with open(RETRIEVAL_HOME_MODEL_PATH, "rb") as file_handle:
            self.retrieval_bundle = pickle.load(file_handle)

        self.ranker = xgb.XGBRanker()
        self.ranker.load_model(RANKER_MODEL_PATH)

    def get_login_users(self, limit: int = 12) -> list[UserSummary]:
        merged = self.user_profiles.merge(
            self.user_features[
                [
                    "user_id",
                    "user_favorite_category",
                    "user_recent_favorite_category",
                    "user_favorite_device",
                    "user_following_channels",
                    "user_total_interactions",
                ]
            ],
            on="user_id",
            how="left",
        )
        merged = merged.sort_values(
            ["user_total_interactions", "recommendation_clicks", "following_channels"],
            ascending=False,
        )
        users: list[UserSummary] = []
        for row in merged.head(limit).itertuples(index=False):
            users.append(
                UserSummary(
                    user_id=int(row.user_id),
                    favorite_category=str(row.user_favorite_category),
                    recent_category=str(row.user_recent_favorite_category),
                    favorite_device=str(row.user_favorite_device),
                    following_channels=int(row.user_following_channels),
                    total_interactions=int(row.user_total_interactions),
                )
            )
        return users

    def user_exists(self, user_id: int) -> bool:
        return int(user_id) in self.user_feature_map

    def available_categories(self) -> list[str]:
        categories = {
            self._clean_text(category, "")
            for category in self.videos["category"].dropna().unique().tolist()
        }
        categories.discard("")
        return sorted(categories)

    def create_user(self, channel_name: str, favorite_category: str = "Education") -> int:
        channel_name = self._clean_text(channel_name, "").strip()
        if not channel_name:
            raise ValueError("El nombre del canal es obligatorio.")

        favorite_category = "Education"
        now = datetime.now(timezone.utc)
        user_id = self._next_numeric_id(self.user_profiles, "user_id", self.user_features, self.interactions)
        channel_id = self._next_numeric_id(self.channels, "channel_id")

        profile_row = {
            "user_id": user_id,
            "account_role": "creator",
            "is_creator": 1,
            "owned_channel_id": channel_id,
            "first_seen_at": now.isoformat(timespec="seconds"),
            "last_seen_at": now.isoformat(timespec="seconds"),
            "total_interactions": 0,
            "unique_videos_seen": 0,
            "unique_creators_seen": 0,
            "likes_given": 0,
            "comments_given": 0,
            "subscriptions_made": 0,
            "recommendation_clicks": 0,
            "following_channels": 0,
        }
        user_feature_row = self._new_user_feature_row(user_id, favorite_category, now)
        channel_row = self._new_channel_row(channel_id, user_id, channel_name, favorite_category, now)

        self.user_profiles = self._append_dataframe_row(self.user_profiles, profile_row)
        self.user_features = self._append_dataframe_row(self.user_features, user_feature_row)
        self.channels = self._append_dataframe_row(self.channels, channel_row)

        self._append_csv_row(USERS_PROFILES_PATH, self.user_profiles.columns, profile_row)
        self._append_csv_row(USER_FEATURES_PATH, self.user_features.columns, user_feature_row)
        self._append_csv_row(CHANNELS_PATH, self.channels.columns, channel_row)

        self.user_profile_map[user_id] = profile_row
        self.user_feature_map[user_id] = user_feature_row
        self.channel_map[channel_id] = channel_row
        self.channels_by_creator.setdefault(user_id, []).append(channel_id)
        self.creator_video_ids.setdefault(user_id, [])
        self.seen_by_user.setdefault(user_id, set())
        self.historical_history_by_user.setdefault(user_id, [])
        self.retrieval_bundle.setdefault("user_preferences", {})[user_id] = {
            "favorite_category": "unknown",
            "recent_favorite_category": "unknown",
        }
        return user_id

    def create_video(
        self,
        user_id: int,
        title: str,
        category: str,
        duration_minutes: int,
        keywords: str = "",
    ) -> int:
        if user_id not in self.user_feature_map:
            raise KeyError(f"No existe user_id={user_id}")

        title = self._clean_text(title, "").strip()
        if not title:
            raise ValueError("El titulo es obligatorio.")

        category = self._normalize_category(category)
        duration_seconds = max(30, min(int(duration_minutes) * 60, 4 * 60 * 60))
        now = datetime.now(timezone.utc)
        channel_id = self._ensure_user_channel(user_id, category)
        video_id = self._next_numeric_id(self.videos, "video_id", self.video_features)
        keywords = self._clean_text(keywords, "").strip() or self._default_keywords_for_category(category)
        thumbnail_dev_url, thumbnail_real_url = self._generated_thumbnail_urls(video_id, title, category)

        video_row = {
            "video_id": video_id,
            "channel_id": channel_id,
            "creator_user_id": user_id,
            "titulo": title,
            "video_duration_s": duration_seconds,
            "published_at": now.date().isoformat(),
            "category": category,
            "que_pasa": keywords,
            "first_interaction_at": "",
            "last_interaction_at": "",
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
            "thumbnail_url": thumbnail_dev_url,
            "thumbnail_source": "created_in_studio",
            "thumbnail_dev_url": thumbnail_dev_url,
            "thumbnail_real_url": thumbnail_real_url,
            "thumbnail_real_source": "loremflickr_real_topic_seed",
        }
        video_feature_row = self._new_video_feature_row(video_row)

        self.videos = self._append_dataframe_row(self.videos, video_row)
        self.video_features = self._append_dataframe_row(self.video_features, video_feature_row)
        self._append_csv_row(VIDEOS_PATH, self.videos.columns, video_row)
        self._append_csv_row(VIDEO_FEATURES_PATH, self.video_features.columns, video_feature_row)

        self.video_catalog_map[video_id] = video_row
        self.video_feature_map[video_id] = video_feature_row
        self.creator_video_ids.setdefault(user_id, [])
        self.creator_video_ids[user_id].insert(0, video_id)
        self._update_channel_after_video(channel_id, video_row)
        self._add_video_to_retrieval_indexes(video_id, channel_id, category)
        self.search_index = self._build_search_index()
        return video_id

    @staticmethod
    def _append_dataframe_row(df: pd.DataFrame, row: dict[str, object]) -> pd.DataFrame:
        aligned_row = {column: row.get(column, "") for column in df.columns}
        return pd.concat([df, pd.DataFrame([aligned_row])], ignore_index=True)

    @staticmethod
    def _append_csv_row(path: Path, columns: pd.Index, row: dict[str, object]) -> None:
        with open(path, "a", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=list(columns))
            writer.writerow({column: row.get(column, "") for column in columns})

    @staticmethod
    def _next_numeric_id(primary_df: pd.DataFrame, column: str, *extra_dfs: pd.DataFrame) -> int:
        max_id = 0
        for df in (primary_df, *extra_dfs):
            if column not in df.columns or df.empty:
                continue
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            if not values.empty:
                max_id = max(max_id, int(values.max()))
        return max_id + 1

    def _normalize_category(self, category: str) -> str:
        raw_category = self._clean_text(category, "Gaming")
        normalized = self._normalize_search_text(raw_category)
        alias = SEARCH_CATEGORY_ALIASES.get(normalized)
        if alias:
            return alias
        available = {self._normalize_search_text(item): item for item in self.available_categories()}
        return available.get(normalized, raw_category.title())

    def _new_user_feature_row(self, user_id: int, favorite_category: str, now: datetime) -> dict[str, object]:
        return {
            "user_id": user_id,
            "user_total_interactions": 0,
            "user_unique_videos": 0,
            "user_unique_categories": 0,
            "user_unique_creators": 0,
            "user_avg_watch_percent": 0.0,
            "user_completion_rate": 0.0,
            "user_like_rate": 0.0,
            "user_comment_rate": 0.0,
            "user_subscription_rate": 0.0,
            "user_recommended_share": 0.0,
            "user_avg_implicit_score": 0.0,
            "user_avg_video_freshness": 1.0,
            "user_active_days": 0,
            "user_first_timestamp": now.isoformat(timespec="seconds"),
            "user_last_timestamp": now.isoformat(timespec="seconds"),
            "user_recommendation_clicks": 0,
            "user_recommendation_impressions": 0,
            "user_click_from_reco_rate": 0.0,
            "user_favorite_category": favorite_category,
            "user_favorite_device": "Mobile",
            "user_favorite_time_slot": "Morning",
            "user_favorite_channel_id": 0,
            "user_favorite_creator_id": 0,
            "user_recent_favorite_category": favorite_category,
            "user_recent_favorite_channel_id": 0,
            "user_recent_favorite_creator_id": 0,
            "user_recent_interactions_30d": 0,
            "user_following_channels": 0,
            "user_active_window_days": 0,
        }

    def _new_channel_row(
        self,
        channel_id: int,
        user_id: int,
        channel_name: str,
        primary_category: str,
        now: datetime,
    ) -> dict[str, object]:
        return {
            "channel_id": channel_id,
            "creator_user_id": user_id,
            "channel_name": channel_name,
            "primary_category": primary_category,
            "channel_created_at": now.date().isoformat(),
            "first_publish_at": "",
            "last_publish_at": "",
            "total_videos": 0,
            "total_views": 0,
            "total_likes": 0,
            "total_comments": 0,
            "total_subscriptions_generated": 0,
            "followers_total": 0,
            "avg_video_watch_percent": 0.0,
            "avg_video_ctr": 0.0,
            "avg_video_like_rate": 0.0,
        }

    def _ensure_user_channel(self, user_id: int, category: str) -> int:
        profile_row = self.user_profile_map.get(user_id, {})
        channel_id = self._resolve_owned_channel_id(user_id, profile_row)
        if channel_id is not None and channel_id in self.channel_map:
            return int(channel_id)

        now = datetime.now(timezone.utc)
        channel_id = self._next_numeric_id(self.channels, "channel_id")
        channel_name = f"creator_{user_id}"
        channel_row = self._new_channel_row(channel_id, user_id, channel_name, category, now)
        self.channels = self._append_dataframe_row(self.channels, channel_row)
        self._append_csv_row(CHANNELS_PATH, self.channels.columns, channel_row)
        self.channel_map[channel_id] = channel_row
        self.channels_by_creator.setdefault(user_id, []).append(channel_id)

        if user_id in self.user_profile_map:
            self.user_profile_map[user_id]["owned_channel_id"] = channel_id
            self.user_profile_map[user_id]["is_creator"] = 1
            self.user_profile_map[user_id]["account_role"] = "creator"
            mask = self.user_profiles["user_id"].astype(int) == int(user_id)
            self.user_profiles.loc[mask, ["owned_channel_id", "is_creator", "account_role"]] = [channel_id, 1, "creator"]
            self.user_profiles.to_csv(USERS_PROFILES_PATH, index=False)
        return channel_id

    def _new_video_feature_row(self, video_row: dict[str, object]) -> dict[str, object]:
        channel_id = int(video_row["channel_id"])
        channel = self.channel_map.get(channel_id, {})
        return {
            "video_id": int(video_row["video_id"]),
            "channel_id": channel_id,
            "creator_user_id": int(video_row["creator_user_id"]),
            "video_category": str(video_row["category"]),
            "video_duration_s_clean": int(video_row["video_duration_s"]),
            "total_views": 0,
            "total_likes": 0,
            "total_comments": 0,
            "suscriptores_ganados": 0,
            "veces_recomendado": 0,
            "total_clics": 0,
            "video_like_rate": 0.0,
            "video_comment_rate": 0.0,
            "video_subscription_rate": 0.0,
            "video_avg_watch_percent": 0.0,
            "video_ctr": 0.0,
            "video_popularity_log": 0.0,
            "video_engagement_score": 0.0,
            "video_age_days": 0,
            "video_freshness_score": 1.0,
            "is_recent_upload": 1,
            "video_discovery_score": 0.4,
            "has_title_metadata": 1,
            "has_keyword_metadata": 1,
            "channel_followers_total": self._clean_int(channel.get("followers_total"), 0),
            "channel_total_videos": self._clean_int(channel.get("total_videos"), 0) + 1,
            "channel_views_per_video": 0.0,
            "channel_engagement_score": 0.0,
        }

    @staticmethod
    def _default_keywords_for_category(category: str) -> str:
        defaults = {
            "Gaming": "gameplay, partida, reto, trucos, reaccion",
            "Sports": "partido, entrenamiento, resumen, jugadas, analisis",
            "Music": "musica, directo, cancion, artista, reaccion",
            "Education": "tutorial, aprender, explicacion, consejos, ejemplos",
            "Tech": "tecnologia, gadgets, review, setup, herramientas",
            "News": "noticias, actualidad, resumen, contexto, analisis",
            "Comedy": "humor, reaccion, amigos, situacion, divertido",
            "Lifestyle": "rutina, vlog, dia, experiencia, consejos",
        }
        return defaults.get(category, f"{category.lower()}, video, recomendacion, nuevo")

    @staticmethod
    def _generated_thumbnail_urls(video_id: int, title: str, category: str) -> tuple[str, str]:
        title_text = quote_plus(f"{category} | {title[:80]}")
        dev_colors = {
            "Gaming": "7c3aed",
            "Sports": "16a34a",
            "Music": "db2777",
            "Education": "2563eb",
            "Tech": "0f172a",
            "News": "ea580c",
            "Comedy": "dc2626",
            "Lifestyle": "be185d",
        }
        color = dev_colors.get(category, "ef4444")
        dev_url = f"https://placehold.co/640x360/{color}/ffffff/png?text={title_text}"
        real_topic = quote_plus(category.lower())
        real_url = f"https://loremflickr.com/640/360/{real_topic}?lock={int(video_id)}"
        return dev_url, real_url

    def _update_channel_after_video(self, channel_id: int, video_row: dict[str, object]) -> None:
        channel = self.channel_map.get(channel_id)
        if not channel:
            return
        published_at = str(video_row["published_at"])
        channel["total_videos"] = self._clean_int(channel.get("total_videos"), 0) + 1
        channel["last_publish_at"] = published_at
        if not self._clean_text(channel.get("first_publish_at"), ""):
            channel["first_publish_at"] = published_at
        mask = self.channels["channel_id"].astype(int) == int(channel_id)
        for column, value in channel.items():
            if column in self.channels.columns:
                self.channels.loc[mask, column] = value
        self.channels.to_csv(CHANNELS_PATH, index=False)

    def _add_video_to_retrieval_indexes(self, video_id: int, channel_id: int, category: str) -> None:
        bundle = self.retrieval_bundle

        def prepend_unique(container: list[int], item: int, max_len: int = 1000) -> list[int]:
            return [item] + [existing for existing in container if int(existing) != item][: max_len - 1]

        bundle["global_recent"] = prepend_unique([int(item) for item in bundle.get("global_recent", [])], video_id)
        category_top = bundle.setdefault("category_top_videos", {})
        recent_category_top = bundle.setdefault("recent_category_top_videos", {})
        channel_top = bundle.setdefault("channel_top_videos", {})
        category_top[category] = prepend_unique([int(item) for item in category_top.get(category, [])], video_id)
        recent_category_top[category] = prepend_unique([int(item) for item in recent_category_top.get(category, [])], video_id)
        channel_top[channel_id] = prepend_unique([int(item) for item in channel_top.get(channel_id, [])], video_id)

    def _ensure_local_session_store(self) -> None:
        if LOCAL_SESSION_EVENTS_PATH.exists():
            return
        with open(LOCAL_SESSION_EVENTS_PATH, "w", newline="", encoding="utf-8") as file_handle:
            writer = csv.writer(file_handle)
            writer.writerow(["actor_type", "actor_id", "video_id", "event_type", "event_ts"])

    def _ensure_local_search_store(self) -> None:
        if LOCAL_SEARCH_EVENTS_PATH.exists():
            return
        with open(LOCAL_SEARCH_EVENTS_PATH, "w", newline="", encoding="utf-8") as file_handle:
            writer = csv.writer(file_handle)
            writer.writerow(["actor_type", "actor_id", "query", "result_count", "clicked_video_id", "event_type", "event_ts"])

    def _ensure_local_engagement_store(self) -> None:
        if LOCAL_ENGAGEMENT_EVENTS_PATH.exists():
            return
        with open(LOCAL_ENGAGEMENT_EVENTS_PATH, "w", newline="", encoding="utf-8") as file_handle:
            writer = csv.writer(file_handle)
            writer.writerow(["actor_type", "actor_id", "video_id", "channel_id", "event_type", "event_ts"])

    def _load_local_session_events(self) -> None:
        if not LOCAL_SESSION_EVENTS_PATH.exists():
            return
        runtime_events = pd.read_csv(LOCAL_SESSION_EVENTS_PATH)
        if runtime_events.empty:
            return

        runtime_events = runtime_events[runtime_events["event_type"] == "view"]
        for row in runtime_events.itertuples(index=False):
            actor_type = str(row.actor_type)
            actor_id = str(row.actor_id)
            video_id = int(row.video_id)
            if actor_type == "user":
                self._register_state_view(self.session_states, int(actor_id), video_id, persist=False)
            elif actor_type == "guest":
                self._register_state_view(self.guest_states, actor_id, video_id, persist=False)

    def _load_local_engagement_events(self) -> None:
        if not LOCAL_ENGAGEMENT_EVENTS_PATH.exists():
            return
        runtime_events = pd.read_csv(LOCAL_ENGAGEMENT_EVENTS_PATH)
        if runtime_events.empty:
            return

        for row in runtime_events.itertuples(index=False):
            actor_type = str(row.actor_type)
            actor_id = str(row.actor_id)
            video_id = self._clean_int(getattr(row, "video_id", 0), 0)
            channel_id = self._clean_int(getattr(row, "channel_id", 0), 0)
            event_type = str(row.event_type)
            store = self.session_states if actor_type == "user" else self.guest_states
            key = int(actor_id) if actor_type == "user" else actor_id
            state = store.setdefault(key, SessionState())
            if event_type == "like" and video_id in self.video_feature_map:
                state.liked_video_ids.add(video_id)
                state.engaged_video_ids.append(video_id)
            elif event_type == "subscribe" and channel_id in self.channel_map:
                state.subscribed_channel_ids.add(channel_id)

    def _append_local_event(self, actor_type: str, actor_id: str, video_id: int) -> None:
        with open(LOCAL_SESSION_EVENTS_PATH, "a", newline="", encoding="utf-8") as file_handle:
            writer = csv.writer(file_handle)
            writer.writerow(
                [
                    actor_type,
                    actor_id,
                    int(video_id),
                    "view",
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ]
            )

    def _append_search_event(
        self,
        actor_type: str,
        actor_id: str,
        query: str,
        result_count: int,
        event_type: str,
        clicked_video_id: int | None = None,
    ) -> None:
        if not str(query or "").strip():
            return
        with open(LOCAL_SEARCH_EVENTS_PATH, "a", newline="", encoding="utf-8") as file_handle:
            writer = csv.writer(file_handle)
            writer.writerow(
                [
                    actor_type,
                    actor_id,
                    str(query).strip(),
                    int(result_count),
                    "" if clicked_video_id is None else int(clicked_video_id),
                    event_type,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ]
            )

    def _append_engagement_event(
        self,
        actor_type: str,
        actor_id: str,
        video_id: int,
        channel_id: int,
        event_type: str,
    ) -> None:
        with open(LOCAL_ENGAGEMENT_EVENTS_PATH, "a", newline="", encoding="utf-8") as file_handle:
            writer = csv.writer(file_handle)
            writer.writerow(
                [
                    actor_type,
                    actor_id,
                    int(video_id),
                    int(channel_id),
                    event_type,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ]
            )

    @staticmethod
    def _register_state_view(
        store: dict[object, SessionState],
        actor_id: object,
        video_id: int,
        persist: bool = False,
    ) -> SessionState:
        state = store.setdefault(actor_id, SessionState())
        refreshed_history = [existing_video_id for existing_video_id in state.recent_video_ids if existing_video_id != video_id]
        refreshed_history.append(video_id)
        state.recent_video_ids = deque(refreshed_history[-SESSION_HISTORY_LIMIT:], maxlen=SESSION_HISTORY_LIMIT)
        state.views_in_session += 1
        return state

    def reset_session(self, user_id: int) -> None:
        self.session_states.pop(user_id, None)

    def register_view(self, user_id: int, video_id: int, source: str = "direct", query: str = "") -> None:
        if user_id not in self.user_feature_map or video_id not in self.video_feature_map:
            return

        self._register_state_view(self.session_states, user_id, video_id)
        self._append_local_event("user", str(user_id), video_id)
        if source == "search":
            self._append_search_event("user", str(user_id), query, 0, "search_click", clicked_video_id=video_id)

    def register_guest_view(self, guest_id: str, video_id: int, source: str = "direct", query: str = "") -> None:
        if video_id not in self.video_feature_map:
            return

        self._register_state_view(self.guest_states, guest_id, video_id)
        self._append_local_event("guest", guest_id, video_id)
        if source == "search":
            self._append_search_event("guest", guest_id, query, 0, "search_click", clicked_video_id=video_id)

    def register_like(self, user_id: int, video_id: int) -> dict[str, object]:
        if user_id not in self.user_feature_map or video_id not in self.video_feature_map:
            raise KeyError(f"No existe user_id={user_id} o video_id={video_id}")
        state = self._register_state_engagement(self.session_states, user_id, video_id, "like")
        channel_id = self._clean_int(self.video_feature_map[video_id].get("channel_id"), 0)
        self._append_engagement_event("user", str(user_id), video_id, channel_id, "like")
        return {"ok": True, "liked": video_id in state.liked_video_ids, "session_views": state.views_in_session}

    def register_guest_like(self, guest_id: str, video_id: int) -> dict[str, object]:
        if video_id not in self.video_feature_map:
            raise KeyError(f"No existe video_id={video_id}")
        state = self._register_state_engagement(self.guest_states, guest_id, video_id, "like")
        channel_id = self._clean_int(self.video_feature_map[video_id].get("channel_id"), 0)
        self._append_engagement_event("guest", guest_id, video_id, channel_id, "like")
        return {"ok": True, "liked": video_id in state.liked_video_ids, "session_views": state.views_in_session}

    def register_subscription(self, user_id: int, channel_id: int) -> dict[str, object]:
        if user_id not in self.user_feature_map or channel_id not in self.channel_map:
            raise KeyError(f"No existe user_id={user_id} o channel_id={channel_id}")
        state = self.session_states.setdefault(user_id, SessionState())
        state.subscribed_channel_ids.add(channel_id)
        self._append_engagement_event("user", str(user_id), 0, channel_id, "subscribe")
        return {"ok": True, "subscribed": channel_id in state.subscribed_channel_ids}

    def register_guest_subscription(self, guest_id: str, channel_id: int) -> dict[str, object]:
        if channel_id not in self.channel_map:
            raise KeyError(f"No existe channel_id={channel_id}")
        state = self.guest_states.setdefault(guest_id, SessionState())
        state.subscribed_channel_ids.add(channel_id)
        self._append_engagement_event("guest", guest_id, 0, channel_id, "subscribe")
        return {"ok": True, "subscribed": channel_id in state.subscribed_channel_ids}

    @staticmethod
    def _register_state_engagement(
        store: dict[object, SessionState],
        actor_id: object,
        video_id: int,
        event_type: str,
    ) -> SessionState:
        state = store.setdefault(actor_id, SessionState())
        if event_type == "like":
            state.liked_video_ids.add(video_id)
        state.engaged_video_ids.append(video_id)
        return state

    def get_user_snapshot(self, user_id: int) -> dict[str, object]:
        user_row = self.user_feature_map.get(user_id)
        if user_row is None:
            raise KeyError(f"No existe user_id={user_id}")

        profile_row = self.user_profile_map.get(user_id, {})
        account_role = str(profile_row.get("account_role", "viewer"))
        raw_is_creator = profile_row.get("is_creator", 0)
        is_creator = 0 if pd.isna(raw_is_creator) else int(raw_is_creator)
        owned_channel_id = self._resolve_owned_channel_id(user_id, profile_row)
        owned_channel = self.channel_map.get(owned_channel_id, {}) if owned_channel_id is not None else {}
        session_state = self.session_states.get(user_id)
        has_signal = self._user_has_recommendation_signal(user_id)

        return {
            "user_id": user_id,
            "favorite_category": str(user_row.get("user_favorite_category", "General")) if has_signal else "Sin historial",
            "recent_category": str(user_row.get("user_recent_favorite_category", "General")) if has_signal else "Sin historial",
            "following_channels": int(user_row.get("user_following_channels", 0)),
            "total_interactions": int(user_row.get("user_total_interactions", 0)),
            "recent_interactions_30d": int(user_row.get("user_recent_interactions_30d", 0)),
            "account_role": account_role,
            "account_label": "creador" if is_creator == 1 else "espectador",
            "owned_channel_name": str(owned_channel.get("channel_name", "sin canal propio")),
            "uploaded_videos": len(self.creator_video_ids.get(user_id, [])),
            "session_views": 0 if session_state is None else int(session_state.views_in_session),
            "session_focus": self._build_session_focus(user_id),
        }

    def home_page(self, user_id: int) -> dict[str, object]:
        user_snapshot = self.get_user_snapshot(user_id)
        has_signal = self._user_has_recommendation_signal(user_id)
        candidate_ids = self._home_candidate_ids(user_id, limit=DEFAULT_VIDEO_LIMIT) if has_signal else []
        ranked_videos = self._rank_candidates(user_id, candidate_ids, surface_value="home") if has_signal else []
        shelves = self._build_home_shelves(user_id, ranked_videos) if has_signal else []
        hero_video = ranked_videos[0] if ranked_videos else None
        hero_videos = self._home_hero_videos(shelves, ranked_videos)
        return {
            "user": user_snapshot,
            "sidebar": self._build_sidebar(user_id),
            "hero_video": hero_video,
            "hero_videos": hero_videos,
            "shelves": shelves,
            "model_stack": "multi_source_home + pairwise_xgboost + session_blend",
            "active_page": "home",
            "needs_search": not has_signal,
        }

    def watch_page(self, user_id: int, video_id: int) -> dict[str, object]:
        user_snapshot = self.get_user_snapshot(user_id)
        video = self._video_card(video_id)
        if video is None:
            raise KeyError(f"No existe video_id={video_id}")
        video = self._apply_user_video_state(video, user_id)

        next_candidate_ids = self._watch_next_candidate_ids(user_id, video_id, limit=120)
        next_up = self._rank_watch_next_candidates(user_id, next_candidate_ids, video_id)
        next_groups = self._build_watch_next_groups_for_user(user_id, video_id, video, next_up)
        return {
            "user": user_snapshot,
            "sidebar": self._build_sidebar(user_id),
            "video": video,
            "next_up": next_up[:20],
            "next_groups": next_groups,
            "model_stack": "item_based_watch_next + pairwise_xgboost + session_blend",
            "active_page": "watch",
        }

    def search_page(self, user_id: int, query: str) -> dict[str, object]:
        results = self.search_videos(query)
        if str(query or "").strip():
            self._append_search_event("user", str(user_id), query, len(results), "search")
        return {
            "user": self.get_user_snapshot(user_id),
            "sidebar": self._build_sidebar(user_id),
            "query": query,
            "results": results,
            "result_count": len(results),
            "search_stack": "token_index + trigram_fuzzy + pairwise_search_rank",
            "active_page": "search",
        }

    def profile_page(self, user_id: int) -> dict[str, object]:
        created_videos = [self._video_card(video_id) for video_id in self.creator_video_ids.get(user_id, [])[:12]]
        return {
            "user": self.get_user_snapshot(user_id),
            "sidebar": self._build_sidebar(user_id),
            "created_videos": [video for video in created_videos if video is not None],
            "active_page": "profile",
        }

    def studio_page(self, user_id: int) -> dict[str, object]:
        created_videos = [self._video_card(video_id) for video_id in self.creator_video_ids.get(user_id, [])[:8]]
        return {
            "user": self.get_user_snapshot(user_id),
            "sidebar": self._build_sidebar(user_id),
            "created_videos": [video for video in created_videos if video is not None],
            "active_page": "studio",
        }

    def history_page(self, user_id: int) -> dict[str, object]:
        return {
            "user": self.get_user_snapshot(user_id),
            "sidebar": self._build_sidebar(user_id),
            "history_items": self._history_page_cards_for_user(user_id),
            "active_page": "history",
        }

    def control_panel_page(self, user_id: int) -> dict[str, object]:
        return {
            "user": self.get_user_snapshot(user_id),
            "sidebar": self._build_sidebar(user_id),
            "dashboard": self._build_user_dashboard(user_id),
            "active_page": "control",
        }

    def guest_home_page(self, guest_id: str) -> dict[str, object]:
        guest_snapshot = self.get_guest_snapshot(guest_id)
        has_guest_signal = self._guest_has_recommendation_signal(guest_id)
        candidate_scores = self._guest_home_scores(guest_id) if has_guest_signal else {}
        ranked_videos = self._cards_from_scores(candidate_scores, limit=DEFAULT_VIDEO_LIMIT) if has_guest_signal else []
        shelves = self._build_guest_home_shelves(ranked_videos, guest_snapshot, guest_id)
        hero_videos = self._home_hero_videos(shelves, ranked_videos)
        return {
            "user": guest_snapshot,
            "sidebar": self._build_guest_sidebar(guest_id),
            "hero_video": ranked_videos[0] if ranked_videos else None,
            "hero_videos": hero_videos,
            "shelves": shelves,
            "model_stack": "guest_session + global_recent",
            "active_page": "home",
            "needs_search": not has_guest_signal,
        }

    def guest_watch_page(self, guest_id: str, video_id: int) -> dict[str, object]:
        video = self._video_card(video_id)
        if video is None:
            raise KeyError(f"No existe video_id={video_id}")
        video = self._apply_guest_video_state(video, guest_id)

        next_candidate_ids = self._watch_next_candidate_ids_for_seen(self._guest_seen_videos(guest_id), video_id, limit=120)
        next_up = self._cards_from_ids(next_candidate_ids, limit=20)
        next_groups = self._build_watch_next_groups_for_guest(guest_id, video_id, video, next_up)
        return {
            "user": self.get_guest_snapshot(guest_id),
            "sidebar": self._build_guest_sidebar(guest_id),
            "video": video,
            "next_up": next_up,
            "next_groups": next_groups,
            "model_stack": "guest_watch_next",
            "active_page": "watch",
        }

    def guest_search_page(self, guest_id: str, query: str) -> dict[str, object]:
        results = self.search_videos(query)
        if str(query or "").strip():
            self._append_search_event("guest", guest_id, query, len(results), "search")
        return {
            "user": self.get_guest_snapshot(guest_id),
            "sidebar": self._build_guest_sidebar(guest_id),
            "query": query,
            "results": results,
            "result_count": len(results),
            "search_stack": "token_index + trigram_fuzzy + pairwise_search_rank",
            "active_page": "search",
        }

    def guest_profile_page(self, guest_id: str) -> dict[str, object]:
        return {
            "user": self.get_guest_snapshot(guest_id),
            "sidebar": self._build_guest_sidebar(guest_id),
            "created_videos": [],
            "active_page": "profile",
        }

    def guest_studio_page(self, guest_id: str) -> dict[str, object]:
        return {
            "user": self.get_guest_snapshot(guest_id),
            "sidebar": self._build_guest_sidebar(guest_id),
            "created_videos": [],
            "active_page": "studio",
        }

    def guest_history_page(self, guest_id: str) -> dict[str, object]:
        return {
            "user": self.get_guest_snapshot(guest_id),
            "sidebar": self._build_guest_sidebar(guest_id),
            "history_items": self._history_page_cards_for_guest(guest_id),
            "active_page": "history",
        }

    def guest_control_panel_page(self, guest_id: str) -> dict[str, object]:
        return {
            "user": self.get_guest_snapshot(guest_id),
            "sidebar": self._build_guest_sidebar(guest_id),
            "dashboard": self._build_guest_dashboard(guest_id),
            "active_page": "control",
        }

    def _build_historical_history(self) -> dict[int, list[int]]:
        histories: dict[int, list[int]] = {}
        ordered_interactions = self.interactions.sort_values("timestamp", ascending=False)
        for user_id, group in ordered_interactions.groupby("user_id", sort=False):
            unique_video_ids: list[int] = []
            seen_video_ids: set[int] = set()
            for video_id in group["video_id"]:
                video_id_int = int(video_id)
                if video_id_int in seen_video_ids:
                    continue
                seen_video_ids.add(video_id_int)
                unique_video_ids.append(video_id_int)
                if len(unique_video_ids) == HISTORICAL_HISTORY_LIMIT:
                    break
            histories[int(user_id)] = unique_video_ids
        return histories

    def _build_creator_video_index(self) -> dict[int, list[int]]:
        catalog = self.videos.copy()
        catalog["published_at"] = pd.to_datetime(catalog["published_at"], errors="coerce")
        catalog = catalog.sort_values("published_at", ascending=False)
        creator_index: dict[int, list[int]] = {}
        for creator_user_id, group in catalog.groupby("creator_user_id", sort=False):
            creator_index[int(creator_user_id)] = [int(video_id) for video_id in group["video_id"].head(20)]
        return creator_index

    def _build_channels_by_creator(self) -> dict[int, list[int]]:
        creator_channels: dict[int, list[int]] = {}
        for row in self.channels.itertuples(index=False):
            creator_user_id = int(row.creator_user_id)
            creator_channels.setdefault(creator_user_id, []).append(int(row.channel_id))
        return creator_channels

    def _build_followed_channels_by_user(self) -> dict[int, list[int]]:
        followed_channels: dict[int, list[int]] = {}
        for user_id, channel_id in self.follow_pairs:
            followed_channels.setdefault(int(user_id), []).append(int(channel_id))
        return followed_channels

    def _resolve_owned_channel_id(self, user_id: int, profile_row: dict[str, object]) -> int | None:
        raw_owned_channel_id = profile_row.get("owned_channel_id")
        if raw_owned_channel_id is not None and not pd.isna(raw_owned_channel_id):
            return int(raw_owned_channel_id)
        creator_channels = self.channels_by_creator.get(user_id, [])
        return creator_channels[0] if creator_channels else None

    def _build_session_focus(self, user_id: int) -> str:
        state = self.session_states.get(user_id)
        if state is None or not state.recent_video_ids:
            if not self._user_has_recommendation_signal(user_id):
                return "Sin historial: empieza buscando"
            favorite_category = str(self.user_feature_map[user_id].get("user_favorite_category", "General"))
            return f"Base historica: {favorite_category}"

        category_counter: Counter[str] = Counter()
        for decay_weight, video_id in self._iter_session_signal_weights(state):
            video_row = self.video_feature_map.get(video_id)
            if video_row is None:
                continue
            category_counter[str(video_row.get("video_category", "General"))] += decay_weight

        if not category_counter:
            return "Sesion activa sin foco claro"
        return f"Sesion activa: {category_counter.most_common(1)[0][0]}"

    def _build_sidebar(self, user_id: int) -> dict[str, object]:
        user_snapshot = self.get_user_snapshot(user_id)
        session_state = self.session_states.get(user_id)
        session_history_ids = [] if session_state is None else list(reversed(session_state.recent_video_ids))
        session_history_set = set(session_history_ids)

        session_history = [self._history_card(video_id, "Sesion") for video_id in session_history_ids]
        historical_history = [
            self._history_card(video_id, "Base")
            for video_id in self.historical_history_by_user.get(user_id, [])
            if video_id not in session_history_set
        ]
        return {
            "session_history": [card for card in session_history if card is not None][:6],
            "historical_history": [card for card in historical_history if card is not None][:6],
            "session_views": user_snapshot["session_views"],
            "session_focus": user_snapshot["session_focus"],
        }

    def get_guest_snapshot(self, guest_id: str) -> dict[str, object]:
        state = self.guest_states.get(guest_id)
        session_views = 0 if state is None else int(state.views_in_session)
        session_focus = self._build_guest_session_focus(guest_id)
        return {
            "user_id": guest_id[-6:].upper(),
            "favorite_category": "Exploracion general",
            "recent_category": "Exploracion general",
            "following_channels": 0,
            "total_interactions": session_views,
            "recent_interactions_30d": session_views,
            "account_role": "guest",
            "account_label": "invitado local",
            "owned_channel_name": "sin cuenta registrada",
            "uploaded_videos": 0,
            "session_views": session_views,
            "session_focus": session_focus,
        }

    def _build_guest_sidebar(self, guest_id: str) -> dict[str, object]:
        state = self.guest_states.get(guest_id)
        session_history_ids = [] if state is None else list(reversed(state.recent_video_ids))
        session_history = [self._history_card(video_id, "Sesion") for video_id in session_history_ids]
        return {
            "session_history": [card for card in session_history if card is not None][:6],
            "historical_history": [],
            "session_views": 0 if state is None else int(state.views_in_session),
            "session_focus": self._build_guest_session_focus(guest_id),
        }

    def _guest_has_recommendation_signal(self, guest_id: str) -> bool:
        state = self.guest_states.get(guest_id)
        return state is not None and bool(state.recent_video_ids or state.engaged_video_ids)

    def _user_has_recommendation_signal(self, user_id: int) -> bool:
        if self.seen_by_user.get(user_id):
            return True
        user_row = self.user_feature_map.get(user_id, {})
        if self._clean_int(user_row.get("user_total_interactions"), 0) > 0:
            return True
        state = self.session_states.get(user_id)
        return state is not None and bool(state.recent_video_ids or state.engaged_video_ids)

    def _build_guest_session_focus(self, guest_id: str) -> str:
        state = self.guest_states.get(guest_id)
        if state is None or not state.recent_video_ids:
            return "Invitado sin historial: mezcla de tendencias y novedad"

        category_counter: Counter[str] = Counter()
        for decay_weight, video_id in self._iter_session_signal_weights(state):
            video_row = self.video_feature_map.get(video_id)
            if video_row is None:
                continue
            category_counter[str(video_row.get("video_category", "General"))] += decay_weight
        if not category_counter:
            return "Invitado sin foco claro"
        return f"Invitado centrado ahora en: {category_counter.most_common(1)[0][0]}"

    def _build_user_dashboard(self, user_id: int) -> dict[str, object]:
        state = self.session_states.get(user_id)
        seen_videos = self._combined_seen_videos(user_id)
        historical_seen = set(self.seen_by_user.get(user_id, set()))
        session_seen = set() if state is None else set(state.recent_video_ids)
        category_signals = self._category_signal_rows(state)
        channel_signals = self._channel_signal_rows(state)
        has_signal = self._user_has_recommendation_signal(user_id)
        candidate_ids = self._home_candidate_ids(user_id, limit=80) if has_signal else []
        ranked_candidates = self._rank_candidates(user_id, candidate_ids, surface_value="home")[:12] if has_signal else []
        event_counts = self._local_event_counts("user", str(user_id))
        session_weight = self._session_weight(user_id)

        return {
            "mode": "Usuario registrado",
            "model_stack": "Retrieval multi-source -> Pairwise XGBoost ranker -> blend de sesion",
            "catalog": self._catalog_dashboard_stats(),
            "activity": self._activity_payload(
                seen_total=len(seen_videos),
                historical_seen=len(historical_seen),
                session_seen=len(session_seen),
                liked_count=0 if state is None else len(state.liked_video_ids),
                subscribed_count=0 if state is None else len(state.subscribed_channel_ids),
                event_counts=event_counts,
                session_weight=session_weight,
            ),
            "signals": {
                "category_rows": category_signals,
                "channel_rows": channel_signals,
                "historical_weight_pct": round((1.0 - session_weight) * 100, 1),
                "session_weight_pct": round(session_weight * 100, 1),
            },
            "recommendations": ranked_candidates,
            "quality": self._quality_dashboard(
                recommendations=ranked_candidates,
                seen_videos=seen_videos,
                session_categories={row["name"] for row in category_signals},
                has_signal=has_signal,
            ),
            "pipeline_steps": self._pipeline_steps(),
            "controls": self._control_rules(),
        }

    def _build_guest_dashboard(self, guest_id: str) -> dict[str, object]:
        state = self.guest_states.get(guest_id)
        seen_videos = self._guest_seen_videos(guest_id)
        scores = self._guest_home_scores(guest_id)
        ranked_candidates = self._cards_from_scores(scores, limit=12)
        event_counts = self._local_event_counts("guest", guest_id)
        session_weight = 1.0 if state is not None and (state.recent_video_ids or state.engaged_video_ids) else 0.0

        return {
            "mode": "Invitado local",
            "model_stack": "Guest session -> item/category/channel retrieval -> ranking ligero por score",
            "catalog": self._catalog_dashboard_stats(),
            "activity": self._activity_payload(
                seen_total=len(seen_videos),
                historical_seen=0,
                session_seen=len(seen_videos),
                liked_count=0 if state is None else len(state.liked_video_ids),
                subscribed_count=0 if state is None else len(state.subscribed_channel_ids),
                event_counts=event_counts,
                session_weight=session_weight,
            ),
            "signals": {
                "category_rows": self._category_signal_rows(state),
                "channel_rows": self._channel_signal_rows(state),
                "historical_weight_pct": 0.0,
                "session_weight_pct": round(session_weight * 100, 1),
            },
            "recommendations": ranked_candidates,
            "quality": self._quality_dashboard(
                recommendations=ranked_candidates,
                seen_videos=seen_videos,
                session_categories={row["name"] for row in self._category_signal_rows(state)},
                has_signal=bool(state is not None and (state.recent_video_ids or state.engaged_video_ids)),
            ),
            "pipeline_steps": self._pipeline_steps(),
            "controls": self._control_rules(),
        }

    def _catalog_dashboard_stats(self) -> dict[str, object]:
        category_counts = self.videos["category"].fillna("unknown").astype(str).value_counts().head(8)
        recent_count = int(pd.to_numeric(self.video_features["is_recent_upload"], errors="coerce").fillna(0).sum())
        return {
            "total_videos": int(len(self.videos)),
            "total_channels": int(len(self.channels)),
            "total_users": int(len(self.user_features)),
            "recent_videos": recent_count,
            "category_rows": [
                {
                    "name": str(category),
                    "count": int(count),
                    "pct": round((int(count) / max(len(self.videos), 1)) * 100, 1),
                }
                for category, count in category_counts.items()
            ],
        }

    def _activity_payload(
        self,
        seen_total: int,
        historical_seen: int,
        session_seen: int,
        liked_count: int,
        subscribed_count: int,
        event_counts: dict[str, int],
        session_weight: float,
    ) -> dict[str, object]:
        return {
            "seen_total": seen_total,
            "historical_seen": historical_seen,
            "session_seen": session_seen,
            "liked_count": liked_count,
            "subscribed_count": subscribed_count,
            "local_views": event_counts.get("views", 0),
            "local_searches": event_counts.get("searches", 0),
            "local_likes": event_counts.get("likes", 0),
            "local_subscriptions": event_counts.get("subscriptions", 0),
            "session_weight_pct": round(session_weight * 100, 1),
        }

    def _category_signal_rows(self, state: SessionState | None) -> list[dict[str, object]]:
        if state is None:
            return []
        scores = self._session_preference_maps_from_state(state)["category_scores"]
        return [
            {"name": str(category), "score": round(float(score), 3), "pct": round(float(score) * 100, 1)}
            for category, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:6]
        ]

    def _channel_signal_rows(self, state: SessionState | None) -> list[dict[str, object]]:
        if state is None:
            return []
        scores = self._session_preference_maps_from_state(state)["channel_scores"]
        rows: list[dict[str, object]] = []
        for channel_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:6]:
            channel = self.channel_map.get(int(channel_id), {})
            rows.append(
                {
                    "name": self._clean_text(channel.get("channel_name"), f"channel_{channel_id}"),
                    "score": round(float(score), 3),
                    "pct": round(float(score) * 100, 1),
                }
            )
        return rows

    def _local_event_counts(self, actor_type: str, actor_id: str) -> dict[str, int]:
        counts = {"views": 0, "searches": 0, "likes": 0, "subscriptions": 0}
        if LOCAL_SESSION_EVENTS_PATH.exists():
            events = pd.read_csv(LOCAL_SESSION_EVENTS_PATH)
            if not events.empty:
                mask = (events["actor_type"].astype(str) == actor_type) & (events["actor_id"].astype(str) == actor_id)
                counts["views"] = int(mask.sum())
        if LOCAL_SEARCH_EVENTS_PATH.exists():
            events = pd.read_csv(LOCAL_SEARCH_EVENTS_PATH)
            if not events.empty:
                mask = (events["actor_type"].astype(str) == actor_type) & (events["actor_id"].astype(str) == actor_id)
                counts["searches"] = int(mask.sum())
        if LOCAL_ENGAGEMENT_EVENTS_PATH.exists():
            events = pd.read_csv(LOCAL_ENGAGEMENT_EVENTS_PATH)
            if not events.empty:
                mask = (events["actor_type"].astype(str) == actor_type) & (events["actor_id"].astype(str) == actor_id)
                actor_events = events[mask]
                counts["likes"] = int((actor_events["event_type"].astype(str) == "like").sum())
                counts["subscriptions"] = int((actor_events["event_type"].astype(str) == "subscribe").sum())
        return counts

    def _quality_dashboard(
        self,
        recommendations: list[dict[str, object]],
        seen_videos: set[int],
        session_categories: set[str],
        has_signal: bool,
    ) -> dict[str, object]:
        total = len(recommendations)
        repeated_seen = sum(1 for video in recommendations if int(video["video_id"]) in seen_videos)
        categories = [str(video.get("category", "unknown")) for video in recommendations]
        channels = [int(video.get("channel_id", 0)) for video in recommendations]
        new_count = sum(1 for video in recommendations if int(video.get("is_recent_upload", 0)) == 1)
        session_match = (
            sum(1 for category in categories if category in session_categories)
            if session_categories
            else 0
        )
        unique_categories = len(set(categories))
        unique_channels = len(set(channels))
        diversity_pct = round((unique_categories / max(total, 1)) * 100, 1)
        channel_diversity_pct = round((unique_channels / max(total, 1)) * 100, 1)
        freshness_pct = round((new_count / max(total, 1)) * 100, 1)
        session_match_pct = round((session_match / max(total, 1)) * 100, 1)
        seen_filter_ok = repeated_seen == 0
        enough_candidates = total >= 8 if has_signal else total == 0
        diversity_ok = unique_categories >= min(3, total) if total else not has_signal
        session_ok = session_match_pct >= 35 if session_categories and total else True
        freshness_ok = freshness_pct > 0 if total else True
        health_items = [
            self._health_item("Filtro de vistos", seen_filter_ok, f"{repeated_seen} repetidos en top {total}"),
            self._health_item("Candidatos suficientes", enough_candidates, f"{total} recomendaciones generadas"),
            self._health_item("Diversidad de temas", diversity_ok, f"{unique_categories} categorias en top {total}"),
            self._health_item("Coherencia con sesion", session_ok, f"{session_match_pct}% encaja con temas activos"),
            self._health_item("Novedad controlada", freshness_ok, f"{freshness_pct}% videos nuevos"),
        ]
        ok_count = sum(1 for item in health_items if item["status"] == "ok")
        warn_count = sum(1 for item in health_items if item["status"] == "warn")
        global_metrics = self._global_quality_metrics()
        return {
            "summary": {
                "status": "correcto" if warn_count == 0 else "revisar",
                "ok_count": ok_count,
                "warn_count": warn_count,
                "message": (
                    "El recomendador esta sano para esta sesion."
                    if warn_count == 0
                    else "Hay senales que conviene revisar antes de defender la demo."
                ),
            },
            "health_items": health_items,
            "local_metrics": [
                {"label": "Vistos repetidos", "value": repeated_seen, "detail": "deberia ser 0"},
                {"label": "Diversidad categorias", "value": f"{diversity_pct}%", "detail": f"{unique_categories} temas"},
                {"label": "Diversidad canales", "value": f"{channel_diversity_pct}%", "detail": f"{unique_channels} canales"},
                {"label": "Videos nuevos", "value": f"{freshness_pct}%", "detail": "frescura/cold-start"},
                {"label": "Match sesion", "value": f"{session_match_pct}%", "detail": "coherencia actual"},
            ],
            "explain_demo": [
                "Ocultamos videos ya vistos para no repetir contenido.",
                "La sesion actual cambia la home, pero con limite para no crear una burbuja instantanea.",
                "El ranker ordena candidatos usando senales de usuario, video, canal, frescura y afinidad.",
                "La calidad se mira con checks locales y con metricas offline del ranker/retrieval.",
            ],
            "global_metrics": global_metrics,
            "offline_report": self._offline_quality_report(),
        }

    @staticmethod
    def _health_item(name: str, ok: bool, detail: str) -> dict[str, str]:
        return {
            "name": name,
            "status": "ok" if ok else "warn",
            "label": "OK" if ok else "Revisar",
            "detail": detail,
        }

    def _global_quality_metrics(self) -> list[dict[str, str]]:
        metrics: list[dict[str, str]] = []
        retrieval_metrics = self._read_json(RETRIEVAL_METRICS_PATH)
        ranker_metrics = self._read_json(RANKER_COMPARISON_PATH)
        diagnostics = self._read_json(RETRIEVAL_DIAGNOSTICS_PATH)

        retrieval_test = retrieval_metrics.get("test", {}) if isinstance(retrieval_metrics, dict) else {}
        if retrieval_test:
            metrics.append(
                {
                    "label": "Retrieval category_hit@50",
                    "value": self._format_pct(float(retrieval_test.get("category_hit@50", 0))),
                    "detail": "recupera bien el tema aunque no siempre el item exacto",
                }
            )
            metrics.append(
                {
                    "label": "Retrieval candidatos",
                    "value": str(round(float(retrieval_test.get("mean_candidates_returned", 0)), 1)),
                    "detail": "media de candidatos por usuario evaluado",
                }
            )

        pairwise_test = {}
        if isinstance(ranker_metrics, dict):
            pairwise_test = ranker_metrics.get("pairwise_xgboost", {}).get("test", {})
        if pairwise_test:
            metrics.append(
                {
                    "label": "Ranker precision@1",
                    "value": self._format_pct(float(pairwise_test.get("precision_at_1", 0))),
                    "detail": "capacidad de poner un buen candidato arriba",
                }
            )
            metrics.append(
                {
                    "label": "Ranker NDCG@10",
                    "value": self._format_pct(float(pairwise_test.get("ndcg_at_10", 0))),
                    "detail": "calidad del orden en el top 10",
                }
            )

        conclusion = diagnostics.get("conclusion", {}) if isinstance(diagnostics, dict) else {}
        if conclusion:
            metrics.append(
                {
                    "label": "Diagnostico",
                    "value": "tema OK",
                    "detail": str(conclusion.get("root_cause", ""))[:120],
                }
            )
        return metrics

    def _offline_quality_report(self) -> dict[str, object]:
        report = self._read_json(QUALITY_SUMMARY_PATH)
        if not report:
            return {}

        ranking_metrics = report.get("ranking_metrics", {})
        negative_sampling = report.get("negative_sampling", {})
        candidate_pool = report.get("candidate_pool", {})
        retrieval_metrics = report.get("retrieval_metrics", {})
        retrieval_test = retrieval_metrics.get("test", {}) if isinstance(retrieval_metrics, dict) else {}
        return {
            "available": True,
            "ranking": report.get("charts", {}).get("ranking", []),
            "ux": report.get("charts", {}).get("ux", []),
            "negative_sources": report.get("charts", {}).get("negative_sources", []),
            "candidate_pool": candidate_pool.get("by_split", []),
            "candidate_warning": candidate_pool.get("warning", ""),
            "takeaways": report.get("demo_takeaway", []),
            "headline_metrics": [
                {
                    "label": "Usuarios evaluados",
                    "value": str(ranking_metrics.get("evaluated_users", "n/a")),
                    "detail": "usuarios del split test con candidatos",
                },
                {
                    "label": "Coverage@10",
                    "value": self._format_pct(float(ranking_metrics.get("coverage_at_10", 0))),
                    "detail": "catalogo distinto recomendado",
                },
                {
                    "label": "Repeat vistos@10",
                    "value": self._format_pct(float(ranking_metrics.get("seen_repeat_rate_at_10", 0))),
                    "detail": "cuanto se cuelan vistos historicos",
                },
                {
                    "label": "Retrieval category_hit@50",
                    "value": self._format_pct(float(retrieval_test.get("category_hit@50", 0))),
                    "detail": "recupera el tema correcto",
                },
            ],
            "negative_summary": [
                {
                    "label": "Positivos",
                    "value": str(negative_sampling.get("positive_rows", "n/a")),
                    "detail": "alta retencion o accion positiva",
                },
                {
                    "label": "Negativos observados",
                    "value": str(negative_sampling.get("observed_negative_rows", "n/a")),
                    "detail": "baja retencion sin acciones positivas",
                },
                {
                    "label": "Negativos sinteticos",
                    "value": str(negative_sampling.get("synthetic_negative_rows", "n/a")),
                    "detail": "no vistos plausibles",
                },
                {
                    "label": "Hard proxy",
                    "value": self._format_pct(float(negative_sampling.get("synthetic_hard_proxy_rate", 0))),
                    "detail": "sinteticos afines por categoria/follow",
                },
            ],
        }

    @staticmethod
    def _read_json(path: Path) -> dict[str, object]:
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as file_handle:
                return json.load(file_handle)
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _format_pct(value: float) -> str:
        return f"{value * 100:.1f}%"

    @staticmethod
    def _pipeline_steps() -> list[dict[str, str]]:
        return [
            {
                "title": "1. Retrieval",
                "body": "Saca candidatos desde collaborative filtering, categorias preferidas, canales vistos/seguidos y videos recientes.",
            },
            {
                "title": "2. Ranking",
                "body": "Ordena candidatos con un ranker pairwise XGBoost usando features de usuario, video, canal, frescura y afinidad.",
            },
            {
                "title": "3. Sesion actual",
                "body": "Recalcula la home con las ultimas vistas. Tiene peso limitado para no convertir dos likes en una burbuja total.",
            },
            {
                "title": "4. Filtros de producto",
                "body": "No vuelve a mostrar vistos, separa listas por objetivo y usa cuotas suaves para mantener variedad.",
            },
        ]

    @staticmethod
    def _control_rules() -> list[dict[str, str]]:
        return [
            {
                "name": "Vistos",
                "value": "bloqueados",
                "detail": "Si el video ya se ha visto en historial o sesion, se excluye de home/watch-next.",
            },
            {
                "name": "Likes",
                "value": "peso bajo",
                "detail": "Refuerzan tema, pero no dominan la recomendacion.",
            },
            {
                "name": "Suscripciones",
                "value": "afinidad de canal",
                "detail": "Afectan sobre todo dentro de temas ya relevantes, no fuerzan todo el feed.",
            },
            {
                "name": "Novedad",
                "value": "boost controlado",
                "detail": "Los videos nuevos pueden entrar aunque no tengan historial, pero deben competir con el ranker.",
            },
        ]

    def _history_card(self, video_id: int, source_label: str) -> dict[str, object] | None:
        card = self._video_card(video_id)
        if card is None:
            return None
        return {
            "video_id": card["video_id"],
            "title": card["title"],
            "channel_name": card["channel_name"],
            "category": card["category"],
            "source_label": source_label,
        }

    def _history_page_cards_for_user(self, user_id: int) -> list[dict[str, object]]:
        state = self.session_states.get(user_id)
        ordered_ids: list[tuple[int, str]] = []
        seen_ids: set[int] = set()

        if state is not None:
            for video_id in reversed(state.recent_video_ids):
                video_id_int = int(video_id)
                if video_id_int in seen_ids:
                    continue
                seen_ids.add(video_id_int)
                ordered_ids.append((video_id_int, "Sesion actual"))

        historical_rows = self.interactions[self.interactions["user_id"] == user_id].sort_values("timestamp", ascending=False)
        for video_id in historical_rows["video_id"]:
            video_id_int = int(video_id)
            if video_id_int in seen_ids:
                continue
            seen_ids.add(video_id_int)
            ordered_ids.append((video_id_int, "Historial base"))
            if len(ordered_ids) >= HISTORY_PAGE_LIMIT:
                break

        return self._history_page_cards(ordered_ids, user_id=user_id)

    def _history_page_cards_for_guest(self, guest_id: str) -> list[dict[str, object]]:
        state = self.guest_states.get(guest_id)
        if state is None:
            return []

        ordered_ids: list[tuple[int, str]] = []
        seen_ids: set[int] = set()
        for video_id in reversed(state.recent_video_ids):
            video_id_int = int(video_id)
            if video_id_int in seen_ids:
                continue
            seen_ids.add(video_id_int)
            ordered_ids.append((video_id_int, "Sesion local"))

        return self._history_page_cards(ordered_ids, guest_id=guest_id)

    def _history_page_cards(
        self,
        ordered_ids: list[tuple[int, str]],
        user_id: int | None = None,
        guest_id: str | None = None,
    ) -> list[dict[str, object]]:
        cards: list[dict[str, object]] = []
        for video_id, source_label in ordered_ids[:HISTORY_PAGE_LIMIT]:
            card = self._video_card(video_id)
            if card is None:
                continue
            if user_id is not None:
                card = self._apply_user_video_state(card, user_id)
            elif guest_id is not None:
                card = self._apply_guest_video_state(card, guest_id)
            card["source_label"] = source_label
            cards.append(card)
        return cards

    def _build_search_index(self) -> SearchIndex:
        documents: list[SearchDocument] = []
        token_index: dict[str, list[int]] = defaultdict(list)
        trigram_index: dict[str, list[int]] = defaultdict(list)

        for row in self.videos.itertuples(index=False):
            video_id = int(getattr(row, "video_id"))
            title = self._clean_text(getattr(row, "titulo", None), "")
            if not title:
                continue

            category = self._clean_text(getattr(row, "category", None), "")
            tags = self._clean_text(getattr(row, "que_pasa", None), "")
            searchable_text = f"{title} {category} {tags}"
            normalized_title = self._normalize_search_text(title)
            normalized_searchable = self._normalize_search_text(searchable_text)
            tokens = self._tokenize_search_text(normalized_searchable)
            trigrams = self._char_ngrams(normalized_searchable)
            document_index = len(documents)
            documents.append(
                SearchDocument(
                    video_id=video_id,
                    title=title,
                    normalized_title=normalized_title,
                    tokens=tokens,
                    trigrams=trigrams,
                    views=self._clean_int(getattr(row, "total_views", 0), 0),
                )
            )
            for token in tokens:
                token_index[token].append(document_index)
            for trigram in trigrams:
                trigram_index[trigram].append(document_index)

        return SearchIndex(
            documents=documents,
            token_index=dict(token_index),
            trigram_index=dict(trigram_index),
        )

    def search_videos(self, query: str, limit: int = 36) -> list[dict[str, object]]:
        query_norm = self._normalize_search_text(query)
        if len(query_norm) < 2:
            return []

        query_tokens = set(query_norm.split())
        query_trigrams = self._char_ngrams(query_norm) if len(query_norm) >= 3 else set()
        candidate_indexes = self._collect_search_candidates(query_norm, query_tokens, query_trigrams, limit)
        if not candidate_indexes:
            return []
        candidate_indexes = self._filter_search_candidates_by_category_intent(candidate_indexes, query_norm)
        if not candidate_indexes:
            return []

        scores = {
            candidate_index: self._score_search_candidate(candidate_index, query_norm, query_tokens, query_trigrams)
            for candidate_index in candidate_indexes
        }
        candidate_indexes = self._filter_search_candidates_by_intent(candidate_indexes, scores, query_tokens)
        if not candidate_indexes:
            return []
        ordered_indexes = sorted(
            candidate_indexes,
            key=lambda candidate_index: (
                scores[candidate_index],
                self.search_index.documents[candidate_index].views,
                self.search_index.documents[candidate_index].title,
            ),
            reverse=True,
        )
        deduped_indexes = self._dedupe_search_titles(ordered_indexes)
        ranked_indexes = self._pairwise_search_rank(deduped_indexes, scores)
        cards: list[dict[str, object]] = []
        for rank, index in enumerate(ranked_indexes[:limit], start=1):
            card = self._video_card(self.search_index.documents[index].video_id)
            if card is None:
                continue
            card["search_rank"] = rank
            card["search_score"] = round(float(scores[index]), 4)
            card["search_ranker"] = "pairwise_search_rank"
            cards.append(card)
        return cards

    def search_suggestions(self, query: str, limit: int = 8) -> list[dict[str, object]]:
        query = str(query or "").strip()
        suggestions: list[dict[str, object]] = []
        for video in self.search_videos(query, limit=limit):
            suggestions.append(
                {
                    "video_id": video["video_id"],
                    "title": video["title"],
                    "channel_name": video["channel_name"],
                    "category": video["category"],
                    "thumbnail_url": video["thumbnail_url"],
                    "duration_text": video["duration_text"],
                    "rank": video["search_rank"],
                    "score": video["search_score"],
                }
            )
        return suggestions

    def _collect_search_candidates(
        self,
        query_norm: str,
        query_tokens: set[str],
        query_trigrams: set[str],
        result_limit: int = 36,
    ) -> list[int]:
        phrase_indexes = {
            index
            for index, document in enumerate(self.search_index.documents)
            if query_norm in document.normalized_title
        }
        if phrase_indexes and len(query_tokens) > 1:
            return list(phrase_indexes)

        direct_indexes: set[int] = set()
        direct_indexes.update(phrase_indexes)
        if query_tokens:
            token_lists = [
                self.search_index.token_index.get(token, [])
                for token in sorted(query_tokens, key=lambda token: len(self.search_index.token_index.get(token, [])))
            ]
            if len(token_lists) == 1:
                direct_indexes.update(token_lists[0])
            else:
                token_counts: dict[int, int] = {}
                min_matches = max(1, math.ceil(len(query_tokens) * 0.55))
                for indexes in token_lists:
                    for index in indexes:
                        token_counts[index] = token_counts.get(index, 0) + 1
                for index, count in token_counts.items():
                    if count >= min_matches:
                        direct_indexes.add(index)

        if direct_indexes:
            return list(direct_indexes)

        fuzzy_indexes: set[int] = set()
        if query_trigrams:
            trigram_counts: dict[int, int] = {}
            for trigram in query_trigrams:
                for index in self.search_index.trigram_index.get(trigram, []):
                    trigram_counts[index] = trigram_counts.get(index, 0) + 1
            min_overlap = max(2, math.ceil(len(query_trigrams) * 0.45))
            for index, count in sorted(trigram_counts.items(), key=lambda item: item[1], reverse=True)[:2000]:
                if count >= min_overlap:
                    fuzzy_indexes.add(index)
                if len(fuzzy_indexes) >= result_limit * 4:
                    break

        return list(fuzzy_indexes)

    def _filter_search_candidates_by_category_intent(self, candidate_indexes: list[int], query_norm: str) -> list[int]:
        category = SEARCH_CATEGORY_ALIASES.get(query_norm)
        if category is None:
            return candidate_indexes

        filtered_indexes = [
            index
            for index in candidate_indexes
            if self._clean_text(
                self.video_feature_map.get(self.search_index.documents[index].video_id, {}).get("video_category"),
                "",
            )
            == category
        ]
        return filtered_indexes or candidate_indexes

    def _filter_search_candidates_by_intent(
        self,
        candidate_indexes: list[int],
        scores: dict[int, float],
        query_tokens: set[str],
    ) -> list[int]:
        if not candidate_indexes:
            return []

        exact_indexes = [
            index
            for index in candidate_indexes
            if query_tokens and query_tokens.issubset(self.search_index.documents[index].tokens)
        ]
        if exact_indexes:
            return exact_indexes

        max_score = max(scores[index] for index in candidate_indexes)
        min_score = max(1.35, max_score * 0.68)
        return [index for index in candidate_indexes if scores[index] >= min_score]

    def _score_search_candidate(
        self,
        candidate_index: int,
        query_norm: str,
        query_tokens: set[str],
        query_trigrams: set[str],
    ) -> float:
        document = self.search_index.documents[candidate_index]
        score = 0.0
        if document.normalized_title.startswith(query_norm):
            score += 3.0
        if query_norm in document.normalized_title:
            score += 2.0
        if query_tokens:
            score += len(query_tokens & document.tokens) / len(query_tokens)
        if query_trigrams:
            score += 1.5 * (len(query_trigrams & document.trigrams) / len(query_trigrams))
        score += difflib.SequenceMatcher(None, query_norm, document.normalized_title).ratio()
        return score

    def _dedupe_search_titles(self, ordered_indexes: list[int]) -> list[int]:
        unique_indexes: list[int] = []
        seen_titles: set[str] = set()
        seen_title_families: set[str] = set()
        for index in ordered_indexes:
            normalized_title = self.search_index.documents[index].normalized_title
            title_family = self._search_title_family(normalized_title)
            if normalized_title in seen_titles or title_family in seen_title_families:
                continue
            seen_titles.add(normalized_title)
            seen_title_families.add(title_family)
            unique_indexes.append(index)
        return unique_indexes

    @staticmethod
    def _search_title_family(normalized_title: str) -> str:
        title = normalized_title
        prefixes = [
            "lo que aprendi ",
            "lo mas importante ",
            "en pocos minutos ",
            "la explicacion clara ",
            "lo que nadie te dice de ",
            "mi experiencia con ",
            "esto cambio todo ",
            "nunca mas hare esto ",
            "no lo vas a entender ",
            "no hagas lo mismo que yo ",
            "guia 2026 ",
        ]
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if title.startswith(prefix):
                    title = title[len(prefix) :].strip()
                    changed = True
        return " ".join(title.split()[:8])

    def _pairwise_search_rank(self, candidate_indexes: list[int], scores: dict[int, float]) -> list[int]:
        if len(candidate_indexes) <= 1:
            return candidate_indexes

        limit = min(250, len(candidate_indexes))
        head = candidate_indexes[:limit]
        wins = {index: 0 for index in head}
        for position, left in enumerate(head):
            for right in head[position + 1 :]:
                winner = self._pairwise_search_winner(left, right, scores)
                wins[winner] += 1

        ranked_head = sorted(
            head,
            key=lambda index: (
                wins[index],
                scores[index],
                self.search_index.documents[index].views,
                self.search_index.documents[index].title.lower(),
            ),
            reverse=True,
        )
        return ranked_head + candidate_indexes[limit:]

    def _pairwise_search_winner(self, left: int, right: int, scores: dict[int, float]) -> int:
        left_score = scores[left]
        right_score = scores[right]
        if left_score != right_score:
            return left if left_score > right_score else right

        left_views = self.search_index.documents[left].views
        right_views = self.search_index.documents[right].views
        if left_views != right_views:
            return left if left_views > right_views else right

        left_title = self.search_index.documents[left].title.lower()
        right_title = self.search_index.documents[right].title.lower()
        return left if left_title <= right_title else right

    @staticmethod
    def _normalize_search_text(text: object) -> str:
        raw = "" if text is None or pd.isna(text) else str(text)
        raw = raw.lower().strip()
        raw = unicodedata.normalize("NFKD", raw)
        raw = "".join(character for character in raw if not unicodedata.combining(character))
        raw = re.sub(r"[^a-z0-9\s]", " ", raw)
        return re.sub(r"\s+", " ", raw).strip()

    @staticmethod
    def _tokenize_search_text(normalized_text: str) -> set[str]:
        return {token for token in normalized_text.split() if len(token) > 2 and token not in SEARCH_STOPWORDS}

    @staticmethod
    def _char_ngrams(normalized_text: str, n: int = 3) -> set[str]:
        condensed = normalized_text.replace(" ", "")
        if not condensed:
            return set()
        if len(condensed) <= n:
            return {condensed}
        return {condensed[index : index + n] for index in range(len(condensed) - n + 1)}

    def _home_candidate_ids(self, user_id: int, limit: int) -> list[int]:
        historical_scores = self._historical_home_scores(user_id)
        session_scores = self._session_home_scores(user_id)
        blended_scores = self._blend_score_maps(historical_scores, session_scores, self._session_weight(user_id))
        return top_items(blended_scores, limit)

    def _historical_home_scores(self, user_id: int) -> dict[int, float]:
        bundle = self.retrieval_bundle
        cf_model = bundle["cf_model"]
        seen_videos = self._combined_seen_videos(user_id)
        return build_multisource_scores(
            user_id=user_id,
            seen_videos=seen_videos,
            user_histories=cf_model["user_histories"],
            user_norms=cf_model["user_norms"],
            item_users=cf_model["item_users"],
            item_neighbors=cf_model["item_neighbors"],
            user_preferences=bundle["user_preferences"],
            user_channel_index=bundle["user_channel_index"],
            category_top_videos=bundle["category_top_videos"],
            recent_category_top_videos=bundle["recent_category_top_videos"],
            channel_top_videos=bundle["channel_top_videos"],
            global_recent=bundle["global_recent"],
        )

    def _session_home_scores(self, user_id: int) -> dict[int, float]:
        state = self.session_states.get(user_id)
        if state is None or (not state.recent_video_ids and not state.engaged_video_ids):
            return {}

        bundle = self.retrieval_bundle
        cf_model = bundle["cf_model"]
        seen_videos = self._combined_seen_videos(user_id)
        candidate_scores: dict[int, float] = {}

        for decay_weight, video_id in self._iter_session_signal_weights(state):
            video_row = self.video_feature_map.get(video_id)
            if video_row is None:
                continue

            for rank_index, (similarity, candidate_video_id) in enumerate(cf_model["item_neighbors"].get(video_id, [])[:80], start=1):
                if candidate_video_id in seen_videos or candidate_video_id == video_id:
                    continue
                candidate_scores[int(candidate_video_id)] = candidate_scores.get(int(candidate_video_id), 0.0) + (
                    0.55 * decay_weight * float(similarity) / math.sqrt(rank_index)
                )

            channel_id = int(video_row["channel_id"])
            for rank_index, candidate_video_id in enumerate(bundle["channel_top_videos"].get(channel_id, [])[:20], start=1):
                if candidate_video_id in seen_videos or candidate_video_id == video_id:
                    continue
                candidate_scores[int(candidate_video_id)] = candidate_scores.get(int(candidate_video_id), 0.0) + (
                    0.18 * decay_weight / math.sqrt(rank_index)
                )

            category = str(video_row["video_category"])
            for rank_index, candidate_video_id in enumerate(bundle["category_top_videos"].get(category, [])[:40], start=1):
                if candidate_video_id in seen_videos or candidate_video_id == video_id:
                    continue
                candidate_scores[int(candidate_video_id)] = candidate_scores.get(int(candidate_video_id), 0.0) + (
                    0.28 * decay_weight / math.sqrt(rank_index)
                )

        self._add_subscription_channel_scores(state, candidate_scores, seen_videos, set(), bundle, boost=0.10)
        return candidate_scores

    def _guest_home_scores(self, guest_id: str) -> dict[int, float]:
        state = self.guest_states.get(guest_id)
        if state is None or (not state.recent_video_ids and not state.engaged_video_ids and not state.subscribed_channel_ids):
            return {int(video_id): 1.0 / math.sqrt(index) for index, video_id in enumerate(self.retrieval_bundle["global_recent"][:200], start=1)}

        bundle = self.retrieval_bundle
        cf_model = bundle["cf_model"]
        seen_videos = self._guest_seen_videos(guest_id)
        candidate_scores: dict[int, float] = {int(video_id): 0.03 / math.sqrt(index) for index, video_id in enumerate(bundle["global_recent"][:80], start=1)}
        for decay_weight, video_id in self._iter_session_signal_weights(state):
            video_row = self.video_feature_map.get(video_id)
            if video_row is None:
                continue

            for rank_index, (similarity, candidate_video_id) in enumerate(cf_model["item_neighbors"].get(video_id, [])[:80], start=1):
                if candidate_video_id in seen_videos or candidate_video_id == video_id:
                    continue
                candidate_scores[int(candidate_video_id)] = candidate_scores.get(int(candidate_video_id), 0.0) + (
                    0.60 * decay_weight * float(similarity) / math.sqrt(rank_index)
                )

            channel_id = int(video_row["channel_id"])
            for rank_index, candidate_video_id in enumerate(bundle["channel_top_videos"].get(channel_id, [])[:20], start=1):
                if candidate_video_id in seen_videos or candidate_video_id == video_id:
                    continue
                candidate_scores[int(candidate_video_id)] = candidate_scores.get(int(candidate_video_id), 0.0) + (
                    0.18 * decay_weight / math.sqrt(rank_index)
                )

            category = str(video_row["video_category"])
            for rank_index, candidate_video_id in enumerate(bundle["category_top_videos"].get(category, [])[:50], start=1):
                if candidate_video_id in seen_videos or candidate_video_id == video_id:
                    continue
                candidate_scores[int(candidate_video_id)] = candidate_scores.get(int(candidate_video_id), 0.0) + (
                    0.32 * decay_weight / math.sqrt(rank_index)
                )
        self._add_subscription_channel_scores(state, candidate_scores, seen_videos, set(), bundle, boost=0.10)
        return candidate_scores

    def _watch_next_candidate_ids(self, user_id: int, current_video_id: int, limit: int) -> list[int]:
        seen_videos = self._combined_seen_videos(user_id)
        return self._watch_next_candidate_ids_for_seen(seen_videos, current_video_id, limit)

    def _watch_next_candidate_ids_for_seen(self, seen_videos: set[int], current_video_id: int, limit: int) -> list[int]:
        bundle = self.retrieval_bundle
        cf_model = bundle["cf_model"]
        current_video = self.video_feature_map.get(current_video_id)
        if current_video is None:
            return []

        candidate_scores: dict[int, float] = {}

        item_neighbors = cf_model["item_neighbors"].get(current_video_id, [])
        for rank_index, (similarity, candidate_video_id) in enumerate(item_neighbors[:100], start=1):
            if candidate_video_id in seen_videos or candidate_video_id == current_video_id:
                continue
            candidate_scores[int(candidate_video_id)] = candidate_scores.get(int(candidate_video_id), 0.0) + 0.55 * float(similarity) / math.sqrt(rank_index)

        channel_id = int(current_video["channel_id"])
        for rank_index, candidate_video_id in enumerate(bundle["channel_top_videos"].get(channel_id, [])[:30], start=1):
            if candidate_video_id in seen_videos or candidate_video_id == current_video_id:
                continue
            candidate_scores[int(candidate_video_id)] = candidate_scores.get(int(candidate_video_id), 0.0) + 0.25 / math.sqrt(rank_index)

        category = str(current_video["video_category"])
        for rank_index, candidate_video_id in enumerate(bundle["category_top_videos"].get(category, [])[:60], start=1):
            if candidate_video_id in seen_videos or candidate_video_id == current_video_id:
                continue
            candidate_scores[int(candidate_video_id)] = candidate_scores.get(int(candidate_video_id), 0.0) + 0.15 / math.sqrt(rank_index)

        for rank_index, candidate_video_id in enumerate(bundle["global_recent"][:40], start=1):
            if candidate_video_id in seen_videos or candidate_video_id == current_video_id:
                continue
            candidate_scores[int(candidate_video_id)] = candidate_scores.get(int(candidate_video_id), 0.0) + 0.05 / math.sqrt(rank_index)

        return top_items(candidate_scores, limit)

    def _build_watch_next_groups_for_user(
        self,
        user_id: int,
        current_video_id: int,
        current_card: dict[str, object],
        related_cards: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        seen_videos = self._combined_seen_videos(user_id)
        current_category = str(current_card["category"])
        channel_cards = self._same_channel_watch_cards(current_video_id, seen_videos, limit=20)

        for_you_candidate_ids = self._home_candidate_ids(user_id, limit=160)
        for_you_cards = self._rank_candidates(user_id, for_you_candidate_ids, surface_value="home", exclude_video_id=current_video_id)[:20]
        all_cards = self._merge_watch_cards([related_cards, channel_cards, for_you_cards], limit=24)

        return self._watch_groups_payload(
            all_cards=all_cards,
            related_cards=[card for card in related_cards if card["category"] == current_category][:20],
            channel_cards=channel_cards,
            for_you_cards=for_you_cards,
            channel_title=str(current_card["channel_name"]),
        )

    def _build_watch_next_groups_for_guest(
        self,
        guest_id: str,
        current_video_id: int,
        current_card: dict[str, object],
        related_cards: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        seen_videos = self._guest_seen_videos(guest_id)
        current_category = str(current_card["category"])
        channel_cards = self._same_channel_watch_cards(current_video_id, seen_videos, limit=20)
        for_you_scores = self._guest_home_scores(guest_id)
        for_you_cards = [card for card in self._cards_from_scores(for_you_scores, limit=40) if card["video_id"] != current_video_id][:20]
        all_cards = self._merge_watch_cards([related_cards, channel_cards, for_you_cards], limit=24)

        return self._watch_groups_payload(
            all_cards=all_cards,
            related_cards=[card for card in related_cards if card["category"] == current_category][:20],
            channel_cards=channel_cards,
            for_you_cards=for_you_cards,
            channel_title=str(current_card["channel_name"]),
        )

    def _same_channel_watch_cards(self, current_video_id: int, seen_videos: set[int], limit: int) -> list[dict[str, object]]:
        current_video = self.video_feature_map.get(current_video_id)
        if current_video is None:
            return []

        channel_id = self._clean_int(current_video.get("channel_id"), 0)
        candidate_ids: list[int] = []
        for candidate_video_id in self.retrieval_bundle["channel_top_videos"].get(channel_id, [])[:80]:
            candidate_id = int(candidate_video_id)
            if candidate_id == current_video_id or candidate_id in seen_videos:
                continue
            candidate_ids.append(candidate_id)
            if len(candidate_ids) >= limit:
                break
        return self._cards_from_ids(candidate_ids, limit=limit)

    @staticmethod
    def _merge_watch_cards(card_groups: list[list[dict[str, object]]], limit: int) -> list[dict[str, object]]:
        merged_cards: list[dict[str, object]] = []
        used_ids: set[int] = set()
        group_positions = [0 for _ in card_groups]
        while len(merged_cards) < limit:
            added_any = False
            for group_index, group in enumerate(card_groups):
                while group_positions[group_index] < len(group):
                    card = group[group_positions[group_index]]
                    group_positions[group_index] += 1
                    if int(card["video_id"]) in used_ids:
                        continue
                    merged_cards.append(card)
                    used_ids.add(int(card["video_id"]))
                    added_any = True
                    break
                if len(merged_cards) >= limit:
                    break
            if not added_any:
                break
        return merged_cards

    @staticmethod
    def _watch_groups_payload(
        all_cards: list[dict[str, object]],
        related_cards: list[dict[str, object]],
        channel_cards: list[dict[str, object]],
        for_you_cards: list[dict[str, object]],
        channel_title: str,
    ) -> list[dict[str, object]]:
        groups = [
            {"key": "all", "title": "Todos", "items": all_cards},
            {"key": "related", "title": "Relacionados", "items": related_cards},
            {"key": "channel", "title": channel_title[:18], "items": channel_cards},
            {"key": "for_you", "title": "Para ti", "items": for_you_cards},
        ]
        return [group for group in groups if group["items"]]

    def _guest_seen_videos(self, guest_id: str) -> set[int]:
        state = self.guest_states.get(guest_id)
        return set() if state is None else set(state.recent_video_ids) | set(state.engaged_video_ids)

    def _combined_seen_videos(self, user_id: int) -> set[int]:
        seen_videos = set(self.seen_by_user.get(user_id, set()))
        state = self.session_states.get(user_id)
        if state is not None:
            seen_videos.update(state.recent_video_ids)
            seen_videos.update(state.engaged_video_ids)
        return seen_videos

    def _session_weight(self, user_id: int) -> float:
        state = self.session_states.get(user_id)
        if state is None or (not state.recent_video_ids and not state.engaged_video_ids):
            return 0.0
        view_weight = min(0.30, 0.035 * len(state.recent_video_ids))
        like_weight = min(0.03, 0.008 * len(state.engaged_video_ids))
        return min(0.34, view_weight + like_weight)

    def _iter_session_video_weights(self, state: SessionState) -> list[tuple[float, int]]:
        video_ids = list(state.recent_video_ids)
        total_videos = len(video_ids)
        weighted_videos: list[tuple[float, int]] = []
        for index, video_id in enumerate(video_ids):
            recency_rank = index + 1
            decay_weight = 0.25 + 0.75 * (recency_rank / total_videos)
            weighted_videos.append((decay_weight, video_id))
        return weighted_videos

    def _iter_session_signal_weights(self, state: SessionState) -> list[tuple[float, int]]:
        weighted_videos = self._iter_session_video_weights(state)
        engaged_video_ids = list(state.engaged_video_ids)
        total_engagements = len(engaged_video_ids)
        for index, video_id in enumerate(engaged_video_ids):
            recency_rank = index + 1
            engagement_weight = 0.04 + 0.06 * (recency_rank / max(total_engagements, 1))
            weighted_videos.append((engagement_weight, video_id))
        return weighted_videos

    def _add_subscription_channel_scores(
        self,
        state: SessionState,
        candidate_scores: dict[int, float],
        seen_videos: set[int],
        exclude_ids: set[int],
        bundle: dict[str, object],
        boost: float,
    ) -> None:
        if not state.subscribed_channel_ids:
            return

        allowed_categories = set(self._session_preference_maps_from_state(state)["category_scores"])
        if not allowed_categories:
            return

        for channel_id in state.subscribed_channel_ids:
            for rank_index, candidate_video_id in enumerate(bundle["channel_top_videos"].get(channel_id, [])[:35], start=1):
                candidate_id = int(candidate_video_id)
                if candidate_id in seen_videos or candidate_id in exclude_ids:
                    continue
                candidate_row = self.video_feature_map.get(candidate_id)
                if candidate_row is None:
                    continue
                candidate_category = self._clean_text(candidate_row.get("video_category"), "")
                if candidate_category not in allowed_categories:
                    continue
                candidate_scores[candidate_id] = candidate_scores.get(candidate_id, 0.0) + boost / math.sqrt(rank_index)

    @staticmethod
    def _normalize_scores(scores: dict[int, float]) -> dict[int, float]:
        if not scores:
            return {}
        max_score = max(scores.values())
        if max_score <= 0:
            return {video_id: 0.0 for video_id in scores}
        return {video_id: float(score) / float(max_score) for video_id, score in scores.items()}

    def _blend_score_maps(
        self,
        historical_scores: dict[int, float],
        session_scores: dict[int, float],
        session_weight: float,
    ) -> dict[int, float]:
        if not session_scores or session_weight <= 0:
            return historical_scores

        historical_norm = self._normalize_scores(historical_scores)
        session_norm = self._normalize_scores(session_scores)
        blended_scores: dict[int, float] = {}
        for video_id in set(historical_norm) | set(session_norm):
            blended_scores[video_id] = (
                (1.0 - session_weight) * historical_norm.get(video_id, 0.0)
                + session_weight * session_norm.get(video_id, 0.0)
            )
        return blended_scores

    def _rank_candidates(
        self,
        user_id: int,
        candidate_ids: list[int],
        surface_value: str,
        exclude_video_id: int | None = None,
    ) -> list[dict[str, object]]:
        if not candidate_ids:
            return []

        rows = [self._build_feature_row(user_id, video_id, surface_value) for video_id in candidate_ids if video_id != exclude_video_id]
        if not rows:
            return []

        features_df = pd.DataFrame(rows)[self.feature_columns].copy()
        for column in CATEGORICAL_COLUMNS:
            if column in features_df.columns:
                allowed_values = CATEGORICAL_SAFE_VALUES.get(column, set())
                fallback = CATEGORICAL_FALLBACKS.get(column, "unknown")
                features_df[column] = features_df[column].fillna(fallback).astype(str)
                if allowed_values:
                    features_df[column] = features_df[column].where(features_df[column].isin(allowed_values), fallback)
                features_df[column] = features_df[column].astype("category")
        for column in features_df.columns:
            if column in CATEGORICAL_COLUMNS:
                continue
            if features_df[column].dtype == "object":
                features_df[column] = features_df[column].fillna("unknown").astype("category")
            else:
                features_df[column] = pd.to_numeric(features_df[column], errors="coerce").fillna(0)

        scores = self.ranker.predict(features_df)
        score_series = pd.Series(scores)
        if len(score_series) > 1:
            normalized_scores = ((score_series - score_series.min()) / (score_series.max() - score_series.min() + 1e-9)).tolist()
        else:
            normalized_scores = [1.0]

        ranked_cards: list[dict[str, object]] = []
        session_blend_weight = self._session_weight(user_id) * 0.85
        for row, score, normalized_score in zip(rows, scores, normalized_scores):
            card = self._video_card(int(row["video_id"]))
            if card is None:
                continue
            session_bonus = self._session_rerank_bonus(user_id, int(row["video_id"]))
            final_score = (1.0 - session_blend_weight) * float(normalized_score) + session_blend_weight * session_bonus
            card["ranker_score"] = round(float(score), 6)
            card["final_score"] = round(float(final_score), 6)
            card["session_bonus"] = round(float(session_bonus), 6)
            card["user_follows_channel"] = int(row["user_follows_channel"])
            card["recent_category_match"] = int(row["recent_category_match"])
            card["channel_current_interest_score"] = round(float(row["channel_current_interest_score"]), 4)
            ranked_cards.append(card)

        ranked_cards.sort(key=lambda item: (item["final_score"], item["ranker_score"]), reverse=True)
        return ranked_cards

    def _rank_watch_next_candidates(
        self,
        user_id: int,
        candidate_ids: list[int],
        current_video_id: int,
        limit: int = 20,
    ) -> list[dict[str, object]]:
        current_video = self.video_feature_map.get(current_video_id)
        if current_video is None:
            return []

        current_category = self._clean_text(current_video.get("video_category"), "")
        current_channel_id = self._clean_int(current_video.get("channel_id"), 0)
        ranked_cards = self._rank_candidates(user_id, candidate_ids, surface_value="home", exclude_video_id=current_video_id)
        candidate_order = {int(video_id): index for index, video_id in enumerate(candidate_ids)}

        for card in ranked_cards:
            order_index = candidate_order.get(int(card["video_id"]), len(candidate_order))
            retrieval_context_score = 1.0 / math.sqrt(order_index + 1)
            same_category_bonus = 1.0 if card["category"] == current_category else 0.0
            same_channel_bonus = 0.35 if int(card["channel_id"]) == current_channel_id else 0.0
            card["watch_context_score"] = round(
                0.58 * same_category_bonus
                + same_channel_bonus
                + 0.27 * retrieval_context_score
                + 0.15 * float(card.get("final_score", 0.0)),
                6,
            )

        ranked_cards.sort(
            key=lambda item: (
                item["watch_context_score"],
                item["category"] == current_category,
                item["final_score"],
            ),
            reverse=True,
        )
        return ranked_cards[:limit]

    def _session_rerank_bonus(self, user_id: int, video_id: int) -> float:
        state = self.session_states.get(user_id)
        if state is None or (not state.recent_video_ids and not state.engaged_video_ids):
            return 0.0

        session_signals = self._session_preference_maps(user_id)
        video_row = self.video_feature_map.get(video_id)
        if video_row is None:
            return 0.0

        video_category = str(video_row.get("video_category", "General"))
        channel_id = int(video_row.get("channel_id", 0))
        category_score = session_signals["category_scores"].get(video_category, 0.0)
        channel_score = session_signals["channel_scores"].get(channel_id, 0.0)
        freshness_score = 0.12 if int(video_row.get("is_recent_upload", 0)) == 1 else 0.0
        return min(1.0, 0.65 * category_score + 0.25 * channel_score + freshness_score)

    def _session_preference_maps(self, user_id: int) -> dict[str, dict[object, float]]:
        state = self.session_states.get(user_id)
        if state is None:
            return {"category_scores": {}, "channel_scores": {}}
        return self._session_preference_maps_from_state(state)

    def _session_preference_maps_from_state(self, state: SessionState) -> dict[str, dict[object, float]]:
        if not state.recent_video_ids and not state.engaged_video_ids:
            return {"category_scores": {}, "channel_scores": {}}
        category_scores: dict[str, float] = {}
        channel_scores: dict[int, float] = {}
        max_weight = 0.0
        for decay_weight, video_id in self._iter_session_signal_weights(state):
            video_row = self.video_feature_map.get(video_id)
            if video_row is None:
                continue
            category = str(video_row.get("video_category", "General"))
            channel_id = int(video_row.get("channel_id", 0))
            category_scores[category] = category_scores.get(category, 0.0) + decay_weight
            channel_scores[channel_id] = channel_scores.get(channel_id, 0.0) + decay_weight
            max_weight = max(max_weight, category_scores[category], channel_scores[channel_id])

        if max_weight <= 0:
            return {"category_scores": {}, "channel_scores": {}}

        return {
            "category_scores": {category: score / max_weight for category, score in category_scores.items()},
            "channel_scores": {channel_id: score / max_weight for channel_id, score in channel_scores.items()},
        }

    def _build_feature_row(self, user_id: int, video_id: int, surface_value: str) -> dict[str, object]:
        user_row = self.user_feature_map[user_id]
        video_row = self.video_feature_map[video_id]
        channel_id = int(video_row["channel_id"])
        channel_row = self.user_channel_map.get((user_id, channel_id), {})
        follows_channel = 1 if (user_id, channel_id) in self.follow_pairs else int(channel_row.get("user_follows_channel", 0))

        feature_row: dict[str, object] = {
            "user_id": user_id,
            "video_id": video_id,
            "channel_id": channel_id,
            "creator_user_id": int(video_row["creator_user_id"]),
        }

        for column in self.feature_columns:
            if column == "surface":
                feature_row[column] = surface_value
                continue
            if column in user_row:
                feature_row[column] = user_row.get(column, 0)
                continue
            if column in video_row:
                feature_row[column] = video_row.get(column, 0)
                continue

        feature_row["channel_recent_interactions_30d"] = int(channel_row.get("channel_recent_interactions_30d", 0))
        feature_row["channel_recent_watch_percent_30d"] = float(channel_row.get("channel_recent_watch_percent_30d", 0))
        feature_row["channel_recent_implicit_score_30d"] = float(channel_row.get("channel_recent_implicit_score_30d", 0))
        feature_row["days_since_last_channel_watch"] = int(channel_row.get("days_since_last_channel_watch", 999))
        feature_row["user_follows_channel"] = follows_channel
        feature_row["stale_follow_flag"] = int(channel_row.get("stale_follow_flag", 1 if follows_channel else 0))
        feature_row["channel_current_interest_score"] = float(channel_row.get("channel_current_interest_score", 0))

        feature_row["category_matches_user_pref"] = int(str(video_row["video_category"]) == str(user_row.get("user_favorite_category", "unknown")))
        feature_row["recent_category_match"] = int(str(video_row["video_category"]) == str(user_row.get("user_recent_favorite_category", "unknown")))
        feature_row["creator_matches_user_pref"] = int(int(video_row["creator_user_id"]) == int(user_row.get("user_favorite_creator_id", 0)))
        feature_row["recent_creator_match"] = int(int(video_row["creator_user_id"]) == int(user_row.get("user_recent_favorite_creator_id", 0)))
        feature_row["channel_matches_user_pref"] = int(int(video_row["channel_id"]) == int(user_row.get("user_favorite_channel_id", 0)))
        feature_row["recent_channel_match"] = int(int(video_row["channel_id"]) == int(user_row.get("user_recent_favorite_channel_id", 0)))
        feature_row["item_vs_user_score_gap"] = round(
            float(video_row.get("video_engagement_score", 0)) - float(user_row.get("user_avg_implicit_score", 0)),
            4,
        )
        feature_row["freshness_vs_user_avg"] = round(
            float(video_row.get("video_freshness_score", 0)) - float(user_row.get("user_avg_video_freshness", 0)),
            4,
        )

        return feature_row

    def _session_categories(self, user_id: int) -> set[str]:
        state = self.session_states.get(user_id)
        if state is None:
            return set()

        categories: set[str] = set()
        for video_id in list(state.recent_video_ids) + list(state.engaged_video_ids):
            video_row = self.video_feature_map.get(int(video_id))
            if video_row is None:
                continue
            category = self._clean_text(video_row.get("video_category"), "")
            if category:
                categories.add(category)
        return categories

    def _connected_history_cards(
        self,
        user_id: int,
        exclude_ids: set[int],
        exclude_categories: set[str],
        limit: int,
    ) -> list[dict[str, object]]:
        history_video_ids = self.historical_history_by_user.get(user_id, [])
        if not history_video_ids and user_id not in self.followed_channels_by_user:
            return []

        category_scores: dict[str, float] = {}
        category_counts: Counter[str] = Counter()
        channel_scores: dict[int, float] = {}
        for rank_index, video_id in enumerate(history_video_ids, start=1):
            video_row = self.video_feature_map.get(int(video_id))
            if video_row is None:
                continue
            weight = 1.0 / math.sqrt(rank_index)
            category = self._clean_text(video_row.get("video_category"), "")
            channel_id = self._clean_int(video_row.get("channel_id"), 0)
            if category:
                category_counts[category] += 1
                category_scores[category] = category_scores.get(category, 0.0) + weight
            if channel_id:
                channel_scores[channel_id] = channel_scores.get(channel_id, 0.0) + weight

        for channel_id in self.followed_channels_by_user.get(user_id, []):
            channel_row = self.user_channel_map.get((user_id, int(channel_id)), {})
            current_interest = float(channel_row.get("channel_current_interest_score", 0.0))
            if current_interest <= 0:
                continue
            channel_scores[int(channel_id)] = channel_scores.get(int(channel_id), 0.0) + min(0.35, 0.20 * current_interest)

        seen_videos = self._combined_seen_videos(user_id)
        blocked_categories = {category for category in exclude_categories if category and category != "unknown"}
        valid_categories = {
            category
            for category, score in category_scores.items()
            if category_counts[category] >= 2 and score >= 0.8 and category not in blocked_categories
        }
        if not valid_categories:
            return []

        candidate_scores: dict[int, float] = {}
        bundle = self.retrieval_bundle

        for channel_id, signal_strength in channel_scores.items():
            for rank_index, candidate_video_id in enumerate(bundle["channel_top_videos"].get(channel_id, [])[:50], start=1):
                candidate_id = int(candidate_video_id)
                if candidate_id in seen_videos or candidate_id in exclude_ids:
                    continue
                candidate_row = self.video_feature_map.get(candidate_id)
                if candidate_row is None:
                    continue
                candidate_category = self._clean_text(candidate_row.get("video_category"), "")
                if candidate_category in blocked_categories:
                    continue
                if candidate_category not in valid_categories:
                    continue
                candidate_scores[candidate_id] = candidate_scores.get(candidate_id, 0.0) + (
                    0.70 * signal_strength / math.sqrt(rank_index)
                )

        for category in valid_categories:
            signal_strength = category_scores[category]
            for rank_index, candidate_video_id in enumerate(bundle["category_top_videos"].get(category, [])[:80], start=1):
                candidate_id = int(candidate_video_id)
                if candidate_id in seen_videos or candidate_id in exclude_ids:
                    continue
                candidate_scores[candidate_id] = candidate_scores.get(candidate_id, 0.0) + (
                    0.45 * signal_strength / math.sqrt(rank_index)
                )

        if not candidate_scores:
            return []

        candidate_ids = top_items(candidate_scores, 120)
        ranked_cards = self._rank_candidates(user_id, candidate_ids, surface_value="home")
        return ranked_cards[:limit]

    def _recent_upload_cards(self, exclude_ids: set[int], limit: int) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        for video_id in self.retrieval_bundle["global_recent"][:600]:
            candidate_id = int(video_id)
            if candidate_id in exclude_ids:
                continue
            card = self._video_card(candidate_id)
            if card is None or int(card["is_recent_upload"]) != 1:
                continue
            items.append(card)
            if len(items) == limit:
                break
        return items

    def _preferred_categories_for_user(self, user_id: int, ranked_videos: list[dict[str, object]]) -> list[str]:
        categories: list[str] = []
        user_row = self.user_feature_map.get(user_id, {})
        for category in list(self._session_categories(user_id)) + [
            self._clean_text(user_row.get("user_recent_favorite_category"), ""),
            self._clean_text(user_row.get("user_favorite_category"), ""),
        ]:
            if category and category not in categories:
                categories.append(category)

        for video in ranked_videos[:18]:
            category = str(video.get("category", ""))
            if category and category != "unknown" and category not in categories:
                categories.append(category)
            if len(categories) >= 5:
                break
        return categories

    def _preferred_categories_for_guest(self, guest_id: str, ranked_videos: list[dict[str, object]]) -> list[str]:
        categories: list[str] = []
        state = self.guest_states.get(guest_id)
        if state is not None:
            for video_id in list(state.recent_video_ids) + list(state.engaged_video_ids):
                video_row = self.video_feature_map.get(int(video_id))
                if video_row is None:
                    continue
                category = self._clean_text(video_row.get("video_category"), "")
                if category and category not in categories:
                    categories.append(category)

        for video in ranked_videos[:18]:
            category = str(video.get("category", ""))
            if category and category != "unknown" and category not in categories:
                categories.append(category)
            if len(categories) >= 5:
                break
        return categories

    def _personalized_recent_upload_cards(
        self,
        preferred_categories: list[str],
        ranked_videos: list[dict[str, object]],
        exclude_ids: set[int],
        seen_ids: set[int],
        limit: int = 12,
    ) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        item_ids: set[int] = set()
        category_counts: Counter[str] = Counter()
        allowed_categories = set(preferred_categories)

        def add_card(card: dict[str, object] | None) -> bool:
            if card is None:
                return False
            video_id = int(card["video_id"])
            category = str(card.get("category", "unknown"))
            if video_id in exclude_ids or video_id in seen_ids or video_id in item_ids:
                return False
            if int(card.get("is_recent_upload", 0)) != 1:
                return False
            if allowed_categories and category not in allowed_categories:
                return False
            if category_counts[category] >= 4:
                return False
            items.append(card)
            item_ids.add(video_id)
            category_counts[category] += 1
            return len(items) == limit

        for video in ranked_videos:
            if add_card(video):
                return items

        recent_by_category = self.retrieval_bundle.get("recent_category_top_videos", {})
        for category in preferred_categories:
            for video_id in recent_by_category.get(category, [])[:80]:
                if add_card(self._video_card(int(video_id))):
                    return items

        return items

    def _discovery_cards(
        self,
        ranked_videos: list[dict[str, object]],
        used_ids: set[int],
        avoid_categories: set[str],
        limit: int = 12,
    ) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        category_counts: Counter[str] = Counter()
        fallback_cards = self._cards_from_ids([int(video_id) for video_id in self.retrieval_bundle["global_recent"][:220]], limit=80)

        for video in list(ranked_videos) + fallback_cards:
            video_id = int(video["video_id"])
            category = str(video.get("category", "unknown"))
            if video_id in used_ids:
                continue
            if category in avoid_categories and len(items) < max(4, limit // 2):
                continue
            if category_counts[category] >= 3:
                continue
            items.append(video)
            used_ids.add(video_id)
            category_counts[category] += 1
            if len(items) == limit:
                break

        return items

    def _fill_home_shelf(
        self,
        title: str,
        description: str,
        items: list[dict[str, object]],
        fallback_pool: list[dict[str, object]],
        used_ids: set[int],
        limit: int = 12,
        max_per_category: int | None = None,
    ) -> dict[str, object]:
        final_items: list[dict[str, object]] = []
        category_counts: Counter[str] = Counter()

        for video in list(items) + list(fallback_pool):
            video_id = int(video["video_id"])
            if video_id in used_ids:
                continue
            if video_id in {int(item["video_id"]) for item in final_items}:
                continue
            category = str(video.get("category", "unknown"))
            if max_per_category is not None and category_counts[category] >= max_per_category:
                continue
            final_items.append(video)
            used_ids.add(video_id)
            category_counts[category] += 1
            if len(final_items) == limit:
                break

        if len(final_items) < limit:
            for video in fallback_pool:
                video_id = int(video["video_id"])
                if video_id in used_ids:
                    continue
                if video_id in {int(item["video_id"]) for item in final_items}:
                    continue
                final_items.append(video)
                used_ids.add(video_id)
                if len(final_items) == limit:
                    break

        return {"title": title, "description": description, "items": final_items}

    def _build_home_shelves(self, user_id: int, ranked_videos: list[dict[str, object]]) -> list[dict[str, object]]:
        used_ids: set[int] = set()
        seen_ids = self._combined_seen_videos(user_id)

        def pick(
            title: str,
            description: str,
            predicate,
            limit: int = 12,
            max_per_category: int | None = None,
        ) -> dict[str, object] | None:
            items: list[dict[str, object]] = []
            category_counts: Counter[str] = Counter()
            for video in ranked_videos:
                if video["video_id"] in used_ids:
                    continue
                if not predicate(video):
                    continue
                category = str(video.get("category", "unknown"))
                if max_per_category is not None and category_counts[category] >= max_per_category:
                    continue
                items.append(video)
                used_ids.add(video["video_id"])
                category_counts[category] += 1
                if len(items) == limit:
                    break
            if not items:
                return None
            return {"title": title, "description": description, "items": items}

        preferred_categories = self._preferred_categories_for_user(user_id, ranked_videos)
        new_ranked_items = self._personalized_recent_upload_cards(
            preferred_categories,
            ranked_videos,
            exclude_ids=used_ids,
            seen_ids=seen_ids,
            limit=12,
        )
        new_shelf = self._fill_home_shelf(
            "Videos nuevos",
            "Novedad personalizada con los temas que estas viendo.",
            new_ranked_items,
            [],
            used_ids,
        )

        for_you_shelf = pick(
            "Para ti",
            "Mezcla principal: historial reciente, sesion actual y ranker final.",
            lambda video: True,
            max_per_category=5,
        )
        if for_you_shelf is None:
            for_you_shelf = self._fill_home_shelf(
                "Para ti",
                "Mezcla principal: historial reciente, sesion actual y ranker final.",
                [],
                ranked_videos,
                used_ids,
                max_per_category=5,
            )

        dominant_current_categories = {
            str(video["category"])
            for video in (for_you_shelf or {}).get("items", [])[:8]
            if str(video.get("category", "unknown")) != "unknown"
        }
        dominant_current_categories.update(self._session_categories(user_id))
        connected_items = self._connected_history_cards(
            user_id,
            exclude_ids=used_ids,
            exclude_categories=dominant_current_categories,
            limit=12,
        )
        connected_shelf = self._fill_home_shelf(
            "Canales y temas que ya viste",
            "Recupera intereses reales del historial/follows. Si se agotan, rellena con candidatos personalizados.",
            connected_items,
            [video for video in ranked_videos if int(video["video_id"]) not in used_ids],
            used_ids,
            max_per_category=6,
        )
        discovery_categories = {
            str(video.get("category", "unknown"))
            for video in list(new_shelf.get("items", [])) + list(for_you_shelf.get("items", []))
        }
        discovery_items = self._discovery_cards(ranked_videos, used_ids, discovery_categories, limit=12)
        discovery_shelf = {
            "title": "Nuevos descubrimientos",
            "description": "Exploracion fuera del foco principal.",
            "items": discovery_items,
        }

        return [shelf for shelf in [new_shelf, for_you_shelf, connected_shelf, discovery_shelf] if shelf.get("items")]

    @staticmethod
    def _home_hero_videos(
        shelves: list[dict[str, object]],
        ranked_videos: list[dict[str, object]],
        limit: int = 5,
    ) -> list[dict[str, object]]:
        hero_items: list[dict[str, object]] = []
        seen_ids: set[int] = set()

        for shelf in shelves:
            if not shelf:
                continue
            if str(shelf.get("title", "")).lower() != "videos nuevos":
                continue
            for video in shelf.get("items", []):
                if RecommendationService._has_placeholder_title(video):
                    continue
                video_id = int(video["video_id"])
                if video_id in seen_ids:
                    continue
                seen_ids.add(video_id)
                hero_items.append(video)
                if len(hero_items) == limit:
                    return hero_items

        for video in ranked_videos:
            if RecommendationService._has_placeholder_title(video):
                continue
            video_id = int(video["video_id"])
            if video_id in seen_ids:
                continue
            seen_ids.add(video_id)
            hero_items.append(video)
            if len(hero_items) == limit:
                break

        return hero_items

    @staticmethod
    def _has_placeholder_title(video: dict[str, object]) -> bool:
        video_id = int(video.get("video_id", 0))
        title = str(video.get("title", "")).strip().lower()
        return title == f"video {video_id}" or not title

    def _guest_session_related_cards(
        self,
        guest_id: str,
        ranked_videos: list[dict[str, object]],
        used_ids: set[int],
        limit: int,
    ) -> list[dict[str, object]]:
        state = self.guest_states.get(guest_id)
        if state is None:
            return []
        session_categories = {
            self._clean_text(self.video_feature_map.get(int(video_id), {}).get("video_category"), "")
            for video_id in list(state.recent_video_ids) + list(state.engaged_video_ids)
        }
        session_categories.discard("")
        items = [
            video
            for video in ranked_videos
            if int(video["video_id"]) not in used_ids and str(video["category"]) in session_categories
        ]
        return items[:limit]

    def _build_guest_home_shelves(
        self,
        ranked_videos: list[dict[str, object]],
        guest_snapshot: dict[str, object],
        guest_id: str = "",
    ) -> list[dict[str, object]]:
        used_ids: set[int] = set()
        seen_ids = self._guest_seen_videos(guest_id)

        def pick(
            title: str,
            description: str,
            predicate,
            limit: int = 12,
            max_per_category: int | None = None,
        ) -> dict[str, object] | None:
            items: list[dict[str, object]] = []
            category_counts: Counter[str] = Counter()
            for video in ranked_videos:
                if video["video_id"] in used_ids or not predicate(video):
                    continue
                category = str(video.get("category", "unknown"))
                if max_per_category is not None and category_counts[category] >= max_per_category:
                    continue
                items.append(video)
                used_ids.add(video["video_id"])
                category_counts[category] += 1
                if len(items) == limit:
                    break
            return None if not items else {"title": title, "description": description, "items": items}

        focus_label = str(guest_snapshot["session_focus"]).replace("Invitado centrado ahora en: ", "")
        preferred_categories = self._preferred_categories_for_guest(guest_id, ranked_videos)
        new_ranked_items = self._personalized_recent_upload_cards(
            preferred_categories,
            ranked_videos,
            exclude_ids=used_ids,
            seen_ids=seen_ids,
            limit=12,
        )
        new_shelf = self._fill_home_shelf(
            "Videos nuevos",
            "Novedad personalizada con los temas que estas viendo.",
            new_ranked_items,
            [],
            used_ids,
        )

        for_you_shelf = pick(
            "Para ti",
            "La home se recalcula con lo que has abierto en esta sesion sin bloquearse en un solo tema.",
            lambda video: True,
            max_per_category=5,
        )
        if for_you_shelf is None:
            for_you_shelf = self._fill_home_shelf(
                "Para ti",
                "La home se recalcula con lo que has abierto en esta sesion sin bloquearse en un solo tema.",
                [],
                ranked_videos,
                used_ids,
                max_per_category=5,
            )

        session_items = self._guest_session_related_cards(guest_id, ranked_videos, used_ids, limit=12)
        session_shelf = self._fill_home_shelf(
            f"Temas abiertos en esta sesion: {focus_label}",
            "Mantiene el hilo de lo que estas viendo ahora. Si se agota, rellena sin ocultar la lista.",
            session_items,
            [video for video in ranked_videos if int(video["video_id"]) not in used_ids],
            used_ids,
            max_per_category=8,
        )
        discovery_categories = {
            str(video.get("category", "unknown"))
            for video in list(new_shelf.get("items", [])) + list(for_you_shelf.get("items", []))
        }
        discovery_items = self._discovery_cards(ranked_videos, used_ids, discovery_categories, limit=12)
        discovery_shelf = {
            "title": "Nuevos descubrimientos",
            "description": "Exploracion fuera del foco principal.",
            "items": discovery_items,
        }
        return [shelf for shelf in [new_shelf, for_you_shelf, session_shelf, discovery_shelf] if shelf.get("items")]

    def _cards_from_scores(self, scores: dict[int, float], limit: int) -> list[dict[str, object]]:
        cards: list[dict[str, object]] = []
        for video_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]:
            card = self._video_card(int(video_id))
            if card is None:
                continue
            card["ranker_score"] = round(float(score), 6)
            card["final_score"] = round(float(score), 6)
            card["session_bonus"] = round(min(1.0, float(score)), 6)
            cards.append(card)
        return cards

    def _cards_from_ids(self, candidate_ids: list[int], limit: int) -> list[dict[str, object]]:
        cards: list[dict[str, object]] = []
        for video_id in candidate_ids[:limit]:
            card = self._video_card(int(video_id))
            if card is not None:
                cards.append(card)
        return cards

    def _video_card(self, video_id: int) -> dict[str, object] | None:
        catalog_row = self.video_catalog_map.get(video_id)
        feature_row = self.video_feature_map.get(video_id)
        if catalog_row is None or feature_row is None:
            return None

        channel_id = int(feature_row["channel_id"])
        channel = self.channel_map.get(channel_id, {})
        title = self._clean_text(catalog_row.get("titulo"), f"Video {video_id}")
        thumbnail_dev_url = self._clean_text(catalog_row.get("thumbnail_dev_url"), "")
        if not thumbnail_dev_url:
            thumbnail_dev_url = self._clean_text(catalog_row.get("thumbnail_url"), "")
        thumbnail_real_url = self._clean_text(catalog_row.get("thumbnail_real_url"), "")
        thumbnail_url = thumbnail_real_url or thumbnail_dev_url
        total_likes = self._clean_int(catalog_row.get("total_likes"), 0)
        total_comments = self._clean_int(catalog_row.get("total_comments"), 0)
        followers_total = self._clean_int(channel.get("followers_total"), 0)
        return {
            "video_id": int(video_id),
            "channel_id": channel_id,
            "title": title,
            "category": self._clean_text(feature_row.get("video_category"), "General"),
            "channel_name": self._clean_text(channel.get("channel_name"), f"channel_{channel_id}"),
            "channel_followers_text": self._format_count(followers_total, "suscriptores"),
            "thumbnail_url": thumbnail_url,
            "thumbnail_real_url": thumbnail_real_url or thumbnail_url,
            "thumbnail_dev_url": thumbnail_dev_url or thumbnail_url,
            "duration_text": self._format_duration(self._clean_int(feature_row.get("video_duration_s_clean"), 0)),
            "views_text": self._format_views(self._clean_int(catalog_row.get("total_views"), 0)),
            "likes_text": self._format_count(total_likes, "likes"),
            "comments_text": self._format_count(total_comments, "comentarios"),
            "total_likes": total_likes,
            "total_comments": total_comments,
            "freshness_text": self._format_age(self._clean_int(feature_row.get("video_age_days"), 0)),
            "thumbnail_seed": int(video_id),
            "is_recent_upload": self._clean_int(feature_row.get("is_recent_upload"), 0),
            "channel_current_interest_score": 0.0,
            "user_follows_channel": 0,
            "recent_category_match": 0,
            "session_bonus": 0.0,
        }

    def _apply_user_video_state(self, video: dict[str, object], user_id: int) -> dict[str, object]:
        state = self.session_states.get(user_id)
        video_id = int(video["video_id"])
        channel_id = int(video["channel_id"])
        is_liked = state is not None and video_id in state.liked_video_ids
        is_subscribed = (user_id, channel_id) in self.follow_pairs or (
            state is not None and channel_id in state.subscribed_channel_ids
        )
        video["user_liked_video"] = is_liked
        video["user_subscribed_channel"] = is_subscribed
        return video

    def _apply_guest_video_state(self, video: dict[str, object], guest_id: str) -> dict[str, object]:
        state = self.guest_states.get(guest_id)
        video_id = int(video["video_id"])
        channel_id = int(video["channel_id"])
        video["user_liked_video"] = state is not None and video_id in state.liked_video_ids
        video["user_subscribed_channel"] = state is not None and channel_id in state.subscribed_channel_ids
        return video

    @staticmethod
    def _clean_text(value: object, fallback: str) -> str:
        if value is None or pd.isna(value):
            return fallback
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return fallback
        return text

    @staticmethod
    def _clean_int(value: object, fallback: int) -> int:
        if value is None or pd.isna(value):
            return fallback
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _format_duration(duration_seconds: int) -> str:
        duration_seconds = max(duration_seconds, 0)
        minutes, seconds = divmod(duration_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    @staticmethod
    def _format_views(views: int) -> str:
        if views >= 1_000_000:
            return f"{views / 1_000_000:.1f} M visualizaciones"
        if views >= 1_000:
            return f"{views / 1_000:.1f} mil visualizaciones"
        return f"{views} visualizaciones"

    @staticmethod
    def _format_count(value: int, label: str) -> str:
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f} M {label}"
        if value >= 1_000:
            return f"{value / 1_000:.1f} mil {label}"
        return f"{value} {label}"

    @staticmethod
    def _format_age(days: int) -> str:
        if days <= 0:
            return "Hoy"
        if days < 7:
            return f"Hace {days} dias"
        if days < 30:
            return f"Hace {days // 7} semanas"
        if days < 365:
            return f"Hace {days // 30} meses"
        return f"Hace {days // 365} anos"


@lru_cache(maxsize=1)
def get_recommendation_service() -> RecommendationService:
    return RecommendationService()
