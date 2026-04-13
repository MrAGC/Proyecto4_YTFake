from __future__ import annotations

import csv
import json
import math
import pickle
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd
import xgboost as xgb

from train_retrieval_cf import top_items
from train_retrieval_multisource import build_multisource_scores


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RECO_DIR = BASE_DIR / "reco_output_v2"
MODELS_DIR = BASE_DIR / "models"

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
RANKER_MODEL_PATH = MODELS_DIR / "ranker_compare_v1" / "pairwise_xgboost" / "ranker_model.json"
RETRIEVAL_HOME_MODEL_PATH = MODELS_DIR / "retrieval_multisource_v1" / "retrieval_model.pkl"

CATEGORICAL_COLUMNS = [
    "surface",
    "user_favorite_category",
    "user_favorite_device",
    "user_favorite_time_slot",
    "user_recent_favorite_category",
    "video_category",
]

DEFAULT_VIDEO_LIMIT = 240
SESSION_HISTORY_LIMIT = 12
HISTORICAL_HISTORY_LIMIT = 10


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
    views_in_session: int = 0


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
        self.seen_by_user = (
            self.interactions.groupby("user_id", sort=False)["video_id"]
            .agg(lambda values: {int(value) for value in values})
            .to_dict()
        )
        self.historical_history_by_user = self._build_historical_history()
        self.creator_video_ids = self._build_creator_video_index()
        self.channels_by_creator = self._build_channels_by_creator()
        self.session_states: dict[int, SessionState] = {}
        self.guest_states: dict[str, SessionState] = {}
        self._ensure_local_session_store()
        self._load_local_session_events()

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

    def _ensure_local_session_store(self) -> None:
        if LOCAL_SESSION_EVENTS_PATH.exists():
            return
        with open(LOCAL_SESSION_EVENTS_PATH, "w", newline="", encoding="utf-8") as file_handle:
            writer = csv.writer(file_handle)
            writer.writerow(["actor_type", "actor_id", "video_id", "event_type", "event_ts"])

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

    def register_view(self, user_id: int, video_id: int) -> None:
        if user_id not in self.user_feature_map or video_id not in self.video_feature_map:
            return

        self._register_state_view(self.session_states, user_id, video_id)
        self._append_local_event("user", str(user_id), video_id)

    def register_guest_view(self, guest_id: str, video_id: int) -> None:
        if video_id not in self.video_feature_map:
            return

        self._register_state_view(self.guest_states, guest_id, video_id)
        self._append_local_event("guest", guest_id, video_id)

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

        return {
            "user_id": user_id,
            "favorite_category": str(user_row.get("user_favorite_category", "General")),
            "recent_category": str(user_row.get("user_recent_favorite_category", "General")),
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
        candidate_ids = self._home_candidate_ids(user_id, limit=DEFAULT_VIDEO_LIMIT)
        ranked_videos = self._rank_candidates(user_id, candidate_ids, surface_value="home")
        shelves = self._build_home_shelves(user_id, ranked_videos)
        hero_video = ranked_videos[0] if ranked_videos else None
        return {
            "user": user_snapshot,
            "sidebar": self._build_sidebar(user_id),
            "hero_video": hero_video,
            "shelves": shelves,
            "model_stack": "multi_source_home + pairwise_xgboost + session_blend",
            "active_page": "home",
        }

    def watch_page(self, user_id: int, video_id: int) -> dict[str, object]:
        user_snapshot = self.get_user_snapshot(user_id)
        video = self._video_card(video_id)
        if video is None:
            raise KeyError(f"No existe video_id={video_id}")

        next_candidate_ids = self._watch_next_candidate_ids(user_id, video_id, limit=120)
        next_up = self._rank_candidates(user_id, next_candidate_ids, surface_value="home", exclude_video_id=video_id)
        return {
            "user": user_snapshot,
            "sidebar": self._build_sidebar(user_id),
            "video": video,
            "next_up": next_up[:20],
            "model_stack": "item_based_watch_next + pairwise_xgboost + session_blend",
            "active_page": "watch",
        }

    def search_page(self, user_id: int, query: str) -> dict[str, object]:
        return {
            "user": self.get_user_snapshot(user_id),
            "sidebar": self._build_sidebar(user_id),
            "query": query,
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

    def guest_home_page(self, guest_id: str) -> dict[str, object]:
        guest_snapshot = self.get_guest_snapshot(guest_id)
        candidate_scores = self._guest_home_scores(guest_id)
        ranked_videos = self._cards_from_scores(candidate_scores, limit=DEFAULT_VIDEO_LIMIT)
        shelves = self._build_guest_home_shelves(ranked_videos, guest_snapshot)
        return {
            "user": guest_snapshot,
            "sidebar": self._build_guest_sidebar(guest_id),
            "hero_video": ranked_videos[0] if ranked_videos else None,
            "shelves": shelves,
            "model_stack": "guest_session + global_recent",
            "active_page": "home",
        }

    def guest_watch_page(self, guest_id: str, video_id: int) -> dict[str, object]:
        video = self._video_card(video_id)
        if video is None:
            raise KeyError(f"No existe video_id={video_id}")

        next_candidate_ids = self._watch_next_candidate_ids_for_seen(self._guest_seen_videos(guest_id), video_id, limit=120)
        next_up = self._cards_from_ids(next_candidate_ids, limit=20)
        return {
            "user": self.get_guest_snapshot(guest_id),
            "sidebar": self._build_guest_sidebar(guest_id),
            "video": video,
            "next_up": next_up,
            "model_stack": "guest_watch_next",
            "active_page": "watch",
        }

    def guest_search_page(self, guest_id: str, query: str) -> dict[str, object]:
        return {
            "user": self.get_guest_snapshot(guest_id),
            "sidebar": self._build_guest_sidebar(guest_id),
            "query": query,
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

    def _resolve_owned_channel_id(self, user_id: int, profile_row: dict[str, object]) -> int | None:
        raw_owned_channel_id = profile_row.get("owned_channel_id")
        if raw_owned_channel_id is not None and not pd.isna(raw_owned_channel_id):
            return int(raw_owned_channel_id)
        creator_channels = self.channels_by_creator.get(user_id, [])
        return creator_channels[0] if creator_channels else None

    def _build_session_focus(self, user_id: int) -> str:
        state = self.session_states.get(user_id)
        if state is None or not state.recent_video_ids:
            favorite_category = str(self.user_feature_map[user_id].get("user_favorite_category", "General"))
            return f"Base historica: {favorite_category}"

        category_counter: Counter[str] = Counter()
        for decay_weight, video_id in self._iter_session_video_weights(state):
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
            "session_history": [card for card in session_history if card is not None],
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
            "session_history": [card for card in session_history if card is not None],
            "historical_history": [],
            "session_views": 0 if state is None else int(state.views_in_session),
            "session_focus": self._build_guest_session_focus(guest_id),
        }

    def _build_guest_session_focus(self, guest_id: str) -> str:
        state = self.guest_states.get(guest_id)
        if state is None or not state.recent_video_ids:
            return "Invitado sin historial: mezcla de tendencias y novedad"

        category_counter: Counter[str] = Counter()
        for decay_weight, video_id in self._iter_session_video_weights(state):
            video_row = self.video_feature_map.get(video_id)
            if video_row is None:
                continue
            category_counter[str(video_row.get("video_category", "General"))] += decay_weight
        if not category_counter:
            return "Invitado sin foco claro"
        return f"Invitado centrado ahora en: {category_counter.most_common(1)[0][0]}"

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
        if state is None or not state.recent_video_ids:
            return {}

        bundle = self.retrieval_bundle
        cf_model = bundle["cf_model"]
        seen_videos = self._combined_seen_videos(user_id)
        candidate_scores: dict[int, float] = {}

        for decay_weight, video_id in self._iter_session_video_weights(state):
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

        return candidate_scores

    def _guest_home_scores(self, guest_id: str) -> dict[int, float]:
        state = self.guest_states.get(guest_id)
        if state is None or not state.recent_video_ids:
            return {int(video_id): 1.0 / math.sqrt(index) for index, video_id in enumerate(self.retrieval_bundle["global_recent"][:200], start=1)}

        bundle = self.retrieval_bundle
        cf_model = bundle["cf_model"]
        seen_videos = self._guest_seen_videos(guest_id)
        candidate_scores: dict[int, float] = {int(video_id): 0.08 / math.sqrt(index) for index, video_id in enumerate(bundle["global_recent"][:80], start=1)}
        for decay_weight, video_id in self._iter_session_video_weights(state):
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

    def _guest_seen_videos(self, guest_id: str) -> set[int]:
        state = self.guest_states.get(guest_id)
        return set() if state is None else set(state.recent_video_ids)

    def _combined_seen_videos(self, user_id: int) -> set[int]:
        seen_videos = set(self.seen_by_user.get(user_id, set()))
        state = self.session_states.get(user_id)
        if state is not None:
            seen_videos.update(state.recent_video_ids)
        return seen_videos

    def _session_weight(self, user_id: int) -> float:
        state = self.session_states.get(user_id)
        if state is None or not state.recent_video_ids:
            return 0.0
        return min(0.35, 0.09 * len(state.recent_video_ids))

    def _iter_session_video_weights(self, state: SessionState) -> list[tuple[float, int]]:
        video_ids = list(state.recent_video_ids)
        total_videos = len(video_ids)
        weighted_videos: list[tuple[float, int]] = []
        for index, video_id in enumerate(video_ids):
            recency_rank = total_videos - index
            decay_weight = 0.55 + 0.45 * (recency_rank / total_videos)
            weighted_videos.append((decay_weight, video_id))
        return weighted_videos

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
                features_df[column] = features_df[column].fillna("unknown").astype("category")
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
        session_blend_weight = self._session_weight(user_id) * 0.5
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

    def _session_rerank_bonus(self, user_id: int, video_id: int) -> float:
        state = self.session_states.get(user_id)
        if state is None or not state.recent_video_ids:
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
        if state is None or not state.recent_video_ids:
            return {"category_scores": {}, "channel_scores": {}}

        category_scores: dict[str, float] = {}
        channel_scores: dict[int, float] = {}
        max_weight = 0.0
        for decay_weight, video_id in self._iter_session_video_weights(state):
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

    def _build_home_shelves(self, user_id: int, ranked_videos: list[dict[str, object]]) -> list[dict[str, object]]:
        user_snapshot = self.get_user_snapshot(user_id)
        recent_category = str(user_snapshot["recent_category"])
        favorite_category = str(user_snapshot["favorite_category"])
        session_focus_label = str(user_snapshot["session_focus"]).replace("Sesion activa: ", "")
        used_ids: set[int] = set()

        def pick(title: str, predicate, limit: int = 12, fallback_predicate=None) -> dict[str, object] | None:
            items: list[dict[str, object]] = []
            for video in ranked_videos:
                if video["video_id"] in used_ids:
                    continue
                if not predicate(video):
                    continue
                items.append(video)
                used_ids.add(video["video_id"])
                if len(items) == limit:
                    break
            if not items and fallback_predicate is not None:
                for video in ranked_videos:
                    if video["video_id"] in used_ids:
                        continue
                    if not fallback_predicate(video):
                        continue
                    items.append(video)
                    used_ids.add(video["video_id"])
                    if len(items) == limit:
                        break
            if not items:
                return None
            return {"title": title, "items": items}

        topical_category = recent_category if recent_category != "unknown" else favorite_category
        shelves = [
            pick("Para ti", lambda video: True),
            pick(
                "Sigue con tu base",
                lambda video: video["category"] == topical_category,
                fallback_predicate=lambda video: video["recent_category_match"] == 1,
            ),
            pick(
                f"Lo que abre tu sesion actual: {session_focus_label}",
                lambda video: video["session_bonus"] > 0.15,
                fallback_predicate=lambda video: video["ranker_score"] > 0,
            ),
            pick(
                "Canales con los que sigues conectado",
                lambda video: video["channel_current_interest_score"] > 0 or video["user_follows_channel"] == 1,
                fallback_predicate=lambda video: video["ranker_score"] > 0,
            ),
            pick("Nuevos que te pueden gustar", lambda video: video["is_recent_upload"] == 1),
            pick(
                f"Mas sobre {favorite_category}",
                lambda video: video["category"] == favorite_category,
                fallback_predicate=lambda video: video["ranker_score"] > 0,
            ),
        ]
        return [shelf for shelf in shelves if shelf is not None]

    def _build_guest_home_shelves(
        self,
        ranked_videos: list[dict[str, object]],
        guest_snapshot: dict[str, object],
    ) -> list[dict[str, object]]:
        used_ids: set[int] = set()

        def pick(title: str, predicate, limit: int = 12) -> dict[str, object] | None:
            items: list[dict[str, object]] = []
            for video in ranked_videos:
                if video["video_id"] in used_ids or not predicate(video):
                    continue
                items.append(video)
                used_ids.add(video["video_id"])
                if len(items) == limit:
                    break
            return None if not items else {"title": title, "items": items}

        focus_label = str(guest_snapshot["session_focus"]).replace("Invitado centrado ahora en: ", "")
        shelves = [
            pick("Para ti ahora", lambda video: True),
            pick(f"Lo que va con tu sesion: {focus_label}", lambda video: video["session_bonus"] > 0.15),
            pick("Videos nuevos", lambda video: video["is_recent_upload"] == 1),
            pick("Sigue explorando", lambda video: True),
        ]
        return [shelf for shelf in shelves if shelf is not None]

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
        title = str(catalog_row.get("titulo") or "").strip() or f"Video {video_id}"
        return {
            "video_id": int(video_id),
            "channel_id": channel_id,
            "title": title,
            "category": str(feature_row.get("video_category", "General")),
            "channel_name": str(channel.get("channel_name", f"channel_{channel_id}")),
            "thumbnail_url": str(catalog_row.get("thumbnail_url", "")).strip(),
            "duration_text": self._format_duration(int(feature_row.get("video_duration_s_clean", 0))),
            "views_text": self._format_views(int(catalog_row.get("total_views", 0))),
            "freshness_text": self._format_age(int(feature_row.get("video_age_days", 0))),
            "thumbnail_seed": int(video_id),
            "is_recent_upload": int(feature_row.get("is_recent_upload", 0)),
            "channel_current_interest_score": 0.0,
            "user_follows_channel": 0,
            "recent_category_match": 0,
            "session_bonus": 0.0,
        }

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
