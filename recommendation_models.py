from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator


def _clean_tag_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        raw_values = value.split(",")
    else:
        raw_values = value

    cleaned: list[str] = []
    seen: set[str] = set()

    for raw in raw_values:
        token = str(raw).strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        cleaned.append(token)

    return cleaned


def _clean_text_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        raw_values = [value]
    else:
        raw_values = value

    cleaned: list[str] = []
    for raw in raw_values:
        text = str(raw).strip()
        if text:
            cleaned.append(text)

    return cleaned


def _clean_video_id_list(value: Any) -> list[int]:
    if value is None:
        return []

    raw_values = [value] if isinstance(value, int) else value

    cleaned: list[int] = []
    seen: set[int] = set()

    for raw in raw_values:
        video_id = int(raw)
        if video_id <= 0 or video_id in seen:
            continue
        seen.add(video_id)
        cleaned.append(video_id)

    return cleaned


def _clean_positive_int_list(value: Any) -> list[int]:
    if value is None:
        return []

    if isinstance(value, int):
        raw_values = [value]
    else:
        raw_values = value

    cleaned: list[int] = []
    seen: set[int] = set()
    for raw in raw_values:
        item = int(raw)
        if item <= 0 or item in seen:
            continue
        seen.add(item)
        cleaned.append(item)

    return cleaned


class ModeloDominio(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class RolCuenta(str, Enum):
    VIEWER = "viewer"
    CREATOR = "creator"
    VIEWER_CREATOR = "viewer_creator"


class SuperficieRecomendacion(str, Enum):
    BUSQUEDA = "busqueda"
    HOME = "home"
    WATCH_NEXT = "watch_next"
    NOTIFICACION = "notificacion"
    SUSCRIPCIONES = "suscripciones"


class EstrategiaRetrieval(str, Enum):
    COSINE_SIMILARITY = "cosine_similarity"
    USER_BASED_CF = "user_based_cf"
    ITEM_BASED_CF = "item_based_cf"
    MULTI_SOURCE_HOME = "multi_source_home"
    SESION_USUARIO = "sesion_usuario"
    VIDEO_ACTUAL = "video_actual"
    CANALES_SEGUIDOS = "canales_seguidos"
    VIDEOS_RECIENTES = "videos_recientes"


class FuenteDescubrimiento(str, Enum):
    BUSQUEDA = "busqueda"
    RECOMENDADO_HOME = "recomendado_home"
    RECOMENDADO_VIDEO = "recomendado_video"
    NOTIFICACION = "notificacion"
    SUSCRIPCIONES = "suscripciones"
    DIRECTO = "directo"


class SenalesEngagement(ModeloDominio):
    tiempo_visto_s: float = Field(default=0, ge=0)
    porcentaje_visto: float = Field(default=0, ge=0, le=1)
    like: bool = False
    comentario: bool = False
    suscripcion: bool = False

    @computed_field(return_type=bool)
    @property
    def completion_flag(self) -> bool:
        return self.porcentaje_visto >= 0.9

    @computed_field(return_type=bool)
    @property
    def senal_positiva(self) -> bool:
        return (
            self.porcentaje_visto >= 0.7
            or self.like
            or self.comentario
            or self.suscripcion
        )

    @computed_field(return_type=bool)
    @property
    def senal_negativa_clara(self) -> bool:
        return (
            self.porcentaje_visto <= 0.2
            and not self.like
            and not self.comentario
            and not self.suscripcion
        )


class ConsultaBusqueda(ModeloDominio):
    texto: str = Field(min_length=1)
    timestamp: datetime
    retrieval_usado: EstrategiaRetrieval = EstrategiaRetrieval.COSINE_SIMILARITY
    videos_recuperados: list[int] = Field(default_factory=list)
    videos_rankeados: list[int] = Field(default_factory=list, max_length=15)
    video_seleccionado_id: int | None = Field(default=None, gt=0)

    @field_validator("videos_recuperados", "videos_rankeados", mode="before")
    @classmethod
    def _normalize_video_lists(cls, value: Any) -> list[int]:
        return _clean_video_id_list(value)

    @model_validator(mode="after")
    def _check_clicked_video(self) -> Self:
        if (
            self.video_seleccionado_id is not None
            and self.video_seleccionado_id not in self.videos_rankeados
        ):
            raise ValueError("El video seleccionado debe estar dentro del ranking mostrado.")
        return self

    @computed_field(return_type=bool)
    @property
    def tiene_click(self) -> bool:
        return self.video_seleccionado_id is not None

    @computed_field(return_type=list[int])
    @property
    def videos_no_elegidos(self) -> list[int]:
        if self.video_seleccionado_id is None:
            return list(self.videos_rankeados)
        return [
            video_id
            for video_id in self.videos_rankeados
            if video_id != self.video_seleccionado_id
        ]


class ImpresionRecomendacion(ModeloDominio):
    superficie: SuperficieRecomendacion
    timestamp: datetime
    retrieval_usado: EstrategiaRetrieval
    video_contexto_id: int | None = Field(default=None, gt=0)
    videos_recuperados: list[int] = Field(default_factory=list)
    videos_rankeados: list[int] = Field(default_factory=list, max_length=15)
    video_clicado_id: int | None = Field(default=None, gt=0)

    @field_validator("videos_recuperados", "videos_rankeados", mode="before")
    @classmethod
    def _normalize_video_lists(cls, value: Any) -> list[int]:
        return _clean_video_id_list(value)

    @model_validator(mode="after")
    def _check_coherence(self) -> Self:
        if (
            self.retrieval_usado == EstrategiaRetrieval.VIDEO_ACTUAL
            and self.video_contexto_id is None
        ):
            raise ValueError("Si el retrieval sale del video actual, hace falta un video_contexto_id.")

        if self.video_clicado_id is not None and self.video_clicado_id not in self.videos_rankeados:
            raise ValueError("El video clicado debe estar dentro de la impresion rankeada.")

        return self

    @computed_field(return_type=bool)
    @property
    def hubo_interes(self) -> bool:
        return self.video_clicado_id is not None

    @computed_field(return_type=list[int])
    @property
    def videos_no_elegidos(self) -> list[int]:
        if self.video_clicado_id is None:
            return list(self.videos_rankeados)
        return [video_id for video_id in self.videos_rankeados if video_id != self.video_clicado_id]


class InteraccionVideo(ModeloDominio):
    video_id: int = Field(gt=0)
    channel_id: int | None = Field(default=None, gt=0)
    creator_user_id: int | None = Field(default=None, gt=0)
    published_at: datetime | None = None
    timestamp: datetime
    fuente: FuenteDescubrimiento
    senales: SenalesEngagement = Field(default_factory=SenalesEngagement)

    @computed_field(return_type=bool)
    @property
    def vino_de_recomendacion(self) -> bool:
        return self.fuente in {
            FuenteDescubrimiento.RECOMENDADO_HOME,
            FuenteDescubrimiento.RECOMENDADO_VIDEO,
            FuenteDescubrimiento.NOTIFICACION,
        }

    @computed_field(return_type=float)
    @property
    def implicit_score(self) -> float:
        bonus_click_reco = 0.05 if self.vino_de_recomendacion else 0.0
        score = (
            0.55 * self.senales.porcentaje_visto
            + 0.15 * int(self.senales.like)
            + 0.10 * int(self.senales.comentario)
            + 0.15 * int(self.senales.suscripcion)
            + bonus_click_reco
        )
        return round(score, 4)

    @computed_field(return_type=bool)
    @property
    def positive_label(self) -> bool:
        return self.senales.senal_positiva or self.vino_de_recomendacion

    @computed_field(return_type=bool)
    @property
    def negative_label(self) -> bool:
        return self.senales.senal_negativa_clara and not self.vino_de_recomendacion

    @computed_field(return_type=int)
    @property
    def edad_video_dias(self) -> int:
        if self.published_at is None:
            return 999
        return max((self.timestamp - self.published_at).days, 0)


class ParPreferencia(ModeloDominio):
    user_id: int = Field(gt=0)
    superficie: SuperficieRecomendacion
    timestamp: datetime
    ganador_video_id: int = Field(gt=0)
    perdedor_video_id: int = Field(gt=0)
    video_contexto_id: int | None = Field(default=None, gt=0)


class SeguimientoCanal(ModeloDominio):
    user_id: int = Field(gt=0)
    channel_id: int = Field(gt=0)
    creator_user_id: int | None = Field(default=None, gt=0)
    followed_at: datetime
    source_video_id: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _avoid_self_follow(self) -> Self:
        if self.creator_user_id is not None and self.user_id == self.creator_user_id:
            raise ValueError("Un usuario no puede seguir su propio canal en este modelo.")
        return self


class CanalCreador(ModeloDominio):
    channel_id: int = Field(gt=0)
    owner_user_id: int = Field(gt=0)
    nombre_canal: str = Field(min_length=1)
    categoria_principal: str = Field(min_length=1)
    creado_en: datetime | None = None
    ultimo_video_publicado_en: datetime | None = None
    total_videos: int = Field(default=0, ge=0)
    total_views: int = Field(default=0, ge=0)
    total_likes: int = Field(default=0, ge=0)
    total_comments: int = Field(default=0, ge=0)
    followers_total: int = Field(default=0, ge=0)

    @computed_field(return_type=float)
    @property
    def likes_por_view(self) -> float:
        if self.total_views == 0:
            return 0.0
        return round(self.total_likes / self.total_views, 4)

    @computed_field(return_type=bool)
    @property
    def canal_establecido(self) -> bool:
        return self.total_videos >= 5 or self.followers_total >= 100


class UsuarioPerfil(ModeloDominio):
    user_id: int = Field(gt=0)
    rol_cuenta: RolCuenta = RolCuenta.VIEWER
    creado_en: datetime | None = None
    actualizado_en: datetime | None = None
    historial_busquedas: list[str] = Field(default_factory=list)
    videos_semilla_recientes: list[int] = Field(default_factory=list)
    categorias_preferidas: list[str] = Field(default_factory=list)
    parametros_preferidos: list[str] = Field(default_factory=list)
    canales_seguidos: list[int] = Field(default_factory=list)
    canal_propio_id: int | None = Field(default=None, gt=0)
    sesiones_totales: int = Field(default=0, ge=0)
    sesiones_solo_busqueda: int = Field(default=0, ge=0)
    impresiones_recomendadas: int = Field(default=0, ge=0)
    clics_en_recomendados: int = Field(default=0, ge=0)
    likes_dados: int = Field(default=0, ge=0)
    suscripciones_generadas: int = Field(default=0, ge=0)
    videos_publicados: int = Field(default=0, ge=0)
    tiempo_total_visto_s: float = Field(default=0, ge=0)
    forzar_notificaciones: bool | None = None

    @field_validator("historial_busquedas", mode="before")
    @classmethod
    def _normalize_search_history(cls, value: Any) -> list[str]:
        return _clean_text_list(value)

    @field_validator("videos_semilla_recientes", mode="before")
    @classmethod
    def _normalize_seed_videos(cls, value: Any) -> list[int]:
        return _clean_video_id_list(value)

    @field_validator("categorias_preferidas", "parametros_preferidos", mode="before")
    @classmethod
    def _normalize_tag_lists(cls, value: Any) -> list[str]:
        return _clean_tag_list(value)

    @field_validator("canales_seguidos", mode="before")
    @classmethod
    def _normalize_followed_channels(cls, value: Any) -> list[int]:
        return _clean_positive_int_list(value)

    @model_validator(mode="after")
    def _check_sessions(self) -> Self:
        if self.sesiones_solo_busqueda > self.sesiones_totales:
            raise ValueError("Las sesiones solo busqueda no pueden superar a las sesiones totales.")
        return self

    @computed_field(return_type=bool)
    @property
    def es_cold_start(self) -> bool:
        return (
            not self.historial_busquedas
            and not self.videos_semilla_recientes
            and not self.canales_seguidos
            and self.tiempo_total_visto_s == 0
            and self.clics_en_recomendados == 0
        )

    @computed_field(return_type=float)
    @property
    def ctr_recomendaciones(self) -> float:
        if self.impresiones_recomendadas == 0:
            return 0.0
        return round(self.clics_en_recomendados / self.impresiones_recomendadas, 4)

    @computed_field(return_type=bool)
    @property
    def usa_la_app_como_buscador(self) -> bool:
        return (
            self.sesiones_solo_busqueda >= 3
            and self.impresiones_recomendadas >= 10
            and self.clics_en_recomendados == 0
        )

    @computed_field(return_type=bool)
    @property
    def es_creador(self) -> bool:
        return self.rol_cuenta in {RolCuenta.CREATOR, RolCuenta.VIEWER_CREATOR}

    @computed_field(return_type=bool)
    @property
    def debe_recibir_notificaciones(self) -> bool:
        if self.forzar_notificaciones is not None:
            return self.forzar_notificaciones
        return not self.usa_la_app_como_buscador


class VideoCatalogo(ModeloDominio):
    video_id: int = Field(gt=0)
    channel_id: int = Field(gt=0)
    creator_user_id: int = Field(gt=0)
    titulo: str = ""
    category: str = Field(min_length=1)
    duracion_s: int = Field(ge=1)
    published_at: datetime | None = None
    parametros_contenido: list[str] = Field(default_factory=list)
    total_views: int = Field(default=0, ge=0)
    total_likes: int = Field(default=0, ge=0)
    like_rate: float = Field(default=0, ge=0)
    total_comments: int = Field(default=0, ge=0)
    comment_rate: float = Field(default=0, ge=0)
    suscriptores_ganados: int = Field(default=0, ge=0)
    subscription_rate: float = Field(default=0, ge=0)
    avg_watch_percent: float = Field(default=0, ge=0, le=1)
    avg_watch_time_s: float = Field(default=0, ge=0)
    veces_recomendado: int = Field(default=0, ge=0)
    total_clics: int = Field(default=0, ge=0)
    click_through_rate: float = Field(default=0, ge=0)

    @field_validator("parametros_contenido", mode="before")
    @classmethod
    def _normalize_parameters(cls, value: Any) -> list[str]:
        return _clean_tag_list(value)

    @computed_field(return_type=bool)
    @property
    def listo_para_busqueda(self) -> bool:
        return bool(self.titulo.strip()) and bool(self.parametros_contenido)

    @computed_field(return_type=bool)
    @property
    def requiere_enriquecimiento_metadata(self) -> bool:
        return not self.listo_para_busqueda

    @computed_field(return_type=str)
    @property
    def texto_indexable(self) -> str:
        parts = [self.titulo, self.category, " ".join(self.parametros_contenido)]
        return " | ".join(part for part in parts if part)

    @computed_field(return_type=float)
    @property
    def engagement_score_base(self) -> float:
        like_rate = min(max(self.like_rate, 0.0), 1.0)
        comment_rate = min(max(self.comment_rate, 0.0), 1.0)
        subscription_rate = min(max(self.subscription_rate, 0.0), 1.0)
        ctr = min(max(self.click_through_rate, 0.0), 1.0)

        score = (
            0.35 * self.avg_watch_percent
            + 0.25 * like_rate
            + 0.15 * comment_rate
            + 0.15 * subscription_rate
            + 0.10 * ctr
        )
        return round(score, 4)

    @computed_field(return_type=int)
    @property
    def antiguedad_dias(self) -> int:
        if self.published_at is None:
            return 999
        return max((datetime.utcnow() - self.published_at).days, 0)

    @computed_field(return_type=float)
    @property
    def freshness_score(self) -> float:
        return round(float(np.exp(-self.antiguedad_dias / 45)), 4)


class SesionRecomendacion(ModeloDominio):
    session_id: str = Field(min_length=1)
    user_id: int = Field(gt=0)
    inicio: datetime
    fin: datetime | None = None
    feed_inicial_vacio: bool = False
    ranker_compartido: str = "pairwise_ranker_contextual"
    retrieval_busqueda: EstrategiaRetrieval = EstrategiaRetrieval.COSINE_SIMILARITY
    retrieval_home: EstrategiaRetrieval = EstrategiaRetrieval.MULTI_SOURCE_HOME
    retrieval_watch_next: EstrategiaRetrieval = EstrategiaRetrieval.ITEM_BASED_CF
    retrieval_suscripciones: EstrategiaRetrieval = EstrategiaRetrieval.CANALES_SEGUIDOS
    consultas: list[ConsultaBusqueda] = Field(default_factory=list)
    recomendaciones: list[ImpresionRecomendacion] = Field(default_factory=list)
    interacciones_video: list[InteraccionVideo] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_dates(self) -> Self:
        if self.fin is not None and self.fin < self.inicio:
            raise ValueError("La fecha de fin no puede ser anterior al inicio de sesion.")
        return self

    @computed_field(return_type=bool)
    @property
    def hubo_click_en_recomendados(self) -> bool:
        return any(impresion.hubo_interes for impresion in self.recomendaciones)

    @computed_field(return_type=bool)
    @property
    def solo_uso_buscador(self) -> bool:
        if not self.consultas:
            return False
        return not self.hubo_click_en_recomendados

    @computed_field(return_type=list[ParPreferencia])
    @property
    def pares_pairwise(self) -> list[ParPreferencia]:
        pairs: list[ParPreferencia] = []

        for consulta in self.consultas:
            if consulta.video_seleccionado_id is None:
                continue

            for perdedor_id in consulta.videos_no_elegidos:
                pairs.append(
                    ParPreferencia(
                        user_id=self.user_id,
                        superficie=SuperficieRecomendacion.BUSQUEDA,
                        timestamp=consulta.timestamp,
                        ganador_video_id=consulta.video_seleccionado_id,
                        perdedor_video_id=perdedor_id,
                    )
                )

        for impresion in self.recomendaciones:
            if impresion.video_clicado_id is None:
                continue

            for perdedor_id in impresion.videos_no_elegidos:
                pairs.append(
                    ParPreferencia(
                        user_id=self.user_id,
                        superficie=impresion.superficie,
                        timestamp=impresion.timestamp,
                        ganador_video_id=impresion.video_clicado_id,
                        perdedor_video_id=perdedor_id,
                        video_contexto_id=impresion.video_contexto_id,
                    )
                )

        return pairs

    @computed_field(return_type=list[int])
    @property
    def videos_semilla_para_siguiente_sesion(self) -> list[int]:
        ordered = sorted(
            self.interacciones_video,
            key=lambda interaction: interaction.timestamp,
            reverse=True,
        )

        seeds: list[int] = []
        seen: set[int] = set()

        for interaction in ordered:
            if not interaction.positive_label or interaction.video_id in seen:
                continue
            seen.add(interaction.video_id)
            seeds.append(interaction.video_id)
            if len(seeds) == 10:
                break

        return seeds

    @computed_field(return_type=list[int])
    @property
    def canales_semilla_para_home(self) -> list[int]:
        ordered = sorted(
            self.interacciones_video,
            key=lambda interaction: interaction.timestamp,
            reverse=True,
        )

        channels: list[int] = []
        seen: set[int] = set()
        for interaction in ordered:
            if interaction.channel_id is None or not interaction.positive_label:
                continue
            if interaction.channel_id in seen:
                continue
            seen.add(interaction.channel_id)
            channels.append(interaction.channel_id)
            if len(channels) == 5:
                break

        return channels

    @computed_field(return_type=float)
    @property
    def implicit_score_total(self) -> float:
        return round(
            sum(interaction.implicit_score for interaction in self.interacciones_video),
            4,
        )


__all__ = [
    "CanalCreador",
    "ConsultaBusqueda",
    "EstrategiaRetrieval",
    "FuenteDescubrimiento",
    "ImpresionRecomendacion",
    "InteraccionVideo",
    "ParPreferencia",
    "RolCuenta",
    "SeguimientoCanal",
    "SenalesEngagement",
    "SesionRecomendacion",
    "SuperficieRecomendacion",
    "UsuarioPerfil",
    "VideoCatalogo",
]
