"""
generate_datasets.py
────────────────────
Lee el dataset original de YouTube y genera dos CSVs limpios:

  data/usuarios.csv  → historial de interacciones por usuario (1 fila = 1 vista)
  data/videos.csv    → catálogo de vídeos con métricas agregadas  (1 fila = 1 vídeo)

Limpieza aplicada:
  - Columna 'liked': normaliza '0'/'no' → 0 | '1'/'yes'/'2' → 1 | NaN → 0
  - Columna 'category': unifica variantes sucias (MUsic→Music, gamingg→Gaming, etc.)
"""

import os
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA
# ─────────────────────────────────────────────────────────────────────────────
print("Cargando dataset original…")
df = pd.read_csv("data/youtube recommendation dataset.csv")
print(f"  {len(df):,} filas cargadas  |  columnas: {list(df.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. LIMPIEZA — columna 'liked'
#    Valores encontrados: '0', '1', '2', 'no', 'yes', NaN
# ─────────────────────────────────────────────────────────────────────────────
liked_map = {"0": 0, "1": 1, "2": 1, "no": 0, "yes": 1}
df["liked"] = df["liked"].map(liked_map).fillna(0).astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# 3. LIMPIEZA — columna 'category'
#    Variantes sucias identificadas en el dataset:
#      MUsic / music → Music
#      gamingg       → Gaming
#      Ed            → Education
#      COMEDY        → Comedy
#      'Tech '       → Tech  (espacio trailing)
# ─────────────────────────────────────────────────────────────────────────────
category_map = {
    "Music":     "Music",
    "MUsic":     "Music",
    "music":     "Music",
    "Gaming":    "Gaming",
    "gamingg":   "Gaming",
    "Education": "Education",
    "Ed":        "Education",
    "Comedy":    "Comedy",
    "COMEDY":    "Comedy",
    "Tech":      "Tech",
    "Tech ":     "Tech",
    "Sports":    "Sports",
    "News":      "News",
    "Lifestyle": "Lifestyle",
}
before = df["category"].nunique()
df["category"] = df["category"].map(category_map).fillna(df["category"])
after = df["category"].nunique()
print(f"  Categorías: {before} variantes → {after} categorías limpias: {sorted(df['category'].unique())}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PARSEAR TIMESTAMP  (vectorizado)
#    El campo mezcla ISO-8601 ("2024-07-09 10:22:22") y Unix epoch ("1744178555")
# ─────────────────────────────────────────────────────────────────────────────
ts = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
epoch_mask = ts.isna()
if epoch_mask.any():
    numeric_vals = pd.to_numeric(df["timestamp"][epoch_mask], errors="coerce")
    ts.loc[epoch_mask] = pd.to_datetime(numeric_vals, unit="s", errors="coerce")
df["timestamp"] = ts

# ══════════════════════════════════════════════════════════════════════════════
#  TABLA USUARIOS
#  Una fila = una interacción (usuario × vídeo).
#  Columnas:
#    user_id           – identificador del usuario
#    video_id          – vídeo que vio
#    dispositivo       – desde dónde lo vio (TV/Mobile/Desktop/Tablet)
#    timestamp         – fecha y hora exacta de la vista
#    franja_horaria    – Morning / Afternoon / Evening / Night
#    tiempo_visto_s    – segundos que estuvo viendo el vídeo
#    porcentaje_visto  – % del vídeo completado (0.0 – 1.0)
#    like              – 0/1 si dio like
#    comentario        – 0/1 si comentó
#    suscripcion       – 0/1 si se suscribió tras ver el vídeo
#    fue_recomendado   – 0/1 si el vídeo le fue sugerido por el algoritmo
#    clic_recomendacion– 0/1 si hizo clic en la recomendación (CTR individual)
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerando tabla usuarios…")

users_df = df[[
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
]].copy().rename(columns={
    "device":            "dispositivo",
    "watch_time_of_day": "franja_horaria",
    "watch_time":        "tiempo_visto_s",
    "watch_percent":     "porcentaje_visto",
    "liked":             "like",
    "commented":         "comentario",
    "subscribed_after":  "suscripcion",
    "recommended":       "fue_recomendado",
    "clicked":           "clic_recomendacion",
})

users_df = users_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TABLA VIDEOS
#  Una fila = un vídeo único.  Datos agregados del historial de vistas.
#  Columnas:
#    video_id                  – identificador del vídeo
#    titulo                    – título real (vacío, rellenar manualmente)
#    video_duration_s          – duración total en segundos
#    category                  – categoría limpia
#    que_pasa                  – palabras clave de contenido (vacío: disparos, robots…)
#    total_views               – número total de reproducciones
#    total_likes               – likes acumulados
#    like_rate                 – likes / vistas
#    total_comments            – comentarios acumulados
#    comment_rate              – comentarios / vistas
#    suscriptores_ganados      – usuarios que se suscribieron tras ver el vídeo
#    subscription_rate         – suscriptores_ganados / vistas
#    avg_watch_percent         – media del % de vídeo completado
#    avg_watch_time_s          – media de segundos visto por vista
#    veces_recomendado         – cuántas veces el algoritmo lo sugirió
#    total_clics               – clics recibidos cuando fue recomendado
#    click_through_rate        – total_clics / veces_recomendado
# ══════════════════════════════════════════════════════════════════════════════
print("Generando tabla videos…")

videos_agg = (
    df.groupby("video_id")
    .agg(
        video_duration_s         =("video_duration",  "first"),
        total_views              =("user_id",         "count"),
        total_likes              =("liked",            "sum"),
        total_comments           =("commented",        "sum"),
        suscriptores_ganados     =("subscribed_after", "sum"),
        avg_watch_percent        =("watch_percent",    "mean"),
        avg_watch_time_s         =("watch_time",       "mean"),
        veces_recomendado        =("recommended",      "sum"),
        total_clics              =("clicked",          "sum"),
        # categoría más frecuente para ese video_id (por si hay variantes)
        category                 =("category",         lambda x: x.mode()[0]),
    )
    .reset_index()
)

# Métricas derivadas
videos_agg["like_rate"]          = (videos_agg["total_likes"]    / videos_agg["total_views"]).round(4)
videos_agg["comment_rate"]       = (videos_agg["total_comments"] / videos_agg["total_views"]).round(4)
videos_agg["subscription_rate"]  = (videos_agg["suscriptores_ganados"] / videos_agg["total_views"]).round(4)
videos_agg["click_through_rate"] = (
    videos_agg["total_clics"] / videos_agg["veces_recomendado"].replace(0, np.nan)
).round(4).fillna(0)

videos_agg["avg_watch_percent"] = videos_agg["avg_watch_percent"].round(4)
videos_agg["avg_watch_time_s"]  = videos_agg["avg_watch_time_s"].round(2)

# Columnas vacías para relleno manual posterior
videos_agg.insert(1, "titulo",   "")   # título real del vídeo
videos_agg.insert(4, "que_pasa", "")   # tags de contenido (disparos, saltar, robots…)

# Orden final de columnas
videos_df = videos_agg[[
    "video_id",
    "titulo",
    "video_duration_s",
    "category",
    "que_pasa",
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
]]

# ─────────────────────────────────────────────────────────────────────────────
# 5. GUARDAR
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
users_df.to_csv("data/usuarios.csv",  index=False)
videos_df.to_csv("data/videos.csv",   index=False)

print(f"\n✓ data/usuarios.csv  → {len(users_df):,} filas")
print(f"  Columnas: {list(users_df.columns)}")
print(f"\n✓ data/videos.csv   → {len(videos_df):,} filas")
print(f"  Columnas: {list(videos_df.columns)}")
print("\nPreview usuarios (3 filas):")
print(users_df.head(3).to_string())
print("\nPreview videos (3 filas):")
print(videos_df.head(3).to_string())
