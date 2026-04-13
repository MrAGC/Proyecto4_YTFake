# Datasets y columnas

## Datasets principales en `data/`

### `data/youtube recommendation dataset.csv`

Dataset bruto de origen. Es la fuente desde la que se derivan las tablas limpias.

### `data/usuarios.csv`

Tamano aproximado: 71.39 MB

Una fila representa una interaccion usuario-video.

Columnas:

- `user_id`: identificador del usuario
- `video_id`: identificador del video consumido
- `dispositivo`: dispositivo desde el que se vio
- `timestamp`: momento del evento
- `franja_horaria`: tramo horario de consumo
- `tiempo_visto_s`: segundos vistos
- `porcentaje_visto`: fraccion del video consumida
- `like`: si hubo like
- `comentario`: si hubo comentario
- `suscripcion`: si la vista genero suscripcion
- `fue_recomendado`: si el video vino sugerido por el sistema
- `clic_recomendacion`: si hubo click sobre una recomendacion

### `data/videos.csv`

Tamano aproximado: 3.47 MB

Catalogo base de videos agregado por `video_id`.

Columnas:

- `video_id`
- `titulo`
- `video_duration_s`
- `category`
- `que_pasa`
- `total_views`
- `total_likes`
- `like_rate`
- `total_comments`
- `comment_rate`
- `suscriptores_ganados`
- `subscription_rate`
- `avg_watch_percent`
- `avg_watch_time_s`
- `veces_recomendado`
- `total_clics`
- `click_through_rate`

En el catalogo base, `titulo` y `que_pasa` nacen vacios.

### `data/videos_con_titulos.csv`

Tamano aproximado: 6.02 MB

Version del catalogo donde se completa `titulo`.

### `data/videos_actualizado.csv`

Tamano aproximado: 5.54 MB

Version del catalogo donde se completa `que_pasa`.

### `data/videos_completo.csv`

Tamano aproximado: 9.62 MB

Es el catalogo mas completo del repositorio. Contiene a la vez:

- `titulo` rellenado
- `que_pasa` rellenado

Hoy es el mejor candidato para convertirse en el catalogo activo del pipeline.

## Datasets generados en `reco_output/`

### `reco_output/eventos_limpios.csv`

Tamano aproximado: 89.39 MB

Contiene eventos de usuario ya corregidos y enriquecidos con labels implicitos.

Columnas clave:

- `tiempo_visto_s_clean`
- `watch_percent_clean`
- `completion_flag`
- `positive_label`
- `negative_label`
- `implicit_score`

### `reco_output/user_features.csv`

Tamano aproximado: 12.79 MB

Una fila por usuario con agregados historicos, preferencias y comportamiento medio.

### `reco_output/video_features.csv`

Tamano aproximado: 4.39 MB

Una fila por video con features numericas y banderas de metadata:

- `video_engagement_score`
- `video_popularity_log`
- `has_title_metadata`
- `has_keyword_metadata`

### `reco_output/pair_interactions.csv`

Tamano aproximado: 97.56 MB

Una fila por par `user_id` + `video_id` agregando repeticiones de consumo y resumiendo engagement.

### `reco_output/cf_interactions.csv`

Tamano aproximado: 62.62 MB

Subconjunto positivo para collaborative filtering con `split` temporal:

- `train`
- `validation`
- `test`

### `reco_output/ranking_dataset.csv`

Tamano aproximado: 285.11 MB

Dataset final para ranking. Une:

- interaccion agregada
- features de usuario
- features de video
- variables cruzadas como `category_matches_user_pref`

Es el artefacto mas rico del pipeline actual.
