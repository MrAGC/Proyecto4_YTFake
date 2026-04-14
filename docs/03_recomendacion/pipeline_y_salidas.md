# Pipeline de recomendacion y salidas

## Objetivo del script

`scripts/data/prepare_recommendation_data.py` convierte las tablas limpias de `data/` en datasets utiles para entrenar o evaluar recomendadores.

Entrada activa hoy:

- `data/usuarios.csv`
- `data/videos.csv`

Salida:

- `reco_output/eventos_limpios.csv`
- `reco_output/user_features.csv`
- `reco_output/video_features.csv`
- `reco_output/pair_interactions.csv`
- `reco_output/cf_interactions.csv`
- `reco_output/ranking_dataset.csv`

## Flujo paso a paso

### 1. `build_video_features`

Convierte el catalogo de videos en features modelables:

- limpieza de tipos numericos
- duracion corregida
- tasas acotadas
- popularidad logaritmica
- score base de engagement
- banderas de metadata disponible

## 2. `clean_user_events`

Convierte `usuarios.csv` en eventos robustos para ML:

- limpia binarios
- limita tiempos
- recalcula porcentaje visto si hace falta
- une categoria y duracion del video
- crea labels y `implicit_score`

## 3. `build_user_features`

Agrega comportamiento por usuario:

- intensidad de uso
- diversidad de videos y categorias
- engagement medio
- propension a clicar recomendaciones
- categoria, dispositivo y franja preferidos

## 4. `build_pair_interactions`

Agrega todo a nivel `user_id` + `video_id`:

- numero de interacciones
- rangos temporales
- maxima y media de retencion
- score implicito acumulado
- confianza para CF

## 5. `build_cf_splits`

Filtra solo pares positivos y crea un split temporal por usuario:

- el ultimo positivo va a `test`
- el penultimo positivo va a `validation`
- el resto queda en `train`

Esto es una señal de que se quiere evaluar de forma mas realista que con una particion aleatoria.

## 6. `build_ranking_dataset`

Une:

- pares etiquetados
- features de usuario
- features de video

Y crea features cruzadas como:

- `category_matches_user_pref`
- `watch_vs_user_avg`
- `item_vs_user_score_gap`

## Interpretacion del pipeline

El proyecto ya tiene la estructura de un sistema clasico de recomendacion por dos etapas:

1. recuperacion o filtrado de candidatos
2. ranking final

De hecho, las notas del repo y los modelos `Pydantic` apuntan explicitamente a ese enfoque:

- collaborative filtering para retrieval
- ranking pointwise o pairwise para ordenar candidatos

## Limitacion actual importante

Aunque el repo ya tiene `data/videos_completo.csv`, este pipeline sigue trabajando con `data/videos.csv`. Por eso:

- `has_title_metadata` y `has_keyword_metadata` pueden salir a `0`
- la parte textual aun no esta plenamente integrada en las features activas

## Siguiente mejora natural

La mejora mas directa seria cambiar la entrada del pipeline para usar el catalogo enriquecido completo. Eso permitiria:

- activar mejor la parte de busqueda
- preparar similitud entre videos con mas contexto textual
- enriquecer el ranking con metadata real del item
