# Proyecto4_YTFake

La documentacion ampliada del proyecto esta en `docs/README.md`.

## Estructura rapida

- `app/`: FastAPI, rutas web y capa de serving del recomendador.
- `ml/`: funciones reutilizables de retrieval/collaborative filtering.
- `scripts/data/`: generacion, limpieza, enriquecimiento y preparacion de datasets.
- `scripts/training/`: entrenamiento de modelos.
- `scripts/evaluation/`: auditorias, comparativas y diagnosticos.
- `tests/`: pruebas/prototipos locales.
- `web/`: plantillas HTML y CSS.
- `data/`: CSV activos; `data/viejos/` conserva versiones antiguas.

## CSV necesarios

Estructura recomendada de datasets:

- `data/`: datasets activos
- `data/viejos/`: versiones antiguas, enriquecimientos legacy y duplicados que no se eliminan

### Para generar los datasets base

Necesitas este archivo de entrada:

- `data/youtube recommendation dataset.csv`

Con este comando:

```powershell
.\.venv\Scripts\python.exe scripts\data\generate_datasets.py
```

se generan los CSV base del proyecto:

- `data/usuarios.csv`
- `data/usuarios_perfiles.csv`
- `data/canales.csv`
- `data/seguimientos_canales.csv`
- `data/videos.csv`

### Para preparar el pipeline de recomendacion

`scripts/data/prepare_recommendation_data.py` necesita estos CSV en `data/`:

- `usuarios.csv`
- `videos.csv`
- `canales.csv`
- `seguimientos_canales.csv`

CSV opcional:

- ninguno

Metricas previstas:
- HitRate
- Precision@K
- Coverage
- NDCG

## Preparacion de datos

Para preparar los datos del recomendador:

```powershell
.\.venv\Scripts\python.exe scripts\data\prepare_recommendation_data.py
```

El script genera la salida en `reco_output/` y, si esa carpeta esta bloqueada, en `reco_output_v2/`.

- `eventos_limpios.csv`: eventos usuario-video con limpieza y score implicito.
- `user_features.csv`: features agregadas por usuario.
- `user_channel_features.csv`: afinidad usuario-canal con recencia, follows y stale follow.
- `video_features.csv`: features agregadas por video.
- `creator_features.csv`: features agregadas por canal/creador.
- `pair_interactions.csv`: pares usuario-video agregados.
- `cf_interactions.csv`: interacciones positivas listas para collaborative filtering con split temporal `train/validation/test`.
- `ranking_dataset.csv`: dataset etiquetado para modelos de ranking pointwise/pairwise.

## Criterio actual de preparacion

- Se corrigen valores invalidos en `porcentaje_visto`, tiempos negativos e inconsistencias de duracion.
- Se crea un `implicit_score` usando watch percent, like, comentario, suscripcion y click en recomendacion.
- Se incorporan senales de creador/canal, follows y frescura del video.
- El follow de un canal no domina por si solo: la recencia de consumo pesa mas que el seguimiento historico.
- Se marca como positivo un par usuario-video si hay alta retencion o accion explicita.
- Se marcan negativos claros para el dataset de ranking cuando hay rebote y ninguna accion positiva.

## App web

La app actual conecta el `retrieval` con el `ranker` y expone una demo web con:

- login simple por usuario
- modo invitado local con cookie persistente en el mismo navegador/equipo
- home con estilo YouTube + listas tipo Netflix
- barra lateral con perfil simulado, historial de sesion e historial base
- pagina de visionado con `watch-next`
- mezcla entre perfil historico y sesion actual para recalcular la home al volver del visionado
- buscador preparado como placeholder visual para conectar la busqueda mas adelante
- persistencia temporal de eventos de sesion en `data/local_session_events.csv`

### Arranque

Instala dependencias del proyecto y lanza:

```powershell
.\.venv\Scripts\python.exe main.py
```

La app queda disponible en:

- `http://127.0.0.1:8000/login`

### Modelos usados por la app

- `models/retrieval_multisource_v1/`: retrieval para home
- `models/ranker_compare_v1/pairwise_xgboost/`: ranker principal

### Nota sobre el buscador

Se revisaron las ramas remotas `origin/feature` y `origin/FeatureEngineering` y el repo actual. No hay una implementacion reutilizable del buscador todavia, asi que la interfaz queda preparada para conectarla en la siguiente fase.

## Miniaturas

Para completar `thumbnail_url` y `thumbnail_source` en `data/videos.csv`:

```powershell
.\.venv\Scripts\python.exe scripts\data\update_video_thumbnails.py --limit 100
```

Orden de resolucion:

- Google Custom Search API, si defines `GOOGLE_CSE_API_KEY` y `GOOGLE_CSE_CX`
- scraping best-effort de Google Images
- fallback a miniatura de YouTube si Google bloquea la respuesta

