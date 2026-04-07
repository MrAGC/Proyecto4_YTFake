# Proyecto4_YTFake

La documentacion ampliada del proyecto esta en `docs/README.md`.

## CSV necesarios

Estructura recomendada de datasets:

- `data/`: datasets activos
- `data/viejos/`: versiones antiguas, enriquecimientos legacy y duplicados que no se eliminan

### Para generar los datasets base

Necesitas este archivo de entrada:

- `data/youtube recommendation dataset.csv`

Con este comando:

```powershell
.\.venv\Scripts\python.exe generate_datasets.py
```

se generan los CSV base del proyecto:

- `data/usuarios.csv`
- `data/usuarios_perfiles.csv`
- `data/canales.csv`
- `data/seguimientos_canales.csv`
- `data/videos.csv`

### Para preparar el pipeline de recomendacion

`prepare_recommendation_data.py` necesita estos CSV en `data/`:

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
.\.venv\Scripts\python.exe prepare_recommendation_data.py
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

