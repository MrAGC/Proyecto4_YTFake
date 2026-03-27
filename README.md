# Proyecto4_YTFake

Metricas previstas:
- HitRate
- Precision@K
- Coverage
- NDCG

## Preparacion de datos

La separacion base ya existe en `data/usuarios.csv` y `data/videos.csv`.

Para preparar los datos del recomendador:

```powershell
.\.venv\Scripts\python.exe prepare_recommendation_data.py
```

El script genera en `reco_output/`:

- `eventos_limpios.csv`: eventos usuario-video con limpieza y score implicito.
- `user_features.csv`: features agregadas por usuario.
- `video_features.csv`: features agregadas por video.
- `pair_interactions.csv`: pares usuario-video agregados.
- `cf_interactions.csv`: interacciones positivas listas para collaborative filtering con split temporal `train/validation/test`.
- `ranking_dataset.csv`: dataset etiquetado para modelos de ranking pointwise/pairwise.

## Criterio actual de preparacion

- Se corrigen valores invalidos en `porcentaje_visto`, tiempos negativos e inconsistencias de duracion.
- Se crea un `implicit_score` usando watch percent, like, comentario, suscripcion y click en recomendacion.
- Se marca como positivo un par usuario-video si hay alta retencion o accion explicita.
- Se marcan negativos claros para el dataset de ranking cuando hay rebote y ninguna accion positiva.

