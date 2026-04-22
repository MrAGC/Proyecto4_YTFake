# Proyecto4_YTFake

Este repo usa una carpeta unica de datasets para app, entrenamiento y evaluacion:

- datasets_unificados_usados/

## Estructura minima importante

- app/: FastAPI y serving de recomendaciones.
- ml/: utilidades de retrieval.
- scripts/data/: construccion y preparacion de datasets.
- scripts/training/: entrenamiento del ranker.
- scripts/evaluation/: auditoria y metricas.
- models/: modelos entrenados usados por la app.
- datasets_unificados_usados/: todos los CSV que consume el sistema.

## Carpeta unica de datasets

Todo lo que se usa en ejecucion y entrenamiento sale de:

- datasets_unificados_usados/

Archivos esperados en esa carpeta:

- usuarios.csv
- usuarios_perfiles.csv
- videos.csv
- canales.csv
- seguimientos_canales.csv
- local_session_events.csv
- local_search_events.csv
- local_engagement_events.csv
- user_features.csv
- video_features.csv
- user_channel_features.csv
- cf_interactions.csv
- ranking_dataset.csv
- training_dataset_balanced_v1.csv
- training_dataset_balanced_v1_metadata.json

## Instalacion rapida

En PowerShell, desde la raiz del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## Ejecutar la app

```powershell
python main.py
```

Abrir en navegador:

- http://127.0.0.1:8000/login

## Entrenar el ranker

```powershell
python scripts\training\train_recommender_ranker.py
```

Salida esperada:

- models/ranker_v1/ranker_model.json
- models/ranker_v1/metrics.json
- models/ranker_v1/feature_importance.csv

## Regenerar datasets de recomendacion

Si quieres reconstruir features/datasets dentro de la carpeta unica:

```powershell
python scripts\data\prepare_recommendation_data.py
python scripts\data\build_training_dataset.py
```

## Evaluacion

```powershell
python scripts\evaluation\evaluate_recommender_quality.py
python scripts\evaluation\analyze_training_data.py
python scripts\evaluation\analyze_retrieval_quality.py
```

## Nota operativa

Los modelos no se guardan dentro de datasets_unificados_usados; se mantienen en models/.
La carpeta datasets_unificados_usados solo contiene datos CSV y metadata de dataset.

