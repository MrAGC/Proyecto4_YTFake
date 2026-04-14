# Estructura del repositorio

## Raiz del proyecto

- `README.md`: resumen operativo y comandos principales.
- `main.py`: punto de entrada para levantar la app con Uvicorn.
- `pyproject.toml`: dependencias principales del proyecto.
- `uv.lock`: lockfile del entorno.

La raiz queda reservada para configuracion y arranque. La logica de negocio, scripts y pruebas viven en carpetas separadas.

## App

- `app/webapp.py`: rutas FastAPI, renderizado de templates y cookies de usuario/invitado.
- `app/reco_serving.py`: servicio de recomendacion usado por la web, busqueda, home, watch-next, likes, suscripciones e historial local.
- `app/recommendation_models.py`: modelos Pydantic del dominio del recomendador.

## Modelos reutilizables

- `ml/train_retrieval_cf.py`: collaborative filtering base, principalmente user-based y secundariamente item-based.
- `ml/train_retrieval_multisource.py`: retrieval combinado con senales de categoria, canal y frescura.

Esta carpeta contiene funciones que tambien se importan desde la app.

## Scripts operativos

- `scripts/data/generate_datasets.py`: genera datasets base desde el CSV original.
- `scripts/data/prepare_recommendation_data.py`: crea features y datasets derivados para retrieval/ranking.
- `scripts/data/build_training_dataset.py`: construye el dataset balanceado para entrenar el ranker.
- `scripts/data/update_video_thumbnails.py`: completa URLs de miniaturas.
- `scripts/data/generar_que_pasa.py`: enriquecimiento legacy de keywords.
- `scripts/training/train_recommender_ranker.py`: entrenamiento del ranker principal.
- `scripts/evaluation/analyze_training_data.py`: auditoria de datos de entrenamiento.
- `scripts/evaluation/analyze_retrieval_quality.py`: diagnostico del retrieval.
- `scripts/evaluation/compare_ranking_models.py`: comparativa pointwise/pairwise.

## Frontend

- `web/templates/`: paginas Jinja de login, home, busqueda, watch, perfil, historial y studio.
- `web/static/styles.css`: estilos de la demo.

## Datos y salidas

- `data/`: CSV activos que usa el proyecto.
- `data/viejos/`: CSV antiguos o duplicados conservados para no perder historico.
- `reco_output/` y `reco_output_v2/`: datasets derivados para entrenamiento y serving.
- `models/`: modelos entrenados y artefactos de ranking/retrieval.
- `reports/`: auditorias y diagnosticos generados por scripts.

## Pruebas y prototipos

- `tests/test_buscador.py`: prototipo local del buscador.
- `tests/testdatos.py`: inspeccion simple de categorias del dataset bruto.

## Documentacion

- `docs/02_datos/`: historia, limpieza y columnas de datasets.
- `docs/03_recomendacion/`: arquitectura de retrieval/ranking, diagnosticos y notas del recomendador.
- `docs/04_pydantic/`: modelos Pydantic.
- `docs/05_arquitectura/`: decisiones de arquitectura del modelo/ranker.
