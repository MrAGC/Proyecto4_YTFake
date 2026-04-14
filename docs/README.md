# Documentacion del proyecto

Esta carpeta organiza la documentacion por areas funcionales para que se pueda entender el proyecto sin tener que reconstruir toda la historia leyendo solo scripts y CSV.

## Mapa rapido

- `01_general/vision_general.md`: que problema intenta resolver el proyecto y en que punto esta.
- `01_general/estructura_del_repo.md`: mapa de archivos y carpetas importantes.
- `02_datos/historia_del_dato.md`: reconstruccion de lo que se ha ido haciendo con el dataset.
- `02_datos/datasets_y_columnas.md`: inventario de datasets y significado de sus columnas.
- `02_datos/limpieza_y_enriquecimiento.md`: reglas de limpieza y enriquecimiento observadas en el codigo.
- `03_recomendacion/pipeline_y_salidas.md`: pipeline actual para preparar datos del recomendador.
- `03_recomendacion/entrenamiento_y_validacion.md`: como se entrena Retrieval + Collaborative Filtering y que validaciones se han hecho.
- `03_recomendacion/retrieval_y_ranking_v1.md`: arquitectura elegida, comparativa de modelos y como explicarlo.
- `03_recomendacion/retrieval_diagnostico_v1.md`: por que el retrieval exacto parecia malo y como leerlo bien.
- `03_recomendacion/retrieval_multisource_v1.md`: mejora de retrieval para home y resultados frente al CF puro.
- `03_recomendacion/notas_recomendador.md`: notas estrategicas sobre home, watch-next y busqueda.
- `04_pydantic/README.md`: indice de modelos Pydantic.
- `04_pydantic/modelos_base_y_eventos.md`: enums, modelos base y eventos.
- `04_pydantic/modelos_de_negocio.md`: perfil de usuario, catalogo de videos y sesion de recomendacion.
- `05_arquitectura/modelo_unico_ranker.md`: diseno del ranker compartido para `home`, `watch_next` y listas.

## Idea principal

El proyecto no esta construido como una app final, sino como un entorno de preparacion de datos y diseno de dominio para un sistema de recomendacion estilo YouTube. La parte mas madura es la de datos:

1. partir del dataset bruto
2. limpiarlo y separarlo en `usuarios.csv` y `videos.csv`
3. enriquecer videos con metadatos sinteticos
4. preparar datasets derivados para collaborative filtering y ranking
5. auditar el dataset de ranking y construir una version balanceada para entrenamiento

La app de demo vive en `app/` y usa `web/` para templates/estilos. Los scripts operativos se separan en `scripts/data/`, `scripts/training/` y `scripts/evaluation/`.

## Estado actual resumido

- El dataset fuente vive en `data/youtube recommendation dataset.csv`.
- El catalogo activo de videos vive en `data/videos.csv` y ya es el que usa el pipeline.
- Los enriquecimientos y versiones intermedias antiguas se han movido a `data/viejos/`.
- El pipeline actual genera:
  - `reco_output_v2/cf_interactions.csv` para retrieval collaborative filtering
  - `reco_output_v2/ranking_dataset.csv` para ranking
  - `reco_output_v2/training_dataset_balanced_v1.csv` como dataset balanceado para entrenar el ranker
- Ya existe una primera auditoria de entrenamiento en `reports/training_audit_v1/`.
