# Vision general del proyecto

## Que es este proyecto

Este repositorio modela un recomendador de videos inspirado en YouTube con datos sinteticos o semisinteticos. El objetivo tecnico no parece ser solo "tener un CSV limpio", sino construir la base de un sistema de recomendacion con varias superficies:

- `home` o "Para ti"
- `watch_next` o "despues de ver un video"
- `busqueda`

El foco actual esta en la preparacion del dato y en el modelado del dominio del recomendador.

## Lo que ya esta hecho

Por el codigo y las notas del repo, la secuencia mas probable de trabajo ha sido esta:

1. Se parte de un dataset bruto llamado `youtube recommendation dataset.csv`.
2. Se detectan problemas de calidad en varias columnas (`liked`, `category`, `timestamp`).
3. Se crea `generate_datasets.py` para separar el dataset original en dos tablas mas utiles:
   - historial de eventos de usuario
   - catalogo agregado de videos
4. Se deja `titulo` y `que_pasa` vacios en el catalogo de videos para rellenarlos mas adelante.
5. Se crea `generar_que_pasa.py` para generar keywords sinteticas por categoria.
6. Se genera otra version del catalogo con titulos.
7. Se crea `prepare_recommendation_data.py` para construir datasets de ML:
   - limpieza de eventos
   - features por usuario
   - features por video
   - interacciones agregadas usuario-video
   - splits para collaborative filtering
   - dataset de ranking
8. En paralelo, `recommendation_models.py` define la estructura conceptual del sistema final usando `Pydantic`.

## Lo que el proyecto intenta demostrar

El proyecto ya contiene las piezas tipicas de un recomendador moderno:

- limpieza del dato de consumo
- agregacion por usuario y por item
- labels implicitos
- separacion temporal train/validation/test
- dataset para ranking
- modelos de dominio para retrieval, ranking y sesiones

No hay aun una aplicacion final, un servicio online ni un entrenamiento integrado de extremo a extremo, pero si hay una base muy util para esa siguiente fase.

## Situacion tecnica actual

La carpeta `data/` es la fuente operativa principal. Los scripts trabajan sobre esos archivos:

- `generate_datasets.py` escribe en `data/`
- `prepare_recommendation_data.py` lee `data/usuarios.csv` y `data/videos.csv`
- `reco_output/` guarda los datasets derivados para ML

Punto importante: ahora mismo el pipeline usa `data/videos.csv` como entrada activa del catalogo. Eso significa que el dataset fusionado `data/videos_completo.csv` existe, pero no entra en la preparacion del recomendador hasta que se sustituya la entrada o se ajuste el script.
