# Entrenamiento y validacion del recomendador

## Que estamos entrenando de verdad

La arquitectura correcta para este proyecto no es "un solo CSV gigante y un solo modelo para todo". Lo que tenemos planteado ahora es un sistema de dos capas:

1. `Retrieval` con `Collaborative Filtering`
2. `Ranking` con un ranker compartido que puntua candidatos

Eso encaja con la idea del proyecto:

- `home` y listas tipo Netflix usan un ranker compartido
- `watch_next` usa ese mismo ranker pero con otro contexto
- `search` deberia usar retrieval textual primero y ese ranker como reranker

## Como se reparte cada dataset

### 1. Datos base

Los datos de entrada que sostienen todo son:

- `data/usuarios.csv`: eventos crudos usuario-video
- `data/videos.csv`: catalogo del item
- `data/canales.csv`: estado agregado de los creadores/canales
- `data/seguimientos_canales.csv`: follows usuario-canal

### 2. Dataset para Retrieval + Collaborative Filtering

El retrieval colaborativo sale de `reco_output_v2/cf_interactions.csv`.

Este dataset esta bien planteado para CF porque:

- trabaja a nivel `user_id` + `video_id`
- usa solo pares positivos
- conserva `implicit_score_sum`, `implicit_score_mean` y `cf_confidence`
- tiene split temporal por usuario:
  - `train`: 653712
  - `validation`: 99065
  - `test`: 99807

Esto sirve para entrenar un retrieval colaborativo tipo:

- matrix factorization
- item-item CF
- retrieval por vecinos

Lo importante es que aqui no intentamos ordenar toda la home. Aqui solo queremos recuperar candidatos razonables.

### 3. Dataset para Ranking

El ranking sale primero en `reco_output_v2/ranking_dataset.csv` y despues se transforma en `reco_output_v2/training_dataset_balanced_v1.csv`.

Este dataset esta bien planteado para ranking porque mezcla:

- features de usuario
- features de video
- features usuario-canal
- features cruzadas
- recencia
- follows
- frescura del video
- afinidad reciente e historica

En concreto, ya tenemos senales que van en la direccion correcta:

- `recent_category_match`
- `channel_current_interest_score`
- `days_since_last_channel_watch`
- `stale_follow_flag`
- `video_freshness_score`
- `user_follows_channel`

Esto cuadra con la regla de negocio que querias:

- seguir un canal ayuda
- pero pesa mas lo que el usuario esta viendo ahora

## Que validaciones se han hecho

### 1. Limpieza y normalizacion logica

No hemos hecho "normalizacion" estilo redes neuronales porque el primer modelo es `XGBoost`, y para arboles eso no es obligatorio.

Lo que si se ha hecho, y era lo importante, es:

- corregir tiempos negativos
- corregir duraciones imposibles
- recalcular `watch_percent` cuando el dato venia mal
- acotar tasas a rangos validos
- construir labels y `implicit_score`
- separar senales por usuario, item y canal

Eso esta implementado en `prepare_recommendation_data.py`.

### 2. Auditoria del dataset de ranking

Se ha creado una auditoria especifica en:

- `reports/training_audit_v1/audit_report.md`

Y sale de:

- `analyze_training_data.py`

Hallazgos importantes:

- `ranking_dataset.csv` tenia `894660` filas
- la label estaba fuertemente desbalanceada:
  - positivos: `852584`
  - negativos: `42076`
  - ratio positiva: `0.95297`

Esto era un problema serio porque:

- el modelo iba a aprender que casi todo es positivo
- metricas como `precision@10` podian salir artificialmente bonitas

### 3. Correlaciones y columnas sospechosas

La auditoria no se uso para decir "cuanto mas correlacion, mejor". Se uso para detectar dos cosas:

1. columnas que parecen demasiado cercanas al resultado final
2. columnas duplicadas o casi duplicadas

Ejemplos de correlaciones sospechosas:

- `positive_label` con la label: `1.0`
- `negative_label` con la label: `-1.0`
- `implicit_score_mean`: `0.549679`
- `cf_confidence`: `0.549501`
- `implicit_score_sum`: `0.549501`

Esto no significa que "el dataset sea buenisimo". Significa que habia leakage probable o senales demasiado cerca del outcome observado.

Tambien aparecieron pares casi duplicados:

- `positive_label` <-> `ranking_label`: `1.0`
- `negative_label` <-> `ranking_label`: `1.0`
- `implicit_score_sum` <-> `cf_confidence`: `1.0`
- `creator_matches_user_pref` <-> `channel_matches_user_pref`: `1.0`
- `recent_creator_match` <-> `recent_channel_match`: `1.0`

Esto era justo lo que habia que comprobar para no entrenar a ciegas.

### 4. Correccion: dataset balanceado para el ranker

Para el primer entrenamiento serio del ranker se creo:

- `build_training_dataset.py`
- salida: `reco_output_v2/training_dataset_balanced_v1.csv`

Este dataset ya esta mucho mejor planteado para ranking porque:

- deja `300000` positivos
- deja `300000` negativos
- mezcla negativos observados y sinteticos plausibles
- mantiene split temporal

Resumen:

- filas totales: `600000`
- positivos: `300000`
- negativos: `300000`
- negativos observados: `42076`
- negativos sinteticos: `257924`

Los negativos sinteticos no son aleatorios puros. Se construyen con videos no vistos del usuario, preferiblemente en categorias afines o recientes. Eso hace que el entrenamiento sea mas dificil y mas util.

## Que significan las metricas actuales

El trainer del ranker esta en:

- `train_recommender_ranker.py`

Y los artefactos salen en:

- `models/ranker_v1/`

Metricas actuales del ranker sobre el dataset balanceado:

- `valid_roc_auc`: `0.94682`
- `test_roc_auc`: `0.935157`
- `valid_average_precision`: `0.923533`
- `test_average_precision`: `0.900327`
- `valid_precision@1`: `0.882526`
- `test_precision@1`: `0.872668`

Interpretacion correcta:

- el modelo si esta aprendiendo senales utiles
- la recencia y la afinidad reciente pesan fuerte
- el follow historico no domina

Pero hay un limite importante:

- en `valid` y `test` solo hay unas `2` candidaturas medias por usuario
- por eso `recall@10 = 1.0` no significa que el sistema ya este resuelto

En otras palabras:

- la preparacion de datos ya es razonable
- la evaluacion offline aun no es suficientemente dura para medir una home real

## Entonces, estan bien los datos para lo que queremos o no

### Si, para esta fase

Los datos estan bien planteados para arrancar un sistema tipo YouTube con dos fases:

- `CF` para retrieval
- `ranker` para ordenar candidatos

Motivos:

- el dataset base tiene eventos de consumo
- hay catalogo de videos
- hay creadores/canales
- hay follows
- hay frescura temporal
- hay afinidad usuario-canal
- hay split temporal
- ya hay una auditoria de leakage y desbalance

### Todavia no, para llamarlo sistema final

Todavia faltan cosas si queremos decir que ya replica bien todas las superficies:

- entrenar de verdad el retrieval colaborativo
- generar datasets especificos para `watch_next`
- preparar `search` como retrieval textual + reranking
- evaluar con muchos mas candidatos por usuario
- aprovechar mejor el texto de `titulo` y `que_pasa`

## Conclusiones practicas

La conclusion correcta ahora mismo es esta:

- el planteamiento de datos ya va bien
- la separacion entre retrieval y ranking esta bien hecha
- la auditoria encontro problemas reales
- esos problemas ya se han corregido en una primera version de entrenamiento

No estamos en "modelo final". Estamos en una fase buena para seguir porque ya se ha validado lo importante:

- que no entrenamos completamente a ciegas
- que el dataset original tenia sesgos claros
- que el ranker balanceado aprende senales coherentes con el producto que quieres

## Siguiente paso recomendado

El siguiente paso con mas impacto no es tocar hiperparametros. Es este:

1. entrenar el retrieval colaborativo con `cf_interactions.csv`
2. generar un conjunto de evaluacion con mas candidatos por usuario
3. despues conectar retrieval + ranker en un pipeline de inferencia

Ese orden tiene mas sentido que intentar exprimir mas el ranker actual.
