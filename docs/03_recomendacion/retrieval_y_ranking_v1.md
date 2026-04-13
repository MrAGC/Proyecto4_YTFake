# Retrieval y ranking v1

## Arquitectura elegida

La arquitectura planteada para este proyecto queda asi:

1. `Retrieval` por `Collaborative Filtering`
2. `Ranking` con features extra del usuario, del video y del canal

La division exacta queda asi:

- `Retrieval principal`: `User-based CF`
- `Retrieval secundario`: `Item-based CF`
- `Ranking a comparar`: `PointWise Dense NN` frente a `PairWise XGBoost` con informacion extra

La idea de producto es correcta porque separa dos problemas distintos:

- primero recuperar candidatos razonables
- despues ordenar esos candidatos con mas contexto

## Como funciona cada capa

### 1. Retrieval principal: User-based CF

Pregunta que responde:

- que videos pueden gustarle a este usuario porque se parecen a los que gustaron a usuarios parecidos

Como funciona:

- se mira el historial positivo del usuario
- se buscan usuarios parecidos por videos vistos
- se traen videos que esos usuarios vieron y el usuario actual aun no ha visto

Ventaja:

- capta afinidad colectiva entre usuarios

Problema:

- si el usuario tiene poco historial o el dataset es muy disperso, se vuelve debil

### 2. Retrieval secundario: Item-based CF

Pregunta que responde:

- dado lo que ya vio el usuario, que videos se parecen a esos videos

Como funciona:

- se construyen similitudes video-video a partir de co-consumo
- para cada video ya visto, se sacan vecinos parecidos
- se acumulan esos scores

Ventaja:

- da continuidad, especialmente util para `watch_next`

Problema:

- si el co-consumo entre items es pobre, la recuperacion sigue siendo floja

### 3. Ranking

Pregunta que responde:

- de los candidatos ya recuperados, cuales deberian ir primero

Aqui metemos informacion que el retrieval puro no usa bien:

- interes reciente del usuario
- categoria reciente
- canal reciente
- follow del canal
- frescura del video
- engagement del item
- recencia de consumo del canal
- senales cruzadas usuario-video y usuario-canal

## Datasets que usa cada pieza

### Retrieval CF

Archivo:

- `reco_output_v2/cf_interactions.csv`

Porque encaja:

- trabaja por `user_id + video_id`
- usa solo positivos
- tiene `implicit_score_sum`, `implicit_score_mean` y `cf_confidence`
- tiene split temporal real por usuario

Totales actuales:

- `train`: `653712`
- `validation`: `99065`
- `test`: `99807`
- usuarios train: `99977`
- videos train: `50000`

### Ranking

Archivos:

- `reco_output_v2/training_dataset_balanced_v1.csv`
- `models/ranker_compare_v1/`

Porque encaja:

- mezcla features de usuario, item y canal
- ya incorpora recencia y follows
- esta balanceado a `50/50`
- evita entrenar con el sesgo del dataset original, que estaba en `95.3%` positivos

## Modelos comparados en ranking

### 1. PointWise Dense NN

Script:

- `compare_ranking_models.py`

En esta version se ha usado una `MLPClassifier` densa como baseline pointwise.

Que hace:

- recibe un candidato
- le pone un score independiente
- luego se ordena por score

Ventaja:

- simple de entender
- buen baseline cuando ya tienes features buenas

Limite:

- no aprende la comparacion directa entre candidatos tan bien como un ranker pairwise
- en este repo corre en CPU porque no hay `PyTorch` ni `TensorFlow` instalados

### 2. PairWise XGBoost

Script:

- `compare_ranking_models.py`

Que hace:

- usa `objective = rank:pairwise`
- aprende a poner por delante el candidato bueno frente al malo dentro del mismo grupo de usuario
- usa toda la informacion extra del dataset

Ventaja:

- esta mas alineado con el problema real de ranking
- en este repo usa `CUDA`

## Resultados reales obtenidos

### Retrieval

Archivo:

- `models/retrieval_cf_v1/metrics.json`

Resultado importante:

- el retrieval CF puro sale muy flojo en este dataset

Metricas de `test`:

- `user_based_cf hit@10`: `0.00019`
- `item_based_cf hit@10`: `0.00020`
- `hybrid_cf hit@10`: `0.00019`
- `user_based_cf hit@50`: `0.000902`
- `item_based_cf hit@50`: `0.001022`
- `hybrid_cf hit@50`: `0.000982`

Interpretacion correcta:

- como baseline de retrieval puro, ahora mismo no recupera bien el holdout
- esto no invalida la arquitectura general
- lo que dice es que `memory-based CF` puro no basta con este dataset tal y como esta montado

La causa mas probable es esta:

- muchisimos usuarios
- muchisimos videos
- muy poco historial positivo por usuario
- matriz usuario-video demasiado dispersa

Conclusión tecnica:

- `User-based CF` e `Item-based CF` se pueden mantener como fuentes de candidatos
- pero no deberian ser la unica retrieval si queremos una home fuerte

### Ranking

Archivo:

- `models/ranker_compare_v1/comparison_summary.json`

Ganador actual:

- `pairwise_xgboost`

#### PointWise Dense NN

Metricas de `test`:

- `roc_auc`: `0.826307`
- `average_precision`: `0.797232`
- `precision@1`: `0.827005`
- `ndcg@10`: `0.970518`

#### PairWise XGBoost

Metricas de `test`:

- `roc_auc`: `0.977433`
- `average_precision`: `0.962763`
- `precision@1`: `0.973418`
- `ndcg@10`: `0.991004`

Interpretacion:

- el pairwise gana claramente
- el modelo que mejor encaja ahora mismo con este proyecto es `PairWise XGBoost`
- ademas usa `CUDA` y entrena en GPU

## Features que mas estan pesando

Archivo:

- `models/ranker_compare_v1/pairwise_xgboost/feature_importance.csv`

Las mas importantes ahora mismo son:

- `days_since_last_channel_watch`
- `recent_category_match`
- `category_matches_user_pref`
- `creator_matches_user_pref`
- `stale_follow_flag`
- `channel_current_interest_score`
- `user_avg_implicit_score`
- `user_follows_channel`

Esto es buena senal conceptual porque demuestra que el modelo esta aprendiendo justo lo que queriamos:

- importa mas lo reciente que un follow viejo
- importa el match con lo que esta viendo ahora
- el follow ayuda, pero no domina

## Como explicarselo al profesor

Forma corta y correcta:

- el sistema se ha planteado en dos etapas
- primero se recuperan candidatos con `Collaborative Filtering`
- despues esos candidatos se ordenan con un ranker que usa contexto adicional
- se comparo un modelo `PointWise Dense NN` contra un `PairWise XGBoost`
- el pairwise fue claramente mejor para ranking
- el retrieval CF puro, en cambio, salio flojo porque el dataset es muy disperso

Forma mas tecnica:

- `CF` sirve para reducir el espacio de busqueda
- `ranking` sirve para priorizar con mas senal semantica y contextual
- `precision@1` mide si el primer candidato es el correcto
- `ndcg@10` mide si el orden de los primeros resultados es bueno
- `roc_auc` y `average_precision` ayudan a ver separacion global de positivos y negativos, pero no son la metrica mas importante en ranking
- para retrieval, `hit@k` y `recall@k` son las metricas clave

## Como leer bien las metricas

### Retrieval

- `hit@10`: en cuantos usuarios aparecio el item objetivo dentro del top 10
- `hit@50`: lo mismo pero con top 50
- `recall@k`: cuanto del objetivo conseguido aparece dentro del top K
- `mrr@10`: premia que el item correcto salga muy arriba
- `coverage@50`: cuantos items distintos es capaz de recomendar el sistema

### Ranking

- `precision@1`: si el mejor candidato suele quedar el primero
- `precision@5` y `precision@10`: proporción de positivos dentro del top K
- `ndcg@10`: mide la calidad del orden, no solo si acierta o falla
- `roc_auc`: capacidad general de separar positivos de negativos
- `average_precision`: calidad global sobre los positivos

## Limitaciones honestas del estado actual

Hay que decir esto con claridad si te preguntan:

- el retrieval CF puro no esta funcionando bien todavia
- la evaluacion del ranking sigue siendo offline
- en validacion y test hay pocos candidatos por usuario, asi que algunas metricas siguen siendo optimistas
- `search` todavia no esta integrado como `retrieval textual + reranking`
- `watch_next` aun no tiene dataset especifico de evaluacion

## Decision recomendada a partir de aqui

Si se sigue esta arquitectura, el camino correcto es este:

1. mantener `PairWise XGBoost` como ranker principal actual
2. mantener `User-based CF` e `Item-based CF` como baseline de retrieval
3. mejorar retrieval con mas fuentes de candidatos, porque CF puro no llega
4. crear evaluacion especifica para `home` y `watch_next`

La idea general sigue siendo buena. Lo que nos han dicho los datos es que el `ranking` esta bastante mejor resuelto que el `retrieval`.
