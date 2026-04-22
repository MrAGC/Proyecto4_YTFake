# Graficas para la presentacion

## 1. Comparativa del modelo de ranking

Archivo: `docs/06_memoria/assets/ranking_comparativa_modelos.png`

Uso recomendado:
- explicar por que se eligio `Pairwise XGBoost` frente a `Pointwise Dense NN`,
- defender que el modelo final ordena mejor, no solo clasifica mejor.

Mensaje clave:
`Pairwise XGBoost` mejora claramente `NDCG@10`, `ROC AUC`, `Average Precision` y `Precision@1`.

## 2. Retrieval: item exacto vs acierto tematico

Archivo: `docs/06_memoria/assets/retrieval_item_vs_categoria.png`

Uso recomendado:
- explicar por que el retrieval no se debe defender por acierto exacto del item,
- justificar que para Home importa mucho recuperar bien el tema.

Mensaje clave:
El retrieval exacto por item es bajo, pero `Category Hit@50` es muy alto. Eso significa que el sistema recupera bien el tipo de contenido, aunque no siempre el video exacto del holdout.

## 3. Calidad offline del recomendador

Archivo: `docs/06_memoria/assets/ranking_calidad_offline.png`

Uso recomendado:
- ponerlo en la parte de resultados,
- enseñar que el ranker ordena muy bien dentro del conjunto de candidatos evaluado.

Mensaje clave:
Las metricas de ranking salen muy altas: `NDCG@10`, `MAP@10`, `MRR@10`, `HitRate@10` y `Recall@10`.

## 4. Composicion del negative sampling

Archivo: `docs/06_memoria/assets/negative_sampling_composicion.png`

Uso recomendado:
- explicar como se construyo el dataset de entrenamiento,
- justificar que no todo negativo es igual.

Mensaje clave:
El dataset balanceado mezcla positivos observados, negativos observados y negativos sinteticos.

## 5. Pool de candidatos por split

Archivo: `docs/06_memoria/assets/candidate_pool_por_split.png`

Uso recomendado:
- explicar una limitacion de la evaluacion,
- defender con honestidad que el siguiente paso es evaluar con mas candidatos por usuario.

Mensaje clave:
En test y validacion el numero de candidatos por usuario sigue siendo bajo, asi que las metricas de ranking hay que leerlas con ese contexto.

## 6. Balanceo del dataset de ranking

Archivo: `docs/06_memoria/assets/dataset_balanceo_ranking.png`

Uso recomendado:
- explicar por que hubo que rebalancear el dataset,
- enseñar visualmente el cambio entre dataset original y balanceado.

Mensaje clave:
El dataset original estaba fuertemente sesgado a positivos y se reequilibro para estabilizar el entrenamiento del ranker.
