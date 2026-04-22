# Evaluacion De Calidad Del Recomendador

## 1. Como se ha evaluado

Se usa el split `test` del dataset de ranking balanceado. Para cada usuario se ordenan candidatos con el ranker `pairwise_xgboost` y se calculan metricas @10.

Esta evaluacion sirve para demo y comparacion offline. La mejora pendiente es evaluar con pools de candidatos mas grandes por usuario.

## 2. Ranking

- Precision@10: `0.558746`
- HitRate@10: `1.0`
- Recall@10: `1.0`
- NDCG@10: `0.992039`
- MAP@10: `0.989067`
- MRR@10: `0.989297`

## 3. Experiencia De Usuario

- Coverage@10: `0.667047`
- Diversity categorias@10: `0.764362`
- Diversity canales@10: `0.99957`
- Novelty@10: `0.006805`
- Repeat vistos@10: `0.033202`

## 4. Negative Sampling

- Filas positivas: `300000`
- Filas negativas: `300000`
- Rate de positivos: `0.5`
- Negativos observados: `42076`
- Negativos sinteticos: `257924`
- Hard-negative proxy en sinteticos: `1.0`

Interpretacion:

- Los negativos observados vienen de baja retencion sin acciones positivas.
- Los negativos sinteticos son videos no vistos, preferiblemente de categorias afines o recientes.
- Falta separar explicitamente hard negatives por retrieval/categoria/canal para que el report sea mas defendible.

## 5. Pool De Candidatos

La evaluacion del ranker tiene pocos candidatos por usuario; para demo esta bien, pero para produccion conviene evaluar con 20-100 candidatos realistas por usuario.

- `train`: 93517 usuarios, media 5.133 candidatos, mediana 5.0, max 22
- `test`: 28995 usuarios, media 2.07 candidatos, mediana 2.0, max 10
- `valid`: 28900 usuarios, media 2.075 candidatos, mediana 2.0, max 10

## 6. Retrieval

- Test category_hit@50: `0.997565`
- Test hit@50 exacto: `0.002004`

Lectura: el retrieval exacto por item es demasiado estricto para este dataset, pero recupera muy bien el tema.