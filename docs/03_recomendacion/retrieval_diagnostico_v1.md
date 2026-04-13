# Diagnostico del retrieval

## Que se queria comprobar

Se queria entender si el retrieval basado en `Collaborative Filtering` estaba fallando de verdad o si la metrica usada estaba penalizando candidatos que, para una home estilo YouTube, en realidad si eran validos.

## Datos usados

- `reco_output_v2/cf_interactions.csv`
- `reco_output_v2/video_features.csv`
- `models/retrieval_cf_v1/retrieval_model.pkl`
- `reports/retrieval_diagnostics_v1/diagnostics.json`

## Resultado corto

El retrieval memory-based no esta fallando por falta de datos ni por cold start del item.

Lo que esta pasando es esto:

- la metrica de `hit exacto al mismo video` sale muy baja
- pero el retrieval mete candidatos de la categoria correcta en casi todos los casos

Eso significa que el retrieval esta recuperando bien el tema, pero la evaluacion por item exacto es demasiado dura para este dataset.

## Numeros clave

### Estado del train

- filas train: `653712`
- usuarios train: `99977`
- videos train: `50000`
- positivos medios por usuario: `6.54`
- soporte medio por item en train: `13.07` usuarios

### Holdout

En `validation` y `test`:

- `unseen_item_rate_vs_train = 0.0`
- el item objetivo si existe en train
- el soporte medio del item objetivo ronda `13` usuarios

Esto descarta la explicacion de que el retrieval falle porque el item no exista todavia o porque no tenga soporte.

### Solape real con el historial

En el holdout:

- el item comparte `categoria` con el historial del usuario en ~`56.5%` de los casos
- el item comparte `canal` con el historial del usuario en solo ~`0.15% - 0.18%`

Esto es importante porque describe la forma real del dataset:

- el comportamiento se parece mas a preferencia por `tema/categoria`
- mucho menos a repeticion fuerte del mismo `canal`

### Calidad del retrieval

Metricas exactas por item en `test`:

- `user_based_cf hit@10`: `0.00019`
- `item_based_cf hit@10`: `0.00020`
- `hybrid_cf hit@50`: `0.000982`

Parecen malisimas si solo miramos exact match.

Pero la metrica importante del diagnostico es esta:

- `user_based_cf category_hit@50`: `0.997275`
- `item_based_cf category_hit@50`: `0.997175`
- `hybrid_cf category_hit@50`: `0.997315`

Es decir:

- en casi el `99.7%` de los casos, el retrieval ya mete en top 50 algun video de la categoria correcta

## Conclusión correcta

La conclusion no es "CF no sirve".

La conclusion correcta es esta:

- `CF` puro no acierta bien el `item exacto` del holdout
- pero si recupera muy bien el `tema correcto`
- como la home no necesita necesariamente el mismo item exacto, la evaluacion usada estaba castigando demasiado al retrieval

## Por que pasa esto

Porque este dataset parece capturar mejor una preferencia de tipo:

- "me gustan videos de esta categoria"

que una preferencia de tipo:

- "quiero otra vez exactamente este mismo canal o este mismo item"

Ademas, cada categoria tiene miles de videos. Entonces pedir que el retrieval meta el item exacto en top 50 es mucho mas duro de lo que parece, incluso si ha recuperado muchos candidatos razonables para el ranker.

## Que hay que hacer para mejorarlo

### 1. Mantener `User-based CF` e `Item-based CF`

Se mantienen porque siguen siendo utiles como fuentes de candidatos:

- `User-based CF` como retrieval principal
- `Item-based CF` como retrieval secundario, especialmente para continuidad y `watch_next`

### 2. No usar solo CF

La mejora correcta no es "mas de lo mismo" con CF memory-based.

La mejora correcta es hacer retrieval multi-source:

- `User-based CF`
- `Item-based CF`
- candidatos por categoria reciente
- candidatos por frescura y discovery
- candidatos por canales activos o seguidos cuando tenga sentido

### 3. Medir retrieval con metricas alineadas con producto

No basta con `hit exacto al item`.

Hay que medir tambien:

- `category_hit@K`
- recall de candidatos utiles para el ranker
- metrica final downstream despues del ranking

## Decisión recomendada

La decision recomendada para este proyecto es:

1. mantener `PairWise XGBoost` como ranker principal
2. mantener `User-based CF` e `Item-based CF` como retrieval base
3. ampliar retrieval a una version multi-source
4. dejar de juzgar retrieval solo con exact match al item holdout

Eso es lo mas defendible tecnicamente con los datos actuales.
