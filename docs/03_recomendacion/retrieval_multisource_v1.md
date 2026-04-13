# Retrieval multi-source v1

## Que se ha cambiado

Antes, la retrieval se estaba midiendo principalmente con variantes de `Collaborative Filtering` memory-based:

- `User-based CF`
- `Item-based CF`
- `Hybrid CF`

Ahora se ha implementado una retrieval `v2` para `home` en:

- `train_retrieval_multisource.py`

Esta retrieval mezcla varias fuentes de candidatos:

- `User-based CF`
- `Item-based CF`
- videos de la categoria reciente del usuario
- videos de la categoria favorita del usuario
- videos de canales activos para ese usuario
- videos recientes como apoyo de frescura

## Por que esto tiene sentido

El diagnostico anterior ya habia demostrado dos cosas:

1. el `CF` puro no estaba acertando bien el item exacto holdout
2. pero si estaba recuperando muy bien el tema correcto

Y ademas el dataset mostraba este patron:

- el holdout comparte categoria con el historial del usuario en ~`56.5%`
- comparte canal en solo ~`0.15% - 0.18%`

Eso significa que la retrieval buena para `home` no puede depender solo de co-watch exacto entre items o usuarios. Tiene que mezclar `CF` con senales de tema, actividad reciente y discovery.

## Resultados

### Baseline CF puro

De `models/retrieval_cf_v1/metrics.json`:

#### Test

- mejor `hit@10` entre variantes CF: `0.00020`
- mejor `hit@50` entre variantes CF: `0.001022`
- `category_hit@50` del diagnostico: alrededor de `0.9972`

### Retrieval multi-source v1

De `models/retrieval_multisource_v1/metrics.json`:

#### Validation

- `hit@10`: `0.000262`
- `hit@50`: `0.001282`
- `category_hit@50`: `0.997709`
- `mrr@10`: `0.000071`
- `ndcg@10`: `0.000181`

#### Test

- `hit@10`: `0.000581`
- `hit@50`: `0.002004`
- `category_hit@50`: `0.997565`
- `mrr@10`: `0.000197`
- `ndcg@10`: `0.000453`

## Lectura correcta de la mejora

La mejora importante esta aqui:

- `hit@10` en test sube desde ~`0.00020` a `0.000581`
- `hit@50` en test sube desde ~`0.001022` a `0.002004`

Eso es aproximadamente:

- casi `3x` mejor en `hit@10`
- casi `2x` mejor en `hit@50`

Y al mismo tiempo:

- `category_hit@50` se mantiene altisimo

Es decir:

- no se ha perdido la capacidad de recuperar el tema correcto
- y si ha mejorado la probabilidad de recuperar el item exacto holdout

## Por que esta solucion es la mejor con los datos actuales

No es una mejora por intuicion. Sale de lo que muestran los datos:

- `CF` aporta senal colaborativa real
- la categoria reciente del usuario explica gran parte del comportamiento observado
- los canales activos aportan continuidad sin depender solo del follow historico
- la frescura ayuda a no bloquear la retrieval en contenido viejo

Con este dataset, esa mezcla es mejor que pedirle a `CF` puro que resuelva todo.

## Decision tecnica recomendada

Para el proyecto ahora mismo:

- `home`: usar `multi_source_home`
- `watch_next`: mantener `item_based_cf` como retrieval fuerte de continuidad
- `ranking`: mantener `pairwise_xgboost`

## Limite que sigue abierto

Aunque mejora, esta retrieval sigue evaluandose offline contra un holdout de item exacto. Asi que todavia falta una evaluacion mas cercana al producto real:

- candidate quality para el ranker
- evaluacion end-to-end retrieval + ranking
- medicion separada de `home` y `watch_next`

Pero como siguiente paso realista, esta v2 ya esta mejor planteada que el CF puro.
