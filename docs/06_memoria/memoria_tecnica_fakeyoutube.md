# 1. Introducción y justificación del proyecto

## 1.1 Contexto del proyecto y objetivos generales

El objetivo principal del proyecto es construir la base de datos, lógica de recomendación y capa de serving de un sistema tipo YouTube capaz de trabajar sobre varias superficies de recomendación:

- `Home`, donde se mezclan histórico, sesión actual y novedad.
- `Watch Next`, donde se prioriza continuidad temática y del vídeo actual.
- `Búsqueda`, donde se recuperan candidatos por texto y luego se ordenan.

No se ha planteado como una copia completa de YouTube a nivel de producto, sino como una demostración técnica de cómo diseñar un recomendador moderno con datos implícitos, reglas de negocio y una interfaz funcional para enseñar el resultado.

El enfoque técnico real del proyecto es un pipeline en dos etapas:

1. `Retrieval`: generar un conjunto reducido de candidatos plausibles.
2. `Ranking`: ordenar esos candidatos con un modelo más costoso y con más contexto.

La hipótesis central del trabajo es que, para este problema, las señales implícitas de consumo son más útiles que una valoración explícita estática. Por eso el sistema se apoya sobre todo en:

- porcentaje visto,
- tiempo visto,
- likes,
- comentarios,
- suscripciones,
- clics en recomendaciones,
- recencia de consumo,
- y afinidad reciente con categorías y canales.

## 1.2 Motivación y relevancia en IA y Big Data

El proyecto es relevante porque ataca uno de los problemas más claros en recomendación: el exceso de contenido disponible. En plataformas con miles o millones de ítems, el usuario no necesita solo encontrar algo "parecido", sino encontrar algo adecuado para ese momento, para ese contexto y para esa superficie concreta.

Desde el punto de vista de IA y Big Data, el problema tiene varias dificultades clásicas:

- La matriz usuario-ítem es extremadamente dispersa.
- La mayoría de pares `(usuario, vídeo)` no tienen interacción.
- La preferencia del usuario no es fija, sino dinámica.
- Lo que una persona sigue o le gustó hace meses no siempre coincide con lo que quiere ver hoy.
- La ausencia de interacción no significa automáticamente rechazo.

Esto obliga a trabajar con modelos híbridos, señales temporales y estrategias de evaluación que no se limiten a acertar un ítem exacto, sino que permitan medir calidad temática, ordenación y utilidad del sistema.

## 1.3 Estructura del documento

Este documento recorre el proyecto desde el dataset de origen hasta la versión actual del sistema:

1. diseño conceptual,
2. adquisición y preparación del dato,
3. construcción de features,
4. retrieval,
5. ranking,
6. integración en la app,
7. evaluación y negative sampling,
8. problemas encontrados,
9. resultados y futuras mejoras.

# 2. Planteamiento inicial y diseño del sistema

## 2.1 Arquitectura conceptual del recomendador

La arquitectura implementada es de dos fases desacopladas:

### Retrieval

Reduce el catálogo completo a un subconjunto razonable de candidatos. En el proyecto actual este retrieval combina:

- `User-based CF`,
- `Item-based CF`,
- categorías preferidas y recientes,
- canales con afinidad,
- y frescura de los vídeos.

### Ranking

Recibe esos candidatos y los ordena con un modelo más preciso. Esta etapa utiliza contexto cruzado de:

- usuario,
- vídeo,
- canal,
- frescura,
- afinidad reciente,
- y variable de superficie (`surface`).

## 2.2 Componentes clave del sistema

### Capa de datos

Se apoya en scripts de `scripts/data/`:

- [`scripts/data/generate_datasets.py`](C:/Programas/Proyecto4_YTFake/scripts/data/generate_datasets.py)
- [`scripts/data/prepare_recommendation_data.py`](C:/Programas/Proyecto4_YTFake/scripts/data/prepare_recommendation_data.py)
- [`scripts/data/build_training_dataset.py`](C:/Programas/Proyecto4_YTFake/scripts/data/build_training_dataset.py)
- [`scripts/data/update_video_thumbnails.py`](C:/Programas/Proyecto4_YTFake/scripts/data/update_video_thumbnails.py)

Esta capa no solo genera CSVs, sino que define la lógica real de transformación del proyecto. Aquí se decide cómo pasar de un dataset original plano a una estructura relacional y después a un conjunto de tablas útiles para entrenar modelos. En esta fase se separan usuarios, vídeos, canales, seguimientos y eventos; después se limpian inconsistencias, se agregan interacciones repetidas y se fabrican features ya pensadas para recomendación. En la práctica, esta parte ha sido tan importante como los propios modelos, porque un recomendador con datos mal agregados o mal etiquetados aprende patrones falsos.

### Capa de modelos

Está repartida entre:

- [`ml/train_retrieval_cf.py`](C:/Programas/Proyecto4_YTFake/ml/train_retrieval_cf.py)
- [`ml/train_retrieval_multisource.py`](C:/Programas/Proyecto4_YTFake/ml/train_retrieval_multisource.py)
- [`scripts/evaluation/compare_ranking_models.py`](C:/Programas/Proyecto4_YTFake/scripts/evaluation/compare_ranking_models.py)

En esta capa se entrenan y comparan los dos bloques principales del recomendador. Por un lado está retrieval, que reduce el catálogo a una selección manejable de candidatos; por otro lado está ranking, que decide el orden final. También aquí se generan métricas, artefactos del modelo y reportes de evaluación, de modo que la capa de modelos no se limita a “entrenar una vez”, sino que deja trazabilidad para justificar por qué se eligió un modelo frente a otro.

### Capa de serving y demo

La aplicación web se sirve con FastAPI y Jinja:

- [`app/webapp.py`](C:/Programas/Proyecto4_YTFake/app/webapp.py)
- [`app/reco_serving.py`](C:/Programas/Proyecto4_YTFake/app/reco_serving.py)

Esta parte es la que conecta el pipeline de datos y modelos con la experiencia final de usuario. Aquí no solo se renderiza la web, sino que se aplican reglas de negocio, persistencia local de sesión, filtros de vistos, control de novedad, lógica de búsqueda, creación de cuentas nuevas y subida de vídeos. Es decir, el serving actúa como la capa que traduce la salida del modelo en un comportamiento parecido al de una plataforma real.

Existe también una capa de modelos de dominio con Pydantic en:

- [`app/recommendation_models.py`](C:/Programas/Proyecto4_YTFake/app/recommendation_models.py)

Es importante aclarar que estos modelos Pydantic documentan y estructuran el dominio, pero la app actual no depende de ellos de forma central en cada endpoint. Por tanto, hablar de "validación estricta con Pydantic en toda la API" no sería exacto.

## 2.3 Estrategia de implementación seguida

El desarrollo real del proyecto ha seguido este orden:

1. separar el dataset original en tablas útiles,
2. limpiar y normalizar señales implícitas,
3. construir features por usuario, vídeo y canal,
4. crear un retrieval base con CF,
5. detectar sus limitaciones,
6. ampliar a retrieval multi-source,
7. comparar ranking pointwise vs pairwise,
8. integrar todo en una app funcional,
9. añadir persistencia local de sesión, cuentas nuevas, subida de vídeos y panel de control.

## 2.4 Herramientas y tecnologías utilizadas

### Backend

- Python 3
- FastAPI
- Uvicorn
- Jinja2

### Datos y ML

- Pandas
- NumPy
- Scikit-Learn
- XGBoost

Pandas y NumPy se usan en toda la cadena de transformación y entrenamiento. Scikit-Learn se utilizó para construir el baseline pointwise y para la parte de preprocesado clásico, mientras que XGBoost se eligió como modelo principal de ranking por su buen comportamiento en problemas tabulares, su soporte para entrenamiento en GPU y su ajuste natural a un problema de ordenación pairwise.

### Frontend

- HTML
- CSS
- templates Jinja

### Hardware / entrenamiento

El entrenamiento del modelo `pairwise_xgboost` se ha ejecutado con soporte `cuda`, según consta en:

- [`models/ranker_compare_v1/comparison_summary.json`](C:/Programas/Proyecto4_YTFake/models/ranker_compare_v1/comparison_summary.json)

Esto es relevante porque el modelo de ranking final no se entrenó como una simple prueba académica en CPU, sino como un experimento ya razonablemente cercano a una configuración realista para tablas grandes, aprovechando la GPU disponible para reducir tiempos de entrenamiento y permitir más árboles y más iteraciones.

# 3. Adquisición y preprocesamiento de datos

## 3.1 Fuente de datos

El proyecto parte de un dataset estático:

- `data/youtube recommendation dataset.csv`

Ese dataset se transforma a una estructura más útil mediante:

- [`scripts/data/generate_datasets.py`](C:/Programas/Proyecto4_YTFake/scripts/data/generate_datasets.py)

## 3.2 Datasets activos del proyecto

Los datasets realmente activos hoy no son los antiguos `videos_completo.csv` o similares, sino los que están en `data/` y `reco_output_v2/`.

### Tablas base activas

- `data/usuarios.csv`
- `data/usuarios_perfiles.csv`
- `data/canales.csv`
- `data/seguimientos_canales.csv`
- `data/videos.csv`

Estas tablas representan la estructura mínima del dominio. `usuarios.csv` y `usuarios_perfiles.csv` separan la identidad del usuario y sus rasgos agregados; `canales.csv` permite introducir la idea de creador y seguimiento; `seguimientos_canales.csv` modela la relación de follow; y `videos.csv` concentra el catálogo base con metadatos como categoría, duración, creador, título y recursos visuales para la demo.

### Tablas derivadas del pipeline

- `reco_output_v2/eventos_limpios.csv`
- `reco_output_v2/pair_interactions.csv`
- `reco_output_v2/cf_interactions.csv`
- `reco_output_v2/user_features.csv`
- `reco_output_v2/video_features.csv`
- `reco_output_v2/user_channel_features.csv`
- `reco_output_v2/ranking_dataset.csv`
- `reco_output_v2/training_dataset_balanced_v1.csv`

Estas tablas son las que ya alimentan directamente la parte de recomendación. `eventos_limpios.csv` es la base temporal limpia de interacciones; `pair_interactions.csv` resume afinidad por usuario-vídeo; `cf_interactions.csv` sirve para retrieval; `user_features.csv`, `video_features.csv` y `user_channel_features.csv` recogen el feature engineering final; `ranking_dataset.csv` es el dataset amplio antes del rebalanceo; y `training_dataset_balanced_v1.csv` es la versión preparada para comparar de forma más estable los modelos de ranking.

## 3.3 Limpieza y normalización

El preprocesado real que hace el proyecto incluye:

- corrección de `porcentaje_visto` cuando viene mal o fuera de rango,
- derivación del porcentaje visto desde `tiempo_visto_s` y duración si hace falta,
- truncamiento de tiempos vistos imposibles,
- eliminación práctica de valores negativos,
- normalización temporal,
- y consolidación de eventos repetidos por par usuario-vídeo.

Esto puede verificarse en:

- [`scripts/data/prepare_recommendation_data.py`](C:/Programas/Proyecto4_YTFake/scripts/data/prepare_recommendation_data.py)

## 3.4 Ingeniería de características

### 3.4.1 Features de usuario

Se generan en `user_features.csv` y recogen, entre otras:

- volumen total de interacciones,
- diversidad de vídeos, categorías y creadores,
- media de retención,
- tasas de like, comentario y suscripción,
- clics desde recomendaciones,
- categoría favorita histórica,
- categoría favorita reciente,
- canal y creador favoritos,
- dispositivo más usado,
- franja horaria habitual.

Estas variables intentan resumir el comportamiento global del usuario sin depender de una sola reproducción. La idea es que el modelo pueda distinguir entre alguien que consume casi siempre la misma temática y otro que mezcla muchas categorías, o entre un usuario muy activo y otro que apenas interactúa. También se intenta capturar si el patrón reciente coincide o no con el histórico, porque esa tensión entre gusto estable y cambio reciente es central en el proyecto.

### 3.4.2 Features de vídeo

Se generan en `video_features.csv` e incluyen:

- popularidad,
- engagement,
- CTR,
- frescura,
- novedad,
- presencia de metadatos,
- calidad del canal asociado.

Variables importantes:

- `video_popularity_log`
- `video_engagement_score`
- `video_freshness_score`
- `video_discovery_score`
- `is_recent_upload`
- `has_title_metadata`
- `has_keyword_metadata`

Con estas variables se intenta que el modelo no vea el vídeo solo como un identificador, sino como un objeto con señales de calidad, popularidad y actualidad. Esto es importante porque la recomendación no depende únicamente de “usuarios parecidos”, sino también de si el vídeo es reciente, si tiene un canal con buen rendimiento o si dispone de metadatos suficientes para poder recuperarlo bien tanto en home como en búsqueda.

### 3.4.3 Features usuario-canal

Este bloque es una de las partes más útiles del proyecto, porque mete señales recientes de afinidad por canal:

- `channel_recent_interactions_30d`
- `channel_recent_watch_percent_30d`
- `channel_recent_implicit_score_30d`
- `days_since_last_channel_watch`
- `stale_follow_flag`
- `channel_current_interest_score`

Estas variables permiten expresar algo importante para la lógica del proyecto: seguir un canal no debe pesar más que el consumo reciente real.

Ese punto era una decisión de diseño explícita del proyecto. Desde el principio se quiso evitar un sistema donde suscribirse a un canal secuestrase la recomendación durante demasiado tiempo. Por eso se construyeron variables que no solo miden la existencia del follow, sino su vigencia real. Si el usuario dejó de consumir ese canal hace tiempo, la variable `stale_follow_flag` y la distancia temporal desde el último consumo ayudan a que el modelo rebaje ese efecto.

### 3.4.4 Señal objetivo implícita

La señal principal no es una nota explícita, sino `implicit_score`, calculada con:

- retención,
- like,
- comentario,
- suscripción,
- clic de recomendación.

Además, se generan:

- `positive_label`
- `negative_label`

Esto permite construir datasets para retrieval y ranking sin necesitar estrellas ni valoraciones directas.

La decisión de usar señal implícita es coherente con el tipo de producto que se quería emular. En plataformas como YouTube la mayor parte de la información útil no viene de una nota explícita, sino del comportamiento: cuánto vio el usuario, si abandonó pronto, si hizo click en una recomendación, si dejó like o si acabó suscribiéndose. El proyecto traslada esa idea al dataset de clase para aproximarse a un escenario más realista que una simple clasificación binaria inventada.

## 3.5 Agregación por par usuario-vídeo

El proyecto agrupa reproducciones repetidas del mismo usuario sobre el mismo vídeo en `pair_interactions.csv`. Esto evita tratar cada evento aislado como si fuera una observación completamente independiente y permite crear un score consolidado de afinidad.

## 3.6 División temporal del dataset

La separación para retrieval en `cf_interactions.csv` es temporal:

- `train`: 653.712 filas
- `validation`: 99.065 filas
- `test`: 99.807 filas

Esto reduce el riesgo de leakage temporal al evaluar.

# 4. Modelado del sistema de candidatos: Retrieval

## 4.1 Propósito

La misión del retrieval es filtrar el catálogo antes de aplicar el ranker. No pretende decidir el orden final perfecto, sino recuperar un conjunto de vídeos plausibles.

## 4.2 Modelos implementados

### 4.2.1 Retrieval memory-based inicial

Se implementan:

- `User-based CF`
- `Item-based CF`
- `hybrid_cf`

Esto puede verificarse en:

- [`ml/train_retrieval_cf.py`](C:/Programas/Proyecto4_YTFake/ml/train_retrieval_cf.py)

El primer retrieval se planteó como un collaborative filtering clásico basado en memoria. La idea era construir una versión sencilla y comprensible del sistema antes de pasar a un retrieval más rico. En `User-based CF` se buscan usuarios con patrones parecidos de consumo para heredar vídeos afines. En `Item-based CF` se buscan vídeos que suelen coaparecer en historiales parecidos. El `hybrid_cf` mezcla ambas visiones para no depender solo de un vecindario por usuario o solo de co-consumo entre ítems.

Este enfoque tiene una ventaja clara: es fácil de implementar y de explicar. Permite comprobar rápidamente si el dataset contiene suficiente señal colaborativa. La desventaja también apareció pronto: en un entorno con mucha dispersión, poco histórico por usuario y sesiones cambiantes, el CF exacto recupera mal el ítem concreto que luego apareció en el holdout.

### 4.2.2 Retrieval multi-source actual

El retrieval que usa la app para `Home` ya no es solo CF puro. Se amplió a una mezcla de:

- señales CF,
- categoría favorita reciente,
- categoría favorita histórica,
- canales activos,
- frescura global.

Esto está implementado en:

- [`ml/train_retrieval_multisource.py`](C:/Programas/Proyecto4_YTFake/ml/train_retrieval_multisource.py)

El retrieval multi-source nace como respuesta directa a esa limitación. En lugar de confiar solo en la similitud colaborativa, combina varias fuentes de candidatos y luego las mezcla en un conjunto común. Algunas fuentes vienen del histórico global del usuario, otras de sus categorías recientes, otras de canales con actividad real y otras de la frescura del catálogo. Este diseño es mucho más adecuado para una home tipo YouTube, donde no siempre interesa “el mismo vídeo que habría visto”, sino un conjunto temáticamente bueno sobre el que luego el ranker pueda decidir.

## 4.3 Qué se vio al evaluar retrieval

El retrieval exacto por ítem da métricas bajas:

- `hit@10 = 0.000581`
- `hit@50 = 0.002004`

pero el retrieval por tema funciona muy bien:

- `category_hit@50 = 0.997565`

Esto significa que el sistema suele recuperar el tema correcto, aunque no necesariamente el mismo ítem exacto del holdout. Para una home tipo YouTube esto es mucho más defendible que juzgar solo el acierto exacto por vídeo.

## 4.4 Conclusión técnica sobre retrieval

La conclusión correcta no es que el retrieval "no sirve", sino que:

- el retrieval memory-based exacto es insuficiente como criterio final,
- el retrieval multi-source mejora mucho la recuperación temática,
- y el ranker debe entenderse como la segunda mitad obligatoria del sistema.

Como línea futura, sí tendría sentido explorar modelos de embeddings tipo `Two-Tower`, pero a día de hoy esa parte todavía no está implementada.

Dicho de otra manera, el retrieval del proyecto ya cumple una función útil, pero no se debe presentar como un recomendador terminado. Su papel es generar un buen pool de candidatos con sentido temático y suficiente variedad. La ordenación final y la adaptación fina al momento del usuario recaen en el ranker. Esta separación entre recuperación gruesa y ordenación fina es precisamente una de las ideas más importantes que conviene defender en la memoria.

# 5. Modelado del sistema de ranking

## 5.1 Propósito

Una vez obtenido el pool de candidatos, el ranking decide qué vídeos van arriba y cuáles quedan por debajo. Esta es la fase de precisión fina del sistema.

El proyecto utiliza un enfoque de modelo único con variable `surface`, de forma que el ranker puede reutilizarse en varias superficies con distinto contexto.

## 5.2 Modelo pointwise de referencia

Se entrenó un baseline con `MLPClassifier` de Scikit-Learn:

- enfoque pointwise,
- CPU,
- dataset balanceado,
- clasificación de cada candidato por separado.

El objetivo de este modelo no era ser el sistema final, sino servir como referencia seria para comparar. Antes de elegir un ranker pairwise se construyó un enfoque más clásico de clasificación binaria candidato a candidato. Para ello se tomó `training_dataset_balanced_v1.csv`, se separó en `train`, `valid` y `test` según el split ya preparado, y se aplicó un preprocesado distinto según el tipo de variable.

Las variables categóricas que entraron explícitamente en este baseline fueron:

- `surface`
- `user_favorite_category`
- `user_favorite_device`
- `user_favorite_time_slot`
- `user_recent_favorite_category`
- `video_category`

Estas variables se imputaron con el valor constante `unknown` y después se codificaron con `OrdinalEncoder`, permitiendo además categorías desconocidas en inferencia mediante `unknown_value = -1`. Las variables numéricas se imputaron con `0.0` y luego se escalaron con `StandardScaler`. Finalmente se concatenaron ambos bloques para construir la matriz de entrada del `MLPClassifier`.

La red densa usada fue:

- capas ocultas: `(256, 128, 64)`
- activación: `relu`
- optimizador: `adam`
- `batch_size = 4096`
- `learning_rate_init = 0.001`
- `max_iter = 20`
- `early_stopping = True`
- `validation_fraction = 0.1`
- `n_iter_no_change = 5`
- `random_state = 42`

El modelo se entrenó en CPU y terminó en `18` iteraciones efectivas. En total utilizó `64` features y `480018` filas de entrenamiento, según `comparison_summary.json`. Es importante entender que este modelo predice una probabilidad individual por candidato, sin modelar de manera explícita la relación de orden entre vídeos del mismo usuario.

Resultados en test:

- `NDCG@10 = 0.970518`
- `ROC_AUC = 0.826307`
- `Average Precision = 0.797232`
- `Precision@1 = 0.827005`

## 5.3 Modelo pairwise principal

El modelo principal del proyecto es:

- `XGBoost Ranker`
- objetivo `rank:pairwise`
- entrenamiento con `cuda`

Esto encaja mucho mejor con el problema real: no interesa tanto predecir una nota absoluta, sino aprender qué candidato debe ir por delante de otro en una lista.

El entrenamiento del modelo pairwise se hizo a partir del mismo dataset balanceado, pero con una preparación distinta. Primero se agruparon las filas por `user_id` y se descartaron usuarios con menos de dos candidatos, porque el aprendizaje pairwise necesita comparar elementos dentro del mismo grupo. Después se ordenaron las filas por `user_id` y `video_id` para construir los grupos de ranking que XGBoost usa internamente.

Las columnas categóricas se mantuvieron como `category` de pandas y el modelo se entrenó con `enable_categorical=True`, evitando un one-hot encoding clásico. Las variables numéricas se convirtieron a tipo numérico y los valores faltantes se rellenaron con `0.0`. Este enfoque encaja bien con XGBoost, que suele trabajar muy bien con datos tabulares heterogéneos y no necesita el mismo escalado que una red densa.

Los hiperparámetros principales del ranker fueron:

- `objective = rank:pairwise`
- `n_estimators = 400`
- `max_depth = 8`
- `learning_rate = 0.05`
- `subsample = 0.85`
- `colsample_bytree = 0.85`
- `reg_lambda = 1.0`
- `min_child_weight = 4`
- `tree_method = hist`
- `device = cuda`
- `enable_categorical = True`
- `eval_metric = [ndcg@10, map@10]`
- `early_stopping_rounds = 30`
- `random_state = 42`

En este caso el entrenamiento sí se hizo como un problema de ordenación real. Cada grupo corresponde a los candidatos de un mismo usuario, y el modelo aprende divisiones del espacio de features que favorecen que los positivos queden por encima de los negativos. El proceso se controló con validación y parada temprana; el mejor punto quedó en `best_iteration = 73`. El entrenamiento final usó `475609` filas, `64` features, `89108` grupos de entrenamiento y `23049` grupos de validación.

## 5.4 Métricas del modelo pairwise

Resultados en test:

- `NDCG@10 = 0.991004`
- `ROC_AUC = 0.977433`
- `Average Precision = 0.962763`
- `Precision@1 = 0.973418`

Frente al baseline pointwise, el modelo pairwise sale mejor en todos los indicadores principales que hoy usamos para comparar.

Además de las métricas finales, el proyecto también guardó la importancia por feature en `feature_importance.csv`. Eso permite abrir una línea de interpretación del modelo, algo especialmente útil en una memoria de clase. No se trata solo de decir “gana XGBoost”, sino de mostrar que el modelo aprende a partir de señales con sentido, como afinidad reciente por canal, recencia de consumo y contexto del usuario.

## 5.5 Sobre la interpretación de estas métricas

Estas cifras son muy altas, pero deben leerse con cuidado. El propio proyecto ya lo ha identificado:

- el dataset de ranking fue balanceado a `50/50`,
- el pool de candidatos por usuario en valid/test es pequeño,
- y por eso las métricas offline del ranker son buenas para demostrar ordenación, pero no equivalen todavía a una evaluación de producción con listas largas.

Esta limitación no invalida el modelo, pero sí obliga a explicarlo bien en la memoria.

La lectura correcta es la siguiente: el ranker parece muy sólido para ordenar el conjunto de candidatos que recibe, y por eso es razonable defender la elección de `Pairwise XGBoost` como modelo principal. Sin embargo, no sería honesto vender esas métricas como si ya fueran la medida final de un sistema de producción masivo. La memoria debería dejar claro que el modelo resuelve bien el problema experimental planteado, pero que una validación más realista exigiría pools más grandes y quizá candidatos más duros por usuario.

# 6. Integración y funcionamiento del sistema final

## 6.1 Flujo de trabajo completo

El flujo actual del sistema es:

1. el usuario entra en la app,
2. la app carga cookies o cuenta registrada,
3. `app/reco_serving.py` decide la superficie,
4. se generan candidatos con retrieval,
5. el ranker ordena,
6. se aplican filtros y reglas de negocio,
7. se devuelve la interfaz renderizada por Jinja.

## 6.2 Reglas de negocio importantes

En la capa de serving hay reglas que forman parte del comportamiento real del sistema:

- no volver a enseñar vídeos ya vistos,
- limitar el peso de la sesión para que no domine todo,
- usar follows como afinidad de canal, no como dominio total de la recomendación,
- mezclar novedad con señales personales,
- y preservar actividad local en sesión.

## 6.3 Persistencia de sesión y actividad local

La app guarda actividad local en:

- `data/local_session_events.csv`
- `data/local_search_events.csv`
- `data/local_engagement_events.csv`

Esto permite recalcular recomendaciones dentro de la demo sin reentrenar el modelo.

## 6.4 Búsqueda

La búsqueda ya no es un placeholder. En el estado actual:

- existe endpoint real de búsqueda,
- hay sugerencias en vivo,
- se recuperan resultados por índice textual y lógica fuzzy,
- y la interfaz de resultados ya está integrada en la app.

Por tanto, afirmar que "Search es solo un placeholder de UI" ya no sería correcto.

## 6.5 Cuentas y subida de vídeos

La app actual ya permite:

- iniciar sesión con usuarios del dataset,
- crear una cuenta nueva real para la demo,
- crear canal propio,
- subir vídeos desde Studio,
- y meter esos vídeos nuevos en el catálogo local como cold start.

Esto no existía al inicio del proyecto, pero sí forma parte del estado actual.

# 7. Evaluación, métricas y negative sampling

## 7.1 Evaluación offline

El proyecto ya distingue correctamente entre:

- evaluación de retrieval,
- evaluación de ranking,
- y evaluación de experiencia del recomendador.

Además, ahora existe un reporte específico:

- [`reports/recommender_quality_v1/quality_report.md`](C:/Programas/Proyecto4_YTFake/reports/recommender_quality_v1/quality_report.md)
- [`reports/recommender_quality_v1/quality_summary.json`](C:/Programas/Proyecto4_YTFake/reports/recommender_quality_v1/quality_summary.json)

## 7.2 Métricas relevantes para el proyecto

Las métricas más útiles, según el estado actual del sistema, son:

### Retrieval

- `Hit@K`
- `Recall@K`
- `MRR`
- `NDCG`
- `category_hit@K`

En retrieval estas métricas no cuentan exactamente la misma historia. `Hit@K` y `Recall@K` miden si el sistema recuperó el ítem relevante entre los primeros candidatos. `MRR` y `NDCG` premian además que aparezca arriba y no abajo. Pero en este proyecto fue especialmente importante introducir `category_hit@K`, porque una home como la de YouTube no necesita recuperar siempre el vídeo exacto del holdout para ser útil; muchas veces basta con recuperar bien el tema y dejar que el ranker termine el trabajo.

### Ranking

- `Precision@K`
- `NDCG@K`
- `MAP@K`
- `MRR@K`

Estas métricas sí son más adecuadas para defender el ranker, porque aquí ya importa la calidad del orden. `Precision@K` mide cuánto contenido relevante aparece entre los primeros elementos. `NDCG@K` valora además si lo mejor se coloca arriba del todo. `MAP@K` resume la calidad media de la lista y `MRR@K` refleja cuán pronto aparece el primer acierto realmente útil.

### Experiencia de recomendación

- `Coverage`
- `Diversity`
- `Novelty`
- `Seen repeat rate`

Estas métricas son importantes porque un recomendador no solo se juzga por acertar, sino por cómo se comporta como producto. `Coverage` indica si el sistema utiliza una parte amplia del catálogo o si siempre enseña lo mismo. `Diversity` permite controlar que no se colapse en un solo tema o canal. `Novelty` mide cuánta exposición se da a contenido menos visto o más nuevo. `Seen repeat rate` ayuda a verificar una regla de negocio clave del proyecto: no repetir vídeos ya vistos salvo casos muy concretos.

## 7.3 Estado actual de la evaluación offline

En la evaluación offline añadida recientemente, el ranker da:

- `Precision@10 = 0.558746`
- `HitRate@10 = 1.0`
- `Recall@10 = 1.0`
- `NDCG@10 = 0.992039`
- `MAP@10 = 0.989067`
- `MRR@10 = 0.989297`

Además:

- `Coverage@10 = 0.667047`
- `Diversity categorías@10 = 0.764362`
- `Diversity canales@10 = 0.99957`
- `Novelty@10 = 0.006805`
- `Seen repeat rate@10 = 0.033202`

## 7.4 Lectura correcta de estas métricas

Estas métricas son útiles para enseñar que el ranker ordena bien el pool evaluado, pero el propio proyecto detecta un límite claro:

- en `test`, la media de candidatos por usuario es `2.07`
- en `valid`, la media es `2.075`

Eso significa que la evaluación del ranking aún no es una simulación completa de producción. Para fortalecerla habría que evaluar con entre `20` y `100` candidatos plausibles por usuario.

## 7.5 Negative sampling

Este es uno de los puntos más importantes del proyecto y también uno de los más alineados con la teoría de recomendadores.

### Negativos observados

Se marcan cuando hay:

- retención baja,
- sin like,
- sin comentario,
- sin suscripción,
- sin clic positivo.

### Negativos sintéticos

En el dataset balanceado final se generan negativos plausibles con vídeos:

- no vistos por el usuario,
- preferiblemente en categorías afines,
- o en pools recientes.

Esto está implementado en:

- [`scripts/data/build_training_dataset.py`](C:/Programas/Proyecto4_YTFake/scripts/data/build_training_dataset.py)

Aquí conviene explicar bien la lógica, porque es una de las partes más defendibles del trabajo. En recomendación implícita casi nunca tenemos negativos puros. Que un usuario no haya visto un vídeo no significa automáticamente que no le guste; muchas veces simplemente no se le mostró o había otro candidato mejor. Por eso el proyecto combina dos ideas: negativos observados, cuando sí existe una señal de poco interés, y negativos sintéticos, cuando hace falta construir contraste para que el modelo aprenda a separar buenos candidatos de candidatos plausibles pero peores.

### Composición real del dataset balanceado

Según `training_dataset_balanced_v1_metadata.json`:

- filas totales: `600000`
- positivas: `300000`
- negativas: `300000`
- negativos observados: `42076`
- negativos sintéticos: `257924`

## 7.6 Qué falta todavía en negative sampling

Aunque el negative sampling actual es razonable, todavía falta separar explícitamente:

- `random negatives`
- `hard negatives`
- `retrieval hard negatives`
- `same-category hard negatives`
- `same-channel hard negatives`

Ahora mismo el proyecto ya aproxima negativos difíciles mediante categorías afines y recientes, pero todavía no los etiqueta de forma explícita como estrategia diferenciada de entrenamiento.

Esta es una mejora futura importante porque permitiría defender mejor el entrenamiento del ranker. No es lo mismo un negativo totalmente aleatorio que un vídeo de la misma categoría y del mismo contexto temporal que el usuario no eligió. Cuanto más “difícil” sea el negativo, más aprende el modelo a ordenar con criterio fino. Por eso, aunque la estrategia actual ya es útil, una siguiente versión debería distinguir formalmente entre niveles de dificultad de los negativos.

# 8. Desafíos encontrados y soluciones aplicadas

## 8.1 Sparsity y retrieval exacto

El primer gran problema fue comprobar que el retrieval exacto por ítem daba resultados muy pobres. La solución no fue abandonar retrieval, sino:

- diagnosticar el problema correctamente,
- añadir señales temáticas y de canal,
- y dejar claro que el retrieval se debe medir también por calidad temática.

## 8.2 Cold start

Otro problema fue el comportamiento de cuentas nuevas y vídeos nuevos.

Se resolvió con:

- arranque sin recomendaciones hasta que exista señal real,
- fallback por sesión o novedad en invitados,
- y entrada de vídeos nuevos como cold start en el catálogo local.

## 8.3 Sesgos del dataset de ranking

El `ranking_dataset.csv` original tenía un sesgo positivo muy fuerte:

- `positive_rate = 0.95297`

Esto se corrigió construyendo:

- `training_dataset_balanced_v1.csv`

con una tasa positiva de:

- `0.5`

Esto estabilizó la comparación entre modelos y permitió entrenar ranking de forma más razonable.

## 8.4 Miniaturas y metadatos

Los vídeos originales no traían una presentación útil para la app. Se resolvió con:

- títulos curados,
- palabras clave sintéticas,
- y un pipeline de miniaturas para la interfaz de demo.

## 8.5 Coherencia entre histórico y sesión

También hubo que ajustar el peso de:

- likes,
- suscripciones,
- historial pasado,
- y sesión actual.

La solución final fue dar más valor al comportamiento reciente real, pero sin dejar que unas pocas acciones destruyan toda la diversidad del feed.

# 9. Resultados, conclusiones y trabajo futuro

## 9.1 Resultados principales

El proyecto ha conseguido construir una demo funcional con:

- pipeline de datos completo,
- retrieval multi-source,
- comparación real de rankers,
- app web navegable,
- cuentas nuevas,
- subida de vídeos,
- búsqueda funcional,
- y panel de control con métricas y checks de salud.

## 9.2 Conclusiones técnicas

Las conclusiones más importantes del trabajo son:

1. El retrieval exacto por ítem no es suficiente para juzgar la calidad del sistema.
2. La recuperación temática es mucho más informativa para una home tipo YouTube.
3. El ranking pairwise funciona mejor que el pointwise en este contexto.
4. Las señales recientes y de afinidad por canal/categoría pesan más que los follows estáticos.
5. El negative sampling es una pieza central del sistema y condiciona mucho la calidad del ranker.

## 9.3 Limitaciones actuales

Las principales limitaciones reales del proyecto son:

- el retrieval todavía no usa embeddings densos,
- la evaluación del ranking tiene pocos candidatos por usuario,
- no existe evaluación online real con A/B testing,
- los hard negatives no están separados formalmente por tipo,
- y la búsqueda todavía no usa recuperación semántica vectorial.

## 9.4 Trabajo futuro

Las siguientes mejoras tendrían más sentido:

### Retrieval

- Two-Tower o DSSM para retrieval por embeddings.

### Search

- recuperación semántica con vectores sobre títulos, keywords y metadatos.

### Evaluación

- evaluación offline con pools más grandes,
- métricas por superficie,
- y A/B testing si se quisiera llevar a una demo con usuarios reales.

### Negative sampling

- distinguir y etiquetar negativos aleatorios, observados y hard negatives.

### Serving

- endurecer la integración entre la capa web y los modelos de dominio Pydantic.

# 10. Cierre

En conjunto, el proyecto ya no es solo un ejercicio de limpieza de CSVs o una interfaz visual. Es una base bastante completa para explicar cómo se construye un recomendador moderno:

- desde el dato bruto,
- pasando por features y labels implícitas,
- hasta llegar a retrieval, ranking, serving y evaluación.

La parte más sólida del proyecto hoy es la combinación entre:

- preparación del dato,
- reglas de negocio bien pensadas,
- y una demostración web que permite enseñar el comportamiento del recomendador de manera clara.
