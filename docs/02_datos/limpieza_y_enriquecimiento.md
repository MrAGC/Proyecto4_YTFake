# Limpieza y enriquecimiento de datos

## Limpieza observada en `scripts/data/generate_datasets.py`

### Normalizacion de `liked`

El dataset original mezcla valores como:

- `0`
- `1`
- `2`
- `no`
- `yes`
- nulos

La limpieza aplicada los reduce a binario:

- negativos y nulos -> `0`
- positivos -> `1`

## Normalizacion de `category`

Se corrigen variantes sucias y cambios de mayusculas o espacios, por ejemplo:

- `MUsic` -> `Music`
- `music` -> `Music`
- `gamingg` -> `Gaming`
- `Ed` -> `Education`
- `COMEDY` -> `Comedy`
- `Tech ` -> `Tech`

## Parseo de `timestamp`

El campo mezcla:

- fecha en texto
- epoch en segundos

La estrategia aplicada es:

1. intentar parseo mixto con `pandas`
2. para los fallos, reinterpretar como epoch

## Construccion de tabla de usuarios

La tabla `data/usuarios.csv` es casi una proyeccion limpia del dataset fuente:

- se renombran columnas a nombres en castellano
- se ordena por `user_id` y `timestamp`

## Construccion de tabla de videos

La tabla `data/videos.csv` agrega por `video_id`:

- volumen total
- likes y comentarios
- suscripciones generadas
- retencion media
- clicks desde recomendacion
- categoria mas frecuente

Tambien crea metricas derivadas:

- `like_rate`
- `comment_rate`
- `subscription_rate`
- `click_through_rate`

## Enriquecimiento de metadata textual

### Campo `titulo`

El repo ya contiene una version del catalogo con titulos:

- `data/videos_con_titulos.csv`

No aparece en el codigo un generador automatico de titulos dentro de este repo, asi que probablemente ese enriquecimiento vino de una fase externa o manual.

### Campo `que_pasa`

`generar_que_pasa.py` genera cinco keywords aleatorias por categoria desde listas predefinidas. Eso indica que:

- el campo es sintetico
- no describe cada video de forma unica
- sirve mas como metadata de apoyo que como anotacion fiel de contenido real

## Limpieza adicional en `scripts/data/prepare_recommendation_data.py`

### Sobre videos

Se fuerzan tipos numericos y se corrigen valores fuera de rango:

- duracion valida entre `1` y `14400` segundos
- fallback por mediana de duracion
- tasas truncadas al rango `[0, 1]`
- `log1p(total_views)` como feature de popularidad

Tambien se crea:

- `video_engagement_score`
- `has_title_metadata`
- `has_keyword_metadata`

## Sobre eventos de usuario

Se aplican estas reglas:

- flags binarias recortadas a `0/1`
- tiempo visto no negativo
- tiempo visto no superior a la duracion del video
- reconstruccion de `watch_percent_clean` cuando falta o es invalido

## Labels implicitos

### Positivo

Un evento se marca positivo si ocurre al menos una de estas condiciones:

- `watch_percent_clean >= 0.7`
- like
- comentario
- suscripcion
- click en recomendacion

### Negativo claro

Un evento se marca negativo si ocurre todo esto:

- `watch_percent_clean <= 0.2`
- no hay like
- no hay comentario
- no hay suscripcion
- no hay click

## Score implicito

La formula observada es:

- `0.55 * watch_percent_clean`
- `0.15 * like`
- `0.10 * comentario`
- `0.15 * suscripcion`
- `0.05 * clic_recomendacion`

Esto deja claro que el proyecto valora mas la retencion que las acciones binarias, pero combina ambas.
