# Modelo unico de ranking para varias superficies

## Idea principal

El proyecto puede usar un unico modelo de ranking para varias superficies del producto, siempre que no se le pida hacer todo a la vez sin contexto.

La idea correcta no es:

- entrenar un modelo que genere directamente toda la home

La idea correcta es:

- entrenar un modelo que puntue candidatos
- cambiar segun el caso:
  - el conjunto de candidatos
  - el filtro previo
  - el contexto que se le pasa

Ese mismo ranker puede servir para:

- `home`
- `watch_next`
- `listas` o shelves tipo Netflix
- `search` como reranker final

## Como funciona

La arquitectura recomendada es:

`retrieval o filtro -> candidatos -> ranker unico -> filtros de serving -> reranking ligero -> top K`

El ranker recibe:

- features del usuario
- features del video candidato
- features del contexto
- superficie o tipo de recomendacion

Y devuelve:

- un score de relevancia

Luego la aplicacion usa ese score para ordenar los candidatos y construir la lista final.

## Que cambia en cada superficie

### 1. Home

Objetivo:

- recomendar contenido relevante, pero no completamente repetitivo

Contexto que se pasa:

- `surface = home`
- categorias recientes del usuario
- canales recientes del usuario
- creadores recientes del usuario
- follows activos
- hora o momento de sesion si hace falta
- posicion de sesion del dia si se quiere modelar primera apertura

Tipos de candidatos:

- candidatos por collaborative filtering
- candidatos por intereses recientes
- candidatos de canales activos
- candidatos nuevos o frescos
- candidatos exploratorios

### 2. Watch Next

Objetivo:

- continuidad sobre el video actual
- repetir mas el tipo de contenido

Contexto que se pasa:

- `surface = watch_next`
- `current_video_id`
- `current_channel_id`
- `current_category`
- `current_tags` o `que_pasa`
- senales del usuario si hacen falta

Tipos de candidatos:

- videos del mismo canal
- videos de la misma categoria
- videos parecidos por metadata
- videos que suelen verse despues de ese video
- item-item CF o session-based

### 3. Search

Objetivo:

- rerankear bien resultados relevantes a la query

Importante:

- el ranker no sustituye la busqueda inicial

Arquitectura:

1. primero retrieval textual o semantico
2. despues ranker final

Contexto que se pasa:

- `surface = search`
- `query_text`
- score del retrieval textual
- contexto del usuario

## Listas o shelves en la app

La app puede construir varias listas usando el mismo modelo.

Lo que cambia en cada lista es:

- el filtro de candidatos
- el contexto
- las reglas de negocio

### Lista principal: Para ti

Debe existir una lista principal sin filtro tematico duro.

Esa lista se construye asi:

- candidatos amplios
- contexto del usuario
- `surface = home`
- ranker compartido
- top K final

Esta lista representa lo mejor que el sistema cree que le gustara al usuario sin limitarlo a una sola categoria.

### Shelves tematicas

Ademas de `Para ti`, la app puede construir listas filtradas:

- `Tecnologia para ti`
- `Risas para ti`
- `Musica para ti`
- `Nuevos en tus temas`
- `Mas de canales que sigues`

Ejemplos:

#### Tecnologia para ti

- candidatos: videos de categoria `Tech`
- contexto: `surface = home`
- ordenar con el ranker
- devolver top 10

#### Risas para ti

- candidatos: videos de `Comedy`
- contexto: `surface = home`
- ordenar con el ranker
- devolver top 10

#### Musica para ti

- candidatos: videos de `Music`
- contexto: `surface = home`
- ordenar con el ranker
- devolver top 10

#### Nuevos en tus temas

- candidatos: videos recientes o cold-start
- filtro adicional: categorias o canales cercanos al usuario
- contexto: `surface = home`
- ordenar con el ranker
- devolver top 10

#### Siguiente video

- candidatos: parecidos al video actual
- contexto: `surface = watch_next`
- ordenar con el ranker
- devolver top K

## Que debe decidir el modelo y que debe decidir la capa de reglas

### Lo que debe decidir el modelo

El modelo debe aprender:

- afinidad usuario-video
- afinidad usuario-canal
- interes reciente
- continuidad con el video actual
- relevancia del candidato en ese contexto

### Lo que NO debe decidir el modelo directamente

No es buena idea usar el modelo para imponer reglas duras de producto.

Esas reglas deben vivir en otra capa.

### Lo que debe decidir la capa de reglas o serving

- no mostrar videos ya vistos
- no duplicar el mismo video en varias posiciones de la misma lista
- limitar exceso del mismo canal o tema
- aplicar filtros por shelf
- reservar huecos para novedad o exploracion si hace falta
- aplicar reglas de seguridad, calidad o negocio

## Videos vistos

Los videos ya vistos no deberian volver a mostrarse en condiciones normales.

Eso no se resuelve dentro del modelo.

Debe resolverse en la capa de serving con un filtro posterior al scoring.

## Videos mostrados pero no clicados

No conviene tratarlos todos como `no le gusta`.

Motivo:

- muchas veces el usuario no entra porque habia otro mejor
- no porque el video fuese malo

La forma correcta de tratarlos es como:

- senal negativa debil
- o preferencia relativa frente al video que si fue clicado

Eso encaja mejor con pairwise ranking que con un negativo duro absoluto.

## Cambios que mejoran el sistema de forma clara

Si se quiere que este enfoque funcione bien, estos cambios no son opcionales en la practica:

### 1. Feature obligatoria de superficie

El ranker debe recibir siempre una feature explicita que diga donde esta recomendando.

Ejemplos:

- `surface = home`
- `surface = watch_next`
- `surface = search`

Sin esto, el modelo mezcla objetivos distintos y aprende peor.

### 2. Retrieval distinto por superficie

No se debe usar el mismo retrieval para todo.

Minimo recomendado:

- `home`: retrieval por usuario, intereses recientes, follows activos y novedad
- `watch_next`: retrieval por video actual, canal actual, categoria actual y similitud
- `search`: retrieval textual o semantico primero, luego reranking

### 3. Diversidad en home y listas

Para `home` y shelves tipo Netflix, no basta con ordenar por score.

Hace falta una capa de diversidad o reranking final para evitar:

- demasiados videos del mismo canal
- demasiados videos casi iguales
- repeticion excesiva del mismo tema

`watch_next` necesita menos diversidad que `home`.

### 4. Recencia pesa mas que follow historico

Seguir un canal no debe dominar el resultado.

Debe pesar mas:

- lo visto en las ultimas sesiones
- los canales consumidos recientemente
- la ultima vez que el usuario vio ese canal

El follow debe quedar como senal de apoyo, no como regla dominante.

### 5. Senales negativas explicitas

No se debe entrenar solo con clicks, likes y watch time.

Hace falta modelar al menos:

- rebote rapido
- abandono temprano
- falta de interes reciente
- canal seguido pero ya no consumido
- impresion no clicada como senal debil o pairwise

Sin senales negativas, el sistema se vuelve demasiado repetitivo.

### 6. Search en dos etapas

En busqueda, el ranker no debe sustituir el retrieval inicial.

La forma correcta es:

1. recuperar candidatos por texto o embeddings
2. rerankear esos candidatos con el modelo

### 7. Metricas separadas por superficie

No se debe medir todo con una sola metrica global.

Minimo:

- `home`: relevancia + diversidad + frescura
- `watch_next`: continuidad + retencion
- `search`: calidad del matching de la query + reranking

## Riesgos si esto no se aplica

Si no se aplican estas reglas, el sistema puede fallar aunque use un buen modelo:

- `home` y `watch_next` se mezclan y empeoran ambos
- el sistema repite demasiado siempre el mismo contenido
- los follows viejos pesan demasiado
- los videos nuevos casi no aparecen
- el buscador falla porque el ranker intenta hacer de retrieval
- el usuario queda encerrado en contenido demasiado estrecho

## Decision recomendada para este proyecto

Usar:

- un ranker unico compartido
- collaborative filtering para retrieval
- filtros por categoria, canal, frescura o texto segun la lista
- `search` como retrieval textual + reranking
- `watch_next` con mas continuidad
- `home` con mas variedad controlada
- una lista principal `Para ti` sin filtro duro
- reglas fuera del modelo para excluir vistos y controlar repeticion
