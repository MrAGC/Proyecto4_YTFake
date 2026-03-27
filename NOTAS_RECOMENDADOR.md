# Notas Sobre El Plan De Recomendacion

## Lo que pienso

El plan tiene sentido y esta bien dividido, pero hay una diferencia importante entre las tres partes del sistema:

- `Para ti` y `despues de ver un video` pueden arrancar muy bien con interacciones, historico y collaborative filtering.
- `busqueda` depende mucho mas de texto y metadatos, asi que ahi los titulos, keywords y descripciones son casi obligatorios.

## Sobre titulos y palabras clave

Si, lo tengo en cuenta.

Ahora mismo los datos de recomendacion ya se pueden preparar usando comportamiento del usuario:

- cuanto ve un video
- si da like
- si comenta
- si se suscribe
- si hace clic en recomendaciones

Pero faltan dos cosas importantes para que el sistema quede bien:

- `titulo`
- `metadatos/keywords`

Eso no bloquea el collaborative filtering principal, pero si mejora mucho:

- el buscador
- el item-based CF
- el ranking final
- el caso de videos nuevos con poco historial

## Como encaja cada recomendador

### 1. Para ti

Este es el recomendador principal.

Yo lo haria asi:

- candidate generation con `User-based CF`
- apoyo secundario con `Item-based CF`
- ranking final con modelo `pointwise` o `pairwise`

Aqui lo mas importante son las interacciones historicas de cada usuario.

## 2. Despues de ver un video

Este no deberia depender tanto de todo el historial largo del usuario.

Aqui tiene mas sentido usar:

- el video actual como contexto principal
- similitud entre videos
- categoria
- duracion
- patrones de co-vision
- titulos y keywords cuando esten listos

O sea: es parecido al principal, pero la entrada mas fuerte no es "quien eres", sino "que video estas viendo ahora".

## 3. Busqueda

Este es otro problema distinto.

Aqui el texto manda:

- coincidencia con titulo
- coincidencia con keywords
- coincidencia con metadatos
- popularidad o engagement como desempate

Si quereis usar transformers o embeddings, este es el sitio donde mas sentido tienen desde el principio.

## Orden que yo seguiria

Para un proyecto de clase, yo no intentaria construir las tres cosas a la vez desde cero.

Haria este orden:

1. dejar bien las tablas limpias de usuarios y videos
2. generar titulos y metadatos sinteticos para los videos
3. construir dataset de interacciones para CF
4. sacar candidatos con collaborative filtering
5. construir dataset de ranking
6. evaluar HitRate@K, Precision@K y Coverage
7. despues adaptar lo anterior a los tres escenarios

## Opinion tecnica

La clave es no mezclar problemas diferentes como si fueran uno solo:

- `Para ti` = recomendacion por comportamiento del usuario
- `Despues de ver` = recomendacion por contexto del video actual
- `Busqueda` = retrieval por texto y metadatos

Los tres pueden compartir parte de la infraestructura, pero no deberian entrenarse exactamente igual ni usar las mismas features con el mismo peso.

## Sobre transformers, RAG y esas cosas

Para este proyecto yo seria pragmatico:

- `Transformers/embeddings`: si, sobre todo para busqueda y similitud entre videos
- `RAG`: no lo veo como pieza principal del recomendador; solo tendria sentido si quereis enriquecer o generar metadatos

O sea:

- embeddings si
- CF si
- ranking si
- RAG solo si os ayuda a crear o ampliar informacion textual

## Riesgo principal

El mayor riesgo no es el modelo.

El mayor riesgo es que:

- los titulos/metadatos queden artificiales o poco consistentes
- los labels positivos y negativos no esten bien definidos
- el ranking tenga muy pocos negativos buenos para aprender

Si eso falla, el modelo parecera peor de lo que realmente podria ser.

## Conclusion

Mi opinion es esta:

- vais por buen camino
- la preparacion de datos de interaccion era el paso correcto
- los titulos y keywords aun faltan y son importantes
- no son tan necesarios para arrancar CF, pero si son muy importantes para busqueda y para mejorar ranking
- el proyecto deberia plantearse como `retrieval -> filtering -> ranking -> evaluacion`

En corto:

`Para ti` aprende de usuarios.
`Despues de ver` aprende del video actual.
`Busqueda` aprende del texto.
