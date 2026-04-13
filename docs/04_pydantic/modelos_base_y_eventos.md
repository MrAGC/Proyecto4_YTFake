# Modelos base y eventos

## Base comun

### `ModeloDominio`

Hereda de `BaseModel` y fija reglas comunes:

- `extra="forbid"`: no acepta campos inesperados
- `str_strip_whitespace=True`: limpia espacios en strings
- `validate_assignment=True`: vuelve a validar si se reasignan campos

Es la base de todos los modelos de dominio reales.

## Funciones auxiliares de limpieza

### `_clean_tag_list`

Normaliza listas de tags:

- divide strings por coma
- pasa a minusculas
- quita espacios
- elimina duplicados

Se usa para categorias o parametros preferidos.

### `_clean_text_list`

Normaliza listas de texto libres sin forzar minusculas.

### `_clean_video_id_list`

Normaliza listas de ids de video:

- convierte a entero
- elimina ids no positivos
- elimina duplicados

## Enums del dominio

### `SuperficieRecomendacion`

Representa donde aparece una recomendacion:

- `busqueda`
- `home`
- `watch_next`
- `notificacion`

### `EstrategiaRetrieval`

Representa desde que logica salen los candidatos:

- `cosine_similarity`
- `sesion_usuario`
- `video_actual`

### `FuenteDescubrimiento`

Representa de donde llego el usuario al video:

- `busqueda`
- `recomendado_home`
- `recomendado_video`
- `notificacion`
- `directo`

## Modelos de evento

### `SenalesEngagement`

Bloque atomico de engagement tras entrar a un video.

Campos:

- `tiempo_visto_s`
- `porcentaje_visto`
- `like`
- `comentario`
- `suscripcion`

Campos derivados:

- `completion_flag`: `porcentaje_visto >= 0.9`
- `senal_positiva`: retencion alta o accion explicita
- `senal_negativa_clara`: rebote sin acciones positivas

Este modelo resume la idea base del proyecto: el comportamiento del usuario define labels implicitos.

### `ConsultaBusqueda`

Representa el flujo:

`texto -> retrieval -> ranking -> click`

Campos importantes:

- `texto`
- `timestamp`
- `retrieval_usado`
- `videos_recuperados`
- `videos_rankeados`
- `video_seleccionado_id`

Validaciones:

- normaliza listas de videos
- obliga a que el video clicado este dentro del ranking mostrado

Campos derivados:

- `tiene_click`
- `videos_no_elegidos`

Su papel principal es generar evidencia pairwise:

- video elegido > videos mostrados y no elegidos

### `ImpresionRecomendacion`

Representa una impresion ya rankeada en una superficie concreta.

Campos:

- `superficie`
- `timestamp`
- `retrieval_usado`
- `video_contexto_id`
- `videos_recuperados`
- `videos_rankeados`
- `video_clicado_id`

Validaciones clave:

- si el retrieval sale de `video_actual`, debe existir `video_contexto_id`
- si hay click, el video clicado debe pertenecer al ranking mostrado

Campos derivados:

- `hubo_interes`
- `videos_no_elegidos`

### `InteraccionVideo`

Es el evento final de consumo de un video.

Campos:

- `video_id`
- `timestamp`
- `fuente`
- `senales`

Campos derivados:

- `vino_de_recomendacion`
- `implicit_score`
- `positive_label`
- `negative_label`

Este modelo conecta origen del trafico con calidad de consumo.

### `ParPreferencia`

Es la unidad lista para ranking pairwise.

Campos:

- `user_id`
- `superficie`
- `timestamp`
- `ganador_video_id`
- `perdedor_video_id`
- `video_contexto_id`

No calcula scores; formaliza comparaciones del tipo:

- este video gano frente a este otro para este usuario y contexto
