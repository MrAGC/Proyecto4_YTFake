# Modelos de negocio

## `UsuarioPerfil`

Es el modelo persistente del usuario. No representa un evento puntual, sino memoria acumulada.

Campos principales:

- identidad y fechas:
  - `user_id`
  - `creado_en`
  - `actualizado_en`

- memoria de interes:
  - `historial_busquedas`
  - `videos_semilla_recientes`
  - `categorias_preferidas`
  - `parametros_preferidos`

- contadores de uso:
  - `sesiones_totales`
  - `sesiones_solo_busqueda`
  - `impresiones_recomendadas`
  - `clics_en_recomendados`
  - `likes_dados`
  - `suscripciones_generadas`
  - `tiempo_total_visto_s`

- control de producto:
  - `forzar_notificaciones`

Normalizaciones:

- historial de busquedas como lista de texto limpia
- listas de videos semilla sin ids invalidos
- tags y parametros en minusculas y sin duplicados

Validacion:

- `sesiones_solo_busqueda` no puede ser mayor que `sesiones_totales`

Campos derivados importantes:

- `es_cold_start`
- `ctr_recomendaciones`
- `usa_la_app_como_buscador`
- `debe_recibir_notificaciones`

Interpretacion:

Este modelo esta pensado para decidir estrategia de retrieval y comportamiento del producto, no solo para almacenar estadisticas.

## `VideoCatalogo`

Es el modelo del item servido por el recomendador.

Campos principales:

- identidad y metadata:
  - `video_id`
  - `titulo`
  - `category`
  - `duracion_s`
  - `parametros_contenido`

- engagement agregado:
  - `total_views`
  - `total_likes`
  - `like_rate`
  - `total_comments`
  - `comment_rate`
  - `suscriptores_ganados`
  - `subscription_rate`
  - `avg_watch_percent`
  - `avg_watch_time_s`
  - `veces_recomendado`
  - `total_clics`
  - `click_through_rate`

Normalizacion:

- `parametros_contenido` se limpia como lista de tags

Campos derivados:

- `listo_para_busqueda`
- `requiere_enriquecimiento_metadata`
- `texto_indexable`
- `engagement_score_base`

Interpretacion:

Es un modelo mixto entre:

- documento indexable para busqueda
- item para similitud entre videos
- item rankeable por engagement base

Ademas refleja una idea central del proyecto: para busqueda y cold-start, `titulo` y `parametros_contenido` son casi obligatorios.

## `SesionRecomendacion`

Es el modelo orquestador del dominio. Une consultas, impresiones e interacciones.

Campos principales:

- identidad y tiempo:
  - `session_id`
  - `user_id`
  - `inicio`
  - `fin`

- configuracion del flujo:
  - `feed_inicial_vacio`
  - `ranker_compartido`
  - `retrieval_busqueda`
  - `retrieval_home`
  - `retrieval_watch_next`

- eventos contenidos:
  - `consultas`
  - `recomendaciones`
  - `interacciones_video`

Validacion:

- `fin` no puede ser anterior a `inicio`

Campos derivados:

- `hubo_click_en_recomendados`
- `solo_uso_buscador`
- `pares_pairwise`
- `videos_semilla_para_siguiente_sesion`
- `implicit_score_total`

## Como se conectan los modelos

La relacion conceptual es esta:

1. El usuario entra con un `UsuarioPerfil`.
2. En una `SesionRecomendacion` realiza consultas o recibe impresiones.
3. Cada `ConsultaBusqueda` o `ImpresionRecomendacion` puede producir un click.
4. El consumo real se registra como `InteraccionVideo` con `SenalesEngagement`.
5. De los clicks se derivan `ParPreferencia`.
6. De las interacciones positivas salen semillas para la siguiente sesion.

## Valor arquitectonico

Aunque hoy el repo no implemente el servicio completo, esta capa `Pydantic` ya deja clara la arquitectura deseada:

- retrieval por contexto
- ranking compartido entre superficies
- aprendizaje implicito desde sesiones reales
- persistencia de memoria de usuario
