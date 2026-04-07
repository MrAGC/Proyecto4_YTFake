# Modelos Pydantic

Toda la modelizacion `Pydantic` del proyecto esta concentrada en un solo archivo:

- `recommendation_models.py`

Eso significa que no hay varios modulos de esquemas repartidos por el repo. La ventaja es que la vision del dominio esta muy centralizada y se puede leer como especificacion funcional del sistema.

## Como esta organizada esta documentacion

- `modelos_base_y_eventos.md`
  - funciones de limpieza
  - modelo base
  - enums
  - eventos y etiquetas de interaccion

- `modelos_de_negocio.md`
  - perfil del usuario
  - catalogo de videos
  - sesion completa de recomendacion

## Lectura general

Los modelos Pydantic describen mas el sistema deseado que el pipeline de CSV actual. Son utiles para:

- validar entradas de negocio
- fijar contratos de datos
- formalizar reglas de coherencia
- expresar propiedades derivadas sin recalcularlas fuera del modelo

Se aprecia un diseño orientado a:

- sesiones de uso
- superficies de recomendacion
- retrieval + ranking
- aprendizaje implicito a partir del comportamiento
