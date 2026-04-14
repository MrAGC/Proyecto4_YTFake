# Historia del dato e interpretacion del trabajo realizado

## Reconstruccion del proceso

Por los nombres de archivos, el contenido de los scripts y las notas del repo, la evolucion del proyecto parece haber sido esta:

### 1. Dataset bruto de partida

Se parte de `data/youtube recommendation dataset.csv`, un dataset con eventos de consumo de videos. Ese dataset mezcla columnas de usuario, video, engagement y contexto de visualizacion.

## 2. Primera normalizacion

Con `scripts/data/generate_datasets.py` se hace una separacion en dos tablas:

- `data/usuarios.csv`: una fila por evento usuario-video
- `data/videos.csv`: una fila por video con metricas agregadas

Esa separacion ya enseña una intencion clara de modelado:

- tabla de hechos de interaccion
- tabla de catalogo o dimension de item

## 3. Limpieza de datos sucios

Antes de separar, el script corrige varios problemas del dataset original:

- `liked` mezcla numeros, texto y nulos
- `category` tiene variantes inconsistentes
- `timestamp` mezcla fechas en formato texto y epoch

Esto indica que primero se priorizo dejar el dato consistente antes de pensar en modelos.

## 4. Catalogo de videos incompleto a proposito

En `data/videos.csv` se crean dos campos vacios:

- `titulo`
- `que_pasa`

Eso sugiere que inicialmente el objetivo era preparar el catalogo estructural y dejar el enriquecimiento semantico para una fase posterior.

## 5. Enriquecimiento del catalogo

Despues aparecen dos ramas de enriquecimiento:

- `data/videos_con_titulos.csv`
  - rellena `titulo`
- `data/videos_actualizado.csv`
  - rellena `que_pasa`

La logica de `generar_que_pasa.py` confirma que al menos una parte de ese enriquecimiento es sintetica y guiada por categoria.

## 6. Fusion del catalogo enriquecido

Posteriormente se ha creado `data/videos_completo.csv`, que junta:

- el `titulo` de `videos_con_titulos.csv`
- el `que_pasa` de `videos_actualizado.csv`

Eso deja el catalogo en el estado mas completo del repositorio a nivel de metadatos textuales.

## 7. Preparacion para recomendacion

Con `scripts/data/prepare_recommendation_data.py` el proyecto da el salto de limpieza general a ML para recomendacion:

- eventos limpios
- labels implicitos
- features por usuario
- features por video
- interacciones agregadas
- splits temporales
- dataset de ranking

## 8. Formalizacion del dominio

`app/recommendation_models.py` va un paso mas alla del CSV y modela la aplicacion futura:

- sesiones
- consultas
- impresiones
- clicks
- preferencias pairwise
- perfil persistente del usuario
- catalogo de videos

## Lectura de madurez del proyecto

La parte mas consolidada es la de preparacion del dato. La parte de producto final esta en fase de modelado conceptual. En otras palabras:

- la tuberia de datos esta bastante avanzada
- la capa de servicio o inferencia online todavia no esta implementada

## Decision importante pendiente

El repo conserva versiones antiguas en `data/viejos/`, pero el script `scripts/data/prepare_recommendation_data.py` sigue leyendo `data/videos.csv`. Si se quiere que el pipeline use otros titulos o metadatos enriquecidos, la entrada activa debe actualizarse en `data/videos.csv`.
