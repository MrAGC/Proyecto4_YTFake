# Manual de instalación y uso

## 1. Objetivo de este manual

Este documento explica cómo instalar, arrancar y utilizar la aplicación del proyecto `YTFake`. Está pensado como manual práctico para que otra persona pueda ejecutar la demo localmente, probar las funcionalidades principales y entender qué partes del sistema ya están operativas.

No sustituye a la memoria técnica ni a la documentación de modelos. Su función es más directa: dejar claro qué se necesita, qué comandos hay que lanzar y cómo se usa la aplicación una vez abierta en el navegador.

## 2. Requisitos previos

Antes de arrancar el proyecto conviene comprobar que el entorno cumple estas condiciones:

- sistema operativo con Python disponible,
- Python `3.12` o superior,
- acceso a una terminal PowerShell o equivalente,
- dependencias instalables desde `pyproject.toml`,
- datasets ya presentes en `datasets_unificados_usados/`,
- y modelos ya entrenados dentro de `models/`.

El proyecto está preparado para ejecutarse en local y la app se sirve con FastAPI sobre `127.0.0.1:8000`.

## 3. Estructura mínima necesaria

Para que la aplicación funcione correctamente, el proyecto necesita una estructura mínima de carpetas y archivos.

### 3.1 Código

Las carpetas de código más importantes son:

- `app/`: rutas FastAPI, serving del recomendador y modelos de dominio.
- `ml/`: lógica de retrieval y utilidades del sistema de recomendación.
- `scripts/data/`: generación, limpieza y construcción de datasets.
- `scripts/training/`: scripts de entrenamiento.
- `scripts/evaluation/`: análisis y evaluación offline.
- `web/`: plantillas HTML y estilos de la interfaz.

### 3.2 Datos

La carpeta central de datos usada por la app y por el pipeline es:

- `datasets_unificados_usados/`

Dentro de esa carpeta deben existir, como mínimo, estos archivos:

- `usuarios.csv`
- `usuarios_perfiles.csv`
- `videos.csv`
- `canales.csv`
- `seguimientos_canales.csv`
- `local_session_events.csv`
- `local_search_events.csv`
- `local_engagement_events.csv`
- `user_features.csv`
- `video_features.csv`
- `user_channel_features.csv`
- `cf_interactions.csv`
- `ranking_dataset.csv`
- `training_dataset_balanced_v1.csv`
- `training_dataset_balanced_v1_metadata.json`

Los archivos `local_*` actúan como una base de datos ligera para la demo. Guardan actividad local de sesión, búsquedas y engagement sin necesidad de montar una base de datos relacional completa.

### 3.3 Modelos

La carpeta de modelos debe contener al menos los artefactos usados por la app:

- `models/ranker_compare_v1/`
- `models/retrieval_multisource_v1/`

Si faltan estos modelos, la aplicación puede arrancar, pero la parte de recomendación no funcionará como se espera o dará errores al intentar servir resultados.

## 4. Instalación del proyecto

## 4.1 Crear entorno virtual

Desde la raíz del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Esto crea un entorno virtual aislado para el proyecto. Es recomendable usarlo para no mezclar dependencias con otras aplicaciones.

## 4.2 Instalar dependencias

Con el entorno activado:

```powershell
pip install -e .
```

El proyecto toma sus dependencias desde `pyproject.toml`. Entre las más importantes están:

- `fastapi`
- `uvicorn`
- `jinja2`
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `pydantic`

La instalación en modo editable (`-e`) permite ejecutar el proyecto y modificar el código sin reinstalar el paquete después de cada cambio.

## 4.3 Requisito opcional para gráficas y análisis

Si se quieren generar gráficas para memoria, informes o presentación, hace falta disponer también de `matplotlib`. Si no está instalado en el entorno, los scripts de generación de imágenes no podrán crear los PNG.

## 5. Arranque de la aplicación

Con el entorno activo y situado en la raíz del proyecto:

```powershell
python main.py
```

El punto de entrada del proyecto es:

- [`main.py`](C:/Programas/Proyecto4_YTFake/main.py)

Ese archivo lanza Uvicorn y sirve la app definida en:

- [`app/webapp.py`](C:/Programas/Proyecto4_YTFake/app/webapp.py)

Una vez arrancada, la aplicación queda accesible en:

- [http://127.0.0.1:8000/login](http://127.0.0.1:8000/login)

## 6. Flujo de uso de la aplicación

## 6.1 Pantalla de entrada

La aplicación ofrece tres posibilidades principales:

- iniciar sesión con un usuario existente del dataset,
- crear una cuenta nueva para la demo,
- continuar como invitado.

### Usuario existente

Permite entrar directamente con un usuario ya presente en el dataset y probar recomendaciones sobre un histórico ya construido.

### Crear cuenta nueva

Genera un usuario nuevo real dentro de la demo. Esta opción sirve para probar el comportamiento de cold start, historial local, subida de vídeos y evolución de la home según la actividad nueva.

### Continuar como invitado

No crea una cuenta persistente completa, pero sí permite navegar y guardar señal local en el propio equipo mediante los CSV de sesión. Es útil para probar la interfaz rápidamente.

## 6.2 Comportamiento inicial de una cuenta nueva

En una cuenta recién creada no se fuerza una home falsa con categorías inventadas. El comportamiento esperado es que la aplicación pida primero señal real del usuario. En la práctica, esto significa que el usuario debe empezar buscando o viendo contenido para que el sistema tenga una base sobre la que personalizar.

Este punto es importante, porque es una decisión deliberada del proyecto: no recomendar por inventar, sino empezar a recomendar cuando exista alguna señal mínima de interés.

## 6.3 Home

La ruta principal de uso después de iniciar sesión es:

- `/home`

La home muestra listas personalizadas construidas a partir del retrieval, el ranking y las reglas de negocio del serving. Dependiendo del estado del usuario, pueden aparecer bloques como:

- recomendaciones principales,
- vídeos nuevos con sentido para ese perfil,
- continuidad por canales o temas con los que sigue conectado,
- y descubrimientos más abiertos.

La home evita, en lo posible, repetir vídeos ya vistos y mezcla histórico, sesión reciente y frescura.

## 6.4 Ver un vídeo

La reproducción de un vídeo se hace en:

- `/watch/{video_id}`

En esa pantalla se muestran:

- el vídeo actual,
- recomendaciones laterales,
- botones de like,
- estado de suscripción al canal,
- contador de comentarios o bloque de comentarios de demo.

Al entrar en vídeos y generar nuevas interacciones, la sesión local se actualiza y la app puede recalcular recomendaciones posteriores con más contexto.

## 6.5 Buscador

La búsqueda está disponible en:

- `/search`
- `/api/search/suggest`

El comportamiento esperado es:

- el usuario escribe una consulta,
- recibe sugerencias en vivo,
- abre una página de resultados,
- y desde allí puede entrar a vídeos relacionados con ese término.

La búsqueda no usa todavía un motor semántico vectorial completo, pero sí tiene lógica funcional de recuperación textual y sugerencias integradas en la interfaz.

## 6.6 Historial

La app dispone de pantalla de historial en:

- `/history`

Desde ahí se puede revisar el contenido visto por el usuario actual. Esta parte es importante porque el historial no es solo visual: también alimenta las señales que luego afectan al recomendador.

## 6.7 Panel de control

La ruta del panel es:

- `/control`

Este panel sirve para inspeccionar:

- señales del usuario actual,
- estado de la sesión,
- calidad de la recomendación mostrada,
- checks de salud del feed,
- y métricas offline ya calculadas del sistema.

No es un panel global completo de producción, sino una vista pensada para la demo y la explicación del comportamiento del recomendador para el usuario o sesión activos.

## 6.8 Perfil

La app incluye una vista de perfil:

- `/profile`

Aquí se agrupan datos del usuario dentro de la demo y accesos relacionados con su actividad y su cuenta.

## 6.9 Studio y subida de vídeos

Los usuarios registrados pueden acceder a:

- `/studio`

Desde esta pantalla pueden crear vídeos nuevos indicando los metadatos necesarios. La aplicación guarda esa información en los CSV activos y el vídeo pasa a formar parte del catálogo local, entrando como contenido nuevo o cold start.

Esto es útil para demostrar que el sistema no solo consume un catálogo fijo, sino que puede incorporar contenido nuevo dentro de la demo.

## 7. Qué guarda la app mientras se usa

Durante el uso local, la app persiste actividad en varios CSV. Los más importantes son:

- `datasets_unificados_usados/local_session_events.csv`
- `datasets_unificados_usados/local_search_events.csv`
- `datasets_unificados_usados/local_engagement_events.csv`

Esto permite que, aunque no haya una base de datos tradicional detrás, la sesión deje rastro suficiente para:

- actualizar recomendaciones,
- conservar parte del historial,
- recordar likes,
- recordar suscripciones,
- y registrar búsquedas.

## 8. Entrenamiento y regeneración de artefactos

## 8.1 Regenerar datasets de recomendación

Si se necesita reconstruir la parte de features y datasets derivados:

```powershell
python scripts\data\prepare_recommendation_data.py
python scripts\data\build_training_dataset.py
```

Esto recalcula tablas limpias, features y el dataset de entrenamiento balanceado.

## 8.2 Entrenar el ranker

Para entrenar el modelo de ranking principal:

```powershell
python scripts\training\train_recommender_ranker.py
```

La salida esperada se guarda en:

- `models/ranker_v1/ranker_model.json`
- `models/ranker_v1/metrics.json`
- `models/ranker_v1/feature_importance.csv`

## 8.3 Evaluar el sistema

Para generar evaluación y auditoría:

```powershell
python scripts\evaluation\evaluate_recommender_quality.py
python scripts\evaluation\analyze_training_data.py
python scripts\evaluation\analyze_retrieval_quality.py
```

Estas salidas son útiles para justificar resultados en documentación, paneles y presentación.

## 9. Rutas principales de la app

Las rutas más importantes actualmente son:

- `/login`
- `/register`
- `/guest-login`
- `/logout`
- `/home`
- `/watch/{video_id}`
- `/search`
- `/history`
- `/control`
- `/api/search/suggest`
- `/api/videos/{video_id}/like`
- `/api/channels/{channel_id}/subscribe`
- `/profile`
- `/studio`
- `/studio/videos`

## 10. Problemas habituales y comprobaciones

## 10.1 La app arranca pero no recomienda bien

Conviene revisar:

- que existan los modelos en `models/`,
- que los CSV de `datasets_unificados_usados/` estén presentes,
- que no falten `user_features.csv`, `video_features.csv` o `user_channel_features.csv`,
- y que la cuenta nueva tenga ya alguna señal de uso si se espera personalización real.

## 10.2 Una cuenta nueva no tiene recomendaciones al principio

Eso no es necesariamente un fallo. En el estado actual del proyecto es el comportamiento esperado: primero hace falta que la persona busque o consuma contenido para construir contexto.

## 10.3 Fallan scripts de gráficas o informes

Si ocurre, normalmente se debe a una de estas causas:

- falta `matplotlib`,
- falta alguno de los JSON de métricas,
- o el dataset balanceado no existe todavía.

## 10.4 Fallan scripts de entrenamiento

En ese caso hay que comprobar:

- versión de Python,
- dependencias instaladas,
- existencia del dataset balanceado,
- y disponibilidad de GPU si se pretende repetir el entrenamiento con `cuda`.

El proyecto puede seguir funcionando sin reentrenar en ese momento si ya existen los modelos previamente generados.

## 11. Recomendación de uso para una demo

Si se quiere enseñar el proyecto a otra persona, el flujo más recomendable es este:

1. abrir `/login`,
2. entrar con un usuario existente o crear uno nuevo,
3. mostrar la home,
4. hacer una búsqueda,
5. entrar a varios vídeos,
6. dar algún like o suscribirse a un canal,
7. volver a la home para ver el cambio,
8. abrir el panel `/control`,
9. y, si interesa, enseñar `Studio` creando un vídeo nuevo.

Este recorrido permite ver tanto la parte visual de la app como la lógica del recomendador y su adaptación al comportamiento del usuario.

## 12. Cierre

El proyecto ya puede instalarse y ejecutarse como una demo local funcional. Aunque no sea una plataforma completa de producción, sí integra las piezas más importantes de un sistema moderno de recomendación:

- datos estructurados,
- modelos de retrieval y ranking,
- serving web,
- persistencia ligera de sesión,
- búsqueda,
- panel de control,
- y subida de contenido nuevo.

Por eso este manual debe leerse como guía práctica de uso de una demo avanzada, no solo como instrucciones mínimas para abrir una web.
