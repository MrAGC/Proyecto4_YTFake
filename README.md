# YTFake — Plataforma experimental de recomendación de vídeos

**YTFake** es una aplicación web inspirada en una plataforma de vídeo tipo YouTube, creada como proyecto experimental para trabajar con sistemas de recomendación, ranking de contenido, generación de datasets y evaluación de modelos de Machine Learning.

El objetivo del proyecto no es replicar YouTube como producto real, sino construir un entorno controlado donde poder simular usuarios, vídeos, canales, interacciones y recomendaciones personalizadas de forma medible.

> Proyecto académico / experimental. No está afiliado a YouTube ni a Google.

---

## Qué incluye

- **Aplicación web con FastAPI y Jinja2** para navegar por una experiencia tipo plataforma de vídeos.
- **Sistema de recomendaciones personalizado** basado en señales de usuario, vídeo, canal e historial de interacción.
- **Ranker entrenable con XGBoost** para ordenar candidatos y mejorar la relevancia de los vídeos recomendados.
- **Retrieval multisource** para generar candidatos desde distintas fuentes antes del ranking final.
- **Búsqueda de vídeos** con sugerencias y normalización básica de texto.
- **Usuarios registrados y modo invitado**, con sesiones separadas.
- **Historial, perfil, likes, suscripciones y studio de creación** para simular comportamiento real dentro de la app.
- **Pipelines de datos, entrenamiento y evaluación** separados por carpetas.
- **Datasets unificados en CSV** para facilitar pruebas, entrenamiento y reproducción del sistema.

---

## Vista general del sistema

El proyecto está dividido en tres partes principales:

1. **Web app**  
   Interfaz de usuario, navegación, login, home, búsqueda, reproducción, historial, perfil y studio.

2. **Serving de recomendaciones**  
   Carga datasets, modelos entrenados e índices de búsqueda/retrieval para generar recomendaciones en tiempo de ejecución.

3. **Pipelines offline**  
   Scripts para preparar datos, entrenar modelos, analizar calidad y auditar métricas del recomendador.

---

## Tecnologías principales

- **Python 3.12+**
- **FastAPI**
- **Jinja2**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **XGBoost**
- **Uvicorn**
- **Ruff**

---

## Estructura del repositorio

```txt
.
├── app/                         # Aplicación FastAPI y serving del recomendador
├── ml/                          # Utilidades de retrieval y modelos auxiliares
├── scripts/
│   ├── data/                    # Preparación y construcción de datasets
│   ├── training/                # Entrenamiento del ranker/recomendador
│   └── evaluation/              # Evaluación, auditoría y análisis de calidad
├── web/
│   ├── templates/               # Vistas Jinja2
│   └── static/                  # CSS, JS y recursos estáticos
├── models/                      # Modelos entrenados usados por la app
├── reports/                     # Reportes y diagnósticos generados
├── datasets_unificados_usados/  # CSV principales usados por app y entrenamiento
├── main.py                      # Punto de entrada de la aplicación
└── pyproject.toml               # Configuración del proyecto y dependencias
```

---

## Instalación rápida

Desde la raíz del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

En Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Ejecutar la aplicación

```powershell
python main.py
```

Después abre en el navegador:

```txt
http://127.0.0.1:8000/login
```

La aplicación redirige inicialmente al login y permite entrar con usuarios existentes o crear una sesión de invitado.

---

## Funcionalidades de la app

### Home personalizada

La página principal muestra recomendaciones adaptadas al usuario o a la sesión actual. Para usuarios nuevos o invitados sin historial suficiente, la app guía al usuario hacia la búsqueda para generar primeras señales.

### Búsqueda

Permite buscar vídeos por texto, categorías o temas. El sistema incluye normalización básica, alias de categorías y sugerencias.

### Reproducción e interacción

Al abrir un vídeo se registra la visualización y se actualiza el estado de sesión. También se pueden simular acciones como likes y suscripciones.

### Historial y perfil

La app mantiene historial y datos agregados para representar mejor el comportamiento del usuario.

### Studio

Permite simular la creación de vídeos por parte de un usuario/creador, añadiendo nuevos elementos al catálogo y a las features necesarias para recomendación.

---

## Datasets

El proyecto centraliza los datos en:

```txt
datasets_unificados_usados/
```

Archivos esperados:

```txt
usuarios.csv
usuarios_perfiles.csv
videos.csv
canales.csv
seguimientos_canales.csv
local_session_events.csv
local_search_events.csv
local_engagement_events.csv
user_features.csv
video_features.csv
user_channel_features.csv
cf_interactions.csv
ranking_dataset.csv
training_dataset_balanced_v1.csv
training_dataset_balanced_v1_metadata.json
```

Esta carpeta contiene los CSV y metadata necesarios para ejecución, entrenamiento y evaluación. Los modelos entrenados se mantienen separados en `models/`.

---

## Preparar o regenerar datasets

```powershell
python scripts\data\prepare_recommendation_data.py
python scripts\data\build_training_dataset.py
```

Estos scripts reconstruyen features y datasets de ranking dentro de la carpeta unificada de datos.

---

## Entrenar el ranker

```powershell
python scripts\training\train_recommender_ranker.py
```

Salida esperada:

```txt
models/ranker_v1/ranker_model.json
models/ranker_v1/metrics.json
models/ranker_v1/feature_importance.csv
models/ranker_v1/training_config.json
```

El entrenamiento usa XGBoost y calcula métricas como AUC, average precision, log loss, precision@k y recall@k.

---

## Evaluación y auditoría

```powershell
python scripts\evaluation\evaluate_recommender_quality.py
python scripts\evaluation\analyze_training_data.py
python scripts\evaluation\analyze_retrieval_quality.py
```

Estos scripts ayudan a revisar la calidad del recomendador, detectar problemas en los datos y analizar el comportamiento del retrieval/ranking.

---

## Notas importantes

- El proyecto está pensado como entorno de experimentación, no como plataforma de producción.
- Los datos son simulados o preparados para entrenar y evaluar el sistema.
- La calidad del recomendador depende directamente de los CSV disponibles y de los modelos presentes en `models/`.
- La app necesita que los datasets y modelos esperados existan antes de ejecutarse correctamente.
- Algunas rutas escriben nuevos eventos o filas en CSV locales para simular evolución de sesión e interacción.

---

## Posibles mejoras futuras

- Añadir capturas de pantalla de la interfaz.
- Documentar el flujo exacto de generación de datasets.
- Separar configuración por entorno.
- Añadir tests automatizados para serving, búsqueda y ranking.
- Publicar métricas principales en una tabla comparativa.
- Incluir un diagrama visual del pipeline de recomendación.

---

## Autor

Proyecto desarrollado por **Álex Clariana** como práctica de aplicación web, datos y Machine Learning aplicado a sistemas de recomendación.
