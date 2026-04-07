# Estructura del repositorio

## Raiz del proyecto

- `README.md`: resumen corto del pipeline de preparacion actual.
- `NOTAS_RECOMENDADOR.md`: notas estrategicas sobre como dividir el recomendador en `home`, `watch_next` y `busqueda`.
- `pyproject.toml`: dependencias principales (`pandas`, `numpy`, `pydantic`, `ruff`).
- `main.py`: punto de entrada minimo, hoy sin logica de negocio.
- `testdatos.py`: script pequeño para inspeccionar categorias del dataset bruto.

## Scripts Python clave

- `generate_datasets.py`
  - Lee el dataset bruto de YouTube.
  - Limpia campos conflictivos.
  - Genera `data/usuarios.csv` y `data/videos.csv`.

- `generar_que_pasa.py`
  - Toma el catalogo de videos.
  - Genera keywords sinteticas por categoria para el campo `que_pasa`.
  - Parece ser un paso de enriquecimiento intermedio.

- `prepare_recommendation_data.py`
  - Carga usuarios y videos ya limpios.
  - Construye features por video y por usuario.
  - Genera datasets finales para collaborative filtering y ranking.

- `recommendation_models.py`
  - No prepara CSV.
  - Define el modelo de dominio del recomendador con `Pydantic`.
  - Es la especificacion mas clara de como deberia comportarse el sistema final.

## Carpetas operativas

- `data/`
  - contiene el dataset original y las tablas limpias/enriquecidas
- `reco_output/`
  - contiene datasets derivados listos para analisis o entrenamiento

## Observacion sobre duplicados en raiz

Hay CSV grandes tambien en la raiz del repo (`usuarios.csv`, `videos.csv`, `videos_actualizado.csv`, `youtube recommendation dataset.csv`). A nivel de codigo, la ruta activa y consistente es `data/`, no la raiz. Por tanto, la documentacion toma `data/` como fuente canonical del proyecto.
