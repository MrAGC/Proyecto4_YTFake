# Documentacion del proyecto

Esta carpeta organiza la documentacion por areas funcionales para que se pueda entender el proyecto sin tener que reconstruir toda la historia leyendo solo scripts y CSV.

## Mapa rapido

- `01_general/vision_general.md`: que problema intenta resolver el proyecto y en que punto esta.
- `01_general/estructura_del_repo.md`: mapa de archivos y carpetas importantes.
- `02_datos/historia_del_dato.md`: reconstruccion de lo que se ha ido haciendo con el dataset.
- `02_datos/datasets_y_columnas.md`: inventario de datasets y significado de sus columnas.
- `02_datos/limpieza_y_enriquecimiento.md`: reglas de limpieza y enriquecimiento observadas en el codigo.
- `03_recomendacion/pipeline_y_salidas.md`: pipeline actual para preparar datos del recomendador.
- `04_pydantic/README.md`: indice de modelos Pydantic.
- `04_pydantic/modelos_base_y_eventos.md`: enums, modelos base y eventos.
- `04_pydantic/modelos_de_negocio.md`: perfil de usuario, catalogo de videos y sesion de recomendacion.

## Idea principal

El proyecto no esta construido como una app final, sino como un entorno de preparacion de datos y diseno de dominio para un sistema de recomendacion estilo YouTube. La parte mas madura es la de datos:

1. partir del dataset bruto
2. limpiarlo y separarlo en `usuarios.csv` y `videos.csv`
3. enriquecer videos con metadatos sinteticos
4. preparar datasets derivados para collaborative filtering y ranking

La parte de modelos `Pydantic` define como deberia verse el sistema final de recomendacion, aunque esos modelos todavia no aparecen conectados a una API ni a un entrenamiento en produccion.

## Estado actual resumido

- El dataset fuente vive en `data/youtube recommendation dataset.csv`.
- El catalogo base de videos se genero en `data/videos.csv`.
- Se crearon dos variantes enriquecidas:
  - `data/videos_con_titulos.csv`
  - `data/videos_actualizado.csv`
- Tambien existe un dataset fusionado completo:
  - `data/videos_completo.csv`
- El pipeline de `prepare_recommendation_data.py` sigue leyendo `data/videos.csv`, asi que hoy la preparacion de features no aprovecha automaticamente el catalogo enriquecido completo salvo que se cambie la entrada.
