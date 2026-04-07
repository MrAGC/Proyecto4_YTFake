link data:
https://www.kaggle.com/datasets/iitanshravan/youtube-recommendation-data-for-cleaning-and-ml?resource=download

Modelos:
 ·Retrieval - Collaborative Filtering (User-based CF)
 ·Ranking - PointWise (Dense NN) or PairWise + using extra information
 ·Metrics For Recommendations - HitRate@k y presition@k with Coverage = (items recommended ) / items total

Ruta pasos aplicación:
    Tomar el dataset y dividirlo para limpiar tabla de usuarios y tabla de videos.
    Crear datos sinteticos de titulos y elementos del video.
    Juntar los datos necesarios de cada tabla y hacer feature engenearin para obtener el dataset para recomendar.
    Generar el Collaborative Filtering para obtener las mejores recomendaciones.
    Con las recomendaciones generar el ranking con pointwise y pairwise para ber cual es mas efectivo y rankea mejor.
    Evaluar resultados de los rankings recomendados por HitRate, precision y Coverage.

