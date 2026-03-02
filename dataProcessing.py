import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from TitleGenerator import TitleGenerator

def enrich_youtube_data(file_path):
    # 1. CARGA DE DATOS
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: El archivo {file_path} no existe.")
        return None

    # --- PASO 2: CONVERSIÓN DE COLUMNAS NUMÉRICAS ---
    # Evita el error 'can't multiply sequence by non-int of type float'
    cols_to_fix = ['watch_percent', 'liked', 'commented', 'watch_time', 'video_duration']
    for col in cols_to_fix:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Rellenamos posibles nulos resultantes con 0
    df[cols_to_fix] = df[cols_to_fix].fillna(0)

    # --- PASO 3: GENERACIÓN DE TÍTULOS COHERENTES POR VIDEO_ID ---
    gen = TitleGenerator()
    unique_titles_pool = set()
    # Diccionario para recordar qué título le dimos a cada video_id
    video_id_to_title = {}

    def get_consistent_title(row):
        vid_id = row['video_id']
        category = row['category']
        
        # Si ya hemos generado un título para este video_id, lo devolvemos
        if vid_id in video_id_to_title:
            return video_id_to_title[vid_id]
        
        # Si es un video_id nuevo, generamos uno único
        new_title = gen.generate(category)
        attempts = 0
        while new_title in unique_titles_pool and attempts < 100:
            new_title = gen.generate(category)
            attempts += 1
        
        unique_titles_pool.add(new_title)
        video_id_to_title[vid_id] = new_title
        return new_title

    # Aplicamos la función fila por fila
    df['video_title'] = df.apply(get_consistent_title, axis=1)

    # --- PASO 4: LIMPIEZA TÉCNICA ---
    df = df[df['watch_time'] <= df['video_duration']].copy()

    # --- PASO 5: FEATURE ENGINEERING ---
    # Target: Engagement Score
    df['engagement_score'] = (df['watch_percent'] * 0.6) + (df['liked'] * 0.3) + (df['commented'] * 0.1)
    
    # --- PASO 6: CONVERSIÓN DE TIEMPO ROBUSTA ---
    # Maneja fechas normales y Unix timestamps (1743416259)
    # Intentamos convertir asumiendo segundos Unix primero
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce', unit='s', origin='unix')
    
    # Para las celdas que fallaron (eran texto), usamos el modo mixed
    mask_nulo = df['timestamp_dt'].isna()
    if mask_nulo.any():
        df.loc[mask_nulo, 'timestamp_dt'] = pd.to_datetime(df.loc[mask_nulo, 'timestamp'], errors='coerce', format='mixed')

    # Rellenar fallos con la hora actual y extraer la hora
    df['timestamp_dt'] = df['timestamp_dt'].fillna(pd.Timestamp.now())
    df['hour'] = df['timestamp_dt'].dt.hour

    # --- PASO 7: ENCODING ---
    le = LabelEncoder()
    for col in ['category', 'device', 'watch_time_of_day']:
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))

    return df

# EJECUCIÓN
if __name__ == "__main__":
    # Asegúrate de que el archivo esté en esta ruta
    file_path = 'data/youtube_recommendation_dataset.csv'
    df_final = enrich_youtube_data(file_path)
    
    if df_final is not None:
        df_final.to_csv('data/youtube_recommendation_processed_dataset.csv', index=False)
        print("✅ Dataset procesado con éxito.")
        print(f"Filas procesadas: {len(df_final)}")
        print(f"Títulos únicos generados: {df_final['video_title'].nunique()}")