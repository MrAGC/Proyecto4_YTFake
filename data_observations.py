import re

import pandas as pd

from config_title import OUTPUT_CSV_PATH


def analizar_titulos(path=OUTPUT_CSV_PATH):
    try:
        df = pd.read_csv(path)
        col = "titulo"

        print(f"--- Analisis de calidad: {path} ---")
        print(f"Total de registros: {len(df)}")

        series = df[col].fillna("").astype(str).str.strip()

        vacios = df[series == ""]
        basura = df[series.isin([":", "|", "()", "[]", "{}", "Titulo no disponible"])]
        duplicados = df[df.duplicated(subset=[col], keep=False) & (series != "")]
        muy_cortos = df[series.str.len() < 16]
        muy_largos = df[series.str.len() > 92]
        cierre_raro = df[series.str.contains(r"[:\-,|/]$", regex=True, na=False)]
        puntuacion_rara = df[series.str.contains(r"(?:\?\?|\!\!|\|\||::)", regex=True, na=False)]
        inicio_minuscula = df[series.str.match(r"^[a-záéíóúñ]", na=False)]
        palabras_repetidas = df[
            series.apply(
                lambda texto: bool(re.search(r"\b(\w+)(?: \1){2,}\b", texto, flags=re.IGNORECASE))
            )
        ]

        print(f"\nVacios detectados: {len(vacios)}")
        if not vacios.empty:
            print(vacios[["video_id", "category"]].head().to_string(index=False))

        print(f"\nTitulos basura: {len(basura)}")
        if not basura.empty:
            print(basura[["video_id", col]].head().to_string(index=False))

        print(f"\nTitulos repetidos: {len(duplicados)}")
        if not duplicados.empty:
            print(duplicados[col].value_counts().head(10).to_string())

        print(f"\nMuy cortos (<16): {len(muy_cortos)}")
        print(f"Muy largos (>92): {len(muy_largos)}")
        print(f"Cierre raro: {len(cierre_raro)}")
        print(f"Puntuacion rara: {len(puntuacion_rara)}")
        print(f"Inicio en minuscula: {len(inicio_minuscula)}")
        print(f"Palabras repetidas en bucle: {len(palabras_repetidas)}")

        if len(df) > 0:
            ratio_duplicados = len(duplicados) / len(df)
            print(f"\nRatio de duplicados: {ratio_duplicados:.2%}")
            print(f"Longitud media: {series.str.len().mean():.2f}")
            print(f"Palabras medias: {series.str.split().str.len().mean():.2f}")

        if all(
            len(dataset) == 0
            for dataset in [
                vacios,
                basura,
                duplicados,
                muy_cortos,
                muy_largos,
                cierre_raro,
                puntuacion_rara,
                inicio_minuscula,
                palabras_repetidas,
            ]
        ):
            print("\nTodo correcto. No se detectaron problemas evidentes de formato o unicidad.")

    except FileNotFoundError:
        print("Error: No se encontro el archivo de salida.")


if __name__ == "__main__":
    analizar_titulos()
