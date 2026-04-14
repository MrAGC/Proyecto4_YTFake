import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Cargar el dataset
df = pd.read_csv(PROJECT_ROOT / "data" / "youtube recommendation dataset.csv")

# Contar cuántas categorías únicas hay
num_categorias = df["category"].nunique()

print("Número de categorías:", num_categorias)

# (Opcional) Mostrar cuáles son
print("Categorías:", df["category"].unique())
