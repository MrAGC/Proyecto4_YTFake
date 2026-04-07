import pandas as pd

# Cargar el dataset
df = pd.read_csv("data/actuales/youtube recommendation dataset.csv")

# Contar cuántas categorías únicas hay
num_categorias = df["category"].nunique()

print("Número de categorías:", num_categorias)

# (Opcional) Mostrar cuáles son
print("Categorías:", df["category"].unique())
