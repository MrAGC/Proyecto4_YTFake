from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_PATH = Path("reco_output_v2/ranking_dataset.csv")
OUTPUT_DIR = Path("reports/training_audit_v1")

LEAKAGE_HINT_PATTERNS = [
    "positive",
    "negative",
    "implicit",
    "watch_percent",
    "watch_time",
    "any_",
    "interaction_count",
    "click",
    "like",
    "comment",
    "subscription",
]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No encuentro el dataset: {path}")
    return pd.read_csv(path)


def suspicious_columns(columns: list[str]) -> list[str]:
    flagged: list[str] = []
    for column in columns:
        lower = column.lower()
        if any(pattern in lower for pattern in LEAKAGE_HINT_PATTERNS):
            flagged.append(column)
    return flagged


def build_markdown_report(summary: dict, missing_df: pd.DataFrame, label_corr: pd.DataFrame, high_corr: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Auditoria del dataset de entrenamiento")
    lines.append("")
    lines.append("## Resumen")
    lines.append("")
    lines.append(f"- rows: {summary['rows']}")
    lines.append(f"- columns: {summary['columns']}")
    lines.append(f"- positive_rate: {summary['positive_rate']}")
    lines.append(f"- unique_users: {summary['unique_users']}")
    lines.append(f"- unique_videos: {summary['unique_videos']}")
    lines.append(f"- numeric_columns: {summary['numeric_columns']}")
    lines.append(f"- categorical_or_text_columns: {summary['categorical_columns']}")
    lines.append("")
    lines.append("## Observaciones")
    lines.append("")
    lines.append("- `ranking_dataset.csv` es util para baseline, pero mezcla agregados offline con posible leakage.")
    lines.append("- La auditoria marca columnas sospechosas; no todas son fuga real, pero deben revisarse antes del entrenamiento definitivo.")
    lines.append("- El mayor riesgo actual es el desbalance fuerte de la label y el uso de señales demasiado cercanas al outcome observado.")
    lines.append("")
    lines.append("## Columnas sospechosas de leakage")
    lines.append("")
    for column in summary["suspicious_columns"]:
        lines.append(f"- `{column}`")
    lines.append("")
    lines.append("## Missing values mas importantes")
    lines.append("")
    if missing_df.empty:
        lines.append("- No hay missing values relevantes.")
    else:
        for _, row in missing_df.head(20).iterrows():
            lines.append(f"- `{row['column']}`: {row['missing_count']} ({row['missing_rate']})")
    lines.append("")
    lines.append("## Correlacion con la label")
    lines.append("")
    for _, row in label_corr.head(20).iterrows():
        lines.append(f"- `{row['feature']}`: {row['correlation']}")
    lines.append("")
    lines.append("## Pares de features muy correlacionadas")
    lines.append("")
    if high_corr.empty:
        lines.append("- No se detectaron pares por encima del umbral configurado.")
    else:
        for _, row in high_corr.head(20).iterrows():
            lines.append(f"- `{row['feature_a']}` <-> `{row['feature_b']}`: {row['correlation']}")
    lines.append("")
    lines.append("## Recomendaciones")
    lines.append("")
    lines.append("- Construir un dataset de entrenamiento balanceado con negativos sinteticos y observados.")
    lines.append("- Excluir las columnas mas cercanas al outcome observado del primer trainer serio.")
    lines.append("- Mantener split temporal explicito y no confiar en metricas de clasificacion si la label sigue muy desbalanceada.")
    return "\n".join(lines)


def main() -> None:
    df = load_dataset(DATASET_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    label_col = "ranking_label"
    if label_col not in df.columns:
        raise ValueError("No existe `ranking_label` en el dataset.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = [col for col in df.columns if col not in numeric_cols]

    summary = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "positive_rate": round(float(df[label_col].mean()), 6),
        "unique_users": int(df["user_id"].nunique()) if "user_id" in df.columns else 0,
        "unique_videos": int(df["video_id"].nunique()) if "video_id" in df.columns else 0,
        "numeric_columns": int(len(numeric_cols)),
        "categorical_columns": int(len(object_cols)),
        "suspicious_columns": suspicious_columns(df.columns.tolist()),
    }

    missing_df = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_rate": (df.isna().mean().round(6)).values,
        }
    ).sort_values(["missing_count", "column"], ascending=[False, True])
    missing_df = missing_df.loc[missing_df["missing_count"] > 0].reset_index(drop=True)

    label_corr = (
        df[numeric_cols]
        .corr(numeric_only=True)[label_col]
        .drop(labels=[label_col])
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .reset_index()
        .rename(columns={"index": "feature", label_col: "correlation"})
    )
    label_corr["correlation"] = label_corr["correlation"].round(6)

    corr_matrix = df[numeric_cols].corr(numeric_only=True).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = []
    for col in upper.columns:
        strong = upper[col][upper[col] >= 0.95]
        for idx, corr_value in strong.items():
            high_corr_pairs.append(
                {
                    "feature_a": idx,
                    "feature_b": col,
                    "correlation": round(float(corr_value), 6),
                }
            )
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values("correlation", ascending=False) if high_corr_pairs else pd.DataFrame(columns=["feature_a", "feature_b", "correlation"])

    cardinality_df = pd.DataFrame(
        {
            "column": object_cols,
            "unique_values": [int(df[col].nunique(dropna=False)) for col in object_cols],
        }
    ).sort_values("unique_values", ascending=False)

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    missing_df.to_csv(OUTPUT_DIR / "missing_values.csv", index=False)
    label_corr.to_csv(OUTPUT_DIR / "correlation_to_label.csv", index=False)
    high_corr_df.to_csv(OUTPUT_DIR / "high_feature_correlations.csv", index=False)
    cardinality_df.to_csv(OUTPUT_DIR / "categorical_cardinality.csv", index=False)

    markdown = build_markdown_report(summary, missing_df, label_corr, high_corr_df)
    (OUTPUT_DIR / "audit_report.md").write_text(markdown, encoding="utf-8")

    print("Training data audit completed")
    print(f"  rows: {summary['rows']}")
    print(f"  positive_rate: {summary['positive_rate']}")
    print(f"  suspicious_columns: {len(summary['suspicious_columns'])}")
    print(f"  output_dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
