from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "06_memoria" / "assets"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def setup_style() -> None:
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": "#111111",
            "axes.facecolor": "#181818",
            "axes.edgecolor": "#404040",
            "axes.labelcolor": "#f5f5f5",
            "xtick.color": "#e8e8e8",
            "ytick.color": "#e8e8e8",
            "text.color": "#f5f5f5",
            "axes.titleweight": "bold",
            "axes.titlesize": 18,
            "axes.labelsize": 11,
            "font.size": 11,
            "grid.color": "#323232",
        }
    )


def ensure_output_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def add_value_labels(ax, values, is_percent=True) -> None:
    top = max(values) if values else 0
    offset = top * 0.03 if top else 0.03
    for i, value in enumerate(values):
        label = f"{value * 100:.2f}%" if is_percent else f"{value:,.0f}"
        ax.text(i, value + offset, label, ha="center", va="bottom", fontsize=10)


def save(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def chart_ranking_comparison(summary: dict) -> None:
    metrics = ["ndcg_at_10", "roc_auc", "average_precision", "precision_at_1"]
    labels = ["NDCG@10", "ROC AUC", "Avg Precision", "Precision@1"]
    pointwise = [summary["pointwise_dense_nn"]["test"][m] for m in metrics]
    pairwise = [summary["pairwise_xgboost"]["test"][m] for m in metrics]

    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, pointwise, width, label="Pointwise Dense NN", color="#5B8FF9")
    ax.bar(x + width / 2, pairwise, width, label="Pairwise XGBoost", color="#FF5A36")

    ax.set_title("Comparativa del modelo de ranking")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False)

    for idx, value in enumerate(pointwise):
        ax.text(idx - width / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
    for idx, value in enumerate(pairwise):
        ax.text(idx + width / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)

    save(fig, "ranking_comparativa_modelos.png")


def chart_retrieval_metrics(metrics: dict) -> None:
    test = metrics["test"]
    labels = ["Hit@10", "Hit@50", "MRR@10", "NDCG@10", "Category Hit@50"]
    values = [
        test["hit@10"],
        test["hit@50"],
        test["mrr@10"],
        test["ndcg@10"],
        test["category_hit@50"],
    ]
    colors = ["#7B61FF", "#7B61FF", "#7B61FF", "#7B61FF", "#00C48C"]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(labels, values, color=colors, width=0.64)
    ax.set_title("Retrieval multi-source: item exacto vs acierto tematico")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.35)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    save(fig, "retrieval_item_vs_categoria.png")


def chart_offline_quality(quality: dict) -> None:
    ranking = quality["charts"]["ranking"]
    labels = [item["label"] for item in ranking]
    values = [item["value"] for item in ranking]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, values, color="#FFB020", width=0.65)
    ax.set_title("Calidad offline del recomendador")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.35)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    save(fig, "ranking_calidad_offline.png")


def chart_negative_sampling(quality: dict) -> None:
    negative = quality["charts"]["negative_sources"]
    labels = ["Positivos observados", "Negativos sinteticos", "Negativos observados"]
    values = [item["count"] for item in negative]
    colors = ["#00C48C", "#FF5A36", "#5B8FF9"]

    fig, ax = plt.subplots(figsize=(9, 7))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        wedgeprops={"width": 0.48, "edgecolor": "#111111"},
        textprops={"color": "#f5f5f5"},
    )
    for autotext in autotexts:
        autotext.set_color("#ffffff")
        autotext.set_fontsize(11)
    ax.set_title("Composicion del dataset de entrenamiento balanceado")
    save(fig, "negative_sampling_composicion.png")


def chart_candidate_pool(quality: dict) -> None:
    items = quality["candidate_pool"]["by_split"]
    labels = [item["split"].upper() for item in items]
    mean_values = [item["mean_candidates"] for item in items]
    median_values = [item["median_candidates"] for item in items]
    max_values = [item["max_candidates"] for item in items]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width, mean_values, width, label="Media", color="#5B8FF9")
    ax.bar(x, median_values, width, label="Mediana", color="#00C48C")
    ax.bar(x + width, max_values, width, label="Maximo", color="#FF5A36")

    ax.set_title("Tamano del pool de candidatos por split")
    ax.set_ylabel("Numero de candidatos")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False)

    for idx, value in enumerate(mean_values):
        ax.text(idx - width, value + 0.35, f"{value:.2f}", ha="center", fontsize=9)
    for idx, value in enumerate(median_values):
        ax.text(idx, value + 0.35, f"{value:.0f}", ha="center", fontsize=9)
    for idx, value in enumerate(max_values):
        ax.text(idx + width, value + 0.35, f"{value:.0f}", ha="center", fontsize=9)

    save(fig, "candidate_pool_por_split.png")


def chart_dataset_balance(audit: dict, quality: dict) -> None:
    original_positive = audit["positive_rate"]
    original_negative = 1 - original_positive
    balanced_positive = quality["negative_sampling"]["positive_rate"]
    balanced_negative = 1 - balanced_positive

    labels = ["Original", "Balanceado"]
    positive = [original_positive, balanced_positive]
    negative = [original_negative, balanced_negative]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(labels, positive, label="Positivos", color="#00C48C", width=0.6)
    ax.bar(labels, negative, bottom=positive, label="Negativos", color="#FF5A36", width=0.6)

    ax.set_title("Balanceo del dataset de ranking")
    ax.set_ylabel("Proporcion")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False)

    for idx, value in enumerate(positive):
        ax.text(idx, value / 2, f"{value * 100:.1f}%", ha="center", va="center", fontsize=11)
    for idx, value in enumerate(negative):
        ax.text(idx, positive[idx] + value / 2, f"{value * 100:.1f}%", ha="center", va="center", fontsize=11)

    save(fig, "dataset_balanceo_ranking.png")


def main() -> None:
    setup_style()
    ensure_output_dir()

    ranking_summary = load_json(ROOT / "models" / "ranker_compare_v1" / "comparison_summary.json")
    retrieval_metrics = load_json(ROOT / "models" / "retrieval_multisource_v1" / "metrics.json")
    quality_summary = load_json(ROOT / "reports" / "recommender_quality_v1" / "quality_summary.json")
    training_audit = load_json(ROOT / "reports" / "training_audit_v1" / "summary.json")

    chart_ranking_comparison(ranking_summary)
    chart_retrieval_metrics(retrieval_metrics)
    chart_offline_quality(quality_summary)
    chart_negative_sampling(quality_summary)
    chart_candidate_pool(quality_summary)
    chart_dataset_balance(training_audit, quality_summary)


if __name__ == "__main__":
    main()
