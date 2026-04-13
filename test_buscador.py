from __future__ import annotations

import difflib
import re
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    import tkinter as tk
    from tkinter import ttk
except Exception:  # pragma: no cover - fallback for headless runs
    tk = None
    ttk = None

DATASET_CANDIDATES = (
    "videos_completos.csv",
    "videos_completo.csv",
    "videos_actualizado.csv",
    "videos.csv",
)

@dataclass(frozen=True)
class SearchConfig:
    min_query_chars: int = 2
    max_results: int = 12
    max_candidates_for_fuzzy: int = 2000
    pairwise_candidates: int = 250
    debounce_ms: int = 200
    chunk_size: int = 5000
    progress_every: int = 2000
    use_pairwise: bool = True


@dataclass
class SearchIndex:
    titles: list[str]
    normalized_titles: list[str]
    title_tokens: list[set[str]]
    title_trigrams: list[set[str]]
    views: list[int]
    token_index: dict[str, list[int]]
    trigram_index: dict[str, list[int]]


@dataclass(frozen=True)
class SearchResult:
    title: str
    score: float
    views: int


@dataclass(frozen=True)
class SearchResponse:
    results: list[SearchResult]
    total: int


DEFAULT_CONFIG = SearchConfig()


def find_dataset_path() -> Path:
    for name in DATASET_CANDIDATES:
        path = Path(name)
        if path.exists():
            return path
    raise FileNotFoundError(
        "No se encontro el dataset. Busca alguno de: "
        + ", ".join(DATASET_CANDIDATES)
    )


def normalize_text(text: object) -> str:
    raw = "" if text is None else str(text)
    raw = raw.lower().strip()
    raw = unicodedata.normalize("NFKD", raw)
    raw = "".join(ch for ch in raw if not unicodedata.combining(ch))
    raw = re.sub(r"[^a-z0-9\s]", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def tokenize(normalized: str) -> set[str]:
    return {token for token in normalized.split() if len(token) > 1}


def char_ngrams(normalized: str, n: int = 3) -> set[str]:
    condensed = normalized.replace(" ", "")
    if not condensed:
        return set()
    if len(condensed) <= n:
        return {condensed}
    return {condensed[i : i + n] for i in range(len(condensed) - n + 1)}


def _count_csv_rows(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            total = sum(1 for _ in handle) - 1
        return max(total, 0)
    except OSError:
        return 0


def _format_eta(seconds: float) -> str:
    if seconds <= 0 or seconds == float("inf"):
        return "?"

    minutes, secs = divmod(int(seconds + 0.5), 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _print_progress(label: str, current: int, total: int, start_time: float) -> None:
    if total <= 0:
        message = f"\r{label}: {current} filas"
        sys.stdout.write(message)
        sys.stdout.flush()
        return

    ratio = min(max(current / total, 0.0), 1.0)
    elapsed = time.perf_counter() - start_time
    eta = _format_eta((elapsed / ratio) - elapsed) if ratio > 0 else "?"
    percent = ratio * 100
    message = f"\r{label}: {current}/{total} ({percent:5.1f}%) - ETA {eta}"
    sys.stdout.write(message)
    sys.stdout.flush()


def build_index(
    normalized_titles: list[str],
    progress_label: str | None = None,
    progress_every: int = 2000,
) -> tuple[list[set[str]], dict[str, list[int]], list[set[str]], dict[str, list[int]]]:
    token_index: dict[str, list[int]] = defaultdict(list)
    title_tokens: list[set[str]] = []
    trigram_index: dict[str, list[int]] = defaultdict(list)
    title_trigrams: list[set[str]] = []
    total = len(normalized_titles)
    start_time = time.perf_counter() if progress_label else 0.0

    for idx, title_norm in enumerate(normalized_titles):
        tokens = tokenize(title_norm)
        trigrams = char_ngrams(title_norm)
        title_tokens.append(tokens)
        title_trigrams.append(trigrams)
        for token in tokens:
            token_index[token].append(idx)
        for tri in trigrams:
            trigram_index[tri].append(idx)
        display_idx = idx + 1
        if progress_label and (display_idx == total or display_idx % progress_every == 0):
            _print_progress(progress_label, display_idx, total, start_time)

    if progress_label:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return title_tokens, token_index, title_trigrams, trigram_index


def load_catalog(config: SearchConfig) -> SearchIndex:
    path = find_dataset_path()
    total_rows = _count_csv_rows(path)
    print(f"Cargando dataset '{path.name}' ({total_rows} filas)...")

    header = pd.read_csv(path, nrows=0)
    if "titulo" not in header.columns:
        raise ValueError("El dataset no tiene la columna 'titulo'.")

    has_views = "total_views" in header.columns
    usecols = ["titulo"] + (["total_views"] if has_views else [])

    titles: list[str] = []
    views: list[int] = []
    rows_read = 0
    start_time = time.perf_counter()
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=config.chunk_size):
        raw_count = len(chunk)
        chunk["titulo"] = chunk["titulo"].fillna("").astype(str)
        chunk = chunk[chunk["titulo"].str.strip() != ""]
        titles.extend(chunk["titulo"].tolist())
        if has_views:
            views.extend(chunk["total_views"].fillna(0).astype(int).tolist())
        rows_read += raw_count
        _print_progress("Leyendo", rows_read, total_rows, start_time)
    sys.stdout.write("\n")
    sys.stdout.flush()

    if not has_views:
        views = [0] * len(titles)

    normalized_titles: list[str] = []
    total_titles = len(titles)
    start_time = time.perf_counter()
    for idx, title in enumerate(titles, start=1):
        normalized_titles.append(normalize_text(title))
        if idx == total_titles or idx % config.progress_every == 0:
            _print_progress("Normalizando", idx, total_titles, start_time)
    sys.stdout.write("\n")
    sys.stdout.flush()

    title_tokens, token_index, title_trigrams, trigram_index = build_index(
        normalized_titles,
        progress_label="Indexando",
        progress_every=config.progress_every,
    )

    return SearchIndex(
        titles=titles,
        normalized_titles=normalized_titles,
        title_tokens=title_tokens,
        title_trigrams=title_trigrams,
        views=views,
        token_index=token_index,
        trigram_index=trigram_index,
    )


class SearchEngine:
    def __init__(self, index: SearchIndex, config: SearchConfig = DEFAULT_CONFIG) -> None:
        self.index = index
        self.config = config

    def search(self, query: str) -> SearchResponse:
        query_norm = normalize_text(query)
        if len(query_norm) < self.config.min_query_chars:
            return SearchResponse(results=[], total=0)

        query_tokens = set(query_norm.split())
        query_trigrams = char_ngrams(query_norm) if len(query_norm) >= 3 else set()
        candidates = self._collect_candidates(query_norm, query_tokens, query_trigrams)

        if not candidates:
            return SearchResponse(results=[], total=0)

        scores = {
            idx: self._score_candidate(idx, query_norm, query_tokens, query_trigrams)
            for idx in candidates
        }

        ordered_ids = sorted(
            candidates,
            key=lambda idx: (scores[idx], self.index.views[idx], self.index.titles[idx]),
            reverse=True,
        )

        unique_ids = self._dedupe_titles(ordered_ids)
        ranked_ids = (
            self._pairwise_rank(unique_ids, scores)
            if self.config.use_pairwise
            else unique_ids
        )

        results = [
            SearchResult(
                title=self.index.titles[idx],
                score=scores[idx],
                views=self.index.views[idx],
            )
            for idx in ranked_ids[: self.config.max_results]
        ]
        return SearchResponse(results=results, total=len(unique_ids))

    def _collect_candidates(
        self,
        query_norm: str,
        query_tokens: set[str],
        query_trigrams: set[str],
    ) -> list[int]:
        candidates: set[int] = set()

        for token in query_tokens:
            candidates.update(self.index.token_index.get(token, []))

        if len(candidates) < self.config.max_results and len(query_norm) >= 3:
            for idx, title_norm in enumerate(self.index.normalized_titles):
                if query_norm in title_norm:
                    candidates.add(idx)

        if len(candidates) < self.config.max_results and query_trigrams:
            trigram_counts: dict[int, int] = {}
            for tri in query_trigrams:
                for idx in self.index.trigram_index.get(tri, []):
                    trigram_counts[idx] = trigram_counts.get(idx, 0) + 1
            if trigram_counts:
                ranked = sorted(trigram_counts.items(), key=lambda item: item[1], reverse=True)
                for idx, _count in ranked[: self.config.max_candidates_for_fuzzy]:
                    candidates.add(idx)

        return list(candidates)

    def _score_candidate(
        self,
        idx: int,
        query_norm: str,
        query_tokens: set[str],
        query_trigrams: set[str],
    ) -> float:
        title_norm = self.index.normalized_titles[idx]
        score = 0.0
        if title_norm.startswith(query_norm):
            score += 3.0
        if query_norm in title_norm:
            score += 2.0
        if query_tokens:
            overlap = len(query_tokens & self.index.title_tokens[idx]) / len(query_tokens)
            score += overlap
        if query_trigrams:
            trigram_overlap = len(query_trigrams & self.index.title_trigrams[idx]) / len(query_trigrams)
            score += 1.5 * trigram_overlap
        ratio = difflib.SequenceMatcher(None, query_norm, title_norm).ratio()
        return score + ratio

    def _dedupe_titles(self, ordered_ids: list[int]) -> list[int]:
        unique_ids: list[int] = []
        seen_titles: set[str] = set()
        for idx in ordered_ids:
            title = self.index.titles[idx]
            if title in seen_titles:
                continue
            seen_titles.add(title)
            unique_ids.append(idx)
        return unique_ids

    def _pairwise_rank(self, candidate_ids: list[int], scores: dict[int, float]) -> list[int]:
        if len(candidate_ids) <= 1:
            return candidate_ids

        limit = min(self.config.pairwise_candidates, len(candidate_ids))
        head = candidate_ids[:limit]
        wins = {idx: 0 for idx in head}

        for i, left in enumerate(head):
            for right in head[i + 1 :]:
                winner = self._pairwise_winner(left, right, scores)
                wins[winner] += 1

        ranked_head = sorted(
            head,
            key=lambda idx: (
                wins[idx],
                scores[idx],
                self.index.views[idx],
                self.index.titles[idx].lower(),
            ),
            reverse=True,
        )

        return ranked_head + candidate_ids[limit:]

    def _pairwise_winner(self, left: int, right: int, scores: dict[int, float]) -> int:
        left_score = scores[left]
        right_score = scores[right]
        if left_score != right_score:
            return left if left_score > right_score else right

        left_views = self.index.views[left]
        right_views = self.index.views[right]
        if left_views != right_views:
            return left if left_views > right_views else right

        left_title = self.index.titles[left].lower()
        right_title = self.index.titles[right].lower()
        return left if left_title <= right_title else right


def create_search_engine(config: SearchConfig = DEFAULT_CONFIG) -> SearchEngine:
    index = load_catalog(config)
    return SearchEngine(index=index, config=config)


def run_cli(engine: SearchEngine) -> None:
    config = engine.config
    print("Buscador de YouTube (modo consola).")
    print(f"Escribe al menos {config.min_query_chars} caracteres. Enter vacio para salir.")

    while True:
        query = input("Buscar: ").strip()
        if not query:
            break
        response = engine.search(query)
        if not response.results:
            print("Sin resultados.\n")
            continue
        print(f"Mostrando {len(response.results)} de {response.total} resultados:")
        for result in response.results:
            print(f"- {result.title}")
        print("")


def run_gui(engine: SearchEngine) -> None:
    config = engine.config
    root = tk.Tk()
    root.title("Buscador de YouTube por titulo")
    root.geometry("900x520")

    main_frame = ttk.Frame(root, padding=16)
    main_frame.pack(fill="both", expand=True)

    header = ttk.Label(main_frame, text="Busca videos por titulo")
    header.pack(anchor="w")

    query_var = tk.StringVar()
    entry = ttk.Entry(main_frame, textvariable=query_var, font=("Segoe UI", 12))
    entry.pack(fill="x", pady=(6, 12))

    results_frame = ttk.Frame(main_frame)
    results_frame.pack(fill="both", expand=True)

    scrollbar = ttk.Scrollbar(results_frame, orient="vertical")
    listbox = tk.Listbox(
        results_frame,
        height=16,
        yscrollcommand=scrollbar.set,
        font=("Segoe UI", 11),
    )
    scrollbar.config(command=listbox.yview)

    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    status_var = tk.StringVar(value=f"Escribe al menos {config.min_query_chars} caracteres.")
    status = ttk.Label(main_frame, textvariable=status_var)
    status.pack(anchor="w", pady=(10, 0))

    pending_job = {"id": None}

    def update_results() -> None:
        query = query_var.get()
        listbox.delete(0, tk.END)

        response = engine.search(query)

        if len(normalize_text(query)) < config.min_query_chars:
            status_var.set(f"Escribe al menos {config.min_query_chars} caracteres.")
            return

        if not response.results:
            status_var.set("Sin resultados.")
            return

        status_var.set(
            f"Mostrando {len(response.results)} de {response.total} resultados."
        )
        for result in response.results:
            listbox.insert(tk.END, result.title)

    def schedule_update(*_args: object) -> None:
        if pending_job["id"] is not None:
            root.after_cancel(pending_job["id"])
        pending_job["id"] = root.after(config.debounce_ms, update_results)

    def apply_selection(*_args: object) -> None:
        selection = listbox.curselection()
        if not selection:
            return
        title = listbox.get(selection[0])
        query_var.set(title)
        entry.icursor(tk.END)

    query_var.trace_add("write", schedule_update)
    entry.bind("<Return>", lambda _event: update_results())
    listbox.bind("<<ListboxSelect>>", apply_selection)
    listbox.bind("<Double-Button-1>", apply_selection)
    listbox.bind("<Return>", apply_selection)
    entry.focus()

    root.mainloop()


def main() -> None:
    config = DEFAULT_CONFIG
    engine = create_search_engine(config)

    if tk is None or ttk is None:
        run_cli(engine)
    else:
        run_gui(engine)


if __name__ == "__main__":
    main()
