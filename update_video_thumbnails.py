from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from urllib.parse import quote_plus

import httpx
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
VIDEOS_PATH = BASE_DIR / "data" / "videos.csv"
GOOGLE_SEARCH_URL = "https://www.google.com/search"
GOOGLE_CSE_URL = "https://www.googleapis.com/customsearch/v1"
YOUTUBE_SEARCH_URL = "https://www.youtube.com/results"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/135.0.0.0 Safari/537.36"
    )
}

CATEGORY_THEMES = {
    "Comedy": ("ff3b30", "ffffff", "Comedy"),
    "Education": ("2563eb", "ffffff", "Education"),
    "Gaming": ("7c3aed", "ffffff", "Gaming"),
    "Lifestyle": ("db2777", "ffffff", "Lifestyle"),
    "Music": ("16a34a", "ffffff", "Music"),
    "News": ("ea580c", "ffffff", "News"),
    "Sports": ("0891b2", "ffffff", "Sports"),
    "Tech": ("0f172a", "ffffff", "Tech"),
}


def build_query(title: str, category: str) -> str:
    normalized_title = " ".join(str(title).split())
    normalized_category = " ".join(str(category).split())
    return f"{normalized_title} {normalized_category} youtube thumbnail"


def clean_text(value: object, fallback: str = "") -> str:
    if value is None or pd.isna(value):
        return fallback
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return fallback
    return text


def build_placeholder_thumbnail(row: pd.Series, provider: str) -> tuple[str, str]:
    video_id = clean_text(row.get("video_id"), "0")
    category = clean_text(row.get("category"), "Video")
    title = clean_text(row.get("titulo"), category)

    if provider == "semantic":
        background, foreground, label_category = CATEGORY_THEMES.get(category, ("111827", "ffffff", category))
        label = quote_plus(f"{label_category} | {title}"[:80])
        return f"https://placehold.co/640x360/{background}/{foreground}/png?text={label}", "semantic_text_thumbnail"

    if provider == "loremflickr":
        keyword = {
            "Comedy": "comedy,stage",
            "Education": "education,study",
            "Gaming": "gaming,computer",
            "Lifestyle": "lifestyle,people",
            "Music": "music,concert",
            "News": "news,city",
            "Sports": "sports,stadium",
            "Tech": "technology,coding",
        }.get(category, "video")
        return f"https://loremflickr.com/640/360/{keyword}?lock={quote_plus(video_id)}", "loremflickr_category_seed"

    return f"https://picsum.photos/seed/ytfake-{quote_plus(video_id)}/640/360", "picsum_seed"


def fetch_google_image_official(client: httpx.Client, query: str) -> str | None:
    api_key = os.getenv("GOOGLE_CSE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_CX")
    if not api_key or not cse_id:
        return None

    response = client.get(
        GOOGLE_CSE_URL,
        params={
            "key": api_key,
            "cx": cse_id,
            "searchType": "image",
            "num": 1,
            "safe": "off",
            "q": query,
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    items = payload.get("items") or []
    if not items:
        return None
    return str(items[0].get("link") or "").strip() or None


def fetch_google_image_scrape(client: httpx.Client, query: str) -> str | None:
    response = client.get(
        GOOGLE_SEARCH_URL,
        params={"tbm": "isch", "q": query, "hl": "es"},
        timeout=20,
        follow_redirects=True,
    )
    response.raise_for_status()
    html = response.text

    patterns = [
        r'"ou":"(https:[^"]+)"',
        r'"(https://encrypted-tbn0\.gstatic\.com/images\?q=tbn:[^"]+)"',
        r'imgurl=(https%3A%2F%2F[^&]+)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, html)
        if matches:
            candidate = matches[0]
            return candidate.replace("\\u003d", "=").replace("\\u0026", "&")
    return None


def fetch_youtube_thumbnail_fallback(client: httpx.Client, query: str) -> str | None:
    response = client.get(
        YOUTUBE_SEARCH_URL,
        params={"search_query": query},
        timeout=20,
        follow_redirects=True,
    )
    response.raise_for_status()
    video_ids = re.findall(r'"videoId":"([^"]+)"', response.text)
    if not video_ids:
        return None
    first_video_id = video_ids[0]
    return f"https://i.ytimg.com/vi/{first_video_id}/hqdefault.jpg"


def resolve_thumbnail_url(client: httpx.Client, query: str) -> tuple[str | None, str]:
    try:
        official_url = fetch_google_image_official(client, query)
        if official_url:
            return official_url, "google_cse"
    except Exception:
        pass

    try:
        scraped_url = fetch_google_image_scrape(client, query)
        if scraped_url:
            return scraped_url, "google_scrape"
    except Exception:
        pass

    try:
        youtube_url = fetch_youtube_thumbnail_fallback(client, query)
        if youtube_url:
            return youtube_url, "youtube_fallback"
    except Exception:
        pass

    return None, "not_found"


def update_dataset(
    limit: int | None,
    only_missing: bool,
    fast_placeholder: bool,
    placeholder_provider: str,
    save_every: int,
) -> tuple[int, int]:
    videos = pd.read_csv(VIDEOS_PATH, low_memory=False)
    if "thumbnail_url" not in videos.columns:
        videos["thumbnail_url"] = ""
    if "thumbnail_source" not in videos.columns:
        videos["thumbnail_source"] = ""

    target_mask = videos["thumbnail_url"].fillna("").eq("") if only_missing else pd.Series([True] * len(videos))
    target_indexes = videos.index[target_mask].tolist()
    if limit is not None:
        target_indexes = target_indexes[:limit]

    updated = 0
    attempted = 0
    if fast_placeholder:
        for row_index in target_indexes:
            row = videos.loc[row_index]
            url, source = build_placeholder_thumbnail(row, placeholder_provider)
            attempted += 1
            videos.at[row_index, "thumbnail_url"] = url
            videos.at[row_index, "thumbnail_source"] = source
            updated += 1

        videos.to_csv(VIDEOS_PATH, index=False)
        return attempted, updated

    with httpx.Client(headers=DEFAULT_HEADERS) as client:
        for row_index in target_indexes:
            row = videos.loc[row_index]
            query = build_query(row["titulo"], row["category"])
            url, source = resolve_thumbnail_url(client, query)
            attempted += 1
            if not url:
                continue
            videos.at[row_index, "thumbnail_url"] = url
            videos.at[row_index, "thumbnail_source"] = source
            updated += 1
            if save_every > 0 and attempted % save_every == 0:
                videos.to_csv(VIDEOS_PATH, index=False)
                print(f"Progreso guardado: {attempted} intentadas, {updated} actualizadas", flush=True)

    videos.to_csv(VIDEOS_PATH, index=False)
    return attempted, updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Completa thumbnail_url en data/videos.csv")
    parser.add_argument("--limit", type=int, default=None, help="Numero maximo de filas a procesar")
    parser.add_argument("--all", action="store_true", help="Reprocesa tambien filas que ya tengan thumbnail_url")
    parser.add_argument(
        "--fast-placeholder",
        action="store_true",
        help="Rellena URLs deterministas sin hacer scraping ni peticiones por fila. Es el modo recomendado para demo.",
    )
    parser.add_argument(
        "--placeholder-provider",
        choices=["semantic", "loremflickr", "picsum"],
        default="semantic",
        help="Proveedor usado con --fast-placeholder",
    )
    parser.add_argument("--save-every", type=int, default=25, help="Guarda progreso cada N aciertos en modo online")
    args = parser.parse_args()

    attempted, updated = update_dataset(
        limit=args.limit,
        only_missing=not args.all,
        fast_placeholder=args.fast_placeholder,
        placeholder_provider=args.placeholder_provider,
        save_every=args.save_every,
    )
    print(f"Filas procesadas: {attempted}")
    print(f"Filas actualizadas: {updated}")
    if args.fast_placeholder:
        print(f"Modo rapido: {args.placeholder_provider}")
    else:
        print("Orden de busqueda: Google CSE -> Google scrape -> YouTube fallback")


if __name__ == "__main__":
    main()
