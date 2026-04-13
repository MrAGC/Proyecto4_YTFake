from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

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


def build_query(title: str, category: str) -> str:
    normalized_title = " ".join(str(title).split())
    normalized_category = " ".join(str(category).split())
    return f"{normalized_title} {normalized_category} youtube thumbnail"


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


def update_dataset(limit: int | None, only_missing: bool) -> tuple[int, int]:
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

    videos.to_csv(VIDEOS_PATH, index=False)
    return attempted, updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Completa thumbnail_url en data/videos.csv")
    parser.add_argument("--limit", type=int, default=None, help="Numero maximo de filas a procesar")
    parser.add_argument("--all", action="store_true", help="Reprocesa tambien filas que ya tengan thumbnail_url")
    args = parser.parse_args()

    attempted, updated = update_dataset(limit=args.limit, only_missing=not args.all)
    print(f"Filas procesadas: {attempted}")
    print(f"Filas actualizadas: {updated}")
    print("Orden de busqueda: Google CSE -> Google scrape -> YouTube fallback")


if __name__ == "__main__":
    main()
