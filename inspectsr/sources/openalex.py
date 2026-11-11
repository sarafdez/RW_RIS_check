"""OpenAlex: retraction flag, citation counts and author-disambiguated retraction look-ups."""
from __future__ import annotations

import os
from typing import Iterable

from ..normalize import normalize_doi
from .http import get

API = "https://api.openalex.org"
SOURCE = "OpenAlex"


def _base() -> dict:
    p = {}
    if os.environ.get("OPENALEX_API_KEY"):
        p["api_key"] = os.environ["OPENALEX_API_KEY"]
    return p


def _chunks(xs: list, n: int) -> Iterable[list]:
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def works_by_doi(dois: list[str]) -> dict[str, dict]:
    out = {}
    for chunk in _chunks(sorted(set(dois)), 40):
        data = get(f"{API}/works", {**_base(), "filter": "doi:" + "|".join(chunk), "per-page": 50,
                                    "select": "id,doi,is_retracted,cited_by_count,publication_date,type,authorships"})
        for w in (data or {}).get("results", []):
            d = normalize_doi(w.get("doi"))
            if d:
                out[d] = w
    return out


def authorship_ids(work: dict) -> list[dict]:
    res = []
    for a in work.get("authorships", []) or []:
        au = a.get("author") or {}
        if au.get("id"):
            res.append({"id": au["id"].rsplit("/", 1)[-1], "name": au.get("display_name"),
                        "position": a.get("author_position")})
    return res


def retracted_works_by_authors(author_ids: list[str]) -> dict[str, list[dict]]:
    """For OpenAlex author IDs, return {author_id: [retracted works]} using disambiguated authorship."""
    out: dict[str, list[dict]] = {a: [] for a in author_ids}
    for chunk in _chunks(sorted(set(author_ids)), 40):
        data = get(f"{API}/works", {**_base(), "filter": f"authorships.author.id:{'|'.join(chunk)},is_retracted:true",
                                    "per-page": 200, "select": "id,doi,title,publication_year,authorships"})
        for w in (data or {}).get("results", []):
            ids = {a["id"] for a in authorship_ids(w)}
            for aid in chunk:
                if aid in ids:
                    out[aid].append({"doi": normalize_doi(w.get("doi")), "title": w.get("title"),
                                     "year": w.get("publication_year"), "openalex": w.get("id")})
    return out
