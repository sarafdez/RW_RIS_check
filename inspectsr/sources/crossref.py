"""Crossref REST API: metadata, citation matching and post-publication update notices."""
from __future__ import annotations

import os
from typing import Optional
from urllib.parse import quote

from ..normalize import normalize_doi, iso
from .http import get

API = "https://api.crossref.org"
SOURCE = "Crossref"

RETRACTION_TYPES = {"retraction", "withdrawal", "removal", "partial_retraction"}
CONCERN_TYPES = {"expression_of_concern"}
CORRECTION_TYPES = {"correction", "erratum", "corrigendum", "addendum", "clarification"}


def _params(extra: Optional[dict] = None) -> dict:
    p = dict(extra or {})
    if os.environ.get("CROSSREF_MAILTO"):
        p["mailto"] = os.environ["CROSSREF_MAILTO"]
    return p


def work(doi: str) -> Optional[dict]:
    data = get(f"{API}/works/{quote(doi, safe='/()<>;:')}", _params())
    return data.get("message") if data else None


def match_citation(query: str, rows: int = 5) -> list[dict]:
    data = get(f"{API}/works", _params({
        "query.bibliographic": query[:1000], "rows": rows,
        "select": "DOI,title,author,issued,container-title,type,score",
    }))
    return (data or {}).get("message", {}).get("items", [])


def summarize(msg: dict) -> dict:
    """Pull out the fields we use from a Crossref work record."""
    authors = []
    for a in msg.get("author", []) or []:
        if a.get("family"):
            authors.append(f"{a['family']}, {a.get('given', '')}".strip(", "))
        elif a.get("name"):
            authors.append(a["name"])
    issued = (msg.get("issued") or {}).get("date-parts", [[None]])[0]
    return {
        "doi": normalize_doi(msg.get("DOI")),
        "title": (msg.get("title") or [None])[0],
        "authors": authors,
        "year": issued[0] if issued and issued[0] else None,
        "pub_date": iso(issued) if issued and issued[0] else None,
        "journal": (msg.get("container-title") or [None])[0],
        "type": msg.get("type"),
        "updated_by": [
            {
                "type": (u.get("type") or "").lower(),
                "label": u.get("label") or u.get("type"),
                "doi": normalize_doi(u.get("DOI")),
                "date": iso((u.get("updated") or {}).get("date-parts", [[None]])[0])
                if (u.get("updated") or {}).get("date-parts") else None,
                "asserted_by": u.get("source", "publisher"),
            }
            for u in msg.get("updated-by", []) or []
        ],
        "is_notice_for": [normalize_doi(u.get("DOI")) for u in msg.get("update-to", []) or []],
    }
