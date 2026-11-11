"""Small HTTP client with an on-disk cache and polite rate limiting.

Cached responses make re-runs fast and make results reproducible within the cache window.
No personal data is sent: a generic User-Agent is used. Set CROSSREF_MAILTO / NCBI_API_KEY
as environment variables to join the Crossref polite pool / raise NCBI rate limits.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import requests

DATA_DIR = os.environ.get(
    "INSPECTSR_DATA_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"),
)
CACHE_PATH = os.path.join(DATA_DIR, "http_cache.sqlite")
USER_AGENT = "INSPECT-SR-assistant/0.2 (systematic review integrity screening; research prototype)"
DEFAULT_TTL = 24 * 3600

_lock = threading.Lock()
_last_call: dict[str, float] = {}
# minimum seconds between calls per host
_MIN_INTERVAL = {
    "eutils.ncbi.nlm.nih.gov": 0.11 if os.environ.get("NCBI_API_KEY") else 0.36,
    "api.crossref.org": 0.05,
    "api.openalex.org": 0.1,
    "clinicaltrials.gov": 0.1,
}


class SourceError(RuntimeError):
    pass


def _db():
    os.makedirs(DATA_DIR, exist_ok=True)
    con = sqlite3.connect(CACHE_PATH, timeout=30, check_same_thread=False)
    con.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, fetched REAL, body TEXT)")
    return con


def _throttle(host: str):
    gap = _MIN_INTERVAL.get(host, 0.05)
    with _lock:
        wait = _last_call.get(host, 0) + gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call[host] = time.monotonic()


def get(url: str, params: Optional[dict] = None, *, as_json: bool = True, ttl: int = DEFAULT_TTL,
        timeout: int = 30, retries: int = 3):
    """GET with cache. Returns parsed JSON (or text). Returns None on 404."""
    req = requests.Request("GET", url, params=params).prepare()
    key = req.url
    con = _db()
    try:
        row = con.execute("SELECT fetched, body FROM cache WHERE key=?", (key,)).fetchone()
        if row and time.time() - row[0] < ttl:
            body = row[1]
            if body == "__404__":
                return None
            return json.loads(body) if as_json else body
        host = requests.utils.urlparse(key).hostname or ""
        headers = {"User-Agent": USER_AGENT}
        last_err = None
        for attempt in range(retries):
            _throttle(host)
            try:
                r = requests.get(key, headers=headers, timeout=timeout)
            except requests.RequestException as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 404:
                con.execute("REPLACE INTO cache VALUES (?,?,?)", (key, time.time(), "__404__"))
                con.commit()
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = SourceError(f"{host} returned {r.status_code}")
                time.sleep(2.0 * (attempt + 1))
                continue
            if not r.ok:
                raise SourceError(f"{host} returned {r.status_code}")
            body = r.text
            con.execute("REPLACE INTO cache VALUES (?,?,?)", (key, time.time(), body))
            con.commit()
            return json.loads(body) if as_json else body
        raise SourceError(f"{host} unavailable: {last_err}")
    finally:
        con.close()


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def clear_cache():
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)
