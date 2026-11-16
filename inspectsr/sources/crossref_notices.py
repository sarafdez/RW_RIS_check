"""Crossref's list of retraction and expression-of-concern notices, downloaded once and used offline.

Crossref holds both the Retraction Watch data (source "retraction-watch") and notices deposited by publishers
(source "publisher"). Publisher-deposited retractions include papers that are not in the Retraction Watch file,
so this list catches retractions Retraction Watch is missing, without per-study API calls.
"""
from __future__ import annotations

import csv
import glob
import gzip
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import requests

from ..normalize import iso, normalize_doi, normalize_title, strip_notice_prefix
from .http import DATA_DIR, USER_AGENT

API = "https://api.crossref.org/works"
LOCAL_CSV = os.path.join(DATA_DIR, "crossref_notices.csv")
TYPES = ["retraction", "partial_retraction", "withdrawal", "removal", "expression_of_concern"]
LABELS = {"retraction": "Retraction", "partial_retraction": "Partial retraction", "withdrawal": "Withdrawal",
          "removal": "Removal", "expression_of_concern": "Expression of concern"}
FIELDS = ["target_doi", "notice_doi", "type", "source", "date", "notice_title"]
MAX_AGE_DAYS = 7
# A dated, compressed copy shipped with the app (Crossref metadata is CC0), so the list is available right after a
# restart (e.g. on Streamlit Cloud) while a fresh copy downloads in the background.
SEED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "seed")


def latest_seed() -> Optional[tuple[str, datetime]]:
    files = sorted(glob.glob(os.path.join(SEED_DIR, "crossref_notices_*.csv.gz")))
    if not files:
        return None
    m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(files[-1]))
    return files[-1], datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)


def make_seed(src: str = LOCAL_CSV) -> str:
    os.makedirs(SEED_DIR, exist_ok=True)
    day = datetime.fromtimestamp(os.path.getmtime(src), tz=timezone.utc).strftime("%Y-%m-%d")
    dest = os.path.join(SEED_DIR, f"crossref_notices_{day}.csv.gz")
    with open(src, "rb") as fi, gzip.open(dest, "wb", compresslevel=9) as fo:
        fo.write(fi.read())
    for old in glob.glob(os.path.join(SEED_DIR, "crossref_notices_*.csv.gz")):
        if old != dest:
            os.remove(old)
    return dest


def _get(params: dict, tries: int = 5) -> dict:
    for attempt in range(tries):
        try:
            r = requests.get(API, params=params, headers={"User-Agent": USER_AGENT}, timeout=120)
        except requests.RequestException:
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r.json()
        wait = float(r.headers.get("Retry-After", 0) or 0)
        time.sleep(max(wait, 3 * (attempt + 1)))
    raise RuntimeError("Crossref did not respond")


def download(dest: str = LOCAL_CSV, progress: Optional[Callable[[float, str], None]] = None) -> str:
    """Page through every retraction/withdrawal/removal/expression-of-concern notice (cursor paging, 1 req/s)."""
    totals = {}
    for t in TYPES:
        totals[t] = _get({"filter": f"update-type:{t}", "rows": 0})["message"]["total-results"]
        time.sleep(1.05)
    grand = sum(totals.values()) or 1
    done = 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        for t in TYPES:
            cursor = "*"
            while True:
                data = _get({"filter": f"update-type:{t}", "rows": 1000, "cursor": cursor,
                             "select": "DOI,title,update-to"})["message"]
                items = data.get("items", [])
                if not items:
                    break
                for it in items:
                    notice = normalize_doi(it.get("DOI"))
                    title = (it.get("title") or [""])[0]
                    for u in it.get("update-to", []) or []:
                        ut = (u.get("type") or "").lower()
                        if ut not in TYPES:
                            continue
                        parts = (u.get("updated") or {}).get("date-parts") or [[None]]
                        w.writerow([normalize_doi(u.get("DOI")), notice, ut, u.get("source") or "publisher",
                                    iso(parts[0]) if parts[0] and parts[0][0] else "", title])
                done += len(items)
                if progress:
                    progress(min(done / grand, 1.0), f"Crossref notices: {done:,} of {grand:,}")
                cursor = data.get("next-cursor")
                if not cursor or len(items) < 1000:
                    break
                time.sleep(1.05)
    os.replace(tmp, dest)
    return dest


def is_fresh(path: str = LOCAL_CSV) -> bool:
    if os.path.exists(path):
        return (time.time() - os.path.getmtime(path)) / 86400 < MAX_AGE_DAYS
    seed = latest_seed()
    return bool(seed) and (datetime.now(timezone.utc) - seed[1]).days < MAX_AGE_DAYS


@dataclass
class CrossrefNotices:
    meta: dict
    by_doi: dict = field(default_factory=dict)     # DOI of the paper -> [notice dicts]
    by_title: dict = field(default_factory=dict)   # normalised title (from "Retraction: <title>") -> [notice dicts]

    def find(self, doi: Optional[str], title: Optional[str]) -> list[dict]:
        """Notices for a paper: by DOI; by exact title only when the study has no DOI."""
        if doi:
            return [dict(n, match="doi") for n in self.by_doi.get(doi, [])]
        t = normalize_title(title)
        if t and len(t) >= 30:
            return [dict(n, match="title_exact") for n in self.by_title.get(t, [])]
        return []


def load(path: str = LOCAL_CSV) -> Optional[CrossrefNotices]:
    when = None
    if not os.path.exists(path):
        seed = latest_seed()
        if not seed:
            return None
        path, when = seed
    by_doi, by_title = defaultdict(list), defaultdict(list)
    n = 0
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if not row["target_doi"]:
                continue
            note = {"notice_doi": row["notice_doi"], "type": row["type"], "label": LABELS.get(row["type"], row["type"]),
                    "source": row["source"], "date": row["date"] or None, "notice_title": row["notice_title"]}
            by_doi[row["target_doi"]].append(note)
            stripped = normalize_title(strip_notice_prefix(row["notice_title"]))
            if stripped and len(stripped) >= 30:
                by_title[stripped].append(note)
            n += 1
    meta = {"source": "Crossref retraction and expression-of-concern notices",
            "downloaded_on": (when.strftime("%Y-%m-%d") + " (copy shipped with the app)") if when else
                             datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "n_notices": n, "n_papers": len(by_doi)}
    return CrossrefNotices(meta=meta, by_doi=dict(by_doi), by_title=dict(by_title))
