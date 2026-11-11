"""Retraction Watch database (open via Crossref): loading, indexing and look-ups."""
from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

from ..normalize import author_keys, normalize_doi, normalize_pmid, normalize_title, parse_date, split_name
from .http import DATA_DIR, USER_AGENT

RW_URL = "https://gitlab.com/crossref/retraction-watch-data/-/raw/main/retraction_watch.csv"
LOCAL_CSV = os.path.join(DATA_DIR, "retraction_watch.csv")
SOURCE = "Retraction Watch"

KEEP_COLS = [
    "Record ID", "Title", "Subject", "Journal", "Publisher", "Country", "Author", "URLS", "ArticleType",
    "RetractionDate", "RetractionDOI", "RetractionPubMedID", "OriginalPaperDate", "OriginalPaperDOI",
    "OriginalPaperPubMedID", "RetractionNature", "Reason", "Notes",
]

# Reasons (Retraction Watch vocabulary) that point at research-integrity problems rather than honest error.
INTEGRITY_REASON_TERMS = [
    "falsification/fabrication", "manipulation of", "paper mill", "compromised peer review",
    "concerns/issues about peer review", "fake peer review", "plagiarism", "euphemisms for",
    "duplication of", "salami slicing", "misconduct", "unreliable data", "unreliable results",
    "unreliable image", "concerns/issues about data", "concerns/issues about image",
    "concerns/issues about results", "computer-aided content", "computer-generated", "rogue editor",
    "false/forged", "ethical violations", "lack of irb", "informed/patient consent",
    "human subject welfare", "investigation by ori", "investigation by company/institution",
    "original data and/or images not provided", "hoax paper", "concerns/issues about third party involvement",
]
# Reasons that indicate an honest error or administrative change (still reported, but less weight).
HONEST_ERROR_TERMS = ["error in", "error by journal", "retract and replace", "withdrawn to publish",
                      "withdrawn as out of date", "doing the right thing", "duplication of content through error"]


def download(dest: str = LOCAL_CSV, url: str = RW_URL) -> str:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with requests.get(url, stream=True, timeout=120, headers={"User-Agent": USER_AGENT}) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    os.replace(tmp, dest)
    return dest


def integrity_related(reason: str) -> bool:
    r = (reason or "").lower()
    return any(t in r for t in INTEGRITY_REASON_TERMS)


def clean_reasons(reason: str) -> list[str]:
    return [x.strip().lstrip("+").strip() for x in str(reason or "").split(";") if x.strip()]


@dataclass
class RWIndex:
    df: pd.DataFrame
    meta: dict
    by_doi: dict = field(default_factory=dict)
    by_pmid: dict = field(default_factory=dict)
    by_title: dict = field(default_factory=dict)
    titles: list = field(default_factory=list)
    by_author_full: dict = field(default_factory=dict)
    by_author_initial: dict = field(default_factory=dict)

    # -- look-ups ------------------------------------------------------------
    def rows(self, idx) -> list[dict]:
        return [self.record(i) for i in sorted(set(idx))]

    def record(self, i: int) -> dict:
        r = self.df.iloc[i]
        return {
            "row": int(i),
            "record_id": str(r["Record ID"]),
            "title": r["Title"],
            "journal": r["Journal"],
            "publisher": r["Publisher"],
            "authors": r["Author"],
            "nature": r["RetractionNature"],
            "notice_date": r["_notice_date"],
            "paper_date": r["_paper_date"],
            "reasons": clean_reasons(r["Reason"]),
            "notice_doi": normalize_doi(r["RetractionDOI"]),
            "paper_doi": normalize_doi(r["OriginalPaperDOI"]),
            "paper_pmid": normalize_pmid(r["OriginalPaperPubMedID"]),
            "urls": [u for u in str(r["URLS"] or "").split(";") if u.startswith("http")],
            "article_type": str(r["ArticleType"] or "").strip(";"),
            "country": str(r["Country"] or "").strip(";"),
            "notes": "" if pd.isna(r["Notes"]) else str(r["Notes"]),
        }

    def find_by_doi(self, doi: Optional[str]) -> list[int]:
        return list(self.by_doi.get(doi, [])) if doi else []

    def find_by_pmid(self, pmid: Optional[str]) -> list[int]:
        return list(self.by_pmid.get(pmid, [])) if pmid else []

    def find_by_title(self, title: Optional[str]) -> list[int]:
        t = normalize_title(title)
        return list(self.by_title.get(t, [])) if t else []

    def find_by_author(self, name: str) -> tuple[list[int], str]:
        """Return (rows, specificity). Searches name as given and with given/family swapped."""
        keys = author_keys(name)
        fam, given = split_name(name)
        swapped = author_keys(f"{given}, {fam}") if given else {}
        if "full" in keys:
            hits = set(self.by_author_full.get(keys["full"], set()))
            if "full" in swapped:
                hits |= self.by_author_full.get(swapped["full"], set())
            return sorted(hits), "full name"
        if "initial" in keys:
            hits = set(self.by_author_initial.get(keys["initial"], set()))
            if "initial" in swapped:
                hits |= self.by_author_initial.get(swapped["initial"], set())
            return sorted(hits), "surname + initial"
        return [], "none"


def load(path: str = LOCAL_CSV, auto_download: bool = True) -> RWIndex:
    if not os.path.exists(path):
        if not auto_download:
            raise FileNotFoundError(path)
        download(path)
    df = pd.read_csv(path, dtype=str, low_memory=False, usecols=lambda c: c in KEEP_COLS)
    df = df.fillna("")
    return build_index(df, {
        "source": "Retraction Watch database (Crossref open data)",
        "url": RW_URL,
        "file": path,
        "downloaded_on": datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    })


def build_index(df: pd.DataFrame, meta: dict) -> RWIndex:
    df = df.reset_index(drop=True).copy()
    for c in KEEP_COLS:
        if c not in df.columns:
            df[c] = ""
    df["_doi"] = df["OriginalPaperDOI"].map(normalize_doi)
    df["_pmid"] = df["OriginalPaperPubMedID"].map(normalize_pmid)
    df["_title"] = df["Title"].map(normalize_title)
    df["_notice_date"] = df["RetractionDate"].map(lambda v: (d.isoformat() if (d := parse_date(v)) else None))
    df["_paper_date"] = df["OriginalPaperDate"].map(lambda v: (d.isoformat() if (d := parse_date(v)) else None))

    idx = RWIndex(df=df, meta=dict(meta))
    by_doi, by_pmid, by_title = defaultdict(list), defaultdict(list), defaultdict(list)
    full, initial = defaultdict(set), defaultdict(set)
    for i, (doi, pmid, title, authors) in enumerate(zip(df["_doi"], df["_pmid"], df["_title"], df["Author"])):
        if doi:
            by_doi[doi].append(i)
        if pmid:
            by_pmid[pmid].append(i)
        if title and len(title) >= 10:
            by_title[title].append(i)
        for name in str(authors).split(";"):
            name = name.strip()
            if not name:
                continue
            k = author_keys(name)
            if "full" in k:
                full[k["full"]].add(i)
            if "initial" in k:
                initial[k["initial"]].add(i)
    # A PubMed ID shared by many unrelated records is a data-entry artefact; don't trust it alone.
    idx.by_pmid = {k: v for k, v in by_pmid.items() if len(v) <= 5}
    idx.by_doi, idx.by_title = dict(by_doi), dict(by_title)
    idx.titles = list(by_title.keys())
    idx.by_author_full, idx.by_author_initial = dict(full), dict(initial)

    nature = df["RetractionNature"].value_counts().to_dict()
    dates = df["_notice_date"].dropna()
    idx.meta.update({
        "n_records": int(len(df)),
        "n_by_nature": {str(k): int(v) for k, v in nature.items()},
        "latest_notice": max(dates) if len(dates) else None,
    })
    return idx
