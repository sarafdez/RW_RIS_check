import os
import json
import re
from datetime import datetime, timedelta

import pandas as pd
from rapidfuzz import process, fuzz
import requests


RW_URL = "https://gitlab.com/crossref/retraction-watch-data/-/raw/main/retraction_watch.csv"
_LOCAL_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "retraction_watch.csv")

BAD_TITLES = {
    "editorial",
    "index",
    "correction",
    "corrigendum",
    "erratum",
    "reply",
    "response",
    "commentary",
    "letter",
    "news",
}


def load_retraction_watch():
    if os.path.exists(_LOCAL_CSV):
        rw_df = pd.read_csv(_LOCAL_CSV)
        metadata = {
            "downloaded_on": datetime.utcfromtimestamp(os.path.getmtime(_LOCAL_CSV)).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "n_records": len(rw_df),
            "source": "Local file",
            "url": _LOCAL_CSV,
        }
    else:
        rw_df = pd.read_csv(RW_URL)
        metadata = {
            "downloaded_on": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "n_records": len(rw_df),
            "source": "Retraction Watch public GitHub CSV",
            "url": RW_URL,
        }

    return rw_df, metadata


# Cleaning
def normalize_doi(doi):
    if doi is None or pd.isna(doi):
        return None

    doi = str(doi).lower().strip()

    if doi in {"", "nan", "none"}:
        return None

    doi = (
        doi
        .replace("https://doi.org/", "")
        .replace("http://doi.org/", "")
        .replace("http://dx.doi.org/", "")
        .replace("doi:", "")
        .strip()
    )
    doi = re.sub(r"\.pub\d+$", "", doi)
    return doi or None


def normalize_title(title):
    if title is None or pd.isna(title):
        return None

    title = title.lower()
    title = re.sub(r"[\/\-–—]", " ", title)
    title = re.sub(r"\s+", " ", title)
    title = re.sub(r"[^\w\s]", "", title)

    return title.strip()


def normalize_pubmed_id(pmid):
    if pmid is None or pd.isna(pmid):
        return None
    pmid = re.sub(r"\D", "", str(pmid).strip())
    return pmid or None


def filter_bad_titles(title_norm, min_len=10):
    if not title_norm:
        return False
    if title_norm in BAD_TITLES:
        return False
    if len(title_norm) < min_len:
        return False
    return True


# Checks

def report_basic_checks(df):
    print("\nQUALITY CHECKS")
    print("-" * 40)
    print(f"Total records in ris file: {len(df)}")
    print(f"Missing DOI: {df['doi'].isna().sum()}")
    print(f"Missing title: {df['title_norm'].isna().sum()}")
    print()


def match_by_doi(review_df, rw_df, key="doi_norm"):
    left = review_df.dropna(subset=[key]).copy()
    right = rw_df.dropna(subset=[key]).copy()

    matched = left.merge(
        right,
        on=key,
        how="inner",
        suffixes=("_ris", "_rw"),
    )

    #if "Record ID" in matched.columns:
    #    matched = matched.drop_duplicates(subset=[key, "Record ID"])

    matched["match_type"] = "doi"
    return matched

def match_by_pubmed_id(review_df, rw_df, key="pmid_norm"):
    left = review_df.dropna(subset=[key]).copy()
    right = rw_df.dropna(subset=[key]).copy()

    matched = left.merge(
        right,
        on=key,
        how="inner",
        suffixes=("_ris", "_rw"),
    )

    matched["match_type"] = "pubmed_id"
    return matched


def match_by_title_exact(review_df, rw_df, key="title_norm"):
    left = review_df.dropna(subset=[key]).copy()
    right = rw_df.dropna(subset=[key]).copy()

    matched = left.merge(
        right,
        on=key,
        how="inner",
        suffixes=("_ris", "_rw"),
    )

    if "Record ID" in matched.columns:
        matched = matched.drop_duplicates(subset=[key, "Record ID"])

    matched["match_type"] = "title_exact"
    return matched



def match_by_title_fuzzy(review_df, rw_df, key="title_norm", threshold=88):
    rw_titles = rw_df[key].dropna().astype(str).unique().tolist()

    def best_match(title):
        if title is None or pd.isna(title):
            return (None, None)

        title = str(title).strip()
        if not title:
            return (None, None)

        res = process.extractOne(
            title,
            rw_titles,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )
        if not res:
            return (None, None)

        match, score, *_ = res
        return (match, score)

    df = review_df.copy()
    df[["matched_title_norm", "title_score"]] = df[key].apply(
        lambda t: pd.Series(best_match(t))
    )

    matched = df.dropna(subset=["matched_title_norm"]).copy()

    if matched.empty:
        return pd.DataFrame()

    merged = matched.merge(
        rw_df,
        left_on="matched_title_norm",
        right_on=key,
        how="inner",
        suffixes=("_review", "_rw"),
    )

    merged["fuzzy_confidence"] = merged["title_score"].apply(
        lambda s: "high" if s >= 95 else "low"
    )
    merged["match_type"] = "title_fuzzy"
    return merged
