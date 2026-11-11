"""Normalisation and identifier extraction helpers (pure functions, no I/O)."""
from __future__ import annotations

import html
import re
import unicodedata
from datetime import date, datetime
from typing import Optional

import pandas as pd

DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"]+)", re.IGNORECASE)
PMID_RE = re.compile(r"\bPMID:?\s*(\d{5,9})\b", re.IGNORECASE)
PUBMED_URL_RE = re.compile(r"(?:pubmed\.ncbi\.nlm\.nih\.gov/|ncbi\.nlm\.nih\.gov/pubmed/)(\d{5,9})", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")

# Trial registry identifiers (most common primary registries in WHO ICTRP)
TRIAL_ID_PATTERNS = {
    "ClinicalTrials.gov": re.compile(r"\bNCT\s?\d{8}\b", re.IGNORECASE),
    "ISRCTN": re.compile(r"\bISRCTN\s?\d{8}\b", re.IGNORECASE),
    "ANZCTR": re.compile(r"\bACTRN\s?\d{14}\b", re.IGNORECASE),
    "EU CTR (EudraCT)": re.compile(r"\b(?:EudraCT\s*(?:No\.?|number)?:?\s*)?(20\d{2}-\d{6}-\d{2})\b", re.IGNORECASE),
    "EU CTIS": re.compile(r"\b(20\d{2}-5\d{5}-\d{2}-\d{2})\b"),
    "ChiCTR": re.compile(r"\bChiCTR[-\s]?[A-Z]{0,4}-?\d{6,10}\b", re.IGNORECASE),
    "CTRI": re.compile(r"\bCTRI/\d{4}/\d{2,3}/\d{6}\b", re.IGNORECASE),
    "IRCT": re.compile(r"\bIRCT\d{6,}N\d+\b", re.IGNORECASE),
    "DRKS": re.compile(r"\bDRKS\d{8}\b", re.IGNORECASE),
    "JPRN/UMIN": re.compile(r"\b(?:UMIN\d{9}|jRCT[s]?\d{9,10})\b", re.IGNORECASE),
    "PACTR": re.compile(r"\bPACTR\d{15}\b", re.IGNORECASE),
    "NTR/NL": re.compile(r"\b(?:NTR\d{3,5}|NL\d{4,5})\b"),
}

NOTICE_TITLE_RE = re.compile(
    r"^\s*(retraction( note| notice)?\s*[:\-–—]|retraction (of|to)\b|notice of retraction|expression of concern|"
    r"editorial expression of concern|erratum|corrigendum|correction( to)?\s*[:\-–—]|correction (of|to)\b)",
    re.IGNORECASE,
)


def is_missing(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and pd.isna(x):
        return True
    if isinstance(x, str) and x.strip().lower() in {"", "nan", "none", "unavailable", "0"}:
        return True
    return False


def normalize_doi(doi) -> Optional[str]:
    if is_missing(doi):
        return None
    s = html.unescape(str(doi)).strip().lower()
    s = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    # Some exports put several DOIs in one field; keep the first.
    m = DOI_RE.search(s)
    if not m:
        return None
    s = m.group(1).rstrip(".,;]}")
    # Drop unbalanced closing brackets picked up from surrounding prose, but keep
    # those that belong to the DOI, e.g. 10.1016/s0140-6736(20)31180-6
    while s.endswith(")") and s.count(")") > s.count("("):
        s = s[:-1].rstrip(".,;")
    while s.endswith(">") and s.count(">") > s.count("<"):
        s = s[:-1].rstrip(".,;")
    return s or None


def normalize_pmid(pmid) -> Optional[str]:
    if is_missing(pmid):
        return None
    s = str(pmid).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = re.sub(r"\D", "", s)
    if not s or s == "0" or len(s) > 9:
        return None
    return s


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def normalize_title(title) -> Optional[str]:
    if is_missing(title):
        return None
    t = html.unescape(str(title))
    t = re.sub(r"<[^>]+>", " ", t)            # stray HTML / MathML tags
    t = strip_accents(t).lower()
    t = re.sub(r"^\s*(retracted article|retracted|withdrawn)\s*[:\-–—]\s*", "", t)
    t = re.sub(r"[\/\-–—]", " ", t)
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip() or None


def is_notice_title(title) -> bool:
    return bool(title and NOTICE_TITLE_RE.match(str(title)))


def find_dois(text: str) -> list[str]:
    out = []
    for m in DOI_RE.finditer(text or ""):
        d = normalize_doi(m.group(1))
        if d and d not in out:
            out.append(d)
    return out


def find_pmids(text: str) -> list[str]:
    out = []
    for rx in (PMID_RE, PUBMED_URL_RE):
        for m in rx.finditer(text or ""):
            if m.group(1) not in out:
                out.append(m.group(1))
    return out


def find_trial_ids(text: str) -> list[str]:
    """Return registry IDs as 'Registry: ID' strings, de-duplicated, in a canonical form."""
    found: list[str] = []
    for registry, rx in TRIAL_ID_PATTERNS.items():
        for m in rx.finditer(text or ""):
            raw = m.group(1) if m.groups() else m.group(0)
            ident = re.sub(r"\s+", "", raw).upper()
            if registry == "JPRN/UMIN" and ident.startswith("JRCT"):
                ident = "jRCT" + ident[4:]
            if registry == "ChiCTR":
                ident = "ChiCTR" + ident[6:]
            item = f"{registry}: {ident}"
            if item not in found:
                found.append(item)
    # an EU CTIS number also looks like a EudraCT number prefix; keep both only if distinct
    return found


def find_year(text: str) -> Optional[int]:
    m = YEAR_RE.search(text or "")
    return int(m.group(1)) if m else None


def parse_date(value) -> Optional[date]:
    """Parse the many date formats we meet (RW '6/25/2026 0:00', ISO, '2020 May', [[2020,5,1]])."""
    if is_missing(value):
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (list, tuple)):
        parts = [int(p) for p in value if p is not None]
        if not parts:
            return None
        parts += [1] * (3 - len(parts))
        try:
            return date(*parts[:3])
        except ValueError:
            return None
    s = str(value).strip()
    for fmt in ("%m/%d/%Y %H:%M", "%m/%d/%Y", "%Y-%m-%d", "%Y-%m", "%Y/%m/%d", "%Y %b %d", "%Y %b", "%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    try:
        return pd.to_datetime(s, errors="raise").date()
    except Exception:
        return None


def iso(d) -> Optional[str]:
    d = parse_date(d) if not isinstance(d, date) else d
    return d.isoformat() if d else None


# ---- Author names -------------------------------------------------------------

def split_name(name: str) -> tuple[str, str]:
    """Return (family, given) from 'Family, Given', 'Given Family' or 'Family GI'."""
    name = re.sub(r"\s+", " ", strip_accents(str(name))).strip().strip(".")
    if not name:
        return "", ""
    if "," in name:
        fam, given = name.split(",", 1)
        return fam.strip(), given.strip()
    parts = name.split(" ")
    if len(parts) == 1:
        return parts[0], ""
    # PubMed style 'Smith JA'
    if re.fullmatch(r"[A-Z]{1,3}", parts[-1]):
        return " ".join(parts[:-1]), parts[-1]
    return parts[-1], " ".join(parts[:-1])


def _clean(s: str) -> str:
    return re.sub(r"[^a-z]", "", strip_accents(s).lower())


def author_keys(name: str) -> dict[str, str]:
    """Keys used for author look-up.

    'full'    family + first given name   (specific)
    'initial' family + first initial      (sensitive, many homonyms)
    """
    fam, given = split_name(name)
    fam_c = _clean(fam)
    if not fam_c:
        return {}
    given_tokens = [t for t in re.split(r"[\s\-\.]+", given) if t]
    first = _clean(given_tokens[0]) if given_tokens else ""
    keys = {}
    if first:
        keys["initial"] = f"{fam_c}|{first[0]}"
        if len(first) > 1 and not re.fullmatch(r"[A-Z]{2,3}", given_tokens[0]):
            keys["full"] = f"{fam_c}|{first}"
    return keys


GROUP_RE = re.compile(r"\b(group|collaborat\w*|consortium|investigators|committee|network|trialists|"
                      r"study|trial|team|society|association|working party|writing)\b", re.IGNORECASE)


def is_group_author(name: str) -> bool:
    return bool(GROUP_RE.search(str(name or "")))


def display_name(name: str) -> str:
    fam, given = split_name(name)
    return f"{fam}, {given}".strip(", ")


# ---- Study design categorisation ---------------------------------------------

def detect_design(title: str | None, abstract: str | None, pub_types: list[str]) -> str:
    pts = {p.lower() for p in pub_types or []}
    text = f"{title or ''} {abstract or ''}".lower()
    if is_notice_title(title) or pts & {"retraction of publication", "published erratum", "expression of concern"}:
        return "Notice (not a study)"
    if pts & {"systematic review", "meta-analysis"} or re.search(r"systematic review|meta-analys", title or "", re.I):
        return "Systematic review"
    if "randomized controlled trial" in pts or re.search(r"\brandomi[sz]ed\b|\brandomly (assigned|allocated)\b|\brct\b", text):
        if re.search(r"\bprotocol\b", title or "", re.I):
            return "Trial protocol"
        return "RCT"
    if pts & {"clinical trial", "clinical trial, phase ii", "clinical trial, phase iii", "controlled clinical trial"}:
        return "Other trial"
    if pts & {"observational study", "comparative study"} or re.search(r"cohort|case.control|cross.sectional|registry", text):
        return "Observational"
    if "conference" in " ".join(pts) or "abstract" in " ".join(pts):
        return "Conference abstract"
    return "Unknown"
