"""Input adapters. Every adapter returns a list of Study records in the same standard form.

Supported: RIS, PubMed/MEDLINE (.nbib/.txt), BibTeX, EndNote XML, CSV/TSV/Excel,
Word (.docx), PDF and free text (pasted reference lists or lists of DOIs/PMIDs).
"""
from __future__ import annotations

import io
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import rispy

from .models import Study
from .normalize import (
    detect_design, find_dois, find_pmids, find_trial_ids, find_year, is_missing,
    normalize_doi, normalize_pmid, parse_date, iso,
)


@dataclass
class ParseReport:
    file: str
    format: str
    n_records: int = 0
    warnings: list[str] = field(default_factory=list)


# ---- text decoding -----------------------------------------------------------

def decode_bytes(data: bytes) -> str:
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


# ---- helpers -----------------------------------------------------------------

def _first(*vals):
    for v in vals:
        if isinstance(v, list):
            v = v[0] if v else None
        if not is_missing(v):
            return v
    return None


def _year(*vals) -> Optional[int]:
    for v in vals:
        if is_missing(v):
            continue
        y = find_year(str(v))
        if y:
            return y
    return None


def finalize(study: Study) -> Study:
    """Fill derived fields: identifiers hidden in free text, trial IDs, design."""
    blob = " ".join(filter(None, [study.raw, study.abstract or "", study.title or ""]))
    if not study.doi:
        dois = find_dois(blob)
        study.doi = dois[0] if dois else None
    if not study.pmid:
        pmids = find_pmids(blob)
        study.pmid = pmids[0] if pmids else None
    for t in find_trial_ids(blob):
        if t not in study.trial_ids:
            study.trial_ids.append(t)
    study.design = detect_design(study.title, study.abstract, study.pub_types)
    if not study.identified:
        study.id_confidence = "none"
        study.id_method = "no DOI/PMID supplied"
    return study


# ---- RIS ---------------------------------------------------------------------

def _flatten(v) -> str:
    if isinstance(v, dict):
        return " ".join(_flatten(x) for x in v.values())
    if isinstance(v, (list, tuple)):
        return " ".join(_flatten(x) for x in v)
    return "" if v is None else str(v)


def parse_ris(text: str, name: str) -> list[Study]:
    # rispy needs records to start with TY; strip junk before first TY and normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    i = text.find("TY  -")
    text = text[i:] if i >= 0 else text
    records = rispy.loads(text, skip_unknown_tags=False, enforce_list_tags=False)
    out = []
    for r in records:
        authors = r.get("authors") or r.get("first_authors") or []
        if isinstance(authors, str):
            authors = [authors]
        notes = r.get("notes") or []
        notes = " ".join(notes) if isinstance(notes, list) else str(notes or "")
        urls = r.get("urls") or r.get("url") or []
        urls = " ".join(urls) if isinstance(urls, list) else str(urls)
        title = _first(r.get("primary_title"), r.get("title"), r.get("translated_title"))
        pmid = None
        an = _first(r.get("accession_number"))
        if an and re.fullmatch(r"\d{5,9}", str(an).strip()):
            pmid = str(an).strip()
        # every field except the abstract (which may cite other papers' DOIs), incl. unknown tags
        blob = " ".join(_flatten(v) for k, v in r.items() if k not in ("abstract", "notes_abstract"))
        if not pmid:
            m = re.search(r"\bPMID[:\s]*(\d{5,9})\b|pubmed[^\s]*?/(\d{5,9})", blob, re.I)
            pmid = (m.group(1) or m.group(2)) if m else None
        s = Study(
            uid="",
            raw=" ".join(filter(None, [title, notes, urls, blob if not normalize_doi(_first(r.get("doi"))) else ""])),
            source_file=name, source_format="RIS",
            title=title,
            authors=list(authors),
            year=_year(r.get("year"), r.get("publication_year"), r.get("date")),
            journal=_first(r.get("journal_name"), r.get("secondary_title"), r.get("alternate_title3"),
                           r.get("alternate_title1")),
            doi=normalize_doi(_first(r.get("doi"))),
            pmid=pmid or (find_pmids(urls + " " + notes) or [None])[0],
            abstract=_first(r.get("abstract")),
            pub_date=iso(_first(r.get("date"), r.get("publication_year"), r.get("year"))),
        )
        out.append(finalize(s))
    return out


# ---- PubMed / MEDLINE --------------------------------------------------------

def parse_medline(text: str, name: str) -> list[Study]:
    text = text.replace("\r\n", "\n")
    out = []
    for block in re.split(r"\n\s*\n(?=PMID-)", text.strip()):
        fields: dict[str, list[str]] = {}
        tag = None
        for line in block.split("\n"):
            m = re.match(r"^([A-Z]{2,4})\s*- (.*)$", line)
            if m:
                tag = m.group(1)
                fields.setdefault(tag, []).append(m.group(2).strip())
            elif tag and line.startswith("      "):
                fields[tag][-1] += " " + line.strip()
        if "PMID" not in fields and "TI" not in fields:
            continue
        doi = None
        for v in fields.get("LID", []) + fields.get("AID", []):
            if "[doi]" in v:
                doi = normalize_doi(v.replace("[doi]", ""))
                break
        trial_ids = []
        for si in fields.get("SI", []):
            trial_ids += find_trial_ids(si)
        s = Study(
            uid="", raw=" ".join(fields.get("TI", [])),
            source_file=name, source_format="PubMed/MEDLINE",
            title=" ".join(fields.get("TI", [])) or None,
            authors=fields.get("FAU") or fields.get("AU", []),
            year=_year(" ".join(fields.get("DP", []))),
            journal=_first(fields.get("JT"), fields.get("TA")),
            doi=doi,
            pmid=normalize_pmid(_first(fields.get("PMID"))),
            abstract=" ".join(fields.get("AB", [])) or None,
            pub_date=iso(_first(fields.get("DP"))),
            trial_ids=trial_ids,
            pub_types=fields.get("PT", []),
        )
        out.append(finalize(s))
    return out


# ---- BibTeX ------------------------------------------------------------------

def _bibtex_entries(text: str):
    i = 0
    while True:
        m = re.compile(r"@(\w+)\s*\{", re.M).search(text, i)
        if not m:
            return
        kind = m.group(1).lower()
        depth, j = 1, m.end()
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        body = text[m.end(): j - 1]
        i = j
        if kind in {"comment", "preamble", "string"}:
            continue
        yield kind, body


def _bibtex_fields(body: str) -> dict[str, str]:
    fields = {}
    _, _, rest = body.partition(",")
    pos = 0
    rx = re.compile(r"\s*(\w[\w\-]*)\s*=\s*", re.M)
    while True:
        m = rx.search(rest, pos)
        if not m:
            break
        key = m.group(1).lower()
        j = m.end()
        if j < len(rest) and rest[j] == "{":
            depth, k = 1, j + 1
            while k < len(rest) and depth:
                depth += {"{": 1, "}": -1}.get(rest[k], 0)
                k += 1
            val = rest[j + 1: k - 1]
        elif j < len(rest) and rest[j] == '"':
            k = rest.find('"', j + 1)
            k = len(rest) if k < 0 else k + 1
            val = rest[j + 1: k - 1]
        else:
            k = rest.find(",", j)
            k = len(rest) if k < 0 else k
            val = rest[j:k]
        fields[key] = re.sub(r"[{}]", "", re.sub(r"\s+", " ", val)).strip()
        pos = k
    return fields


def parse_bibtex(text: str, name: str) -> list[Study]:
    out = []
    for _, body in _bibtex_entries(text):
        f = _bibtex_fields(body)
        authors = [a.strip() for a in re.split(r"\s+and\s+", f.get("author", "")) if a.strip()]
        s = Study(
            uid="", raw=f.get("title", ""), source_file=name, source_format="BibTeX",
            title=f.get("title") or None, authors=authors, year=_year(f.get("year"), f.get("date")),
            journal=f.get("journal") or f.get("journaltitle") or f.get("booktitle"),
            doi=normalize_doi(f.get("doi")),
            pmid=normalize_pmid(f.get("pmid")) or (normalize_pmid(f.get("eprint"))
                                                    if f.get("eprinttype", "").lower() == "pubmed" else None),
            abstract=f.get("abstract"),
        )
        s.raw = " ".join(filter(None, [s.title, f.get("note"), f.get("url")]))
        out.append(finalize(s))
    return out


# ---- EndNote XML -------------------------------------------------------------

def _xml_text(el) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


def parse_endnote_xml(text: str, name: str) -> list[Study]:
    root = ET.fromstring(text.encode("utf-8"))
    out = []
    for rec in root.iter("record"):
        title = _xml_text(rec.find("./titles/title"))
        authors = [_xml_text(a) for a in rec.findall("./contributors/authors/author")]
        doi = normalize_doi(_xml_text(rec.find("./electronic-resource-num")))
        an = _xml_text(rec.find("./accession-num"))
        notes = _xml_text(rec.find("./notes"))
        s = Study(
            uid="", raw=" ".join([title, notes]), source_file=name, source_format="EndNote XML",
            title=title or None, authors=authors,
            year=_year(_xml_text(rec.find("./dates/year"))),
            journal=_xml_text(rec.find("./titles/secondary-title")) or _xml_text(rec.find("./periodical/full-title")) or None,
            doi=doi, pmid=normalize_pmid(an) if re.fullmatch(r"\d{5,9}", an or "") else None,
            abstract=_xml_text(rec.find("./abstract")) or None,
        )
        out.append(finalize(s))
    return out


# ---- Tables (CSV / TSV / Excel, e.g. Covidence or Rayyan exports) -------------

COLUMN_ALIASES = {
    "title": ["title", "primary title", "article title", "ti", "study title", "document title"],
    "authors": ["authors", "author", "au", "author(s)", "author full names"],
    "year": ["year", "published year", "publication year", "py", "pub year", "date"],
    "journal": ["journal", "journal/book", "source title", "source", "journal title", "publication title", "so"],
    "doi": ["doi", "di", "doi link"],
    "pmid": ["pmid", "pubmed id", "pubmed", "pm", "accession number"],
    "abstract": ["abstract", "ab"],
    "study": ["study", "study id", "study_id", "reference", "citation", "ref"],
}


def _pick(cols: list[str], key: str) -> Optional[str]:
    lower = {c.lower().strip(): c for c in cols}
    for alias in COLUMN_ALIASES[key]:
        if alias in lower:
            return lower[alias]
    return None


def parse_table(df: pd.DataFrame, name: str, fmt: str) -> list[Study]:
    cols = [str(c) for c in df.columns]
    df.columns = cols
    c = {k: _pick(cols, k) for k in COLUMN_ALIASES}
    out = []
    for _, row in df.iterrows():
        get = lambda k: (None if c[k] is None or is_missing(row[c[k]]) else row[c[k]])
        authors = get("authors")
        if isinstance(authors, str):
            authors = [a.strip() for a in re.split(r";|\s+and\s+|\|", authors) if a.strip()]
        raw = " ".join(str(v) for v in row.values if not is_missing(v))
        s = Study(
            uid="", raw=raw, source_file=name, source_format=fmt,
            title=get("title") or get("study"), authors=authors or [],
            year=_year(get("year")), journal=get("journal"),
            doi=normalize_doi(get("doi")), pmid=normalize_pmid(get("pmid")),
            abstract=get("abstract"),
        )
        if not s.title and not s.doi and not s.pmid:
            continue
        out.append(finalize(s))
    return out


# ---- Free text, Word, PDF ----------------------------------------------------

REF_HEADING_RE = re.compile(
    r"^\s*(references|reference list|bibliography|literature cited|included studies|referanser|litteratur)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
NUMBERED_RE = re.compile(r"^\s*(?:\[\d{1,4}\]|\d{1,4}[\.\)])\s+", re.MULTILINE)


def split_references(text: str) -> list[str]:
    """Split a block of text into individual references."""
    text = text.replace("\r\n", "\n").replace("­", "")
    # If the text has a references heading, keep only what follows the last one.
    heads = list(REF_HEADING_RE.finditer(text))
    if heads:
        text = text[heads[-1].end():]
    text = text.strip()
    if not text:
        return []
    lines = [l for l in text.split("\n")]
    nonempty = [l for l in lines if l.strip()]
    # A plain list of identifiers: one DOI/PMID per line
    if nonempty and all(re.fullmatch(r"\s*(?:\d{5,9}|(?:https?://(?:dx\.)?doi\.org/|doi:\s*)?10\.\d{4,9}/\S+)\s*", l, re.I)
                        for l in nonempty):
        return [l.strip() for l in nonempty]
    numbered = NUMBERED_RE.findall(text)
    if len(numbered) >= 2:
        parts = NUMBERED_RE.split(text)
        refs = [re.sub(r"\s+", " ", p).strip() for p in parts]
    elif re.search(r"\n\s*\n", text):
        refs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    else:
        refs = [l.strip() for l in nonempty]
    # Re-join hyphenated DOI line breaks ("10.1056/NEJ- Moa...") are rare; keep simple
    return [r for r in refs if len(r) >= 15 or DOI_RE_SIMPLE.search(r) or re.fullmatch(r"\d{5,9}", r)
            or find_pmids(r)]


DOI_RE_SIMPLE = re.compile(r"10\.\d{4,9}/\S+")


def _guess_title(ref: str) -> Optional[str]:
    """Best-effort title from Vancouver/APA-style citation: the sentence after the author list."""
    r = re.sub(r"https?://\S+|doi:\s*\S+|10\.\d{4,9}/\S+", "", ref, flags=re.I)
    # Title in quotes: Authors. 2021. "Title". Journal...
    m = re.search(r"[\"“”]([^\"“”]{15,})[\"“”]", r)
    if m:
        return m.group(1).strip().rstrip(".")
    # APA: Authors (2020). Title. Journal...
    m = re.search(r"\((?:19|20)\d{2}[a-z]?\)\.\s*(.+?[\.\?!])\s", r + " ")
    if m:
        return m.group(1).strip().rstrip(".")
    # Vancouver: Authors. Title. Journal. Year;...
    parts = [p.strip() for p in re.split(r"(?<=[\.\?!])\s+(?=[A-Z0-9\[\'\"‘“])", r) if p.strip()]
    if len(parts) >= 2:
        cand = parts[1].rstrip(".")
        if len(cand) > 20:
            return cand
    return None


def _guess_authors(ref: str) -> list[str]:
    head = re.split(r"\.\s|\(\d{4}", ref, maxsplit=1)[0]
    if len(head) > 250 or not re.search(r"[A-Z]", head):
        return []
    names = [n.strip() for n in re.split(r",\s*(?=[A-Z][^,]*\s[A-Z]{1,3}\b)|;\s*|\s&\s|,?\s+and\s+", head) if n.strip()]
    names = [n for n in names if 2 < len(n) < 60 and n.lower() != "et al"]
    return names[:30]


def parse_free_text(text: str, name: str, fmt: str = "Free text") -> list[Study]:
    out = []
    for ref in split_references(text):
        dois, pmids = find_dois(ref), find_pmids(ref)
        if re.fullmatch(r"\d{5,9}", ref.strip()):
            pmids = [ref.strip()]
        s = Study(
            uid="", raw=ref, source_file=name, source_format=fmt,
            title=_guess_title(ref), authors=_guess_authors(ref) if not re.fullmatch(r"\S+", ref) else [],
            year=find_year(ref), doi=dois[0] if dois else None, pmid=pmids[0] if pmids else None,
        )
        if not s.title and not s.identified:
            s.id_notes.append("Could not detect a title; will try to identify via Crossref using the full citation text.")
        out.append(finalize(s))
    return out


def docx_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paras = []
    for p in root.iter(f"{ns}p"):
        paras.append("".join(t.text or "" for t in p.iter(f"{ns}t")))
    # numbered lists in Word lose their numbers; one paragraph per reference
    return "\n\n".join(p for p in paras if p.strip()) if not any(NUMBERED_RE.match(p) for p in paras) \
        else "\n".join(paras)


def pdf_text(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


# ---- dispatcher --------------------------------------------------------------

def sniff_format(name: str, text: Optional[str]) -> str:
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if ext in {"xlsx", "xls"}:
        return "excel"
    if ext == "docx":
        return "docx"
    if ext == "pdf":
        return "pdf"
    head = (text or "")[:5000]
    if ext == "ris" or re.search(r"^TY  - ", head, re.M):
        return "ris"
    if ext == "nbib" or re.search(r"^PMID- \d+", head, re.M):
        return "medline"
    if ext in {"bib", "bibtex"} or re.search(r"^\s*@\w+\s*\{", head, re.M):
        return "bibtex"
    if ext == "xml" and "<record" in head:
        return "endnote_xml"
    if ext in {"csv", "tsv"}:
        return ext
    return "text"


def parse_upload(name: str, data: bytes) -> tuple[list[Study], ParseReport]:
    fmt = sniff_format(name, None)
    text = None
    if fmt not in {"excel", "docx", "pdf"}:
        text = decode_bytes(data)
        fmt = sniff_format(name, text)
    report = ParseReport(file=name, format=fmt)
    try:
        if fmt == "ris":
            studies = parse_ris(text, name)
        elif fmt == "medline":
            studies = parse_medline(text, name)
        elif fmt == "bibtex":
            studies = parse_bibtex(text, name)
        elif fmt == "endnote_xml":
            studies = parse_endnote_xml(text, name)
        elif fmt in {"csv", "tsv"}:
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python", dtype=str)
            studies = parse_table(df, name, fmt.upper())
        elif fmt == "excel":
            studies = parse_table(pd.read_excel(io.BytesIO(data), dtype=str), name, "Excel")
        elif fmt == "docx":
            studies = parse_free_text(docx_text(data), name, "Word")
        elif fmt == "pdf":
            studies = parse_free_text(pdf_text(data), name, "PDF")
            report.warnings.append("PDF text extraction is approximate; check the parsed references below.")
        else:
            studies = parse_free_text(text, name)
    except Exception as e:  # keep the app alive on malformed files
        report.warnings.append(f"Could not parse as {fmt}: {e}")
        studies = []
    report.n_records = len(studies)
    return studies, report


def assign_ids(studies: list[Study]) -> list[Study]:
    for i, s in enumerate(studies, 1):
        s.uid = f"S{i:03d}"
    return studies


def deduplicate(studies: list[Study]) -> tuple[list[Study], int]:
    """Drop exact duplicates across uploaded files (same DOI, PMID or normalised title)."""
    from .normalize import normalize_title
    seen, out, dropped = set(), [], 0
    for s in studies:
        keys = {k for k in [("doi", s.doi), ("pmid", s.pmid), ("title", normalize_title(s.title))] if k[1]}
        if keys & seen:
            dropped += 1
            continue
        seen |= keys
        out.append(s)
    return out, dropped
