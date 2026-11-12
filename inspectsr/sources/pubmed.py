"""PubMed E-utilities: identifiers, publication types, linked notices and trial registrations."""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from typing import Iterable

from ..normalize import find_trial_ids, find_year, normalize_doi, normalize_title
from .http import get

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
SOURCE = "PubMed"

RETRACTION_REFS = {"RetractionIn", "PartialRetractionIn", "RetractedandRepublishedIn"}
CONCERN_REFS = {"ExpressionOfConcernIn"}
CORRECTION_REFS = {"ErratumIn", "CorrectedandRepublishedIn", "UpdateIn"}
COMMENT_REFS = {"CommentIn"}


def _base() -> dict:
    p = {"tool": "retraction-watch-check"}
    if os.environ.get("NCBI_API_KEY"):
        p["api_key"] = os.environ["NCBI_API_KEY"]
    return p


def _chunks(xs: list, n: int) -> Iterable[list]:
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def pmids_for_dois(dois: list[str]) -> list[str]:
    """Search PubMed for many DOIs at once; returns PMIDs (mapping is recovered via efetch)."""
    out: list[str] = []
    for chunk in _chunks(sorted(set(dois)), 40):
        term = " OR ".join(f'"{d}"[doi]' for d in chunk)
        data = get(f"{EUTILS}/esearch.fcgi", {**_base(), "db": "pubmed", "term": term,
                                               "retmode": "json", "retmax": 500})
        out += (data or {}).get("esearchresult", {}).get("idlist", [])
    return out


# PubMed's own stopword list: a stopword with a [ti] tag matches nothing, which makes the whole AND fail.
_STOP = set("""a about again all almost also although always among an and another any are as at be because been before
being between both but by can could did do does done due during each either enough especially etc for found from
further had has have having here how however i if in into is it its itself just kg km made mainly make may mg might ml
mm most mostly must nearly neither no nor obtained of often on our overall perhaps pmid quite rather really regarding
seem seen several should show showed shown shows significantly since so some such than that the their theirs them then
there therefore these they this those through thus to upon use used using various very was we were what when which
while with within without would versus vs""".split())


def _title_term(title: str) -> str | None:
    words = [w for w in (normalize_title(title) or "").split() if w not in _STOP and len(w) > 2 and not w.isdigit()]
    words = sorted(set(words), key=len, reverse=True)[:7]
    if len(words) < 3:
        return None
    return "(" + " AND ".join(f"{w}[ti]" for w in words) + ")"


def search_titles(titles: list[str], batch: int = 10) -> dict[str, dict]:
    """Find PubMed records for many titles with few requests.

    Each title becomes (word1[ti] AND word2[ti] ...) using its most distinctive words; ~10 titles are ORed per
    search. Returns {pmid: record}; the caller decides which record belongs to which title.
    """
    terms = [t for t in (_title_term(x) for x in titles) if t]
    pmids: list[str] = []
    for chunk in _chunks(terms, batch):
        data = get(f"{EUTILS}/esearch.fcgi", {**_base(), "db": "pubmed", "term": " OR ".join(chunk),
                                               "retmode": "json", "retmax": 500})
        pmids += (data or {}).get("esearchresult", {}).get("idlist", [])
    return fetch(pmids) if pmids else {}


def fetch(pmids: list[str]) -> dict[str, dict]:
    """efetch PubMed XML for PMIDs → {pmid: record}"""
    records: dict[str, dict] = {}
    for chunk in _chunks(sorted(set(pmids)), 150):
        xml = get(f"{EUTILS}/efetch.fcgi", {**_base(), "db": "pubmed", "id": ",".join(chunk),
                                             "retmode": "xml"}, as_json=False)
        if not xml:
            continue
        root = ET.fromstring(xml.encode("utf-8"))
        for art in root.iter("PubmedArticle"):
            rec = _parse_article(art)
            records[rec["pmid"]] = rec
    return records


def _text(el) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


def _parse_article(art) -> dict:
    mc = art.find("MedlineCitation")
    pmid = _text(mc.find("PMID"))
    a = mc.find("Article")
    authors = []
    for au in a.findall("./AuthorList/Author"):
        if au.find("LastName") is not None:
            authors.append(f"{_text(au.find('LastName'))}, {_text(au.find('ForeName'))}".strip(", "))
        elif au.find("CollectiveName") is not None:
            authors.append(_text(au.find("CollectiveName")))
    doi = None
    for aid in art.findall("./PubmedData/ArticleIdList/ArticleId"):
        if aid.get("IdType") == "doi":
            doi = normalize_doi(aid.text)
    if not doi:
        for eid in a.findall("ELocationID"):
            if eid.get("EIdType") == "doi":
                doi = normalize_doi(eid.text)
    pub_types = [_text(p) for p in a.findall("./PublicationTypeList/PublicationType")]
    links = []
    for cc in mc.findall("./CommentsCorrectionsList/CommentsCorrections"):
        links.append({
            "ref_type": cc.get("RefType"),
            "source": _text(cc.find("RefSource")),
            "pmid": _text(cc.find("PMID")) or None,
            "year": find_year(_text(cc.find("RefSource"))),
        })
    trials = []
    for db in a.findall("./DataBankList/DataBank"):
        name = _text(db.find("DataBankName"))
        for acc in db.findall("./AccessionNumberList/AccessionNumber"):
            trials += find_trial_ids(f"{name} {_text(acc)}") or ([f"{name}: {_text(acc)}"] if name else [])
    abstract = " ".join(_text(x) for x in a.findall("./Abstract/AbstractText")) or None
    if abstract:
        trials += find_trial_ids(abstract)
    year = find_year(_text(a.find("./Journal/JournalIssue/PubDate")))
    ad = a.find("ArticleDate")
    edate = None
    if ad is not None:
        try:
            edate = f"{int(_text(ad.find('Year'))):04d}-{int(_text(ad.find('Month'))):02d}-{int(_text(ad.find('Day'))):02d}"
        except ValueError:
            pass
    return {
        "pmid": pmid, "doi": doi, "title": _text(a.find("ArticleTitle")) or None,
        "authors": authors, "journal": _text(a.find("./Journal/Title")) or None,
        "year": year, "epub_date": edate, "pub_types": pub_types, "links": links,
        "trial_ids": sorted(set(trials)), "abstract": abstract,
    }


def pubmed_url(pmid: str) -> str:
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


def ref_type_label(ref_type: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", ref_type or "").replace("Of", "of").replace("In", "in")
