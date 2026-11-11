"""Identification and evidence gathering.

For each study we build a Dossier: everything every source said about it, kept verbatim enough
that each proposed INSPECT-SR answer can be traced back to its evidence.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from rapidfuzz import fuzz, process

from .models import Study
from .normalize import (
    detect_design, find_trial_ids, is_notice_title, normalize_title, split_name, strip_accents,
)
from .sources import crossref, openalex, pubmed, registries
from .sources.http import SourceError, now_utc
from .sources.retraction_watch import RWIndex

FUZZY_CUTOFF = 88
FUZZY_HIGH = 95


@dataclass
class RWMatch:
    row: int
    method: str          # doi | pmid | title_exact | title_fuzzy
    score: float = 100.0
    corroborated: bool = True   # author / year agree
    strength: str = "definite"  # definite | probable | possible


@dataclass
class Dossier:
    rw: list[RWMatch] = field(default_factory=list)
    crossref: Optional[dict] = None
    pubmed: Optional[dict] = None
    openalex: Optional[dict] = None
    registrations: dict = field(default_factory=dict)       # trial id -> registry record
    author_rw: dict = field(default_factory=dict)           # author name -> {"rows": [...], "specificity": str}
    author_openalex: dict = field(default_factory=dict)     # author name -> [works]
    errors: list[str] = field(default_factory=list)
    sources_checked: list[str] = field(default_factory=list)
    notice_info: dict = field(default_factory=dict)          # "doi:x" / "pmid:y" -> {"title", "citation"}


@dataclass
class RunInfo:
    started: str = ""
    finished: str = ""
    online: bool = True
    sources: dict = field(default_factory=dict)   # source -> description/version
    errors: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)  # notes on records merged after identification


Progress = Callable[[float, str], None]


def _noop(p: float, msg: str):
    pass


# ---- identification -------------------------------------------------------------

def _title_sim(a: Optional[str], b: Optional[str]) -> float:
    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb:
        return 0.0
    return float(fuzz.token_sort_ratio(na, nb))


def _accept_crossref(study: Study, items: list[dict]) -> Optional[tuple[dict, str, float]]:
    """Choose a Crossref citation match we can trust, or None."""
    for it in items:
        cand_title = (it.get("title") or [None])[0]
        if not cand_title:
            continue
        if is_notice_title(cand_title) and not is_notice_title(study.title):
            continue  # don't resolve a study to its own retraction notice
        year = ((it.get("issued") or {}).get("date-parts") or [[None]])[0][0]
        year_ok = not study.year or not year or abs(int(year) - int(study.year)) <= 1
        if study.title:
            sim = _title_sim(study.title, cand_title)
            if sim >= 97 and year_ok:
                return it, "high", sim
            if sim >= 92 and year_ok:
                return it, "medium", sim
        else:
            nt, raw = normalize_title(cand_title), normalize_title(study.raw)
            if nt and raw and len(nt) >= 20:
                sim = float(fuzz.partial_ratio(nt, raw))
                if sim >= 95 and year_ok:
                    return it, "medium", sim
    return None


def _accept_pubmed(study: Study, records: list[dict]) -> Optional[tuple[dict, str, float]]:
    """Pick the PubMed record that matches the study's title (and year), or None."""
    best, best_sim = None, 0.0
    for rec in records:
        if not rec.get("title") or (is_notice_title(rec["title"]) and not is_notice_title(study.title)):
            continue
        sim = _title_sim(study.title, rec["title"])
        year_ok = not study.year or not rec.get("year") or abs(int(rec["year"]) - int(study.year)) <= 1
        if year_ok and sim > best_sim:
            best, best_sim = rec, sim
    if best and best_sim >= 97:
        return best, "high", best_sim
    if best and best_sim >= 92:
        return best, "medium", best_sim
    return None


def _accept_openalex(study: Study, works: list[dict]) -> Optional[tuple[dict, str, float]]:
    best, best_sim = None, 0.0
    for w in works:
        if not w.get("title") or (is_notice_title(w["title"]) and not is_notice_title(study.title)):
            continue
        y = w.get("publication_year")
        year_ok = not study.year or not y or abs(int(y) - int(study.year)) <= 1
        sim = _title_sim(study.title, w["title"])
        if year_ok and sim > best_sim:
            best, best_sim = w, sim
    if best and best_sim >= 97:
        return best, "high", best_sim
    if best and best_sim >= 92:
        return best, "medium", best_sim
    return None


def identify(studies: list[Study], dossiers: dict[str, Dossier], progress: Progress = _noop,
             use_crossref: bool = False):
    """Fill in missing DOIs/PMIDs and metadata, cheapest source first.

    1. PubMed title search, batched (~10 titles per request; most included health studies are in PubMed).
    2. OpenAlex title search, in parallel (~10 requests/second; covers nearly all Crossref DOIs).
    3. Optional: Crossref citation matching for what is left (slow: 1 request/second on the public API).
    """
    pm_records: dict[str, dict] = {}
    todo = [s for s in studies if not s.identified]

    # 1. PubMed, batched by title
    with_title = [s for s in todo if s.title and len(normalize_title(s.title) or "") >= 20]
    if with_title:
        progress(0.0, f"Searching PubMed for {len(with_title)} references without DOI/PMID")
        try:
            found = pubmed.search_titles([s.title for s in with_title])
            pm_records.update(found)
            cands = list(found.values())
            for s in with_title:
                hit = _accept_pubmed(s, cands)
                if hit:
                    rec, conf, sim = hit
                    s.pmid, s.doi = rec["pmid"], s.doi or rec.get("doi")
                    s.id_method, s.id_confidence = "PubMed title match", conf
                    s.id_notes.append(f"Found in PubMed by title (similarity {sim:.0f}%).")
        except Exception as e:  # network/XML problems shouldn't stop the run
            for s in with_title:
                dossiers[s.uid].errors.append(f"PubMed title search failed: {e}")

    # 2. OpenAlex title search, parallel
    rest = [s for s in todo if not s.identified and s.title and len(normalize_title(s.title) or "") >= 20]
    if rest:
        def _oa(s: Study):
            try:
                return s, openalex.search_title(s.title), None
            except SourceError as e:
                return s, [], str(e)

        with ThreadPoolExecutor(max_workers=4) as ex:
            for i, (s, works, err) in enumerate(ex.map(_oa, rest)):
                progress(i / len(rest), f"Searching OpenAlex for references not in PubMed ({i + 1}/{len(rest)})")
                if err:
                    dossiers[s.uid].errors.append(f"OpenAlex title search failed: {err}")
                    continue
                hit = _accept_openalex(s, works)
                if hit:
                    w, conf, sim = hit
                    from .normalize import normalize_doi, normalize_pmid
                    s.doi = normalize_doi(w.get("doi"))
                    s.pmid = s.pmid or normalize_pmid(((w.get("ids") or {}).get("pmid") or "").rsplit("/", 1)[-1])
                    if s.identified:
                        s.id_method, s.id_confidence = "OpenAlex title match", conf
                        s.id_notes.append(f"Found in OpenAlex by title (similarity {sim:.0f}%).")

    # 3. Crossref citation matching for the rest (optional: slow on the public API)
    rest = [s for s in todo if not s.identified]
    if not use_crossref:
        for s in rest:
            s.id_notes.append("Not found in PubMed or OpenAlex; checked against Retraction Watch by title only.")
        rest = []
    t0 = time.monotonic()
    for i, s in enumerate(rest):
        eta = ""
        if i >= 3:
            left = (time.monotonic() - t0) / i * (len(rest) - i)
            eta = f" · about {left / 60:.0f} min left" if left >= 90 else f" · about {left:.0f} s left"
        progress(i / max(len(rest), 1), f"Crossref citation matching for references not in PubMed ({i + 1}/{len(rest)}){eta}")
        structured = " ".join(filter(None, [s.title, " ".join(s.authors[:3]), str(s.year or ""), s.journal or ""]))
        free_text = s.source_format in ("Free text", "Word", "PDF")
        query = s.raw if (free_text and s.raw) or not s.title else structured
        if not query.strip():
            continue
        try:
            items = crossref.match_citation(query[:500])
        except SourceError as e:
            dossiers[s.uid].errors.append(f"Crossref citation matching failed: {e}")
            continue
        hit = _accept_crossref(s, items)
        if hit:
            it, conf, sim = hit
            s.doi = it["DOI"].lower()
            s.id_method = "Crossref citation match"
            s.id_confidence = conf
            s.id_notes.append(f"DOI found by Crossref citation matching (title similarity {sim:.0f}%).")
        else:
            s.id_notes.append("Not found in PubMed or Crossref; checked by title only.")

    # 2. DOI → PMID (batch), and PubMed records for all PMIDs
    need_pmid = [s for s in studies if s.doi and not s.pmid]
    try:
        if need_pmid:
            progress(0.0, "Looking up PubMed IDs")
            pmids = pubmed.pmids_for_dois([s.doi for s in need_pmid])
            pm_records.update(pubmed.fetch(pmids))
            by_doi = {r["doi"]: r for r in pm_records.values() if r.get("doi")}
            for s in need_pmid:
                if s.doi in by_doi:
                    s.pmid = by_doi[s.doi]["pmid"]
        missing = [s.pmid for s in studies if s.pmid and s.pmid not in pm_records]
        if missing:
            progress(0.5, "Fetching PubMed records")
            pm_records.update(pubmed.fetch(missing))
    except (SourceError, Exception) as e:  # XML/network problems shouldn't stop the run
        for s in studies:
            dossiers[s.uid].errors.append(f"PubMed unavailable: {e}")
    for s in studies:
        rec = pm_records.get(s.pmid) if s.pmid else None
        if rec:
            dossiers[s.uid].pubmed = rec
            dossiers[s.uid].sources_checked.append("PubMed")
            if not s.doi and rec.get("doi"):
                s.doi = rec["doi"]
                s.id_notes.append("DOI taken from PubMed record.")


def _fill_metadata(s: Study, d: Dossier):
    cr, pm = d.crossref or {}, d.pubmed or {}
    s.title = s.title or pm.get("title") or cr.get("title")
    if not s.authors:
        s.authors = pm.get("authors") or cr.get("authors") or []
    s.year = s.year or pm.get("year") or cr.get("year")
    s.journal = s.journal or pm.get("journal") or cr.get("journal")
    s.abstract = s.abstract or pm.get("abstract")
    dates = [x for x in (pm.get("epub_date"), cr.get("pub_date"), s.pub_date) if x and len(x) >= 7]
    s.pub_date = min(dates) if dates else s.pub_date
    if pm.get("pub_types"):
        s.pub_types = pm["pub_types"]
    for t in (pm.get("trial_ids") or []) + find_trial_ids(s.abstract or ""):
        if t not in s.trial_ids:
            s.trial_ids.append(t)
    s.design = detect_design(s.title, s.abstract, s.pub_types)


def gather_online(studies: list[Study], dossiers: dict[str, Dossier], progress: Progress = _noop):
    with_doi = [s for s in studies if s.doi]

    progress(0.0, f"Crossref: notices for {len(with_doi)} DOIs")
    try:
        works = crossref.works_by_doi([s.doi for s in with_doi])
        for s in with_doi:
            msg = works.get(s.doi)
            if msg:
                dossiers[s.uid].crossref = crossref.summarize(msg)
                dossiers[s.uid].sources_checked.append("Crossref")
    except SourceError as e:
        for s in with_doi:
            dossiers[s.uid].errors.append(f"Crossref: {e}")

    progress(0.0, "OpenAlex")
    try:
        works = openalex.works_by_doi([s.doi for s in with_doi])
        for s in with_doi:
            if s.doi in works:
                dossiers[s.uid].openalex = works[s.doi]
                dossiers[s.uid].sources_checked.append("OpenAlex")
    except SourceError as e:
        for s in with_doi:
            dossiers[s.uid].errors.append(f"OpenAlex: {e}")

    for s in studies:
        _fill_metadata(s, dossiers[s.uid])

    # Registries (ClinicalTrials.gov automatically)
    ncts = [(s, t) for s in studies for t in s.trial_ids if t.startswith("ClinicalTrials.gov")]
    for i, (s, t) in enumerate(ncts):
        progress(i / max(len(ncts), 1), f"ClinicalTrials.gov: {i + 1}/{len(ncts)}")
        try:
            rec = registries.ctgov_study(t.split(": ")[1])
            if rec:
                dossiers[s.uid].registrations[t] = rec
                if "ClinicalTrials.gov" not in dossiers[s.uid].sources_checked:
                    dossiers[s.uid].sources_checked.append("ClinicalTrials.gov")
        except SourceError as e:
            dossiers[s.uid].errors.append(f"ClinicalTrials.gov: {e}")

    # Author track records via OpenAlex (disambiguated author IDs) for first and last authors
    progress(0.0, "OpenAlex: research team track record")
    wanted: dict[str, list[tuple[str, str]]] = {}
    for s in with_doi:
        w = dossiers[s.uid].openalex
        if not w:
            continue
        auths = openalex.authorship_ids(w)
        if not auths:
            continue
        picks = [auths[0]] + ([auths[-1]] if len(auths) > 1 else [])
        wanted[s.uid] = [(a["id"], a["name"]) for a in picks]
    all_ids = sorted({aid for v in wanted.values() for aid, _ in v})
    if all_ids:
        try:
            res = openalex.retracted_works_by_authors(all_ids)
            for s in with_doi:
                for aid, name in wanted.get(s.uid, []):
                    others = [w for w in res.get(aid, []) if w.get("doi") != s.doi]
                    dossiers[s.uid].author_openalex[name] = others
        except SourceError as e:
            for uid in wanted:
                dossiers[uid].errors.append(f"OpenAlex authors: {e}")


# ---- Retraction Watch matching ------------------------------------------------------

def _family(name: str) -> str:
    return strip_accents(split_name(name)[0]).lower()


def _corroborate(s: Study, rec: dict) -> bool:
    ok_author = False
    if s.first_author:
        fam = _family(s.first_author)
        ok_author = bool(fam) and fam in strip_accents(str(rec["authors"])).lower()
    ok_year = False
    if s.year and rec.get("paper_date"):
        ok_year = abs(int(rec["paper_date"][:4]) - int(s.year)) <= 1
    return ok_author or ok_year


def match_rw(studies: list[Study], dossiers: dict[str, Dossier], rw: RWIndex, progress: Progress = _noop):
    unmatched: list[Study] = []
    for s in studies:
        d = dossiers[s.uid]
        d.sources_checked.append("Retraction Watch")
        seen = set()
        for r in rw.find_by_doi(s.doi):
            d.rw.append(RWMatch(r, "doi"))
            seen.add(r)
        for r in rw.find_by_pmid(s.pmid):
            if r in seen:
                continue
            rec = rw.record(r)
            sim = _title_sim(s.title, rec["title"]) if s.title else 100.0
            d.rw.append(RWMatch(r, "pmid", sim, sim >= 80, "definite" if sim >= 80 else "possible"))
            seen.add(r)
        for r in rw.find_by_title(s.title):
            if r in seen:
                continue
            rec = rw.record(r)
            # A DOI that disagrees with RW's DOI means a different paper with the same title
            if s.doi and rec["paper_doi"] and rec["paper_doi"] != s.doi:
                continue
            corr = _corroborate(s, rec)
            d.rw.append(RWMatch(r, "title_exact", 100.0, corr, "definite" if corr else "probable"))
            seen.add(r)
        if not d.rw and normalize_title(s.title) and len(normalize_title(s.title)) >= 20:
            unmatched.append(s)

    if unmatched and rw.titles:
        progress(0.0, f"Fuzzy title matching for {len(unmatched)} studies")
        queries = [normalize_title(s.title) for s in unmatched]
        scores = process.cdist(queries, rw.titles, scorer=fuzz.token_sort_ratio,
                               score_cutoff=FUZZY_CUTOFF, workers=-1, dtype=np.uint8)
        for qi, s in enumerate(unmatched):
            row = scores[qi]
            best = int(row.argmax())
            score = float(row[best])
            if score < FUZZY_CUTOFF:
                continue
            for r in rw.by_title[rw.titles[best]]:
                rec = rw.record(r)
                if s.doi and rec["paper_doi"] and rec["paper_doi"] != s.doi:
                    continue
                corr = _corroborate(s, rec)
                strength = "probable" if (score >= FUZZY_HIGH and corr) else "possible"
                dossiers[s.uid].rw.append(RWMatch(r, "title_fuzzy", score, corr, strength))

    # Research team: first and last author in RW (excluding the study's own records)
    for s in studies:
        d = dossiers[s.uid]
        own = {m.row for m in d.rw}
        for name in filter(None, [s.first_author, s.last_author]):
            rows, spec = rw.find_by_author(name)
            rows = [r for r in rows if r not in own]
            d.author_rw[name] = {"rows": rows, "specificity": spec}


# ---- run ---------------------------------------------------------------------------

def _merge_duplicates(studies: list[Study], dossiers: dict, info: RunInfo):
    """After identification, references from different files may turn out to be the same paper."""
    seen: dict[str, Study] = {}
    keep = []
    for s in studies:
        keys = [k for k in (f"doi:{s.doi}" if s.doi else None, f"pmid:{s.pmid}" if s.pmid else None) if k]
        first = next((seen[k] for k in keys if k in seen), None)
        if first:
            info.duplicates.append(f"{s.uid} ({s.source_file}) is the same paper as {first.uid} ({first.source_file}); merged.")
            first.id_notes.append(f"Also supplied as {s.uid} in {s.source_file}.")
            dossiers.pop(s.uid, None)
            continue
        for k in keys:
            seen[k] = s
        keep.append(s)
    studies[:] = keep


def fetch_notice_titles(studies: list[Study], dossiers: dict, rw: RWIndex, progress: Progress = _noop):
    """Titles and citations of the notices themselves (retraction notes, EoCs, errata), fetched in batches.
    The full wording of a notice is only on the journal's website; databases keep its title and citation."""
    refs = pubmed.RETRACTION_REFS | pubmed.CONCERN_REFS | pubmed.CORRECTION_REFS
    pmids, dois = set(), set()
    for s in studies:
        d = dossiers[s.uid]
        pmids |= {l["pmid"] for l in (d.pubmed or {}).get("links", []) if l["pmid"] and l["ref_type"] in refs}
        dois |= {u["doi"] for u in (d.crossref or {}).get("updated_by", []) if u.get("doi")}
        dois |= {rw.record(m.row)["notice_doi"] for m in d.rw if rw.record(m.row)["notice_doi"]}
    if not pmids and not dois:
        return
    progress(0.0, f"Fetching titles of {len(pmids | dois)} notices")
    info = {}
    try:
        for pmid, rec in pubmed.fetch(sorted(pmids)).items():
            cit = " ".join(filter(None, [rec.get("journal"), str(rec.get("year") or "")]))
            info[f"pmid:{pmid}"] = {"title": rec.get("title") or "", "citation": cit}
            if rec.get("doi"):
                info.setdefault(f"doi:{rec['doi']}", info[f"pmid:{pmid}"])
    except Exception as e:
        info["_error"] = str(e)
    try:
        for doi, it in crossref.works_by_doi(sorted(dois)).items():
            issued = ((it.get("issued") or {}).get("date-parts") or [[None]])[0][0]
            cit = " ".join(filter(None, [(it.get("container-title") or [""])[0], str(issued or "")]))
            info.setdefault(f"doi:{doi}", {"title": (it.get("title") or [""])[0], "citation": cit})
    except SourceError as e:
        info["_error"] = str(e)
    for s in studies:
        dossiers[s.uid].notice_info = info


def identify_from_rw(studies: list[Study], rw: RWIndex):
    """First pass, local and instant: a reference with no DOI/PMID whose exact title is in Retraction Watch
    (confirmed by first author or year) takes its identifiers from the Retraction Watch record."""
    for s in studies:
        if s.identified or not s.title:
            continue
        for r in rw.find_by_title(s.title):
            rec = rw.record(r)
            if (rec["paper_doi"] or rec["paper_pmid"]) and _corroborate(s, rec):
                s.doi = s.doi or rec["paper_doi"]
                s.pmid = s.pmid or rec["paper_pmid"]
                s.id_method, s.id_confidence = "Retraction Watch title match", "high"
                s.id_notes.append("Identifiers taken from the matching Retraction Watch record (exact title, "
                                  "confirmed by author or year).")
                break


def run(studies: list[Study], rw: RWIndex, online: bool = True, progress: Progress = _noop,
        use_crossref: bool = False) -> tuple[dict, RunInfo]:
    info = RunInfo(started=now_utc(), online=online)
    dossiers = {s.uid: Dossier() for s in studies}
    info.sources["Retraction Watch"] = (
        f"{rw.meta.get('source')}; file downloaded {rw.meta.get('downloaded_on')}; "
        f"{rw.meta.get('n_records'):,} records; latest notice {rw.meta.get('latest_notice')}"
    )
    identify_from_rw(studies, rw)
    if online:
        identify(studies, dossiers, progress, use_crossref=use_crossref)
        _merge_duplicates(studies, dossiers, info)
        gather_online(studies, dossiers, progress)
        info.sources["Crossref"] = f"REST API (works, updated-by, citation matching), queried {now_utc()}"
        info.sources["PubMed"] = f"NCBI E-utilities (publication types, linked notices, databank IDs), queried {now_utc()}"
        info.sources["OpenAlex"] = f"API (is_retracted, author-disambiguated retractions), queried {now_utc()}"
        info.sources["ClinicalTrials.gov"] = f"API v2 (registration and start dates), queried {now_utc()}"
    else:
        for s in studies:
            s.design = detect_design(s.title, s.abstract, s.pub_types)
    match_rw(studies, dossiers, rw, progress)
    if online:
        fetch_notice_titles(studies, dossiers, rw, progress)
    info.errors = sorted({e for d in dossiers.values() for e in d.errors})
    info.finished = now_utc()
    return dossiers, info
