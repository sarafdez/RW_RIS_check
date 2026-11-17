"""INSPECT-SR check modules.

Each automated check turns a study's Dossier into a *proposed* answer with evidence. The reviewer
always has the last word: INSPECT-SR is a judgement-based tool and the guidance gives no algorithm.

To add a check: write a function `(study, dossier, rw) -> CheckResult` and register it in AUTOMATED.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Callable

from .models import Answer, CheckResult, Evidence, Study
from .normalize import parse_date
from .pipeline import Dossier
from .sources import crossref, pubmed
from .sources.registries import registry_url
from .sources.retraction_watch import HONEST_ERROR_TERMS, RWIndex, integrity_related

DOI = "https://doi.org/{}"


def _ntext(d: Dossier, doi: str | None = None, pmid: str | None = None) -> str:
    n = (d.notice_info.get(f"pmid:{pmid}") if pmid else None) or (d.notice_info.get(f"doi:{doi}") if doi else None)
    if not n or not n.get("title"):
        return ""
    return n["title"] + (f" ({n['citation']})" if n.get("citation") else "")


def _cr_list_evidence(d: Dossier, retraction: bool) -> list[Evidence]:
    """Notices from the downloaded Crossref list (retraction-type or expression of concern)."""
    out = []
    for n in d.cr_list:
        is_ret = n["type"] != "expression_of_concern"
        if is_ret != retraction:
            continue
        out.append(Evidence("Crossref", n["label"], f"Asserted by: {n['source']}", n["date"],
                            DOI.format(n["notice_doi"]) if n["notice_doi"] else None, n["match"],
                            text=n["notice_title"] or ""))
    return out


def _listed_notices(s: Study, ntype: str) -> list[Evidence]:
    """Notices that were in the reference list and merged into this study."""
    return [Evidence("Your reference list", n["type"], "This notice was also in your reference list (merged into the study)",
                     n.get("date"), DOI.format(n["doi"]) if n.get("doi") else None, "your list", text=n.get("title") or "")
            for n in s.merged_notices if n["type"] == ntype]


def _rw_evidence(rec: dict, match, d: Dossier | None = None) -> Evidence:
    reasons = "; ".join(rec["reasons"][:6])
    url = DOI.format(rec["notice_doi"]) if rec["notice_doi"] else (rec["urls"][0] if rec["urls"] else None)
    how = match.method + (f" ({match.score:.0f}%)" if match.method == "title_fuzzy" else "")
    return Evidence("Retraction Watch", rec["nature"], f"{reasons}" + (f" | Note: {rec['notes'][:400]}" if rec["notes"] else ""),
                    rec["notice_date"], url, how, text=_ntext(d, doi=rec["notice_doi"]) if d else "")


def _not_identified_note(s: Study) -> str:
    if s.identified:
        return ""
    return " The study has no DOI or PubMed ID, so only title matching was possible."


# ---- 1.1 Retraction --------------------------------------------------------------

def check_1_1(s: Study, d: Dossier, rw: RWIndex) -> CheckResult:
    ev, possible = [], []
    replace = False
    reinstated = False
    for m in d.rw:
        rec = rw.record(m.row)
        if rec["nature"] == "Retraction":
            (ev if m.strength in ("definite", "probable") else possible).append(_rw_evidence(rec, m, d))
            if any("retract and replace" in r.lower() for r in rec["reasons"]):
                replace = True
        elif rec["nature"] == "Reinstatement":
            reinstated = True
            ev.append(_rw_evidence(rec, m, d))
    for u in (d.crossref or {}).get("updated_by", []):
        if u["type"] in crossref.RETRACTION_TYPES:
            ev.append(Evidence("Crossref", u["label"], f"Asserted by: {u['asserted_by']}", u["date"],
                               DOI.format(u["doi"]) if u["doi"] else None, "doi", text=_ntext(d, doi=u["doi"])))
    pm = d.pubmed or {}
    if "Retracted Publication" in pm.get("pub_types", []):
        ev.append(Evidence("PubMed", "Retracted Publication", "PubMed publication type", None,
                           pubmed.pubmed_url(pm["pmid"]), "pmid"))
    for link in pm.get("links", []):
        if link["ref_type"] in pubmed.RETRACTION_REFS:
            if link["ref_type"] == "RetractedandRepublishedIn":
                replace = True
            ev.append(Evidence("PubMed", pubmed.ref_type_label(link["ref_type"]), link["source"],
                               str(link["year"]) if link["year"] else None,
                               pubmed.pubmed_url(link["pmid"]) if link["pmid"] else None, "pmid",
                               text=_ntext(d, pmid=link["pmid"])))
    ev += _listed_notices(s, "Retraction")
    if d.crossref is None:  # per-DOI Crossref data (Extended) already covers these
        ev += _cr_list_evidence(d, retraction=True)
    oa = d.openalex or {}
    if oa.get("is_retracted"):
        ev.append(Evidence("OpenAlex", "Retracted (is_retracted)", "", None, oa.get("id"), "doi"))

    if s.design == "Notice (not a study)":
        return _notice_record(s, d)
    if ev and not (reinstated and len(ev) == sum(e.kind == "Reinstatement" for e in ev)):
        n_src = len({e.source for e in ev})
        if replace:
            return CheckResult("1.1", Answer.YES,
                               f"Retraction found ({n_src} source(s)), but it is a retract-and-replace: the "
                               "replacement version can be assessed instead.", ev, True, data={"kind": "replace"})
        if reinstated:
            return CheckResult("1.1", Answer.YES, "Retraction found, but Retraction Watch also lists a reinstatement. "
                               "Read the notices.", ev, True, data={"kind": "reinstated"})
        return CheckResult("1.1", Answer.YES,
                           f"Retraction found in {n_src} source(s): {', '.join(sorted({e.source for e in ev}))}. "
                           "A retraction would typically warrant 'serious concerns' for the study, whatever the reason.", ev, needs_review=False, suggests_serious=True,
                           data={"kind": "retraction"})
    if possible:
        return CheckResult("1.1", Answer.UNCLEAR, "Possible retraction: a similar title is in Retraction Watch but the "
                           "match could not be confirmed by DOI/PMID, author or year. Verify manually." +
                           _not_identified_note(s), possible, True, data={"kind": "possible"})
    srcs = ", ".join(dict.fromkeys(d.sources_checked)) or "Retraction Watch"
    if not s.identified and not s.title:
        return CheckResult("1.1", Answer.UNCLEAR, "The study could not be identified (no DOI, PMID or title).", [], True,
                           data={"kind": "unidentified"})
    return CheckResult("1.1", Answer.NO if s.identified else Answer.UNCLEAR,
                       f"No retraction found in: {srcs}." + _not_identified_note(s) +
                       " Tip: notices can appear late, so repeat the check shortly before finishing the review.", [], needs_review=False, data={"kind": "none"})


def _notice_record(s: Study, d: Dossier) -> CheckResult:
    """The reference in the list is itself a notice (corrigendum, erratum, retraction notice, EoC).
    That means the study it belongs to has that notice: report it as such, and name the original if known."""
    from .normalize import notice_kind
    ntype = notice_kind(s.title, s.pub_types, (s.notice_for or {}).get("nature"))
    original, orig_url = None, None
    if s.notice_for and (s.notice_for.get("doi") or s.notice_for.get("pmid")):
        nf = s.notice_for
        original = (f"“{nf['title']}”, " if nf.get("title") else "") + (f"DOI {nf['doi']}" if nf.get("doi")
                                                                        else f"PMID {nf['pmid']}")
        orig_url = DOI.format(nf["doi"]) if nf.get("doi") else pubmed.pubmed_url(nf["pmid"])
    for doi in ([] if original else (d.crossref or {}).get("is_notice_for", []) or []):
        original, orig_url = f"DOI {doi}", DOI.format(doi)
        break
    if not original:
        for link in (d.pubmed or {}).get("links", []):
            if link["ref_type"] in ("ErratumFor", "RetractionOf", "ExpressionOfConcernFor", "CorrectedandRepublishedFrom"):
                original = link["source"] or (f"PMID {link['pmid']}" if link["pmid"] else None)
                orig_url = pubmed.pubmed_url(link["pmid"]) if link["pmid"] else None
                break
    ev = [Evidence("Your reference list", ntype, "The reference itself is the notice", s.pub_date,
                   DOI.format(s.doi) if s.doi else None, "title", text=s.title or "")]
    if original:
        ev.append(Evidence("PubMed" if orig_url and "pubmed" in orig_url else "Crossref", "Original article",
                           original, None, orig_url, "notice link"))
    return CheckResult("1.1", Answer.YES if ntype == "Retraction" else Answer.NO,
                       f"This reference is the {ntype.lower()} notice itself, not the study."
                       + (f" It belongs to: {original}." if original else ""), ev, True,
                       suggests_serious=ntype == "Retraction",
                       data={"kind": "notice_record", "notice_type": ntype, "original": original,
                             "original_url": orig_url})


# ---- 1.2 Expressions of concern and other notices ------------------------------------

def check_1_2(s: Study, d: Dossier, rw: RWIndex) -> CheckResult:
    eoc, other, comments, possible = [], [], [], []
    lifted = False
    for m in d.rw:
        rec = rw.record(m.row)
        if rec["nature"] in ("Expression of concern", "Correction"):
            target = eoc if rec["nature"] == "Expression of concern" else other
            (target if m.strength in ("definite", "probable") else possible).append(_rw_evidence(rec, m, d))
            lifted |= any("eoc lifted" in r.lower() for r in rec["reasons"])
    for u in (d.crossref or {}).get("updated_by", []):
        e = Evidence("Crossref", u["label"], f"Asserted by: {u['asserted_by']}", u["date"],
                     DOI.format(u["doi"]) if u["doi"] else None, "doi", text=_ntext(d, doi=u["doi"]))
        if u["type"] in crossref.CONCERN_TYPES:
            eoc.append(e)
        elif u["type"] in crossref.CORRECTION_TYPES:
            other.append(e)
    for link in (d.pubmed or {}).get("links", []):
        e = Evidence("PubMed", pubmed.ref_type_label(link["ref_type"]), link["source"],
                     str(link["year"]) if link["year"] else None,
                     pubmed.pubmed_url(link["pmid"]) if link["pmid"] else None, "pmid",
                     text=_ntext(d, pmid=link["pmid"]))
        if link["ref_type"] in pubmed.CONCERN_REFS:
            eoc.append(e)
        elif link["ref_type"] in pubmed.CORRECTION_REFS:
            other.append(e)
        elif link["ref_type"] in pubmed.COMMENT_REFS:
            comments.append(e)

    eoc += _listed_notices(s, "Expression of concern")
    if d.crossref is None:
        eoc += _cr_list_evidence(d, retraction=False)
    other += _listed_notices(s, "Correction")
    if eoc:
        msg = "Expression of concern found. Read the notice: is an investigation ongoing? Check the journal site again before finishing the review."
        if lifted:
            msg += " Retraction Watch indicates the expression of concern was later lifted."
        return CheckResult("1.2", Answer.YES, msg, eoc + other + comments, needs_review=True, data={"kind": "eoc"})
    if other:
        return CheckResult("1.2", Answer.UNCLEAR,
                           f"{len(other)} correction/erratum notice(s) found. Corrections are often honest errors; read "
                           "each notice to judge whether it is relevant to trustworthiness.", other + comments, True,
                           data={"kind": "correction"})
    if possible:
        return CheckResult("1.2", Answer.UNCLEAR, "Possible notice found via fuzzy title match; verify.", possible, True,
                           data={"kind": "possible"})
    note = ""
    if comments:
        note = (f" {len(comments)} comment(s)/letter(s) are linked in PubMed (letters or "
                "commentaries). These are not notices, but can be worth a look.")
    return CheckResult("1.2", Answer.NO if s.identified else Answer.UNCLEAR,
                       "No expression of concern or correction found." + note + _not_identified_note(s) +
                       " Also consider PubPeer comments (link on the study page).", comments,
                       needs_review=False, data={"kind": "none", "comments": len(comments)})


# ---- 1.3 Other studies by the research team -------------------------------------------

def author_key(name: str) -> str:
    """One key per person-name, shared by "Schietroma, Mario" and "Mario Schietroma"."""
    from .normalize import author_keys
    k = author_keys(name or "")
    return k.get("full") or k.get("initial") or (name or "").strip().lower()


def check_1_3(s: Study, d: Dossier, rw: RWIndex) -> CheckResult:
    if not s.authors:
        return CheckResult("1.3", Answer.UNCLEAR, "No author names available, so this could not be checked.", [], True)
    ev: list[Evidence] = []
    strong, weak = 0, 0
    lines, triggers = [], []
    from .normalize import author_keys
    people: dict[str, dict] = {}  # structured findings per author, for the Authors review page

    def person(name: str, role: str) -> dict:
        k = author_key(name)
        if k not in people:
            people[k] = {"key": k, "name": name, "role": role, "match": "", "strong": False, "papers": []}
        return people[k]

    def _key(name: str) -> str | None:
        return author_keys(name).get("full")  # full first names only: initials are too common to prove identity

    own_keys_cache: dict = {}

    def coauthor_overlap(rec: dict, name: str) -> bool:
        """Co-authors of the study are also on the retracted paper: a strong sign it is the same research team, not a
        namesake. One shared co-author by full name is enough; with initials only (e.g. pasted references) at least
        two shared co-authors (surname + initial) are required, since initials are common."""
        if name not in own_keys_cache:
            ok = author_keys(name)
            ours_full = {k for a in s.authors if (k := author_keys(a).get("full")) and k != ok.get("full")}
            ours_init = {k for a in s.authors if (k := author_keys(a).get("initial")) and k != ok.get("initial")}
            own_keys_cache[name] = (ours_full, ours_init)
        ours_full, ours_init = own_keys_cache[name]
        theirs_full, theirs_init = set(), set()
        for x in str(rec["authors"]).split(";"):
            k = author_keys(x.strip())
            if k.get("full"):
                theirs_full.add(k["full"])
            if k.get("initial"):
                theirs_init.add(k["initial"])
        return bool(ours_full & theirs_full) or len(ours_init & theirs_init) >= 2

    for name, hit in d.author_rw.items():
        recs = [rw.record(r) for r in hit["rows"]]
        recs = [r for r in recs if r["nature"] in ("Retraction", "Expression of concern")]
        if not recs:
            continue
        integ = [r for r in recs if any(integrity_related(x) for x in r["reasons"])]
        specific = hit["specificity"] == "full name"
        shared = [r for r in integ if coauthor_overlap(r, name)]
        if shared:
            strong += 1
            triggers.append(f"{name} has {len(shared)} other integrity-related retraction(s) in Retraction Watch "
                            f"on papers shared with a co-author of this study.")
        else:
            weak += 1
        role = "first author" if name == s.first_author else "last author"
        p = person(name, role)
        p["strong"] = p["strong"] or bool(shared)
        p["match"] = "shared co-author" if shared else f"name only ({hit['specificity']})"
        shared_ids = {id(r) for r in shared}
        for r in recs:
            p["papers"].append({"title": r["title"], "nature": r["nature"], "date": r["notice_date"],
                                "reasons": r["reasons"], "integrity": any(integrity_related(x) for x in r["reasons"]),
                                "shared_coauthor": id(r) in shared_ids, "doi": r["paper_doi"],
                                "notice": DOI.format(r["notice_doi"]) if r["notice_doi"] else None,
                                "source": "Retraction Watch"})
        lines.append(f"{name} ({role}): {len(recs)} retraction/EoC record(s) in Retraction Watch matched by "
                     f"{hit['specificity']}; {len(integ)} with integrity-related reasons"
                     + (f", {len(shared)} sharing a co-author with this study." if shared else "."))
        for r in sorted(recs, key=lambda r: r["notice_date"] or "", reverse=True)[:8]:
            ev.append(Evidence("Retraction Watch", f"{r['nature']} ({role})",
                               f"{r['title'][:140]} — {'; '.join(r['reasons'][:4])}",
                               r["notice_date"], DOI.format(r["notice_doi"]) if r["notice_doi"] else None,
                               f"author:{hit['specificity']}"))
    oa_hits = 0
    for name, works in d.author_openalex.items():
        if not works:
            continue
        # Cross-check each retracted work in Retraction Watch: only integrity-related retractions count
        # (a retraction for honest error is not a warning sign about the team).
        integ_works = []
        for w in works:
            recs = [rw.record(r) for r in rw.find_by_doi(w.get("doi"))]
            reasons = sorted({x for r in recs for x in r["reasons"]})
            w = {**w, "reasons": reasons, "integrity": any(integrity_related(x) for x in reasons)}
            integ_works.append(w)
        n_int = sum(w["integrity"] for w in integ_works)
        p = person(name, "first author" if author_key(name) == author_key(s.first_author or "") else "last author")
        known = {x["doi"] for x in p["papers"] if x.get("doi")}
        for w in integ_works:
            if w.get("doi") in known:
                continue
            p["papers"].append({"title": w.get("title"), "nature": "Retraction", "date": str(w.get("year") or ""),
                                "reasons": w["reasons"], "integrity": w["integrity"], "shared_coauthor": False,
                                "doi": w.get("doi"), "notice": DOI.format(w["doi"]) if w.get("doi") else w.get("openalex"),
                                "source": "OpenAlex"})
        if n_int:
            p["strong"] = True
            p["match"] = "same OpenAlex author ID"
        if n_int:
            oa_hits += 1
            triggers.append(f"{name} has {n_int} other integrity-related retraction(s) (same OpenAlex author ID).")
        lines.append(f"{name}: {len(works)} other retracted work(s) in OpenAlex (matched by author ID), "
                     f"{n_int} with integrity-related reasons in Retraction Watch.")
        for w in sorted(integ_works, key=lambda w: not w["integrity"])[:8]:
            ev.append(Evidence("OpenAlex", "Retracted work by same author",
                               f"{w['title'] or ''}"[:140] + (f" — {'; '.join(w['reasons'][:3])}" if w["reasons"] else ""),
                               str(w.get("year") or ""), DOI.format(w["doi"]) if w.get("doi") else w.get("openalex"),
                               "author:openalex id"))
    if strong or oa_hits:
        return CheckResult("1.3", Answer.YES,
                           " ".join(lines) + " Confirm that these are the same people and read the notices: a retraction "
                           "for honest error may not warrant concern.", ev, needs_review=True,
                           data={"triggers": triggers, "authors": list(people.values())})
    if weak:
        return CheckResult("1.3", Answer.UNCLEAR,
                           " ".join(lines) + " Matches are by name only (possibly other people with the same name) "
                           "or not integrity-related, so the study is not flagged.", ev, needs_review=False,
                           data={"authors": list(people.values())})
    checked = ", ".join(d.author_rw.keys())
    return CheckResult("1.3", Answer.NO,
                       f"No retractions or expressions of concern found for first/last author ({checked}). "
                       "Only the first and last authors are checked.", [])


# ---- 2.2 Registration timing / absence -----------------------------------------------

def _lag_days(start, submitted) -> int | None:
    a, b = parse_date(start), parse_date(submitted)
    if not a or not b:
        return None
    return (b - a).days


def check_2_2(s: Study, d: Dossier, rw: RWIndex) -> CheckResult:
    if s.design in ("Systematic review", "Notice (not a study)"):
        return CheckResult("2.2", Answer.NA, f"Detected design: {s.design}.", [], False)
    if not s.trial_ids:
        return CheckResult("2.2", Answer.UNCLEAR,
                           "No trial registration number found in the reference, abstract or PubMed record. Check the "
                           "full text; if the trial was not registered, the answer is 'Yes'.", [], True)
    ev, worst, msgs, lags = [], Answer.NO, [], {}
    order = {Answer.NO: 0, Answer.UNCLEAR: 1, Answer.YES: 2}
    pub = parse_date(s.pub_date) or (date(int(s.year), 12, 31) if s.year else None)
    for t in s.trial_ids:
        rec = d.registrations.get(t)
        if not rec:
            ev.append(Evidence(t.split(": ")[0], "Registration (not checked automatically)", t, None, registry_url(t), "trial id"))
            msgs.append(f"{t}: check registration date against start of recruitment manually.")
            worst = max(worst, Answer.UNCLEAR, key=order.get)
            continue
        lag = _lag_days(rec["start_date"], rec["first_submitted"])
        lags[t] = lag
        detail = (f"First submitted {rec['first_submitted']}; study start {rec['start_date']} ({rec['start_type'] or 'n/a'}); "
                  f"status {rec['status']}; enrolment {rec['enrollment']}")
        ev.append(Evidence("ClinicalTrials.gov", "Registration", detail, rec["first_submitted"], rec["history_url"], "trial id"))
        sub = parse_date(rec["first_submitted"])
        if pub and sub and sub > pub:
            msgs.append(f"{t}: registered after publication.")
            worst = Answer.YES
        elif lag is None:
            msgs.append(f"{t}: dates incomplete.")
            worst = max(worst, Answer.UNCLEAR, key=order.get)
        elif lag <= 30:
            msgs.append(f"{t}: registered {'before' if lag <= 0 else f'{lag} days after'} the study start (prospective or near-prospective).")
        elif lag <= 180:
            msgs.append(f"{t}: registered {lag} days after the study start. A short delay is common "
                        "and usually not a concern. If the trial is also registered elsewhere (e.g. ISRCTN), the earliest "
                        "registration counts.")
            worst = max(worst, Answer.UNCLEAR, key=order.get)
        else:
            msgs.append(f"{t}: registered {lag} days (~{lag / 365:.1f} years) after the study start (retrospective).")
            worst = Answer.YES
    rationale = " ".join(msgs) + (" Start dates on ClinicalTrials.gov can change; compare with the first version in the "
                                  "record history." if d.registrations else "")
    return CheckResult("2.2", worst, rationale, ev, needs_review=worst != Answer.NO, data={"lag_days": lags})


AUTOMATED: dict[str, Callable[[Study, Dossier, RWIndex], CheckResult]] = {
    "1.1": check_1_1,
    "1.2": check_1_2,
    "1.3": check_1_3,
    "2.2": check_2_2,
}


def run_checks(s: Study, d: Dossier, rw: RWIndex) -> dict[str, CheckResult]:
    return {cid: fn(s, d, rw) for cid, fn in AUTOMATED.items()}
