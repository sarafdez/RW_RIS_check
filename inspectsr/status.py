"""Plain-language status per study, and one consolidated list of notices per study.

A study can have several flags (e.g. an expression of concern that later became a retraction); its status
is the most severe one, and every flag comes with a short 'why' for the reviewer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from .models import Answer, CheckResult, Study
from .normalize import parse_date

# ordered from most to least severe: (label, colour, icon, group)
STATUSES = [
    ("Retracted", "#d03b3b", "✕", "retracted"),
    ("Retracted and replaced", "#d03b3b", "✕", "retracted"),
    ("Expression of concern", "#ec835a", "!", "concern"),
    ("Possible match", "#898781", "?", "concern"),
    ("Not a study record", "#898781", "?", "concern"),
    ("Correction", "#fab219", "!", "concern"),
    ("Authors' other retractions", "#fab219", "!", "concern"),
    ("Late registration", "#fab219", "!", "concern"),
    ("No notices found", "#0ca30c", "✓", "none"),
]
STATUS_INFO = {s[0]: s for s in STATUSES}
ORDER = {s[0]: i for i, s in enumerate(STATUSES)}
NONE = "No notices found"

DECISIONS = ["Not yet reviewed", "Confirmed", "False match – ignore"]


@dataclass
class Flag:
    status: str
    why: str
    check: str


@dataclass
class Notice:
    type: str
    date: Optional[str]
    sources: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    note: str = ""
    possible: bool = False

    @property
    def sources_label(self) -> str:
        return ", ".join(self.sources)


def event_type(kind: str) -> Optional[str]:
    k = (kind or "").lower()
    if "reinstat" in k:
        return "Reinstatement"
    if "retract" in k or "withdraw" in k or "removal" in k:
        return "Retraction"
    if "concern" in k:
        return "Expression of concern"
    if "correct" in k or "errat" in k or "corrig" in k or "addend" in k or "clarif" in k:
        return "Correction"
    return None


def _days(a: str, b: str) -> Optional[int]:
    da, db = parse_date(a), parse_date(b)
    return abs((da - db).days) if da and db else None


def notices(results: dict[str, CheckResult]) -> list[Notice]:
    """Merge the same notice reported by several sources into one row (same type, dates within 60 days)."""
    items = []
    for cid in ("1.1", "1.2"):
        r = results.get(cid)
        if not r:
            continue
        possible = r.data.get("kind") == "possible"
        for e in r.evidence:
            t = event_type(e.kind)
            if t:
                items.append((t, e, possible))
    out: list[Notice] = []
    full = [x for x in items if x[1].date and len(x[1].date) >= 10]
    partial = [x for x in items if not (x[1].date and len(x[1].date) >= 10)]
    for t, e, possible in sorted(full, key=lambda x: x[1].date):
        n = next((n for n in out if n.type == t and n.date and (_days(n.date, e.date) or 0) <= 60), None)
        if n is None:
            n = Notice(t, e.date, possible=possible)
            out.append(n)
        _add(n, e)
    for t, e, possible in partial:
        year = (e.date or "")[:4]
        n = next((n for n in out if n.type == t and (not year or (n.date or "").startswith(year))), None) \
            or next((n for n in out if n.type == t), None)
        if n is None:
            n = Notice(t, e.date, possible=possible)
            out.append(n)
        _add(n, e)
    return sorted(out, key=lambda n: n.date or "9999")


def _add(n: Notice, e):
    src = e.source if e.source != "Crossref" or "retraction-watch" not in e.detail else "Crossref (from RW)"
    if src not in n.sources:
        n.sources.append(src)
    if e.url and e.url not in n.links:
        n.links.append(e.url)
    if e.source == "Retraction Watch":
        reasons, _, note = e.detail.partition(" | Note: ")
        for x in reasons.split("; "):
            x = x.strip()
            if x and x not in n.reasons and not x.startswith(("Date of Article", "Upgrade/Update", "Updated to", "Notice -")):
                n.reasons.append(x)
        n.note = n.note or note


def flags(s: Study, results: dict[str, CheckResult]) -> list[Flag]:
    out: list[Flag] = []
    ns = notices(results)

    def first(t):
        return next((n for n in ns if n.type == t and not n.possible), None)

    r11, r12 = results.get("1.1"), results.get("1.2")
    k11 = r11.data.get("kind") if r11 else None
    k12 = r12.data.get("kind") if r12 else None
    if k11 in ("retraction", "reinstated"):
        n = first("Retraction")
        why = f"Retraction notice{(' ' + n.date[:10]) if n and n.date else ''} ({n.sources_label if n else 'see evidence'})."
        if n and n.reasons:
            why += f" Reasons: {'; '.join(n.reasons[:3])}."
        if k11 == "reinstated":
            why += " Retraction Watch also lists a reinstatement."
        out.append(Flag("Retracted", why, "1.1"))
    elif k11 == "replace":
        out.append(Flag("Retracted and replaced", "Retracted and replaced by a corrected version: check the "
                        "replacement article instead.", "1.1"))
    elif k11 == "notice_record":
        out.append(Flag("Not a study record", "This reference looks like a retraction or correction notice, not the "
                        "study itself. Replace it with the original article.", "1.1"))
    if k12 == "eoc":
        n = first("Expression of concern")
        why = f"Expression of concern{(' ' + n.date[:10]) if n and n.date else ''} ({n.sources_label if n else 'see evidence'})."
        if n and n.reasons:
            why += f" Reasons: {'; '.join(n.reasons[:3])}."
        out.append(Flag("Expression of concern", why, "1.2"))
    elif k12 == "correction":
        cs = [n for n in ns if n.type == "Correction"]
        out.append(Flag("Correction", f"{len(cs)} correction notice(s). Usually an honest error: read the notice to "
                        "judge whether it affects the results.", "1.2"))
    if k11 == "possible" or k12 == "possible":
        poss = [n for n in ns if n.possible]
        kinds = ", ".join(sorted({n.type.lower() for n in poss})) or "notice"
        out.append(Flag("Possible match", f"A similar title has a {kinds} in Retraction Watch, but DOI, author or year "
                        "do not confirm it is the same paper. Compare the two records.", "1.1"))
    r13 = results.get("1.3")
    if r13 and r13.proposed == Answer.YES:
        trig = r13.data.get("triggers") or ["The first or last author has other retracted papers."]
        out.append(Flag("Authors' other retractions", " ".join(trig) + " Check it is the same person.", "1.3"))
    r22 = results.get("2.2")
    if r22 and r22.proposed == Answer.YES:
        lag = max((v for v in r22.data.get("lag_days", {}).values() if v is not None), default=None)
        why = (f"Trial registered {lag / 365.25:.1f} years after the study started." if lag and lag > 0
               else "Trial registered after publication or long after the study started.")
        out.append(Flag("Late registration", why, "2.2"))
    return sorted(out, key=lambda f: ORDER[f.status])


def study_status(s: Study, results: dict[str, CheckResult], decision: Optional[dict] = None) -> tuple[str, list[Flag]]:
    fl = flags(s, results)
    if decision and decision.get("decision") == DECISIONS[2]:
        return NONE, fl
    return (fl[0].status if fl else NONE), fl


def group(status: str) -> str:
    return STATUS_INFO[status][3]


def identification_label(s: Study) -> str:
    m = s.id_method
    if m == "as supplied" and s.identified:
        return "DOI/PMID in your file"
    if not s.identified:
        return "Not identified (title only)"
    return m


def summary_sentences(rows: list[dict]) -> list[str]:
    """rows: one dict per study with Status, Identification, Study."""
    n = len(rows)
    by = {}
    for r in rows:
        by.setdefault(r["Status"], []).append(r)
    retracted = by.get("Retracted", []) + by.get("Retracted and replaced", [])
    out = [f"**{n} studies** were checked against Retraction Watch, Crossref, PubMed and OpenAlex."]
    if retracted:
        names = ", ".join(r["Study"] for r in retracted[:6]) + (" …" if len(retracted) > 6 else "")
        out.append(f"**{len(retracted)} {'is' if len(retracted) == 1 else 'are'} retracted** ({names}).")
    else:
        out.append("**No retracted studies** were found.")
    concern_bits = []
    for st in ("Expression of concern", "Correction", "Possible match", "Authors' other retractions",
               "Late registration", "Not a study record"):
        k = len(by.get(st, []))
        if k:
            concern_bits.append(f"{k} {st.lower()}")
    if concern_bits:
        out.append("Other findings to look at: " + ", ".join(concern_bits) + ".")
    not_id = [r for r in rows if r["Identification"].startswith("Not identified")]
    if not_id:
        out.append(f"{len(not_id)} reference(s) could not be identified (no DOI or PubMed ID found) and were checked "
                   "against Retraction Watch by title only.")
    out.append("No database is complete: a study without notices is not guaranteed to be clean. Repeat the check "
               "shortly before submitting the review.")
    return out
