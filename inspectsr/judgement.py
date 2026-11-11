"""Suggested domain and overall judgements.

INSPECT-SR deliberately has no algorithm for judgements. These suggestions follow the published
guidance as closely as possible and are always shown as *suggestions* the reviewer can override:

* A retraction (check 1.1 'Yes') typically warrants 'serious concerns' for the domain and study.
* Any other 'Yes' contributes to the domain judgement: suggested as 'some concerns'.
* 'Unclear' answers with no 'Yes': the reviewer needs to decide ('Needs reviewer').
* Overall is at least as severe as the most severe domain. If any domain has serious concerns,
  the overall is serious (reviewers may stop there).
"""
from __future__ import annotations

from dataclasses import dataclass

from .framework import DOMAINS, checks_in_domain
from .models import Answer, CheckResult, Judgement


@dataclass
class DomainStatus:
    domain: int
    suggested: Judgement
    final: Judgement
    answered: int
    total: int
    overridden: bool = False
    reason: str = ""

    @property
    def complete(self) -> bool:
        return self.answered == self.total


def final_answer(result: CheckResult, reviewer: dict | None) -> Answer:
    if reviewer and reviewer.get("answer"):
        return Answer(reviewer["answer"])
    return result.proposed


def suggest_domain(domain: int, results: dict[str, CheckResult], reviewer: dict) -> tuple[Judgement, int, int]:
    checks = checks_in_domain(domain)
    answers = {c.id: final_answer(results[c.id], reviewer.get(c.id)) for c in checks}
    answered = sum(a != Answer.NOT_ASSESSED for a in answers.values())
    yes = [cid for cid, a in answers.items() if a == Answer.YES]
    # A confirmed retraction typically warrants serious concerns. Exception the tool allows: a
    # retract-and-replace, where the automated check proposes 'Yes' without suggesting serious.
    r11 = results.get("1.1")
    if "1.1" in yes and r11 is not None and (r11.suggests_serious or r11.proposed != Answer.YES):
        return Judgement.SERIOUS_CONCERNS, answered, len(checks)
    if yes:
        return Judgement.SOME_CONCERNS, answered, len(checks)
    if any(a == Answer.UNCLEAR for a in answers.values()):
        return Judgement.PENDING, answered, len(checks)
    if answered == 0:
        return Judgement.NOT_ASSESSED, answered, len(checks)
    if answered < len(checks):
        # Partly assessed with no problems found so far: don't claim 'no concerns' for the whole domain.
        return Judgement.NOT_ASSESSED, answered, len(checks)
    return Judgement.NO_CONCERNS, answered, len(checks)


def domain_statuses(results: dict[str, CheckResult], reviewer: dict, overrides: dict) -> dict[int, DomainStatus]:
    out = {}
    for dom in DOMAINS:
        sug, answered, total = suggest_domain(dom, results, reviewer)
        ov = overrides.get(str(dom)) or overrides.get(dom) or {}
        final = Judgement(ov["judgement"]) if ov.get("judgement") else sug
        out[dom] = DomainStatus(dom, sug, final, answered, total, bool(ov.get("judgement")), ov.get("reason", ""))
    return out


def suggest_overall(domains: dict[int, DomainStatus]) -> tuple[Judgement, bool]:
    """Return (judgement, complete). Serious in any domain → serious (assessment can stop)."""
    finals = [d.final for d in domains.values()]
    worst = max(finals, key=lambda j: j.severity)
    if worst == Judgement.SERIOUS_CONCERNS:
        return worst, True
    complete = all(d.final not in (Judgement.NOT_ASSESSED, Judgement.PENDING) for d in domains.values())
    if worst == Judgement.NOT_ASSESSED:
        return Judgement.NOT_ASSESSED, False
    return worst, complete


def overall_label(j: Judgement, complete: bool) -> str:
    if j == Judgement.SERIOUS_CONCERNS or complete:
        return j.value
    if j in (Judgement.NO_CONCERNS, Judgement.NOT_ASSESSED):
        return "Incomplete"
    return f"{j.value} (incomplete)"
