"""INSPECT-SR structure: 4 domains, 21 checks (questions as published at https://inspect-sr.com/)."""
from __future__ import annotations

from dataclasses import dataclass

GUIDANCE_BASE = "https://inspect-sr.com/chapters/check_{}.html"

CITATION = (
    "Wilkinson J, et al. INSPECT-SR tool for assessing trustworthiness of randomised "
    "controlled trials. BMJ 2026;394:e100611. doi:10.1136/bmj-2026-100611"
)
CITATION_URL = "https://doi.org/10.1136/bmj-2026-100611"
SITE_URL = "https://inspect-sr.com/"


@dataclass(frozen=True)
class CheckDef:
    id: str
    domain: int
    question: str
    automation: str  # "auto", "assisted" or "manual"

    @property
    def guidance_url(self) -> str:
        return GUIDANCE_BASE.format(self.id.replace(".", "_"))


DOMAINS = {
    1: "Inspecting post-publication notices",
    2: "Inspecting conduct, governance and transparency",
    3: "Inspecting text and figures",
    4: "Inspecting results in the study",
}

DOMAIN_SHORT = {1: "D1 Notices", 2: "D2 Conduct", 3: "D3 Text & figures", 4: "D4 Results"}

CHECKS: list[CheckDef] = [
    CheckDef("1.1", 1, "Does the study have an associated retraction?", "auto"),
    CheckDef("1.2", 1, "Does the study have an associated expression of concern or other relevant post-publication notice?", "auto"),
    CheckDef("1.3", 1, "Do other studies by the research team highlight causes for concern?", "assisted"),
    CheckDef("2.1", 2, "Are there concerns relating to ethical approval?", "manual"),
    CheckDef("2.2", 2, "Are there concerns relating to the timing or absence of study registration?", "assisted"),
    CheckDef("2.3", 2, "Are there important inconsistencies between the publication and the registration documents?", "manual"),
    CheckDef("2.4", 2, "Is the recruitment of participants implausible?", "manual"),
    CheckDef("2.5", 2, "Are there concerns about the plausibility of conducting the study using the reported methods and resources?", "manual"),
    CheckDef("3.1", 3, "Are there concerns relating to duplicated content, such as text or tables, or text that is incompatible with the study?", "manual"),
    CheckDef("3.2", 3, "Is there evidence of manipulation or duplication of figures?", "manual"),
    CheckDef("4.1", 4, "Are there any unexplained discrepancies between reported data and participant eligibility criteria?", "manual"),
    CheckDef("4.2", 4, "Are numbers of participants allocated to each group implausible given the allocation method?", "manual"),
    CheckDef("4.3", 4, "Are any baseline data implausible?", "manual"),
    CheckDef("4.4", 4, "Are there any discrepancies between results reported in figures, tables, and text?", "manual"),
    CheckDef("4.5", 4, "Are the numbers of participants lost to follow-up implausible?", "manual"),
    CheckDef("4.6", 4, "Are there any unexplained inconsistencies in the numbers of participants?", "manual"),
    CheckDef("4.7", 4, "Are any outcome data, including estimated treatment effects, implausible?", "manual"),
    CheckDef("4.8", 4, "Are the means and variances of integer data impossible?", "manual"),
    CheckDef("4.9", 4, "Are there errors in statistical results?", "manual"),
    CheckDef("4.10", 4, "Are any other contradictions implied by the data?", "manual"),
    CheckDef("4.11", 4, "Are there inconsistencies in descriptions of methods and results across publications describing the study?", "manual"),
]

CHECKS_BY_ID = {c.id: c for c in CHECKS}


def checks_in_domain(domain: int) -> list[CheckDef]:
    return [c for c in CHECKS if c.domain == domain]
