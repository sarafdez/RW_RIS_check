"""Core data model: one standard record per study, plus check results and judgements."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class Answer(str, Enum):
    """INSPECT-SR response options. 'Yes' always means a potentially problematic feature."""
    YES = "Yes"
    NO = "No"
    UNCLEAR = "Unclear"
    NA = "Not applicable"
    NOT_ASSESSED = "Not assessed"


class Judgement(str, Enum):
    NO_CONCERNS = "No concerns"
    SOME_CONCERNS = "Some concerns"
    SERIOUS_CONCERNS = "Serious concerns"
    PENDING = "Needs reviewer"     # automated evidence inconclusive
    NOT_ASSESSED = "Not assessed"  # no checks in this domain answered

    @property
    def severity(self) -> int:
        return {
            Judgement.NOT_ASSESSED: -1,
            Judgement.NO_CONCERNS: 0,
            Judgement.PENDING: 1,
            Judgement.SOME_CONCERNS: 2,
            Judgement.SERIOUS_CONCERNS: 3,
        }[self]


@dataclass
class Evidence:
    """One finding from one source, kept so every proposed answer can be traced."""
    source: str                    # e.g. "Retraction Watch", "Crossref", "PubMed", "OpenAlex", "ClinicalTrials.gov"
    kind: str                      # e.g. "Retraction", "Expression of concern", "Correction", "Registration"
    detail: str = ""
    date: Optional[str] = None     # ISO date of the notice / event
    url: Optional[str] = None
    match: str = ""                # how the record was linked to the study (doi, pmid, title_fuzzy:96, author...)
    text: str = ""                 # title / citation of the notice itself, when available

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CheckResult:
    check_id: str
    proposed: Answer
    rationale: str
    evidence: list[Evidence] = field(default_factory=list)
    needs_review: bool = False
    automated: bool = True
    suggests_serious: bool = False  # e.g. check 1.1 "Yes" typically warrants serious concerns
    data: dict = field(default_factory=dict)  # structured values for charts (e.g. registration lag)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["proposed"] = self.proposed.value
        d["evidence"] = [e.to_dict() for e in self.evidence]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CheckResult":
        d = dict(d)
        d["proposed"] = Answer(d["proposed"])
        d["evidence"] = [Evidence(**e) for e in d.get("evidence", [])]
        return cls(**d)


@dataclass
class Study:
    uid: str                                   # stable id within the session (e.g. "S001")
    raw: str = ""                              # original citation text / record summary
    source_file: str = ""
    source_format: str = ""
    title: Optional[str] = None
    authors: list[str] = field(default_factory=list)  # "Family, Given" where possible
    year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    abstract: Optional[str] = None
    pub_date: Optional[str] = None             # ISO date if known
    trial_ids: list[str] = field(default_factory=list)
    pub_types: list[str] = field(default_factory=list)  # e.g. PubMed publication types
    design: str = "Unknown"                    # detected study design category
    include: bool = True
    # identification / resolution
    id_method: str = "as supplied"             # how DOI/PMID were obtained
    id_confidence: str = "high"                # high / medium / low / none
    id_notes: list[str] = field(default_factory=list)

    @property
    def people(self) -> list[str]:
        """Author list without group/collaboration names."""
        from .normalize import is_group_author
        return [a for a in self.authors if not is_group_author(a)]

    @property
    def first_author(self) -> Optional[str]:
        p = self.people
        return p[0] if p else None

    @property
    def last_author(self) -> Optional[str]:
        p = self.people
        return p[-1] if len(p) > 1 else None

    @property
    def label(self) -> str:
        fa = (self.first_author or (self.authors[0] if self.authors else "Unknown")).split(",")[0]
        etal = " et al." if len(self.authors) > 1 else ""
        yr = f" {self.year}" if self.year else ""
        return f"{fa}{etal}{yr}"

    @property
    def identified(self) -> bool:
        return bool(self.doi or self.pmid)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Study":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})
