"""The assessment: studies + automated results + reviewer decisions, with tables and exports."""
from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass, field
from typing import Optional

import pandas as pd

from . import __version__
from .framework import CHECKS, CHECKS_BY_ID, CITATION, DOMAINS, DOMAIN_SHORT
from .judgement import DomainStatus, domain_statuses, final_answer, overall_label, suggest_overall
from .models import Answer, CheckResult, Judgement, Study
from .pipeline import RunInfo


@dataclass
class Assessment:
    studies: list[Study]
    results: dict[str, dict[str, CheckResult]]
    run: RunInfo
    settings: dict = field(default_factory=dict)                 # e.g. search_date, online
    reviewer: dict = field(default_factory=dict)                 # uid -> check_id -> {answer, note}
    domain_overrides: dict = field(default_factory=dict)         # uid -> domain -> {judgement, reason}
    overall_overrides: dict = field(default_factory=dict)        # uid -> {judgement, reason}

    # ---- judgements -------------------------------------------------------------
    def domains(self, uid: str) -> dict[int, DomainStatus]:
        return domain_statuses(self.results[uid], self.reviewer.get(uid, {}), self.domain_overrides.get(uid, {}))

    def overall(self, uid: str) -> tuple[Judgement, bool, bool]:
        """(judgement, complete, overridden)"""
        ov = self.overall_overrides.get(uid) or {}
        if ov.get("judgement"):
            return Judgement(ov["judgement"]), True, True
        j, complete = suggest_overall(self.domains(uid))
        return j, complete, False

    def answer(self, uid: str, cid: str) -> Answer:
        return final_answer(self.results[uid][cid], self.reviewer.get(uid, {}).get(cid))

    def study(self, uid: str) -> Study:
        return next(s for s in self.studies if s.uid == uid)

    # ---- tables ------------------------------------------------------------------
    def summary_table(self) -> pd.DataFrame:
        rows = []
        for s in self.studies:
            doms = self.domains(s.uid)
            ov, complete, ov_flag = self.overall(s.uid)
            res = self.results[s.uid]
            rows.append({
                "ID": s.uid, "Study": s.label, "Title": s.title, "Year": s.year, "Journal": s.journal,
                "Design (detected)": s.design, "DOI": s.doi, "PMID": s.pmid,
                "Identification": f"{s.id_method} ({s.id_confidence})",
                "Trial IDs": "; ".join(s.trial_ids),
                **{f"{c}": self.answer(s.uid, c) .value for c in ("1.1", "1.2", "1.3", "2.2")},
                **{DOMAIN_SHORT[k]: v.final.value for k, v in doms.items()},
                "Overall": overall_label(ov, complete),
                "Overall (raw)": ov.value,
                "Needs review": sum(r.needs_review and not self.reviewer.get(s.uid, {}).get(cid, {}).get("answer")
                                    for cid, r in res.items()),
                "Source file": s.source_file,
            })
        return pd.DataFrame(rows)

    def checks_table(self) -> pd.DataFrame:
        rows = []
        for s in self.studies:
            for c in CHECKS:
                r = self.results[s.uid][c.id]
                rv = self.reviewer.get(s.uid, {}).get(c.id, {})
                rows.append({
                    "ID": s.uid, "Study": s.label, "Domain": c.domain, "Check": c.id, "Question": c.question,
                    "Automated": r.automated, "Proposed answer": r.proposed.value,
                    "Reviewer answer": rv.get("answer") or "", "Final answer": self.answer(s.uid, c.id).value,
                    "Rationale": r.rationale, "Reviewer note": rv.get("note", ""),
                    "Needs review": r.needs_review, "N evidence": len(r.evidence),
                    "Guidance": c.guidance_url,
                })
        return pd.DataFrame(rows)

    def evidence_table(self) -> pd.DataFrame:
        rows = []
        for s in self.studies:
            for cid, r in self.results[s.uid].items():
                for e in r.evidence:
                    rows.append({"ID": s.uid, "Study": s.label, "Check": cid, "Source": e.source, "Type": e.kind,
                                 "Date": e.date, "Detail": e.detail, "Link": e.url, "Matched by": e.match})
        return pd.DataFrame(rows, columns=["ID", "Study", "Check", "Source", "Type", "Date", "Detail", "Link", "Matched by"])

    def judgements_table(self) -> pd.DataFrame:
        rows = []
        for s in self.studies:
            for dom, st in self.domains(s.uid).items():
                rows.append({"ID": s.uid, "Study": s.label, "Domain": f"D{dom}", "Domain name": DOMAINS[dom],
                             "Suggested": st.suggested.value, "Final": st.final.value, "Overridden": st.overridden,
                             "Reason": st.reason, "Checks answered": f"{st.answered}/{st.total}"})
            ov, complete, flag = self.overall(s.uid)
            rows.append({"ID": s.uid, "Study": s.label, "Domain": "Overall", "Domain name": "Overall study judgement",
                         "Suggested": ov.value, "Final": overall_label(ov, complete), "Overridden": flag,
                         "Reason": (self.overall_overrides.get(s.uid) or {}).get("reason", ""), "Checks answered": ""})
        return pd.DataFrame(rows)

    def provenance_table(self) -> pd.DataFrame:
        rows = [{"Item": "Tool", "Value": f"INSPECT-SR assistant v{__version__}"},
                {"Item": "Framework", "Value": CITATION},
                {"Item": "Run started", "Value": self.run.started},
                {"Item": "Run finished", "Value": self.run.finished},
                {"Item": "Online sources used", "Value": "Yes" if self.run.online else "No (Retraction Watch only)"}]
        if self.settings.get("search_date"):
            rows.append({"Item": "Review search date", "Value": str(self.settings["search_date"])})
        rows += [{"Item": f"Source: {k}", "Value": v} for k, v in self.run.sources.items()]
        rows += [{"Item": "Source error", "Value": e} for e in self.run.errors[:50]]
        return pd.DataFrame(rows)

    # ---- methods text --------------------------------------------------------------
    def methods_text(self) -> str:
        n = len(self.studies)
        n_id = sum(s.identified for s in self.studies)
        srcs = ", ".join(self.run.sources.keys())
        auto = ", ".join(f"{c} ({CHECKS_BY_ID[c].question.rstrip('?')})" for c in ("1.1", "1.2", "1.3", "2.2"))
        return (
            f"The trustworthiness of the {n} included studies was assessed with INSPECT-SR ({CITATION}). "
            f"To support checks {auto}, we used the INSPECT-SR assistant (v{__version__}) on {self.run.finished}. "
            f"It identified {n_id}/{n} studies by DOI or PubMed ID and searched the following sources: {srcs}. "
            "The tool proposed an answer for each automated check, and a reviewer verified it against the cited evidence "
            "(retraction and correction notices, registry records). The remaining checks were answered by reviewers from the "
            "full text. Domain and overall judgements followed the INSPECT-SR guidance: a retraction typically warrants "
            "'serious concerns', and the overall judgement is at least as severe as the most severe domain."
        )

    # ---- exports ------------------------------------------------------------------
    def to_excel(self) -> bytes:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
            sheets = {
                "Summary": self.summary_table().drop(columns=["Overall (raw)"]),
                "Judgements": self.judgements_table(),
                "Checks": self.checks_table(),
                "Evidence": self.evidence_table(),
                "Provenance": self.provenance_table(),
                "Methods text": pd.DataFrame({"Methods text (draft)": [self.methods_text()]}),
            }
            wb = xw.book
            fmt = {
                Judgement.SERIOUS_CONCERNS.value: wb.add_format({"bg_color": "#f4c7c7"}),
                Judgement.SOME_CONCERNS.value: wb.add_format({"bg_color": "#fde8b4"}),
                Judgement.NO_CONCERNS.value: wb.add_format({"bg_color": "#c9ecc9"}),
                Answer.YES.value: wb.add_format({"bg_color": "#f4c7c7"}),
            }
            for name, df in sheets.items():
                df.to_excel(xw, sheet_name=name, index=False)
                ws = xw.sheets[name]
                ws.freeze_panes(1, 0)
                ws.autofilter(0, 0, max(len(df), 1), max(len(df.columns) - 1, 0))
                for i, col in enumerate(df.columns):
                    width = min(60, max(10, int(df[col].astype(str).str.len().quantile(0.9)) if len(df) else 10, len(col) + 2))
                    ws.set_column(i, i, width)
                for val, f in fmt.items():
                    ws.conditional_format(1, 0, max(len(df), 1), len(df.columns) - 1,
                                          {"type": "cell", "criteria": "==", "value": f'"{val}"', "format": f})
        return buf.getvalue()

    def to_json(self) -> str:
        return json.dumps({
            "tool": "inspect-sr-assistant", "version": __version__,
            "run": asdict(self.run), "settings": {k: str(v) for k, v in self.settings.items()},
            "studies": [s.to_dict() for s in self.studies],
            "results": {uid: {cid: r.to_dict() for cid, r in res.items()} for uid, res in self.results.items()},
            "reviewer": self.reviewer, "domain_overrides": self.domain_overrides,
            "overall_overrides": self.overall_overrides,
        }, indent=1, default=str)

    @classmethod
    def from_json(cls, text: str) -> "Assessment":
        d = json.loads(text)
        if d.get("tool") != "inspect-sr-assistant":
            raise ValueError("Not an INSPECT-SR assistant assessment file.")
        return cls(
            studies=[Study.from_dict(s) for s in d["studies"]],
            results={uid: {cid: CheckResult.from_dict(r) for cid, r in res.items()} for uid, res in d["results"].items()},
            run=RunInfo(**d["run"]), settings=d.get("settings", {}),
            reviewer=d.get("reviewer", {}), domain_overrides=d.get("domain_overrides", {}),
            overall_overrides=d.get("overall_overrides", {}),
        )


def build(studies: list[Study], dossiers: dict, run: RunInfo, rw, settings: Optional[dict] = None) -> Assessment:
    from .checks import run_checks
    results = {s.uid: run_checks(s, dossiers[s.uid], rw) for s in studies}
    return Assessment(studies=studies, results=results, run=run, settings=settings or {})
