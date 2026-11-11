"""The assessment: studies + automated results + reviewer decisions, with tables and exports."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from . import __version__
from .framework import CHECKS_BY_ID
from .models import CheckResult, Study
from .pipeline import RunInfo
from .status import DECISIONS, Flag, Notice, group, identification_label, notices, study_status

CHECK_LABELS = {"1.1": "Retraction", "1.2": "Expression of concern / correction",
                "1.3": "Authors' other retractions", "2.2": "Trial registration"}


@dataclass
class Assessment:
    studies: list[Study]
    results: dict[str, dict[str, CheckResult]]
    run: RunInfo
    settings: dict = field(default_factory=dict)
    decisions: dict = field(default_factory=dict)   # uid -> {"decision": one of DECISIONS, "note": str}

    def study(self, uid: str) -> Study:
        return next(s for s in self.studies if s.uid == uid)

    def status(self, uid: str) -> tuple[str, list[Flag]]:
        return study_status(self.study(uid), self.results[uid], self.decisions.get(uid))

    def notices(self, uid: str) -> list[Notice]:
        return notices(self.results[uid])

    def decision(self, uid: str) -> str:
        return (self.decisions.get(uid) or {}).get("decision") or DECISIONS[0]

    # ---- tables ------------------------------------------------------------------
    def summary_table(self) -> pd.DataFrame:
        rows = []
        for s in self.studies:
            status, fl = self.status(s.uid)
            ns = [n for n in self.notices(s.uid) if not n.possible]
            rows.append({
                "ID": s.uid, "Status": status, "Group": group(status), "Study": s.label, "Title": s.title,
                "Journal": s.journal, "Year": s.year,
                "Why": " ".join(f.why for f in fl) if fl else "",
                "Notices": "; ".join(f"{n.type} {n.date[:10] if n.date else ''}".strip() for n in ns),
                "Reasons (Retraction Watch)": "; ".join(dict.fromkeys(r for n in ns for r in n.reasons)),
                "Sources": ", ".join(dict.fromkeys(x for n in ns for x in n.sources)),
                "Reviewer decision": self.decision(s.uid),
                "Reviewer note": (self.decisions.get(s.uid) or {}).get("note", ""),
                "DOI": s.doi, "PMID": s.pmid, "Identification": identification_label(s),
                "Design (detected)": s.design, "Trial IDs": "; ".join(s.trial_ids),
                "Source file": s.source_file,
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            from .status import ORDER
            df = df.sort_values(["Status", "ID"], key=lambda c: c.map(ORDER) if c.name == "Status" else c)
        return df

    def notices_table(self) -> pd.DataFrame:
        rows = []
        for s in self.studies:
            for n in self.notices(s.uid):
                rows.append({"ID": s.uid, "Study": s.label, "Notice": n.type, "Date": n.date,
                             "Possible match only": n.possible, "Reported by": n.sources_label,
                             "Reasons (Retraction Watch)": "; ".join(n.reasons), "Note": n.note,
                             "Links": " ".join(n.links)})
        return pd.DataFrame(rows, columns=["ID", "Study", "Notice", "Date", "Possible match only", "Reported by",
                                           "Reasons (Retraction Watch)", "Note", "Links"])

    def checks_table(self) -> pd.DataFrame:
        rows = []
        for s in self.studies:
            for cid, r in self.results[s.uid].items():
                rows.append({"ID": s.uid, "Study": s.label, "Check": CHECK_LABELS.get(cid, cid),
                             "Question": CHECKS_BY_ID[cid].question if cid in CHECKS_BY_ID else "",
                             "Finding": r.proposed.value, "Explanation": r.rationale, "N evidence": len(r.evidence)})
        return pd.DataFrame(rows)

    def evidence_table(self) -> pd.DataFrame:
        rows = []
        for s in self.studies:
            for cid, r in self.results[s.uid].items():
                for e in r.evidence:
                    rows.append({"ID": s.uid, "Study": s.label, "Check": CHECK_LABELS.get(cid, cid), "Source": e.source,
                                 "Type": e.kind, "Date": e.date, "Detail": e.detail, "Link": e.url, "Matched by": e.match})
        return pd.DataFrame(rows, columns=["ID", "Study", "Check", "Source", "Type", "Date", "Detail", "Link", "Matched by"])

    def provenance_table(self) -> pd.DataFrame:
        rows = [{"Item": "Tool", "Value": f"Retraction Watch check v{__version__}"},
                {"Item": "Run started", "Value": self.run.started},
                {"Item": "Run finished", "Value": self.run.finished}]
        rows += [{"Item": f"Source: {k}", "Value": v} for k, v in self.run.sources.items()]
        rows += [{"Item": "Merged duplicate", "Value": d} for d in self.run.duplicates]
        rows += [{"Item": "Source error", "Value": e} for e in self.run.errors[:50]]
        return pd.DataFrame(rows)

    # ---- methods text --------------------------------------------------------------
    def methods_text(self) -> str:
        n = len(self.studies)
        n_id = sum(s.identified for s in self.studies)
        srcs = ", ".join(self.run.sources.keys())
        return (
            f"On {self.run.finished}, the {n} included studies were checked for retractions and other post-publication "
            f"notices with the Retraction Watch check app (v{__version__}). The app identified {n_id}/{n} studies by DOI or "
            f"PubMed ID and searched the following sources: {srcs}. It also checked for retractions and expressions of "
            "concern among other papers by each study's first and last authors, and compared trial registration dates "
            "with study start dates. A reviewer verified every flagged finding against the cited notices and registry records."
        )

    # ---- exports ------------------------------------------------------------------
    def to_excel(self) -> bytes:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
            sheets = {
                "Summary": self.summary_table().drop(columns=["Group"]),
                "Notices": self.notices_table(),
                "Checks": self.checks_table(),
                "Evidence": self.evidence_table(),
                "Provenance": self.provenance_table(),
                "Methods text": pd.DataFrame({"Methods text (draft)": [self.methods_text()]}),
            }
            wb = xw.book
            fmt = {"Retracted": wb.add_format({"bg_color": "#f4c7c7"}),
                   "Retracted and replaced": wb.add_format({"bg_color": "#f4c7c7"}),
                   "Expression of concern": wb.add_format({"bg_color": "#f9d9cc"}),
                   "Correction": wb.add_format({"bg_color": "#fde8b4"}),
                   "No notices found": wb.add_format({"bg_color": "#c9ecc9"})}
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


def build(studies: list[Study], dossiers: dict, run: RunInfo, rw, settings: Optional[dict] = None) -> Assessment:
    from .checks import run_checks
    results = {s.uid: run_checks(s, dossiers[s.uid], rw) for s in studies}
    return Assessment(studies=studies, results=results, run=run, settings=settings or {})
