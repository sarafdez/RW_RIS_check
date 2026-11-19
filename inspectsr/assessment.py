"""The assessment: studies + automated results + reviewer decisions, with tables and exports."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from . import __version__
from .models import CheckResult, Study
from .pipeline import RunInfo
from .status import (DECISIONS, DEFAULT_REVIEW, NOT_SELECTED, NOT_YET, OUTCOMES, Flag, Notice, group,
                     identification_label, notices, study_status)

def xlsx_col(i: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA."""
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


CHECK_LABELS = {"1.1": "Retraction", "1.2": "Expression of concern / correction",
                "1.3": "Authors' other retractions", "2.2": "Trial registration"}
CHECK_QUESTIONS = {
    "1.1": "Has the study been retracted?",
    "1.2": "Is there an expression of concern or a correction?",
    "1.3": "Do the first or last authors have other retractions for research-integrity reasons?",
    "2.2": "Was the trial registered, and before it started?",
}


@dataclass
class Assessment:
    studies: list[Study]
    results: dict[str, dict[str, CheckResult]]
    run: RunInfo
    settings: dict = field(default_factory=dict)
    decisions: dict = field(default_factory=dict)   # uid -> {"decision": one of DECISIONS, "note": str}
    review_set: dict = field(default_factory=dict)  # uid -> bool: reviewer's choice of which flagged studies to review
    author_reviews: dict = field(default_factory=dict)  # author key -> {"verdict": ..., "note": ...}
    # Flags depend only on the automated results, which never change after the run: cache them.
    _raw: dict = field(default_factory=dict, repr=False)
    _flagged: Optional[list] = field(default=None, repr=False)

    def study(self, uid: str) -> Study:
        if not getattr(self, "_by_uid", None):
            self._by_uid = {s.uid: s for s in self.studies}
        return self._by_uid[uid]

    def status(self, uid: str) -> tuple[str, list[Flag]]:
        return study_status(self.study(uid), self.results[uid], self.decisions.get(uid), self.author_reviews)

    def notices(self, uid: str) -> list[Notice]:
        return notices(self.results[uid])

    def decision(self, uid: str) -> str:
        return (self.decisions.get(uid) or {}).get("decision") or DECISIONS[0]

    def flagged_uids(self) -> list[str]:
        """Studies with at least one flag, most severe first. A study stays in this list after the reviewer
        marks it as a false match, so review progress can be counted and the decision revisited."""
        if self._flagged is None:
            from .status import ORDER
            raw = [(self.raw_status(s.uid), s.uid) for s in self.studies]
            self._flagged = [uid for st, uid in sorted(raw, key=lambda x: (ORDER[x[0]], x[1])) if group(st) != "none"]
        return self._flagged

    def raw_status(self, uid: str) -> str:
        if uid not in self._raw:
            self._raw[uid] = study_status(self.study(uid), self.results[uid])[0]
        return self._raw[uid]

    def selected(self, uid: str) -> bool:
        """Is this flagged study in the reviewer's queue? Default: the serious findings."""
        if uid in self.review_set:
            return bool(self.review_set[uid])
        return self.raw_status(uid) in DEFAULT_REVIEW

    def review_queue(self) -> list[str]:
        return [u for u in self.flagged_uids() if self.selected(u)]

    def review_progress(self) -> tuple[int, int]:
        q = self.review_queue()
        return sum(self.decision(u) != NOT_YET for u in q), len(q)

    def final_decision(self, uid: str) -> str:
        """What the export says: '' if nothing was flagged, 'Not reviewed (not selected)' if flagged but not chosen."""
        if group(self.raw_status(uid)) == "none":
            return ""
        if not self.selected(uid):
            return NOT_SELECTED
        return self.decision(uid)

    # ---- authors -------------------------------------------------------------------
    def authors_table(self) -> pd.DataFrame:
        """One row per author (first/last authors of included studies) with other retractions or concerns."""
        from .status import AUTHOR_UNCHECKED
        rank = {"same OpenAlex author ID": 0, "shared co-author": 1, "name only (full name)": 2,
                "name only (surname + initial)": 3}
        agg: dict[str, dict] = {}
        for s in self.studies:
            r13 = self.results[s.uid].get("1.3")
            for au in (r13.data.get("authors") or []) if r13 else []:
                a = agg.setdefault(au["key"], {"key": au["key"], "name": au["name"], "studies": [], "uids": [],
                                                "papers": {}, "match": au["match"], "strong": False})
                if "," in au["name"] and "," not in a["name"]:
                    a["name"] = au["name"]  # prefer "Family, Given"
                if s.uid not in a["uids"]:
                    a["uids"].append(s.uid)
                    a["studies"].append(f"{s.uid} {s.label} ({au['role']})")
                a["strong"] = a["strong"] or au["strong"]
                if rank.get(au["match"], 9) < rank.get(a["match"], 9):
                    a["match"] = au["match"]
                for p in au["papers"]:
                    a["papers"].setdefault((p.get("doi") or p.get("title") or "").lower(), p)
        rows = []
        for a in agg.values():
            papers = list(a["papers"].values())
            rev = self.author_reviews.get(a["key"]) or {}
            rows.append({
                "key": a["key"], "Author": a["name"], "Flagged lead": a["strong"],
                "Studies in your review": "; ".join(a["studies"]), "N studies": len(a["uids"]),
                "Other retractions/EoCs": len(papers), "Integrity-related": sum(p["integrity"] for p in papers),
                "Match": a["match"], "Your verdict": rev.get("verdict", AUTHOR_UNCHECKED), "Note": rev.get("note", ""),
                "Other papers": " | ".join(f"{p['title'] or ''} ({p['nature']}, {str(p['date'] or '')[:4]}; "
                                           f"{'; '.join(p['reasons'][:3])}) {p.get('notice') or ''}" for p in papers),
                "_papers": papers, "_uids": a["uids"],
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values(["Flagged lead", "Integrity-related", "Other retractions/EoCs"],
                              ascending=[False, False, False]).reset_index(drop=True)

    # ---- figures ------------------------------------------------------------------
    def outcome_counts(self) -> pd.DataFrame:
        f = self.flagged_uids()
        counts = {o: 0 for o in OUTCOMES}
        for u in f:
            counts[self.final_decision(u)] += 1
        return pd.DataFrame({"Outcome": list(counts), "Studies": list(counts.values())})

    def notices_per_year(self) -> pd.DataFrame:
        nt = self.notices_table()
        nt = nt[~nt["Possible match only"] & nt["Date"].fillna("").str.match(r"^\d{4}")]
        if nt.empty:
            return pd.DataFrame(columns=["Year", "Notice", "Notices"])
        nt = nt.assign(Year=nt["Date"].str[:4].astype(int))
        return nt.groupby(["Year", "Notice"]).size().reset_index(name="Notices")

    def studies_per_year(self) -> pd.DataFrame:
        rows = []
        for s in self.studies:
            g = group(self.raw_status(s.uid))
            label = {"retracted": "Retracted", "concern": "Other findings", "none": "No notices found"}[g]
            if self.final_decision(s.uid).startswith("False match"):
                label = "No notices found"
            rows.append({"Year": s.year, "Finding": label})
        df = pd.DataFrame(rows).dropna(subset=["Year"])
        if df.empty:
            return pd.DataFrame(columns=["Year", "Finding", "Studies"])
        df["Year"] = df["Year"].astype(int)
        return df.groupby(["Year", "Finding"]).size().reset_index(name="Studies")

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
                "Selected for review": ("Yes" if self.selected(s.uid) else "No") if fl else "",
                "Reviewer decision": self.final_decision(s.uid),
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
                dec = self.decisions.get(s.uid) or {}
                rows.append({"ID": s.uid, "Study": s.label, "Notice": n.type, "Date": n.date,
                             "Notice title": n.title, "In Retraction Watch": "Yes" if n.in_retraction_watch else "No",
                             "Reported by": n.sources_label, "Possible match only": n.possible,
                             "Reasons (Retraction Watch)": "; ".join(n.reasons), "Retraction Watch note": n.note,
                             "Links": " ".join(n.links), "Reviewer decision": self.final_decision(s.uid),
                             "Reviewer note": dec.get("note", "")})
        return pd.DataFrame(rows, columns=["ID", "Study", "Notice", "Date", "Notice title", "In Retraction Watch",
                                           "Reported by", "Possible match only", "Reasons (Retraction Watch)",
                                           "Retraction Watch note", "Links", "Reviewer decision", "Reviewer note"])

    def checks_table(self) -> pd.DataFrame:
        rows = []
        for s in self.studies:
            for cid, r in self.results[s.uid].items():
                rows.append({"ID": s.uid, "Study": s.label, "Check": CHECK_LABELS.get(cid, cid),
                             "Question": CHECK_QUESTIONS.get(cid, ""),
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

    def not_identified_table(self) -> pd.DataFrame:
        """References for which no DOI or PubMed ID could be found: checked against Retraction Watch by title only."""
        rows = []
        for s in self.studies:
            if s.identified:
                continue
            status = self.raw_status(s.uid)
            rows.append({"ID": s.uid, "Study": s.label, "Title": s.title, "Journal": s.journal, "Year": s.year,
                         "Result of title check": status,
                         "What was tried": " ".join(s.id_notes) or (
                             "No DOI or PubMed ID in your file; no exact title match in Retraction Watch. "
                             + ("Online title searches found nothing." if self.run.online else
                                "Quick check: no online search was done (use Standard to look it up in PubMed).")),
                         "Citation as supplied": s.raw[:500], "Source file": s.source_file,
                         "What to do": "Add the DOI on the Input page and run again, or check the study by hand"})
        return pd.DataFrame(rows, columns=["ID", "Study", "Title", "Journal", "Year", "Result of title check",
                                           "What was tried", "Citation as supplied", "Source file", "What to do"])

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
        srcs = ", ".join(k for k in self.run.sources if k != "Check type")
        return (
            f"On {self.run.finished}, the {n} included studies were checked for retractions and other post-publication "
            f"notices with the Retraction Watch check app (v{__version__}). The app identified {n_id}/{n} studies by DOI or "
            f"PubMed ID and searched the following sources: {srcs}. It also checked for retractions and expressions of "
            "concern among other papers by each study's first and last authors, and compared trial registration dates "
            "with study start dates. A reviewer verified every flagged finding against the cited notices and registry records."
        )

    # ---- exports ------------------------------------------------------------------
    def to_ris(self, drop_excluded: bool = True, replace_notices: bool = True, add_notes: bool = True
               ) -> tuple[str, dict]:
        """Cleaned reference list. Records that came from a RIS file are written back with all their fields."""
        import re
        import rispy
        from .status import EXCLUDE
        list_keys = {rispy.TAG_KEY_MAPPING[t] for t in rispy.LIST_TYPE_TAGS if t in rispy.TAG_KEY_MAPPING}
        out, stats = [], {"written": 0, "excluded": 0, "replaced": 0}
        for s in self.studies:
            dec = self.final_decision(s.uid)
            if drop_excluded and dec == EXCLUDE:
                stats["excluded"] += 1
                continue
            replaced = replace_notices and bool(s.notice_for)
            if replaced:
                m = s.notice_for
                rec = {"type_of_reference": "JOUR", "title": m.get("title"), "authors": m.get("authors"),
                       "year": str(m["year"]) if m.get("year") else None, "journal_name": m.get("journal"),
                       "doi": m.get("doi"), "accession_number": m.get("pmid")}
                stats["replaced"] += 1
            elif s.ris:
                rec = dict(s.ris)
            else:
                rec = {"type_of_reference": "JOUR", "title": s.title, "authors": s.authors,
                       "year": str(s.year) if s.year else None, "journal_name": s.journal, "doi": s.doi,
                       "accession_number": s.pmid}
            rec = {k: v for k, v in rec.items() if v not in (None, "", [])}
            for k in list(rec):
                if k in list_keys and isinstance(rec[k], str):
                    rec[k] = [rec[k]]
            notes = list(rec.get("notes", []))
            if replaced:
                notes.append(f"Retraction check: replaces the notice that was in the list "
                             f"(\"{s.title}\"{', doi ' + s.doi if s.doi else ''}).")
            if add_notes:
                status, _ = self.status(s.uid)
                line = f"Retraction check {self.run.finished[:10]}: {status}"
                if dec:
                    line += f"; decision: {dec}"
                note = (self.decisions.get(s.uid) or {}).get("note")
                if note:
                    line += f"; note: {note}"
                notes.append(line)
            if notes:
                rec["notes"] = notes
            out.append(rec)
            stats["written"] += 1
        text = rispy.dumps(out, skip_unknown_tags=False)
        text = re.sub(r"(?m)^\d+\.\n", "", text)  # rispy numbers records; reference managers don't need it
        return text, stats

    def to_excel(self) -> bytes:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
            sheets = {
                "Summary": self._summary_with_notices(),
                "Notices": self.notices_table(),
                "Authors": self._authors_sheet(),
                "Authors' other papers": self.author_papers_table(),
                "Not identified": self.not_identified_table(),
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
            self._overview_sheet(xw)
            row_colors = {  # whole-row highlighting on the Summary sheet, by status
                "#f4c7c7": ["Retracted", "Retracted and replaced", "Retraction notice in your list"],
                "#f9d9cc": ["Expression of concern", "Expression of concern notice in your list"],
                "#fde8b4": ["Correction", "Authors' other retractions", "Late registration", "Not a study record",
                            "Correction notice in your list"],
                "#e8e8e4": ["Possible match"],
            }
            for name, df in sheets.items():
                df.to_excel(xw, sheet_name=name, index=False)
                if name in ("Authors", "Authors' other papers") and "Your verdict" in df.columns and len(df):
                    col = xlsx_col(list(df.columns).index("Your verdict"))
                    for color, verdict in (("#f4c7c7", "Same person"), ("#e8e8e4", "Different person"),
                                           ("#fde8b4", "Unsure")):
                        xw.sheets[name].conditional_format(
                            1, 0, len(df), len(df.columns) - 1,
                            {"type": "formula", "criteria": f'=${col}2="{verdict}"',
                             "format": wb.add_format({"bg_color": color})})
                if name == "Summary" and "Status" in df.columns and len(df):
                    col = xlsx_col(list(df.columns).index("Status"))
                    for color, statuses in row_colors.items():
                        rf = wb.add_format({"bg_color": color})
                        for stt in statuses:
                            xw.sheets[name].conditional_format(
                                1, 0, len(df), len(df.columns) - 1,
                                {"type": "formula", "criteria": f'=${col}2="{stt}"', "format": rf})
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


    def _authors_sheet(self) -> pd.DataFrame:
        t = self.authors_table()
        if t.empty:
            return pd.DataFrame({"Authors": ["No other retractions found for the first or last authors."]})
        t = t.drop(columns=["key", "_papers", "_uids", "Other papers"]).rename(
            columns={"Flagged lead": "Strong match (shared co-author / author ID)"})
        return t

    def author_papers_table(self) -> pd.DataFrame:
        """One row per other retracted paper, for authors that are strong matches or that you gave a verdict.
        Name-only matches you did not review are left out (often hundreds of papers by namesakes)."""
        from .status import AUTHOR_UNCHECKED
        cols = ["Author", "Your verdict", "Note", "Strong match", "Studies in your review", "Other paper",
                "Notice", "Year", "Integrity-related", "Shares co-authors with your study",
                "Reasons (Retraction Watch)", "Found in", "Link"]
        t = self.authors_table()
        rows = []
        if not t.empty:
            for _, r in t.iterrows():
                if not r["Flagged lead"] and r["Your verdict"] == AUTHOR_UNCHECKED:
                    continue
                papers = sorted(r["_papers"], key=lambda p: (not p.get("shared_coauthor"), not p["integrity"],
                                                             str(p["date"] or "")))
                for p in papers:
                    rows.append({"Author": r["Author"], "Your verdict": r["Your verdict"], "Note": r["Note"],
                                 "Strong match": "Yes" if r["Flagged lead"] else "No",
                                 "Studies in your review": r["Studies in your review"], "Other paper": p["title"],
                                 "Notice": p["nature"], "Year": str(p["date"] or "")[:4],
                                 "Integrity-related": "Yes" if p["integrity"] else "No",
                                 "Shares co-authors with your study": "Yes" if p.get("shared_coauthor") else "",
                                 "Reasons (Retraction Watch)": "; ".join(p["reasons"]), "Found in": p["source"],
                                 "Link": p.get("notice") or ""})
        if not rows:
            return pd.DataFrame({"Authors' other papers": ["No authors with other retractions to list (strong matches "
                                                           "or authors you reviewed)."]})
        return pd.DataFrame(rows, columns=cols)

    def _summary_with_notices(self) -> pd.DataFrame:
        """Summary for Excel: every study, and right below a study any notice from your reference list that was
        merged into it (e.g. a retraction notice or corrigendum), so both records can be found in your library."""
        summ = self.summary_table().drop(columns=["Group"])
        if summ.empty:
            return summ
        rows = []
        studies = {s.uid: s for s in self.studies}
        for _, r in summ.iterrows():
            rows.append({**r.to_dict(), "Linked to": ""})
            for n in studies[r["ID"]].merged_notices:
                rows.append({c: "" for c in summ.columns} | {
                    "ID": n.get("uid") or "", "Status": f"{n['type']} notice in your list",
                    "Study": f"Notice for {r['ID']} {r['Study']}", "Title": n.get("title") or "",
                    "Journal": n.get("journal") or "", "Year": n.get("year") or "",
                    "Why": f"This reference in your list is the {n['type'].lower()} notice for {r['ID']}; it was "
                           "merged into that study (one decision for both).",
                    "Reviewer decision": r["Reviewer decision"], "DOI": n.get("doi") or "",
                    "Identification": "Notice in your list", "Source file": n.get("source_file") or "",
                    "Linked to": r["ID"]})
        return pd.DataFrame(rows, columns=list(summ.columns) + ["Linked to"])

    def _overview_sheet(self, xw):
        """First sheet: key numbers and three native Excel charts."""
        wb = xw.book
        ws = wb.add_worksheet("Overview")
        bold = wb.add_format({"bold": True})
        title = wb.add_format({"bold": True, "font_size": 14})
        ws.write(0, 0, "Retraction check: overview", title)
        ws.write(1, 0, f"Run {self.run.finished} · {self.run.sources.get('Check type', '')}")
        done, total = self.review_progress()
        n_ret = sum(group(self.raw_status(s.uid)) == "retracted" for s in self.studies)
        oc = self.outcome_counts().set_index("Outcome")["Studies"]
        nums = [("Studies checked", len(self.studies)), ("Flagged (any finding)", len(self.flagged_uids())),
                ("Retracted", n_ret), ("Selected for review", total), ("Reviewed", done),
                ("Excluded", int(oc.get("Confirmed – exclude", 0))), ("Kept", int(oc.get("Confirmed – keep", 0))),
                ("False matches", int(oc.get("False match – ignore", 0))),
                ("Not identified (title check only)", sum(not s.identified for s in self.studies))]
        at = self.authors_table()
        if not at.empty:
            leads = at[at["Flagged lead"]]
            nums += [("Authors with other retractions (strong matches)", len(leads)),
                     ("   confirmed as the same person", int((at["Your verdict"] == "Same person").sum())),
                     ("   ruled out (different person)", int((at["Your verdict"] == "Different person").sum()))]
        for i, (k, v) in enumerate(nums):
            ws.write(3 + i, 0, k)
            ws.write(3 + i, 1, v, bold)
        ws.set_column(0, 0, 30)
        ws.set_column(1, 8, 12)

        def table(df: pd.DataFrame, row: int, col: int) -> tuple[int, int]:
            for j, c in enumerate(df.columns):
                ws.write(row, col + j, c, bold)
            for i, r in enumerate(df.itertuples(index=False), 1):
                for j, v in enumerate(r):
                    ws.write(row + i, col + j, v if not isinstance(v, float) or v == v else "")
            return row + len(df), col

        # 1. review outcomes
        r0 = 19
        oc_df = self.outcome_counts()
        ws.write(r0 - 1, 0, "Review of flagged studies", bold)
        last, _ = table(oc_df, r0, 0)
        ch = wb.add_chart({"type": "bar"})
        ch.add_series({"name": "Studies", "categories": ["Overview", r0 + 1, 0, last, 0],
                       "values": ["Overview", r0 + 1, 1, last, 1], "fill": {"color": "#2a78d6"},
                       "data_labels": {"value": True}})
        ch.set_title({"name": "Review of flagged studies"})
        ch.set_legend({"none": True})
        ch.set_y_axis({"reverse": True})
        ws.insert_chart(r0 - 1, 4, ch, {"x_scale": 1.3, "y_scale": 1.0})

        # 2. notices per year (pivot: one column per notice type)
        r1 = last + 18
        npy = self.notices_per_year()
        ws.write(r1 - 1, 0, "Notices per year", bold)
        if not npy.empty:
            piv = npy.pivot(index="Year", columns="Notice", values="Notices").fillna(0).astype(int).reset_index()
            last1, _ = table(piv, r1, 0)
            ch = wb.add_chart({"type": "column", "subtype": "stacked"})
            colors = {"Retraction": "#e34948", "Expression of concern": "#eb6834", "Correction": "#1baf7a",
                      "Reinstatement": "#2a78d6"}
            for j, c in enumerate(piv.columns[1:], 1):
                ch.add_series({"name": ["Overview", r1, j], "categories": ["Overview", r1 + 1, 0, last1, 0],
                               "values": ["Overview", r1 + 1, j, last1, j], "fill": {"color": colors.get(c, "#898781")}})
            ch.set_title({"name": "Notices per year (included studies)"})
            ws.insert_chart(r1 - 1, 6, ch, {"x_scale": 1.3})
            r2 = max(last1, r1 + 15) + 3
        else:
            ws.write(r1, 0, "No notices found.")
            r2 = r1 + 4

        # 3. included studies by publication year and finding
        spy = self.studies_per_year()
        ws.write(r2 - 1, 0, "Included studies by publication year", bold)
        if not spy.empty:
            order = [c for c in ["Retracted", "Other findings", "No notices found"] if c in set(spy["Finding"])]
            piv = spy.pivot(index="Year", columns="Finding", values="Studies").fillna(0).astype(int)[order].reset_index()
            last2, _ = table(piv, r2, 0)
            ch = wb.add_chart({"type": "column", "subtype": "stacked"})
            colors = {"Retracted": "#d03b3b", "Other findings": "#fab219", "No notices found": "#0ca30c"}
            for j, c in enumerate(piv.columns[1:], 1):
                ch.add_series({"name": ["Overview", r2, j], "categories": ["Overview", r2 + 1, 0, last2, 0],
                               "values": ["Overview", r2 + 1, j, last2, j], "fill": {"color": colors[c]}})
            ch.set_title({"name": "Included studies by publication year"})
            ws.insert_chart(r2 - 1, 6, ch, {"x_scale": 1.3})


def build(studies: list[Study], dossiers: dict, run: RunInfo, rw, settings: Optional[dict] = None) -> Assessment:
    from .checks import run_checks
    results = {s.uid: run_checks(s, dossiers[s.uid], rw) for s in studies}
    return Assessment(studies=studies, results=results, run=run, settings=settings or {})
