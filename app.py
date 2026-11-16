"""Retraction Watch check: Streamlit front end.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import os
from datetime import date

import pandas as pd
import streamlit as st

from inspectsr import __version__, charts, explore, inputs, pipeline
from inspectsr.assessment import CHECK_LABELS, Assessment, build
from inspectsr.models import Answer, Study
from inspectsr.sources import retraction_watch as rwmod
from inspectsr.sources.registries import registry_url
from inspectsr.status import (AUTHOR_UNCHECKED, AUTHOR_VERDICTS, DECISIONS, NOT_SELECTED, NOT_YET, STATUS_INFO,
                              group)

st.set_page_config(page_title="Retraction Watch check", page_icon="🔎", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
PAGES = ["1 · Input", "2 · Overview", "3 · Flagged studies", "4 · Authors", "5 · Final overview & export",
         "Explore Retraction Watch"]

st.markdown("""
<style>
.badge{display:inline-block;padding:2px 9px;border-radius:999px;font-size:0.85rem;font-weight:600;
       color:#fff;white-space:nowrap;margin-right:4px}
.badge.dark{color:#0b0b0b}
.muted{color:#898781;font-size:0.85rem}
</style>
""", unsafe_allow_html=True)


def badge(text: str, color: str) -> str:
    dark = color in ("#fab219", "#d9d8d2", "#c3c2b7")
    return f'<span class="badge{" dark" if dark else ""}" style="background:{color}">{text}</span>'


def status_badge(status: str) -> str:
    _, color, ic, _ = STATUS_INFO[status]
    return badge(f"{ic} {status}", color)


# ---- data ------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the Retraction Watch database…")
def get_rw(mtime: float, version: str):
    # `version` is part of the cache key: after an app update the index is rebuilt with the new code,
    # instead of reusing an object built by the old code (which may lack newer fields).
    return rwmod.load()


def rw_index():
    try:
        mtime = os.path.getmtime(rwmod.LOCAL_CSV) if os.path.exists(rwmod.LOCAL_CSV) else 0
        idx = get_rw(mtime, __version__)
        if not hasattr(idx, "by_notice_doi"):  # built by older code: rebuild
            get_rw.clear()
            idx = get_rw(mtime, __version__)
        return idx
    except Exception as e:
        st.error(f"Could not load the Retraction Watch database: {e}")
        st.stop()


ss = st.session_state
ss.setdefault("page", PAGES[0])
if ss.page not in PAGES:  # page renamed or removed by an app update
    ss.page = PAGES[0]
if "_goto" in ss:  # navigation requested by a button in the previous run
    ss.page = ss.pop("_goto")
ss.setdefault("parsed", [])        # list[Study]
ss.setdefault("reports", [])
ss.setdefault("assessment", None)  # Assessment
# When the app is updated while a session is open (e.g. a redeploy on Streamlit Cloud), results stored in the
# session were made by the old code and may lack newer features. Discard them instead of crashing.
_stale = ss.assessment is not None and (not isinstance(ss.assessment, Assessment)
                                        or ss.get("app_version") != __version__)
if _stale:
    ss.assessment = None
    ss["_stale_notice"] = True
ss["app_version"] = __version__
ss.setdefault("input_round", 0)     # bumping this gives the upload and paste boxes fresh, empty widgets


# ---- sidebar -----------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🔎 Retraction Watch check")
    st.caption(f"v{__version__} · checks included studies for retractions and other post-publication notices")
    page = st.radio("Go to", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    rw = rw_index()
    st.markdown("**Retraction Watch database**")
    st.caption(f"{rw.meta['n_records']:,} records · downloaded {rw.meta['downloaded_on']} · "
               f"latest notice {rw.meta.get('latest_notice')}")
    if st.button("Update database", help="Download the latest Retraction Watch CSV from Crossref (~70 MB)"):
        with st.spinner("Downloading…"):
            rwmod.download()
        get_rw.clear()
        st.rerun()
    a: Assessment | None = ss.assessment
    if a:
        st.divider()
        st.markdown(f"**Current check:** {len(a.studies)} studies")
        done, total = a.review_progress()
        if total:
            st.progress(done / total, text=f"Reviewed {done} of {total} flagged")
        st.caption(f"Run {a.run.finished}")


def goto(p: str):
    ss["_goto"] = p


def open_study(uid: str):
    ss["_study"] = uid
    goto(PAGES[2])


# =====================================================================================
# 1. INPUT
# =====================================================================================

def page_input():
    st.title("Check your included studies")
    st.markdown(
        "Upload the **studies included in your review** in any common format. Each study is checked against the "
        "**Retraction Watch** database and, depending on the check type you choose below, PubMed and other sources. "
        "Every finding comes with its evidence, and you decide what to do with each flagged study."
    )
    files = st.file_uploader(
        "Reference files",
        type=["ris", "txt", "nbib", "bib", "xml", "csv", "tsv", "xlsx", "xls", "docx", "pdf"],
        accept_multiple_files=True, key=f"files_{ss.input_round}",
        help="RIS (EndNote, Zotero, Covidence…), PubMed/MEDLINE (.nbib), BibTeX, EndNote XML, CSV/Excel, "
             "Word or PDF reference lists, or plain text.",
    )
    pasted = st.text_area("…or paste references (one per line, numbered list, or a list of DOIs/PMIDs)", height=140,
                          key=f"paste_{ss.input_round}",
                          placeholder="1. Horby P, Lim WS, et al. Dexamethasone in Hospitalized Patients with "
                                      "Covid-19. N Engl J Med. 2021;384:693-704.\n10.1056/NEJMoa1602489\nPMID: 30879366")

    b1, b2, _ = st.columns([2, 2, 6])
    b2.button("Clear all / start over", on_click=_clear_all, width="stretch",
              help="Removes the uploaded files, pasted text, parsed references and previous results.")
    if b1.button("Read references", type="primary", disabled=not (files or pasted.strip()), width="stretch"):
        ss.reports = []
        allst: list[Study] = []
        for f in files or []:
            stds, rep = inputs.parse_upload(f.name, f.getvalue())
            allst += stds
            ss.reports.append(rep)
        if pasted.strip():
            stds = inputs.parse_free_text(pasted, "pasted text")
            allst += stds
            ss.reports.append(inputs.ParseReport("pasted text", "free text", len(stds)))
        _set_parsed(allst)

    if not ss.parsed:
        st.info("Add references above to begin.")
        return

    st.subheader("Review the parsed references")
    rep_df = pd.DataFrame([{"File": r.file, "Format": r.format, "Records": r.n_records,
                            "Warnings": "; ".join(r.warnings)} for r in ss.reports])
    st.dataframe(rep_df, hide_index=True, width="stretch")
    if ss.get("dupes_dropped"):
        st.caption(f"{ss.dupes_dropped} duplicate record(s) across files were removed (same DOI, PMID or title).")

    df = pd.DataFrame([{
        "Include": s.include, "ID": s.uid, "Title": s.title or "", "Authors": "; ".join(s.authors),
        "Year": s.year, "DOI": s.doi or "", "PMID": s.pmid or "", "Trial IDs": "; ".join(s.trial_ids),
        "Design (detected)": s.design, "Source": f"{s.source_file} ({s.source_format})",
        "Citation text": s.raw[:300],
    } for s in ss.parsed])
    st.caption("Untick records that are not included studies (e.g. background references). You can fix a "
               "DOI, PMID or title directly in the table.")
    edited = st.data_editor(
        df, hide_index=True, width="stretch", key="editor",
        disabled=["ID", "Design (detected)", "Source", "Citation text"],
        column_config={"Include": st.column_config.CheckboxColumn(width="small"),
                       "Year": st.column_config.NumberColumn(format="%d", width="small"),
                       "Title": st.column_config.TextColumn(width="large")},
    )
    n_inc = int(edited["Include"].sum())
    n_id = int(((edited["DOI"].str.len() > 0) | (edited["PMID"].str.len() > 0))[edited["Include"]].sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Records", len(edited))
    m2.metric("Included", n_inc)
    m3.metric("With DOI or PMID", n_id)
    m4.metric("Need identification", n_inc - n_id)

    st.subheader("Choose a check type")
    ss.setdefault("check_mode", "standard")
    labels = {"quick": "⚡ Quick", "standard": "✓ Standard (recommended)", "extended": "🔍 Extended"}
    about = {"quick": "Retraction Watch only. No internet. Seconds, even for 10,000 references.",
             "standard": "Adds PubMed: finds missing DOIs, PubMed's own retraction tags and notices, trial "
                         "registrations. About 1 minute per 1,500 references.",
             "extended": "Adds Crossref publisher notices (mainly corrections) and OpenAlex. Slow for large lists "
                         "(Crossref allows 1 request per second)."}
    cols = st.columns(3)
    for col, m in zip(cols, pipeline.MODES):
        col.button(labels[m], key=f"mode_{m}", width="stretch",
                   type="primary" if ss.check_mode == m else "secondary",
                   on_click=lambda m=m: ss.update(check_mode=m))
        col.caption(about[m])
    mode = ss.check_mode

    if st.button(f"Run {labels[mode].split(' ', 1)[1].replace(' (recommended)', '')} check on {n_inc} studies",
                 type="primary", disabled=n_inc == 0):
        studies = _apply_edits(ss.parsed, edited)
        bar = st.progress(0.0, text="Starting…")

        def progress(p: float, msg: str):
            bar.progress(min(max(p, 0.0), 1.0), text=msg)

        dossiers, info = pipeline.run(studies, rw, online=True, progress=progress, mode=mode)
        ss.assessment = build(studies, dossiers, info, rw, {"online": True})
        bar.progress(1.0, text="Done")
        goto(PAGES[1])
        st.rerun()


def _clear_all():
    ss.input_round += 1
    ss.parsed, ss.reports, ss.assessment = [], [], None
    for k in ("editor", "dupes_dropped", "study_sel", "_study"):
        ss.pop(k, None)


def _set_parsed(studies: list[Study]):
    studies = inputs.assign_ids(studies)
    studies, dropped = inputs.deduplicate(studies)
    ss.parsed = inputs.assign_ids(studies)
    ss.dupes_dropped = dropped
    ss.pop("editor", None)


def _apply_edits(parsed: list[Study], edited: pd.DataFrame) -> list[Study]:
    from inspectsr.normalize import normalize_doi, normalize_pmid, detect_design, find_trial_ids
    out = []
    by_id = {s.uid: s for s in parsed}
    for _, r in edited.iterrows():
        if not r["Include"]:
            continue
        s = Study.from_dict(by_id[r["ID"]].to_dict())
        s.title = r["Title"] or None
        s.authors = [x.strip() for x in str(r["Authors"] or "").split(";") if x.strip()]
        s.year = int(r["Year"]) if pd.notna(r["Year"]) else None
        new_doi, new_pmid = normalize_doi(r["DOI"]), normalize_pmid(r["PMID"])
        if (new_doi, new_pmid) != (s.doi, s.pmid):
            s.id_notes.append("Identifier edited by reviewer.")
            s.id_method, s.id_confidence = "as supplied", "high"
        s.doi, s.pmid = new_doi, new_pmid
        s.trial_ids = find_trial_ids(str(r["Trial IDs"] or ""))
        s.design = detect_design(s.title, s.abstract, s.pub_types)
        out.append(s)
    return inputs.assign_ids(out)


# =====================================================================================
# 2. OVERVIEW
# =====================================================================================

def need_assessment() -> Assessment | None:
    if ss.pop("_stale_notice", False):
        st.warning("The app was updated since your last check, so the previous results were cleared. "
                   "Please run the check again on the Input page.")
    if ss.assessment is None:
        st.info("No results yet. Start on the **Input** page.")
        st.button("Go to Input", on_click=goto, args=(PAGES[0],))
        return None
    return ss.assessment


def _doi_link(d):
    return f"https://doi.org/{d}" if d else None


def _clickable_table(df: pd.DataFrame, key: str, cols: list[str], config: dict | None = None):
    """Table where clicking a row opens that study."""
    view = df[cols].assign(DOI=df["DOI"].map(_doi_link)) if "DOI" in cols else df[cols]
    ev = st.dataframe(view, hide_index=True, width="stretch", key=key, on_select="rerun",
                      selection_mode="single-row",
                      column_config={"DOI": st.column_config.LinkColumn(display_text=r"https://doi\.org/(.*)"),
                                     "Title": st.column_config.TextColumn(width="medium"),
                                     "Why": st.column_config.TextColumn(width="large"), **(config or {})})
    rows = ev.selection.rows if ev and ev.selection else []
    if rows:
        open_study(df.iloc[rows[0]]["ID"])
        st.rerun()


def page_overview():
    a = need_assessment()
    if not a:
        return
    summ = a.summary_table()
    st.title("Overview")

    with st.container(border=True):
        st.markdown("#### Summary")
        for line in charts_summary(summ):
            st.markdown(f"- {line}")

    k = st.columns(4)
    n_ret = int((summ["Group"] == "retracted").sum())
    n_con = int((summ["Group"] == "concern").sum())
    n_none = int((summ["Group"] == "none").sum())
    n_noid = int(summ["Identification"].str.startswith("Not identified").sum())
    k[0].metric("✕ Retracted", n_ret)
    k[1].metric("! Other findings", n_con, help="Expressions of concern, corrections, possible matches, "
                "authors with other retractions, late trial registration.")
    k[2].metric("✓ No notices found", n_none)
    k[3].metric("? Not identified", n_noid, help="No DOI or PubMed ID could be found; checked by title only.")

    st.markdown("#### Retracted studies")
    ret = summ[summ["Group"] == "retracted"]
    if ret.empty:
        st.success("No retracted studies found.")
    else:
        st.caption("Click a row to open the study.")
        _clickable_table(ret, "tbl_ret", ["Status", "Study", "Title", "Journal", "Why", "Reviewer decision", "DOI"])

    st.markdown("#### Other findings to check")
    con = summ[summ["Group"] == "concern"]
    if con.empty:
        st.success("Nothing else to check.")
    else:
        st.caption("Each row says why it is listed. Click a row to open the study.")
        _clickable_table(con, "tbl_con", ["Status", "Study", "Title", "Why", "Reviewer decision", "DOI"])

    noid = summ[summ["Identification"].str.startswith("Not identified")]
    if not noid.empty:
        with st.expander(f"{len(noid)} reference(s) could not be identified"):
            st.markdown(
                "For these references no DOI or PubMed ID was found: not in your file, not by an exact title match in "
                "Retraction Watch, not by a PubMed or OpenAlex title search (nor by Crossref, if you switched it on). They were still "
                "checked against **Retraction Watch by title** (exact and similar titles), but not against the other "
                "sources. Add the DOI on the Input page and run again for a full check.")
            st.dataframe(noid[["Study", "Title", "Journal", "Year", "Status", "Source file"]], hide_index=True,
                         width="stretch")

    st.markdown("#### How the studies were identified")
    st.caption("To check a study reliably, the app first needs its DOI or PubMed ID. It tries these steps in order "
               "and stops at the first that works. A step showing 0 just means no study needed it or matched there.")
    steps_def = [("DOI/PMID in your file", "1 · DOI or PubMed ID already in your file"),
                 ("Retraction Watch title match", "2 · Exact title found in Retraction Watch"),
                 ("PubMed title match", "3 · Found by title in PubMed"),
                 ("OpenAlex title match", "4 · Found by title in OpenAlex (Extended)"),
                 ("Crossref citation match", "5 · Found by citation in Crossref (Extended)"),
                 ("Not identified (title only)", "Not identified (title only)")]
    counts = summ["Identification"].value_counts()
    steps = [(label, int(counts.get(key, 0))) for key, label in steps_def]
    st.altair_chart(charts.funnel(steps), width="stretch")
    n_all, n_id = len(summ), int((~summ["Identification"].str.startswith("Not identified")).sum())
    others = [k for k in ("PubMed", "Crossref", "OpenAlex") if k in a.run.sources]
    st.caption(f"Check type: **{a.run.sources.get('Check type', '')}**. After identification, **all {n_all} studies "
               f"were checked against Retraction Watch**: {n_id} by DOI/PubMed ID and title, and {n_all - n_id} by "
               f"title only." + (f" The identified studies were also checked in {', '.join(others)}." if others else ""))
    if a.run.duplicates:
        st.caption(f"{len(a.run.duplicates)} duplicate(s) merged after identification.")

    with st.expander(f"All {len(summ)} studies"):
        st.dataframe(summ.drop(columns=["Group"]).assign(DOI=summ["DOI"].map(_doi_link)), hide_index=True,
                     width="stretch", column_config={"DOI": st.column_config.LinkColumn(display_text=r"https://doi\.org/(.*)")})


def charts_summary(summ: pd.DataFrame) -> list[str]:
    from inspectsr.status import summary_sentences
    srcs = [k for k in ("PubMed", "Crossref", "OpenAlex") if k in ss.assessment.run.sources]
    return summary_sentences(summ.to_dict("records"), ", ".join(srcs))


# =====================================================================================
# 3. FLAGGED STUDIES
# =====================================================================================

def _links(s: Study) -> str:
    links = []
    if s.doi:
        links += [f"[Article (DOI)](https://doi.org/{s.doi})", f"[PubPeer](https://pubpeer.com/search?q={s.doi})"]
    if s.pmid:
        links.append(f"[PubMed](https://pubmed.ncbi.nlm.nih.gov/{s.pmid}/)")
    q = (s.doi or s.title or "")[:200]
    if q:
        links.append(f"[Retraction Watch database](https://retractiondatabase.org/RetractionSearch.aspx#?ttl={q})")
    for t in s.trial_ids:
        links.append(f"[{t.split(': ')[1]}]({registry_url(t)})")
    return " · ".join(links)


def _notices_df(ns) -> pd.DataFrame:
    return pd.DataFrame([{"Notice": n.type + (" (possible match)" if n.possible else ""),
                          "Date": (n.date or "")[:10], "Reported by": n.sources_label,
                          "Reasons (Retraction Watch)": "; ".join(n.reasons), "Note": n.note,
                          "Notice link": n.links[0] if n.links else None} for n in ns])


def _cb_select_all(value: bool):
    a = ss.assessment
    for u in a.flagged_uids():
        a.review_set[u] = value
    ss["queue_round"] = ss.get("queue_round", 0) + 1  # fresh editor so it shows the new ticks


def _cb_decision(uid: str):
    a = ss.assessment
    a.decisions[uid] = {"decision": ss.get(f"dec|{uid}") or DECISIONS[0], "note": ss.get(f"note|{uid}") or ""}


def page_flagged():
    a = need_assessment()
    if not a:
        return
    st.title("Flagged studies")
    summ = a.summary_table().set_index("ID")
    flagged = a.flagged_uids()
    if not flagged:
        st.success("No flagged studies: nothing to review.")
        return
    bar = st.empty()

    with st.expander(f"Choose which flagged studies to review ({len(a.review_queue())} of {len(flagged)} selected)",
                     expanded=False):
        st.caption("Ticked studies are in your review queue and counted in the progress bar. Unticked ones appear "
                   "as “Not reviewed (not selected)” in the export. By default retractions, expressions of concern "
                   "and possible matches are ticked.")
        b1, b2, _ = st.columns([1, 1, 4])
        b1.button("Select all", on_click=_cb_select_all, args=(True,), width="stretch")
        b2.button("Select none", on_click=_cb_select_all, args=(False,), width="stretch")
        q = pd.DataFrame([{"Review?": a.selected(u), "ID": u, "Status": a.raw_status(u),
                           "Study": summ.loc[u, "Study"], "Why": summ.loc[u, "Why"],
                           "Decision": a.decision(u)} for u in flagged])
        edited = st.data_editor(q, hide_index=True, width="stretch", key=f"queue_{ss.get('queue_round', 0)}",
                                disabled=["ID", "Status", "Study", "Why", "Decision"],
                                column_config={"Review?": st.column_config.CheckboxColumn(width="small"),
                                               "Why": st.column_config.TextColumn(width="large")})
        for _, r in edited.iterrows():
            a.review_set[r["ID"]] = bool(r["Review?"])

    done, total = a.review_progress()
    if total:
        bar.progress(done / total, text=f"**Reviewed {done} of {total} studies selected for review**"
                                        + (" · all done ✓" if done == total else ""))
    else:
        bar.info("No studies selected for review. Open “Choose which flagged studies to review” below.")

    show = st.radio("Show", ["To review", "All flagged", "All studies"], horizontal=True, label_visibility="collapsed")
    opts = {"To review": a.review_queue(), "All flagged": flagged, "All studies": list(summ.index)}[show] or flagged
    if "_study" in ss and ss["_study"] not in opts:
        opts = [ss["_study"]] + opts
    if "_study" in ss:
        ss.study_sel = ss.pop("_study")
    if ss.get("study_sel") not in opts:
        ss.study_sel = opts[0]
    i = opts.index(ss.study_sel)
    queue = set(a.review_queue())
    unreviewed = [u for u in opts[i + 1:] + opts[:i] if u in queue and a.decision(u) == NOT_YET]
    c1, c2, c3, c4 = st.columns([1, 5, 1, 1.4])
    if c1.button("← Previous", disabled=i == 0, width="stretch"):
        ss.study_sel = opts[i - 1]
        st.rerun()
    if c3.button("Next →", disabled=i == len(opts) - 1, width="stretch"):
        ss.study_sel = opts[i + 1]
        st.rerun()
    if c4.button("Next unreviewed →", disabled=not unreviewed, width="stretch", type="primary"):
        ss.study_sel = unreviewed[0]
        st.rerun()

    def _label(u):
        st_ = a.raw_status(u)
        return f"{STATUS_INFO[st_][2]} {st_} · {summ.loc[u, 'Study']} · {(summ.loc[u, 'Title'] or '')[:70]}"

    c2.selectbox("Study", opts, key="study_sel", label_visibility="collapsed", format_func=_label)
    uid = ss.study_sel
    s = a.study(uid)
    status, fl = a.status(uid)

    st.markdown(f"### {s.title or s.raw[:200]}")
    meta = " · ".join(filter(None, ["; ".join(s.authors[:6]) + (" et al." if len(s.authors) > 6 else ""),
                                    s.journal, str(s.year) if s.year else None]))
    st.markdown(f"<span class='muted'>{meta}</span>", unsafe_allow_html=True)
    st.markdown(_links(s))

    dec = a.final_decision(uid)
    dec_badge = "" if not dec else badge(
        ("○ " if dec in (NOT_YET, NOT_SELECTED) else "✓ ") + dec,
        {"Confirmed – exclude": "#d03b3b", "Confirmed – keep": "#fab219", "False match – ignore": "#0ca30c"}.get(dec, "#c3c2b7"))
    st.markdown(status_badge(a.raw_status(uid)) + " " + dec_badge, unsafe_allow_html=True)
    for f in fl:
        st.markdown(f"- **{f.status}:** {f.why}")
    if not fl:
        st.markdown("No retraction, expression of concern or correction was found in any source.")

    ns = a.notices(uid)
    st.markdown(f"#### Notices ({len(ns)})")
    if ns:
        st.caption("Each notice is shown once, even when several databases report it. The databases keep only the "
                   "notice's title and citation: open the link to read the full notice on the journal's website.")
        for n in ns:
            _notice_card(n)
    else:
        st.caption("None found.")

    with st.container(border=True):
        st.markdown("**Your decision**")
        d1, d2 = st.columns([2, 5])
        cur = a.decisions.get(uid, {})
        d1.radio("Your decision", DECISIONS, index=DECISIONS.index(cur.get("decision", DECISIONS[0])),
                 key=f"dec|{uid}", on_change=_cb_decision, args=(uid,), label_visibility="collapsed",
                 help="Exclude/keep: the finding is real and you decided what to do with the study. "
                      "False match: a different paper (e.g. similar title); the flags are ignored.")
        d2.text_area("Note", value=cur.get("note", ""), key=f"note|{uid}", on_change=_cb_decision, args=(uid,),
                     placeholder="e.g. Retracted in 2018 for data fabrication: excluded from the meta-analysis", height=100,
                     label_visibility="collapsed")
        st.caption("Saved automatically when you click outside the box. Your decision and note are included in the "
                   "Excel export (Summary and Notices sheets).")

    r13, r22 = a.results[uid].get("1.3"), a.results[uid].get("2.2")
    with st.expander(f"Authors' other retractions — {r13.proposed.value if r13 else 'n/a'}"):
        if r13:
            st.markdown(r13.rationale)
            if r13.evidence:
                st.dataframe(_evidence_df(r13), hide_index=True, width="stretch",
                             column_config={"Link": st.column_config.LinkColumn(display_text="open ↗")})
    with st.expander(f"Trial registration — {r22.proposed.value if r22 else 'n/a'}"):
        if r22:
            st.markdown(r22.rationale)
            if r22.evidence:
                st.dataframe(_evidence_df(r22), hide_index=True, width="stretch",
                             column_config={"Link": st.column_config.LinkColumn(display_text="open ↗")})
    with st.expander("All records found (per source)"):
        ev = pd.concat([_evidence_df(a.results[uid][c]) for c in ("1.1", "1.2") if a.results[uid].get(c)],
                       ignore_index=True)
        if ev.empty:
            st.caption("No records.")
        else:
            st.dataframe(ev, hide_index=True, width="stretch",
                         column_config={"Link": st.column_config.LinkColumn(display_text="open ↗")})
    with st.expander(f"Identification · {s.id_method} · design detected: {s.design}"):
        st.write({"DOI": s.doi, "PMID": s.pmid, "Trial registrations": s.trial_ids,
                  "Publication types (PubMed)": s.pub_types, "Source": f"{s.source_file} ({s.source_format})",
                  "Notes": s.id_notes, "Citation as supplied": s.raw[:500]})


MAIN_SOURCES = ["Retraction Watch", "Crossref", "PubMed", "OpenAlex"]


def _notice_card(n):
    _, color, ic, _ = STATUS_INFO[{"Retraction": "Retracted", "Expression of concern": "Expression of concern",
                                       "Correction": "Correction"}.get(n.type, "Correction")]
    with st.container(border=True):
        head = badge(f"{ic} {n.type}", color) + (f" &nbsp;**{n.date[:10]}**" if n.date else " &nbsp;date unknown")
        if n.possible:
            head += " &nbsp;" + badge("? possible match – verify it is the same paper", "#898781")
        st.markdown(head, unsafe_allow_html=True)
        if n.title:
            st.markdown(f"**Notice:** {n.title}")
        found = []
        for src in MAIN_SOURCES:
            if src == "OpenAlex" and n.type != "Retraction":
                continue  # OpenAlex only records whether a paper is retracted
            if src == "Crossref":
                hit = "Crossref" in n.sources
                only_copy = not hit and "Crossref (from RW)" in n.sources
                found.append("✓ Crossref (copy of Retraction Watch)" if only_copy else f"{'✓' if hit else '✗'} Crossref")
            else:
                found.append(f"{'✓' if src in n.sources else '✗'} {src}")
        if "Your reference list" in n.sources:
            found.insert(0, "✓ your reference list (the reference is the notice itself)")
        st.markdown("**Found in:** " + " · ".join(found))
        if n.reasons:
            st.markdown("**Reasons (Retraction Watch):** " + "; ".join(n.reasons))
        if n.note:
            st.markdown(f"**Retraction Watch note:** {n.note}")
        links = []
        doi_links = [l for l in n.links if "doi.org" in l]
        pm_links = [l for l in n.links if "pubmed" in l]
        if doi_links:
            links.append(f"[Read the notice ↗]({doi_links[0]})")
        if pm_links:
            links.append(f"[Notice in PubMed ↗]({pm_links[0]})")
        if links:
            st.markdown(" · ".join(links))


def _evidence_df(r) -> pd.DataFrame:
    return pd.DataFrame([{"Source": e.source, "Type": e.kind, "Date": e.date, "Detail": e.detail,
                          "Matched by": e.match, "Link": e.url} for e in r.evidence])


# =====================================================================================
# 4. AUTHORS
# =====================================================================================

def _author_links(name: str) -> str:
    from urllib.parse import quote_plus
    from inspectsr.normalize import split_name
    fam, given = split_name(name)
    initials = "".join(w[0] for w in given.replace("-", " ").split() if w)
    pm = quote_plus(f"{fam} {initials}[Author]")
    full = quote_plus(f"{given} {fam}".strip())
    return (f"[PubMed author search ↗](https://pubmed.ncbi.nlm.nih.gov/?term={pm}) · "
            f"[ORCID search ↗](https://orcid.org/orcid-search/search?searchQuery={full}) · "
            f"[Google Scholar ↗](https://scholar.google.com/scholar?q=author%3A%22{full}%22)")


def _set_author_verdict(key: str, verdict: str):
    a = ss.assessment
    cur = a.author_reviews.get(key, {})
    a.author_reviews[key] = {"verdict": verdict, "note": cur.get("note", "")}


def _set_author_note(key: str):
    a = ss.assessment
    cur = a.author_reviews.get(key, {})
    a.author_reviews[key] = {"verdict": cur.get("verdict", AUTHOR_UNCHECKED), "note": ss.get(f"anote|{key}", "")}


WHY_SAME = {"shared co-author": "Likely the same person: some of those papers share co-authors with your study.",
            "same OpenAlex author ID": "Likely the same person: OpenAlex links those papers to the same author profile."}


def _papers_table(papers: list[dict]):
    df = pd.DataFrame([{"Paper": p["title"], "Notice": p["nature"], "Year": str(p["date"] or "")[:4],
                        "Reasons (Retraction Watch)": "; ".join(p["reasons"]), "Link": p.get("notice")}
                       for p in papers])
    st.dataframe(df, hide_index=True, width="stretch",
                 column_config={"Link": st.column_config.LinkColumn(display_text="open ↗"),
                                "Paper": st.column_config.TextColumn(width="large")})


def page_authors():
    a = need_assessment()
    if not a:
        return
    st.title("Authors with other retractions")
    st.markdown("For every included study, the app looked up the **first and last author** in Retraction Watch. "
                "The authors below have **other** retracted papers. That can be a warning sign about the research "
                "team. Check that it is really the same person.")
    t = a.authors_table()
    leads = t[t["Flagged lead"]] if not t.empty else t
    if leads.empty:
        st.success("None of the first or last authors has other retractions for research-integrity reasons.")
    else:
        done = int((leads["Your verdict"] != AUTHOR_UNCHECKED).sum())
        st.progress(done / len(leads), text=f"**{done} of {len(leads)} authors checked**"
                                            + (" ✓" if done == len(leads) else ""))
        st.caption("How to check: open one or two of the other papers. Same field, same institution, shared "
                   "co-authors? Then it is the same person. “Different person” removes the warning from their studies.")
        for _, r in leads.iterrows():
            papers = r["_papers"]
            n, n_int = len(papers), int(r["Integrity-related"])
            with st.container(border=True):
                verdict = r["Your verdict"]
                import re as _re
                roles = []
                for x in r["Studies in your review"].split("; "):
                    m = _re.match(r"\S+ (.*) \((first|last) author\)$", x)
                    roles.append(f"{m.group(2)} author of *{m.group(1)}*" if m else x)
                head = f"**{r['Author']}** — " + "; ".join(roles)
                st.markdown(head)
                st.markdown(f"Has **{n} other retracted paper{'s' if n != 1 else ''} or concern{'s' if n != 1 else ''}**"
                            f" ({n_int} for research-integrity reasons, e.g. unreliable data, misconduct or paper "
                            f"mills). {WHY_SAME.get(r['Match'], '')}")
                with st.expander(f"Show the {n} paper{'s' if n != 1 else ''}"):
                    _papers_table(papers)
                    st.markdown(_author_links(r["Author"]))
                st.markdown("**Is this the same person?**")
                c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
                for col, (label, v) in zip((c1, c2, c3), [("Same person", "Same person"),
                                                          ("Different person", "Different person"),
                                                          ("Not sure", "Unsure")]):
                    col.button(("✓ " if verdict == v else "") + label, key=f"av|{r['key']}|{v}", width="stretch",
                               type="primary" if verdict == v else "secondary",
                               on_click=_set_author_verdict, args=(r["key"], v))
                c4.text_input("Note", value=r["Note"], key=f"anote|{r['key']}", label_visibility="collapsed",
                              placeholder="Note (optional)", on_change=_set_author_note, args=(r["key"],))

    others = t[~t["Flagged lead"]] if not t.empty else t
    if not others.empty:
        with st.expander(f"Possible namesakes ({len(others)}): same name only, not flagged"):
            st.caption("These authors share a name with someone who has a retracted paper, but nothing else links "
                       "them (no shared co-authors), so they are often different people. They don't flag any study. "
                       "If you know one is the same person, mark it, and the warning is added to their studies.")
            for _, r in others.iterrows():
                st.markdown(f"**{r['Author']}** — {r['Studies in your review']} · "
                            f"{len(r['_papers'])} paper(s) under this name")
                c1, c2, _ = st.columns([1, 1, 4])
                for col, (label, v) in zip((c1, c2), [("Same person", "Same person"), ("Different person", "Different person")]):
                    col.button(("✓ " if r["Your verdict"] == v else "") + label, key=f"av|{r['key']}|{v}",
                               width="stretch", type="primary" if r["Your verdict"] == v else "secondary",
                               on_click=_set_author_verdict, args=(r["key"], v))
                with st.expander("Show the papers"):
                    _papers_table(r["_papers"])


# =====================================================================================
# NOTICE FIGURES (shown on the final overview page)
# =====================================================================================

def _notice_figures(a: Assessment):
    nt = a.notices_table()
    nt = nt[~nt["Possible match only"]]
    if nt.empty:
        return
    studies = {s.uid: s for s in a.studies}

    st.markdown("#### Timeline of notices")
    st.caption("When each study with a notice was published, and when its notices appeared. A study can have "
               "several notices, e.g. an expression of concern followed by a retraction.")
    pub = [{"ID": u, "Study": studies[u].label, "Event": "Publication",
            "Date": studies[u].pub_date or (f"{studies[u].year}-07-01" if studies[u].year else None),
            "Detail": studies[u].title} for u in nt["ID"].unique()]
    tl = pd.concat([pd.DataFrame(pub), nt.rename(columns={"Notice": "Event", "Reported by": "Detail"})
                    [["ID", "Study", "Event", "Date", "Detail"]]], ignore_index=True)
    tl = tl[tl["Date"].fillna("").str.len() >= 7]
    tl["Study"] = tl["ID"] + " " + tl["Study"]
    ch = charts.notice_timeline(tl)
    if ch is not None:
        st.altair_chart(ch, width="stretch")
    ret = nt[nt["Notice"] == "Retraction"].dropna(subset=["Date"])
    yrs = []
    for uid, r in ret.groupby("ID")["Date"].min().items():
        s = studies[uid]
        p = s.pub_date or (f"{s.year}-07-01" if s.year else None)
        if p and len(r) >= 7:
            yrs.append((pd.to_datetime(r) - pd.to_datetime(p)).days / 365.25)
    if yrs:
        st.caption(f"Median time from publication to retraction: **{pd.Series(yrs).median():.1f} years** "
                   f"(range {min(yrs):.1f}–{max(yrs):.1f}).")

    ret_n = nt[nt["Notice"] == "Retraction"][["ID", "Study", "Reasons (Retraction Watch)"]]
    per = (ret_n.assign(Reason=ret_n["Reasons (Retraction Watch)"].str.split("; ")).explode("Reason")
           .query("Reason.str.len() > 0", engine="python").drop_duplicates(["ID", "Reason"]))
    if not per.empty:
        st.markdown("#### Why were they retracted?")
        n_st, n_r = per["ID"].nunique(), per["Reason"].nunique()
        st.caption(f"Retraction Watch lists several reasons per retraction, so {n_st} retracted "
                   f"stud{'y' if n_st == 1 else 'ies'} give {n_r} different reasons. Each bar shows how many "
                   "of the retracted studies have that reason.")
        c1, c2 = st.columns([3, 2], gap="large")
        rdf = per.groupby("Reason")["ID"].nunique().sort_values(ascending=False).head(12) \
            .rename_axis("Item").reset_index(name="Studies")
        c1.altair_chart(explore._hbar(rdf, value="Studies", axis_title="Retracted studies"), width="stretch")
        if n_st <= 10:
            c2.dataframe(per.groupby(["ID", "Study"])["Reason"].apply("; ".join).reset_index()
                         .drop(columns="ID").rename(columns={"Reason": "Reasons (Retraction Watch)"}),
                         hide_index=True, width="stretch",
                         column_config={"Reasons (Retraction Watch)": st.column_config.TextColumn(width="large")})

    st.markdown("#### Which sources reported each notice")
    st.caption("Sources differ in coverage and timing, so no single database is enough. "
               "“Crossref (from RW)” is Retraction Watch data redistributed by Crossref.")
    sa = nt.assign(Source=nt["Reported by"].str.split(", ")).explode("Source")
    st.altair_chart(charts.source_agreement(sa.assign(Flag=sa["Notice"])), width="content")


# =====================================================================================
# 4. FINAL OVERVIEW & EXPORT
# =====================================================================================

def page_export():
    a = need_assessment()
    if not a:
        return
    import altair as alt
    st.title("Final overview & export")
    done, total = a.review_progress()
    oc = a.outcome_counts().set_index("Outcome")["Studies"]
    n_ret = sum(group(a.raw_status(s.uid)) == "retracted" for s in a.studies)
    n_notices = int(a.notices_per_year()["Notices"].sum()) if not a.notices_per_year().empty else 0
    k = st.columns(7)
    k[6].metric("Not identified", sum(not s.identified for s in a.studies),
                help="No DOI or PubMed ID found: checked against Retraction Watch by title only. Listed below.")
    k[0].metric("Studies checked", len(a.studies))
    k[1].metric("Flagged", len(a.flagged_uids()))
    k[2].metric("Retracted", n_ret)
    k[3].metric("Notices", n_notices, help="Retractions, expressions of concern and corrections, merged across sources.")
    k[4].metric("Reviewed", f"{done}/{total}", help="Of the studies you selected for review.")
    k[5].metric("Excluded", int(oc.get("Confirmed – exclude", 0)))
    if total and done < total:
        st.warning(f"{total - done} selected study(ies) are not reviewed yet. They will be exported as “Not yet reviewed”.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### Review of flagged studies")
        ocd = a.outcome_counts()
        ocd = ocd[ocd["Studies"] > 0]
        colors = {"Confirmed – exclude": "#d03b3b", "Confirmed – keep": "#fab219", "False match – ignore": "#0ca30c",
                  "Not yet reviewed": "#898781", "Not reviewed (not selected)": "#d9d8d2"}
        bars = alt.Chart(ocd).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("Studies:Q", title=None, axis=alt.Axis(tickMinStep=1, grid=False)),
            y=alt.Y("Outcome:N", sort=list(colors), title=None, axis=alt.Axis(labelLimit=240)),
            color=alt.Color("Outcome:N", scale=alt.Scale(domain=list(colors), range=list(colors.values())), legend=None),
            tooltip=["Outcome", "Studies"])
        st.altair_chart((bars + bars.mark_text(align="left", dx=4, color="#898781").encode(text="Studies:Q"))
                        .properties(height=alt.Step(30), width="container"), width="stretch")
    with c2:
        st.markdown("#### Notices per year")
        npy = a.notices_per_year()
        if npy.empty:
            st.caption("No notices found.")
        else:
            types = [t for t in explore.NATURES if t in set(npy["Notice"])]
            ch = alt.Chart(npy).mark_bar(stroke="white", strokeWidth=1).encode(
                x=alt.X("Year:O", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Notices:Q", title=None, axis=alt.Axis(tickMinStep=1)),
                color=alt.Color("Notice:N", scale=alt.Scale(domain=types, range=[explore.NATURE_COLORS[t] for t in types]),
                                legend=alt.Legend(title=None, orient="bottom")),
                tooltip=["Year", "Notice", "Notices"]).properties(height=240, width="container")
            st.altair_chart(ch, width="stretch")

    st.markdown("#### Included studies by publication year")
    spy = a.studies_per_year()
    if not spy.empty:
        order = ["Retracted", "Other findings", "No notices found"]
        ch = alt.Chart(spy).mark_bar(stroke="white", strokeWidth=1).encode(
            x=alt.X("Year:O", title=None, axis=alt.Axis(labelAngle=0, labelOverlap=True)),
            y=alt.Y("Studies:Q", title=None, axis=alt.Axis(tickMinStep=1)),
            color=alt.Color("Finding:N", scale=alt.Scale(domain=order, range=["#d03b3b", "#fab219", "#0ca30c"]),
                            legend=alt.Legend(title=None, orient="bottom")),
            order=alt.Order("Finding:N"),
            tooltip=["Year", "Finding", "Studies"]).properties(height=240, width="container")
        st.altair_chart(ch, width="stretch")

    _notice_figures(a)

    noid = a.not_identified_table()
    st.markdown(f"#### Not identified ({len(noid)})")
    if noid.empty:
        st.caption("Every study was identified by DOI or PubMed ID.")
    else:
        st.caption("No DOI or PubMed ID could be found for these references, so they were checked against Retraction "
                   "Watch **by title only** (exact and similar titles) and not against the other sources. A clean "
                   "result here is less certain. Add the DOI on the Input page and run again, or check them by hand. "
                   "They are listed in the **Not identified** sheet of the Excel workbook.")
        st.dataframe(noid[["Study", "Title", "Journal", "Year", "Result of title check", "Source file"]],
                     hide_index=True, width="stretch",
                     column_config={"Title": st.column_config.TextColumn(width="large")})

    st.markdown("#### Download")
    st.markdown("The Excel workbook starts with an **Overview** sheet (these numbers and charts), followed by "
                "**Summary** (one row per study: status, why, whether you selected it for review, your decision and "
                "note), **Notices**, **Not identified**, **Checks**, **Evidence** (every record from every source), **Provenance** and a "
                "draft **Methods text**.")
    d1, d2 = st.columns(2)
    stamp = date.today().isoformat()
    d1.download_button("⬇ Excel workbook", a.to_excel(), f"retraction-check_{stamp}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary",
                       width="stretch")
    d2.download_button("⬇ Summary (CSV)", a.summary_table().drop(columns=["Group"]).to_csv(index=False),
                       f"retraction-check_summary_{stamp}.csv", mime="text/csv", width="stretch")
    st.markdown("#### Cleaned reference list (RIS)")
    st.caption("A RIS file to import back into EndNote, Zotero or Covidence. References that came from a RIS file "
               "are written back with all their original fields.")
    r1, r2, r3 = st.columns(3)
    drop = r1.checkbox("Leave out studies you excluded", value=True)
    repl = r2.checkbox("Replace notices (e.g. a corrigendum) by the original article", value=True,
                       help="Only when the original article could be identified. If the original was already in "
                            "your list, the notice was merged into it during the check.")
    notes = r3.checkbox("Add the check result and your decision as a note (N1)", value=True)
    ris_text, rs = a.to_ris(drop_excluded=drop, replace_notices=repl, add_notes=notes)
    st.download_button(f"⬇ Cleaned RIS ({rs['written']} references)", ris_text.encode("utf-8"),
                       f"included-studies_checked_{stamp}.ris", mime="application/x-research-info-systems")
    bits = []
    if rs["excluded"]:
        bits.append(f"{rs['excluded']} excluded study(ies) left out")
    if rs["replaced"]:
        bits.append(f"{rs['replaced']} notice(s) replaced by the original article")
    merged = [x for x in a.run.duplicates if "is a notice for" in x]
    if merged:
        bits.append(f"{len(merged)} notice(s) merged into their original article during the check")
    if bits:
        st.caption(" · ".join(bits) + ".")

    st.markdown("#### Draft methods text")
    st.code(a.methods_text(), language=None, wrap_lines=True)
    with st.expander("Provenance (database versions and query dates)"):
        st.dataframe(a.provenance_table(), hide_index=True, width="stretch")
    if a.run.errors:
        st.warning("Some sources returned errors during the run; affected checks may be incomplete. Re-run later.")


# =====================================================================================
# EXPLORE RETRACTION WATCH
# =====================================================================================

def page_explore():
    explore.render(rw)


{
    PAGES[0]: page_input,
    PAGES[1]: page_overview,
    PAGES[2]: page_flagged,
    PAGES[3]: page_authors,
    PAGES[4]: page_export,
    PAGES[5]: page_explore,
}[page]()
