"""Retraction Watch check: Streamlit front end.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import os
from datetime import date

import pandas as pd
import streamlit as st

from inspectsr import __version__, charts, inputs, pipeline
from inspectsr.assessment import Assessment, build
from inspectsr import explore
from inspectsr.framework import CHECKS_BY_ID, DOMAINS, DOMAIN_SHORT, checks_in_domain
from inspectsr.judgement import overall_label, suggest_overall
from inspectsr.models import Answer, Judgement, Study
from inspectsr.sources import retraction_watch as rwmod
from inspectsr.sources.registries import registry_url

st.set_page_config(page_title="Retraction Watch check", page_icon="🔎", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
PAGES = ["1 · Input", "2 · Overview", "3 · Study assessments", "4 · Notices & evidence", "5 · Export & methods",
         "Explore Retraction Watch"]
ANSWER_OPTIONS = ["(use proposed)", Answer.YES.value, Answer.NO.value, Answer.UNCLEAR.value, Answer.NA.value]
JUDGEMENT_OPTIONS = ["(use suggested)", Judgement.NO_CONCERNS.value, Judgement.SOME_CONCERNS.value,
                     Judgement.SERIOUS_CONCERNS.value]

st.markdown("""
<style>
.badge{display:inline-block;padding:2px 9px;border-radius:999px;font-size:0.82rem;font-weight:600;
       color:#fff;white-space:nowrap;margin-right:4px}
.badge.dark{color:#0b0b0b}
.muted{color:#898781;font-size:0.85rem}
.check-q{font-weight:600}
</style>
""", unsafe_allow_html=True)

ANSWER_BADGE = {
    Answer.YES.value: ("#d03b3b", "✕ Yes"),
    Answer.NO.value: ("#0ca30c", "✓ No"),
    Answer.UNCLEAR.value: ("#898781", "? Unclear"),
    Answer.NA.value: ("#c3c2b7", "– N/A"),
    Answer.NOT_ASSESSED.value: ("#d9d8d2", "– Not assessed"),
}


def badge(text: str, color: str) -> str:
    dark = color in ("#fab219", "#d9d8d2", "#c3c2b7")
    return f'<span class="badge{" dark" if dark else ""}" style="background:{color}">{text}</span>'


def judgement_badge(label: str) -> str:
    base = charts._overall_base(label)
    color, ic = charts.STATUS.get(base, ("#d9d8d2", "–"))
    return badge(f"{ic} {label}", color)


def answer_badge(a: str) -> str:
    color, text = ANSWER_BADGE.get(a, ("#d9d8d2", a))
    return badge(text, color)


# ---- data ------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the Retraction Watch database…")
def get_rw(mtime: float):
    return rwmod.load()


def rw_index():
    try:
        mtime = os.path.getmtime(rwmod.LOCAL_CSV) if os.path.exists(rwmod.LOCAL_CSV) else 0
        return get_rw(mtime)
    except Exception as e:
        st.error(f"Could not load the Retraction Watch database: {e}")
        st.stop()


ss = st.session_state
ss.setdefault("page", PAGES[0])
if "_goto" in ss:  # navigation requested by a button in the previous run
    ss.page = ss.pop("_goto")
ss.setdefault("parsed", [])        # list[Study]
ss.setdefault("reports", [])
ss.setdefault("assessment", None)  # Assessment


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
        st.markdown(f"**Current assessment:** {len(a.studies)} studies")
        st.caption(f"Run {a.run.finished}")


def goto(p: str):
    ss["_goto"] = p


# =====================================================================================
# 1. INPUT
# =====================================================================================

def page_input():
    st.title("Check your included studies")
    st.markdown(
        "Upload the **studies included in your review** in any common format. Each study is identified and checked "
        "against the Retraction Watch database, Crossref, PubMed, OpenAlex and ClinicalTrials.gov. Every finding comes "
        "with its evidence, and you can override every proposed answer."
    )
    files = st.file_uploader(
        "Reference files",
        type=["ris", "txt", "nbib", "bib", "xml", "csv", "tsv", "xlsx", "xls", "docx", "pdf"],
        accept_multiple_files=True,
        help="RIS (EndNote, Zotero, Covidence…), PubMed/MEDLINE (.nbib), BibTeX, EndNote XML, CSV/Excel, "
             "Word or PDF reference lists, or plain text.",
    )
    pasted = st.text_area("…or paste references (one per line, numbered list, or a list of DOIs/PMIDs)", height=140,
                          placeholder="1. Horby P, Lim WS, et al. Dexamethasone in Hospitalized Patients with "
                                      "Covid-19. N Engl J Med. 2021;384:693-704.\n10.1056/NEJMoa1602489\nPMID: 30879366")

    if st.button("Read references", type="primary", disabled=not (files or pasted.strip())):
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
    st.caption("Untick records that are not included studies (e.g. notices or background references). You can fix a "
               "DOI, PMID or title directly in the table. Records without DOI/PMID will be looked up via Crossref.")
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

    if st.button(f"Run checks on {n_inc} studies", type="primary", disabled=n_inc == 0):
        studies = _apply_edits(ss.parsed, edited)
        bar = st.progress(0.0, text="Starting…")

        def progress(p: float, msg: str):
            bar.progress(min(max(p, 0.0), 1.0), text=msg)

        dossiers, info = pipeline.run(studies, rw, online=True, progress=progress)
        ss.assessment = build(studies, dossiers, info, rw, {"online": True})
        bar.progress(1.0, text="Done")
        goto(PAGES[1])
        st.rerun()


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
    if ss.assessment is None:
        st.info("No assessment yet. Start on the **Input** page.")
        st.button("Go to Input", on_click=goto, args=(PAGES[0],))
        return None
    return ss.assessment


def page_overview():
    a = need_assessment()
    if not a:
        return
    summ = a.summary_table()
    st.title("Overview")
    if a.run.duplicates:
        st.caption(" ".join(a.run.duplicates))
    serious = summ["Overall (raw)"].eq(Judgement.SERIOUS_CONCERNS.value).sum()
    some = summ["Overall (raw)"].eq(Judgement.SOME_CONCERNS.value).sum()
    k = st.columns(5)
    k[0].metric("Studies", len(summ))
    k[1].metric("✕ Serious concerns", int(serious))
    k[2].metric("! Some concerns", int(some))
    k[3].metric("Checks to verify", int(summ["Needs review"].sum()),
                help="Automated proposals flagged for the reviewer (unclear evidence, notices to read, name matches).")
    k[4].metric("Identified (DOI/PMID)", f"{sum(s.identified for s in a.studies)}/{len(a.studies)}")

    st.markdown("#### Judgements by study")
    st.caption("Suggested judgements are a starting point: confirm or override them on the **Study assessments** "
               "page. Domains 3–4 need the full text and stay *Not assessed* until you answer them.")
    st.altair_chart(charts.traffic_light(summ), width="content")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Share of studies per judgement**")
        st.altair_chart(charts.domain_summary(summ), width="stretch")
    with c2:
        st.markdown("**Identification**")
        steps = [("References supplied", len(a.studies) + len(a.run.duplicates)),
                 ("Unique studies", len(a.studies)),
                 ("With DOI or PMID", sum(s.identified for s in a.studies)),
                 ("Found in PubMed", sum(bool(s.pmid) for s in a.studies)),
                 ("With trial registration ID", sum(bool(s.trial_ids) for s in a.studies))]
        st.altair_chart(charts.funnel(steps), width="stretch")

    st.markdown("#### Study table")
    view = summ.drop(columns=["Overall (raw)"])
    f1, f2 = st.columns([2, 3])
    flt = f1.selectbox("Show", ["All studies", "Serious or some concerns", "Has checks to verify", "RCTs only"])
    if flt == "Serious or some concerns":
        view = view[summ["Overall (raw)"].isin([Judgement.SERIOUS_CONCERNS.value, Judgement.SOME_CONCERNS.value])]
    elif flt == "Has checks to verify":
        view = view[view["Needs review"] > 0]
    elif flt == "RCTs only":
        view = view[view["Design (detected)"] == "RCT"]
    view = view.assign(DOI=view["DOI"].map(lambda d: f"https://doi.org/{d}" if d else None))
    st.dataframe(view, hide_index=True, width="stretch", column_config={
        "DOI": st.column_config.LinkColumn(display_text=r"https://doi\.org/(.*)"),
        "Title": st.column_config.TextColumn(width="medium"),
        "1.1": st.column_config.TextColumn("1.1 Retraction", help=CHECKS_BY_ID["1.1"].question),
        "1.2": st.column_config.TextColumn("1.2 EoC/notices", help=CHECKS_BY_ID["1.2"].question),
        "1.3": st.column_config.TextColumn("1.3 Team", help=CHECKS_BY_ID["1.3"].question),
        "2.2": st.column_config.TextColumn("2.2 Registration", help=CHECKS_BY_ID["2.2"].question),
    })
    st.button("Open study assessments →", on_click=goto, args=(PAGES[2],))


# =====================================================================================
# 3. STUDY ASSESSMENTS
# =====================================================================================

def _links(s: Study) -> str:
    links = []
    if s.doi:
        links += [f"[DOI](https://doi.org/{s.doi})", f"[PubPeer](https://pubpeer.com/search?q={s.doi})"]
    if s.pmid:
        links.append(f"[PubMed](https://pubmed.ncbi.nlm.nih.gov/{s.pmid}/)")
    q = (s.doi or s.title or "")[:200]
    if q:
        links.append(f"[Retraction Watch database](https://retractiondatabase.org/RetractionSearch.aspx#?ttl={q})")
    for t in s.trial_ids:
        links.append(f"[{t.split(': ')[1]}]({registry_url(t)})")
    return " · ".join(links)


def _evidence_df(r) -> pd.DataFrame:
    return pd.DataFrame([{"Source": e.source, "Type": e.kind, "Date": e.date, "Detail": e.detail,
                          "Matched by": e.match, "Link": e.url} for e in r.evidence])


def page_studies():
    a = need_assessment()
    if not a:
        return
    st.title("Study assessments")
    summ = a.summary_table().set_index("ID")
    opts = [s.uid for s in a.studies]
    only = st.toggle("Only studies with checks to verify", value=False)
    if only:
        opts = [u for u in opts if summ.loc[u, "Needs review"] > 0] or opts
    ss.setdefault("study_sel", opts[0])
    if ss.study_sel not in opts:
        ss.study_sel = opts[0]
    nav1, nav2, nav3 = st.columns([1, 6, 1])
    i = opts.index(ss.study_sel)
    if nav1.button("← Previous", disabled=i == 0, width="stretch"):
        ss.study_sel = opts[i - 1]
        st.rerun()
    if nav3.button("Next →", disabled=i == len(opts) - 1, width="stretch"):
        ss.study_sel = opts[i + 1]
        st.rerun()
    nav2.selectbox("Study", opts, key="study_sel", label_visibility="collapsed",
                   format_func=lambda u: f"{u} · {summ.loc[u, 'Study']} · {charts.icon(charts._overall_base(summ.loc[u, 'Overall']))} "
                                         f"{summ.loc[u, 'Overall']} · {(summ.loc[u, 'Title'] or '')[:70]}")
    uid = ss.study_sel
    s = a.study(uid)

    # ---- header
    st.markdown(f"### {s.title or s.raw[:200]}")
    meta = " · ".join(filter(None, [
        "; ".join(s.authors[:6]) + (" et al." if len(s.authors) > 6 else ""),
        s.journal, str(s.year) if s.year else None]))
    st.markdown(f"<span class='muted'>{meta}</span>", unsafe_allow_html=True)
    st.markdown(_links(s))
    with st.expander(f"Identification and metadata · design detected: **{s.design}** · "
                     f"{s.id_method} ({s.id_confidence})"):
        st.write({"DOI": s.doi, "PMID": s.pmid, "Trial registrations": s.trial_ids,
                  "Publication types (PubMed)": s.pub_types, "Source": f"{s.source_file} ({s.source_format})",
                  "Notes": s.id_notes, "Citation as supplied": s.raw[:500]})
        if s.design not in ("RCT", "Unknown", "Other trial"):
            st.warning(f"The checks are designed for randomised controlled trials; this record looks like: {s.design}.")

    # ---- judgement strip
    doms = a.domains(uid)
    ov, complete, ov_flag = a.overall(uid)
    strip = " ".join(f"{DOMAIN_SHORT[d]} {judgement_badge(st_.final.value)}" for d, st_ in doms.items())
    st.markdown(f"{strip} &nbsp; **Overall** {judgement_badge(overall_label(ov, complete))}", unsafe_allow_html=True)

    rv = a.reviewer.get(uid, {})
    dov = a.domain_overrides.get(uid, {})

    for dom, name in DOMAINS.items():
        stt = doms[dom]
        title = f"Domain {dom}: {name} — {charts.icon(stt.final.value)} {stt.final.value} ({stt.answered}/{stt.total} checks answered)"
        with st.expander(title, expanded=dom in (1, 2)):
            for c in checks_in_domain(dom):
                r = a.results[uid][c.id]
                cur = rv.get(c.id, {})
                st.markdown(f"<span class='check-q'>{c.id} {c.question}</span> "
                            f"<a href='{c.guidance_url}' target='_blank' class='muted'>guidance ↗</a>",
                            unsafe_allow_html=True)
                cols = st.columns([2, 5])
                with cols[0]:
                    if r.automated:
                        st.markdown(f"Proposed: {answer_badge(r.proposed.value)}"
                                    + (" <span class='muted'>· verify</span>" if r.needs_review else ""),
                                    unsafe_allow_html=True)
                    else:
                        st.markdown("<span class='muted'>Reviewer check (full text needed)</span>", unsafe_allow_html=True)
                    opts = ANSWER_OPTIONS if r.automated else ["(not assessed)"] + ANSWER_OPTIONS[1:]
                    st.selectbox("Your answer", opts, index=_idx(opts, cur.get("answer")),
                                 key=f"ans|{uid}|{c.id}", label_visibility="collapsed",
                                 on_change=_cb_check, args=(uid, c.id))
                    st.text_input("Note", value=cur.get("note", ""), key=f"note|{uid}|{c.id}",
                                  placeholder="Reason / where in the paper", label_visibility="collapsed",
                                  on_change=_cb_check, args=(uid, c.id))
                with cols[1]:
                    if r.automated:
                        st.markdown(r.rationale)
                        if r.evidence:
                            st.dataframe(_evidence_df(r), hide_index=True, width="stretch",
                                         column_config={"Link": st.column_config.LinkColumn(display_text="open ↗"),
                                                        "Detail": st.column_config.TextColumn(width="large")})
                st.divider()
            cur = dov.get(str(dom), {})
            jc1, jc2 = st.columns([2, 5])
            with jc1:
                st.markdown(f"**Domain judgement** · suggested: {judgement_badge(stt.suggested.value)}", unsafe_allow_html=True)
                st.selectbox("Domain judgement", JUDGEMENT_OPTIONS, index=_idx(JUDGEMENT_OPTIONS, cur.get("judgement")),
                             key=f"dj|{uid}|{dom}", label_visibility="collapsed",
                             on_change=_cb_domain, args=(uid, dom))
            with jc2:
                st.text_input("Reason for domain judgement", value=cur.get("reason", ""), key=f"dr|{uid}|{dom}",
                              placeholder="Report the reason for this judgement",
                              on_change=_cb_domain, args=(uid, dom))

    st.markdown("#### Overall study judgement")
    cur = a.overall_overrides.get(uid, {})
    oc1, oc2 = st.columns([2, 5])
    with oc1:
        sug, sug_complete = suggest_overall(doms)
        st.markdown(f"Suggested: {judgement_badge(overall_label(sug, sug_complete))}", unsafe_allow_html=True)
        st.selectbox("Overall", JUDGEMENT_OPTIONS, index=_idx(JUDGEMENT_OPTIONS, cur.get("judgement")),
                     key=f"oj|{uid}", label_visibility="collapsed", on_change=_cb_overall, args=(uid,))
    with oc2:
        st.text_input("Reason", value=cur.get("reason", ""), key=f"or|{uid}",
                      placeholder="e.g. Retracted for data integrity concerns (check 1.1)",
                      on_change=_cb_overall, args=(uid,))
    st.caption("Guidance: the overall judgement should be at least as severe as the most severe domain. Several "
               "'some concerns' domains may together justify 'serious concerns'. Studies with serious concerns are "
               "usually excluded; pre-specify in the protocol how 'some concerns' studies are handled.")


def _idx(options: list[str], value) -> int:
    return options.index(value) if value in options else 0


# Callbacks write reviewer input into the assessment *before* the rerun, so every badge is current.
def _cb_check(uid: str, cid: str):
    a = ss.assessment
    choice = ss.get(f"ans|{uid}|{cid}") or ""
    note = ss.get(f"note|{uid}|{cid}") or ""
    ans = None if choice.startswith("(") else choice
    rv = a.reviewer.setdefault(uid, {})
    if ans or note:
        rv[cid] = {"answer": ans, "note": note}
    else:
        rv.pop(cid, None)


def _cb_domain(uid: str, dom: int):
    a = ss.assessment
    choice = ss.get(f"dj|{uid}|{dom}") or ""
    reason = ss.get(f"dr|{uid}|{dom}") or ""
    j = None if choice.startswith("(") else choice
    dov = a.domain_overrides.setdefault(uid, {})
    if j or reason:
        dov[str(dom)] = {"judgement": j, "reason": reason}
    else:
        dov.pop(str(dom), None)


def _cb_overall(uid: str):
    a = ss.assessment
    choice = ss.get(f"oj|{uid}") or ""
    reason = ss.get(f"or|{uid}") or ""
    j = None if choice.startswith("(") else choice
    if j or reason:
        a.overall_overrides[uid] = {"judgement": j, "reason": reason}
    else:
        a.overall_overrides.pop(uid, None)


# =====================================================================================
# 4. NOTICES & EVIDENCE
# =====================================================================================

def _event(kind: str) -> str | None:
    k = kind.lower()
    if "reinstat" in k:
        return "Reinstatement"
    if "retract" in k or "withdraw" in k or "removal" in k:
        return "Retraction"
    if "concern" in k:
        return "Expression of concern"
    if "correct" in k or "errat" in k or "corrig" in k:
        return "Correction"
    return None


def page_notices():
    a = need_assessment()
    if not a:
        return
    st.title("Notices & evidence")
    ev = a.evidence_table()
    sd = None

    notices = ev[ev["Check"].isin(["1.1", "1.2"])].copy()
    notices["Event"] = notices["Type"].map(_event)
    notices = notices.dropna(subset=["Event"])
    # Year-only dates (PubMed links) would be plotted on 1 January; keep exact dates on the timeline.
    dated = notices[notices["Date"].fillna("").str.len() >= 10]
    tl = dated.drop_duplicates(["ID", "Event", "Date"])[["ID", "Study", "Event", "Date", "Detail"]]
    pubs = [{"ID": s.uid, "Study": s.label, "Event": "Publication", "Date": s.pub_date or (f"{s.year}-07-01" if s.year else None),
             "Detail": s.title} for s in a.studies if s.uid in set(tl["ID"])]
    tl = pd.concat([pd.DataFrame(pubs), tl], ignore_index=True)
    tl["Study"] = tl["ID"] + " " + tl["Study"]

    st.markdown("#### Timeline of post-publication notices")
    st.caption("When each flagged study was published and when notices appeared. Notices can appear years after "
               "publication, so repeat the check shortly before finishing the review.")
    ch = charts.notice_timeline(tl, sd)
    if ch is not None:
        st.altair_chart(ch, width="stretch")
        lag = dated[dated["Event"] == "Retraction"].groupby("ID")["Date"].min()
        pubd = {s.uid: s.pub_date or (f"{s.year}-07-01" if s.year else None) for s in a.studies}
        yrs = [(pd.to_datetime(d) - pd.to_datetime(pubd[u])).days / 365.25 for u, d in lag.items() if pubd.get(u)]
        if yrs:
            st.caption(f"Median time from publication to retraction among flagged studies: "
                       f"**{pd.Series(yrs).median():.1f} years** (range {min(yrs):.1f}–{max(yrs):.1f}).")
    else:
        st.info("No retractions, expressions of concern or corrections found.")

    st.markdown("#### Which sources flagged each study?")
    st.caption("Number of notice records per source. Sources differ in coverage and timing, so a single "
               "database is not enough.")
    flags = notices.assign(Flag=notices["Event"].map(
        lambda e: "Retraction" if e == "Retraction" else "EoC / correction"))
    sa = charts.source_agreement(flags)
    if sa is not None:
        st.altair_chart(sa, width="content")
    else:
        st.info("Nothing flagged.")
    c2 = st.container()
    with c2:
        st.markdown("#### Reasons given in Retraction Watch")
        rwn = notices[notices["Source"] == "Retraction Watch"]
        rows = []
        for _, r in rwn[rwn["Event"] != "Correction"].iterrows():
            for reason in str(r["Detail"]).split(" | ")[0].split("; "):
                if reason and not reason.startswith(("Date of Article", "Upgrade/Update", "Updated to", "Notice -")):
                    rows.append({"ID": r["ID"], "Reason": reason})
        rdf = pd.DataFrame(rows, columns=["ID", "Reason"]).drop_duplicates()
        rdf = rdf.groupby("Reason")["ID"].nunique().reset_index(name="Studies").sort_values("Studies", ascending=False).head(12)
        rb = charts.reasons_bar(rdf)
        if rb is not None:
            st.altair_chart(rb, width="stretch")
        else:
            st.info("No Retraction Watch notices among included studies.")

    st.markdown("#### Trial registration timing (check 2.2)")
    lag_rows = [{"Study": f"{s.uid} {s.label}", "Trial": t, "Lag": lag}
                for s in a.studies for t, lag in a.results[s.uid]["2.2"].data.get("lag_days", {}).items() if lag is not None]
    rl = charts.registration_lag(pd.DataFrame(lag_rows, columns=["Study", "Trial", "Lag"]))
    if rl is not None:
        st.caption("Days between the study start and first registration on ClinicalTrials.gov (negative = registered "
                   "before start). Guidance: registration soon after recruitment starts may not warrant concerns.")
        st.altair_chart(rl, width="stretch")
    no_id = [f"{s.uid} {s.label}" for s in a.studies if not s.trial_ids and s.design in ("RCT", "Other trial", "Unknown")]
    if no_id:
        st.caption(f"No registration number found for {len(no_id)} study(ies): {', '.join(no_id)}. Check the full text.")

    st.markdown("#### Research team track record (check 1.3)")
    team = ev[ev["Check"] == "1.3"]
    if team.empty:
        st.info("No retractions or expressions of concern found for first or last authors.")
    else:
        st.caption("Other retractions/EoCs by the first or last author. Name matches can be namesakes: confirm identity.")
        st.dataframe(team.drop(columns=["Check"]), hide_index=True, width="stretch",
                     column_config={"Link": st.column_config.LinkColumn(display_text="open ↗")})

    st.markdown("#### All evidence")
    st.dataframe(ev, hide_index=True, width="stretch",
                 column_config={"Link": st.column_config.LinkColumn(display_text="open ↗")})


# =====================================================================================
# 5. EXPORT
# =====================================================================================

def page_export():
    a = need_assessment()
    if not a:
        return
    st.title("Export & methods")
    st.markdown("Everything needed to document the assessment: judgements, every check with its proposed and final "
                "answer, the evidence behind it, and which database versions were searched.")
    c1, c2 = st.columns(2)
    stamp = date.today().isoformat()
    c1.download_button("⬇ Excel workbook (all tables)", a.to_excel(), f"retraction-check_{stamp}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary",
                       width="stretch")
    c2.download_button("⬇ Summary (CSV)", a.summary_table().drop(columns=["Overall (raw)"]).to_csv(index=False),
                       f"retraction-check_summary_{stamp}.csv", mime="text/csv", width="stretch")

    st.markdown("#### Draft methods text")
    st.code(a.methods_text(), language=None, wrap_lines=True)

    st.markdown("#### Provenance")
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
    PAGES[2]: page_studies,
    PAGES[3]: page_notices,
    PAGES[4]: page_export,
    PAGES[5]: page_explore,
}[page]()
