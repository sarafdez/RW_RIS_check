"""'Explore Retraction Watch' page: filter and analyse the whole database."""
from __future__ import annotations

import re
from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

from .charts import CAT, INK_MUTED
from .sources.retraction_watch import RWIndex, integrity_related

FIELDS = {
    "HSC": "Health sciences", "BLS": "Biology & life sciences", "PHY": "Physical sciences",
    "B/T": "Business & technology", "SOC": "Social sciences", "ENV": "Environment", "HUM": "Humanities",
}
NATURES = ["Retraction", "Expression of concern", "Correction", "Reinstatement"]
NATURE_COLORS = dict(zip(NATURES, CAT[:4]))


@st.cache_data(show_spinner="Preparing the database for exploration…")
def prepare(_rw: RWIndex, version: str) -> pd.DataFrame:
    """Derived columns used by the explorer. `version` keys the cache to the downloaded file."""
    df = _rw.df
    out = pd.DataFrame({
        "Record ID": df["Record ID"],
        "Title": df["Title"],
        "Authors": df["Author"].str.replace(";", "; ", regex=False).str.strip("; "),
        "Journal": df["Journal"],
        "Publisher": df["Publisher"],
        "Nature": df["RetractionNature"],
        "Notice date": pd.to_datetime(df["_notice_date"], errors="coerce"),
        "Paper date": pd.to_datetime(df["_paper_date"], errors="coerce"),
        "Reasons": df["Reason"].map(lambda r: [x.strip().lstrip("+").strip() for x in str(r).split(";") if x.strip()]),
        "Countries": df["Country"].map(lambda c: [x.strip() for x in str(c).split(";") if x.strip()] or ["Unknown"]),
        "Fields": df["Subject"].map(lambda x: sorted({FIELDS.get(m, m) for m in re.findall(r"\(([A-Z/]+)\)", str(x))})),
        "Article types": df["ArticleType"].map(lambda x: [t.strip() for t in str(x).split(";") if t.strip()]),
        "DOI": df["_doi"],
        "Notice DOI": df["RetractionDOI"],
    })
    out["Notice year"] = out["Notice date"].dt.year
    out["Years to notice"] = (out["Notice date"] - out["Paper date"]).dt.days / 365.25
    out["Integrity-related"] = out["Reasons"].map(lambda rs: any(integrity_related(r) for r in rs))
    out["_search"] = (out["Title"].fillna("") + " | " + out["Authors"].fillna("") + " | " + out["Journal"].fillna("")
                      + " | " + out["DOI"].fillna("")).str.lower()
    return out


def _top(series_of_lists: pd.Series, n: int = 15, exclude: tuple = ()) -> pd.DataFrame:
    s = series_of_lists.explode().dropna()
    if exclude:
        s = s[~s.str.startswith(exclude)]
    vc = s.value_counts().head(n)
    return vc.rename_axis("Item").reset_index(name="Records")


def _hbar(df: pd.DataFrame, color: str = CAT[0]) -> alt.Chart:
    bars = alt.Chart(df).mark_bar(color=color, cornerRadiusEnd=4).encode(
        x=alt.X("Records:Q", title=None, axis=alt.Axis(grid=False, format="~s")),
        y=alt.Y("Item:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300)),
        tooltip=["Item", alt.Tooltip("Records:Q", format=",")],
    )
    labels = bars.mark_text(align="left", dx=4, color=INK_MUTED).encode(text=alt.Text("Records:Q", format=","))
    return (bars + labels).properties(height=alt.Step(22), width="container")


def render(rw: RWIndex):
    st.title("Explore Retraction Watch")
    st.caption(f"{rw.meta.get('source')} · {rw.meta['n_records']:,} records · downloaded {rw.meta['downloaded_on']}. "
               "Each record is one notice (retraction, expression of concern, correction or reinstatement) for one paper.")
    data = prepare(rw, str(rw.meta.get("downloaded_on")))

    # ---- filters
    years = data["Notice year"].dropna()
    y0, y1 = int(years.min()), int(years.max())
    with st.container(border=True):
        f1, f2, f3 = st.columns([2, 2, 3])
        natures = f1.multiselect("Notice type", NATURES, default=["Retraction"])
        yr = f2.slider("Notice year", y0, y1, (max(y0, 2000), y1))
        query = f3.text_input("Search title, author, journal or DOI", placeholder="e.g. Fujii, or 10.1056/")
        g1, g2, g3 = st.columns(3)
        fields = g1.multiselect("Field", list(FIELDS.values()))
        types = g2.multiselect("Article type", _top(data["Article types"], 15)["Item"].tolist(),
                               placeholder="All (e.g. choose Clinical Study)")
        countries = g3.multiselect("Country", _top(data["Countries"], 60)["Item"].tolist())
        integrity_only = st.checkbox("Only integrity-related reasons (misconduct, fabrication, paper mills, "
                                     "manipulation, plagiarism…)")

    m = data["Nature"].isin(natures or NATURES) & data["Notice year"].between(*yr)
    if fields:
        m &= data["Fields"].map(lambda fs: bool(set(fs) & set(fields)))
    if types:
        m &= data["Article types"].map(lambda ts: bool(set(ts) & set(types)))
    if countries:
        m &= data["Countries"].map(lambda cs: bool(set(cs) & set(countries)))
    if integrity_only:
        m &= data["Integrity-related"]
    if query.strip():
        m &= data["_search"].str.contains(query.strip().lower(), regex=False)
    df = data[m]

    if df.empty:
        st.info("No records match these filters.")
        return

    # ---- headline numbers
    yrs = df["Years to notice"].dropna()
    yrs = yrs[yrs >= 0]
    k = st.columns(5)
    k[0].metric("Records", f"{len(df):,}")
    k[1].metric("Journals", f"{df['Journal'].nunique():,}")
    k[2].metric("Integrity-related", f"{df['Integrity-related'].mean():.0%}")
    k[3].metric("Median years to notice", f"{yrs.median():.1f}" if len(yrs) else "–",
                help="Time from the paper's publication to the notice.")
    k[4].metric("Notice within 1 year", f"{(yrs <= 1).mean():.0%}" if len(yrs) else "–")

    # ---- over time
    st.markdown("#### Notices per year")
    per_year = df.groupby(["Notice year", "Nature"]).size().reset_index(name="Records")
    per_year["Notice year"] = per_year["Notice year"].astype(int)
    shown = [n for n in NATURES if n in set(per_year["Nature"])]
    chart = alt.Chart(per_year).mark_bar(stroke="white", strokeWidth=1).encode(
        x=alt.X("Notice year:O", title=None, axis=alt.Axis(labelAngle=0, labelOverlap=True)),
        y=alt.Y("Records:Q", title=None, axis=alt.Axis(format="~s")),
        color=alt.Color("Nature:N", scale=alt.Scale(domain=shown, range=[NATURE_COLORS[n] for n in shown]),
                        legend=alt.Legend(title=None, orient="bottom") if len(shown) > 1 else None),
        order=alt.Order("Nature:N"),
        tooltip=["Notice year", "Nature", alt.Tooltip("Records:Q", format=",")],
    ).properties(height=280, width="container")
    st.altair_chart(chart, width="stretch")
    if yr[1] == date.today().year:
        st.caption(f"{yr[1]} is a partial year. Large spikes are usually mass retractions by one publisher "
                   "(e.g. paper-mill special issues): filter by publisher or year to see them.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### Time from publication to notice")
        if len(yrs):
            h = pd.DataFrame({"Years": yrs.clip(upper=20)})
            hist = alt.Chart(h).mark_bar(color=CAT[0]).encode(
                x=alt.X("Years:Q", bin=alt.Bin(step=1), title="Years after publication (20 = 20 or more)"),
                y=alt.Y("count():Q", title=None, axis=alt.Axis(format="~s")),
                tooltip=[alt.Tooltip("count():Q", title="Records", format=",")],
            ).properties(height=260, width="container")
            st.altair_chart(hist, width="stretch")
    with c2:
        st.markdown("#### Fields")
        st.altair_chart(_hbar(_top(df["Fields"], 10)), width="stretch")

    st.markdown("#### Most common reasons")
    st.caption("A notice usually lists several reasons. Procedural tags (dates, notice updates) are left out.")
    st.altair_chart(_hbar(_top(df["Reasons"], 15, exclude=("Date of", "Upgrade/Update", "Updated to", "Notice -"))),
                    width="stretch")

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.markdown("#### Countries (author affiliations)")
        st.altair_chart(_hbar(_top(df["Countries"], 12)), width="stretch")
    with c4:
        st.markdown("#### Publishers")
        st.altair_chart(_hbar(_top(df["Publisher"].map(lambda x: [x] if x else ["Unknown"]), 12)), width="stretch")

    st.markdown("#### Journals")
    st.altair_chart(_hbar(_top(df["Journal"].map(lambda x: [x] if x else ["Unknown"]), 15)), width="stretch")

    # ---- records
    st.markdown("#### Records")
    table = df.sort_values("Notice date", ascending=False).assign(
        **{"Notice date": df["Notice date"].dt.date, "Paper date": df["Paper date"].dt.date,
           "Reasons": df["Reasons"].map("; ".join), "Countries": df["Countries"].map("; ".join),
           "DOI": df["DOI"].map(lambda d: f"https://doi.org/{d}" if d else None)}
    )[["Notice date", "Nature", "Title", "Authors", "Journal", "Publisher", "Countries", "Paper date", "Reasons", "DOI"]]
    st.caption(f"Showing the {min(len(table), 1000):,} most recent of {len(table):,} matching records.")
    st.dataframe(table.head(1000), hide_index=True, width="stretch", column_config={
        "DOI": st.column_config.LinkColumn(display_text=r"https://doi\.org/(.*)"),
        "Title": st.column_config.TextColumn(width="large"),
    })
    st.download_button("⬇ Download matching records (CSV)", table.to_csv(index=False).encode("utf-8"),
                       "retraction_watch_filtered.csv", mime="text/csv")
