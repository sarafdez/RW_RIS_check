import io
import os
import json
import pandas as pd
import rispy
import streamlit as st

from utils import (
    load_retraction_watch,
    normalize_doi,
    normalize_title,
    match_by_doi,
    match_by_title_exact,
    match_by_title_fuzzy,
)

# ---- Config ----
st.set_page_config(page_title="RIS ↔ Retraction Watch overlap", layout="wide")


st.title("RIS ↔ Retraction Watch overlap")

# ---- App info ----
st.markdown(
    """
This app matches references in a **RIS** file to the **Retraction Watch** database.

How matching works:
1. **Normalize DOIs** and **normalize titles** (lowercasing + stripping punctuation).
2. **DOI exact match** (fast + most reliable).
3. **Exact title match** (on normalized titles).
4. **Fuzzy title matching** using token-set similarity.
"""
)

# Fixed fuzzy threshold 
FUZZY_THRESHOLD = 90


def _read_ris(uploaded_file) -> pd.DataFrame:
    text = uploaded_file.getvalue().decode("utf-8", errors="replace")
    records = rispy.load(io.StringIO(text))
    df = pd.DataFrame(records)

    if "doi" not in df.columns:
        df["doi"] = None
    if "primary_title" not in df.columns:
        df["primary_title"] = None

    df = df.copy()
    df["doi"] = df["doi"].apply(normalize_doi)
    df["title_norm"] = df["primary_title"].apply(normalize_title)
    return df


# ---- Load RW ----
rw_df, rw_meta = load_retraction_watch()
rw_df["doi"] = rw_df["OriginalPaperDOI"].apply(normalize_doi)
rw_df["title_norm"] = rw_df["Title"].apply(normalize_title)

colA, colB, colC = st.columns([2, 1, 1])
with colA:
    uploaded = st.file_uploader("Upload a RIS file", type=["ris", "txt"])
with colB:
    st.metric("Retraction Watch records", f"{len(rw_df):,}")
    if rw_meta.get("downloaded_on"):
        st.caption(f"Downloaded on: {rw_meta['downloaded_on']}")
with colC:
    st.metric("Unique DOIs in RW", f"{rw_df['doi'].dropna().nunique():,}")
    if rw_meta.get("url"):
        with st.expander("Retraction Watch metadata", expanded=False):
            st.write({
                "source": rw_meta.get("source"),
                "downloaded_on": rw_meta.get("downloaded_on"),
                "records": rw_meta.get("n_records"),
                "source_url": rw_meta.get("source_url"),
            })

if not uploaded:
    st.info("Upload a RIS file to begin.")
    st.stop()

review_df = _read_ris(uploaded)

# ---- Quality checks ----
qc1, qc2, qc3 = st.columns(3)
qc1.metric("RIS records", f"{len(review_df):,}")
qc2.metric("Missing DOI", int(review_df["doi"].isna().sum()))
qc3.metric("Missing title", int(review_df["title_norm"].isna().sum()))

# ---- Matching ----
with st.spinner("Running title matching…"):
    doi_matches = match_by_doi(review_df, rw_df)
    exact_matches = match_by_title_exact(review_df, rw_df)
    fuzzy_matches = match_by_title_fuzzy(
        review_df,
        rw_df,
        threshold=FUZZY_THRESHOLD,
    )


# ---- Pull matched RW rows for display ----
rw_cols = ["Title", "RetractionNature", "Reason", "OriginalPaperDOI", "doi"]

rw_doi = rw_df[rw_df["doi"].isin(doi_matches["doi"].dropna())][rw_cols].drop_duplicates().copy()

rw_exact = rw_df[rw_df["title_norm"].isin(exact_matches["title_norm"].dropna())][rw_cols].drop_duplicates().copy()

# this removes very short titles from visualization (not from matching)
rw_fuzzy = rw_df[
    (rw_df["title_norm"].str.len() >= 5)
    & (rw_df["title_norm"].isin(fuzzy_matches["matched_title_norm"].dropna()))
][rw_cols].drop_duplicates().copy()


def _doi_url(doi: str) -> str:
    """Turn a DOI string into a doi.org URL (empty string if missing)."""
    if doi is None:
        return ""
    doi = str(doi).strip()
    if doi == "" or doi.lower() == "nan":
        return ""
    return f"https://doi.org/{doi}"

def _prep_for_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "OriginalPaperDOI" in out.columns:
        out["OriginalPaperDOI"] = out["OriginalPaperDOI"].apply(_doi_url)
    return out

# ---- Summary ----
st.subheader("Results")
res1, res2, res3 = st.columns(3)
res1.metric("DOI matches", int(len(rw_doi)))
res2.metric("Exact title matches", int(len(rw_exact)))
res3.metric("Fuzzy title matches (excluding very short titles)", int(len(rw_fuzzy)))

# Tabs
tabs = st.tabs(["DOI matches", "Exact title matches", "Fuzzy title matches", "Raw RIS"])

doi_col_config = {
    "OriginalPaperDOI": st.column_config.LinkColumn(
        "OriginalPaperDOI",
        display_text=r"https?://doi\.org/(.*)",
        help="Opens the DOI on doi.org",
    )
}

with tabs[0]:
    if len(rw_doi) == 0:
        st.write("No DOI matches found.")
    else:
        st.dataframe(_prep_for_display(rw_doi), use_container_width=True, column_config=doi_col_config)

with tabs[1]:
    if len(rw_exact) == 0:
        st.write("No exact title matches found.")
    else:
        st.dataframe(_prep_for_display(rw_exact), use_container_width=True, column_config=doi_col_config)

with tabs[2]:
    if len(rw_fuzzy) == 0:
        st.write("No fuzzy title matches found.")
    else:
        st.dataframe(_prep_for_display(rw_fuzzy), use_container_width=True, column_config=doi_col_config)

with tabs[3]:
    st.dataframe(review_df, use_container_width=True)

# ---- Download ----
st.subheader("Download")

combined = (
    pd.concat(
        [
            rw_doi.assign(match_type="doi"),
            rw_exact.assign(match_type="title_exact"),
            rw_fuzzy.assign(match_type="title_fuzzy"),
        ],
        ignore_index=True,
    )
    .drop_duplicates(subset=["doi", "Title", "match_type"])
)

csv_bytes = combined.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download matched Retraction Watch rows (CSV)",
    data=csv_bytes,
    file_name="retraction_watch_matches.csv",
    mime="text/csv",
)
