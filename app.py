import io
import os
import json
import pandas as pd
import rispy
import streamlit as st
import time

from utils import *

# ---- Config ----
st.set_page_config(page_title="RIS ↔ Retraction Watch overlap", layout="wide")


st.title("RIS ↔ Retraction Watch overlap")

# ---- App info ----
st.markdown(
    """
This app matches references in a **RIS** file to the **Retraction Watch** database.

The app downloads the Retraction Watch database at most once every 24h.


How matching works:
1. **Normalize DOIs** and **normalize titles** (lowercasing + stripping punctuation).
2. **DOI exact match** (fast + most reliable).
3. **Exact title match** (on normalized titles) and filter out bad/short titles.
4. **Fuzzy title matching** using token-set similarity.



"""
)

# ---- Variables and functions ----
FUZZY_THRESHOLD = 95

def _read_ris(uploaded_file) -> pd.DataFrame:
    text = uploaded_file.getvalue().decode("utf-8", errors="replace")
    records = rispy.load(io.StringIO(text))
    df = pd.DataFrame(records)

    if "doi" not in df.columns:
        df["doi"] = None
    if "primary_title" not in df.columns:
        df["primary_title"] = None

    df = df.copy()
    df["doi_norm"] = df["doi"].apply(normalize_doi)
    df["title_norm"] = df["primary_title"].apply(normalize_title)
    df["title_ok"] = df["title_norm"].apply(filter_bad_titles)
        
    return df

# this function is cached to avoid re-downloading RW data too often
@st.cache_data(ttl=24 * 3600, show_spinner="Loading Retraction Watch database…")
def get_retraction_watch():
    rw_df, meta = load_retraction_watch()
    rw_df = rw_df.copy()
    rw_df["doi_norm"] = rw_df["OriginalPaperDOI"].apply(normalize_doi)
    rw_df["title_norm"] = rw_df["Title"].apply(normalize_title)
    return rw_df, meta

def _doi_url(doi: str) -> str:
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

doi_col_config = {
    "OriginalPaperDOI": st.column_config.LinkColumn(
        "OriginalPaperDOI",
        display_text=r"https?://doi\.org/(.*)",
        help="Opens the DOI on doi.org",
    )
}


# ---- Load RW ----
rw_df, rw_meta = get_retraction_watch()
rw_df["title_ok"] = rw_df["title_norm"].apply(filter_bad_titles)


# ---- Options ----
colA, colB, colC = st.columns([2, 1, 1])
with colA:
    uploaded = st.file_uploader("Upload a RIS file", type=["ris", "txt"])
with colB:
    st.metric("Retraction Watch records", f"{len(rw_df):,}")
    if rw_meta.get("downloaded_on"):
        st.caption(f"Downloaded on: {rw_meta['downloaded_on']}")
with colC:
    st.metric("Unique DOIs in RW", f"{rw_df['doi_norm'].dropna().nunique():,}")
    if rw_meta.get("url"):
        with st.expander("Retraction Watch metadata", expanded=False):
            st.write({
                "source": rw_meta.get("source"),
                "downloaded_on": rw_meta.get("downloaded_on"),
                "records": rw_meta.get("n_records"),
                "source_url": rw_meta.get("url"),
            })

if not uploaded:
    st.info("Upload a RIS file to begin.")
    st.stop()

st.info("In progress: Fuzzy title matching is optional and can be slow. Select the option below to enable it.")
run_fuzzy = st.checkbox("Run fuzzy title matching", value=False)

# ---- Load review data ----
review_df = _read_ris(uploaded)


# ---- Quality checks ----
qc1, qc2, qc3 = st.columns(3)
qc1.metric("RIS records", f"{len(review_df):,}")
qc2.metric("Missing DOI", int(review_df["doi"].isna().sum()))
qc3.metric("Missing title", int(review_df["title_norm"].isna().sum()))


# ---- Matching ----

with st.spinner("Running title matching…"):
    start = time.perf_counter()
    
    doi_matches = match_by_doi(review_df, rw_df)
    exact_matches = match_by_title_exact(review_df[review_df["title_ok"]], rw_df[rw_df["title_ok"]])
    
    if run_fuzzy:
        fuzzy_matches = match_by_title_fuzzy(
            review_df[review_df["title_ok"]],
            rw_df[rw_df["title_ok"]],
            threshold=FUZZY_THRESHOLD,
        )
    else:
        fuzzy_matches = pd.DataFrame() 
    
    elapsed = time.perf_counter() - start
    
st.success(f"Matching completed in {elapsed:.2f} seconds")


# ---- Filtering ----

rw_cols = ["Title", "primary_title", "Author", "RetractionNature", "Reason", "OriginalPaperDOI", "doi", "urls"]
rw_doi = doi_matches[rw_cols].copy()
rw_exact = exact_matches[rw_cols].copy()

if run_fuzzy and not fuzzy_matches.empty:
    rw_fuzzy = fuzzy_matches[rw_cols].copy()
else:
    rw_fuzzy = pd.DataFrame(columns=rw_cols)
    

# ---- Summary ----
st.subheader("Results")
st.caption("Manually verify these results.")

res1, res2, res3 = st.columns(3)
res1.metric("DOI matches", int(len(rw_doi)))
res2.metric("Exact title matches", int(len(rw_exact)))
res3.metric("Fuzzy title matches (excluding very short titles)", int(len(rw_fuzzy)))


# ---- Show results ----
tabs = st.tabs(["DOI matches", "Exact title matches", "Fuzzy title matches", "All matches - unique", "Raw RIS"])
st.caption("Title-> RW, primary_title-> RIS, OriginalPaperDOI-> RW, doi-> RIS")

combined = (
    pd.concat(
        [
            rw_doi.assign(match_type="doi"),
            rw_exact.assign(match_type="title_exact"),
            rw_fuzzy.assign(match_type="title_fuzzy"),
        ],
        ignore_index=True,
    )
    #.drop_duplicates(subset=["doi", "Title", "match_type"])
)

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
    if not run_fuzzy:
        st.info("Fuzzy title matching is turned off. Enable it above to run.")
    if len(rw_fuzzy) == 0:
        st.write("No fuzzy title matches found.")
    else:
        st.dataframe(_prep_for_display(rw_fuzzy), use_container_width=True, column_config=doi_col_config)
        
with tabs[3]:
    if len(combined) == 0:
        st.write("No matches found.")
    else:
        combined_unique = combined.drop_duplicates()
        st.dataframe(_prep_for_display(combined_unique), use_container_width=True, column_config=doi_col_config)

with tabs[4]:
    st.dataframe(review_df, use_container_width=True)


# ---- Download ----
st.subheader("Download")

csv_bytes = combined.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download matched Retraction Watch rows (CSV)",
    data=csv_bytes,
    file_name="retraction_watch_matches.csv",
    mime="text/csv",
)
