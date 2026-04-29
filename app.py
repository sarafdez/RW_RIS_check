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
    "For more information on retracted papers, visit the "
    "[Retraction Watch website](https://retractionwatch.com/)."
)

st.markdown(
    """
**How to use**
1. Upload a RIS file using the uploader below.
2. The app matches your references against the Retraction Watch database — by DOI, exact title, and fuzzy title.
3. Review matches in the **Results** tabs and manually verify flagged records.
4. Download the matched records as CSV.

**What it does**
- Normalizes DOIs (strips URL prefixes) and titles (lowercase, punctuation removed)
- Matches by exact DOI, exact title, and fuzzy title similarity
- Fuzzy matches are flagged as high confidence (≥ 95) or low confidence (88–94)

**Limitations**
1. The Retraction Watch database is comprehensive but not exhaustive — retractions missing from it will not be flagged regardless of match quality.
2. Match failures are possible even for papers that are in RW: DOIs are often absent in older references, and titles can differ enough between a citation and the RW record to fall below the matching threshold. A clean result does not guarantee the absence of retracted papers.
"""
)


# ---- Variables and functions ----
FUZZY_THRESHOLD = 88

def _read_ris(uploaded_file) -> pd.DataFrame:
    text = uploaded_file.getvalue().decode("utf-8", errors="replace")
    records = rispy.load(io.StringIO(text))
    df = pd.DataFrame(records)

    if "doi" not in df.columns:
        df["doi"] = None
    if "primary_title" not in df.columns:
        df["primary_title"] = None
    if "title" not in df.columns:
        df["title"] = None

    df = df.copy()
    df["doi_norm"]  = df["doi"].apply(normalize_doi)
    # T1 → primary_title; TI → title. Use whichever is present.
    df["primary_title"] = df["primary_title"].combine_first(df["title"])
    df["title_norm"] = df["primary_title"].apply(normalize_title)
    df["title_ok"]  = df["title_norm"].apply(filter_bad_titles)

    return df

# this function is cached to avoid re-downloading RW data too often
@st.cache_data(ttl=24 * 3600, show_spinner="Loading Retraction Watch database…")
def get_retraction_watch():
    rw_df, meta = load_retraction_watch()
    rw_df = rw_df.copy()
    rw_df["doi_norm"]  = rw_df["OriginalPaperDOI"].apply(normalize_doi)

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
    if "RetractionDOI" in out.columns:
        out["RetractionDOI"] = out["RetractionDOI"].apply(_doi_url)
    return out

doi_col_config = {
    "OriginalPaperDOI": st.column_config.LinkColumn(
        "RW DOI",
        display_text=r"https?://doi\.org/(.*)",
        help="DOI from Retraction Watch — opens on doi.org",
    ),
    "RetractionDOI": st.column_config.LinkColumn(
        "Retraction Notice DOI",
        display_text=r"https?://doi\.org/(.*)",
        help="DOI of the retraction notice — opens on doi.org",
    ),
    "Title": st.column_config.TextColumn("RW Title"),
    "primary_title": st.column_config.TextColumn("Your Title (RIS)"),
    "Author": st.column_config.TextColumn("Author (RW)"),
    "Journal": st.column_config.TextColumn("Journal"),
    "RetractionDate": st.column_config.TextColumn("Retraction Date"),
    "RetractionNature": st.column_config.TextColumn("Retraction Nature"),
    "Reason": st.column_config.TextColumn("Reason"),
    "doi": st.column_config.TextColumn("Your DOI (RIS)"),
}

fuzzy_col_config = {
    **doi_col_config,
    "title_score": st.column_config.NumberColumn(
        "Match Score (%)",
        format="%.1f",
        help="Token-sort similarity score (0–100). ≥95 = high confidence, 88–94 = low confidence.",
    ),
    "fuzzy_confidence": st.column_config.TextColumn(
        "Confidence",
        help="High: score ≥ 95. Low: score 88–94 — review carefully.",
    ),
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
    
    doi_matches    = match_by_doi(review_df, rw_df)
    exact_matches  = match_by_title_exact(review_df[review_df["title_ok"]], rw_df[rw_df["title_ok"]])
    
    if run_fuzzy:
        fuzzy_matches = match_by_title_fuzzy(
            review_df[review_df["title_ok"]],
            rw_df[rw_df["title_ok"]],
            threshold=FUZZY_THRESHOLD,
        )
        # exclude titles already caught by exact matching
        if not fuzzy_matches.empty and not exact_matches.empty:
            exact_title_norms = set(exact_matches["title_norm"].dropna())
            fuzzy_matches = fuzzy_matches[
                ~fuzzy_matches["matched_title_norm"].isin(exact_title_norms)
            ]
    else:
        fuzzy_matches = pd.DataFrame()
    
    elapsed = time.perf_counter() - start
    
st.success(f"Matching completed in {elapsed:.2f} seconds")


# ---- Filtering ----

rw_cols = ["Title", "primary_title", "Author", "Journal", "RetractionDate", "RetractionNature", "Reason", "RetractionDOI", "OriginalPaperDOI", "doi"]
rw_doi   = doi_matches[rw_cols].copy()
rw_exact = exact_matches[rw_cols].copy()

rw_fuzzy_cols = rw_cols + ["title_score", "fuzzy_confidence"]

if run_fuzzy and not fuzzy_matches.empty:
    rw_fuzzy = fuzzy_matches[rw_fuzzy_cols].copy()
else:
    rw_fuzzy = pd.DataFrame(columns=rw_fuzzy_cols)
    

# ---- Summary ----
st.subheader("Results")
st.caption("Manually verify these results.")

res1, res2, res3 = st.columns(3)
res1.metric("DOI matches", int(len(rw_doi)))
res2.metric("Exact title matches", int(len(rw_exact)))
res3.metric("Fuzzy title matches", int(len(rw_fuzzy)))


# ---- Show results ----
st.caption(
    "Column sources — RW database: RW Title, Author, Journal, Retraction Date, Type, Reason, RW DOI | "
    "Your RIS file: Your Title (RIS), Your DOI (RIS) | "
    "Fuzzy tab only: Match Score (%), Confidence"
)
tabs = st.tabs(["DOI matches", "Exact title matches", "Fuzzy title matches", "All matches (unique)", "Raw RIS"])

combined = pd.concat(
    [
        rw_doi.assign(match_type="doi"),
        rw_exact.assign(match_type="title_exact"),
        rw_fuzzy.assign(match_type="title_fuzzy"),
    ],
    ignore_index=True,
)

combined_unique = combined.copy()
for col in combined_unique.columns:
    combined_unique[col] = combined_unique[col].map(
        lambda x: repr(x) if isinstance(x, (list, dict, set, tuple)) else x
    )
combined_unique = combined_unique.drop_duplicates(subset=["OriginalPaperDOI", "Title"], keep="first")

with tabs[0]:
    st.caption(f"{len(rw_doi)} row(s)")
    if len(rw_doi) == 0:
        st.info("No DOI matches found.")
    else:
        st.dataframe(_prep_for_display(rw_doi), use_container_width=True, column_config=doi_col_config)

with tabs[1]:
    st.caption(f"{len(rw_exact)} row(s)")
    if len(rw_exact) == 0:
        st.info("No exact title matches found.")
    else:
        st.dataframe(_prep_for_display(rw_exact), use_container_width=True, column_config=doi_col_config)

with tabs[2]:
    if not run_fuzzy:
        st.info("Fuzzy title matching is turned off. Enable it above to run.")
    else:
        st.caption(f"{len(rw_fuzzy)} row(s) — high confidence ≥ 95, low confidence 88–94")
        if len(rw_fuzzy) == 0:
            st.info("No fuzzy title matches found.")
        else:
            st.dataframe(_prep_for_display(rw_fuzzy), use_container_width=True, column_config=fuzzy_col_config)

with tabs[3]:
    st.caption(f"{len(combined_unique)} row(s) (deduplicated)")
    if len(combined_unique) == 0:
        st.info("No matches found.")
    else:
        st.dataframe(_prep_for_display(combined_unique), use_container_width=True, column_config=doi_col_config)

with tabs[4]:
    st.caption(f"{len(review_df)} row(s)")
    st.dataframe(review_df, use_container_width=True)


# ---- Download ----
st.subheader("Download")

csv_bytes = combined_unique.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download matched Retraction Watch rows (CSV)",
    data=csv_bytes,
    file_name="retraction_watch_matches.csv",
    mime="text/csv",
)
