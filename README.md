# INSPECT-SR assistant

Automated support for the **[INSPECT-SR](https://inspect-sr.com/)** tool, which assesses the trustworthiness of
randomised controlled trials included in systematic reviews.

You upload the studies included in your review in almost any format. The app identifies each study, searches
several sources and proposes answers to the INSPECT-SR checks that can be automated. Each proposal links to its
evidence, and the reviewer can override any of them. The full INSPECT-SR assessment is recorded, including the 17
checks that need the full text, and exported with provenance for the review's methods section.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the Retraction Watch database (~70 MB, Crossref open data) to `data/`.
Click **Load demo studies** to try it.

## What it does

| INSPECT-SR check | Automated? | Sources |
|---|---|---|
| 1.1 Retraction | Yes | Retraction Watch (DOI, PMID, exact and fuzzy title), Crossref update notices, PubMed publication types and linked notices, OpenAlex |
| 1.2 Expression of concern / other notices | Yes | Same, plus comments linked in PubMed |
| 1.3 Other studies by the research team | Search automated, reviewer judges | Retraction Watch author names (also with names swapped), OpenAlex disambiguated author IDs |
| 2.2 Timing or absence of registration | Search automated, reviewer judges | Trial IDs from the record, abstract and PubMed; ClinicalTrials.gov dates |
| All other checks (2.1, 2.3–2.5, 3.x, 4.x) | Reviewer | Recorded in the app with links to the guidance |

The suggested domain and overall judgements follow the guidance:
- a retraction typically warrants *serious concerns*;
- any other "Yes" contributes to the domain judgement;
- the overall judgement is at least as severe as the worst domain.

They are always shown as suggestions.

**Inputs:** RIS, PubMed/MEDLINE (`.nbib`), BibTeX, EndNote XML, CSV/TSV/Excel (e.g. Covidence or Rayyan exports),
Word (`.docx`), PDF, or pasted text: numbered reference lists, or lists of DOIs/PMIDs. References without a DOI or
PMID are identified with Crossref citation matching. Duplicates across files are merged.

**Outputs:**
- Overview: a robvis-style traffic-light plot, summary bars and an identification funnel.
- Per-study assessment pages.
- A timeline of notices against your search date.
- A chart showing which sources flagged each study.
- Retraction reasons and registration-timing charts.
- An Excel workbook with sheets Summary, Judgements, Checks, Evidence, Provenance and Methods text.
- A JSON file to save and resume an assessment.

## Structure

```
app.py                      Streamlit UI
inspectsr/
  framework.py              INSPECT-SR domains and 21 checks
  models.py                 Study, CheckResult, Evidence, Answer, Judgement
  inputs.py                 input adapters (one per format) → Study
  normalize.py              DOI/PMID/title/author normalisation, trial-ID and design detection
  pipeline.py               identification + evidence gathering → Dossier per study
  checks.py                 automated checks (add a new check here and register it in AUTOMATED)
  judgement.py              suggested domain / overall judgements
  assessment.py             reviewer decisions, tables, Excel/JSON export, methods text
  charts.py                 Altair charts
  sources/                  retraction_watch, crossref, pubmed, openalex, registries, http (cached)
tests/                      offline tests (pytest)
examples/                   demo inputs
```

## Privacy and reproducibility

Only identifiers and titles of published articles are sent to public APIs. To work offline, turn off *online
sources*; only the downloaded Retraction Watch file is then used. API responses are cached for 24 hours in
`data/http_cache.sqlite`. Optional environment variables: `CROSSREF_MAILTO`, `NCBI_API_KEY`, `OPENALEX_API_KEY`.

## Limits

- No source is complete, so a clean result does not prove the absence of notices. Also check the journal website, as the guidance advises.
- Author matching by name finds namesakes. Treat check 1.3 hits as leads to verify.
- ClinicalTrials.gov start dates can change. Compare with the first version in the record history.

## Citation

Wilkinson J, et al. INSPECT-SR tool for assessing trustworthiness of randomised controlled trials.
*BMJ* 2026;394:e100611. doi:10.1136/bmj-2026-100611
