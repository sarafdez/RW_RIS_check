# Retraction Watch check

Checks the studies included in a systematic review for retractions, expressions of concern, corrections and other
warning signs. It also lets you explore the whole Retraction Watch database.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the Retraction Watch database (~70 MB, Crossref open data) to `data/`.

## Pages

1. **Input**: upload RIS, PubMed/MEDLINE (`.nbib`), BibTeX, EndNote XML, CSV/TSV/Excel (e.g. Covidence or
   Rayyan exports), Word (`.docx`) or PDF reference lists, or paste references or a list of DOIs/PMIDs.
   - References without a DOI or PMID are identified via Crossref.
   - Duplicates across files are merged.
   - You can untick or edit records before running.
2. **Overview**: a traffic-light plot of judgements per study, summary bars, an identification funnel and the study table.
3. **Study assessments**: for each study, the proposed answers with their evidence. You can override any answer
   or judgement and record a reason.
4. **Notices & evidence**:
   - a timeline of notices;
   - which sources flagged each study;
   - retraction reasons;
   - registration timing;
   - the research team's other retractions;
   - all evidence.
5. **Export & methods**: an Excel workbook (Summary, Judgements, Checks, Evidence, Provenance, Methods text), a CSV
   summary and draft methods text.
6. **Explore Retraction Watch**: filter the database by notice type, year, field, article type, country,
   integrity-related reasons or free text. Shows:
   - notices per year;
   - time from publication to notice;
   - top reasons, countries, publishers and journals;
   - matching records, downloadable as CSV.

## What is checked

| Check | Sources |
|---|---|
| Retraction | Retraction Watch (DOI, PMID, exact and fuzzy title), Crossref update notices, PubMed publication types and linked notices, OpenAlex |
| Expression of concern / correction | Same sources, plus comments linked in PubMed |
| Authors' other retractions | Retraction Watch author names (also with names swapped), OpenAlex disambiguated author IDs |
| Trial registration timing | Trial IDs from the record, abstract and PubMed; ClinicalTrials.gov dates |

Flagged studies are listed with the reason for each flag, and you record your own decision for each one.

## Structure

```
app.py                      Streamlit UI
inspectsr/
  inputs.py                 input adapters (one per format) → Study
  normalize.py              DOI/PMID/title/author normalisation, trial-ID and design detection
  pipeline.py               identification + evidence gathering → Dossier per study
  checks.py                 automated checks (register new ones in AUTOMATED)
  judgement.py              suggested domain / overall judgements
  assessment.py             reviewer decisions, tables, Excel export, methods text
  charts.py, explore.py     charts and the Retraction Watch explorer
  sources/                  retraction_watch, crossref, pubmed, openalex, registries, http (cached)
tests/                      offline tests (pytest)
examples/                   demo inputs for testing
```

## Privacy and limits

- Only identifiers and titles of published articles are sent to public APIs (Crossref, NCBI, OpenAlex, ClinicalTrials.gov).
- No source is complete, so a clean result does not prove the absence of notices.
- Author matching by name can find namesakes; verify check 1.3 hits.
- ClinicalTrials.gov start dates can change; check the record history.
