"""Offline tests: parsing, normalisation, Retraction Watch matching, checks and judgements."""
import pandas as pd
import pytest

from inspectsr import inputs
from inspectsr.assessment import Assessment
from inspectsr.checks import check_1_1, check_1_2, check_1_3, check_2_2, run_checks
from inspectsr.status import notices, study_status
from inspectsr.models import Answer, Study
from inspectsr.normalize import (author_keys, detect_design, find_trial_ids, is_group_author, is_notice_title,
                                 normalize_doi, normalize_pmid, normalize_title)
from inspectsr.pipeline import Dossier, RunInfo, match_rw
from inspectsr.sources.retraction_watch import build_index


# ---- normalisation ---------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("https://doi.org/10.1056/NEJMoa2021436", "10.1056/nejmoa2021436"),
    ("https://dx.doi.org/10.1056/NEJMoa2021436.", "10.1056/nejmoa2021436"),
    ("doi: 10.1016/S0140-6736(20)31180-6", "10.1016/s0140-6736(20)31180-6"),
    ("(see 10.1000/xyz123)", "10.1000/xyz123"),
    ("unavailable", None), ("", None), (None, None),
])
def test_normalize_doi(raw, expected):
    assert normalize_doi(raw) == expected


def test_normalize_pmid_ignores_placeholders():
    assert normalize_pmid("0") is None
    assert normalize_pmid("12345678.0") == "12345678"


def test_title_strips_retracted_prefix_and_html():
    assert normalize_title("RETRACTED ARTICLE: Effect of <i>X</i> on Y") == "effect of x on y"


def test_notice_vs_retracted_article_titles():
    assert is_notice_title("Retraction: Effect of X on Y")
    assert is_notice_title("Expression of concern: Effect of X")
    assert not is_notice_title("RETRACTED: Effect of X on Y")  # the article itself, labelled


def test_trial_ids():
    ids = find_trial_ids("Registered at ClinicalTrials.gov (NCT01690806) and ISRCTN35739639; ACTRN12611000123456")
    assert "ClinicalTrials.gov: NCT01690806" in ids and "ISRCTN: ISRCTN35739639" in ids
    assert any(i.startswith("ANZCTR") for i in ids)


def test_author_keys_and_groups():
    assert author_keys("Schietroma, Mario")["full"] == "schietroma|mario"
    assert author_keys("Mario Schietroma")["full"] == "schietroma|mario"
    assert "full" not in author_keys("Schietroma M") and author_keys("Schietroma M")["initial"] == "schietroma|m"
    assert is_group_author("The RECOVERY Collaborative Group")
    assert not is_group_author("Horby, Peter")


def test_design_detection():
    assert detect_design("A randomized controlled trial of X", None, []) == "RCT"
    assert detect_design("Retraction: A trial", None, []) == "Notice (not a study)"
    assert detect_design("X", None, ["Systematic Review"]) == "Systematic review"


# ---- parsers ---------------------------------------------------------------------
RIS = """﻿TY  - JOUR
AU  - Smith, John
AU  - Doe, Jane
TI  - Effect of drug X on outcome Y: a randomised trial
JO  - Some Journal
PY  - 2019
DO  - https://doi.org/10.1234/ABC.2019.1
N1  - Trial registration NCT01234567
ER  -

TY  - JOUR
T1  - A study without a DOI about something
AU  - Roe, Richard
PY  - 2018
AN  - 29876543
ER  -
"""


def test_parse_ris_with_bom_and_fields():
    studies, rep = inputs.parse_upload("x.ris", RIS.encode("utf-8"))
    assert rep.format == "ris" and len(studies) == 2
    a, b = studies
    assert a.doi == "10.1234/abc.2019.1" and a.year == 2019 and a.authors[0] == "Smith, John"
    assert "ClinicalTrials.gov: NCT01234567" in a.trial_ids and a.design == "RCT"
    assert b.title.startswith("A study") and b.pmid == "29876543"


def test_parse_medline():
    txt = """PMID- 30879366
TI  - Continuous versus interrupted use of vaginal probiotics: a randomized
      controlled study.
FAU - Elsharkawy, Ibrahim
DP  - 2021 Feb
LID - 10.1080/14767058.2019.1588246 [doi]
PT  - Journal Article
PT  - Retracted Publication
SI  - ClinicalTrials.gov/NCT03017625
"""
    studies, rep = inputs.parse_upload("x.nbib", txt.encode())
    s = studies[0]
    assert rep.format == "medline" and s.pmid == "30879366" and s.doi == "10.1080/14767058.2019.1588246"
    assert "controlled study" in s.title and "Retracted Publication" in s.pub_types
    assert "ClinicalTrials.gov: NCT03017625" in s.trial_ids


def test_parse_bibtex():
    bib = "@article{k1, title={{Effect} of {X} on {Y}}, author={Smith, John and Doe, Jane}, year={2020}, doi={10.1000/abc}}"
    s = inputs.parse_upload("x.bib", bib.encode())[0][0]
    assert s.title == "Effect of X on Y" and s.authors == ["Smith, John", "Doe, Jane"] and s.doi == "10.1000/abc"


def test_parse_csv_covidence_style():
    csv = "Title,Authors,Published Year,Journal,DOI\nEffect of X,Smith J; Doe J,2020,J Med,10.1000/abc\n"
    s = inputs.parse_upload("x.csv", csv.encode())[0][0]
    assert s.title == "Effect of X" and s.year == 2020 and s.authors == ["Smith J", "Doe J"]


def test_free_text_numbered_list_and_ids():
    text = """References
1. Horby P, Lim WS. Dexamethasone in Hospitalized Patients with Covid-19. N Engl J Med. 2021;384:693-704. doi:10.1056/NEJMoa2021436
2. Monticone M, Ambrosini E. 'Regent Suit' training improves recovery of motor activities: a randomized controlled trial. Clin Rehabil. 2013;27:792-802.
3. PMID: 30879366
"""
    studies = inputs.parse_free_text(text, "t")
    assert len(studies) == 3
    assert studies[0].doi == "10.1056/nejmoa2021436" and studies[0].year == 2021
    assert studies[1].title.startswith("'Regent Suit' training") and studies[1].authors[0] == "Monticone M"
    assert studies[2].pmid == "30879366"


def test_list_of_identifiers():
    studies = inputs.parse_free_text("10.1056/NEJMoa1602489\nhttps://doi.org/10.1000/abc\n30879366", "t")
    assert [s.doi or s.pmid for s in studies] == ["10.1056/nejmoa1602489", "10.1000/abc", "30879366"]


def test_deduplicate():
    a = Study("", title="Same title here please", doi="10.1000/a")
    b = Study("", title="Same title here please!")
    c = Study("", title="Different", doi="10.1000/c")
    out, dropped = inputs.deduplicate([a, b, c])
    assert dropped == 1 and len(out) == 2


# ---- Retraction Watch matching and checks -----------------------------------------------
def _row(rid, title, author, nature, rdate, pdate, doi, pmid, rdoi, reason):
    return {"Record ID": rid, "Title": title, "Author": author, "RetractionNature": nature,
            "RetractionDate": rdate, "OriginalPaperDate": pdate, "OriginalPaperDOI": doi,
            "OriginalPaperPubMedID": pmid, "RetractionDOI": rdoi, "Reason": reason}


def _rw():
    rows = [
        _row("1", "Effect of drug X on outcome Y: a randomised trial", "John Smith;Jane Doe", "Retraction",
             "5/1/2021 0:00", "1/1/2019 0:00", "10.1234/abc.2019.1", "0", "10.1234/ret.1",
             "+Falsification/Fabrication of Data;+Investigation by Journal/Publisher;"),
        _row("2", "Another paper by the same team about Z", "John Smith;Jane Doe;Al Other", "Retraction",
             "5/1/2022 0:00", "1/1/2020 0:00", "10.1234/other", "0", "10.1234/ret.2", "+Paper Mill;"),
        _row("3", "Pain relief after knee surgery with drug Q", "Ann Lee", "Expression of concern",
             "3/3/2023 0:00", "1/1/2017 0:00", "unavailable", "0", "10.9/eoc", "+Concerns/Issues about Data;"),
        _row("4", "Thymectomy trial", "Gil Wolfe", "Correction", "5/25/2017 0:00", "8/11/2016 0:00",
             "10.1056/nejmoa1602489", "27509100", "10.1056/x", "+Error in Methods;"),
    ]
    return build_index(pd.DataFrame(rows), {"source": "test"})


def _run(studies):
    rw = _rw()
    for i, s in enumerate(studies, 1):
        s.uid = f"S{i:03d}"
    dossiers = {s.uid: Dossier() for s in studies}
    match_rw(studies, dossiers, rw)
    return rw, dossiers


def _smith():
    return Study("", title="Effect of drug X on outcome Y: a randomised trial", authors=["Smith, John", "Doe, Jane"],
                 year=2019, doi="10.1234/abc.2019.1")


def test_retraction_by_doi_is_serious():
    s = _smith()
    rw, d = _run([s])
    r = check_1_1(s, d[s.uid], rw)
    assert r.proposed == Answer.YES and r.suggests_serious
    status, fl = study_status(s, run_checks(s, d[s.uid], rw))
    assert status == "Retracted" and "Falsification" in fl[0].why


def test_team_check_finds_other_retraction_but_not_own():
    s = _smith()
    rw, d = _run([s])
    r = check_1_3(s, d[s.uid], rw)
    assert r.proposed == Answer.YES and r.needs_review
    assert r.evidence and all("Another paper" in e.detail for e in r.evidence)  # own retraction excluded


def test_team_check_name_only_match_is_not_flagged():
    s = Study("", title="Unrelated trial of something else entirely", authors=["Smith, John", "Brown, Bob"],
              year=2019, doi="10.1234/unrelated")
    rw, d = _run([s])
    r = check_1_3(s, d[s.uid], rw)
    assert r.proposed == Answer.UNCLEAR and not r.needs_review  # possible namesake: shown, not flagged


def test_eoc_by_exact_title_with_corroboration():
    s = Study("", title="Pain relief after knee surgery with drug Q", authors=["Lee, Ann"], year=2017)
    rw, d = _run([s])
    assert check_1_2(s, d[s.uid], rw).proposed == Answer.YES


def test_fuzzy_title_without_corroboration_is_unclear():
    s = Study("", title="Pain relief after knee surgery using drug Q", authors=["Brown, Bob"], year=2005)
    rw, d = _run([s])
    assert d[s.uid].rw and d[s.uid].rw[0].method == "title_fuzzy"
    assert check_1_2(s, d[s.uid], rw).proposed == Answer.UNCLEAR


def test_correction_is_unclear_not_yes():
    s = Study("", title="Thymectomy trial", authors=["Wolfe, Gil"], year=2016, doi="10.1056/nejmoa1602489")
    rw, d = _run([s])
    assert check_1_2(s, d[s.uid], rw).proposed == Answer.UNCLEAR
    assert check_1_1(s, d[s.uid], rw).proposed == Answer.NO


def test_registration_rules():
    tid = "ClinicalTrials.gov: NCT00000001"
    s = Study("", title="A randomized trial", year=2015, trial_ids=[tid], design="RCT")
    d = Dossier(registrations={tid: {"first_submitted": "2012-09-19", "start_date": "2008-01", "start_type": "ACTUAL",
                                     "status": "COMPLETED", "enrollment": 100, "history_url": "u"}})
    r = check_2_2(s, d, None)
    assert r.proposed == Answer.YES and r.data["lag_days"][tid] > 1000
    d.registrations[tid]["first_submitted"] = "2007-12-01"
    assert check_2_2(s, d, None).proposed == Answer.NO
    assert check_2_2(Study("", design="RCT"), Dossier(), None).proposed == Answer.UNCLEAR


def test_false_match_decision_clears_status_and_excel_exports():
    s = _smith()
    rw, d = _run([s])
    a = Assessment([s], {s.uid: run_checks(s, d[s.uid], rw)}, RunInfo(started="t", finished="t"))
    assert a.status(s.uid)[0] == "Retracted"
    a.decisions[s.uid] = {"decision": "False match – ignore", "note": "different paper"}
    assert a.status(s.uid)[0] == "No notices found"
    assert a.summary_table().iloc[0]["Reviewer note"] == "different paper"
    # still counted as a flagged study that has been reviewed
    assert a.flagged_uids() == [s.uid] and a.review_progress() == (1, 1)
    assert len(a.to_excel()) > 1000


def test_notices_are_merged_across_sources():
    from inspectsr.models import CheckResult, Evidence
    ev = [Evidence("Retraction Watch", "Retraction", "Paper Mill; Investigation by Journal/Publisher", "2021-05-01",
                   "https://doi.org/10.1/r", "doi"),
          Evidence("Crossref", "Retraction", "Asserted by: retraction-watch", "2021-05-01", "https://doi.org/10.1/r", "doi"),
          Evidence("Crossref", "Retraction", "Asserted by: publisher", "2021-04-20", "https://doi.org/10.1/r", "doi"),
          Evidence("PubMed", "Retraction in", "J X. 2021", "2021", None, "pmid"),
          Evidence("PubMed", "Retracted Publication", "PubMed publication type", None, None, "pmid")]
    res = {"1.1": CheckResult("1.1", Answer.YES, "", ev, data={"kind": "retraction"})}
    ns = notices(res)
    assert len(ns) == 1
    assert ns[0].sources == ["Crossref", "Retraction Watch", "Crossref (from RW)", "PubMed"]
    assert ns[0].reasons == ["Paper Mill", "Investigation by Journal/Publisher"]
    assert ns[0].date == "2021-05-01"  # Retraction Watch date wins over the publisher's 2021-04-20


def test_notice_title_prefers_rw_and_skips_article_titles():
    from inspectsr.models import CheckResult, Evidence
    ev = [Evidence("Crossref", "Retraction", "Asserted by: publisher", "2020-05-22", "u1", "doi",
                   text="RETRACTED: A different article (The Lancet 2020)"),
          Evidence("PubMed", "Retraction in", "Lancet 2020", "2020", "u2", "pmid", text="Retraction—Study title (Lancet)"),
          Evidence("Retraction Watch", "Retraction", "Unreliable Data", "2020-06-05", "u3", "doi")]
    ns = notices({"1.1": CheckResult("1.1", Answer.YES, "", ev, data={"kind": "retraction"})})
    assert len(ns) == 1 and ns[0].title == "Retraction—Study title (Lancet)" and ns[0].date == "2020-06-05"


def test_correction_and_possible_match_are_concerns_not_retractions():
    s = Study("", title="Thymectomy trial", authors=["Wolfe, Gil"], year=2016, doi="10.1056/nejmoa1602489")
    rw, d = _run([s])
    assert study_status(s, run_checks(s, d[s.uid], rw))[0] == "Correction"
    s2 = Study("", title="Pain relief after knee surgery using drug Q", authors=["Brown, Bob"], year=2005)
    rw, d = _run([s2])
    assert study_status(s2, run_checks(s2, d[s2.uid], rw))[0] == "Possible match"


# ---- identification helpers --------------------------------------------------------
def test_pubmed_title_term_drops_pubmed_stopwords():
    from inspectsr.sources.pubmed import _title_term
    t = _title_term("Latent Inhibition Reduces Nocebo Nausea, Even Without Deception")
    assert "without[ti]" not in t and "deception[ti]" in t


def test_accept_pubmed_requires_similar_title_and_year():
    from inspectsr.pipeline import _accept_pubmed
    s = Study("", title="Randomized Trial of Thymectomy in Myasthenia Gravis", year=2016)
    recs = [{"title": "Retraction: Randomized Trial of Thymectomy in Myasthenia Gravis", "year": 2016},
            {"title": "Randomized trial of thymectomy in myasthenia gravis.", "year": 2016, "pmid": "1"},
            {"title": "Something else entirely about gravis", "year": 2016, "pmid": "2"}]
    hit = _accept_pubmed(s, recs)
    assert hit and hit[0]["pmid"] == "1" and hit[1] == "high"
    assert _accept_pubmed(Study("", title=s.title, year=2005), recs[1:]) is None


def test_identify_from_retraction_watch_first_pass():
    from inspectsr.pipeline import identify_from_rw
    s = Study("", title="Effect of drug X on outcome Y: a randomised trial", authors=["Smith, John"], year=2019)
    identify_from_rw([s], _rw())
    assert s.doi == "10.1234/abc.2019.1" and s.id_method == "Retraction Watch title match"


def test_ris_doi_in_unusual_field():
    ris = "TY  - JOUR\nTI  - A trial of something important here\nM3  - doi: 10.5555/abc.123\nER  - \n"
    s = inputs.parse_upload("x.ris", ris.encode())[0][0]
    assert s.doi == "10.5555/abc.123"
