"""Trial registries. ClinicalTrials.gov is queried automatically; other registries are linked."""
from __future__ import annotations

from typing import Optional

from .http import get

CTGOV_API = "https://clinicaltrials.gov/api/v2/studies/{}"
SOURCE = "ClinicalTrials.gov"

REGISTRY_URLS = {
    "ClinicalTrials.gov": "https://clinicaltrials.gov/study/{id}?tab=history",
    "ISRCTN": "https://www.isrctn.com/{id}",
    "ANZCTR": "https://www.anzctr.org.au/TrialSearch.aspx#&&searchTxt={id}",
    "EU CTR (EudraCT)": "https://www.clinicaltrialsregister.eu/ctr-search/search?query={id}",
    "EU CTIS": "https://euclinicaltrials.eu/search-for-clinical-trials/?lang=en&EUCT={id}",
    "ChiCTR": "https://www.chictr.org.cn/searchprojEN.html?regno={id}",
    "CTRI": "https://ctri.nic.in/",
    "IRCT": "https://irct.behdasht.gov.ir/search/result?query={id}",
    "DRKS": "https://drks.de/search/en/trial/{id}",
    "JPRN/UMIN": "https://rctportal.niph.go.jp/en/",
    "PACTR": "https://pactr.samrc.ac.za/",
    "NTR/NL": "https://onderzoekmetmensen.nl/en",
}
ICTRP_URL = "https://trialsearch.who.int/Trial2.aspx?TrialID={id}"


def split_trial_id(item: str) -> tuple[str, str]:
    reg, _, ident = item.partition(": ")
    return reg, ident


def registry_url(item: str) -> str:
    reg, ident = split_trial_id(item)
    return REGISTRY_URLS.get(reg, ICTRP_URL).format(id=ident)


FIELDS = ("protocolSection.statusModule,protocolSection.identificationModule,"
          "protocolSection.designModule.enrollmentInfo")


def ctgov_studies(ncts: list[str], chunk: int = 100) -> dict[str, dict]:
    """Batch look-up: up to 100 trials per request."""
    out: dict[str, dict] = {}
    ids = sorted(set(ncts))
    for i in range(0, len(ids), chunk):
        part = ids[i:i + chunk]
        data = get("https://clinicaltrials.gov/api/v2/studies",
                   {"filter.ids": ",".join(part), "fields": FIELDS, "pageSize": len(part)})
        for st in (data or {}).get("studies", []):
            nct = st.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
            if nct:
                out[nct] = _summ(nct, st)
    return out


def ctgov_study(nct: str) -> Optional[dict]:
    data = get(CTGOV_API.format(nct), {"fields": FIELDS})
    return _summ(nct, data) if data else None


def _summ(nct: str, data: dict) -> dict:
    ps = data.get("protocolSection", {})
    st = ps.get("statusModule", {})
    return {
        "id": nct,
        "title": ps.get("identificationModule", {}).get("briefTitle"),
        "first_submitted": st.get("studyFirstSubmitDate"),
        "first_posted": (st.get("studyFirstPostDateStruct") or {}).get("date"),
        "start_date": (st.get("startDateStruct") or {}).get("date"),
        "start_type": (st.get("startDateStruct") or {}).get("type"),
        "primary_completion": (st.get("primaryCompletionDateStruct") or {}).get("date"),
        "status": st.get("overallStatus"),
        "enrollment": (ps.get("designModule", {}).get("enrollmentInfo") or {}).get("count"),
        "url": f"https://clinicaltrials.gov/study/{nct}",
        "history_url": f"https://clinicaltrials.gov/study/{nct}?tab=history",
    }
