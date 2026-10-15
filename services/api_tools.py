"""
services/api_tools.py
External tools for the ToolExecutionAgent: PubChem PUG REST (properties), NCBI PubMed E-utilities
(search + real article titles) and the local EGFR potency model.

Every network call goes through `call_with_recovery`, which retries, then falls back to an offline answer and
records what happened. The orchestrator surfaces those records in the agent trace (the self-healing part of
the PurpleRain loop). Successful live answers are cached for the life of the process.
"""

import json
import time
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Tuple

from services.affinity_model import EGFRAffinityModel

USER_AGENT = {"User-Agent": "OmniBio-Agent/2.0"}
_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}


def _get_json(url: str, timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers=USER_AGENT)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def call_with_recovery(tool: str, key: str, live: Callable[[], Dict[str, Any]],
                       fallback: Callable[[], Dict[str, Any]], attempts: int = 2,
                       events: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Runs `live` up to `attempts` times; on repeated failure returns `fallback()`. Logs each outcome to `events`."""
    events = events if events is not None else []
    if (tool, key) in _CACHE:
        events.append({"tool": tool, "key": key, "outcome": "cache"})
        return _CACHE[(tool, key)]
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            result = live()
            _CACHE[(tool, key)] = result
            events.append({"tool": tool, "key": key, "outcome": "live", "attempt": attempt})
            return result
        except Exception as exc:  # network down, HTTP error, malformed payload
            last_error = f"{type(exc).__name__}: {exc}"[:120]
            if attempt < attempts:
                time.sleep(0.2 * attempt)
    events.append({"tool": tool, "key": key, "outcome": "recovered_with_fallback", "error": last_error})
    return fallback()


class ChemTools:
    """PubChem properties plus the local EGFR potency model."""

    @staticmethod
    def fetch_pubchem_properties(smiles: str, events: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        def live():
            url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
                   f"{urllib.parse.quote(smiles, safe='')}/property/MolecularWeight,MolecularFormula,XLogP,TPSA/JSON")
            props = _get_json(url, timeout=3)["PropertyTable"]["Properties"][0]
            return {
                "status": "success", "source": "PubChem PUG REST", "cid": props.get("CID"),
                "formula": props.get("MolecularFormula"),
                "molecular_weight": float(props.get("MolecularWeight", 0.0)),
                "xlogp": props.get("XLogP"), "tpsa": props.get("TPSA"),
            }

        def fallback():
            return {"status": "offline", "source": "Not retrieved (PubChem unreachable)", "molecular_weight": None}

        return call_with_recovery("PubChem", smiles, live, fallback, events=events)

    @staticmethod
    def predict_egfr_potency(smiles: str) -> Dict[str, Any]:
        return EGFRAffinityModel.load().predict(smiles)


# Real papers used when PubMed cannot be reached.
OFFLINE_EVIDENCE = [
    {"pmid": "29151359", "title": "Osimertinib in Untreated EGFR-Mutated Advanced Non-Small-Cell Lung Cancer (FLAURA)",
     "journal": "N Engl J Med 2018;378:113-125"},
    {"pmid": "33958875", "title": "Zhou et al. case report: sequential T790M and trans/cis C797S under EGFR TKIs",
     "journal": "OncoTargets and Therapy 2021;14:2847-2851"},
]


class LiteratureTools:
    """NCBI PubMed E-utilities: esearch for ids, esummary for the real titles."""

    @staticmethod
    def search_pubmed_evidence(query: str, max_results: int = 3, events: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        def live():
            base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
            ids = _get_json(f"{base}esearch.fcgi?db=pubmed&retmode=json&retmax={max_results}"
                            f"&term={urllib.parse.quote(query)}", timeout=3)["esearchresult"]["idlist"]
            if not ids:
                raise LookupError("no PubMed hits")
            summ = _get_json(f"{base}esummary.fcgi?db=pubmed&retmode=json&id={','.join(ids)}", timeout=3)["result"]
            return {
                "status": "success", "source": "PubMed E-utilities", "count": len(ids), "pmids": ids,
                "top_articles": [{"pmid": i, "title": summ[i].get("title", ""), "journal": summ[i].get("source", "")}
                                 for i in ids if i in summ],
            }

        def fallback():
            return {"status": "offline", "source": "Offline evidence (PubMed unreachable)", "count": len(OFFLINE_EVIDENCE),
                    "pmids": [a["pmid"] for a in OFFLINE_EVIDENCE], "top_articles": OFFLINE_EVIDENCE}

        return call_with_recovery("PubMed", query, live, fallback, events=events)
