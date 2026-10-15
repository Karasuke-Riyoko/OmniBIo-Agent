"""
scripts/run_benchmarks.py
Measures the runtime numbers the README quotes and stores them in data/benchmarks.json
(next to the affinity metrics written by scripts/train_affinity_model.py).

- Fusion latency: Perceiver + tanh-gated cross-attention on the real H&E patch tokens, median of 50 runs.
- End-to-end pipeline latency with network tools forced offline (so the number does not depend on PubChem/PubMed).

Run:  python scripts/run_benchmarks.py
"""

import json
import os
import platform
import sys
import time
from unittest import mock

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from agents.orchestrator import DEFAULT_CANDIDATE_COMPOUNDS, MultiAgentOrchestrator  # noqa: E402
from services import histology  # noqa: E402
from services.featurizers import smiles_features, text_tokens  # noqa: E402

BENCH = os.path.join(ROOT, "data", "benchmarks.json")


def main():
    orch = MultiAgentOrchestrator()
    histo = histology.extract(histology.SAMPLE_IMAGE)
    chem = np.stack([smiles_features(c["smiles"], 1024) for c in DEFAULT_CANDIDATE_COMPOUNDS])
    text = text_tokens(["EGFR exon 19 deletion with T790M and cis C797S", "Stage IV lung adenocarcinoma"])

    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        orch.fusion_pipeline.execute_fusion(histo["tokens"], chem, text)
        times.append((time.perf_counter() - t0) * 1000)

    with open(os.path.join(ROOT, "data", "sample_case.json"), encoding="utf-8") as f:
        case = json.load(f)
    e2e = []
    with mock.patch("urllib.request.urlopen", side_effect=OSError("offline")), mock.patch("time.sleep"):
        for _ in range(5):
            t0 = time.perf_counter()
            orch.run_full_pipeline(case)
            e2e.append((time.perf_counter() - t0) * 1000)

    bench = {}
    if os.path.exists(BENCH):
        with open(BENCH, encoding="utf-8") as f:
            bench = json.load(f)
    bench["runtime"] = {
        "fusion_latency_ms_median": round(float(np.median(times)), 1),
        "fusion_latency_ms_p90": round(float(np.percentile(times, 90)), 1),
        "fusion_inputs": f"{histo['n_patches']} H&E patch tokens, {len(chem)} SMILES token sets, {len(text)} text tokens",
        "pipeline_latency_ms_median_offline": round(float(np.median(e2e)), 1),
        "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}, CPU only",
        "generated": time.strftime("%Y-%m-%d"),
    }
    with open(BENCH, "w", encoding="utf-8") as f:
        json.dump(bench, f, indent=2)
    print(json.dumps(bench["runtime"], indent=2))


if __name__ == "__main__":
    main()
