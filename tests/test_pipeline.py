"""
tests/test_pipeline.py
Unit & Integration Test Suite for OmniBio-Agent.
Validates:
1. Two-Stage Spatial Perceiver Resampler (M=64 latent compression)
2. Decoupled tanh-Gated Cross-Attention and Gibbs Free Energy
3. Chou-Talalay Combination Index (CI) synergy engine (SAVANNAH, MARIPOSA)
4. Spatial Pharmacokinetic stromal penetration penalty
5. Clonal cis vs trans C797S resistance branching
6. Orchestrator synergy evaluation and Zhou et al. (PMID 33958875) longitudinal timeline
"""

import sys
import os
import json
import unittest
from unittest import mock
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASE_FILE = os.path.join(ROOT_DIR, "data", "sample_case.json")
sys.path.insert(0, ROOT_DIR)

from models.fusion import SpatialPerceiverResampler, DecoupledTanhGatedCrossAttention, OmniBioFusionPipeline
from services.combination_synergy import CombinationSynergyEngine
from agents.orchestrator import MultiAgentOrchestrator


class TestOmniBioAgent(unittest.TestCase):
    def setUp(self):
        self.pipeline = OmniBioFusionPipeline(d_vis=512, d_mol=1024, d_text=768, d_shared=256)
        self.orchestrator = MultiAgentOrchestrator()

    def test_perceiver_resampler_compression(self):
        """Verifies that variable-length patch tokens N_vis compress to fixed M=64 latents."""
        resampler = SpatialPerceiverResampler(d_in=512, d_model=256, n_latents=64)
        x_vis = np.random.randn(250, 512)
        z_vis, attn = resampler.resample(x_vis)
        self.assertEqual(z_vis.shape, (64, 256))

    def test_decoupled_tanh_gated_mha(self):
        """Verifies tanh-gated Cross-Attention and Gibbs Free Energy calculation."""
        gated = DecoupledTanhGatedCrossAttention(d_model=256)
        h_clin = np.random.randn(4, 256)
        z_vis = np.random.randn(64, 256)
        x_chem = np.random.randn(10, 256)

        out = gated.fuse(h_clin, z_vis, x_chem)
        self.assertEqual(len(out["fused_vector"]), 256)
        self.assertLess(out["predicted_delta_g_kcal_mol"], 0.0)

    def test_chou_talalay_combination_synergy(self):
        """Verifies mathematical Chou-Talalay CI calculation and clinical rules."""
        # Test Strong Synergism (SAVANNAH CI = 0.38)
        ci_res = CombinationSynergyEngine.calculate_ci(d1=15.0, dx1=80.0, d2=20.0, dx2=100.0, alpha=1.0)
        self.assertLess(ci_res["combination_index"], 0.7)
        self.assertTrue(ci_res["is_synergistic"])

        # Test SAVANNAH evaluation
        savannah = CombinationSynergyEngine.evaluate_regimen("SAVANNAH", is_met_amplified=True)
        self.assertEqual(savannah["ci"], 0.38)
        self.assertIn("RECOMMENDED", savannah["verdict"])

        # Test MARIPOSA evaluation
        mariposa = CombinationSynergyEngine.evaluate_regimen("MARIPOSA")
        self.assertEqual(mariposa["ci"], 0.44)
        self.assertEqual(mariposa["mPFS_months"], 23.7)

        # Test cis-C797S antagonism for Gefitinib + Osimertinib
        cis_test = CombinationSynergyEngine.evaluate_regimen("Gefitinib_Osimertinib", is_cis_c797s=True)
        self.assertGreater(cis_test["ci"], 1.1)
        self.assertIn("CONTRAINDICATED", cis_test["verdict"])

    def test_spatial_pk_penetration(self):
        """Verifies high MW penalty in desmoplastic stroma."""
        pk_small = MultiAgentOrchestrator.calculate_spatial_pk_penetration(mw=400.0, stroma_pct=65.0)
        pk_protac = MultiAgentOrchestrator.calculate_spatial_pk_penetration(mw=900.0, stroma_pct=65.0)
        self.assertGreater(pk_small["penetration_efficiency_pct"], pk_protac["penetration_efficiency_pct"])

    def _run_sample_case(self):
        with open(CASE_FILE, "r", encoding="utf-8") as f:
            case = json.load(f)
        # Force the offline fallbacks so the suite is deterministic and network-free.
        with mock.patch("urllib.request.urlopen", side_effect=OSError("offline")):
            return self.orchestrator.run_full_pipeline(case)

    def test_orchestrator_synergy_savannah_mariposa(self):
        """Verifies the orchestrator scores SAVANNAH and MARIPOSA via CombinationSynergyEngine."""
        result = self._run_sample_case()
        synergy = {r["regimen"]: r for r in result["combination_synergy"]}

        savannah = synergy["Osimertinib + Savolitinib"]
        self.assertEqual(savannah["combination_index"], 0.38)
        self.assertTrue(savannah["is_synergistic"])
        self.assertIn("SAVANNAH", savannah["trial"])

        mariposa = synergy["Amivantamab + Lazertinib"]
        self.assertEqual(mariposa["combination_index"], 0.44)
        self.assertEqual(mariposa["median_pfs_months"], 23.7)
        self.assertTrue(any("anticoagulation" in g for g in mariposa["guardrails"]))

        # Zhou et al. latest sequencing is cis-C797S, so Gefitinib + Osimertinib must be antagonistic.
        gef_osi = synergy["Osimertinib + Gefitinib"]
        self.assertGreater(gef_osi["combination_index"], 1.1)
        self.assertIn("CONTRAINDICATED", gef_osi["verdict"])

        self.assertTrue(any(t["agent"] == "CombinationSynergyAgent" for t in result["agent_trace"]))

    def test_orchestrator_zhou_longitudinal_timeline(self):
        """Verifies the Zhou et al. (PMID 33958875) treatment timeline is reflected in the output."""
        result = self._run_sample_case()
        timeline = result["longitudinal_timeline"]

        self.assertEqual(timeline["pmid"], "33958875")
        self.assertEqual(result["patient_summary"]["id"], "PT-ZHOU-2021-OTT")
        self.assertEqual([l["line"] for l in timeline["lines"]], [1, 2, 3, 4])
        self.assertEqual([l["pfs_months"] for l in timeline["lines"]], [7.0, 10.0, 8.0, 9.0])
        self.assertEqual(timeline["lines"][2]["start_month"], 17.0)
        self.assertEqual(timeline["total_tki_months"], 34.0)
        self.assertEqual(timeline["lines"][1]["resistance_snapshot"], "post_osimertinib")
        self.assertTrue(any("C797S (in trans)" in m["variant"] for m in timeline["lines"][1]["mutations_at_progression"]))
        self.assertTrue(any("C797S (in cis)" in m["variant"] for m in timeline["lines"][2]["mutations_at_progression"]))

    def test_visium_spatial_endpoint(self):
        """Verifies /api/visium serves the LDV Labs 10x Visium overlay the Modality 1 toggle renders."""
        from app import app
        response = app.test_client().get("/api/visium")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertIn("escape_threshold", data)
        self.assertTrue(data["spots"])
        for spot in data["spots"]:
            for key in ("barcode", "pxl_x", "pxl_y", "escape_score", "expression_log2_umi"):
                self.assertIn(key, spot)
            self.assertTrue(0.0 <= spot["escape_score"] <= 1.0)
        self.assertTrue(any(s["escape_score"] >= data["escape_threshold"] for s in data["spots"]))


if __name__ == "__main__":
    unittest.main()
