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
7. Spatial escape scoring, Gi* hotspots and the /api/visium endpoint
8. EGFR potency model, H&E patch extraction, tool self-healing and the self-correcting critic loop
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
        self.assertAlmostEqual(out["tanh_gates"]["tanh_alpha_vis"], float(np.tanh(0.05)), places=4)

    def test_chou_talalay_combination_synergy(self):
        """Verifies mathematical Chou-Talalay CI calculation and clinical rules."""
        # Test Strong Synergism (SAVANNAH CI = 0.38)
        ci_res = CombinationSynergyEngine.calculate_ci(d1=15.0, dx1=80.0, d2=20.0, dx2=100.0, alpha=1.0)
        self.assertLess(ci_res["combination_index"], 0.7)
        self.assertTrue(ci_res["is_synergistic"])

        # Median-effect fits recover the curve parameters and the CI is computed, not looked up.
        savannah = CombinationSynergyEngine.evaluate_regimen("SAVANNAH", is_met_amplified=True)
        self.assertLess(savannah["ci"], 0.7)
        self.assertIn("RECOMMENDED", savannah["verdict"])
        for fit in savannah["calculation"]["fits"].values():
            self.assertGreater(fit["r"], 0.95)
        self.assertEqual([r["fa"] for r in savannah["calculation"]["fa_ci"]], [0.5, 0.75, 0.9])

        mariposa = CombinationSynergyEngine.evaluate_regimen("MARIPOSA")
        self.assertLess(mariposa["ci"], 0.7)
        self.assertEqual(mariposa["mPFS_months"], 23.7)

        # Allele phasing flips Gefitinib + Osimertinib from synergy (trans) to antagonism (cis).
        cis_test = CombinationSynergyEngine.evaluate_regimen("Gefitinib_Osimertinib", is_cis_c797s=True)
        trans_test = CombinationSynergyEngine.evaluate_regimen("Gefitinib_Osimertinib", is_cis_c797s=False)
        self.assertGreater(cis_test["ci"], 1.1)
        self.assertLess(trans_test["ci"], 0.9)
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
        with mock.patch("urllib.request.urlopen", side_effect=OSError("offline")), mock.patch("time.sleep"):
            return self.orchestrator.run_full_pipeline(case)

    def test_orchestrator_synergy_savannah_mariposa(self):
        """Verifies the orchestrator scores SAVANNAH and MARIPOSA via CombinationSynergyEngine."""
        result = self._run_sample_case()
        synergy = {r["regimen"]: r for r in result["combination_synergy"]}

        savannah = synergy["Osimertinib + Savolitinib"]
        self.assertLess(savannah["combination_index"], 0.7)
        self.assertTrue(savannah["is_synergistic"])
        self.assertIn("SAVANNAH", savannah["trial"])

        mariposa = synergy["Amivantamab + Lazertinib"]
        self.assertLess(mariposa["combination_index"], 0.7)
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
            for key in ("barcode", "pxl_x", "pxl_y", "escape_score", "gi_star_z", "expression_log2_umi"):
                self.assertIn(key, spot)
            self.assertTrue(0.0 <= spot["escape_score"] <= 1.0)
        self.assertGreater(data["n_hotspots"], 0)
        self.assertTrue(data["recommendation"]["spatial_met_niche"])

    def test_spatial_escape_and_gi_star(self):
        """Escape rises with MET/AXL; Gi* flags a cluster of high values and not an isolated one."""
        from services import spatial
        low = spatial.escape_score({"EGFR": 9, "MET": 2, "AXL": 2, "CD274": 2})
        high = spatial.escape_score({"EGFR": 7, "MET": 9.5, "AXL": 7, "CD274": 5})
        self.assertLess(low, 0.5)
        self.assertGreater(high, spatial.ESCAPE_THRESHOLD)

        xy = np.array([[x, y] for y in range(6) for x in range(6)], dtype=float)
        values = np.zeros(36)
        values[[0, 1, 6, 7]] = 1.0          # 2x2 cluster in one corner
        values[35] = 1.0                    # isolated high value
        z = spatial.getis_ord_gi_star(xy, values, band=1.5)
        self.assertGreater(z[0], 1.96)
        self.assertLess(z[35], z[0])

    def test_affinity_model_and_histology_tokens(self):
        """The ChEMBL-trained model returns bounded pIC50s; the H&E tile yields one 16-dim token per patch."""
        from services.affinity_model import EGFRAffinityModel
        from services import histology
        pred = EGFRAffinityModel.load().predict("COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC")
        self.assertTrue(3.0 < pred["predicted_pic50"] < 12.0)
        self.assertLess(pred["estimated_delta_g_kcal_mol"], 0.0)

        histo = histology.extract(histology.SAMPLE_IMAGE)
        rows, cols = histo["grid"]
        self.assertEqual(histo["tokens"].shape, (rows * cols, len(histology.FEATURE_NAMES)))
        self.assertTrue(0 < histo["composition"]["stroma_pct_of_tissue"] < 100)

        bench_path = os.path.join(ROOT_DIR, "data", "benchmarks.json")
        with open(bench_path, encoding="utf-8") as f:
            bench = json.load(f)
        self.assertGreater(bench["egfr_affinity"]["pearson_r"], 0.5)

    def test_tool_self_healing_and_critic_loop(self):
        """Failed tool calls are retried then healed; the critic loop demotes Osimertinib for cis-C797S and converges."""
        result = self._run_sample_case()
        events = result["tool_events"]
        self.assertTrue(events)
        self.assertTrue(all(e["outcome"] in ("recovered_with_fallback", "cache", "live") for e in events))

        loop = result["critic_loop"]
        self.assertTrue(loop["converged"])
        self.assertGreaterEqual(len(loop["iterations"]), 2)
        ranked = result["ranked_compounds"]
        self.assertTrue(ranked[0]["name"].startswith("BLU-945"))
        osi = next(d for d in ranked if d["name"].startswith("Osimertinib"))
        self.assertIn("COVALENT_ANCHOR_LOST", osi["violations"])
        self.assertLess(osi["composite_score"], osi["planner_score"])
        erl = next(d for d in ranked if d["name"] == "Erlotinib")
        self.assertIn("CONTRAINDICATED", erl["critic_status"])


if __name__ == "__main__":
    unittest.main()
