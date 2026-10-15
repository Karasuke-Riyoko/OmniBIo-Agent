"""
tests/test_pipeline.py
Unit & Integration Test Suite for OmniBio-Agent (Argon Upgraded).
Validates:
1. Spatial Perceiver Resampler (M=64 latent compression)
2. Decoupled tanh-Gated Cross-Attention stability and Thermodynamic Free Energy
3. Spatial Pharmacokinetics stromal penetration penalty
4. C797S resistance branching and conformational PDB ensemble support
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.fusion import SpatialPerceiverResampler, DecoupledTanhGatedCrossAttention, OmniBioFusionPipeline
from agents.orchestrator import MultiAgentOrchestrator


class TestOmniBioAgentArgon(unittest.TestCase):
    def setUp(self):
        self.pipeline = OmniBioFusionPipeline(d_vis=512, d_mol=1024, d_text=768, d_shared=256)
        self.orchestrator = MultiAgentOrchestrator()

    def test_perceiver_resampler_compression(self):
        """Verifies that variable-length patch tokens N_vis compress to fixed M=64 latents."""
        resampler = SpatialPerceiverResampler(d_in=512, d_model=256, n_latents=64)
        x_vis = np.random.randn(250, 512)  # 250 slide patches
        z_vis, attn = resampler.resample(x_vis)

        self.assertEqual(z_vis.shape, (64, 256))
        self.assertEqual(attn.shape, (64, 250 + 64))

    def test_decoupled_tanh_gated_mha(self):
        """Verifies tanh-gated Cross-Attention and Gibbs Free Energy calculation."""
        gated = DecoupledTanhGatedCrossAttention(d_model=256)
        h_clin = np.random.randn(4, 256)
        z_vis = np.random.randn(64, 256)
        x_chem = np.random.randn(10, 256)

        out = gated.fuse(h_clin, z_vis, x_chem)

        self.assertEqual(len(out["fused_vector"]), 256)
        self.assertIn("predicted_pkd", out)
        self.assertIn("predicted_delta_g_kcal_mol", out)
        # Delta G must be negative for spontaneous thermodynamic binding
        self.assertLess(out["predicted_delta_g_kcal_mol"], 0.0)

        # Verify tanh gate values
        self.assertAlmostEqual(out["tanh_gates"]["tanh_alpha_vis"], np.tanh(0.05), places=3)
        self.assertAlmostEqual(out["tanh_gates"]["tanh_alpha_chem"], np.tanh(0.05), places=3)

    def test_spatial_pk_penetration(self):
        """Verifies high MW penalty (e.g. PROTACs) in desmoplastic stroma."""
        pk_small = MultiAgentOrchestrator.calculate_spatial_pk_penetration(mw=400.0, stroma_pct=65.0)
        pk_protac = MultiAgentOrchestrator.calculate_spatial_pk_penetration(mw=900.0, stroma_pct=65.0)

        # High MW PROTAC should have lower tissue penetration than small molecule TKI
        self.assertGreater(pk_small["penetration_efficiency_pct"], pk_protac["penetration_efficiency_pct"])

    def test_orchestrator_c797s_branching(self):
        """Verifies full multi-agent loop with C797S resistance surveillance."""
        case_data = {
            "patient_id": "TEST-PT",
            "diagnosis": "NSCLC Stage IV",
            "clinical_summary": "T790M relapse with ctDNA C797S trace",
            "genomic_mutations": [{"gene": "EGFR", "mutation": "L858R"}, {"gene": "EGFR", "mutation": "T790M"}],
            "candidate_compounds": [
                {
                    "name": "Osimertinib",
                    "smiles": "CC(=O)N1CCN(CC1)C2=CC(=C(C=C2NC3=NC=CC(=N3)C4=CN(C5=CC=CC=C54)C)NC(=O)C=C)OC",
                    "target_pdb": "4ZAU",
                    "class": "3rd-Gen TKI",
                    "molecular_weight": 499.6,
                },
                {
                    "name": "BLU-945",
                    "smiles": "CC(C)NC1=NC=C(C=N1)C2=CN(C3=CC=CC=C32)C4=CC(=C(C=C4)F)NC(=O)CC5=CC=NC=C5",
                    "target_pdb": "6LUD",
                    "class": "4th-Gen TKI",
                    "molecular_weight": 526.6,
                },
                {
                    "name": "Erlotinib",
                    "smiles": "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
                    "target_pdb": "1M17",
                    "class": "1st-Gen TKI",
                    "molecular_weight": 393.4,
                },
            ],
            "histopathology": {"stromal_desmoplasia_pct": 60.0, "til_phenotype": "Immune-Excluded"},
        }

        output = self.orchestrator.run_full_pipeline(case_data)
        ranked = output["ranked_compounds"]

        self.assertEqual(ranked[0]["name"], "Osimertinib")
        self.assertIn("FIRST_LINE", ranked[0]["critic_status"])

        # BLU-945 should be prioritized as second-line C797S hedge
        blu = next(c for c in ranked if "BLU-945" in c["name"])
        self.assertIn("C797S", blu["critic_status"])

        # Erlotinib must be flagged contraindicated
        erl = next(c for c in ranked if "Erlotinib" in c["name"])
        self.assertIn("CONTRAINDICATED", erl["critic_status"])

    def test_combination_recommendations(self):
        """Verifies Osimertinib + BLU-945 is proposed when C797S is subclonal alongside T790M."""
        import json
        case_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_case.json")
        with open(case_path, encoding="utf-8") as f:
            case_data = json.load(f)

        output = self.orchestrator.run_full_pipeline(case_data)
        combos = output["combination_recommendations"]

        self.assertTrue(combos)
        top = combos[0]
        self.assertTrue(any("Osimertinib" in n for n in top["regimen"]))
        self.assertTrue(any("BLU-945" in n for n in top["regimen"]))
        self.assertIn("C797S", top["status"])


if __name__ == "__main__":
    unittest.main()
