"""
tests/test_pipeline.py
Unit & Integration Test Suite for OmniBio-Agent.
Validates:
1. Two-Stage Spatial Perceiver Resampler (M=64 latent compression)
2. Decoupled tanh-Gated Cross-Attention and Gibbs Free Energy
3. Chou-Talalay Combination Index (CI) synergy engine (SAVANNAH, MARIPOSA)
4. Spatial Pharmacokinetic stromal penetration penalty
5. Clonal cis vs trans C797S resistance branching
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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


if __name__ == "__main__":
    unittest.main()
