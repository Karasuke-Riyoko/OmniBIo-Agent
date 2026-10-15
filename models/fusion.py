"""
models/fusion.py
Two-Stage Spatial Perceiver Resampler + Decoupled tanh-Gated Cross-Attention Engine.
Upgraded with Gemini 4 Argon's mathematical formulations for the Multimodal AI Hackathon 2026.

Solves:
1. Softmax Dilution (Attention Sink): Compresses N_vis (1,000-15,000 patches) -> M=64 latent queries
2. Gradient Imbalance: Decoupled tanh-gated Cross-Attention between Morpological Latents & Chemical Graphs

Weights are seeded random initialisations: the architecture runs end to end on real inputs (H&E patch tokens,
SMILES features, clinical text), but it has not been trained, so its outputs are shapes, gates, attention
statistics and timing, not predictions. Potency predictions come from services/affinity_model.py.
"""

import time

import numpy as np
from typing import Dict, Any, Tuple


class SpatialPerceiverResampler:
    """
    Stage 1: Compresses N_vis variable-length spatial patches into M=64 fixed morphological latent tokens.
    Eliminates Softmax Dilution and reduces quadratic complexity O(N_vis^2) to O(M * N_vis).
    """

    def __init__(self, d_in: int = 512, d_model: int = 256, n_latents: int = 64, n_heads: int = 4, seed: int = 42):
        np.random.seed(seed)
        self.d_in = d_in
        self.d_model = d_model
        self.n_latents = n_latents
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Learnable latent queries Z_0 in R^(M x d_model)
        self.z_0 = np.random.randn(n_latents, d_model) * np.sqrt(2.0 / d_model)

        # Visual projection
        self.w_proj = np.random.randn(d_in, d_model) * np.sqrt(2.0 / d_in)

        # Cross-Attention Weights
        scale = np.sqrt(2.0 / d_model)
        self.w_q = np.random.randn(d_model, d_model) * scale
        self.w_k = np.random.randn(d_model, d_model) * scale
        self.w_v = np.random.randn(d_model, d_model) * scale
        self.w_out = np.random.randn(d_model, d_model) * scale

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    @staticmethod
    def _layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + eps)

    def resample(self, x_vis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compresses input visual patches x_vis (N_vis x d_in) into z_vis (M x d_model).
        Returns: (z_vis, attention_weights)
        """
        if x_vis.ndim == 1:
            x_vis = x_vis[np.newaxis, :]

        # Project visual patches
        proj_vis = self._layer_norm(np.dot(x_vis, self.w_proj))

        # Context KV is concatenation of visual patches and learnable latents [X_vis; Z_0]
        kv_context = np.concatenate([proj_vis, self.z_0], axis=0)  # (N_vis + M, d_model)
        q_context = self.z_0  # (M, d_model)

        m = self.n_latents
        n_kv = kv_context.shape[0]

        # Multi-Head Attention projections
        q = np.dot(q_context, self.w_q).reshape(m, self.n_heads, self.d_k).swapaxes(0, 1)
        k = np.dot(kv_context, self.w_k).reshape(n_kv, self.n_heads, self.d_k).swapaxes(0, 1)
        v = np.dot(kv_context, self.w_v).reshape(n_kv, self.n_heads, self.d_k).swapaxes(0, 1)

        scores = np.matmul(q, k.swapaxes(-1, -2)) / np.sqrt(self.d_k)
        attn_weights = self._softmax(scores, axis=-1)

        head_out = np.matmul(attn_weights, v).swapaxes(0, 1).reshape(m, self.d_model)
        attn_out = np.dot(head_out, self.w_out)

        z_vis = self._layer_norm(self.z_0 + attn_out)
        return z_vis, np.mean(attn_weights, axis=0)


class DecoupledTanhGatedCrossAttention:
    """
    Stage 2: Decoupled tanh-Gated Cross-Attention.
    Prevents Softmax dilution by having Clinical/Genomic tokens attend separately to:
    1. Morphological Resampled Latents Z_vis via tanh(alpha_vis)
    2. Molecular Graph Tokens X_chem via tanh(alpha_chem)
    """

    def __init__(self, d_model: int = 256, n_heads: int = 4, seed: int = 42):
        np.random.seed(seed)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Zero-initialized learnable scalar gates (guarantees stable warm-start)
        self.alpha_vis = 0.05  # Initialized near zero
        self.alpha_chem = 0.05

        scale = np.sqrt(2.0 / d_model)
        # Visual cross-attention
        self.w_qv = np.random.randn(d_model, d_model) * scale
        self.w_kv = np.random.randn(d_model, d_model) * scale
        self.w_vv = np.random.randn(d_model, d_model) * scale
        self.w_out_v = np.random.randn(d_model, d_model) * scale

        # Chemical cross-attention
        self.w_qc = np.random.randn(d_model, d_model) * scale
        self.w_kc = np.random.randn(d_model, d_model) * scale
        self.w_vc = np.random.randn(d_model, d_model) * scale
        self.w_out_c = np.random.randn(d_model, d_model) * scale

        # FFN
        self.w_ffn1 = np.random.randn(d_model, d_model * 2) * scale
        self.w_ffn2 = np.random.randn(d_model * 2, d_model) * np.sqrt(2.0 / (d_model * 2))

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    @staticmethod
    def _layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + eps)

    def _cross_attention_block(self, q_ctx: np.ndarray, kv_ctx: np.ndarray, w_q, w_k, w_v, w_out) -> np.ndarray:
        nq = q_ctx.shape[0]
        nkv = kv_ctx.shape[0]

        q = np.dot(q_ctx, w_q).reshape(nq, self.n_heads, self.d_k).swapaxes(0, 1)
        k = np.dot(kv_ctx, w_k).reshape(nkv, self.n_heads, self.d_k).swapaxes(0, 1)
        v = np.dot(kv_ctx, w_v).reshape(nkv, self.n_heads, self.d_k).swapaxes(0, 1)

        scores = np.matmul(q, k.swapaxes(-1, -2)) / np.sqrt(self.d_k)
        attn = self._softmax(scores, axis=-1)

        head_out = np.matmul(attn, v).swapaxes(0, 1).reshape(nq, self.d_model)
        return np.dot(head_out, w_out)

    def fuse(self, h_clin: np.ndarray, z_vis: np.ndarray, x_chem: np.ndarray) -> Dict[str, Any]:
        """
        Calculates:
        H_tilde = H_clin + tanh(alpha_vis) * MHA(Q=H_clin, KV=Z_vis)
                         + tanh(alpha_chem) * MHA(Q=H_clin, KV=X_chem)
        """
        if h_clin.ndim == 1:
            h_clin = h_clin[np.newaxis, :]
        if x_chem.ndim == 1:
            x_chem = x_chem[np.newaxis, :]

        # Visual attention update
        vis_attn_out = self._cross_attention_block(h_clin, z_vis, self.w_qv, self.w_kv, self.w_vv, self.w_out_v)
        # Chemical attention update
        chem_attn_out = self._cross_attention_block(h_clin, x_chem, self.w_qc, self.w_kc, self.w_vc, self.w_out_c)

        gate_vis = float(np.tanh(self.alpha_vis))
        gate_chem = float(np.tanh(self.alpha_chem))

        # Gated residual update
        h_fused = h_clin + gate_vis * vis_attn_out + gate_chem * chem_attn_out
        h_norm = self._layer_norm(h_fused)

        # FFN block
        ffn = np.maximum(0, np.dot(h_norm, self.w_ffn1))
        fused_tokens = self._layer_norm(h_norm + np.dot(ffn, self.w_ffn2))

        # Mean pooled unified embedding
        unified_vec = np.mean(fused_tokens, axis=0)

        norm_val = float(np.linalg.norm(unified_vec))

        return {
            "fused_vector": unified_vec.tolist(),
            "fused_vector_norm": round(norm_val, 4),
            "tanh_gates": {
                "alpha_vis_raw": round(self.alpha_vis, 4),
                "tanh_alpha_vis": round(gate_vis, 4),
                "alpha_chem_raw": round(self.alpha_chem, 4),
                "tanh_alpha_chem": round(gate_chem, 4),
            },
            "latent_shapes": {
                "clinical_tokens": list(h_clin.shape),
                "resampled_visual_latents": list(z_vis.shape),
                "chemical_graph_tokens": list(x_chem.shape),
                "fused_output_dim": self.d_model,
            },
        }


class OmniBioFusionPipeline:
    """End-to-End Orchestrator wrapping Stage 1 (Perceiver) and Stage 2 (tanh-gated MHA)."""

    def __init__(self, d_vis: int = 512, d_mol: int = 1024, d_text: int = 768, d_shared: int = 256):
        self.perceiver = SpatialPerceiverResampler(d_in=d_vis, d_model=d_shared, n_latents=64)
        self.gated_mha = DecoupledTanhGatedCrossAttention(d_model=d_shared)

        # Projectors for text & chemistry to d_shared
        self.w_text_proj = np.random.randn(d_text, d_shared) * np.sqrt(2.0 / d_text)
        self.w_chem_proj = np.random.randn(d_mol, d_shared) * np.sqrt(2.0 / d_mol)

    def execute_fusion(self, x_vis: np.ndarray, x_chem: np.ndarray, x_text: np.ndarray) -> Dict[str, Any]:
        t0 = time.perf_counter()
        # Stage 1: Spatial Perceiver Compression (N_vis -> 64 latents)
        z_vis, perceiver_attn = self.perceiver.resample(x_vis)

        # Project text & chem to shared 256
        if x_text.ndim == 1:
            x_text = x_text[np.newaxis, :]
        if x_chem.ndim == 1:
            x_chem = x_chem[np.newaxis, :]

        h_clin = np.dot(x_text, self.w_text_proj)
        h_chem = np.dot(x_chem, self.w_chem_proj)

        # Stage 2: Decoupled tanh-Gated Cross-Attention
        fusion_out = self.gated_mha.fuse(h_clin, z_vis, h_chem)
        fusion_out["perceiver_compression_ratio"] = f"{x_vis.shape[0]} → {z_vis.shape[0]}"
        fusion_out["n_patches"] = int(x_vis.shape[0])
        fusion_out["n_latents"] = int(z_vis.shape[0])
        # Mean entropy of each latent's attention over patches, relative to uniform (1.0 = uniform).
        p = np.clip(perceiver_attn, 1e-12, 1.0)
        fusion_out["perceiver_attention_entropy_rel"] = round(float((-(p * np.log(p)).sum(-1)).mean() / np.log(p.shape[-1])), 4)

        # Cosine alignments
        u_mean = np.mean(h_clin, axis=0)
        v_mean = np.mean(h_chem, axis=0)
        z_mean = np.mean(z_vis, axis=0)

        def cos_sim(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

        fusion_out["alignment_metrics"] = {
            "vision_text_alignment": round(cos_sim(z_mean, u_mean), 4),
            "chemistry_text_alignment": round(cos_sim(v_mean, u_mean), 4),
            "vision_chemistry_alignment": round(cos_sim(z_mean, v_mean), 4),
        }
        fusion_out["fusion_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        fusion_out["weights"] = "untrained (seeded initialisation)"
        return fusion_out
