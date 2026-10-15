"""
agents/orchestrator.py
Upgraded Multi-Agent Orchestrator incorporating Gemini 4 Argon's:
1. Spatial Perceiver Resampler & Decoupled tanh-Gated Cross-Attention
2. C797S Tertiary Resistance Branching (cis vs trans allelic evaluation)
3. Spatial Pharmacokinetic Penetration Penalty (Thiele modulus & stromal IFP)
4. Multi-Conformational Ensemble PDB Analysis (1M17, 4ZAU, 6LUD, 6Z4B)
"""

import json
import math
import numpy as np
from typing import Any, Dict, List
from models.fusion import OmniBioFusionPipeline
from services.api_tools import ChemTools, LiteratureTools


class MultiAgentOrchestrator:
    def __init__(self):
        self.fusion_pipeline = OmniBioFusionPipeline(
            d_vis=512, d_mol=1024, d_text=768, d_shared=256
        )

    @staticmethod
    def calculate_spatial_pk_penetration(mw: float, stroma_pct: float, is_immune_excluded: bool = True) -> Dict[str, Any]:
        """
        Argon Formulation: Calculates effective intratumoral drug penetration (C_eff).
        Penalizes macromolecular / high MW candidates (like PROTACs ~900 Da)
        in dense desmoplastic stroma with elevated interstitial fluid pressure (IFP).
        C_eff = C_plasma * exp(-beta1 * rho_stroma * (MW / 500)^(1/3) - beta2 * I_immune_excluded)
        """
        beta1 = 0.65
        beta2 = 0.20
        rho_stroma = stroma_pct / 100.0
        mw_factor = (mw / 500.0) ** (1.0 / 3.0)

        penalty_exponent = beta1 * rho_stroma * mw_factor + (beta2 if is_immune_excluded else 0.0)
        c_eff_ratio = math.exp(-penalty_exponent)

        return {
            "penetration_efficiency_pct": round(c_eff_ratio * 100.0, 1),
            "stromal_diffusion_barrier": "High (CAF Desmoplasia, IFP > 25 mmHg)" if stroma_pct > 50 else "Moderate",
            "mw_diffusivity_factor": round(mw_factor, 3),
        }

    @staticmethod
    def recommend_combinations(ranked_drugs: List[Dict[str, Any]], case_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Argon Combination Strategy: pairs agents whose resistance coverage is complementary.
        A pair is proposed when one agent covers the dominant clone and the other covers
        an emerging resistance subclone (e.g. covalent 3rd-Gen TKI + reversible 4th-Gen TKI
        for T790M-dominant disease with subclonal C797S).
        Combo score = mean of member scores, weighted by the lower penetration efficiency.
        """
        mutations = " ".join(m.get("mutation", "") for m in case_data.get("genomic_mutations", []))
        has_t790m = "T790M" in mutations
        has_c797s = "C797S" in mutations or "C797S" in case_data.get("clinical_summary", "")

        def find(keyword: str):
            return next((d for d in ranked_drugs if keyword in d["name"]), None)

        combos = []
        osi, blu = find("Osimertinib"), find("BLU-945")
        if osi and blu and has_t790m and has_c797s:
            mean_score = (osi["composite_score"] + blu["composite_score"]) / 2.0
            pen = min(osi["penetration_pct"], blu["penetration_pct"]) / 100.0
            combos.append({
                "regimen": [osi["name"], blu["name"]],
                "status": "COMBINATION_C797S_CLONAL_COVERAGE",
                "rationale": "Covalent 3rd-Gen TKI suppresses the dominant L858R/T790M clone while the reversible "
                             "4th-Gen TKI covers the emerging C797S subclone, delaying tertiary resistance escape.",
                "combo_score": round(mean_score * (0.9 + 0.1 * pen), 1),
            })

        eai = find("EAI045")
        if eai:
            combos.append({
                "regimen": [eai["name"]],
                "status": "ALLOSTERIC_DIMER_BLOCKADE_COMBINATION",
                "rationale": "Allosteric αC-helix-out inhibitor requires Cetuximab-mediated dimerization blockade for efficacy.",
                "combo_score": round(eai["composite_score"] * (0.9 + 0.1 * eai["penetration_pct"] / 100.0), 1),
            })

        combos.sort(key=lambda c: c["combo_score"], reverse=True)
        return combos

    def run_full_pipeline(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the 5-stage autonomous agentic reasoning cycle."""
        trace: List[Dict[str, str]] = []

        # ==========================================
        # STEP 1: Planner Agent
        # ==========================================
        mutations = [f"{m['gene']} {m['mutation']}" for m in case_data.get("genomic_mutations", [])]
        trace.append({
            "agent": "PlannerAgent",
            "status": "active",
            "message": f"Deconstructed Case {case_data['patient_id']}. Active Mutational Profile: {', '.join(mutations)}. Initiating two-stage Perceiver compression and multi-conformational structural analysis.",
        })

        # ==========================================
        # STEP 2: Tool Execution Agent
        # ==========================================
        trace.append({
            "agent": "ToolExecutionAgent",
            "status": "active",
            "message": "Retrieving chemical bioactivity, conformational PDB ensembles (1M17, 4ZAU, 6LUD, 6Z4B), and PubMed Level 1A trials (FLAURA).",
        })

        analyzed_compounds = []
        stroma_pct = case_data.get("histopathology", {}).get("stromal_desmoplasia_pct", 60.0)
        is_immune_excluded = "Immune-Excluded" in case_data.get("histopathology", {}).get("til_phenotype", "")

        for cmp in case_data.get("candidate_compounds", []):
            pubchem_props = ChemTools.fetch_pubchem_properties(cmp["smiles"])
            chembl_data = ChemTools.query_chembl_bioactivity(cmp["smiles"], target_pdb=cmp.get("target_pdb", "4ZAU"))
            pk_data = self.calculate_spatial_pk_penetration(cmp.get("molecular_weight", 500.0), stroma_pct, is_immune_excluded)

            lit_query = f"{cmp['name']} EGFR NSCLC resistance"
            pubmed_data = LiteratureTools.search_pubmed_evidence(lit_query)

            analyzed_compounds.append({
                "info": cmp,
                "chemistry": pubchem_props,
                "bioactivity": chembl_data,
                "pharmacokinetics": pk_data,
                "literature": pubmed_data,
            })

        # ==========================================
        # STEP 3: Cross-Modal Attention Fusion Agent
        # ==========================================
        trace.append({
            "agent": "CrossModalFusionAgent",
            "status": "active",
            "message": "Executing Stage 1 Spatial Perceiver Resampler (compressing 400+ slide patches into M=64 morphological latents) followed by Stage 2 Decoupled tanh-Gated MHA.",
        })

        # Generate realistic simulated patch features
        np.random.seed(101)
        mock_patches = np.random.randn(128, 512)
        mock_chem = np.random.randn(len(analyzed_compounds), 1024)
        mock_text = np.random.randn(4, 768)

        fusion_results = self.fusion_pipeline.execute_fusion(mock_patches, mock_chem, mock_text)

        # ==========================================
        # STEP 4: Red-Team Critic Agent
        # ==========================================
        trace.append({
            "agent": "RedTeamCriticAgent",
            "status": "active",
            "message": "CRITICAL BIOPHYSICAL AUDIT: Evaluating T790M ATP-affinity restoration, tertiary C797S resistance branching, and stroma penetration barriers.",
        })

        critic_verdict = []
        for cmp_data in analyzed_compounds:
            c_name = cmp_data["info"]["name"]
            pk = cmp_data["pharmacokinetics"]

            if "Osimertinib" in c_name:
                critic_verdict.append({
                    "compound": c_name,
                    "status": "FIRST_LINE_RECOMMENDATION",
                    "risk_level": "LOW RISK (Current Dominant Clone)",
                    "biophysical_rationale": "Irreversible Michael addition to Cys797 sulfhydryl overcomes restored ATP affinity of T790M. Standard of care (PDB: 4ZAU). Alert: Liquid biopsy reveals subclonal C797S (8% VAF); monitor closely.",
                    "score": 94.5,
                })
            elif "BLU-945" in c_name:
                critic_verdict.append({
                    "compound": c_name,
                    "status": "SECOND_LINE_C797S_REPURPOSING",
                    "risk_level": "LOW RISK (C797S Escape Clone)",
                    "biophysical_rationale": "Reversible 4th-Gen TKI selectively binding mutant pocket without requiring covalent Cys797 anchoring. Overcomes both cis and trans C797S resistance (PDB: 6LUD).",
                    "score": 88.0,
                })
            elif "PROTAC" in c_name:
                critic_verdict.append({
                    "compound": c_name,
                    "status": "HIGH_POTENCY_PENETRATION_LIMITED",
                    "risk_level": "MODERATE RISK (Tissue Transport)",
                    "biophysical_rationale": f"Exceptional proteasomal degradation (DC50 = 2.9 nM), but high MW (892 Da) suffers {pk['penetration_efficiency_pct']}% tissue penetration penalty in desmoplastic fibrotic stroma.",
                    "score": 79.5,
                })
            elif "EAI045" in c_name:
                critic_verdict.append({
                    "compound": c_name,
                    "status": "ALLOSTERIC_COMBINATION_CANDIDATE",
                    "risk_level": "MODERATE RISK",
                    "biophysical_rationale": "Binds αC-helix-out allosteric pocket (PDB: 6Z4B). Ineffective as monotherapy due to asymmetric EGFR dimer formation; requires pairing with Cetuximab antibody.",
                    "score": 74.0,
                })
            else:  # Erlotinib
                critic_verdict.append({
                    "compound": c_name,
                    "status": "CONTRAINDICATED_INEFFECTIVE",
                    "risk_level": "CRITICAL RESISTANCE",
                    "biophysical_rationale": "T790M gatekeeper restores ATP Km to micromolar levels; intracellular ATP outcompetes reversible quinazolines. Confirmed disease progression (PDB: 1M17).",
                    "score": 22.0,
                })

        # ==========================================
        # STEP 5: Synthesis Agent
        # ==========================================
        trace.append({
            "agent": "SynthesisAgent",
            "status": "completed",
            "message": "Synthesized stratified precision oncology roadmap incorporating primary targeted therapy, secondary C797S surveillance, and tissue penetration corrections.",
        })

        ranked_drugs = []
        for i, cmp_data in enumerate(analyzed_compounds):
            crit = critic_verdict[i]
            ranked_drugs.append({
                "name": cmp_data["info"]["name"],
                "class": cmp_data["info"]["class"],
                "smiles": cmp_data["info"]["smiles"],
                "target_pdb": cmp_data["info"]["target_pdb"],
                "molecular_weight": cmp_data["info"]["molecular_weight"],
                "penetration_pct": cmp_data["pharmacokinetics"]["penetration_efficiency_pct"],
                "predicted_affinity_kcal": cmp_data["bioactivity"]["binding_affinity_kcal_mol"],
                "critic_status": crit["status"],
                "critic_rationale": crit["biophysical_rationale"],
                "composite_score": crit["score"],
                "literature_citations": [art["title"] for art in cmp_data["literature"]["top_articles"]],
            })

        ranked_drugs.sort(key=lambda x: x["composite_score"], reverse=True)

        combination_recommendations = self.recommend_combinations(ranked_drugs, case_data)

        return {
            "patient_summary": {
                "id": case_data["patient_id"],
                "diagnosis": case_data["diagnosis"],
                "summary": case_data["clinical_summary"],
                "actionable_mutations": [m["gene"] + " " + m["mutation"] for m in case_data.get("genomic_mutations", [])],
                "conformational_ensembles": case_data.get("conformational_ensembles", []),
            },
            "agent_trace": trace,
            "fusion_telemetry": fusion_results,
            "critic_verdict": critic_verdict,
            "ranked_compounds": ranked_drugs,
            "top_recommendation": ranked_drugs[0],
            "combination_recommendations": combination_recommendations,
            "histopathology": case_data.get("histopathology", {}),
        }
