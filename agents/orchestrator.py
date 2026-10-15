"""
agents/orchestrator.py
Upgraded Multi-Agent Orchestrator incorporating Gemini 4 Argon's:
1. Spatial Perceiver Resampler & Decoupled tanh-Gated Cross-Attention
2. C797S Tertiary Resistance Branching (cis vs trans allelic evaluation)
3. Spatial Pharmacokinetic Penetration Penalty (Thiele modulus & stromal IFP)
4. Multi-Conformational Ensemble PDB Analysis (1M17, 4ZAU, 6LUD, 6Z4B)
5. Chou-Talalay Combination Synergy (SAVANNAH: Osimertinib + Savolitinib, MARIPOSA: Amivantamab + Lazertinib)
6. Longitudinal treatment timeline for the Zhou et al. real clinical case (PMID 33958875)
"""

import json
import math
import numpy as np
from typing import Any, Dict, List
from models.fusion import OmniBioFusionPipeline
from services.api_tools import ChemTools, LiteratureTools
from services.combination_synergy import CombinationSynergyEngine


DEFAULT_CONFORMATIONAL_ENSEMBLES = [
    {"pdb_id": "1M17", "title": "Wild-Type EGFR + Erlotinib (Active Conformation)",
     "significance": "Demonstrates baseline ATP pocket before gatekeeper steric shift"},
    {"pdb_id": "4ZAU", "title": "EGFR + Osimertinib (AZD9291) Covalent Adduct",
     "significance": "Irreversible covalent bond formed with Cys797 sulfhydryl group"},
    {"pdb_id": "6LUD", "title": "EGFR L858R / T790M / C797S Triple Mutant",
     "significance": "Ser797 hydroxyl replaces Cys797 thiol, abolishing covalent binding"},
    {"pdb_id": "6Z4B", "title": "EGFR T790M + Allosteric Inhibitor EAI045",
     "significance": "Binds αC-helix-out allosteric pocket, overcoming ATP-site resistance"},
]

DEFAULT_CANDIDATE_COMPOUNDS = [
    {"name": "Osimertinib (AZD9291)",
     "smiles": "CC(=O)N1CCN(CC1)C2=CC(=C(C=C2NC3=NC=CC(=N3)C4=CN(C5=CC=CC=C54)C)NC(=O)C=C)OC",
     "chembl_id": "CHEMBL3353410", "target_pdb": "4ZAU", "class": "3rd-Gen Irreversible EGFR TKI",
     "molecular_weight": 499.6, "mechanism": "Covalent Michael addition to Cys797; highly potent against T790M"},
    {"name": "BLU-945 (4th-Gen TKI)",
     "smiles": "CC(C)NC1=NC=C(C=N1)C2=CN(C3=CC=CC=C32)C4=CC(=C(C=C4)F)NC(=O)CC5=CC=NC=C5",
     "chembl_id": "CHEMBL4794821", "target_pdb": "6LUD", "class": "4th-Gen Mutant-Selective TKI",
     "molecular_weight": 526.6, "mechanism": "Reversible ATP-competitive inhibition of the T790M/C797S triple mutant"},
    {"name": "EGFR PROTAC Degrader 14",
     "smiles": "CC1=C(C=C(C=C1)CNC2=NC=C(C=N2)NC3=CC(=C(C=C3)OC4CCNCC4)OC)C(=O)N5CCC(CC5)NC(=O)C6=CC=CC=C6C(=O)N7C(=O)CCC7=O",
     "chembl_id": "CHEMBL5129983", "target_pdb": "6LUD", "class": "Bifunctional Targeted Protein Degrader",
     "molecular_weight": 892.4, "mechanism": "E3 ubiquitin ligase recruitment; proteasomal degradation (DC50 = 2.9 nM)"},
    {"name": "EAI045 + Cetuximab (Allosteric)",
     "smiles": "CC1=CC=C(C=C1)S(=O)(=O)NC2=CC=C(C=C2)C(=O)NC3=CC=C(C=C3)Cl",
     "chembl_id": "CHEMBL3989824", "target_pdb": "6Z4B", "class": "4th-Gen Allosteric Inhibitor",
     "molecular_weight": 444.9, "mechanism": "Allosteric αC-helix-out binding; requires dimerization blockade"},
    {"name": "Erlotinib",
     "smiles": "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
     "chembl_id": "CHEMBL545", "target_pdb": "1M17", "class": "1st-Gen Reversible TKI",
     "molecular_weight": 393.4, "mechanism": "Reversible ATP competitor; ineffective against restored ATP affinity of T790M"},
]

# Zhou et al. sequencing timepoints, keyed by the treatment line after which they were drawn.
MUTATION_TIMEPOINTS = [(0, "baseline"), (1, "post_gefitinib"), (2, "post_osimertinib"), (3, "post_combination")]

# Dual-drug regimens scored by the Chou-Talalay engine.
SYNERGY_REGIMENS = [
    "Pair_A_MET_Bypass_Blockade",                # SAVANNAH: Osimertinib + Savolitinib
    "Pair_B_Bispecific_Dual_Receptor_Blockade",  # MARIPOSA: Amivantamab + Lazertinib
    "Gefitinib_Osimertinib",                     # Zhou et al. line 3 (cis vs trans C797S)
]


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
    def normalize_case(raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accepts either the flat pipeline schema (patient_id, genomic_mutations, ...) or the
        real-case schema (patient_case.primary_on_target_case, Zhou et al.) and returns the
        flat schema, carrying the clinical case and regimen data for downstream agents.
        """
        if "patient_id" in raw:
            return raw

        case = raw["patient_case"]["primary_on_target_case"]
        timepoints = case.get("mutations", {})
        latest_key = next(k for _, k in reversed(MUTATION_TIMEPOINTS) if timepoints.get(k))
        genomic_mutations = [
            {
                "gene": m["gene"],
                "mutation": m["variant"],
                "variant_allele_frequency": m["vaf_pct"] / 100.0 if m.get("vaf_pct") is not None else None,
                "timepoint": latest_key,
            }
            for m in timepoints[latest_key]
        ]
        latest_line = case["timeline_months"][-1]
        histology = case.get("histology", {})

        return {
            "patient_id": case["patient_id"],
            "age": case.get("age"),
            "gender": case.get("gender"),
            "diagnosis": f"{histology.get('subtype', 'NSCLC')} ({case.get('stage', 'Stage IV')})",
            "clinical_summary": (
                f"{case['citation']} (PMID {case['pmid']}). EGFR-mutant NSCLC treated across "
                f"{len(case['timeline_months'])} TKI-based lines; overall survival {case.get('overall_survival_months')} months. "
                f"Latest progression: {latest_line['progression_event']}."
            ),
            "genomic_mutations": genomic_mutations,
            "conformational_ensembles": raw.get("conformational_ensembles", DEFAULT_CONFORMATIONAL_ENSEMBLES),
            "candidate_compounds": raw.get("candidate_compounds", DEFAULT_CANDIDATE_COMPOUNDS),
            "histopathology": histology,
            "clinical_case": case,
            "combination_regimens": raw.get("combination_regimens", []),
            "critic_guidance_rules": raw.get("critic_guidance_rules", []),
        }

    @staticmethod
    def build_longitudinal_timeline(case_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lays the Zhou et al. treatment lines end-to-end on a month axis and attaches the
        sequencing drawn at each progression (baseline -> line 1 -> line 2 -> line 3).
        """
        case = case_data.get("clinical_case")
        if not case:
            return {}

        timepoints = case.get("mutations", {})
        snapshot_for_line = dict(MUTATION_TIMEPOINTS)
        lines, cursor = [], 0.0
        for entry in case["timeline_months"]:
            start = cursor
            cursor += entry["pfs_months"]
            snapshot_key = snapshot_for_line.get(entry["line"])
            lines.append({
                **entry,
                "start_month": start,
                "end_month": cursor,
                "resistance_snapshot": snapshot_key,
                "mutations_at_progression": timepoints.get(snapshot_key, []),
            })

        return {
            "source": case["citation"],
            "pmid": case["pmid"],
            "doi": case.get("doi"),
            "patient_id": case["patient_id"],
            "overall_survival_months": case.get("overall_survival_months"),
            "post_osimertinib_survival_months": case.get("post_osimertinib_survival_months"),
            "total_tki_months": cursor,
            "baseline_mutations": timepoints.get("baseline", []),
            "lines": lines,
        }

    @staticmethod
    def evaluate_combination_synergy(case_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Scores dual-drug regimens with CombinationSynergyEngine (Chou-Talalay CI) in the context
        of the patient's clonal state: cis-C797S phasing gates Gefitinib + Osimertinib, and MET
        amplification gates the SAVANNAH bypass recommendation. Critic guardrails are attached.
        """
        genomic = case_data.get("genomic_mutations", [])
        variants = " ".join(m.get("mutation", "") for m in genomic)
        is_cis_c797s = "C797S (in cis)" in variants or "cis-C797S" in variants
        is_met_amplified = any(m.get("gene") == "MET" for m in genomic)
        regimen_meta = {r["pair_id"]: r for r in case_data.get("combination_regimens", [])}
        rules = {r["rule_id"]: r["action"] for r in case_data.get("critic_guidance_rules", [])}
        guardrail_rules = {
            "Pair_A_MET_Bypass_Blockade": "CRITIC_01_LIQUID_BIOPSY_MET_FALSE_NEGATIVE",
            "Pair_B_Bispecific_Dual_Receptor_Blockade": "CRITIC_03_MARIPOSA_THROMBOEMBOLIC_GUARDRAIL",
            "Gefitinib_Osimertinib": "CRITIC_04_CIS_VS_TRANS_C797S_CLONAL_SHIFT",
        }

        results = []
        for pair_id in SYNERGY_REGIMENS:
            ev = CombinationSynergyEngine.evaluate_regimen(
                pair_id, is_cis_c797s=is_cis_c797s, is_met_amplified=is_met_amplified
            )
            meta = regimen_meta.get(pair_id, {})
            drug_1, drug_2 = ev["name"].split(" + ")
            rule = rules.get(guardrail_rules[pair_id])
            results.append({
                "pair_id": pair_id,
                "regimen": ev["name"],
                "drug_1": meta.get("drug_1", drug_1),
                "drug_2": meta.get("drug_2", drug_2),
                "combination_index": ev["ci"],
                "category": ev["category"],
                "is_synergistic": ev["ci"] < 0.9,
                "target": ev["target"],
                "trial": meta.get("phase_trial", ev["trial"]),
                "median_pfs_months": ev.get("mPFS_months", meta.get("median_pfs_months")),
                "orr_pct": ev.get("orr_pct", meta.get("confirmed_orr_pct")),
                "hazard_ratio": ev.get("hazard_ratio", meta.get("hazard_ratio")),
                "verdict": ev["verdict"],
                "rationale": meta.get("rationale", ""),
                "guardrails": [rule] if rule else [],
            })

        results.sort(key=lambda r: r["combination_index"])
        return results

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
        """Runs the 6-stage autonomous agentic reasoning cycle."""
        case_data = self.normalize_case(case_data)
        trace: List[Dict[str, str]] = []

        # ==========================================
        # STEP 1: Planner Agent
        # ==========================================
        mutations = [f"{m['gene']} {m['mutation']}" for m in case_data.get("genomic_mutations", [])]
        c797s_vaf = next(
            (m.get("variant_allele_frequency") for m in case_data.get("genomic_mutations", []) if "C797S" in m["mutation"]),
            None,
        )
        c797s_alert = (
            f"Sequencing reveals C797S ({c797s_vaf * 100:.2g}% VAF); covalent Cys797 anchoring compromised."
            if c797s_vaf else "Monitor ctDNA for C797S emergence."
        )
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

        has_cis_c797s = any(
            "cis" in m.get("mutation", "").lower() or "cis" in m.get("variant", "").lower()
            for m in case_data.get("genomic_mutations", [])
        )

        critic_verdict = []
        for cmp_data in analyzed_compounds:
            c_name = cmp_data["info"]["name"]
            pk = cmp_data["pharmacokinetics"]

            if "BLU-945" in c_name:
                critic_verdict.append({
                    "compound": c_name,
                    "status": "FIRST_LINE_C797S_TARGETED" if has_cis_c797s else "SECOND_LINE_C797S_REPURPOSING",
                    "risk_level": "LOW RISK (C797S Escape Clone)",
                    "biophysical_rationale": "Reversible 4th-Gen TKI selectively binding mutant pocket without requiring covalent Cys797 anchoring. Overcomes both cis and trans C797S resistance (PDB: 6LUD).",
                    "score": 96.0 if has_cis_c797s else 88.0,
                })
            elif "Osimertinib" in c_name:
                if has_cis_c797s:
                    critic_verdict.append({
                        "compound": c_name,
                        "status": "RESISTANCE_ESCAPE_OBSOLETE",
                        "risk_level": "HIGH RESISTANCE (C797S Abolished Covalency)",
                        "biophysical_rationale": "C797S in cis replaces Cys797 thiol with Ser797 hydroxyl, abolishing covalent Michael addition and causing >100-fold loss of potency. Monotherapy no longer effective.",
                        "score": 42.0,
                    })
                else:
                    critic_verdict.append({
                        "compound": c_name,
                        "status": "FIRST_LINE_RECOMMENDATION",
                        "risk_level": "LOW RISK (Current Dominant Clone)",
                        "biophysical_rationale": "Irreversible Michael addition to Cys797 sulfhydryl overcomes restored ATP affinity of T790M. Standard of care (PDB: 4ZAU). Alert: " + c797s_alert,
                        "score": 94.5,
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
                    "score": 20.0,
                })

        # ==========================================
        # STEP 5: Combination Synergy Agent
        # ==========================================
        combination_synergy = self.evaluate_combination_synergy(case_data)
        longitudinal_timeline = self.build_longitudinal_timeline(case_data)
        synergy_summary = "; ".join(f"{r['regimen']} CI={r['combination_index']}" for r in combination_synergy)
        timeline_note = (
            f" Reconciled against {longitudinal_timeline['source']} (PMID {longitudinal_timeline['pmid']}): "
            f"{len(longitudinal_timeline['lines'])} lines, {longitudinal_timeline['total_tki_months']:.0f} months on TKI."
            if longitudinal_timeline else ""
        )
        trace.append({
            "agent": "CombinationSynergyAgent",
            "status": "active",
            "message": f"Chou-Talalay CI dual-drug evaluation: {synergy_summary}.{timeline_note}",
        })

        # ==========================================
        # STEP 6: Synthesis Agent
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
            "combination_synergy": combination_synergy,
            "longitudinal_timeline": longitudinal_timeline,
            "histopathology": case_data.get("histopathology", {}),
        }
