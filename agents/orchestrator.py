"""
agents/orchestrator.py
PurpleRain-style multi-agent orchestrator for OmniBio-Agent.

Pipeline (each step appears in the agent trace the UI animates):
1. PlannerAgent           decomposes the case and drafts a ranking from predicted potency + tissue penetration only.
2. ToolExecutionAgent     PubChem / PubMed calls through call_with_recovery (retry, then fallback; logged).
3. CrossModalFusionAgent  Spatial Perceiver Resampler on real H&E patch tokens + tanh-gated cross-attention with
                          SMILES features and clinical text (untrained weights; reports shapes, gates, latency).
4. RedTeamCriticAgent     self-correcting loop: audits the current top candidates against resistance guardrails,
                          applies corrections, re-ranks, and repeats until an audit finds nothing new.
5. CombinationSynergyAgent Chou-Talalay CIs computed from dose-response fits, gated by allele phasing and the
                          spatial MET-niche flag from the Visium layer.
6. SynthesisAgent         final stratified dossier.
"""

import math
from typing import Any, Dict, List

import numpy as np

from models.fusion import OmniBioFusionPipeline
from services import histology, spatial
from services.api_tools import ChemTools, LiteratureTools
from services.combination_synergy import CombinationSynergyEngine
from services.featurizers import smiles_features, text_tokens


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

# binding_mode drives the critic's guardrails; evidence is the strongest clinical evidence tier for this setting.
DEFAULT_CANDIDATE_COMPOUNDS = [
    {"name": "Osimertinib (AZD9291)",
     "smiles": "CC(=O)N1CCN(CC1)C2=CC(=C(C=C2NC3=NC=CC(=N3)C4=CN(C5=CC=CC=C54)C)NC(=O)C=C)OC",
     "chembl_id": "CHEMBL3353410", "target_pdb": "4ZAU", "class": "3rd-Gen Irreversible EGFR TKI",
     "molecular_weight": 499.6, "binding_mode": "covalent_cys797", "evidence": "phase3",
     "mechanism": "Covalent Michael addition to Cys797; highly potent against T790M"},
    {"name": "BLU-945 (4th-Gen TKI)",
     "smiles": "CC(C)NC1=NC=C(C=N1)C2=CN(C3=CC=CC=C32)C4=CC(=C(C=C4)F)NC(=O)CC5=CC=NC=C5",
     "chembl_id": "CHEMBL4794821", "target_pdb": "6LUD", "class": "4th-Gen Mutant-Selective TKI",
     "molecular_weight": 526.6, "binding_mode": "reversible_mutant_selective", "evidence": "phase1",
     "mechanism": "Reversible ATP-competitive inhibition of the T790M/C797S triple mutant"},
    {"name": "EGFR PROTAC Degrader 14",
     "smiles": "CC1=C(C=C(C=C1)CNC2=NC=C(C=N2)NC3=CC(=C(C=C3)OC4CCNCC4)OC)C(=O)N5CCC(CC5)NC(=O)C6=CC=CC=C6C(=O)N7C(=O)CCC7=O",
     "chembl_id": "CHEMBL5129983", "target_pdb": "6LUD", "class": "Bifunctional Targeted Protein Degrader",
     "molecular_weight": 892.4, "binding_mode": "degrader", "evidence": "preclinical",
     "mechanism": "E3 ubiquitin ligase recruitment; proteasomal degradation (DC50 = 2.9 nM)"},
    {"name": "EAI045 + Cetuximab (Allosteric)",
     "smiles": "CC1=CC=C(C=C1)S(=O)(=O)NC2=CC=C(C=C2)C(=O)NC3=CC=C(C=C3)Cl",
     "chembl_id": "CHEMBL3989824", "target_pdb": "6Z4B", "class": "4th-Gen Allosteric Inhibitor",
     "molecular_weight": 444.9, "binding_mode": "allosteric", "evidence": "preclinical",
     "mechanism": "Allosteric αC-helix-out binding; requires dimerization blockade"},
    {"name": "Erlotinib",
     "smiles": "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
     "chembl_id": "CHEMBL553", "target_pdb": "1M17", "class": "1st-Gen Reversible TKI",
     "molecular_weight": 393.4, "binding_mode": "reversible_1st_gen", "evidence": "phase3",
     "mechanism": "Reversible ATP competitor; ineffective against restored ATP affinity of T790M"},
]

EVIDENCE_WEIGHT = {"phase3": 1.0, "phase2": 0.8, "phase1": 0.6, "preclinical": 0.3}

# Zhou et al. sequencing timepoints, keyed by the treatment line after which they were drawn.
MUTATION_TIMEPOINTS = [(0, "baseline"), (1, "post_gefitinib"), (2, "post_osimertinib"), (3, "post_combination")]

# Dual-drug regimens scored by the Chou-Talalay engine.
SYNERGY_REGIMENS = [
    "Pair_A_MET_Bypass_Blockade",                # SAVANNAH: Osimertinib + Savolitinib
    "Pair_B_Bispecific_Dual_Receptor_Blockade",  # MARIPOSA: Amivantamab + Lazertinib
    "Gefitinib_Osimertinib",                     # Zhou et al. line 3 (cis vs trans C797S)
]

PENETRATION_FLOOR_PCT = 55.0
MAX_CRITIC_ITERATIONS = 6
CRITIC_AUDIT_WIDTH = 3


def resistance_context(case_data: Dict[str, Any]) -> Dict[str, bool]:
    variants = " ".join(m.get("mutation", "") for m in case_data.get("genomic_mutations", []))
    return {
        "t790m": "T790M" in variants,
        "c797s": "C797S" in variants,
        "cis_c797s": "C797S (in cis)" in variants or "cis-C797S" in variants,
        "met_amplified": any(m.get("gene") == "MET" for m in case_data.get("genomic_mutations", [])),
    }


def critic_guardrails(drug: Dict[str, Any], ctx: Dict[str, bool]) -> List[Dict[str, Any]]:
    """Every guardrail the drug violates in this resistance context, with the score factor it costs."""
    mode, found = drug["binding_mode"], []
    if mode == "covalent_cys797" and ctx["c797s"]:
        found.append({"rule": "COVALENT_ANCHOR_LOST", "factor": 0.4 if ctx["cis_c797s"] else 0.75,
                      "status": "RESISTANCE_ESCAPE_OBSOLETE",
                      "why": "C797S replaces the Cys797 thiol the covalent warhead needs"
                             + (" (in cis with T790M)" if ctx["cis_c797s"] else "")})
    if mode == "reversible_1st_gen" and ctx["t790m"]:
        found.append({"rule": "GATEKEEPER_RESISTANCE", "factor": 0.25, "status": "CONTRAINDICATED_INEFFECTIVE",
                      "why": "T790M restores ATP affinity and blocks 1st-gen reversible TKIs"})
    if drug["penetration_pct"] < PENETRATION_FLOOR_PCT:
        found.append({"rule": "STROMAL_PENETRATION", "factor": max(0.6, drug["penetration_pct"] / PENETRATION_FLOOR_PCT),
                      "status": "HIGH_POTENCY_PENETRATION_LIMITED",
                      "why": f"only {drug['penetration_pct']}% predicted penetration through the stroma"})
    if mode == "allosteric":
        found.append({"rule": "MONOTHERAPY_INSUFFICIENT", "factor": 0.8, "status": "ALLOSTERIC_COMBINATION_CANDIDATE",
                      "why": "allosteric inhibitors need dimerisation blockade (Cetuximab)"})
    if drug["evidence"] == "preclinical":
        found.append({"rule": "PRECLINICAL_ONLY", "factor": 0.9, "status": "INVESTIGATIONAL",
                      "why": "no clinical data in this setting"})
    return found


class MultiAgentOrchestrator:
    def __init__(self):
        self.fusion_pipeline = OmniBioFusionPipeline(
            d_vis=len(histology.FEATURE_NAMES), d_mol=1024, d_text=768, d_shared=256
        )

    @staticmethod
    def calculate_spatial_pk_penetration(mw: float, stroma_pct: float, is_immune_excluded: bool = True) -> Dict[str, Any]:
        """
        Effective intratumoral penetration: C_eff / C_plasma = exp(-beta1 * rho_stroma * (MW / 500)^(1/3) - beta2 * I_excl).
        Penalises high-MW agents (e.g. ~900 Da PROTACs) in dense desmoplastic stroma.
        """
        beta1, beta2 = 0.65, 0.20
        rho_stroma = stroma_pct / 100.0
        mw_factor = (mw / 500.0) ** (1.0 / 3.0)
        c_eff_ratio = math.exp(-(beta1 * rho_stroma * mw_factor + (beta2 if is_immune_excluded else 0.0)))
        return {
            "penetration_efficiency_pct": round(c_eff_ratio * 100.0, 1),
            "stromal_diffusion_barrier": "High (desmoplastic stroma)" if stroma_pct > 50 else "Moderate",
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

    @staticmethod
    def evaluate_combination_synergy(case_data: Dict[str, Any], spatial_met_niche: bool = False) -> List[Dict[str, Any]]:
        """
        CIs computed by CombinationSynergyEngine from dose-response fits, in the patient's clonal context:
        cis-C797S phasing gates Gefitinib + Osimertinib; MET amplification (or a spatial MET-high niche awaiting
        confirmation) gates the SAVANNAH recommendation. Critic guardrails are attached.
        """
        ctx = resistance_context(case_data)
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
                pair_id, is_cis_c797s=ctx["cis_c797s"], is_met_amplified=ctx["met_amplified"]
            )
            verdict = ev["verdict"]
            if pair_id == "Pair_A_MET_Bypass_Blockade" and spatial_met_niche and not ctx["met_amplified"]:
                verdict = "SPATIAL MET NICHE: CONFIRM MET AMPLIFICATION ON TISSUE"
            meta = regimen_meta.get(pair_id, {})
            drug_1, drug_2 = ev["name"].split(" + ")
            rule = rules.get(guardrail_rules[pair_id])
            calc = ev["calculation"]
            results.append({
                "pair_id": pair_id,
                "regimen": ev["name"],
                "drug_1": meta.get("drug_1", drug_1),
                "drug_2": meta.get("drug_2", drug_2),
                "combination_index": ev["ci"],
                "ci_by_effect": [{"fa": r["fa"], "ci": r["combination_index"]} for r in calc["fa_ci"]],
                "ci_context": calc["context"],
                "ci_alpha": calc["alpha"],
                "ci_inputs": "illustrative dose-response (data/dose_response_illustrative.json)",
                "category": ev["category"],
                "is_synergistic": ev["ci"] < 0.9,
                "target": ev["target"],
                "trial": meta.get("phase_trial", ev["trial"]),
                "median_pfs_months": ev.get("mPFS_months", meta.get("median_pfs_months")),
                "orr_pct": ev.get("orr_pct", meta.get("confirmed_orr_pct")),
                "hazard_ratio": ev.get("hazard_ratio", meta.get("hazard_ratio")),
                "verdict": verdict,
                "rationale": meta.get("rationale", ""),
                "guardrails": [rule] if rule else [],
            })

        results.sort(key=lambda r: r["combination_index"])
        return results

    @staticmethod
    def planner_score(drug: Dict[str, Any]) -> float:
        """Resistance-blind draft score: predicted potency, tissue penetration and evidence tier."""
        potency = min(max((drug["predicted_pic50"] - 5.0) / 4.0, 0.0), 1.0)
        return 100.0 * (0.55 * potency + 0.30 * drug["penetration_pct"] / 100.0 + 0.15 * EVIDENCE_WEIGHT[drug["evidence"]])

    @staticmethod
    def self_correcting_critic(drugs: List[Dict[str, Any]], ctx: Dict[str, bool]) -> Dict[str, Any]:
        """
        Iteratively audits the current top-CRITIC_AUDIT_WIDTH candidates. Each audit applies any guardrail
        corrections it finds and re-ranks; the loop stops when an audit finds nothing new (converged).
        """
        for d in drugs:
            d["score"] = d["planner_score"]
            d["violations"] = []
        audited, iterations = set(), []
        for it in range(1, MAX_CRITIC_ITERATIONS + 1):
            ranking = sorted(drugs, key=lambda d: d["score"], reverse=True)
            top = ranking[:CRITIC_AUDIT_WIDTH]
            corrections = []
            for d in top:
                if d["name"] in audited:
                    continue
                audited.add(d["name"])
                for v in critic_guardrails(d, ctx):
                    d["score"] *= v["factor"]
                    d["violations"].append(v)
                    corrections.append(f"{d['name'].split(' (')[0]}: {v['rule']} (x{v['factor']:.2f})")
            iterations.append({
                "iteration": it,
                "audited": [d["name"] for d in top],
                "corrections": corrections,
                "top_after": sorted(drugs, key=lambda d: d["score"], reverse=True)[0]["name"],
            })
            if not corrections:
                break
        # Audit everything once more so non-top candidates still carry their flags in the dossier.
        for d in drugs:
            if d["name"] not in audited:
                for v in critic_guardrails(d, ctx):
                    d["score"] *= v["factor"]
                    d["violations"].append(v)
        return {"iterations": iterations, "converged": not iterations[-1]["corrections"]}

    def run_full_pipeline(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the six-agent cycle and returns everything the UI renders."""
        case_data = self.normalize_case(case_data)
        ctx = resistance_context(case_data)
        trace: List[Dict[str, str]] = []

        # STEP 1: Planner
        mutations = [f"{m['gene']} {m['mutation']}" for m in case_data.get("genomic_mutations", [])]
        trace.append({
            "agent": "PlannerAgent", "status": "active",
            "message": f"Deconstructed case {case_data['patient_id']}. Active mutational profile: {', '.join(mutations)}. "
                       f"Drafting a resistance-blind ranking of {len(case_data.get('candidate_compounds', []))} candidates "
                       "from predicted potency, penetration and evidence; the critic will audit it.",
        })

        # STEP 2: Tools (self-healing calls) + histology + potency model
        histo = histology.extract()
        stroma_pct = histo["composition"]["stroma_pct_of_tissue"]
        is_immune_excluded = "Immune-Excluded" in case_data.get("histopathology", {}).get("til_phenotype", "")
        tool_events: List[Dict[str, Any]] = []

        drugs = []
        for cmp in case_data.get("candidate_compounds", []):
            pubchem = ChemTools.fetch_pubchem_properties(cmp["smiles"], events=tool_events)
            potency = ChemTools.predict_egfr_potency(cmp["smiles"])
            mw = pubchem.get("molecular_weight") or cmp.get("molecular_weight", 500.0)
            pk = self.calculate_spatial_pk_penetration(mw, stroma_pct, is_immune_excluded)
            literature = LiteratureTools.search_pubmed_evidence(f"{cmp['name'].split(' (')[0]} EGFR NSCLC resistance",
                                                                events=tool_events)
            drug = {
                "name": cmp["name"], "class": cmp["class"], "smiles": cmp["smiles"], "target_pdb": cmp["target_pdb"],
                "binding_mode": cmp.get("binding_mode", "reversible_mutant_selective"),
                "evidence": cmp.get("evidence", "preclinical"),
                "molecular_weight": round(mw, 1),
                "penetration_pct": pk["penetration_efficiency_pct"],
                "predicted_pic50": potency["predicted_pic50"],
                "predicted_ic50_nm": potency["predicted_ic50_nm"],
                "predicted_affinity_kcal": potency["estimated_delta_g_kcal_mol"],
                "pubchem_source": pubchem["source"],
                "literature": literature,
            }
            drug["planner_score"] = round(self.planner_score(drug), 1)
            drugs.append(drug)

        live = sum(e["outcome"] == "live" for e in tool_events)
        cached = sum(e["outcome"] == "cache" for e in tool_events)
        healed = [e for e in tool_events if e["outcome"] == "recovered_with_fallback"]
        heal_note = (f"{len(healed)} failed after retry and were healed with offline fallbacks "
                     f"({', '.join(sorted({e['tool'] for e in healed}))})." if healed else "no failures to heal.")
        trace.append({
            "agent": "ToolExecutionAgent", "status": "active",
            "message": f"{len(tool_events)} PubChem/PubMed calls: {live} live, {cached} cached; {heal_note} "
                       f"EGFR potency predicted by the ChEMBL-trained model; H&E tile measured at {stroma_pct}% stroma.",
        })

        # STEP 3: Cross-modal fusion on real inputs
        chem_tokens = np.stack([smiles_features(d["smiles"], dim=1024) for d in drugs])
        clinical = [case_data.get("clinical_summary", ""), "Mutations: " + "; ".join(mutations),
                    f"Histology: {case_data.get('diagnosis', '')}, stroma {stroma_pct}%"]
        fusion_results = self.fusion_pipeline.execute_fusion(histo["tokens"], chem_tokens, text_tokens(clinical))
        trace.append({
            "agent": "CrossModalFusionAgent", "status": "active",
            "message": f"Stage 1 Spatial Perceiver Resampler compressed {fusion_results['n_patches']} real H&E patch tokens "
                       f"into {fusion_results['n_latents']} morphological latents; Stage 2 tanh-gated cross-attention fused "
                       f"them with {len(drugs)} SMILES token sets and {len(clinical)} clinical text tokens in "
                       f"{fusion_results['fusion_latency_ms']} ms (untrained weights).",
        })

        # STEP 4: Self-correcting red-team critic
        critic = self.self_correcting_critic(drugs, ctx)
        steps = " ".join(
            f"Pass {it['iteration']}: " + ("; ".join(it["corrections"]) if it["corrections"] else "no new violations, converged.")
            for it in critic["iterations"])
        trace.append({
            "agent": "RedTeamCriticAgent", "status": "active",
            "message": f"Self-correcting audit of the planner's draft ({'cis' if ctx['cis_c797s'] else 'trans'}-C797S context). {steps}",
        })

        # STEP 5: Spatial layer + combination synergy
        spatial_result = spatial.analyze()
        combination_synergy = self.evaluate_combination_synergy(case_data, spatial_result["recommendation"]["spatial_met_niche"])
        longitudinal_timeline = self.build_longitudinal_timeline(case_data)
        synergy_summary = "; ".join(f"{r['regimen']} CI={r['combination_index']}" for r in combination_synergy)
        timeline_note = (
            f" Reconciled against {longitudinal_timeline['source']} (PMID {longitudinal_timeline['pmid']}): "
            f"{len(longitudinal_timeline['lines'])} lines, {longitudinal_timeline['total_tki_months']:.0f} months on TKI."
            if longitudinal_timeline else ""
        )
        trace.append({
            "agent": "CombinationSynergyAgent", "status": "active",
            "message": f"Chou-Talalay CI at ED50 from median-effect fits: {synergy_summary}. "
                       f"Spatial layer: {spatial_result['recommendation']['summary']}{timeline_note}",
        })

        # STEP 6: Synthesis
        ranked = sorted(drugs, key=lambda d: d["score"], reverse=True)
        ranked_drugs = []
        for i, d in enumerate(ranked):
            if d["violations"]:
                worst = min(d["violations"], key=lambda v: v["factor"])
                status = worst["status"]
            elif i == 0:
                status = "FIRST_LINE_C797S_TARGETED" if ctx["c797s"] else "FIRST_LINE_RECOMMENDATION"
            else:
                status = "SECOND_LINE_CANDIDATE"
            if i == 0 and d["violations"] and min(v["factor"] for v in d["violations"]) >= 0.8:
                status = "FIRST_LINE_C797S_TARGETED" if ctx["c797s"] else "FIRST_LINE_RECOMMENDATION"
            rationale = (f"Predicted EGFR pIC50 {d['predicted_pic50']} (IC50 ≈ {d['predicted_ic50_nm']} nM), "
                         f"{d['penetration_pct']}% stromal penetration. Planner draft {d['planner_score']} → critic "
                         f"{round(d['score'], 1)}. " +
                         (" ".join(f"{v['rule']}: {v['why']}." for v in d["violations"]) or "No guardrail violations."))
            ranked_drugs.append({
                "name": d["name"], "class": d["class"], "smiles": d["smiles"], "target_pdb": d["target_pdb"],
                "molecular_weight": d["molecular_weight"], "penetration_pct": d["penetration_pct"],
                "predicted_pic50": d["predicted_pic50"], "predicted_affinity_kcal": d["predicted_affinity_kcal"],
                "planner_score": d["planner_score"], "composite_score": round(d["score"], 1),
                "critic_status": status, "critic_rationale": rationale,
                "violations": [v["rule"] for v in d["violations"]],
                "literature_citations": [a["title"] for a in d["literature"]["top_articles"]],
                "literature_source": d["literature"]["source"],
            })

        trace.append({
            "agent": "SynthesisAgent", "status": "completed",
            "message": f"Final ranking after {len(critic['iterations'])} critic passes: "
                       + ", ".join(f"#{i + 1} {d['name'].split(' (')[0]} ({d['composite_score']})" for i, d in enumerate(ranked_drugs[:3]))
                       + ". Dossier includes C797S surveillance and stromal penetration corrections.",
        })

        combination_recommendations = self.recommend_combinations(ranked_drugs, case_data)

        return {
            "patient_summary": {
                "id": case_data["patient_id"],
                "diagnosis": case_data["diagnosis"],
                "summary": case_data["clinical_summary"],
                "actionable_mutations": mutations,
                "conformational_ensembles": case_data.get("conformational_ensembles", []),
            },
            "agent_trace": trace,
            "tool_events": tool_events,
            "critic_loop": critic,
            "fusion_telemetry": {k: v for k, v in fusion_results.items() if k != "fused_vector"},
            "ranked_compounds": ranked_drugs,
            "top_recommendation": ranked_drugs[0],
            "combination_recommendations": combination_recommendations,
            "combination_synergy": combination_synergy,
            "longitudinal_timeline": longitudinal_timeline,
            "histopathology": {**case_data.get("histopathology", {}), "measured_from_image": histo["composition"],
                               "image_source": histo["source_note"]},
            "spatial_recommendation": spatial_result["recommendation"],
        }
