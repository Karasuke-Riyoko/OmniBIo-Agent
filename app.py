"""
app.py
Production Web Application for OmniBio-Agent.
Multimodal AI Hackathon 2026 (IIT Mandi, Augli.ai, PurpleRain Tech, LDV Labs).
"""

import json
import os
from flask import Flask, jsonify, render_template
from agents.orchestrator import MultiAgentOrchestrator

app = Flask(__name__)
orchestrator = MultiAgentOrchestrator()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASE_FILE = os.path.join(BASE_DIR, "data", "sample_case.json")


def load_patient_case():
    with open(CASE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/case", methods=["GET"])
def get_case():
    """Returns the patient case profile."""
    return jsonify(load_patient_case())


@app.route("/api/analyze", methods=["GET", "POST"])
def analyze():
    """Executes the full multimodal multi-agent pipeline."""
    case_data = load_patient_case()
    results = orchestrator.run_full_pipeline(case_data)
    return jsonify(results)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "OmniBio-Agent", "version": "1.0.0"})


if __name__ == "__main__":
    print("\n=======================================================")
    print("   OmniBio-Agent: Multimodal Precision Oncology Copilot")
    print("   Multimodal AI Hackathon 2026 (IIT Mandi / Augli / PurpleRain / LDV)")
    print("=======================================================")
    print("-> Server running at: http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
