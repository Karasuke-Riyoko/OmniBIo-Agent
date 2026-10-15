"""
app.py
Production Web Application for OmniBio-Agent.
Multimodal AI Hackathon 2026 (IIT Mandi, Augli.ai, PurpleRain Tech, LDV Labs).
"""

import json
import os
from flask import Flask, abort, jsonify, render_template, request, send_file
from agents.orchestrator import MultiAgentOrchestrator
from services import histology, spatial

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
    return jsonify({"status": "healthy", "service": "OmniBio-Agent", "version": "2.0.0"})


@app.route("/api/visium", methods=["GET"])
def get_visium_overlay():
    """Visium-geometry spots with computed escape scores, Gi* hotspots and the spatial recommendation."""
    return jsonify(spatial.analyze())


@app.route("/api/histology", methods=["GET"])
def get_histology():
    """Composition and patch grid measured from the current H&E tile."""
    return jsonify(histology.summary())


@app.route("/api/histology", methods=["POST"])
def upload_histology():
    """Replace the H&E tile used by the Perceiver (multipart field 'image')."""
    if "image" not in request.files:
        return jsonify({"error": "send the tile as multipart field 'image'"}), 400
    try:
        histology.save_upload(request.files["image"].read())
    except Exception as exc:
        return jsonify({"error": f"could not read image: {exc}"}), 400
    return jsonify(histology.summary())


@app.route("/api/histology", methods=["DELETE"])
def reset_histology():
    histology.reset_upload()
    return jsonify(histology.summary())


@app.route("/api/histology/image", methods=["GET"])
def get_histology_image():
    return send_file(histology.current_image_path(), max_age=0)


@app.route("/api/pdb/<pdb_id>", methods=["GET"])
def get_pdb(pdb_id):
    """Serves a locally cached structure from data/pdb/<ID>.pdb so the 3D viewer works offline."""
    path = os.path.join(BASE_DIR, "data", "pdb", f"{pdb_id.upper()}.pdb")
    if not pdb_id.isalnum() or not os.path.exists(path):
        abort(404)
    return send_file(path, mimetype="chemical/x-pdb", max_age=0)


@app.route("/api/benchmarks", methods=["GET"])
def get_benchmarks():
    """Metrics produced by scripts/train_affinity_model.py and scripts/run_benchmarks.py."""
    path = os.path.join(BASE_DIR, "data", "benchmarks.json")
    if not os.path.exists(path):
        return jsonify({}), 404
    with open(path, encoding="utf-8") as f:
        return jsonify(json.load(f))


if __name__ == "__main__":
    print("\n=======================================================")
    print("   OmniBio-Agent: Multimodal Precision Oncology Copilot")
    print("   Multimodal AI Hackathon 2026 (IIT Mandi / Augli / PurpleRain / LDV)")
    print("=======================================================")
    print("-> Server running at: http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
