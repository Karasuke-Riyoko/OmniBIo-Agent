"""
run.py
One-click launcher for OmniBio-Agent.
Multimodal AI Hackathon 2026 (IIT Mandi, Augli.ai, PurpleRain Tech, LDV Labs).
"""

import sys
import webbrowser
import threading
import time
from app import app

def open_browser():
    time.sleep(1.2)
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("   🚀 OMNIBIO-AGENT | MULTIMODAL AI HACKATHON 2026")
    print("   IIT Mandi (KamandPrompt) · Augli.ai · PurpleRain Tech · LDV Labs")
    print("="*60)
    print("-> Tri-Modal Fusion: Histopathology + 3D PDB / SMILES + PubMed Text")
    print("-> PurpleRain Agentic Core: Planner -> Tool -> Critic -> Synthesis")
    print("-> Opening dashboard at: http://127.0.0.1:5000")
    print("="*60 + "\n")

    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
