"""
scripts/full_autopilot.py
100% Hands-Free Autonomous Loop:
Antigravity (Coordinator & Sandbox) <--> Claude Code CLI (Native Execution)

No copy-pasting. No browser windows.
Runs Claude CLI directly in the desktop repository, applies code, tests, and self-heals!
"""

import sys
import os
import subprocess
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def run_claude(prompt: str) -> str:
    """Executes Claude Code CLI natively inside the project folder."""
    print(f"\n[Claude CLI] 🚀 Invoking Claude Code with task...")
    cmd = [
        "claude.exe",
        "-p",
        "--dangerously-skip-permissions",
        prompt
    ]
    res = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True)
    out = res.stdout + "\n" + res.stderr
    print(f"[Claude CLI] 🏁 Finished execution.")
    return out

def run_tests() -> tuple[bool, str]:
    """Runs automated unit tests in the local sandbox."""
    res = subprocess.run(
        [sys.executable, "-m", "unittest", "tests/test_pipeline.py"],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )
    return res.returncode == 0, res.stdout + "\n" + res.stderr

def autonomous_loop(initial_task: str, max_iterations: int = 5):
    print("\n" + "="*70)
    print("   🤖 100% HANDS-FREE AUTONOMOUS CLAUDE-ANTIGRAVITY LOOP")
    print("="*70)
    print(f"Project: {BASE_DIR}")
    print(f"Initial Task: {initial_task[:80]}...\n")

    current_prompt = initial_task
    iteration = 1

    while iteration <= max_iterations:
        print(f"\n--- [CYCLE {iteration}/{max_iterations}] Running Claude Code ---")
        claude_output = run_claude(current_prompt)

        print("\n--- [CYCLE {iteration}] Testing Sandbox ---")
        passed, test_log = run_tests()

        if passed:
            print("\n" + "="*60)
            print("   🎉 SUCCESS! ALL TESTS PASSED (100% GREEN)!")
            print("="*60)
            print("Project is fully updated and verified on your Desktop.")
            print("Launch UI: python run.py\n")
            return True
        else:
            print(f"\n[!] Tests failed on iteration {iteration}. Triggering Autonomous Self-Healing...")
            print(test_log[:300] + "...")
            current_prompt = f"""
            The previous changes caused a test failure in tests/test_pipeline.py.
            Here is the exact traceback:
            {test_log}

            SELF-HEALING MANDATE:
            Inspect the codebase, identify the root cause, and fix the code so that `python -m unittest tests/test_pipeline.py` passes 100%.
            """
            iteration += 1
            time.sleep(2)

    print("\n[!] Max iterations reached. Please inspect tests/test_pipeline.py.")
    return False

if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Enhance the project with Argon's combination therapy recommendations. "
        "Review data/sample_case.json and agents/orchestrator.py to ensure BLU-945 and Osimertinib "
        "have high ranking scores and all unit tests pass with python -m unittest tests/test_pipeline.py."
    )
    autonomous_loop(task)
