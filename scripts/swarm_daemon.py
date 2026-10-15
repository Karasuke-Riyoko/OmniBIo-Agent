"""
scripts/swarm_daemon.py
Autonomous Continuous Self-Healing Loop:
Gemini Argon (Research) <-> Antigravity (Local Sandbox / File I/O) <-> Claude (Max-Thinking Code Engine)

Monitors:
- .swarm_sync/argon_input.md   -> Ingests clinical research from Argon
- .swarm_sync/CLAUDE_TASK.md   -> Dynamic prompt with test feedback & Max-Thinking directives
- .swarm_sync/claude_reply.md  -> Automatically extracts code blocks, writes files, and runs tests
- Auto-catches tracebacks and updates CLAUDE_TASK.md for self-healing iteration
"""

import os
import sys
import time
import json
import re
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SYNC_DIR = BASE_DIR / ".swarm_sync"
SYNC_DIR.mkdir(exist_ok=True)

STATE_FILE = SYNC_DIR / "state.json"
ARGON_INPUT = SYNC_DIR / "argon_input.md"
CLAUDE_TASK = SYNC_DIR / "CLAUDE_TASK.md"
CLAUDE_REPLY = SYNC_DIR / "claude_reply.md"

def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"phase": "INITIAL", "iteration": 0, "tests_passed": False, "last_error": None}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

def run_tests():
    """Runs tests and returns (passed: bool, output: str)"""
    res = subprocess.run(
        [sys.executable, "-m", "unittest", "tests/test_pipeline.py"],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )
    return res.returncode == 0, res.stdout + "\n" + res.stderr

def extract_and_write_code_blocks(content: str) -> list[str]:
    """
    Parses markdown code blocks formatted as:
    ### File: models/fusion.py
    ```python
    ...
    ```
    or ```python file="models/fusion.py"
    """
    written_files = []
    
    # Pattern 1: ### File: <filepath> \n ```language \n <code> ```
    pattern1 = re.compile(r"###\s+File:\s*([^\n\r]+)[\r\n]+```[a-zA-Z0-9_-]*[\r\n]+(.*?)```", re.DOTALL)
    for match in pattern1.finditer(content):
        rel_path = match.group(1).strip().replace("\\", "/")
        code = match.group(2)
        target = BASE_DIR / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code, encoding="utf-8")
        written_files.append(rel_path)

    # Pattern 2: ```python file="<filepath>" \n <code> ```
    pattern2 = re.compile(r"```[a-zA-Z0-9_-]*\s+file=[\"']([^\"']+)[\"'][\r\n]+(.*?)```", re.DOTALL)
    for match in pattern2.finditer(content):
        rel_path = match.group(1).strip().replace("\\", "/")
        code = match.group(2)
        target = BASE_DIR / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code, encoding="utf-8")
        written_files.append(rel_path)

    return written_files

def generate_claude_prompt(argon_data: str, error_trace: str = None, iteration: int = 1):
    """Formats an exhaustive prompt tailored for Claude with Max/Extended Thinking."""
    error_section = ""
    if error_trace:
        error_section = f"""
### [CRITICAL TEST FAILURE - ITERATION {iteration}]
The local test sandbox executed `python -m unittest tests/test_pipeline.py` and encountered this failure:
```
{error_trace.strip()}
```
**SELF-HEALING MANDATE**:
1. Spend your thinking tokens pinpointing the exact line, tensor shape, or assertion mismatch in the traceback.
2. Fix the error completely in your output files. Do not stop until all edge cases are healed.
"""

    prompt = f"""# ROLE: PRINCIPAL AI ENGINEER (MAX-THINKING / EXTENDED THINKING MODE)
Project: OmniBio-Agent (Multimodal AI Hackathon 2026 - IIT Mandi, Augli.ai, PurpleRain Tech, LDV Labs)
Workspace: C:\\Users\\Asus\\Desktop\\multimodal_ai_hackathon_2026
Loop Iteration: {iteration}

### EXTENDED THINKING DIRECTIVE:
You have maximum thinking budget. Spend extensive thinking tokens to:
1. Cross-reference all tensor projection shapes, softmax partitions, and mathematical constraints.
2. Verify all biophysical drug-resistance pathways (EGFR L858R, T790M, C797S cis/trans, MET bypass).
3. Ensure zero imports fail, zero stubs exist, and all asynchronous / Flask endpoints return HTTP 200.

{error_section}

### CURRENT INPUT FROM GEMINI 4 ARGON (RESEARCH INTELLIGENCE):
{argon_data if argon_data else "Using existing baseline biophysical and clinical dataset."}

### FORMAT INSTRUCTION FOR FILE UPDATES:
For every file you create or update, format it EXACTLY like this so our automated daemon writes it to disk:

### File: path/to/filename.ext
```language
// full complete code here (no placeholders)
```

Generate the verified code files now.
"""
    CLAUDE_TASK.write_text(prompt, encoding="utf-8")

def main_loop():
    print("\n" + "="*70)
    print("   🔁 AUTONOMOUS SWARM LOOP DAEMON ACTIVE")
    print("   Gemini Argon -> Antigravity Sandbox -> Claude (Max Thinking)")
    print("="*70)
    print(f"Monitoring folder: {SYNC_DIR}")
    print("1. Paste Argon research into:  .swarm_sync/argon_input.md")
    print("2. Copy Claude prompt from:    .swarm_sync/CLAUDE_TASK.md")
    print("3. Paste Claude code into:     .swarm_sync/claude_reply.md")
    print("="*70 + "\n")

    last_argon_mtime = 0
    last_claude_mtime = 0

    state = load_state()

    # Initial baseline check
    passed, test_log = run_tests()
    state["tests_passed"] = passed
    state["last_error"] = None if passed else test_log
    save_state(state)
    print(f"[*] Baseline Test Status: {'✅ PASSED (4/4 tests)' if passed else '❌ FAILED'}")

    # Generate initial Claude prompt
    generate_claude_prompt(
        argon_data=ARGON_INPUT.read_text(encoding="utf-8") if ARGON_INPUT.exists() else None,
        error_trace=state["last_error"],
        iteration=state["iteration"]
    )
    print("[+] Ready: Generated .swarm_sync/CLAUDE_TASK.md")

    while True:
        try:
            # 1. Check for Argon update
            if ARGON_INPUT.exists():
                mtime = ARGON_INPUT.stat().st_mtime
                if mtime > last_argon_mtime:
                    last_argon_mtime = mtime
                    print("\n[Argon] 📥 New Research Input Detected! Updating Claude Task...")
                    argon_text = ARGON_INPUT.read_text(encoding="utf-8")
                    state["phase"] = "ARGON_INGESTED"
                    state["iteration"] += 1
                    generate_claude_prompt(argon_text, error_trace=state["last_error"], iteration=state["iteration"])
                    save_state(state)
                    print(f"[Argon] 🔄 Updated .swarm_sync/CLAUDE_TASK.md (Iteration {state['iteration']})")

            # 2. Check for Claude reply update
            if CLAUDE_REPLY.exists():
                mtime = CLAUDE_REPLY.stat().st_mtime
                if mtime > last_claude_mtime:
                    last_claude_mtime = mtime
                    print("\n[Claude] 💻 New Code Detected! Extracting and writing files...")
                    reply_text = CLAUDE_REPLY.read_text(encoding="utf-8")
                    files = extract_and_write_code_blocks(reply_text)
                    if files:
                        print(f"[Claude] ✍️  Wrote {len(files)} files: {', '.join(files)}")
                    else:
                        print("[Claude] ⚠️  No formatted file blocks found. Use '### File: path/to/file'")

                    # Immediate Sandbox Execution & Test
                    print("[Sandbox] 🧪 Executing automated test suite...")
                    passed, test_log = run_tests()
                    state["tests_passed"] = passed
                    state["last_error"] = None if passed else test_log
                    save_state(state)

                    if passed:
                        print("\n" + "="*50)
                        print("   🎉 ALL TESTS PASSED! 100% GREEN (CYCLE COMPLETE)")
                        print("="*50)
                        print("Dashboard running at: http://127.0.0.1:5000\n")
                    else:
                        print("\n[Sandbox] ❌ Tests failed! Triggering Self-Healing Loop back to Claude...")
                        state["phase"] = "SELF_HEALING"
                        generate_claude_prompt(
                            argon_data=ARGON_INPUT.read_text(encoding="utf-8") if ARGON_INPUT.exists() else None,
                            error_trace=test_log,
                            iteration=state["iteration"]
                        )
                        print(f"[Loop] 🔁 Injected error trace into .swarm_sync/CLAUDE_TASK.md for Claude to fix!")

            time.sleep(2)
        except KeyboardInterrupt:
            print("\nStopping Swarm Daemon.")
            break
        except Exception as e:
            print(f"[!] Daemon Error: {e}")
            time.sleep(3)

if __name__ == "__main__":
    main_loop()
