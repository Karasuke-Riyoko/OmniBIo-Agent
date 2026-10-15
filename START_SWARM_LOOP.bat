@echo off
title AUTONOMOUS SWARM LOOP (ARGON - ANTIGRAVITY - CLAUDE)
color 0A
echo ======================================================================
echo    STARTING CONTINUOUS SELF-HEALING SWARM LOOP
echo    Gemini Argon (Research) -> Antigravity (Sandbox) -> Claude (Max Thinking)
echo ======================================================================
echo.
start "" "%~dp0.swarm_sync"
python "%~dp0scripts\swarm_daemon.py"
pause
