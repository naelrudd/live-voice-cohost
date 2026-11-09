@echo off
title Jarvis-Finance
cd /d "%~dp0"

echo [Jarvis] Menyalakan engine...
start "" "http://127.0.0.1:8848/control"
".venv\Scripts\python.exe" -m jarvis.main
pause