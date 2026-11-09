@echo off
title Setup - live-voice-cohost
cd /d "%~dp0"

echo [live-voice-cohost] Membuka Setup Wizard...
echo (Tutup window GUI atau gunakan setup.bat --cli untuk mode terminal)
echo.

".venv\Scripts\python.exe" -m jarvis.setup %*
pause