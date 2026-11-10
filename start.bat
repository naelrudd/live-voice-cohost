@echo off
title Jarvis-Finance
cd /d "%~dp0"

echo [Jarvis] Menyalakan engine via CLI...
rem cohost start otomatis buka browser via kimi-webbridge (jika aktif) / default browser
"%~dp0cohost.bat" start
pause