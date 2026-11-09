@echo off
setlocal
title Install - live-voice-cohost
cd /d "%~dp0"

echo.
echo ================================================================
echo   live-voice-cohost - Installer
echo ================================================================
echo.

rem --- locate Python 3.11+ ---
set "PY="
for %%C in (py python) do (
    if not defined PY (
        %%C --version >nul 2>&1 && set "PY=%%C"
    )
)
if not defined PY (
    echo [ERROR] Python 3.11+ tidak ditemukan.
    echo Install dari https://www.python.org/downloads/ ^(centang "Add to PATH"^).
    echo.
    pause
    exit /b 1
)
echo [1/4] Python terdeteksi: %PY%

rem --- venv ---
if not exist ".venv\Scripts\python.exe" (
    echo [2/4] Membuat virtual environment .venv ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Gagal membuat venv.
        pause
        exit /b 1
    )
) else (
    echo [2/4] .venv sudah ada, skip.
)

rem --- install + daftarkan command `cohost` ---
echo [3/4] Install dependencies ^(sekali saja, +/- 1 menit^) ...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>&1
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 (
    echo [ERROR] Gagal install dependencies. Cek koneksi internet.
    pause
    exit /b 1
)

rem --- setup wizard ---
echo [4/4] Membuka konfigurasi...
echo.
".venv\Scripts\python.exe" -m jarvis.cli setup
if errorlevel 1 (
    echo [ERROR] Setup wizard gagal.
    pause
    exit /b 1
)

echo.
echo Selesai! Jalankan dengan perintah:
echo   .venv\Scripts\cohost.exe start   ^(atau^)   cohost.bat start
echo.
pause
endlocal