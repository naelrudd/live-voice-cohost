@echo off
rem `cohost` command shim — jalankan dari folder repo (mis. `cohost.bat setup`).
rem Setelah install.bat, `.venv\Scripts\cohost.exe` juga tersedia sebagai
rem console script asli bila Anda menambah `.venv\Scripts` ke PATH.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [cohost] Belum di-install. Jalankan install.bat dulu.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m jarvis.cli %*
exit /b %errorlevel%