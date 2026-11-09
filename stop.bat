@echo off
setlocal
cd /d "%~dp0"
for /f "tokens=5" %%P in ('netstat -aon ^| findstr 8848 ^| findstr LISTENING') do (
    taskkill /F /PID %%P >nul 2>&1
    echo [Jarvis] Stopped PID %%P
)
endlocal