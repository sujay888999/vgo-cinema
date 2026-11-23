@echo off
REM One-shot launcher for the FastAPI backend.

REM Force UTF-8 codepage BEFORE doing anything else — cmd defaults to GBK on
REM zh-CN Windows which often fails to find Chinese paths.
chcp 65001 >nul 2>&1

setlocal
set "HERE=%~dp0"
cd /d "%HERE%backend"
call "venv\Scripts\activate.bat"

set "VENV_BIN=%CD%\venv\Scripts"
if exist "%VENV_BIN%\uvicorn.exe" (
    "%VENV_BIN%\uvicorn.exe" backend.main:app --host 127.0.0.1 --port 8081 --log-level info
) else (
    echo [ERROR] uvicorn.exe not found at %VENV_BIN%\uvicorn.exe
    exit /b 1
)
endlocal
