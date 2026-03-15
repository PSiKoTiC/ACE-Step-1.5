@echo off
setlocal enabledelayedexpansion
title ACE-Step Custom Setup v11 — Full Install (Autodetect)

echo.
echo ============================================================
echo  ACE-Step Custom Setup v11 — Full Install (Autodetect)
echo ============================================================
echo.
echo  This script will:
echo    0. Detect your GPU, VRAM, and system memory
echo    1. Recommend an installation profile for your hardware
echo    2. Clone ACE-Step v1.5 from GitHub
echo    3. Set up the Python environment (uv sync)
echo    4. Download models for your chosen profile
echo    5. Apply VRAM + GUI patches
echo    6. Install .env settings + Start_Custom.bat
echo.
echo  Prerequisites:
echo    - git must be installed and on PATH
echo    - uv must be installed (or embedded Python present)
echo.

REM ── Ask for install directory ────────────────────────────────
set /p "INSTALL_DIR=Enter install directory (e.g. C:\ace-step): "
if "!INSTALL_DIR!"=="" (
    echo ERROR: No directory specified.
    pause
    exit /b 1
)

REM ── Verify git is available ──────────────────────────────────
where git >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo.
    echo ERROR: git is not installed or not on PATH.
    echo Please install git from: https://git-scm.com/download/win
    pause
    exit /b 1
)

REM ── Locate this script's directory ───────────────────────────
set "SCRIPT_DIR=%~dp0"
if "!SCRIPT_DIR:~-1!"=="\" set "SCRIPT_DIR=!SCRIPT_DIR:~0,-1!"

set "INSTALL_PY=!SCRIPT_DIR!\install.py"
if not exist "!INSTALL_PY!" (
    echo ERROR: install.py not found in: !SCRIPT_DIR!
    echo Make sure install.bat and install.py are in the same folder.
    pause
    exit /b 1
)

REM ── Check if uv is available ─────────────────────────────────
where uv >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    echo [INFO] uv found — will use uv for environment setup.
    goto :RunInstall
)

REM ── Check for embedded Python ────────────────────────────────
if exist "!INSTALL_DIR!\python_embeded\python.exe" (
    echo [INFO] Embedded Python found.
    goto :RunInstall
)

REM ── uv not found — offer to install it ───────────────────────
echo.
echo uv is not installed. Would you like to install it now?
echo (uv is the recommended Python package manager for ACE-Step)
echo.
set /p "INSTALL_UV=Install uv? (Y/N): "
if /i "!INSTALL_UV!"=="Y" (
    echo Installing uv...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    echo.
    echo uv installed. You may need to restart your terminal for PATH changes.
    echo Re-run this script after restarting your terminal.
    pause
    exit /b 0
)

REM ── Fall back to system Python ───────────────────────────────
where python >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    echo [INFO] Using system Python.
    python "!INSTALL_PY!" "!INSTALL_DIR!"
    goto :Done
)

echo ERROR: No Python found (uv, embedded, or system).
echo Please install uv or Python first.
pause
exit /b 1

:RunInstall
REM Try uv first, then embedded, then system
where uv >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    echo [INFO] Running installer with uv...
    uv run --no-project python "!INSTALL_PY!" "!INSTALL_DIR!"
    goto :Done
)

if exist "!INSTALL_DIR!\python_embeded\python.exe" (
    echo [INFO] Running installer with embedded Python...
    "!INSTALL_DIR!\python_embeded\python.exe" "!INSTALL_PY!" "!INSTALL_DIR!"
    goto :Done
)

python "!INSTALL_PY!" "!INSTALL_DIR!"

:Done
echo.
echo ============================================================
echo  Setup complete. See summary above for results.
echo  Launch ACE-Step with: Start_Custom.bat
echo  (located in: !INSTALL_DIR!\Start_Custom.bat)
echo ============================================================
echo.
pause
endlocal
