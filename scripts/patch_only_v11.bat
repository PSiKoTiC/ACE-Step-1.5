@echo off
setlocal enabledelayedexpansion
title ACE-Step Patch Only v11

echo.
echo ============================================================
echo  ACE-Step Patch Only v11
echo ============================================================
echo.
echo  This script applies (or re-applies) all patches to an
echo  existing ACE-Step installation WITHOUT cloning or
echo  downloading models.
echo.
echo  Use this if you already have ACE-Step installed and just
echo  want to apply or re-apply the VRAM + GUI fixes.
echo.

REM ── Locate this script's directory ───────────────────────────
set "SCRIPT_DIR=%~dp0"
if "!SCRIPT_DIR:~-1!"=="\" set "SCRIPT_DIR=!SCRIPT_DIR:~0,-1!"

set "INSTALL_PY=!SCRIPT_DIR!\install.py"
if not exist "!INSTALL_PY!" (
    echo ERROR: install.py not found in: !SCRIPT_DIR!
    pause
    exit /b 1
)

REM ── Auto-detect or ask for ACE-Step root ─────────────────────
if exist "%CD%\acestep\llm_inference.py" (
    set "ACESTEP_ROOT=%CD%"
    echo  Auto-detected ACE-Step at: !ACESTEP_ROOT!
    goto :Confirm
)

REM Check if this script is inside the ace-step folder
set "PARENT=%SCRIPT_DIR%\.."
pushd "!PARENT!"
set "PARENT_ABS=%CD%"
popd
if exist "!PARENT_ABS!\acestep\llm_inference.py" (
    set "ACESTEP_ROOT=!PARENT_ABS!"
    echo  Auto-detected ACE-Step at: !ACESTEP_ROOT!
    goto :Confirm
)

set /p "ACESTEP_ROOT=Enter path to your ACE-Step folder: "
if not exist "!ACESTEP_ROOT!\acestep" (
    echo ERROR: 'acestep' subfolder not found at: !ACESTEP_ROOT!
    pause
    exit /b 1
)

:Confirm
echo.
echo  Will apply patches to: !ACESTEP_ROOT!
echo.
set /p "OK=Continue? (Y/N): "
if /i not "!OK!"=="Y" (
    echo Cancelled.
    pause
    exit /b 0
)

REM ── Find Python and run ──────────────────────────────────────
if exist "!ACESTEP_ROOT!\python_embeded\python.exe" (
    echo [INFO] Using embedded Python...
    "!ACESTEP_ROOT!\python_embeded\python.exe" "!INSTALL_PY!" --patch-only "!ACESTEP_ROOT!"
    goto :Done
)

where uv >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    echo [INFO] Using uv...
    cd /d "!ACESTEP_ROOT!"
    uv run python "!INSTALL_PY!" --patch-only "!ACESTEP_ROOT!"
    goto :Done
)

where python >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    echo [INFO] Using system Python...
    python "!INSTALL_PY!" --patch-only "!ACESTEP_ROOT!"
    goto :Done
)

echo ERROR: No Python found.
pause
exit /b 1

:Done
echo.
pause
endlocal
