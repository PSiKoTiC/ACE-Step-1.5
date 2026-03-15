@echo off
setlocal enabledelayedexpansion
title ACE-Step Re-patch

echo.
echo ============================================================
echo  ACE-Step Re-patch
echo ============================================================
echo.
echo  Re-applies all custom patches without pulling updates.
echo  Your .env settings will NOT be touched.
echo.

REM ── Verify we're in the right place ──────────────────────────
set "ACESTEP_ROOT=%~dp0"
if "!ACESTEP_ROOT:~-1!"=="\" set "ACESTEP_ROOT=!ACESTEP_ROOT:~0,-1!"

if not exist "!ACESTEP_ROOT!\acestep" (
    echo ERROR: 'acestep' folder not found.
    echo Run this script from your ACE-Step root directory.
    pause
    exit /b 1
)

set "PATCH_DIR=!ACESTEP_ROOT!\_custom_patches"
if not exist "!PATCH_DIR!\install.py" (
    echo ERROR: Patch files not found at: !PATCH_DIR!
    echo Re-run the original installer to restore them.
    pause
    exit /b 1
)

REM ── Find Python and run ──────────────────────────────────────
if exist "!ACESTEP_ROOT!\python_embeded\python.exe" (
    echo [INFO] Using embedded Python...
    "!ACESTEP_ROOT!\python_embeded\python.exe" "!PATCH_DIR!\install.py" --patch-only "!ACESTEP_ROOT!"
    goto :Done
)

where uv >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    echo [INFO] Using uv...
    cd /d "!ACESTEP_ROOT!"
    uv run python "!PATCH_DIR!\install.py" --patch-only "!ACESTEP_ROOT!"
    goto :Done
)

where python >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    echo [INFO] Using system Python...
    python "!PATCH_DIR!\install.py" --patch-only "!ACESTEP_ROOT!"
    goto :Done
)

echo ERROR: No Python found.
pause
exit /b 1

:Done
echo.
pause
endlocal
