@echo off
setlocal enabledelayedexpansion
title ACE-Step Update + Re-patch

echo.
echo ============================================================
echo  ACE-Step Update + Re-patch
echo ============================================================
echo.
echo  This script will:
echo    1. Pull the latest code from GitHub (git pull)
echo    2. Update dependencies (uv sync)
echo    3. Re-apply custom patches
echo.
echo  Your .env settings will NOT be touched.
echo.

REM ── Verify we're in the right place ──────────────────────────
if not exist "%~dp0acestep" (
    echo ERROR: 'acestep' folder not found.
    echo Run this script from your ACE-Step root directory.
    pause
    exit /b 1
)

set "ACESTEP_ROOT=%~dp0"
if "!ACESTEP_ROOT:~-1!"=="\" set "ACESTEP_ROOT=!ACESTEP_ROOT:~0,-1!"

REM ── Verify patch files exist ─────────────────────────────────
set "PATCH_DIR=!ACESTEP_ROOT!\_custom_patches"
if not exist "!PATCH_DIR!\install.py" (
    echo ERROR: Patch files not found at: !PATCH_DIR!
    echo Re-run the original installer to restore them.
    pause
    exit /b 1
)

REM ── Step 1: Git pull ─────────────────────────────────────────
echo.
echo [Step 1] Pulling latest code from GitHub...
echo.
cd /d "!ACESTEP_ROOT!"
git pull
if !ERRORLEVEL! NEQ 0 (
    echo.
    echo WARNING: git pull had issues. This may be fine if you have local changes.
    echo Continuing with dependency update and patches...
    echo.
)

REM ── Step 2: uv sync ─────────────────────────────────────────
echo.
echo [Step 2] Updating dependencies...
echo.
if exist "!ACESTEP_ROOT!\python_embeded\python.exe" (
    echo [INFO] Embedded Python detected — skipping uv sync
) else (
    where uv >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        uv sync
    ) else (
        echo WARNING: uv not found — skipping dependency update
    )
)

REM ── Step 3: Re-apply patches ─────────────────────────────────
echo.
echo [Step 3] Re-applying custom patches...
echo.

if exist "!ACESTEP_ROOT!\python_embeded\python.exe" (
    "!ACESTEP_ROOT!\python_embeded\python.exe" "!PATCH_DIR!\install.py" --patch-only "!ACESTEP_ROOT!"
    goto :Done
)

where uv >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    cd /d "!ACESTEP_ROOT!"
    uv run python "!PATCH_DIR!\install.py" --patch-only "!ACESTEP_ROOT!"
    goto :Done
)

where python >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    python "!PATCH_DIR!\install.py" --patch-only "!ACESTEP_ROOT!"
    goto :Done
)

echo ERROR: No Python found.
pause
exit /b 1

:Done
echo.
echo ============================================================
echo  Update complete. See patch summary above.
echo  Launch with: Start_Custom.bat
echo ============================================================
echo.
pause
endlocal
