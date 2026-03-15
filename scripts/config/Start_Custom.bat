@echo off
setlocal enabledelayedexpansion

REM ACE-Step Gradio Web UI Launcher
call :LoadEnvFile

REM ==================== Configuration ====================
if not defined PORT        set PORT=7860
if not defined SERVER_NAME set SERVER_NAME=0.0.0.0
if not defined LANGUAGE    set LANGUAGE=en

REM Default to SFT model if not overridden by .env
if not defined CONFIG_PATH   set CONFIG_PATH=--config_path acestep-v15-sft
if not defined LM_MODEL_PATH set LM_MODEL_PATH=--lm_model_path acestep-5Hz-lm-4B

REM Force init on startup so --config_path is respected before GPU tier auto-select kicks in
if not defined INIT_SERVICE  set INIT_SERVICE=--init_service true

REM ==================== Build Argument String ====================
set "FULL_ARGS=--port %PORT% --server-name %SERVER_NAME% --language %LANGUAGE%"

if not "%CONFIG_PATH%"==""   set "FULL_ARGS=!FULL_ARGS! %CONFIG_PATH%"
if not "%LM_MODEL_PATH%"=="" set "FULL_ARGS=!FULL_ARGS! %LM_MODEL_PATH%"
if not "%INIT_SERVICE%"==""  set "FULL_ARGS=!FULL_ARGS! %INIT_SERVICE%"
if not "%INIT_LLM%"==""      set "FULL_ARGS=!FULL_ARGS! %INIT_LLM%"
if not "%BATCH_SIZE%"==""    set "FULL_ARGS=!FULL_ARGS! %BATCH_SIZE%"
if not "%AUTH_USERNAME%"=="" set "FULL_ARGS=!FULL_ARGS! %AUTH_USERNAME%"
if not "%AUTH_PASSWORD%"=="" set "FULL_ARGS=!FULL_ARGS! %AUTH_PASSWORD%"
if not "%LM_BACKEND%"==""  set "FULL_ARGS=!FULL_ARGS! %LM_BACKEND%"

set "FULL_ARGS=!FULL_ARGS! --device cuda"

REM ==================== Debug Output ====================
echo ================================================
echo  ACE-Step Launcher
echo ================================================
echo  Config Path  : [%CONFIG_PATH%]
echo  LM Model     : [%LM_MODEL_PATH%]
echo  Init Service : [%INIT_SERVICE%]
echo  Auth User    : [%AUTH_USERNAME%]
echo  Port         : [%PORT%]
echo  Language     : [%LANGUAGE%]
echo  Init LLM     : [%INIT_LLM%]
echo  Batch Size   : [%BATCH_SIZE%]
echo  LM Backend   : [%LM_BACKEND%]
echo  PyTorch Alloc: [%PYTORCH_CUDA_ALLOC_CONF%]
echo.
echo  Full args: !FULL_ARGS!
echo ================================================
echo.

REM ==================== Launch ====================
if exist "%~dp0python_embeded\python.exe" (
    echo [INFO] Using embedded Python...
    "%~dp0python_embeded\python.exe" "%~dp0acestep\acestep_v15_pipeline.py" !FULL_ARGS!
) else (
    echo [INFO] Using uv / system Python...
    uv run acestep !FULL_ARGS!
)

pause
endlocal
goto :eof

REM ================================================================
:LoadEnvFile
REM Reads .env from the same directory as this .bat file.
REM Uses for /f re-tokenization on every value to strip trailing \r
REM that CRLF .env files inject, which corrupts arguments silently.
REM ================================================================
set "ENV_FILE=%~dp0.env"
if not exist "%ENV_FILE%" (
    echo [WARN] .env file not found at: %ENV_FILE%
    exit /b 0
)

echo [Config] Loading configuration from .env file...

for /f "usebackq tokens=1,* delims==" %%K in ("%ENV_FILE%") do (
    if not "%%K"=="" (
        set "_k=%%K"
        if not "!_k:~0,1!"=="#" (

            REM Strip \r from value by re-tokenizing through for /f
            set "_v=%%L"
            if defined _v (
                for /f "tokens=* delims= " %%S in ("!_v!") do set "_v=%%S"
            )

            if /i "!_k!"=="ACESTEP_CONFIG_PATH"   if defined _v set "CONFIG_PATH=--config_path !_v!"
            if /i "!_k!"=="ACESTEP_LM_MODEL_PATH" if defined _v set "LM_MODEL_PATH=--lm_model_path !_v!"
            if /i "!_k!"=="ACESTEP_AUTH_USERNAME"  if defined _v set "AUTH_USERNAME=--auth-username !_v!"
            if /i "!_k!"=="ACESTEP_AUTH_PASSWORD"  if defined _v set "AUTH_PASSWORD=--auth-password !_v!"
            if /i "!_k!"=="ACESTEP_INIT_LLM"       if defined _v set "INIT_LLM=--init_llm !_v!"
            if /i "!_k!"=="ACESTEP_BATCH_SIZE"     if defined _v set "BATCH_SIZE=--batch_size !_v!"
            if /i "!_k!"=="ACESTEP_INIT_SERVICE"   if defined _v set "INIT_SERVICE=--init_service !_v!"
            if /i "!_k!"=="ACESTEP_LM_BACKEND"    if defined _v set "LM_BACKEND=--backend !_v!"
            if /i "!_k!"=="PORT"        if defined _v set "PORT=!_v!"
            if /i "!_k!"=="SERVER_NAME" if defined _v set "SERVER_NAME=!_v!"
            if /i "!_k!"=="LANGUAGE"    if defined _v set "LANGUAGE=!_v!"

            REM PyTorch env var - set directly into process environment (not a CLI arg)
            if /i "!_k!"=="PYTORCH_CUDA_ALLOC_CONF" if defined _v set "PYTORCH_CUDA_ALLOC_CONF=!_v!"
        )
    )
)

echo [Config] Configuration loaded.
exit /b 0
