@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo [1/3] Checking Python ...
where python >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH.
    echo Install from https://www.python.org/downloads/
    exit /b 1
)

for /f "tokens=*" %%v in ('python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"') do set PYVER=%%v
echo     Using Python %PYVER%

REM ---------- 2. Virtual environment + dependencies ----------
echo [2/3] Setting up virtual environment ...

if not exist ".venv" (
    python -m venv .venv
    echo     Created .venv
) else (
    echo     .venv already exists, skipping creation
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q
echo     Dependencies installed

REM ---------- 3. Ollama ----------
echo [3/3] Checking Ollama ...

where ollama >nul 2>&1
if errorlevel 1 (
    echo     Ollama is not installed. Download from https://ollama.com
    echo     Skipping model pull.
    goto :done
)

for /f "tokens=*" %%m in ('.venv\Scripts\python.exe -c "import yaml; cfg=yaml.safe_load(open('config.yaml',encoding='utf-8')); print(cfg['embedding']['model'] if cfg.get('embedding',{}).get('provider')=='ollama' else '')"') do set EMB_MODEL=%%m
for /f "tokens=*" %%m in ('.venv\Scripts\python.exe -c "import yaml; cfg=yaml.safe_load(open('config.yaml',encoding='utf-8')); print(cfg['llm']['model'] if cfg.get('llm',{}).get('provider')=='ollama' else '')"') do set LLM_MODEL=%%m

if not "%EMB_MODEL%"=="" (
    echo     Pulling embedding model: %EMB_MODEL% ...
    ollama pull %EMB_MODEL%
)
if not "%LLM_MODEL%"=="" (
    echo     Pulling LLM model: %LLM_MODEL% ...
    ollama pull %LLM_MODEL%
)

:done
echo.
echo Setup complete!
echo.
echo   Activate the environment:  .venv\Scripts\activate.bat
echo   Quick start:
echo     python cli.py ingest ^<pdf^>     Import a paper
echo     python cli.py ask ^<question^>   Ask a question
echo     start.bat                       Start web servers
echo.
