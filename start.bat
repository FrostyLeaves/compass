@echo off
cd /d "%~dp0"

set "VIRTUAL_ENV=%~dp0.venv"
set "PATH=%VIRTUAL_ENV%\Scripts;%PATH%"

for /f "tokens=*" %%h in ('python -c "import yaml; cfg=yaml.safe_load(open('config.yaml',encoding='utf-8')); print(cfg.get('api',{}).get('host','localhost'))"') do set API_HOST=%%h
for /f "tokens=*" %%p in ('python -c "import yaml; cfg=yaml.safe_load(open('config.yaml',encoding='utf-8')); print(cfg.get('api',{}).get('port',8000))"') do set API_PORT=%%p
echo Starting backend on %API_HOST%:%API_PORT% ...
start "Compass-Backend" cmd /k "cd /d %~dp0 && set "VIRTUAL_ENV=%~dp0.venv" && set "PATH=%~dp0.venv\Scripts;%PATH%" && python -m uvicorn api:app --reload --host %API_HOST% --port %API_PORT%"

echo Starting frontend on :5173 ...
start "Compass-Frontend" cmd /k "cd /d %~dp0\web && npm run dev"
