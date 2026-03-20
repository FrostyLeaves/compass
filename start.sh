#!/usr/bin/env bash
# Start Compass backend + frontend dev servers

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
source .venv/bin/activate

trap 'kill 0' EXIT

API_HOST=$(python -c "import yaml; cfg=yaml.safe_load(open('config.yaml')); print(cfg.get('api',{}).get('host','localhost'))")
API_PORT=$(python -c "import yaml; cfg=yaml.safe_load(open('config.yaml')); print(cfg.get('api',{}).get('port',8000))")
echo "Starting backend on ${API_HOST}:${API_PORT} ..."
uvicorn api:app --reload --host "$API_HOST" --port "$API_PORT" &

echo "Starting frontend on :5173 ..."
cd web && npm run dev &

wait
