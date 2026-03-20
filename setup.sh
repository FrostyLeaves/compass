#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- Colors ----------
green()  { printf "\033[32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }
red()    { printf "\033[31m%s\033[0m\n" "$*"; }

# ---------- 1. Python ----------
green "==> [1/3] Checking Python ..."

PYTHON=""
for cmd in python3.12 python3.13 python3; do
    if command -v "$cmd" &>/dev/null; then
        major=$("$cmd" -c "import sys; print(sys.version_info[0])")
        minor=$("$cmd" -c "import sys; print(sys.version_info[1])")
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON="$(command -v "$cmd")"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    red "Error: Python >= 3.10 is required but not found."
    red "Install via: brew install python@3.12"
    exit 1
fi
green "    Using $PYTHON ($($PYTHON --version))"

# ---------- 2. Virtual environment + dependencies ----------
green "==> [2/3] Setting up virtual environment ..."

if [ ! -d ".venv" ]; then
    "$PYTHON" -m venv .venv
    green "    Created .venv"
else
    yellow "    .venv already exists, skipping creation"
fi

source .venv/bin/activate
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q
green "    Dependencies installed"

# ---------- 3. Ollama ----------
green "==> [3/3] Checking Ollama ..."

if ! command -v ollama &>/dev/null; then
    yellow "    Ollama is not installed."
    read -rp "    Install Ollama now? [Y/n] " ans
    ans="${ans:-Y}"
    if [[ "$ans" =~ ^[Yy]$ ]]; then
        if [[ "$(uname)" == "Darwin" ]]; then
            brew install ollama
        else
            curl -fsSL https://ollama.com/install.sh | sh
        fi
    else
        yellow "    Skipped Ollama installation. You can install it later from https://ollama.com"
    fi
fi

if command -v ollama &>/dev/null; then
    EMB_MODEL=$(python -c "
import yaml
cfg = yaml.safe_load(open('config.yaml'))
if cfg.get('embedding',{}).get('provider') == 'ollama':
    print(cfg['embedding']['model'])
" 2>/dev/null)

    LLM_MODEL=$(python -c "
import yaml
cfg = yaml.safe_load(open('config.yaml'))
if cfg.get('llm',{}).get('provider') == 'ollama':
    print(cfg['llm']['model'])
" 2>/dev/null)

    pull_model() {
        local model="$1" role="$2"
        if [ -n "$model" ]; then
            if ollama list 2>/dev/null | grep -q "^$model"; then
                yellow "    $role: $model (already pulled)"
            else
                green "    Pulling $role: $model ..."
                ollama pull "$model"
            fi
        fi
    }

    pull_model "$EMB_MODEL" "embedding"
    pull_model "$LLM_MODEL" "llm"
fi

# ---------- Done ----------
echo ""
green "Setup complete!"
echo ""
echo "  Activate the environment:  source .venv/bin/activate"
echo "  Quick start:"
echo "    python cli.py ingest <pdf>     Import a paper"
echo "    python cli.py ask <question>   Ask a question"
echo "    ./start.sh                     Start web servers"
echo ""
