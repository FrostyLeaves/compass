# Compass

Compass is a local paper knowledge base for PDF ingestion, retrieval, and question answering.

It combines:

- a Python CLI for importing papers and running search / Q&A
- a FastAPI backend
- a React web UI
- an MCP server for tool-based access
- a local Qdrant-backed vector store

The project is designed for working with academic PDFs end to end: download or import papers, convert them to markdown, chunk and embed them, store them locally, and query them later through search or chat.

## What This Project Does

Compass supports these workflows:

- ingest a local PDF, a directory of PDFs, or a remote PDF URL
- parse PDFs into markdown with `marker-pdf`
- generate embeddings and store chunks in Qdrant
- run semantic search over the paper collection
- answer questions grounded in retrieved paper content
- browse papers in a web interface
- expose read-only MCP tools for external clients

When the API server is running, the CLI automatically uses the web API for write operations. This avoids file-lock conflicts with the local Qdrant database.

## Requirements

- Python 3.10+
- Node.js and npm
- Optional but recommended: Ollama for local embedding / LLM usage

## Initialization

### macOS / Linux

Run the setup script:

```bash
./setup.sh
```

This script will:

- create `.venv` if needed
- install Python dependencies from `requirements.txt`
- install frontend dependencies in `web/`
- optionally install / prepare Ollama models based on `config.yaml`

Then activate the virtual environment:

```bash
source .venv/bin/activate
```

### Windows

Run:

```bat
setup.bat
```

Then activate the environment:

```bat
.venv\Scripts\activate
```

## Manual Setup

If you do not want to use the setup script:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cd web && npm install
```

## Configuration

Main configuration lives in `config.yaml`.

Important sections:

- `embedding`
- `llm`
- `storage`
- `ingest`
- `retrieval`
- `api`

By default, local data is stored under:

- `./data/qdrant`
- `./data/papers`

## Quick Start

Import a paper:

```bash
python cli.py ingest <pdf-url>
```

Ask a question:

```bash
python cli.py ask "What problem does this paper solve?"
```

Search the local paper base:

```bash
python cli.py search "transformer attention"
```

## Running the App

Start backend and frontend together:

```bash
./start.sh
```

This starts:

- FastAPI backend on the host / port defined in `config.yaml`
- Vite frontend on `http://localhost:5173`

You can also start them separately:

```bash
source .venv/bin/activate
uvicorn api:app --reload --host localhost --port 8000
```

```bash
cd web
npm run dev
```

## CLI Commands

Examples:

```bash
python cli.py ingest <pdf-path-or-url>
python cli.py batch <url-list.txt>
python cli.py list
python cli.py search "<query>"
python cli.py ask "<question>"
python cli.py remove <paper_id>
python cli.py reindex [paper_id]
python cli.py audit
```

## MCP

The project also includes an MCP server in `server.py`.

Run it in stdio mode with:

```bash
source .venv/bin/activate
python server.py
```

When the FastAPI app is running, MCP is also mounted over HTTP at `/mcp`.

## Notes

- The root also contains `app.py`, which provides a Streamlit-based interface.
- The frontend lives in `web/`.
- The backend entrypoint is `api.py`.
