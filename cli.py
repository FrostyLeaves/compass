"""Compass — Paper Search Engine CLI entry."""

import argparse
import sys
import io
import time
from pathlib import Path

# Fix Windows console encoding for Unicode output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Ensure core is importable
sys.path.insert(0, str(Path(__file__).parent))

_STAGE_LABELS = {
    "downloading": "Downloading PDF",
    "hashing": "Hashing file",
    "dedup": "Checking duplicates",
    "skipped": "Already ingested",
    "parsing": "Parsing PDF",
    "cleaning": "Cleaning markdown",
    "keywords": "Extracting keywords",
    "detecting_lang": "Detecting language",
    "translating": "Translating",
    "embedding": "Embedding",
    "storing": "Storing",
}


def _cli_progress(stage, current, total):
    """Shared progress callback for all CLI commands."""
    label = _STAGE_LABELS.get(stage, stage)
    if stage == "downloading" and total > 1:
        mb_done = current / (1024 * 1024)
        mb_total = total / (1024 * 1024)
        print(f"\r  {label}: {mb_done:.1f}/{mb_total:.1f} MB", end="", flush=True)
        if current >= total:
            print()
    elif total > 1:
        print(f"\r  {label}: {current}/{total}", end="", flush=True)
        if current >= total:
            print()
    else:
        status = "done" if current >= total else "..."
        print(f"\r  {label}: {status}    ", end="", flush=True)
        if current >= total:
            print()


def _elapsed(t0):
    """Format elapsed seconds as human-readable string."""
    s = time.time() - t0
    if s < 60:
        return f"{s:.1f}s"
    return f"{int(s // 60)}m{int(s % 60)}s"


def _print_mode():
    """Print a note when using API server mode."""
    from core.client import is_server_running
    if is_server_running():
        print("[via API server]")


def _ingest_kwargs(args):
    """Build common kwargs for ingest/batch from parsed args."""
    from core.client import is_server_running
    kwargs = {}
    if not is_server_running():
        if args.embedding_provider:
            kwargs["embedding_provider"] = args.embedding_provider
        if args.embedding_model:
            kwargs["embedding_model"] = args.embedding_model
    return kwargs


def cmd_ingest(args):
    from core.client import ingest_paper

    target = args.path
    kwargs = _ingest_kwargs(args)

    _print_mode()
    print(f"Ingesting: {target}")
    t0 = time.time()
    try:
        results = ingest_paper(target, progress_callback=_cli_progress, **kwargs)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    for result in results:
        if result.get("error"):
            print(f"  FAIL: {result.get('title', target)}: {result['error']}")
        else:
            print(f"  OK: {result['title']} ({result['chunks_count']} chunks)")
            if result.get("source_url"):
                print(f"      Source URL: {result['source_url']}")

    total = sum(r.get("chunks_count", 0) for r in results)
    if len(results) > 1:
        print(f"\nDone: {len(results)} papers, {total} chunks total ({_elapsed(t0)})")
    else:
        print(f"  Finished in {_elapsed(t0)}")


def cmd_batch(args):
    from core.client import ingest_paper

    txt_path = Path(args.file)
    if not txt_path.exists():
        print(f"Error: {txt_path} not found")
        sys.exit(1)

    lines = txt_path.read_text(encoding="utf-8").splitlines()
    urls = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    if not urls:
        print("No URLs found in file.")
        return

    kwargs = _ingest_kwargs(args)

    _print_mode()
    print(f"Batch import: {len(urls)} URLs from {txt_path.name}\n")

    ok_count = 0
    fail_count = 0
    total_chunks = 0
    t0_all = time.time()

    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] {url}")
        t0 = time.time()
        try:
            results = ingest_paper(url, progress_callback=_cli_progress, **kwargs)
            for result in results:
                if result.get("error"):
                    print(f"  FAIL: {result.get('title', url)}: {result['error']}")
                    fail_count += 1
                else:
                    print(f"  OK: {result['title']} ({result['chunks_count']} chunks, {_elapsed(t0)})")
                    ok_count += 1
                    total_chunks += result.get("chunks_count", 0)
        except Exception as e:
            print(f"  FAIL: {e}")
            fail_count += 1

    print(f"\nDone: {ok_count} succeeded, {fail_count} failed, {total_chunks} chunks total ({_elapsed(t0_all)})")


def cmd_search(args):
    from core.client import is_server_running, search

    kwargs = {}
    if not is_server_running():
        if args.embedding_provider:
            kwargs["embedding_provider"] = args.embedding_provider
        if args.embedding_model:
            kwargs["embedding_model"] = args.embedding_model

    _print_mode()
    print(f"Searching: {args.query}")
    t0 = time.time()
    results = search(args.query, top_k=args.top_k, **kwargs)
    print(f"  Found {len(results)} results ({_elapsed(t0)})\n")

    if not results:
        return

    for i, item in enumerate(results, 1):
        print(f"{'='*60}")
        print(f"[{i}] {item['title']}  (score: {item['score']:.3f})")
        print(f"    Source: {item.get('source_path', '')}")
        print(f"    Chunk: #{item['chunk_index']}")
        print(f"{'-'*60}")
        text = item["text"][:300]
        if len(item["text"]) > 300:
            text += "..."
        print(text)


def cmd_ask(args):
    from core.client import is_server_running, ask

    kwargs = {}
    if not is_server_running():
        if args.embedding_provider:
            kwargs["embedding_provider"] = args.embedding_provider
        if args.embedding_model:
            kwargs["embedding_model"] = args.embedding_model
        if args.provider:
            kwargs["provider"] = args.provider
        if args.model:
            kwargs["model"] = args.model

    _print_mode()
    print(f"Question: {args.question}")
    print("  Searching...", end="", flush=True)
    t0 = time.time()
    answer, sources = ask(args.question, top_k=args.top_k, **kwargs)
    t_total = _elapsed(t0)

    if not sources:
        print(f" no results ({t_total})")
        print("No relevant papers found in Compass.")
        return

    print(f" {len(sources)} papers found, answer generated ({t_total})\n")
    print(answer)
    print(f"\n{'='*60}")
    print(f"Sources ({len(sources)}):")
    for s in sources:
        score = f"  score: {s['score']:.3f}" if 'score' in s else ""
        print(f"  - {s['title']}{score}")


def cmd_list(args):
    from core.client import list_papers

    _print_mode()
    print("Loading papers...", end="", flush=True)
    papers = list_papers()
    if not papers:
        print(" empty")
        print("Knowledge base is empty.")
        return

    print(f" {len(papers)} papers\n")
    for p in papers:
        chunks = p.get("chunks_count", 0)
        kw = p.get("keywords", [])
        print(f"  [{chunks:3d} chunks] {p['title']}")
        if kw:
            print(f"             Keywords: {', '.join(kw)}")
        print(f"             {p.get('path', '')}")


def cmd_remove(args):
    from core.client import remove

    _print_mode()
    print(f"Removing: {args.paper_id}...", end="", flush=True)
    if remove(args.paper_id):
        print(" done")
    else:
        print(" not found")


def cmd_reindex(args):
    from core.client import is_server_running, reindex

    kwargs = {}
    if not is_server_running():
        if args.embedding_provider:
            kwargs["embedding_provider"] = args.embedding_provider
        if args.embedding_model:
            kwargs["embedding_model"] = args.embedding_model

    _print_mode()
    paper_id = getattr(args, "paper_id", None)
    print(f"Reindexing: {paper_id or 'all papers'}...")
    t0 = time.time()

    if not is_server_running():
        kwargs["progress_callback"] = _cli_progress

    try:
        results = reindex(paper_id=paper_id, **kwargs)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    for result in results:
        if result.get("error"):
            print(f"  FAIL: {result.get('title', '?')}: {result['error']}")
        else:
            kw = result.get("keywords", [])
            print(f"  OK: {result['title']} ({result['chunks_count']} chunks, {len(kw)} keywords)")
            if kw:
                print(f"      Keywords: {', '.join(kw)}")

    if len(results) > 1:
        ok = sum(1 for r in results if not r.get("error"))
        total_chunks = sum(r.get("chunks_count", 0) for r in results)
        print(f"\nDone: {ok}/{len(results)} papers reindexed, {total_chunks} chunks total ({_elapsed(t0)})")
    else:
        print(f"  Finished in {_elapsed(t0)}")


def cmd_audit(args):
    from core.client import is_server_running, audit

    def _progress(stem, translating=None, removed=None, title_fixed=None):
        if title_fixed:
            print(f"  [{stem}] Fixed title -> {title_fixed}")
        elif translating:
            print(f"  [{stem}] Translating to {translating}...")
        elif removed:
            print(f"  [{stem}] Removed {removed}")
        else:
            print(f"  Scanning: {stem}")

    _print_mode()
    print("Auditing papers...")
    t0 = time.time()
    kwargs = {}
    if not is_server_running():
        kwargs["progress_callback"] = _progress
    result = audit(**kwargs)
    print(f"\nDone: {result['papers_scanned']} papers scanned, "
          f"{result['titles_fixed']} titles fixed, "
          f"{result['translations_added']} translations added, "
          f"{result['translations_removed']} translations removed "
          f"({_elapsed(t0)})")


def main():
    parser = argparse.ArgumentParser(description="Compass — Paper Search Engine CLI")
    parser.add_argument("--embedding-provider", default=None)
    parser.add_argument("--embedding-model", default=None)

    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Import PDF(s) into Compass")
    p_ingest.add_argument("path", help="PDF file path, directory, or https:// URL")

    # batch
    p_batch = sub.add_parser("batch", help="Batch import PDFs from a text file (one URL per line)")
    p_batch.add_argument("file", help="Text file with one PDF URL per line")

    # search
    p_search = sub.add_parser("search", help="Search for relevant paper chunks")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--top-k", type=int, default=5)

    # ask
    p_ask = sub.add_parser("ask", help="Ask a question (search + LLM answer)")
    p_ask.add_argument("question", help="Your question")
    p_ask.add_argument("--top-k", type=int, default=5)
    p_ask.add_argument("--provider", default=None, help="LLM provider: ollama/claude/openai/codex")
    p_ask.add_argument("--model", default=None, help="LLM model name")

    # list
    sub.add_parser("list", help="List all ingested papers")

    # remove
    p_remove = sub.add_parser("remove", help="Remove a paper from Compass")
    p_remove.add_argument("paper_id", help="Paper id")

    # reindex
    p_reindex = sub.add_parser("reindex", help="Re-embed and regenerate keywords for papers (skip PDF parsing)")
    p_reindex.add_argument("paper_id", nargs="?", default=None, help="Paper id (omit for all papers)")

    # audit
    sub.add_parser("audit", help="Audit and fix translation files for all papers")

    args = parser.parse_args()
    commands = {
        "ingest": cmd_ingest,
        "batch": cmd_batch,
        "search": cmd_search,
        "ask": cmd_ask,
        "list": cmd_list,
        "remove": cmd_remove,
        "reindex": cmd_reindex,
        "audit": cmd_audit,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
