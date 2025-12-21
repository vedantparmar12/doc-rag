# Quick Start Guide

Get your MCP Docs RAG server running in 5 minutes!

## Step 1: Install Dependencies (1 min)

```bash
cd mcp-docs-rag
uv sync
```

## Step 2: Configure Environment (1 min)

```bash
# Copy example env
cp .env.example .env

# Edit .env file
# Required: Set DOCS_FOLDER and OPENAI_API_KEY
```

Example `.env`:
```bash
DOCS_FOLDER=C:/Users/yourname/company-docs
OPENAI_API_KEY=sk-your-key-here
ENABLE_SEMANTIC_SEARCH=true
```

## Step 3: Build Index (1-2 min)

```bash
# Fast index (metadata only, instant search)
uv run python -m indexer.build_index --mode fast

# OR Full index (with semantic search, slower)
uv run python -m indexer.build_index --mode full
```

## Step 4: Test the Server (30 sec)

```bash
# Start server in test mode
uv run python server.py
```

Keep this running in one terminal.

## Step 5: Connect to VS Code (1 min)

Add to VS Code `settings.json` (Cmd/Ctrl + Shift + P → "Open Settings (JSON)"):

```json
{
  "mcp.servers": {
    "docs-rag": {
      "command": "uv",
      "args": ["run", "python", "C:/path/to/mcp-docs-rag/server.py"],
      "env": {
        "DOCS_FOLDER": "C:/path/to/your/docs",
        "OPENAI_API_KEY": "${env:OPENAI_API_KEY}"
      }
    }
  }
}
```

## Step 6: Try It! (30 sec)

In VS Code:
```
@docs-rag search for "windows vm"
```

You should see search results!

---

## Example Queries to Try

```
@docs-rag What pipelines do we have for deployment?
@docs-rag Show me all Kubernetes configs
@docs-rag How do I set up a Windows VM?
@docs-rag List the documentation structure
@docs-rag Summarize our CI/CD docs
```

---

## Troubleshooting

**Server won't start?**
- Check `.env` has correct paths
- Verify DOCS_FOLDER exists and contains .md files

**No results found?**
- Rebuild index: `uv run python -m indexer.build_index`
- Check docs are in correct folder structure

**Search is slow?**
- Install ripgrep: `choco install ripgrep` (Windows) or `brew install ripgrep` (Mac)
- Disable semantic search: `export ENABLE_SEMANTIC_SEARCH=false`

---

## Next Steps

- Read the full [README.md](README.md) for all features
- Check [config/](config/) for more examples
- Customize search scoring in `search/hybrid_search.py`

Happy searching! 🔍