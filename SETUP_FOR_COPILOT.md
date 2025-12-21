# Setup Guide for GitHub Copilot

**Perfect setup for using MCP Docs RAG with GitHub Copilot - NO OpenAI API key needed!**

---

## Why This Works Great with Copilot

✅ **GitHub Copilot** is your LLM (no need for OpenAI GPT)
✅ **Local HuggingFace embeddings** for semantic search (free!)
✅ **3000+ .md files?** No problem - only search results sent to Copilot
✅ **Total cost:** $0

---

## Quick Setup (5 Minutes)

### 1. Install Dependencies (2 min)

```bash
cd mcp-docs-rag
uv sync
```

This installs:
- `sentence-transformers` (local embeddings, FREE)
- `docling` (document processing)
- `mcp` (Model Context Protocol)

### 2. Configure Environment (1 min)

```bash
cp .env.example .env
```

Edit `.env`:
```bash
# REQUIRED
DOCS_FOLDER=C:/path/to/your/team/docs

# EMBEDDING (FREE!)
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2

# SEARCH
ENABLE_SEMANTIC_SEARCH=true
ENABLE_IMAGE_UNDERSTANDING=false

# NO API KEYS NEEDED!
```

### 3. Build Index (2 min)

```bash
# Build index with local embeddings
uv run python -m indexer.build_index --mode full
```

**What happens:**
- Scans all 3000 .md files
- Extracts metadata (titles, headings, content)
- Generates embeddings locally (no API calls!)
- Saves to `.index/` folder

**Time:** ~1-2 seconds per file = 50-100 minutes for 3000 files
**Cost:** $0

### 4. Configure VS Code (30 sec)

Add to VS Code `settings.json`:

```json
{
  "mcp.servers": {
    "docs-rag": {
      "command": "uv",
      "args": [
        "run",
        "python",
        "C:/path/to/mcp-docs-rag/server.py"
      ],
      "env": {
        "DOCS_FOLDER": "C:/path/to/your/docs",
        "EMBEDDING_PROVIDER": "local",
        "ENABLE_SEMANTIC_SEARCH": "true"
      }
    }
  }
}
```

### 5. Test It! (30 sec)

In VS Code with Copilot:

```
You: @docs-rag search for "windows vm"

Copilot: Found 3 documentation matches for 'windows vm':

1. **Windows VM Setup Guide**
   Path: `platform-team/iaas/windows-vm.md`
   Score: 95.0 (exact match)

   [Uses this context to answer your question]
```

---

## How It Works (No Context Overflow!)

### Architecture

```
3000 .md files in folders
        ↓
[One-time indexing] → .index/ (stored locally)
        ↓
User asks: "how to deploy?"
        ↓
[MCP Server]
  - Searches index (fast, local)
  - Finds top 5 matches
        ↓
Returns 5 results to Copilot (small payload!)
        ↓
[GitHub Copilot]
  - Receives search results
  - Generates answer using GitHub's LLM
```

**Key:** Only search results (5-10 docs) are sent to Copilot, NOT all 3000 files!

### Example

```
You: @docs-rag how do I create a Windows VM?

MCP Server: [Searches 3000 files in 50ms]
            Finds: windows-vm.md, azure-vm.md, vm-troubleshooting.md

            Returns to Copilot:
            "Found 3 docs:
            1. windows-vm.md - Complete guide...
            2. azure-vm.md - Azure specific...
            3. vm-troubleshooting.md - Common issues..."

GitHub Copilot: [Receives 3 docs, ~2KB total]
                "To create a Windows VM, follow these steps:
                1. Run az group create...
                2. Configure VM size...
                Based on: windows-vm.md"

You: Perfect! No context overflow even with 3000 files!
```

---

## Performance with 3000 Files

| Operation | Time | Details |
|-----------|------|---------|
| Initial indexing | 50-100 min | One-time (can run overnight) |
| Keyword search | 10-50ms | Using ripgrep |
| Semantic search | 200-500ms | Local embeddings |
| Index size | ~50-100MB | Metadata + embeddings |
| Context sent to Copilot | ~2-10KB | Only top results |

**No context overflow because:**
- Index is stored locally (not sent to Copilot)
- Only search results (5-10 docs) are sent
- Each search result is ~500-1000 tokens
- Total per query: ~5000 tokens max

---

## Embedding Options

### Option 1: Fast & Free (Recommended)

```bash
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

- **Speed:** Fast (even on CPU)
- **Dimension:** 384
- **Quality:** Good for most cases
- **Cost:** $0

### Option 2: Better Quality (Still Free!)

```bash
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-mpnet-base-v2
```

- **Speed:** Slower (but still local)
- **Dimension:** 768
- **Quality:** Higher quality
- **Cost:** $0

### Option 3: Multilingual (Free!)

```bash
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
```

- **Languages:** 50+ languages
- **Dimension:** 384
- **Use case:** Mixed language docs
- **Cost:** $0

---

## Example Queries

### Keyword Search (Fastest)

```
@docs-rag search for "pipeline config"
→ Returns: All docs mentioning "pipeline" and "config"
→ Time: 10-50ms
```

### Semantic Search

```
@docs-rag how do I deploy containerized applications?
→ Returns: Docs about Docker, Kubernetes, even if they don't say "deploy"
→ Time: 200-500ms
```

### Folder-Specific

```
@docs-rag search for "vm" in folder "platform-team/iaas"
→ Returns: Only VMs from iaas folder
→ Time: 10-50ms
```

### Get Full Doc

```
@docs-rag get file "platform-team/ci-cd/pipelines.md"
→ Returns: Complete file content
→ Time: <10ms
```

### List Structure

```
@docs-rag list documentation structure
→ Returns: Folder tree with all docs
→ Time: <10ms
```

---

## Troubleshooting

### Index is slow to build

**Solution 1:** Use faster embedding model
```bash
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Faster
```

**Solution 2:** Disable semantic search for now
```bash
ENABLE_SEMANTIC_SEARCH=false
uv run python -m indexer.build_index --mode fast
```
(Builds in ~3-5 minutes for 3000 files)

### Search returns no results

**Check:**
1. Index was built: `ls .index/`
2. Docs folder is correct: `ls $DOCS_FOLDER`
3. Rebuild index: `uv run python -m indexer.build_index`

### Copilot not using results

**Check:**
1. MCP server is running: `uv run python server.py`
2. VS Code settings.json has correct path
3. Try restarting VS Code

### Out of memory during indexing

**Solution:** Process in batches
```bash
# Index one team at a time
DOCS_FOLDER=/path/to/platform-team uv run python -m indexer.build_index
DOCS_FOLDER=/path/to/backend-team uv run python -m indexer.build_index
# Then merge indexes (script coming soon)
```

---

## Advanced: GPU Acceleration (Optional)

If you have an NVIDIA GPU, speed up embeddings:

```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# sentence-transformers will auto-use GPU
```

**Speedup:** ~5-10x faster indexing!

---

## Comparison: Local vs API Embeddings

| Feature | Local HF | HF API | OpenAI |
|---------|----------|--------|--------|
| **Cost** | $0 | $0 (free tier) | ~$0.02/1M tokens |
| **Speed (indexing)** | Medium | Fast | Fast |
| **Speed (search)** | Fast | Medium | Medium |
| **Privacy** | ✅ Local | ❌ API call | ❌ API call |
| **Offline** | ✅ Yes | ❌ No | ❌ No |
| **Quality** | Good | Good | Excellent |

**Recommendation for 3000 files:** Local HuggingFace - free, private, good quality!

---

## Summary: Your Perfect Setup

```bash
# .env
DOCS_FOLDER=/path/to/3000-files
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2
ENABLE_SEMANTIC_SEARCH=true

# No API keys needed!
# Works with GitHub Copilot!
# Handles 3000+ files!
# $0 cost!
```

**Total setup time:** 5 minutes + indexing time
**Total cost:** $0
**Context overflow:** Never (only results sent to Copilot)

---

## Next Steps

1. ✅ Build index: `uv run python -m indexer.build_index --mode full`
2. ✅ Start server: `uv run python server.py`
3. ✅ Test in VS Code: `@docs-rag search for "test"`
4. ✅ Read [USAGE.md](USAGE.md) for all features

**Questions?** Check [README.md](README.md) or open an issue!

Happy searching with Copilot! 🚀✨
