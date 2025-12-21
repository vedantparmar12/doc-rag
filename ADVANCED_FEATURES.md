# Advanced Features: Hybrid Chunking & Reranking

## Overview

Your MCP Docs RAG server now includes TWO powerful features for **better search quality**:

1. **Advanced Hybrid Chunking** - Smarter document splitting
2. **Reranking** - Better result ordering

Both are **FREE** and run **locally**!

---

## 1. Advanced Hybrid Chunking

### What It Does

Instead of naively splitting documents by character count, hybrid chunking:

✅ **Preserves document structure** (headings, paragraphs, code blocks)
✅ **Maintains heading hierarchy** in chunks
✅ **Respects semantic boundaries** (doesn't split mid-paragraph)
✅ **Token-aware** (uses actual tokenizer, not estimates)
✅ **Better for RAG** (more coherent chunks)

### How It Works

```python
# Without hybrid chunking (BAD):
"# Windows VM Setup\n## Prerequisites\nYou need Azure..."
→ Split at 500 chars
→ "# Windows VM Setup\n## Prerequisites\nYou ne"  ❌ Broken!

# With hybrid chunking (GOOD):
"# Windows VM Setup\n## Prerequisites\nYou need Azure..."
→ Split at section boundaries
→ Chunk 1: "# Windows VM Setup\n## Prerequisites\nYou need Azure..."  ✅
→ Chunk 2: "## Step 1: Create Resource Group\n..."  ✅
```

### Example Output

Each chunk includes heading hierarchy:

```json
{
  "content": "To create a Windows VM, first configure...",
  "heading_hierarchy": [
    "Windows VM Setup",
    "Prerequisites",
    "Azure Subscription"
  ],
  "chunk_index": 2,
  "token_count": 487,
  "file_path": "platform-team/iaas/windows-vm.md"
}
```

### Why It Matters for Your 3000 Files

**Problem:** Naive chunking breaks context
```
Query: "How to troubleshoot Windows VM boot issues?"
Bad chunk: "...issues. Checkmemory settings..." ❌ Split mid-sentence
```

**Solution:** Hybrid chunking preserves context
```
Query: "How to troubleshoot Windows VM boot issues?"
Good chunk: "## Troubleshooting Boot Issues\nCheck memory settings..." ✅ Complete section
```

### Configuration

```bash
# .env
CHUNK_MAX_TOKENS=512          # Max tokens per chunk
CHUNK_WITH_HEADINGS=true      # Include heading hierarchy
CHUNK_MERGE_PEERS=true        # Merge small adjacent chunks
```

---

## 2. Reranking

### What It Does

Reranking improves search results in 2 steps:

1. **Fast Retrieval:** Get top 50 candidates (fast but rough)
2. **Reranking:** Score each candidate precisely (slower but accurate)

**Result:** Better top 10 results!

### How It Works

```
Query: "How to deploy containerized applications?"

Step 1: Fast Retrieval (embeddings)
  → 50 candidates in 200ms
  → Good recall, but order may be imperfect

Step 2: Reranking (cross-encoder)
  → Score each of 50 candidates
  → Takes 500ms but very accurate
  → Return top 10 best matches

Total: 700ms for MUCH better results!
```

### With vs Without Reranking

**Without Reranking:**
```
Query: "windows vm troubleshooting"

Results:
1. windows-vm.md (score: 0.85)
2. linux-vm.md (score: 0.82) ← Wrong OS!
3. vm-networking.md (score: 0.80)
4. vm-troubleshooting.md (score: 0.78) ← Should be #1!
5. azure-vm.md (score: 0.75)
```

**With Reranking:**
```
Query: "windows vm troubleshooting"

Results (reranked):
1. vm-troubleshooting.md (score: 0.94) ✅ Perfect!
2. windows-vm.md (score: 0.91) ✅ Also great!
3. azure-vm.md (score: 0.85)
4. vm-networking.md (score: 0.72)
5. linux-vm.md (score: 0.58) ← Pushed down correctly
```

### Why It Helps

**Embeddings (bi-encoder):**
- Fast but imprecise
- Encodes query and docs separately
- Can mis-rank results

**Reranker (cross-encoder):**
- Slower but very accurate
- Scores query + document pair directly
- Better understands relevance

### Performance

| Operation | Time | Quality |
|-----------|------|---------|
| **Retrieval only** | 200ms | Good |
| **Retrieval + Reranking** | 700ms | Excellent |

**For 3000 files:** Still fast enough!

### Configuration

```bash
# .env
ENABLE_RERANKING=true  # Recommended for better quality!
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2  # Fast

# Or for even better quality (slower):
# RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
```

---

## Do You Need Reranking?

### When Reranking Helps Most

✅ **Conceptual queries:** "how to deploy", "best practices"
✅ **Ambiguous queries:** "vm issues", "pipeline config"
✅ **Multi-term queries:** "windows vm deployment troubleshooting"
✅ **3000+ documents:** More candidates to rerank

### When It's Less Critical

❌ **Exact keyword searches:** "windows-vm.md"
❌ **Very specific queries:** "ERROR_CODE_12345"
❌ **< 100 documents:** Less noise to filter

### Recommendation

**For your 3000 .md files:** **YES, enable reranking!**

**Why:**
- More documents = more potential false positives
- Reranking fixes the ordering
- 700ms is still fast for better quality
- FREE - runs locally

---

## Combined Power: Chunking + Reranking

### The Full Pipeline

```
3000 .md files
     ↓
[Advanced Hybrid Chunking]
  - Split by sections
  - Preserve headings
  - Token-aware
     ↓
~15,000 chunks (5 per file average)
     ↓
[Build Index with Embeddings]
     ↓
User query: "troubleshoot windows vm"
     ↓
[Fast Retrieval]
  - Get top 50 chunks (200ms)
     ↓
[Reranking]
  - Score all 50 precisely (500ms)
  - Return best 10
     ↓
GitHub Copilot
  - Receives top 10 highest quality chunks
  - Generates excellent answer!
```

### Example

**Query:** "How do I configure networking for Windows VMs?"

**Step 1: Hybrid Chunking (during indexing)**
```
windows-vm.md split into:
  Chunk 1: "# Windows VM Setup" + "## Prerequisites"
  Chunk 2: "## Networking Configuration" + network details ← Perfect!
  Chunk 3: "## Troubleshooting" + issues
```

**Step 2: Fast Retrieval (query time)**
```
Embeddings find top 50 chunks mentioning:
  - Windows, VM, networking, configuration
  - Includes some false positives (Linux VMs, general networking)
```

**Step 3: Reranking (query time)**
```
Cross-encoder precisely scores each chunk:
  Best: "## Networking Configuration for Windows VMs" ← Exact match!
  Good: "## Advanced Networking" from windows-vm.md
  Lower: "## Networking" from linux-vm.md ← Filtered out
```

**Result:** Copilot gets PERFECT context for the answer!

---

## Setup Guide

### 1. Enable Both Features

```bash
# .env
ENABLE_SEMANTIC_SEARCH=true
ENABLE_RERANKING=true
CHUNK_WITH_HEADINGS=true
CHUNK_MAX_TOKENS=512
```

### 2. Rebuild Index

```bash
# Rebuild with new chunking
uv run python -m indexer.build_index --mode full
```

This will:
- Use hybrid chunking (preserves structure)
- Generate embeddings per chunk
- Save to .index/

### 3. Start Server

```bash
uv run python server.py
```

Reranker loads automatically on first search!

### 4. Test It

```
@docs-rag search for "windows vm troubleshooting"
```

You should see `match_type: "grep+reranked"` or `"semantic+reranked"`

---

## Performance Comparison

### Without These Features

| Metric | Value |
|--------|-------|
| Chunking | Character-based (bad quality) |
| Search time | 50-200ms (fast) |
| Result quality | Medium |
| False positives | High |

### With Advanced Chunking Only

| Metric | Value |
|--------|-------|
| Chunking | Structure-aware (good quality) |
| Search time | 50-200ms (fast) |
| Result quality | Good |
| False positives | Medium |

### With Chunking + Reranking

| Metric | Value |
|--------|-------|
| Chunking | Structure-aware (good quality) |
| Search time | 200-700ms (still fast!) |
| Result quality | **Excellent** |
| False positives | **Low** |

---

## Troubleshooting

### Reranking is slow

**Try faster model:**
```bash
RERANKER_MODEL=cross-encoder/ms-marco-TinyBERT-L-2-v2  # Fastest
```

**Or disable for keyword searches:**
```bash
# Only use reranking for semantic search
# Fast search will skip reranking
```

### Chunks are too small/large

**Adjust chunking:**
```bash
CHUNK_MAX_TOKENS=768  # Larger chunks
# or
CHUNK_MAX_TOKENS=384  # Smaller chunks
```

### Out of memory

**Use smaller reranker:**
```bash
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-2-v2
```

---

## FAQ

**Q: Is reranking necessary?**
A: No, but highly recommended for 3000+ files. It significantly improves quality for ~500ms extra.

**Q: Does reranking need API keys?**
A: No! Completely free and local using HuggingFace models.

**Q: Can I customize chunking strategy?**
A: Yes! Edit `indexer/doc_indexer.py` → `AdvancedChunkingConfig` class.

**Q: Does this increase index size?**
A: Slightly. Chunks have more metadata (heading hierarchy), but negligible.

**Q: Works with GitHub Copilot?**
A: Absolutely! Copilot receives the reranked results - even better quality!

---

## Recommended Setup (Your 3000 Files)

```bash
# .env - Optimal configuration
DOCS_FOLDER=/path/to/3000-files

# Embeddings (FREE, local)
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Reranking (FREE, better quality)
ENABLE_RERANKING=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Advanced chunking
CHUNK_MAX_TOKENS=512
CHUNK_WITH_HEADINGS=true
CHUNK_MERGE_PEERS=true

# Search
ENABLE_SEMANTIC_SEARCH=true
MAX_RESULTS=10
```

**Result:**
- ✅ Best possible search quality
- ✅ Still fast (< 1 second)
- ✅ Completely free
- ✅ Works with GitHub Copilot
- ✅ No API keys needed

---

## Summary

| Feature | What | Why | Cost |
|---------|------|-----|------|
| **Hybrid Chunking** | Smart doc splitting | Better chunks = better search | $0 |
| **Reranking** | Precise result scoring | Better ordering = better results | $0 |

**Both together = Maximum search quality for your 3000 files!**

**Next:** Read [SETUP_FOR_COPILOT.md](SETUP_FOR_COPILOT.md) for complete setup!
