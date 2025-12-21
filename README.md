# MCP Documentation RAG Server

**Smart documentation search for team-organized markdown repositories**

## Overview

This MCP server provides intelligent documentation search over markdown files organized in team/category folders. It uses a **hybrid search approach** combining:

- ⚡ **Fast text search** (ripgrep) for exact/fuzzy matches
- 🧠 **Semantic search** (vector embeddings) for conceptual queries
- 🖼️ **Image understanding** (Docling VLM) for diagrams and screenshots
- 📁 **Folder-aware search** (filter by team/category)

## Use Cases

- "How do I create a Windows VM?" → Finds `iaas/windows-vm.md`
- "Which pipeline for frontend deployment?" → Finds `ci-cd/frontend-pipeline.md`
- "What config for Kubernetes?" → Finds all k8s-related configs with examples
- "Show me architecture diagrams" → Extracts and describes images from docs

## Architecture

```
Your Docs Folder Structure:
├── platform-team/
│   ├── ci-cd/
│   │   ├── pipelines.md
│   │   └── github-actions.md
│   ├── paas/
│   │   └── kubernetes-setup.md
│   └── iaas/
│       ├── windows-vm.md
│       └── linux-vm.md
├── backend-team/
│   └── api-docs.md
└── frontend-team/
    └── deployment.md

MCP Server:
┌─────────────────────────────────────┐
│     Fast Search (ripgrep)           │ ← "windows vm" → instant results
├─────────────────────────────────────┤
│   Semantic Search (embeddings)      │ ← "how to deploy" → conceptual match
├─────────────────────────────────────┤
│  Image Understanding (Docling VLM)  │ ← processes diagrams, screenshots
└─────────────────────────────────────┘
          ↓
    VS Code (MCP Client)
```

## Features

### 🔍 Multi-Tier Search Strategy

1. **Keyword Search** - Fast grep-based search
   ```python
   search_docs(query="windows vm", team="platform-team")
   # Returns: iaas/windows-vm.md with relevance score
   ```

2. **Semantic Search** - Understanding intent
   ```python
   semantic_search(query="how to deploy containerized apps")
   # Returns: paas/kubernetes-setup.md, ci-cd/docker-pipeline.md
   ```

3. **Folder Filtering** - Team/category aware
   ```python
   search_docs(query="pipeline", folder="platform-team/ci-cd")
   # Only searches in ci-cd folder
   ```

4. **Image-Aware** - Understands diagrams
   ```python
   get_doc_with_images(path="iaas/windows-vm.md")
   # Returns: Markdown + image descriptions
   ```

### 🛠️ MCP Tools

| Tool | Description | Use Case |
|------|-------------|----------|
| `search_docs` | Fast keyword search | "find windows vm docs" |
| `semantic_search` | Conceptual search | "how do I deploy?" |
| `get_file` | Get full file content | "show me pipeline config" |
| `list_structure` | Browse folder tree | "what teams/categories exist?" |
| `summarize_topic` | Multi-doc summary | "summarize all k8s docs" |
| `find_configs` | Extract code blocks | "show all pipeline YAML configs" |

## Installation

```bash
# Navigate to project
cd mcp-docs-rag

# Install dependencies
uv sync

# Set up environment
cp .env.example .env
# Edit .env with your settings
```

## Configuration

### 1. Point to Your Docs Folder

```bash
# .env
DOCS_FOLDER=/path/to/your/team/docs
OPENAI_API_KEY=sk-...
```

### 2. Build Index (One-time)

```bash
# Quick index (metadata only - instant)
uv run python -m indexer.build_index --mode fast

# Full index (with embeddings - slower but semantic search)
uv run python -m indexer.build_index --mode full

# With image understanding (Docling VLM)
uv run python -m indexer.build_index --mode full --enable-vlm
```

### 3. Start MCP Server

```bash
# Stdio mode (for VS Code)
uv run python server.py

# HTTP mode (for testing)
uv run python server.py --transport http --port 3000
```

### 4. Configure VS Code

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
        "DOCS_FOLDER": "C:/path/to/your/docs"
      }
    }
  }
}
```

Or use Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "docs-rag": {
      "command": "uv",
      "args": ["run", "python", "/path/to/mcp-docs-rag/server.py"],
      "env": {
        "DOCS_FOLDER": "/path/to/your/docs"
      }
    }
  }
}
```

## Usage Examples

### In VS Code with MCP

```
User: @docs-rag How do I create a Windows VM?
Assistant: [Uses search_docs tool]
> Found Windows VM documentation in platform-team/iaas/windows-vm.md
> Here's the step-by-step guide...

User: @docs-rag What pipeline should I run for frontend deployment?
Assistant: [Uses search_docs with folder filter]
> Found CI/CD pipeline in platform-team/ci-cd/frontend-pipeline.md
> Use the `frontend-deploy.yml` pipeline with these config parameters...

User: @docs-rag Show me all Kubernetes configs
Assistant: [Uses find_configs tool]
> Found 3 YAML configs across 2 files:
> ```yaml
> apiVersion: v1
> kind: Service
> ...
> ```
```

### In Claude Desktop

```
You: Summarize all our infrastructure documentation
Claude: [Uses summarize_topic tool]
> Analyzed 8 documents about infrastructure...
> Overview: Your infrastructure uses a hybrid cloud approach...

You: List the documentation structure
Claude: [Uses list_structure tool]
> Documentation Structure:
> 📁 platform-team/ (12 docs)
>   📁 ci-cd/ (4 docs)
>     📄 pipelines.md
>   📁 iaas/ (5 docs)
>     📄 windows-vm.md
```

## Performance Comparison

| Search Type | Method | Speed | Use Case |
|-------------|--------|-------|----------|
| Keyword | ripgrep | ~10ms | "windows vm", "pipeline config" |
| Fuzzy | rapidfuzz | ~50ms | "windowz vm" (typos) |
| Semantic | Embeddings | ~200ms | "how to deploy containers" |
| Summary | LLM | ~2s | "summarize all docs about X" |

**Strategy:** The server automatically chooses the fastest method:
1. Try exact match (instant)
2. Try ripgrep search (10-50ms)
3. Fall back to semantic if needed (200ms)

## Advanced Features

### Folder-Aware Search

```bash
# Search only in specific team folder
search_docs(query="pipeline", folder="platform-team/ci-cd")

# List specific team structure
list_structure(path="backend-team", depth=2)
```

### Image Understanding (Optional)

Enable VLM to understand diagrams and screenshots:

```bash
# Build index with image understanding
uv run python -m indexer.build_index --enable-vlm

# Now searches include image descriptions
search_docs(query="architecture diagram")
# Returns: "Diagram shows microservices architecture with API Gateway..."
```

### Config Extraction

```bash
# Find all YAML pipeline configs
find_configs(query="pipeline", language="yaml")

# Find Terraform infrastructure code
find_configs(query="terraform", language="hcl")
```

## Folder Structure Example

Your documentation should be organized like:

```
team-docs/
├── platform-team/
│   ├── ci-cd/
│   │   ├── pipelines.md           # GitHub Actions pipelines
│   │   ├── gitlab-ci.md           # GitLab CI configs
│   │   └── jenkins.md             # Jenkins setup
│   ├── paas/
│   │   ├── kubernetes.md          # K8s cluster setup
│   │   ├── helm-charts.md         # Helm deployment
│   │   └── docker-compose.md      # Local development
│   └── iaas/
│       ├── windows-vm.md          # Windows VM provisioning
│       ├── linux-vm.md            # Linux VM setup
│       └── networking.md          # Network configuration
├── backend-team/
│   ├── api-guidelines.md
│   ├── database-setup.md
│   └── deployment.md
├── frontend-team/
│   ├── build-process.md
│   ├── deployment.md
│   └── environment-config.md
└── security-team/
    ├── access-control.md
    ├── secret-management.md
    └── compliance.md
```

## Troubleshooting

### Index is outdated

```bash
# Rebuild index
uv run python -m indexer.build_index --docs /path/to/docs

# Or delete and restart (will auto-rebuild)
rm -rf .index
uv run python server.py
```

### Search is slow

1. **Disable semantic search** if you don't need it:
   ```bash
   export ENABLE_SEMANTIC_SEARCH=false
   ```

2. **Install ripgrep** for faster text search:
   ```bash
   # Windows (Chocolatey)
   choco install ripgrep

   # macOS
   brew install ripgrep

   # Linux
   apt-get install ripgrep  # Debian/Ubuntu
   ```

3. **Reduce result limit**:
   ```bash
   export MAX_RESULTS=5
   ```

### Images not being understood

1. **Enable VLM** in config:
   ```bash
   export ENABLE_IMAGE_UNDERSTANDING=true
   ```

2. **Rebuild index** with VLM:
   ```bash
   uv run python -m indexer.build_index --enable-vlm
   ```

Note: VLM processing is slower but makes diagrams searchable.

## Architecture Details

### Hybrid Search Strategy

```
User Query: "windows vm"
     ↓
┌────────────────────────┐
│  1. Exact Match        │ → ✓ Found "windows-vm.md" (100ms, score: 100)
└────────────────────────┘
     ↓ (if needed)
┌────────────────────────┐
│  2. Ripgrep Search     │ → Find content matches (10ms, score: 80-90)
└────────────────────────┘
     ↓ (if needed)
┌────────────────────────┐
│  3. Fuzzy Match        │ → Handle typos (50ms, score: 60-80)
└────────────────────────┘
     ↓ (if needed)
┌────────────────────────┐
│  4. Semantic Search    │ → Conceptual match (200ms, score: 60-100)
└────────────────────────┘
```

**Result:** Fast searches return in 10-100ms, complex queries take 200-500ms

### Index Structure

```
.index/
├── index.json          # Metadata for all files
│   ├── path, title, headings
│   ├── content, images
│   └── modified timestamp
└── embeddings.npy      # Vector embeddings (optional)
    └── [1536-dim vectors]
```

## Development

### Running Tests

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html
```

### Adding New Tools

1. Create tool in `tools/` directory
2. Register in `server.py`:
   ```python
   @app.list_tools()
   async def list_tools():
       return [
           Tool(name="my_new_tool", ...)
       ]

   @app.call_tool()
   async def call_tool(name, arguments):
       if name == "my_new_tool":
           return await my_tool_handler(**arguments)
   ```

### Debugging

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
uv run python server.py
```

View MCP communication:

```bash
# Logs are written to stderr
uv run python server.py 2>mcp-debug.log
```

## FAQ

**Q: Do I need to rebuild the index when docs change?**
A: Yes, run `uv run python -m indexer.build_index` after significant changes. For minor updates, the search will still work but may show stale metadata.

**Q: Can I use this without OpenAI?**
A: For keyword search, yes! Just set `ENABLE_SEMANTIC_SEARCH=false`. Summarization requires an LLM.

**Q: How do I search across multiple teams?**
A: Don't specify a folder filter - it will search all teams by default.

**Q: Can I customize the search ranking?**
A: Yes, modify the scoring in `search/hybrid_search.py`. Exact matches get 100, fuzzy 60-80, semantic varies.

**Q: Does it work with non-markdown files?**
A: Currently only `.md` files. You could extend it to support other formats using Docling's conversion.

## Roadmap

- [ ] Auto-rebuild index on file changes (watch mode)
- [ ] Support for other file formats (PDF, DOCX via Docling)
- [ ] Caching for faster repeated queries
- [ ] Multi-language support
- [ ] Integration with other vector stores (Pinecone, Weaviate)
- [ ] Web UI for browsing docs
- [ ] Analytics: most searched terms, popular docs

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Credits

Built with:
- **MCP**: Model Context Protocol for VS Code/Claude integration
- **Docling**: Document processing and image understanding
- **OpenAI**: Embeddings and summarization
- **ripgrep**: Fast text search
- **rapidfuzz**: Fuzzy string matching

---

**Questions?** Open an issue or reach out!

**Happy Documenting!** 📚✨