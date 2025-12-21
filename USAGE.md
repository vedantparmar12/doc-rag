# Usage Guide

Complete guide to using the MCP Docs RAG server.

## Table of Contents

1. [Basic Search](#basic-search)
2. [Advanced Search](#advanced-search)
3. [Configuration](#configuration)
4. [Best Practices](#best-practices)
5. [Real-World Examples](#real-world-examples)

---

## Basic Search

### Simple Keyword Search

```
@docs-rag search for "windows vm"
```

**What happens:**
1. Server searches filenames for "windows vm"
2. Searches content using ripgrep (fast!)
3. Returns top results with relevance scores

**Example output:**
```
Found 3 documentation matches for 'windows vm':

1. **Windows VM Setup Guide**
   Path: `platform-team/iaas/windows-vm.md`
   Team: platform-team | Category: iaas
   Score: 95.0 (exact match)

   Complete guide to provisioning Windows VMs in Azure...
```

### Semantic Search

For conceptual questions:

```
@docs-rag how do I deploy a containerized application?
```

**What happens:**
1. Query is embedded using OpenAI
2. Compares against all doc embeddings
3. Returns semantically similar docs

**Good for:**
- "How to..." questions
- Conceptual understanding
- When exact keywords don't match

---

## Advanced Search

### Folder-Specific Search

Search only in specific team/category:

```
@docs-rag search for "pipeline" in folder "platform-team/ci-cd"
```

**Use cases:**
- You know which team owns the doc
- Reduce noise from other teams
- Faster results

### List Documentation Structure

See what docs are available:

```
@docs-rag list the documentation structure
```

**Output:**
```
Documentation Structure: team-docs
📁 platform-team/ (12 docs)
  📁 ci-cd/ (4 docs)
    📄 pipelines.md (8.3 KB)
    📄 github-actions.md (5.1 KB)
  📁 iaas/ (5 docs)
    📄 windows-vm.md (12.4 KB)
```

### Get Full File

Read complete documentation file:

```
@docs-rag get file "platform-team/iaas/windows-vm.md"
```

**Returns:**
- Full markdown content
- File metadata (size, team, category)
- Image descriptions if VLM enabled

### Find Config Examples

Extract code blocks:

```
@docs-rag find configs for "github actions pipeline" language "yaml"
```

**Returns:**
```yaml
name: Deploy Frontend
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      ...
```

### Summarize Topic

Get AI summary of all related docs:

```
@docs-rag summarize topic "kubernetes deployment"
```

**What happens:**
1. Finds all docs about topic
2. Reads top 5 most relevant
3. Uses GPT-4o-mini to generate summary

**Output:**
```
# Summary: kubernetes deployment

Analyzed 3 documents:
- Kubernetes Setup (platform-team/paas/kubernetes.md)
- Helm Charts (platform-team/paas/helm-charts.md)
- K8s Deployment Guide (backend-team/deployment.md)

## Overview

Kubernetes deployment in our infrastructure uses Helm charts...
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCS_FOLDER` | (required) | Path to your docs |
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `ENABLE_SEMANTIC_SEARCH` | `true` | Use embeddings for search |
| `ENABLE_IMAGE_UNDERSTANDING` | `false` | Process images with VLM |
| `MAX_RESULTS` | `10` | Max search results |
| `INDEX_PATH` | `.index` | Index storage location |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |

### Performance Tuning

#### Fast Mode (Keyword Only)

```bash
export ENABLE_SEMANTIC_SEARCH=false
```

**Pros:**
- 10x faster searches
- No OpenAI API calls for search
- Works offline (except summarize)

**Cons:**
- No conceptual matching
- Exact keywords only

#### With Semantic Search

```bash
export ENABLE_SEMANTIC_SEARCH=true
```

**Pros:**
- Understands questions
- Finds related concepts
- Better for "how to" queries

**Cons:**
- Slower initial index build
- Uses OpenAI embeddings API

#### With Image Understanding

```bash
export ENABLE_IMAGE_UNDERSTANDING=true
uv run python -m indexer.build_index --enable-vlm
```

**Pros:**
- Diagrams become searchable
- Image descriptions in results

**Cons:**
- Much slower indexing
- Requires Docling VLM models

---

## Best Practices

### Organizing Your Docs

**Good structure:**
```
team-docs/
├── platform-team/        # Clear team name
│   ├── ci-cd/           # Specific category
│   │   ├── README.md    # Overview
│   │   └── pipelines.md # Specific topic
```

**Bad structure:**
```
docs/
├── file1.md
├── file2.md
├── random/
│   └── stuff.md
```

**Tips:**
- Use team names as top-level folders
- Group by category (ci-cd, iaas, paas, etc.)
- Use descriptive filenames (`windows-vm.md` not `doc1.md`)
- Include README.md in each folder

### Writing Searchable Docs

**Good markdown:**
```markdown
# Windows VM Setup Guide

## Prerequisites
- Azure subscription
- Terraform installed

## Steps

### 1. Create Resource Group
```bash
az group create --name my-rg --location eastus
```

### Configuration
```yaml
vm_config:
  size: Standard_D4s_v3
  os: Windows Server 2022
```
```

**Tips:**
- Use descriptive headings
- Include code blocks with language tags
- Add alt text to images
- Use tables for structured data

### When to Rebuild Index

**Rebuild after:**
- Adding new docs
- Major doc updates
- Changing VLM settings
- Folder restructuring

**Command:**
```bash
uv run python -m indexer.build_index
```

**Auto-rebuild coming soon!** (See roadmap)

---

## Real-World Examples

### Example 1: New Developer Onboarding

**Scenario:** New dev needs to deploy to production

```
Developer: @docs-rag how do I deploy to production?

Server: [Searches "deploy production"]
Found CI/CD pipeline docs and deployment guides

Developer: @docs-rag show me the backend deployment pipeline

Server: [Returns backend-team/deployment.md with configs]

Developer: @docs-rag what are the prerequisites?

Server: [Extracts prerequisites section]
```

### Example 2: Debugging Infrastructure Issue

**Scenario:** VM won't start, need troubleshooting

```
DevOps: @docs-rag search for "vm troubleshooting" in "platform-team/iaas"

Server: [Finds iaas/vm-troubleshooting.md]

DevOps: @docs-rag get file "platform-team/iaas/vm-troubleshooting.md"

Server: [Returns full troubleshooting guide]
```

### Example 3: Architecture Review

**Scenario:** Understanding overall system architecture

```
Architect: @docs-rag list structure

Server: [Shows all team folders and categories]

Architect: @docs-rag summarize topic "architecture"

Server: [Analyzes all architecture docs]
Analyzed 5 documents across 3 teams...
Overview: The system uses microservices architecture with...

Architect: @docs-rag find configs for "api gateway"

Server: [Extracts all API gateway configs]
```

### Example 4: Compliance Audit

**Scenario:** Security audit needs docs

```
Auditor: @docs-rag search for "security" folder "security-team"

Server: [Lists all security docs]

Auditor: @docs-rag get file "security-team/access-control.md"

Server: [Returns access control policies]

Auditor: @docs-rag find configs for "rbac" language "yaml"

Server: [Extracts RBAC configurations]
```

---

## Tips & Tricks

### Faster Searches

1. **Install ripgrep** for 10x faster text search
2. **Use folder filters** to narrow scope
3. **Disable semantic search** if you don't need it
4. **Keep index updated** for accurate results

### Better Results

1. **Use specific terms** ("windows vm" vs "vm")
2. **Filter by folder** when you know the team
3. **Use semantic search** for questions
4. **Combine tools** (search → get file → find configs)

### Working with Large Doc Sets

1. **Build index incrementally**:
   ```bash
   # Index one team at a time
   DOCS_FOLDER=team-docs/platform-team uv run python -m indexer.build_index
   ```

2. **Use folder filtering** in searches

3. **Cache frequently accessed docs** (coming soon)

---

## Keyboard Shortcuts (VS Code)

| Shortcut | Action |
|----------|--------|
| `@docs-rag` | Invoke MCP tool |
| `Ctrl+Space` | Tool completion |
| `Ctrl+Click` | Jump to file (if supported) |

---

## Next Steps

- Explore [README.md](README.md) for complete feature list
- Check [QUICKSTART.md](QUICKSTART.md) for setup
- See [config/](config/) for configuration examples
- Read [search/hybrid_search.py](search/hybrid_search.py) to customize ranking

**Questions?** Open an issue or check the FAQ in README.md!
