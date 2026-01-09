# MCP Skills for Documentation RAG

This folder contains predefined skills (workflows) for the Documentation RAG MCP server.

## What are Skills?

Skills are reusable, named workflows that combine multiple MCP tool calls into common patterns. They make it easier for users to accomplish complex tasks with simple commands.

## Available Skills

### 1. `deep-search`
**Purpose:** Comprehensive search across all features (keyword + semantic + tables)

**Usage:** `/deep-search kubernetes deployment`

**What it does:**
- Searches documents with keyword search
- Performs semantic search
- Searches tables specifically
- Combines and ranks all results
- Provides rich context with tables, images, and links

### 2. `find-table`
**Purpose:** Quickly find and display specific tables

**Usage:** `/find-table pricing tiers`

**What it does:**
- Searches for tables matching the query
- Displays table with headers and data
- Shows related documentation
- Provides context about the table

### 3. `explain-topic`
**Purpose:** Get comprehensive explanation of a topic from documentation

**Usage:** `/explain-topic CI/CD pipeline`

**What it does:**
- Searches for relevant documents
- Extracts key information
- Summarizes the topic
- Includes code examples and configs
- Shows related topics

### 4. `find-config`
**Purpose:** Extract configuration examples (YAML, JSON, etc.)

**Usage:** `/find-config kubernetes deployment yaml`

**What it does:**
- Searches for configuration files
- Extracts code blocks
- Shows examples with context
- Provides related configurations

### 5. `explore-docs`
**Purpose:** Browse and explore documentation structure

**Usage:** `/explore-docs platform-team`

**What it does:**
- Lists folder structure
- Shows available documents
- Displays document summaries
- Helps navigate large documentation sets

### 6. `quick-answer`
**Purpose:** Fast answer to specific questions

**Usage:** `/quick-answer how to deploy to production`

**What it does:**
- Fast semantic search
- Returns most relevant sections
- Provides concise answer
- Includes source references

## How to Use

### In GitHub Copilot Chat
```
/deep-search kubernetes
```

### In Codex CLI
```
@docs-rag /find-table pricing
```

### In Claude Desktop
```
Use the deep-search skill to find information about deployment
```

## Skill Development

To add new skills, create a new file in this directory following the pattern:
- `skill-name.json` - Skill definition
- Clear purpose and usage instructions
- Step-by-step tool invocations
