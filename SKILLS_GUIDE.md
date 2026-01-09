# 🎯 Skills Guide - MCP Documentation RAG

> **Predefined workflows for common documentation tasks**

---

## What are Skills?

Skills are **predefined workflows** that combine multiple MCP tool calls into common patterns. Instead of manually calling tools, you use simple skill commands like `/deep-search` or `/find-table`.

**Think of them as macros or shortcuts for common tasks!**

---

## 📚 Available Skills (6 Total)

### 1. 🔍 `/deep-search` - Comprehensive Search

**Purpose:** Search across ALL features (keyword + semantic + tables)

**Usage:**
```
/deep-search kubernetes deployment
/deep-search CI/CD pipeline configuration
/deep-search troubleshooting windows VM
```

**What it does:**
1. Fast keyword search (ripgrep)
2. Semantic search (embeddings)
3. Table search
4. Combines and ranks results
5. Returns rich context with tables, images, links

**When to use:**
- You want the most thorough search
- Looking for information across multiple docs
- Need context from tables and images

---

### 2. 📊 `/find-table` - Table Search

**Purpose:** Find specific tables in documentation

**Usage:**
```
/find-table pricing tiers
/find-table deployment options
/find-table API endpoints
/find-table configuration settings
```

**What it does:**
1. Searches for tables matching description
2. Gets full context (heading, caption)
3. Displays formatted table with source
4. Shows related documentation

**When to use:**
- Looking for comparison matrices
- Need pricing information
- Want configuration options in table format
- Searching for structured data

---

### 3. 💡 `/explain-topic` - Topic Explanation

**Purpose:** Comprehensive explanation of any topic

**Usage:**
```
/explain-topic kubernetes deployment
/explain-topic microservices architecture
/explain-topic infrastructure as code
/explain-topic CI/CD pipeline
```

**What it does:**
1. Semantic search for related docs
2. Finds exact mentions
3. Extracts relevant tables
4. Gets code examples
5. Creates comprehensive summary

**Output includes:**
- Overview and definition
- Key concepts
- How it works
- Configuration examples
- Best practices
- Common issues
- Related topics

**When to use:**
- Learning about a new topic
- Need comprehensive overview
- Want examples and best practices
- Onboarding new team members

---

### 4. ⚙️ `/find-config` - Configuration Search

**Purpose:** Extract configuration examples and code snippets

**Usage:**
```
/find-config kubernetes deployment yaml
/find-config terraform aws
/find-config docker compose
/find-config github actions workflow
/find-config nginx configuration
```

**What it does:**
1. Searches for config files
2. Extracts code blocks
3. Shows with syntax highlighting
4. Provides context and usage notes

**When to use:**
- Need example configurations
- Looking for YAML/JSON templates
- Want to see how to configure something
- Searching for shell scripts

---

### 5. 🗂️ `/explore-docs` - Browse Structure

**Purpose:** Explore documentation structure and contents

**Usage:**
```
/explore-docs
/explore-docs platform-team
/explore-docs backend-team/api
/explore-docs infrastructure
```

**What it does:**
1. Lists folder structure
2. Shows document summaries
3. Counts tables, images, links
4. Provides quick statistics

**When to use:**
- New to the documentation
- Want to see what's available
- Looking for specific team's docs
- Need overview of structure

---

### 6. ⚡ `/quick-answer` - Fast Q&A

**Purpose:** Get fast, concise answer to specific question

**Usage:**
```
/quick-answer how to deploy to production
/quick-answer what is the default timeout
/quick-answer how to configure SSL
/quick-answer where are logs stored
```

**What it does:**
1. Semantic search for relevant sections
2. Extracts specific answer
3. Checks tables for data
4. Provides concise response with source

**When to use:**
- Need quick answer
- Have specific question
- Want direct information
- Don't need comprehensive explanation

---

## 🎮 How to Use Skills

### In GitHub Copilot Chat

```
User: /deep-search kubernetes deployment

Copilot uses the deep-search skill to:
1. Search docs
2. Search tables
3. Combine results
4. Provide comprehensive answer
```

### In Codex CLI

```bash
$ codex chat

> @docs-rag /find-table pricing tiers

Codex invokes find-table skill on docs-rag MCP server
```

### In Claude Desktop

```
User: Use the explain-topic skill to explain microservices

Claude: [Invokes explain-topic skill]
         [Returns comprehensive explanation]
```

---

## 📖 Skill Comparison

| Skill | Speed | Depth | Best For |
|-------|-------|-------|----------|
| **quick-answer** | ⚡⚡⚡ Very Fast | ⭐ Brief | Specific questions |
| **find-table** | ⚡⚡ Fast | ⭐⭐ Focused | Table data |
| **find-config** | ⚡⚡ Fast | ⭐⭐ Focused | Code examples |
| **deep-search** | ⚡ Moderate | ⭐⭐⭐⭐ Deep | Thorough research |
| **explore-docs** | ⚡⚡ Fast | ⭐⭐ Overview | Navigation |
| **explain-topic** | ⚡ Moderate | ⭐⭐⭐⭐⭐ Very Deep | Learning |

---

## 💡 Usage Tips

### Combining Skills

You can use multiple skills in sequence:

```
1. /explore-docs platform-team
   → See what's available

2. /deep-search kubernetes
   → Find relevant docs

3. /find-table deployment options
   → Get specific table

4. /find-config kubernetes deployment yaml
   → Get example config

5. /explain-topic kubernetes deployment
   → Understand it deeply
```

### Choosing the Right Skill

**Need quick info?**
→ Use `/quick-answer`

**Looking for tables?**
→ Use `/find-table`

**Need code examples?**
→ Use `/find-config`

**Want comprehensive info?**
→ Use `/explain-topic` or `/deep-search`

**Exploring docs?**
→ Use `/explore-docs`

---

## 🔧 Skill Configuration

Skills are defined in `skills/*.json` files:

```
skills/
├── README.md                 # This guide
├── skills.json              # Manifest of all skills
├── deep-search.json         # Deep search workflow
├── find-table.json          # Table search workflow
├── explain-topic.json       # Topic explanation workflow
├── find-config.json         # Config search workflow
├── explore-docs.json        # Documentation browsing workflow
└── quick-answer.json        # Quick Q&A workflow
```

Each skill defines:
- Name and description
- Parameters
- Workflow steps
- Output format

---

## 📝 Example Workflows

### Research Workflow

```
1. /explore-docs infrastructure
   → Discover available docs

2. /deep-search kubernetes cluster setup
   → Find relevant information

3. /find-table node requirements
   → Check specifications

4. /find-config kubernetes cluster yaml
   → Get example config

5. /explain-topic kubernetes high availability
   → Understand best practices
```

### Troubleshooting Workflow

```
1. /quick-answer why is pod crashing
   → Get quick diagnostic info

2. /find-table error codes
   → Check error reference

3. /deep-search pod troubleshooting
   → Find detailed guides

4. /find-config pod debug yaml
   → Get debug configuration
```

### Learning Workflow

```
1. /explore-docs
   → See all topics

2. /explain-topic chosen-topic
   → Learn comprehensively

3. /find-table topic-comparison
   → See options

4. /find-config topic-example
   → Practice with examples
```

---

## 🎯 Real-World Examples

### Example 1: New Developer Onboarding

**Task:** Learn about the deployment process

```
Dev: /explain-topic deployment process

Result:
# Deployment Process - Comprehensive Explanation

## Overview
Our deployment process uses GitOps with ArgoCD...

## Key Concepts
- Continuous Deployment
- GitOps workflow
- Kubernetes manifests

## How It Works
1. Developer pushes to main branch
2. CI pipeline runs tests
3. ArgoCD syncs to cluster
...

## Configuration & Setup
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
...
```

## Best Practices
- Always use staging first
- Tag releases properly
...

## Related Topics
- CI/CD Pipeline
- Kubernetes Deployment
- GitOps Principles
```

---

### Example 2: Finding Pricing Information

**Task:** Compare pricing tiers

```
Manager: /find-table pricing tiers

Result:
## Found 2 table(s) for 'pricing tiers'

### Table 1: Cloud Provider Pricing
**Source:** infrastructure/cloud-pricing.md

| Tier | vCPUs | RAM | Storage | Price/Month |
|------|-------|-----|---------|-------------|
| Basic | 2 | 4GB | 50GB | $29 |
| Pro | 4 | 16GB | 200GB | $99 |
| Enterprise | 8 | 32GB | 500GB | $299 |

**Context:** Our infrastructure costs vary by tier...

### Table 2: Support Pricing
...
```

---

### Example 3: Debugging Production Issue

**Task:** Find why service is failing

```
Ops: /quick-answer common causes of 502 errors

Result:
## Answer

502 Bad Gateway errors typically occur when:
1. Backend service is down
2. Timeout configuration too low
3. Network connectivity issues

## Details

According to our troubleshooting guide, check:
- Service health: `kubectl get pods`
- Logs: `kubectl logs <pod-name>`
- Timeouts in nginx config

## Source

📁 operations/troubleshooting.md
🔗 Section: HTTP 502 Errors

"When you see 502 errors, the gateway cannot reach
the backend service. First verify the service is
running with `kubectl get svc`..."
```

---

## 🚀 Advanced Usage

### Custom Workflows

You can create custom skills by adding new JSON files to the `skills/` folder:

```json
{
  "name": "my-custom-skill",
  "description": "My custom workflow",
  "parameters": {
    "param1": {
      "type": "string",
      "required": true
    }
  },
  "workflow": [
    {
      "step": 1,
      "action": "search_docs",
      "params": {
        "query": "${param1}"
      }
    }
  ]
}
```

---

## 📊 Performance

| Skill | Avg Time | Tools Used |
|-------|----------|------------|
| quick-answer | < 1s | 1-2 tools |
| find-table | 1-2s | 2-3 tools |
| find-config | 1-2s | 1-2 tools |
| explore-docs | 1-2s | 1 tool |
| deep-search | 2-5s | 3-4 tools |
| explain-topic | 3-7s | 4-5 tools |

*Times are for 900 markdown files with all features enabled*

---

## ✅ Best Practices

1. **Start simple**: Use `/quick-answer` or `/explore-docs` first
2. **Be specific**: Better queries get better results
3. **Use right skill**: Match skill to your need (see comparison table)
4. **Combine skills**: Build workflows with multiple skills
5. **Iterate**: Refine queries based on results

---

## 🎓 Summary

**6 Skills Available:**
1. `/deep-search` - Comprehensive search
2. `/find-table` - Table lookup
3. `/explain-topic` - Topic explanation
4. `/find-config` - Config examples
5. `/explore-docs` - Browse structure
6. `/quick-answer` - Fast Q&A

**All skills are:**
- ✅ Ready to use
- ✅ No configuration needed
- ✅ Work with all MCP clients
- ✅ Optimized for 900+ files
- ✅ Include tables, images, links

**Start using skills today to make documentation search 10x easier!** 🚀
