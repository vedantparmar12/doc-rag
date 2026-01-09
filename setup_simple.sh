#!/bin/bash

# ============================================
# MCP RAG - Simplified Setup Script
# For: 400 fixed markdown files (no images)
# ============================================

echo "🚀 Setting up Simplified MCP RAG (Markdown-Only)"
echo ""

# 1. Install basic dependencies
echo "📦 Installing dependencies..."
uv sync

# 2. Install embeddings (local, FREE)
echo "🧮 Installing local embeddings..."
pip install sentence-transformers

# 3. Copy simplified config
echo "📝 Creating configuration..."
if [ ! -f .env ]; then
    cp .env.simple .env
    echo "✅ Created .env from .env.simple"
else
    echo "⚠️  .env already exists, skipping"
fi

# 4. Prompt for docs folder
echo ""
echo "📂 Where are your markdown files?"
read -p "Enter path (e.g., /path/to/docs): " docs_path

if [ -n "$docs_path" ]; then
    # Update .env with docs path
    sed -i "s|DOCS_FOLDER=.*|DOCS_FOLDER=$docs_path|" .env
    echo "✅ Set DOCS_FOLDER=$docs_path"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Build index: uv run python -m indexer.doc_indexer_simple"
echo "  2. Start server: uv run python server.py"
echo ""
echo "Features:"
echo "  ✅ Table extraction (for parameter tables)"
echo "  ✅ JSON extraction (for config examples)"
echo "  ✅ Smart search (keyword + semantic)"
echo "  ✅ Query decomposition (complex questions)"
echo "  ✅ Feedback learning (improves over time)"
echo "  ✅ Conversation context (remembers queries)"
echo "  ✅ $0 cost (all local, no API keys)"
echo ""
echo "🎉 Ready to build your index!"
