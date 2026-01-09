#!/usr/bin/env python3
"""
Create a zip archive of the MCP Docs RAG project.
"""

import zipfile
import os
from pathlib import Path

def create_zip():
    """Create zip archive of the project."""
    # Source folder
    source_dir = Path(__file__).parent

    # Destination zip file on desktop
    desktop = Path.home() / "Desktop"
    zip_path = desktop / "mcp-docs-rag-optimized.zip"

    print(f"Creating archive: {zip_path}")
    print(f"Source: {source_dir}")

    # Create zip file
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through all files
        for root, dirs, files in os.walk(source_dir):
            # Skip .venv, __pycache__, .git, .index
            dirs[:] = [d for d in dirs if d not in ['.venv', '__pycache__', '.git', '.index', 'node_modules']]

            for file in files:
                # Skip .pyc files
                if file.endswith('.pyc'):
                    continue

                file_path = Path(root) / file
                arcname = file_path.relative_to(source_dir.parent)

                print(f"  Adding: {arcname}")
                zipf.write(file_path, arcname)

    # Get file size
    size_mb = zip_path.stat().st_size / (1024 * 1024)

    print(f"\n✓ Archive created successfully!")
    print(f"  Location: {zip_path}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"\nExtract and use:")
    print(f"  1. Extract the zip file")
    print(f"  2. cd mcp-docs-rag")
    print(f"  3. bash install_optimizations.sh")
    print(f"  4. Edit .env and set DOCS_FOLDER")
    print(f"  5. uv run python -m indexer.build_index --mode full")
    print(f"  6. uv run python server.py")

if __name__ == "__main__":
    create_zip()
