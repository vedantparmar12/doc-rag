#!/usr/bin/env python3
"""
CLI tool for building the documentation index.
"""

import sys
import asyncio
from doc_indexer import main

if __name__ == "__main__":
    asyncio.run(main())
