#!/usr/bin/env python3
"""Test environment variable loading."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
print("🔧 Loading .env file...")
load_dotenv()

# Check if .env file exists
env_file = Path(".env")
print(f"📁 .env file exists: {env_file.exists()}")

if env_file.exists():
    print(f"📄 .env file size: {env_file.stat().st_size} bytes")
    print("\n📝 .env file contents:")
    with open(env_file) as f:
        for line_num, line in enumerate(f, 1):
            if line.strip() and not line.strip().startswith('#'):
                key = line.split('=')[0]
                print(f"  {line_num}: {key}=***")

print("\n🔍 Environment variables:")
for key in ["OPENROUTER_API_KEY", "HF_HOME"]:
    value = os.getenv(key)
    if value:
        print(f"  ✅ {key}: {value[:10]}..." if len(value) > 10 else f"  ✅ {key}: {value}")
    else:
        print(f"  ❌ {key}: Not set")