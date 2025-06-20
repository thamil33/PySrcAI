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
for key in ["OPENROUTER_API_KEY", "HF_API_TOKEN", "HF_HOME"]:
    value = os.getenv(key)
    if value:
        print(f"  ✅ {key}: {value[:10]}..." if len(value) > 10 else f"  ✅ {key}: {value}")
    else:
        print(f"  ❌ {key}: Not set")

print("\n🧪 Testing HuggingFace API:")
import requests

hf_token = os.getenv("HF_API_TOKEN")
if hf_token:
    headers = {"Authorization": f"Bearer {hf_token}"}
    print(f"  🔑 Using token: {hf_token[:10]}...")
    
    # Test the whoami endpoint
    try:
        print("  📡 Testing whoami endpoint...")
        response = requests.get("https://huggingface.co/api/whoami", headers=headers)
        print(f"  📊 Response status: {response.status_code}")
        if response.status_code == 200:
            user_info = response.json()
            print(f"  ✅ API token valid for user: {user_info.get('name', 'Unknown')}")
        else:
            print(f"  ❌ API token invalid: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"  ❌ Error testing whoami API: {e}")
    
    # Test the actual inference API we're using
    try:
        print("  📡 Testing inference API...")
        inference_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/BAAI/bge-base-en-v1.5"
        response = requests.post(
            inference_url,
            headers=headers,
            json={"inputs": ["test"], "options": {"wait_for_model": True}}
        )
        print(f"  📊 Inference response status: {response.status_code}")
        if response.status_code == 200:
            print(f"  ✅ Inference API working")
        else:
            print(f"  ❌ Inference API error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"  ❌ Error testing inference API: {e}")
else:
    print("  ❌ No HF_API_TOKEN found")
