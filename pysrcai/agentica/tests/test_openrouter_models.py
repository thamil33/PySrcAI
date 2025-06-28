"""Debug script to investigate OpenRouter free models."""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_openrouter_models():
    """Check available OpenRouter models and specifically look for free models."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment variables")
        return

    print("🔍 Investigating OpenRouter Models")
    print("=" * 50)

    # Get all available models
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        print("📡 Fetching available models...")
        response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)
        response.raise_for_status()
        data = response.json()

        models = data.get("data", [])
        print(f"✅ Found {len(models)} total models")

        # Filter for free models
        free_models = []
        mistral_models = []

        for model in models:
            model_id = model.get("id", "")
            pricing = model.get("pricing", {})

            # Check if it's a free model (pricing shows 0)
            prompt_cost = float(pricing.get("prompt", "1"))
            completion_cost = float(pricing.get("completion", "1"))

            if prompt_cost == 0 and completion_cost == 0:
                free_models.append(model)

            # Also collect Mistral models specifically
            if "mistral" in model_id.lower():
                mistral_models.append(model)

        print(f"\n💰 Free Models Found ({len(free_models)}):")
        print("-" * 30)
        for model in free_models[:10]:  # Show first 10
            model_id = model.get("id", "")
            name = model.get("name", "")
            print(f"  🆓 {model_id}")
            if name:
                print(f"     {name}")

        print(f"\n🤖 Mistral Models Found ({len(mistral_models)}):")
        print("-" * 30)
        for model in mistral_models[:10]:  # Show first 10
            model_id = model.get("id", "")
            name = model.get("name", "")
            pricing = model.get("pricing", {})
            prompt_cost = float(pricing.get("prompt", "0"))
            completion_cost = float(pricing.get("completion", "0"))

            is_free = prompt_cost == 0 and completion_cost == 0
            status = "🆓 FREE" if is_free else f"💵 ${prompt_cost}/${completion_cost}"

            print(f"  {status} {model_id}")
            if name:
                print(f"     {name}")

        # Test specific models
        test_models = [
            "mistralai/mistral-small-3.1-24b-instruct:free",
            "mistralai/mistral-small-24b-instruct-2501:free",
            "mistralai/mistral-small-3.1-24b-instruct",
            "mistralai/mistral-small-24b-instruct-2501"
        ]

        print(f"\n🧪 Testing Specific Models:")
        print("-" * 30)

        for model_name in test_models:
            print(f"\n🔬 Testing: {model_name}")
            test_payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "Hello, can you say 'test successful'?"}],
                "max_tokens": 10,
                "temperature": 0.1
            }

            try:
                test_response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=test_payload,
                    timeout=30
                )

                if test_response.status_code == 200:
                    result = test_response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    print(f"  ✅ SUCCESS: {content[:50]}...")
                else:
                    print(f"  ❌ FAILED: {test_response.status_code} - {test_response.text[:200]}")

            except Exception as e:
                print(f"  ❌ ERROR: {str(e)}")

    except Exception as e:
        print(f"❌ Error fetching models: {str(e)}")

if __name__ == "__main__":
    check_openrouter_models()
