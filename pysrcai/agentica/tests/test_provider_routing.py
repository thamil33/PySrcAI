"""Test the specific combination causing the 404 error."""

import os
import sys
from dotenv import load_dotenv

# Add the pyscrai path so we can import the OpenRouterLLM directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pyscrai'))

from pysrcai.agentica.adapters.llm.openrouter_adapter import OpenRouterLLM

load_dotenv()

def test_provider_routing_combinations():
    """Test different combinations to isolate the 404 issue."""
    
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not set")
        return
    
    print("🧪 Testing Provider Routing Combinations")
    print("=" * 50)
    
    # Test the problematic model with different provider configurations
    test_cases = [
        {
            "name": "2501:free + NO provider routing",
            "model": "mistralai/mistral-small-24b-instruct-2501:free",
            "provider": None
        },
        {
            "name": "2501:free + Simple provider routing",
            "model": "mistralai/mistral-small-24b-instruct-2501:free", 
            "provider": {"sort": "price"}
        },
        {
            "name": "2501:free + Full provider routing (FAILING CASE)",
            "model": "mistralai/mistral-small-24b-instruct-2501:free",
            "provider": {
                "allow_fallbacks": True,
                "data_collection": "deny",
                "sort": "price"
            }
        },
        {
            "name": "3.1:free + Full provider routing",
            "model": "mistralai/mistral-small-3.1-24b-instruct:free",
            "provider": {
                "allow_fallbacks": True,
                "data_collection": "deny", 
                "sort": "price"
            }
        }
    ]
    
    for case in test_cases:
        print(f"\n🔬 {case['name']}")
        print(f"   Model: {case['model']}")
        print(f"   Provider: {case['provider']}")
        
        try:
            llm_kwargs = {
                "model": case['model'],
                "temperature": 0.7,
                "max_tokens": 20,
                "app_name": "pyscrai_test",
                "site_url": "https://github.com/test",
            }
            
            if case['provider']:
                llm_kwargs["provider"] = case['provider']
            
            llm = OpenRouterLLM(**llm_kwargs)
            response = llm.invoke("Say 'OK' only.")
            print(f"   ✅ SUCCESS: {response}")
            
        except Exception as e:
            print(f"   ❌ FAILED: {str(e)}")

if __name__ == "__main__":
    test_provider_routing_combinations()
