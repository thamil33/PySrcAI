"""Test OpenRouterLLM class directly without importing other adapters."""

import os
import sys
from dotenv import load_dotenv

# Add the pyscrai path so we can import the OpenRouterLLM directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pyscrai'))

# Import just the OpenRouterLLM class
from pysrcai.agentica.adapters.llm.openrouter_adapter import OpenRouterLLM

# Load environment variables
load_dotenv()

def test_openrouter_llm_direct():
    """Test the OpenRouterLLM class directly."""
    
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not set")
        return
    
    print("🧪 Testing OpenRouterLLM Class (Direct Import)")
    print("=" * 50)
    
    # Test the specific model from the LangGraph example
    test_configs = [
        {
            "name": "Current LangGraph Example Model",
            "model": "mistralai/mistral-small-24b-instruct-2501:free",
            "provider": {
                "allow_fallbacks": True,
                "data_collection": "deny",
                "sort": "price"
            }
        },
        {
            "name": "Default Template Model",
            "model": "mistralai/mistral-small-3.1-24b-instruct:free",
            "provider": None
        },
        {
            "name": "No Free Suffix (working version)",
            "model": "mistralai/mistral-small-24b-instruct-2501",
            "provider": None
        }
    ]
    
    for config in test_configs:
        print(f"\n🔬 Testing: {config['name']}")
        print(f"   Model: {config['model']}")
        if config['provider']:
            print(f"   Provider: {config['provider']}")
        
        try:
            # Create the LLM instance with exact same parameters as LangGraph example
            llm_kwargs = {
                "model": config['model'],
                "temperature": 0.7,
                "max_tokens": 50,
                "app_name": "pyscrai_example",
                "site_url": "https://github.com/tyler-richardson/pyscrai_workstation",
            }
            
            if config['provider']:
                llm_kwargs["provider"] = config['provider']
            
            llm = OpenRouterLLM(**llm_kwargs)
            
            print("   ✅ LLM instance created successfully")
            
            # Test a simple invoke
            response = llm.invoke("Say 'test successful' and nothing else.")
            print(f"   ✅ Response: {response[:100]}...")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            print(f"   📝 Error type: {type(e).__name__}")
            
            # Check if it's a requests exception for more details
            if "requests" in str(type(e)):
                print(f"   🔍 Detailed error: {repr(e)}")

if __name__ == "__main__":
    test_openrouter_llm_direct()
