"""Test the exact OpenRouterLLM class that's causing the 404 error."""

import os
from dotenv import load_dotenv
from pyscrai.agentica.adapters.llm import OpenRouterLLM

# Load environment variables
load_dotenv()

def test_openrouter_llm_class():
    """Test the OpenRouterLLM class with different model configurations."""
    
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not set")
        return
    
    print("🧪 Testing OpenRouterLLM Class")
    print("=" * 40)
    
    # Test configurations that should work based on our debug results
    test_configs = [
        {
            "name": "Mistral Small 3.1 (free)",
            "model": "mistralai/mistral-small-3.1-24b-instruct:free"
        },
        {
            "name": "Mistral Small 2501 (free)", 
            "model": "mistralai/mistral-small-24b-instruct-2501:free"
        },
        {
            "name": "Mistral Small 3.1 (no free suffix)",
            "model": "mistralai/mistral-small-3.1-24b-instruct"
        },
        {
            "name": "Mistral Small 2501 (no free suffix)",
            "model": "mistralai/mistral-small-24b-instruct-2501"
        }
    ]
    
    for config in test_configs:
        print(f"\n🔬 Testing: {config['name']}")
        print(f"   Model: {config['model']}")
        
        try:
            # Create the LLM instance
            llm = OpenRouterLLM(
                model=config['model'],
                temperature=0.7,
                max_tokens=50,
                app_name="pyscrai_debug",
                site_url="https://github.com/tyler-richardson/pyscrai_workstation"
            )
            
            print("   ✅ LLM instance created successfully")
            
            # Test a simple invoke
            response = llm.invoke("Say 'Hello World' and nothing else.")
            print(f"   ✅ Response: {response[:100]}...")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            print(f"   📝 Error type: {type(e).__name__}")
            
            # If it's a request exception, let's see more details
            if hasattr(e, 'response'):
                print(f"   📊 Status code: {e.response.status_code}")
                print(f"   📄 Response text: {e.response.text[:300]}")

if __name__ == "__main__":
    test_openrouter_llm_class()
