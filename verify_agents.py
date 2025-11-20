import os
import sys
from dotenv import load_dotenv

# Mock Settings to control environment for testing
from unittest.mock import patch

# Add api directory to path
sys.path.append(os.path.join(os.getcwd(), "api"))

def test_agent_config():
    print("Testing Agent Configuration...")
    
    # Test 1: Default Configuration (Gemini)
    print("\nTest 1: Default Configuration (Expect Gemini)")
    with patch.dict(os.environ, {}, clear=True):
        # We need to reload config and agents to pick up env changes
        if 'config' in sys.modules: del sys.modules['config']
        if 'api.graph.agents.router_agent' in sys.modules: del sys.modules['api.graph.agents.router_agent']
        
        from api.graph.agents.router_agent import router_agent
        # Inspect the llm object inside router_agent if possible, 
        # but since it's a local variable in the function, we can't easily inspect it without running it.
        # However, we can inspect the imports or mock the classes.
        
        # Let's mock Gemini3Client and ChatOpenAI to see which one is called.
        with patch('api.graph.agents.router_agent.Gemini3Client') as mock_gemini, \
             patch('api.graph.agents.router_agent.ChatOpenAI') as mock_openai:
            
            # Trigger the agent function (we need a mock state)
            try:
                router_agent({"messages": []})
            except Exception as e:
                # It might fail due to missing API keys or other things, but we just want to see the constructor call
                pass
            
            if mock_gemini.called:
                print("SUCCESS: Gemini3Client was called.")
                print(f"Args: {mock_gemini.call_args}")
            else:
                print("FAILURE: Gemini3Client was NOT called.")
                
            if mock_openai.called:
                print("FAILURE: ChatOpenAI was called unexpectedly.")

    # Test 2: Custom Configuration (OpenAI)
    print("\nTest 2: Custom Configuration (Expect OpenAI)")
    with patch.dict(os.environ, {
        "ROUTER_AGENT_URL": "http://localhost:1234/v1",
        "ROUTER_AGENT_API_KEY": "test-key",
        "ROUTER_AGENT_MODEL": "test-model"
    }, clear=True):
        if 'config' in sys.modules: del sys.modules['config']
        from api.config import get_settings
        get_settings.cache_clear()
        
        from api.graph.agents.router_agent import router_agent
        
        with patch('api.graph.agents.router_agent.Gemini3Client') as mock_gemini, \
             patch('api.graph.agents.router_agent.ChatOpenAI') as mock_openai:
            
            try:
                router_agent({"messages": []})
            except Exception:
                pass
                
            if mock_openai.called:
                print("SUCCESS: ChatOpenAI was called.")
                print(f"Args: {mock_openai.call_args}")
            else:
                print("FAILURE: ChatOpenAI was NOT called.")
                
            if mock_gemini.called:
                print("FAILURE: Gemini3Client was called unexpectedly.")

    # Test 3: Code Agent Tool Enablement
    print("\nTest 3: Code Agent (Expect Code Execution Enabled)")
    with patch.dict(os.environ, {}, clear=True):
        if 'api.graph.agents.code_agent' in sys.modules: del sys.modules['api.graph.agents.code_agent']
        from api.graph.agents.code_agent import code_agent
        
        with patch('api.graph.agents.code_agent.Gemini3Client') as mock_gemini:
            try:
                code_agent({"messages": []})
            except Exception:
                pass
            
            if mock_gemini.return_value.enable_code_execution.called:
                print("SUCCESS: enable_code_execution was called.")
            else:
                print("FAILURE: enable_code_execution was NOT called.")

    # Test 4: General Agent Tool Enablement
    print("\nTest 4: General Agent (Expect Google Search Enabled)")
    with patch.dict(os.environ, {}, clear=True):
        if 'api.graph.agents.general_agent' in sys.modules: del sys.modules['api.graph.agents.general_agent']
        from api.graph.agents.general_agent import general_agent
        
        with patch('api.graph.agents.general_agent.Gemini3Client') as mock_gemini:
            try:
                general_agent({"messages": []})
            except Exception:
                pass
            
            if mock_gemini.return_value.enable_google_search.called:
                print("SUCCESS: enable_google_search was called.")
            else:
                print("FAILURE: enable_google_search was NOT called.")

if __name__ == "__main__":
    try:
        test_agent_config()
    except Exception as e:
        print(f"An error occurred: {e}")
