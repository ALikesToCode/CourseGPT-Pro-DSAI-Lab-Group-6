#!/usr/bin/env python3
"""Test script for Router API endpoints."""

import requests
import json
import time
import sys

BASE_URL = "https://Alovestocode-router-router-zero.hf.space"

def test_healthcheck():
    """Test the health check endpoint."""
    print("Testing GET /health...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print(f"Content-Type: {response.headers.get('content-type', 'unknown')}")
        print(f"Response length: {len(response.text)} bytes")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"Response: {json.dumps(data, indent=2)}")
                return True
            except json.JSONDecodeError:
                print(f"Not JSON. First 200 chars: {response.text[:200]}")
                return False
        else:
            print(f"Error: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generate():
    """Test the generate endpoint."""
    print("\nTesting POST /v1/generate...")
    try:
        payload = {
            "prompt": "You are a router agent. User query: What is 2+2?",
            "max_new_tokens": 100,
            "temperature": 0.2,
            "top_p": 0.9
        }
        response = requests.post(
            f"{BASE_URL}/v1/generate",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120  # Longer timeout for model loading
        )
        print(f"Status: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"Response keys: {list(result.keys())}")
                if "text" in result:
                    print(f"Generated text (first 300 chars): {result['text'][:300]}...")
                else:
                    print(f"Full response: {json.dumps(result, indent=2)}")
                return True
            except json.JSONDecodeError:
                print(f"Not JSON. First 200 chars: {response.text[:200]}")
                return False
        else:
            print(f"Error: {response.text[:500]}")
            return False
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradio_api():
    """Test Gradio's built-in API endpoint."""
    print("\nTesting Gradio API /api/predict...")
    try:
        # Gradio creates /api/predict endpoints automatically
        # We need to find the function index - usually 0 for the first function
        payload = {
            "data": [
                "You are a router agent. User query: What is 2+2?",
                100,
                0.2,
                0.9
            ],
            "fn_index": 0
        }
        response = requests.post(
            f"{BASE_URL}/api/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Gradio API Response: {json.dumps(result, indent=2)[:500]}...")
            return True
        else:
            print(f"Error: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"Exception: {e}")
        return False

def test_root():
    """Test the root endpoint."""
    print("\nTesting GET / (root)...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type', 'unknown')}")
        print(f"Response length: {len(response.text)} chars")
        if response.status_code == 200:
            return True
        return False
    except Exception as e:
        print(f"Exception: {e}")
        return False

def main():
    """Run all API tests."""
    print("=" * 60)
    print("Router API Test Suite")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}\n")
    
    # Wait a moment for Space to be ready
    print("Waiting 3 seconds for Space to be ready...")
    time.sleep(3)
    
    results = []
    
    # Test endpoints
    results.append(("Root", test_root()))
    results.append(("Health Check", test_healthcheck()))
    results.append(("Generate", test_generate()))
    results.append(("Gradio API", test_gradio_api()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(result[1] for result in results)
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
