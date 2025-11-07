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
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Exception: {e}")
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
            timeout=60  # Longer timeout for model loading
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response keys: {list(result.keys())}")
            if "text" in result:
                print(f"Generated text (first 200 chars): {result['text'][:200]}...")
            else:
                print(f"Full response: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Exception: {e}")
        return False

def test_gradio_ui():
    """Test the Gradio UI endpoint."""
    print("\nTesting GET /gradio (UI redirect target)...")
    try:
        response = requests.get(f"{BASE_URL}/gradio", timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response length: {len(response.text)} chars")
            print(f"Response type: {response.headers.get('content-type', 'unknown')}")
            return True
        else:
            print(f"Error: {response.text[:200]}")
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
    print("Waiting 5 seconds for Space to be ready...")
    time.sleep(5)
    
    results = []
    
    # Test endpoints
    results.append(("Health Check", test_healthcheck()))
    results.append(("Generate", test_generate()))
    results.append(("Gradio UI", test_gradio_ui()))
    
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
