#!/usr/bin/env python3
"""
Test script for ZeroGPU LLM Inference API
Usage: python test_api.py
"""

import requests
import json
import sys

API_URL = "https://Alovestocode-ZeroGPU-LLM-Inference.hf.space"

def test_api():
    """Test the API endpoint"""
    print("=" * 60)
    print("Testing ZeroGPU LLM Inference API")
    print("=" * 60)
    
    # Test 1: Check if space is accessible
    print("\n1. Checking if space is accessible...")
    try:
        response = requests.get(API_URL, timeout=10)
        if response.status_code == 200:
            print("   ✅ Space is accessible")
        else:
            print(f"   ⚠️  Space returned status {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 2: Check API info
    print("\n2. Checking API info...")
    try:
        response = requests.get(f"{API_URL}/api/info", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ API info endpoint accessible")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    # Test 3: Try the API endpoint
    print("\n3. Testing API endpoint...")
    payload = {
        "data": [
            "Solve a quadratic equation using Python",
            "",
            "- Provide step-by-step solution",
            "",
            "intermediate",
            "math, python",
            "Router-Qwen3-32B-8bit",
            256,  # Small token count for quick test
            0.2,
            0.9
        ],
        "fn_index": 0
    }
    
    try:
        print(f"   Sending request to {API_URL}/api/predict...")
        response = requests.post(
            f"{API_URL}/api/predict",
            json=payload,
            timeout=120  # Longer timeout for model loading
        )
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ API is working!")
            result = response.json()
            print(f"\n   Response structure:")
            if isinstance(result, dict):
                print(f"   Keys: {list(result.keys())}")
                if "data" in result:
                    print(f"   Data length: {len(result['data'])}")
                    if len(result['data']) > 0:
                        print(f"   First output preview: {str(result['data'][0])[:200]}")
            else:
                print(f"   Result: {str(result)[:300]}")
            return True
        else:
            print(f"   ❌ API returned status {response.status_code}")
            print(f"   Response: {response.text[:500]}")
            return False
            
    except requests.exceptions.Timeout:
        print("   ⚠️  Request timed out (this might be normal for first request due to model loading)")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_api()
    print("\n" + "=" * 60)
    if success:
        print("✅ API test completed successfully!")
    else:
        print("⚠️  API test had issues. The space might still be building.")
        print("   Wait a few minutes and try again, or check the space status at:")
        print(f"   {API_URL}")
    print("=" * 60)
    sys.exit(0 if success else 1)

