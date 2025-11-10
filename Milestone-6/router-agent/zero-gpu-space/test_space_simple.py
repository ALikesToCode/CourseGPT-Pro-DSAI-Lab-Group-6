#!/usr/bin/env python3
"""
Simple test script for ZeroGPU Space AWQ models
Tests basic connectivity and model availability without gradio_client
"""

import json
import urllib.request
import urllib.parse

API_URL = "https://Alovestocode-ZeroGPU-LLM-Inference.hf.space"

def test_space_status():
    """Test Space status via Hugging Face API"""
    print("=" * 60)
    print("Testing ZeroGPU Space with AWQ Models")
    print("=" * 60)
    
    print("\n1. Checking Space status via Hugging Face API...")
    try:
        url = "https://huggingface.co/api/spaces/Alovestocode/ZeroGPU-LLM-Inference"
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read())
            space_id = data.get('id', 'unknown')
            runtime_stage = data.get('runtime', {}).get('stage', 'unknown')
            hardware = data.get('hardware', {}).get('current', 'unknown')
            
            print(f"   Space ID: {space_id}")
            print(f"   Runtime Stage: {runtime_stage}")
            print(f"   Hardware: {hardware}")
            
            if runtime_stage == "RUNNING":
                print("   ✅ Space is RUNNING")
                return True
            else:
                print(f"   ⚠️  Space is {runtime_stage} (may be building or sleeping)")
                return False
    except Exception as e:
        print(f"   ❌ Failed to check status: {e}")
        return False

def test_space_connectivity():
    """Test basic HTTP connectivity to Space"""
    print("\n2. Testing Space HTTP connectivity...")
    try:
        req = urllib.request.Request(API_URL)
        req.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(req, timeout=10) as response:
            code = response.getcode()
            if code == 200:
                print(f"   ✅ Space is accessible (HTTP {code})")
                return True
            else:
                print(f"   ⚠️  Space returned HTTP {code}")
                return False
    except urllib.error.HTTPError as e:
        if e.code == 200:
            print(f"   ✅ Space is accessible (HTTP {e.code})")
            return True
        else:
            print(f"   ⚠️  Space returned HTTP {e.code}")
            return False
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False

def main():
    """Run all tests"""
    status_ok = test_space_status()
    connectivity_ok = test_space_connectivity()
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Space Status: {'✅ OK' if status_ok else '⚠️  Not Running'}")
    print(f"Connectivity: {'✅ OK' if connectivity_ok else '❌ Failed'}")
    
    print("\nExpected AWQ Models:")
    print("  - Router-Qwen3-32B-AWQ → Alovestocode/router-qwen3-32b-merged-awq")
    print("  - Router-Gemma3-27B-AWQ → Alovestocode/router-gemma3-merged-awq")
    
    print("\n" + "=" * 60)
    if status_ok and connectivity_ok:
        print("✅ Space is ready for testing!")
        print("\nTo test the API with full functionality:")
        print("  pip install gradio_client")
        print("  python test_api_gradio_client.py")
    elif connectivity_ok:
        print("⚠️  Space is accessible but may still be building")
        print("   Wait a few minutes and check again")
    else:
        print("❌ Space connectivity issues detected")
    print("=" * 60)

if __name__ == "__main__":
    main()

