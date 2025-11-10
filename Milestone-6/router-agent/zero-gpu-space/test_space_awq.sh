#!/bin/bash
# Quick test script to verify ZeroGPU Space is working with AWQ models

API_URL="https://Alovestocode-ZeroGPU-LLM-Inference.hf.space"

echo "=========================================="
echo "Testing ZeroGPU Space with AWQ Models"
echo "=========================================="

echo -e "\n1. Checking Space status..."
SPACE_STATUS=$(curl -s "https://huggingface.co/api/spaces/Alovestocode/ZeroGPU-LLM-Inference/status" | python3 -c "import sys, json; print(json.load(sys.stdin).get('runtime', {}).get('stage', 'UNKNOWN'))" 2>/dev/null)
echo "   Space Status: $SPACE_STATUS"

if [ "$SPACE_STATUS" != "RUNNING" ]; then
    echo "   ⚠️  Space is not RUNNING (Status: $SPACE_STATUS)"
    echo "   The Space may still be building or sleeping."
    echo "   Wait a few minutes and try again."
    exit 1
fi

echo "   ✅ Space is RUNNING"

echo -e "\n2. Checking Space accessibility..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL")
if [ "$HTTP_CODE" -eq 200 ]; then
    echo "   ✅ Space is accessible (HTTP $HTTP_CODE)"
else
    echo "   ❌ Space returned HTTP $HTTP_CODE"
    exit 1
fi

echo -e "\n3. Checking API info endpoint..."
API_INFO=$(curl -s "${API_URL}/api/info" 2>/dev/null | head -c 200)
if [ -n "$API_INFO" ]; then
    echo "   ✅ API info endpoint accessible"
    echo "   Response preview: ${API_INFO:0:100}..."
else
    echo "   ⚠️  API info endpoint not accessible (may be normal)"
fi

echo -e "\n=========================================="
echo "✅ Basic connectivity test PASSED"
echo ""
echo "Space URL: $API_URL"
echo ""
echo "To test the API with Python:"
echo "  pip install gradio_client"
echo "  python test_api_gradio_client.py"
echo ""
echo "Expected models:"
echo "  - Router-Qwen3-32B-AWQ (Alovestocode/router-qwen3-32b-merged-awq)"
echo "  - Router-Gemma3-27B-AWQ (Alovestocode/router-gemma3-merged-awq)"
echo "=========================================="

