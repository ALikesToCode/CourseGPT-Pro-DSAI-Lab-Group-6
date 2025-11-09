#!/bin/bash
# Simple test script for Hugging Face Space (no Python dependencies)

SPACE_URL="https://alovestocode-zerogpu-llm-inference.hf.space"

echo "=========================================="
echo "Testing Hugging Face Space: ZeroGPU"
echo "=========================================="
echo ""

echo "1. Checking Space accessibility..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$SPACE_URL")
if [ "$HTTP_CODE" == "200" ]; then
    echo "   ✅ Space is accessible (HTTP $HTTP_CODE)"
else
    echo "   ❌ Space returned HTTP $HTTP_CODE"
    exit 1
fi

echo ""
echo "2. Checking API info endpoint..."
API_INFO=$(curl -s "$SPACE_URL/api/info" 2>&1)
if echo "$API_INFO" | grep -q "api_name\|endpoints"; then
    echo "   ✅ API info endpoint accessible"
    echo "$API_INFO" | python3 -m json.tool 2>/dev/null | head -20 || echo "   (Raw response received)"
else
    echo "   ⚠️  API info response: $API_INFO"
fi

echo ""
echo "3. Checking Space status..."
STATUS=$(curl -s "https://huggingface.co/api/spaces/Alovestocode/ZeroGPU-LLM-Inference" 2>&1 | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('runtime', {}).get('stage', 'UNKNOWN'))" 2>/dev/null || echo "UNKNOWN")
echo "   Space Status: $STATUS"

echo ""
echo "=========================================="
if [ "$HTTP_CODE" == "200" ]; then
    echo "✅ Basic connectivity test PASSED"
    echo ""
    echo "Space URL: $SPACE_URL"
    echo "To test the API with Python, install gradio_client:"
    echo "  pip install gradio_client"
    echo "  python test_api_gradio_client.py"
else
    echo "❌ Basic connectivity test FAILED"
fi
echo "=========================================="

