---
title: ZeroGPU-LLM-Inference
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Streaming LLM chat with web search and controls
---

# 🧠 ZeroGPU LLM Inference

A modern, user-friendly Gradio interface for **token-streaming, chat-style inference** across a wide variety of Transformer models—powered by ZeroGPU for free GPU acceleration on Hugging Face Spaces.

## ✨ Key Features

### 🎨 Modern UI/UX
- **Clean, intuitive interface** with organized layout and visual hierarchy
- **Collapsible advanced settings** for both simple and power users
- **Smooth animations and transitions** for better user experience
- **Responsive design** that works on all screen sizes
- **Copy-to-clipboard** functionality for easy sharing of responses

### 🔍 Web Search Integration
- **Real-time DuckDuckGo search** with background threading
- **Configurable timeout** and result limits
- **Automatic context injection** into system prompts
- **Smart toggle** - search settings auto-hide when disabled

### 💡 Smart Features
- **Thought vs. Answer streaming**: `<think>…</think>` blocks shown separately as "💭 Thought"
- **Working cancel button** - immediately stops generation without errors
- **Debug panel** for prompt engineering insights
- **Duration estimates** based on model size and settings
- **Example prompts** to help users get started
- **Dynamic system prompts** with automatic date insertion

### 🎯 Model Variety
- **30+ LLM options** from leading providers (Qwen, Microsoft, Meta, Mistral, etc.)
- Models ranging from **135M to 32B+** parameters
- Specialized models for **reasoning, coding, and general chat**
- **Efficient model loading** - one at a time with automatic cache clearing

### ⚙️ Advanced Controls
- **Generation parameters**: max tokens, temperature, top-k, top-p, repetition penalty
- **Web search settings**: max results, chars per result, timeout
- **Custom system prompts** with dynamic date insertion
- **Organized in collapsible sections** to keep interface clean

## 🔄 Supported Models

### Compact Models (< 2B)
- **SmolLM2-135M-Instruct** - Tiny but capable
- **SmolLM2-360M-Instruct** - Lightweight conversation
- **Taiwan-ELM-270M/1.1B** - Multilingual support
- **Qwen3-0.6B/1.7B** - Fast inference

### Mid-Size Models (2B-8B)
- **Qwen3-4B/8B** - Balanced performance
- **Phi-4-mini** (4.3B) - Reasoning & Instruct variants
- **MiniCPM3-4B** - Efficient mid-size
- **Gemma-3-4B-IT** - Instruction-tuned
- **Llama-3.2-Taiwan-3B** - Regional optimization
- **Mistral-7B-Instruct** - Classic performer
- **DeepSeek-R1-Distill-Llama-8B** - Reasoning specialist

### Large Models (14B+)
- **Qwen3-14B** - Strong general purpose
- **Apriel-1.5-15b-Thinker** - Multimodal reasoning
- **gpt-oss-20b** - Open GPT-style
- **Qwen3-32B** - Top-tier performance

## 🚀 How It Works

1. **Select Model** - Choose from 30+ pre-configured models
2. **Configure Settings** - Adjust generation parameters or use defaults
3. **Enable Web Search** (optional) - Get real-time information
4. **Start Chatting** - Type your message or use example prompts
5. **Stream Response** - Watch as tokens are generated in real-time
6. **Cancel Anytime** - Stop generation mid-stream if needed

### Technical Flow

1. User message enters chat history
2. If search enabled, background thread fetches DuckDuckGo results
3. Search snippets merge into system prompt (within timeout limit)
4. Selected model pipeline loads on ZeroGPU (bf16→f16→f32 fallback)
5. Prompt formatted with thinking mode detection
6. Tokens stream to UI with thought/answer separation
7. Cancel button available for immediate interruption
8. Memory cleared after generation for next request

## ⚙️ Generation Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Max Tokens | 64-16384 | 1024 | Maximum response length |
| Temperature | 0.1-2.0 | 0.7 | Creativity vs focus |
| Top-K | 1-100 | 40 | Token sampling pool size |
| Top-P | 0.1-1.0 | 0.9 | Nucleus sampling threshold |
| Repetition Penalty | 1.0-2.0 | 1.2 | Reduce repetition |

## 🌐 Web Search Settings

| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| Max Results | Integer | 4 | Number of search results |
| Max Chars/Result | Integer | 50 | Character limit per result |
| Search Timeout | 0-30s | 5s | Maximum wait time |

## 💻 Local Development

```bash
# Clone the repository
git clone https://huggingface.co/spaces/Luigi/ZeroGPU-LLM-Inference
cd ZeroGPU-LLM-Inference

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

## 🎨 UI Design Philosophy

The interface follows these principles:

1. **Simplicity First** - Core features immediately visible
2. **Progressive Disclosure** - Advanced options hidden but accessible
3. **Visual Hierarchy** - Clear organization with groups and sections
4. **Feedback** - Status indicators and helpful messages
5. **Accessibility** - Responsive, keyboard-friendly, with tooltips

## 🔧 Customization

### Adding New Models

Edit `MODELS` dictionary in `app.py`:

```python
"Your-Model-Name": {
    "repo_id": "org/model-name",
    "description": "Model description",
    "params_b": 7.0  # Size in billions
}
```

### Modifying UI Theme

Adjust theme parameters in `gr.Blocks()`:

```python
theme=gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="purple",
    # ... more options
)
```

## 📊 Performance

- **Token streaming** for responsive feel
- **Background search** doesn't block UI
- **Efficient memory** management with cache clearing
- **ZeroGPU acceleration** for fast inference
- **Optimized loading** with dtype fallbacks

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Additional model integrations
- UI/UX enhancements
- Performance optimizations
- Bug fixes and testing
- Documentation improvements

## 📝 License

Apache 2.0 - See LICENSE file for details

## 🙏 Acknowledgments

- Built with [Gradio](https://gradio.app)
- Powered by [Hugging Face Transformers](https://huggingface.co/transformers)
- Uses [ZeroGPU](https://huggingface.co/zero-gpu-explorers) for acceleration
- Search via [DuckDuckGo](https://duckduckgo.com)

---

**Made with ❤️ for the open source community**
