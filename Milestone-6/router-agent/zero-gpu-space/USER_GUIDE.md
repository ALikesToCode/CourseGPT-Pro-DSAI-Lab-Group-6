# 📖 User Guide - ZeroGPU LLM Inference

## Quick Start (5 Minutes)

### 1. Choose Your Model
The model dropdown shows 30+ options organized by size:
- **Compact (<2B)**: Fast, lightweight - great for quick responses
- **Mid-size (2-8B)**: Best balance of speed and quality
- **Large (14B+)**: Highest quality, slower but more capable

**Recommendation for beginners**: Start with `Qwen3-4B-Instruct-2507`

### 2. Try an Example Prompt
Click on any example below the chat box to get started:
- "Explain quantum computing in simple terms"
- "Write a Python function..."
- "What are the latest developments..." (requires web search)

### 3. Start Chatting!
Type your message and press Enter or click "📤 Send"

## Core Features

### 💬 Chat Interface

The main chat area shows:
- Your messages on one side
- AI responses with a 🤖 avatar
- Copy button on each message
- Smooth streaming as tokens generate

**Tips:**
- Press Enter to send (Shift+Enter for new line)
- Click Copy button to save responses
- Scroll up to review history
- Use Clear Chat to start fresh

### 🤖 Model Selection

**When to use each size:**

| Model Size | Best For | Speed | Quality |
|------------|----------|-------|---------|
| <2B | Quick questions, testing | ⚡⚡⚡ | ⭐⭐ |
| 2-8B | General chat, coding help | ⚡⚡ | ⭐⭐⭐ |
| 14B+ | Complex reasoning, long-form | ⚡ | ⭐⭐⭐⭐ |

**Specialized Models:**
- **Phi-4-mini-Reasoning**: Math, logic problems
- **Qwen2.5-Coder**: Programming tasks
- **DeepSeek-R1-Distill**: Step-by-step reasoning
- **Apriel-1.5-15b-Thinker**: Multimodal understanding

### 🔍 Web Search

Enable this when you need:
- Current events and news
- Recent information (after model training cutoff)
- Facts that change frequently
- Real-time data

**How it works:**
1. Toggle "🔍 Enable Web Search"
2. Web search settings accordion appears
3. System prompt updates automatically
4. Search runs in background (won't block chat)
5. Results injected into context

**Settings explained:**
- **Max Results**: How many search results to fetch (4 is good default)
- **Max Chars/Result**: Limit length per result (50 prevents overwhelming context)
- **Search Timeout**: Maximum wait time (5s recommended)

### 📝 System Prompt

This defines the AI's personality and behavior.

**Default prompts:**
- Without search: Helpful, creative assistant
- With search: Includes search results and current date

**Customization ideas:**
```
You are a professional code reviewer...
You are a creative writing coach...
You are a patient tutor explaining concepts simply...
You are a technical documentation writer...
```

## Advanced Features

### 🎛️ Advanced Generation Parameters

Click the accordion to reveal these controls:

#### Max Tokens (64-16384)
- **What it does**: Sets maximum response length
- **Lower (256-512)**: Quick, concise answers
- **Medium (1024)**: Balanced (default)
- **Higher (2048+)**: Long-form content, detailed explanations

#### Temperature (0.1-2.0)
- **What it does**: Controls randomness/creativity
- **Low (0.1-0.3)**: Focused, deterministic (good for facts, code)
- **Medium (0.7)**: Balanced creativity (default)
- **High (1.2-2.0)**: Very creative, unpredictable (stories, brainstorming)

#### Top-K (1-100)
- **What it does**: Limits token choices to top K most likely
- **Lower (10-20)**: More focused
- **Medium (40)**: Balanced (default)
- **Higher (80-100)**: More varied vocabulary

#### Top-P (0.1-1.0)
- **What it does**: Nucleus sampling threshold
- **Lower (0.5-0.7)**: Conservative choices
- **Medium (0.9)**: Balanced (default)
- **Higher (0.95-1.0)**: Full vocabulary range

#### Repetition Penalty (1.0-2.0)
- **What it does**: Reduces repeated words/phrases
- **Low (1.0-1.1)**: Allows some repetition
- **Medium (1.2)**: Balanced (default)
- **High (1.5+)**: Strongly avoids repetition (may hurt coherence)

### Preset Configurations

**For Creative Writing:**
```
Temperature: 1.2
Top-P: 0.95
Top-K: 80
Max Tokens: 2048
```

**For Code Generation:**
```
Temperature: 0.3
Top-P: 0.9
Top-K: 40
Max Tokens: 1024
Repetition Penalty: 1.1
```

**For Factual Q&A:**
```
Temperature: 0.5
Top-P: 0.85
Top-K: 30
Max Tokens: 512
Enable Web Search: Yes
```

**For Reasoning Tasks:**
```
Model: Phi-4-mini-Reasoning or DeepSeek-R1
Temperature: 0.7
Max Tokens: 2048
```

## Tips & Tricks

### 🎯 Getting Better Results

1. **Be Specific**: "Write a Python function to sort a list" → "Write a Python function that sorts a list of dictionaries by a specific key"

2. **Provide Context**: "Explain recursion" → "Explain recursion to someone learning programming for the first time, with a simple example"

3. **Use System Prompts**: Define role/expertise in system prompt instead of every message

4. **Iterate**: Use follow-up questions to refine responses

5. **Experiment with Models**: Try different models for the same task

### ⚡ Performance Tips

1. **Start Small**: Test with smaller models first
2. **Adjust Max Tokens**: Don't request more than you need
3. **Use Cancel**: Stop bad generations early
4. **Clear Cache**: Clear chat if experiencing slowdowns
5. **One Task at a Time**: Don't send multiple requests simultaneously

### 🔍 When to Use Web Search

**✅ Good use cases:**
- "What happened in the latest SpaceX launch?"
- "Current cryptocurrency prices"
- "Recent AI research papers"
- "Today's weather in Paris"

**❌ Don't need search for:**
- General knowledge questions
- Code writing/debugging
- Math problems
- Creative writing
- Theoretical explanations

### 💭 Understanding Thinking Mode

Some models output `<think>...</think>` blocks:

```
<think>
Let me break this down step by step...
First, I need to consider...
</think>

Here's the answer: ...
```

**In the UI:**
- Thinking shows as "💭 Thought"
- Answer shows separately
- Helps you see the reasoning process

**Best for:**
- Complex math problems
- Multi-step reasoning
- Debugging logic
- Learning how AI thinks

## Troubleshooting

### Generation is Slow
- Try a smaller model
- Reduce Max Tokens
- Disable web search if not needed
- Clear chat history

### Responses are Repetitive
- Increase Repetition Penalty
- Reduce Temperature slightly
- Try different model

### Responses are Random/Nonsensical
- Decrease Temperature
- Reduce Top-P
- Reduce Top-K
- Try more stable model

### Web Search Not Working
- Check timeout isn't too short
- Verify internet connection
- Try increasing Max Results
- Check search query in debug panel

### Cancel Button Doesn't Work
- Wait a moment (might be processing)
- Refresh page if persists
- Check browser console for errors

## Keyboard Shortcuts

- **Enter**: Send message
- **Shift+Enter**: New line in input
- **Ctrl+C**: Copy (when text selected)
- **Ctrl+A**: Select all in input

## Best Practices

### For Beginners
1. Start with example prompts
2. Use default settings initially
3. Try 2-4 different models
4. Gradually explore advanced settings
5. Read responses fully before replying

### For Power Users
1. Create custom system prompts
2. Fine-tune parameters per task
3. Use debug panel for prompt engineering
4. Experiment with model combinations
5. Utilize web search strategically

### For Developers
1. Study the debug output
2. Test code generation thoroughly
3. Use lower temperature for determinism
4. Compare multiple models
5. Save working configurations

## Privacy & Safety

- **No data collection**: Conversations not stored permanently
- **Model limitations**: May produce incorrect information
- **Verify important info**: Don't rely solely on AI for critical decisions
- **Web search**: Uses DuckDuckGo (privacy-focused)
- **Open source**: Code is transparent and auditable

## Support & Feedback

Found a bug? Have a suggestion?
- Check GitHub issues
- Submit feature requests
- Contribute improvements
- Share your use cases

---

**Happy chatting! 🎉**
