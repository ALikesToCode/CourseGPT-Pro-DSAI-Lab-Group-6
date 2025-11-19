# CourseGPT Pro - User Guide

Welcome to CourseGPT Pro! This guide will help you use the AI educational assistant system effectively.

---

## Table of Contents

1. [What is CourseGPT Pro?](#1-what-is-coursegpt-pro)
2. [Getting Started](#2-getting-started)
3. [How to Ask Questions](#3-how-to-ask-questions)
4. [Uploading Documents](#4-uploading-documents)
5. [Understanding Responses](#5-understanding-responses)
6. [Example Use Cases](#6-example-use-cases)
7. [Tips for Best Results](#7-tips-for-best-results)
8. [Troubleshooting](#8-troubleshooting)
9. [Frequently Asked Questions](#9-frequently-asked-questions)

---

## 1. What is CourseGPT Pro?

CourseGPT Pro is your AI-powered study companion that helps with:

- 📚 **General Education**: Explanations of concepts across various subjects
- 💻 **Programming Help**: Code examples, debugging, and programming tutorials
- ➗ **Math Problems**: Step-by-step solutions with detailed explanations
- 📄 **Document Q&A**: Ask questions about your uploaded course materials

### How It Works

1. You ask a question (with or without uploading a document)
2. Our AI analyzes your query and routes it to the right specialist
3. The specialized AI agent provides a tailored response
4. You can continue the conversation with follow-up questions

---

## 2. Getting Started

### Accessing CourseGPT Pro

**Option 1: Web Interface**
- Visit: [Your deployment URL]
- No account required for basic use
- Upload documents and ask questions instantly

**Option 2: API Access**
- For developers and advanced users
- See [API Documentation](api_doc.md) for details

### First Steps

1. **Start Simple**: Begin with a straightforward question
   - Example: "What is photosynthesis?"

2. **Try Different Topics**: Test programming, math, or general questions
   - Programming: "How do I create a list in Python?"
   - Math: "Solve: x² + 5x + 6 = 0"
   - General: "Explain the causes of World War I"

3. **Upload a Document** (optional): Add course materials for context
   - Supported formats: PDF
   - Max size: 10MB (recommended)

---

## 3. How to Ask Questions

### Input Format

**Basic Question:**
```
Your Question: "Explain Newton's laws of motion"
```

**Question with Document:**
```
1. Upload your lecture notes (PDF)
2. Ask: "Summarize the key points from page 5"
```

**Follow-up Questions:**
```
First: "What is recursion in programming?"
Follow-up: "Can you show me an example in Python?"
Follow-up: "What are the advantages and disadvantages?"
```

### What You Can Ask

#### ✅ Good Questions

- **Specific**: "How do I implement a binary search in Java?"
- **Clear**: "Explain the Pythagorean theorem with an example"
- **Contextual**: "Based on the uploaded PDF, what are the main topics in Chapter 3?"

#### ❌ Avoid

- **Too Vague**: "Help me with coding"
- **Multiple Unrelated Questions**: "Explain calculus AND help me debug my code AND write an essay"
- **Inappropriate Content**: Offensive, harmful, or unethical requests

---

## 4. Uploading Documents

### Supported File Types

Currently supported:
- ✅ **PDF** - Lecture notes, textbooks, research papers

### How to Upload

**Step 1: Prepare Your Document**
- Ensure file is under 10MB
- Text-based PDFs work best
- Scanned PDFs supported (with OCR)

**Step 2: Upload**
- Click "Upload File" or "Choose File"
- Select your PDF
- Wait for upload confirmation

**Step 3: Ask Questions**
- Reference the document in your question
- Examples:
  - "What are the key formulas in this document?"
  - "Summarize section 2"
  - "Explain the diagram on page 7"

### Document Processing

**What Happens Behind the Scenes:**
1. Text is extracted from your PDF
2. Content is analyzed for relevance
3. AI uses document context to answer your question
4. Your document is NOT stored permanently (privacy-friendly)

---

## 5. Understanding Responses

### Response Types

#### Programming Responses

**Format:**
- Clear explanation
- Code examples with syntax highlighting
- Comments explaining each step

**Example:**
```
Question: "How do I sort a list in Python?"

Response:
"You can sort a list in Python using the built-in `sort()` method or `sorted()` function.

**Method 1: In-place sorting with sort()**
```python
my_list = [3, 1, 4, 1, 5, 9, 2]
my_list.sort()
print(my_list)  # Output: [1, 1, 2, 3, 4, 5, 9]
```

**Method 2: Create new sorted list with sorted()**
```python
my_list = [3, 1, 4, 1, 5, 9, 2]
sorted_list = sorted(my_list)
print(sorted_list)  # Output: [1, 1, 2, 3, 4, 5, 9]
```

**Key Differences:**
- `sort()` modifies the original list
- `sorted()` returns a new list, leaving the original unchanged
```
"
```

#### Math Responses

**Format:**
- Step-by-step solution
- Clear explanations at each step
- Mathematical notation (LaTeX)

**Example:**
```
Question: "Solve: 2x + 5 = 15"

Response:
"Let's solve this equation step by step:

**Step 1:** Subtract 5 from both sides
2x + 5 - 5 = 15 - 5
2x = 10

**Step 2:** Divide both sides by 2
2x / 2 = 10 / 2
x = 5

**Solution:** x = 5

**Verification:**
2(5) + 5 = 10 + 5 = 15 ✓
"
```

#### General Education Responses

**Format:**
- Clear definitions
- Explanations with examples
- Structured information

**Example:**
```
Question: "What is photosynthesis?"

Response:
"Photosynthesis is the process by which plants convert light energy into chemical energy.

**Key Components:**
1. **Inputs:**
   - Carbon dioxide (CO₂) from air
   - Water (H₂O) from soil
   - Light energy from sun

2. **Process:**
   - Occurs in chloroplasts
   - Chlorophyll absorbs light
   - Chemical reactions convert CO₂ and H₂O

3. **Outputs:**
   - Glucose (C₆H₁₂O₆) - food for plant
   - Oxygen (O₂) - released into air

**Formula:**
6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂
"
```

### Response Time

- **Simple questions**: 1-3 seconds
- **Complex questions**: 3-7 seconds
- **With document upload**: 5-10 seconds
- **Long documents**: Up to 15 seconds

---

## 6. Example Use Cases

### Use Case 1: Homework Help

**Scenario:** Student needs help with algebra homework

**Interaction:**
```
Student: "Solve the quadratic equation: x² - 7x + 12 = 0"

CourseGPT: [Provides step-by-step factoring solution]

Student: "Can you show me how to solve it using the quadratic formula too?"

CourseGPT: [Demonstrates quadratic formula method]
```

### Use Case 2: Coding Tutorial

**Scenario:** Learning loops in Python

**Interaction:**
```
Student: "How do for loops work in Python?"

CourseGPT: [Explains syntax with simple example]

Student: "Can you show me a more complex example with nested loops?"

CourseGPT: [Provides nested loop example with explanation]

Student: "How is this different from a while loop?"

CourseGPT: [Compares for vs while loops]
```

### Use Case 3: Lecture Note Review

**Scenario:** Preparing for exam using lecture notes

**Interaction:**
```
Student: [Uploads lecture PDF on cell biology]

Student: "What are the main points about mitosis?"

CourseGPT: [Extracts and summarizes mitosis content from PDF]

Student: "Create a comparison table between mitosis and meiosis"

CourseGPT: [Generates comparison table]
```

### Use Case 4: Debugging Code

**Scenario:** Student's code has an error

**Interaction:**
```
Student: "My Python code gives 'IndexError: list index out of range'. Here's the code:
```python
numbers = [1, 2, 3]
print(numbers[3])
```
What's wrong?"

CourseGPT: "The error occurs because Python lists use 0-based indexing. Your list has 3 elements at indices [0, 1, 2], but you're trying to access index 3, which doesn't exist.

**Fix:**
```python
numbers = [1, 2, 3]
print(numbers[2])  # Accesses the last element (value: 3)
```

**Remember:** For a list of length n, valid indices are 0 to n-1."
```

### Use Case 5: Concept Explanation

**Scenario:** Understanding a difficult concept

**Interaction:**
```
Student: "What is Big O notation?"

CourseGPT: [Explains Big O with simple examples]

Student: "Can you give me examples comparing O(1), O(n), and O(n²)?"

CourseGPT: [Provides code examples for each]

Student: "Which sorting algorithms have O(n log n) complexity?"

CourseGPT: [Lists merge sort, quick sort, heap sort with explanations]
```

---

## 7. Tips for Best Results

### 📝 Writing Effective Questions

**DO:**
- ✅ Be specific about what you need
- ✅ Provide context (your level, what you've tried)
- ✅ Ask one main question at a time
- ✅ Use proper terminology when possible

**DON'T:**
- ❌ Ask extremely broad questions
- ❌ Expect the AI to do assignments for you
- ❌ Combine unrelated topics in one question

### 📄 Uploading Documents

**DO:**
- ✅ Use clear, text-based PDFs
- ✅ Reference specific sections or pages
- ✅ Ask focused questions about the content

**DON'T:**
- ❌ Upload unrelated documents
- ❌ Expect perfect results from low-quality scans
- ❌ Upload documents with sensitive personal information

### 🔄 Having Conversations

**DO:**
- ✅ Ask follow-up questions for clarification
- ✅ Request different approaches or explanations
- ✅ Ask for examples to understand better

**DON'T:**
- ❌ Switch topics abruptly without context
- ❌ Assume the AI remembers everything from previous sessions

---

## 8. Troubleshooting

### Common Issues

#### Issue: "No response" or long wait times

**Possible Causes:**
- Server is processing a complex query
- Large document upload
- Network connection issues

**Solutions:**
- Wait up to 15 seconds for complex queries
- Try refreshing the page
- Check your internet connection
- Simplify your question

#### Issue: Response doesn't match my question

**Possible Causes:**
- Question was unclear or ambiguous
- AI routed to wrong specialist

**Solutions:**
- Rephrase your question more specifically
- Add keywords (e.g., "Python code for..." or "Math problem:")
- Try breaking down into smaller questions

#### Issue: PDF upload fails

**Possible Causes:**
- File too large (>10MB)
- Unsupported file format
- Network issue

**Solutions:**
- Compress PDF or split into smaller files
- Ensure file is PDF format
- Try uploading again

#### Issue: Code examples don't work

**Possible Causes:**
- Code requires specific environment/libraries
- Version-specific syntax
- Copy-paste formatting issues

**Solutions:**
- Ask about required libraries or environment
- Specify your programming language version
- Check for proper indentation (especially in Python)

### Error Messages

| Error | Meaning | Solution |
|-------|---------|----------|
| "Service unavailable" | Backend is down | Try again in a few minutes |
| "Invalid file type" | Unsupported file format | Use PDF files only |
| "Request timeout" | Query took too long | Simplify question or try again |
| "Rate limit exceeded" | Too many requests | Wait a minute and try again |

---

## 9. Frequently Asked Questions

### General Questions

**Q: Is my data private?**
A: Yes. Uploaded documents are processed in memory and not permanently stored. Conversations are isolated by session and user ID.

**Q: Can I use this for exams or tests?**
A: CourseGPT Pro is designed for learning and homework help. Using it during closed-book exams may violate academic integrity policies. Check your institution's rules.

**Q: What subjects can I ask about?**
A: Programming, mathematics, sciences, humanities, and general education topics. The system specializes in technical subjects but can handle a wide range of educational queries.

**Q: Can I get help in languages other than English?**
A: Currently, the system is optimized for English queries. Support for other languages may vary.

### Technical Questions

**Q: What programming languages are supported?**
A: The system can help with Python, JavaScript, Java, C++, C, Go, and many others. Python has the most extensive training data.

**Q: Can it solve any math problem?**
A: The system handles a wide range of math from elementary to college level, including algebra, calculus, statistics, and geometry. Very advanced graduate-level mathematics may have limitations.

**Q: What if the code doesn't run?**
A: Ask for clarification or debugging help. Specify your error message and the AI can help troubleshoot.

**Q: Can I ask multiple questions in one session?**
A: Yes! You can have multi-turn conversations. Each follow-up maintains context from previous messages in the same session.

### Usage Questions

**Q: How many questions can I ask?**
A: There may be rate limits to ensure fair use (typically 10-20 requests per minute). Check with your system administrator.

**Q: Can I save conversations?**
A: This depends on your interface. Some implementations may offer conversation history. Check your specific deployment's features.

**Q: How do I report a problem or give feedback?**
A: Contact your system administrator or use the feedback form if available in your interface.

**Q: Can I use this on my phone?**
A: Yes, if you're accessing via a web interface. The system is responsive and works on mobile browsers.

---

## Getting Help

### Support Resources

- 📖 **Technical Documentation**: [technical_doc.md](technical_doc.md)
- 🔌 **API Guide**: [api_doc.md](api_doc.md)
- 📋 **Project Overview**: [overview.md](overview.md)

### Contact

- **Issues & Bugs**: [GitHub Issues Page]
- **General Questions**: [Your support email]
- **Feature Requests**: [Feedback form or email]

---

## Happy Learning! 🎓

We hope CourseGPT Pro helps make your learning journey more effective and enjoyable. Remember:

- 🎯 Ask specific questions
- 📚 Use documents for context
- 💬 Follow up for clarification
- 🔄 Practice explaining concepts back to solidify understanding

*Good luck with your studies!*

---

*Last Updated: 2025-01-19*
*User Guide Version: 1.0*
