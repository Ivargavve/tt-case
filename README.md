# 🔍 Mums AB Document Search Engine

Mums AB wants to build an internal search engine to quickly find relevant information in a collection of PDF documents. The PDFs can contain everything from product data to organizational structure and they want to be able to search through them using natural language queries.

## 🎯 Task

Create a search engine that pass a number of tests defined under `tests/`. You may also expand the test suit with tests that you believe is relevant for a search engine like this. You can use any technology within reasonable limits, but the goal is to demonstrate how you:

- Write code that is sustainable, testable, portable, and extensible.
- Understand LLM/RAG concepts.
- Understand efficient token usage.

### ✨ Features

- Natural language search queries
- Semantic search using embeddings
- Document chunking and context preservation
- Relevance-based result ranking
- Support for PDF documents
- Efficient token management

### 🛠️ Technical Details

The search engine should be built using the following technologies and concepts:

- **RAG (Retrieval Augmented Generation)**: Combines document retrieval with LLM capabilities
- **Vector Database**: Stores and indexes document embeddings for efficient semantic search

### ⚠️ What to Avoid

- Don't build a solution tailored to specific PDF
- Don't hardcode answers to specific questions
- Don't assume specific document structure or content
- Don't create solutions that only work with the provided test documents


## 🚀 Setup

1. Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root folder and add your OpenAI API key. **Make sure to handle your API key with care.**
```
OPENAI_API_KEY="sk-proj-hb8..."
```

## 🧪 Tests

Run the tests with:

```bash
python3 -m pytest tests/
```
