# LangChain Documents Chat

A Streamlit app for chatting with your documents using retrieval-augmented generation (RAG). Upload PDF, TXT, or DOCX files, index them locally with FAISS, and ask questions with conversational memory.

## Features

- Multi-format upload: PDF, TXT, and DOCX
- Local vector search with FAISS
- Conversational RAG with history-aware retrieval
- Cost-optimized defaults: `gpt-4o-mini` + `text-embedding-3-small`

## Requirements

- Python 3.11+
- OpenAI API key

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:

```bash
cp .env.example .env
```

Edit `.env` and set your `OPENAI_API_KEY`.

## Run

```bash
streamlit run app.py
```

## Configuration

All settings are optional except `OPENAI_API_KEY`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat model for answers |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for indexing |
| `OPENAI_TEMPERATURE` | `0` | Lower = more factual |
| `RETRIEVAL_K` | `4` | Number of chunks retrieved per question |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |

For maximum savings, try `OPENAI_CHAT_MODEL=gpt-4.1-nano` in `.env`.

## Cost notes

- Embeddings run once when you click **Process documents** (very low cost with `text-embedding-3-small`).
- Chat queries dominate ongoing cost; `gpt-4o-mini` is the default balance of quality and price.

## Project structure

```
app.py                      # Streamlit UI
config.py                   # Environment settings
services/
  document_loader.py        # PDF / TXT / DOCX extraction
  ingestion.py              # Chunking, embeddings, FAISS
  rag_chain.py              # History-aware RAG chain
```

## Usage

1. Upload one or more documents in the sidebar.
2. Click **Process documents** and wait for indexing to finish.
3. Ask questions in the chat. Follow-up questions use conversation history.

Re-processing new uploads clears the chat and replaces the index.
