# LangChain Documents Chat

A **Retrieval-Augmented Generation (RAG)** app built with LangChain and Streamlit. Upload your own documents, build a local vector index, and chat with an LLM that answers from retrieved context—not from memory alone.

## What is RAG?

**RAG** grounds every answer in your uploaded files. Instead of asking the model to guess, the app:

1. **Ingests** your PDF, TXT, or DOCX files and splits them into searchable chunks
2. **Embeds** those chunks with OpenAI and stores them in a local **FAISS** vector index
3. **Retrieves** the most relevant chunks for each question
4. **Generates** an answer using only that retrieved context

That means responses stay tied to your documents, with less hallucination and better follow-up handling via conversational memory.

## RAG features

- **Document-grounded answers** — the model is instructed to answer only from retrieved context
- **History-aware retrieval** — follow-up questions are rewritten so retrieval still works mid-conversation
- **Local vector search** — FAISS indexes run on your machine after processing
- **Multi-format upload** — PDF, TXT, and DOCX
- **Cost-optimized defaults** — `gpt-4o-mini` + `text-embedding-3-small`

## Requirements

- Python 3.11+ (3.12 supported)
- OpenAI API key

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
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

Restart Streamlit after installing new packages or changing `.env`.

## Configuration

All settings are optional except `OPENAI_API_KEY`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat model for RAG answers |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for the vector index |
| `OPENAI_TEMPERATURE` | `0` | Lower = more factual |
| `RETRIEVAL_K` | `4` | Chunks retrieved per question |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |

For maximum savings, try `OPENAI_CHAT_MODEL=gpt-4.1-nano` in `.env`.

## Cost notes

- Embeddings run once when you click **Process documents** (very low cost with `text-embedding-3-small`).
- Chat queries dominate ongoing cost; `gpt-4o-mini` is the default balance of quality and price.

## RAG pipeline

```
Upload docs → Chunk & embed → FAISS index
                                    ↓
User question → History-aware retrieval → Top-K chunks → LLM answer
```

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
2. Click **Process documents** to build the RAG index.
3. Ask questions in the chat. Follow-up questions use conversation history and fresh retrieval.

Re-processing new uploads clears the chat and replaces the index.
