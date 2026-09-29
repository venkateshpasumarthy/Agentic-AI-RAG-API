# RAG-based AI Chatbot: Agentic AI eBook

A Retrieval-Augmented Generation chatbot built with **Python, LangGraph, Pinecone, Gemini (or OpenAI) and FastAPI** (plus an optional Streamlit UI).
It answers questions **only** from the *Agentic AI* eBook and refuses anything outside it.

Every response returns: **the answer**, **the retrieved context chunks** (with page + similarity), and a **confidence score**.

---

## Architecture

```
                      ONE-TIME INGESTION (src/ingestion.py)
 PDF ──► PyPDFLoader ──► RecursiveCharacterTextSplitter ──► Gemini embeddings ──► Pinecone
        (per page)        (1000 chars / 200 overlap)       (gemini-embedding-001,   (1536-d, cosine)
                                                              1536-d)

                      QUERY TIME (src/graph.py, served by app.py)
 POST /chat ──► ┌──────────┐   best score ≥ threshold   ┌──────────┐
   {"query"}    │ retrieve │ ─────────────────────────► │ generate │ ──► END
                │ top-k    │                            │ (Gemini, │
                └──────────┘   best score < threshold   │ grounded)│
                     │        ┌──────────┐              └──────────┘
                     └──────► │  refuse  │ ──► END
                              └──────────┘
```

### Components

| File | Responsibility |
|---|---|
| `src/config.py` | Env loading, provider selection, model names, chunking params, retrieval threshold |
| `src/models.py` | Provider factory — returns the embedding model and chat LLM for `LLM_PROVIDER` (`gemini` or `openai`) |
| `src/ingestion.py` | Download PDF → load → chunk → create Pinecone index → embed & upsert in rate-limit-friendly batches (idempotent) |
| `src/graph.py` | LangGraph `StateGraph`: `AgentState`, `retrieve` / `generate` / `refuse` nodes, conditional routing |
| `src/calibrate.py` | Probes your live index to suggest a `RELEVANCE_THRESHOLD` for your embedding model |
| `app.py` | FastAPI app: `POST /chat`, `GET /health` |
| `streamlit_app.py` | Optional chat UI with a sidebar showing chunks + score |
| `tests_sample_queries.py` | 6 benchmark queries (5 in-document, 1 out-of-scope) with PASS/FAIL |

### Graph state

```python
class AgentState(TypedDict):
    question: str
    context: List[RetrievedChunk]   # {text, page, score}
    answer: str
    score: float                    # confidence in [0, 1]
```

### How strict grounding works (three layers)

1. **Retrieval gate**: if *no* retrieved chunk reaches `RELEVANCE_THRESHOLD`, the graph routes to `refuse` and the LLM is never called. This is cheap and prevents hallucination on off-topic questions.
2. **Prompt**: the LLM is told to use *only* the numbered context passages, with no outside knowledge.
3. **Structured output**: the LLM must return `answerable: bool`. If it decides the context is insufficient, we return the refusal message with score `0.0`.

### Confidence score

For grounded answers, `confidence_score` is the **mean cosine similarity of the chunks the answer was built from** (clamped to [0, 1]). Refusals always return `0.0`. It's a retrieval-based heuristic, not a calibrated probability, and the useful range differs by embedding model — see **Tuning** below.

### Why Gemini by default

This project uses **Gemini** (`gemini-embedding-001` for embeddings, `gemini-flash-latest` for chat) as the default provider, because that's the API key available for this assignment. OpenAI is kept as a drop-in alternative — see `LLM_PROVIDER` below. Both are configured to produce 1536-dimensional vectors so the same Pinecone index setup works either way.

### Design decisions

- **Provider factory (`src/models.py`)**: swapping `LLM_PROVIDER=gemini` ↔ `openai` in `.env` changes both the embedding model and the LLM with no code edits, and both target 1536 dimensions.
- **Deterministic chunk IDs** (`chunk-00001`…) make re-running ingestion an overwrite, not a duplicate.
- **Batched ingestion with retry/backoff**: embeddings are upserted in small batches with a pause between them and exponential-backoff retry, since Gemini's free tier has tighter rate limits than OpenAI's.
- **Dependency injection** in `build_rag_graph(vectorstore=..., llm=...)` allows testing the graph without live services.
- **Plain `def` endpoint** in FastAPI, because the graph makes blocking network calls and FastAPI will run it in a threadpool.
- **`pinecone` instead of `pinecone-client`**: the old package is deprecated and conflicts with `langchain-pinecone`.

---

## Setup

Requires Python 3.10+, a Gemini API key, and a (free) Pinecone API key.

```bash
git clone <your-repo-url> rag-agentic-ai && cd rag-agentic-ai

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # then edit .env with your keys
```

**Get a Gemini key**: https://aistudio.google.com/apikey (free tier available).
**Get a Pinecone key**: https://app.pinecone.io → API Keys (free tier available).

`.env`:
```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-index
```

## 1. Ingest the document

```bash
python -m src.ingestion --download
```

This downloads the eBook to `data/Ebook-Agentic-AI.pdf`, creates the Pinecone serverless index (1536-d, cosine) and upserts all chunks in small batches (pausing between batches to stay under Gemini's free-tier rate limits — expect this to take a few minutes for the full eBook).

If the automatic download fails (Google Drive rate limits), save the PDF manually at that path and run `python -m src.ingestion`.

Other flags: `--reset` (drop & rebuild the index), `--pdf PATH`, `--index NAME`.

If you hit `429` errors during ingestion, increase `EMBED_BATCH_PAUSE` in `.env` (e.g. `2.0`) and re-run — it's safe to re-run since chunk IDs are deterministic.

## 2. Calibrate the relevance threshold (recommended)

Cosine similarity ranges differ by embedding model, so run this once against your live index before trusting the default:

```bash
python -m src.calibrate
```

It prints the best similarity score for 4 in-document questions and 3 off-topic questions, and suggests a `RELEVANCE_THRESHOLD` to put in `.env`. The shipped default (`0.55` for Gemini) is a reasonable starting point but **not verified against the actual eBook**, since I don't have your document indexed.

## 3. Run the API

```bash
uvicorn app:app --reload
```
Interactive docs: http://localhost:8000/docs

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Agentic AI according to the eBook?"}'
```

Response shape:
```json
{
  "answer": "...",
  "retrieved_chunks": [
    {"text": "...", "page": 4, "score": 0.62}
  ],
  "confidence_score": 0.55
}
```

## 3b. (Optional) Streamlit UI

```bash
streamlit run streamlit_app.py
```

## 4. Run the test queries

With the API running:
```bash
python tests_sample_queries.py
```
Or without a server:
```bash
python tests_sample_queries.py --direct
```

| # | Query | Expected |
|---|---|---|
| 1 | What is Agentic AI according to the eBook? | Grounded answer |
| 2 | How do AI agents differ from traditional automation systems? | Grounded answer |
| 3 | What are the core components of an Agentic Architecture? | Grounded answer |
| 4 | What role does memory play in Agentic AI workflows? | Grounded answer |
| 5 | What are the main risks or challenges of deploying agentic systems? | Grounded answer |
| 6 | Who won the 2022 FIFA World Cup? | Refusal, `confidence_score = 0.0` |

The script prints the answer, confidence, retrieved chunks and a PASS/FAIL per query.

## Using OpenAI instead of Gemini

Set in `.env`:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key
```
Nothing else changes — `src/models.py` handles the switch. Re-run ingestion (`--reset`) if you switch providers after already indexing, since embeddings from different models aren't compatible in the same index.

## Tuning

| Env var | Default | Effect |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` or `openai` |
| `TOP_K` | `4` | Chunks retrieved per query |
| `RELEVANCE_THRESHOLD` | `0.55` (gemini) / `0.30` (openai) | Raise if off-topic questions slip through; lower if valid questions get refused. Use `python -m src.calibrate` to find your value |
| `GEMINI_LLM_MODEL` | `gemini-flash-latest` | Any Gemini chat model |
| `EMBED_BATCH_SIZE` / `EMBED_BATCH_PAUSE` | `50` / `1.0`s | Ingestion batching, tune down for stricter free-tier rate limits |

## Sample test run

Run against a live index (Gemini + Pinecone, `TOP_K=8`, `RELEVANCE_THRESHOLD=0.55`):

```
[1] What is Agentic AI according to the eBook?
    Result: FAIL (refused) — see "Known limitation" below

[2] How do AI agents differ from traditional automation systems?
    Confidence: 0.725 — PASS
    "Traditional automation systems... execute predefined rules and logic.
     In contrast, Agentic AI adapts to different situations, handles
     unstructured inputs, and is capable of autonomous decision-making..."

[3] What are the core components of an Agentic Architecture?
    Confidence: 0.758 — PASS
    "The core components of an Agentic AI system include Perception,
     Reasoning, Planning, Learning, and Execution."

[4] What role does memory play in Agentic AI workflows?
    Confidence: 0.737 — PASS
    "Memory in Agentic AI involves both long-term memory (LTM) and
     short-term memory (STM)..."

[5] What are the main risks or challenges of deploying agentic systems?
    Confidence: 0.755 — PASS
    "...communication and coordination difficulties, and conflict
     management issues..."

[6] Who won the 2022 FIFA World Cup?
    Confidence: 0.0 — PASS (correctly refused, best similarity 0.49 < threshold)

5/6 checks passed.
```

### Known limitation

Query [1] is refused even though the eBook does define Agentic AI (page 7 onward). The retriever keeps surfacing the cover page, table of contents, and section-header chunks instead of the actual defining paragraph, because those pages repeat the phrase "Agentic AI" heavily and score similarly high on cosine similarity without containing much substantive content. This is a known weakness of plain chunk-level retrieval on documents with repetitive titles/headers.

Two directions to fix it, not implemented here to keep the pipeline simple:
- **Header/boilerplate filtering** during ingestion (drop very short chunks, cover pages, TOC pages).
- **Reranking** the top-k chunks with a cross-encoder before generation, instead of relying on raw cosine similarity alone.

Both the retrieval gate and the LLM's `answerable` check worked as designed here: rather than guessing at a definition from a table of contents, the system correctly declined instead of hallucinating one.

## Troubleshooting

- **`Missing environment variables`**: `.env` is missing or not in the project root.
- **`429` during ingestion**: raise `EMBED_BATCH_PAUSE`/lower `EMBED_BATCH_SIZE` in `.env` and re-run (idempotent).
- **Index dimension mismatch**: the index must be 1536-d. Re-run ingestion with `--reset`.
- **Every question refused / everything answered**: run `python -m src.calibrate` and set `RELEVANCE_THRESHOLD` from its suggestion.
- **`ModuleNotFoundError: src`**: run commands from the repo root (`python -m src.ingestion`, not `python src/ingestion.py`).