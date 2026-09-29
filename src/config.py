"""Environment setup & constants shared across the project."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent

# --- Provider: "gemini" (default) or "openai" -------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

# --- Credentials / services -------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# --- Models -----------------------------------------------------------------
# Gemini. "gemini-flash-latest" is Google's rolling alias for the current Flash
# model, so it keeps working as versions retire. Pin a specific name via env if wanted.
GEMINI_LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-flash-latest")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
# OpenAI
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# Both providers are configured to output 1536-d vectors, so the Pinecone index
# is 1536-d either way. (Gemini embeddings are truncated via output_dimensionality.)
EMBEDDING_DIM = 1536

# --- Ingestion --------------------------------------------------------------
PDF_DRIVE_ID = "15VLphKcY23_fpYxN62UEQRri_psRVfP9"
PDF_PATH = ROOT_DIR / "data" / "Ebook-Agentic-AI.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "50"))
EMBED_BATCH_PAUSE = float(os.getenv("EMBED_BATCH_PAUSE", "1.0"))  # seconds, helps free-tier rate limits

# --- Retrieval / grounding --------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "4"))
# Similarity scales differ per embedding model, so the default differs.
# Tune with:  python -m src.calibrate
_DEFAULT_THRESHOLD = "0.55" if LLM_PROVIDER == "gemini" else "0.30"
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", _DEFAULT_THRESHOLD))

REFUSAL_MESSAGE = "I cannot answer based on the provided document."


def require_env() -> None:
    """Fail fast with a readable message if credentials are missing."""
    if LLM_PROVIDER not in ("gemini", "openai"):
        raise RuntimeError("LLM_PROVIDER must be 'gemini' or 'openai'.")
    needed = [("PINECONE_API_KEY", PINECONE_API_KEY)]
    if LLM_PROVIDER == "gemini":
        needed.append(("GOOGLE_API_KEY", GOOGLE_API_KEY))
    else:
        needed.append(("OPENAI_API_KEY", OPENAI_API_KEY))
    missing = [n for n, v in needed if not v]
    if missing:
        raise RuntimeError(
            f"Missing environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and fill them in."
        )
