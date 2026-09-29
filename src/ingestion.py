"""ETL: PDF -> chunks -> embeddings -> Pinecone.

Usage:
    python -m src.ingestion --download          # download PDF, then ingest
    python -m src.ingestion                     # ingest existing PDF
    python -m src.ingestion --reset             # drop & rebuild the index
"""
import argparse
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from src import config
from src.models import get_embeddings


def download_pdf(dest: Path = config.PDF_PATH) -> Path:
    """Fetch the eBook from Google Drive into data/."""
    import gdown

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = gdown.download(id=config.PDF_DRIVE_ID, output=str(dest), quiet=False)
    if not out or not dest.exists():
        raise RuntimeError(
            "Automatic download failed. Download the PDF manually from the "
            f"Google Drive link and save it to {dest}."
        )
    return dest


def ensure_index(pc: Pinecone, name: str, reset: bool = False) -> None:
    """Create the serverless index (1536-dim, cosine) if it doesn't exist."""
    existing = pc.list_indexes().names()
    if name in existing and reset:
        print(f"Deleting existing index '{name}'...")
        pc.delete_index(name)
        existing = pc.list_indexes().names()

    if name not in existing:
        print(f"Creating index '{name}' (dim={config.EMBEDDING_DIM}, cosine)...")
        pc.create_index(
            name=name,
            dimension=config.EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(1)
    else:
        print(f"Index '{name}' already exists - upserting into it.")


def run_ingestion(pdf_path: Path, index_name: str, reset: bool = False) -> PineconeVectorStore:
    config.require_env()
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"{pdf_path} not found. Run with --download or place the PDF there manually."
        )

    # 1. Load
    docs = PyPDFLoader(str(pdf_path)).load()
    print(f"Loaded {len(docs)} pages.")

    # 2. Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
    print(f"Split into {len(chunks)} chunks.")

    # 3. Index setup
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    ensure_index(pc, index_name, reset=reset)

    # 4. Embed + upsert in small batches with retry (friendly to free-tier rate limits).
    #    Deterministic ids => re-running is idempotent.
    store = PineconeVectorStore(index_name=index_name, embedding=get_embeddings())
    ids = [f"chunk-{i:05d}" for i in range(len(chunks))]
    size = config.EMBED_BATCH_SIZE
    for start in range(0, len(chunks), size):
        batch, batch_ids = chunks[start : start + size], ids[start : start + size]
        _with_retry(lambda: store.add_documents(batch, ids=batch_ids))
        print(f"  upserted {min(start + size, len(chunks))}/{len(chunks)}")
        time.sleep(config.EMBED_BATCH_PAUSE)
    print(f"Done. {len(chunks)} vectors in '{index_name}'.")
    return store


def _with_retry(fn, attempts: int = 5):
    """Retry with exponential backoff (handles 429 / transient errors)."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if i == attempts - 1:
                raise
            wait = 2 ** (i + 1)
            print(f"  retrying in {wait}s after error: {str(exc)[:120]}")
            time.sleep(wait)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the Agentic AI eBook into Pinecone.")
    parser.add_argument("--pdf", default=str(config.PDF_PATH), help="Path to the PDF")
    parser.add_argument("--index", default=config.PINECONE_INDEX_NAME, help="Pinecone index name")
    parser.add_argument("--download", action="store_true", help="Download the PDF first")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate the index")
    args = parser.parse_args()

    if args.download or not Path(args.pdf).exists():
        download_pdf(Path(args.pdf))
    run_ingestion(Path(args.pdf), args.index, reset=args.reset)


if __name__ == "__main__":
    main()
