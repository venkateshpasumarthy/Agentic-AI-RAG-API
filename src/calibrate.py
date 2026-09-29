"""Pick a RELEVANCE_THRESHOLD for your embedding model.

    python -m src.calibrate

Embeds each probe query (cheap; no LLM call) and prints the best Pinecone
similarity. Suggests a threshold between the weakest in-document query and the
strongest off-topic query.
"""
from langchain_pinecone import PineconeVectorStore

from src import config
from src.models import get_embeddings

IN_DOC = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
]
OFF_TOPIC = [
    "Who won the 2022 FIFA World Cup?",
    "What is the capital of Australia?",
    "Give me a recipe for chocolate cake.",
]


def best_score(store, q: str) -> float:
    res = store.similarity_search_with_score(q, k=config.TOP_K)
    return max((s for _, s in res), default=0.0)


def main() -> None:
    config.require_env()
    store = PineconeVectorStore(index_name=config.PINECONE_INDEX_NAME, embedding=get_embeddings())
    print(f"Provider: {config.LLM_PROVIDER} | current threshold: {config.RELEVANCE_THRESHOLD}\n")

    in_scores = [best_score(store, q) for q in IN_DOC]
    off_scores = [best_score(store, q) for q in OFF_TOPIC]
    for q, s in zip(IN_DOC, in_scores):
        print(f"  in-doc   {s:.3f}  {q}")
    for q, s in zip(OFF_TOPIC, off_scores):
        print(f"  off-topic{s:.3f}  {q}")

    lo, hi = min(in_scores), max(off_scores)
    print(f"\nweakest in-doc = {lo:.3f} | strongest off-topic = {hi:.3f}")
    if lo > hi:
        print(f"Suggested RELEVANCE_THRESHOLD ≈ {(lo + hi) / 2:.2f}  (put it in .env)")
    else:
        print("Scores overlap - the threshold alone can't separate them. Pick a value just above "
              "the off-topic max; the LLM's `answerable` flag will catch the rest.")


if __name__ == "__main__":
    main()
