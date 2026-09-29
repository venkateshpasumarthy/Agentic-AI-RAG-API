"""LangGraph workflow: retrieve -> (generate | refuse) -> END.

Strict grounding is enforced in three layers:
  1. Retrieval gate  - if no chunk clears RELEVANCE_THRESHOLD, we refuse
                       without even calling the LLM.
  2. Prompt          - the LLM may only use the supplied context.
  3. Structured out  - the LLM must flag `answerable=false` when the context
                       doesn't contain the answer; we then return the refusal.
"""
from typing import List, Optional, TypedDict

from langchain_pinecone import PineconeVectorStore
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src import config
from src.models import get_embeddings, get_llm


class RetrievedChunk(TypedDict):
    text: str
    page: Optional[int]  # 1-indexed page in the PDF
    score: float  # cosine similarity from Pinecone (higher = closer)


class AgentState(TypedDict):
    question: str
    context: List[RetrievedChunk]
    answer: str
    score: float  # confidence score in [0, 1]


class GroundedAnswer(BaseModel):
    answerable: bool = Field(
        description="True only if the context contains enough information to answer."
    )
    answer: str = Field(description="The answer, using only the context. Empty if not answerable.")


SYSTEM_PROMPT = f"""You are a strict question-answering assistant for a single document:
an eBook about Agentic AI.

Rules:
- Answer using ONLY the numbered context passages provided. Do not use outside knowledge.
- If the context does not contain enough information to answer, set answerable=false.
- Be concise and faithful to the source. Do not invent details.
- When helpful, cite passages inline like [1], [2].
"""


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_rag_graph(
    index_name: str = config.PINECONE_INDEX_NAME,
    vectorstore=None,
    llm=None,
    top_k: int = config.TOP_K,
    threshold: float = config.RELEVANCE_THRESHOLD,
):
    """Build & compile the RAG graph. `vectorstore` / `llm` are injectable for tests."""
    if vectorstore is None or llm is None:
        config.require_env()
    if vectorstore is None:
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=get_embeddings())
    if llm is None:
        llm = get_llm()
    structured_llm = llm.with_structured_output(GroundedAnswer)

    # ---- Nodes -------------------------------------------------------------
    def retrieve_node(state: AgentState) -> dict:
        results = vectorstore.similarity_search_with_score(state["question"], k=top_k)
        chunks: List[RetrievedChunk] = []
        for doc, score in results:
            page = doc.metadata.get("page")
            chunks.append(
                {
                    "text": doc.page_content,
                    "page": int(page) + 1 if isinstance(page, (int, float)) else None,
                    "score": round(float(score), 4),
                }
            )
        chunks.sort(key=lambda c: c["score"], reverse=True)
        return {"context": chunks}

    def generate_node(state: AgentState) -> dict:
        relevant = [c for c in state["context"] if c["score"] >= threshold]
        context_str = "\n\n".join(
            f"[{i}] (page {c['page']}) {c['text']}" for i, c in enumerate(relevant, 1)
        )
        result: GroundedAnswer = structured_llm.invoke(
            [
                ("system", SYSTEM_PROMPT),
                ("human", f"Context:\n{context_str}\n\nQuestion: {state['question']}"),
            ]
        )
        if not result.answerable:
            return {"answer": config.REFUSAL_MESSAGE, "score": 0.0}

        # Confidence = mean similarity of the chunks the answer was built from.
        confidence = max(0.0, min(1.0, _mean([c["score"] for c in relevant])))
        return {"answer": result.answer.strip(), "score": round(confidence, 3)}

    def refuse_node(state: AgentState) -> dict:
        return {"answer": config.REFUSAL_MESSAGE, "score": 0.0}

    # ---- Routing -----------------------------------------------------------
    def route_after_retrieve(state: AgentState) -> str:
        best = max((c["score"] for c in state["context"]), default=0.0)
        return "generate" if best >= threshold else "refuse"

    # ---- Graph -------------------------------------------------------------
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("refuse", refuse_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_conditional_edges(
        "retrieve", route_after_retrieve, {"generate": "generate", "refuse": "refuse"}
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("refuse", END)

    return workflow.compile()


def ask(graph, question: str) -> AgentState:
    """Convenience wrapper used by the API, UI and tests."""
    return graph.invoke({"question": question, "context": [], "answer": "", "score": 0.0})
