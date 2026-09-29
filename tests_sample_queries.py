"""Run the benchmark queries.

    python tests_sample_queries.py            # against a running API (localhost:8000)
    python tests_sample_queries.py --url http://host:port
    python tests_sample_queries.py --direct   # call the graph directly, no server
"""
import argparse
import json
import sys
import urllib.request

QUERIES = [
    ("What is Agentic AI according to the eBook?", False),
    ("How do AI agents differ from traditional automation systems?", False),
    ("What are the core components of an Agentic Architecture?", False),
    ("What role does memory play in Agentic AI workflows?", False),
    ("What are the main risks or challenges of deploying agentic systems?", False),
    ("Who won the 2022 FIFA World Cup?", True),  # must be refused
]


def call_api(url: str, query: str) -> dict:
    req = urllib.request.Request(
        f"{url.rstrip('/')}/chat",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--direct", action="store_true")
    args = parser.parse_args()

    if args.direct:
        from src.graph import ask, build_rag_graph

        graph = build_rag_graph()

        def run(q):
            r = ask(graph, q)
            return {
                "answer": r["answer"],
                "retrieved_chunks": r["context"],
                "confidence_score": r["score"],
            }
    else:
        def run(q):
            return call_api(args.url, q)

    failures = 0
    for i, (query, should_refuse) in enumerate(QUERIES, 1):
        out = run(query)
        refused = out["confidence_score"] == 0.0
        ok = refused if should_refuse else not refused
        failures += not ok
        print(f"\n{'=' * 78}\n[{i}] {query}\n{'-' * 78}")
        print(f"Answer     : {out['answer']}")
        print(f"Confidence : {out['confidence_score']}")
        for j, c in enumerate(out["retrieved_chunks"], 1):
            snippet = c["text"].replace("\n", " ")[:110]
            print(f"  chunk {j} (p.{c['page']}, sim={c['score']}): {snippet}...")
        print(f"Result     : {'PASS' if ok else 'FAIL'} "
              f"(expected {'refusal' if should_refuse else 'grounded answer'})")

    print(f"\n{len(QUERIES) - failures}/{len(QUERIES)} checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
