"""
Run RAGAS retrieval evaluation on a Milvus-backed retriever.

Example:
  python scripts/ragas_eval.py --eval-jsonl data\\ragas_eval_2023.jsonl --top-k 5 --out data\\ragas_eval_2023_results.json
"""

import argparse
import json
import sys
from pathlib import Path


def load_eval_items(jsonl_path: Path):
    items = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def build_ragas_dataset(items, store, embedder, top_k):
    dataset = []
    for item in items:
        question = item["question"]
        ground_truth = item.get("ground_truth") or ""
        gt_contexts = item.get("contexts") or []
        if not ground_truth and gt_contexts:
            ground_truth = gt_contexts[0]

        query_vec = embedder.encode_queries(question)[0]
        results = store.search(query_embedding=query_vec, top_k=top_k)
        contexts = [r.content for r in results]

        dataset.append(
            {
                "user_input": question,
                "retrieved_contexts": contexts,
                "reference": ground_truth,
            }
        )
    return dataset


def main():
    parser = argparse.ArgumentParser(description="RAGAS retrieval eval")
    parser.add_argument("--eval-jsonl", required=True, help="JSONL with question/ground_truth/contexts")
    parser.add_argument("--collection", default="rag_chunks", help="Milvus collection name")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k retrieval size")
    parser.add_argument("--model", default="BAAI/bge-m3", help="Embedding model name")
    parser.add_argument("--device", default="cuda", help="Device for embedding model")
    parser.add_argument("--metrics", default="precision,recall,relevancy", help="Comma list of metrics")
    parser.add_argument("--llm-provider", default="openai", help="LLM provider for RAGAS (default: openai)")
    parser.add_argument("--llm-client", default="openai", help="LLM client type (default: openai)")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="LLM model name for RAGAS")
    parser.add_argument("--llm-api-key", default="", help="LLM API key (optional, falls back to env)")
    parser.add_argument("--llm-base-url", default="", help="LLM base URL (optional)")
    parser.add_argument("--out", default="", help="Output JSON file for results")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from embedding import BGEEmbedder, MilvusVectorStore, EmbeddingConfig, MilvusConfig

    try:
        from ragas import evaluate
        from ragas.llms import InstructorLLM
        import ragas.metrics as legacy_metrics
        from datasets import Dataset
    except Exception as e:
        print(f"RAGAS not available: {e}")
        sys.exit(1)

    llm_client = None
    if args.llm_client == "openai":
        try:
            from openai import OpenAI
        except Exception as e:
            print(f"OpenAI client not available: {e}")
            sys.exit(1)
        llm_client = OpenAI(
            api_key=args.llm_api_key or None,
            base_url=args.llm_base_url or None,
        )
    else:
        print(f"Unsupported llm-client: {args.llm_client}")
        sys.exit(1)

    llm = InstructorLLM(
        client=llm_client,
        model=args.llm_model,
        provider=args.llm_provider,
    )

    metric_map = {
        "precision": legacy_metrics._context_precision,
        "recall": legacy_metrics._context_recall,
        "relevancy": legacy_metrics._context_entity_recall,
        "entity_recall": legacy_metrics._context_entity_recall,
    }
    selected = []
    for name in [m.strip() for m in args.metrics.split(",") if m.strip()]:
        if name in metric_map:
            metric = metric_map[name]
            if hasattr(metric, "llm"):
                metric.llm = llm
            selected.append(metric)
    if not selected:
        print("No valid metrics selected.")
        sys.exit(1)

    embedder = BGEEmbedder(EmbeddingConfig(model_name=args.model, device=args.device))
    store = MilvusVectorStore(MilvusConfig(collection_name=args.collection))
    store.load_collection()

    items = load_eval_items(Path(args.eval_jsonl))
    data = build_ragas_dataset(items, store, embedder, args.top_k)
    dataset = Dataset.from_list(data)

    results = evaluate(dataset, metrics=selected)
    print(results)

    if args.out:
        out_path = Path(args.out)
        records = results.to_pandas().to_dict(orient="records")
        summary = getattr(results, "scores", None)
        payload = {
            "summary": summary,
            "records": records,
        }
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
