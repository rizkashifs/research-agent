def run(args: dict) -> str:
    query = args.get("query", "").strip()
    if not query:
        return "Error: recall requires a non-empty query."

    try:
        from retrieval.vector_store import VectorStore
        vs = VectorStore()
        if vs.count() == 0:
            return "No notes found in memory yet."
        results = vs.query(query, n_results=3)
    except Exception as e:
        return f"Recall failed: {e}"

    if not results:
        return "No relevant notes found."

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] Topic: {r['topic']}\n    {r['content']}")
    return "Recalled notes:\n\n" + "\n\n".join(lines)
