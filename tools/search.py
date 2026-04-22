try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


def run(args: dict) -> str:
    query = args.get("query", "").strip()
    if not query:
        return "Error: search requires a non-empty query."

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except Exception as e:
        return f"Search failed: {e}"

    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        href = r.get("href", "")
        body = r.get("body", "")
        lines.append(f"[{i}] {title}\n    URL: {href}\n    {body}")

    return "\n\n".join(lines)
