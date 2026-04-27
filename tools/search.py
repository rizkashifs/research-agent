from datetime import datetime, timezone

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


VALID_TIMELIMITS = {"d", "w", "m", "y"}


def run(args: dict) -> str:
    query = args.get("query", "").strip()
    if not query:
        return "Error: search requires a non-empty query."

    max_results = args.get("max_results", 5)
    try:
        max_results = int(max_results)
    except (TypeError, ValueError):
        return "Error: max_results must be an integer."
    max_results = max(1, min(max_results, 10))

    timelimit = args.get("timelimit")
    if timelimit is not None:
        timelimit = str(timelimit).strip().lower()
        if timelimit not in VALID_TIMELIMITS:
            return "Error: timelimit must be one of: d, w, m, y."

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results, timelimit=timelimit))
    except Exception as e:
        return f"Search failed: {e}"

    if not results:
        return "No results found."

    searched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    freshness = f", timelimit={timelimit}" if timelimit else ""
    lines = [f"Search query: {query}", f"Searched at: {searched_at}{freshness}"]

    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        href = r.get("href", "")
        body = r.get("body", "")
        lines.append(f"[{i}] {title}\n    URL: {href}\n    {body}")

    return "\n\n".join(lines)
