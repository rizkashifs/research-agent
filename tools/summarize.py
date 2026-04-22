import re


def run(args: dict) -> str:
    text = args.get("text", "").strip()
    max_sentences = int(args.get("max_sentences", 5))

    if not text:
        return "Error: summarize requires non-empty text."

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= max_sentences:
        return text

    # Naive extractive: pick first sentence, then every Nth to fill quota
    step = max(1, len(sentences) // max_sentences)
    selected = []
    seen = set()
    # Always include first sentence
    selected.append(sentences[0])
    seen.add(0)
    idx = step
    while len(selected) < max_sentences and idx < len(sentences):
        if idx not in seen:
            selected.append(sentences[idx])
            seen.add(idx)
        idx += step

    return " ".join(selected)
