_session_context: dict = {"session_id": None}


def set_session_id(session_id: int | None):
    _session_context["session_id"] = session_id


def run(args: dict) -> str:
    content = args.get("content", "").strip()
    topic = args.get("topic", "general").strip()

    if not content:
        return "Error: save_note requires non-empty content."

    session_id = _session_context.get("session_id")

    # Persist to long-term SQLite memory
    try:
        from memory.long_term import LongTermMemory
        ltm = LongTermMemory()
        ltm.save_note(content=content, topic=topic, session_id=session_id)
    except Exception as e:
        return f"Note could not be saved to DB: {e}"

    # Also upsert into vector store
    try:
        from retrieval.vector_store import VectorStore
        vs = VectorStore()
        vs.upsert(content=content, topic=topic, session_id=str(session_id or ""))
    except Exception as e:
        pass  # Vector store is optional — don't fail the tool call

    return f"Note saved successfully under topic '{topic}'."
