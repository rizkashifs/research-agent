#!/usr/bin/env python3
"""Evaluation harness — runs all test cases and reports per-case results."""
import json
import shutil
import sys
import tempfile
import os
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from guardrails.input_validator import validate_query
from guardrails.output_validator import validate_output, extract_search_urls


def run_case(tc: dict, eval_db_path: str, eval_chroma_path: str) -> dict:
    """
    Run a single test case with its own isolated DB paths.
    Returns a result dict: passed, reasons, tool_calls_made, answer, iterations_used.
    """
    query = tc["query"]
    expected_tools = tc.get("expected_tool_calls", [])
    expected_keywords = tc.get("expected_answer_contains", [])
    max_iters = tc.get("max_iterations_allowed", 8)
    expect_rejection = tc.get("expect_guardrail_rejection", False)

    reasons: list[str] = []
    tool_calls_made: list[str] = []
    answer = ""
    iterations_used = 0

    # --- Guardrail check ---
    valid, err = validate_query(query)
    if not valid:
        if expect_rejection:
            return {
                "passed": True, "reasons": [], "tool_calls_made": [],
                "answer": f"[REJECTED] {err}", "iterations_used": 0,
                "guardrail_triggered": True,
            }
        else:
            return {
                "passed": False,
                "reasons": [f"Unexpected guardrail rejection: {err}"],
                "tool_calls_made": [], "answer": f"[REJECTED] {err}",
                "iterations_used": 0, "guardrail_triggered": True,
            }

    if expect_rejection:
        reasons.append("Expected guardrail rejection but query passed validation")
        return {
            "passed": False, "reasons": reasons, "tool_calls_made": [],
            "answer": "", "iterations_used": 0, "guardrail_triggered": False,
        }

    # --- Run agent with isolated memory ---
    try:
        import config as cfg
        original_db = cfg.SQLITE_DB_PATH
        original_chroma = cfg.CHROMA_DB_PATH

        cfg.SQLITE_DB_PATH = eval_db_path
        cfg.CHROMA_DB_PATH = eval_chroma_path

        from agent.agent import ResearchAgent
        from memory.short_term import ShortTermMemory
        from memory.long_term import LongTermMemory
        from tools.registry import ToolRegistry
        from tools.save_note import set_session_id
        from llm.factory import get_llm_client

        ltm = LongTermMemory(db_path=eval_db_path)
        session_id = ltm.create_session(query)
        set_session_id(session_id)

        # Patch VectorStore to use eval chroma path
        import retrieval.vector_store as vs_module
        original_init = vs_module.VectorStore.__init__

        def patched_init(self_vs):
            import chromadb
            import hashlib
            self_vs.client = chromadb.PersistentClient(path=eval_chroma_path)
            self_vs.ef = vs_module.TFIDFEmbeddingFunction()
            self_vs.collection = self_vs.client.get_or_create_collection(
                name=cfg.CHROMA_COLLECTION,
                embedding_function=self_vs.ef,
                metadata={"hnsw:space": "cosine"},
            )

        vs_module.VectorStore.__init__ = patched_init

        stm = ShortTermMemory()
        llm = get_llm_client()
        registry = ToolRegistry()
        agent = ResearchAgent(
            llm_client=llm, registry=registry, short_term=stm,
            long_term=ltm, max_iterations=max_iters,
        )
        state = agent.run(query)

        vs_module.VectorStore.__init__ = original_init
        cfg.SQLITE_DB_PATH = original_db
        cfg.CHROMA_DB_PATH = original_chroma

        search_urls = extract_search_urls(state)
        answer = validate_output(state, search_urls)
        tool_calls_made = state.tool_calls_made
        iterations_used = state.iteration

    except NotImplementedError as e:
        reasons.append(f"Provider not implemented: {e}")
        return {
            "passed": False, "reasons": reasons, "tool_calls_made": [],
            "answer": str(e), "iterations_used": 0, "guardrail_triggered": False,
        }
    except Exception as e:
        reasons.append(f"Agent error: {e}")
        return {
            "passed": False, "reasons": reasons, "tool_calls_made": [],
            "answer": str(e), "iterations_used": 0, "guardrail_triggered": False,
        }

    # --- Scoring ---
    if expected_tools:
        missing = [t for t in expected_tools if t not in tool_calls_made]
        if missing:
            reasons.append(f"Missing expected tool calls: {missing}")

    if expected_keywords:
        answer_lower = answer.lower()
        missing_kw = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
        if missing_kw:
            reasons.append(f"Answer missing expected keywords: {missing_kw}")

    if iterations_used > max_iters:
        reasons.append(f"Used {iterations_used} iterations, exceeded max {max_iters}")

    return {
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "tool_calls_made": tool_calls_made,
        "answer": answer[:200] + ("..." if len(answer) > 200 else ""),
        "iterations_used": iterations_used,
        "guardrail_triggered": False,
    }


def main():
    cases_path = Path(__file__).parent / "test_cases.json"
    test_cases = json.loads(cases_path.read_text())

    passed = 0
    total = len(test_cases)

    print(f"\n{'='*60}")
    print(f"Research Agent Evaluation — {total} test cases")
    print(f"{'='*60}\n")

    for tc in test_cases:
        qid = tc["id"]
        desc = tc.get("description", "")
        print(f"Running {qid}: {desc}")
        print(f"  Query: {tc['query'][:80]!r}")

        # Isolated temp paths per test case
        tmp_dir = tempfile.mkdtemp(prefix=f"eval_{qid}_")
        eval_db = os.path.join(tmp_dir, "eval_memory.db")
        eval_chroma = os.path.join(tmp_dir, "chroma")

        try:
            result = run_case(tc, eval_db, eval_chroma)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        status = "PASS" if result["passed"] else "FAIL"
        if result["passed"]:
            passed += 1

        print(f"  Status: {status}")
        if result.get("guardrail_triggered"):
            print(f"  Guardrail triggered: YES")
        else:
            print(f"  Tool calls: {result['tool_calls_made']}")
            print(f"  Iterations: {result['iterations_used']}")
        if result["reasons"]:
            for r in result["reasons"]:
                print(f"  Reason: {r}")
        print()

    print(f"{'='*60}")
    print(f"Passed {passed}/{total} test cases")
    print(f"{'='*60}\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
