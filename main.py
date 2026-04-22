#!/usr/bin/env python3
"""CLI entry point for the Research Agent."""
import argparse
import sys

from guardrails.input_validator import validate_query
from guardrails.output_validator import validate_output, extract_search_urls
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from tools.registry import ToolRegistry
from tools.save_note import set_session_id
from agent.agent import ResearchAgent
from llm.factory import get_llm_client


def parse_args():
    parser = argparse.ArgumentParser(description="Research Agent CLI")
    parser.add_argument("query", nargs="?", help="Research question")
    parser.add_argument(
        "--continue", dest="cont", action="store_true",
        help="Restore last session's short-term buffer"
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore memory, start completely clean"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    query = args.query
    if not query:
        query = input("Enter your research query: ").strip()

    # --- Input guardrails ---
    valid, err = validate_query(query)
    if not valid:
        print(f"\n[Rejected] {err}\n")
        sys.exit(1)

    # --- Memory setup ---
    ltm = LongTermMemory()
    session_id = ltm.create_session(query)
    set_session_id(session_id)

    stm = ShortTermMemory()

    if args.cont and not args.fresh:
        saved = ltm.load_last_short_term_snapshot()
        if saved:
            stm.load(saved)
            print(f"[Memory] Restored {len(saved)} messages from last session.\n")
        else:
            print("[Memory] No previous session found — starting fresh.\n")
    elif args.fresh:
        print("[Memory] Starting fresh — ignoring all prior memory.\n")

    # --- Run agent ---
    try:
        llm = get_llm_client()
    except NotImplementedError as e:
        print(f"\n[Error] {e}\n")
        sys.exit(1)

    registry = ToolRegistry()
    agent = ResearchAgent(
        llm_client=llm,
        registry=registry,
        short_term=stm,
        long_term=ltm if not args.fresh else None,
    )

    state = agent.run(query)

    # --- Output guardrails ---
    search_urls = extract_search_urls(state)
    final_answer = validate_output(state, search_urls)

    # --- Persist short-term buffer ---
    if not args.fresh:
        ltm.save_short_term_snapshot(session_id, stm.to_list())

    print("\n" + "="*60)
    print("FINAL ANSWER")
    print("="*60)
    print(final_answer)
    print()


if __name__ == "__main__":
    main()
