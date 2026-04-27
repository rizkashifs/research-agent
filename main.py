#!/usr/bin/env python3
"""CLI entry point for the Research Agent."""
import argparse
import sys
import os
import re

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
    parser.add_argument(
        "--output", action="store_true",
        help="Save the final answer as a markdown file in the 'results' directory"
    )
    parser.add_argument(
        "--online", action="store_true",
        help="Enable internet search for latest/current information"
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    """Convert a string to a filesystem-friendly slug."""
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '_', text).strip('_')
    return text or "research_result"


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

    if args.online:
        print("[Mode] Online mode enabled - internet search is available.\n")
    else:
        print("[Mode] Offline mode enabled - use --online to enable internet search.\n")

    # --- Run agent ---
    try:
        llm = get_llm_client()
    except NotImplementedError as e:
        print(f"\n[Error] {e}\n")
        sys.exit(1)

    registry = ToolRegistry(internet_enabled=args.online)
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

    # --- Save to file ---
    if args.output:
        try:
            output_dir = "results"
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{slugify(query)}.md"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# Research Task: {query}\n\n")
                f.write(final_answer)
            print(f"[File] Final answer saved to: {filepath}\n")
        except Exception as e:
            print(f"[Error] Could not save to file: {e}\n")


if __name__ == "__main__":
    main()
