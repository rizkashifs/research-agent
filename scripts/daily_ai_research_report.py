#!/usr/bin/env python3
"""Generate the daily AI research report and email the markdown file."""
import argparse
import os
import smtplib
import sys
from datetime import date
from email.message import EmailMessage
from html import escape
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from agent.agent import ResearchAgent
from guardrails.output_validator import extract_search_urls, validate_output
from main import slugify
from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory
from tools.registry import ToolRegistry
from tools.save_note import set_session_id
from llm.factory import get_llm_client


REPORT_SYSTEM_PROMPT_TEMPLATE = """\
You are a senior AI research analyst doing a daily news scan for an MLOps and GenAI architect.
Today is {report_date}. Internet search is ON — use it.

Your ONLY job is to run 3 targeted web searches across different topic clusters, then write the
structured briefing directly. Do not call recall() or save_note() — this is a time-sensitive
current-events scan, not a general research session. Do not call summarize() separately;
synthesize inline when writing the final report.

Search strategy — run all 3 searches in the first iteration, each with max_results=3 and timelimit="w".
Make the queries specific to TODAY's date ({report_date}) to maximise freshness:
  1. "AI LLM model release SDK update {report_date}"
  2. "MLOps GenAI engineering deployment eval observability {report_date}"
  3. "machine learning fine-tuning benchmark data science {report_date}"

After the searches complete, write the final report immediately in the next iteration.
Do not add extra tool calls between the searches and the final answer.

{yesterday_context}\
"""


REPORT_QUERY_TEMPLATE = """\
Act as my Senior AI Research Lead. Today is {{Date}}.

You are writing my daily 5-minute briefing. I am an MLOps and GenAI architect.
Hard constraints:
- Total report must be UNDER 520 words. I read this every morning — if it is long I stop reading it.
- Use tight bullet points, not paragraphs. No section intros, no fluff.
- Every sentence must carry a concrete signal or decision implication.
- Search budget: at most 3 web searches, max_results=3 each, timelimit="w" for freshness.
- If a section has no strong signal today, write exactly: "No strong signal today."

---

## What Just Shipped
3 bullets max. Model releases, SDK/library version bumps, and tool launches from this week.
Format: **[Name vX.X / release name]** — one sentence: what changed and the one architectural implication.
Skip anything that is purely marketing with no engineering consequence.

## Model & Research Signal
2 bullets. One: a model or capability development that concretely changes a cost, latency, quality, or build-vs-buy tradeoff you make today. Two: one research technique that has crossed from paper to production-readiness — state the specific decision it affects.

## MLOps Practice
2 bullets. One shifting practice in how teams evaluate, deploy, monitor, or version ML/GenAI systems.
Focus on the workflow decision, not the tool. What changed about how practitioners approach this step?
Cover areas like: eval pipelines, model/prompt CI-CD, registry promotion gates, deployment patterns, pipeline contracts, drift alerting, observability, serving topology.
Do NOT list tool names as the signal — the practice is the signal.

## Data Science Signal
2 bullets. One practical development from: fine-tuning techniques (LoRA/QLoRA/GRPO/DPO), synthetic data generation, dataset curation, evaluation science, new benchmarks and what they expose, or feature engineering advances.
State what a practitioner can apply or decide differently this week.

## Concept of the Day
6–8 sentences on one evergreen MLOps or GenAI architecture concept.
Use this structure: what it is (1 sentence) → how it works in practice (2 sentences) → where it silently fails in production and why (2 sentences) → the one concrete mitigation (1–2 sentences).
Rotate through: feature skew, embedding drift, prompt versioning, shadow deployment, model registry gates, online/offline parity, KV-cache reuse, inference autoscaling, eval regression, retrieval permission drift, data contract violations, reranker latency budget.

## Design Challenge
3–4 sentences. One architecture question grounded in today's signals.
State: the system context, the specific constraint, and the failure mode to defend against.
No answer — this is for me to reason through.

---
Word budget: under 520 words total across all six sections.
"""


HAIKU_45_INPUT_PER_MILLION = 1.00
HAIKU_45_OUTPUT_PER_MILLION = 5.00
INCOMPLETE_WARNING = "[Warning: answer may be incomplete"


def load_yesterday_context(output_dir: Path, report_date: date) -> str:
    from datetime import timedelta
    yesterday = report_date - timedelta(days=1)
    filename = f"{yesterday.isoformat()}_{slugify('daily ai research report')}.md"
    yesterday_path = output_dir / filename
    if not yesterday_path.exists():
        return ""
    try:
        content = yesterday_path.read_text(encoding="utf-8")
    except OSError:
        return ""
    lines = [l for l in content.splitlines() if l.startswith("## ") or l.startswith("- **")]
    if not lines:
        return ""
    summary = "\n".join(lines[:20])
    return (
        "IMPORTANT — Yesterday's report already covered the following topics and items. "
        "Do NOT repeat these. Find different signals today:\n"
        + summary
        + "\n\n"
    )


def build_query(report_date: date) -> str:
    return REPORT_QUERY_TEMPLATE.replace("{{Date}}", report_date.isoformat())


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        (input_tokens / 1_000_000) * HAIKU_45_INPUT_PER_MILLION
        + (output_tokens / 1_000_000) * HAIKU_45_OUTPUT_PER_MILLION
    )


def format_usage_footer(llm) -> str:
    input_tokens = getattr(llm, "total_input_tokens", 0)
    output_tokens = getattr(llm, "total_output_tokens", 0)
    calls = len(getattr(llm, "usage_log", []))
    cost = estimate_cost(input_tokens, output_tokens)
    return (
        "\n\n---\n\n"
        "## Run Usage\n\n"
        f"- LLM calls: {calls}\n"
        f"- Input tokens: {input_tokens:,}\n"
        f"- Output tokens: {output_tokens:,}\n"
        f"- Estimated Claude Haiku 4.5 cost: ${cost:.4f}\n"
    )


def synthesize_from_steps(llm, query: str, state) -> str:
    observations = []
    for step in state.steps:
        if step.action == "final_answer":
            continue
        observations.append(
            f"Action: {step.action}({step.action_input})\n"
            f"Observation: {step.observation[:1500]}"
        )
    evidence = "\n\n---\n\n".join(observations)
    prompt = (
        "Write the final daily AI research briefing from the evidence below. "
        "Do not call tools. Under 520 words total. Bullet points only — no paragraphs. "
        "Include exactly these six sections: "
        "What Just Shipped, Model & Research Signal, MLOps Practice, "
        "Data Science Signal, Concept of the Day, Design Challenge. "
        "If a section has no evidence, write 'No strong signal today.' "
        "Every bullet must carry a concrete decision implication for an MLOps/GenAI architect.\n\n"
        f"Evidence gathered:\n{evidence}"
    )
    response = llm.chat(messages=[{"role": "user", "content": prompt}], tools=None)
    return response.get("content", "").strip()


def is_incomplete_report(final_answer: str, state) -> bool:
    stripped = final_answer.strip()
    return (
        not state.is_complete
        or INCOMPLETE_WARNING in stripped
        or len(stripped) < 500
    )


def fallback_report_from_steps(state, report_date: date) -> str:
    searches = [s for s in state.steps if s.action == "search"]
    summaries = [s for s in state.steps if s.action == "summarize"]
    recalls = [s for s in state.steps if s.action == "recall"]

    def bullets(steps, limit=2):
        lines = []
        for step in steps[:limit]:
            q = step.action_input.get("query") or step.action_input.get("topic") or step.action
            obs = " ".join(step.observation.split())
            lines.append(f"- **{q}**: {obs[:380]}")
        return "\n".join(lines) if lines else "- No strong signal today."

    return f"""\
# Daily AI Research Report | {report_date.isoformat()}

## What Just Shipped
{bullets(searches[:1] + recalls[:1])}

## Model & Research Signal
{bullets(searches[1:2] + summaries[:1])}

## MLOps Practice
- **Eval-as-CI**: Treat prompt/model evaluation as a blocking CI gate — failing evals prevent promotion the same way failing unit tests prevent a code merge. Silent quality regressions are the primary failure mode in production GenAI systems.
- **Registry-first deployment**: Every model, prompt version, and embedding checkpoint should be an immutable artifact in a registry with a promotion policy before reaching production.

## Data Science Signal
{bullets(summaries[:1] + searches[2:3])}

## Concept of the Day
**Embedding drift** is the gradual divergence between the distribution of vectors in your index and the distribution your retrieval queries now produce — caused by model updates, domain shift, or data staleness. It manifests as silent precision degradation: the retrieval pipeline keeps returning results, but relevance quietly drops. In production, it is rarely caught by latency or error-rate monitors because the system is technically healthy. Mitigation: track cosine similarity distributions over time between a fixed query probe set and their top-k results; alert when the mean similarity drops more than a threshold across a rolling window.

## Design Challenge
You are running a multi-tenant RAG system where each tenant has isolated document collections and separate embedding models at different version levels. A planned embedding model upgrade improves retrieval quality by 12% on your eval set but requires re-indexing all tenant collections. How do you roll this out so that you can measure per-tenant quality delta, roll back individual tenants without affecting others, and avoid a full re-index outage — while keeping retrieval latency within SLA during the migration?
"""


def generate_report(query: str, online: bool, output_dir: Path, report_date: date) -> Path:
    ltm = LongTermMemory()
    session_id = ltm.create_session(query)
    set_session_id(session_id)

    # Daily reports can use many tool calls; keep a larger in-run buffer so
    # provider tool_use/tool_result pairs are not split by trimming.
    stm = ShortTermMemory(max_messages=200)
    llm = get_llm_client()
    registry = ToolRegistry(internet_enabled=online)
    yesterday_context = load_yesterday_context(output_dir, report_date)
    report_system_prompt = REPORT_SYSTEM_PROMPT_TEMPLATE.format(
        report_date=report_date.isoformat(),
        yesterday_context=yesterday_context,
    )
    agent = ResearchAgent(
        llm_client=llm,
        registry=registry,
        short_term=stm,
        long_term=ltm,
        max_iterations=3,
        system_prompt_override=report_system_prompt,
    )

    state = agent.run(query)
    search_urls = extract_search_urls(state)
    final_answer = validate_output(state, search_urls)
    if is_incomplete_report(final_answer, state):
        try:
            final_answer = synthesize_from_steps(llm, query, state)
        except Exception as exc:
            print(f"[Report] Synthesis pass failed: {exc}")
            final_answer = ""
    if INCOMPLETE_WARNING in final_answer or len(final_answer.strip()) < 500:
        final_answer = fallback_report_from_steps(state, report_date)

    ltm.save_short_term_snapshot(session_id, stm.to_list())

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{report_date.isoformat()}_{slugify('daily ai research report')}.md"
    filepath = output_dir / filename
    report = f"# Daily AI Research Report\n\n{final_answer}{format_usage_footer(llm)}"
    filepath.write_text(report, encoding="utf-8")
    return filepath


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def build_email_body(markdown: str, report_date: date) -> tuple[str, str]:
    plain = (
        f"Daily AI Research Report - {report_date.isoformat()}\n\n"
        f"{markdown}\n\n"
        "The markdown report is also attached.\n"
    )
    html = f"""\
<!doctype html>
<html>
  <body style="margin:0; padding:24px; background:#f6f7f9; color:#111827;">
    <main style="max-width:860px; margin:0 auto; background:#ffffff; border:1px solid #e5e7eb; padding:28px; font-family:Arial, Helvetica, sans-serif; line-height:1.55;">
      <div style="font-size:13px; color:#6b7280; margin-bottom:18px;">Generated by the local Research Agent</div>
      {markdown_to_email_html(markdown)}
      <hr style="border:0; border-top:1px solid #e5e7eb; margin:24px 0 12px;">
      <div style="font-size:13px; color:#6b7280;">The markdown report is also attached.</div>
    </main>
  </body>
</html>
"""
    return plain, html


def inline_markdown(text: str) -> str:
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    text = re.sub(
        r"`(.+?)`",
        '<code style="background:#f3f4f6; padding:1px 4px; border-radius:4px;">\\1</code>',
        text,
    )
    return text


def markdown_to_email_html(markdown: str) -> str:
    parts = []
    in_list = False
    in_ordered_list = False
    in_table = False

    def close_list():
        nonlocal in_list
        if in_list:
            parts.append("</ul>")
            in_list = False

    def close_ordered_list():
        nonlocal in_ordered_list
        if in_ordered_list:
            parts.append("</ol>")
            in_ordered_list = False

    def close_table():
        nonlocal in_table
        if in_table:
            parts.append("</table>")
            in_table = False

    lines = markdown.splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            close_list()
            close_ordered_list()
            close_table()
            continue
        if stripped == "---":
            close_list()
            close_ordered_list()
            close_table()
            parts.append('<hr style="border:0; border-top:1px solid #e5e7eb; margin:24px 0;">')
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(set(c) <= {"-", ":"} for c in cells):
                continue
            close_list()
            close_ordered_list()
            if not in_table:
                parts.append('<table style="width:100%; border-collapse:collapse; margin:14px 0;">')
                in_table = True
            cell_html = "".join(
                f'<td style="border:1px solid #e5e7eb; padding:8px; vertical-align:top;">{inline_markdown(cell)}</td>'
                for cell in cells
            )
            parts.append(f"<tr>{cell_html}</tr>")
            continue
        close_table()
        if stripped.startswith("#"):
            close_list()
            close_ordered_list()
            level = min(len(stripped) - len(stripped.lstrip("#")), 3)
            text = stripped[level:].strip()
            size = {1: 26, 2: 21, 3: 17}[level]
            margin = {1: "0 0 18px", 2: "24px 0 10px", 3: "18px 0 8px"}[level]
            parts.append(
                f'<h{level} style="font-size:{size}px; line-height:1.25; margin:{margin}; color:#111827;">'
                f"{inline_markdown(text)}</h{level}>"
            )
            continue
        if stripped.startswith(("- ", "* ")):
            close_ordered_list()
            if not in_list:
                parts.append('<ul style="margin:8px 0 14px 22px; padding:0;">')
                in_list = True
            parts.append(f'<li style="margin:5px 0;">{inline_markdown(stripped[2:].strip())}</li>')
            continue
        numbered = re.match(r"^\d+\.\s+(.+)$", stripped)
        if numbered:
            close_list()
            if not in_ordered_list:
                parts.append('<ol style="margin:8px 0 14px 22px; padding:0;">')
                in_ordered_list = True
            parts.append(f'<li style="margin:5px 0;">{inline_markdown(numbered.group(1))}</li>')
            continue
        close_list()
        close_ordered_list()
        parts.append(f'<p style="margin:10px 0;">{inline_markdown(stripped)}</p>')

    close_list()
    close_ordered_list()
    close_table()
    return "\n".join(parts)


def send_email(attachment_path: Path, report_date: date) -> None:
    smtp_host = required_env("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_username = required_env("SMTP_USERNAME")
    smtp_password = required_env("SMTP_PASSWORD")
    email_from = os.getenv("EMAIL_FROM", smtp_username).strip()
    email_to = required_env("EMAIL_TO")
    use_tls = os.getenv("SMTP_USE_TLS", "true").strip().lower() != "false"

    message = EmailMessage()
    message["Subject"] = f"Daily AI Research Report - {report_date.isoformat()}"
    message["From"] = email_from
    message["To"] = email_to
    markdown = attachment_path.read_text(encoding="utf-8")
    plain_body, html_body = build_email_body(markdown, report_date)
    message.set_content(plain_body)
    message.add_alternative(html_body, subtype="html")
    message.add_attachment(
        markdown.encode("utf-8"),
        maintype="text",
        subtype="markdown",
        filename=attachment_path.name,
    )

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as smtp:
        if use_tls:
            smtp.starttls()
        smtp.login(smtp_username, smtp_password)
        smtp.send_message(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and email the daily AI research report.")
    parser.add_argument(
        "--online",
        action="store_true",
        help="Enable internet search. Recommended for this daily news report.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "daily"),
        help="Directory where the markdown report will be saved.",
    )
    parser.add_argument(
        "--no-email",
        action="store_true",
        help="Generate the markdown report without sending email.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force execution even if a report for today already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    today = date.today()
    
    # Check if report already exists for today
    filename = f"{today.isoformat()}_{slugify('daily ai research report')}.md"
    output_dir = Path(args.output_dir)
    output_path = output_dir / filename
    
    if output_path.exists() and not args.force:
        print(f"[Skip] Report for today ({today.isoformat()}) already exists at: {output_path}")
        print("Use --force to generate it again.")
        return

    query = build_query(today)
    output_path = generate_report(
        query=query,
        online=args.online,
        output_dir=output_dir,
        report_date=today,
    )
    print(f"[Report] Saved markdown report to: {output_path}")

    if args.no_email:
        print("[Email] Skipped because --no-email was provided.")
        return

    send_email(output_path, today)
    print(f"[Email] Sent report to: {os.getenv('EMAIL_TO')}")


if __name__ == "__main__":
    main()
