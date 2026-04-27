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


REPORT_QUERY_TEMPLATE = """\
Act as my Senior AI Research Lead. Today is {{Date}}. Provide a high-signal report on:

Run compactly: use at most 3 web searches total, save only one concise note, and complete the final report within 3 agent iterations.

GenAI Ops: One practical pattern for agentic orchestration, LLM evaluation, agent observability, or prompt/retrieval testing.

MLOps Platform/Infra: One major update or practical tradeoff involving Vertex AI, SageMaker, Databricks, MLflow, Kubeflow, Ray, Airflow, Kubernetes, feature stores, model registries, serving stacks, or experiment tracking.

Deployment/Release Engineering: One useful pattern for model/prompt CI/CD, canary rollout, shadow testing, rollback, environment promotion, model registry gates, or reproducible pipelines.

RAG/Data Quality: One practical signal on retrieval quality, data contracts, chunking/indexing, reranking, embedding/version drift, freshness SLAs, or knowledge-base governance.

Cost/Performance Engineering: One concrete update or tactic for latency, inference cost, caching, batching, model routing, quantization, GPU/serverless tradeoffs, or evaluation cost control.

Production Reliability/Security: One production concern such as drift detection, data leakage, PII handling, access control, prompt injection, eval regressions, incident response, SLOs, or observability. Avoid regulatory summaries unless they directly change an engineering decision.

Deep Dive: A 2-sentence technical breakdown of one evergreen MLOps or GenAI architecture concept like feature skew, online/offline parity, model registry gates, embedding drift, CI/CD for prompts, shadow deployments, or inference autoscaling.

Design Challenge: One thought experiment question based on today's news to test my architecture skills.
Constraint: No marketing fluff or listicles-focus on scalability, security, and technical feasibility.
"""


HAIKU_45_INPUT_PER_MILLION = 1.00
HAIKU_45_OUTPUT_PER_MILLION = 5.00
INCOMPLETE_WARNING = "[Warning: answer may be incomplete"


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
            "Action: "
            f"{step.action}({step.action_input})\n"
            f"Observation: {step.observation[:1800]}"
        )
    evidence = "\n\n---\n\n".join(observations)
    prompt = (
        "Write the final daily AI research report from the evidence below. "
        "Do not call tools. Keep it concise but useful for an MLOps and GenAI architect. "
        "Include these sections: GenAI Ops, MLOps Platform/Infra, "
        "Deployment/Release Engineering, RAG/Data Quality, Cost/Performance Engineering, "
        "Production Reliability/Security, Deep Dive, Design Challenge.\n\n"
        f"Original request:\n{query}\n\n"
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
    notes = [s for s in state.steps if s.action == "save_note"]

    def bullets(steps, limit=3):
        lines = []
        for step in steps[:limit]:
            query = step.action_input.get("query") or step.action_input.get("topic") or step.action
            observation = " ".join(step.observation.split())
            lines.append(f"- **{query}**: {observation[:420]}")
        return "\n".join(lines) if lines else "- No strong signal captured in this run."

    return f"""\
# AI Research Report | {report_date.isoformat()}

## GenAI Ops
{bullets(recalls[:1] + searches[:1], limit=2)}

## MLOps Platform/Infra
{bullets(searches[1:2] + recalls[1:2], limit=2)}

## Deployment/Release Engineering
- Use model registry gates, canary/shadow evaluation, and rollback metadata as the default release path for models, prompts, and retrieval pipelines. This run did not produce a dedicated release-engineering final answer before the iteration cap, so treat this as the architectural fallback.

## RAG/Data Quality
{bullets(recalls[2:3] + searches[2:3], limit=2)}

## Cost/Performance Engineering
{bullets(summaries[:1] + searches[1:2], limit=2)}

## Production Reliability/Security
- Prioritize drift alerts, access-control tests, prompt-injection regression tests, and incident runbooks over broad regulatory summaries unless the rule changes an engineering decision.

## Deep Dive
- **Model/prompt release gates**: Treat prompts, embedding models, rerankers, and model versions as deployable artifacts with immutable versions and eval thresholds. The practical failure mode is not only bad output quality; it is silent regressions in latency, retrieval permissions, and rollback reproducibility.

## Design Challenge
- You are deploying a multi-tenant RAG assistant with model registry promotion, cross-encoder reranking, and cost-sensitive inference routing. Where do you place the canary gate so it catches retrieval permission regressions, latency spikes, and answer-quality drift before full production rollout?

## Evidence Captured
{bullets(searches + summaries + notes, limit=6)}
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
    agent = ResearchAgent(
        llm_client=llm,
        registry=registry,
        short_term=stm,
        long_term=ltm,
        max_iterations=3,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    today = date.today()
    query = build_query(today)
    output_path = generate_report(
        query=query,
        online=args.online,
        output_dir=Path(args.output_dir),
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
