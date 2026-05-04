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
You are a senior ML educator and AI research analyst writing a daily briefing for an MLOps and \
Data Science architect who also works with GenAI/LLMs/Agentic AI.
Today is {report_date}. Internet search is ON — use it.

Your job has two parts:
1. Run 3 targeted web searches to gather today's news signals (for What's New).
2. Write a learning-first briefing: the Concept of the Day, Deployment Practice, and Design \
Challenge come from your deep knowledge — grounded in today's signals where relevant, but not \
limited to them. These sections must teach something concrete and memorable every day.

Do not call recall() or save_note(). Do not call summarize() separately; synthesize inline.

Search strategy — run all 3 searches in the first iteration, each with max_results=3 and timelimit="w":
  1. "AI LLM GenAI model release tool update {report_date}"
  2. "MLOps data science ML deployment monitoring practice {report_date}"
  3. "machine learning research benchmark fine-tuning statistics {report_date}"

After the searches complete, write the full report in the next iteration.
Do not add extra tool calls between the searches and the final answer.

{yesterday_context}\
"""


REPORT_QUERY_TEMPLATE = """\
Act as my Senior ML Educator and Research Lead. Today is {{Date}}.

I am an MLOps and Data Science architect with focus also on GenAI/LLMs/Agentic AI.
My primary goal is to learn something every day — conceptually and operationally.
Staying current on news is secondary.

Hard constraints:
- Total report UNDER 570 words. If it is long I stop reading it.
- Tight bullet points, not paragraphs. No section intros, no fluff.
- Every sentence carries a concrete signal, learning point, or decision implication.
- Search budget: at most 3 web searches, max_results=3 each, timelimit="w".
- If What's New has no strong signal, write: "No strong signal today."

---

## What's New
4 bullets max. Covers all domains: model/tool releases, MLOps practice shifts, DS/ML research
crossing into production. Format: **[thing]** — signal + one decision implication.
Skip pure marketing. Include classical ML, data engineering, LLMs, and MLOps tooling.

## Concept of the Day
One concept chosen from the rotation below. Use this exact structure:
- **What it is**: 1 sentence, plain definition — no jargon unless defined.
- **How it works**: 2 sentences, concrete example with real numbers or a before/after scenario.
- **Where it silently fails**: 2 sentences — specific production failure mode and why it is hard to catch.
- **Decision rule**: 1 sentence — a threshold, heuristic, or "if X then Y" the reader can apply immediately.

Rotate across these three domains equally — do not favour GenAI:
- Classical ML/stats: bias-variance tradeoff, model calibration, feature leakage, data leakage, \
train-test contamination, overfitting signals, class imbalance, A/B test design, statistical power, \
type I/II errors, p-value traps, confidence intervals, covariate shift, concept drift, \
distribution shift detection, cross-validation design, regularisation (L1/L2/elastic net), learning curves.
- DS practice: train-serve skew, feature engineering patterns, target encoding pitfalls, \
feature store design, data contracts, schema drift, label noise, dataset versioning, \
EDA failure modes, outlier handling, missing data strategies, stratified sampling.
- MLOps/GenAI: shadow deployment mechanics, canary rollout, model registry promotion gates, \
blue-green model swap, eval regression, embedding drift, RAG retrieval quality, prompt versioning, \
agentic loop failure modes, context window management, KV-cache reuse, inference autoscaling, \
cost attribution, online/offline parity, reranker latency budget.

## Quick Concept
One concept from a DIFFERENT domain than the Concept of the Day above. Use this compact structure:
- **What it is**: 1 sentence definition.
- **Why it matters**: 1 sentence — one concrete production implication.
- **Decision rule**: 1 sentence — a heuristic or "if X then Y" to apply immediately.

The two concepts must never be from the same domain on the same day.

## Deployment Practice
One operational pattern — something a practitioner wires up or decides during deployment or operations.
Structure:
- **Pattern**: 1 sentence — what it is.
- **When to use it**: 1 sentence — the trigger condition or signal that calls for this pattern.
- **What breaks without it**: 1 sentence — the specific, concrete failure mode.
- **How to implement it**: 1–2 sentences — one detail that makes it work in practice (a threshold, \
a tool contract, a data structure, a decision rule).

Rotate through: shadow deployment, canary release, blue-green model swap, model promotion gate, \
feature flag for model variants, rollback strategy, serving topology design, request batching, \
latency budget allocation, per-call cost attribution, pipeline contract enforcement, \
data drift alerting, experiment tracking discipline, training pipeline idempotency, \
load testing ML endpoints, A/B test instrumentation, SLA definition for ML systems.

## Design Challenge
3–4 sentences. One architecture problem. Rotate domains: classical ML system, data pipeline, \
GenAI/LLM, MLOps infrastructure, DS workflow — do not default to LLMs.
State: system context + one concrete numeric constraint (latency, dataset size, cost, team size) \
+ the specific decision to make + the failure mode to defend against.
End with **Consider:** followed by 2–3 pointed hints that scaffold thinking without giving the answer.

---

**Rule of Thumb:** One practitioner heuristic relevant to today's concept or challenge — \
one sentence, something concrete enough to remember and repeat.

---
Word budget: under 570 words total.
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
        "Write the final daily briefing from the evidence below for an MLOps and Data Science "
        "architect who also works with GenAI/LLMs. Do not call tools. Under 570 words total. "
        "Include exactly these five sections plus a closing Rule of Thumb line:\n"
        "1. What's New — 4 bullets max, all domains, format: **[thing]** — signal + implication.\n"
        "2. Concept of the Day — one concept (full depth) with: What it is / How it works "
        "(example with numbers) / Where it silently fails / Decision rule.\n"
        "3. Quick Concept — one concept from a DIFFERENT domain than #2, compact: "
        "What it is / Why it matters / Decision rule (3 lines total).\n"
        "4. Deployment Practice — one operational pattern with: Pattern / When to use it "
        "/ What breaks without it / How to implement it.\n"
        "5. Design Challenge — system context + numeric constraint + decision + failure mode + "
        "Consider: 2–3 hints. Rotate domains across classical ML, data pipeline, MLOps, GenAI.\n"
        "End with: **Rule of Thumb:** one memorable practitioner heuristic.\n"
        "If What's New has no evidence write 'No strong signal today.' "
        "Every line must carry a concrete learning point or decision implication.\n\n"
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

## What's New
{bullets(searches[:2] + recalls[:1])}

## Concept of the Day
- **What it is**: Train-serve skew is the divergence between the feature distribution seen during model training and the distribution of the same features at serving time.
- **How it works**: A model trained on batch-aggregated features (e.g. 7-day rolling mean computed offline) receives real-time point-in-time values at inference — the two pipelines compute the same feature name but with different logic or timing windows, producing systematically different inputs. A fraud model trained on monthly average transaction velocity will underestimate risk for a new account with 3 days of history.
- **Where it silently fails**: The model and serving pipeline both appear healthy — no errors, no latency spikes. Prediction quality degrades silently because the model has never seen the true inference-time distribution. Standard monitoring (error rate, latency, null rate) does not catch it.
- **Decision rule**: If your training pipeline and serving pipeline compute the same feature independently, they will diverge — enforce a single feature definition served from a feature store used by both.

## Quick Concept
- **What it is**: Concept drift is the change over time in the statistical relationship between input features and the target variable a model was trained to predict.
- **Why it matters**: A model predicting customer churn trained on pre-pandemic behaviour may silently degrade as spending patterns shift — the inputs look identical but the meaning of those patterns has changed, making staleness invisible to standard data quality checks.
- **Decision rule**: Schedule retraining triggers on business-cycle boundaries (seasonal shifts, market events) not just on data volume — time elapsed is often a better drift proxy than rows processed.

## Deployment Practice
- **Pattern**: Shadow deployment runs a new model version in parallel with production, receiving the same live traffic, but its outputs are logged rather than served to users.
- **When to use it**: Before any model swap where you lack sufficient offline eval coverage of production traffic distribution — especially on skewed or long-tail inputs.
- **What breaks without it**: You discover latency regressions, edge-case failures, or cost overruns only after the model is live, when rollback is expensive and user-visible.
- **How to implement it**: Route 100% of requests to both models; gate the shadow path behind a feature flag; compare output distributions (not just accuracy) and p99 latency over 24–48 hours before promoting.

## Design Challenge
You are deploying a new gradient-boosted model to replace a 2-year-old logistic regression on a customer churn prediction pipeline. The new model improves AUC by 6% on your holdout set (80k rows, 18 months of data). Inference runs in batch overnight. Your constraint: rollback must complete within one pipeline cycle (24 hours) and you cannot re-train within that window. The failure mode to defend against: the new model performs worse on a customer segment that is underrepresented in training data but high-value in production.

**Consider:** How do you slice your holdout eval to surface segment-level performance before promoting? What artefacts do you version so rollback is a config change, not a redeploy? How do you monitor for the segment degradation signal in the first 72 hours post-swap?

---

**Rule of Thumb:** A 6% AUC lift on a global holdout means nothing if it comes from the majority class — always check segment-level performance before promoting a model that serves a heterogeneous population.
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
