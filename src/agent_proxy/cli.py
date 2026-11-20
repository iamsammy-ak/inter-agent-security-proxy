"""Command-line interface: ``agent-proxy scan | mask | serve | verify-audit | bench | demo``."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .audit import AuditLog, verify_audit_log
from .config import DEFAULT_PORT, load_settings
from .injection.classifier import load_classifier
from .pii.detectors import PIIDetector
from .pii.masker import Masker
from .pii.names import SpacyNameFinder
from .pii.vault import TokenVault
from .policy import PolicyFile, load_policy
from .proxy import SecurityProxy

app = typer.Typer(
    help="Security proxy for agent-to-agent traffic: PII tokenisation, injection defence, least-privilege tool "
    "policy and a tamper-evident audit log.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

CLASSIFIER_HELP = "auto (embedding model, else hashing), bge-small, hashing or off"


def _read(text: Optional[str], file: Optional[Path]) -> str:
    if file is not None:
        return file.read_text(encoding="utf-8")
    if text is None or text == "-":
        return sys.stdin.read()
    return text


def _proxy(classifier: str, allow: list[str], names: bool = False) -> SecurityProxy:
    detector = PIIDetector(extra_finders=[SpacyNameFinder()] if names else ())
    return SecurityProxy(PolicyFile(link_allowlist=allow), classifier=load_classifier(classifier),
                         pii_detector=detector)


@app.command()
def scan(
    text: Optional[str] = typer.Argument(None, help="text to scan; '-' or omitted reads stdin"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="read the text from a file"),
    classifier: str = typer.Option("auto", help=CLASSIFIER_HELP),
    allow: list[str] = typer.Option([], "--allow-domain", help="host allowed in links and images (repeatable)"),
    as_json: bool = typer.Option(False, "--json", help="print the full report as JSON"),
) -> None:
    """Check text for prompt injection and PII. Exit code 1 when an injection is detected."""
    report = _proxy(classifier, allow).inspect(_read(text, file), session_id="cli")
    verdict = report.verdict
    if as_json:
        console.print_json(report.model_dump_json())
    else:
        style = "bold red" if verdict.is_injection else "bold green"
        clf = "off" if verdict.classifier_score is None else f"{verdict.classifier_score:.3f}"
        console.print(Text(f"injection: {verdict.is_injection}  score {verdict.score:.3f}  "
                           f"(rules {verdict.rule_score:.3f}, classifier {clf})", style=style))
        if verdict.hits:
            table = Table("rule", "category", "weight", "where", "excerpt", show_edge=False, header_style="bold")
            for hit in verdict.hits:
                table.add_row(hit.rule_id, hit.category, f"{hit.weight:.2f}", hit.view, Text(hit.excerpt[:60]))
            console.print(table)
        console.print(f"PII: {report.pii_counts or 'none'}")
    raise typer.Exit(code=1 if verdict.is_injection else 0)


@app.command()
def mask(
    text: Optional[str] = typer.Argument(None, help="text to mask; '-' or omitted reads stdin"),
    file: Optional[Path] = typer.Option(None, "--file", "-f"),
    session: str = typer.Option("cli", help="session id: the same value gets the same token within a session"),
    names: bool = typer.Option(False, help="also mask person names with spaCy en_core_web_sm"),
    as_json: bool = typer.Option(False, "--json", help="print masked text and entities as JSON"),
) -> None:
    """Replace PII with typed tokens such as <EMAIL_7f3a9c1e>."""
    settings = load_settings()
    detector = PIIDetector(extra_finders=[SpacyNameFinder()] if names else ())
    result = Masker(TokenVault(settings.secret), detector).mask(_read(text, file), session)
    if as_json:
        console.print_json(result.model_dump_json())
    else:
        console.print(Text(result.text))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(DEFAULT_PORT),
    policy: Optional[Path] = typer.Option(None, help="YAML policy (default: the bundled demo policy)"),
    audit: Path = typer.Option(Path("audit.jsonl"), help="audit log file (JSON Lines, hash-chained)"),
    offline: bool = typer.Option(False, help="use the mock upstream even if OPENROUTER_API_KEY is set"),
    classifier: str = typer.Option("auto", help=CLASSIFIER_HELP),
    names: bool = typer.Option(False, help="mask person names with spaCy"),
) -> None:
    """Run the HTTP proxy (OpenAI-compatible /v1/chat/completions, /v1/inspect, /v1/relay)."""
    import uvicorn

    from .demo import demo_policy_path
    from .llm import select_upstream
    from .server import create_app

    settings = load_settings()
    if policy is None:
        policy = demo_policy_path()
        console.print("[yellow]No --policy given: using the bundled demo policy with public demo keys.[/]")
    detector = PIIDetector(extra_finders=[SpacyNameFinder()] if names else ())
    proxy = SecurityProxy(load_policy(policy), vault=TokenVault(settings.secret),
                          classifier=load_classifier(classifier), pii_detector=detector,
                          audit=AuditLog(audit, key=settings.audit_key))
    upstream = select_upstream(settings, offline=offline)
    console.print(f"upstream: {upstream.name}; listening on http://{host}:{port}")
    uvicorn.run(create_app(proxy, upstream), host=host, port=port, log_level="info")


@app.command("verify-audit")
def verify_audit(
    path: Path = typer.Argument(..., help="audit log to check"),
    key_env: str = typer.Option("AGENT_PROXY_AUDIT_KEY", help="environment variable holding the HMAC key, if any"),
    expect_head: Optional[str] = typer.Option(None, help="pinned hash of the last record (detects truncation)"),
) -> None:
    """Verify the hash chain (and HMACs when a key is set). Exit code 1 on any tampering."""
    result = verify_audit_log(path, key=os.environ.get(key_env) or None, expected_head=expect_head)
    if result.ok:
        console.print(f"[bold green]OK[/]: {result.records} records, head {result.head}")
    else:
        console.print(f"[bold red]TAMPERED[/] at record {result.error_seq}: {result.error} "
                      f"({result.records} records verified before it)")
        raise typer.Exit(code=1)


@app.command()
def bench(
    suite: str = typer.Option("all", help="all, injection, pii or latency"),
    out: Path = typer.Option(Path("benchmarks/results"), help="output directory for JSON and Markdown"),
    download: bool = typer.Option(False, help="download the deepset dataset first (network)"),
) -> None:
    """Run the benchmarks and write results (the injection suite needs the downloaded dataset)."""
    from .datasets import download_deepset
    from .evaluation import run_suites

    if download:
        download_deepset()
    run_suites(suite, out, f"agent-proxy bench --suite {suite}", log=console.print)


@app.command()
def demo(
    svg: Optional[Path] = typer.Option(None, help="also save the output as an SVG terminal recording"),
    classifier: str = typer.Option("auto", help=CLASSIFIER_HELP),
    names: Optional[bool] = typer.Option(None, "--names/--no-names", help="mask person names (default: if spaCy "
                                         "and en_core_web_sm are installed)"),
) -> None:
    """Offline walk-through: a poisoned tool result passes from one agent to another through the proxy."""
    from .demo import run_demo

    recorder = Console(width=100, record=True) if svg else console
    outcome = run_demo(recorder, load_classifier(classifier), names=names)
    if svg:
        svg.parent.mkdir(parents=True, exist_ok=True)
        markup = recorder.export_svg(title="agent-proxy demo")
        svg.write_text(re.sub(r"\s*<!--.*?-->", "", markup, count=1), encoding="utf-8")  # drop the exporter banner
        console.print(f"saved {svg.name}")
    if not (outcome.injection_detected and outcome.audit_ok):
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

