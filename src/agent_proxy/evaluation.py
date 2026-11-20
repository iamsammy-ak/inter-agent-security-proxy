"""Benchmarks: injection detection, PII detection and per-message latency.

Used by ``agent-proxy bench`` and ``benchmarks/run_benchmarks.py``. Results are plain dictionaries so they can be
saved as JSON and rendered as Markdown without extra dependencies.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .datasets import DATASET_URL, load_split
from .injection.classifier import InjectionClassifier
from .injection.detector import InjectionDetector
from .injection.embeddings import ModelUnavailable
from .models import PIIType
from .pii.detectors import PIIDetector, PIISpan
from .pii.masker import Masker
from .pii.names import SpacyNameFinder, spacy_model_available
from .pii.vault import TokenVault
from .policy import PolicyFile
from .proxy import SecurityProxy
from .synth import agentic_suite, pii_corpus

LINK_ALLOWLIST = ("example-corp.com",)


class Metrics(BaseModel):
    """Confusion counts and the usual ratios (0 when undefined)."""

    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    fpr: float

    @classmethod
    def from_counts(cls, tp: int, fp: int, fn: int, tn: int = 0) -> Metrics:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        return cls(tp=tp, fp=fp, fn=fn, tn=tn, precision=round(precision, 4), recall=round(recall, 4),
                   f1=round(f1, 4), fpr=round(fpr, 4))


def binary_metrics(y_true: Iterable[int], y_pred: Iterable[bool]) -> Metrics:
    tp = fp = fn = tn = 0
    for truth, pred in zip(y_true, y_pred, strict=True):
        if pred and truth:
            tp += 1
        elif pred:
            fp += 1
        elif truth:
            fn += 1
        else:
            tn += 1
    return Metrics.from_counts(tp, fp, fn, tn)


def _load_classifiers(names: Iterable[str]) -> dict[str, InjectionClassifier]:
    loaded: dict[str, InjectionClassifier] = {}
    for name in names:
        try:
            loaded[name] = InjectionClassifier.load(name)  # type: ignore[arg-type]
        except ModelUnavailable:
            continue
    return loaded


def _configs(classifiers: dict[str, InjectionClassifier]) -> dict[str, InjectionDetector]:
    configs = {"rules": InjectionDetector(None, link_allowlist=LINK_ALLOWLIST)}
    for name, clf in classifiers.items():
        configs[f"classifier ({name})"] = InjectionDetector(clf, use_rules=False, link_allowlist=LINK_ALLOWLIST)
        configs[f"rules + classifier ({name})"] = InjectionDetector(clf, link_allowlist=LINK_ALLOWLIST)
    return configs


def _example(text: str) -> str:
    """One-line, truncated copy of a dataset text for the error sample."""
    flat = " ".join(text.split())
    return flat[:110] + ("..." if len(flat) > 110 else "")


def evaluate_injection(
    data_dir: Path | None = None, classifier_names: Iterable[str] = ("bge-small", "hashing")
) -> dict[str, Any]:
    """Rules vs classifier vs combined on the deepset test split and on the synthetic agentic suite."""
    classifiers = _load_classifiers(classifier_names)
    configs = _configs(classifiers)
    test = load_split("test", data_dir)
    labels = [r.label for r in test]
    deepset: dict[str, Any] = {}
    errors: dict[str, list[str]] = {}
    for name, detector in configs.items():
        flags = [detector.inspect(r.text).is_injection for r in test]
        deepset[name] = binary_metrics(labels, flags).model_dump()
        if name.startswith("rules + classifier") and not errors:
            fn = [_example(r.text) for r, f in zip(test, flags, strict=True) if r.label and not f]
            fp = [_example(r.text) for r, f in zip(test, flags, strict=True) if f and not r.label]
            errors = {"config": [name], "missed_injections": fn[:6], "false_alarms": fp[:6]}
    suite = agentic_suite()
    agentic: dict[str, Any] = {}
    for name, detector in configs.items():
        flags = [detector.inspect(item.text).is_injection for item in suite]
        per_transform: dict[str, list[bool]] = defaultdict(list)
        for item, flag in zip(suite, flags, strict=True):
            if item.label:
                per_transform[item.transform].append(flag)
        benign = [flag for item, flag in zip(suite, flags, strict=True) if not item.label]
        agentic[name] = {
            "overall": binary_metrics([i.label for i in suite], flags).model_dump(),
            "recall_by_transform": {t: round(sum(v) / len(v), 4) for t, v in per_transform.items()},
            "benign_false_positive_rate": round(sum(benign) / len(benign), 4),
        }
    return {
        "dataset": {"name": "deepset/prompt-injections", "url": DATASET_URL, "license": "Apache-2.0",
                    "split": "test", "rows": len(test), "injections": sum(labels)},
        "classifiers": {
            name: {"threshold": c.threshold, "cv_metrics_on_train": c.artefact.cv_metrics, "C": c.artefact.C}
            for name, c in classifiers.items()
        },
        "deepset_test": deepset,
        "errors_rules_plus_classifier": errors,
        "agentic_suite": {
            "description": "synthetic: 12 hand-written attack seeds x 10 obfuscations, plus 40 hand-written benign "
                           "agent messages; written by the rules' author, so optimistic for the rules",
            "attacks": sum(i.label for i in suite), "benign": sum(1 - i.label for i in suite),
            "results": agentic,
        },
    }


def _match(gold: list[tuple[PIIType, int, int]], pred: list[PIISpan]) -> tuple[int, list[bool], list[bool]]:
    """Greedy one-to-one matching: same type and intersection-over-union >= 0.5."""
    gold_hit = [False] * len(gold)
    pred_hit = [False] * len(pred)
    for gi, (gtype, gs, ge) in enumerate(gold):
        for pi, span in enumerate(pred):
            if pred_hit[pi] or span.type != gtype:
                continue
            inter = max(0, min(ge, span.end) - max(gs, span.start))
            union = max(ge, span.end) - min(gs, span.start)
            if union and inter / union >= 0.5:
                gold_hit[gi] = pred_hit[pi] = True
                break
    return sum(gold_hit), gold_hit, pred_hit


def evaluate_pii(n: int = 600, seed: int = 7, *, names: bool | None = None) -> dict[str, Any]:
    """Span-level precision and recall per PII class on the seeded synthetic corpus."""
    corpus = pii_corpus(n, seed)
    use_names = spacy_model_available() if names is None else names
    detectors = {"regex + validators": PIIDetector()}
    if use_names:
        detectors["regex + validators + spaCy names"] = PIIDetector(extra_finders=[SpacyNameFinder()])
    results: dict[str, Any] = {}
    for label, detector in detectors.items():
        counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # tp, fp, fn
        exact = 0
        for item in corpus:
            gold = [g for g in item.spans if ("spaCy" in label) or g[0] is not PIIType.PERSON]
            pred = detector.detect(item.text)
            _, gold_hit, pred_hit = _match(gold, pred)
            for (gtype, gs, ge), hit in zip(gold, gold_hit, strict=True):
                counts[gtype.value][0 if hit else 2] += 1
                exact += any(p.type == gtype and p.start == gs and p.end == ge for p in pred) if hit else 0
            for span, hit in zip(pred, pred_hit, strict=True):
                if not hit:
                    counts[span.type.value][1] += 1
        per_type = {t: Metrics.from_counts(tp, fp, fn).model_dump(exclude={"tn", "fpr"})
                    for t, (tp, fp, fn) in sorted(counts.items())}
        tp, fp, fn = (sum(c[i] for c in counts.values()) for i in range(3))
        results[label] = {"overall": Metrics.from_counts(tp, fp, fn).model_dump(exclude={"tn", "fpr"}),
                          "exact_boundary_rate": round(exact / tp, 4) if tp else 0.0, "per_type": per_type}
    return {
        "corpus": {"sentences": len(corpus), "seed": seed,
                   "italian_share": round(sum(c.lang == "it" for c in corpus) / len(corpus), 3),
                   "gold_spans": sum(len(c.spans) for c in corpus),
                   "hard_negatives": sum(c.negatives for c in corpus),
                   "out_of_scope_variants": sum(c.variants for c in corpus),
                   "note": "synthetic values from agent_proxy.synth (seeded); not real people"},
        "matching": "same class and intersection-over-union >= 0.5",
        "results": results,
    }


def _percentiles(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)

    def pct(q: float) -> float:
        return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 3)

    return {"p50_ms": pct(0.50), "p95_ms": pct(0.95), "mean_ms": round(statistics.fmean(ordered), 3)}


def _classifier_stage(classifier: InjectionClassifier) -> Callable[[str], object]:
    return lambda text: classifier.score([text])


def _screen_stage(proxy: SecurityProxy) -> Callable[[str], object]:
    return lambda text: proxy.screen(text, session_id="bench", source="agent", kind="tool_result")


def measure_latency(
    groups: dict[str, list[str]], classifier_names: Iterable[str] = ("bge-small", "hashing"), repeats: int = 3
) -> dict[str, Any]:
    """Per-message wall time of each stage and of the whole screening step, per group of messages.

    Each message is timed ``repeats`` times and the fastest run kept, which filters out pauses caused by other
    processes on the machine; percentiles are then taken across messages."""
    classifiers = _load_classifiers(classifier_names)
    masker = Masker(TokenVault(b"bench"))
    rules = InjectionDetector(None, link_allowlist=LINK_ALLOWLIST)
    stages: dict[str, Callable[[str], object]] = {
        "pii masking": lambda t: masker.mask(t, "bench"),
        "rules + hidden-text + decoding": rules.inspect,
    }
    for name, clf in classifiers.items():
        stages[f"classifier ({name})"] = _classifier_stage(clf)
    for name, clf in classifiers.items():
        proxy = SecurityProxy(PolicyFile(link_allowlist=list(LINK_ALLOWLIST)), vault=TokenVault(b"bench"),
                              classifier=clf)
        stages[f"full screen with {name} (detect + mask + spotlight)"] = _screen_stage(proxy)
    warmup = next(iter(groups.values()))[:5]
    for fn in stages.values():  # model sessions, regex caches
        for text in warmup:
            fn(text)
    results: dict[str, Any] = {}
    for group, texts in groups.items():
        timings: dict[str, Any] = {}
        for name, fn in stages.items():
            samples = []
            for text in texts:
                runs = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    fn(text)
                    runs.append((time.perf_counter() - start) * 1000)
                samples.append(min(runs))
            timings[name] = _percentiles(samples)
        lengths = sorted(len(t) for t in texts)
        results[group] = {"messages": len(texts), "median_chars": lengths[len(lengths) // 2],
                          "max_chars": lengths[-1], "stages": timings}
    return {"machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
            "method": f"best of {repeats} runs per message, then percentiles across messages",
            "groups": results}


def latency_groups(data_dir: Path | None = None) -> dict[str, list[str]]:
    """Short messages (deepset test split, agentic suite, 100 synthetic PII sentences) and 24 long synthetic tool
    results of 4-6 thousand characters, where chunked classification dominates."""
    short = [item.text for item in agentic_suite()] + [c.text for c in pii_corpus(100, seed=11)]
    try:
        short = [r.text for r in load_split("test", data_dir)] + short
    except FileNotFoundError:
        pass
    sentences = [c.text for c in pii_corpus(400, seed=13)]
    long: list[str] = []
    cursor = 0
    for i in range(24):
        target = 4000 + 85 * i
        parts: list[str] = []
        while sum(len(p) + 1 for p in parts) < target:
            parts.append(sentences[cursor % len(sentences)])
            cursor += 1
        long.append("\n".join(parts))
    return {"short messages": short, "long tool results": long}


# ----------------------------------------------------------------------------------------------- output


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def injection_markdown(result: dict[str, Any]) -> str:
    lines = ["# Injection detection", "",
             f"Data: [{result['dataset']['name']}]({result['dataset']['url']}) ({result['dataset']['license']}), "
             f"{result['dataset']['split']} split: {result['dataset']['rows']} rows, "
             f"{result['dataset']['injections']} injections. Classifiers were trained on the train split only; "
             "thresholds were chosen by 5-fold cross-validation on train.", "",
             "| Detector | Precision | Recall | F1 | False-positive rate |", "|---|---:|---:|---:|---:|"]
    for name, m in result["deepset_test"].items():
        lines.append(f"| {name} | {_fmt(m['precision'])} | {_fmt(m['recall'])} | {_fmt(m['f1'])} | {_fmt(m['fpr'])} |")
    suite = result["agentic_suite"]
    transforms = list(next(iter(suite["results"].values()))["recall_by_transform"])
    heading = f"## Agentic suite ({suite['attacks']} attacks, {suite['benign']} benign; {suite['description']})"
    lines += ["", heading, "",
              "| Detector | " + " | ".join(transforms) + " | Benign FPR |",
              "|---|" + "---:|" * (len(transforms) + 1)]
    for name, r in suite["results"].items():
        cells = " | ".join(_fmt(r["recall_by_transform"][t]) for t in transforms)
        lines.append(f"| {name} | {cells} | {_fmt(r['benign_false_positive_rate'])} |")
    errors = result.get("errors_rules_plus_classifier") or {}
    if errors:
        lines += ["", f"## Error sample ({errors['config'][0]}, deepset test)", "", "Missed injections:"]
        lines += [f"- `{e}`" for e in errors["missed_injections"]] or ["- none"]
        lines += ["", "False alarms on benign text:"]
        lines += [f"- `{e}`" for e in errors["false_alarms"]] or ["- none"]
    return "\n".join(lines) + "\n"


def pii_markdown(result: dict[str, Any]) -> str:
    c = result["corpus"]
    lines = ["# PII detection", "",
             f"Synthetic corpus: {c['sentences']} sentences (seed {c['seed']}, {c['italian_share']:.0%} Italian), "
             f"{c['gold_spans']} gold spans ({c['out_of_scope_variants']} in spellings the detectors do not target, "
             f"such as `name [at] example [dot] com` or dotted phone numbers), {c['hard_negatives']} hard negatives. "
             f"Matching: {result['matching']}.", ""]
    for label, r in result["results"].items():
        o = r["overall"]
        lines += [f"## {label}", "",
                  f"Overall precision {_fmt(o['precision'])}, recall {_fmt(o['recall'])}, F1 {_fmt(o['f1'])}; "
                  f"exact boundaries on {r['exact_boundary_rate']:.1%} of matches.", "",
                  "| Class | Precision | Recall | F1 | TP | FP | FN |", "|---|---:|---:|---:|---:|---:|---:|"]
        for t, m in r["per_type"].items():
            lines.append(f"| {t} | {_fmt(m['precision'])} | {_fmt(m['recall'])} | {_fmt(m['f1'])} | {m['tp']} | "
                         f"{m['fp']} | {m['fn']} |")
        lines.append("")
    return "\n".join(lines)


def latency_markdown(result: dict[str, Any]) -> str:
    lines = ["# Latency per message", "", f"{result['machine']}; single process, CPU only; {result['method']}."]
    for group, g in result["groups"].items():
        lines += ["", f"## {group}: {g['messages']} messages (median {g['median_chars']} characters, "
                      f"max {g['max_chars']})", "",
                  "| Stage | p50 (ms) | p95 (ms) | mean (ms) |", "|---|---:|---:|---:|"]
        for name, t in g["stages"].items():
            lines.append(f"| {name} | {t['p50_ms']:.2f} | {t['p95_ms']:.2f} | {t['mean_ms']:.2f} |")
    return "\n".join(lines) + "\n"


def write_result(out_dir: Path, name: str, result: dict[str, Any], markdown: str, command: str) -> Path:
    """Save ``<name>.json`` and ``<name>.md`` (with the reproduce command) and return the JSON path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"command": command, **result}
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / f"{name}.md").write_text(f"{markdown}\nReproduce: `{command}`\n", encoding="utf-8")
    return path


def run_suites(suite: str, out_dir: Path, command: str, log: Callable[[str], None] = print) -> None:
    """Run ``all`` or one of ``injection``, ``pii``, ``latency`` and write the results."""
    if suite in ("all", "injection"):
        try:
            result = evaluate_injection()
        except FileNotFoundError as exc:
            log(f"skipping injection benchmark: {exc}")
        else:
            write_result(out_dir, "injection", result, injection_markdown(result), command)
            for name, m in result["deepset_test"].items():
                log(f"injection  {name:<34} P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}")
    if suite in ("all", "pii"):
        result = evaluate_pii()
        write_result(out_dir, "pii", result, pii_markdown(result), command)
        for label, r in result["results"].items():
            o = r["overall"]
            log(f"pii        {label:<34} P {o['precision']:.3f}  R {o['recall']:.3f}  F1 {o['f1']:.3f}")
    if suite in ("all", "latency"):
        result = measure_latency(latency_groups())
        write_result(out_dir, "latency", result, latency_markdown(result), command)
        for group, g in result["groups"].items():
            for name, t in g["stages"].items():
                log(f"latency    {group[:5]} {name:<56} p50 {t['p50_ms']:.2f} ms  p95 {t['p95_ms']:.2f} ms")
