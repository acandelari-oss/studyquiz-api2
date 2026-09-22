#!/usr/bin/env python3
"""Standalone model-native PDF structure benchmark.

This script is intentionally isolated from DOUNO ingestion, chunking,
taxonomy, persistence, and Source Structure heuristics.  It sends a PDF
directly to the OpenAI Responses API as an input_file and asks the model to
return a machine-readable academic document structure.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover - exercised as CLI failure
    raise SystemExit(
        "OpenAI SDK is not available. Run this script inside the backend venv."
    ) from exc


# Optional pricing configuration.  Keep this empty unless intentionally
# benchmarking with a known pricing snapshot.  CLI flags can override it.
PRICING_USD_PER_1M_TOKENS: Dict[str, Dict[str, float]] = {}


CONFIDENCE_VALUES = ["HIGH", "MEDIUM", "LOW"]
MAX_REPRESENTATIVE_LOCAL_STRUCTURES = 10
MAX_NOTES = 5


DOCUMENT_STRUCTURE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "document_title",
        "document_type",
        "structure_confidence",
        "sections",
        "local_structure_summary",
        "notes",
    ],
    "properties": {
        "document_title": {"type": "string"},
        "document_type": {"type": "string"},
        "structure_confidence": {"type": "string", "enum": CONFIDENCE_VALUES},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "title",
                    "level",
                    "parent",
                    "start_page",
                    "end_page",
                    "confidence",
                ],
                "properties": {
                    "title": {"type": "string"},
                    "level": {"type": "integer", "minimum": 1, "maximum": 6},
                    "parent": {"type": ["string", "null"]},
                    "start_page": {"type": ["integer", "null"], "minimum": 1},
                    "end_page": {"type": ["integer", "null"], "minimum": 1},
                    "confidence": {"type": "string", "enum": CONFIDENCE_VALUES},
                },
            },
        },
        "local_structure_summary": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "detected",
                "types",
                "representative_examples",
            ],
            "properties": {
                "detected": {"type": "boolean"},
                "types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 10,
                },
                "representative_examples": {
                    "type": "array",
                    "maxItems": MAX_REPRESENTATIVE_LOCAL_STRUCTURES,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "page",
                            "type",
                            "title",
                            "confidence",
                        ],
                        "properties": {
                            "page": {"type": ["integer", "null"], "minimum": 1},
                            "type": {"type": "string"},
                            "title": {"type": "string"},
                            "confidence": {"type": "string", "enum": CONFIDENCE_VALUES},
                        },
                    },
                },
            },
        },
        "notes": {
            "type": "array",
            "maxItems": MAX_NOTES,
            "items": {"type": "string"},
        },
    },
}


LOCAL_STRUCTURE_SUMMARY_SCHEMA: Dict[str, Any] = (
    DOCUMENT_STRUCTURE_SCHEMA["properties"]["local_structure_summary"]
)


SYSTEM_INSTRUCTIONS = """\
You are an expert academic document-structure analyst.

Reconstruct the canonical document hierarchy intended by the source document.
Use the PDF itself, including visual/layout information when the model has
access to it.

Keep the output compact. Do not attempt to exhaustively describe the PDF. Do
not exhaustively describe local structures. Do not list ordinary body content.
Prefer concise titles and structured data over
natural-language explanations.

Do not promote local enumerations, questions, examples, tables, figures,
captions, headers, footers, page numbers, or artifacts to the global hierarchy
unless they genuinely function as sections of the document.

Report only representative local-structure examples, capped at 10 for the
entire document. Use notes only for exceptional document-level observations
that affect interpretation of the hierarchy, capped at 5.

Return only the structured JSON requested by the schema.
"""


USER_TASK = """\
Analyze this PDF and reconstruct its canonical academic/semantic hierarchy.

For the main document hierarchy, include only genuine major sections,
subsections, and lower-level sections. For each section, return only title,
level, parent title if any, start_page, end_page, and confidence.

Do not list ordinary body content. Do not exhaustively enumerate local
structures. Instead, summarize whether local/non-global structures were
detected, list their broad types, and provide at most 10 representative
examples for the whole document.
"""


@dataclass(frozen=True)
class UsageSummary:
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    reasoning_tokens: Optional[int]


@dataclass(frozen=True)
class ResponseDiagnostics:
    response_status: Optional[str]
    incomplete_reason: Optional[str]
    response_max_output_tokens: Optional[int]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    reasoning_tokens: Optional[int]
    raw_output_chars: int
    output_item_statuses: List[str]


@dataclass(frozen=True)
class CostSummary:
    input_cost: Optional[float]
    output_cost: Optional[float]
    total_cost: Optional[float]


class BenchmarkResponseError(ValueError):
    def __init__(self, message: str, diagnostics: Optional[ResponseDiagnostics] = None):
        super().__init__(message)
        self.diagnostics = diagnostics


def load_dotenv_if_present(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def pdf_page_count(pdf_path: Path) -> Optional[int]:
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(pdf_path)).pages)
    except Exception:
        return None


def encode_pdf_file_data(pdf_path: Path) -> str:
    encoded = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
    return f"data:application/pdf;base64,{encoded}"


def build_response_input(pdf_path: Path) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": USER_TASK},
                {
                    "type": "input_file",
                    "filename": pdf_path.name,
                    "file_data": encode_pdf_file_data(pdf_path),
                    "detail": "high",
                },
            ],
        }
    ]


def build_text_config() -> Dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": "document_structure_benchmark",
            "strict": True,
            "schema": DOCUMENT_STRUCTURE_SCHEMA,
        }
    }


def response_text(response: Any) -> str:
    direct = getattr(response, "output_text", None)
    if direct:
        return direct

    parts: List[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def parse_structured_result(
    raw_text: str,
    diagnostics: Optional[ResponseDiagnostics] = None,
) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        if diagnostics and (
            diagnostics.response_status == "incomplete"
            or diagnostics.incomplete_reason
        ):
            raise BenchmarkResponseError(
                "Response incomplete before valid JSON could be parsed: "
                f"{exc}",
                diagnostics,
            ) from exc
        raise BenchmarkResponseError(
            "Completed response did not contain valid structured JSON: "
            f"{exc}",
            diagnostics,
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError("Model JSON response must be an object.")
    for required_key in DOCUMENT_STRUCTURE_SCHEMA["required"]:
        if required_key not in parsed:
            raise ValueError(f"Model JSON response is missing '{required_key}'.")
    return parsed


def _get_attr_or_key(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def extract_usage(response: Any) -> UsageSummary:
    usage = _get_attr_or_key(response, "usage")
    output_details = _get_attr_or_key(usage, "output_tokens_details")
    reasoning_tokens = _get_attr_or_key(output_details, "reasoning_tokens")

    return UsageSummary(
        input_tokens=_get_attr_or_key(usage, "input_tokens"),
        output_tokens=_get_attr_or_key(usage, "output_tokens"),
        total_tokens=_get_attr_or_key(usage, "total_tokens"),
        reasoning_tokens=reasoning_tokens,
    )


def extract_response_diagnostics(response: Any, raw_text: str) -> ResponseDiagnostics:
    incomplete_details = _get_attr_or_key(response, "incomplete_details")
    usage = extract_usage(response)
    output_item_statuses: List[str] = []
    for item in _get_attr_or_key(response, "output") or []:
        status = _get_attr_or_key(item, "status")
        if status:
            output_item_statuses.append(str(status))

    return ResponseDiagnostics(
        response_status=_get_attr_or_key(response, "status"),
        incomplete_reason=_get_attr_or_key(incomplete_details, "reason"),
        response_max_output_tokens=_get_attr_or_key(response, "max_output_tokens"),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        raw_output_chars=len(raw_text),
        output_item_statuses=output_item_statuses,
    )


def format_response_diagnostics(diagnostics: ResponseDiagnostics) -> str:
    output_item_statuses = (
        ", ".join(diagnostics.output_item_statuses)
        if diagnostics.output_item_statuses
        else "n/a"
    )
    return "\n".join(
        [
            f"response_status: {diagnostics.response_status or 'n/a'}",
            f"incomplete_reason: {diagnostics.incomplete_reason or 'n/a'}",
            (
                "response_max_output_tokens: "
                f"{diagnostics.response_max_output_tokens if diagnostics.response_max_output_tokens is not None else 'n/a'}"
            ),
            f"input_tokens: {diagnostics.input_tokens if diagnostics.input_tokens is not None else 'n/a'}",
            f"output_tokens: {diagnostics.output_tokens if diagnostics.output_tokens is not None else 'n/a'}",
            f"total_tokens: {diagnostics.total_tokens if diagnostics.total_tokens is not None else 'n/a'}",
            (
                "reasoning_tokens: "
                f"{diagnostics.reasoning_tokens if diagnostics.reasoning_tokens is not None else 'n/a'}"
            ),
            f"raw_output_chars: {diagnostics.raw_output_chars}",
            f"output_item_statuses: {output_item_statuses}",
        ]
    )


def estimate_cost(
    usage: UsageSummary,
    model: str,
    input_cost_per_1m: Optional[float] = None,
    output_cost_per_1m: Optional[float] = None,
) -> CostSummary:
    model_pricing = PRICING_USD_PER_1M_TOKENS.get(model, {})
    input_rate = input_cost_per_1m if input_cost_per_1m is not None else model_pricing.get("input")
    output_rate = output_cost_per_1m if output_cost_per_1m is not None else model_pricing.get("output")

    input_cost = (
        usage.input_tokens * input_rate / 1_000_000
        if usage.input_tokens is not None and input_rate is not None
        else None
    )
    output_cost = (
        usage.output_tokens * output_rate / 1_000_000
        if usage.output_tokens is not None and output_rate is not None
        else None
    )
    total_cost = (
        input_cost + output_cost
        if input_cost is not None and output_cost is not None
        else None
    )
    return CostSummary(input_cost, output_cost, total_cost)


def validate_model_available(client: OpenAI, model: str) -> None:
    try:
        model_ids = {model_info.id for model_info in client.models.list().data}
    except Exception as exc:
        raise RuntimeError(f"Could not list available OpenAI models: {exc}") from exc

    if model not in model_ids:
        sample = ", ".join(sorted(model_ids)[:20])
        raise ValueError(
            f"Model '{model}' was not returned by this project's model list. "
            f"Available model sample: {sample}"
        )


def run_model_benchmark(
    client: OpenAI,
    pdf_path: Path,
    model: str,
    max_output_tokens: int,
) -> Dict[str, Any]:
    start = time.perf_counter()
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=build_response_input(pdf_path),
        text=build_text_config(),
        max_output_tokens=max_output_tokens,
        store=False,
    )
    elapsed = time.perf_counter() - start
    raw_text = response_text(response)
    diagnostics = extract_response_diagnostics(response, raw_text)
    parsed = parse_structured_result(raw_text, diagnostics=diagnostics)
    return {
        "response": response,
        "diagnostics": diagnostics,
        "raw_text": raw_text,
        "parsed": parsed,
        "elapsed_seconds": elapsed,
    }


def format_tree(result: Dict[str, Any]) -> str:
    sections = result.get("sections") or []
    lines: List[str] = []
    for section in sections:
        level = max(1, int(section.get("level") or 1))
        indent = "  " * (level - 1)
        pages = _format_page_range(section.get("start_page"), section.get("end_page"))
        confidence = section.get("confidence") or "n/a"
        lines.append(f"{indent}- {section.get('title', '')} [{confidence}, {pages}]")
    return "\n".join(lines) if lines else "(no document hierarchy returned)"


def format_local_structures(result: Dict[str, Any]) -> str:
    summary = result.get("local_structure_summary") or {}
    if not summary.get("detected"):
        return "(none detected)"

    lines = []
    types = summary.get("types") or []
    if types:
        lines.append(f"types: {', '.join(str(item) for item in types)}")

    examples = (summary.get("representative_examples") or [])[
        :MAX_REPRESENTATIVE_LOCAL_STRUCTURES
    ]
    if not examples:
        return "\n".join(lines) if lines else "(none returned)"

    lines.append("representative examples:")
    for item in examples:
        page = item.get("page")
        page_text = f"page {page}" if page is not None else "page n/a"
        lines.append(
            f"- {item.get('type', 'local')} | {page_text} | "
            f"{item.get('title', '')} [{item.get('confidence', 'n/a')}]"
        )
    return "\n".join(lines)


def _format_page_range(start_page: Any, end_page: Any) -> str:
    if start_page is None and end_page is None:
        return "pages n/a"
    if start_page == end_page or end_page is None:
        return f"page {start_page}"
    if start_page is None:
        return f"through page {end_page}"
    return f"pages {start_page}-{end_page}"


def money(value: Optional[float]) -> str:
    return "n/a" if value is None else f"${value:.6f}"


def print_report(
    *,
    model: str,
    pdf_path: Path,
    pages: Optional[int],
    diagnostics: ResponseDiagnostics,
    usage: UsageSummary,
    cost: CostSummary,
    elapsed_seconds: float,
    result: Dict[str, Any],
    raw_text: str,
) -> None:
    print("================================================")
    print("MODEL STRUCTURE BENCHMARK")
    print("================================================")
    print()
    print(f"MODEL: {model}")
    print(f"FILE: {pdf_path}")
    print(f"PAGES: {pages if pages is not None else 'n/a'}")
    print()
    print("RESPONSE")
    print(format_response_diagnostics(diagnostics))
    print()
    print("USAGE")
    print(f"input_tokens: {usage.input_tokens if usage.input_tokens is not None else 'n/a'}")
    print(f"output_tokens: {usage.output_tokens if usage.output_tokens is not None else 'n/a'}")
    print(f"total_tokens: {usage.total_tokens if usage.total_tokens is not None else 'n/a'}")
    print(f"reasoning_tokens: {usage.reasoning_tokens if usage.reasoning_tokens is not None else 'n/a'}")
    print()
    print("ESTIMATED COST")
    print(f"input: {money(cost.input_cost)}")
    print(f"output: {money(cost.output_cost)}")
    print(f"total: {money(cost.total_cost)}")
    print()
    print(f"TIME: {elapsed_seconds:.2f}s")
    print()
    print("DOCUMENT ANALYSIS")
    print(f"document_type: {result.get('document_type', 'n/a')}")
    print(f"structure_confidence: {result.get('structure_confidence', 'n/a')}")
    print()
    print("TREE")
    print(format_tree(result))
    print()
    print("LOCAL STRUCTURES NOT PROMOTED")
    print(format_local_structures(result))
    print()
    print("RAW STRUCTURED RESULT")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()
    print("RAW MODEL TEXT")
    print(raw_text)
    print("================================================")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark model-native PDF structure understanding."
    )
    parser.add_argument("pdf", nargs="?", help="Path to a PDF file.")
    parser.add_argument("--model", help="Single model ID to benchmark.")
    parser.add_argument("--models", nargs="+", help="One or more model IDs to benchmark sequentially.")
    parser.add_argument("--list-models", action="store_true", help="List model IDs available to this OpenAI project and exit.")
    parser.add_argument("--skip-model-validation", action="store_true", help="Skip the models.list() availability check.")
    parser.add_argument("--max-output-tokens", type=int, default=6000)
    parser.add_argument("--input-cost-per-1m", type=float, default=None)
    parser.add_argument("--output-cost-per-1m", type=float, default=None)
    parser.add_argument("--env-file", default=".env", help="Optional dotenv file to load before reading OPENAI_API_KEY.")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    load_dotenv_if_present(Path(args.env_file))

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is unavailable.", file=sys.stderr)
        return 2

    client = OpenAI(api_key=api_key)

    if args.list_models:
        try:
            for model in sorted(model_info.id for model_info in client.models.list().data):
                print(model)
        except Exception as exc:
            print(f"ERROR: Could not list OpenAI models: {exc}", file=sys.stderr)
            return 2
        return 0

    if not args.pdf:
        print("ERROR: PDF path is required unless --list-models is used.", file=sys.stderr)
        return 2

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        print(f"ERROR: PDF file not found: {pdf_path}", file=sys.stderr)
        return 2
    if pdf_path.suffix.lower() != ".pdf":
        print(f"ERROR: Expected a .pdf file, got: {pdf_path.name}", file=sys.stderr)
        return 2

    models = args.models or ([args.model] if args.model else [])
    if not models:
        print("ERROR: Provide --model <MODEL> or --models <MODEL...>.", file=sys.stderr)
        return 2

    pages = pdf_page_count(pdf_path)
    overall_status = 0

    for model in models:
        try:
            if not args.skip_model_validation:
                validate_model_available(client, model)
            benchmark = run_model_benchmark(
                client=client,
                pdf_path=pdf_path,
                model=model,
                max_output_tokens=args.max_output_tokens,
            )
            usage = extract_usage(benchmark["response"])
            cost = estimate_cost(
                usage,
                model,
                input_cost_per_1m=args.input_cost_per_1m,
                output_cost_per_1m=args.output_cost_per_1m,
            )
            print_report(
                model=model,
                pdf_path=pdf_path,
                pages=pages,
                diagnostics=benchmark["diagnostics"],
                usage=usage,
                cost=cost,
                elapsed_seconds=benchmark["elapsed_seconds"],
                result=benchmark["parsed"],
                raw_text=benchmark["raw_text"],
            )
        except BenchmarkResponseError as exc:
            overall_status = 1
            print("================================================", file=sys.stderr)
            print("MODEL STRUCTURE BENCHMARK FAILED", file=sys.stderr)
            print("================================================", file=sys.stderr)
            print(f"MODEL: {model}", file=sys.stderr)
            print(f"FILE: {pdf_path}", file=sys.stderr)
            print(f"ERROR: {exc}", file=sys.stderr)
            if exc.diagnostics:
                print("RESPONSE DIAGNOSTICS", file=sys.stderr)
                print(format_response_diagnostics(exc.diagnostics), file=sys.stderr)
        except Exception as exc:
            overall_status = 1
            print("================================================", file=sys.stderr)
            print("MODEL STRUCTURE BENCHMARK FAILED", file=sys.stderr)
            print("================================================", file=sys.stderr)
            print(f"MODEL: {model}", file=sys.stderr)
            print(f"FILE: {pdf_path}", file=sys.stderr)
            print(f"ERROR: {exc}", file=sys.stderr)

    return overall_status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
