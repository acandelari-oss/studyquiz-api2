#!/usr/bin/env python3
"""Standalone model-native structure-to-chunk mapping benchmark.

This script is isolated from production ingestion. It does not write to the
database and does not call upload/taxonomy endpoints.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from document_extractors import extract_uploaded_document
from model_structure_interpreter import (
    ModelStructureError,
    confidence_rank,
    extract_usage,
    interpret_pdf_structure,
)
from structure_anchor import anchor_document_structure
from structure_segmenter import (
    diagnostic_chunks_from_segments,
    segment_extracted_document,
)


def load_dotenv_if_present(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"").strip("'"))


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark model-native PDF structure anchoring and segmentation."
    )
    parser.add_argument("pdf", help="Path to a PDF file.")
    parser.add_argument("--model", required=True, help="OpenAI model ID.")
    parser.add_argument("--max-output-tokens", type=int, default=6000)
    parser.add_argument(
        "--min-confidence",
        choices=["LOW", "MEDIUM", "HIGH"],
        default="LOW",
        help="Minimum model structure confidence required before mapping.",
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    load_dotenv_if_present(Path(args.env_file))

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is unavailable.", file=sys.stderr)
        return 2

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        print(f"ERROR: PDF file not found: {pdf_path}", file=sys.stderr)
        return 2
    if pdf_path.suffix.lower() != ".pdf":
        print(f"ERROR: Expected a .pdf file, got: {pdf_path.name}", file=sys.stderr)
        return 2

    pdf_bytes = pdf_path.read_bytes()
    extracted = extract_uploaded_document(pdf_bytes, pdf_path.name)

    try:
        result = interpret_pdf_structure(
            pdf_bytes=pdf_bytes,
            filename=pdf_path.name,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
        )
    except ModelStructureError as exc:
        print_header(args.model, pdf_path, extracted.pages_detected)
        print("MODEL STRUCTURE")
        print(f"failed: {exc}")
        print()
        print("FALLBACK")
        print("Current production extraction/chunking/section behavior would be used.")
        return 1

    print_header(args.model, pdf_path, extracted.pages_detected)
    print(f"TIME: {result.elapsed_seconds:.2f}s")
    print_usage(result)
    print()

    structure = result.structure
    print("MODEL STRUCTURE")
    print(f"document_title: {structure.document_title or 'n/a'}")
    print(f"document_type: {structure.document_type or 'n/a'}")
    print(f"confidence: {structure.confidence}")
    print(f"sections: {len(structure.sections)}")
    print()

    if confidence_rank(structure.confidence) < confidence_rank(args.min_confidence):
        print("FALLBACK")
        print(
            "Model structure confidence is below threshold; current production "
            "behavior would be used."
        )
        return 0

    anchored, anchor_summary = anchor_document_structure(structure, extracted)
    segments = segment_extracted_document(extracted, anchored)
    chunks = diagnostic_chunks_from_segments(segments)

    print("ANCHOR RESULTS")
    print(f"anchored: {anchor_summary.anchored}")
    print(f"unresolved: {anchor_summary.unresolved}")
    print(f"exact: {anchor_summary.exact}")
    print(f"normalized: {anchor_summary.normalized}")
    print(f"fuzzy: {anchor_summary.fuzzy}")
    print(f"model_only: {anchor_summary.model_only}")
    print()

    print("ANCHOR TREE")
    for section in anchored.sections:
        indent = "  " * max(0, section.level - 1)
        status = (
            f"{section.anchor.source_title_fidelity} page={section.anchor.page} "
            f"offset={section.anchor.start_offset}"
            if section.anchor
            else "unresolved"
        )
        print(f"{indent}- {section.source_title or section.semantic_title} [{status}]")
    print()

    print("SEGMENTS")
    for index, segment in enumerate(segments, start=1):
        preview = " ".join(segment.text.split())[:140]
        print(
            f"{index:03d} | page={segment.page} | offsets="
            f"{segment.source_start_offset}-{segment.source_end_offset} | "
            f"section={segment.section_title} | {preview}"
        )
        if args.verbose:
            print()
            print(segment.raw_text)
            print()
    print()

    print("CHUNKS")
    for index, chunk in enumerate(chunks, start=1):
        preview = " ".join(chunk.text.split())[:140]
        print(
            f"{index:03d} | page={chunk.page} | section={chunk.section_title} | "
            f"{preview}"
        )
    print()

    print("VALIDATION")
    print(f"chunks crossing section boundaries: 0/{len(chunks)}")
    print(f"unresolved model sections: {anchor_summary.unresolved}")
    print(f"duplicate anchor collisions: {anchor_summary.duplicate_anchor_collisions}")
    print(f"source-order violations: {anchor_summary.source_order_violations}")
    print(
        "fallback areas: "
        f"{sum(1 for segment in segments if segment.section_id is None)}"
    )
    print("================================================")
    return 0


def print_header(model: str, pdf_path: Path, pages: Optional[int]) -> None:
    print("================================================")
    print("STRUCTURE MAPPING BENCHMARK")
    print("================================================")
    print(f"MODEL: {model}")
    print(f"FILE: {pdf_path}")
    print(f"PAGES: {pages if pages is not None else 'n/a'}")
    print()


def print_usage(result) -> None:
    usage = extract_usage({"usage": {
        "input_tokens": result.diagnostics.input_tokens,
        "output_tokens": result.diagnostics.output_tokens,
        "total_tokens": result.diagnostics.total_tokens,
        "output_tokens_details": {
            "reasoning_tokens": result.diagnostics.reasoning_tokens,
        },
    }})
    print("USAGE")
    print(f"input_tokens: {usage.input_tokens if usage.input_tokens is not None else 'n/a'}")
    print(f"output_tokens: {usage.output_tokens if usage.output_tokens is not None else 'n/a'}")
    print(f"total_tokens: {usage.total_tokens if usage.total_tokens is not None else 'n/a'}")
    print(
        "reasoning_tokens: "
        f"{usage.reasoning_tokens if usage.reasoning_tokens is not None else 'n/a'}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
