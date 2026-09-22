#!/usr/bin/env python3
"""Universal isolated structure-mapping benchmark for PDF, DOCX, and PPTX."""

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

from canonical_document import adapt_document
from universal_structure_interpreter import UniversalStructureError, interpret_document_structure
from universal_structure_mapping import build_mapping_result


def load_dotenv_if_present(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"").strip("'"))


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark universal document structure mapping."
    )
    parser.add_argument("file", help="Path to PDF, DOCX, or PPTX.")
    parser.add_argument("--model", required=True, help="OpenAI model ID.")
    parser.add_argument("--max-output-tokens", type=int, default=6000)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    load_dotenv_if_present(Path(args.env_file))
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is unavailable.", file=sys.stderr)
        return 2

    path = Path(args.file).expanduser().resolve()
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        return 2

    file_bytes = path.read_bytes()
    document = adapt_document(file_bytes, path.name)

    try:
        interpretation = interpret_document_structure(
            document=document,
            original_file_bytes=file_bytes if document.file_type == "pdf" else None,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
        )
        result = build_mapping_result(document, interpretation.structure)
    except UniversalStructureError as exc:
        interpretation = None
        print_header(document)
        print("MODEL STRUCTURE")
        print(f"failed: {exc}")
        result = build_mapping_result(document, None, interpretation_failed=True)
        print_result(result, verbose=args.verbose)
        return 1

    print_header(document)
    print("MODEL STRUCTURE")
    print(f"document_title: {interpretation.structure.document_title or 'n/a'}")
    print(f"document_type: {interpretation.structure.document_type or 'n/a'}")
    print(f"confidence: {interpretation.structure.confidence}")
    print(f"sections: {len(interpretation.structure.sections)}")
    print()
    print("USAGE / TIME")
    print(f"time: {interpretation.elapsed_seconds:.2f}s")
    print(f"input_tokens: {interpretation.diagnostics.input_tokens or 'n/a'}")
    print(f"output_tokens: {interpretation.diagnostics.output_tokens or 'n/a'}")
    print(f"total_tokens: {interpretation.diagnostics.total_tokens or 'n/a'}")
    print()
    print_result(result, verbose=args.verbose)
    return 0


def print_header(document) -> None:
    print("================================================")
    print("UNIVERSAL STRUCTURE MAPPING BENCHMARK")
    print("================================================")
    print("DOCUMENT")
    print(f"file: {document.filename}")
    print(f"type: {document.file_type}")
    print(f"units: {len(document.units)}")
    print(f"blocks: {len(document.blocks)}")
    print()


def print_result(result, *, verbose: bool = False) -> None:
    print("ANCHOR RESULTS")
    print(f"anchored: {result.sections_anchored}")
    print(f"unresolved: {result.unresolved_sections}")
    print(f"exact: {result.exact_count}")
    print(f"normalized: {result.normalized_count}")
    print(f"fuzzy: {result.fuzzy_count}")
    print(f"model_only: {result.model_only_count}")
    print()
    print("MAPPING DECISION")
    print(f"structure_status: {result.structure_status}")
    print(f"mapping_status: {result.mapping_status}")
    print(f"mapping_confidence: {result.mapping_confidence}")
    print()
    print("SEGMENTS")
    for index, segment in enumerate(result.segments, start=1):
        preview = " ".join(segment.text.split())[:140]
        print(
            f"{index:03d} | {segment.source_start.unit_type}="
            f"{segment.source_start.unit_index} | section={segment.section_path} | "
            f"{preview}"
        )
        if verbose:
            print(segment.raw_text)
    print()
    print("CHUNKS")
    for index, chunk in enumerate(result.diagnostic_chunks, start=1):
        preview = " ".join(chunk.text.split())[:140]
        print(f"{index:03d} | {chunk.source_unit} | section={chunk.section_path} | {preview}")
    print()
    print("VALIDATION")
    print(f"boundary crossing: {result.diagnostics.get('chunks_crossing_boundaries', 0)}")
    print(f"unresolved: {result.unresolved_sections}")
    print(f"collisions: {result.duplicate_anchor_collisions}")
    print(f"order violations: {result.source_order_violations}")
    print(f"fallback areas: {sum(1 for segment in result.segments if not segment.section_id)}")
    print("================================================")


if __name__ == "__main__":
    raise SystemExit(main())
