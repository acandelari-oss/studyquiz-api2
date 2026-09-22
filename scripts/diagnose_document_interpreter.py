#!/usr/bin/env python3
"""Read-only diagnostic for the clean document interpreter pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from document_interpreter_pipeline import interpret_document_to_chunks


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only diagnostic for DOUNO's clean document interpreter. "
            "It does not upload, embed, write to the database, or run taxonomy."
        )
    )
    parser.add_argument("path", help="Local PDF, DOCX, or PPTX file to inspect")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of the human-readable report",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=20,
        help="Maximum number of chunks to display in the readable report",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=180,
        help="Maximum preview length for each displayed chunk",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    path = Path(args.path).expanduser().resolve()
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        return 2
    if not path.is_file():
        print(f"ERROR: Not a file: {path}", file=sys.stderr)
        return 2

    result = interpret_document_to_chunks(
        path.read_bytes(),
        path.name,
        document_id=f"diagnostic:{path.name}",
    )
    payload = diagnostic_payload(result)

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print_readable_report(
            payload,
            max_chunks=max(0, args.max_chunks),
            preview_chars=max(40, args.preview_chars),
        )
    return 0


def diagnostic_payload(result) -> Dict[str, Any]:
    interpretation = result.interpretation
    return {
        "document": {
            "document_id": interpretation.document_id,
            "title": interpretation.document_title,
            "source_format": interpretation.source_format,
            "status": interpretation.status,
            "confidence": interpretation.confidence,
            "has_authoritative_structure": interpretation.has_authoritative_structure,
        },
        "summary": {
            "sections": len(interpretation.sections),
            "unresolved_blocks": len(interpretation.unresolved_blocks),
            "chunks": len(result.chunks),
        },
        "diagnostics": interpretation.diagnostics,
        "sections": [
            {
                "section_id": section.section_id,
                "title": section.title,
                "display_title": section.display_title,
                "level": section.level,
                "parent_id": section.parent_id,
                "source_order": section.source_order,
                "confidence": section.confidence,
                "origin": section.origin,
                "owned_blocks": len(section.owned_blocks),
                "metadata": section.metadata,
            }
            for section in interpretation.sections
        ],
        "unresolved_blocks": [
            {
                "block_id": block.block_id,
                "source_order": block.source_order,
                "page": block.page,
                "role": block.role,
                "preview": _preview(block.text, 220),
            }
            for block in interpretation.unresolved_blocks
        ],
        "chunks": [
            {
                "index": index,
                "section_id": chunk.section_id,
                "section_title": chunk.section_title,
                "section_path": chunk.section_path,
                "page": chunk.page,
                "source_block_ids": list(chunk.source_block_ids),
                "source_order_start": chunk.source_order_start,
                "source_order_end": chunk.source_order_end,
                "unresolved": chunk.unresolved,
                "char_count": len(chunk.chunk_text),
                "preview": _preview(chunk.chunk_text, 260),
            }
            for index, chunk in enumerate(result.chunks, start=1)
        ],
    }


def print_readable_report(
    payload: Dict[str, Any],
    *,
    max_chunks: int,
    preview_chars: int,
) -> None:
    document = payload["document"]
    summary = payload["summary"]
    print("================================================")
    print("DOUNO CLEAN DOCUMENT INTERPRETER DIAGNOSTIC")
    print("================================================")
    print("DOCUMENT")
    print(f"file: {document['title']}")
    print(f"type: {document['source_format']}")
    print(f"status: {document['status']}")
    print(f"confidence: {document['confidence']}")
    print(f"authoritative: {document['has_authoritative_structure']}")
    print()
    print("SUMMARY")
    print(f"sections: {summary['sections']}")
    print(f"unresolved_blocks: {summary['unresolved_blocks']}")
    print(f"chunks: {summary['chunks']}")
    print()
    print("HIERARCHY")
    if not payload["sections"]:
        print("(no accepted sections)")
    for section in payload["sections"]:
        indent = "  " * max(0, section["level"] - 1)
        parent = f" parent={section['parent_id']}" if section["parent_id"] else ""
        print(
            f"{indent}- {section['title']} "
            f"[id={section['section_id']} level={section['level']} "
            f"conf={section['confidence']} blocks={section['owned_blocks']}{parent}]"
        )
    print()
    if payload["unresolved_blocks"]:
        print("UNRESOLVED BLOCKS")
        for block in payload["unresolved_blocks"][:10]:
            print(
                f"- order={block['source_order']} page={block['page']} "
                f"role={block['role']} | {block['preview']}"
            )
        remaining = len(payload["unresolved_blocks"]) - 10
        if remaining > 0:
            print(f"... {remaining} more unresolved blocks")
        print()
    print("CHUNKS")
    for chunk in payload["chunks"][:max_chunks]:
        preview = _preview(chunk["preview"], preview_chars)
        print(
            f"{chunk['index']:03d} | page={chunk['page']} | "
            f"section={chunk['section_path']} | chars={chunk['char_count']} | {preview}"
        )
    remaining_chunks = len(payload["chunks"]) - max_chunks
    if remaining_chunks > 0:
        print(f"... {remaining_chunks} more chunks")
    print("================================================")


def _preview(text: str, limit: int) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 1)].rstrip() + "…"


if __name__ == "__main__":
    raise SystemExit(main())
