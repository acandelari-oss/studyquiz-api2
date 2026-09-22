"""Isolated document interpretation pipeline.

This is the future production seam, kept deliberately disconnected from live
upload.  It proves the intended flow:

raw file -> format-specific interpreter -> validated hierarchy -> safe chunks

Only DOCX is enabled here for now because its source-native heading evidence is
already reliable enough for the clean interpreter contract.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from document_extractors import DocumentExtractionError
from document_hierarchy_contracts import SOURCE_FORMAT_DOCX, SOURCE_FORMAT_PDF, SOURCE_FORMAT_PPTX
from document_interpreter_chunk_adapter import (
    InterpreterChunk,
    chunks_from_interpreted_document,
)
from document_interpreter_contract import DocumentInterpreterResult
from document_interpreter_docx import interpret_docx_document
from document_interpreter_pdf import interpret_pdf_document
from document_interpreter_pptx import interpret_pptx_document


@dataclass(frozen=True)
class InterpretedDocumentPipelineResult:
    interpretation: DocumentInterpreterResult
    chunks: List[InterpreterChunk]


def interpret_document_to_chunks(
    file_bytes: bytes,
    filename: str,
    *,
    document_id: str,
) -> InterpretedDocumentPipelineResult:
    """Run the isolated interpreter pipeline for supported formats."""

    source_format = _source_format_from_filename(filename)
    if source_format == SOURCE_FORMAT_DOCX:
        interpretation = interpret_docx_document(
            file_bytes,
            filename,
            document_id=document_id,
        )
    elif source_format == SOURCE_FORMAT_PPTX:
        interpretation = interpret_pptx_document(
            file_bytes,
            filename,
            document_id=document_id,
        )
    elif source_format == SOURCE_FORMAT_PDF:
        interpretation = interpret_pdf_document(
            file_bytes,
            filename,
            document_id=document_id,
        )
    else:
        raise DocumentExtractionError(
            (
                "The clean document interpreter currently supports PDF, DOCX, and PPTX."
            )
        )

    return InterpretedDocumentPipelineResult(
        interpretation=interpretation,
        chunks=chunks_from_interpreted_document(interpretation),
    )


def _source_format_from_filename(filename: str) -> str:
    extension = os.path.splitext(filename or "")[1].lower().strip()
    if extension == ".docx":
        return SOURCE_FORMAT_DOCX
    if extension == ".pdf":
        return SOURCE_FORMAT_PDF
    if extension == ".pptx":
        return SOURCE_FORMAT_PPTX
    raise DocumentExtractionError(
        "Unsupported document format for the clean document interpreter."
    )
