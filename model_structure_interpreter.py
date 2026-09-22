"""Isolated model-native document structure interpreter.

This module is intentionally not wired into the production ingestion path.
It provides a reusable, structured wrapper around the OpenAI Responses API for
standalone structure-mapping experiments.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


CONFIDENCE_VALUES = ("HIGH", "MEDIUM", "LOW")
SECTION_SCOPE_VALUES = ("global", "local", "unknown")
ANALYZER_NAME = "model_native_pdf_structure"
ANALYZER_VERSION = "g1-prototype"
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
        "structure_confidence": {"type": "string", "enum": list(CONFIDENCE_VALUES)},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "source_title",
                    "semantic_title",
                    "level",
                    "parent",
                    "start_page",
                    "end_page",
                    "confidence",
                    "scope",
                ],
                "properties": {
                    "source_title": {"type": ["string", "null"]},
                    "semantic_title": {"type": "string"},
                    "level": {"type": "integer", "minimum": 1, "maximum": 8},
                    "parent": {"type": ["string", "null"]},
                    "start_page": {"type": ["integer", "null"], "minimum": 1},
                    "end_page": {"type": ["integer", "null"], "minimum": 1},
                    "confidence": {"type": "string", "enum": list(CONFIDENCE_VALUES)},
                    "scope": {"type": "string", "enum": list(SECTION_SCOPE_VALUES)},
                },
            },
        },
        "local_structure_summary": {
            "type": "object",
            "additionalProperties": False,
            "required": ["detected", "types", "representative_examples"],
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
                        "required": ["page", "type", "title", "confidence"],
                        "properties": {
                            "page": {"type": ["integer", "null"], "minimum": 1},
                            "type": {"type": "string"},
                            "title": {"type": "string"},
                            "confidence": {
                                "type": "string",
                                "enum": list(CONFIDENCE_VALUES),
                            },
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


SYSTEM_INSTRUCTIONS = """\
You are an expert academic document-structure analyst.

Reconstruct the canonical hierarchy intended by the source document. Use the
PDF itself, including visual/layout information when available.

Preserve source wording for headings whenever you can identify an actual
heading in the source. Put that literal source heading in source_title. Use
semantic_title only for a concise normalized/descriptive interpretation.
Never use semantic_title as a substitute for source_title when the source
heading is not visible or uncertain.

Keep the output compact. Do not attempt to exhaustively describe the PDF. Do
not list ordinary body content. Prefer concise titles and structured data over
natural-language explanations.

Do not promote local enumerations, questions, examples, tables, figures,
captions, headers, footers, page numbers, or artifacts to the global hierarchy
unless they genuinely function as document sections.

Report only representative local-structure examples, capped at 10 for the
entire document. Use notes only for exceptional document-level observations
that affect interpretation of the hierarchy, capped at 5.

Return only the structured JSON requested by the schema.
"""


USER_TASK = """\
Analyze this PDF and reconstruct its canonical academic/semantic hierarchy.

For each real section, return source_title when the heading wording appears in
the source, semantic_title, level, parent source title if any, start_page,
end_page, confidence, and scope.

Do not list ordinary body content. Do not exhaustively enumerate local
structures. Instead, summarize whether local/non-global structures were
detected, list their broad types, and provide at most 10 representative
examples for the whole document.
"""


@dataclass(frozen=True)
class ModelSection:
    section_id: str
    source_title: Optional[str]
    semantic_title: str
    level: int
    parent_id: Optional[str]
    parent_title: Optional[str]
    source_order: int
    start_page: Optional[int]
    end_page: Optional[int]
    confidence: str
    scope: str


@dataclass(frozen=True)
class DocumentStructure:
    document_title: str
    document_type: str
    confidence: str
    analyzer: str
    analyzer_version: str
    sections: List[ModelSection]
    local_structure_summary: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


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
class InterpretationResult:
    structure: DocumentStructure
    diagnostics: ResponseDiagnostics
    raw_text: str
    elapsed_seconds: float


class ModelStructureError(ValueError):
    def __init__(self, message: str, diagnostics: Optional[ResponseDiagnostics] = None):
        super().__init__(message)
        self.diagnostics = diagnostics


def encode_pdf_file_data(pdf_bytes: bytes) -> str:
    encoded = base64.b64encode(pdf_bytes).decode("ascii")
    return f"data:application/pdf;base64,{encoded}"


def build_response_input(pdf_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": USER_TASK},
                {
                    "type": "input_file",
                    "filename": filename,
                    "file_data": encode_pdf_file_data(pdf_bytes),
                    "detail": "high",
                },
            ],
        }
    ]


def build_text_config() -> Dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": "douno_document_structure_g1",
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


def _get_attr_or_key(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def extract_usage(response: Any) -> UsageSummary:
    usage = _get_attr_or_key(response, "usage")
    output_details = _get_attr_or_key(usage, "output_tokens_details")
    return UsageSummary(
        input_tokens=_get_attr_or_key(usage, "input_tokens"),
        output_tokens=_get_attr_or_key(usage, "output_tokens"),
        total_tokens=_get_attr_or_key(usage, "total_tokens"),
        reasoning_tokens=_get_attr_or_key(output_details, "reasoning_tokens"),
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


def parse_model_response(
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
            raise ModelStructureError(
                "Response incomplete before valid JSON could be parsed: "
                f"{exc}",
                diagnostics,
            ) from exc
        raise ModelStructureError(
            "Completed response did not contain valid structured JSON: "
            f"{exc}",
            diagnostics,
        ) from exc

    if not isinstance(parsed, dict):
        raise ModelStructureError("Model JSON response must be an object.", diagnostics)

    for key in DOCUMENT_STRUCTURE_SCHEMA["required"]:
        if key not in parsed:
            raise ModelStructureError(
                f"Model JSON response is missing '{key}'.",
                diagnostics,
            )
    return parsed


def structure_from_model_json(data: Dict[str, Any]) -> DocumentStructure:
    raw_sections = data.get("sections") or []
    title_to_id: Dict[str, str] = {}
    sections: List[ModelSection] = []

    for index, raw in enumerate(raw_sections, start=1):
        section_id = f"section-{index:04d}"
        source_title = _clean_optional_text(raw.get("source_title"))
        semantic_title = (
            _clean_optional_text(raw.get("semantic_title"))
            or source_title
            or f"Section {index}"
        )
        parent_title = _clean_optional_text(raw.get("parent"))
        parent_id = title_to_id.get(_title_key(parent_title)) if parent_title else None
        level = _coerce_int(raw.get("level"), default=1, minimum=1, maximum=8)
        confidence = _enum_value(raw.get("confidence"), CONFIDENCE_VALUES, "LOW")
        scope = _enum_value(raw.get("scope"), SECTION_SCOPE_VALUES, "unknown")

        section = ModelSection(
            section_id=section_id,
            source_title=source_title,
            semantic_title=semantic_title,
            level=level,
            parent_id=parent_id,
            parent_title=parent_title,
            source_order=index,
            start_page=_coerce_optional_int(raw.get("start_page")),
            end_page=_coerce_optional_int(raw.get("end_page")),
            confidence=confidence,
            scope=scope,
        )
        sections.append(section)

        for title in (source_title, semantic_title):
            key = _title_key(title)
            if key and key not in title_to_id:
                title_to_id[key] = section_id

    return DocumentStructure(
        document_title=str(data.get("document_title") or "").strip(),
        document_type=str(data.get("document_type") or "").strip(),
        confidence=_enum_value(
            data.get("structure_confidence"),
            CONFIDENCE_VALUES,
            "LOW",
        ),
        analyzer=ANALYZER_NAME,
        analyzer_version=ANALYZER_VERSION,
        sections=sections,
        local_structure_summary=data.get("local_structure_summary") or {},
        notes=list((data.get("notes") or [])[:MAX_NOTES]),
    )


def interpret_pdf_structure(
    *,
    pdf_bytes: bytes,
    filename: str,
    model: str,
    client: Optional[Any] = None,
    max_output_tokens: int = 6000,
) -> InterpretationResult:
    if not pdf_bytes:
        raise ModelStructureError("PDF bytes are empty.")
    if not filename:
        raise ModelStructureError("Filename is required.")

    if client is None:
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ModelStructureError(
                "OpenAI SDK is not available in this environment."
            ) from exc
        client = OpenAI()

    started = time.perf_counter()
    try:
        response = client.responses.create(
            model=model,
            instructions=SYSTEM_INSTRUCTIONS,
            input=build_response_input(pdf_bytes, filename),
            text=build_text_config(),
            max_output_tokens=max_output_tokens,
            store=False,
        )
    except Exception as exc:
        raise ModelStructureError(f"Model structure request failed: {exc}") from exc

    elapsed = time.perf_counter() - started
    raw_text = response_text(response)
    diagnostics = extract_response_diagnostics(response, raw_text)
    parsed = parse_model_response(raw_text, diagnostics=diagnostics)
    return InterpretationResult(
        structure=structure_from_model_json(parsed),
        diagnostics=diagnostics,
        raw_text=raw_text,
        elapsed_seconds=elapsed,
    )


def confidence_rank(value: str) -> int:
    return {"LOW": 1, "MEDIUM": 2, "HIGH": 3}.get((value or "").upper(), 0)


def _clean_optional_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _title_key(value: Optional[str]) -> str:
    return " ".join((value or "").casefold().split())


def _coerce_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError):
        integer = default
    return max(minimum, min(maximum, integer))


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return None
    return integer if integer >= 1 else None


def _enum_value(value: Any, allowed: tuple[str, ...], default: str) -> str:
    text = str(value or "").strip()
    return text if text in allowed else default
