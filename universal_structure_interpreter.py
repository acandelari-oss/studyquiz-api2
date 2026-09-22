"""Format-independent model structure interpreter for the isolated G2 pipeline."""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from canonical_document import CanonicalDocumentInput, SourcePosition


CONFIDENCE_VALUES = ("HIGH", "MEDIUM", "LOW")
SECTION_SCOPE_VALUES = ("global", "local", "unknown")
ANALYZER_NAME = "universal_model_structure"
ANALYZER_VERSION = "g2-prototype"
MAX_STRUCTURED_INPUT_CHARS = 80_000


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
                    "expected_start",
                    "expected_end",
                    "confidence",
                    "scope",
                ],
                "properties": {
                    "source_title": {"type": ["string", "null"]},
                    "semantic_title": {"type": "string"},
                    "level": {"type": "integer", "minimum": 1, "maximum": 8},
                    "parent": {"type": ["string", "null"]},
                    "expected_start": {
                        "type": ["object", "null"],
                        "additionalProperties": False,
                        "required": [
                            "unit_type",
                            "unit_index",
                            "block_index",
                            "start_offset",
                            "end_offset",
                        ],
                        "properties": {
                            "unit_type": {"type": ["string", "null"]},
                            "unit_index": {"type": ["integer", "null"], "minimum": 1},
                            "block_index": {"type": ["integer", "null"], "minimum": 1},
                            "start_offset": {"type": ["integer", "null"], "minimum": 0},
                            "end_offset": {"type": ["integer", "null"], "minimum": 0},
                        },
                    },
                    "expected_end": {
                        "type": ["object", "null"],
                        "additionalProperties": False,
                        "required": [
                            "unit_type",
                            "unit_index",
                            "block_index",
                            "start_offset",
                            "end_offset",
                        ],
                        "properties": {
                            "unit_type": {"type": ["string", "null"]},
                            "unit_index": {"type": ["integer", "null"], "minimum": 1},
                            "block_index": {"type": ["integer", "null"], "minimum": 1},
                            "start_offset": {"type": ["integer", "null"], "minimum": 0},
                            "end_offset": {"type": ["integer", "null"], "minimum": 0},
                        },
                    },
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
                    "maxItems": 10,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["unit", "type", "title", "confidence"],
                        "properties": {
                            "unit": {"type": ["string", "null"]},
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
        "notes": {"type": "array", "maxItems": 5, "items": {"type": "string"}},
    },
}


SYSTEM_INSTRUCTIONS = """\
You are an expert academic document-structure analyst.

Reconstruct the canonical hierarchy intended by the source material. The input
may be a native PDF file or a deterministic structured representation of a
DOCX/PPTX file.

Preserve source wording for headings whenever you can identify an actual
heading. Put literal heading wording in source_title. Use semantic_title only
for normalized interpretation. Never treat semantic_title as source-exact.

Role, style, slide, paragraph, bullet, and heading metadata are evidence, not
truth. Decide hierarchy from the full document context.

Do not promote local enumerations, examples, questions, tables, captions,
headers, footers, page numbers, or artifacts to the global hierarchy unless
they genuinely function as document sections.

Return compact JSON matching the schema. Do not describe ordinary body content.
"""


USER_TASK = """\
Return the source document's canonical structure. For each genuine section,
include source_title, semantic_title, level, parent, expected_start,
expected_end, confidence, and scope.
"""


@dataclass(frozen=True)
class DocumentSection:
    section_id: str
    source_title: Optional[str]
    semantic_title: str
    level: int
    parent_id: Optional[str]
    parent_title: Optional[str]
    source_order: int
    expected_start: Optional[SourcePosition]
    expected_end: Optional[SourcePosition]
    confidence: str
    scope: str


@dataclass(frozen=True)
class DocumentStructure:
    document_title: str
    document_type: str
    confidence: str
    analyzer: str
    analyzer_version: str
    sections: List[DocumentSection]
    local_structure_summary: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


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


class UniversalStructureError(ValueError):
    def __init__(self, message: str, diagnostics: Optional[ResponseDiagnostics] = None):
        super().__init__(message)
        self.diagnostics = diagnostics


def build_structured_text_representation(document: CanonicalDocumentInput) -> str:
    lines = [
        f"FILE_TYPE: {document.file_type}",
        f"FILENAME: {document.filename}",
        f"TITLE: {document.title}",
        "",
    ]
    for unit in document.units:
        lines.append(
            f"[unit={unit.unit_type}:{unit.unit_index} | label={unit.display_label or ''}]"
        )
        for block in unit.blocks:
            hint_parts = [
                f"block={block.source_position.block_index}",
                f"role_hint={block.role_hint}",
            ]
            if block.style_hint:
                hint_parts.append(f"style={block.style_hint}")
            if block.native_hierarchy_hint is not None:
                hint_parts.append(f"hierarchy_hint={block.native_hierarchy_hint}")
            lines.append(f"[{' | '.join(hint_parts)}]")
            lines.append(block.raw_text or block.text)
            lines.append("")
    text = "\n".join(lines).strip()
    if len(text) > MAX_STRUCTURED_INPUT_CHARS:
        return text[:MAX_STRUCTURED_INPUT_CHARS] + "\n[TRUNCATED_FOR_G2_PROTOTYPE]"
    return text


def build_response_input(
    document: CanonicalDocumentInput,
    *,
    original_file_bytes: Optional[bytes] = None,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": USER_TASK}]
    if document.file_type == "pdf" and original_file_bytes:
        encoded = base64.b64encode(original_file_bytes).decode("ascii")
        content.append(
            {
                "type": "input_file",
                "filename": document.filename,
                "file_data": f"data:application/pdf;base64,{encoded}",
                "detail": "high",
            }
        )
    else:
        content.append(
            {
                "type": "input_text",
                "text": build_structured_text_representation(document),
            }
        )
    return [{"role": "user", "content": content}]


def build_text_config() -> Dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": "douno_universal_document_structure_g2",
            "strict": True,
            "schema": DOCUMENT_STRUCTURE_SCHEMA,
        }
    }


def interpret_document_structure(
    *,
    document: CanonicalDocumentInput,
    model: str,
    client: Optional[Any] = None,
    original_file_bytes: Optional[bytes] = None,
    max_output_tokens: int = 6000,
) -> InterpretationResult:
    if client is None:
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover
            raise UniversalStructureError("OpenAI SDK is unavailable.") from exc
        client = OpenAI()

    started = time.perf_counter()
    try:
        response = client.responses.create(
            model=model,
            instructions=SYSTEM_INSTRUCTIONS,
            input=build_response_input(
                document,
                original_file_bytes=original_file_bytes,
            ),
            text=build_text_config(),
            max_output_tokens=max_output_tokens,
            store=False,
        )
    except Exception as exc:
        raise UniversalStructureError(f"Model structure request failed: {exc}") from exc

    elapsed = time.perf_counter() - started
    raw_text = response_text(response)
    diagnostics = extract_response_diagnostics(response, raw_text)
    parsed = parse_model_response(raw_text, diagnostics)
    return InterpretationResult(
        structure=structure_from_model_json(parsed),
        diagnostics=diagnostics,
        raw_text=raw_text,
        elapsed_seconds=elapsed,
    )


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


def extract_response_diagnostics(response: Any, raw_text: str) -> ResponseDiagnostics:
    usage = _get(response, "usage")
    details = _get(usage, "output_tokens_details")
    incomplete = _get(response, "incomplete_details")
    statuses = []
    for item in _get(response, "output") or []:
        status = _get(item, "status")
        if status:
            statuses.append(str(status))
    return ResponseDiagnostics(
        response_status=_get(response, "status"),
        incomplete_reason=_get(incomplete, "reason"),
        response_max_output_tokens=_get(response, "max_output_tokens"),
        input_tokens=_get(usage, "input_tokens"),
        output_tokens=_get(usage, "output_tokens"),
        total_tokens=_get(usage, "total_tokens"),
        reasoning_tokens=_get(details, "reasoning_tokens"),
        raw_output_chars=len(raw_text),
        output_item_statuses=statuses,
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
            raise UniversalStructureError(
                f"Response incomplete before valid JSON could be parsed: {exc}",
                diagnostics,
            ) from exc
        raise UniversalStructureError(
            f"Completed response did not contain valid structured JSON: {exc}",
            diagnostics,
        ) from exc
    if not isinstance(parsed, dict):
        raise UniversalStructureError("Model response must be a JSON object.", diagnostics)
    return parsed


def structure_from_model_json(data: Dict[str, Any]) -> DocumentStructure:
    title_to_id: Dict[str, str] = {}
    sections: List[DocumentSection] = []
    for index, raw in enumerate(data.get("sections") or [], start=1):
        section_id = f"section-{index:04d}"
        source_title = _clean_optional(raw.get("source_title"))
        semantic_title = _clean_optional(raw.get("semantic_title")) or source_title or f"Section {index}"
        parent_title = _clean_optional(raw.get("parent"))
        parent_id = title_to_id.get(_title_key(parent_title)) if parent_title else None
        section = DocumentSection(
            section_id=section_id,
            source_title=source_title,
            semantic_title=semantic_title,
            level=_coerce_int(raw.get("level"), 1, 1, 8),
            parent_id=parent_id,
            parent_title=parent_title,
            source_order=index,
            expected_start=_position_from_json(raw.get("expected_start")),
            expected_end=_position_from_json(raw.get("expected_end")),
            confidence=_enum(raw.get("confidence"), CONFIDENCE_VALUES, "LOW"),
            scope=_enum(raw.get("scope"), SECTION_SCOPE_VALUES, "unknown"),
        )
        sections.append(section)
        for value in (source_title, semantic_title):
            key = _title_key(value)
            if key and key not in title_to_id:
                title_to_id[key] = section_id
    return DocumentStructure(
        document_title=str(data.get("document_title") or "").strip(),
        document_type=str(data.get("document_type") or "").strip(),
        confidence=_enum(data.get("structure_confidence"), CONFIDENCE_VALUES, "LOW"),
        analyzer=ANALYZER_NAME,
        analyzer_version=ANALYZER_VERSION,
        sections=sections,
        local_structure_summary=data.get("local_structure_summary") or {},
        notes=list((data.get("notes") or [])[:5]),
    )


def confidence_rank(value: str) -> int:
    return {"LOW": 1, "MEDIUM": 2, "HIGH": 3}.get((value or "").upper(), 0)


def _position_from_json(value: Any) -> Optional[SourcePosition]:
    if not isinstance(value, dict):
        return None
    unit_type = str(value.get("unit_type") or "").strip()
    if not unit_type:
        return None
    if value.get("unit_index") is None or value.get("block_index") is None:
        return None
    return SourcePosition(
        unit_type=unit_type,
        unit_index=_coerce_int(value.get("unit_index"), 1, 1, 1_000_000),
        block_index=_coerce_int(value.get("block_index"), 1, 1, 1_000_000),
        start_offset=_coerce_int(value.get("start_offset"), 0, 0, 100_000_000)
        if value.get("start_offset") is not None
        else 0,
        end_offset=_coerce_int(value.get("end_offset"), 0, 0, 100_000_000)
        if value.get("end_offset") is not None
        else None,
    )


def _get(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _clean_optional(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _title_key(value: Optional[str]) -> str:
    return " ".join((value or "").casefold().split())


def _coerce_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError):
        integer = default
    return max(minimum, min(maximum, integer))


def _enum(value: Any, allowed: tuple[str, ...], default: str) -> str:
    text = str(value or "").strip()
    return text if text in allowed else default
